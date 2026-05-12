import copy
import os

import einops
import torch
from ml_logger import logger

from .arrays import apply_dict, batch_to_device, to_device, to_np
from .timer import Timer


def cycle(dl):
    while True:
        for data in dl:
            yield data


class EMA:
    """
    empirical moving average
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        eval_freq=100000,
        save_parallel=False,
        n_reference=8,
        bucket=None,
        train_device="cuda",
        save_checkpoints=False,
        start_from_batch=0,
        end_before_batch=1200,
        train_label_model=False,
        label_dataset=None,
        label_dataset_eval=None,
        label_model_eval_freq=None,
        use_llm_labels=True,
        prompt_handler='',
        task_dir=''

    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints

        self.start_from_batch = start_from_batch
        self.end_before_batch = end_before_batch
        self.train_label_model = train_label_model
        self.label_dataset = label_dataset
        self.label_dataset_eval = label_dataset_eval
        self.label_model_eval_freq = label_model_eval_freq
        self.use_llm_labels = use_llm_labels
        self.prompt_handler = prompt_handler
        self.task_dir = task_dir

        if self.label_dataset is not None:
            self.label_dataloader = cycle(
                torch.utils.data.DataLoader(
                    self.label_dataset,
                    batch_size=train_batch_size,
                    num_workers=0,
                    shuffle=True,
                    pin_memory=True,
                )
            )
        
        if self.label_dataset_eval is not None:
            self.label_dataloader_eval = cycle(
                torch.utils.data.DataLoader(
                    self.label_dataset_eval,
                    batch_size=train_batch_size,
                    num_workers=0,
                    shuffle=True,
                    pin_memory=True,
                )
            )

        self.step_start_ema = step_start_ema

        assert (
            eval_freq % save_freq == 0
        ), f"eval_freq must be a multiple of save_freq, but got {eval_freq} and {save_freq} respectively"
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.eval_freq = eval_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset

        if dataset is not None:
            self.dataloader = cycle(
                torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=train_batch_size,
                    num_workers=0,
                    shuffle=True,
                    pin_memory=True,
                )
            )
            self.dataloader_vis = cycle(
                torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=1,
                    num_workers=0,
                    shuffle=True,
                    pin_memory=True,
                )
            )

        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

        self.evaluator = None
        self.device = train_device

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def finish_training(self):
        if self.step % self.save_freq == 0:
            self.save()
        if self.evaluator is not None:
            del self.evaluator

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps):
        timer = Timer()

        if getattr(self, '_first_time_train', None) is None:
            self._first_time_train = False
            if not self.train_label_model:
                print(f"[WARNING] starting from batch {self.start_from_batch} to collect labels")
                self.batch_idx = self.start_from_batch
                for _ in range(self.start_from_batch):
                    next(self.dataloader)

        for _ in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader if not self.train_label_model else self.label_dataloader)
                batch = batch_to_device(batch, device=self.device)

                extra = dict()
                if not self.train_label_model:
                    extra['batch_idx'] = self.batch_idx
                    extra['use_llm_labels'] = self.use_llm_labels
                    extra['prompt_handler'] = self.prompt_handler
                    extra['task_dir'] = self.task_dir
                    self.batch_idx += 1
                    loss, infos = self.model.collect_label_batch(**batch, **extra)
                else:
                    loss, infos = self.model.loss(**batch, **extra)
                
                loss = loss / self.gradient_accumulate_every
                loss.backward()

                if not self.train_label_model and self.batch_idx >= self.end_before_batch:
                    break

            if not self.train_label_model and self.batch_idx >= self.end_before_batch:
                    break

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            
            if self.train_label_model and self.label_model_eval_freq is not None and self.step % self.label_model_eval_freq == 0:
                sum_eval_info = dict()
                self.model.label_model.eval()
                with torch.no_grad():
                    for k in range(10):
                        eval_batch = next(self.label_dataloader_eval)
                        eval_batch = batch_to_device(eval_batch, device=self.device)
                        info = self.model.eval_label_model(**eval_batch)
                        for key in info.keys():
                            if not sum_eval_info.get(key):
                                sum_eval_info[key] = []
                            sum_eval_info[key].append(info[key])
                sum_eval_info_str = " | ".join(
                    [f"eval_{key}: {sum(val) / len(val):8.4f}" for key, val in sum_eval_info.items()]
                )
                logger.print(
                    f"{self.step}: {sum_eval_info_str}"
                )
                self.model.label_model.train()

            if self.step % self.save_freq == 0:
                if self.train_label_model:
                    self.save_label_model()
                else:
                    self.save()

            if self.step % self.log_freq == 0:
                infos_str = " | ".join(
                    [f"{key}: {val:8.4f}" for key, val in infos.items()]
                )
                logger.print(
                    f"{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}"
                )
                metrics = {k: v.detach().item() for k, v in infos.items()}
                logger.log(
                    step=self.step, loss=loss.detach().item(), **metrics, flush=True
                )

            self.step += 1

    def evaluate(self):
        assert (
            self.evaluator is not None
        ), "Method `evaluate` can not be called when `self.evaluator` is None. Set evaluator with `self.set_evaluator` first."
        self.evaluator.evaluate(load_step=self.step)

    def save(self):
        """
        saves model and ema to disk;
        syncs to storage bucket if a bucket is specified
        """

        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        savepath = os.path.join(self.bucket, logger.prefix, "checkpoint")
        os.makedirs(savepath, exist_ok=True)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f"state_{self.step}.pt")
        else:
            savepath = os.path.join(savepath, "state.pt")
        torch.save(data, savepath)
        logger.print(f"[ utils/training ] Saved model to {savepath}")

    def save_label_model(self):
        data = {
            "step": self.step,
            "model": self.model.label_model.state_dict(),
            "ema": self.ema_model.label_model.state_dict(),
        }
        savepath = os.path.join(self.bucket, logger.prefix, "checkpoint")
        os.makedirs(savepath, exist_ok=True)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f"label_model_{self.step}.pt")
        else:
            savepath = os.path.join(savepath, "label_model.pt")
        torch.save(data, savepath)
        logger.print(f"[ utils/training ] Saved model to {savepath}")

    def load(self):
        """
        loads model and ema from disk
        """

        loadpath = os.path.join(self.bucket, logger.prefix, "checkpoint/state.pt")
        data = torch.load(loadpath)

        self.step = data["step"]
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])
