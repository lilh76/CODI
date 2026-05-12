from typing import Optional
import os
import pickle

import numpy as np
import torch
from ml_logger import logger

import diffuser.utils as utils
from diffuser.utils.launcher_util import build_config_from_dict
from diffuser.utils.stitching import trajectory_stitching_parallel


class MADSynthesizer:
    def __init__(
        self,
        n_gen=4,
        gen_batch_size=4,
        check_discrepancy=True,
        partially_noise=True,
        times_of_regen_upper_limit=3,
        total_times_of_regen_upper_limit=10,
        use_composition_tech=False,
        use_legal_model=False,
        recon_threshold=0.0003,
        max_path_length_stitch=25,
        each_ig_step_num=10,
        test_ret=100,
        include_labels=False,
        verbose=False,
    ):
        self.n_gen = n_gen
        self.gen_batch_size = gen_batch_size
        self.check_discrepancy = check_discrepancy
        self.partially_noise = partially_noise
        self.times_of_regen_upper_limit = times_of_regen_upper_limit
        self.total_times_of_regen_upper_limit = total_times_of_regen_upper_limit
        self.use_composition_tech = use_composition_tech
        self.use_legal_model = use_legal_model
        self.recon_threshold = recon_threshold
        self.max_path_length_stitch = max_path_length_stitch
        self.each_ig_step_num = each_ig_step_num
        self.test_ret = test_ret
        self.include_labels = include_labels

        self.initialized = False
        self.verbose = verbose

    def synthesize(self, load_step=None):
        assert (
            self.initialized is True
        ), "Evaluator should be initialized before evaluation."

        Config = self.Config
        loadpath = os.path.join(self.log_dir, "checkpoint")

        utils.set_seed(Config.seed)

        if Config.save_checkpoints:
            assert load_step is not None
            loadpath = os.path.join(loadpath, f"state_{load_step}.pt")
        else:
            loadpath = os.path.join(loadpath, "state.pt")
            
        print(loadpath)

        state_dict = torch.load(loadpath, map_location=Config.device)
        state_dict["model"] = {
            k: v
            for k, v in state_dict["model"].items()
            if "value_diffusion_model." not in k
        }
        state_dict["ema"] = {
            k: v
            for k, v in state_dict["ema"].items()
            if "value_diffusion_model." not in k
        }

        self.trainer.step = state_dict["step"]
        self.trainer.model.load_state_dict(state_dict["model"])
        self.trainer.ema_model.load_state_dict(state_dict["ema"])

        save_traj_dir = os.path.join(
            self.log_dir,
            f"syn_results/step_{load_step}-ddim"
            if getattr(Config, "use_ddim_sample", False)
            else f"syn_results/step_{load_step}",
        )

        if self.check_discrepancy:
            save_traj_dir = save_traj_dir + "-check_discrepancy" + str(self.recon_threshold)

        if self.partially_noise:
            save_traj_dir = save_traj_dir + "-partially_noise"

        if self.test_ret:
            save_traj_dir = save_traj_dir + f"-test_ret_{Config.test_ret}"

        if self.max_path_length_stitch:
            save_traj_dir = save_traj_dir + f"-path_len_{self.max_path_length_stitch}"

        if self.include_labels:
            save_traj_dir = save_traj_dir + "-include_labels"

        if self.use_composition_tech:
            save_traj_dir = save_traj_dir + "-use_composition_tech"

        if self.rewrite_cgw:
            save_traj_dir = save_traj_dir + f"-cg_{self.trainer.ema_model.condition_guidance_w}"

        if not os.path.exists(save_traj_dir):
            os.makedirs(save_traj_dir)
        
        normed_observations, actions, rewards, avail = trajectory_stitching_parallel(
            args=Config, 
            n_gen=self.n_gen, 
            gen_batch_size=self.gen_batch_size, 
            trainer=self.trainer, 
            dataset=self.dataset, 
            horizon=Config.horizon, 
            n_agents=Config.n_agents, 
            discrete_action=Config.discrete_action, 
            cond_return=Config.test_ret,
            include_labels=Config.include_labels,
            times_of_regen_upper_limit=self.times_of_regen_upper_limit, 
            total_times_of_regen_upper_limit=self.total_times_of_regen_upper_limit, 
            use_composition_tech=self.use_composition_tech, 
            threshold=self.recon_threshold,
            max_path_length_stitch=self.max_path_length_stitch,
            partially_noise=self.partially_noise,
            each_ig_step_num=self.each_ig_step_num,
            verbose=self.verbose,
            save_traj_dir=save_traj_dir,
            normalizer = self.normalizer,
        )

        if not self.discrete_action:
            actions = self.normalizer.unnormalize(actions, "actions")
        observations = self.normalizer.unnormalize(normed_observations, "observations")

        if self.dataset.include_states:
            states = observations[:, :, :, -self.dataset.state_dim:]
            states = np.mean(states, axis=2)
            observations = observations[:, :, :, :-self.dataset.state_dim]

        np.save(os.path.join(save_traj_dir, "obs.npy"), observations)
        np.save(os.path.join(save_traj_dir, "acs.npy"), actions)
        np.save(os.path.join(save_traj_dir, "avail.npy"), avail)
        np.save(os.path.join(save_traj_dir, "rew.npy"), rewards)
        if self.dataset.include_states:
            np.save(os.path.join(save_traj_dir, "state.npy"), states)

    def init(self, log_dir: str, condition_guidance_w: Optional[float] = None, **kwargs):
        assert self.initialized is False, "Synthesizer can only be initialized once."

        self.log_dir = log_dir
        with open(os.path.join(log_dir, "parameters.pkl"), "rb") as f:
            params = pickle.load(f)

        Config = build_config_from_dict(params["Config"])
        
        self.Config = Config = build_config_from_dict(kwargs, Config)
        self.Config.joint_inv = getattr(Config, "joint_inv", False)
        self.Config.use_return_to_go = getattr(Config, "use_return_to_go", False)
        self.Config.use_ddim_sample = getattr(Config, "use_ddim_sample", False)
        self.Config.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.configure(log_dir)
        torch.backends.cudnn.benchmark = True

        with open(os.path.join(log_dir, "model_config.pkl"), "rb") as f:
            model_config = pickle.load(f)

        with open(os.path.join(log_dir, "diffusion_config.pkl"), "rb") as f:
            diffusion_config = pickle.load(f)

        with open(os.path.join(log_dir, "trainer_config.pkl"), "rb") as f:
            trainer_config = pickle.load(f)

        with open(os.path.join(log_dir, "dataset_config.pkl"), "rb") as f:
            dataset_config = pickle.load(f)

        self.rewrite_cgw = False
        if condition_guidance_w is not None:
            print(f"Set condition guidance weight to {condition_guidance_w}")
            diffusion_config._dict["condition_guidance_w"] = condition_guidance_w
            self.rewrite_cgw = True
            
        diffusion_config._dict["use_legal_model"] = self.use_legal_model

        self.dataset = dataset_config()
        self.normalizer = self.dataset.normalizer
        self.mask_generator = self.dataset.mask_generator

        model = model_config()
        diffusion = diffusion_config(model)
        self.trainer = trainer_config(diffusion, None, None)

        if Config.use_ddim_sample:
            print(f"\n Use DDIM Sampler of {Config.n_ddim_steps} Step(s) \n")
            self.trainer.model.set_ddim_scheduler(Config.n_ddim_steps)
            self.trainer.ema_model.set_ddim_scheduler(Config.n_ddim_steps)

        self.discrete_action = Config.discrete_action

        self.initialized = True
        