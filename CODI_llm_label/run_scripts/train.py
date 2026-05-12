import argparse
import os
import random
import sys
sys.path.append(".")

import diffuser.utils as utils
import torch
import yaml
from diffuser.utils.launcher_util import (
    build_config_from_dict,
    discover_latest_checkpoint_path,
)


def main(Config, RUN):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    utils.set_seed(Config.seed)
    dataset_extra_kwargs = dict()

    # configs that does not exist in old yaml files
    Config.discrete_action = getattr(Config, "discrete_action", False)
    Config.num_actions = getattr(Config, "num_actions", 0)
    Config.state_loss_weight = getattr(Config, "state_loss_weight", None)
    Config.opponent_loss_weight = getattr(Config, "opponent_loss_weight", None)
    Config.use_seed_dataset = getattr(Config, "use_seed_dataset", False)
    Config.residual_attn = getattr(Config, "residual_attn", True)
    Config.use_temporal_attention = getattr(Config, "use_temporal_attention", True)
    Config.env_ts_condition = getattr(Config, "env_ts_condition", False)
    Config.states_condition = getattr(Config, "states_condition", False)
    Config.use_return_to_go = getattr(Config, "use_return_to_go", False)
    Config.use_zero_padding = getattr(Config, "use_zero_padding", True)
    Config.use_inv_dyn = getattr(Config, "use_inv_dyn", True)
    Config.use_fwd_dyn = getattr(Config, "use_fwd_dyn", True)
    Config.use_rwd_model = getattr(Config, "use_rwd_model", True)
    Config.use_legal_model = getattr(Config, "use_legal_model", True)
    Config.pred_future_padding = getattr(Config, "pred_future_padding", False)
    if not hasattr(Config, "agent_condition_type"):
        if Config.decentralized_execution:
            Config.agent_condition_type = "single"
        else:
            Config.agent_condition_type = "all"

    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#

    # SequenceAugDataset
    dataset_config = utils.Config(
        'datasets.SequenceAugDataset',
        savepath="dataset_config.pkl",
        env_type=Config.env_type,
        env=Config.dataset,
        n_agents=Config.n_agents,
        horizon=Config.horizon,
        history_horizon=Config.history_horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        max_n_episodes=Config.max_n_episodes,
        use_padding=Config.use_padding,
        use_action=Config.use_action,
        discrete_action=Config.discrete_action,
        max_path_length=Config.max_path_length,
        include_returns=Config.returns_condition,
        include_rewards=Config.use_rwd_model,
        include_labels=Config.include_labels,
        label_model_hidden_dim=Config.hidden_dim,
        include_env_ts=Config.env_ts_condition,
        include_states=Config.states_condition,
        returns_scale=Config.returns_scale,
        discount=Config.discount,
        termination_penalty=Config.termination_penalty,
        agent_share_parameters=utils.config.import_class(
            Config.model
        ).agent_share_parameters,
        use_seed_dataset=Config.use_seed_dataset,
        seed=Config.seed,
        use_inv_dyn=Config.use_inv_dyn,
        decentralized_execution=Config.decentralized_execution,
        use_zero_padding=Config.use_zero_padding,
        agent_condition_type=Config.agent_condition_type,
        pred_future_padding=Config.pred_future_padding,
        circular_shift=Config.circular_shift,
        shift_ratio=Config.shift_ratio,
        syn_dataset_dir=Config.syn_dataset_dir,
        **dataset_extra_kwargs,
    )

    data_encoder_config = utils.Config(
        getattr(Config, "data_encoder", "utils.IdentityEncoder"),
        savepath="data_encoder_config.pkl",
    )

    dataset = dataset_config()
    # renderer = render_config()
    data_encoder = data_encoder_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#


    diffusion_config = utils.Config(
        'models.LabelModelWrapper',
        savepath="diffusion_config.pkl",
        n_agents=Config.n_agents,
        horizon=Config.horizon,
        history_horizon=Config.history_horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        discrete_action=Config.discrete_action,
        num_actions=Config.num_actions,
        hidden_dim=Config.hidden_dim,
        dropout=getattr(Config, 'dropout', 0.0),
        device=Config.device,
    )

    label_dataset = None
    label_dataset_eval = None

    if getattr(Config, 'train_label_model', False):
        cnts = list(range(Config.start_from_batch, Config.end_before_batch))
        random.shuffle(cnts)
        train_cnts = cnts[:int(len(cnts)*Config.train_split)]
        eval_cnts = cnts[int(len(cnts)*Config.train_split):]

        label_dataset_config = utils.Config(
            'datasets.SequenceLLMLabelsDataset',
            cnts=train_cnts,
            savepath="label_dataset_config.pkl",
            env_type=Config.env_type,
            env=Config.dataset,
            n_agents=Config.n_agents,
            horizon=Config.horizon,
            use_llm_labels=Config.use_llm_labels,
            task_dir=Config.task_dir
        )

        label_dataset_eval_config = utils.Config(
            'datasets.SequenceLLMLabelsDataset',
            cnts=eval_cnts,
            savepath="label_dataset_eval_config.pkl",
            env_type=Config.env_type,
            env=Config.dataset,
            n_agents=Config.n_agents,
            horizon=Config.horizon,
            use_llm_labels=Config.use_llm_labels,
            task_dir=Config.task_dir
        )
        label_dataset=label_dataset_config()
        label_dataset_eval=label_dataset_eval_config()

        
    trainer_config = utils.Config(
        utils.Trainer,
        savepath="trainer_config.pkl",
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        eval_freq=Config.eval_freq,
        save_parallel=Config.save_parallel,
        bucket=logger.root,
        n_reference=Config.n_reference,
        train_device=Config.device,
        save_checkpoints=Config.save_checkpoints,
        start_from_batch=getattr(Config, 'start_from_batch', 0),
        end_before_batch=getattr(Config, 'end_before_batch', 1200),
        train_label_model=getattr(Config, 'train_label_model', False),
        label_dataset=label_dataset,
        label_dataset_eval=label_dataset_eval,
        label_model_eval_freq=getattr(Config, 'label_model_eval_freq', None),
        use_llm_labels=getattr(Config, 'use_llm_labels', True),
        prompt_handler=getattr(Config, 'prompt_handler', ''),
        task_dir=getattr(Config, 'task_dir', ''),
    )

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    diffusion = diffusion_config()
    trainer = trainer_config(diffusion, dataset, renderer=None)

    if Config.continue_training:
        loadpath = discover_latest_checkpoint_path(
            os.path.join(trainer.bucket, logger.prefix, "checkpoint")
        )
        if loadpath is not None:
            state_dict = torch.load(loadpath, map_location=Config.device)
            logger.print(
                f"\nLoaded checkpoint from {loadpath} (step {state_dict['step']})\n",
                color="green",
            )
            trainer.step = state_dict["step"]
            trainer.model.load_state_dict(state_dict["model"])
            trainer.ema_model.load_state_dict(state_dict["ema"])


    # -----------------------------------------------------------------------------#
    # --------------------------------- main loop ---------------------------------#
    # -----------------------------------------------------------------------------#

    n_epochs = int((Config.n_train_steps - trainer.step) // Config.n_steps_per_epoch)

    for i in range(n_epochs):
        logger.print(f"Epoch {i} / {n_epochs} | {logger.prefix}")
        trainer.train(n_train_steps=Config.n_steps_per_epoch)
    trainer.finish_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=str, default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.SafeLoader)

    from ml_logger import RUN, logger

    Config = build_config_from_dict(exp_specs)

    Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    job_name = Config.job_name.format(**vars(Config))
    RUN.prefix, RUN.job_name, _ = RUN(
        script_path=__file__,
        exp_name=exp_specs["exp_name"],
        job_name=job_name + f"/{Config.seed}",
    )

    logger.configure(RUN.prefix, root=RUN.script_root)
    # logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.remove("parameters.pkl")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))
    logger.log_text(
        """
                    charts:
                    - yKey: loss
                      xKey: steps
                    - yKey: a0_loss
                      xKey: steps
                    """,
        filename=".charts.yml",
        dedent=True,
        overwrite=True,
    )
    logger.save_yaml(exp_specs, "exp_specs.yml")

    main(Config, RUN)
