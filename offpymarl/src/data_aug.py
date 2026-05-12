import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
import json
from datetime import datetime
import numpy as np

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.offline_buffer import OfflineBuffer, MADiffOfflineBuffer
from components.transforms import OneHot

from synthesize import trajectory_stitching_parallel

def run(_run, _config, _log):

    # check args sanity
    print(_config["use_cuda"])
    _config = args_sanity_check(_config, _log)
    
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    

    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    results_save_dir = args.results_save_dir
    if args.use_wandb:
        args.use_tensorboard = False
    # assert args.use_tensorboard and args.use_wandb
    
    
    if args.use_tensorboard and not args.evaluate:
        # only log tensorboard when in training mode
        tb_exp_direc = os.path.join(results_save_dir, 'logs')
        logger.setup_tb(tb_exp_direc)
        
    
    
    if args.use_wandb and not args.evaluate:
        wandb_run_name = args.results_save_dir.split('/')
        wandb_run_name = "/".join(wandb_run_name[wandb_run_name.index("results")+1:])
        wandb_exp_direc = os.path.join(results_save_dir, 'logs')
        logger.setup_wandb(wandb_exp_direc, project=args.wandb_project_name, name=wandb_run_name,
                           run_id=args.resume_id, config=args)
    # write config file
    config_str = json.dumps(vars(args), indent=4)
    with open(os.path.join(results_save_dir, "config.json"), "w") as f:
        f.write(config_str)
    # set model save dir
    args.model_save_dir = os.path.join(results_save_dir, 'models')

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)

def run_sequential(args, logger):

    # In offline training, we use t_max to denote iterations
    
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    for k, v in env_info.items():
        setattr(args, k, v)

    # Create Offline Data
    match args.env:
        case "sc2":
            args.map_name = args.env_args["map_name"]
        case "gymma":
            env_name, args.map_name = args.env_args['key'].split(':')
            args.env = env_name
            # if 'SimpleTag' in args.map_name:
            #     args.n_agents = args.n_agents - 1
        case _:
                raise NotImplementedError("Do not support such envs: {}".format(args.env))

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    if args.agent == 'madiff':
        offline_buffer_ = OfflineBuffer(args, args.map_name, args.offline_data_quality,
                args.offline_bottom_data_path, args.offline_max_buffer_size, 
                shuffle=args.offline_data_shuffle) # device defauly cpu
        madiff_dataset = MADiffOfflineBuffer(args, offline_buffer_).dataset
        mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args, madiff_dataset=madiff_dataset)
    else:
        mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
        
    if args.agent == 'madiff':
        offline_buffer = MADiffOfflineBuffer(args, offline_buffer_)
    else:
        offline_buffer = OfflineBuffer(args, args.map_name, args.offline_data_quality,
                                    args.offline_bottom_data_path, args.offline_max_buffer_size, 
                                    shuffle=args.offline_data_shuffle) # device defauly cpu
        
    dataset = offline_buffer.dataset
    if args.transform_h5:
        # print(dir(dataset.fields))
        # print(dataset.fields.observations.shape)  # [n_traj, seq_len, n_agents, obs_shape + n_agents(id)]
        # print(dataset.fields.actions.shape)  # [n_traj, seq_len + h2 - 1, n_agents, 1]
        # print(dataset.fields.rewards.shape)  # [n_traj, seq_len + h2 - 1, n_agents, 1]
        # print(dataset.fields.terminals.shape)  # [n_traj, seq_len + h2 - 1, n_agents, 1]
        # print(dataset.fields.random_labels.shape)  # [n_traj, seq_len, n_agents, 1]

        observations = dataset.fields.observations
        actions = dataset.fields.actions
        rewards = dataset.fields.rewards
        
        if actions.shape[1] > observations.shape[1]:
            actions = actions[:, : observations.shape[1]]
        
        if rewards.shape[1] > observations.shape[1]:
            rewards = rewards[:, : observations.shape[1]]
            
        observations = observations[:, :-1]
        actions = actions[:, :-1]
        rewards = rewards[:, :-1]
        
        print(observations.shape, actions.shape, rewards.shape)
        save_traj_dir = os.path.join(args.results_save_dir, 'h52npy')
        if not os.path.exists(save_traj_dir):
            os.makedirs(save_traj_dir)
        np.save(os.path.join(save_traj_dir, "obs.npy"), observations) # [bs, seq_len, n_agents, obs_shape]
        np.save(os.path.join(save_traj_dir, "acs.npy"), actions) # [bs, seq_len, n_agents, 1]
        np.save(os.path.join(save_traj_dir, "rew.npy"), rewards) # [bs, seq_len, n_agents, 1]
        
        if hasattr(dataset.fields, 'random_labels'):
            random_labels = dataset.fields.random_labels
            if random_labels.shape[1] > observations.shape[1]:
                random_labels = random_labels[:, : observations.shape[1]]
            print(random_labels.shape)
            np.save(os.path.join(save_traj_dir, "labels.npy"), random_labels) # [bs, seq_len, n_agents, 1]
            
        print()
        print(save_traj_dir)
        print()
        return

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        print("use cuda")
        learner.cuda()

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
    
    if args.checkpoint_path != "":
        dataset = offline_buffer.dataset
        agent = mac.agent
        
        analyze_data = False
        syn_data = False

        if analyze_data:
            bs = len(dataset) // 2
            episode_sample = offline_buffer.sample_bs(bs)
            if isinstance(episode_sample, dict):
                episode_sample = to_device_recursive(episode_sample, args.device)
            else: # EpisodeBatch
                if episode_sample.device != args.device:
                    episode_sample.to(args.device)
            
            obs_target = episode_sample['x'][..., agent.diffusion.action_dim :] # [bs, h1+h2, n_agents, obs_shape] # h1=0
            
            obs = obs_target.clone()
            obs[:, args.history_horizon + 1 : ] = 0.0
            conditions = {
                "x": obs,
                "masks": episode_sample['cond']['masks'],
            }
            if args.returns_condition:
                returns = episode_sample['returns']
            else:
                returns = None
            attention_masks = episode_sample['attention_masks'] * 0.0 + 1.0
            
            samples = agent.diffusion.conditional_sample(
                conditions,
                returns=returns,
                env_ts=None,
                attention_masks=attention_masks,
            )
            samples = samples[:, args.history_horizon:] # [bs, h2, n_agents, obs_shape]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"/home/offpymarl/dataset_analysis/diffusion/{timestamp}"
            os.makedirs(save_dir, exist_ok=True)

            obs_target = obs_target[:, args.history_horizon:] # [bs, h2, n_agents, obs_shape]
            bs, h, n_agents, obs_shape = obs_target.shape
            
            obs_target = obs_target.reshape(bs * h, -1).cpu()
            obs_target_unnorm = dataset.normalizer.unnormalize(obs_target, 'observations')
            obs_target_unnorm = obs_target_unnorm.reshape(bs, h, n_agents, obs_shape)
            obs_gen_unnorm = samples.reshape(bs * h, -1).cpu()
            obs_gen_unnorm = dataset.normalizer.unnormalize(obs_gen_unnorm, 'observations')
            obs_gen_unnorm = obs_gen_unnorm.reshape(bs, h, n_agents, obs_shape)
            
            th.save({
                'obs_target': obs_target_unnorm,
                'obs_gen': obs_gen_unnorm,     
                'conditions': {k: v.cpu() if isinstance(v, th.Tensor) else v for k, v in conditions.items()},
                'returns': returns.cpu() if isinstance(returns, th.Tensor) else returns,
                'attention_masks': attention_masks.cpu()
            }, os.path.join(save_dir, 'diffusion_outputs.pt'))

            print(f"Saved analysis data to {os.path.join(save_dir, 'diffusion_outputs.pt')}")
        
        if syn_data:
            normed_observations, actions, rewards = trajectory_stitching_parallel(args, dataset, agent)

            if not dataset.discrete_action and dataset.use_action:
                actions = dataset.normalizer.unnormalize(actions, "actions")
            observations = dataset.normalizer.unnormalize(normed_observations, "observations")
            n_agents = observations.shape[-2]
            observations = observations[..., : - n_agents]

            save_traj_dir = os.path.join(args.results_save_dir, 'syn')

            if not os.path.exists(save_traj_dir):
                os.makedirs(save_traj_dir)

            np.save(os.path.join(save_traj_dir, "obs.npy"), observations) # [bs, seq_len, n_agents, obs_shape]
            np.save(os.path.join(save_traj_dir, "acs.npy"), actions) # [bs, seq_len, n_agents, 1]
            np.save(os.path.join(save_traj_dir, "rew.npy"), rewards) # [bs, seq_len, n_agents, 1]
            # if self.dataset.include_states:
            #     np.save(os.path.join(save_traj_dir, "state.npy"), states)
        
        if analyze_data or syn_data:
            return
    
    logger.console_logger.info("Beginning  offline training with {} iterations".format(args.t_max))
    train_sequential(args, logger, learner, runner, offline_buffer)

    if args.save_model:
        save_path = os.path.join(args.model_save_dir, str(args.t_max))
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Save final model checkpoint in {}".format(save_path))
        learner.save_models(save_path)
    
    runner.close_env()
    logger.console_logger.info("Finish Training")

def to_device_recursive(data, device):
    if isinstance(data, dict):
        return {k: to_device_recursive(v, device) for k, v in data.items()}
    elif isinstance(data, th.Tensor):
        return data.to(device)
    else:
        return data

def train_sequential(args, logger, learner, runner, offline_buffer):
    t_env = 0
    episode = 0
    t_max = args.t_max
    model_save_time = 0
    last_test_T = 0
    last_log_T = 0
    start_time = time.time()
    last_time = start_time
    test_time_total = 0

    batch_size_train = args.offline_batch_size
    batch_size_run = args.batch_size_run # num of parellel envs
    n_test_runs = max(1, args.test_nepisode//batch_size_run)
    test_start_time = time.time()

    test_time_total += time.time() - test_start_time

    while t_env < t_max:
        episode_sample = offline_buffer.sample(batch_size_train)
        if isinstance(episode_sample, dict):
            episode_sample = to_device_recursive(episode_sample, args.device)
        else: # EpisodeBatch
            if episode_sample.device != args.device:
                episode_sample.to(args.device)
   
        learner.train(episode_sample, t_env, episode)
        
        t_env += 1
        episode += batch_size_run
        # Execute test runs once in a while & final evaluation
        if ((t_env - last_test_T) / args.test_interval >= 1 or t_env >= t_max):
            test_start_time = time.time()
            test_time_total += time.time() - test_start_time

            logger.console_logger.info("Step: {}/{}".format(t_env, t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}. FPS {:.2f}. Test time cost: {}".format(
                time_left(last_time, last_test_T, t_env, t_max), time_str(time.time() - start_time), (t_env - last_test_T) / (time.time() - last_time), time_str(test_time_total)
            ))
            last_time = time.time()
            last_test_T = t_env
        
        if args.save_model and (t_env-model_save_time >= args.save_model_interval or model_save_time==0):
            save_path = os.path.join(args.model_save_dir, str(t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            learner.save_models(save_path)
            model_save_time = t_env
        
        if (t_env - last_log_T) >= args.log_interval:
            last_log_T = t_env
            logger.log_stat("episode", episode, t_env)
            logger.print_recent_stats()

        
    if args.save_model:
        save_path = os.path.join(args.model_save_dir, str(t_env))
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving models to {}".format(save_path))
        learner.save_models(save_path)
        model_save_time = t_env

    logger.console_logger.info("Finish training sequential")
            
      
def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config