import collections
import os
from typing import Callable, List, Optional

import gym
import numpy as np
import torch
from ddpg_agent import DDPGAgent

import h5py


class StackWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_n = np.array(obs)
        reward_n = np.array(reward)
        done_n = np.array(done)
        return obs_n, reward_n, done_n, info

    def reset(self):
        obs = self.env.reset()
        obs_n = np.array(obs)
        return obs_n


class PretrainedPreyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, scenario_name: str):
        assert scenario_name in ["simple_tag", "simple_world"], scenario_name
        # XXX: Pass in `device` as an argument?
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prey = DDPGAgent(
            num_in_pol=env.observation_space[-1].shape[0],
            num_out_pol=2,
            num_in_critic=env.observation_space[-1].shape[0]
            + 2,
        )
        self.prey.to(self.device)

        load_path = os.path.join(
            os.path.dirname(__file__),
            "data/mpe",
            scenario_name,
            "pretrained_adv_model.pt",
        )
        prey_params = torch.load(load_path, map_location=self.device)["agent_params"][
            -1
        ]
        self.prey.load_params_without_optims(prey_params)
        self.prey.policy.eval()
        self.prey.target_policy.eval()

        super().__init__(env)

        self.prey_obs = None
        # rewrite env attributes to remove prey
        self.n = env.n - 1
        self.action_space = env.action_space[:-1]
        self.observation_space = env.observation_space[:-1]

    def step(self, action):
        prey_obs = torch.tensor(
            self.prey_obs, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        prey_action = self.prey.step(prey_obs, explore=False)[0].detach().cpu().numpy()
        action = [*action, prey_action]
        obs, reward, done, info = self.env.step(action)
        self.prey_obs = obs[-1]
        return obs[:-1], reward[:-1], done[:-1], info

    def reset(self):
        obs = self.env.reset()
        self.prey_obs = obs[-1]
        return obs[:-1]


def make_env(scenario_name, benchmark=False, **kwargs):
    """
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    """
    import multiagent.scenarios as scenarios
    from multiagent.environment import MultiAgentEnv

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.benchmark_data,
            **kwargs,
        )
    else:
        env = MultiAgentEnv(
            world, scenario.reset_world, scenario.reward, scenario.observation, **kwargs
        )
    if scenario_name in ["simple_tag", "simple_world"]:
        env = PretrainedPreyWrapper(env, scenario_name)
    return StackWrapper(env)


def load_environment(name, **kwargs):
    if type(name) != str:
        # name is already an environment
        return name

    idx = name.find("-")
    env_name, data_split = name[:idx], name[idx + 1 :]

    env = make_env(env_name, discrete_action=True, **kwargs)
    if hasattr(env, "metadata"):
        assert isinstance(env.metadata, dict)
    else:
        env.metadata = {}
    env.metadata["data_split"] = data_split
    env.metadata["name"] = env_name
    env.metadata["global_feats"] = []
    return env


def sequence_dataset(env, syn_dataset_dir, preprocess_fn: List[Callable] = [], seed: int = None):
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "data/grid_mpe",
        env.metadata["name"],
        env.metadata["data_split"],
    )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset directory not found: {}".format(dataset_path))

    def _read_data(h5_path, max_data_size, shuffle):
        data = {}
        with h5py.File(h5_path, 'r') as f:
            for k in f.keys():
                added_data = f[k][:]
                if k not in data:
                    data[k] = added_data
                else:
                    data[k] = np.concatenate((data[k], added_data), axis=0)
        if not shuffle and data[list(data.keys())[0]].shape[0] > max_data_size:
            data = {k: v[-max_data_size:] for k, v in data.items()}

        keys = list(data.keys())
        original_data_size = data[keys[0]].shape[0]
        data_size = min(original_data_size, max_data_size)

        if shuffle:
            shuffled_idx = np.random.choice(original_data_size, data_size, replace=False)
            data = {k: v[shuffled_idx] for k, v in data.items()}
        return data

    if not isinstance(dataset_path, list) and os.path.exists(dataset_path):
        data_path_list = [dataset_path] 
    else:
        data_path_list = dataset_path

    h5_paths = []
    for final_data_path in data_path_list:
        h5_paths.extend([os.path.join(final_data_path, f) for f in sorted(os.listdir(final_data_path)) if f.endswith(".h5")])
    # print(h5_paths)
    
    max_buffer_size = 100000
    shuffle = True
    max_data_size_per_file = max_buffer_size // len(h5_paths)
    dataset = [_read_data(h5_paths[i], max_data_size_per_file, shuffle) for i in range(len(h5_paths))]
    data = {k: np.concatenate([v[k] for v in dataset], axis=0) for k in dataset[0].keys()}
    # print(self.data)
    keys = list(data.keys())
    buffer_size = data[keys[0]].shape[0]

    if shuffle:
        shuffled_idx = np.random.choice(buffer_size, buffer_size, replace=False)
        data = {k: v[shuffled_idx] for k, v in data.items()}

    for i in range(data['actions'].shape[0]):
        episode_data = {}
        terminal_idx = np.nonzero(data['terminated'][i])[0][0]
        episode_data["terminals"] = np.repeat(data['terminated'][i][:terminal_idx + 1], env.n).reshape(terminal_idx + 1, env.n)
        # episode_data["states"] = data['state'][i][:terminal_idx + 1]
        episode_data["observations"] = data['obs'][i][:terminal_idx + 1]  # (50, 2, 15)
        episode_data["legal_actions"] = data['avail_actions'][i][:terminal_idx + 1]
        episode_data["rewards"] = np.expand_dims(data['reward'][i][:terminal_idx + 1], axis=-1)
        episode_data["actions"] = data['actions'][i][:terminal_idx + 1]  # (50, 2, 1)
        yield episode_data



if __name__ == "__main__":
    env = make_env("simple_tag", discrete_action=True)

    obs = env.reset()
    for _ in range(5):
        obs, reward, done, info = env.step(
            [act_space.sample() for act_space in env.action_space]
        )
