import os
from typing import Any, Dict, List, Optional, Callable

import gym
import numpy as np
from smac.env import StarCraft2Env

import h5py


import torch
import torch.nn as nn
import torch.optim as optim

class SMAC(gym.Env):
    """Environment wrapper SMAC."""

    metadata = {}

    def __init__(self, map_name: str, add_agent_ids_to_obs: bool = True):
        self._environment = StarCraft2Env(map_name=map_name, obs_last_action=False)
        self._agents = [f"agent_{n}" for n in range(self._environment.n_agents)]
        self.num_agents = len(self._agents)
        self.num_actions = self._environment.n_actions
        self._done = False
        self.max_episode_length = self._environment.episode_limit
        self.add_agent_ids_to_obs = add_agent_ids_to_obs

        if add_agent_ids_to_obs:
            self.one_hot_agent_ids = []
            for i in range(self.num_agents):
                agent_id = np.eye(self.num_agents)[i]
                self.one_hot_agent_ids.append(agent_id)
            self.one_hot_agent_ids = np.stack(self.one_hot_agent_ids, axis=0)

        self.observation_space = [
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    self._environment.get_obs_size() + self.num_agents
                    if add_agent_ids_to_obs
                    else self._environment.get_obs_size(),
                ),
            )
            for _ in range(self.num_agents)
        ]
        self.action_space = [
            gym.spaces.Discrete(n=self.num_actions) for _ in range(self.num_agents)
        ]

    def reset(self):
        """Resets the env."""

        # Reset the environment
        self._environment.reset()
        self._done = False

        observation = np.array(self.environment.get_obs())
        if self.add_agent_ids_to_obs:
            observation = np.concatenate([self.one_hot_agent_ids, observation], axis=1)
        return observation

    def step(self, actions: np.ndarray):
        """Steps in env."""

        # Step the SMAC environment
        reward, self._done, self._info = self._environment.step(actions)
        reward_n = np.array([reward for _ in range(self.num_agents)])
        done_n = np.array([self._done for _ in range(self.num_agents)])

        # Get the next observation
        next_observation = np.array(self._environment.get_obs())
        if self.add_agent_ids_to_obs:
            next_observation = np.concatenate(
                [self.one_hot_agent_ids, next_observation], axis=1
            )
        return next_observation, reward_n, done_n, self._info

    def env_done(self) -> bool:
        """Check if env is done."""
        return self._done

    def get_legal_actions(self) -> List:
        """Get legal actions from the environment."""
        legal_actions = []
        for i, _ in enumerate(self._agents):
            legal_actions.append(
                np.array(self._environment.get_avail_agent_actions(i), dtype="float32")
            )
        return np.array(legal_actions)

    def get_stats(self) -> Optional[Dict]:
        """Return extra stats to be logged."""
        return self._environment.get_stats()

    @property
    def agents(self) -> List:
        """Agents still alive in env (not done)."""
        return self._agents

    @property
    def possible_agents(self) -> List:
        """All possible agents in env."""
        return self._agents

    @property
    def environment(self):
        """Returns the wrapped environment."""
        return self._environment

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)


def load_environment(name, **kwargs):
    if type(name) is not str:
        # name is already an environment
        return name

    idx = name.find("-")
    env_name, data_split = name[:idx], name[idx + 1 :]

    env = SMAC(env_name, **kwargs)
    if hasattr(env, "metadata"):
        assert isinstance(env.metadata, dict)
    else:
        env.metadata = {}
    env.metadata["data_split"] = data_split
    env.metadata["name"] = env_name
    env.metadata["global_feats"] = ["states"]
    return env


def sequence_dataset(env, syn_dataset_dir, preprocess_fn: List[Callable] = []):
    """dataset_path = os.path.join(
        os.path.dirname(__file__),
        "data/smac",
        env.metadata["name"],
        env.metadata["data_split"],
    )"""
    dataset_path = syn_dataset_dir
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
    keys = list(data.keys())
    buffer_size = data[keys[0]].shape[0]

    if shuffle:
        shuffled_idx = np.random.choice(buffer_size, buffer_size, replace=False)
        data = {k: v[shuffled_idx] for k, v in data.items()}
    
    for i in range(data['actions'].shape[0]):
        episode_data = {}
        terminal_idx = np.nonzero(data['terminated'][i])[0][0]
        episode_data["terminals"] = np.repeat(data['terminated'][i][:terminal_idx + 1], env.num_agents).reshape(terminal_idx + 1, env.num_agents)
        episode_data["states"] = data['state'][i][:terminal_idx + 1]
        episode_data["observations"] = data['obs'][i][:terminal_idx + 1]  # (50, 2, 15)
        episode_data["legal_actions"] = data['avail_actions'][i][:terminal_idx + 1]
        episode_data["rewards"] = np.expand_dims(data['reward'][i][:terminal_idx + 1], axis=-1) # [seq_len, 1, 1]
        episode_data["actions"] = data['actions'][i][:terminal_idx + 1]  # (50, 2, 1)
        episode_data["labels"] = np.repeat(data['random_labels'][i][np.newaxis, :, np.newaxis], terminal_idx + 1, axis=0)
        yield episode_data
