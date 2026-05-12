import importlib
from typing import Callable, List, Optional

import numpy as np
import torch

from diffuser.datasets.buffer import ReplayBuffer
from diffuser.datasets.normalization import DatasetNormalizer
from diffuser.datasets.preprocessing import get_preprocess_fn
from diffuser.utils.mask_generator import MultiAgentMaskGenerator
from diffuser.utils.arrays import array_circular_shift
import torch.nn as nn
import os


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        env_type: str = "d4rl",
        env: str = "hopper-medium-replay",
        syn_dataset_dir: str = "",
        n_agents: int = 2,
        horizon: int = 64,
        normalizer: str = "LimitsNormalizer",
        preprocess_fns: List[Callable] = [],
        use_action: bool = True,
        discrete_action: bool = False,
        max_path_length: int = 1000,
        max_n_episodes: int = 10000,
        termination_penalty: float = 0,
        use_padding: bool = True,  # when path_length is smaller than max_path_length
        discount: float = 0.99,
        returns_scale: float = 400.0,
        include_returns: bool = False,
        include_env_ts: bool = False,
        include_states: bool = False,
        history_horizon: int = 0,
        agent_share_parameters: bool = False,
        use_seed_dataset: bool = False,
        decentralized_execution: bool = False,
        use_inv_dyn: bool = True,
        use_zero_padding: bool = True,
        agent_condition_type: str = "all",  # single
        pred_future_padding: bool = False,
        seed: Optional[int] = None,
    ):
        if use_seed_dataset:
            assert (
                env_type == "mpe"
            ), f"Seed dataset only supported for MPE, not {env_type}"

        assert agent_condition_type in ["single", "all", "random"], agent_condition_type
        self.agent_condition_type = agent_condition_type

        env_mod_name = {
            "mpe": "diffuser.datasets.mpe",
            "grid_mpe": "diffuser.datasets.grid_mpe",
            "smac": "diffuser.datasets.smac_env",
        }[env_type]
        env_mod = importlib.import_module(env_mod_name)

        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = env_mod.load_environment(env)
        self.global_feats = env.metadata["global_feats"]

        self.use_inv_dyn = use_inv_dyn
        self.returns_scale = returns_scale
        self.n_agents = n_agents
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None, None]
        self.use_padding = use_padding
        self.use_action = use_action
        self.discrete_action = discrete_action
        self.include_returns = include_returns
        self.include_env_ts = include_env_ts
        self.include_states = include_states
        self.decentralized_execution = decentralized_execution
        self.use_zero_padding = use_zero_padding
        self.pred_future_padding = pred_future_padding

        if env_type == "mpe":
            if use_seed_dataset:
                itr = env_mod.sequence_dataset(env, syn_dataset_dir, self.preprocess_fn, seed=seed)
            else:
                itr = env_mod.sequence_dataset(env, syn_dataset_dir, self.preprocess_fn)
        elif env_type == "grid_mpe":
            itr = env_mod.sequence_dataset(env, syn_dataset_dir, self.preprocess_fn)
        elif env_type == "smac" or env_type == "smacv2":
            itr = env_mod.sequence_dataset(env, syn_dataset_dir, self.preprocess_fn)
        else:
            itr = env_mod.sequence_dataset(env, syn_dataset_dir, self.preprocess_fn)

        fields = ReplayBuffer(
            n_agents,
            max_n_episodes,
            max_path_length,
            termination_penalty,
            global_feats=self.global_feats,
            use_zero_padding=self.use_zero_padding,
        )
        for _, episode in enumerate(itr):
            # concatenate global_feats to individual observations
            if self.include_states:
                assert self.global_feats == ["states"]
                for global_key in self.global_feats:
                    episode["observations"] = np.concatenate([episode["observations"], np.stack([episode[global_key]] * episode["observations"].shape[1], axis=1)], axis=2)
                # print(episode["observations"].shape)  # (H, 3, 30)
                # print(episode["states"].shape)  # (H, 48)
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(
            fields,
            normalizer,
            path_lengths=fields["path_lengths"],
            agent_share_parameters=agent_share_parameters,
            global_feats=self.global_feats,
        )

        self.observation_dim = fields.observations.shape[-1]
        if self.include_states:
            self.state_dim = fields.states.shape[-1]
        self.action_dim = fields.actions.shape[-1] if self.use_action else 0
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths

        self.indices = self.make_indices(fields.path_lengths)
        self.mask_generator = MultiAgentMaskGenerator(
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
            history_horizon=self.history_horizon,
            action_visible=not use_inv_dyn,
        )

        if self.discrete_action:
            # smac has discrete actions, so we only need to normalize observations
            self.normalize(["observations"])
        else:
            self.normalize()

        self.pad_future()
        if self.history_horizon > 0:
            self.pad_history()

        print(fields)

    def pad_future(self, keys: List[str] = None):
        if keys is None:
            keys = ["normed_observations", "rewards", "terminals"]
            if "legal_actions" in self.fields.keys:
                keys.append("legal_actions")
            if self.use_action:
                if self.discrete_action:
                    keys.append("actions")
                else:
                    keys.append("normed_actions")

        for key in keys:
            shape = self.fields[key].shape
            if self.use_zero_padding:
                self.fields[key] = np.concatenate(
                    [
                        self.fields[key],
                        np.zeros(
                            (shape[0], self.horizon - 1, *shape[2:]),
                            dtype=self.fields[key].dtype,
                        ),
                    ],
                    axis=1,
                )
            else:
                self.fields[key] = np.concatenate(
                    [
                        self.fields[key],
                        np.repeat(
                            self.fields[key][:, -1:],
                            self.horizon - 1,
                            axis=1,
                        ),
                    ],
                    axis=1,
                )

    def pad_history(self, keys: List[str] = None):
        if keys is None:
            keys = ["normed_observations", "rewards", "terminals"]
            if "legal_actions" in self.fields.keys:
                keys.append("legal_actions")
            if self.use_action:
                if self.discrete_action:
                    keys.append("actions")
                else:
                    keys.append("normed_actions")

        for key in keys:
            shape = self.fields[key].shape
            if self.use_zero_padding:
                self.fields[key] = np.concatenate(
                    [
                        np.zeros(
                            (shape[0], self.history_horizon, *shape[2:]),
                            dtype=self.fields[key].dtype,
                        ),
                        self.fields[key],
                    ],
                    axis=1,
                )
            else:
                self.fields[key] = np.concatenate(
                    [
                        np.repeat(
                            self.fields[key][:, :1],
                            self.history_horizon,
                            axis=1,
                        ),
                        self.fields[key],
                    ],
                    axis=1,
                )

    def normalize(self, keys: List[str] = None):
        """
        normalize fields that will be predicted by the diffusion model
        """
        if keys is None:
            keys = ["observations", "actions"] if self.use_action else ["observations"]

        for key in keys:
            shape = self.fields[key].shape
            array = self.fields[key].reshape(shape[0] * shape[1], *shape[2:])
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(shape)

    def make_indices(self, path_lengths: np.ndarray):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """

        indices = []
        for i, path_length in enumerate(path_lengths):
            if self.use_padding:
                max_start = path_length - 1
            else:
                max_start = path_length - self.horizon
                if max_start < 0:
                    continue

            # get `end` and `mask_end` for each `start`
            for start in range(max_start):
                end = start + self.horizon
                mask_end = min(end, path_length)
                indices.append((i, start, end, mask_end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations: np.ndarray, agent_idx: Optional[int] = None):
        """
        condition on current observations for planning
        """

        ret_dict = {}
        # if self.decentralized_execution:
        if self.agent_condition_type == "single":
            cond_observations = np.zeros_like(observations[: self.history_horizon + 1])
            cond_observations[:, agent_idx] = observations[
                : self.history_horizon + 1, agent_idx
            ]
            ret_dict["agent_idx"] = torch.LongTensor([[agent_idx]])
        elif self.agent_condition_type == "all":
            cond_observations = observations[: self.history_horizon + 1]
        ret_dict[(0, self.history_horizon + 1)] = cond_observations
        return ret_dict

    def __len__(self):
        if self.agent_condition_type == "single":
            return len(self.indices) * self.n_agents
        else:
            return len(self.indices)

    def __getitem__(self, idx: int, agent_idx: Optional[int] = None):
        if self.agent_condition_type == "single":
            path_ind, start, end, mask_end = self.indices[idx // self.n_agents]
            agent_mask = np.zeros(self.n_agents, dtype=bool)
            agent_mask[idx % self.n_agents] = 1
        elif self.agent_condition_type == "all":
            path_ind, start, end, mask_end = self.indices[idx]
            agent_mask = np.ones(self.n_agents, dtype=bool)
        elif self.agent_condition_type == "random":
            path_ind, start, end, mask_end = self.indices[idx]
            # randomly generate 0 or 1 agent_masks
            agent_mask = np.random.randint(0, 2, self.n_agents, dtype=bool)

        # shift by `self.history_horizon`
        history_start = start
        start = history_start + self.history_horizon
        end = end + self.history_horizon
        mask_end = mask_end + self.history_horizon

        observations = self.fields.normed_observations[path_ind, history_start:end]
        if self.use_action:
            if self.discrete_action:
                actions = self.fields.actions[path_ind, history_start:end]
            else:
                actions = self.fields.normed_actions[path_ind, history_start:end]

        if self.use_action:
            trajectories = np.concatenate([actions, observations], axis=-1)
        else:
            trajectories = observations

        if self.use_inv_dyn:
            cond_masks = self.mask_generator(observations.shape, agent_mask)
            cond_trajectories = observations.copy()
        else:
            cond_masks = self.mask_generator(trajectories.shape, agent_mask)
            cond_trajectories = trajectories.copy()
        cond_trajectories[: self.history_horizon, ~agent_mask] = 0.0
        cond = {
            "x": cond_trajectories,
            "masks": cond_masks,
        }

        loss_masks = np.zeros((observations.shape[0], observations.shape[1], 1))
        if self.pred_future_padding:
            loss_masks[self.history_horizon :] = 1.0
        else:
            loss_masks[self.history_horizon : mask_end - history_start] = 1.0
        if self.use_inv_dyn:
            loss_masks[self.history_horizon, agent_mask] = 0.0

        attention_masks = np.zeros((observations.shape[0], observations.shape[1], 1))
        attention_masks[self.history_horizon : mask_end - history_start] = 1.0
        attention_masks[: self.history_horizon, agent_mask] = 1.0

        batch = {
            "x": trajectories,
            "cond": cond,
            "loss_masks": loss_masks,
            "attention_masks": attention_masks,
        }

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start : -self.horizon + 1]
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum(axis=0).squeeze(-1)
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch["returns"] = returns

        if self.include_env_ts:
            env_ts = (
                np.arange(history_start, start + self.horizon) - self.history_horizon
            )
            env_ts[np.where(env_ts < 0)] = self.max_path_length
            env_ts[np.where(env_ts >= self.max_path_length)] = self.max_path_length
            batch["env_ts"] = env_ts

        if "legal_actions" in self.fields.keys:
            batch["legal_actions"] = self.fields.legal_actions[
                path_ind, history_start:end
            ]

        return batch
    

class SequenceAugDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        env_type: str = "d4rl",
        env: str = "hopper-medium-replay",
        syn_dataset_dir: str = "",
        n_agents: int = 2,
        horizon: int = 64,
        normalizer: str = "LimitsNormalizer",
        preprocess_fns: List[Callable] = [],
        use_action: bool = True,
        discrete_action: bool = False,
        max_path_length: int = 1000,
        max_n_episodes: int = 10000,
        termination_penalty: float = 0,
        use_padding: bool = True,  # when path_length is smaller than max_path_length
        discount: float = 0.99,
        returns_scale: float = 400.0,
        include_returns: bool = False,
        include_rewards: bool = False,
        include_labels: bool = False,
        include_env_ts: bool = False,
        include_states: bool = False,
        history_horizon: int = 0,
        agent_share_parameters: bool = False,
        use_seed_dataset: bool = False,
        decentralized_execution: bool = False,
        use_inv_dyn: bool = True,
        use_zero_padding: bool = True,
        agent_condition_type: str = "all",  # single
        pred_future_padding: bool = False,
        circular_shift: bool = False,
        shift_ratio: float = 0.1,
        seed: Optional[int] = None,
    ):
        if use_seed_dataset:
            assert (
                env_type == "mpe"
            ), f"Seed dataset only supported for MPE, not {env_type}"

        assert agent_condition_type in ["single", "all", "random"], agent_condition_type
        self.agent_condition_type = agent_condition_type

        env_mod_name = {
            "lbf": "diffuser.datasets.lbf",
            "mpe": "diffuser.datasets.mpe",
            "grid_mpe": "diffuser.datasets.grid_mpe",
            "smac": "diffuser.datasets.smac_env",
            "smacv2": "diffuser.datasets.smacv2_env",
        }[env_type]
        env_mod = importlib.import_module(env_mod_name)

        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = env_mod.load_environment(env)
        self.global_feats = env.metadata["global_feats"]

        self.use_inv_dyn = use_inv_dyn
        self.returns_scale = returns_scale
        self.n_agents = n_agents
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None, None]
        self.use_padding = use_padding
        self.use_action = use_action
        self.discrete_action = discrete_action
        self.include_returns = include_returns
        self.include_rewards = include_rewards
        self.include_labels = include_labels
        self.include_env_ts = include_env_ts
        self.include_states = include_states
        self.decentralized_execution = decentralized_execution
        self.use_zero_padding = use_zero_padding
        self.pred_future_padding = pred_future_padding
        self.circular_shift = circular_shift
        self.shift_ratio = shift_ratio

        if env_type == "mpe":
            if use_seed_dataset:
                itr = env_mod.sequence_dataset(env, syn_dataset_dir, self.preprocess_fn, seed=seed)
            else:
                itr = env_mod.sequence_dataset(env, syn_dataset_dir, self.preprocess_fn)
        else:
            itr = env_mod.sequence_dataset(env, syn_dataset_dir, self.preprocess_fn)

        fields = ReplayBuffer(
            n_agents,
            max_n_episodes,
            max_path_length,
            termination_penalty,
            global_feats=self.global_feats,
            use_zero_padding=self.use_zero_padding,
        )
        for _, episode in enumerate(itr):
            # concatenate global_feats to individual observations
            if self.include_states:
                assert self.global_feats == ["states"]
                for global_key in self.global_feats:
                    episode["observations"] = np.concatenate([episode["observations"], np.stack([episode[global_key]] * episode["observations"].shape[1], axis=1)], axis=2)
                # print(episode["observations"].shape)  # (H, 3, 30)
                # print(episode["states"].shape)  # (H, 48)
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(
            fields,
            normalizer,
            path_lengths=fields["path_lengths"],
            agent_share_parameters=agent_share_parameters,
            global_feats=self.global_feats,
        )

        self.observation_dim = fields.observations.shape[-1]
        if self.include_states:
            self.state_dim = fields.states.shape[-1]
        self.action_dim = fields.actions.shape[-1] if self.use_action else 0
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths

        if self.circular_shift:
            self.agent_shift = self.find_rewarding_traj_seg()
        self.indices = self.make_indices(fields.path_lengths)
        self.mask_generator = MultiAgentMaskGenerator(
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
            history_horizon=self.history_horizon,
            action_visible=not use_inv_dyn,
        )

        if self.discrete_action:
            # smac has discrete actions, so we only need to normalize observations
            self.normalize(["observations"])
        else:
            self.normalize()

        self.pad_future()
        if self.history_horizon > 0:
            self.pad_history()

        print(fields)
        '''
        spread:
            observations: (20000, 26, 3, 17)
            normed_observations: (20000, 33, 3, 17)
            actions: (20000, 26, 3, 1)
            normed_actions: (20000, 33, 3, 1)
            rewards: (20000, 33, 3, 1)
            terminals: (20000, 33, 3, 1)
            timeouts: (20000, 26, 3, 1)
        3m:
            observations: (20000, 60, 3, 78)
            normed_observations: (20000, 67, 3, 78)
            actions: (20000, 67, 3, 1)
            rewards: (20000, 67, 3, 1)
            terminals: (20000, 67, 3, 1)
            legal_actions: (20000, 67, 3, 9)
            states: (20000, 60, 48)
        '''
        
    def pad_future(self, keys: List[str] = None):
        if keys is None:
            keys = ["normed_observations", "rewards", "terminals"]
            if "legal_actions" in self.fields.keys:
                keys.append("legal_actions")
            if "labels" in self.fields.keys:
                keys.append("labels")
            if self.use_action:
                if self.discrete_action:
                    keys.append("actions")
                else:
                    keys.append("normed_actions")

        for key in keys:
            shape = self.fields[key].shape
            if self.use_zero_padding:
                self.fields[key] = np.concatenate(
                    [
                        self.fields[key],
                        np.zeros(
                            (shape[0], self.horizon - 1, *shape[2:]),
                            dtype=self.fields[key].dtype,
                        ),
                    ],
                    axis=1,
                )
            else:
                self.fields[key] = np.concatenate(
                    [
                        self.fields[key],
                        np.repeat(
                            self.fields[key][:, -1:],
                            self.horizon - 1,
                            axis=1,
                        ),
                    ],
                    axis=1,
                )

    def pad_history(self, keys: List[str] = None):
        if keys is None:
            keys = ["normed_observations", "rewards", "terminals"]
            if "legal_actions" in self.fields.keys:
                keys.append("legal_actions")
            if "labels" in self.fields.keys:
                keys.append("labels")
            if self.use_action:
                if self.discrete_action:
                    keys.append("actions")
                else:
                    keys.append("normed_actions")

        for key in keys:
            shape = self.fields[key].shape
            if self.use_zero_padding:
                self.fields[key] = np.concatenate(
                    [
                        np.zeros(
                            (shape[0], self.history_horizon, *shape[2:]),
                            dtype=self.fields[key].dtype,
                        ),
                        self.fields[key],
                    ],
                    axis=1,
                )
            else:
                self.fields[key] = np.concatenate(
                    [
                        np.repeat(
                            self.fields[key][:, :1],
                            self.history_horizon,
                            axis=1,
                        ),
                        self.fields[key],
                    ],
                    axis=1,
                )

    def normalize(self, keys: List[str] = None):
        """
        normalize fields that will be predicted by the diffusion model
        """
        if keys is None:
            keys = ["observations", "actions"] if self.use_action else ["observations"]

        for key in keys:
            shape = self.fields[key].shape
            array = self.fields[key].reshape(shape[0] * shape[1], *shape[2:])
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(shape)

    def find_rewarding_traj_seg(self):
        """
        return a table of binary indicating whether a traj_seg is of high quality
        """
        # self.discounts: (self.max_path_length, 1, 1), self.fields.rewards: (n_traj, self.max_path_length + self.horizon - 1, n_agent, 1)
        rewards = self.fields.rewards[:, :, 0, 0]  # (n_traj, self.max_path_length)
        discounts = self.discounts.reshape(-1)[None, :]  # (1, self.max_path_length)
        discounted_rewards = np.hstack((rewards, np.zeros((rewards.shape[0], self.horizon - 1))))
        discounted_rewards[:, :self.max_path_length] = discounts * rewards[:, :self.max_path_length]  # (n_traj, self.max_path_length + self.horizon - 1)
        returns = np.sum([discounted_rewards[:, i: i + self.max_path_length] for i in range(self.horizon)], axis=0)  # (n_traj, self.max_path_length)

        shift_threshold = np.percentile(returns, (1 - self.shift_ratio) * 100, axis=0)
        return returns > shift_threshold

    def make_indices(self, path_lengths: np.ndarray):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """

        indices = []
        for i, path_length in enumerate(path_lengths):
            if self.use_padding:
                max_start = path_length - 1
            else:
                max_start = path_length - self.horizon
                if max_start < 0:
                    continue

            # get `end` and `mask_end` for each `start`
            # for start in range(max_start):
            #     end = start + self.horizon
            #     mask_end = min(end, path_length)
            #     indices.append((i, start, end, mask_end))

            for start in range(0, max_start + 1, 1):  # temporary
                end = start + self.horizon
                mask_end = min(end, path_length)
                indices.append((i, start, end, mask_end, 0))
                if self.circular_shift and self.agent_shift[i][start]:
                    for shift_stride in range(1, self.n_agents):
                        indices.append((i, start, end, mask_end, shift_stride))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations: np.ndarray, agent_idx: Optional[int] = None):
        """
        condition on current observations for planning
        """

        ret_dict = {}
        # if self.decentralized_execution:
        if self.agent_condition_type == "single":
            cond_observations = np.zeros_like(observations[: self.history_horizon + 1])
            cond_observations[:, agent_idx] = observations[
                : self.history_horizon + 1, agent_idx
            ]
            ret_dict["agent_idx"] = torch.LongTensor([[agent_idx]])
        elif self.agent_condition_type == "all":
            cond_observations = observations[: self.history_horizon + 1]
        ret_dict[(0, self.history_horizon + 1)] = cond_observations
        return ret_dict

    def __len__(self):
        if self.agent_condition_type == "single":
            return len(self.indices) * self.n_agents
        else:
            return len(self.indices)

    def __getitem__(self, idx: int, agent_idx: Optional[int] = None):
        if self.agent_condition_type == "single":
            path_ind, start, end, mask_end, shift_stride = self.indices[idx // self.n_agents]
            agent_mask = np.zeros(self.n_agents, dtype=bool)
            agent_mask[idx % self.n_agents] = 1
        elif self.agent_condition_type == "all":
            path_ind, start, end, mask_end, shift_stride = self.indices[idx]
            agent_mask = np.ones(self.n_agents, dtype=bool)
        elif self.agent_condition_type == "random":
            path_ind, start, end, mask_end, shift_stride = self.indices[idx]
            # randomly generate 0 or 1 agent_masks
            agent_mask = np.random.randint(0, 2, self.n_agents, dtype=bool)

        # shift by `self.history_horizon`
        history_start = start
        start = history_start + self.history_horizon
        end = end + self.history_horizon
        mask_end = mask_end + self.history_horizon
        
        observations = self.fields.normed_observations[path_ind, history_start:end]
        observations = array_circular_shift(observations, 1, shift_stride)
        if self.use_action:
            if self.discrete_action:
                actions = self.fields.actions[path_ind, history_start:end]
            else:
                actions = self.fields.normed_actions[path_ind, history_start:end]
            actions = array_circular_shift(actions, 1, shift_stride)

        if self.use_action:
            trajectories = np.concatenate([actions, observations], axis=-1)
        else:
            trajectories = observations

        if self.use_inv_dyn:
            cond_masks = self.mask_generator(observations.shape, agent_mask)
            cond_trajectories = observations.copy()
        else:
            cond_masks = self.mask_generator(trajectories.shape, agent_mask)
            cond_trajectories = trajectories.copy()
        cond_trajectories[: self.history_horizon, ~agent_mask] = 0.0
        cond = {
            "x": cond_trajectories,
            "masks": cond_masks,
        }

        loss_masks = np.zeros((observations.shape[0], observations.shape[1], 1))
        if self.pred_future_padding:
            loss_masks[self.history_horizon :] = 1.0
        else:
            loss_masks[self.history_horizon : mask_end - history_start] = 1.0
        if self.use_inv_dyn:
            loss_masks[self.history_horizon, agent_mask] = 0.0
            
        loss_masks_world_models = np.zeros((observations.shape[0], observations.shape[1], 1))
        loss_masks_world_models[self.history_horizon : mask_end - history_start] = 1.0
        if self.use_inv_dyn:
            loss_masks_world_models[self.history_horizon, agent_mask] = 0.0

        attention_masks = np.zeros((observations.shape[0], observations.shape[1], 1))
        attention_masks[self.history_horizon : mask_end - history_start] = 1.0
        attention_masks[: self.history_horizon, agent_mask] = 1.0

        batch = {
            "x": trajectories,
            "cond": cond,
            "loss_masks": loss_masks,
            "loss_masks_world_models": loss_masks_world_models,
            "attention_masks": attention_masks,
        }

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start : -self.horizon + 1]
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum(axis=0).squeeze(-1)
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch["returns"] = returns

        if self.include_rewards:
            rewards = self.fields.rewards[path_ind, start : end]  # (100, 2, 1)
            rewards = np.mean(rewards, axis=1)  # (100, 1)
            batch["rewards"] = rewards

        if self.include_env_ts:
            env_ts = (
                np.arange(history_start, start + self.horizon) - self.history_horizon
            )
            env_ts[np.where(env_ts < 0)] = self.max_path_length
            env_ts[np.where(env_ts >= self.max_path_length)] = self.max_path_length
            batch["env_ts"] = env_ts

        if "legal_actions" in self.fields.keys:
            batch["legal_actions"] = self.fields.legal_actions[
                path_ind, history_start:end
            ]
            batch["legal_actions"] = array_circular_shift(batch["legal_actions"], 1, shift_stride)

        if self.include_labels:
            labels = self.fields.labels[path_ind, start : end]  # (1, n_agents, 1)
            batch["labels"] = labels
        
        return batch

class ValueDataset(SequenceDataset):
    """
    adds a value field to the datapoints for training the value function
    """

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.include_returns is True

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        value_batch = {
            "x": batch["x"],
            "cond": batch["cond"],
            "returns": batch["returns"].mean(axis=-1),
        }
        return value_batch


class BCSequenceDataset(SequenceDataset):
    def __init__(
        self,
        env_type: str = "d4rl",
        env: str = "hopper-medium-replay",
        syn_dataset_dir: str = "",
        n_agents: int = 2,
        normalizer: str = "LimitsNormalizer",
        preprocess_fns: List[Callable] = [],
        max_path_length: int = 1000,
        max_n_episodes: int = 10000,
        discrete_action: bool = False,
        agent_share_parameters: bool = False,
    ):
        super().__init__(
            env_type=env_type,
            env=env,
            syn_dataset_dir=syn_dataset_dir,
            n_agents=n_agents,
            normalizer=normalizer,
            preprocess_fns=preprocess_fns,
            max_path_length=max_path_length,
            max_n_episodes=max_n_episodes,
            discrete_action=discrete_action,
            agent_share_parameters=agent_share_parameters,
            horizon=1,
            history_horizon=0,
            use_action=True,
            termination_penalty=0.0,
            use_padding=False,
            discount=1.0,
            include_returns=False,
        )

    def __getitem__(self, idx: int):
        path_ind, start, end, _ = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        if self.discrete_action:
            actions = self.fields.actions[path_ind, start:end]
        else:
            actions = self.fields.normed_actions[path_ind, start:end]

        batch = {"observations": observations, "actions": actions}
        return batch
