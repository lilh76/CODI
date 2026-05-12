from typing import List, Optional

import numpy as np
import torch

import scipy.interpolate as interpolate
from einops import rearrange, repeat

def array_circular_shift(ar, ax, s):
    index = [slice(None)] * len(ar.shape)
    index[ax] = slice(s, None)
    part1 = ar[tuple(index)]
    index[ax] = slice(0, s)
    part2 = ar[tuple(index)]
    return np.concatenate((part1, part2), axis=ax)

# ----------------- copied from madiff: diffuser.datasets.buffer ----------------- #
def atleast_nd(x, n: int):
    while x.ndim < n:
        x = np.expand_dims(x, axis=-1)
    return x


class ReplayBuffer:
    def __init__(
        self,
        n_agents: int,
        max_n_episodes: int,
        max_path_length: int,
        termination_penalty: float,
        global_feats: List[str] = ["states"],
        use_zero_padding: bool = True,
    ):
        self._dict = {
            "path_lengths": np.zeros(max_n_episodes, dtype=int),
        }
        self._count = 0
        self.n_agents = n_agents
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty
        self.global_feats = global_feats
        self.use_zero_padding = use_zero_padding

    def __repr__(self):
        return "[ datasets/buffer ] Fields:\n" + "\n".join(
            f"    {key}: {val.shape}" for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self["path_lengths"])

    def _add_keys(self, path):
        if hasattr(self, "keys"):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        """
        can access fields with `buffer.observations`
        instead of `buffer['observations']`
        """
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items() if k != "path_lengths"}.items()

    def _allocate(self, key, array):
        assert key not in self._dict
        dim = array.shape[-1]
        if len(array.shape) == 3:
            shape = (self.max_n_episodes, self.max_path_length, self.n_agents, dim)
        else:
            assert len(array.shape) == 2, f"Invalid shape {array.shape} of {key}"
            shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)

    def add_path(self, path):
        # path[key] shape: (path_length, n_agents, dim)
        path_length = len(path["observations"])
        assert path_length <= self.max_path_length

        # NOTE(zbzhu): agents must terminate together
        all_terminals = np.any(path["terminals"], axis=1)
        if all_terminals.any():
            assert (bool(all_terminals[-1]) is True) and (not all_terminals[:-1].any())

        # if first path added, set keys based on contents
        self._add_keys(path)

        # add tracked keys in path
        for key in self.keys:
            if key in self.global_feats:  # all agents share the same global state
                array = atleast_nd(path[key], n=2)
            else:
                array = atleast_nd(path[key], n=3)
            if key not in self._dict:
                self._allocate(key, array)
            if not self.use_zero_padding and key not in ["rewards"]:
                self._dict[key][self._count] = array[-1]
            self._dict[key][self._count, :path_length] = array

        # penalize early termination
        if all_terminals.any() and self.termination_penalty is not None:
            if "timeouts" in path:
                assert not path[
                    "timeouts"
                ].any(), "Penalized a timeout episode for early termination"
            self._dict["rewards"][
                self._count, path_length - 1
            ] += self.termination_penalty

        # record path length
        self._dict["path_lengths"][self._count] = path_length

        # increment path counter
        self._count += 1

    def truncate_path(self, path_ind, step):
        old = self._dict["path_lengths"][path_ind]
        new = min(step, old)
        self._dict["path_lengths"][path_ind] = new

    def finalize(self):
        # remove extra slots
        for key in self.keys + ["path_lengths"]:
            self._dict[key] = self._dict[key][: self._count]
        self._add_attributes()
        print(f"[ datasets/buffer ] Finalized replay buffer | {self._count} episodes")

        debug = False
        if debug:
            print("[ datasets/buffer ] Fields:\n" + "\n".join(
                f"    {key}: {val.shape}" for key, val in self.items()
            ))
            assert 0


# ----------------- copied from madiff: diffuser.datasets.normalization ----------------- #

POINTMASS_KEYS = ["observations", "actions", "next_observations", "deltas"]

# -----------------------------------------------------------------------------#
# --------------------------- multi-field normalizer --------------------------#
# -----------------------------------------------------------------------------#


class DatasetNormalizer:
    def __init__(
        self,
        dataset,
        normalizer,
        global_feats: List[str] = ["states"],
        agent_share_parameters=False,
        path_lengths=None,
    ):
        dataset = flatten(
            dataset, path_lengths
        )  # dataset from `ReplayBuffer` object to python dict

        self.n_agents = dataset["observations"].shape[1]
        self.observation_dim = dataset["observations"].shape[-1]
        self.action_dim = (
            dataset["actions"].shape[-1] if "actions" in dataset.keys() else 0
        )
        self.global_feats = global_feats
        self.agent_share_parameters = agent_share_parameters

        if type(normalizer) is str:
            normalizer = eval(normalizer)

        self.normalizers = {}
        for key, val in dataset.items():
            try:
                if key in global_feats or self.agent_share_parameters:
                    self.normalizers[key] = normalizer(val.reshape(-1, val.shape[-1]))
                else:
                    self.normalizers[key] = [
                        normalizer(val[:, i]) for i in range(val.shape[1])
                    ]
            except Exception:
                print(f"[ utils/normalization ] Skipping {key} | {normalizer}")
            # key: normalizer(val)
            # for key, val in dataset.items()

    def __repr__(self):
        string = ""
        for key, normalizer in self.normalizers.items():
            string += f"{key}: {normalizer}]\n"
        return string

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, x, key):
        if key in self.global_feats or self.agent_share_parameters:
            return self.normalizers[key].normalize(x)
        else:
            return np.stack(
                [
                    self.normalizers[key][i].normalize(x[..., i, :])
                    for i in range(x.shape[-2])
                ],
                axis=-2,
            )

    def unnormalize(self, x, key):
        if key in self.global_feats or self.agent_share_parameters:
            return self.normalizers[key].unnormalize(x)
        else:
            return np.stack(
                [
                    self.normalizers[key][i].unnormalize(x[..., i, :])
                    for i in range(x.shape[-2])
                ],
                axis=-2,
            )


def flatten(dataset, path_lengths):
    """
    flattens dataset of { key: [ n_episodes x max_path_lenth x dim ] }
        to { key : [ (n_episodes * sum(path_lengths)) x dim ]}
    """

    flattened = {}
    for key, xs in dataset.items():
        assert len(xs) == len(path_lengths)
        flattened[key] = np.concatenate(
            [x[:length] for x, length in zip(xs, path_lengths)], axis=0
        )
    return flattened


# -----------------------------------------------------------------------------#
# ------------------------------- @TODO: remove? ------------------------------#
# -----------------------------------------------------------------------------#


class PointMassDatasetNormalizer(DatasetNormalizer):
    def __init__(self, preprocess_fns, dataset, normalizer, keys=POINTMASS_KEYS):
        reshaped = {}
        for key, val in dataset.items():
            dim = val.shape[-1]
            reshaped[key] = val.reshape(-1, dim)

        self.observation_dim = reshaped["observations"].shape[1]
        self.action_dim = reshaped["actions"].shape[1]

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizers = {key: normalizer(reshaped[key]) for key in keys}


# -----------------------------------------------------------------------------#
# -------------------------- single-field normalizers -------------------------#
# -----------------------------------------------------------------------------#


class Normalizer:
    """
    parent class, subclass by defining the `normalize` and `unnormalize` methods
    """

    def __init__(self, X):
        X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return (
            f"""[ Normalizer ] dim: {self.mins.size}\n    -: """
            f"""{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n"""
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()


class DebugNormalizer(Normalizer):
    """
    identity function
    """

    def normalize(self, x, *args, **kwargs):
        return x

    def unnormalize(self, x, *args, **kwargs):
        return x


class GaussianNormalizer(Normalizer):
    """
    normalizes to zero mean and unit variance
    """

    def __init__(self, X, *args, **kwargs):
        super().__init__(X=X, *args, **kwargs)
        self.means = X.mean(axis=0)
        self.stds = X.std(axis=0)
        self.z = 1

    def __repr__(self):
        return (
            f"""[ Normalizer ] dim: {self.mins.size}\n    """
            f"""means: {np.round(self.means, 2)}\n    """
            f"""stds: {np.round(self.z * self.stds, 2)}\n"""
        )

    def normalize(self, x):
        return (x - self.means) / self.stds

    def unnormalize(self, x):
        return x * self.stds + self.means


class LimitsNormalizer(Normalizer):
    """
    maps [ xmin, xmax ] to [ -1, 1 ]
    """

    def normalize(self, x):
        # [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins + 1e-20)
        # [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        """
        x : [ -1, 1 ]
        """
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = np.clip(x, -1, 1)

        # [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.0

        return x * (self.maxs - self.mins) + self.mins


class SafeLimitsNormalizer(LimitsNormalizer):
    """
    functions like LimitsNormalizer, but can handle data for which a dimension is constant
    """

    def __init__(self, *args, eps=1, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(len(self.mins)):
            if self.mins[i] == self.maxs[i]:
                print(
                    f"""
                    [ utils/normalization ] Constant data in dimension {i} | """
                    f"""max = min = {self.maxs[i]}"""
                )
                self.mins -= eps
                self.maxs += eps


# -----------------------------------------------------------------------------#
# ------------------------------- CDF normalizer ------------------------------#
# -----------------------------------------------------------------------------#


class CDFNormalizer(Normalizer):
    """
    makes training data uniform (over each dimension) by transforming it with marginal CDFs
    """

    def __init__(self, X):
        super().__init__(atleast_2d(X))
        self.dim = X.shape[1]
        self.cdfs = [CDFNormalizer1d(X[:, i]) for i in range(self.dim)]

    def __repr__(self):
        return f"[ CDFNormalizer ] dim: {self.mins.size}\n" + "    |    ".join(
            f"{i:3d}: {cdf}" for i, cdf in enumerate(self.cdfs)
        )

    def wrap(self, fn_name, x):
        shape = x.shape
        # reshape to 2d
        x = x.reshape(-1, self.dim)
        out = np.zeros_like(x)
        for i, cdf in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            out[:, i] = fn(x[:, i])
        return out.reshape(shape)

    def normalize(self, x):
        return self.wrap("normalize", x)

    def unnormalize(self, x):
        return self.wrap("unnormalize", x)


class CDFNormalizer1d:
    """
    CDF normalizer for a single dimension
    """

    def __init__(self, X):
        assert X.ndim == 1
        X = X.astype(np.float32)
        if X.max() == X.min():
            self.constant = True
        else:
            self.constant = False
            quantiles, cumprob = empirical_cdf(X)
            self.fn = interpolate.interp1d(quantiles, cumprob)
            self.inv = interpolate.interp1d(cumprob, quantiles)

            self.xmin, self.xmax = quantiles.min(), quantiles.max()
            self.ymin, self.ymax = cumprob.min(), cumprob.max()

    def __repr__(self):
        return f"[{np.round(self.xmin, 2):.4f}, {np.round(self.xmax, 2):.4f}"

    def normalize(self, x):
        if self.constant:
            return x

        x = np.clip(x, self.xmin, self.xmax)
        # [ 0, 1 ]
        y = self.fn(x)
        # [ -1, 1 ]
        y = 2 * y - 1
        return y

    def unnormalize(self, x, eps=1e-4):
        """
        X : [ -1, 1 ]
        """

        # [ -1, 1 ] --> [ 0, 1 ]
        if self.constant:
            return x

        x = (x + 1) / 2.0

        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            print(
                f"""[ dataset/normalization ] Warning: out of range in unnormalize: """
                f"""[{x.min()}, {x.max()}] | """
                f"""x : [{self.xmin}, {self.xmax}] | """
                f"""y: [{self.ymin}, {self.ymax}]"""
            )

        x = np.clip(x, self.ymin, self.ymax)

        y = self.inv(x)
        return y


def empirical_cdf(sample):
    # https://stackoverflow.com/a/33346366

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob


def atleast_2d(x):
    if x.ndim < 2:
        x = x[:, None]
    return x


# ----------------- copied from madiff: diffuser.datasets.preprocessing ----------------- #

def compose(*fns):
    def _fn(x):
        for fn in fns:
            x = fn(x)
        return x

    return _fn


def get_preprocess_fn(fn_names, env):
    fns = [eval(name)(env) for name in fn_names]
    return compose(*fns)

# ---------------- copied from madiff: diffuser.utils.mask_generator ----------------- #

class MultiAgentMaskGenerator:
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        # obs mask setup
        history_horizon: int = 10,
        # action mask
        action_visible: bool = False,
    ):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.history_horizon = history_horizon
        self.action_visible = action_visible

    def __call__(self, shape: tuple, agent_mask: np.ndarray):
        if len(shape) == 4:
            B, T, _, D = shape  # b t a f
        else:
            B = None
            T, _, D = shape  # t a f
        if self.action_visible:
            assert D == (self.action_dim + self.observation_dim)
        else:
            assert D == self.observation_dim

        # generate obs mask
        steps = np.arange(0, T)
        obs_mask = np.tile(
            (steps < self.history_horizon + 1).reshape(T, 1), (1, self.observation_dim)
        )

        # generate action mask
        if self.action_visible:
            action_mask = np.tile((steps < self.history_horizon).reshape(T, 1), (1, D))

        visible_mask = obs_mask # [h1+h2, obs_shape]
        if self.action_visible:
            visible_mask = np.concatenate([action_mask, visible_mask], dim=-1)

        # the history of invisible agents are conditioned to be always zero
        invisible_mask = np.tile((steps < self.history_horizon).reshape(T, 1), (1, D)) # [h1+h2, obs_shape]
        # agent_mask[a_idx] = True if agent a_idx is visible -> mask[a_idx] = visible_mask
        mask = np.stack([invisible_mask, visible_mask], axis=0)[agent_mask.astype(int)]
        mask = rearrange(mask, "a t f -> t a f")
        if B is not None:
            mask = repeat(mask, "t a f -> b t a f", b=B)

        return mask


# ---------------- copied from madiff: diffuser.datasets.sequence ----------------- #

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        itr,
        n_agents: int = 2,
        horizon: int = 4,
        normalizer: str = "LimitsNormalizer",
        # preprocess_fns: List[Callable] = [],
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
        include_env_ts: bool = False,
        history_horizon: int = 0,
        agent_share_parameters: bool = False,
        # use_seed_dataset: bool = False,
        decentralized_execution: bool = False,
        use_inv_dyn: bool = True,
        use_zero_padding: bool = True,
        agent_condition_type: str = "single",
        pred_future_padding: bool = False,
        circular_shift: bool = False,
        shift_ratio: float = 0.1,
    ):


        assert agent_condition_type in ["single", "all", "random"], agent_condition_type
        self.agent_condition_type = agent_condition_type

        self.global_feats = ["states"]

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
        self.include_env_ts = include_env_ts
        self.decentralized_execution = decentralized_execution
        self.use_zero_padding = use_zero_padding
        self.pred_future_padding = pred_future_padding
        self.circular_shift = circular_shift
        self.shift_ratio = shift_ratio

        fields = ReplayBuffer(
            n_agents,
            max_n_episodes,
            max_path_length,
            termination_penalty,
            global_feats=self.global_feats,
            use_zero_padding=self.use_zero_padding,
        )
        for _, episode in enumerate(itr):
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

        if self.discrete_action: # 20250806 here
            # smac has discrete actions, so we only need to normalize observations
            self.normalize(["observations"])
        else:
            self.normalize()

        self.pad_future()
        if self.history_horizon > 0:
            self.pad_history()

        # print(fields)
        '''
        observations: (20000, 26, 3, 17)
        normed_observations: (20000, 33, 3, 17)
        actions: (20000, 33, 3, 1)
        rewards: (20000, 33, 3, 1)
        terminals: (20000, 33, 3, 1)
        legal_actions: (20000, 33, 3, 5)
        '''

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
            array = self.fields[key].reshape(shape[0] * shape[1], *shape[2:]) # [bs * h, n_agents, obs_shape]
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
        if self.circular_shift:
            observations = array_circular_shift(observations, 1, shift_stride)
        if self.use_action:
            if self.discrete_action:
                actions = self.fields.actions[path_ind, history_start:end]
            else:
                actions = self.fields.normed_actions[path_ind, history_start:end]
            if self.circular_shift:
                actions = array_circular_shift(actions, 1, shift_stride)

        if self.use_action:
            trajectories = np.concatenate([actions, observations], axis=-1)
        else:
            trajectories = observations

        if self.use_inv_dyn:
            cond_masks = self.mask_generator(observations.shape, agent_mask) # [h, n_agents, obs_shape]
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
        if self.pred_future_padding: # 1
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
            if self.circular_shift:
                batch["legal_actions"] = array_circular_shift(batch["legal_actions"], 1, shift_stride)
            

        return batch
