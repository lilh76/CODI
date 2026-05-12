import os
import h5py
import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim

from .madiff_sequence import SequenceDataset

############## DataBatch ##############
class OfflineDataBatch():
    def __init__(self, data, batch_size, max_seq_length, device='cpu') -> None:
        self.data = data
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length # None if taken all length
        self.device = device
        for k, v in self.data.items():
            # (batch_size, T, n_agents, *shape)
            # truncate here, interface directly in offlinebuffer
            self.data[k] = v[:, :max_seq_length].to(self.device)
    
    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data:
                return self.data[item]
            elif hasattr(self, item):
                return getattr(self, item)
            else:
                raise ValueError('Cannot index OfflineDataBatch with key "{}"'.format(item))
        else:
            raise ValueError('Cannot index OfflineDataBatch with key "{}"'.format(item))

    def to(self, device=None):
        if device is None:
            device = self.device
        for k, v in self.data.items():
            self.data[k] = v.to(device)
        self.device = device # update self.device
    
    def keys(self):
        return list(self.data.keys())
    
    def assign(self, key, value):
        if key in self.data:
            assert 0, "Cannot assign to existing key"
        self.data[key] = value


############## OfflineBuffer ##############
class OfflineBufferH5():
    def __init__(self, args, map_name, quality,
                 data_path='', # deepest folder
                 max_buffer_size=2000,
                 device='cpu',
                 shuffle=True) -> None:
        self.args = args
        self.base_data_folder = args.offline_data_folder
        self.map_name = map_name 
        # map name is an abstract name, can be map_name in sc2/scenario_name in mpe
        self.quality = quality
        # set data_path
        if not isinstance(data_path, list) and os.path.exists(data_path):
            self.data_path_list = [data_path] 
        else:
            self.data_path_list = []
            for i, quality_i in enumerate(quality.split("_")): # e.g. "medium_expert"
                data_path_i = data_path[i] if isinstance(data_path, list) else data_path
                self.data_path_list.append(self.get_data_path(map_name, quality_i, data_path_i))

        self.h5_paths = []
        for final_data_path in self.data_path_list:
            self.h5_paths.extend([os.path.join(final_data_path, f) for f in sorted(os.listdir(final_data_path)) if f.endswith(".h5")])
        print(self.h5_paths)
        #self.h5_paths = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path)]

        self.max_buffer_size = 100000000 if max_buffer_size <= 0 else max_buffer_size
        self.device = device # device does not work actually.
        self.shuffle = shuffle
        max_data_size_per_file = self.max_buffer_size // len(self.h5_paths)
        dataset = [self._read_data(self.h5_paths[i], max_data_size_per_file, shuffle) for i in range(len(self.h5_paths))]
        self.data = {
            k: np.concatenate([v[k] for v in dataset], axis=0) for k in dataset[0].keys()
        }
        traj_len_lst = self.data['terminated'].squeeze(-1).argmax(axis=1)
        returns = self.data['reward'].squeeze(-1).sum(axis=-1)
        print(f'Loaded {len(traj_len_lst)} trajectories. \
                Max traj len: {traj_len_lst.max()}, min traj len: {traj_len_lst.min()}. \
                Max return: {returns.max()}, min return: {returns.min()}.')
        self.keys = list(self.data.keys())
        self.buffer_size = self.data[self.keys[0]].shape[0]

        if shuffle:
            # shuffle again
            shuffled_idx = np.random.choice(self.buffer_size, self.buffer_size, replace=False)
            self.data = {k: v[shuffled_idx] for k, v in self.data.items()}
            
        if hasattr(self.args, "better_ratio"):
            returns_real = self.data['reward'].squeeze(-1).sum(axis=-1)  # (n_real, )
            
            threshold = np.percentile(returns_real, 100 * (1 - self.args.better_ratio))
            
            self.high_return_indices = np.where(returns_real >= threshold)[0]

    def get_data_path(self, map_name, quality, data_path):
        data_path = os.path.join(self.base_data_folder, self.args.env, map_name, quality, data_path)
        if all([".h5" not in f for f in os.listdir(data_path)]):
            # automatically find a folder
            existing_folders = [f for f in sorted(os.listdir(data_path)) if os.path.isdir(os.path.join(data_path, f))]
            assert len(existing_folders) > 0
            return os.path.join(data_path, existing_folders[-1])
        else:
            return data_path   

    def _read_data(self, h5_path, max_data_size, shuffle):
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
            
        data['state'] = data['obs'].reshape(data['state'].shape[0], data['state'].shape[1], -1)
            
        return data

    @staticmethod
    def max_t_filled(filled):
        return th.sum(filled, 1).max(0)[0]
    
    def can_sample(self, batch_size):
        return self.buffer_size >= batch_size

    def sample(self, batch_size):
        
        # First, find all valid trajectories (length >= 10)
        if self.args.agent == 'transformer':
            valid_indices = []
            for i in range(self.buffer_size):
                # Count the number of non-zero filled steps to determine trajectory length
                traj_length = np.sum(self.data['filled'][i])
                if traj_length >= self.args.window_size + 1:
                    valid_indices.append(i)
            sampled_ep_idx = np.random.choice(valid_indices, batch_size, replace=False)
        else:
            if hasattr(self.args, "better_ratio") and self.args.better_ratio > 0:
                
                sampled_ep_idx = np.random.choice(self.high_return_indices, batch_size, replace=False)
            else:
                sampled_ep_idx = np.random.choice(self.buffer_size, batch_size, replace=False)
        
        sampled_data = {k: th.tensor(v[sampled_ep_idx]) for k, v in self.data.items()}
        if self.args.use_corrected_terminated and "corrected_terminated" in sampled_data:
            sampled_data["terminated"] = sampled_data["corrected_terminated"]
        """sampled_data = {}
        for k, v in self.data.items():
            dtype = self.scheme[k].get("dtype", th.float32) if self.scheme is not None and k in self.scheme else th.float32
            sampled_data[k] = th.tensor(v[sampled_ep_idx], dtype=dtype)"""
            
        max_ep_t = self.max_t_filled(filled=sampled_data['filled']).item()
        
        offline_data_batch = OfflineDataBatch(data=sampled_data, 
                                              batch_size=batch_size, 
                                              max_seq_length=max_ep_t, 
                                              device=self.device)
        return offline_data_batch


class OfflineBufferNpy():
    def __init__(self, args, map_name, quality, data_path='', max_buffer_size=2000, device='cpu', shuffle=True, real_buffer=None) -> None:
        self.args = args
        self.map_name = map_name 
        # map name is an abstract name, can be map_name in sc2/scenario_name in mpe
        self.quality = quality
        # set data_path

        self.max_buffer_size = 100000000 if max_buffer_size <= 0 else max_buffer_size
        self.device = device # device does not work actually.
        self.shuffle = shuffle

        self.data = self._read_data(data_path, real_buffer)
        
        # Print dataset information
        traj_len_lst = self.data['terminated'].squeeze(-1).argmax(axis=1)
        returns = self.data['reward'].squeeze(-1).sum(axis=-1)
        print(f'Loaded {len(traj_len_lst)} trajectories. \
                Max traj len: {traj_len_lst.max()}, min traj len: {traj_len_lst.min()}. \
                Max return: {returns.max()}, min return: {returns.min()}.')
        
        self.keys = list(self.data.keys())
        self.buffer_size = self.data[self.keys[0]].shape[0]

        if shuffle:
            # shuffle again
            shuffled_idx = np.random.choice(self.buffer_size, self.buffer_size, replace=False)
            self.data = {k: v[shuffled_idx] for k, v in self.data.items()}
            
        if self.data['obs'].shape[-1] - args.obs_shape == args.n_agents:
            self.data['obs'] = self.data['obs'][..., : - args.n_agents]

    def _read_data(self, syn_dataset_dir, real_buffer=None):

        assert os.path.isdir(syn_dataset_dir)
        observations = np.load(os.path.join(syn_dataset_dir, "obs.npy")) # [bs, seq_len, n_agents, obs_shape]
            
        if observations.shape[-1] - self.args.obs_shape == self.args.n_agents:
            observations = observations[..., : - self.args.n_agents]
        
        actions = np.load(os.path.join(syn_dataset_dir, "acs.npy")) # [bs, seq_len, n_agents, action_dim=1]
        if os.path.exists(os.path.join(syn_dataset_dir, "rew.npy")):
            rewards = np.load(os.path.join(syn_dataset_dir, "rew.npy")) # [bs, seq_len, n_agents, 1]
        else:
            rewards = np.zeros_like(observations)[..., :1]
        if os.path.exists(os.path.join(syn_dataset_dir, "state.npy")):
            states = np.load(os.path.join(syn_dataset_dir, "state.npy"))
        else:
            obs_shape = observations.shape
            state_shape = obs_shape[:-2] + (obs_shape[-2] * obs_shape[-1],)
            states = observations.reshape(state_shape)

        observations = observations[:self.max_buffer_size].astype(np.float32)
        states = states[:self.max_buffer_size].astype(np.float32)
        actions = actions[:self.max_buffer_size].astype(np.int64)
        if len(rewards.shape) == 4:
            rewards = rewards[:self.max_buffer_size, :, 0, :].astype(np.float32)
        elif len(rewards.shape) == 3:
            rewards = rewards[:self.max_buffer_size].astype(np.float32)
        else:
            assert 0
        
        dones = np.zeros((observations.shape[0], observations.shape[1], 1), dtype=np.uint8)
        dones[:, -1, :] = 1
        filled = np.ones_like(dones, dtype=int)

        dones = dones[:self.max_buffer_size]
        filled = filled[:self.max_buffer_size]
        
        actions_onehot = np.array(th.nn.functional.one_hot(th.squeeze(th.from_numpy(actions), -1).to(th.int64), num_classes=self.args.n_actions).type(th.FloatTensor))

        if os.path.exists(os.path.join(syn_dataset_dir, "avail.npy")):
            avail_actions = np.load(os.path.join(syn_dataset_dir, "avail.npy"))
        else:
            avail_actions = np.ones_like(actions_onehot, dtype=np.int32)

        data = {
            "actions": actions,
            "actions_onehot": actions_onehot,
            "avail_actions": avail_actions,
            "filled": filled,
            "obs": observations, 
            "reward": rewards,
            "state": states,
            "terminated": dones,
        }

        return data

    @staticmethod
    def max_t_filled(filled):
        return th.sum(filled, 1).max(0)[0]
    
    def can_sample(self, batch_size):
        return self.buffer_size >= batch_size

    def sample(self, batch_size):
        sampled_ep_idx = np.random.choice(self.buffer_size, batch_size, replace=False)
        sampled_data = {k: th.tensor(v[sampled_ep_idx]) for k, v in self.data.items()}
        """sampled_data = {}
        for k, v in self.data.items():
            dtype = self.scheme[k].get("dtype", th.float32) if self.scheme is not None and k in self.scheme else th.float32
            sampled_data[k] = th.tensor(v[sampled_ep_idx], dtype=dtype)"""
            
        max_ep_t = self.max_t_filled(filled=sampled_data['filled']).item()
        offline_data_batch = OfflineDataBatch(data=sampled_data, 
                                              batch_size=batch_size, 
                                              max_seq_length=max_ep_t, 
                                              device=self.device)
        
        return offline_data_batch

class OfflineBuffer():
    def __init__(self, args, map_name, quality,
                data_path='', # deepest folder
                max_buffer_size=2000,
                device='cpu',
                shuffle=True) -> None:
        
        syn_dataset = getattr(args, 'syn_dataset', None)
        max_syn_buffer_size = getattr(args, 'syn_max', max_buffer_size)
        assert max_buffer_size > 0 or max_syn_buffer_size > 0, f"Buffer size must > 0! Real size: {max_buffer_size}, Syn size: {max_syn_buffer_size} "

        if max_buffer_size > 0 and (max_syn_buffer_size <= 0 or syn_dataset is None or syn_dataset == ''): # only real data
            self.contain_real = True
            self.contain_syn = False
            assert data_path is not None and data_path != '', "No real data path!"
            if args.offline_data_type=="h5":
                self.buffer = OfflineBufferH5(args, map_name, quality, data_path, max_buffer_size, device, shuffle)
                self.buffer_size = self.buffer.buffer_size
            else:
                raise NotImplementedError("Do not support offline data type: {}".format(args.offline_data_type))
            print(f"Loaded {self.buffer_size} real trajs and {0} syn trajs")
        elif (max_buffer_size <= 0 or data_path is None or data_path == '') and max_syn_buffer_size > 0: # only syn data
            self.contain_real = False
            self.contain_syn = True
            assert syn_dataset is not None and syn_dataset != '', "No syn data path!"
            self.buffer = OfflineBufferNpy(args, map_name, quality, syn_dataset, max_syn_buffer_size, device, shuffle)
            self.buffer_size = self.buffer.buffer_size
            print(f"Loaded {0} real trajs and {self.buffer_size} syn trajs")
        elif max_buffer_size > 0 and max_syn_buffer_size > 0: # both
            self.contain_real = True
            self.contain_syn = True
            assert data_path is not None and data_path != '', "No real data path!"
            assert syn_dataset is not None and syn_dataset != '', "No syn data path!"
            if args.offline_data_type=="h5":
                self.buffer = OfflineBufferH5(args, map_name, quality, data_path, max_buffer_size, device, shuffle)
            else:
                raise NotImplementedError("Do not support offline data type: {}".format(args.offline_data_type))
            self.syn_buffer = OfflineBufferNpy(args, map_name, quality, syn_dataset, max_syn_buffer_size, device, shuffle, self.buffer)
            self.buffer_size = self.buffer.buffer_size + self.syn_buffer.buffer_size
            print(f"Loaded {self.buffer.buffer_size} real trajs and {self.syn_buffer.buffer_size} syn trajs")

        print(f'Buffer size: {self.buffer_size}')
        
        self.shuffle = shuffle
        self.device = device
        self.args = args
        
        if hasattr(self.args, "better_ratio"):
            returns_real = self.buffer.data['reward'].squeeze(-1).sum(axis=-1)  # (n_real, )
            
            if hasattr(self, 'syn_buffer'):
            
                returns_syn = self.syn_buffer.data['reward'].squeeze(-1).sum(axis=-1)  # (n_syn, )
            
                all_returns = np.concatenate([returns_real, returns_syn])
            
            else:
                all_returns = returns_real
            
            threshold = np.percentile(all_returns, 100 * (1 - self.args.better_ratio))
            
            self.high_return_indices = np.where(all_returns >= threshold)[0]
    
    def can_sample(self, batch_size):
        return self.buffer_size >= batch_size

    def sample(self, batch_size):
        if self.contain_real + self.contain_syn == 1:
            return self.buffer.sample(batch_size)
        else:
            n_real = self.buffer.buffer_size
            n_syn = self.syn_buffer.buffer_size
            total_size = n_real + n_syn

            if hasattr(self.args, "better_ratio") and self.args.better_ratio > 0:
                
                sampled_ep_idx = np.random.choice(self.high_return_indices, batch_size, replace=False)

            else:
                sampled_ep_idx = np.random.choice(total_size, batch_size, replace=False)
            ori_sampled_ep_idx = sampled_ep_idx[sampled_ep_idx < n_real]
            syn_sampled_ep_idx = sampled_ep_idx[sampled_ep_idx >= n_real] - n_real
            
            keys = [k for k in self.buffer.keys if k in self.syn_buffer.keys]

            sampled_data = {}
            for k in sorted(keys): # ['actions', 'actions_onehot', 'avail_actions', 'filled', 'obs', 'reward', 'state', 'terminated']
                ori_data = self.buffer.data[k][ori_sampled_ep_idx, :-1]  # [bs1, seq_len_1, *dim]
                syn_data = self.syn_buffer.data[k][syn_sampled_ep_idx, :-1]  # [bs1, seq_len_2, *dim]
                seq_len_1 = ori_data.shape[1]
                seq_len_2 = syn_data.shape[1]
                if self.args.env == 'sc2' and k == 'terminated':
                    syn_data[:, -2] = 1
                max_seq_len = max(seq_len_1, seq_len_2)
                if seq_len_1 < max_seq_len:
                    pad_len = max_seq_len - seq_len_1
                    pad_width = [(0, 0)] * ori_data.ndim
                    pad_width[1] = (0, pad_len)
                    ori_data = np.pad(ori_data, pad_width, mode='constant', constant_values=0)
                if seq_len_2 < max_seq_len:
                    pad_len = max_seq_len - seq_len_2
                    pad_width = [(0, 0)] * syn_data.ndim
                    pad_width[1] = (0, pad_len)
                    syn_data = np.pad(syn_data, pad_width, mode='constant', constant_values=0)
                if k == 'obs':
                    n_agents = ori_data.shape[-2]
                    if ori_data.shape[-1] - syn_data.shape[-1] == n_agents:
                        ori_data = ori_data[..., :-n_agents]
                    if syn_data.shape[-1] - ori_data.shape[-1] == n_agents:
                        syn_data = syn_data[..., :-n_agents]
                if k == 'state':
                    kk = 'obs'
                    ori_obs = self.buffer.data[kk][ori_sampled_ep_idx, :-1]  # [bs1, seq_len_1, *dim]
                    n_agents = ori_obs.shape[-2]
                    if ori_obs.shape[-2] * ori_obs.shape[-1] - syn_data.shape[-1] == n_agents * n_agents:
                        ori_obs = ori_obs[..., :-n_agents]
                    if ori_obs.shape[-2] * ori_obs.shape[-1] - syn_data.shape[-1] == - n_agents * n_agents:
                        syn_data = syn_data.reshape(syn_data.shape[0], syn_data.shape[1], n_agents, -1)
                        syn_data = syn_data[..., :-n_agents].reshape(syn_data.shape[0], syn_data.shape[1], -1)
                    if ori_obs.shape[0] > 0:
                        ori_data = ori_obs.reshape(ori_data.shape[0], ori_data.shape[1], -1)
                    else:
                        ori_data = ori_obs
                # print(k, ori_data.shape, syn_data.shape)
                if ori_data.shape[0] == 0:
                    sampled_data[k] = th.tensor(syn_data)
                elif syn_data.shape[0] == 0:
                    sampled_data[k] = th.tensor(ori_data)
                else:
                    sampled_data[k] = th.tensor(np.concatenate([ori_data, syn_data], axis=0))
            
            if self.shuffle:
                shuffled_idx = np.random.choice(batch_size, batch_size, replace=False)
                sampled_data = {k: v[shuffled_idx] for k, v in sampled_data}
            
            max_ep_t = self.buffer.max_t_filled(filled=sampled_data['filled']).item()
            offline_data_batch = OfflineDataBatch(data=sampled_data, batch_size=batch_size, max_seq_length=max_ep_t, device=self.device)
            return offline_data_batch

class DataSaver():
    def __init__(self, save_path, logger=None, max_size=2000) -> None:
        self.save_path = save_path
        self.max_size = max_size
        #self.episode_batch = []
        self.data_batch = []
        self.cur_size = 0
        self.part_cnt = 0
        self.logger = logger
        os.makedirs(save_path, exist_ok=True)
    
    def append(self, data):
        self.data_batch.append(data) # data \in OfflineDataBatch/EpisodeBatch
        self.cur_size += data[list(data.keys())[0]].shape[0]
        #if len(self.episode_batch) >= self.max_size:
        if self.cur_size >= self.max_size:
            self.save_batch()
    
    def save_batch(self):
        #if len(self.data_batch) == 0:
        if self.cur_size == 0:
            return
        keys = list(self.data_batch[0].keys())
        data_dict = {k: [] for k in keys}
        for data in self.data_batch:
            for k in keys:
                if isinstance(data[k], th.Tensor):
                    data_dict[k].append(data[k].numpy())
                else:
                    data_dict[k].append(data[k])
                    
        # concatenate e.g. [(x, T, n_agents, *shape), ...] -> [max_size, T, n_agents, *shape]
        data_dict = {k: np.concatenate(v) for k, v in data_dict.items()}
        save_file = os.path.join(self.save_path, "part_{}.h5".format(self.part_cnt))
        with h5py.File(save_file, 'w') as file:
            for k, v in data_dict.items():
                file.create_dataset(k, data=v, compression='gzip', compression_opts=9)
        if self.logger is not None:
            self.logger.console_logger.info("Save offline buffer to {} with {} episodes".format(save_file, self.cur_size))
        else:
            print("Save offline buffer to {} with {} episodes".format(save_file, self.cur_size))
        self.data_batch.clear()
        self.cur_size = 0
        self.part_cnt += 1
    
    def close(self):
        self.save_batch()

def sequence_dataset(offline_buffer):
    data = offline_buffer.buffer.data

    actions = data['actions']
    avail_actions = data['avail_actions']
    filled = data['filled']
    obs = data['obs']
    reward = data['reward']
    state = data['state']
    
    n_episodes = actions.shape[0]
    n_agents = actions.shape[2]

    onehot_matrix = np.eye(n_agents)[np.newaxis, :, :] # (1, n_agents, n_agents)

    for i in range(n_episodes):
        ep_length = filled[i].sum().item()
        episode_data = {}
        onehot = np.broadcast_to(onehot_matrix, (ep_length, n_agents, n_agents))
        episode_data["observations"] = np.concatenate([obs[i][:ep_length], onehot], axis=-1)
        episode_data["legal_actions"] = avail_actions[i][:ep_length]
        shape = list(reward[i][:ep_length].shape)
        shape[-1] = n_agents
        episode_data["rewards"] = np.broadcast_to(reward[i][:ep_length], tuple(shape))
        episode_data["actions"] = actions[i][:ep_length].squeeze(-1)
        episode_data["terminals"] = np.zeros(
            (ep_length, n_agents), dtype=bool
        )
        episode_data["terminals"][-1] = True
        
        if 'random_labels' in data:
            episode_data["random_labels"] = np.tile(data['random_labels'][i], (shape[0], 1))
            
        yield episode_data


def cycle(dl):
    while True:
        for data in dl:
            yield data

class MADiffOfflineBuffer():
    def __init__(self, args, offline_buffer: OfflineBuffer):
        assert args.agent == 'madiff', "MADiffOfflineBuffer only supports madiff_ctce config"
        args.max_path_length = offline_buffer.buffer.data['filled'].shape[1]
        self.args = args
        self.dataset = SequenceDataset(
            sequence_dataset(offline_buffer),
            n_agents=args.n_agents,
            horizon=args.horizon, # 4
            history_horizon=args.history_horizon, # 20
            normalizer=args.normalizer, # CDFNormalizer
            max_n_episodes=args.offline_max_buffer_size, # 50k
            use_padding=args.use_padding, # True
            use_action=args.use_action, # True
            discrete_action=args.discrete_action, # True
            max_path_length=args.max_path_length,
            include_returns=args.returns_condition, # True
            include_rewards=args.rewards_condition, # True
            include_env_ts=False,
            returns_scale=args.returns_scale, # 6
            discount=args.discount, # 0.99
            termination_penalty=args.termination_penalty, # 0.0
            agent_share_parameters=True,
            use_inv_dyn=True,
            decentralized_execution=args.decentralized_execution, # False
            use_zero_padding=args.use_zero_padding, # False
            agent_condition_type="single" if args.decentralized_execution else "all", # all
            pred_future_padding=args.pred_future_padding, # True
            circular_shift = args.circular_shift,
            shift_ratio = args.shift_ratio,
        )
        self.dataloader = cycle(
            th.utils.data.DataLoader(
                self.dataset,
                batch_size=args.offline_batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
            )
        )
        
        print(f'Dataset size: {len(self.dataset)}')
    
    def sample(self, batch_size):
        return next(self.dataloader)
        #return self.buffer.sample(batch_size)

    def sample_bs(self, batch_size):
        # batch_size could be large, the final batch size is still the number of traj fragments
        temp_dataloader = cycle(
            th.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
            )
        )
        return next(temp_dataloader)