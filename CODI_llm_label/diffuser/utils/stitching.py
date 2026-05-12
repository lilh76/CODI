import datetime
from enum import Enum
import numpy as np
import torch as th
import torch.nn.functional as F
from ml_logger import logger
import os

from .arrays import to_torch
from .imbalance import check_imbalance_parallel


class PathStatus(Enum):
    IN_PROGRESS = 0
    COMPLETED = 1
    ABANDONED = 2

def check_state_pair(args, trainer, state, state_prime, discrete_action, threshold, normalizer):
    with th.no_grad():
        obs_comb = th.concat([state, state_prime], dim=-1)  # (bs, na, 2 * obs_dim)
        action = trainer.ema_model.inv_model(obs_comb.reshape(obs_comb.shape[0], -1)).reshape(obs_comb.shape[0], obs_comb.shape[1], -1)  #  (bs, na, output_dim)
        # action should be converted from logits into onehot encoding
        if discrete_action:
            if hasattr(trainer.ema_model, 'legal_model'):
                xxx = state.reshape(state.shape[0], -1)
                legality_logits = trainer.ema_model.legal_model(xxx)
                legality_logits = legality_logits.reshape(state.shape[0], state.shape[1], -1)  # [bs, n_agents, n_actions]
                legal_mask = (th.sigmoid(legality_logits) > 0.5).float()
                masked_action = action.clone()
                masked_action = masked_action - (1 - legal_mask) * 1e10
                indices = th.argmax(masked_action, dim=-1)
                
                # print('legality_logits', legality_logits)
                # print('legal_mask', legal_mask)
                # print('action', action)
                # print('masked_action', masked_action)
                # print('indices', indices)
            else:
                indices = th.argmax(action, dim=-1)
            action_onehot = F.one_hot(indices, num_classes=action.size(-1)).float()
            ao_comb = th.concat([action_onehot, state], dim=-1)
        else:
            assert 0
            ao_comb = th.concat([action, state], dim=-1)  # (bs, na, n_actions + obs_dim)
        state_prime_recon = trainer.ema_model.fwd_model(ao_comb.reshape(ao_comb.shape[0], -1)).reshape(ao_comb.shape[0], ao_comb.shape[1], -1)  # (bs, na, obs_dim)
        recon_error = th.mean((state_prime - state_prime_recon) ** 2)
        
        # print(threshold)
        # print(recon_error)
        # print(state_prime)
        # print(state_prime_recon)
        # print(ao_comb)
        valid = recon_error < threshold
        if not valid:
            return valid, None, None, None

        # assert valid
        try:
            reward = trainer.ema_model.rwd_model(ao_comb.reshape(ao_comb.shape[0], -1))
        except:
            ao_comb = th.cat([ao_comb, state_prime], dim=-1)
            reward = trainer.ema_model.rwd_model(ao_comb.reshape(ao_comb.shape[0], -1))
              
        reward = reward.reshape(ao_comb.shape[0], 1, 1).repeat(1, ao_comb.shape[1], 1) # [bs=1, n_agents, 1]

        action = action.unsqueeze(1).detach() # [bs, n_agents, n_actions, 1]
        if discrete_action:
            if hasattr(trainer.ema_model, 'legal_model'):
                action = th.argmax(masked_action, dim=-1, keepdim=True)
                avail = legal_mask
                # print('final action', action[..., 0])
            else:
                action = th.argmax(action, dim=-1, keepdim=True)
                avail = action_onehot * 0 + 1
                
    return valid, action, reward, avail

def generate_from_obs(args, obs, trainer, dataset, num_episodes, n_agents, cond_return, cond_rtg, cond_labels=None, use_composition_tech=False):
    if cond_rtg is None:
        returns = (cond_return * th.ones(num_episodes, 1, n_agents)).to(args.device)  
    else:
        returns = cond_rtg
    if cond_labels is not None:
        returns = (returns, cond_labels)
    obs = to_torch(obs, device=args.device)
    cond_obs = F.pad(obs, (0, 0, 0, 0, 0, dataset.horizon - 1))
    cond_masks = (th.arange(0, dataset.horizon) < 1).reshape(1, dataset.horizon, 1, 1).repeat(cond_obs.shape[0], 1, cond_obs.shape[2], cond_obs.shape[3])
    conditions = {"x": to_torch(cond_obs, device=args.device), "masks": to_torch(cond_masks, device=args.device)}
    samples = trainer.ema_model.conditional_sample(conditions, returns=returns, use_composition_tech=use_composition_tech)  # (num_episodes, hrz, ag, obs_dim)
    # logger.print(samples)
    return samples

def generate_from_partially_noised_seq_parallel(args, trainer, dataset, n_agents, cond_return, cur_obs_seq, bad_agent, cond_rtg=None):
    num_episodes = len(cur_obs_seq)
    if cond_rtg is None:
        returns = (cond_return * th.ones(num_episodes, 1, n_agents)).to(args.device)  
    else:
        returns = cond_rtg
    # cur_obs_seq is a list of (seq, na, dim). cond_obs_start (bs, 1, na, dim)
    cond_obs = np.zeros((len(cur_obs_seq), dataset.horizon, cur_obs_seq[0].shape[1], cur_obs_seq[0].shape[2]))
    cond_masks = np.ones_like(cond_obs, dtype=bool)  # (bs, hrz, na, dim)
    for i, (array, agent_arr) in enumerate(zip(cur_obs_seq, bad_agent)):
        cond_obs[i, :array.shape[0], :, :] = array
        cond_masks[i, :, agent_arr, :] = False
    cond_masks[:, 0, :, :] = True
    conditions = {"x": to_torch(cond_obs, device=args.device), "masks": to_torch(cond_masks, device=args.device)}
    samples = trainer.ema_model.conditional_sample(conditions, returns=returns)  # (num_episodes, hrz, ag, obs_dim)
    return samples


def trajectory_stitching_parallel(
        args, 
        n_gen, 
        gen_batch_size, 
        trainer, 
        dataset, 
        horizon, 
        n_agents, 
        discrete_action, 
        cond_return, 
        include_labels,
        times_of_regen_upper_limit, 
        total_times_of_regen_upper_limit, 
        use_composition_tech, 
        partially_noise, 
        threshold, 
        max_path_length_stitch, 
        each_ig_step_num, 
        verbose=False,
        save_traj_dir='',
        normalizer=None,
    ):
    
    
    soft_n_gen = n_gen + gen_batch_size
    path_status = [PathStatus.IN_PROGRESS] * soft_n_gen
    times_of_regen_list = [0] * soft_n_gen
    total_times_of_regen_list = [0] * soft_n_gen

    sampled_ep_idx = np.random.choice(dataset.n_episodes, soft_n_gen)
    sampled_obs = dataset.fields.normed_observations[sampled_ep_idx]
    init_obs = sampled_obs[:, 0:1, :, :]  # (n_gen, 1, ag, obs_dim)
    
    if max_path_length_stitch is None:
        seq_len = dataset.max_path_length + 1
    else:
        seq_len = max_path_length_stitch
        
    gen_obs_seq = np.hstack([init_obs, np.zeros((soft_n_gen, seq_len, n_agents, dataset.observation_dim))])  # (n_gen, seq_len + 1, ag, obs_dim)
    gen_action_seq = np.zeros((soft_n_gen, seq_len, n_agents, dataset.action_dim)) # print(dataset.action_dim) # 1
    gen_avail_seq = None
    gen_reward_seq = np.zeros((soft_n_gen, seq_len, n_agents, 1))
    cur = [0] * soft_n_gen
            
    start_time = datetime.datetime.now()
    last_saved_count = 0
    while len([x for x in path_status if x == PathStatus.COMPLETED]) < n_gen:
        completed_count = len([x for x in path_status if x == PathStatus.COMPLETED])
        IN_PROGRESS_count = len([x for x in path_status if x == PathStatus.IN_PROGRESS])
        ABANDONED_count = len([x for x in path_status if x == PathStatus.ABANDONED])
        if completed_count // 100 > last_saved_count // 100:
            current_time = datetime.datetime.now()
            elapsed_time = current_time - start_time
            progress_file = os.path.join(save_traj_dir, 'progress.txt')
            time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            with open(progress_file, 'a') as f:
                f.write(f"\n{time_str}      spent {elapsed_time}      COMPLETED {completed_count}      IN_PROGRESS {IN_PROGRESS_count}      ABANDONED {ABANDONED_count}\n")
            last_saved_count = completed_count
        
        batch_idx = [idx for idx, v in enumerate(path_status) if v == PathStatus.IN_PROGRESS][:gen_batch_size] # len(batch_idx) < gen_batch_size may happen
        cond_obs = np.stack([gen_obs_seq[idx][cur[idx]: cur[idx] + 1] for idx in batch_idx], axis=0)  # (bs, 1, ag, obs_dim)
        # achieved_return = [gen_reward_seq[idx][ : cur[idx], 0, 0].sum() for idx in batch_idx]  # list with len = bs
        # print(1, [gen_obs_seq[idx][cur[idx]: cur[idx] + 1] for idx in batch_idx])
        # print(2, achieved_return)
        last_cur = cur[:]  # copy
        
        # idx = 0
        # print([gen_reward_seq[idx][ : cur[idx], 0, 0] for idx in batch_idx])
        achieved_returns = np.stack([gen_reward_seq[idx][ : last_cur[idx], 0, 0].sum() for idx in batch_idx], axis=0) # (bs, )
        cond_rtg = th.tensor(achieved_returns * (-1) + cond_return, device=args.device, dtype=th.float32).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, n_agents) # [num_episodes, 1, n_agents]
        # cond_rtg = None
        
        cond_labels = None
        if include_labels:
            cond_labels = (cond_rtg * 0).unsqueeze(-1) # [bs, 1, n_agents, 1]
            cond_labels[:, :, :] = 1
        
        seg = generate_from_obs(args, cond_obs, trainer, dataset, len(batch_idx), n_agents, cond_return, cond_rtg, cond_labels, use_composition_tech)  # (bs, hrz, ag, obs_dim)
        for i, idx in enumerate(batch_idx):
            for j in range(1, horizon):
                valid, action, reward, avail = check_state_pair(args, trainer, seg[i: i + 1, j - 1], seg[i: i + 1, j], discrete_action, threshold, normalizer)
                if valid:
                    cur[idx] += 1
                    times_of_regen_list[idx] = 0
                    gen_obs_seq[idx][cur[idx]] = seg[i: i + 1, j: j + 1].cpu()  # seg[i: i + 1, j: j + 1]: (1, 1, ag, obs_dim)
                    gen_action_seq[idx][cur[idx] - 1] = action.cpu()
                    if gen_avail_seq is None:
                        n_actions = avail.shape[-1]
                        gen_avail_seq = np.zeros((soft_n_gen, seq_len, n_agents, n_actions))
                    gen_avail_seq[idx][cur[idx] - 1] = avail.cpu()
                    gen_reward_seq[idx][cur[idx] - 1] = reward.cpu()
                    # achieved_return = gen_reward_seq[idx][ : cur[idx] - 1, 0, 0].sum()
                    if cur[idx] >= seq_len:
                        # traj_return = gen_reward_seq[idx][ : cur[idx] - 1, 0, 0].sum()
                        path_status[idx] = PathStatus.COMPLETED
                        # print(1111, gen_reward_seq[idx][ : cur[idx] - 1, 0, 0])
                        # assert 0
                        break
                else:
                    times_of_regen_list[idx] += 1
                    total_times_of_regen_list[idx] += 1
                    break
            if times_of_regen_list[idx] >= times_of_regen_upper_limit or total_times_of_regen_list[idx] >= total_times_of_regen_upper_limit:
                path_status[idx] = PathStatus.ABANDONED
        
        if partially_noise and sum(cur) - sum(last_cur) > 0:
            assert 0
            assert not include_labels
            assert not use_composition_tech
            if verbose:
                logger.print("AFTER SEG GEN")
                logger.print(f"cur: {cur}")
                logger.print(f"last_cur: {last_cur}")
            batch_idx_not_abandoned = [idx for idx in batch_idx if path_status[idx] != PathStatus.ABANDONED]
            if verbose:
                logger.print(f"batch_idx_not_abandoned: {batch_idx_not_abandoned}")
            obs_balance_to_be_checked = [gen_obs_seq[idx, last_cur[idx]: cur[idx] + 1] for idx in batch_idx_not_abandoned]
            action_balance_to_be_checked = [gen_action_seq[idx, last_cur[idx]: cur[idx] + 1] for idx in batch_idx_not_abandoned]
            reward_balance_to_be_checked = [gen_reward_seq[idx] for idx in batch_idx_not_abandoned]
            bad_agent = check_imbalance_parallel(args, trainer, obs_balance_to_be_checked, action_balance_to_be_checked, discrete_action, each_ig_step_num)
            if verbose:
                logger.print(f"bad_agent: {bad_agent}")

            bad_agent_i = [i for i, bad_agent_list in enumerate(bad_agent) if bad_agent_list]
            if verbose:
                logger.print(f"bad_agent_i: {bad_agent_i}")
            batch_idx_imbalanced = [batch_idx_not_abandoned[i] for i in bad_agent_i]
            if verbose:
                logger.print(f"batch_idx_imbalanced: {batch_idx_imbalanced}")
            bad_agent = [bad_agent[i] for i in bad_agent_i]
            if verbose:
                logger.print(f"bad_agent: {bad_agent}")
            if batch_idx_imbalanced:
                imbalanced_obs_seq = [obs_balance_to_be_checked[i] for i in bad_agent_i]
                imbalanced_rwd_seq = [reward_balance_to_be_checked[i] for i in bad_agent_i]

                achieved_returns = [x[: last_cur[idx], 0, 0].sum() for x in imbalanced_rwd_seq]
                achieved_returns = np.stack(achieved_returns, axis=0)
                cond_rtg = th.tensor(achieved_returns * (-1) + cond_return, device=args.device, dtype=th.float32).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, n_agents) # [num_episodes, 1, n_agents]
                seg = generate_from_partially_noised_seq_parallel(args, trainer, dataset, n_agents, cond_return, imbalanced_obs_seq, bad_agent, cond_rtg)

                gen_seg_len = [len(obs) for obs in imbalanced_obs_seq]
                if verbose:
                    logger.print(f"gen_seg_len: {gen_seg_len}")
                for idx in batch_idx_imbalanced:
                    cur[idx] = last_cur[idx]
                if verbose:
                    logger.print(f"cur: {cur}")
                for i in range(seg.shape[0]):
                    for j in range(1, gen_seg_len[i]):
                        valid, action, reward = check_state_pair(args, trainer, seg[i: i + 1, j - 1], seg[i: i + 1, j], discrete_action, threshold, normalizer)
                        idx = batch_idx_imbalanced[i]
                        if valid:
                            cur[idx] += 1
                            times_of_regen_list[idx] = 0
                            gen_obs_seq[idx][cur[idx]] = seg[i: i + 1, j: j + 1].cpu()  # seg[i: i + 1, j: j + 1]: (1, 1, ag, obs_dim)
                            gen_action_seq[idx][cur[idx] - 1] = action.cpu()
                            gen_reward_seq[idx][cur[idx] - 1] = reward.cpu()
                        else:
                            path_status[idx] = PathStatus.IN_PROGRESS
                            break
                if verbose:
                    logger.print(f"AFTER PARTIALLY DENOISING cur: {cur}")
        
        cm, ab, ip = tuple(path_status.count(x) for x in (PathStatus.COMPLETED, PathStatus.ABANDONED, PathStatus.IN_PROGRESS))
        logger.print(f"COMPLETED {cm}/{soft_n_gen}\tABANDONED {ab}/{gen_batch_size}\tIN_PROCESS {ip}/{soft_n_gen}\t\
                        Time passed: {str(datetime.datetime.now() - start_time).split('.')[0]}.\n")

        abandoned_idx = [idx for idx in range(soft_n_gen) if path_status[idx] == PathStatus.ABANDONED]
        for idx in abandoned_idx:
            cur[idx] = 0
            times_of_regen_list[idx] = 0
            total_times_of_regen_list[idx] = 0
            sampled_ep_idx = np.random.choice(dataset.n_episodes, 1)
            sampled_obs = dataset.fields.normed_observations[sampled_ep_idx]
            init_obs = sampled_obs[0, 0, :, :]  # (1, 1, ag, obs_dim)
            gen_obs_seq[idx][0] = init_obs
            path_status[idx] = PathStatus.IN_PROGRESS     

    completed_idx = np.array([i for i, v in enumerate(path_status) if v == PathStatus.COMPLETED])[:n_gen]
    # print(gen_obs_seq[completed_idx, :-1, :, :][0])
    # print(gen_obs_seq[completed_idx, :-1, :, :].shape)
    # print(gen_action_seq[completed_idx][0, :, :, 0])
    # print(gen_action_seq[completed_idx].shape)
    # print(gen_reward_seq[completed_idx][0, :, :, 0])
    # print(gen_reward_seq[completed_idx][0, :, 0, 0])
    # print(gen_reward_seq[completed_idx][0, :, 0, 0].sum())
    # print(gen_reward_seq[completed_idx].shape)
    return gen_obs_seq[completed_idx, :-1, :, :], gen_action_seq[completed_idx], gen_reward_seq[completed_idx], gen_avail_seq[completed_idx]
