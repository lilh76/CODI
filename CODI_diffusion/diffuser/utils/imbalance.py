import numpy as np
import torch as th

from .arrays import to_device, to_np, to_torch


# unroll only one path
def unroll_s_a_path(path, each_ig_step_num):
    """
    Now the full path covers from S_0 to S_T, and ig_step_num = battle_limit * each_ig_step_num
    full path = S_0, P_01, P_02, ..., P_0(n-1), S_1, P_11, ..., S_(T-1), P(T-1)1, ..., P(T-1)(n-1), S_T
    :param path:
    :return:
    """
    unrolled_path = []
    for pos, next_pos in zip(path[:-1], path[1:]):
        pos, next_pos = np.asarray(pos), np.asarray(next_pos)
        step_sizes = (next_pos - pos) / each_ig_step_num
        each_unrolled_s_a_path = [step_sizes * i_step + pos for i_step in range(each_ig_step_num)]
        unrolled_path += each_unrolled_s_a_path
    unrolled_path.append(path[-1])  # Next pos here means the terminated state
    step_sizes = np.asarray(unrolled_path[0:-1]) - np.asarray(unrolled_path[1:])
    # print('---step size shape is {}---'.format(step_sizes.shape))
    unrolled_path = np.asarray(unrolled_path[0:-1])
    return unrolled_path, step_sizes


def get_integrated_gradients(s_a_path, trainer):
    # s_a_path: (seq, na * dim), rwd_model: na * dim -> 1
    # print(s_a_path.shape)
    # assert 0
    team_reward = trainer.ema_model.rwd_model(s_a_path)
    grads = th.autograd.grad(team_reward, s_a_path, th.ones_like(team_reward))
    return grads[0]


def check_imbalance(args, trainer, cur_obs_seq, cur_action_seq, each_ig_step_num=10):
    # cur_obs_seq: list of (bs, 1, na, obs_dim)
    # rwd model: (bs * seq, na * (action_dim + obs_dim)) -> (bs * seq, 1)
    cur_obs_seq = np.concatenate(cur_obs_seq, axis=1)
    cur_action_seq = np.concatenate(cur_action_seq, axis=1)
    ao_comb = np.concatenate([cur_action_seq, cur_obs_seq], axis=-1)  # (bs, seq, na, dim)
    n_agents = ao_comb.shape[2]

    bad_agent = []
    randomly_check = False
    if randomly_check:
        if np.random.rand() < 0.99:
            numbers = range(cur_obs_seq.shape[2])
            random_numbers = list(np.random.choice(numbers, size=1 if np.random.rand() < 0.5 else 2, replace=False))
            return random_numbers
        else:
            return []
    else:
        target_q_val = []
        # (bs, seq, na, dim) -> (seq, na * dim)
        for raw_path in ao_comb:
            raw_path = raw_path.reshape(raw_path.shape[0], -1)
            unrolled_full_path, full_step_size = unroll_s_a_path(raw_path, each_ig_step_num)
            unrolled_full_path, full_step_size = to_torch(unrolled_full_path, device=args.device), to_torch(full_step_size, device=args.device)
            unrolled_full_path.requires_grad_()
            ex = get_integrated_gradients(unrolled_full_path, trainer)
            q_ex = ex.clone().detach()
            for i in range(1, full_step):
                q_ex[:-i * each_ig_step_num] = q_ex[:-i * each_ig_step_num] + (0.99 ** i) + ex[i * each_ig_step_num:]
            ex = q_ex

            ex = ex * full_step_size
            ex = to_np(ex)
            full_step = len(raw_path) - 1
            for loc in range(full_step):
                agent_ex = ex[loc * each_ig_step_num:]
                agent_ex = np.sum(agent_ex, axis=0)
                agent_ex = np.reshape(agent_ex, (n_agents, -1))
                target_q_val.append(np.sum(agent_ex, axis=1))

            agent_rank = np.array([list(np.argsort(-vals)) for vals in target_q_val])
            agent_rank_mean = np.mean(agent_rank, axis=0)
            bad_agent.append(np.where(agent_rank_mean > 4 / 3)[0].tolist())
        return bad_agent


def check_imbalance_parallel(args, trainer, obs_seq, action_seq, discrete_action, each_ig_step_num):
    # list of array (seq, na, dim)
    ao_comb = []
    # print(obs_seq[0].shape) # [h2, n_agents, obs_shape]
    # print(action_seq[0].shape) # [h2, n_agents, 1]
    for i in range(len(obs_seq)):
        action2bconcat = action_seq[i] # [h2, n_agents, 1]
        if discrete_action:
            action2bconcat = np.array(th.nn.functional.one_hot(th.squeeze(th.from_numpy(action_seq[i]), -1).to(th.int64), num_classes=trainer.model.num_actions).type(th.FloatTensor))
        h2 = action2bconcat.shape[0]
        if obs_seq[i].shape[0] > action2bconcat.shape[0]:
            assert obs_seq[i].shape[0] == action2bconcat.shape[0] + 1
            # ao = np.concatenate([action2bconcat, obs_seq[i][:-1, ...]], axis=-1)
            ao = np.concatenate([action2bconcat, obs_seq[i][:-1, ...], obs_seq[i][1:, ...]], axis=-1)
        else:
            next_obs = np.concatenate([
                obs_seq[i][1:], 
                np.zeros_like(obs_seq[i][-1:])
            ], axis=0)
            ao = np.concatenate([action2bconcat, obs_seq[i], next_obs], axis=-1)
        ao_comb.append(ao)
    n_agents = ao_comb[0].shape[1]

    bad_agent = []
    randomly_check = False
    if randomly_check:
        if np.random.rand() < 0.5:
            numbers = range(obs_seq[0].shape[1])
            random_numbers = list(np.random.choice(numbers, size=1 if np.random.rand() < 0.5 else 2, replace=False))
            return random_numbers
        else:
            return []
    else:
        # (bs, seq, na, dim) -> (seq, na * dim)
        for raw_path in ao_comb:
            raw_path = raw_path.reshape(raw_path.shape[0], -1) # [h2, n_agents, dim] -> [h2, n_agents * dim]
            if len(raw_path) == 1:
                bad_agent.append([])
                continue
            unrolled_full_path, full_step_size = unroll_s_a_path(raw_path, each_ig_step_num)
            # print(unrolled_full_path.shape) # [10?, n_agents * dim]
            unrolled_full_path, full_step_size = to_torch(unrolled_full_path, device=args.device), to_torch(full_step_size, device=args.device)
            unrolled_full_path.requires_grad_()
            ex = get_integrated_gradients(unrolled_full_path, trainer)

            ex = ex * full_step_size
            ex = to_np(ex)
            full_step = len(raw_path) - 1
            target_val = []
            for loc in range(full_step):
                agent_ex = ex[loc * each_ig_step_num:]
                agent_ex = np.sum(agent_ex, axis=0)
                agent_ex = np.reshape(agent_ex, (n_agents, -1))
                target_val.append(np.sum(agent_ex, axis=1))

            agent_rank = np.array([list(np.argsort(-vals)) for vals in target_val])
            agent_rank_mean = np.mean(agent_rank, axis=0)
            bad_agent.append(np.where(agent_rank_mean > 4 / 3)[0].tolist())
        return bad_agent
