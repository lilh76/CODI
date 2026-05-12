# Imports
import argparse
import gym
import random
import numpy as np
import copy
from scipy.spatial import cKDTree
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from Algorithms import CVAE
from Algorithms import model_trainer
from Algorithms import Value_net
from Algorithms import WGAN_Reward
from Algorithms import Reward_Functions
import copy

from dataload import get_trajs

def normal_pdf(x, mu, sigma):
    dist =Independent(Normal(mu, sigma), 1)
    probability = torch.exp(dist.log_prob(x))
    return probability
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default = "Hopper")    # OpenAI gym environment name to save file
    parser.add_argument("--env", default="hopper-medium-v2")        # OpenAI gym environment name
    parser.add_argument("--diff", default = "Med")              # D4RL difficulty
    parser.add_argument("--seed", default=19636, type=int)   # Sets Gym, PyTorch and Numpy seeds #(19636 ,17,22481)
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--reward_function", default = "MLP", type = str)
    args = parser.parse_args()
    # Load environment
    environment = args.env_name
    if args.diff == 'Med':
        Diff_v = 'Medium'
        Diff_gen = 'Medium'
        Diff_dm = 'Med'
    elif args.diff =='MedExp':
        Diff_v = 'MediumExp'
        Diff_gen ='MediumExpert'
        Diff_dm ='MedExp'
    elif args.diff =='MedRep':
        Diff_v = 'MedRep'
        Diff_gen ='MediumReplay'
        Diff_dm ='MedRep'
    elif args.diff == 'Exp':
        Diff_v = 'Exp'
        Diff_gen = 'Expert'
        Diff_dm = 'Expert'
    env0 = args.env
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # TODO: change this to your dataset path
    original_data_path =  None
    
    
    obss, actions, rewards, dones = get_trajs(original_data_path)


    obss = torch.Tensor(obss).to(device)
    obss = obss.reshape(obss.shape[0], obss.shape[1], obss.shape[2]*obss.shape[3])
    states = obss[:, :-1, :]
    next_states = obss[:, 1:, :]
    actions = torch.Tensor(actions).to(device)
    rewards = torch.Tensor(rewards).to(device)
    dones = torch.Tensor(dones).to(device)

    actions = actions.reshape(actions.shape[0]*actions.shape[1], actions.shape[2]*actions.shape[3])
    states = states.reshape(states.shape[0] * states.shape[1], states.shape[2])
    next_states = next_states.reshape(next_states.shape[0] * next_states.shape[1], next_states.shape[2])
    rewards = rewards.reshape(rewards.shape[0] * rewards.shape[1], rewards.shape[2])
    dones = dones.reshape(dones.shape[0] * dones.shape[1], dones.shape[2])


    dataset_new = [states, actions, rewards, next_states, dones]
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    max_action = 8

    latent_dim = 2*action_dim

    if len(states)< 9000000:
        hidden_dim = 128
        hidden_dim_val = 256
        latent_dim_rew = 2
        hidden_dim_rew=512
        hidden_dim_dm = 200
    else:
        hidden_dim = 750
        hidden_dim_val = 256
        latent_dim_rew = 2
        hidden_dim_rew=512
        hidden_dim_dm = 200

    seed = args.seed
    if seed == 19636:
        s_idx = 1
    elif seed == 17:
        s_idx = 2
    elif seed == 22481:
        s_idx = 3


    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    subset_size = 400



    agent = CVAE.VAE(state_dim, action_dim, latent_dim,max_action, hidden_dim=hidden_dim, device=device).to(device)
    agent.load_state_dict(torch.load(f"PlannerModels/{environment}/{Diff_gen}-VAE", map_location = device))


    if args.reward_function == "MLP":
        reward_function= Reward_Functions.MLP_train(state_dim, action_dim,hidden_dim = 512, batch_size=256,
                                                    device=device)
        reward_function.Rew.load_state_dict(torch.load(f"PlannerModels/{environment}/Rewards_MLP_{Diff_dm}",
                                                       map_location = device))
    elif args.reward_function == "WGAN":
        reward_function = WGAN_Reward.GAN(state_dim, action_dim, latent_dim=2, hidden_dim=512, l2_reg_D=0.0001,
                                          device=device)
        reward_function.discriminator.load_state_dict(
            torch.load(f"PlannerModels/{environment}/{Diff_gen}-Rewards-Discriminator", map_location=device))
        reward_function.generator.load_state_dict(
            torch.load(f"PlannerModels/{environment}/{Diff_gen}-Rewards-Generator",map_location=device))

    dm0 = model_trainer.Dynamics_Model(state_dim,hidden_dim=hidden_dim_dm).to(device)
    dm1 = model_trainer.Dynamics_Model(state_dim,hidden_dim=hidden_dim_dm).to(device)
    dm2 = model_trainer.Dynamics_Model(state_dim,hidden_dim=hidden_dim_dm).to(device)
    dm3 = model_trainer.Dynamics_Model(state_dim,hidden_dim=hidden_dim_dm).to(device)
    dm4 = model_trainer.Dynamics_Model(state_dim,hidden_dim=hidden_dim_dm).to(device)
    dm5 = model_trainer.Dynamics_Model(state_dim,hidden_dim=hidden_dim_dm).to(device)
    dm6 = model_trainer.Dynamics_Model(state_dim,hidden_dim=hidden_dim_dm).to(device)
    dm0.load_state_dict(torch.load(f"PlannerModels/{environment}/state{Diff_dm}DM-0.pt", map_location=device))
    dm1.load_state_dict(torch.load(f"PlannerModels/{environment}/state{Diff_dm}DM-1.pt", map_location=device))
    dm2.load_state_dict(torch.load(f"PlannerModels/{environment}/state{Diff_dm}DM-2.pt", map_location=device))
    dm3.load_state_dict(torch.load(f"PlannerModels/{environment}/state{Diff_dm}DM-3.pt", map_location=device))
    dm4.load_state_dict(torch.load(f"PlannerModels/{environment}/state{Diff_dm}DM-4.pt", map_location=device))
    dm5.load_state_dict(torch.load(f"PlannerModels/{environment}/state{Diff_dm}DM-5.pt", map_location=device))
    dm6.load_state_dict(torch.load(f"PlannerModels/{environment}/state{Diff_dm}DM-6.pt", map_location=device))

    if env0 == 'halfcheetah-medium-v2':
        models = [dm1, dm2, dm4, dm5, dm6]
    elif env0 == 'hopper-medium-v2':
        models = [dm0, dm1, dm3, dm4, dm6]
    elif env0 =='walker2d-medium-v2':
        models = [dm1, dm2, dm4, dm5, dm6]
    # elif env0 == 'halfcheetah-medium-expert-v2':
    #     models = [dm0, dm1, dm2, dm3, dm6]
    elif env0 == 'halfcheetah-mediumexpert-v2':
        models = [dm0, dm1, dm2, dm3, dm6]
    elif env0 == 'hopper-medium-expert-v2':
        models = [dm0, dm3, dm4, dm5, dm6]
    elif env0 =='walker2d-medium-expert-v2':
        models = [dm0, dm2, dm4, dm5, dm6]
    elif env0 == 'hopper-medium-replay-v2':
        models = [dm1, dm3, dm4, dm5, dm6]
    elif env0 == 'halfcheetah-medium-replay-v2':
        models = [dm1, dm2, dm3, dm4, dm5]
    elif env0 == 'walker2d-medium-replay-v2':
        models = [dm0, dm3, dm4, dm5, dm6]
    elif env0 == 'hopper-expert-v2':
        models = [dm0, dm3, dm4, dm5, dm6]
    elif env0 == 'halfcheetah-expert-v2':
        models = [dm0, dm1, dm3, dm4, dm6]
    elif env0 == 'walker2d-expert-v2':
        models = [dm0, dm1, dm3, dm4, dm6]
    else:
        print("Error - check model selection")

    if environment == 'Hopper':
        r_star = 0.1
    elif environment == 'Halfcheetah':
        r_star = 3.0
    elif environment == 'Walker2d':
        r_star = 1.0

    num_traj_changes = []
    k=-1
    while k <5:
        print(f"######################\n    Iteration {k}     \n######################")
        torch.cuda.empty_cache()
        epochs = 10
        iterations = 100
        Value = Value_net.Agent(state_dim,hidden_dim=hidden_dim_val, device=device)
        grad_steps = 0

        if k>-1:
            dataset_new = torch.load(f"Data/{environment}/state{Diff_dm}_new_data_iter{k-1}-S{s_idx}.pt", map_location=device)

            #dataset_new = torch.load(f"Data/{environment}/{Diff_dm}_iter{k - 1}-S{s_idx}-P{p_hat}.pt", map_location=device)
        print("Value function training .....")
        
        for epoch in range(epochs):
            # Training #
            Value.train(dataset_new, iterations)
            grad_steps += iterations

            if epoch % 10 == 0:
                print("Epoch ", epoch, "out of ", epochs)
        print("..... Value function trained ")
        torch.save(Value.value.state_dict(), f"Models/{environment}/{Diff_v}JustValue-S{s_idx}-iter{k}")
        states_np = dataset_new[0].detach().cpu().numpy()
        states = torch.Tensor(states_np).to(device)
        actions_np = dataset_new[1].detach().cpu().numpy()
        actions = torch.Tensor(actions_np).to(device)
        rewards_np = dataset_new[2].detach().cpu().numpy()
        rewards = torch.Tensor(rewards_np).to(device)
        next_states_np = dataset_new[3].detach().cpu().numpy()
        next_states = torch.Tensor(next_states_np).to(device)
        dones_np = dataset_new[4].detach().cpu().numpy()
        dones = torch.Tensor(dones_np).to(device)




        if(states_np.ndim >=3):
            states_np = states_np.reshape(states_np.shape[0]*states_np.shape[1], states_np.shape[2])
        if( next_states_np.ndim >=3):
            next_states_np = next_states_np.reshape(next_states_np.shape[0]*next_states_np.shape[1], next_states_np.shape[2])
        if( actions_np.ndim >=3):
            actions_np = actions_np.reshape(actions_np.shape[0]*actions_np.shape[1], actions_np.shape[2])
        if( rewards_np.ndim >=3):
            rewards_np = rewards_np.reshape(rewards_np.shape[0]*rewards_np.shape[1], rewards_np.shape[2])
        if( dones_np.ndim >=3):
            dones_np = dones_np.reshape(dones_np.shape[0]*dones_np.shape[1],dones_np.shape[2])

        idx_traj = []
        what_traj = []
        traj_scores = []
        tr = 0
        r = 0
        score = 0
        l = 0
        terminals_traj = np.zeros_like(dones_np)
        

        for i in range(len(dataset_new[0])):
            what_traj.append(tr)
            score += rewards_np[i]
            terminals_traj[i] = False
            l += 1
            if dones_np[i] == True:
                idx_traj.append(i + 1)
                tr += 1
                traj_scores.append(score)
                terminals_traj[i] = True
                score = 0
                l = 0

            # elif l == env._max_episode_steps -1:
            elif l == 20000 -1:

                idx_traj.append(i + 1)  # index of last value in array
                terminals_traj[i] = True
                tr += 1
                traj_scores.append(score)
                score = 0
                l = 0

        if k == -1:
            states_np = states_np[0:idx_traj[-1]]
            actions_np = actions_np[0:idx_traj[-1]]
            rewards_np = rewards_np[0:idx_traj[-1]]
            next_states_np = next_states_np[0:idx_traj[-1]]
            dones_np = terminals_traj[0:idx_traj[-1]]
            states = torch.Tensor(states_np).to(device)
            actions = torch.Tensor(actions_np).to(device)
            rewards = torch.Tensor(rewards_np).to(device)
            next_states =torch.Tensor(next_states_np).to(device)
            dones = torch.Tensor(dones_np).to(device)
            dataset_new = [states, actions, rewards, next_states, dones]

        new_state = []
        new_next_state = []
        for i in range(0, len(states_np), 1000*26):
            new_state.extend(states_np[i: i+300*26])
            new_next_state.extend(next_states_np[i: i+300*26])
        print("creating neighbourhood")
        kd_tree1 = cKDTree(states_np[: subset_size * 26])
        kd_tree2 = cKDTree(next_states_np[ :subset_size * 26])
        print("neighbourhood created")

        states_new = torch.empty((1, state_dim)).to(device)
        actions_new = torch.empty((1, action_dim)).to(device)
        rewards_new = torch.empty((1)).to(device)
        next_states_new = torch.empty((1, state_dim)).to(device)
        dones_new = torch.empty((1)).to(device)

        idx_t = 0
        state_base = states[0]
        states_new[0] = state_base
        next_state_base = next_states[0]
        id = 0
        i = 0
        traj_changes = 0
        for t in range(len(idx_traj)):


            states_traj = torch.empty((1001, state_dim)).to(device)
            actions_traj = torch.empty((1001, action_dim)).to(device)
            rewards_traj = torch.empty((1001)).to(device)
            next_states_traj = torch.empty((1001, state_dim)).to(device)
            dones_traj = torch.empty((1001)).to(device)
            fin = 0.0
            len_traj = 0
            n_c_t = 0
            r_traj = 0
            tr_idx = 0
            if t == 0:
                state_base = states[0]
                states_traj[tr_idx] = state_base
                next_state_base = next_states[0]
            elif t > 0:
                id = idx_traj[t - 1]
                state_base = states[id]
                # states_new[i] = state_base
                states_traj[tr_idx] = state_base
                next_state_base = next_states[id]
            



            while not fin:

                indexes1 = kd_tree1.query_ball_point(state_base.detach().cpu().numpy().reshape(1, state_dim), r=r_star)[0]
                indexes2 = kd_tree2.query_ball_point(next_state_base.detach().cpu().numpy().reshape(1, state_dim), r=r_star)[0]
                
                
                indexes = np.concatenate((indexes1, indexes2))
                indexes = indexes.astype(int)

                if len(indexes) == 0:

                    next_states_traj[tr_idx] = next_states[id]
                    actions_traj[tr_idx] = actions[id]
                    rewards_traj[tr_idx] = rewards[id]
                    dones_traj[tr_idx] = dones[id].float()
                    fin = dones[id].float()
                    if id +1 == len(states):
                        state_base = states[id]
                        states_traj[tr_idx + 1] = state_base
                        next_state_base = next_states[id]
                    else:
                        state_base = states[id + 1]
                        states_traj[tr_idx + 1] = state_base
                        next_state_base = next_states[id + 1]
                    r_traj += rewards[id].detach().cpu().item()
                    id += 1
                    i += 1
                    len_traj += 1
                    if len_traj > 25:
                        fin = 1.0
                        if tr_idx + 1 < dones_traj.shape[0]:
                            dones_traj[tr_idx] = 1.
                    if id ==len(states):
                        fin = 1.0
                        dones_traj[tr_idx] = 1.
                    tr_idx += 1
                    continue
                
                neighbour_next_states = np.take(next_states_np, indices=indexes, axis=0)
                nb_ns_tens = torch.from_numpy(neighbour_next_states).to(device)

                state_base_rep = state_base.repeat(len(nb_ns_tens), 1)
                mu0, sigma0 = models[0].forward(state_base)
                mu1, sigma1 = models[1].forward(state_base)
                mu2, sigma2 = models[2].forward(state_base)
                mu3, sigma3 = models[3].forward(state_base)
                mu4, sigma4 = models[4].forward(state_base)
                p_base0 = normal_pdf(next_state_base, mu0, sigma0).detach().cpu().item()
                p_base1 = normal_pdf(next_state_base, mu1, sigma1).detach().cpu().item()
                p_base2 = normal_pdf(next_state_base, mu2, sigma2).detach().cpu().item()
                p_base3 = normal_pdf(next_state_base, mu3, sigma3).detach().cpu().item()
                p_base4 = normal_pdf(next_state_base, mu4, sigma4).detach().cpu().item()
                p_base = np.mean((p_base0, p_base1, p_base2, p_base3, p_base4)) #changed from max to mean

                # print("nb,ns_shape: ", nb_ns_tens.shape)

                p0 = normal_pdf(nb_ns_tens, mu0, sigma0).detach().cpu().numpy()
                p1 = normal_pdf(nb_ns_tens, mu1, sigma1).detach().cpu().numpy()
                p2 = normal_pdf(nb_ns_tens, mu2, sigma2).detach().cpu().numpy()
                p3 = normal_pdf(nb_ns_tens, mu3, sigma3).detach().cpu().numpy()
                p4 = normal_pdf(nb_ns_tens, mu4, sigma4).detach().cpu().numpy()
                p_test = np.min((p0,p1,p2,p3,p4), axis = 0)
                if np.max(p_test) > p_base:  ##Potenially a new state with higher probability of happening

                    idx_neighbours = [idx for idx, v in enumerate(p_test) if v > p_base]
                    new_states_tilde = nb_ns_tens[idx_neighbours]

                    v1_base, v2_base = Value.value(next_state_base)
                    v_base = torch.min(v1_base, v2_base)

                    v1, v2 = Value.value(new_states_tilde)
                    v = torch.min(v1, v2)
                    idx_v = torch.argmax(v)
                    idx_main = indexes[idx_neighbours[idx_v]]

                    z_vae = torch.randn(1, latent_dim, dtype=torch.float32, device=device)
                    z_gan = torch.randn(1, 2, dtype=torch.float32, device=device)
                    st_b = state_base.reshape(1, state_dim)
                    ns_t = new_states_tilde[idx_v].reshape(1, state_dim)
                    # fake_a, fake_r = agent.generator(st_b, ns_t, z)
                    fake_a = agent.decode(st_b, ns_t, z_vae)
                    if args.reward_function == "MLP":
                        fake_r = reward_function.Rew.forward(st_b, fake_a, ns_t)
                    elif args.reward_function == "WGAN":
                        # fake_r = reward_function.generator(st_b, fake_a, ns_t)
                        
                        z_gan = torch.randn(1, 2, dtype=torch.float32, device=device)
                        fake_r = reward_function.generator(st_b, fake_a, ns_t, z_gan)

                    if torch.max(v) > v_base:  # should bias towards seen actions by doing this
                        n_c_t += 1

                        if tr_idx + 1 < next_states_traj.shape[0]:
                            next_states_traj[tr_idx] = new_states_tilde[idx_v]
                            actions_traj[tr_idx] = fake_a
                            rewards_traj[tr_idx] = fake_r
                            dones_traj[tr_idx] = dones[idx_main].float()
                        r_traj += fake_r.detach().cpu().item()
                        
                        fin = dones[idx_main].float()
                    else:
                        if tr_idx + 1 < next_states_traj.shape[0]:
                            actions_traj[tr_idx] = actions[id]
                            rewards_traj[tr_idx] = rewards[id]
                            next_states_traj[tr_idx] = next_states[id]
                            dones_traj[tr_idx] = dones[id].float()
                        r_traj += rewards[id].detach().cpu().item()
                        
                        fin = dones[id].float()
                        idx_main = id
                    if idx_main+1 == len(states):
                        fin = 1.0
                        dones_traj[tr_idx] = 1.0
                    else:
                        state_base = states[idx_main + 1]

                        if tr_idx + 1 < states_traj.shape[0]:
                            states_traj[tr_idx + 1] = state_base

                        next_state_base = next_states[idx_main + 1]
                        id = idx_main + 1
                        i += 1
                        len_traj += 1
                else:
                    next_states_traj[tr_idx] = next_states[id]
                    actions_traj[tr_idx] = actions[id]
                    rewards_traj[tr_idx] = rewards[id]
                    dones_traj[tr_idx] = dones[id].float()
                    fin = dones[id].float()
                    if id +1 == len(states):
                        state_base = states[id]#doesn't matter as this will be deleted later
                        states_traj[tr_idx + 1] = state_base
                        next_state_base = next_states[id]#doesn't matter as this will be deleted later
                    else:
                        state_base = states[id + 1]
                        states_traj[tr_idx + 1] = state_base
                        next_state_base = next_states[id + 1]
                    r_traj += rewards[id].detach().cpu().item()
                    id += 1
                    i += 1
                    len_traj += 1
                # print("len_traj", len_traj)
                # if len_traj > env._max_episode_steps -2:
                # if len_traj > 20000 -2:
                if len_traj > 25:
                    fin = 1.0
                    if tr_idx + 1 < dones_traj.shape[0]:
                        dones_traj[tr_idx] = 1.
                if id ==len(states):
                    fin = 1.0
                    dones_traj[tr_idx] = 1.
                tr_idx += 1
            print("Trajectory", t, "out of ", len(idx_traj), "has new length", len_traj, "number of changes", n_c_t,
                 "\n Score of traj",np.round(r_traj,2),"Score difference between new and old trajectory", np.round(r_traj - traj_scores[t], 2))
            



            # if r_traj > traj_scores[t]+0.1*np.abs(traj_scores[t]): # we need to see a 10% improvement
            if r_traj > traj_scores[t] + 0.01*np.abs(traj_scores[t]): 
                print("Trajectory", t, "out of ", len(idx_traj), "has new length", len_traj, "number of changes", n_c_t,
                     "\n Score of traj",np.round(r_traj,2),"Score difference between new and old trajectory", np.round(r_traj - traj_scores[t], 2))
                traj_changes += 1
                # concatenate new trajector into states tensor
                states_traj = states_traj[:len_traj]
                actions_traj = actions_traj[:len_traj]
                rewards_traj = rewards_traj[:len_traj]
                next_states_traj = next_states_traj[:len_traj]
                dones_traj = dones_traj[:len_traj]

                if (len(rewards_new.shape)) ==2:
                    rewards_new = rewards_new.squeeze(-1)
                if (len(dones_new.shape)) ==2:
                    dones_new = dones_new.squeeze(-1)

                states_new = torch.cat((states_new, states_traj), 0)
                actions_new = torch.cat((actions_new, actions_traj), 0)
                rewards_new = torch.cat((rewards_new, rewards_traj), 0)
                next_states_new = torch.cat((next_states_new, next_states_traj), 0)
                dones_new = torch.cat((dones_new, dones_traj), 0)
            else:
                # concatenate old tensor into states tensor
                if t == 0:

                    if (len(rewards_new.shape)) ==1:
                        rewards_new = rewards_new.unsqueeze(-1)
                    if (len(dones_new.shape)) ==1:
                        dones_new = dones_new.unsqueeze(-1)

                    rewards = rewards.reshape(-1, 1)
                    dones = dones.reshape(-1, 1)

                    states_new = torch.cat((states_new, states[0: idx_traj[0]]), 0)
                    actions_new = torch.cat((actions_new, actions[0: idx_traj[0] ]), 0)
                    rewards_new = torch.cat((rewards_new, rewards[0: idx_traj[0] ]), 0)
                    next_states_new = torch.cat((next_states_new, next_states[0: idx_traj[0] ]), 0)
                    dones_new = torch.cat((dones_new, dones[0: idx_traj[0] ]), 0)
                else:
                    
                    if (len(rewards_new.shape)) ==1:
                        rewards_new = rewards_new.unsqueeze(-1)
                    if (len(dones_new.shape)) ==1:
                        dones_new = dones_new.unsqueeze(-1)

                    rewards = rewards.reshape(-1, 1)
                    dones = dones.reshape(-1, 1)

                    states_new = torch.cat((states_new, states[idx_traj[t - 1] : idx_traj[t] ]), 0)
                    actions_new = torch.cat((actions_new, actions[idx_traj[t - 1] : idx_traj[t] ]), 0)
                    rewards_new = torch.cat((rewards_new, rewards[idx_traj[t - 1]: idx_traj[t] ]), 0)
                    next_states_new = torch.cat((next_states_new, next_states[idx_traj[t - 1] : idx_traj[t] ]), 0)
                    dones_new = torch.cat((dones_new, dones[idx_traj[t - 1] : idx_traj[t] ]), 0)

            # print("one traj")

            if dones_new[-1] != True:
                print('Error final done is False')
        eps = traj_changes
        print(traj_changes, "out of", len(idx_traj), "trajectories were changed")
        num_traj_changes.append(traj_changes)
        states_new = states_new[1:].detach().cpu()
        actions_new = actions_new[1:].detach().cpu()
        rewards_new = rewards_new[1:].detach().cpu()
        next_states_new = next_states_new[1:].detach().cpu()
        dones_new = dones_new[1:].detach().cpu()
        dataset_new = [states_new, actions_new, rewards_new, next_states_new, dones_new]
        torch.save(dataset_new,f"Data/{environment}/state{Diff_dm}_new_data_iter{k}-S{s_idx}.pt")
        # torch.save(dataset_new, f"Data/{environment}/{Diff_dm}_iter{k}-S{s_idx}-P{p_hat}.pt")
        k+=1
    print("Trajectory changes", num_traj_changes)
