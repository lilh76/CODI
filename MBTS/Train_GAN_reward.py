# Imports
import gym
import random
import numpy as np
import copy
import pickle
import os
import torch
# import d4rl
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from Algorithms import WGAN_Reward

from dataload import get_trajs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default = "Hopper")    # OpenAI gym environment name to save file
    parser.add_argument("--env", default="hopper-medium-v2")        # OpenAI gym environment name
    parser.add_argument("--diff", default = "Med")              # D4RL difficulty
    parser.add_argument("--seed", default=42, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--latent_dim", default = 64, type = int)   #laten dim for gan
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--lr", default = 3e-4)            # learning rate of models
    parser.add_argument("--epochs", default = 30000, type =int)   #Number of epochs
    parser.add_argument("--iterations", default = 1, type = int)   #Number of iterations
    parser.add_argument("--l2_reg_D", default = 0.0001)
    args = parser.parse_args()
    if not os.path.exists("./PlannerModels"):
        os.makedirs("./PlannerModels")

    print("Inputs:", args.env_name, args.env, args.diff)
    if args.diff == 'Rand':
        file_name = 'Random'
    elif args.diff =='MedRep':
        file_name = 'MediumReplay'
    elif args.diff =='MedExp':
        file_name ='MediumExpert'
    elif args.diff == 'Med':
        file_name = 'Medium'
    else:
        file_name = args.diff

    environment = args.env_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    # TODO: change this to your dataset path
    original_data_path =  None

    obss, actions, rewards, dones = get_trajs(original_data_path)
    obss = torch.Tensor(obss).to(device)
    obss = obss.reshape(obss.shape[0] , obss.shape[1], obss.shape[2]*obss.shape[3])
    actions = actions.reshape(actions.shape[0] ,  actions.shape[1], actions.shape[2]*actions.shape[3])
    states = obss[:, :-1, :]
    next_states = obss[:, 1:, :]

    states = states.reshape(states.shape[0]*(states.shape[1]), states.shape[2])
    next_states = next_states.reshape(next_states.shape[0]*(next_states.shape[1]), next_states.shape[2])
    actions = actions.reshape(actions.shape[0]*(actions.shape[1]), actions.shape[2])
    rewards = rewards.reshape(rewards.shape[0]*(rewards.shape[1]),rewards.shape[2])

    actions = torch.Tensor(actions).to(device)
    rewards = torch.Tensor(rewards).to(device)
    dones = torch.Tensor(dones).to(device)

    replay_buffer = [states, actions, rewards, next_states, dones]
    print("...data conversion complete")

    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    max_action = 8

    hidden_dim = 512
    latent_dim = 2
    epochs = 30000
    iterations = 1

    agent = WGAN_Reward.GAN(state_dim, action_dim, latent_dim,hidden_dim = hidden_dim,  l2_reg_D=0.0001, device=device)

    grad_steps = 0

    discriminator_test_loss = []
    generator_test_loss = []
    gen_accuracy = []
    for epoch in range(epochs):
        # Training #
        agent.train(replay_buffer, iterations)
        grad_steps += iterations

        if epoch % 10000 == 0 :
            print("Epoch ", epoch, " out of ", epochs, "Generator loss = ",
                  np.round(agent.generator_loss_history[epoch],6), "Discriminator loss = ",
                  np.round(agent.discriminator_loss_history[epoch],6)) # "Test Accuracy = ", np.round(test_acc,6),


    gen_loss = agent.generator_loss_history
    dis_loss = agent.discriminator_loss_history

    torch.save(agent.discriminator.state_dict(), f"PlannerModels/{environment}/{file_name}-Rewards-Discriminator")
    torch.save(agent.generator.state_dict(), f"PlannerModels/{environment}/{file_name}-Rewards-Generator")

