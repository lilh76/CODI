# Import modules
import gym
import random
import numpy as np
import time
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.distributions.normal import Normal

from Algorithms import model_trainer

from dataload import get_trajs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default = "Hopper", type = str)    # OpenAI gym environment name to save file
    parser.add_argument("--env", default="hopper-medium-v2", type = str)        # OpenAI gym environment name
    parser.add_argument("--diff", default = "MedRep", type =str)              # D4RL difficulty
    parser.add_argument("--batch_size", default=32, type=int)      # Batch size for both actor and critic
    parser.add_argument("--lr", default = 3e-4, type = float)            # learning rate of models
    parser.add_argument("--max_epochs", default = 3, type = int)   #Number of epochs
    parser.add_argument("--patience", default = 50, type = int)   #num of iterations to wait for a lower loss before halting

    args = parser.parse_args()

    if not os.path.exists("./PlannerModels"):
        os.makedirs("./PlannerModels")
    # Load D4Rl dataset
    environment = args.env_name
    Diff = args.diff
    # env = gym.make(args.env)
    # dataset = d4rl.qlearning_dataset(env)

    # TODO: change this to your dataset path
    original_data_path =  None
    
    obss, actions, rewards, dones = get_trajs(original_data_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obss = torch.Tensor(obss).to(device)
    obss = obss.reshape(obss.shape[0] , obss.shape[1], obss.shape[2]*obss.shape[3])
    actions = actions.reshape(actions.shape[0] ,  actions.shape[1], actions.shape[2]*actions.shape[3])
    states = obss[:, :-1, :]
    next_states = obss[:, 1:, :]

    states = states.reshape(states.shape[0]*(states.shape[1]), states.shape[2])
    next_states = next_states.reshape(next_states.shape[0]*(next_states.shape[1]), next_states.shape[2])
    actions = actions.reshape(actions.shape[0]*(actions.shape[1]), actions.shape[2])

    actions = torch.Tensor(actions).to(device)
    rewards = torch.Tensor(rewards).to(device)
    dones = torch.Tensor(dones).to(device)
    # Set parameters
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    batch_size = args.batch_size
    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    max_action = 8

    lr_model = args.lr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    max_epochs = args.max_epochs
    patience = args.patience

    num_models = 7
    seeds = np.array((977748, 585286, 741286, 148614, 409960, 466256, 93834))
    validation_losses = []
    for model_id in range(num_models):
        # Set seeds
        seed = seeds[model_id].item()
        # env.seed(seed)
        # env.action_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Convert data to DataLoader
        print("Pre processing data...")
        replay_buffer_env = []
        # for i in range(len(dataset["observations"])):
        # 	replay_buffer_env.append((dataset["observations"][i], dataset["actions"][i], dataset["rewards"][i],
        # 							  dataset["next_observations"][i] , dataset["terminals"][i]))
        for i in range(len(states)):
            replay_buffer_env.append((states, actions, rewards,
                                        next_states , dones))
        random.shuffle(replay_buffer_env)
        replay_buffer_train = data.DataLoader(replay_buffer_env[0: int(0.8 * len(replay_buffer_env))], batch_size=batch_size, shuffle=True)
        replay_buffer_val = data.DataLoader(replay_buffer_env[int(0.8 * len(replay_buffer_env)):len(replay_buffer_env)], batch_size=batch_size, shuffle=True)
        print("data processed!")
        # Initialise dynamics model
        model = model_trainer.Dynamics_Model(state_dim).to(device)
        model_optimizer = torch.optim.Adam(model.parameters(), lr=lr_model)
        # Train (save best model based on validation loss, stop training after no improvement for patience)
        val_loss = []
        start = time.time()
        for epoch in range(max_epochs):
            

            t, v = model_trainer.train_model(model, model_optimizer, replay_buffer_train, replay_buffer_val, device)
            val_loss.append(v)
            if v == np.min(val_loss):
                torch.save(model.state_dict( keep_vars = True), f"./PlannerModels/{environment}/state{Diff}DM-%s.pt" % model_id)
                best_model = epoch
                v_best = v
                print("Model improvement...saved", "Epoch", best_model, "Loss %.4f" % v_best)
            if np.sum(val_loss[-patience:] <= np.min(val_loss)) < 1:
                print("No improvement for", patience,  "updates.  Model best at epoch", best_model, "Model loss %.4f" % v_best)
                break
            print("Model", model_id, "Epoch", epoch, "Current loss %.4f" % v, "Best loss %.4f" % v_best)

        end = time.time()
        print("Total model training time", end-start)
        validation_losses.append(v_best)
    print(np.round(validation_losses, 3))
