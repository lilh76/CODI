import collections
import os
from typing import Callable, List, Optional

import gym
import numpy as np
import torch
from ddpg_agent import DDPGAgent


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
        # syn_dataset_dir: Pass in `device` as an argument?
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prey = DDPGAgent(
            num_in_pol=env.observation_space[-1].shape[0],
            num_out_pol=env.action_space[-1].shape[0],
            num_in_critic=env.observation_space[-1].shape[0]
            + env.action_space[-1].shape[0],
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
    # if scenario_name in ["simple_tag", "simple_world"]:
    #     env = PretrainedPreyWrapper(env, scenario_name)
    return StackWrapper(env)


def load_environment(name, **kwargs):
    if type(name) is not str:
        # name is already an environment
        return name

    idx = name.find("-")
    env_name, data_split = name[:idx], name[idx + 1 :]

    env = make_env(env_name, **kwargs)
    if hasattr(env, "metadata"):
        assert isinstance(env.metadata, dict)
    else:
        env.metadata = {}
    env.metadata["data_split"] = data_split
    env.metadata["name"] = env_name
    env.metadata["global_feats"] = []

    print("data_split:",data_split, "env_name:", env_name)
    # print(dir(env))
    '''
    ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__enter__', '__eq__', '__exit__', '__format__', '__ge__', 
    '__getattr__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', 
    '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 
    'action_space', 'class_name', 'close', 'compute_reward', 'env', 'metadata', 
    'observation_space', 'render', 'reset', 'reward_range', 'seed', 'spec', 'step', 'unwrapped']
    '''
    # assert 0

    env.action_space = None
    env.class_name = None
    env.close = None
    env.compute_reward = None
    env.env = None
    # env.metadata = None
    env.observation_space = None
    env.render = None
    env.reset = None
    env.reward_range = None
    env.seed = None
    env.step = None

    # print(env.metadata.keys()) # dict_keys(['render.modes', 'data_split', 'name', 'global_feats'])
    env.metadata['data_split'] = None
    env.metadata['name'] = None
    # env.metadata['global_feats'] = None

    return env


def sequence_dataset(env, syn_dataset_dir, preprocess_fn: List[Callable] = [], seed: int = None):

    syn_dataset_dir = ''
    print(f'Loading dataset {syn_dataset_dir} !!!!!')
    print(f'Loading dataset {syn_dataset_dir} !!!!!')
    print(f'Loading dataset {syn_dataset_dir} !!!!!')

    observations = np.load(os.path.join(syn_dataset_dir, "obs.npy"))
    actions = np.load(os.path.join(syn_dataset_dir, "acs.npy"))
    if os.path.exists(os.path.join(syn_dataset_dir, "rew.npy")):
        rewards = np.load(os.path.join(syn_dataset_dir, "rew.npy"))
    else:
        rewards = np.zeros_like(observations)[..., :1]
    if os.path.exists(os.path.join(syn_dataset_dir, "labels.npy")):
        labels = np.load(os.path.join(syn_dataset_dir, "labels.npy"))
    else:
        labels = np.zeros_like(observations)[..., :1]
        
    print(observations.shape)
    print(actions.shape)
    print(rewards.shape)
    print(labels.shape)
        
    dones = np.zeros_like(rewards)
    data_ = collections.defaultdict(list)
    for obs, act, rew, done, label in zip(observations, actions, rewards, dones, labels):
        data_["observations"] = obs
        data_["actions"] = act
        data_["rewards"] = rew
        data_["labels"] = label
        data_["terminals"] = done
        data_["timeouts"] = np.zeros_like(data_["terminals"])
        data_["terminals"][-1][:] = 0.0
        data_["timeouts"][-1][:] = 1.0
        
        episode_data = {}
        for k in data_:
            episode_data[k] = np.array(data_[k])
        yield episode_data
        data_ = collections.defaultdict(list)
