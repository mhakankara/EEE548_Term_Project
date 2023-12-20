# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 20:49:08 2023


@author: Abdallah

Implementation/Recreation attempt of the seminal paper:
- "Playing Atari with Deep Reinforcement Learning"
- (Main) Paper: https://arxiv.org/pdf/1312.5602.pdf
- Also Helpful: https://doi.org/10.1038/nature14236

The implementation is Torch-based and relies on OpenAI's Gymnasium for ATARI game simulation.
- https://pytorch.org/get-started/previous-versions/#v1101

Stables Baselines 3's ReplayBuffer and MaxAndSkipEnv are used for generalizability/reproducibility.
- http://jmlr.org/papers/v22/20-1364.html
"""

import os
import argparse
import gym
import torch
import random
import numpy as np
from tqdm import tqdm

from torch.nn import Module, Sequential, Conv2d, ReLU, Flatten, Linear, init

from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv


class DQN(Module):
    """ DQN - Deep Q-Network Definition """
    def __init__(self, n_actions):
        super().__init__()
        self.network = Sequential(
            Conv2d(4, 16, 8, stride=4),
            ReLU(),
            
            Conv2d(16, 32, 4, stride=2),
            ReLU(),
            
            Flatten(),
            Linear(2592, 256),
            ReLU(),
            
            Linear(256, n_actions)
            )
        
        # Initialize the layers with Xavier Initialization
        for layer in self.network:
            if isinstance(layer, (Conv2d, Linear)):
                init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.network(x / 255.)


def Eval(env, env_name,
         weights_path='', n_steps=10_000,
         bs=32, gamma=0.99, epsilon=0.05, seed=0,
         device='cuda', savepath='Eval_Plots') -> None:
    """ Evaluate DQN """
    
    # Initialize action-value function Q with random weights
    q_net = DQN(env.action_space.n).to(device)
    q_net.load_state_dict(torch.load(weights_path))
    
    epoch = 0
    
    rewards_list = []
    
    average_q_values = []
    
    progress_bar = tqdm(total=n_steps)
    
    while (epoch <= n_steps):

        died = False
        
        total_rewards = 0
        max_q_values = []

        # Initialise sequence s_1 = \{x_1\} and preprocessed sequenced φ_1 = φ(s_1)
        seq = env.reset()

        # Fire after reset / "do nothing"
        for _ in range(random.randint(1, 30)):
            seq, _, _, info = env.step(1)
        
        # Play until death
        while not died and (epoch <= n_steps):
            current_life = info['lives']
                        
            # With probability ε select a random action a.
            q_values = q_net(torch.Tensor(seq).unsqueeze(0).to(device))
            if (random.random() < epsilon):
                action = np.array(env.action_space.sample())
            # Otherwise select a = max_a Q^{*}(φ(s_t), a; θ)
            else:
                action = torch.argmax(q_values, dim=1).item()

            # Calculate and append average Q value
            max_q_values.append(q_values.max().item())

            # Execute action a in emulator and observe reward r_t and image x_{t+1}
            next_seq, r, died, info = env.step(action)

            died = True if (info['lives'] < current_life) else False

            total_rewards += r
            
            seq = next_seq
            
            epoch += 1
            
            progress_bar.update()
        
        # Append results if not NaN/None/[]
        if not total_rewards:
            rewards_list.append(0.)
        else:
            rewards_list.append(total_rewards)
            
        if not max_q_values:
            average_q_values.append(0.)
        else:
            average_q_values.append(np.mean(max_q_values))
            
    with open(fr'{savepath}\{env_name.title()}_reward_{seed}_EVAL.npy', 'wb') as f:
        np.save(f, np.array(rewards_list))
    
    with open(fr'{savepath}\{env_name.title()}_Q_{seed}_EVAL.npy', 'wb') as f:
        np.save(f, np.array(average_q_values))
    
    with open(fr'{savepath}\summary.txt', 'a') as f:
        f.write(f'\n{env_name.title()} seed={seed} averages for evaluation: \tReward = {np.mean(rewards_list):.10} \tQ = {np.mean(average_q_values):.10}\n')
        
    return


class AtariCropWrapper(gym.Wrapper):
    def __init__(self, env, game_name):
        super(AtariCropWrapper, self).__init__(env)
        self.game_name = game_name
        self.crop_values = self.get_crop_values(game_name)
        
        # Determine the dimensions after cropping
        self.crop_height = self.crop_values[2] - self.crop_values[0]
        self.crop_width = self.crop_values[3] - self.crop_values[1]
        
        # Set the observation space to the cropped dimensions
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.crop_height, self.crop_width, 3), dtype=np.uint8)

    def get_crop_values(self, game_name):
        # Define crop values for different Atari games
        crop_values = {
            'BeamRider': (30, 5, 195, 160),
            'Breakout': (30, 5, 195, 160),
            'Enduro': (30, 10, 195, 150),
            'Pong': (35, 5, 190, 160),
            'Qbert': (30, 5, 190, 160),
            'Seaquest': (30, 10, 190, 160),
            'SpaceInvaders': (30, 5, 190, 160),
        }
        return crop_values.get(game_name, (0, 0, 210, 160))  # Default to (0, 0, 210, 160) if game_name not found

    def preprocess_observation(self, observation):
        # Crop observation without converting to grayscale
        cropped = observation[self.crop_values[0]:self.crop_values[2], self.crop_values[1]:self.crop_values[3], :]
        return cropped
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.preprocess_observation(observation), reward, done, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        observation = self.env.reset()
        return self.preprocess_observation(observation)


def create_env(env_name, k=4, seed=0, crop=False) -> gym.Env:
    # env = gym.make(f'ALE/{env_name}-v5')
    env = gym.make(f'{env_name}NoFrameskip-v4') # Better behavior
    # env = gym.make(f'{env_name}NoFrameskip-v4', render_mode='human') # Better behavior

    # Replicate paper's conditions (grayscale, frameskip, etc.)
    if crop:
        env = AtariCropWrapper(env, env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, k)
    env = MaxAndSkipEnv(env, skip=k)
    
    # Fix initial seed
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    return env

if __name__ == '__main__':
    # Environments considered in the paper
    env_list = ['BeamRider', 'Breakout', 'Enduro', 'Pong', 'Qbert', 'Seaquest', 'SpaceInvaders']

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)                  
    parser.add_argument('--env', default=env_list[0], choices=env_list)
    parser.add_argument('--savepath', default='')
    parser.add_argument('--weights', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', default=10_000, type=int)
    parser.add_argument('--epsilon', default=0.05, type=float)
    parser.add_argument('--cropenv', action='store_true')
    args = parser.parse_args()     
    
    savepath = os.path.join(os.getcwd(), args.savepath, 'EVAL')
    weights = os.path.join(os.getcwd(), *args.weights.split('/'))
    
    # Create folder for saving plots and .npy files, if doesn't already exist
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available. Switching to CPU ...')
        device = 'cpu'
    else:
        device = torch.device(args.device)
    
    # Create env
    # k = 3 if args.env == 'SpaceInvaders' else 4 # Lasers disappear, apparently
    k = 4
    env = create_env(args.env, k=k, seed=args.seed, crop=args.cropenv)
    
    print('Environment:', args.env)
    print('Device:', device)
    
    # Maximum N allowed due to mem limits === 750_000 (<1_000_000)
    Eval(env, args.env, 
         weights_path=weights, n_steps=args.steps,
         seed=args.seed, savepath=savepath,
         device=device, epsilon=args.epsilon)
    
    env.close()    
