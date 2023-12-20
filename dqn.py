# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 01:00:13 2023

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
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

from torch.nn import Module, Sequential, Conv2d, ReLU, Flatten, Linear, init
from torch.nn.functional import huber_loss
from torch.nn.utils import clip_grad_norm_

from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.buffers import ReplayBuffer

# os.chdir('/auto/data2/alkilani/rl_2023')


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


def DQL(env, env_name,
        N=1_000_000, n_epochs=5_000_000,
        update_f=4, bs=32,
        gamma=0.99, replay_start_size=50_000,
        initial_explore=1., final_explore=0.1,
        explore_steps=1_000_000, device='cuda',
        seed=0, savepath='', lr_scheduler=False, clip_grads=False) -> None:
    """ DQL - Deep Q-Learning """
    
    # Initialize replay memory D to capacity N
    D = ReplayBuffer(N, env.observation_space, env.action_space, device,
                      optimize_memory_usage=True, handle_timeout_termination=False)

    # Initialize action-value function Q with random weights
    q_net = DQN(env.action_space.n).to(device)
    
    if lr_scheduler:
        optim = torch.optim.RMSprop(
            q_net.parameters(),
            lr=0.01, alpha=0.95, eps=0.01, momentum=0.95
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99999)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=25_000, eta_min=0.00025)
    else:
        optim = torch.optim.RMSprop(
            q_net.parameters(),
            lr=0.00025, alpha=0.95, eps=0.01, momentum=0.95
        )
    
    # Gradient clipping
    if clip_grads:
        max_grad_norm = 1.0
    
    epoch = 0
    
    smoothed_rewards = []
    rewards_list = []
    
    smoothed_q_values = []
    average_q_values = []
    
    progress_bar = tqdm(total=n_epochs)
    
    scheduler_str = '_lr_scheduler' if lr_scheduler else ''
    
    while (epoch <= n_epochs):

        died = False
        
        total_rewards = 0
        max_q_values = []

        # Initialise sequence s_1 = \{x_1\} and preprocessed sequenced φ_1 = φ(s_1)
        seq = env.reset()

        # Fire after reset / "do nothing"
        for _ in range(random.randint(1, 30)):
            seq, _, _, info = env.step(1)
        
        # Play until death
        while not died:
            current_life = info['lives']
            
            # Probability ε, decays with each epoch
            epsilon = max(
                (final_explore - initial_explore) / explore_steps * epoch + initial_explore,
                final_explore
                )
            
            # With probability ε select a random action a.
            if (random.random() < epsilon):
                action = np.array(env.action_space.sample())
            # Otherwise select a = max_a Q^{*}(φ(s_t), a; θ)
            else:
                q_values = q_net(torch.Tensor(seq).unsqueeze(0).to(device))
                action = torch.argmax(q_values, dim=1).item()

            # Execute action a in emulator and observe reward r_t and image x_{t+1}
            next_seq, r, died, info = env.step(action)

            died = (info['lives'] < current_life) or died

            # Set s_{t+1} = s_t, a_t, x_{t+1} and preprocess φ_{t+1} = φ(s_{t+1})
            real_next_seq = next_seq.copy()

            total_rewards += r
            # Reward clipping
            r = np.sign(r)
            
            # Store transition (φ_t, a_t, r_t, φ_{t+1}) in D
            D.add(seq, real_next_seq, action, r, died, info)

            seq = next_seq
            
            # Minibatch updates
            if (epoch >= replay_start_size) and (epoch % update_f == 0):
                # Sample random minibatch of transitions (φ_j , a_j , r_j , φ_{j +1} ) from D
                minibatch = D.sample(bs)
                with torch.no_grad():
                    max_q_value, _ = q_net(minibatch.next_observations).max(dim=1)
                    y = minibatch.rewards.flatten() + gamma * max_q_value * (1 - minibatch.dones.flatten())
                current_q_value = q_net(minibatch.observations).gather(1, minibatch.actions).squeeze()
                loss = huber_loss(y, current_q_value)
                
                # Calculate and append average Q value
                max_q_values.append(current_q_value.max().item())

                # Perform a gradient step according to equation 3 of the paper
                optim.zero_grad()
                # Calculate gradients and perform backward pass
                loss.backward()
                if clip_grads:
                    # Clip gradients to the specified range [-max_grad_norm, max_grad_norm]
                    clip_grad_norm_(q_net.parameters(), max_grad_norm)
                optim.step()
                                
                # Update scheduler
                if lr_scheduler and scheduler.get_last_lr()[-1] > 0.00025:
                    scheduler.step()
                
                # # Update scheduler
                # if lr_scheduler:
                #     scheduler.step(epoch - replay_start_size)

            epoch += 1
            
            # Plot and save
            plot_period = n_epochs // 100
            if (epoch % plot_period == 0) and (epoch > 0):
                smoothed_rewards.append(np.mean(rewards_list))
                rewards_list = []
                
                plt.figure()
                plt.plot(smoothed_rewards)
                plt.title(f'Average Reward on {env_name.title()}')
                plt.xlabel('Training Epochs')
                plt.ylabel('Average Reward per Episode')
                plt.savefig(fr'{savepath}/{env_name.title()}_reward_{seed}{scheduler_str}.png')
                plt.close()
                
                smoothed_q_values.append(np.mean(average_q_values))
                average_q_values = []
                
                plt.figure()
                plt.plot(smoothed_q_values)
                plt.title(f'Average Q on {env_name.title()}')
                plt.xlabel('Training Epochs')
                plt.ylabel('Average Action Value (Q)')
                plt.savefig(fr'{savepath}/{env_name.title()}_Q_{seed}{scheduler_str}.png')
                plt.close()
                
                with open(fr'{savepath}/{env_name.title()}_reward_{seed}{scheduler_str}.npy', 'wb') as f:
                    np.save(f, np.array(smoothed_rewards))
                
                with open(fr'{savepath}/{env_name.title()}_Q_{seed}{scheduler_str}.npy', 'wb') as f:
                    np.save(f, np.array(smoothed_q_values))
                    
                # Save model
                # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended
                torch.save(q_net.state_dict(), fr'{savepath}/{env_name.title()}_{seed}{scheduler_str}.pth')
                
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


def create_env(env_name, k=4, seed=0) -> gym.Env:
    # env = gym.make(f'ALE/{env_name}-v5')
    env = gym.make(f'{env_name}NoFrameskip-v4') # Better behavior
    # env = gym.make(f'{env_name}NoFrameskip-v4', render_mode='human') # Better behavior

    # Replicate paper's conditions (grayscale, frameskip, etc.)
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

def clean_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    # Environments considered in the paper
    env_list = ['BeamRider', 'Breakout', 'Enduro', 'Pong', 'Qbert', 'Seaquest', 'SpaceInvaders']

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)                  
    parser.add_argument('--env', default=env_list[0], choices=env_list)
    parser.add_argument('--savepath', default='Env_Plots')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', default=1_000_000, type=int)
    parser.add_argument('--N', default=750_00, type=int)              
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.add_argument('--clipgrads', action='store_true')
    args = parser.parse_args()     
    
    savepath = os.path.join(os.getcwd(), args.savepath)
    
    # Create folder for saving plots and .npy files, if doesn't already exist
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available. Switching to CPU ...')
        device = 'cpu'
    else:
        device = torch.device(args.device)
        clean_gpu_cache()
    
    # Create env
    # k = 3 if args.env == 'SpaceInvaders' else 4 # Lasers disappear, apparently
    k = 4
    env = create_env(args.env, k=k, seed=args.seed)
    
    print('Environment:', args.env)
    print('Device:', device)
    
    # Maximum N allowed due to mem limits === 750_000 (<1_000_000)
    DQL(env, args.env, n_epochs=args.epochs, N=args.N, update_f=k, device=device, seed=args.seed, savepath=savepath, lr_scheduler=args.lr_scheduler, clip_grads=args.clipgrads)
    
    env.close()
            


