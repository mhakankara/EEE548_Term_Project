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

from torch.nn import Module, Sequential, Conv2d, ReLU, Flatten, Linear
from torch.nn.functional import huber_loss

from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.buffers import ReplayBuffer


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

    def forward(self, x):
        return self.network(x / 255.)


def DQL(env, env_name,
        N=1_000_000, n_epochs=5_000_000,
        update_f=4, bs=32,
        gamma=0.99, replay_start_size=50_000,
        initial_explore=1., final_explore=0.1,
        explore_steps=1_000_000, device='cuda',
        seed=0, savepath='.\Env_Plots', lr_scheduler=False) -> None:
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
    else:
        optim = torch.optim.RMSprop(
            q_net.parameters(),
            lr=0.00025, alpha=0.95, eps=0.01, momentum=0.95
        )
    
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

            done = True if (info['lives'] < current_life) else False

            # Set s_{t+1} = s_t, a_t, x_{t+1} and preprocess φ_{t+1} = φ(s_{t+1})
            real_next_seq = next_seq.copy()

            total_rewards += r
            # Reward clipping
            r = np.sign(r)
            
            # Store transition (φ_t, a_t, r_t, φ_{t+1}) in D
            D.add(seq, real_next_seq, action, r, done, info)

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
                loss.backward()
                optim.step()
                
                # Update scheduler
                if lr_scheduler and scheduler.get_last_lr() > 0.00025:
                    scheduler.step()

            epoch += 1
            
            # Plot and save
            if (epoch % 50_000 == 0) and (epoch > 0):
                smoothed_rewards.append(np.mean(rewards_list))
                rewards_list = []
                
                plt.figure()
                plt.plot(smoothed_rewards)
                plt.title(f'Average Reward on {env_name.title()}')
                plt.xlabel('Training Epochs')
                plt.ylabel('Average Reward per Episode')
                plt.savefig(fr'{savepath}\{env_name.title()}_reward_{seed}{scheduler_str}.png')
                plt.close()
                
                smoothed_q_values.append(np.mean(average_q_values))
                average_q_values = []
                
                plt.figure()
                plt.plot(smoothed_q_values)
                plt.title(f'Average Q on {env_name.title()}')
                plt.xlabel('Training Epochs')
                plt.ylabel('Average Action Value (Q)')
                plt.savefig(fr'{savepath}\{env_name.title()}_Q_{seed}{scheduler_str}.png')
                plt.close()
                
                with open(fr'{savepath}\{env_name.title()}_reward_{seed}{scheduler_str}.npy', 'wb') as f:
                    np.save(f, np.array(smoothed_rewards))
                
                with open(fr'{savepath}\{env_name.title()}_Q_{seed}{scheduler_str}.npy', 'wb') as f:
                    np.save(f, np.array(smoothed_q_values))
                    
                # Save model
                # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended
                torch.save(q_net.state_dict(), fr'{savepath}\{env_name.title()}_{seed}{scheduler_str}.pth')
                
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

def create_env(env_name, k=4, seed=0) -> gym.Env:
    # env = gym.make(f'ALE/{env_name}-v5')
    env = gym.make(f'{env_name}NoFrameskip-v4') # Better behavior
    # env = gym.make(f'{env_name}NoFrameskip-v4', render_mode='human') # Better behavior

    
    # Replicate paper's conditions (grayscale, frameskip, etc.)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, k)
    env = MaxAndSkipEnv(env, skip=k)
    
    # Fix initial seed
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.random.seed(seed)
    torch.manual_seed(seed)
    
    return env

if __name__ == '__main__':
    # Environments considered in the paper
    env_list = ['BeamRider', 'Breakout', 'Enduro', 'Pong', 'Qbert', 'Seaquest', 'SpaceInvaders']

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)                  
    parser.add_argument('--env', default=env_list[0], choices=env_list)
    parser.add_argument('--savepath', default='.\Env_Plots')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--N', default=750_00, type=int)                  
    parser.add_argument('--lr_scheduler', action='store_true')
    args = parser.parse_args()     
    
    # Create folder for saving plots and .npy files, if doesn't already exist
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available. Switching to CPU ...')
        device = 'cpu'
    else:
        device = args.device
    
    # Create env
    k = 3 if args.env == 'SpaceInvaders' else 4 # Lasers disappear, apparently
    env = create_env(args.env, k=k, seed=args.seed)
    
    print('Environment:', args.env)
    print('Device:', device)
    
    # Maximum N allowed due to mem limits === 750_000 (<1_000_000)
    DQL(env, args.env, N=args.N, update_f=k, device=device, seed=args.seed, savepath=args.savepath, lr_scheduler=args.lr_scheduler)
    
    env.close()
            


