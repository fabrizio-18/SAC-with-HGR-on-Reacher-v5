import argparse
import numpy as np
import gymnasium as gym
import warnings
import torch
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
from utils import NormalizedActions, plot_success_rates
from policy import *


#plot_success_rates('HER_buffer_plots/success_rates.npy', 'HGR_buffer_plots/success_rates.npy')
def evaluate(env=None, n_episodes=1000, render=False):
    agent = Policy()
    agent.load('HGR_buffer_model.pt')
    env = NormalizedActions(gym.make("Reacher-v5", max_episode_steps=50))
    if render:
        env = NormalizedActions(gym.make("Reacher-v5", render_mode="human", max_episode_steps=50))
        
    #print(env.action_space.shape[0], env.observation_space.shape[0])
    rewards = []
    successes = []
    for episode in range(n_episodes):
        total_reward = 0
        done = False
        s, _ = env.reset()
        g = [s[4], s[5]]
        env.render()  
        while not done:
            s = torch.tensor(s, dtype=torch.float32)
            with eval_mode(agent.sac):
                action = agent.sac.actor.act(s, g, True)
            #print(action)
            s, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            total_reward += reward
        success = agent.isSuccess(info=info)
        successes.append(int(success))
        rewards.append(total_reward)
    env.close()
    print(f'Mean Reward: {np.mean(rewards):.4f}, Success Rate: {np.mean(successes):.4f}')
    


def train():
    agent = Policy()
    agent.train()
    agent.save()


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    args = parser.parse_args()
    
    if args.train:
        train()

    if args.evaluate:
        evaluate(render=args.render)

    
if __name__ == '__main__':
    main()
