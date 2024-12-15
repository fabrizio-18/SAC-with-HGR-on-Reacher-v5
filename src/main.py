import argparse
import numpy as np
from policy import Policy
import gymnasium as gym
import warnings
import torch
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt


def evaluate(env=None, n_episodes=10, render=False):
    agent = Policy()
    agent.load()
    env = gym.make("Reacher-v5", max_episode_steps=100)
    if render:
        env = gym.make("Reacher-v5", render_mode="human", max_episode_steps=100)
        
    rewards = []
    for episode in range(n_episodes):
        total_reward = 0
        done = False
        s, _ = env.reset()
        env.render()  
        while not done:
            action = agent(s)
            
            s, reward, terminated, truncated, info = env.step(action)
            #print(s['achieved_goal'])
            done = terminated or truncated
            #print(f"episode : {episode}, {truncated}")
            total_reward += reward
        
        rewards.append(total_reward)
    env.close()
    print('Mean Reward:', np.mean(rewards))
    


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
