import gymnasium as gym
import gymnasium_robotics
import torch
import torch.nn as nn
from models.SAC import SAC
from models.replayBuffer import ReplayBuffer
from utils import *
import numpy as np

gym.register_envs(gymnasium_robotics)


class Policy(nn.Module):

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device if device else torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.env = gym.make("Reacher-v5", render_mode="rgb_array", max_episode_steps=100)
        
        self.state_size = self.env.observation_space.shape[0]
        self.goal_size = 2
        self.action_size = self.env.action_space.shape[0]
        #print(self.state_size, self.action_size, self.goal_size)
        self.buffer = ReplayBuffer(1500000, reward_fun=self.reward_fun)
        self.gamma = 0.98
        self.tau = 0.01
        self.hidden_size = 256
        self.lr = 1e-3
        self.sac = SAC(self.env, self.state_size, self.action_size, self.goal_size, self.buffer, self.gamma, self.tau, self.hidden_size, self.lr, self.device, self.reward_fun)
        self.env.reset()
        
        
    def forward(self, state):
        a = self.sac.actor.get_action(state, [state[4], state[5]])
        return a

    def train(self):
        rewards, losses = self.sac.train()
        plot(rewards, 'reward')
        plot(losses, 'loss')

        
    
    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    
    def reward_fun(self, achieved_goal, action, desired_goal):  # vectorized
        achieved_goal = np.array(achieved_goal) 
        desired_goal = np.array(desired_goal)    
        action = np.array(action) 
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if distance <0.05:
            r = -distance - 0.1 * (np.linalg.norm(action, axis=-1)) ** 2 
        else:
            r = - 0.1 * (np.linalg.norm(action, axis=-1)) ** 2 
        return r
    


