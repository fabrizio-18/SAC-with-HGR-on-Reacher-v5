import gymnasium as gym
import torch
import torch.nn as nn
from models.SAC import SAC
from models.replayBuffer import PrioritizedHERReplayBuffer
from utils import *


class Policy(nn.Module):

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device if device else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.env = gym.make("FetchPickAndPlace-v3", render_mode="rgb_array", max_episode_steps=50)
        
        self.state_size = self.env.observation_space['observation'].shape[0]
        self.goal_size = self.env.observation_space['desired_goal'].shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.buffer = PrioritizedHERReplayBuffer(1500000, self.state_size, self.action_size, self.goal_size, reward_fun=self.reward_fun)
        self.gamma = 0.98
        self.tau = 0.01
        self.hidden_size = 256
        self.lr = 1e-3
        self.sac = SAC(self.env, self.state_size, self.action_size, self.goal_size, self.buffer, self.gamma, self.tau, self.hidden_size, self.lr, self.device, self.reward_fun)
        self.env.reset()
        
        
    def forward(self, state):
        a = self.sac.actor.get_action(state['observation'], state['desired_goal'])
        return a

    def train(self):
        rewards, losses = self.sac.train()
        #print(rewards, losses)
        self.save()
        plot(rewards, 'reward')
        plot(losses.detach().numpy(), 'loss')

        
    
    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    
    def reward_fun(self, achieved_goal, desired_goal, info):  # vectorized
        return self.env.env.env.env.compute_reward(achieved_goal, desired_goal, info=info)
    


