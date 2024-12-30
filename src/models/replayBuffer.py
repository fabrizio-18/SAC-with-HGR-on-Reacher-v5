import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        self.obs = np.empty((capacity, obs_shape), dtype=np.float32)
        self.next_obs = np.empty((capacity, obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx
    
    def push(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obs[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obs[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)

        obs = self.obs[idxs]
        next_obs = self.next_obs[idxs]

        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        next_obs = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32)

        actions = torch.as_tensor(self.actions[idxs], device = self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device = self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device = self.device)

        return obs, actions, rewards, next_obs, not_dones_no_max
    