import numpy as np
import torch
import threading

class HERReplayBuffer:
    def __init__(self, obs_shape, action_shape, goal_shape, capacity, device, max_episode_steps):
        self.capacity = capacity
        self.device = device
        self.T = max_episode_steps
        self.size = self.capacity // self.T

        self.current_size = 0
        self.n_transition_stored = 0

        self.replay_k = 4
        self.future_p = 1 - (1./(1 + self.replay_k))

        self.buffers = {'obs': np.empty([self.size, self.T+1, obs_shape], dtype=np.float32),
                        'achieved_goals': np.empty([self.size, self.T+1, goal_shape], dtype=np.float32),
                        'goals': np.empty([self.size, self.T, goal_shape], dtype=np.float32),
                        'actions':np.empty([self.size, self.T, action_shape], dtype=np.float32),
                        'next_obs': np.empty([self.size, self.T+1, obs_shape], dtype=np.float32),
                        'next_achieved_goals': np.empty([self.size, self.T+1, goal_shape], dtype=np.float32),
                        'rewards': np.empty([self.size, self.T], dtype=np.float32),
                        'dones': np.empty([self.size, self.T], dtype=np.float32),
                        }
        
        self.lock = threading.Lock()

    
    def store_episode(self, obs, achieved_goals, goals, actions, next_obs, next_achieved_goals, rewards, dones):
        with self.lock:
            idxs = self._get_storage_idx()
            self.buffers['obs'][idxs] = obs
            self.buffers['achieved_goals'][idxs] = achieved_goals
            self.buffers['goals'][idxs] = goals
            self.buffers['actions'][idxs] = actions
            self.buffers['next_obs'][idxs] = next_obs
            self.buffers['next_achieved_goals'][idxs] = next_achieved_goals
            self.buffers['rewards'][idxs] = rewards
            self.buffers['dones'][idxs] = dones
            self.n_transition_stored += self.T

    
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = self.her_sampler(temp_buffers, batch_size)
        transitions = {key: torch.as_tensor(value, device=self.device, dtype=torch.float32) for key, value in transitions.items()}

        return transitions

    def her_sampler(self, buffers, batch_size):
        current_episodes = buffers['actions'].shape[0]
        episode_idxs = np.random.randint(0, current_episodes, batch_size)
        t_samples = np.random.randint(0, self.T, size=batch_size)
        transitions = {key: buffers[key][episode_idxs, t_samples].copy() for key in buffers.keys()}
        
        ##### HER indices #####
        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (self.T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indices]

        future_achieved_goals = buffers['achieved_goals'][episode_idxs[her_indices], future_t]
        transitions['goals'] = transitions['goals'].copy()  
        transitions['goals'][her_indices] = future_achieved_goals

        transitions['rewards'] = np.expand_dims(self.reward_fun(transitions['next_achieved_goals'], transitions['actions'], transitions['goals']), axis=1)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        return transitions
    
    def reward_fun(self, achieved_goal, action, desired_goal): 
        achieved_goal = np.array(achieved_goal) 
        desired_goal = np.array(desired_goal)    
        action = np.array(action) 
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        
        rewards = np.where(
            distance < 0.05,  #Condition
            -0.1 * (np.linalg.norm(action, axis=-1) ** 2),  #for small distance
            -distance - 0.1 * (np.linalg.norm(action, axis=-1) ** 2)  #for large distance
        )

        return rewards
    

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx