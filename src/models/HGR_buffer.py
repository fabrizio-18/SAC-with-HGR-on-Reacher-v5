import numpy as np
import torch
import threading
from utils import SumSegmentTree, MinSegmentTree, LinearSchedule

class HGRReplayBuffer:
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
        
        #### HGR prioritization ####
        self._length_weight = int((self.T + 1) * self.T / 2) # 1275
        self.weight_of_transition = np.empty([self.size, self._length_weight]) # 30,000 (# of episodes in buffer) x 1275 (# of combinations between transition and future goals)
        self.td_of_transition = np.empty([self.size, self._length_weight]) # 30,000 x 1275
        self._idx_state_and_future = np.empty(self._length_weight, dtype=list) # 1275
        _idx = 0
        for i in range(self.T):
            for j in range(i, self.T):
                self._idx_state_and_future[_idx] = [i, j + 1]
                _idx += 1

        it_capacity = 1  #Iterator for computing capacity of buffer
        while it_capacity < self.size:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)

        self._max_transition_priority = 1.0

        self.alpha = 0.3
        self.beta = 0.3

        self.beta_schedule = LinearSchedule(self.size, 1.0, self.beta)
        self.lock = threading.Lock()

    
    def store_episode(self, obs, achieved_goals, goals, actions, next_obs, next_achieved_goals, rewards, dones):
        with self.lock:
            idx = self._get_storage_idx()
            self.buffers['obs'][idx] = obs
            self.buffers['achieved_goals'][idx] = achieved_goals
            self.buffers['goals'][idx] = goals
            self.buffers['actions'][idx] = actions
            self.buffers['next_obs'][idx] = next_obs
            self.buffers['next_achieved_goals'][idx] = next_achieved_goals
            self.buffers['rewards'][idx] = rewards
            self.buffers['dones'][idx] = dones
            self.n_transition_stored += self.T
        
        if isinstance(idx, np.int64):
            rollout_batch_size = 1
            idx_ep = np.array([idx])
        elif isinstance(idx, np.ndarray):
            rollout_batch_size = idx.shape[0]
        else:
            rollout_batch_size = None

        _default_priority_ep = self._max_transition_priority ** self.alpha
        for k in range(rollout_batch_size):
            idx = idx_ep[k]
            self._it_sum[idx] = _default_priority_ep
            self._it_min[idx] = _default_priority_ep

        self.weight_of_transition[idx_ep] = \
            (np.ones((rollout_batch_size, self._length_weight)) * self._max_transition_priority) ** self.alpha
        self.td_of_transition[idx_ep] = (np.ones((rollout_batch_size, self._length_weight)) * self._max_transition_priority)

    
    def sample(self, batch_size, step):
        self.beta = self.beta_schedule.value(step)
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]

        ##### Computing episodes indices ##### 
        episode_idxs, weights_episodes = self._sample_episode_indices(batch_size)

        transitions, weights_transitions, transitions_idxs = self.her_sampler(temp_buffers, episode_idxs, batch_size)
        transitions = {key: torch.as_tensor(value, device=self.device, dtype=torch.float32) for key, value in transitions.items()}

        weights = weights_episodes * weights_transitions
        weights = weights / weights.max()
        weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)
        return transitions, episode_idxs, transitions_idxs, weights

    def her_sampler(self, buffers, episode_idxs, batch_size):
        transitions = {}
        transition_idxs = np.zeros(batch_size, dtype=np.int64)
        weights_transitions = np.zeros(batch_size, dtype=np.float32)
        t_samples = np.zeros(batch_size, dtype=np.int64)
        future_t = np.zeros(batch_size, dtype=np.int64)

        for i in range(batch_size):
            ##### Weights for transitions within the sampled episode #####
            weight_prob = self.weight_of_transition[episode_idxs[i]] / self.weight_of_transition[episode_idxs[i]].sum()

            ##### Sample a transition using the weights #####
            idx = np.random.choice(len(weight_prob), p=weight_prob)
            transition_idxs[i] = idx
            t_samples[i], future_t[i] = self._idx_state_and_future[idx]

            ##### IS weights for transitions #####
            weights_transitions[i] = (self.weight_of_transition[episode_idxs[i], idx] * self._length_weight) ** (-self.beta)

        ##### Extract transitions using the sampled states and futures #####
        for key in buffers.keys():
            transitions[key] = buffers[key][episode_idxs, t_samples].copy()
        
        ##### HER indices #####
        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)

        future_achieved_goals = buffers['achieved_goals'][episode_idxs[her_indices], future_t[her_indices]] 
        transitions['goals'][her_indices] = future_achieved_goals

        transitions['rewards'] = np.expand_dims(self.reward_fun(transitions['next_achieved_goals'], transitions['actions'], transitions['goals']), axis=1)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        return transitions, weights_transitions, transition_idxs
    
    def _sample_episode_indices(self, batch_size):
        """
        Sample episode indices using proportional prioritization.
        """
        episode_idxs = []
        weights = []

        ##### Total priority sum #####
        p_total = self._it_sum.sum(0, self.current_size)

        for _ in range(batch_size):
            mass = np.random.uniform() * p_total
            idx = self._it_sum.find_prefixsum_idx(mass)
            episode_idxs.append(idx)

            # Compute weights
            p_sample = self._it_sum[idx] / p_total
            weight = (p_sample * self.current_size) ** (-self.beta)
            weights.append(weight)

        weights = np.array(weights)

        return np.array(episode_idxs), weights
    

    def update_priorities(self, ep_idxs, priorities, transition_idxs):
        priorities = priorities.detach().numpy()
        for ep_idx, priority_of_transition, transition_idx in zip(ep_idxs, priorities, transition_idxs):

            ##### Update weight for transitions in 1 episode #####
            self.weight_of_transition[ep_idx, transition_idx] = priority_of_transition ** self.alpha
            self.td_of_transition[ep_idx, transition_idx] = priority_of_transition

            ##### Update weight for all episodes #####
            priority_of_episode = self.td_of_transition[ep_idx].mean()
            self._it_sum[ep_idx] = priority_of_episode ** self.alpha
            self._it_min[ep_idx] = priority_of_episode ** self.alpha

            ##### Update maximal transition priority #####
            self._max_transition_priority = max(self._max_transition_priority, priority_of_transition)


    
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