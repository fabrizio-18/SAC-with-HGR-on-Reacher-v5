import numpy as np
import threading

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for HER experience replay.
    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {}
        for key in episode_batch:
            transitions[key] = episode_batch[key][episode_idxs, t_samples].copy()

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = np.minimum((t_samples + 1 + future_offset), T - 1)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['achieved_goals'][episode_idxs[her_indexes], future_t]
        transitions['goals'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['achieved_goals', 'goals']}
        reward_params['info'] = info
        transitions['rewards'] = reward_fun(achieved_goal=reward_params['achieved_goals'],
                                            desired_goal=reward_params['goals'],
                                            info=reward_params['info'])

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['actions'].shape[0] == batch_size_in_transitions)

        return transitions, [episode_idxs, t_samples]

    return _sample_her_transitions

class SumSegmentTree:
    def __init__(self, capacity):
        assert capacity > 0 and (capacity & (capacity - 1)) == 0, "Capacity must be a power of 2."
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)

    def sum(self, start=0, end=None):
        if end is None:
            end = self.capacity
        start += self.capacity
        end += self.capacity

        result = 0.0
        while start < end:
            if start % 2 == 1:
                result += self.tree[start]
                start += 1
            if end % 2 == 1:
                end -= 1
                result += self.tree[end]
            start //= 2
            end //= 2
        return result

    def find_prefixsum_idx(self, prefixsum):
        idx = 1
        while idx < self.capacity:
            if self.tree[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self.tree[2 * idx]
                idx = 2 * idx + 1
        return idx - self.capacity

    def update(self, idx, value):
        idx += self.capacity
        self.tree[idx] = value
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]
            idx //= 2

class PrioritizedHERReplayBuffer:
    def __init__(self, max_size, obs_dim, action_dim, goal_dim, alpha=0.8, replay_k=4, reward_fun=None, T=50):
        """
        Args:
            max_size (int): Maximum number of transitions the buffer can store.
            obs_dim (int): Dimension of the observations.
            action_dim (int): Dimension of the actions.
            future_p (float): Probability of sampling future goals.
            alpha (float): Priority exponent (how much prioritization is used).
            replay_k (int): Ratio of HER to regular samples.
            reward_fun (function): Function to compute the reward.
        """
        self.max_size = max_size
        self.alpha = alpha
        self.replay_k = replay_k
        self.reward_fun = reward_fun
        self.T = T
        self.rollout_batch_size = max_size // self.T
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim

        # Current size and position for adding new data
        self.size = 0
        self.ptr = 0

        # Data storage
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.goals = np.zeros((max_size, goal_dim), dtype=np.float32)
        self.achieved_goals = np.zeros((max_size, goal_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)

        # Prioritization structures
        self.sum_tree = SumSegmentTree(2 ** (max_size - 1).bit_length())
        self.max_priority = 1.0  # Initialize with max priority

        # HER transition sampler
        self.her_sampler = make_sample_her_transitions('future', replay_k, reward_fun)

        # Thread lock for safety
        self.lock = threading.Lock()

    def add(self, obs, action, reward, next_obs, goal, achieved_goal, done):
        """
        Add a transition to the buffer.
        """
        with self.lock:
            self.obs[self.ptr] = obs
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.next_obs[self.ptr] = next_obs
            self.goals[self.ptr] = goal
            self.achieved_goals[self.ptr] = achieved_goal
            self.dones[self.ptr] = done

            self.sum_tree.update(self.ptr, self.max_priority ** self.alpha)

            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of transitions with prioritization and HER.
        Args:
            batch_size (int): Number of transitions to sample.
            beta (float): Importance sampling correction factor.
        Returns:
            dict: Sampled transitions with importance sampling weights.
        """
        with self.lock:
            if self.size == 0:
                raise ValueError("Replay buffer is empty!")

            indices = np.array([self.sum_tree.find_prefixsum_idx(np.random.uniform(0, self.sum_tree.sum(0, self.size)))
                                 for _ in range(batch_size)])

            
            probabilities = np.array([self.sum_tree.sum(idx, idx + 1) for idx in indices]) / self.sum_tree.sum(0, self.size)
            weights = (self.size * probabilities) ** (-beta)
            weights /= weights.max()  # Normalize
            
            #print(self.obs)
            episode_batch = {
                'observations': self.obs.reshape(-1, self.T, self.obs_dim),  # Reshape to (rollout_batch_size, T, obs_dim)
                'actions': self.actions.reshape(-1, self.T, self.action_dim),
                'rewards': self.rewards.reshape(-1, self.T),
                'next_observations': self.next_obs.reshape(-1, self.T, self.obs_dim),
                'goals': self.goals.reshape(-1, self.T, self.goal_dim),
                'achieved_goals': self.achieved_goals.reshape(-1, self.T, self.goal_dim),
                'dones': self.dones.reshape(-1, self.T),
            }

            transitions, _ = self.her_sampler(episode_batch, batch_size)
            print(transitions['observations'])
            return {
                'obs': transitions['observations'],
                'actions': transitions['actions'],
                'rewards': transitions['rewards'],
                'next_obs': transitions['next_observations'],
                'goals': transitions['goals'],
                'dones': transitions['dones'],
                'indices': indices,
                'weights': weights
            }

    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled transitions.
        Args:
            indices (np.ndarray): Indices of the sampled transitions.
            priorities (np.ndarray): Updated priority values.
        """
        with self.lock:
            for idx, priority in zip(indices, priorities):
                self.sum_tree.update(idx, max(priority, 1e-6) ** self.alpha)
            self.max_priority = max(self.max_priority, priorities.max())
