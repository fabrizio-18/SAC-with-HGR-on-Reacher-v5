import numpy as np
import random


class ReplayBuffer:

    def __init__(self, buffer_size=1000000, alpha=0.8, reward_fun=None):
        self.buffer_size = buffer_size
        self.position = 0   
        self.total_transitions = 0  
        self.episodes = []
        self.episode_priorities = []
        self.full = False
        self.max_episode_length = 100
        self.alpha = alpha
        self.reward_fun = reward_fun

        self._length_weight = int((self.max_episode_length + 1) * self.max_episode_length / 2)
        self.size_in_episodes = self.buffer_size // self.max_episode_length
        self.priority_of_transition = np.empty([self.size_in_episodes, self._length_weight])
        self.td_of_transition = np.empty([self.size_in_episodes, self._length_weight])
        self._idx_state_and_future = np.empty(self._length_weight, dtype=list)

        _idx = 0
        for i in range(self.max_episode_length):
            for j in range(i, self.max_episode_length):
                self._idx_state_and_future[_idx] = [i, j + 1]
                _idx += 1
    
    def push(self, state, action, next_state, goal, done, priority, episode_idx, step_count):
        transition = {
        "state": state,
        "action": action,
        "next_state": next_state,
        "goal": goal,
        "done": done,
        "episode_idx": episode_idx,
        "step_count": step_count
        }
        #print(f"Im the {step_count} transition")
        if len(self.episodes) <= episode_idx:
            self.episodes.append([])  # Ensure episode buffer exists
        self.episodes[episode_idx].append(transition)
        self.total_transitions += 1
        
        # Check if episode ends
        if done or step_count == self.max_episode_length:
            # Update buffer position
            self.position = (self.position + 1) % self.size_in_episodes
            self.full = self.full or self.position == 0

            episode_transitions = self.episodes[episode_idx]
            for i, t in enumerate(episode_transitions):
                for j in range(i + 1, len(episode_transitions)):
                    self.priority_of_transition[episode_idx, i * self.max_episode_length // 2 + j] = priority

            # Calculate episode priority and update the array
            episode_priority = sum(self.priority_of_transition[episode_idx]) / self._length_weight  # Aggregate priority
            if len(self.episode_priorities) < self.size_in_episodes:
                self.episode_priorities.append(episode_priority)
            else:
                self.episode_priorities[self.position] = episode_priority

            # Store episode transitions in flattened priority buffer
            

                
    
    def sample(self,batch_size=256, beta=0.5):
        scaled_priorities = [p ** self.alpha for p in self.episode_priorities]
        tot_priority = sum(scaled_priorities)
        episode_probabilities = [p / tot_priority for p in scaled_priorities]
        
        # Sample batch_size episodes based on the probabilities
        sampled_episode_indices = np.random.choice(len(self.episodes), batch_size, p=episode_probabilities)
        states = []
        actions = []
        next_states = []
        rewards = []
        goals = []
        indices = []
        dones = []
        weights = []
        
        #for t in episode:
        #    print(t['step_count'])
        #print(transition['step_count'])
        
        for episode_idx in sampled_episode_indices:
            episode = self.episodes[episode_idx]
            probabilities = []
            Z = 0
            for i, j in self._idx_state_and_future:
                priority = self.priority_of_transition[episode_idx, (i * self.max_episode_length // 2 + j) -1] ** self.alpha
                probabilities.append(priority)
                Z += priority

            # Normalize probabilities
            probabilities = [p / Z for p in probabilities]

            #transition = random.sample(episode[:-1], 1)[0]
            #valid_indices = np.arange(transition['step_count']+1, len(episode)+1)
            #print(valid_indices)
            #valid_probabilities = probabilities[transition['step_count']:]
            #print(len(valid_probabilities))
            #valid_probabilities /= np.sum(valid_probabilities)
            # Sample a transition
            flat_idx = np.random.choice(self._length_weight, p=probabilities)
            transition_idx, goal_idx = self._idx_state_and_future[flat_idx]
            #print(transition_idx, goal_idx)
            transition = episode[transition_idx]
            transition_goal = episode[goal_idx-1]

            state = transition["state"]
            action = transition["action"]
            next_state = transition["next_state"]
            goal = [transition_goal["state"][8], transition_goal["state"][9]]
            done = transition["done"]
            rewards.append(self.reward_fun([state[8],state[9]], action, goal))
            dones.append(done)
            states.append(state)
            actions.append(action)
            next_states.append(next_state) 
            goals.append(goal)
            indices.append(goal_idx)  # Store index for tracking

            w_n = (1 / (len(self.episodes) * episode_probabilities[episode_idx])) ** beta
            w_ij = (self._length_weight * probabilities[flat_idx]) ** -beta
            weights.append(w_n * w_ij)
            
        weights = weights / np.max(weights)

        return np.array(states), np.array(actions), np.array(next_states), np.array(goals), np.array(dones), np.array(rewards),  np.array(weights), indices, sampled_episode_indices
    
    
    def update_priorities(self, episode_indices, indices, priorities):
       priorities = priorities.tolist()
       for episode_idx, idx, priority in zip(episode_indices, indices, priorities):
        # Update the priority of the specific (transition, goal) pair
        self.priority_of_transition[episode_idx, idx] = priority[0]
        
        # Recompute the total priority for the episode
        self.episode_priorities[episode_idx] = self.priority_of_transition[episode_idx, :].mean()
   

    def __len__(self):
        return self.buffer_size if self.full else self.position
    

    """
    def sample_episode(self):
        if len(self.episodes) == 0:
            raise ValueError("No episodes in buffer to sample.")
        scaled_priorities = [p ** self.alpha for p in self.episode_priorities]
        tot_priority = sum(scaled_priorities)
        probabilities = [p / tot_priority for p in scaled_priorities]
        #print(len(self.episodes), len(probabilities))
        episode_idx = np.random.choice(len(self.episodes), p=probabilities)
        episode_prob = probabilities[episode_idx]
        episode = self.episodes[episode_idx]
        return episode, episode_prob, episode_idx
    """


    """
     def update_episode_prior(self, episode_idx, td_error):
        self.episode_priorities[episode_idx] = td_error.item() + 1e-6

    def update_experience_prior(self, td_error, idx, episode_idx):
        #print(td_error)
        self.episodes[episode_idx][idx]["priority"] = (td_error.abs().item()) + 1e-6
    """