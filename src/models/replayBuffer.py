import numpy as np
import random


class ReplayBuffer:

    def __init__(self, buffer_size=1000000, alpha=0.8):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0   
        self.episodes = []
        self.episode_priorities = []
        self.full = False
        self.max_episode_length = 70
        self.alpha = alpha
    
    def push(self, state, action, next_state, done, goal, hindsight_goal, step_count, episode_idx, priority=10.0):
        transition = {
        "state": state,
        "action": action,
        "next_state": next_state,
        "goal": goal,
        "hindsight_goal": hindsight_goal,
        "priority": priority,
        "episode_idx": episode_idx,
        "step_count": step_count
        }
        #print(f"Im the {step_count} transition")
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        #else:
        #    self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.buffer_size
        self.full = self.full or self.position == 0


        if done or step_count == self.max_episode_length:
            episode_priority = 10
            self.episode_priorities.append(episode_priority)
            episode_transitions = [t for t in self.buffer if t["episode_idx"] == episode_idx]
            self.episodes.append(episode_transitions)  # Store the current episode
            
    
    def sample(self, episode, beta=0.4):
        transition = random.sample(episode[:-1], 1)[0]
        #for t in episode:
        #    print(t['step_count'])
        #print(transition['step_count'])
        
        Z = sum([t['priority'] ** self.alpha for t in episode])

        probabilities = [t["priority"] ** self.alpha / Z for t in episode]
        
        valid_indices = np.arange(transition['step_count']+1, len(episode)+1)
        #print(valid_indices)
        valid_probabilities = probabilities[transition['step_count']:]
        #print(len(valid_probabilities))
        valid_probabilities /= np.sum(valid_probabilities)

        index = np.random.choice(len(valid_indices), p=valid_probabilities)
        transition_goal = episode[index-1]

        state = transition["state"]
        action = transition["action"]
        next_state = transition["next_state"]
        goal = transition["hindsight_goal"]
        hindsight_goal = transition_goal["hindsight_goal"]
        
        P_ij = probabilities[index]
        return state, action, next_state, goal, hindsight_goal, P_ij, index
    
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
    
    def update_episode_prior(self, episode_idx, td_error):
        self.episode_priorities[episode_idx] = td_error.item() + 1e-6

    def update_experience_prior(self, td_error, idx, episode_idx):
        #print(td_error)
        self.episodes[episode_idx][idx]["priority"] = (td_error.abs().item()) + 1e-6

    def __len__(self):
        return self.buffer_size if self.full else self.position