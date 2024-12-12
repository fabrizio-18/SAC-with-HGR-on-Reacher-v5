import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from models.actor import Actor
from models.critic import Critic
from models.valueNet import ValueNetwork
from utils import *

class SAC:
    def __init__(self, env, state_size, action_size, goal_size, buffer, gamma = 0.98, tau=0.95, hidden_size=256, lr=1e-3, device='cpu', get_reward=None):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.buffer = buffer
        self.env = env
        self.MSE = nn.MSELoss()
        self.log_alpha = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = - action_size
        self.beta = 0.5
        self.action_size = action_size
        self.get_reward = get_reward

        self.actor = Actor(state_size, action_size, goal_size, hidden_size, device=self.device).to(self.device)
        self.critic1 = Critic(state_size, action_size, goal_size, hidden_size, device=self.device).to(self.device)
        self.critic2 = Critic(state_size, action_size, goal_size, hidden_size, device=self.device).to(self.device)
        self.value_net = ValueNetwork(state_size, goal_size,hidden_size, device=self.device).to(self.device)
        self.target_value_net = ValueNetwork(state_size, goal_size, hidden_size, device=self.device).to(self.device)
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        self.value_net_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def soft_update_networks(self, local_model, target_model):
        for t_params, e_params in zip(target_model.parameters(), local_model.parameters()):
            t_params.data.copy_(self.tau * e_params.data + (1 - self.tau) * t_params.data)

    def train(self):
        self.critic1.train()
        self.critic2.train()
        self.actor.train()
        self.value_net.train()
        
        n_epochs = 1000
        steps = 50
        episode = 0
        update_freq = 2
        batch_size  = 256
        epsilon = 0.3
        epsilon_decay = 0.995

        episode_rewards = []  
        losses = []   

        with tqdm(range(n_epochs), desc="Epochs", unit="epoch") as pbar:
            for epoch in pbar:
                s, _ = self.env.reset()
                state = s['observation']
                goal = s['desired_goal']
                hindsight_goal = s['achieved_goal']
                step_count = 0
                episode_reward = 0
                #episode_loss = 0
                critic_loss1 = torch.zeros(1, device=self.device)
                critic_loss2 = torch.zeros(1, device=self.device)
                critic_loss = torch.zeros(1, device=self.device)
                value_loss = torch.zeros(1, device=self.device)
                actor_loss = torch.zeros(1, device=self.device)
                alpha_loss = torch.zeros(1, device=self.device)
                
                ##### Rollout one episode #####
                for step in tqdm(range(steps), desc=f"Rollout (Epoch {epoch+1})", leave=False, unit="step"):
                    ##### Action selected with epsilon-greedy policy #####
                    action = self.actor.get_action(state, goal)
                    noise = torch.distributions.normal.Normal(0, 0.2).sample(action.shape).cpu().numpy()
                    #print(action, noise)
                    action = action + noise if random.random() > epsilon else self.env.action_space.sample()
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    step_count += 1
                    #print(f'Im step count: {step_count}')
                    #print(f"\n{hindsight_goal} \n, {state}")
                    self.buffer.add(state, action, reward, next_state['observation'], goal, hindsight_goal, done)
                    state = next_state['observation']
                    hindsight_goal = next_state['achieved_goal']
                    goal = next_state['desired_goal']
                    
                if done or step_count >= steps:
                    episode += 1
                    step_count = 0  

                episode_rewards.append(episode_reward)
                #print(f"The episode buffer contains: {len(self.buffer.episodes)}")
                #print(f"I'm the first episode: {self.buffer.episodes[0]}")
                #print(self.buffer.buffer[-1])
                
                if episode % update_freq == 0 and self.buffer.size > batch_size:
                    
                    ##### Episode prioritization #####
                    batch = self.buffer.sample(batch_size)
                    
                    states = torch.tensor(batch['obs'], dtype=torch.float32, device=self.device)
                    actions = torch.tensor(batch['actions'], dtype=torch.float32, device=self.device)
                    rewards = torch.tensor(batch['rewards'], dtype=torch.float32, device=self.device) ##### Recalculated rewards (HER) #####
                    next_states = torch.tensor(batch['next_obs'], dtype=torch.float32, device=self.device)
                    goals = torch.tensor(batch['goals'], dtype=torch.float32, device=self.device)
                    #print(states)
                    dones = torch.tensor(batch['dones'], dtype=torch.float32, device=self.device)
                    weights = torch.tensor(batch['weights'], dtype=torch.float32, device=self.device)


                    
                    with torch.no_grad():
                        next_a_pred, next_log_prob = self.actor.evaluate(next_states, goals)
                        q_pred1_target = self.critic1(next_states, goals, next_a_pred)
                        q_pred2_target = self.critic2(next_states, goals, next_a_pred)
                        
                        v_target = torch.min(q_pred1_target, q_pred2_target) - self.alpha * next_log_prob

                    v_pred = self.value_net(states, goals)
                    value_loss = self.MSE(v_pred, v_target.detach())

                    q_pred1 = self.critic1(states, goals, actions)
                    q_pred2 = self.critic2(states, goals, actions)
                    target_values = self.target_value_net(next_states, goals)
                    td_targets = rewards.unsqueeze(1) + self.gamma * target_values * (1 - dones.unsqueeze(1))
                    priorities =  (td_targets - torch.min(q_pred1,q_pred2)).abs()
                    #td_error = torch.clamp(td_error, min=-10.0, max=10.0)  

                    #print(td_error.shape)

                    ##### Update priorities #####
                    self.buffer.update_priorities(batch['indices'], priorities)

                    #episode_td_error = self.compute_episode_td_error(sampled_episode, goal)
                    #self.buffer.update_episode_prior(episode_idx, episode_td_error)
                    

                    critic_loss1 = (weights *  (q_pred1 - td_targets.detach()).pow(2)).mean()
                    critic_loss2 = (weights *  (q_pred2 - td_targets.detach()).pow(2)).mean()
                                   

                    a_pred, log_prob = self.actor.evaluate(states, goals)
                    q_pred_new1 = self.critic1(states, goals, a_pred)
                    q_pred_new2 = self.critic2(states, goals, a_pred)
                    q_pred_new = torch.min(q_pred_new1, q_pred_new2)
                    actor_loss = (self.alpha * log_prob - q_pred_new.detach()).mean() 

                    alpha_loss = (-(self.log_alpha * (next_log_prob.detach() + self.target_entropy))).mean()
                    
                    #critic_loss1 = critic_loss1 / torch.max(weights)
                    #critic_loss2 = critic_loss2 / torch.max(weights)
                    #print(f"\n{np.max(weights)}")

                    #print(value_loss, critic_loss, actor_loss)
                    self.value_net_optimizer.zero_grad()
                    value_loss.backward()
                    self.value_net_optimizer.step()
                    
                    self.critic1_optimizer.zero_grad()
                    self.critic2_optimizer.zero_grad()
                    critic_loss1.backward()
                    critic_loss2.backward()
                    #torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
                    #torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
                    self.critic1_optimizer.step()
                    self.critic2_optimizer.step()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                    self.actor_optimizer.step()
                    
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    #torch.nn.utils.clip_grad_norm_(self.log_alpha, max_norm=1.0)
                    self.alpha_optimizer.step()

                    #self.soft_update_networks(self.critic1, self.target_critic1)
                    #self.soft_update_networks(self.critic2, self.target_critic2)
                    self.soft_update_networks(self.value_net, self.target_value_net)

                    critic_loss = (critic_loss1 + critic_loss2) / 2
                    losses.append(critic_loss)
                    #print(weights)

                epoch += 1
                epsilon *= epsilon_decay
                if critic_loss.item() !=0:
                    pbar.set_postfix({
                        "Critic Loss": f"{critic_loss.item():.4f}",
                        "Value Loss": f"{value_loss.item():.4f}",
                        "Actor Loss": f"{actor_loss.item():.4f}",
                        "Episode Loss": f"{critic_loss.item() + value_loss.item() + actor_loss.item(): .4f}"
                    })

                

        return episode_rewards, losses
        






















        '''def compute_episode_td_error(self, episode, goal):
        td_errors = []
        #print(episode)
        for experience in episode:
            
            state = experience["state"]
            state = torch.tensor(state, dtype=torch.float32, device=self.device, requires_grad=True)

            action = experience["action"]
            action = torch.tensor(action, dtype=torch.float32, device=self.device, requires_grad=True)

            next_state = experience['next_state']
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device, requires_grad=True)

            hindsight_goal = experience["hindsight_goal"]
            hindsight_goal = torch.tensor(hindsight_goal, dtype=torch.float32, device=self.device, requires_grad=True)

            #print(f"\n{hindsight_goal, goal}")
            reward = self.get_reward(hindsight_goal, goal)
            #print(f"\n{reward}")
            #with torch.no_grad():
            #    next_a_pred, next_log_prob = self.actor.evaluate(next_state,goal)
            #    q_pred1_target = self.critic1(next_state, goal, next_a_pred)
            #    q_pred2_target = self.critic2(next_state, goal, next_a_pred)
                
            #    v_target = torch.min(q_pred1_target, q_pred2_target) - next_log_prob
            
            q_pred1 = self.critic1(state, goal, action)
            q_pred2 = self.critic2(state, goal, action)
            v_target = self.target_value_net(state, goal)
            td_error = reward + self.gamma * v_target - torch.min(q_pred1, q_pred2)
            td_error = torch.clamp(td_error, min=-10.0, max=10.0)
            td_errors.append(td_error.abs())

        # Compute mean absolute TD error for the entire episode
        td_errors = torch.FloatTensor(td_errors)
        #print(td_errors.shape)
        episode_td_error = torch.tensor(td_errors).mean()
        return episode_td_error
        '''
                        

            