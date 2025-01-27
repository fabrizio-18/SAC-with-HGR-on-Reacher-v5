import torch
import numpy as np
import torch.nn.functional as F
from models import actor, critic

class SAC():
    def __init__(self, obs_shape, action_shape, goal_shape, device, hidden_dim=256, init_alpha=0.1, lr=1e-3, gamma=0.98, tau=0.01, batch_size=256):
        self.device = device
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor_update_freq = 2
        self.critic_target_update_freq = 2

        self.actor = actor.Actor(obs_shape=obs_shape, action_shape=action_shape, goal_shape=goal_shape, hidden_dim=hidden_dim, device=device)
        self.critic = critic.Critic(obs_shape=obs_shape, action_shape=action_shape, goal_shape=goal_shape, hidden_dim=hidden_dim, device=device)
        self.target_critic = critic.Critic(obs_shape=obs_shape, action_shape=action_shape, goal_shape=goal_shape, hidden_dim=hidden_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32).to(device)
        self.log_alpha.requires_grad = True
        
        self.target_entropy = -action_shape

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
    
        self.train()
        self.target_critic.train()
        

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    

    ##### Uncomment this method if you are using HER #####
    #def update_critic(self, obs, goal, action, reward, next_obs, done):
        with torch.no_grad():

            dist = self.actor(next_obs, goal)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

            target_Q1, target_Q2 = self.target_critic(next_obs, next_action, goal)
            target_V = torch.min(target_Q1,target_Q2) - self.alpha.detach() * log_prob
            done = done.unsqueeze(1)
            target_Q = reward + (self.gamma * target_V * (1 - done))

        current_Q1, current_Q2 = self.critic(obs, action, goal)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        ##### Update Critic #####
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.detach()


    ##### Uncomment this method if you're using HGR buffer #####
    def update_critic(self, obs, goal, action, reward, next_obs, done, episode_idxs, transitions_idxs, weights, replay_buffer):
        with torch.no_grad():

            dist = self.actor(next_obs, goal)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

            target_Q1, target_Q2 = self.target_critic(next_obs, next_action, goal)
            target_V = torch.min(target_Q1,target_Q2) - self.alpha.detach() * log_prob
            done = done.unsqueeze(1)
            target_Q = reward + (self.gamma * target_V * (1 - done))

        current_Q1, current_Q2 = self.critic(obs, action, goal)
        loss_Q1 = F.mse_loss(current_Q1, target_Q, reduction='none')
        loss_Q2 = F.mse_loss(current_Q2, target_Q, reduction='none') 

        weighted_loss_Q1 = loss_Q1 * weights
        weighted_loss_Q2 = loss_Q2 * weights

        critic_loss = weighted_loss_Q1.mean() + weighted_loss_Q2.mean()

        priorities = (target_Q - torch.min(current_Q1, current_Q2)).abs()
        
        ##### Update Critic #####
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        replay_buffer.update_priorities(episode_idxs, priorities, transitions_idxs)
        return critic_loss.detach()
    
    def update_actor(self, obs, goal):

        dist = self.actor(obs, goal)
        action = dist.rsample() 
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        actor_Q1, actor_Q2 = self.critic(obs, action, goal)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        
        ##### Update Actor #####
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        ##### Update alpha #####
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (-self.alpha * (log_prob + self.target_entropy).detach()).mean()

        alpha_loss.backward()
        self.log_alpha_optimizer.step()


    def soft_update_params(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data +
                                    (1 - self.tau) * target_param.data)
            
    
    def update(self, replay_buffer, step):
        ##### Sampling using the HER_buffer class #####
        #transitions = replay_buffer.sample(self.batch_size)

        ##### Sampling using the HGR_buffer class #####
        transitions, episode_idxs, transitions_idxs, weights = replay_buffer.sample(self.batch_size, step)
        
        obs = transitions['obs']
        goal = transitions['goals']
        action = transitions['actions']
        next_obs = transitions['next_obs']
        reward = transitions['rewards']
        done = transitions['dones']

        ##### If you're using HER buffer #####
        #critic_loss = self.update_critic(obs, goal, action, reward, next_obs, done)

        
        ##### If you're using HGR buffer #####
        critic_loss = self.update_critic(obs, goal, action, reward, next_obs, done, episode_idxs, transitions_idxs, weights, replay_buffer)

        if step % self.actor_update_freq == 0:
            self.update_actor(obs, goal)
        
        if step % self.critic_target_update_freq == 0:
            self.soft_update_params(self.critic, self.target_critic)
        
        return critic_loss