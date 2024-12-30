import gymnasium as gym
import os
import torch
import numpy as np
import time
import torch.nn as nn
from utils import *
from models import SAC, replayBuffer, HER_buffer, HGR_buffer


class Policy(nn.Module):

    def __init__(self, device=torch.device('cpu'), seed=999):
        super(Policy, self).__init__()
        set_seed_everywhere(seed)
        self.device = device if device else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.dir = os.getcwd()
        self.env = NormalizedActions(gym.make("Reacher-v5", render_mode="rgb_array", max_episode_steps=50))
        self.obs_shape = self.env.observation_space.shape[0]
        self.action_shape = self.env.action_space.shape[0]
        self.goal_shape = 2
        self.hidden_dim = 256
        self.lr = 1e-3
        self.gamma = 0.98
        self.tau = 0.01
        self.batch_size = 256
        self.max_episode_steps = 50
        self.epsilon = 0.3
        self.eps_dacay = 0.99
        self.sac = SAC.SAC(obs_shape=self.obs_shape, action_shape=self.action_shape, goal_shape=self.goal_shape, device=self.device, 
                           hidden_dim=self.hidden_dim, lr=self.lr, gamma=self.gamma, tau=self.tau, batch_size=self.batch_size)
        
        #### if you want to use the simple buffer ####
        #self.buffer = replayBuffer.ReplayBuffer(obs_shape=self.obs_shape, action_shape=self.action_shape, capacity=1500000, device=self.device)
        
        #### if you want to use HER buffer ####
        #self.buffer = HER_buffer.HERReplayBuffer(obs_shape=self.obs_shape, action_shape=self.action_shape, goal_shape=self.goal_shape, capacity=1500000, device=self.device, max_episode_steps=self.max_episode_steps)
        
        #### if you want to use HGR buffer ####
        self.buffer = HGR_buffer.HGRReplayBuffer(obs_shape=self.obs_shape, action_shape=self.action_shape, goal_shape=self.goal_shape, capacity=1500000, device=self.device, max_episode_steps=self.max_episode_steps)
        self.step = 0
        self.env.reset()
        
        
    def train(self, train_steps=30000, eval_step=100):
        episode = 0
        episode_reward = 0
        done = True
        print("Training started...")
        start_time = time.time()
        success_count = 0
        rewards = []
        critic_losses = []
        success_rates = []
        episodes = []

        while self.step < train_steps:
            
            obs, _ = self.env.reset()
            goal = [obs[4], obs[5]]
            episode_reward = 0
            ##### Rollout one episode #####
            ep_obs, ep_achieved_goals, ep_goals, ep_actions, ep_next_obs, ep_next_achieved_goals, ep_rewards, ep_dones = [], [], [], [], [], [], [], []
            for step in range(self.max_episode_steps):
                ##### Action selected with epsilon-greedy policy #####
                action = self.sac.actor.act(obs, goal)
                noise = torch.distributions.normal.Normal(0, 0.2).sample(action.shape).cpu().numpy()
                action = action + noise if random.random() > self.epsilon else self.env.action_space.sample()
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                done = float(done)
                #done_no_max = 0 if step + 1 == self.max_episode_steps else done

                episode_reward += reward
                #Push the transition in the buffer. To use if you want to use the simple replayBuffer class
                #self.buffer.push(obs, action, reward, next_obs, done, done_no_max)

                #Storing the transitions of the episode
                ep_obs.append(obs.copy())
                ep_achieved_goals.append([obs[8].copy() + obs[4].copy(), obs[9].copy() + obs[5].copy()])
                ep_goals.append([obs[4].copy(), obs[5].copy()])
                ep_actions.append(action.copy())
                ep_next_obs.append(next_obs.copy())
                ep_next_achieved_goals.append([next_obs[8].copy() + next_obs[4].copy(), next_obs[9].copy() + next_obs[5].copy()])
                ep_rewards.append(reward)
                ep_dones.append(done)

                obs = next_obs
            
            ep_obs.append(obs.copy())
            ep_achieved_goals.append([obs[8].copy() + obs[4].copy(), obs[9].copy() + obs[5].copy()])
            ep_next_obs.append(next_obs.copy())
            ep_next_achieved_goals.append([next_obs[8].copy() + next_obs[4].copy(), next_obs[9].copy() + next_obs[5].copy()])

            self.buffer.store_episode(ep_obs, ep_achieved_goals, ep_goals, ep_actions, ep_next_obs, ep_next_achieved_goals, ep_rewards, ep_dones)

            success = self.isSuccess(info=info)
            success_count += int(success)
            success_rate = success_count / episode if episode > 0 else 0
                
            if done or step_count >= self.max_episode_steps:
                episode += 1
                step_count = 0 

            

            if self.buffer.n_transition_stored > self.batch_size:
                critic_loss = self.sac.update(self.buffer, self.step)

                rewards.append(episode_reward)
                critic_losses.append(critic_loss)
                success_rates.append(success_rate)
                episodes.append(self.step)
                if self.step % 100 == 0:
                    print(f"Epoch: {self.step +1}, Critic Loss: {critic_loss:.4f}, Episode Reward: {episode_reward:.4f}, "
                          f"Rollout Episode: {episode}, Success Rate: {success_rate:.4f}, Time: {time.time() - start_time:.4f}")
            
            

            #evaluate agent 
            if self.step % eval_step == 0:
                print("Periodic agent evaluation...")
                self.evaluate()

            self.step += 1
            self.epsilon *= self.eps_dacay
        print(f"Training complete.")
        
        #save_plots(rewards, critic_losses, success_rates, episodes, output_dir='simple_buffer_plots')
        save_plots(rewards, critic_losses, success_rates, episodes, output_dir='HGR_buffer_plots')

    def evaluate(self, eval_episodes=3):
        avg_reward = 0
        for _ in range(eval_episodes):
            obs, _ = self.env.reset()
            goal = [obs[4], obs[5]]
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                with eval_mode(self.sac):
                    action = self.sac.actor.act(obs, goal, True)
                #print(action)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_step +=1
            #print(info)
            avg_reward += episode_reward
        avg_reward = avg_reward/eval_episodes
        print(f'Average reward of evaluation at step {self.step}: {avg_reward:.4f}')

    def isSuccess(self, info, treshold=0.05):
        if np.abs(info['reward_dist']) < treshold:
            return True
        
        return False
    
    def save(self, path='model.pt'):
        torch.save({
            'policy_state_dict': self.state_dict(),  # Save Policy class state
            'sac': {
                'actor_state_dict': self.sac.actor.state_dict(),
                'critic_state_dict': self.sac.critic.state_dict(),
                'target_critic_state_dict': self.sac.target_critic.state_dict(),
                'actor_optimizer_state_dict': self.sac.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.sac.critic_optimizer.state_dict(),
                'log_alpha_optimizer_state_dict': self.sac.log_alpha_optimizer.state_dict(),
                'log_alpha': self.sac.log_alpha.detach().cpu().numpy(),  # Save log_alpha as a plain value
            },
            'step': self.step,  # Save current training step
            'epsilon': self.epsilon  # Save epsilon value
        }, path)
        print(f"Model and components saved to {path}")


    def load(self, path='model.pt'):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['policy_state_dict'])  # Load Policy class state

        # Load SAC components
        self.sac.actor.load_state_dict(checkpoint['sac']['actor_state_dict'])
        self.sac.critic.load_state_dict(checkpoint['sac']['critic_state_dict'])
        self.sac.target_critic.load_state_dict(checkpoint['sac']['target_critic_state_dict'])
        self.sac.actor_optimizer.load_state_dict(checkpoint['sac']['actor_optimizer_state_dict'])
        self.sac.critic_optimizer.load_state_dict(checkpoint['sac']['critic_optimizer_state_dict'])
        self.sac.log_alpha_optimizer.load_state_dict(checkpoint['sac']['log_alpha_optimizer_state_dict'])
        self.sac.log_alpha = torch.tensor(checkpoint['sac']['log_alpha'], device=self.device, requires_grad=True)

        # Restore training state
        self.step = checkpoint['step']
        self.epsilon = checkpoint['epsilon']

        print(f"Model and components loaded from {path}")



    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    

    
    