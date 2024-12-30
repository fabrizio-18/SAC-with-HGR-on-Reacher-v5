import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, goal_shape, hidden_dim=256, log_std_min=-10, log_std_max=2, device='cpu'):
        super(Actor, self).__init__()

        self.device = device
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(obs_shape + goal_shape, hidden_dim)
        nn.init.orthogonal_(self.linear1.weight.data)
        self.linear1.bias.data.fill_(0.0)
        
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.orthogonal_(self.linear2.weight.data)
        self.linear2.bias.data.fill_(0.0)

        self.mean_linear = nn.Linear(hidden_dim, action_shape)
        nn.init.orthogonal_(self.mean_linear.weight.data)
        self.mean_linear.bias.data.fill_(0.0)

        self.log_std_linear = nn.Linear(hidden_dim, action_shape)
        nn.init.orthogonal_(self.log_std_linear.weight.data)
        self.log_std_linear.bias.data.fill_(0.0)

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=-1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        mu = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)                                                                    
        std = log_std.exp()
        
        dist = SquashedNormal(mu, std)

        return dist  
    
    def act(self, obs, goal, eval=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        goal = torch.FloatTensor(goal).unsqueeze(0).to(self.device)

        dist = self.forward(obs, goal)
        action = dist.sample() if not eval else dist.mean
        action = torch.clamp(action, -1, 1)

        action  = action.cpu().detach().numpy()
        #print(action[0])
        return action[0]
    
    