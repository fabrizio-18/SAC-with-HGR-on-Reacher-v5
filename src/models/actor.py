import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, state_size, action_size, goal_size, hidden_size=256, init_w=3e-3, log_std_min=-20, log_std_max=2, device='cpu'):
        super(Actor, self).__init__()
        self.device = device

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_size + goal_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, action_size)
        self.mean_linear.weight.data = torch.nn.init.uniform_(self.mean_linear.weight, -init_w, init_w)
        self.mean_linear.bias.data = torch.nn.init.uniform_(self.mean_linear.bias, -init_w, init_w)

        
        self.log_std_linear = nn.Linear(hidden_size, action_size)
        self.log_std_linear.weight.data = torch.nn.init.uniform_(self.log_std_linear.weight, -init_w, init_w)
        self.log_std_linear.bias.data = torch.nn.init.uniform_(self.log_std_linear.bias, -init_w, init_w)  
    
    def forward(self, state, goal):
        x = F.relu(self.linear1(torch.cat([state, goal], dim=-1)))
        x = F.relu(self.linear2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        dist = Normal(mean, log_std.exp())
        return dist
    
    def evaluate(self, state, goal, epsilon=1e-6):
        dist = self.forward(state, goal)

        action = torch.tanh(dist.sample())

        #''''torch.log(1 - action.pow(2) + epsilon'''' is an adjustment that compensates for the squeezing of the tanh transformation
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1)
        #log_prob = log_prob - torch.log(1 - action.pow(2) + epsilon).sum(dim=-1)
        
        return action, log_prob
    
    def get_action(self, state, goal):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        goal = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
        dist = self.forward(state, goal)
        action = torch.tanh(dist.sample())

        action = action.cpu().detach().numpy()
        return action[0]


