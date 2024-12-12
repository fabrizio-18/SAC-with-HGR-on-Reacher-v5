import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_size, action_size, goal_size, hidden_size = 256, init_w=3e-3, device='cpu'):
        super(Critic, self).__init__()
        self.device = device
        self.linear1 = nn.Linear(state_size + goal_size + action_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight = torch.nn.init.uniform_(self.linear3.weight, -init_w, init_w)
        self.linear3.bias = torch.nn.init.uniform_(self.linear3.bias, -init_w, init_w)


    def forward(self, state, goal, action):
        #state = torch.FloatTensor(state)
        #goal = torch.FloatTensor(goal)
        #action = torch.FloatTensor(action)
                
        x = torch.cat([state, goal, action],dim=-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x