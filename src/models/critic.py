import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape, goal_shape, hidden_dim=256, device='cpu'):
        super(Critic, self).__init__()

        self.device = device

        self.Q1 = nn.Sequential (
                nn.Linear(obs_shape + action_shape + goal_shape, hidden_dim), 
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1) #fc3
            ) 

        self.Q2 = nn.Sequential(
            nn.Linear(obs_shape + action_shape + goal_shape, hidden_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1) #fc3
        )
        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

    def forward(self, obs, action, goal):
        x = torch.cat([obs, action, goal], dim=-1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2


    