"""Critic Networks (Soft Q-functions)"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    """Soft Q-function approximator"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        
        return q_value
