import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
	"""Hidden layer initialization"""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, num_agents=2, fc1_units=256, fc2_units=256):
        
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_agents = num_agents
        
        self.action_size = action_size
        self.state_size = state_size
        
        self.fc1 = nn.Linear(state_size * num_agents, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        self.bn1 = nn.BatchNorm1d(fc1_units)

        
        self.reset_parameters()

    def reset_parameters(self):
    	"""Initilaize the weights using uniform distribution"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        
        """Build an actor (policy) network that maps states -> actions."""
        
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)        
        x = F.leaky_relu(self.fc1(state))
        x = self.bn1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        output = torch.tanh(x)
        
        return output


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, num_agents=2, fcs1_units=256, fc2_units=256):
        
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.action_size = action_size
        self.state_size = state_size
        self.num_agents = num_agents
        
        self.fcs1 = nn.Linear(state_size * num_agents, fcs1_units)
        self.fc2 = nn.Linear(action_size * num_agents + fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
        self.bn1 = nn.BatchNorm1d(fcs1_units)        
        self.reset_parameters()

    def reset_parameters(self):
    	"""Initilaize the weights using uniform distribution"""
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)         
        xs = F.leaky_relu(self.fcs1(state))
        xs = self.bn2(xs)
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        output = self.fc3(x)
        
        return output
