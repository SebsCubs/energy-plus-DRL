import torch
import torch.nn as nn
import numpy as np

class Policy(nn.Module):
    def __init__(self, input_shape, action_size):
        super(Policy, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(np.prod(input_shape), 512)
        self.fc2 = nn.Linear(512, action_size)
        self.fc3 = nn.Linear(512, 1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.elu = nn.ELU()
        self.input_shape = input_shape
        self.action_size = action_size

        
    def forward(self, x):
        x = self.flatten(x)
        x = self.elu(self.fc1(x))
        action_probs = self.softmax(self.fc2(x))
        state_value = self.fc3(x)
        return action_probs, state_value
    
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self(state)
        action = np.random.choice(self.action_size, p=action_probs.numpy().squeeze())
        return action