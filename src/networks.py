import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Policy_Network(nn.Module):
    def __init__(self, input_size:int, output_size:int, use_softmax:bool):
        super(Policy_Network, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 512)
        self.use_softmax = use_softmax

        self.action_head = nn.Linear(512, output_size)

    def forward(self, x):
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        if self.use_softmax:
            action_probs = F.softmax(self.action_head(x), dim=-1)
        else:
            action_probs = self.action_head(x) # for cross entropy loss
        return action_probs
    
    def set_softmax(self, t:bool):
        self.use_softmax = t
        

class Value_Network(nn.Module):
    def __init__(self, input_size:int, output_size:int):
        super(Value_Network, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 512)

        self.value_head = nn.Linear(512, output_size)

    def forward(self, x):
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.value_head(x) 
