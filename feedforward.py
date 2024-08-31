# feedforward.py
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class DropoutLayer(nn.Module):
    def __init__(self, p):
        super(DropoutLayer, self).__init__()
        self.dropout = nn.Dropout(p)
    
    def forward(self, x):
        return self.dropout(x)

class ActivationLayer(nn.Module):
    def forward(self, x):
        return F.relu(x)
