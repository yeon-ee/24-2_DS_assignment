# feedforward.py
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Feedforward Layer
class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.relu(self.fc1(x)))

# Dropout Layer
class DropoutLayer(nn.Module):
    def __init__(self, p: float) -> None:
        super(DropoutLayer, self).__init__()
        self.dropout = nn.Dropout(p)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x)

# Activation Layer
class ActivationLayer(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)