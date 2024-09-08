# residual.py
import torch.nn as nn
from torch import Tensor

# Residual Connection
class ResidualConnection(nn.Module):
    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        return x + sublayer(x)