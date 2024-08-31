# residual.py
import torch.nn as nn

class ResidualConnection(nn.Module):
    def forward(self, x, sublayer):
        return x + sublayer(x)
