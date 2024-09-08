# normalization.py
import torch.nn as nn
from torch import Tensor

# Layer Normalization
class LayerNormalization(nn.Module):
    def __init__(self, d_model: int) -> None:
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)