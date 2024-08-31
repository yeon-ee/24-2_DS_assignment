# normalization.py
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, d_model):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        return self.norm(x)
