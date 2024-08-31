# attention.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class QueryLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(QueryLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x):
        return self.linear(x)

class KeyLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(KeyLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x):
        return self.linear(x)

class ValueLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(ValueLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x):
        return self.linear(x)

class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return torch.matmul(attention, v), attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.query_layers = QueryLayer(d_model, n_heads)
        self.key_layers = KeyLayer(d_model, n_heads)
        self.value_layers = ValueLayer(d_model, n_heads)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads * d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        q = self.query_layers(x).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        k = self.key_layers(x).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        v = self.value_layers(x).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        
        attn_output, attn = self.attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_model)
        return self.fc(attn_output)
