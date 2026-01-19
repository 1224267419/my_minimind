import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, hidden_states):
        variance = hidden_states.var(-1, unbiased=False) + self.variance_epsilon
        hidden_states = hidden_states / torch.sqrt(variance)
        return self.weight * hidden_states
