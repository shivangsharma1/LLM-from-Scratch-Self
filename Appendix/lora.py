import math
import torch
import torch.nn as nn


class LoraLayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=10):
        super().__init__()

        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_normal(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):

        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLora(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoraLayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x) 
