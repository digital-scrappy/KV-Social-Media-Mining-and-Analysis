#!/usr/bin/env python3
import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Feature-level gated fusion of projected image and text features"""

    def __init__(self, input_dim: int):
        super().__init__()
        # Gating network A: [u;v] -> g
        self.gate_net = nn.Sequential(
            nn.Linear(2 * input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        g = self.gate_net(torch.cat([u, v], dim=-1))
        return g * torch.tanh(u) + (1 - g) * torch.tanh(v)


class LinearFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight1 = nn.Parameter(torch.rand(1))
        self.weight2 = nn.Parameter(torch.rand(1))

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.weight1 * u + self.weight2 * v
