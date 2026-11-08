#!/usr/bin/env python3

import torch.nn as nn
import torch.nn.functional as F


class EmbedBranch(nn.Module):
    """Projects frozen-encoder features to the common dimension d, L2-normalized."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=-1)
