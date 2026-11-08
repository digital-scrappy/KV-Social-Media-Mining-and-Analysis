#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalProjectionLoss(nn.Module):
    """Reference OrthogonalProjectionLoss (FOP main.py) with meme labels standing in for
    identity labels. Applied to the fused embeddings z:

        q_i   = z_i / max(||z_i||_2, eps)
        L_OP  = 1 - mean_{(i,j) in P} q_i.q_j + 0.7 * mean_{(i,j) in N} |q_i.q_j|

    P is the distinct same-label pairs (diagonal excluded), N the different-label pairs.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        q = F.normalize(features, p=2, dim=1, eps=self.epsilon)

        same = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
        eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        mask_pos = same.masked_fill(eye, False).float()
        mask_neg = (~same).float()

        dot_prod = q @ q.T
        pos_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + self.epsilon)
        neg_mean = (mask_neg * dot_prod).abs().sum() / (mask_neg.sum() + self.epsilon)

        return (1.0 - pos_mean) + 0.7 * neg_mean
