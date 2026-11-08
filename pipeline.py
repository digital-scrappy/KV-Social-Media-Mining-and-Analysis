#!/usr/bin/env python3

import torch
import torch.nn as nn

from fusion import GatedFusion, LinearFusion
from projections import EmbedBranch


class MultimodalHatefulMemeModel(nn.Module):
    """
    config["FUSION"] picks "gated" or "linear"
    """

    def __init__(self, config: dict, img_dim: int, text_dim: int):
        super().__init__()
        d = config["COMMON_DIM"]
        self.img_proj = EmbedBranch(img_dim, d, config["DROPOUT"])
        self.text_proj = EmbedBranch(text_dim, d, config["DROPOUT"])
        self.fusion = GatedFusion(d) if config["FUSION"] == "gated" else LinearFusion()
        self.head = nn.Linear(d, 2)

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor):
        """Returns (raw logits, fused embedding z); z feeds the orthogonal projection loss."""
        z = self.fusion(self.img_proj(image_emb), self.text_proj(text_emb))
        return self.head(z), z

    def get_scores(
        self, image_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> torch.Tensor:
        """Hateful-class probability per meme: softmax(logits)[:, 1]."""
        logits, _ = self.forward(image_emb, text_emb)
        return torch.softmax(logits, dim=-1)[:, 1]
