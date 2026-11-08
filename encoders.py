#!/usr/bin/env python3
import os

import torch
import torch.nn as nn
from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPModel,
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer
from utils import DEVICE


class ImageEncoder(nn.Module):
    """Frozen image encoder; `kind` picks the loading/forward path per architecture."""

    def __init__(self, model_name, kind):
        super().__init__()
        self.kind = kind
        if kind == "clip":
            self.proc = AutoImageProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()
        elif kind == "emb":
            self.model = SentenceTransformer(model_name).to(DEVICE)
        else:
            self.proc = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()

    @torch.no_grad()
    def __call__(self, imgs):
        """imgs: list of PIL.Image, raw (no resize/normalize applied upstream)."""
        if self.kind == "clip":
            emb = self.model.get_image_features(
                **self.proc(images=imgs, return_tensors="pt").to(DEVICE)
            )
        elif self.kind == "emb":
            emb = self.model.encode(
                [{"image": im} for im in imgs],
                batch_size=len(imgs),
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        else:
            emb = self.model(
                **self.proc(images=imgs, return_tensors="pt").to(DEVICE)
            ).last_hidden_state[:, 0]
        return emb.float()


class TextEncoder(nn.Module):
    def __init__(self, model_name, pooling="mean", max_len=128):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
        self.pooling = pooling
        self.max_len = max_len

    @torch.no_grad()
    def __call__(self, texts):
        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(DEVICE)
        h = self.model(**enc).last_hidden_state
        if self.pooling == "cls":
            return h[:, 0].float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        return ((h * m).sum(1) / m.sum(1).clamp(min=1)).float()


def extract_image_embeddings(rows, data_dir, model_name, kind, batch_size=64):
    """One embedding per row (same order), via a frozen ImageEncoder. Run once; the
    caller is responsible for caching the result to disk."""
    enc = ImageEncoder(model_name, kind)
    out = []
    for i in range(0, len(rows), batch_size):
        imgs = [
            Image.open(os.path.join(data_dir, r["img"])).convert("RGB")
            for r in rows[i : i + batch_size]
        ]
        out.append(enc(imgs).cpu())
    return torch.cat(out).numpy()


def extract_text_embeddings(rows, model_name, pooling="mean", max_len=128, batch_size=64):
    """One embedding per row (same order), via a frozen TextEncoder. Run once; the
    caller is responsible for caching the result to disk."""
    enc = TextEncoder(model_name, pooling, max_len)
    out = []
    for i in range(0, len(rows), batch_size):
        texts = [r["text"] for r in rows[i : i + batch_size]]
        out.append(enc(texts).cpu())
    return torch.cat(out).numpy()
