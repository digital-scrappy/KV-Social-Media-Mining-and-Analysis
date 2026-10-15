"""Shared code for the unimodal hateful-meme baselines.
Both me and Jakob import this file so that the classifier, training loop, evaluation and seeds are identical across all four experiments.
"""

import itertools
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPModel,
    AutoModel,
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------- DATA LOADING
def load_split(data_dir, name):
    """Reads {data_dir}/{name}.jsonl -> list of dicts (id, img, text, label)."""
    rows = []
    with open(os.path.join(data_dir, f"{name}.jsonl"), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def describe_splits(data_dir, names=("train", "dev", "test")):
    """Prints split sizes and class counts (if the labels exist)."""
    for n in names:
        rows = load_split(data_dir, n)
        labels = [r["label"] for r in rows if "label" in r]
        if labels:
            pos = int(sum(labels))
            print(
                f"{n}: {len(rows)} examples | hateful=1: {pos} | non-hateful=0: {len(labels) - pos}"
            )
        else:
            print(f"{n}: {len(rows)} examples | no labels")


def get_labels(rows):
    return np.array([r["label"] for r in rows], dtype=np.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------- EXTRACTING EMBEDDINGS
@torch.no_grad()
def extract_image_embeddings(rows, data_dir, model_name, kind, batch_size=64):
    """kind: 'clip' -> CLIPModel.get_image_features (projected, e.g. 512-d)
             'cls'  -> AutoModel last_hidden_state[:, 0] (CLS token, e.g. DINOv2 768-d)
    Image pixels only; the processor does the required resize + normalization."""
    proc = AutoImageProcessor.from_pretrained(model_name)
    if kind == "clip":
        model = CLIPModel.from_pretrained(mode_name).to(DEVICE).eval()
    if kind == "emb":
        model = SentenceTransformer(model_name).to(DEVICE).eval()
    else:
        model = AutoModel.from_pretrained(mode_name).to(DEVICE).eval()
    out = []
    for i in range(0, len(rows), batch_size):
        imgs = [
            Image.open(os.path.join(data_dir, r["img"])).convert("RGB")
            for r in rows[i : i + batch_size]
        ]
        inputs = proc(images=imgs, return_tensors="pt").to(DEVICE)
        if kind == "clip":
            emb = model.get_image_features(**inputs)
        elif kind == "emb":
            emb = model.encode([{"image": img} for img in inputs])
        else:
            emb = model(**inputs).last_hidden_state[:, 0]
        if not isinstance(emb, torch.Tensor):
            emb = emb.pooler_output
        out.append(emb.float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def extract_text_embeddings(
    rows, model_name, pooling="mean", max_len=128, batch_size=64
):
    """pooling: 'mean' (attention-masked mean of last hidden state) or 'cls'.
    Truncation to max_len tokens; meme text only."""
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    out = []
    for i in range(0, len(rows), batch_size):
        texts = [r["text"] for r in rows[i : i + batch_size]]
        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(DEVICE)
        h = model(**enc).last_hidden_state
        if pooling == "cls":
            emb = h[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1).float()
            emb = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(emb.float().cpu())
    return torch.cat(out).numpy()


# ------------------------------------------------------------ CLASSIFIER
class LinearProbe(nn.Module):
    """Single linear layer, one output logit. Only in_dim changes."""

    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)  # logit


def standardize(train, *others):
    """z-score using TRAIN statistics only (same for all four experiments)."""
    mu, sd = train.mean(0, keepdims=True), train.std(0, keepdims=True) + 1e-8
    return [(a - mu) / sd for a in (train, *others)]


@torch.no_grad()
def predict_scores(model, X, batch_size=1024):
    """Sigmoid of the logit = hateful-class score in [0, 1]."""
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    s = [
        torch.sigmoid(model(X[i : i + batch_size].to(DEVICE))).cpu()
        for i in range(0, len(X), batch_size)
    ]
    return torch.cat(s).numpy()


def auroc(y_true, scores):
    """Single evaluation function used everywhere (continuous scores)."""
    return float(roc_auc_score(y_true, scores))


def train_linear(Xtr, ytr, Xva, yva, lr, wd, bs, epochs=100, seed=0, patience=15):
    """Adam + BCE-with-logits. Keeps the epoch with the best validation AUROC.
    Returns (best_state_dict, best_val_auroc, best_epoch)."""
    set_seed(seed)
    model = LinearProbe(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    best, best_state, best_ep, bad = -1.0, None, -1, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g)
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx].to(DEVICE)), ytr_t[idx].to(DEVICE))
            loss.backward()
            opt.step()
        val = auroc(yva, predict_scores(model, Xva))
        if val > best:
            best, best_ep, bad = val, ep, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            bad += 1
            if bad >= patience:
                break
    return best_state, best, best_ep


def grid_search(Xtr, ytr, Xva, yva, grid, seed=0, epochs=100):
    """Same grid for every experiment. Selection by validation AUROC."""
    results = []
    for lr, wd, bs in itertools.product(grid["lr"], grid["wd"], grid["bs"]):
        _, val, ep = train_linear(
            Xtr, ytr, Xva, yva, lr, wd, bs, epochs=epochs, seed=seed
        )
        results.append(
            {"lr": lr, "wd": wd, "bs": bs, "val_auroc": val, "best_epoch": ep}
        )
    results.sort(key=lambda r: -r["val_auroc"])
    return results


def final_runs(Xtr, ytr, Xva, yva, Xte, yte, best, seeds, epochs=100):
    """Retrains with the best hyperparameters for each seed.
    Returns per-seed val/test AUROC (test only if yte is not None) and scores of seed seeds[0]."""
    per_seed, first = [], None
    for s in seeds:
        state, val, ep = train_linear(
            Xtr,
            ytr,
            Xva,
            yva,
            best["lr"],
            best["wd"],
            best["bs"],
            epochs=epochs,
            seed=s,
        )
        model = LinearProbe(Xtr.shape[1]).to(DEVICE)
        model.load_state_dict(state)
        row = {"seed": s, "val_auroc": val, "best_epoch": ep}
        va_scores = predict_scores(model, Xva)
        te_scores = predict_scores(model, Xte) if Xte is not None else None
        if te_scores is not None and yte is not None:
            row["test_auroc"] = auroc(yte, te_scores)
        per_seed.append(row)
        if first is None:
            first = {"state": state, "val_scores": va_scores, "test_scores": te_scores}
    return per_seed, first


def summarize(per_seed, key):
    v = [r[key] for r in per_seed if key in r]
    return (float(np.mean(v)), float(np.std(v))) if v else (None, None)
