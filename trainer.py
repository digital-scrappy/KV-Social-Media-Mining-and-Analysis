#!/usr/bin/env python3

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from losses import OrthogonalProjectionLoss
from utils import DEVICE


class HatefulMemeTrainer:
    """Trains the multimodal model"""

    def __init__(self, model: nn.Module, config: dict, checkpoint: bool = True):
        self.model = model
        self.config = config
        self.checkpoint = checkpoint
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"]
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.op_loss = OrthogonalProjectionLoss()
        self.lambda_op = config["LAMBDA_OP"]
        self.best_val_auroc = 0.0
        self.patience_counter = 0

    def train_epoch(self, train_loader) -> dict:
        self.model.train()
        totals = {"loss": 0.0, "ce_loss": 0.0, "op_loss": 0.0}

        for batch in tqdm(train_loader, desc="Training"):
            images, texts, labels = (
                batch[k].to(DEVICE) for k in ("image", "text", "label")
            )
            logits, z = self.model(images, texts)
            ce = self.ce_loss(logits, labels)
            op = self.op_loss(z, labels)
            loss = ce + self.lambda_op * op

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for key, value in zip(totals, (loss, ce, op)):
                totals[key] += value.item()

        return {key: value / len(train_loader) for key, value in totals.items()}

    @torch.no_grad()
    def predict(self, data_loader):
        """(labels, hateful-class scores) in dataset order - dev/test loaders are unshuffled.
        Labels are the -1 placeholder for the unlabeled test split (see main.get_embeddings)."""
        self.model.eval()
        labels, scores = [], []
        for batch in tqdm(data_loader, desc="Predicting"):
            scores.append(
                self.model.get_scores(
                    batch["image"].to(DEVICE), batch["text"].to(DEVICE)
                )
                .cpu()
                .numpy()
            )
            labels.append(batch["label"].numpy())
        return np.concatenate(labels), np.concatenate(scores)

    def evaluate(self, data_loader) -> float:
        return float(roc_auc_score(*self.predict(data_loader)))

    def train(self, train_loader, val_loader, epochs: int, patience: int) -> list:
        """Full training loop wit hearly stopping"""
        history = []
        best_state = None

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            losses = self.train_epoch(train_loader)
            val_auroc = self.evaluate(val_loader)
            history.append({"epoch": epoch, **losses, "val_auroc": val_auroc})
            print(f"Train loss: {losses['loss']:.4f} | Val AUROC: {val_auroc:.4f}")

            if val_auroc > self.best_val_auroc:
                self.best_val_auroc = val_auroc
                self.patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
                if self.checkpoint:
                    self.save_checkpoint(epoch, val_auroc)
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self.model.load_state_dict(best_state)
        return history

    def save_checkpoint(self, epoch: int, val_auroc: float):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "val_auroc": val_auroc,
                "config": self.config,
            },
            Path(self.config["OUTPUT_DIR"]) / "best_model.pt",
        )
        print(f"Saved checkpoint (val AUROC {val_auroc:.4f})")
