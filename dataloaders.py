#!/usr/bin/env python3

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class HatefulMemesDataset(Dataset):
    """Wraps precomputed (image, text) embeddings + labels for one split."""

    def __init__(self, image_emb, text_emb, labels):
        self.image_emb = image_emb
        self.text_emb = text_emb
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": torch.as_tensor(self.image_emb[idx], dtype=torch.float32),
            "text": torch.as_tensor(self.text_emb[idx], dtype=torch.float32),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


class ClassBalancedBatchSampler(Sampler):
    """Draws train batches that keep the split's class ratio, so OrthogonalProjectionLoss
    always sees both same-label and different-label pairs
    """

    def __init__(self, labels, batch_size: int, generator: torch.Generator):
        labels = np.asarray(labels)
        self.pos_idx = np.where(labels == 1)[0]
        self.neg_idx = np.where(labels == 0)[0]
        self.generator = generator
        self.n_pos = round(batch_size * len(self.pos_idx) / len(labels))
        self.n_neg = batch_size - self.n_pos

    def _shuffled(self, idx):
        return idx[torch.randperm(len(idx), generator=self.generator).numpy()]

    def __iter__(self):
        pos, neg = self._shuffled(self.pos_idx), self._shuffled(self.neg_idx)
        for i in range(len(self)):
            batch = np.concatenate(
                [
                    pos[i * self.n_pos : (i + 1) * self.n_pos],
                    neg[i * self.n_neg : (i + 1) * self.n_neg],
                ]
            )
            yield self._shuffled(batch).tolist()

    def __len__(self) -> int:
        return min(len(self.pos_idx) // self.n_pos, len(self.neg_idx) // self.n_neg)


def get_dataloaders(config: dict, embeddings: dict, seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    dataloaders = {}
    for split, (image_emb, text_emb, labels, _ids) in embeddings.items():
        dataset = HatefulMemesDataset(image_emb, text_emb, labels)
        if split == "train":
            dataloaders[split] = DataLoader(
                dataset,
                batch_sampler=ClassBalancedBatchSampler(
                    labels, config["BATCH_SIZE"], generator
                ),
                num_workers=config["NUM_WORKERS"],
            )
        else:
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=config["BATCH_SIZE"],
                shuffle=False,
                num_workers=config["NUM_WORKERS"],
            )
    return dataloaders
