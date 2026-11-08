#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataloaders import get_dataloaders
from main import get_embeddings
from pipeline import MultimodalHatefulMemeModel
from utils import DEVICE


@torch.no_grad()
def main(config_path: str, split: str):
    with open(config_path) as f:
        config = json.load(f)

    embeddings = get_embeddings(config, splits=(split,))
    img_emb, text_emb, _labels, ids = embeddings[split]

    checkpoint_path = Path(config["OUTPUT_DIR"]) / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = MultimodalHatefulMemeModel(config, img_emb.shape[1], text_emb.shape[1])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    scores = [
        model.get_scores(batch["image"].to(DEVICE), batch["text"].to(DEVICE))
        .cpu()
        .numpy()
        for batch in get_dataloaders(config, embeddings)[split]
    ]

    out_path = Path(config["OUTPUT_DIR"]) / f"predictions_{split}.csv"
    pd.DataFrame(
        {"meme_id": ids, "hateful_probability": np.concatenate(scores)}
    ).to_csv(out_path, index=False)
    print(
        f"Wrote {len(ids)} predictions to {out_path} "
        f"(checkpoint epoch {checkpoint['epoch']}, val AUROC {checkpoint['val_auroc']:.4f})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write meme_id,hateful_probability predictions"
    )
    parser.add_argument(
        "config", help="Config of the trained run to load the checkpoint from"
    )
    parser.add_argument(
        "--split", default="test", help="Split to score (default: test)"
    )
    args = parser.parse_args()
    main(args.config, args.split)
