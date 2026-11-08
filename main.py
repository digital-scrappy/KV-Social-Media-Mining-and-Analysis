#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataloaders import get_dataloaders
from encoders import extract_image_embeddings, extract_text_embeddings
from pipeline import MultimodalHatefulMemeModel
from trainer import HatefulMemeTrainer
from utils import DEVICE, set_seed

EMBEDDINGS_DIR = Path("embeddings_cache")


def _sanitize(model_name: str) -> str:
    return model_name.replace("/", "__")


def _load_rows(data_dir: str, split: str) -> list:
    with open(Path(data_dir) / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _load_or_extract(path: Path, extract_fn):
    if path.exists():
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    emb = extract_fn()
    np.save(path, emb)
    return emb


def get_embeddings(config: dict, splits=("train", "dev")) -> dict:
    img_key = f"{_sanitize(config['IMG_MODEL'])}_{config['IMG_KIND']}"
    text_key = (
        f"{_sanitize(config['TEXT_MODEL'])}_{config['POOLING']}_{config['MAX_LEN']}"
    )

    embeddings = {}
    for split in splits:
        rows = _load_rows(config["DATA_DIR"], split)
        img_emb = _load_or_extract(
            EMBEDDINGS_DIR / img_key / f"{split}.npy",
            lambda: extract_image_embeddings(
                rows, config["DATA_DIR"], config["IMG_MODEL"], config["IMG_KIND"]
            ),
        )
        text_emb = _load_or_extract(
            EMBEDDINGS_DIR / text_key / f"{split}.npy",
            lambda: extract_text_embeddings(
                rows, config["TEXT_MODEL"], config["POOLING"], config["MAX_LEN"]
            ),
        )
        labels = np.array([r.get("label", -1) for r in rows], dtype=np.int64)
        print(f"{split}: image {img_emb.shape}, text {text_emb.shape}")
        embeddings[split] = (img_emb, text_emb, labels, [r["id"] for r in rows])

    return embeddings


def run_one_seed(
    config: dict,
    seed: int,
    embeddings: dict,
    img_dim: int,
    text_dim: int,
    checkpoint: bool,
):
    """Trains and selects one seed"""
    set_seed(seed)
    model = MultimodalHatefulMemeModel(config, img_dim, text_dim).to(DEVICE)
    dataloaders = get_dataloaders(config, embeddings, seed=seed)
    trainer = HatefulMemeTrainer(model, config, checkpoint=checkpoint)

    history = trainer.train(
        dataloaders["train"], dataloaders["dev"], config["EPOCHS"], config["PATIENCE"]
    )
    row = {
        "seed": seed,
        "val_auroc": trainer.best_val_auroc,
        "trainable_params": sum(p.numel() for p in model.parameters()),
    }
    dev_predictions = trainer.predict(dataloaders["dev"]) if checkpoint else None
    return row, history, dev_predictions


def main(config_path: str):
    with open(config_path) as f:
        config = json.load(f)

    out_dir = Path(config["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    embeddings = get_embeddings(config)
    img_dim = embeddings["train"][0].shape[1]
    text_dim = embeddings["train"][1].shape[1]

    seeds = config["SEEDS"]
    per_seed = []
    for i, seed in enumerate(seeds):
        print(f"\n=== Seed {seed} ({i + 1}/{len(seeds)}) ===")
        row, history, dev_predictions = run_one_seed(
            config, seed, embeddings, img_dim, text_dim, checkpoint=(i == 0)
        )
        per_seed.append(row)
        print(f"Seed {seed}: val AUROC {row['val_auroc']:.4f}")

        if i == 0:
            pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
            dev_labels, dev_scores = dev_predictions
            pd.DataFrame(
                {
                    "id": embeddings["dev"][3],
                    "label": dev_labels,
                    "score": dev_scores,
                }
            ).to_csv(out_dir / "val_scores.csv", index=False)

    val_aurocs = [r["val_auroc"] for r in per_seed]
    print(
        f"\nVal AUROC over {len(seeds)} seeds: "
        f"{np.mean(val_aurocs):.4f} +/- {np.std(val_aurocs):.4f}"
    )
    print(f"Trainable parameters: {per_seed[0]['trainable_params']:,}")

    summary = {
        "exp": config["EXP_NAME"],
        "model": f"{config['IMG_MODEL']}+{config['TEXT_MODEL']}",
        "modality": config["MODALITY"],
        "fusion": config["FUSION"],
        "lambda_op": config["LAMBDA_OP"],
        "common_dim": config["COMMON_DIM"],
        "dropout": config["DROPOUT"],
        "trainable_params": per_seed[0]["trainable_params"],
        "seeds": seeds,
        "per_seed": per_seed,
        "val_auroc_mean": float(np.mean(val_aurocs)),
        "val_auroc_std": float(np.std(val_aurocs)),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal hateful meme classification"
    )
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
