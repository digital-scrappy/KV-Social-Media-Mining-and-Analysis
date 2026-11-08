#!/usr/bin/env python
import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from main import get_embeddings, run_one_seed

DEFAULT_LAMBDAS = [0.0, 0.1, 0.25, 0.5, 1.0]

GRID_KEYS = {
    "lr": "LR",
    "wd": "WEIGHT_DECAY",
    "bs": "BATCH_SIZE",
    "common_dim": "COMMON_DIM",
    "dropout": "DROPOUT",
}


def _report(results: list, out_dir: Path, filename: str) -> dict:
    results = sorted(results, key=lambda r: -r["val_auroc"])
    frame = pd.DataFrame(results)
    frame.to_csv(out_dir / filename, index=False)
    print("\n" + frame.to_string(index=False))
    return results[0]


def grid_search(
    base_config: dict, embeddings: dict, img_dim: int, text_dim: int, seed: int
) -> list:
    grid = base_config["GRID"]
    results = []
    for values in itertools.product(*grid.values()):
        combo = dict(zip(grid, values))
        config = {
            **base_config,
            **{GRID_KEYS[k]: v for k, v in combo.items()},
            "LAMBDA_OP": 0.0,
        }
        label = " ".join(f"{k}={v}" for k, v in combo.items())
        print(f"\n=== {label} (seed {seed}) ===")
        row, _, _ = run_one_seed(
            config, seed, embeddings, img_dim, text_dim, checkpoint=False
        )
        print(f"{label}: val AUROC {row['val_auroc']:.4f}")
        results.append({**combo, "seed": seed, "val_auroc": row["val_auroc"]})
    return results


def lambda_sweep(
    base_config: dict,
    embeddings: dict,
    img_dim: int,
    text_dim: int,
    seed: int,
    lambdas: list,
) -> list:
    results = []
    for lam in lambdas:
        config = {**base_config, "LAMBDA_OP": lam}
        print(f"\n=== LAMBDA_OP={lam} (seed {seed}) ===")
        row, _, _ = run_one_seed(
            config, seed, embeddings, img_dim, text_dim, checkpoint=False
        )
        print(f"LAMBDA_OP={lam}: val AUROC {row['val_auroc']:.4f}")
        results.append({"lambda_op": lam, "seed": seed, "val_auroc": row["val_auroc"]})
    return results


def main(config_path: str, lambdas: list):
    with open(config_path) as f:
        base_config = json.load(f)

    embeddings = get_embeddings(base_config)
    img_dim = embeddings["train"][0].shape[1]
    text_dim = embeddings["train"][1].shape[1]
    seed = base_config["SEEDS"][0]

    out_dir = Path("outputs") / f"{base_config['EXP_NAME']}_search"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n### Stage 1: grid search {', '.join(base_config['GRID'])} at LAMBDA_OP=0 ###"
    )
    best = _report(
        grid_search(base_config, embeddings, img_dim, text_dim, seed),
        out_dir,
        "grid_search.csv",
    )
    recommended = {GRID_KEYS[k]: best[k] for k in base_config["GRID"]}
    print(
        f"\nBest stage-1 hyperparameters: {recommended} (val AUROC {best['val_auroc']:.4f})"
    )

    print("\n### Stage 2: sweep LAMBDA_OP with the stage-1 winner fixed ###")
    best_lambda = _report(
        lambda_sweep(
            {**base_config, **recommended}, embeddings, img_dim, text_dim, seed, lambdas
        ),
        out_dir,
        "lambda_sweep.csv",
    )
    recommended["LAMBDA_OP"] = best_lambda["lambda_op"]
    with open(out_dir / "recommended_hparams.json", "w") as f:
        json.dump(recommended, f, indent=2)
    print(
        f"\nRecommended hyperparameters written to {out_dir / 'recommended_hparams.json'}:"
    )
    print(json.dumps(recommended, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-stage hyperparameter search by validation AUROC"
    )
    parser.add_argument(
        "config", help="Base config (encoder pair + architecture + GRID)"
    )
    parser.add_argument(
        "--lambdas",
        default=",".join(map(str, DEFAULT_LAMBDAS)),
        help="Comma-separated candidate LAMBDA_OP values for stage 2",
    )
    args = parser.parse_args()
    main(args.config, [float(x) for x in args.lambdas.split(",")])
