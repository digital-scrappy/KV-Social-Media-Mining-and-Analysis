#!/usr/bin/env python
# coding: utf-8

# # Unimodal hateful-meme baseline (template)
# One notebook per experiment. Copy this file, change ONLY the config cell, run top to bottom.
# Shared logic lives in `common.py` so all four runs use identical training/eval code.


import os, json, numpy as np, pandas as pd, torch
import argparse
import common as C


# ---------------- CONFIG (the only cell that differs per experiment) ----------------
DEFAULTS = {
    "EXP_NAME": "image_A",
    "MODALITY": "image",
    "MODEL_NAME": "openai/clip-vit-base-patch32",
    # CHANGED: removed stray `a` after the comma (syntax error)
    "IMG_KIND": "clip",
    "POOLING": "mean",
    "MAX_LEN": 128,
    "DATA_DIR": "data",
    # CHANGED: removed "OUT_DIR" here. The f-string used EXP_NAME, which doesn't
    # exist as a variable at that point (NameError). It is now derived after the
    # JSON is loaded (see bottom).
    "SEEDS": [0, 1, 2],
    "GRID": {
        "lr": [1e-4, 1e-3, 1e-2],
        "wd": [0, 1e-4, 1e-2],
        "bs": [64, 256],
    },
    "EPOCHS": 100,
    "USE_TEST": False,
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("device:", C.DEVICE)
    C.describe_splits(DATA_DIR)

    # ---------------- 1. Extract + cache embeddings (run once) ----------------
    train, dev, test = (C.load_split(DATA_DIR, n) for n in ["train", "dev", "test"])

    def embed(rows):
        if MODALITY == "image":
            return C.extract_image_embeddings(rows, DATA_DIR, MODEL_NAME, IMG_KIND)
        return C.extract_text_embeddings(rows, MODEL_NAME, POOLING, MAX_LEN)

    emb = {}
    for name, rows in [("train", train), ("dev", dev), ("test", test)]:
        path = f"{OUT_DIR}/emb_{name}.npy"
        if os.path.exists(path):
            emb[name] = np.load(path)
        else:
            emb[name] = embed(rows)
            np.save(path, emb[name])
        print(name, emb[name].shape)
    EMB_DIM = emb["train"].shape[1]

    # In[ ]:

    # ---------------- 2. Prepare features / labels ----------------
    ytr, yva = C.get_labels(train), C.get_labels(dev)
    yte = C.get_labels(test) if "label" in test[0] else None
    Xtr, Xva, Xte = C.standardize(emb["train"], emb["dev"], emb["test"])

    # In[ ]:

    # ---------------- 3. Hyperparameter search (validation AUROC, seed = SEEDS[0]) ----------------
    search = C.grid_search(Xtr, ytr, Xva, yva, GRID, seed=SEEDS[0], epochs=EPOCHS)
    pd.DataFrame(search).to_csv(f"{OUT_DIR}/grid_search.csv", index=False)
    best = search[0]
    print("best:", best)
    pd.DataFrame(search).head(10)

    # In[ ]:

    # ---------------- 4. Final runs over all seeds ----------------
    per_seed, first = C.final_runs(
        Xtr,
        ytr,
        Xva,
        yva,
        Xte if USE_TEST else None,
        yte if USE_TEST else None,
        best,
        SEEDS,
        epochs=EPOCHS,
    )
    df = pd.DataFrame(per_seed)
    print(df)
    print("val  AUROC mean/std:", C.summarize(per_seed, "val_auroc"))
    if USE_TEST:
        print("test AUROC mean/std:", C.summarize(per_seed, "test_auroc"))

    # In[ ]:

    # ---------------- 5. Save everything needed by the report / error analysis ----------------
    torch.save(first["state"], f"{OUT_DIR}/classifier_seed{SEEDS[0]}.pt")
    pd.DataFrame(
        {"id": [r["id"] for r in dev], "label": yva, "score": first["val_scores"]}
    ).to_csv(f"{OUT_DIR}/val_scores.csv", index=False)
    if USE_TEST and first["test_scores"] is not None:
        pd.DataFrame(
            {"id": [r["id"] for r in test], "score": first["test_scores"]}
        ).to_csv(f"{OUT_DIR}/test_scores.csv", index=False)
    json.dump(
        {
            "exp": EXP_NAME,
            "model": MODEL_NAME,
            "modality": MODALITY,
            "emb_dim": int(EMB_DIM),
            "img_kind": IMG_KIND if MODALITY == "image" else None,
            "pooling": POOLING if MODALITY == "text" else None,
            "max_len": MAX_LEN if MODALITY == "text" else None,
            "optimizer": "Adam",
            "epochs_max": EPOCHS,
            "seeds": SEEDS,
            "grid": GRID,
            "best_hparams": best,
            "per_seed": per_seed,
        },
        open(f"{OUT_DIR}/summary.json", "w"),
        indent=2,
    )
    print("saved to", OUT_DIR)

    # ## Error analysis (run after BOTH partners have finished)
    # Load `outputs/<best_image>/val_scores.csv` and `outputs/<best_text>/val_scores.csv`, merge on `id`, then pick the 5 dev examples with the largest disagreement at threshold 0.5.

    # In[ ]:

    # BEST_IMG, BEST_TXT = 'image_A', 'text_A'   # fill in after comparing validation AUROC
    # a = pd.read_csv(f'outputs/{BEST_IMG}/val_scores.csv').rename(columns={'score': 'img_score'})
    # b = pd.read_csv(f'outputs/{BEST_TXT}/val_scores.csv').rename(columns={'score': 'txt_score'})[['id', 'txt_score']]
    # m = a.merge(b, on='id')
    # m['disagree'] = (m.img_score >= 0.5) != (m.txt_score >= 0.5)   # fixed threshold 0.5
    # m['gap'] = (m.img_score - m.txt_score).abs()
    # dev_text = {r['id']: r['text'] for r in C.load_split(DATA_DIR, 'dev')}
    # m['text'] = m.id.map(dev_text)
    # m[m.disagree].sort_values('gap', ascending=False).head(5)


# CHANGED: config is now JSON instead of key=value lines (your requirement).
# Values in the JSON override DEFAULTS, so a file only needs the keys that differ.
# This also removes the use of the undefined `log`, and JSON keeps lists, floats
# and booleans typed (the old parser would have returned everything as strings).


def load_config(path):
    with open(path) as f:
        return {**DEFAULTS, **json.load(f)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="MemeBaseline",
        description="generates a baseline result based on config file",
    )
    parser.add_argument("config_file")
    args = parser.parse_args()

    cfg = load_config(args.config_file)

    cfg.setdefault("OUT_DIR", f"outputs/{cfg['EXP_NAME']}")

    globals().update(cfg)
