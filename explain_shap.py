#!/usr/bin/env python3
import json, os
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from PIL import Image
import common as C

DATA_DIR = "data"
exp = "outputs/img_embeddinggemma2"
out_dir = f"{exp}/shap"
os.makedirs(out_dir, exist_ok=True)

cfg = json.load(open(f"{exp}/summary.json"))
scorer = C.ImageScorer(exp, cfg["model"], cfg["img_kind"])

img_files = ["54206.png", "79042.png", "73914.png", "74058.png", "02568.png"]

# val_scores.csv for the sanity check (seed-0 probe, same as the scorer)
val = pd.read_csv(f"{exp}/val_scores.csv").set_index("id")
logit = lambda p: float(np.log(p / (1 - p)))


def load_native(fname):
    """Full image, no crop/resize: the model's own processor handles resizing."""
    im = Image.open(os.path.join(DATA_DIR, "img", fname)).convert("RGB")
    return np.array(im)


for f in img_files:
    stem = Path(f).stem
    arr = load_native(f)

    # sanity check: scorer on the native image should match val_scores.csv
    out = float(scorer(arr[None])[0])
    ref = logit(val.loc[int(stem), "score"]) if int(stem) in val.index else float("nan")
    print(f"{stem}: shape={arr.shape}  scorer logit={out:.3f}  csv logit={ref:.3f}")

    masker = shap.maskers.Image("blur(64,64)", arr.shape)
    explainer = shap.Explainer(scorer, masker)
    sv = explainer(arr[None], max_evals=1000, batch_size=50)

    shap.image_plot(sv, show=False)
    plt.savefig(f"{out_dir}/{stem}.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"saved {out_dir}/{stem}.png")
