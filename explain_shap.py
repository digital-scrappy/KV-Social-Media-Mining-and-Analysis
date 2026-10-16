#!/usr/bin/env python3
import json, os
import numpy as np
import shap
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from PIL import Image
import common as C

DATA_DIR = "data"
exp = "outputs/img_embeddinggemma2"
cfg = json.load(open(f"{exp}/summary.json"))
scorer = C.ImageScorer(exp, cfg["model"], cfg["img_kind"])

img_ids = ["54206", "79042", "73914", "74058", "02568"]

# id -> image path, taken from the data itself (avoids zero-padding issues)
dev = {int(r["id"]): r["img"] for r in C.load_split(DATA_DIR, "dev")}


def load_224(path):
    im = Image.open(path).convert("RGB")
    s = 224 / min(im.size)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
    l, t = (im.width - 224) // 2, (im.height - 224) // 2
    return np.array(im.crop((l, t, l + 224, t + 224)))


imgs = np.stack([load_224(os.path.join(DATA_DIR, dev[i])) for i in img_ids])

masker = shap.maskers.Image("blur(32,32)", imgs[0].shape)
explainer = shap.Explainer(scorer, masker, output_names=["hateful logit"])
sv = explainer(imgs, max_evals=1000, batch_size=50)

os.makedirs(f"{exp}/shap", exist_ok=True)
for k, i in enumerate(img_ids):
    shap.image_plot(sv[k : k + 1], show=False)
    plt.savefig(f"{exp}/shap/{i}.png", dpi=150, bbox_inches="tight")
    plt.close("all")
