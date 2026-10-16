#!/usr/bin/env python3

import shap, json, numpy as np
from PIL import Image
import common as C

exp = "outputs/img_embeddinggemma2"
cfg = json.load(open(f"{exp}/summary.json"))
scorer = C.ImageScorer(exp, cfg["model"], cfg["img_kind"])

img_ids = [54206, 79042, 73914, 74058, 2568]


def load_224(path):
    im = Image.open(path).convert("RGB")
    s = 224 / min(im.size)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
    l, t = (im.width - 224) // 2, (im.height - 224) // 2
    return np.array(im.crop((l, t, l + 224, t + 224)))


imgs = [load_224("data/img/{i}.png") for i in img_ids]
masker = shap.maskers.Image("blur(32,32)", img.shape)
explainer = shap.Explainer(scorer, masker, output_names=["hateful logit"])
sv = explainer(imgs, max_evals=1000, batch_size=50)
plt = shap.image_plot(sv)
plt.save()
