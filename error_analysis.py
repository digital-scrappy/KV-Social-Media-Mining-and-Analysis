#!/usr/bin/env python
"""Candidate validation examples for the error analysis (Assignment 2).

Merges the dev scores of the image-only, text-only and fusion runs and lists, at a
threshold (default 0.5):
  FP / FN           confident mistakes of the fusion model
  FIXES_BOTH        fusion right, image-only AND text-only wrong
  FIXES_IMG         fusion right, image-only wrong, text-only right
  FIXES_TXT         fusion right, text-only wrong, image-only right
  BREAKS            fusion wrong, image-only or text-only right
Pick >=6 across categories, open data/<img_path> and write the explanation yourself.
Writes outputs/error_candidates.csv.

Usage: uv run error_analysis.py [--k 4] [--thr 0.5]
"""
import argparse
import json

import pandas as pd

RUNS = {
    "s_img": "outputs/img_embeddinggemma2/val_scores.csv",
    "s_txt": "outputs/text_mpnet/val_scores.csv",
    "s_fus": "outputs/multimodal_embeddinggemma_mpnet/val_scores.csv",
}
DEV = "data/dev.jsonl"


def load(col):
    d = pd.read_csv(RUNS[col])
    d["label"] = d["label"].astype(int)
    return d.rename(columns={"score": col})[["id", "label", col]]


def main(k, thr):
    df = load("s_img")
    for col in ("s_txt", "s_fus"):
        df = df.merge(load(col).drop(columns="label"), on="id", validate="1:1")
    meta = pd.DataFrame([json.loads(line) for line in open(DEV, encoding="utf-8")])
    meta = meta.rename(columns={"img": "img_path"})[["id", "img_path", "text"]]
    df = df.merge(meta, on="id", validate="1:1")

    for col in RUNS:
        df[col + "_ok"] = (df[col] >= thr).astype(int) == df["label"]
    df["conf"] = (df["s_fus"] - thr).abs()
    f, i, t = df.s_fus_ok, df.s_img_ok, df.s_txt_ok

    cats = {
        "FP": df[(df.label == 0) & ~f],
        "FN": df[(df.label == 1) & ~f],
        "FIXES_BOTH": df[f & ~i & ~t],
        "FIXES_IMG": df[f & ~i & t],
        "FIXES_TXT": df[f & ~t & i],
        "BREAKS": df[~f & (i | t)],
    }
    rows = []
    for name, sub in cats.items():
        top = sub.sort_values("conf", ascending=False).head(k)
        print(f"\n== {name}: {len(sub)} total, showing top {len(top)} by fusion confidence")
        for _, r in top.iterrows():
            print(
                f"  id {r['id']}  label={r['label']}  img={r.s_img:.2f} "
                f"txt={r.s_txt:.2f} fus={r.s_fus:.2f}  data/{r.img_path}\n"
                f"      \"{r.text}\""
            )
            rows.append({"category": name, **r[["id", "label", "s_img", "s_txt", "s_fus", "img_path", "text"]]})
    pd.DataFrame(rows).to_csv("outputs/error_candidates.csv", index=False)
    print("\nWrote outputs/error_candidates.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()
    main(args.k, args.thr)
