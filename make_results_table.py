#!/usr/bin/env python
"""Build a LaTeX results table from outputs/*/summary.json (one file per run).

Image and text rows are the Assignment 1 baselines; multimodal rows are the new
Assignment 2 runs. Best run per modality is bolded.

Usage (from the repo root):
    uv run make_results_table.py
    uv run make_results_table.py --digits 3 --ddof 1 --out results_table.tex
"""
import argparse
import json
from pathlib import Path

import numpy as np


def latex_escape(s):
    for ch in "&%$#_{}":
        s = s.replace(ch, "\\" + ch)
    return s


def load_runs(outputs_dir, ddof):
    runs = []
    for path in sorted(Path(outputs_dir).glob("*/summary.json")):
        with open(path) as f:
            s = json.load(f)
        vals = [r["val_auroc"] for r in s["per_seed"]]
        runs.append(
            {
                "model": s["model"],
                "modality": s["modality"],
                "fusion": s.get("fusion"),
                "lambda_op": s.get("lambda_op"),
                # Assignment 1 probes are nn.Linear(emb_dim, 1): emb_dim weights + 1 bias
                "params": s.get("trainable_params") or (s["emb_dim"] + 1 if "emb_dim" in s else None),
                # ddof=0 is the population std main.py reports; --ddof 1 for sample std
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=ddof)),
                "n_seeds": len(vals),
            }
        )
    return runs


def display_name(r):
    """Encoder basename, or encoder pair + fusion/lambda for multimodal rows."""
    if r["modality"] != "multimodal":
        return r["model"].split("/")[-1]
    lam = r["lambda_op"]
    if r["fusion"] == "gated" and lam > 0:
        return f"gated + OP ($\\lambda$={lam:g})"
    return f"{r['fusion']} ($\\lambda$={lam:g})"


def build_table(runs, digits):
    best = {}
    for r in runs:
        current = best.get(r["modality"])
        if current is None or r["mean"] > current["mean"]:
            best[r["modality"]] = r

    best_uni = max(r["mean"] for r in runs if r["modality"] != "multimodal")
    order = {"image": 0, "text": 1, "multimodal": 2}
    runs = sorted(runs, key=lambda r: (order[r["modality"]], -r["mean"]))
    seeds_txt = "/".join(str(n) for n in sorted({r["n_seeds"] for r in runs}))

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{llrcc}",
        "\\toprule",
        "Modality & Model & Trainable params & Val AUROC & $\\Delta$ vs best unimodal \\\\",
        "\\midrule",
    ]
    previous = None
    for r in runs:
        if previous is not None and r["modality"] != previous:
            lines.append("\\midrule")
        previous = r["modality"]
        raw = display_name(r)
        name = raw if r["modality"] == "multimodal" else "\\texttt{" + latex_escape(raw) + "}"
        params = f"{r['params']:,}" if r["params"] else "--"
        score = f"{r['mean']:.{digits}f} $\\pm$ {r['std']:.{digits}f}"
        if best[r["modality"]] is r:
            score = f"\\textbf{{{score}}}"
        gain = f"{r['mean'] - best_uni:+.{digits}f}" if r["modality"] == "multimodal" else "--"
        lines.append(f"{r['modality'].capitalize()} & {name} & {params} & {score} & {gain} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Validation AUROC (mean $\\pm$ std over {seeds_txt} seeds). Image and "
        "text rows are the Assignment 1 baselines (linear probes); multimodal rows use "
        "EmbeddingGemma~2 (image) and all-mpnet-base-v2 (text). $\\Delta$ is the absolute "
        "AUROC gain over the best unimodal mean. Best run per modality in bold.}",
        "\\label{tab:results}",
        "\\end{table}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LaTeX results table from summary.json files")
    ap.add_argument("--outputs", default="outputs", help="folder with one subfolder per experiment")
    ap.add_argument("--digits", type=int, default=3)
    ap.add_argument("--ddof", type=int, default=0, choices=[0, 1], help="0 = population std, 1 = sample std")
    ap.add_argument("--out", default="results_table.tex")
    args = ap.parse_args()

    table = build_table(load_runs(args.outputs, args.ddof), args.digits)
    print(table)
    Path(args.out).write_text(table + "\n")
    print(f"\n% written to {args.out}")
