#!/usr/bin/env python
"""Build a LaTeX results table from outputs/*/summary.json (one file per encoder).

Usage (from the repo root):
    uv run make_results_table.py
    uv run make_results_table.py --outputs outputs --digits 3 --ddof 1 --out results_table.tex

Reads `model`, `modality` and `per_seed` from every summary.json, computes the
mean and std of val_auroc (and test_auroc, if the runs have it) over the seeds,
and prints a booktabs table. The best encoder of each modality is bolded.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def latex_escape(s):
    for ch in ["\\", "&", "%", "$", "#", "_", "{", "}"]:
        if ch != "\\":
            s = s.replace(ch, "\\" + ch)
    return s


def mean_std(per_seed, key, ddof):
    vals = [r[key] for r in per_seed if key in r]
    if not vals:
        return None
    # ddof=0 matches common.summarize (np.std default); use --ddof 1 for sample std
    return float(np.mean(vals)), float(np.std(vals, ddof=ddof)), len(vals)


def load_runs(outputs_dir, ddof):
    runs = []
    for path in sorted(Path(outputs_dir).glob("*/summary.json")):
        with open(path) as f:
            s = json.load(f)
        runs.append(
            {
                "exp": s.get("exp", path.parent.name),
                "model": s["model"],
                "modality": s["modality"],
                "val": mean_std(s["per_seed"], "val_auroc", ddof),
                "test": mean_std(s["per_seed"], "test_auroc", ddof),
            }
        )
    return runs


def fmt(ms, digits, bold):
    if ms is None:
        return "--"
    m, sd, _ = ms
    txt = f"{m:.{digits}f} $\\pm$ {sd:.{digits}f}"
    return f"\\textbf{{{txt}}}" if bold else txt


def build_table(runs, digits):
    has_test = any(r["test"] is not None for r in runs)
    key = "test" if has_test else "val"  # bold the best by test if available, else val

    # best per modality
    best = {}
    for r in runs:
        if r[key] is None:
            continue
        cur = best.get(r["modality"])
        if cur is None or r[key][0] > cur[key][0]:
            best[r["modality"]] = r

    order = {"image": 0, "text": 1}
    runs = sorted(runs, key=lambda r: (order.get(r["modality"], 9), -(r["val"][0] if r["val"] else 0)))

    n_seeds = sorted({r["val"][2] for r in runs if r["val"]})
    seeds_txt = "/".join(map(str, n_seeds)) if n_seeds else "?"

    cols = "llc" + ("c" if has_test else "")
    head = "Modality & Encoder & Val AUROC" + (" & Test AUROC" if has_test else "") + " \\\\"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        head,
        "\\midrule",
    ]
    prev = None
    for r in runs:
        if prev is not None and r["modality"] != prev:
            lines.append("\\midrule")
        prev = r["modality"]
        is_best = best.get(r["modality"]) is r
        name = "\\texttt{" + latex_escape(r["model"].split("/")[-1]) + "}"
        row = f"{r['modality'].capitalize()} & {name} & {fmt(r['val'], digits, is_best and key == 'val')}"
        if has_test:
            row += f" & {fmt(r['test'], digits, is_best and key == 'test')}"
        lines.append(row + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{AUROC (mean $\\pm$ std over {seeds_txt} seeds) of the linear probe on each frozen encoder. "
        "Best encoder per modality in bold.}",
        "\\label{tab:results}",
        "\\end{table}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LaTeX results table from summary.json files")
    ap.add_argument("--outputs", default="outputs", help="folder with one subfolder per experiment")
    ap.add_argument("--digits", type=int, default=3)
    ap.add_argument("--ddof", type=int, default=0, choices=[0, 1], help="0 = population std (as in common.py), 1 = sample std")
    ap.add_argument("--out", default="results_table.tex")
    args = ap.parse_args()

    runs = load_runs(args.outputs, args.ddof)
    if not runs:
        raise SystemExit(f"No summary.json found under {args.outputs}/*/")
    table = build_table(runs, args.digits)
    print(table)
    Path(args.out).write_text(table + "\n")
    print(f"\n% written to {args.out}")
