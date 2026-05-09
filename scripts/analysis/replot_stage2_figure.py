#!/usr/bin/env python3
"""Re-plot Stage 2 patching figure from existing patching_continuation_results.json.

Why: cells B (job 334716) and D (job 335340) ran with the pre-`b975763`
title generator that hardcoded "som→phantom_som" regardless of --reverse.
This standalone script re-renders the 4-panel figure with a corrected
direction-aware title, in-place over the old PNG.

Usage:
    .venv/bin/python3 scripts/analysis/replot_stage2_figure.py \\
        --results <dir>/patching_continuation_results.json \\
        [--output <dir>/patching_continuation_curves.png]   # default = same dir
        [--direction-override forward|reverse]              # default = read config
        [--tier-override strong|reverse]                    # default = read config
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("token_overlap_to_source", "Token overlap → source\n(1=patched matches source position-by-position)"),
    ("token_overlap_to_target", "Token overlap → target\n(higher = patch had no effect)"),
    ("ld_to_source",             "Levenshtein dist → source\n(0=identical, max=~50)"),
    ("ld_to_target",             "Levenshtein dist → target\n(higher = patch pulled away from target)"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--direction-override", choices=["forward", "reverse"], default=None)
    p.add_argument("--tier-override", choices=["strong", "reverse"], default=None)
    args = p.parse_args()

    results_path = Path(args.results)
    output_path = Path(args.output) if args.output else results_path.parent / "patching_continuation_curves.png"

    data = json.loads(results_path.read_text(encoding="utf-8"))
    cfg = data["config"]
    per_task = data["per_task"]
    n_tasks = len(per_task)
    n_layers = len(per_task[0]["per_layer"])

    # Derive direction from config (config dumps args namespace ish; or read
    # output_dir name as fallback heuristic). Currently config has source_mode
    # / target_mode but not the --reverse flag explicitly.
    direction = args.direction_override
    if direction is None:
        # Heuristic: results dir name contains "reverse" or "rev"
        dir_name = str(results_path.parent).lower()
        if "reverse" in dir_name or "_rev_" in dir_name or dir_name.endswith("rev"):
            direction = "reverse"
        else:
            direction = "forward"

    tier = args.tier_override
    if tier is None:
        dir_name = str(results_path.parent).lower()
        if "revtasks" in dir_name or "reverse_curated" in dir_name:
            tier = "reverse"
        else:
            tier = "strong"

    src = cfg["source_mode"]
    tgt = cfg["target_mode"]
    site = cfg["site"]
    step = cfg["step"]
    max_tok = cfg["max_new_tokens"]

    if direction == "reverse":
        direction_label = f"{tgt}→{src} (reverse)"
    else:
        direction_label = f"{src}→{tgt} (forward)"

    # Aggregate per-layer mean ± std
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    layer_idx = list(range(n_layers))
    for (metric, title), ax in zip(METRICS, axes.flat):
        vals = np.asarray([[pl[metric] for pl in t["per_layer"]] for t in per_task])
        mean = vals.mean(axis=0)
        std = vals.std(axis=0, ddof=1) if n_tasks > 1 else np.zeros_like(mean)
        ax.plot(layer_idx, mean, "-o", markersize=4, label=f"mean (N={n_tasks})")
        ax.fill_between(layer_idx, mean - std, mean + std, alpha=0.2, label="±1 std")
        ax.set_xlabel("Layer index (0=embedding, ≥1=post-block)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Stage 2B Continuation Activation Patching — {direction_label} "
        f"({site} N={n_tasks} {tier}-tier task × step_{step:03d}, "
        f"max_new_tokens={max_tok})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {output_path}")
    print(f"  direction = {direction}, tier = {tier}, N = {n_tasks}, layers = {n_layers}")


if __name__ == "__main__":
    main()
