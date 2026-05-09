#!/usr/bin/env python3
"""Stage 2 heterogeneity figure — per-task scatter + violin + median/IQR overlay.

Addresses audit constraint **G8** (report task heterogeneity, not only mean
layer effects). Companion to `patching_continuation_curves.png` which shows
mean ± std band only — this version shows the full per-task distribution
so reviewers can verify the mean isn't carried by a few outlier tasks.

Outputs:
- `<cell_dir>/patching_heterogeneity_curves.png` — 4-panel scatter+violin
- prints per-layer median + IQR table to stdout

Usage:
    .venv/bin/python3 scripts/analysis/stage2_heterogeneity_figure.py \\
        --results <cell_dir>/patching_continuation_results.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("token_overlap_to_source", "Token overlap → source"),
    ("token_overlap_to_target", "Token overlap → target"),
    ("ld_to_source",             "Levenshtein dist → source"),
    ("ld_to_target",             "Levenshtein dist → target"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--direction-override", choices=["forward", "reverse"], default=None)
    p.add_argument("--tier-override", choices=["strong", "reverse"], default=None)
    args = p.parse_args()

    results_path = Path(args.results)
    output_path = Path(args.output) if args.output else (
        results_path.parent / "patching_heterogeneity_curves.png"
    )

    data = json.loads(results_path.read_text(encoding="utf-8"))
    cfg = data["config"]
    per_task = data["per_task"]
    n_tasks = len(per_task)
    n_layers = len(per_task[0]["per_layer"])

    direction = args.direction_override
    if direction is None:
        dir_name = str(results_path.parent).lower()
        direction = "reverse" if ("reverse" in dir_name or "_rev_" in dir_name) else "forward"
    tier = args.tier_override
    if tier is None:
        dir_name = str(results_path.parent).lower()
        tier = "reverse" if "revtasks" in dir_name or "reverse_curated" in dir_name else "strong"

    src = cfg["source_mode"]
    tgt = cfg["target_mode"]
    direction_label = f"{tgt}→{src} (reverse)" if direction == "reverse" else f"{src}→{tgt} (forward)"

    # Per-task per-layer grids
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    layer_idx = list(range(n_layers))
    plot_layers = [0, 5, 11, 17, 23, 29, 35]  # canonical sampled layers
    rng = np.random.default_rng(seed=42)  # reproducible jitter

    print(f"\n=== Heterogeneity stats ({direction_label}, N={n_tasks}) ===\n")

    for (metric, title), ax in zip(METRICS, axes.flat):
        vals = np.asarray(
            [[pl[metric] for pl in t["per_layer"]] for t in per_task]
        )  # (N, n_layers)

        mean = vals.mean(axis=0)
        median = np.median(vals, axis=0)
        q25 = np.percentile(vals, 25, axis=0)
        q75 = np.percentile(vals, 75, axis=0)

        # Violin at sampled layers (canonical L0/5/11/17/23/29/35)
        violin_data = [vals[:, L] for L in plot_layers]
        violin_pos = plot_layers
        parts = ax.violinplot(violin_data, positions=violin_pos, widths=2.5,
                              showmeans=False, showmedians=False, showextrema=False)
        for body in parts['bodies']:
            body.set_alpha(0.18)
            body.set_facecolor("steelblue")
            body.set_edgecolor("steelblue")

        # Per-task scatter dots (jittered) at sampled layers
        for L in plot_layers:
            jitter = rng.normal(0, 0.4, size=n_tasks)
            ax.scatter(np.full(n_tasks, L) + jitter, vals[:, L],
                       s=8, alpha=0.45, color="navy", edgecolors="none", zorder=3)

        # Median line + IQR band across all layers
        ax.fill_between(layer_idx, q25, q75, alpha=0.15, color="orange",
                        label=f"IQR (Q25-Q75)")
        ax.plot(layer_idx, median, "-", color="orange", lw=2,
                label=f"median (N={n_tasks})", zorder=4)
        ax.plot(layer_idx, mean, "--", color="darkred", lw=1.5, alpha=0.7,
                label=f"mean (N={n_tasks})", zorder=4)

        ax.set_xlabel("Layer index (0=embedding, ≥1=post-block)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        ax.set_xticks(plot_layers)

        print(f"## metric: {metric}")
        print("| Layer | mean | median | IQR (Q25-Q75) |")
        print("|---|---|---|---|")
        for L in plot_layers:
            print(f"| L{L:>2} | {mean[L]:.3f} | {median[L]:.3f} | "
                  f"[{q25[L]:.3f}, {q75[L]:.3f}] |")
        print()

    fig.suptitle(
        f"Stage 2 Patching Heterogeneity — {direction_label} "
        f"(N={n_tasks} {tier}-tier × per-task scatter + violin + median/IQR)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
