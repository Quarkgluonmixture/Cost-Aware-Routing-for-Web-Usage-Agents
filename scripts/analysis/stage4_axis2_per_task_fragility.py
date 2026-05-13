#!/usr/bin/env python3
"""Axis-2 per-task fragility check (response to /stress W2 attack).

The /stress reviewer (2026-05-12) attacked the §5.7 three-axis hierarchy
on grounds that the axis-2 cosine gap mean of 0.0114 is reported without
per-task variance check. By analogy to h1_per_task_fragility (which found
H1 dichotomy holds for only 11% of (task, step) pairs strict), axis-2 at
roughly one-third the magnitude is plausibly dominated by 2-3 outlier
tasks.

This script computes per-task cosine gap distribution at L23 for:
  - Axis-2 pair: P-text vs P-SoM (flat-text, prompt swap) — main test
  - Axis-2 pair: DOM vs P-prompt (hierarchical, prompt swap) — secondary
  - Axis-1 pair: DOM vs P-text (DOM-prompt, text swap) — magnitude reference
  - Axis-3 pair: P-SoM vs SoM (image-axis reference) — calibration scale

For each task: average mode hidden states across the 2 steps the task
contributes, then compute cosine gap between the two pair modes at L23.
Report median, IQR, fraction-above-threshold (0.005 and 0.010), top/bottom
5 tasks.

Outputs:
  - docs/checkpoints/mechanism/results/axis2_per_task_fragility.md
  - results/phantom_paper/figures/fig_axis2_per_task_fragility.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLS_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states_v2_fixed.npz"
DEFAULT_RED_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_reddit/hidden_states_v2_fixed.npz"
DEFAULT_MD = ROOT / "docs/checkpoints/mechanism/results/axis2_per_task_fragility.md"
DEFAULT_FIG = ROOT / "results/phantom_paper/figures/fig_axis2_per_task_fragility.png"

# (mode_a, mode_b, label, axis, color)
PAIRS = [
    ("phantom_text", "phantom_som",  "P-text ↔ P-SoM   (axis-2 flat-text)",   "axis-2", "#d62728"),
    ("dom",          "phantom_prompt","DOM ↔ P-prompt  (axis-2 hierarchical)", "axis-2", "#ff7f0e"),
    ("dom",          "phantom_text",  "DOM ↔ P-text     (axis-1 reference)",   "axis-1", "#1f77b4"),
    ("phantom_som",  "som",           "P-SoM ↔ SoM     (axis-3 image ref)",    "axis-3", "#9467bd"),
]

L_TARGET = 23  # paper §5.7 axis-2 peak layer


def cosine_gap(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def compute_per_task_cosine(npz_path: Path, layer: int):
    """For each task ID, compute mode mean hidden state across steps, then
    cosine gap between mode pairs.

    Returns dict[pair_label] = {task_id: cosine_gap}
    """
    d = np.load(npz_path, allow_pickle=True)
    H = d["hidden_states"]  # (N, L, D)
    ml = d["mode_labels_str"]
    tids = d["task_ids"]

    unique_tasks = sorted(set(int(t) for t in tids))
    unique_modes = sorted(set(ml.tolist()))

    # For each (task, mode), average hidden state across steps at target layer
    task_mode_mean = {}  # (task_id, mode) -> hidden (D,)
    for t in unique_tasks:
        for m in unique_modes:
            mask = (tids == t) & (ml == m)
            if mask.sum() == 0:
                continue
            task_mode_mean[(t, m)] = H[mask, layer].mean(axis=0)

    per_pair = {}
    for a, b, label, axis, _color in PAIRS:
        per_task = {}
        for t in unique_tasks:
            if (t, a) in task_mode_mean and (t, b) in task_mode_mean:
                per_task[t] = cosine_gap(task_mode_mean[(t, a)], task_mode_mean[(t, b)])
        per_pair[label] = {"per_task": per_task, "axis": axis}

    return per_pair, unique_tasks


def summarize(per_pair: dict, layer: int):
    """For each pair, compute median, IQR, fraction-above-threshold."""
    summary = {}
    for label, info in per_pair.items():
        vals = np.array(list(info["per_task"].values()))
        n = len(vals)
        summary[label] = {
            "axis": info["axis"],
            "n": n,
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std()),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
            "p10": float(np.percentile(vals, 10)),
            "p90": float(np.percentile(vals, 90)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "frac_gt_005": float((vals > 0.005).mean()),
            "frac_gt_010": float((vals > 0.010).mean()),
            "frac_gt_020": float((vals > 0.020).mean()),
            "per_task_sorted": sorted(info["per_task"].items(), key=lambda kv: kv[1], reverse=True),
        }
    return summary


def write_md(cls_sum: dict, red_sum: dict, layer: int, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Axis-2 per-task fragility check",
        "",
        f"Per-task cosine gap distribution at L{layer} (axis-2 peak per §5.7 / Exp 1).",
        f"Each task averaged across its 2 steps; cosine gap computed between mode pairs.",
        "",
        "**Defuse target**: /stress W2 attack — axis-2 mean 0.0114 might be dominated by 2-3 outlier tasks.",
        "",
        "## Classifieds (24 tasks)",
        "",
        "| Pair | Axis | Mean | Median | IQR | min | max | % > 0.005 | % > 0.010 | % > 0.020 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, s in cls_sum.items():
        iqr = f"[{s['p25']:.4f}, {s['p75']:.4f}]"
        lines.append(
            f"| {label} | {s['axis']} | {s['mean']:.4f} | {s['median']:.4f} | {iqr} | "
            f"{s['min']:.4f} | {s['max']:.4f} | "
            f"{s['frac_gt_005']:.0%} | {s['frac_gt_010']:.0%} | {s['frac_gt_020']:.0%} |"
        )
    lines += [
        "",
        "## Reddit (24 tasks)",
        "",
        "| Pair | Axis | Mean | Median | IQR | min | max | % > 0.005 | % > 0.010 | % > 0.020 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, s in red_sum.items():
        iqr = f"[{s['p25']:.4f}, {s['p75']:.4f}]"
        lines.append(
            f"| {label} | {s['axis']} | {s['mean']:.4f} | {s['median']:.4f} | {iqr} | "
            f"{s['min']:.4f} | {s['max']:.4f} | "
            f"{s['frac_gt_005']:.0%} | {s['frac_gt_010']:.0%} | {s['frac_gt_020']:.0%} |"
        )

    # Top/bottom 5 tasks for the main axis-2 pair
    main_label = next(k for k in cls_sum.keys() if "P-text ↔ P-SoM" in k)
    for site_name, summ in [("classifieds", cls_sum), ("reddit", red_sum)]:
        s = summ[main_label]
        lines += [
            "",
            f"## Top 5 axis-2 tasks ({site_name}, P-text ↔ P-SoM @ L23)",
            "",
            "| Task ID | Cosine gap |",
            "|---|---:|",
        ]
        for tid, val in s["per_task_sorted"][:5]:
            lines.append(f"| {tid} | {val:.4f} |")
        lines += [
            "",
            f"## Bottom 5 axis-2 tasks ({site_name}, P-text ↔ P-SoM @ L23)",
            "",
            "| Task ID | Cosine gap |",
            "|---|---:|",
        ]
        for tid, val in s["per_task_sorted"][-5:]:
            lines.append(f"| {tid} | {val:.4f} |")

    lines += [
        "",
        "## Verdict",
        "",
        f"Read the `% > 0.010` column for the axis-2 P-text↔P-SoM pair:",
        f"- cls: **{cls_sum[main_label]['frac_gt_010']:.0%}** of 24 tasks above the L23 axis-2 mean magnitude",
        f"- reddit: **{red_sum[main_label]['frac_gt_010']:.0%}** of 24 tasks above",
        "",
        f"Interpretation tree:",
        f"- If both ≥ 50% → axis-2 signal **broad**, /stress W2 attack defused, §5.7 framing OK",
        f"- If both 25-50% → axis-2 signal **modest but present**, §5.7 needs to add 'task-conditional sparse' qualifier",
        f"- If both < 25% → axis-2 signal **aggregate artifact**, §5.7 three-axis claim must downgrade to 'axis-1 + image-axis with axis-2 weak per-task'",
        "",
        f"Median values: cls={cls_sum[main_label]['median']:.4f}, reddit={red_sum[main_label]['median']:.4f}.",
        f"Compare to mean: cls={cls_sum[main_label]['mean']:.4f}, reddit={red_sum[main_label]['mean']:.4f}.",
        f"If median << mean, the distribution is right-skewed → outlier-driven (consistent with /stress W2 attack).",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"summary → {out}")


def plot(cls_sum: dict, red_sum: dict, layer: int, cls_per_pair: dict, red_per_pair: dict, out: Path):
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for col_idx, (site, summ, per_pair) in enumerate([
        ("classifieds", cls_sum, cls_per_pair),
        ("reddit", red_sum, red_per_pair),
    ]):
        # Top row: histograms per pair
        ax_h = axes[0, col_idx]
        for label, info in per_pair.items():
            vals = np.array(list(info["per_task"].values()))
            color = {
                "axis-1": "#1f77b4",
                "axis-2": "#d62728" if "flat" in label else "#ff7f0e",
                "axis-3": "#9467bd",
            }[info["axis"]]
            ax_h.hist(vals, bins=20, alpha=0.6, label=label, color=color)
        ax_h.axvline(0.005, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="0.005 threshold")
        ax_h.axvline(0.010, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="0.010 threshold")
        ax_h.set_xlabel(f"Cosine gap at L{layer}")
        ax_h.set_ylabel("Tasks")
        ax_h.set_title(f"{site}: per-task cosine gap distribution")
        ax_h.legend(fontsize=7, loc="upper right")
        ax_h.grid(True, alpha=0.3)

        # Bottom row: per-pair box+swarm
        ax_b = axes[1, col_idx]
        labels = list(per_pair.keys())
        data = [list(per_pair[l]["per_task"].values()) for l in labels]
        bp = ax_b.boxplot(data, vert=True, patch_artist=True, labels=[l.split("(")[0].strip() for l in labels])
        for patch, label in zip(bp["boxes"], labels):
            axis = per_pair[label]["axis"]
            patch.set_facecolor({
                "axis-1": "#1f77b4",
                "axis-2": "#d62728" if "flat" in label else "#ff7f0e",
                "axis-3": "#9467bd",
            }[axis])
            patch.set_alpha(0.6)
        # overlay individual task points
        for i, vals in enumerate(data):
            jitter = np.random.normal(0, 0.04, size=len(vals))
            ax_b.scatter(np.full(len(vals), i + 1) + jitter, vals, color="k", s=8, alpha=0.5)
        ax_b.axhline(0.010, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax_b.set_ylabel(f"Cosine gap at L{layer}")
        ax_b.set_title(f"{site}: per-task box + swarm")
        ax_b.grid(True, alpha=0.3)
        plt.setp(ax_b.get_xticklabels(), rotation=15, ha="right")

    fig.suptitle("Axis-2 per-task fragility check (/stress W2 defuse target)", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"figure → {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cls-npz", type=Path, default=DEFAULT_CLS_NPZ)
    p.add_argument("--red-npz", type=Path, default=DEFAULT_RED_NPZ)
    p.add_argument("--layer", type=int, default=L_TARGET)
    p.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    p.add_argument("--output-fig", type=Path, default=DEFAULT_FIG)
    args = p.parse_args()

    np.random.seed(0)
    print(f"Loading cls: {args.cls_npz}")
    cls_per_pair, cls_tasks = compute_per_task_cosine(args.cls_npz, args.layer)
    print(f"  {len(cls_per_pair)} pairs, {len(cls_tasks)} tasks")

    print(f"Loading reddit: {args.red_npz}")
    red_per_pair, red_tasks = compute_per_task_cosine(args.red_npz, args.layer)
    print(f"  {len(red_per_pair)} pairs, {len(red_tasks)} tasks")

    cls_sum = summarize(cls_per_pair, args.layer)
    red_sum = summarize(red_per_pair, args.layer)

    write_md(cls_sum, red_sum, args.layer, args.output_md)
    plot(cls_sum, red_sum, args.layer, cls_per_pair, red_per_pair, args.output_fig)


if __name__ == "__main__":
    main()
