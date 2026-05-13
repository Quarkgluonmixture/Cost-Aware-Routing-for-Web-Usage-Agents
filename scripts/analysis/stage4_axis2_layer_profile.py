#!/usr/bin/env python3
"""Exp 1: Axis-2 (prompt-family) layer profile.

Method 4.2 cosine gap at L17 places (DOM, P-prompt) and (P-text, P-SoM) into
two text-format clusters with prompt-family making essentially zero geometric
contribution to that single layer. The four-fold drop-in property + forest
plot show P-SoM uniquely earns drop-one hero status, but axis-2 mechanism is
not visible at L17.

This script asks: across ALL 37 layers, where does prompt-family contribute
to residual-stream geometry?

Pairs computed:
  - axis-2-only (prompt swap, text fixed):
      DOM <-> P-prompt    (both hierarchical AXTree)
      P-text <-> P-SoM    (both flat [SOM_MARKS])
  - axis-1-only (text swap, prompt fixed) — reference:
      DOM <-> P-text      (both DOM-prompt)
      P-prompt <-> P-SoM  (both SoM-prompt)
  - image-axis reference (scale calibration):
      P-SoM <-> SoM

Outputs:
  - docs/checkpoints/mechanism/results/axis2_layer_profile.md
  - results/phantom_paper/figures/fig_axis2_prompt_layer_profile.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Pipeline audit P0-7 fix (2026-05-13): shared paired helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paired_npz_helpers import (  # noqa: E402
    load_v2_npz, paired_rows, paired_cosine_gap_per_layer, task_bootstrap_ci,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ_CLS = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states_v2_fixed.npz"
DEFAULT_NPZ_RED = ROOT / "results/mechanistic/stage4_multimode_b1_reddit/hidden_states_v2_fixed.npz"
DEFAULT_OUT_MD = ROOT / "docs/checkpoints/mechanism/results/axis2_layer_profile.md"
DEFAULT_OUT_FIG = ROOT / "results/phantom_paper/figures/fig_axis2_prompt_layer_profile.png"

PAIRS = [
    # (mode_a, mode_b, label, group, color, linestyle)
    ("dom",          "phantom_prompt", "DOM ↔ P-prompt  (axis-2 only, hierarchical)", "axis-2", "#d62728", "-"),
    ("phantom_text", "phantom_som",    "P-text ↔ P-SoM  (axis-2 only, flat)",         "axis-2", "#ff7f0e", "-"),
    ("dom",          "phantom_text",   "DOM ↔ P-text    (axis-1 only, DOM-prompt)",   "axis-1", "#1f77b4", "--"),
    ("phantom_prompt","phantom_som",   "P-prompt ↔ P-SoM (axis-1 only, SoM-prompt)",  "axis-1", "#2ca02c", "--"),
    ("phantom_som",  "som",            "P-SoM ↔ SoM     (image-axis reference)",      "image",  "#9467bd", ":"),
]


def compute_pair_curves(npz_path: Path, n_boot: int = 1000) -> tuple[dict, int]:
    """Per-task paired cosine gap per layer with bootstrap CI band.

    P0-7 fix (2026-05-13): Previously computed `means[m] = H[mask].mean(axis=0)` then
    `cosine_gap(means[a][L], means[b][L])` — single cosine between two pooled means.
    That mixes task-content variance into the "layer profile" claim. Now uses
    (task_id, step) inner-join via `_paired_npz_helpers.paired_rows` then averages
    per-task cosine gap, plus task-level bootstrap CI (1000 resamples) for the
    paper-grade peak-layer precision claim (P1-6 freebie).
    """
    npz = load_v2_npz(npz_path)
    n_layers = npz["H"].shape[1]
    assert n_layers == 37, f"expected 37 layers (embed + 36 blocks), got {n_layers}"

    rng = np.random.default_rng(seed=20260513)
    curves = {}
    for a, b, label, group, color, ls in PAIRS:
        try:
            Ha, Hb, keys = paired_rows(npz, a, b)
        except (KeyError, ValueError) as e:
            print(f"  skip {label}: {e}")
            continue
        if len(keys) == 0:
            continue
        point, ci_lo, ci_hi = task_bootstrap_ci(
            Ha, Hb, keys, paired_cosine_gap_per_layer,
            n_boot=n_boot, rng=rng,
        )
        curves[label] = {
            "curve": point.astype(np.float64),
            "ci_lo": ci_lo.astype(np.float64),
            "ci_hi": ci_hi.astype(np.float64),
            "n_paired": len(keys),
            "n_unique_tasks": len(set(k[0] for k in keys)),
            "group": group,
            "color": color,
            "linestyle": ls,
            "mode_a": a,
            "mode_b": b,
            "peak_L": int(np.argmax(point)),
            "peak_gap": float(point.max()),
            "peak_ci_lo": float(ci_lo[int(np.argmax(point))]),
            "peak_ci_hi": float(ci_hi[int(np.argmax(point))]),
            "L17": float(point[17]) if n_layers > 17 else None,
            "L17_ci_lo": float(ci_lo[17]) if n_layers > 17 else None,
            "L17_ci_hi": float(ci_hi[17]) if n_layers > 17 else None,
            "L4": float(point[4]) if n_layers > 4 else None,
            "L0": float(point[0]),
            "L_last": float(point[-1]),
        }
    return curves, n_layers


def _site_rows(curves: dict, n_layers: int) -> list[str]:
    out = [
        f"| Pair | Group | L0 | L4 | L17 [CI] | L{n_layers-1} | Peak L | Peak gap [95% CI] | n_paired |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, info in curves.items():
        out.append(
            f"| {label} | {info['group']} | {info['L0']:.4f} | {info['L4']:.4f} | "
            f"{info['L17']:.4f} [{info['L17_ci_lo']:.4f}, {info['L17_ci_hi']:.4f}] | "
            f"{info['L_last']:.4f} | **L{info['peak_L']}** | "
            f"{info['peak_gap']:.4f} [{info['peak_ci_lo']:.4f}, {info['peak_ci_hi']:.4f}] | "
            f"{info['n_paired']} |"
        )
    return out


def write_md(curves_cls: dict, curves_red: dict, n_layers: int, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    # Dynamic N from curves (P2-1 fix: was hardcoded "288 ex")
    n_cls = next(iter(curves_cls.values()))["n_paired"] if curves_cls else 0
    n_red = next(iter(curves_red.values()))["n_paired"] if curves_red else 0
    n_tasks_cls = next(iter(curves_cls.values()))["n_unique_tasks"] if curves_cls else 0
    n_tasks_red = next(iter(curves_red.values()))["n_unique_tasks"] if curves_red else 0
    lines = [
        "# Exp 1 — Axis-2 (prompt-family) layer profile — per-task paired (v2 NPZ)",
        "",
        "**P0-7 + P1-6 fix (2026-05-13)**: per-task paired cosine gap per layer with",
        "task-level bootstrap 95% CI (1000 resamples). Previous version computed cosine",
        "of pooled mode-means — mixed task-content variance into 'layer profile' claim.",
        "Now uses (task_id, step) inner-join via `_paired_npz_helpers.paired_rows` then",
        "averages per-task cosine gap. CI is from resampling tasks (NOT (task,step) rows)",
        "with replacement, preserving within-task step paired structure.",
        "",
        "**Question**: Method 4.2 at L17 shows prompt-family makes ~0 geometric contribution to residual stream.",
        "But forest plot drop-one places P-SoM as unique hero, implying axis-2 (prompt) contributes",
        "behaviorally. **Where in the model does axis-2 act?**",
        "",
        "**Method**: For each prompt-only pair (text format fixed, prompt swap), compute paired",
        "per-task cosine gap across 37 layers. Overlay axis-1-only (text swap, prompt fixed) +",
        "image-axis P-SoM↔SoM reference curves to calibrate scale.",
        "",
        f"## Results — classifieds site (stage4_multimode_b1_cls, {n_cls} paired rows across {n_tasks_cls} unique tasks)",
        "",
    ]
    lines += _site_rows(curves_cls, n_layers)
    lines += [
        "",
        f"## Results — reddit site (stage4_multimode_b1_reddit, {n_red} paired rows across {n_tasks_red} unique tasks)",
        "",
    ]
    lines += _site_rows(curves_red, n_layers)
    lines += [
        "",
        "## Interpretation",
        "",
        "Three hypotheses about axis-2 mechanism layer:",
        "",
        "1. **Truly null geometry** — axis-2 pair curves flat <0.01 at all layers. Prompt-family bypasses residual stream entirely (acts at attention pattern or output head). → Next: Exp 3 logit lens or Exp 4 attention probe.",
        "2. **Late-layer spike** — axis-2 pair curves spike at L25+ but flat at mid-layer. Prompt prior re-emerges at output decoding. → Next: Exp 5 late-layer patching.",
        "3. **Early-layer spike absorbed** — axis-2 pair curves spike at L0-L5 then collapse to ~0. Prompt embedding effect absorbed by mid-layer fusion. → Next: Exp 3 logit lens to verify if it re-emerges in output distribution.",
        "",
        "Compare peak layers above against axis-1 (text-format) pairs and image-axis reference (~0.04 magnitude).",
        "If axis-2 pair peak CI overlaps 0, hypothesis 1 holds; if CI lower-bound > 0.005, hypothesis 2 or 3.",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"summary → {out}")


def plot(curves_cls: dict, curves_red: dict, n_layers: int, out: Path):
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, curves, title in [(axes[0], curves_cls, "classifieds"), (axes[1], curves_red, "reddit")]:
        layers = np.arange(n_layers)
        for label, info in curves.items():
            lw = 2.5 if info["group"] == "axis-2" else 1.5
            alpha = 1.0 if info["group"] == "axis-2" else 0.7
            ax.plot(layers, info["curve"], color=info["color"], linestyle=info["linestyle"],
                    linewidth=lw, alpha=alpha, label=label)
            # P1-6 fix (2026-05-13): bootstrap CI band, lighter shade per pair
            ax.fill_between(layers, info["ci_lo"], info["ci_hi"],
                            color=info["color"], alpha=0.15, linewidth=0)
        ax.axhline(0.01, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axvline(17, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Layer index (L0 = embedding, L36 = final block)")
        ax.set_title(f"{title}  (axis-2 = solid, axis-1 = dashed, image = dotted, 95% CI band)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")

    axes[0].set_ylabel("Cosine gap")
    fig.suptitle("Exp 1: Axis-2 (prompt-family) layer profile — where does the prompt act?", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"figure → {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cls-npz", type=Path, default=DEFAULT_NPZ_CLS)
    p.add_argument("--red-npz", type=Path, default=DEFAULT_NPZ_RED)
    p.add_argument("--output-md", type=Path, default=DEFAULT_OUT_MD)
    p.add_argument("--output-fig", type=Path, default=DEFAULT_OUT_FIG)
    args = p.parse_args()

    print(f"Loading cls: {args.cls_npz}")
    curves_cls, n_layers_cls = compute_pair_curves(args.cls_npz)
    print(f"  {len(curves_cls)} pairs, {n_layers_cls} layers")

    print(f"Loading reddit: {args.red_npz}")
    curves_red, n_layers_red = compute_pair_curves(args.red_npz)
    print(f"  {len(curves_red)} pairs, {n_layers_red} layers")

    assert n_layers_cls == n_layers_red, f"layer count mismatch cls={n_layers_cls} red={n_layers_red}"

    write_md(curves_cls, curves_red, n_layers_cls, args.output_md)
    plot(curves_cls, curves_red, n_layers_cls, args.output_fig)


if __name__ == "__main__":
    main()
