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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ_CLS = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz"
DEFAULT_NPZ_RED = ROOT / "results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz"
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


def cosine_gap(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def compute_pair_curves(npz_path: Path) -> tuple[dict, int, dict]:
    d = np.load(npz_path, allow_pickle=True)
    H = d["hidden_states"]
    ml = d["mode_labels_str"]
    n_layers = H.shape[1]
    means = {}
    for m in {p[0] for p in PAIRS} | {p[1] for p in PAIRS}:
        mask = ml == m
        if mask.sum() == 0:
            continue
        means[m] = H[mask].mean(axis=0)

    curves = {}
    for a, b, label, group, color, ls in PAIRS:
        if a not in means or b not in means:
            continue
        curve = np.array([cosine_gap(means[a][L], means[b][L]) for L in range(n_layers)])
        curves[label] = {
            "curve": curve,
            "group": group,
            "color": color,
            "linestyle": ls,
            "mode_a": a,
            "mode_b": b,
            "peak_L": int(np.argmax(curve)),
            "peak_gap": float(curve.max()),
            "L17": float(curve[17]) if n_layers > 17 else None,
            "L4": float(curve[4]) if n_layers > 4 else None,
            "L0": float(curve[0]),
            "L_last": float(curve[-1]),
        }
    return curves, n_layers, means


def write_md(curves_cls: dict, curves_red: dict, n_layers: int, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Exp 1 — Axis-2 (prompt-family) layer profile",
        "",
        "**Question**: Method 4.2 at L17 shows prompt-family makes ~0 geometric contribution to residual stream",
        "(P-SoM↔P-text 0.0028, DOM↔P-prompt 0.0013). But forest plot drop-one places P-SoM as unique hero,",
        "implying axis-2 (prompt) contributes behaviorally. **Where in the model does axis-2 act?**",
        "",
        "**Method**: For each prompt-only pair (text format fixed, prompt swap), compute full 37-layer cosine gap.",
        "Overlay axis-1-only (text swap, prompt fixed) + image-axis P-SoM↔SoM reference curves to calibrate scale.",
        "",
        "## Results — classifieds site (stage4_multimode_b1_cls, 288 ex)",
        "",
        f"| Pair | Group | L0 | L4 | L17 | L{n_layers-1} | Peak L | Peak gap |",
        f"|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, info in curves_cls.items():
        lines.append(
            f"| {label} | {info['group']} | {info['L0']:.4f} | {info['L4']:.4f} | {info['L17']:.4f} | "
            f"{info['L_last']:.4f} | **L{info['peak_L']}** | {info['peak_gap']:.4f} |"
        )

    lines += [
        "",
        "## Results — reddit site (stage4_multimode_b1_reddit, 288 ex)",
        "",
        f"| Pair | Group | L0 | L4 | L17 | L{n_layers-1} | Peak L | Peak gap |",
        f"|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, info in curves_red.items():
        lines.append(
            f"| {label} | {info['group']} | {info['L0']:.4f} | {info['L4']:.4f} | {info['L17']:.4f} | "
            f"{info['L_last']:.4f} | **L{info['peak_L']}** | {info['peak_gap']:.4f} |"
        )

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
        "Compare peak layers above against axis-1 (text-format) pairs (the established mechanism with L17 peak) and image-axis reference (~0.04 magnitude). If axis-2 pair peak < 0.01 at all layers, hypothesis 1 holds.",
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
        ax.axhline(0.01, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axvline(17, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Layer index (L0 = embedding, L36 = final block)")
        ax.set_title(f"{title}  (axis-2 = solid, axis-1 = dashed, image = dotted)", fontsize=10)
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
    curves_cls, n_layers_cls, _ = compute_pair_curves(args.cls_npz)
    print(f"  {len(curves_cls)} pairs, {n_layers_cls} layers")

    print(f"Loading reddit: {args.red_npz}")
    curves_red, n_layers_red, _ = compute_pair_curves(args.red_npz)
    print(f"  {len(curves_red)} pairs, {n_layers_red} layers")

    assert n_layers_cls == n_layers_red, f"layer count mismatch cls={n_layers_cls} red={n_layers_red}"

    write_md(curves_cls, curves_red, n_layers_cls, args.output_md)
    plot(curves_cls, curves_red, n_layers_cls, args.output_fig)


if __name__ == "__main__":
    main()
