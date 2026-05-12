#!/usr/bin/env python3
"""Exp 3: Logit lens at late layers — does axis-2 prompt-family signal
re-emerge in output distribution even though mid-layer residual stream
shows only weak (~0.011) signal at L23?

Method: Apply Qwen3-VL-4B's lm_head + final_norm to each per-layer hidden
state mean, get a token distribution per (mode, layer). For each axis-2
pair (P-text vs P-SoM at same task) compute:
  - top-1 token disagreement rate per layer
  - KL divergence (P-text || P-SoM) per layer
  - log-prob gap on canonical SoM-prompt vs DOM-prompt action tokens
    (e.g., "click" vs "search", "_pick_", json keys)

This is Wu et al. tool-calling "knows but says differently" mirror: if
axis-2 cosine gap is 0.011 at L23 but output KL is large at L30-L36,
prompt prior is amplified by late-layer decoding into different output.

Inputs:
  results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz
  results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz

Outputs:
  docs/checkpoints/mechanism/results/axis2_logit_lens.md
  results/phantom_paper/figures/fig_axis2_logit_lens.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLS_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz"
DEFAULT_RED_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz"
DEFAULT_MD = ROOT / "docs/checkpoints/mechanism/results/axis2_logit_lens.md"
DEFAULT_FIG = ROOT / "results/phantom_paper/figures/fig_axis2_logit_lens.png"
MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"

AXIS_2_PAIRS = [
    ("phantom_text", "phantom_som", "P-text vs P-SoM  (axis-2 flat-text)"),
    ("dom",          "phantom_prompt", "DOM vs P-prompt  (axis-2 hierarchical)"),
]
AXIS_1_PAIRS = [
    ("dom",           "phantom_text",   "DOM vs P-text    (axis-1 DOM-prompt)"),
    ("phantom_prompt","phantom_som",    "P-prompt vs P-SoM (axis-1 SoM-prompt)"),
]


def load_lm_head_and_norm(device="cuda"):
    """Load Qwen3-VL-4B lm_head + final_norm from HF cache (offline)."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"  loading model (lm_head + norm only)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    # Qwen3-VL nests under .model.language_model
    if hasattr(model, "language_model"):
        norm = model.language_model.model.norm
    elif hasattr(model.model, "norm"):
        norm = model.model.norm
    else:
        raise RuntimeError(f"cannot locate final norm in {type(model).__name__}")
    lm_head = model.lm_head
    return tokenizer, lm_head, norm, model


@torch.no_grad()
def logits_at_layer(hidden: torch.Tensor, lm_head, norm) -> torch.Tensor:
    """hidden: (D,) → logits (V,) after final_norm + lm_head."""
    h = hidden.unsqueeze(0).to(lm_head.weight.device).to(lm_head.weight.dtype)
    h = norm(h)
    logits = lm_head(h).squeeze(0)
    return logits


def kl_divergence(p_logits, q_logits) -> float:
    """KL(P || Q) with softmax on logits."""
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum().item()
    return kl


def top1_agree(p_logits, q_logits) -> bool:
    return torch.argmax(p_logits).item() == torch.argmax(q_logits).item()


def compute_pair_logit_lens(npz: Path, pair_pairs: list, lm_head, norm, n_layers_use: int):
    d = np.load(npz, allow_pickle=True)
    H = d["hidden_states"]  # (N, L, D)
    ml = d["mode_labels_str"]
    means = {}
    for m in {p[0] for p in pair_pairs} | {p[1] for p in pair_pairs}:
        mask = ml == m
        if mask.sum() == 0:
            continue
        means[m] = H[mask].mean(axis=0)

    result = {}
    for a, b, label in pair_pairs:
        if a not in means or b not in means:
            continue
        layer_kl = []
        layer_disagree = []
        for L in range(n_layers_use):
            h_a = torch.tensor(means[a][L])
            h_b = torch.tensor(means[b][L])
            l_a = logits_at_layer(h_a, lm_head, norm)
            l_b = logits_at_layer(h_b, lm_head, norm)
            layer_kl.append(kl_divergence(l_a, l_b))
            layer_disagree.append(0.0 if top1_agree(l_a, l_b) else 1.0)
        result[label] = {
            "kl": np.array(layer_kl),
            "disagree": np.array(layer_disagree),
            "mode_a": a, "mode_b": b,
            "peak_kl_L": int(np.argmax(layer_kl)),
            "peak_kl": float(np.max(layer_kl)),
        }
    return result, n_layers_use


def write_md(cls_axis2, cls_axis1, red_axis2, red_axis1, n_layers, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Exp 3 — Logit lens at late layers (axis-2 vs axis-1)",
        "",
        "Apply Qwen3-VL-4B's final_norm + lm_head to per-layer per-mode mean hidden states.",
        "For each axis-isolated pair, compute KL(mode_a || mode_b) and top-1 token disagreement",
        "across all 37 layers. This probes whether axis-2 cosine signal (L23 peak 0.011) gets",
        "amplified into output distribution divergence by late-layer decoding.",
        "",
        "## Classifieds site",
        "",
        "### Axis-2 (prompt-family) pairs:",
        "",
        "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, info in cls_axis2.items():
        kl = info["kl"]
        lines.append(
            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
        )
    lines += ["", "### Axis-1 (text-format) pairs:", "",
              "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, info in cls_axis1.items():
        kl = info["kl"]
        lines.append(
            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
        )

    lines += ["", "## Reddit site", "",
              "### Axis-2 (prompt-family) pairs:", "",
              "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, info in red_axis2.items():
        kl = info["kl"]
        lines.append(
            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
        )
    lines += ["", "### Axis-1 (text-format) pairs:", "",
              "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, info in red_axis1.items():
        kl = info["kl"]
        lines.append(
            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "Three hypotheses tested:",
        "",
        "- **H_A (axis-2 absent from output)**: axis-2 KL flat <0.1 at all layers → prompt-family",
        "  effect bypasses logit lens, only visible via attention heads or runtime decoding.",
        "- **H_B (axis-2 amplified at output)**: axis-2 KL peak at L30+ ≫ cosine 0.011 magnitude →",
        "  late-layer decoding amplifies prompt prior into output divergence (Wu et al. tool calling",
        "  'knows but says differently' mirror).",
        "- **H_C (axis-2 tracks residual stream)**: axis-2 KL peak at L23 same as cosine peak →",
        "  prompt prior signal proportional to mid-layer geometry, no amplification.",
        "",
        "Cross-site replication should hold for any of the three. Compare axis-2 KL magnitudes to",
        "axis-1 KL magnitudes to see whether 3-4x ratio in cosine space persists at output level.",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"summary → {out}")


def plot(cls_a2, cls_a1, red_a2, red_a1, n_layers, out: Path):
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    layers = np.arange(n_layers)
    for ax, a2, a1, site in [(axes[0], cls_a2, cls_a1, "classifieds"),
                              (axes[1], red_a2, red_a1, "reddit")]:
        for label, info in a2.items():
            ax.plot(layers, info["kl"], color="#d62728" if "DOM" in label else "#ff7f0e",
                    linewidth=2.5, label=label)
        for label, info in a1.items():
            ax.plot(layers, info["kl"], color="#1f77b4" if "DOM" in label else "#2ca02c",
                    linestyle="--", linewidth=1.5, alpha=0.7, label=label)
        ax.axvline(17, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axvline(23, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Layer index")
        ax.set_title(f"{site}  (axis-2 solid, axis-1 dashed)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("KL divergence (logit lens)")
    fig.suptitle("Exp 3: Output distribution divergence per layer via logit lens", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"figure → {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cls-npz", type=Path, default=DEFAULT_CLS_NPZ)
    p.add_argument("--red-npz", type=Path, default=DEFAULT_RED_NPZ)
    p.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    p.add_argument("--output-fig", type=Path, default=DEFAULT_FIG)
    args = p.parse_args()

    print("Loading model...")
    tokenizer, lm_head, norm, _ = load_lm_head_and_norm(device="cuda")

    print("\n[cls] axis-2 pairs:")
    cls_a2, n_L = compute_pair_logit_lens(args.cls_npz, AXIS_2_PAIRS, lm_head, norm, 37)
    print(f"  done, {len(cls_a2)} pairs")

    print("[cls] axis-1 pairs:")
    cls_a1, _ = compute_pair_logit_lens(args.cls_npz, AXIS_1_PAIRS, lm_head, norm, 37)
    print(f"  done, {len(cls_a1)} pairs")

    print("[reddit] axis-2 pairs:")
    red_a2, _ = compute_pair_logit_lens(args.red_npz, AXIS_2_PAIRS, lm_head, norm, 37)
    print("[reddit] axis-1 pairs:")
    red_a1, _ = compute_pair_logit_lens(args.red_npz, AXIS_1_PAIRS, lm_head, norm, 37)

    write_md(cls_a2, cls_a1, red_a2, red_a1, n_L, args.output_md)
    plot(cls_a2, cls_a1, red_a2, red_a1, n_L, args.output_fig)


if __name__ == "__main__":
    main()
