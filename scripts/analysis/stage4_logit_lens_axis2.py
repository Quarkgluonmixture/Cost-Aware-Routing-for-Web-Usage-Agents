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
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLS_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states_v2_fixed.npz"
DEFAULT_RED_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_reddit/hidden_states_v2_fixed.npz"
DEFAULT_MD = ROOT / "docs/checkpoints/mechanism/results/axis2_logit_lens.md"
DEFAULT_FIG = ROOT / "results/phantom_paper/figures/fig_axis2_logit_lens.png"
MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
# Bug 5 fix (/codex-stress methodology audit 2026-05-12): pin HF revision
# to match HiddenStateExtractor + Stage 2B / Stage 4 v2 extraction. Previously
# unpinned, so logit lens KL applied `norm + lm_head` from an arbitrary cached
# revision to hidden states extracted under a pinned revision — making KL
# magnitudes non-reproducible across machines or cache states.
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

AXIS_2_PAIRS = [
    ("phantom_text", "phantom_som", "P-text vs P-SoM  (axis-2 flat-text)"),
    ("dom",          "phantom_prompt", "DOM vs P-prompt  (axis-2 hierarchical)"),
]
AXIS_1_PAIRS = [
    ("dom",           "phantom_text",   "DOM vs P-text    (axis-1 DOM-prompt)"),
    ("phantom_prompt","phantom_som",    "P-prompt vs P-SoM (axis-1 SoM-prompt)"),
]


def load_lm_head_and_norm(device="cuda"):
    """Load Qwen3-VL-4B lm_head + final_norm from HF cache (offline).

    Pipeline audit P0-8 fix (2026-05-13): model loaded in fp32 instead of
    bfloat16. Sub-permille mean-diff cosine geometry + KL between similar
    distributions requires >3-decimal precision; bf16 mantissa = 7 bits
    quantizes 4th-decimal of KL to noise. fp32 lm_head ~1.5GB VRAM on Qwen3-VL-4B
    (150k vocab × 2560 dim × 4 bytes) — fits comfortably on shared DGX.
    """
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, revision=MODEL_REVISION, trust_remote_code=True
    )
    print(f"  loading Qwen3VLForConditionalGeneration (lm_head + norm only, revision={MODEL_REVISION[:12]}..., fp32 P0-8)")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, revision=MODEL_REVISION, dtype=torch.float32,
        device_map=device, trust_remote_code=True,
    )
    # Qwen3-VL structure (verified via p79/mechanistic/activation_patching.py):
    #   model.model.language_model.layers  (36 decoder layers, no embedding included)
    #   model.model.language_model.norm    (final RMSNorm, sibling of layers)
    #   model.lm_head                       (top-level projection)
    norm = model.model.language_model.norm
    lm_head = model.lm_head
    print(f"  norm: {type(norm).__name__}, lm_head: {type(lm_head).__name__}")
    return tokenizer, lm_head, norm, model


@torch.no_grad()
def logits_at_layer(hidden: torch.Tensor, lm_head, norm) -> torch.Tensor:
    """hidden: (D,) → logits (V,) after final_norm + lm_head.

    P0-8 fix: keep fp32 throughout. Previous version coerced to lm_head.weight.dtype
    which was bf16 — destroyed 4th-decimal of KL between similar distributions.
    """
    h = hidden.unsqueeze(0).to(lm_head.weight.device).float()
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


def compute_pair_logit_lens(npz_path: Path, pair_pairs: list, lm_head, norm, n_layers_use: int):
    """Per-task paired logit-lens KL (P0-1 fix, 2026-05-13).

    For each (mode_a, mode_b) pair: compute KL between decoded logits at
    every (task_id, step) common to both modes, then average across paired
    rows. Also report legacy KL-of-decoded-means as deprecated proxy with ratio.

    Previous version (commit 9f3f516 and earlier) computed
    KL(decode(mean(h_a)), decode(mean(h_b))) — KL between two decoded mode-means.
    By Jensen + softmax non-linearity, this differs from per-task KL averaged
    (E_t[KL(decode(h_a_t), decode(h_b_t))]). The per-task version is the
    paper-grade quantity; of-means is reported for monotonicity check only.

    Audit defuse fire (`stage4_logit_lens_per_task.py`, 2026-05-13 morning)
    measured ratio per-task / of-means in [1.1, 3.9] across cls + reddit ×
    axis-1/axis-2 pairs — direction consistent, magnitude understated by
    of-means. Hero claim §1.2 "amplification" survives terminology fix.
    """
    from _paired_npz_helpers import load_v2_npz, paired_rows

    npz = load_v2_npz(npz_path)
    H = npz["H"]  # (N, L, D), fp32
    assert H.shape[1] == 37, f"expected 37 layers (embed + 36 blocks), got {H.shape[1]}"

    # Also pre-compute mode-means for of-means proxy
    means = {}
    for m in {p[0] for p in pair_pairs} | {p[1] for p in pair_pairs}:
        mask = npz["mode"] == m
        if mask.sum() == 0:
            continue
        means[m] = H[mask].mean(axis=0)

    result = {}
    for a, b, label in pair_pairs:
        if a not in means or b not in means:
            continue
        try:
            Ha, Hb, keys = paired_rows(npz, a, b)
        except (KeyError, ValueError) as e:
            print(f"  skip {label}: {e}")
            continue
        if len(keys) == 0:
            continue
        n_paired = len(keys)
        print(f"  {label}: {n_paired} paired (task_id, step) rows")

        kl_per_task = np.zeros((n_paired, n_layers_use), dtype=np.float32)
        kl_of_means = np.zeros(n_layers_use, dtype=np.float32)
        disagree_per_task = np.zeros((n_paired, n_layers_use), dtype=np.float32)

        for L in range(n_layers_use):
            # KL of decoded mode-means (legacy proxy)
            l_a_mean = logits_at_layer(torch.tensor(means[a][L]), lm_head, norm)
            l_b_mean = logits_at_layer(torch.tensor(means[b][L]), lm_head, norm)
            kl_of_means[L] = kl_divergence(l_a_mean, l_b_mean)

            # Per-task paired KL (paper-grade)
            for ti in range(n_paired):
                l_a = logits_at_layer(torch.tensor(Ha[ti, L]), lm_head, norm)
                l_b = logits_at_layer(torch.tensor(Hb[ti, L]), lm_head, norm)
                kl_per_task[ti, L] = kl_divergence(l_a, l_b)
                disagree_per_task[ti, L] = 0.0 if top1_agree(l_a, l_b) else 1.0

        kl_per_task_mean = kl_per_task.mean(axis=0)
        kl_per_task_std = kl_per_task.std(axis=0)
        # Avoid div by zero in ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(kl_of_means > 1e-9, kl_per_task_mean / kl_of_means, np.nan)

        peak_L_per_task = int(np.argmax(kl_per_task_mean))
        peak_L_of_means = int(np.argmax(kl_of_means))

        result[label] = {
            # Paper-grade (paired)
            "kl_per_task_mean": kl_per_task_mean,
            "kl_per_task_std": kl_per_task_std,
            "peak_L_per_task": peak_L_per_task,
            "peak_kl_per_task": float(kl_per_task_mean[peak_L_per_task]),
            "disagree_mean": disagree_per_task.mean(axis=0),
            # Legacy proxy (of-means)
            "kl_of_means": kl_of_means,
            "peak_L_of_means": peak_L_of_means,
            "peak_kl_of_means": float(kl_of_means[peak_L_of_means]),
            # Diagnostic
            "ratio_per_task_over_means": ratio,
            "n_paired": n_paired,
            "paired_keys": keys,
            "mode_a": a, "mode_b": b,
            # Backcompat aliases
            "kl": kl_per_task_mean,
            "disagree": disagree_per_task.mean(axis=0),
            "peak_kl_L": peak_L_per_task,
            "peak_kl": float(kl_per_task_mean[peak_L_per_task]),
        }
    return result, n_layers_use


def _pair_row(label: str, info: dict) -> str:
    """Render one pair as a markdown row with both per-task + of-means columns."""
    pt = info["kl_per_task_mean"]
    om = info["kl_of_means"]
    pt_std = info["kl_per_task_std"]
    pt_peak = info["peak_L_per_task"]
    om_peak = info["peak_L_of_means"]
    # Ratio at per-task peak layer (where the headline is)
    om_at_pt_peak = om[pt_peak]
    r = info["peak_kl_per_task"] / max(om_at_pt_peak, 1e-9)
    return (
        f"| {label} | **L{pt_peak}** | {info['peak_kl_per_task']:.4f} ± {pt_std[pt_peak]:.4f} | "
        f"L{om_peak} | {info['peak_kl_of_means']:.4f} | **{r:.2f}×** | "
        f"{pt[17]:.4f} | {pt[23]:.4f} | {info['n_paired']} |"
    )


def write_md(cls_axis2, cls_axis1, red_axis2, red_axis1, n_layers, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Exp 3 — Logit lens at late layers (axis-2 vs axis-1) — per-task paired (v2 NPZ)",
        "",
        "P0-1 fix (2026-05-13): now computes BOTH per-task paired KL (paper-grade) AND",
        "KL-of-decoded-mode-means (legacy proxy) with ratio per pair. Per-task KL is the",
        "paper-grade quantity; of-means reported for monotonicity check only.",
        "",
        "P0-8 fix (2026-05-13): lm_head + norm now loaded in fp32 (not bf16). bf16 mantissa",
        "= 7 bits quantizes 4th-decimal of KL between similar distributions to noise. fp32",
        "preserves sub-permille precision needed for the cosine-causal disjoint claim.",
        "",
        "**Interpretation of ratio per-task / of-means**:",
        "- ratio ≈ 1 → 'amplification' framing terminology-fix-able (KL-of-means is defensible proxy)",
        "- ratio ≫ 1 (>2×) → per-task signal MUCH stronger; paper UNDERSTATES mechanism",
        "- ratio ≪ 1 (<0.5×) → KL-of-means inflates; 'amplification' hero claim collapses",
        "",
    ]

    def site_section(name: str, axis2: dict, axis1: dict):
        out_lines = [f"## {name}", ""]
        for axis_label, axis_results in [
            ("Axis-2 (prompt-family) pairs", axis2),
            ("Axis-1 (text-format) pairs", axis1),
        ]:
            out_lines.append(f"### {axis_label}")
            out_lines.append("")
            out_lines.append(
                "| Pair | Peak L per-task | Peak KL per-task ± std | "
                "Peak L of-means | Peak KL of-means | "
                "Ratio @ per-task peak | KL @ L17 (per-task) | KL @ L23 (per-task) | n_paired |"
            )
            out_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for label, info in axis_results.items():
                out_lines.append(_pair_row(label, info))
            out_lines.append("")
        return out_lines

    lines += site_section("Classifieds site", cls_axis2, cls_axis1)
    lines += site_section("Reddit site", red_axis2, red_axis1)

    lines += [
        "## Interpretation",
        "",
        "Three hypotheses tested:",
        "",
        "- **H_A (axis-2 absent from output)**: axis-2 per-task KL flat <0.1 at all layers → prompt-family",
        "  effect bypasses logit lens, only visible via attention heads or runtime decoding.",
        "- **H_B (axis-2 amplified at output)**: axis-2 per-task KL peak at L21-L25 ≫ cosine 0.005-0.009 magnitude →",
        "  late-layer decoding amplifies prompt prior into output divergence (Wu et al. tool calling",
        "  'knows but says differently' mirror).",
        "- **H_C (axis-2 tracks residual stream)**: axis-2 per-task KL peak at L36 same as cosine peak →",
        "  prompt prior signal proportional to mid-layer geometry, no amplification.",
        "",
        "Cross-site replication should hold for any of the three.",
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
