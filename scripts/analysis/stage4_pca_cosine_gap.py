#!/usr/bin/env python3
"""Stage 4 Method 4.2: PCA cosine gap analysis of phantom routing space.

Ports Tool Calling Linear Steerable Circuit method (Anonymous 2026 ACL, validated
on Qwen3-4B) to Qwen3-VL-4B (B1). Tests whether phantom routing space modes are
mechanistically distinct in hidden state geometry layer-by-layer.

Three analyses per (mode pair, layer):
  A. Cosine gap between mean hidden states
  B. AUROC: project hidden states onto (mean_A - mean_B) direction, predict mode
  C. Per-(mode, layer) PCA top-10 variance explained

Outputs:
  - results/mechanistic/stage4_multimode_b1_cls/method42_metrics.json
  - docs/checkpoints/stage4_method42_results.md
  - results/phantom_paper/figures/fig_stage4_pca_cosine_gap.png
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states_v2_fixed.npz"
DEFAULT_OUT_JSON = ROOT / "results/mechanistic/stage4_multimode_b1_cls/method42_metrics.json"
DEFAULT_OUT_MD = ROOT / "docs/checkpoints/stage4_method42_results.md"
DEFAULT_OUT_FIG = ROOT / "results/phantom_paper/figures/fig_stage4_pca_cosine_gap.png"

MODES = ["dom", "phantom_text", "phantom_prompt", "phantom_som", "som", "vision"]
DISPLAY = {"dom": "DOM", "phantom_text": "P-text", "phantom_prompt": "P-prompt",
           "phantom_som": "P-SoM", "som": "SoM", "vision": "Vision"}


def cosine_gap(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))


def pair_key(a: str, b: str) -> str:
    """Canonical pair key using MODES index order (matches itertools.combinations output)."""
    i, j = MODES.index(a), MODES.index(b)
    return f"{MODES[min(i, j)]}_vs_{MODES[max(i, j)]}"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--output-fig", type=Path, default=DEFAULT_OUT_FIG)
    args = parser.parse_args()
    NPZ = args.input
    OUT_JSON = args.output_json
    OUT_MD = args.output_md
    OUT_FIG = args.output_fig

    d = np.load(NPZ, allow_pickle=True)
    H = d["hidden_states"]
    mode_labels = d["mode_labels_str"]
    task_ids = d["task_ids"] if "task_ids" in d.files else None
    n_layers = H.shape[1]
    print(f"[stage4] loaded {H.shape[0]} examples × {n_layers} layers × {H.shape[2]} dim")

    states = {m: H[mode_labels == m] for m in MODES}
    means = {m: states[m].mean(axis=0) for m in MODES}  # each (37, 2560)
    print(f"[stage4] per-mode counts: " + ", ".join(f"{m}={len(states[m])}" for m in MODES))

    # Per-mode task_id mapping for leave-one-task-out (Bug 3 fix, codex
    # methodology audit 2026-05-12: previous AUROC fit direction on the
    # same examples used to evaluate → inflated, not held-out decodability).
    mode_task_ids = {m: task_ids[mode_labels == m] if task_ids is not None else None
                     for m in MODES}

    pairs = list(combinations(MODES, 2))
    cos_gap = np.zeros((len(pairs), n_layers))
    auroc_in_sample = np.zeros((len(pairs), n_layers))
    auroc_lototask = np.zeros((len(pairs), n_layers))  # leave-one-task-out CV
    for pi, (m1, m2) in enumerate(pairs):
        for L in range(n_layers):
            c1, c2 = means[m1][L], means[m2][L]
            cos_gap[pi, L] = cosine_gap(c1, c2)
            direction = (c1 - c2) / (np.linalg.norm(c1 - c2) + 1e-9)
            s1 = states[m1][:, L, :] @ direction
            s2 = states[m2][:, L, :] @ direction
            y = np.concatenate([np.ones(len(s1)), np.zeros(len(s2))])
            scores = np.concatenate([s1, s2])
            try:
                auroc_in_sample[pi, L] = roc_auc_score(y, scores)
            except Exception:
                auroc_in_sample[pi, L] = 0.5

            # Leave-one-task-out CV — only when task_ids are available
            tids_m1 = mode_task_ids[m1]
            tids_m2 = mode_task_ids[m2]
            if tids_m1 is None or tids_m2 is None:
                auroc_lototask[pi, L] = np.nan
                continue
            # Tasks that appear in BOTH modes (paper-grade design has all
            # tasks in all modes, so this is usually all 24)
            common_tasks = sorted(set(tids_m1.tolist()) & set(tids_m2.tolist()))
            if len(common_tasks) < 3:
                auroc_lototask[pi, L] = np.nan
                continue
            fold_aurocs = []
            for held_out_tid in common_tasks:
                # Train: all examples whose task_id != held_out_tid
                train_mask_m1 = tids_m1 != held_out_tid
                train_mask_m2 = tids_m2 != held_out_tid
                test_mask_m1 = tids_m1 == held_out_tid
                test_mask_m2 = tids_m2 == held_out_tid
                if (train_mask_m1.sum() == 0 or train_mask_m2.sum() == 0 or
                        test_mask_m1.sum() == 0 or test_mask_m2.sum() == 0):
                    continue
                train_c1 = states[m1][train_mask_m1, L, :].mean(0)
                train_c2 = states[m2][train_mask_m2, L, :].mean(0)
                train_dir = (train_c1 - train_c2) / (np.linalg.norm(train_c1 - train_c2) + 1e-9)
                test_s1 = states[m1][test_mask_m1, L, :] @ train_dir
                test_s2 = states[m2][test_mask_m2, L, :] @ train_dir
                test_y = np.concatenate([np.ones(len(test_s1)), np.zeros(len(test_s2))])
                test_scores = np.concatenate([test_s1, test_s2])
                if len(np.unique(test_y)) < 2:
                    continue
                try:
                    fold_aurocs.append(roc_auc_score(test_y, test_scores))
                except Exception:
                    pass
            auroc_lototask[pi, L] = float(np.mean(fold_aurocs)) if fold_aurocs else np.nan

    pca_var = np.zeros((len(MODES), n_layers))
    for mi, mode in enumerate(MODES):
        X = states[mode]  # (n, 37, 2560)
        for L in range(n_layers):
            if X.shape[0] >= 11:
                n_comp = min(10, X.shape[0] - 1)
                pca_var[mi, L] = PCA(n_components=n_comp).fit(X[:, L, :]).explained_variance_ratio_.sum()

    peak = {}
    for pi, (m1, m2) in enumerate(pairs):
        L = int(np.argmax(cos_gap[pi]))
        peak[f"{m1}_vs_{m2}"] = {
            "layer": L,
            "gap": float(cos_gap[pi, L]),
            "auroc_in_sample_at_peak": float(auroc_in_sample[pi, L]),
            "auroc_lototask_at_peak": (
                float(auroc_lototask[pi, L])
                if not np.isnan(auroc_lototask[pi, L]) else None
            ),
        }

    # Replace NaN with None for JSON serializability
    def _nan_to_none(arr):
        return [None if np.isnan(x) else float(x) for x in arr]

    metrics = {
        "n_examples": int(H.shape[0]), "n_layers": int(n_layers), "n_modes": len(MODES),
        "modes": MODES, "n_per_mode": {m: int(len(states[m])) for m in MODES},
        "pairwise_cosine_gap": {f"{m1}_vs_{m2}": cos_gap[pi].tolist()
                                  for pi, (m1, m2) in enumerate(pairs)},
        "pairwise_auroc_in_sample": {f"{m1}_vs_{m2}": auroc_in_sample[pi].tolist()
                                       for pi, (m1, m2) in enumerate(pairs)},
        "pairwise_auroc_lototask": {f"{m1}_vs_{m2}": _nan_to_none(auroc_lototask[pi])
                                      for pi, (m1, m2) in enumerate(pairs)},
        "pca_top10_var_ratio": {m: pca_var[mi].tolist() for mi, m in enumerate(MODES)},
        "peak_disruption_layers": peak,
        "auroc_protocol_note": (
            "auroc_in_sample fits mode-mean direction on all examples and scores those "
            "same examples (inflated, NOT held-out decodability). auroc_lototask is "
            "leave-one-task-out cross-validation: for each held-out task, fit direction "
            "on the remaining tasks' means, then score the held-out task's examples. "
            "Report lototask as the paper-grade linear-readability metric; in-sample is "
            "kept for descriptive comparison only. Bug 3 fix per codex methodology audit "
            "2026-05-12."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"[stage4] metrics → {OUT_JSON}")

    write_summary(metrics, OUT_MD)
    plot(cos_gap, auroc_lototask, pairs, pca_var, OUT_FIG)


def write_summary(m: dict, out: Path) -> None:
    sorted_pairs = sorted(m["peak_disruption_layers"].items(),
                           key=lambda x: -x[1]["gap"])
    lines = [
        "# Stage 4 Method 4.2: PCA Cosine Gap Analysis",
        "",
        f"**Data**: {m['n_examples']} examples × {m['n_layers']} layers × {m['n_modes']} modes (Qwen3-VL-4B B1 cls)",
        f"**Per-mode n**: " + ", ".join(f"{DISPLAY[k]}={v}" for k, v in m['n_per_mode'].items()),
        "",
        "**AUROC protocol** (Bug 3 fix, codex methodology audit 2026-05-12): paper-grade "
        "metric is `auroc_lototask` = leave-one-task-out cross-validation (fit mode-mean "
        "direction on training tasks, score held-out task). `auroc_in_sample` (fit + score "
        "on same examples) is reported for descriptive comparison only; treat any in-sample "
        "≥0.95 as expected algebraic separability, NOT held-out linear-readability.",
        "",
        "## Peak disruption layer per mode pair",
        "",
        "Sorted by cosine gap magnitude (= geometric distance between mode means in hidden space):",
        "",
        "| Mode pair | Peak layer | Cosine gap | AUROC (in-sample) | AUROC (lototask) |",
        "|---|---|---|---|---|",
    ]
    for k, v in sorted_pairs:
        m1, m2 = k.split("_vs_")
        lototask_val = v.get("auroc_lototask_at_peak")
        lototask_str = f"{lototask_val:.3f}" if lototask_val is not None else "n/a"
        lines.append(
            f"| {DISPLAY[m1]} vs {DISPLAY[m2]} | L{v['layer']:02d} | {v['gap']:.4f} | "
            f"{v['auroc_in_sample_at_peak']:.3f} | {lototask_str} |"
        )

    # Mid-layer (L17) snapshot — paper §5 disruption locus
    L17_section = ["", "## L17 cosine gap snapshot (paper §5 disruption locus)", ""]
    L17_section.append("| Mode pair | L17 cosine gap | L17 AUROC in-sample | L17 AUROC lototask |")
    L17_section.append("|---|---|---|---|")
    pairs = list(combinations(MODES, 2))
    for pi, (m1, m2) in enumerate(pairs):
        gap = m["pairwise_cosine_gap"][f"{m1}_vs_{m2}"][17]
        a_in = m["pairwise_auroc_in_sample"][f"{m1}_vs_{m2}"][17]
        a_lo = m["pairwise_auroc_lototask"][f"{m1}_vs_{m2}"][17]
        a_lo_str = f"{a_lo:.3f}" if a_lo is not None else "n/a"
        L17_section.append(f"| {DISPLAY[m1]} vs {DISPLAY[m2]} | {gap:.4f} | {a_in:.3f} | {a_lo_str} |")
    lines.extend(L17_section)

    # Phantom-arm specific anchor — P-SoM cosine to each baseline mode at L17
    psom_section = ["", "## P-SoM vs baseline modes (paper §5 HERO arm)", "",
                     "P-SoM identity test: is P-SoM closer to SoM (prompt-axis sibling) or DOM (text-axis sibling)?",
                     ""]
    psom_section.append("| L | P-SoM↔DOM | P-SoM↔SoM | P-SoM↔Vision | P-SoM↔P-text | P-SoM↔P-prompt |")
    psom_section.append("|---|---|---|---|---|---|")
    for L in [0, 8, 11, 17, 24, 30, 36]:
        row = [f"L{L:02d}"]
        for other in ["dom", "som", "vision", "phantom_text", "phantom_prompt"]:
            key = pair_key("phantom_som", other)
            row.append(f"{m['pairwise_cosine_gap'][key][L]:.4f}")
        psom_section.append("| " + " | ".join(row) + " |")
    lines.extend(psom_section)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[stage4] summary → {out}")


def plot(cos_gap, auroc, pairs, pca_var, out):
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    pair_labels = [f"{DISPLAY[m1]}↔{DISPLAY[m2]}" for m1, m2 in pairs]

    ax = axes[0, 0]
    im = ax.imshow(cos_gap, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pair_labels, fontsize=7)
    ax.set_xlabel("Layer index")
    ax.set_title("(a) Pairwise cosine gap (geometric distance between mode means)")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.imshow(auroc, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pair_labels, fontsize=7)
    ax.set_xlabel("Layer index")
    ax.set_title("(b) Pairwise AUROC (project onto Δ-mean direction, classify)")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    for mi, mode in enumerate(MODES):
        ax.plot(pca_var[mi], label=DISPLAY[mode], linewidth=1.5)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Top-10 PCA cumulative variance explained")
    ax.set_title("(c) Per-mode within-cluster dimensionality")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    psom_idx = {f"{m1}_vs_{m2}": i for i, (m1, m2) in enumerate(pairs)}
    for other in ["dom", "som", "vision", "phantom_text", "phantom_prompt"]:
        key = pair_key("phantom_som", other)
        ax.plot(cos_gap[psom_idx[key]], label=f"P-SoM ↔ {DISPLAY[other]}", linewidth=1.5)
    ax.axvline(17, color="red", linestyle=":", alpha=0.5, label="L17 (Stage 2 disruption locus)")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Cosine gap to P-SoM")
    ax.set_title("(d) P-SoM identity — closest sibling per layer")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle("Stage 4 Method 4.2: Phantom routing space hidden state geometry (Qwen3-VL-4B B1 cls)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[stage4] figure → {out}")


if __name__ == "__main__":
    main()
