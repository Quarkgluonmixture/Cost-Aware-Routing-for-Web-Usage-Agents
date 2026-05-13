#!/usr/bin/env python3
"""Exp 3 v3: Per-task logit lens KL (defuse for /stress v5 OOB attack).

The /stress v5 audit (2026-05-13) flagged that `stage4_logit_lens_axis2.py`
computes `KL(softmax(lm_head(mean_h_A)), softmax(lm_head(mean_h_B)))` — i.e.,
KL between decoded-averaged-means — not the paper's implied per-task
amplification `E_task[KL(softmax(lm_head(h_A_t)), softmax(lm_head(h_B_t)))]`.
By Jensen's inequality + non-linearity of softmax, these can differ in
either direction. Plan.md §1.2 "amplification 8-44×" hero claim depends on
which interpretation is true.

This script computes BOTH and outputs the ratio per (pair, site, layer).

  - KL_per_task = mean over tasks of KL(decode(h_A_task), decode(h_B_task))
  - KL_of_means = KL(decode(mean(h_A)), decode(mean(h_B)))   [original method]
  - Ratio       = KL_per_task / KL_of_means

Per-task pairing requires `task_ids` in NPZ. Both v2 NPZs include task_ids.

Outputs:
  docs/checkpoints/mechanism/results/axis2_logit_lens_per_task_2026-05-13.md
  /tmp/stage4_logit_lens_per_task.done   (Tier 1 file marker for /stress monitor)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLS_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states_v2_fixed.npz"
DEFAULT_RED_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_reddit/hidden_states_v2_fixed.npz"
DEFAULT_MD = ROOT / "docs/checkpoints/mechanism/results/axis2_logit_lens_per_task_2026-05-13.md"
DONE_MARKER = Path("/tmp/stage4_logit_lens_per_task.done")
MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
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
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    AutoTokenizer.from_pretrained(MODEL_PATH, revision=MODEL_REVISION, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, revision=MODEL_REVISION, dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    )
    return model.lm_head, model.model.language_model.norm


@torch.no_grad()
def logits_at_layer(hidden: torch.Tensor, lm_head, norm) -> torch.Tensor:
    h = hidden.unsqueeze(0).to(lm_head.weight.device).to(lm_head.weight.dtype)
    h = norm(h)
    return lm_head(h).squeeze(0)


def kl_divergence(p_logits, q_logits) -> float:
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    p = log_p.exp()
    return float((p * (log_p - log_q)).sum().item())


def compute_per_task_and_mean_kl(npz_path: Path, pairs, lm_head, norm, n_layers: int) -> Dict:
    """For each pair × layer, compute BOTH:
       - KL_per_task (paired by task_id, then averaged)
       - KL_of_means (decode the mode-mean, single KL)
    """
    d = np.load(npz_path, allow_pickle=True)
    H = d["hidden_states"]              # (N, L, D), float32
    ml = d["mode_labels_str"]
    tids = d["task_ids"]
    print(f"[per-task-kl] {npz_path.name}: shape={H.shape}, modes={set(ml.tolist())}")

    # Pre-compute mode means once
    means = {}
    for m in set(ml.tolist()):
        mask = ml == m
        if mask.sum() == 0:
            continue
        means[m] = H[mask].mean(axis=0)  # (L, D)

    # Pre-compute per-mode task → hidden state index (assume 1 example per task per mode)
    mode_task_to_idx = {}
    for m in set(ml.tolist()):
        mask = ml == m
        mode_tids = tids[mask]
        mode_hidx = np.where(mask)[0]
        mode_task_to_idx[m] = dict(zip(mode_tids.tolist(), mode_hidx.tolist()))

    result = {}
    for a, b, label in pairs:
        if a not in means or b not in means:
            continue
        # Common task IDs between modes a and b
        common_tids = sorted(set(mode_task_to_idx[a].keys()) & set(mode_task_to_idx[b].keys()))
        print(f"[per-task-kl]   {label}: {len(common_tids)} paired tasks")

        kl_per_task = np.zeros((len(common_tids), n_layers))
        kl_of_means = np.zeros(n_layers)
        for L in range(n_layers):
            # KL of means (original method)
            l_a_mean = logits_at_layer(torch.tensor(means[a][L]), lm_head, norm)
            l_b_mean = logits_at_layer(torch.tensor(means[b][L]), lm_head, norm)
            kl_of_means[L] = kl_divergence(l_a_mean, l_b_mean)
            # Per-task KL averaged
            for ti, tid in enumerate(common_tids):
                idx_a = mode_task_to_idx[a][tid]
                idx_b = mode_task_to_idx[b][tid]
                h_a = torch.tensor(H[idx_a, L, :])
                h_b = torch.tensor(H[idx_b, L, :])
                l_a = logits_at_layer(h_a, lm_head, norm)
                l_b = logits_at_layer(h_b, lm_head, norm)
                kl_per_task[ti, L] = kl_divergence(l_a, l_b)

        kl_per_task_mean = kl_per_task.mean(axis=0)
        kl_per_task_std = kl_per_task.std(axis=0)
        # Avoid div by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(kl_of_means > 1e-9, kl_per_task_mean / kl_of_means, np.nan)

        result[label] = {
            "n_tasks_paired": len(common_tids),
            "common_tids": common_tids,
            "mode_a": a, "mode_b": b,
            "kl_per_task_mean": kl_per_task_mean.tolist(),
            "kl_per_task_std": kl_per_task_std.tolist(),
            "kl_of_means": kl_of_means.tolist(),
            "ratio_per_task_over_means": ratio.tolist(),
            "peak_L_per_task": int(np.argmax(kl_per_task_mean)),
            "peak_kl_per_task": float(np.max(kl_per_task_mean)),
            "peak_L_of_means": int(np.argmax(kl_of_means)),
            "peak_kl_of_means": float(np.max(kl_of_means)),
        }
        print(f"[per-task-kl]   peak per-task L{result[label]['peak_L_per_task']} "
              f"KL={result[label]['peak_kl_per_task']:.4f} vs "
              f"of-means L{result[label]['peak_L_of_means']} "
              f"KL={result[label]['peak_kl_of_means']:.4f} "
              f"ratio={result[label]['peak_kl_per_task']/max(result[label]['peak_kl_of_means'],1e-9):.2f}x")
    return result


def write_md(cls_a2, cls_a1, red_a2, red_a1, n_layers, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Exp 3 v3 — Per-task logit lens KL vs KL-of-means (defuse 2026-05-13)",
        "",
        "**Audit target** (`/stress v5 OOB attack 1, 2026-05-13`): `stage4_logit_lens_axis2.py:114`",
        "computes `KL(decode(mean_h_A), decode(mean_h_B))` — KL of decoded-averaged-means.",
        "Plan.md §1.2 hero claim 'amplification 8-44×' depends on this matching the paper's",
        "implied per-task amplification `E_task[KL(decode(h_A_t), decode(h_B_t))]`.",
        "By Jensen's inequality + non-linearity of softmax, these can differ.",
        "",
        "This script computes BOTH on the same v2 NPZ and reports the ratio.",
        "",
        "**Interpretation**:",
        "- ratio ≈ 1 → 'amplification' framing terminology-fix-able (KL-of-means is a defensible proxy)",
        "- ratio ≫ 1 (>2×) → per-task signal is MUCH stronger; paper UNDERSTATES mechanism",
        "- ratio ≪ 1 (<0.5×) → KL-of-means inflates; 'amplification' hero claim collapses",
        "",
    ]

    def site_section(site_label: str, a2: dict, a1: dict):
        lines.append(f"## {site_label}")
        lines.append("")
        for section_name, results_dict in [("Axis-2 (prompt-family)", a2), ("Axis-1 (text-format)", a1)]:
            lines.append(f"### {section_name}")
            lines.append("")
            lines.append("| Pair | Peak L per-task | Peak KL per-task ± std | Peak L of-means | Peak KL of-means | Ratio @ per-task peak |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for label, info in results_dict.items():
                pt_peak = info["peak_L_per_task"]
                of_peak = info["peak_L_of_means"]
                pt_kl = info["peak_kl_per_task"]
                of_kl = info["peak_kl_of_means"]
                pt_std = info["kl_per_task_std"][pt_peak]
                # Ratio at per-task peak layer
                of_at_pt_peak = info["kl_of_means"][pt_peak]
                r = pt_kl / max(of_at_pt_peak, 1e-9)
                lines.append(
                    f"| {label} | **L{pt_peak}** | {pt_kl:.4f} ± {pt_std:.4f} | "
                    f"L{of_peak} | {of_kl:.4f} | **{r:.2f}×** |"
                )
            lines.append("")

    site_section("Classifieds (cls)", cls_a2, cls_a1)
    site_section("Reddit (red)", red_a2, red_a1)

    lines.append("## Verdict logic")
    lines.append("")
    lines.append("- All 8 ratios (2 sites × 4 pairs) within [0.5, 2.0] → terminology-only fix")
    lines.append("- Any ratio > 2 → mechanism stronger than reported, paper UNDERSTATES")
    lines.append("- Any ratio < 0.5 → 'amplification' hero claim REJECTED, §1.2 rewrite required")
    lines.append("")

    out.write_text("\n".join(lines))
    print(f"[per-task-kl] wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls-npz", type=Path, default=DEFAULT_CLS_NPZ)
    parser.add_argument("--red-npz", type=Path, default=DEFAULT_RED_NPZ)
    parser.add_argument("--out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--n-layers", type=int, default=37)
    args = parser.parse_args()

    print(f"[per-task-kl] loading lm_head + norm")
    lm_head, norm = load_lm_head_and_norm()

    print(f"[per-task-kl] === cls site ===")
    cls_axis2 = compute_per_task_and_mean_kl(args.cls_npz, AXIS_2_PAIRS, lm_head, norm, args.n_layers)
    cls_axis1 = compute_per_task_and_mean_kl(args.cls_npz, AXIS_1_PAIRS, lm_head, norm, args.n_layers)

    print(f"[per-task-kl] === reddit site ===")
    red_axis2 = compute_per_task_and_mean_kl(args.red_npz, AXIS_2_PAIRS, lm_head, norm, args.n_layers)
    red_axis1 = compute_per_task_and_mean_kl(args.red_npz, AXIS_1_PAIRS, lm_head, norm, args.n_layers)

    write_md(cls_axis2, cls_axis1, red_axis2, red_axis1, args.n_layers, args.out)

    # Also dump JSON for downstream analysis
    json_out = args.out.with_suffix(".json")
    json_out.write_text(json.dumps({
        "cls_axis2": cls_axis2, "cls_axis1": cls_axis1,
        "red_axis2": red_axis2, "red_axis1": red_axis1,
        "n_layers": args.n_layers,
        "audit_origin": "/stress v5 OOB attack 1, 2026-05-13",
        "method_note": ("KL_per_task = E_task[KL(decode(h_A_t), decode(h_B_t))]; "
                        "KL_of_means = KL(decode(mean(h_A)), decode(mean(h_B))). "
                        "Ratio diagnoses Jensen artifact in original mode-mean script."),
    }, indent=2))
    print(f"[per-task-kl] wrote {json_out}")

    DONE_MARKER.write_text(f"done at {__import__('datetime').datetime.now().isoformat()}\n")
    print(f"[per-task-kl] marker → {DONE_MARKER}")


if __name__ == "__main__":
    main()
