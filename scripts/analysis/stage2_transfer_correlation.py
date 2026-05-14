#!/usr/bin/env python3
"""Stage 2 transfer-vs-disruption correlation analysis.

Question: when mid-layer (L11/L17) patching disrupts target generation, does
the patched output drift TOWARD source (genuine transfer / fusion) or just
AWAY from target (destruction)?

Method (LD-based per-task; no re-run needed):
- For each cell × layer L:
    Δ_to_target(task, L) = ld_to_target(task, L) - ld_to_target(task, L35)
    Δ_to_source(task, L) = ld_to_source(task, L) - ld_to_source(task, L35)
- L35 baseline = no-effective-patch reference.
- Pearson correlation across tasks:
    + Strong NEGATIVE → tasks where target shifts more (Δ_to_target↑) also show
      patched output closer to source (Δ_to_source↓) → genuine TRANSFER, just
      hidden by mean-level greedy decode lock-in (sub-population mechanism).
    + ZERO / POSITIVE → no source bias; patching just destroys target →
      paper §5 must reframe "fusion locus" → "disruption locus".

Per-cell verdict at L11 + L17 (mid-layer probe).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CELLS = {
    "A":  ROOT / "results/mechanistic/stage2b_curated_b1_cls_myriad",
    "B":  ROOT / "results/mechanistic/stage2c_reverse_curated_b1_cls_myriad",
    "C":  ROOT / "results/mechanistic/stage2b_2x2_fwd_revtasks_myriad",
    "D":  ROOT / "results/mechanistic/stage2c_2x2_rev_strongtasks_myriad",
    "F":  ROOT / "results/mechanistic/stage2b_cellf_fwd_reddit_strong_myriad",
    "G":  ROOT / "results/mechanistic/stage2c_cellg_rev_reddit_reverse_myriad",
    "Cr": ROOT / "results/mechanistic/stage2b_cellcr_reddit_fwd_revtier_myriad",
    "Dr": ROOT / "results/mechanistic/stage2c_celldr_reddit_rev_strongtier_myriad",
    # Stage 3 valid cells (text-only attribution: target=phantom_text)
    "Ht_cls": ROOT / "results/mechanistic/stage3_cellht_cls_fwd_text_myriad",
    "Ht_red": ROOT / "results/mechanistic/stage3_cellht_red_fwd_text_myriad",
    "Hp_cls": ROOT / "results/mechanistic/stage3_cellhp_cls_fwd_prompt_myriad",
    "Hp_red": ROOT / "results/mechanistic/stage3_cellhp_red_fwd_prompt_myriad",
}

CELL_LABELS = {
    "A":  "cls fwd × strong (target=phantom_som)",
    "B":  "cls rev × reverse (target=phantom_som)",
    "C":  "cls fwd × reverse (target=phantom_som)",
    "D":  "cls rev × strong (target=phantom_som)",
    "F":  "reddit fwd × strong (target=phantom_som)",
    "G":  "reddit rev × reverse (target=phantom_som)",
    "Cr": "reddit fwd × reverse (target=phantom_som)",
    "Dr": "reddit rev × strong (target=phantom_som)",
    "Ht_cls": "cls fwd × strong (target=phantom_TEXT) Stage 3",
    "Ht_red": "reddit fwd × strong (target=phantom_TEXT) Stage 3",
    "Hp_cls": "cls fwd × strong (target=phantom_PROMPT) Stage 3",
    "Hp_red": "reddit fwd × strong (target=phantom_PROMPT) Stage 3",
}


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = (sum((xs[i] - mx) ** 2 for i in range(n))) ** 0.5
    dy = (sum((ys[i] - my) ** 2 for i in range(n))) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def per_task_layer_delta(per_task, layer, baseline_layer=35):
    """Return list of (Δ_to_target, Δ_to_source) per task at given layer vs baseline."""
    out = []
    for t in per_task:
        ld_t_L = ld_s_L = ld_t_b = ld_s_b = None
        for lr in t["per_layer"]:
            if lr["layer"] == layer:
                ld_t_L = lr["ld_to_target"]
                ld_s_L = lr["ld_to_source"]
            if lr["layer"] == baseline_layer:
                ld_t_b = lr["ld_to_target"]
                ld_s_b = lr["ld_to_source"]
        if None not in (ld_t_L, ld_s_L, ld_t_b, ld_s_b):
            out.append((ld_t_L - ld_t_b, ld_s_L - ld_s_b))
    return out


def analyze_cell(cell_id: str, path: Path):
    data = json.load(open(path / "patching_continuation_results.json"))
    per_task = data["per_task"]
    n = len(per_task)
    print(f"\n## Cell {cell_id} — {CELL_LABELS[cell_id]} (N={n})")
    print()
    print("| Layer | mean Δ_to_tgt | mean Δ_to_src | Pearson(Δ_tgt, Δ_src) | Verdict |")
    print("|---|---:|---:|---:|---|")

    for L in [0, 5, 11, 17, 23, 29]:
        pairs = per_task_layer_delta(per_task, L)
        if not pairs:
            continue
        d_tgts, d_srcs = zip(*pairs)
        mean_dt = statistics.mean(d_tgts)
        mean_ds = statistics.mean(d_srcs)
        rho = pearson(list(d_tgts), list(d_srcs))
        # Verdict logic:
        # - L11/L17 mid-layer probe is most informative
        # - rho < -0.4 → strong transfer evidence (target↓ ↔ source↑ correlated)
        # - rho < -0.2 → modest transfer evidence
        # - rho ~ 0 → destruction without transfer
        # - rho > 0.2 → patched output drifts BOTH away from target AND away from source
        #   (catastrophic destruction)
        verdict = ""
        if L in (11, 17):
            if rho < -0.4:
                verdict = "🟢 TRANSFER (strong)"
            elif rho < -0.2:
                verdict = "🟡 TRANSFER (modest)"
            elif rho < 0.1:
                verdict = "🔴 DISRUPTION only"
            else:
                verdict = "⚫ CATASTROPHIC (away from both)"
        print(f"| L{L:2d} | {mean_dt:+.2f} | {mean_ds:+.2f} | {rho:+.3f} | {verdict} |")


def main():
    print("# Stage 2 Transfer-vs-Disruption Correlation Analysis")
    print()
    print("**Question**: when mid-layer patching disrupts target (Δ_to_target > 0),")
    print("does the patched output drift toward source (Δ_to_source < 0) per-task?")
    print()
    print("Pearson correlation across tasks: strong negative → genuine TRANSFER")
    print("(sub-population fusion); zero or positive → DISRUPTION without transfer.")
    print()
    print("Verdict thresholds at L11/L17 (mid-layer probe):")
    print("- 🟢 ρ < -0.4: strong transfer evidence")
    print("- 🟡 -0.4 ≤ ρ < -0.2: modest transfer evidence")
    print("- 🔴 -0.2 ≤ ρ < 0.1: disruption without transfer")
    print("- ⚫ ρ ≥ 0.1: catastrophic destruction (away from both target AND source)")
    print()
    for cid, path in CELLS.items():
        if not (path / "patching_continuation_results.json").exists():
            print(f"\n⚠️ Cell {cid} skipped (no data at {path})")
            continue
        analyze_cell(cid, path)


if __name__ == "__main__":
    main()
