#!/usr/bin/env python3
"""H1 per-task fragility check on existing format-variation data.

For each (task_id, step) pair, compute per-layer cosine gap variant↔som
and find peak layer per variant. Aggregate: do marks-like variants peak
L17+ AND AXTree-DOM peak L4 PER INDIVIDUAL TASK?

If ≥80% of (task, step) pairs show this dichotomy → robust, not driven
by few tasks. If <60% → average artifact.

Output: docs/checkpoints/mechanism/results/h1_per_task_fragility.md
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "results/mechanistic/stage4_format_variation_b1_cls/hidden_states.npz"
OUT_MD = ROOT / "docs/checkpoints/mechanism/results/h1_per_task_fragility.md"

MARKS_LIKE = ["som_standard", "browser_use_at", "appagent_id", "tarsier_typed",
               "plain_numbered", "xml_tagged", "hash_id_control"]  # 7 flat-list (incl no-int control)
CONTROL_NO_LIST = ["plain_sentence"]
AXTREE = "dom"
BASELINE = "som"


def cosine_gap(a, b):
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    d = np.load(NPZ, allow_pickle=True)
    H = d["hidden_states"]
    ml = d["mode_labels_str"]
    tids = d["task_ids"]
    steps = d["step_indices"]
    n_layers = H.shape[1]

    # Per (task, step), gather per-variant hidden state
    task_step_data = defaultdict(dict)
    for i, (m, tid, st) in enumerate(zip(ml.tolist(), tids.tolist(), steps.tolist())):
        task_step_data[(tid, st)][m] = H[i]

    # Per (task, step), per variant: compute peak layer of variant↔som cosine gap
    per_task_peaks = defaultdict(dict)
    for key, hs in task_step_data.items():
        if BASELINE not in hs:
            continue
        som_h = hs[BASELINE]
        for v in MARKS_LIKE + CONTROL_NO_LIST + [AXTREE]:
            if v not in hs:
                continue
            curve = np.array([cosine_gap(hs[v][L], som_h[L]) for L in range(n_layers)])
            per_task_peaks[key][v] = int(np.argmax(curve))

    # Aggregate: per-task dichotomy verdict
    n_tasks = len(per_task_peaks)
    print(f"Loaded {n_tasks} (task, step) pairs")

    # Verdict 1: AXTree-DOM peak ≤ L10 (early peak) per task?
    dom_early = sum(1 for v in per_task_peaks.values() if v.get(AXTREE, 99) <= 10)
    # Verdict 2: ≥ 4/7 marks-like variants peak ≥ L20 (late) per task?
    marks_late = 0
    for v in per_task_peaks.values():
        n_late = sum(1 for m in MARKS_LIKE if v.get(m, 0) >= 20)
        if n_late >= 4:
            marks_late += 1
    # Verdict 3: BOTH conditions per task (strongest dichotomy)
    both_strict = 0
    for v in per_task_peaks.values():
        cond1 = v.get(AXTREE, 99) <= 10
        n_late = sum(1 for m in MARKS_LIKE if v.get(m, 0) >= 20)
        cond2 = n_late >= 4
        if cond1 and cond2:
            both_strict += 1

    # Per-task peak-layer distribution for AXTree vs flat-list
    dom_peaks = [v.get(AXTREE, -1) for v in per_task_peaks.values() if AXTREE in v]
    marks_avg_peaks = []
    for v in per_task_peaks.values():
        marks = [v[m] for m in MARKS_LIKE if m in v]
        if marks:
            marks_avg_peaks.append(np.mean(marks))

    write_md(n_tasks, dom_early, marks_late, both_strict,
              dom_peaks, marks_avg_peaks, per_task_peaks)


def write_md(n, dom_early, marks_late, both, dom_peaks, marks_avg_peaks, per_task_peaks):
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# H1 per-task fragility check",
        "",
        f"**Sample**: {n} (task, step) pairs from format_variation_b1_cls",
        "",
        "## Aggregate verdict per individual (task, step) pair",
        "",
        f"- **AXTree-DOM peak ≤ L10** (early image-axis peak): {dom_early}/{n} = **{100*dom_early/n:.0f}%**",
        f"- **≥4/7 marks-like variants peak ≥ L20** (late image-axis peak): {marks_late}/{n} = **{100*marks_late/n:.0f}%**",
        f"- **BOTH conditions** (strict dichotomy per task): {both}/{n} = **{100*both/n:.0f}%**",
        "",
        "## Per-task peak-layer distribution",
        "",
        f"AXTree-DOM peak layer: mean = **{np.mean(dom_peaks):.1f}**, std = {np.std(dom_peaks):.1f}, range L{min(dom_peaks):02d}-L{max(dom_peaks):02d}",
        f"Marks-like (avg across 7) peak layer: mean = **{np.mean(marks_avg_peaks):.1f}**, std = {np.std(marks_avg_peaks):.1f}",
        f"**Separation** = marks - dom = **{np.mean(marks_avg_peaks) - np.mean(dom_peaks):+.1f} layers**",
        "",
        "## Verdict",
        "",
    ]
    if both / n >= 0.8:
        lines.append("→ **H1 ROBUST per-task**: dichotomy holds for ≥80% of individual tasks. Not driven by few outliers. Paper-grade defensible.")
    elif both / n >= 0.6:
        lines.append("→ **H1 MODERATE per-task**: dichotomy holds for 60-80% of tasks. Honest framing needed; reviewer may probe per-task heterogeneity.")
    else:
        lines.append("→ **H1 WEAK per-task**: dichotomy is averaged effect, not per-task universal. Paper §5 framing must acknowledge per-task variability.")
    lines.append("")

    # Detail table: top 5 and bottom 5 (task, step) pairs by separation
    sep_per_task = []
    for key, peaks in per_task_peaks.items():
        if AXTREE not in peaks:
            continue
        marks_avg = np.mean([peaks[m] for m in MARKS_LIKE if m in peaks])
        sep = marks_avg - peaks[AXTREE]
        sep_per_task.append((key, sep, peaks[AXTREE], marks_avg))
    sep_per_task.sort(key=lambda x: x[1], reverse=True)

    lines.append("## Top 5 dichotomy-confirming (task, step) pairs (largest separation)")
    lines.append("")
    lines.append("| Task ID | Step | AXTree peak | Marks avg peak | Separation |")
    lines.append("|---|---|---|---|---|")
    for (tid, st), sep, dom_p, marks_avg in sep_per_task[:5]:
        lines.append(f"| {tid} | {st} | L{dom_p:02d} | L{marks_avg:.1f} | **{sep:+.1f}** |")

    lines.append("")
    lines.append("## Bottom 5 (task, step) pairs (smallest / inverse separation)")
    lines.append("")
    lines.append("| Task ID | Step | AXTree peak | Marks avg peak | Separation |")
    lines.append("|---|---|---|---|---|")
    for (tid, st), sep, dom_p, marks_avg in sep_per_task[-5:]:
        lines.append(f"| {tid} | {st} | L{dom_p:02d} | L{marks_avg:.1f} | {sep:+.1f} |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"summary → {OUT_MD}")
    print(f"\n=== KEY VERDICT ===")
    print(f"  AXTree-DOM peak ≤ L10: {100*dom_early/n:.0f}%")
    print(f"  ≥4/7 marks-like peak ≥ L20: {100*marks_late/n:.0f}%")
    print(f"  BOTH (strict dichotomy per task): {100*both/n:.0f}%")
    print(f"  Mean separation: {np.mean(marks_avg_peaks) - np.mean(dom_peaks):+.1f} layers")


if __name__ == "__main__":
    main()
