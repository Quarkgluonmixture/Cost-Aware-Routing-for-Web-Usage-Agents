#!/usr/bin/env python3
"""L2 partial-trajectory AUROC sanity check — F1 defuse.

v5 §3.1 cites fig0g `ep_mean_verbalized` AUROC (≥ 0.7 in 13/19 cells) as anchor
for L2 runtime trigger threshold. But fig0g AUROC is computed on *full-episode*
mean (analyze_confidence_calibration.py:187: np.mean(verb_vals) over all steps).
L2 fires at step ≥ 3 with *partial* trajectory.

This script computes prefix-k AUROC for k ∈ {1, 2, 3, 5, 8, episode_full} and
reports the gap. If step-3 AUROC drops far below the cited ≥ 0.7, v5 L2 verbose
anchor is category error.

Per /stress F1, this is the "1 thing to fix tonight" defuse.

Usage:
    python3 scripts/analysis/l2_partial_trajectory_auroc.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_archive_simulation import ARCHIVE_RUNS

REPO = Path(__file__).resolve().parents[2]
PHASE1_ROOT = REPO / "results/visualwebarena/phase1"

PREFIX_K = [1, 2, 3, 5, 8, 9999]  # 9999 = episode full


def extract_episode_verbalized(steps_file: Path) -> list[float]:
    """Read verbalized confidence per step from steps_v2.jsonl."""
    vals = []
    if not steps_file.exists():
        return vals
    try:
        with steps_file.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                conf = rec.get("confidence", {})
                if isinstance(conf, dict):
                    v = conf.get("verbalized")
                    if v is not None:
                        vals.append(float(v))
    except OSError:
        pass
    return vals


def cell_partial_auroc(baseline: str, site: str, mode: str) -> dict:
    """For each task in cell, compute prefix-k mean verbalized + episode outcome.

    Then bootstrap AUROC vs outcome for each k.
    """
    sub = ARCHIVE_RUNS.get((baseline, site, mode))
    if sub is None:
        return {"error": "no archive entry"}
    ep_dir = PHASE1_ROOT / sub / "episodes"
    if not ep_dir.is_dir():
        return {"error": "dir missing"}

    # Collect (mean_prefix_k_verbalized, outcome) per task
    per_k_verb: dict[int, list[float]] = {k: [] for k in PREFIX_K}
    outcomes: list[int] = []
    n_tasks = 0
    n_no_verb = 0

    for summary_f in ep_dir.glob(f"{site}_task_*_summary_v2.json"):
        try:
            summary = json.loads(summary_f.read_text())
        except json.JSONDecodeError:
            continue
        tid = int(summary["task_id"])
        ok = int(summary.get("success", False))
        steps_f = ep_dir / f"{site}_task_{tid}_steps_v2.jsonl"
        verb_vals = extract_episode_verbalized(steps_f)
        if not verb_vals:
            n_no_verb += 1
            continue
        n_tasks += 1
        for k in PREFIX_K:
            prefix = verb_vals[:k] if k != 9999 else verb_vals
            if not prefix:
                # if k=2 and episode only has 1 step, skip
                per_k_verb[k].append(np.nan)
            else:
                per_k_verb[k].append(float(np.mean(prefix)))
        outcomes.append(ok)

    # AUROC per k (drop NaN entries)
    out = {"n_tasks": n_tasks, "n_no_verb": n_no_verb, "auroc": {}, "n_valid": {}}
    if n_tasks == 0:
        return out
    outcomes_arr = np.array(outcomes)
    if outcomes_arr.sum() == 0 or outcomes_arr.sum() == n_tasks:
        out["error"] = "all-same-outcome (degenerate)"
        return out
    for k in PREFIX_K:
        v = np.array(per_k_verb[k])
        mask = ~np.isnan(v)
        n_valid = int(mask.sum())
        out["n_valid"][k] = n_valid
        if n_valid < 10 or outcomes_arr[mask].sum() in (0, n_valid):
            out["auroc"][k] = None
            continue
        try:
            # AUROC: lower verbalized → higher fail prob → use 1-verb as score for "fail" positive
            auroc = roc_auc_score(1 - outcomes_arr[mask], 1.0 - v[mask])
            out["auroc"][k] = float(auroc)
        except ValueError:
            out["auroc"][k] = None
    return out


def main():
    cells = []
    # B0 archive (where verbalized exists)
    for site in ["classifieds", "reddit"]:
        for mode in ["dom", "som", "vision", "phantom_text", "phantom_prompt", "phantom_som"]:
            if (("B0", site, mode)) in ARCHIVE_RUNS:
                cells.append(("B0", site, mode))

    report = {"cells": {}, "summary": {"k_below_0.65": 0, "k_above_0.7": 0}}
    print(f"Running partial-trajectory AUROC on {len(cells)} cells...")
    md_rows = ["# L2 partial-trajectory AUROC — F1 defuse"]
    md_rows.append("")
    md_rows.append("> v5 §3.1 claims fig0g `ep_mean_verbalized` AUROC ≥ 0.7 anchors L2 runtime trigger. But fig0g uses *full-episode* mean. L2 fires at step ≥ 3 with *partial* trajectory. This script computes prefix-k AUROC for k ∈ {1, 2, 3, 5, 8, full}.")
    md_rows.append("")
    md_rows.append("**Hypothesis**: if step-3 AUROC ≥ 0.65 → v5 anchor partially salvaged (with calibrated threshold). If step-3 AUROC < 0.6 → v5 L2 verbose trigger 不能 paper-grade 落地.")
    md_rows.append("")
    md_rows.append("## AUROC by prefix-k per cell")
    md_rows.append("")
    md_rows.append("| Cell | n_tasks | n_no_verb | k=1 | k=2 | k=3 | k=5 | k=8 | k=full | k=3 vs full Δ |")
    md_rows.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for baseline, site, mode in cells:
        result = cell_partial_auroc(baseline, site, mode)
        cell_id = f"{baseline}_{site}_{mode}"
        report["cells"][cell_id] = result
        if "error" in result and result.get("n_tasks", 0) == 0:
            md_rows.append(f"| {cell_id} | — | — | error: {result['error']} | | | | | | |")
            continue
        n_tasks = result.get("n_tasks", 0)
        n_no_verb = result.get("n_no_verb", 0)
        auroc = result.get("auroc", {})
        a1 = auroc.get(1)
        a2 = auroc.get(2)
        a3 = auroc.get(3)
        a5 = auroc.get(5)
        a8 = auroc.get(8)
        afull = auroc.get(9999)
        delta = (a3 - afull) if (a3 is not None and afull is not None) else None
        def fmt(x): return f"{x:.3f}" if x is not None else "n/a"
        md_rows.append(f"| {cell_id} | {n_tasks} | {n_no_verb} | {fmt(a1)} | {fmt(a2)} | {fmt(a3)} | {fmt(a5)} | {fmt(a8)} | {fmt(afull)} | {fmt(delta)} |")
        # update summary
        if a3 is not None:
            if a3 < 0.65: report["summary"]["k_below_0.65"] += 1
            if afull is not None and afull >= 0.7: report["summary"]["k_above_0.7"] += 1

    md_rows.append("")
    md_rows.append("## Summary — verdict on v5 §3.1 L2 trigger anchor")
    md_rows.append("")
    md_rows.append(f"- Cells where k=3 AUROC < 0.65 (anchor not viable): **{report['summary']['k_below_0.65']}**")
    md_rows.append(f"- Cells where full-episode AUROC ≥ 0.7 (v5 cited threshold): {report['summary']['k_above_0.7']}")
    md_rows.append("")
    md_rows.append("**Verdict**: if `k=3 AUROC < 0.65` in majority of cells → v5 §3.1 L2 verbose AUROC anchor is category error. L2 falls back to cycle-only triggers (max_repeat / url_revisit) which are more directly computable at runtime.")

    out_md = REPO / "docs/checkpoints/router/l2_partial_traj_auroc_2026-05-16.md"
    out_json = REPO / "docs/checkpoints/router/l2_partial_traj_auroc_2026-05-16.json"
    out_md.write_text("\n".join(md_rows))
    out_json.write_text(json.dumps(report, indent=2))
    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_json}")
    print()
    print("\n".join(md_rows[-12:]))


if __name__ == "__main__":
    main()
