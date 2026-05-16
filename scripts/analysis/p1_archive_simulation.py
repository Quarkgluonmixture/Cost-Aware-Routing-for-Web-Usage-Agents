#!/usr/bin/env python3
"""P1 (rule-based) router archive simulation — directional confidence only.

CRITICAL CAVEATS (read first):
- This is SANITY-CHECK ONLY, NOT preregistration lock substrate. Archive uses
  the same task IDs Phase 1a will use (cls 0-233 / red 0-209) and outcomes are
  pre-§107 + pre-§139.8 fix. Output numbers are *directional* — they cannot
  enter the paper as P1 SR claims. See proposals_v4 §A + Option C reframe.
- B0 only (no B1 / B2 archive). B0 capability profile may not transfer.
- cls phantom_prompt only 4 ep aborted -> cls = 5-mode evaluation.

Purpose:
    Audit whether P1 v3 design (`else -> phantom_som`) yields better archive SR
    than `else -> dom` (3-mode P1) on the same routing tree. Quantifies P1's
    actual phantom-mode reliance vs phantom-mode SR delivery.

Method:
    1. For each task, extract obs_1 features from step-0 record's state_digest:
       - dom_size = state_digest.text_length (== observation_dom.txt file size,
         empirically verified on task 0: 2674 bytes both)
       - dom_complexity = state_digest.dom_complexity (line count of AXTree)
       - intent = task config (external/visualwebarena/config_files/vwa/test_<site>/<tid>.json)
    2. Apply 5 router variants:
       - P1_v3_6mode: current design (search -> dom; complex -> som; else -> phantom_som)
       - P1_3mode:    same tree but else -> dom (phantom-free counterfactual)
       - always_dom:  baseline lower bound
       - always_som:  baseline single-mode
       - oracle:      perfect routing (upper bound)
    3. For each task t with routed mode m, look up archive outcome[t][m].
    4. Aggregate: SR, phantom routing share, phantom path SR vs DOM path SR.

Usage:
    python3 scripts/analysis/p1_archive_simulation.py [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PHASE1_ROOT = REPO / "results/visualwebarena/phase1"
VWA_CONFIG = REPO / "external/visualwebarena/config_files/vwa"


ARCHIVE_RUNS: dict[tuple[str, str, str], str] = {
    ("B0", "classifieds", "dom"):           "B0_3mode_classifieds_20260413/phase1_dom_router_0",
    ("B0", "classifieds", "som"):           "B0_3mode_classifieds_20260413/phase1_som_router_0",
    ("B0", "classifieds", "vision"):        "B0_3mode_classifieds_20260413/phase1_vision_router_0",
    ("B0", "classifieds", "phantom_text"):  "B0_phantom_text_classifieds_20260427/phase1_phantom_dom_router_0",
    ("B0", "classifieds", "phantom_prompt"):"B0_phantom_prompt_classifieds_20260504/phase1_phantom_prompt_router_0",
    ("B0", "classifieds", "phantom_som"):   "B0_phantom_som_classifieds_20260426/phase1_phantom_som_router_0",
    ("B0", "reddit", "dom"):                "B0_3mode_reddit_20260422/phase1_dom_router_0",
    ("B0", "reddit", "som"):                "B0_3mode_reddit_20260422/phase1_som_router_0",
    ("B0", "reddit", "vision"):             "B0_3mode_reddit_20260422/phase1_vision_router_0",
    ("B0", "reddit", "phantom_text"):       "B0_phantom_text_reddit_20260427/phase1_phantom_dom_router_0",
    ("B0", "reddit", "phantom_prompt"):     "B0_phantom_prompt_reddit_20260429/phase1_phantom_prompt_router_0",
    ("B0", "reddit", "phantom_som"):        "B0_phantom_som_reddit_20260428/phase1_phantom_som_router_0",
}

MODES = ["dom", "som", "vision", "phantom_text", "phantom_prompt", "phantom_som"]
CELLS = [("B0", "classifieds"), ("B0", "reddit")]

# P1 v3 rule (per proposals_v4)
SEARCH_RE = re.compile(r"\b(find|search|locate|how many|how much)\b", re.I)
DOM_SIZE_THRESHOLD = 12000
DOM_COMPLEXITY_THRESHOLD = 500


def load_cell_outcomes(baseline: str, site: str, min_ep: int = 50):
    """Load (task_id -> mode -> success) intersection across modes with >= min_ep eps."""
    per_mode: dict[str, dict[int, bool]] = {}
    retained: list[str] = []
    skipped: list[str] = []
    for mode in MODES:
        sub = ARCHIVE_RUNS.get((baseline, site, mode))
        if sub is None:
            skipped.append(f"{mode}: no archive entry"); continue
        ep_dir = PHASE1_ROOT / sub / "episodes"
        if not ep_dir.is_dir():
            skipped.append(f"{mode}: dir missing"); continue
        per_task: dict[int, bool] = {}
        for f in ep_dir.glob(f"{site}_task_*_summary_v2.json"):
            try:
                rec = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            tid = int(rec["task_id"])
            per_task[tid] = bool(rec.get("success", False))
        if len(per_task) < min_ep:
            skipped.append(f"{mode}: only {len(per_task)} ep (< {min_ep})"); continue
        per_mode[mode] = per_task; retained.append(mode)
    if not per_mode:
        return {}, retained, skipped
    common = set.intersection(*(set(per_mode[m]) for m in retained))
    matrix = {tid: {m: per_mode[m][tid] for m in retained} for tid in sorted(common)}
    return matrix, retained, skipped


def load_task_features(baseline: str, site: str, task_ids: list[int]) -> dict[int, dict]:
    """Extract obs_1 features per task from DOM-mode archive step-0 + task config.

    obs_1 features are mode-agnostic (entry page DOM is the same regardless of
    which mode the agent will use). We pull from DOM-mode archive because it
    has the cleanest AXTree representation matching P1's threshold calibration.
    """
    dom_sub = ARCHIVE_RUNS.get((baseline, site, "dom"))
    if dom_sub is None:
        return {}
    ep_dir = PHASE1_ROOT / dom_sub / "episodes"
    feats: dict[int, dict] = {}
    for tid in task_ids:
        steps_file = ep_dir / f"{site}_task_{tid}_steps_v2.jsonl"
        cfg_file = VWA_CONFIG / f"test_{site}" / f"{tid}.json"
        if not steps_file.exists() or not cfg_file.exists():
            continue
        try:
            with steps_file.open() as f:
                step0 = json.loads(f.readline())
            cfg = json.loads(cfg_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        sd = step0.get("state_digest", {})
        feats[tid] = {
            "dom_size": sd.get("text_length", 0),
            "dom_complexity": sd.get("dom_complexity", 0),
            "intent": cfg.get("intent", ""),
            "has_image": cfg.get("image") not in (None, "None", ""),
        }
    return feats


def route_p1_v3_6mode(feat: dict) -> str:
    if SEARCH_RE.search(feat["intent"]):
        return "dom"
    if feat["dom_size"] > DOM_SIZE_THRESHOLD or feat["dom_complexity"] > DOM_COMPLEXITY_THRESHOLD:
        return "som"
    return "phantom_som"


def route_p1_3mode(feat: dict) -> str:
    """Counterfactual: drop phantom_som, fallback to dom."""
    if SEARCH_RE.search(feat["intent"]):
        return "dom"
    if feat["dom_size"] > DOM_SIZE_THRESHOLD or feat["dom_complexity"] > DOM_COMPLEXITY_THRESHOLD:
        return "som"
    return "dom"


def route_always(mode: str):
    return lambda _: mode


def route_oracle(matrix_row: dict[str, bool]) -> str:
    """Pick any mode that succeeds; if none, return 'dom' as proxy."""
    for m, ok in matrix_row.items():
        if ok:
            return m
    return "dom"


def evaluate_router(matrix, features, route_fn, oracle: bool = False) -> dict:
    """Score router on archive. Returns SR + per-routed-mode breakdown."""
    sr_hits = 0
    n_eval = 0
    routed_counts: Counter = Counter()
    routed_sr: dict[str, list[int]] = {}
    missing_mode: list[int] = []
    for tid, row in matrix.items():
        if tid not in features and not oracle:
            continue
        if oracle:
            mode = route_oracle(row)
        else:
            mode = route_fn(features[tid])
        routed_counts[mode] += 1
        if mode not in row:
            missing_mode.append(tid)
            continue
        ok = int(row[mode])
        sr_hits += ok
        routed_sr.setdefault(mode, []).append(ok)
        n_eval += 1
    return {
        "sr_pct": 100.0 * sr_hits / n_eval if n_eval else float("nan"),
        "n_eval": n_eval,
        "routed_counts": dict(routed_counts),
        "per_mode_sr": {m: (100.0 * sum(v) / len(v), len(v)) for m, v in routed_sr.items()},
        "missing_mode_count": len(missing_mode),
    }


def render_cell(baseline: str, site: str, matrix, modes, skipped, features, results) -> str:
    lines = [
        f"### {baseline}_{site}",
        f"",
        f"- n tasks (intersection): **{len(matrix)}**, modes retained: `{modes}`",
        f"- features extracted: **{len(features)}** tasks (config + step-0 parsed)",
    ]
    if skipped:
        lines.append(f"- skipped modes: {skipped}")
    lines.append("")
    lines.append("**Router comparison**:")
    lines.append("")
    lines.append("| Router | SR (%) | N | Routed mode distribution |")
    lines.append("|---|---:|---:|---|")
    for name, r in results.items():
        dist_str = ", ".join(
            f"{m}={c} ({100*c/len(matrix):.1f}%)"
            for m, c in sorted(r["routed_counts"].items(), key=lambda x: -x[1])
        )
        lines.append(f"| {name} | {r['sr_pct']:.2f} | {r['n_eval']} | {dist_str} |")
    lines.append("")
    lines.append("**Per-routed-mode SR** (P1_v3_6mode breakdown — when router routes to X, what's the archive SR?):")
    lines.append("")
    lines.append("| Routed mode | N | SR (%) |")
    lines.append("|---|---:|---:|")
    for m, (sr, n) in sorted(results["P1_v3_6mode"]["per_mode_sr"].items(), key=lambda x: -x[1][1]):
        lines.append(f"| {m} | {n} | {sr:.2f} |")
    lines.append("")
    p1_v3 = results["P1_v3_6mode"]
    p1_3m = results["P1_3mode"]
    delta = p1_v3["sr_pct"] - p1_3m["sr_pct"]
    phantom_share = 100 * p1_v3["routed_counts"].get("phantom_som", 0) / len(matrix)
    lines.append(f"**P1 phantom dependency**: {phantom_share:.1f}% of tasks routed to phantom_som by v3.")
    lines.append(f"**P1 v3 (6-mode) vs P1 3-mode delta**: {delta:+.2f}pp ({'phantom HELPS' if delta > 0 else 'phantom HURTS' if delta < 0 else 'phantom NEUTRAL'})")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", default="docs/checkpoints/router/p1_archive_simulation_2026-05-16.md")
    ap.add_argument("--out-json", default="docs/checkpoints/router/p1_archive_simulation_2026-05-16.json")
    args = ap.parse_args()

    report: dict = {"run_date": datetime.utcnow().isoformat() + "Z", "cells": {}}
    md_blocks: list[str] = [
        "# P1 (rule-based) router archive simulation — SANITY-CHECK ONLY",
        "",
        "> ⚠️ **NOT preregistration lock substrate.** Archive uses same task IDs Phase 1a will use; outcomes are pre-§107 + pre-§139.8 fix. Numbers below are *directional* — they answer 'does P1 v3 design have honest phantom-dependency signal?', NOT 'P1 SR is X%'. Real P1 SR claim must come from Phase 1a fresh-data 5-fold CV. See `proposals_v4.md` §A for methodology reframe.",
        "",
        f"Run date: `{report['run_date']}`",
        "",
        "## Method recap",
        "",
        "- **P1 v3 rule** (from `proposals_v4.md` decide_p1_v3):",
        "  - if `intent` matches search regex (`find|search|locate|how many|how much`) → `dom`",
        "  - elif `dom_size > 12000` OR `dom_complexity > 500` → `som`",
        "  - else → `phantom_som`",
        "- **P1 3-mode counterfactual**: same tree but else → `dom` (drops phantom dependency)",
        "- Features: step-0 `state_digest.{text_length, dom_complexity}` + task config `intent`",
        "- Outcome: archive `success` per (task_id, mode)",
        "",
        "## Per-cell results",
        "",
    ]

    for baseline, site in CELLS:
        matrix, modes, skipped = load_cell_outcomes(baseline, site)
        if not matrix:
            md_blocks.append(f"### {baseline}_{site}\n\n_(no archive data)_\n"); continue
        features = load_task_features(baseline, site, list(matrix.keys()))
        # restrict matrix to tasks with features
        m2 = {tid: matrix[tid] for tid in matrix if tid in features}
        results = {
            "P1_v3_6mode": evaluate_router(m2, features, route_p1_v3_6mode),
            "P1_3mode":    evaluate_router(m2, features, route_p1_3mode),
            "always_dom":  evaluate_router(m2, features, route_always("dom")),
            "always_som":  evaluate_router(m2, features, route_always("som")) if "som" in modes else {"sr_pct": float("nan"), "n_eval": 0, "routed_counts": {}, "per_mode_sr": {}, "missing_mode_count": 0},
            "always_phantom_som": evaluate_router(m2, features, route_always("phantom_som")) if "phantom_som" in modes else {"sr_pct": float("nan"), "n_eval": 0, "routed_counts": {}, "per_mode_sr": {}, "missing_mode_count": 0},
            "oracle": evaluate_router(m2, features, None, oracle=True),
        }
        report["cells"][f"{baseline}_{site}"] = {
            "n_tasks": len(m2), "modes_retained": modes, "modes_skipped": skipped,
            "results": results,
        }
        md_blocks.append(render_cell(baseline, site, m2, modes, skipped, features, results))

    md_blocks.append("## Cross-cell summary\n")
    md_blocks.append("| Cell | always_dom | P1_3mode | P1_v3_6mode | Δ (6-3) | always_phantom_som | oracle |")
    md_blocks.append("|---|---:|---:|---:|---:|---:|---:|")
    for cell, info in report["cells"].items():
        r = info["results"]
        delta = r["P1_v3_6mode"]["sr_pct"] - r["P1_3mode"]["sr_pct"]
        md_blocks.append(
            f"| {cell} | {r['always_dom']['sr_pct']:.2f} | {r['P1_3mode']['sr_pct']:.2f} | "
            f"{r['P1_v3_6mode']['sr_pct']:.2f} | {delta:+.2f} | "
            f"{r['always_phantom_som']['sr_pct']:.2f} | {r['oracle']['sr_pct']:.2f} |"
        )
    md_blocks.append("")
    md_blocks.append("## Interpretation guide\n")
    md_blocks.append("- If P1_v3_6mode > P1_3mode by ≥ 1pp → phantom dependency adds archive SR (directional confidence that v3 design is sensible)")
    md_blocks.append("- If P1_v3_6mode ≈ P1_3mode (within ±0.5pp) → phantom path SR ≈ DOM path SR on routed-tasks; phantom value is cost not SR (paper §6 should frame Pareto, not lift)")
    md_blocks.append("- If P1_v3_6mode < P1_3mode by > 1pp → phantom_som hurts on the 'simple non-search' task slice; v3 rule needs revision (likely v5: tighter complexity threshold or alternative else-branch)")
    md_blocks.append("")
    md_blocks.append("**Reminder**: this is correlated-population evidence on B0 only. Phase 1a 6-cell fresh data is the paper-grade test.")
    md_blocks.append("")

    out_md = REPO / args.out_md
    out_json = REPO / args.out_json
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_blocks))
    out_json.write_text(json.dumps(report, indent=2))
    print(f"Wrote: {out_md}\nWrote: {out_json}")
    # also print summary table to stdout
    print("\n" + "\n".join(md_blocks[-15:]))


if __name__ == "__main__":
    main()
