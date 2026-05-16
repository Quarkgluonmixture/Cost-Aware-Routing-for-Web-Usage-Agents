#!/usr/bin/env python3
"""Train L1 learned router per cell from Pass-1 baseline data.

Phase 1a v7 Pass-2 prerequisite. Reads Pass-1 (`router_on=False`) baseline
per-task outcomes for a (baseline, site) cell, derives per-task oracle-best
mode label, trains LR with balanced class weight + in-fold StandardScaler
Pipeline, dumps pickle to `results/phantom_paper/l1_router/<baseline>_<site>_lr.pkl`.

Feature schema MUST match `p79/policies/learned_router.py:extract_task_features`
(8-dim: site / has_image / 4 intent regex / intent_tok_count / axtree_elements).

Usage:
    # Train one cell
    python3 scripts/analysis/train_l1_router.py --baseline B0 --site classifieds

    # Train all 6 Phase 1a cells
    python3 scripts/analysis/train_l1_router.py --all

    # Override Pass-1 run discovery
    python3 scripts/analysis/train_l1_router.py --baseline B0 --site classifieds \\
        --pass1-run-glob 'results/visualwebarena/phase1/B0_*_classifieds_*'
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
PHASE1_ROOT = REPO / "results/visualwebarena/phase1"
VWA_CONFIG = REPO / "external/visualwebarena/config_files/vwa"
OUT_DIR = REPO / "results/phantom_paper/l1_router"

MODES = ["dom", "som", "vision", "phantom_text", "phantom_prompt", "phantom_som"]

CELLS = [
    ("B0", "classifieds"), ("B0", "reddit"),
    ("B1", "classifieds"), ("B1", "reddit"),
    ("B2", "classifieds"), ("B2", "reddit"),
]


def find_pass1_runs(baseline: str, site: str, run_glob: str | None = None) -> list[Path]:
    """Discover Pass-1 baseline run directories for this (baseline, site) cell.

    Default heuristic: directory name starts with `<baseline>_` and contains
    `_<site>_`. Excludes directories with `router_learned_` suffix (Pass-2).
    """
    if run_glob:
        return sorted(Path(REPO).glob(run_glob))
    candidates = []
    for d in PHASE1_ROOT.glob(f"{baseline}_*_{site}_*"):
        if not d.is_dir():
            continue
        if "router_learned" in d.name:
            continue
        candidates.append(d)
    return sorted(candidates)


def collect_per_task_outcomes(run_dirs: list[Path], site: str) -> dict[int, dict[str, bool]]:
    """Collect per-task per-mode success from condition_summary_v2.json across runs.

    Returns: {task_id: {mode: success_bool}}.
    """
    matrix: dict[int, dict[str, bool]] = {}
    for run_dir in run_dirs:
        for cond_dir in run_dir.iterdir():
            if not cond_dir.is_dir():
                continue
            # cond_dir like phase1_dom_router_0 / phase1_phantom_som_router_0
            cond_id = cond_dir.name
            if cond_id == "phase1_learned_router":
                continue  # skip Pass-2 router conditions
            # Extract mode from condition_id (phase1_<mode>_router_0)
            parts = cond_id.split("_")
            if len(parts) < 3 or parts[0] != "phase1":
                continue
            mode_tokens = parts[1:-2]
            mode = "_".join(mode_tokens)
            # Normalize legacy "phantom_dom" → "phantom_text" (CLAUDE.md note)
            if mode == "phantom_dom":
                mode = "phantom_text"
            if mode not in MODES:
                continue
            ep_dir = cond_dir / "episodes"
            if not ep_dir.is_dir():
                continue
            for summary_f in ep_dir.glob(f"{site}_task_*_summary_v2.json"):
                try:
                    rec = json.loads(summary_f.read_text())
                except json.JSONDecodeError:
                    continue
                tid = int(rec["task_id"])
                success = bool(rec.get("success", False))
                matrix.setdefault(tid, {})[mode] = success
    return matrix


def extract_features_per_task(
    task_ids: list[int],
    site: str,
    pass1_run_dir: Path,
) -> dict[int, dict]:
    """Read task config (intent / image) + step-0 state_digest (axtree_element_count)
    to build feature dict per task.

    `pass1_run_dir` is used to locate step-0 JSONL for axtree_element_count;
    we use the DOM condition's step-0 (mode-agnostic page).
    """
    feats: dict[int, dict] = {}
    # Pick first dom condition for step-0 features (any baseline mode would do —
    # entry-page DOM is mode-agnostic per p1_archive_simulation.py:78-95).
    dom_cond_dir = pass1_run_dir / "phase1_dom_router_0"
    if not dom_cond_dir.is_dir():
        # Try any condition that's available
        cands = [d for d in pass1_run_dir.iterdir() if d.is_dir() and d.name.startswith("phase1_")]
        if not cands:
            return feats
        dom_cond_dir = cands[0]
    ep_dir = dom_cond_dir / "episodes"
    for tid in task_ids:
        cfg_file = VWA_CONFIG / f"test_{site}" / f"{tid}.json"
        steps_file = ep_dir / f"{site}_task_{tid}_steps_v2.jsonl"
        if not cfg_file.exists():
            continue
        try:
            cfg = json.loads(cfg_file.read_text())
        except json.JSONDecodeError:
            continue
        intent = cfg.get("intent", "")
        has_image = cfg.get("image") not in (None, "None", "", [])
        # Read step-0 axtree count from steps JSONL
        axtree_element_count = 0
        if steps_file.exists():
            try:
                with steps_file.open() as f:
                    step0 = json.loads(f.readline())
                sd = step0.get("state_digest", {})
                axtree_element_count = int(sd.get("dom_complexity", 0))
            except (json.JSONDecodeError, OSError):
                pass
        feats[tid] = {
            "intent": intent,
            "has_image": has_image,
            "intent_tok_count": len(intent.split()),
            "axtree_element_count": axtree_element_count,
        }
    return feats


def derive_oracle_label(outcomes: dict[str, bool]) -> str:
    """Pick oracle-best mode per task.

    Tie-break: MODES priority order (dom > som > vision > phantom_text >
    phantom_prompt > phantom_som). Matches l1_archive_simulation.py:63-69.

    If no mode succeeded, label = "dom" (majority-class fallback, mirrors archive sim).
    """
    for m in MODES:
        if outcomes.get(m, False):
            return m
    return "dom"


def build_design_matrix(
    matrix: dict[int, dict[str, bool]],
    features: dict[int, dict],
    site: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build (X 8-dim, y_label, task_ids) for LR training.

    Feature columns match p79/policies/learned_router.py:65-89.
    """
    from p79.policies.learned_router import (
        COLOR_RE, SEARCH_RE, COMPARE_RE, NAV_RE,
    )
    rows = []
    labels = []
    task_ids = []
    for tid in sorted(matrix.keys()):
        if tid not in features:
            continue
        feat = features[tid]
        intent = feat["intent"]
        row = [
            1.0 if site == "classifieds" else 0.0,
            1.0 if feat["has_image"] else 0.0,
            1.0 if COLOR_RE.search(intent) else 0.0,
            1.0 if SEARCH_RE.search(intent) else 0.0,
            1.0 if COMPARE_RE.search(intent) else 0.0,
            1.0 if NAV_RE.search(intent) else 0.0,
            float(feat["intent_tok_count"]),
            float(feat["axtree_element_count"]),
        ]
        rows.append(row)
        labels.append(derive_oracle_label(matrix[tid]))
        task_ids.append(tid)
    return np.array(rows, dtype=float), np.array(labels), task_ids


def train_and_dump(baseline: str, site: str, run_glob: str | None = None) -> dict:
    """Train LR for one (baseline, site) cell + dump pickle.

    Returns metadata dict with train stats.
    """
    runs = find_pass1_runs(baseline, site, run_glob)
    if not runs:
        return {"error": f"no Pass-1 runs found for {baseline} {site}"}
    print(f"[{baseline} {site}] Found {len(runs)} Pass-1 run dir(s)")
    for r in runs:
        print(f"  - {r.name}")

    # Collect outcomes across all run dirs
    matrix = {}
    for r in runs:
        sub = collect_per_task_outcomes([r], site)
        for tid, modes in sub.items():
            matrix.setdefault(tid, {}).update(modes)
    if not matrix:
        return {"error": f"no episode summaries found for {baseline} {site}"}
    print(f"[{baseline} {site}] Collected {len(matrix)} tasks with outcomes")

    # Extract features (use first run dir for step-0 features)
    features = extract_features_per_task(list(matrix.keys()), site, runs[0])
    print(f"[{baseline} {site}] Extracted features for {len(features)} tasks")

    # Build design matrix
    X, y, task_ids = build_design_matrix(matrix, features, site)
    if len(X) == 0:
        return {"error": f"empty design matrix after intersection for {baseline} {site}"}
    label_dist = dict(Counter(y))
    print(f"[{baseline} {site}] Design matrix: X.shape={X.shape}, labels={label_dist}")

    # Train LR Pipeline with in-fold StandardScaler
    preprocessor = ColumnTransformer(
        transformers=[("scale_numeric", StandardScaler(), [6, 7])],
        remainder="passthrough",
    )
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])
    pipeline.fit(X, y)

    # Predictions distribution sanity (in-sample, just for transparency)
    in_sample_preds = pipeline.predict(X)
    pred_dist = dict(Counter(in_sample_preds))
    print(f"[{baseline} {site}] In-sample prediction dist: {pred_dist}")

    # Dump pickle
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{baseline}_{site}_lr.pkl"
    with out_path.open("wb") as f:
        pickle.dump(pipeline, f)
    print(f"[{baseline} {site}] Wrote: {out_path}")

    # Also write companion metadata JSON for paper-grade provenance
    meta = {
        "baseline": baseline,
        "site": site,
        "n_tasks": len(X),
        "pass1_run_dirs": [r.name for r in runs],
        "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
        "in_sample_pred_distribution": {str(k): int(v) for k, v in pred_dist.items()},
        "feature_columns": [
            "site_cls", "has_image",
            "color_intent", "search_intent", "compare_intent", "nav_intent",
            "intent_tok_count", "axtree_element_count",
        ],
        "modes_in_label_set": list(MODES),
        "class_weight": "balanced",
        "scaler_columns": [6, 7],
    }
    meta_path = OUT_DIR / f"{baseline}_{site}_lr_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"path": str(out_path), "meta": meta}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", help="B0 | B1 | B2 (omit if --all)")
    ap.add_argument("--site", help="classifieds | reddit (omit if --all)")
    ap.add_argument("--all", action="store_true", help="train all 6 Phase 1a cells")
    ap.add_argument("--pass1-run-glob", default=None,
                    help="Override Pass-1 run discovery glob (per single-cell)")
    args = ap.parse_args()

    if args.all:
        targets = CELLS
    elif args.baseline and args.site:
        targets = [(args.baseline, args.site)]
    else:
        ap.error("Either --all or (--baseline + --site) required")

    results = {}
    n_ok = 0
    n_fail = 0
    for baseline, site in targets:
        print(f"\n=== Training {baseline} × {site} ===")
        try:
            res = train_and_dump(baseline, site, args.pass1_run_glob)
            results[f"{baseline}_{site}"] = res
            if "error" in res:
                n_fail += 1
                print(f"[{baseline} {site}] FAIL: {res['error']}")
            else:
                n_ok += 1
        except Exception as e:
            n_fail += 1
            results[f"{baseline}_{site}"] = {"error": str(e)}
            print(f"[{baseline} {site}] EXCEPTION: {e}")

    print(f"\n=== Summary: {n_ok}/{n_ok + n_fail} cells trained ===")
    if n_fail > 0:
        print(f"  {n_fail} cell(s) failed; check Pass-1 run completion")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
