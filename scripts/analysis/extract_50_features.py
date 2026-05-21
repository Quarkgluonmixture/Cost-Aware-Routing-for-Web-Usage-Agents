#!/usr/bin/env python3
"""Stage 1: Deterministic raw feature extraction for L1 learned router.

A2.5 Chunk A (B-996 expanded, /stress 2026-05-18). Reads Pass-1 baseline outcomes
+ task configs + step-0 records, produces:

- 5 numeric features (step-0 mode-agnostic + task config): dom_complexity, text_length,
  tokens_input_text, intent_token_count, reasoning_difficulty.
- 15 task binary features: has_reference_image + 14 intent regex matches.
- Raw intent text (held for Stage 2 fold-local TF-IDF, NOT vectorized here).

site + capability_tier are NOT included in the candidate pool — they are cell-constant
within a per-cell pickle architecture (Q1=C + (E''')). Cell identity is implicit via
runtime pickle selection per (baseline, site).

Output: results/phantom_paper/l1_router/raw_features_phase1a.npz
        + companion JSON with task_ids + cell_ids + filter logs.

Used by: scripts/analysis/train_l1_router_with_mi.py (Stage 2 fold-local TF-IDF + global
pooled MI), then by refactored train_l1_router.py (Stage 3 per-cell × per-fold LR).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

from p79.policies.router_features import (
    INTENT_REGEX,
    MODES,
    compute_intent_binaries,
    derive_oracle_label,
    difficulty_to_int,
)

REPO = Path(__file__).resolve().parents[2]
PHASE1_ROOT = REPO / "results/visualwebarena/phase1"
VWA_CONFIG = REPO / "external/visualwebarena/config_files/vwa"
OUT_DIR = REPO / "results/phantom_paper/l1_router"

# MODES / INTENT_REGEX / compute_intent_binaries / derive_oracle_label now live in
# p79.policies.router_features (router /stress B-1806/B-1807): single source of truth
# shared with the serve-time predictor + archive sim — was copy-pasted and drifting.

# All Phase 1a cells (B0/B1/B2 × cls/red). B2 cells may be empty pre-Phase-1a fire.
CELLS = [
    ("B0", "classifieds"), ("B0", "reddit"),
    ("B1", "classifieds"), ("B1", "reddit"),
    ("B2", "classifieds"), ("B2", "reddit"),
]

# INTENT_REGEX moved to p79.policies.router_features (B-1807). has_ref_image is the
# 15th binary, computed from the task config (not a regex).

# Schema version for downstream consumers
FEATURE_SCHEMA_VERSION = "2026-05-18-a2.5-chunk-a"


def find_pass1_runs(baseline: str, site: str) -> list[Path]:
    """Discover Pass-1 baseline run dirs for a (baseline, site) cell.

    Excludes Pass-2 router conditions (`router_learned_` suffix).
    """
    candidates = []
    if not PHASE1_ROOT.is_dir():
        return candidates
    for d in PHASE1_ROOT.glob(f"{baseline}_*_{site}_*"):
        if not d.is_dir():
            continue
        if "router_learned" in d.name:
            continue
        candidates.append(d)
    return sorted(candidates)


def collect_per_task_outcomes(run_dirs: list[Path], site: str) -> dict[int, dict[str, bool]]:
    """Per-task per-mode success from condition_summary_v2 episodes.

    Mirrors train_l1_router.py:70-106 logic. Returns {task_id: {mode: success_bool}}.
    Legacy phantom_dom → phantom_text normalization (CLAUDE.md note).
    """
    matrix: dict[int, dict[str, bool]] = {}
    for run_dir in run_dirs:
        for cond_dir in run_dir.iterdir():
            if not cond_dir.is_dir():
                continue
            cond_id = cond_dir.name
            if cond_id == "phase1_learned_router":
                continue
            parts = cond_id.split("_")
            if len(parts) < 3 or parts[0] != "phase1":
                continue
            mode_tokens = parts[1:-2]
            mode = "_".join(mode_tokens)
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
                except (json.JSONDecodeError, OSError):
                    continue
                tid = int(rec["task_id"])
                success = bool(rec.get("success", False))
                matrix.setdefault(tid, {})[mode] = success
    return matrix


def read_step0_features(
    pass1_run_dir: Path, site: str, task_id: int
) -> Optional[dict[str, Any]]:
    """Read DOM-mode step-0 record to extract numeric features.

    Picks first available DOM condition in run_dir (any DOM-like condition works —
    entry-page is mode-agnostic at step-0 per train_l1_router.py:121-128).
    Returns dict with dom_complexity, text_length, tokens_input_text — or None if missing.
    """
    # Try preferred DOM-condition names first
    dom_candidates = ["phase1_dom_router_0", "phase1_dom_baseline_0", "phase1_dom_0"]
    cond_dir = None
    for name in dom_candidates:
        p = pass1_run_dir / name
        if p.is_dir():
            cond_dir = p
            break
    if cond_dir is None:
        for d in pass1_run_dir.iterdir():
            if d.is_dir() and "dom" in d.name and d.name.startswith("phase1_"):
                cond_dir = d
                break
    if cond_dir is None:
        # Fallback to any phase1_ condition
        cands = [
            d for d in pass1_run_dir.iterdir()
            if d.is_dir() and d.name.startswith("phase1_") and "router_learned" not in d.name
        ]
        if not cands:
            return None
        cond_dir = cands[0]

    steps_file = cond_dir / "episodes" / f"{site}_task_{task_id}_steps_v2.jsonl"
    if not steps_file.exists():
        return None
    try:
        with steps_file.open() as f:
            step0 = json.loads(f.readline())
    except (json.JSONDecodeError, OSError):
        return None
    sd = step0.get("state_digest", {}) or {}
    tokens = step0.get("tokens", {}) or {}
    return {
        "dom_complexity": int(sd.get("dom_complexity", 0) or 0),
        "text_length": int(sd.get("text_length", 0) or 0),
        "tokens_input_text": int(tokens.get("input_text", 0) or 0),
    }


def read_task_config(site: str, task_id: int) -> Optional[dict[str, Any]]:
    """Read VWA task config for intent + reference image + difficulty annotations.

    Returns dict with intent, has_reference_image, reasoning_difficulty — or None on
    missing/malformed config. Difficulty fields default to 0 when absent (some tasks
    lack VWA difficulty annotations).
    """
    cfg_file = VWA_CONFIG / f"test_{site}" / f"{task_id}.json"
    if not cfg_file.exists():
        return None
    try:
        cfg = json.loads(cfg_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    image = cfg.get("image")
    has_ref_image = image not in (None, "None", "", [])
    return {
        "intent": cfg.get("intent", "") or "",
        "has_reference_image": bool(has_ref_image),
        # F1 (B-1805): VWA stores these as ordinal strings ("easy"/"medium"/"hard"),
        # not ints — int("medium") crashed Stage 1 here. difficulty_to_int maps them
        # (train ≡ serve via shared router_features).
        "reasoning_difficulty": difficulty_to_int(cfg.get("reasoning_difficulty")),
        "visual_difficulty": difficulty_to_int(cfg.get("visual_difficulty")),
        "overall_difficulty": difficulty_to_int(cfg.get("overall_difficulty")),
    }


def build_cell_records(baseline: str, site: str) -> dict[str, Any]:
    """Build per-task raw feature records for a (baseline, site) cell.

    Returns dict with arrays (task_ids, numeric matrix, binary matrix, intents, labels)
    plus filter stats. Skips tasks with no-success outcomes (B-995 filter).
    """
    runs = find_pass1_runs(baseline, site)
    cell_id = f"{baseline}_{site}"
    if not runs:
        return {
            "cell_id": cell_id,
            "baseline": baseline,
            "site": site,
            "n_runs": 0,
            "n_total_tasks": 0,
            "n_filtered_no_success": 0,
            "n_kept": 0,
            "all_task_ids": [],
            "no_success_task_ids": [],
            "oracle_provenance": {
                "n_single_success": 0, "n_multi_success": 0,
                "n_no_success": 0, "n_total_universe": 0, "note": "no Pass-1 runs",
            },
            "task_ids": [],
            "intents": [],
            "X_numeric": np.zeros((0, 5), dtype=float),
            "X_binary": np.zeros((0, 15), dtype=int),
            "labels": [],
            "label_distribution": {},
            "error": "no Pass-1 runs found",
        }

    # Union outcomes across runs (newer overwrites older per train_l1_router.py:228)
    matrix: dict[int, dict[str, bool]] = {}
    for r in runs:
        sub = collect_per_task_outcomes([r], site)
        for tid, modes in sub.items():
            matrix.setdefault(tid, {}).update(modes)

    task_ids_sorted = sorted(matrix.keys())
    n_total = len(task_ids_sorted)
    filtered_no_success = 0

    records_task_id = []
    records_intent = []
    records_numeric = []
    records_binary = []
    records_label = []

    # C1 (B-1808): the FULL routable task universe (incl. no-success tasks). Pass-2
    # routes every task and the runtime hard-fails on any task missing from
    # fold_assignment, so the fold generator must cover this set even though only the
    # labeled rows are trained on (separate "trainable rows" from "routable universe").
    all_task_ids = list(task_ids_sorted)
    no_success_task_ids: list[int] = []
    # G1 (B-1809): oracle-label provenance. Labels are N=1 draws (single Pass-1 run per
    # condition); multi-success tasks (label = cheapest of >=2 succeeding modes) are the
    # tie-break / noise-sensitive subset that a self-oracle ceiling must contextualize.
    n_single_success = 0
    n_multi_success = 0

    pass1_run_for_step0 = runs[0]  # use first run dir for step-0 (per train_l1_router.py:234)

    for tid in task_ids_sorted:
        outcomes = matrix[tid]
        label = derive_oracle_label(outcomes)
        if label is None:
            filtered_no_success += 1
            no_success_task_ids.append(tid)
            continue
        if sum(1 for m in MODES if outcomes.get(m, False)) >= 2:
            n_multi_success += 1
        else:
            n_single_success += 1

        step0 = read_step0_features(pass1_run_for_step0, site, tid)
        cfg = read_task_config(site, tid)
        if cfg is None:
            # Task config missing — skip rather than fabricate
            continue

        intent = cfg["intent"]
        intent_tok_count = len(intent.split())
        # 5 numeric features (mode-agnostic step-0 + task config)
        numeric = np.array(
            [
                step0["dom_complexity"] if step0 else 0,
                step0["text_length"] if step0 else 0,
                step0["tokens_input_text"] if step0 else 0,
                intent_tok_count,
                cfg["reasoning_difficulty"],
            ],
            dtype=float,
        )
        # 15 binary features (has_ref_image + 14 intent regex)
        intent_bins = compute_intent_binaries(intent)
        binary = np.array(
            [int(cfg["has_reference_image"])]
            + [intent_bins[name] for name in sorted(INTENT_REGEX.keys())],
            dtype=int,
        )

        records_task_id.append(tid)
        records_intent.append(intent)
        records_numeric.append(numeric)
        records_binary.append(binary)
        records_label.append(label)

    n_kept = len(records_task_id)
    X_numeric = (
        np.vstack(records_numeric) if records_numeric else np.zeros((0, 5), dtype=float)
    )
    X_binary = (
        np.vstack(records_binary) if records_binary else np.zeros((0, 15), dtype=int)
    )
    label_dist = dict(Counter(records_label))

    return {
        "cell_id": cell_id,
        "baseline": baseline,
        "site": site,
        "n_runs": len(runs),
        "pass1_run_dirs": [r.name for r in runs],
        "n_total_tasks": n_total,
        "n_filtered_no_success": filtered_no_success,
        "n_kept": n_kept,
        # C1 (B-1808): full routable universe (incl. no-success) for fold coverage.
        "all_task_ids": all_task_ids,
        "no_success_task_ids": no_success_task_ids,
        # G1 (B-1809): oracle-label provenance (N=1; multi-success = tie-break sensitive).
        "oracle_provenance": {
            "n_single_success": n_single_success,
            "n_multi_success": n_multi_success,
            "n_no_success": filtered_no_success,
            "n_total_universe": n_total,
            "note": (
                "Oracle labels are N=1 draws (single Pass-1 run per condition). "
                "Multi-success tasks (label = cheapest of >=2 succeeding modes) are "
                "tie-break/noise sensitive. Report a self-oracle noise ceiling in paper "
                "§6 before claiming the router learns signal beyond oracle variance "
                "(router /stress G1 B-1809)."
            ),
        },
        "task_ids": records_task_id,
        "intents": records_intent,
        "X_numeric": X_numeric,
        "X_binary": X_binary,
        "labels": records_label,
        "label_distribution": label_dist,
    }


def extract_all_cells(cells: Optional[list[tuple[str, str]]] = None) -> dict[str, Any]:
    """Extract raw features for all (baseline, site) cells.

    Returns master dict with per-cell records + cross-cell pool arrays for Stage 2 use.
    """
    cells = cells or CELLS
    per_cell = {}
    for baseline, site in cells:
        print(f"\n=== Extracting {baseline} × {site} ===")
        rec = build_cell_records(baseline, site)
        per_cell[rec["cell_id"]] = rec
        if rec.get("error"):
            print(f"  ⚠️  {rec['error']}")
        else:
            print(
                f"  n_runs={rec['n_runs']}, n_total={rec['n_total_tasks']}, "
                f"n_filtered={rec['n_filtered_no_success']}, n_kept={rec['n_kept']}"
            )
            print(f"  label_distribution: {rec['label_distribution']}")

    # Cross-cell pooled arrays (for Stage 2 fold-local TF-IDF + global pooled MI)
    pooled_intents: list[str] = []
    pooled_X_numeric: list[np.ndarray] = []
    pooled_X_binary: list[np.ndarray] = []
    pooled_labels: list[str] = []
    pooled_task_ids: list[int] = []
    pooled_cell_ids: list[str] = []
    # C1 (B-1808): full routable universe (labeled + no-success) so the fold generator
    # covers every Pass-2 task; trained only on the labeled pooled arrays above.
    pooled_all_task_ids: list[int] = []
    pooled_all_cell_ids: list[str] = []

    for cell_id, rec in per_cell.items():
        for tid in rec.get("all_task_ids", []):
            pooled_all_task_ids.append(tid)
            pooled_all_cell_ids.append(cell_id)
        if rec["n_kept"] == 0:
            continue
        pooled_intents.extend(rec["intents"])
        pooled_X_numeric.append(rec["X_numeric"])
        pooled_X_binary.append(rec["X_binary"])
        pooled_labels.extend(rec["labels"])
        pooled_task_ids.extend(rec["task_ids"])
        pooled_cell_ids.extend([cell_id] * rec["n_kept"])

    X_pooled_numeric = (
        np.vstack(pooled_X_numeric) if pooled_X_numeric else np.zeros((0, 5), dtype=float)
    )
    X_pooled_binary = (
        np.vstack(pooled_X_binary) if pooled_X_binary else np.zeros((0, 15), dtype=int)
    )

    feature_names_numeric = [
        "dom_complexity",
        "text_length",
        "tokens_input_text",
        "intent_token_count",
        "reasoning_difficulty",
    ]
    feature_names_binary = ["has_reference_image"] + sorted(INTENT_REGEX.keys())

    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names_numeric": feature_names_numeric,
        "feature_names_binary": feature_names_binary,
        "n_numeric_features": 5,
        "n_binary_features": 15,
        "n_raw_features_total": 20,
        "tfidf_max_features": 30,
        "n_candidate_pool_total": 50,
        "cells_in_pool": list(per_cell.keys()),
        "cells_present_in_pool": [
            cid for cid, rec in per_cell.items() if rec["n_kept"] > 0
        ],
        "per_cell": per_cell,
        "pooled": {
            "intents": pooled_intents,
            "X_numeric": X_pooled_numeric,
            "X_binary": X_pooled_binary,
            "labels": pooled_labels,
            "task_ids": pooled_task_ids,
            "cell_ids": pooled_cell_ids,
            "n_total": len(pooled_intents),
            # C1: full routable universe (incl. no-success) for fold coverage.
            "all_task_ids": pooled_all_task_ids,
            "all_cell_ids": pooled_all_cell_ids,
            "n_universe_total": len(pooled_all_task_ids),
        },
    }


def save_npz(extracted: dict[str, Any], out_path: Path) -> None:
    """Save extracted features as NPZ + companion JSON metadata."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pooled = extracted["pooled"]
    np.savez_compressed(
        out_path,
        X_numeric=pooled["X_numeric"],
        X_binary=pooled["X_binary"],
        labels=np.array(pooled["labels"]),
        task_ids=np.array(pooled["task_ids"], dtype=int),
        cell_ids=np.array(pooled["cell_ids"]),
        intents=np.array(pooled["intents"], dtype=object),
        # C1 (B-1808): full routable universe (incl. no-success) for fold coverage.
        all_task_ids=np.array(pooled.get("all_task_ids", pooled["task_ids"]), dtype=int),
        all_cell_ids=np.array(pooled.get("all_cell_ids", pooled["cell_ids"])),
    )

    # Companion JSON with schema + per-cell summary
    meta_path = out_path.with_suffix(".json")
    meta = {
        "schema_version": extracted["schema_version"],
        "feature_names_numeric": extracted["feature_names_numeric"],
        "feature_names_binary": extracted["feature_names_binary"],
        "n_numeric_features": extracted["n_numeric_features"],
        "n_binary_features": extracted["n_binary_features"],
        "n_raw_features_total": extracted["n_raw_features_total"],
        "tfidf_max_features": extracted["tfidf_max_features"],
        "n_candidate_pool_total": extracted["n_candidate_pool_total"],
        "cells_in_pool": extracted["cells_in_pool"],
        "cells_present_in_pool": extracted["cells_present_in_pool"],
        "n_pooled_total": pooled["n_total"],
        "per_cell_summary": {
            cid: {
                "baseline": rec["baseline"],
                "site": rec["site"],
                "n_runs": rec["n_runs"],
                "pass1_run_dirs": rec.get("pass1_run_dirs", []),
                "n_total_tasks": rec["n_total_tasks"],
                "n_filtered_no_success": rec["n_filtered_no_success"],
                "n_kept": rec["n_kept"],
                "n_routable_universe": len(rec.get("all_task_ids", [])),
                "label_distribution": rec["label_distribution"],
                "oracle_provenance": rec.get("oracle_provenance", {}),
                "error": rec.get("error"),
            }
            for cid, rec in extracted["per_cell"].items()
        },
        "note_cell_constant_excluded": (
            "site and capability_tier features are NOT in the candidate pool. "
            "These are cell-constant within a per-cell pickle architecture (Q1=C + (E''')); "
            "cell identity is implicit via runtime pickle selection per (baseline, site)."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nWrote: {out_path}")
    print(f"Wrote: {meta_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default=str(OUT_DIR / "raw_features_phase1a.npz"),
        help="Output NPZ path (companion JSON written alongside).",
    )
    ap.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="Subset of cells to extract (e.g. B0_classifieds B1_reddit). Default = all 6.",
    )
    args = ap.parse_args()

    cells = None
    if args.cells:
        cells = []
        for cid in args.cells:
            parts = cid.split("_", 1)
            if len(parts) != 2 or parts[0] not in ("B0", "B1", "B2"):
                print(f"⚠️  Skipping invalid cell id: {cid}", file=sys.stderr)
                continue
            cells.append((parts[0], parts[1]))

    extracted = extract_all_cells(cells)
    save_npz(extracted, Path(args.output))

    print("\n=== Summary ===")
    print(f"Cells present in pool: {extracted['cells_present_in_pool']}")
    print(f"Pooled total tasks: {extracted['pooled']['n_total']}")
    print(f"Numeric features: {extracted['feature_names_numeric']}")
    print(f"Binary features: {extracted['feature_names_binary']}")
    print(
        f"Candidate pool: {extracted['n_raw_features_total']} raw + "
        f"{extracted['tfidf_max_features']} TF-IDF = {extracted['n_candidate_pool_total']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
