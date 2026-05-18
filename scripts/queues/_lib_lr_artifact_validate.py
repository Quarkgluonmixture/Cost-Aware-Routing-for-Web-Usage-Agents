#!/usr/bin/env python3
"""LR artifact validate-each-pickle preflight (B-1603 /stress A2.10 P1-5-A 2026-05-18).

Called by `queue_phase1_router_paper_grade.sh` Gate 4 AFTER existence check passes.
Loads every fold-aware artifact via the runtime loader path (not raw `pickle.load()`)
to catch corrupt pickle / numpy version mismatch / sklearn version drift /
partial-write / dim-mismatch failures at gate time rather than at first runtime task.

Existence check alone is overconfident: paths exist ≠ pickle loadable. Empirical
failure modes this preflight catches:
  1. Corrupt pickle (write interrupted mid-stream)
  2. numpy version mismatch (numpy.ndarray serialization changed between versions)
  3. sklearn version drift (Pipeline / Vectorizer attribute layout changed)
  4. selected_idx_fold{k}.json missing required `selected_mask` key
  5. cell_meta missing required `thresholds_per_fold` key
  6. fold_assignment.json malformed task_id → fold_index dict

Exit codes:
  0 — all artifacts load cleanly
  1 — at least one artifact failed to load
  2 — schema-level integrity check failed (key missing, dtype wrong)

Usage:
  .venv/bin/python3 scripts/queues/_lib_lr_artifact_validate.py
  .venv/bin/python3 scripts/queues/_lib_lr_artifact_validate.py --artifacts-dir <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct invocation from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from p79.policies.learned_router import (  # noqa: E402
    load_cell_meta,
    load_fold_assignment,
    load_lr_pipeline_fold,
    load_selected_idx_fold,
    load_vectorizer_fold,
)

CELLS = [
    f"{baseline}_{site}"
    for baseline in ("B0", "B1", "B2")
    for site in ("classifieds", "reddit")
]
N_FOLDS = 5


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LR artifact validate-each-pickle preflight"
    )
    parser.add_argument(
        "--artifacts-dir",
        default="results/phantom_paper/l1_router",
        help="Directory containing LR fold-aware artifacts",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_dir():
        print(f"  ✗ artifacts_dir not a directory: {artifacts_dir}")
        return 1

    failures: list[str] = []

    # Shared fold-local feature machinery (5 folds × 2 = 10 paths)
    for k in range(N_FOLDS):
        vec = load_vectorizer_fold(artifacts_dir, k)
        if vec is None:
            failures.append(f"vectorizer_fold{k}.pkl load returned None")
        else:
            # Light schema check: should have .transform() method
            if not hasattr(vec, "transform"):
                failures.append(
                    f"vectorizer_fold{k}.pkl loaded but missing .transform "
                    f"(not a TfidfVectorizer?)"
                )

        mask, names = load_selected_idx_fold(artifacts_dir, k)
        if mask is None:
            failures.append(f"selected_idx_fold{k}.json load returned None")
        else:
            if mask.dtype.kind != "b":
                failures.append(
                    f"selected_idx_fold{k}.json mask dtype is {mask.dtype}, expected bool"
                )

    # Per-cell artifacts (6 cells × (2 meta + 5 LR-heads) = 42 paths)
    for cell_id in CELLS:
        fa = load_fold_assignment(artifacts_dir, cell_id)
        if not fa:
            failures.append(
                f"{cell_id}_fold_assignment.json load returned empty dict"
            )

        meta = load_cell_meta(artifacts_dir, cell_id)
        if not meta:
            failures.append(f"{cell_id}_lr_meta.json load returned empty dict")
        elif "thresholds_per_fold" not in meta:
            failures.append(
                f"{cell_id}_lr_meta.json missing required 'thresholds_per_fold' key"
            )

        for k in range(N_FOLDS):
            pipe = load_lr_pipeline_fold(artifacts_dir, cell_id, k)
            if pipe is None:
                failures.append(f"{cell_id}_lr_fold{k}.pkl load returned None")
            else:
                # Light schema check: should have .predict_proba + .classes_
                if not hasattr(pipe, "predict_proba"):
                    failures.append(
                        f"{cell_id}_lr_fold{k}.pkl loaded but missing .predict_proba"
                    )
                if not hasattr(pipe, "classes_"):
                    failures.append(
                        f"{cell_id}_lr_fold{k}.pkl loaded but missing .classes_"
                    )

    if failures:
        print(f"  ✗ {len(failures)} artifact validation failures:")
        for f in failures:
            print(f"      - {f}")
        return 2 if any("missing" in f for f in failures) else 1

    print(
        f"  ✓ {N_FOLDS * 2 + len(CELLS) * (2 + N_FOLDS)} artifacts validated "
        f"({N_FOLDS} shared-fold × 2 + {len(CELLS)} cells × 7 each)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
