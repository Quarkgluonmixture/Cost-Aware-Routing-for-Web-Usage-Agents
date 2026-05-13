"""Paired (task_id, step, mode) helpers for v2 NPZ analysis.

Pipeline audit P0-1 / P0-7 fix substrate (2026-05-13). Previously
`stage4_logit_lens_axis2.py`, `stage4_axis2_layer_profile.py`, and
`stage4_robustness.py:test_b_per_task_simple` re-implemented the same
"per-mode → per-task → mean over steps → paired with sibling mode" loop.
Each implementation drifted slightly — extracting to one helper keeps
paper-grade analyses mechanically consistent across scripts and lets
P0-7 share bootstrap CI logic with future per-task analyses.

Schema expected on v2 NPZ files: `hidden_states (N, L, D)`, `mode_labels_str`,
`task_ids`, `step_indices`. See `run_stage4_multimode_extract.py` provenance
sidecar for canonical extractor output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


def load_v2_npz(npz_path: Path) -> dict:
    """Load NPZ with the standard v2 schema and validate required keys.

    Returns dict with fp32-promoted hidden states (P0-8 fix substrate —
    callers should not need to think about dtype at sub-permille magnitudes).
    """
    d = np.load(npz_path, allow_pickle=True)
    for k in ("hidden_states", "mode_labels_str", "task_ids", "step_indices"):
        if k not in d.files:
            raise KeyError(f"{npz_path}: missing key {k}")
    return {
        "H": d["hidden_states"].astype(np.float32, copy=False),
        "mode": d["mode_labels_str"],
        "tid": d["task_ids"],
        "step": d["step_indices"],
    }


def task_mode_step_index(npz: dict, mode: str) -> dict[tuple[int, int], int]:
    """Return {(task_id, step) -> row_idx} for given mode. Asserts uniqueness.

    Raises ValueError on duplicate (tid, step) within a mode (would indicate
    an extraction-script bug — same task+step+mode extracted twice).
    """
    mask = npz["mode"] == mode
    idx = np.where(mask)[0]
    out: dict[tuple[int, int], int] = {}
    for i in idx:
        key = (int(npz["tid"][i]), int(npz["step"][i]))
        if key in out:
            raise ValueError(
                f"duplicate (tid={key[0]}, step={key[1]}) for mode={mode!r}: "
                f"rows {out[key]} and {i}"
            )
        out[key] = int(i)
    return out


def paired_rows(
    npz: dict, mode_a: str, mode_b: str
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Return (H_a, H_b, keys) inner-joined on (task_id, step).

    H_a, H_b shape: (n_paired, n_layers, hidden_dim)
    keys: list of (task_id, step) in row order, for downstream provenance.
    """
    idx_a = task_mode_step_index(npz, mode_a)
    idx_b = task_mode_step_index(npz, mode_b)
    common = sorted(set(idx_a) & set(idx_b))
    rows_a = np.array([idx_a[k] for k in common])
    rows_b = np.array([idx_b[k] for k in common])
    return npz["H"][rows_a], npz["H"][rows_b], common


def paired_cosine_gap_per_layer(Ha: np.ndarray, Hb: np.ndarray) -> np.ndarray:
    """Per-task-paired cosine gap, averaged over paired axis.

    Ha, Hb: (n_paired, n_layers, D). Returns (n_layers,).

    Note: this is E_pair[1 - cos(h_a, h_b)] which is the paper-grade quantity.
    The naive E_pair[h_a] then cos(mean_a, mean_b) is a different number
    (smaller in magnitude under Jensen for cosine distance). The two
    differ for sub-permille mean-diff regimes documented in v2 retraction.
    """
    dot = (Ha * Hb).sum(axis=-1)  # (n, L)
    norm_a = np.linalg.norm(Ha, axis=-1)  # (n, L)
    norm_b = np.linalg.norm(Hb, axis=-1)
    gap = 1.0 - dot / (norm_a * norm_b + 1e-9)  # (n, L)
    return gap.mean(axis=0)  # (L,)


def task_bootstrap_ci(
    Ha: np.ndarray,
    Hb: np.ndarray,
    keys: list[tuple[int, int]],
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample tasks (NOT (task, step) pairs) with replacement, n_boot times.

    Returns (point_estimate, ci_lo, ci_hi), each shape (n_layers,).
    fn(Ha_b, Hb_b) -> scalar-per-layer statistic.

    Task-level resampling (not row-level) is the paper-grade choice — captures
    task-to-task variability while preserving within-task step paired structure.
    Seed default 20260513 matches stage4_robustness.py RNG idiom.
    """
    rng = rng or np.random.default_rng(seed=20260513)
    tids = np.array([k[0] for k in keys])
    unique_tids = np.unique(tids)
    point = fn(Ha, Hb)
    samples = np.empty((n_boot, point.shape[0]), dtype=np.float32)
    for b in range(n_boot):
        sel = rng.choice(unique_tids, size=len(unique_tids), replace=True)
        rows = np.concatenate([np.where(tids == t)[0] for t in sel])
        samples[b] = fn(Ha[rows], Hb[rows])
    lo = np.percentile(samples, 2.5, axis=0)
    hi = np.percentile(samples, 97.5, axis=0)
    return point, lo, hi
