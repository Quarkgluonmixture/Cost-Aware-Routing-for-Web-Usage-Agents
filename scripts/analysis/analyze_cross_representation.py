#!/usr/bin/env python3
"""Per-run diagnostic; not part of the 4-layer evidence framework.

Cross-representation task-level analysis for Phase 1 experiments.

Consumes episode_reason_rows.csv + episode summary JSONs to produce
task-granularity cross-representation comparisons (oracle ceiling,
exclusive sets, cost-at-success, reason stability, router signals).

Usage:
    python3 scripts/analysis/analyze_cross_representation.py \
        --run-dir results/visualwebarena/phase1/<RUN_ID> \
        [--reason-diag-dir <path>] \
        [--output-dir <path>] \
        [--skip-plots] \
        [--priority p0|p1|p2|all]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional deps -- graceful fallback
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from matplotlib_venn import venn2, venn3

    HAS_VENN = True
except ImportError:
    HAS_VENN = False

try:
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUMMARY_RE = re.compile(r"^(?P<site>.+)_task_(?P<task_id>\d+)_summary_v2\.json$")

# Visual signal keywords for R1 feature extraction
COLOR_WORDS = {
    "red", "blue", "green", "yellow", "black", "white", "pink", "purple",
    "orange", "brown", "grey", "gray", "silver", "gold", "beige", "navy",
}
VISUAL_PATTERNS = re.compile(
    r"(?:image|photo|picture|screenshot|logo|icon|thumbnail|color|colour|"
    r"looks?\s+like|visually|appearance|background|font|size|layout)",
    re.IGNORECASE,
)
NUMERIC_PATTERNS = re.compile(
    r"(?:cheapest|most expensive|highest|lowest|price|cost|\$\d|"
    r"how many|count|number of|total|average|more than|less than|"
    r"at least|at most|\d+\s*(?:items?|results?|listings?))",
    re.IGNORECASE,
)
NAV_VERB_PATTERNS = re.compile(
    r"(?:find|navigate|go to|open|click|search for|locate|browse|visit)",
    re.IGNORECASE,
)


_OUTPUT_FILENAME_PREFIXES = (
    "A1_", "A2_", "A3_", "A4_", "A4b_", "A5_", "A6_",
    "B1_", "B2_", "B3_",
    "R1_", "R2_", "R3_",
    "cross_representation_",
)


class OutputDirs:
    """Manages output subdirectories (tables/, plots/, root for JSON)."""

    def __init__(self, base: Path):
        self.base = base
        self.tables = base / "tables"
        self.plots = base / "plots"

    def ensure(self) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        self.tables.mkdir(exist_ok=True)
        self.plots.mkdir(exist_ok=True)
        # Clean previous outputs to avoid stale files from prior runs.
        # Restricted to known prefixes so user/external files placed in
        # the same directory are preserved (single-site runs use
        # base == out_root, where users may place notes or other JSON).
        for f in self.tables.glob("*.csv"):
            if f.name.startswith(_OUTPUT_FILENAME_PREFIXES):
                f.unlink(missing_ok=True)
        for f in self.plots.glob("*.png"):
            if f.name.startswith(_OUTPUT_FILENAME_PREFIXES):
                f.unlink(missing_ok=True)
        for f in self.base.glob("*.json"):
            if f.name.startswith(_OUTPUT_FILENAME_PREFIXES):
                f.unlink(missing_ok=True)

    def all_outputs(self) -> List[str]:
        """List all produced files relative to base."""
        produced = []
        for p in sorted(self.base.rglob("*")):
            if p.is_file():
                produced.append(str(p.relative_to(self.base)))
        return produced


def _safe_ratio(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _pct(v: float) -> str:
    return f"{v:.1%}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def resolve_reason_diagnostics(run_dir: Path, explicit: Optional[Path]) -> Path:
    """Find episode_reason_rows.csv, searching common locations."""
    candidates: List[Path] = []
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        if p.is_dir():
            f = p / "episode_reason_rows.csv"
            if f.is_file():
                return f
        print(f"[WARN] --reason-diag-dir {explicit} not usable, searching defaults")

    # Default location
    rd = run_dir / "analysis" / "reason_diagnostics" / "episode_reason_rows.csv"
    if rd.is_file():
        return rd

    # Live diagnostics (pick newest)
    live_base = run_dir / "analysis" / "reason_diagnostics_live"
    if live_base.is_dir():
        for d in sorted(live_base.iterdir(), reverse=True):
            f = d / "episode_reason_rows.csv"
            if f.is_file():
                candidates.append(f)
        if candidates:
            # Merge all live CSVs (different conditions may be in different dirs)
            return _merge_live_csvs(candidates, run_dir)

    # Not found -- try to generate
    print("[INFO] episode_reason_rows.csv not found, attempting auto-generation...")
    import subprocess

    script = Path(__file__).parent / "analyze_reason_diagnostics.py"
    cmd = [sys.executable, str(script), "--run-dir", str(run_dir), "--skip-similarity"]
    try:
        # Bound the auto-gen run so we don't hang forever if the
        # subprocess deadlocks (no progress reporting back here).
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        print("[ERROR] Auto-generation timed out after 1800s")
        sys.exit(1)
    if result.returncode != 0:
        print(f"[ERROR] Auto-generation failed:\n{result.stderr[:500]}")
        sys.exit(1)
    rd = run_dir / "analysis" / "reason_diagnostics" / "episode_reason_rows.csv"
    if rd.is_file():
        return rd

    print("[ERROR] Cannot locate or generate episode_reason_rows.csv")
    sys.exit(1)


def _merge_live_csvs(csv_paths: List[Path], run_dir: Path) -> Path:
    """Merge multiple live-diagnostic CSVs (deduplicate by condition+task)."""
    dfs = []
    for p in csv_paths:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"[WARN] Cannot read {p}: {e}")
    if not dfs:
        print("[ERROR] No readable live CSVs")
        sys.exit(1)
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["condition_id", "site", "task_id"], keep="last")
    return _save_merged_csv(merged, run_dir)


def _save_merged_csv(df: pd.DataFrame, run_dir: Path) -> Path:
    """Save merged CSV to a temp location and return path."""
    out = run_dir / "analysis" / "reason_diagnostics_live" / "_merged_episode_reason_rows.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def load_reason_rows(csv_path: Path) -> pd.DataFrame:
    """Load episode_reason_rows.csv."""
    df = pd.read_csv(csv_path)
    # Normalize types
    df["task_id"] = df["task_id"].astype(int)
    df["success"] = df["success"].astype(bool)
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce").fillna(0).astype(int)
    return df


def load_episode_summaries(run_dir: Path) -> pd.DataFrame:
    """Load all *_summary_v2.json into a DataFrame. Corrupt files are
    skipped with a warning so we don't silently lose data."""
    rows = []
    n_skipped = 0
    for p in run_dir.glob("*/episodes/*_summary_v2.json"):
        try:
            d = _read_json(p)
            rows.append(d)
        except Exception as exc:
            n_skipped += 1
            print(f"  [WARN] failed to read {p.name}: {exc}")
    if n_skipped:
        print(f"  [WARN] skipped {n_skipped} corrupt episode summaries")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "task_id" in df.columns:
        df["task_id"] = df["task_id"].astype(int)
    return df


def load_condition_meta(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load condition_meta.json for each condition."""
    metas = {}
    for p in run_dir.glob("*/condition_meta.json"):
        try:
            d = _read_json(p)
            cid = d.get("condition_id", p.parent.name)
            metas[cid] = d
        except Exception:
            pass
    return metas


def load_task_configs(run_dir: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Load task_configs/*.json keyed by (site, task_id)."""
    configs = {}
    tc_dir = run_dir / "task_configs"
    if not tc_dir.is_dir():
        return configs
    pat = re.compile(r"^(?P<site>.+)_task_(?P<task_id>\d+)\.json$")
    for p in tc_dir.glob("*.json"):
        m = pat.match(p.name)
        if m:
            try:
                d = _read_json(p)
                configs[(m.group("site"), int(m.group("task_id")))] = d
            except Exception:
                pass
    return configs


# ---------------------------------------------------------------------------
# Core: build task pivot
# ---------------------------------------------------------------------------


def _condition_to_mode(cond_metas: Dict[str, Dict], reason_df: pd.DataFrame) -> Dict[str, str]:
    """Map condition_id -> observation_mode."""
    mapping = {}
    # From condition_meta
    for cid, meta in cond_metas.items():
        mode = meta.get("observation_mode")
        if mode:
            mapping[cid] = mode
    # Fallback: from reason_rows observation_mode column
    if "observation_mode" in reason_df.columns:
        for cid in reason_df["condition_id"].unique():
            if cid not in mapping:
                subset = reason_df[reason_df["condition_id"] == cid]
                modes = subset["observation_mode"].dropna().unique()
                if len(modes) == 1:
                    mapping[cid] = modes[0]
                elif len(modes) > 1:
                    print(
                        f"  [WARN] condition_id={cid} has multiple observation_modes "
                        f"{modes.tolist()} in reason_df; dropping (its tasks will be "
                        f"excluded from per-mode analysis)"
                    )
    return mapping


def build_task_pivot(
    reason_df: pd.DataFrame, cond_metas: Dict[str, Dict],
) -> Tuple[pd.DataFrame, List[str]]:
    """Pivot reason_df from (condition × task) rows to one row per (site, task_id)
    with per-mode columns. Returns (pivot_df, sorted_modes)."""
    cond_mode = _condition_to_mode(cond_metas, reason_df)
    # Add mode column
    df = reason_df.copy()
    df["mode"] = df["condition_id"].map(cond_mode)
    df = df.dropna(subset=["mode"])

    modes = sorted(df["mode"].unique())

    # Base task-level info (from first available row per task)
    task_cols = ["site", "task_id", "task_type", "eval_type", "task_intent", "require_reset"]
    available_task_cols = [c for c in task_cols if c in df.columns]
    base = df.drop_duplicates(subset=["site", "task_id"])[available_task_cols].copy()

    # Per-mode columns
    per_mode_fields = ["success", "reason_bucket", "steps", "final_action_type", "fallback_finish", "page_unchanged_rate", "has_effective_action", "url_unique_count"]
    for mode in modes:
        mode_df = df[df["mode"] == mode][["site", "task_id"] + per_mode_fields].copy()
        mode_df = mode_df.rename(columns={f: f"{mode}_{f}" for f in per_mode_fields})
        base = base.merge(mode_df, on=["site", "task_id"], how="outer")

    base = base.sort_values(["site", "task_id"]).reset_index(drop=True)
    return base, modes


# ---------------------------------------------------------------------------
# False-positive detection (N/A FP + Eval FP; visual_fp removed in §95)
# ---------------------------------------------------------------------------


def _infer_benchmark(run_dir: Optional[Path]) -> str:
    """Infer benchmark from run_dir path (e.g. results/webarena/... -> 'webarena')."""
    if run_dir is not None:
        for part in run_dir.parts:
            if part == "webarena":
                return "webarena"
    return "visualwebarena"


def _mark_false_positives(
    pivot: pd.DataFrame, modes: List[str], *, run_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Add is_na_task + {mode}_na_fp + {mode}_eval_fp + {mode}_success_adj columns.

    N/A FP: any mode + N/A task + raw success + ~agent_finished.
    Eval FP (§95 simplified):
      string_match + success + ~agent_finished → always E-FP
      program_html + success + ~agent_finished + ~has_effective_action → E-FP
      url_match excluded: navigating to correct page without finish is legitimate.
    """
    from p79.experiment.analysis import _load_na_task_ids

    # episode_reason_rows.csv does not currently carry a `benchmark`
    # column, so this almost always falls through to path-based
    # inference; left here in case the schema gains the column later.
    _bm = "visualwebarena"
    if "benchmark" in pivot.columns and not pivot.empty:
        _bm = str(pivot["benchmark"].iloc[0])
    elif run_dir is not None:
        _bm = _infer_benchmark(run_dir)
    na_ids_by_site: Dict[str, set] = {}
    for site in pivot["site"].unique():
        na_ids_by_site[site] = _load_na_task_ids(str(site), _bm)

    pivot["is_na_task"] = pivot.apply(
        lambda r: int(r["task_id"]) in na_ids_by_site.get(r["site"], set()),
        axis=1,
    )

    # Compute agent_finished per mode (active finish, not fallback).
    # Mirror canonical p79.experiment.analysis.compute_adjusted_success:
    #   - na_fp is strict: missing/unknown agent_finished → flag as FP
    #     (we accomplish this by setting af_col=False when data missing,
    #      which makes nfp_col = is_na & success & ~False = True)
    #   - eval_fp is permissive: missing/unknown → skip (don't flag)
    #     (gated by af_known_col below)
    for m in modes:
        fat_col = f"{m}_final_action_type"
        ff_col = f"{m}_fallback_finish"
        af_col = f"{m}_agent_finished"
        af_known_col = f"{m}_agent_finished_known"
        if fat_col in pivot.columns and ff_col in pivot.columns:
            ff_series = pivot[ff_col].fillna(False).astype(bool)
            pivot[af_col] = (
                pivot[fat_col].astype(str).str.lower().isin(["finish", "stop"])
                & ~ff_series
            )
            pivot[af_known_col] = pivot[fat_col].notna()
        else:
            # Missing schema → treat as "agent_finished unknown".
            pivot[af_col] = False  # strict for na_fp
            pivot[af_known_col] = False  # blocks eval_fp

    # Mark false positives and build adjusted success columns
    na_fp_count: Dict[str, int] = {}
    eval_fp_count: Dict[str, int] = {}

    _is_string_match = pd.Series(False, index=pivot.index)
    _is_program_html = pd.Series(False, index=pivot.index)
    if "eval_type" in pivot.columns:
        _et = pivot["eval_type"].astype(str)
        _is_string_match = _et.str.contains("string_match", na=False)
        _is_program_html = _et.str.contains("program_html", na=False)

    for m in modes:
        scol = f"{m}_success"
        nfp_col = f"{m}_na_fp"
        efp_col = f"{m}_eval_fp"
        adj_col = f"{m}_success_adj"
        af_col = f"{m}_agent_finished"
        af_known_col = f"{m}_agent_finished_known"
        if scol not in pivot.columns:
            continue
        pivot[nfp_col] = pivot["is_na_task"] & (pivot[scol] == True) & ~pivot[af_col]
        # Eval FP: string_match always; program_html + ~has_effective_action.
        # Gate on af_known to avoid over-flagging when agent_finished can't be
        # determined from data (matches canonical compute_adjusted_success).
        _hea_col = f"{m}_has_effective_action"
        _hea = pivot[_hea_col].fillna(True).astype(bool) if _hea_col in pivot.columns else pd.Series(True, index=pivot.index)
        _efp_eligible_m = _is_string_match | (_is_program_html & ~_hea)
        pivot[efp_col] = (
            _efp_eligible_m
            & (pivot[scol] == True)
            & pivot[af_known_col]
            & ~pivot[af_col]
            & ~pivot[nfp_col]
        )
        pivot[adj_col] = pivot[scol].copy()
        pivot.loc[pivot[nfp_col] | pivot[efp_col], adj_col] = False
        na_fp_count[m] = int(pivot[nfp_col].sum())
        eval_fp_count[m] = int(pivot[efp_col].sum())

    n_na = int(pivot["is_na_task"].sum())
    fp_na_total = sum(na_fp_count.values())
    fp_eval_total = sum(eval_fp_count.values())
    print(f"  N/A FP: {n_na} N/A tasks, {fp_na_total} false positives {na_fp_count}")
    print(f"  Eval FP: {fp_eval_total} false positives {eval_fp_count}")
    return pivot, na_fp_count, eval_fp_count


# ---------------------------------------------------------------------------
# P0: Core cross-comparison
# ---------------------------------------------------------------------------


def a1_task_result_matrix(pivot: pd.DataFrame, modes: List[str], dirs: OutputDirs) -> Path:
    """A1: Save the full task × condition result matrix."""
    path = dirs.tables / "A1_task_result_matrix.csv"
    pivot.to_csv(path, index=False)
    print(f"  A1: {len(pivot)} tasks → tables/{path.name}")
    return path


def _compute_set_metrics(
    pivot: pd.DataFrame, modes: List[str], success_suffix: str = "_success",
) -> Dict[str, Any]:
    """Compute set analysis metrics using the given success column suffix.

    Args:
        success_suffix: "_success" for raw, "_success_adj" for adjusted.
    """
    n_tasks = len(pivot)
    if n_tasks == 0:
        return {}

    mode_sets: Dict[str, set] = {}
    mode_sr: Dict[str, float] = {}
    for m in modes:
        col = f"{m}{success_suffix}"
        if col in pivot.columns:
            s = set(pivot.loc[pivot[col] == True, ["site", "task_id"]].apply(tuple, axis=1))
            mode_sets[m] = s
            mode_sr[m] = _safe_ratio(len(s), n_tasks)
        else:
            mode_sets[m] = set()
            mode_sr[m] = 0.0

    all_sets = [s for s in mode_sets.values() if s]
    union_set = set().union(*all_sets) if all_sets else set()
    intersection_set = set.intersection(*all_sets) if all_sets else set()

    union_sr = _safe_ratio(len(union_set), n_tasks)
    intersection_sr = _safe_ratio(len(intersection_set), n_tasks)
    best_single = max(mode_sr.values()) if mode_sr else 0.0
    best_single_mode = max(mode_sr, key=mode_sr.get) if mode_sr else ""

    # Feature oracle: per (task_type, eval_type) pick best mode
    feature_oracle_sr = 0.0
    feature_oracle_choices: Dict[str, Any] = {}
    type_cols = []
    if "task_type" in pivot.columns:
        type_cols.append("task_type")
    if "eval_type" in pivot.columns:
        type_cols.append("eval_type")

    if type_cols:
        grouped = pivot.groupby(type_cols)
        weighted_successes = 0
        for group_key, group_df in grouped:
            group_sr: Dict[str, float] = {}
            for m in modes:
                col = f"{m}{success_suffix}"
                if col in group_df.columns:
                    group_sr[m] = group_df[col].sum() / len(group_df) if len(group_df) else 0
            if group_sr:
                best_mode = max(group_sr, key=group_sr.get)
                weighted_successes += group_sr[best_mode] * len(group_df)
                feature_oracle_choices[str(group_key)] = {
                    "best_mode": best_mode,
                    "sr": group_sr[best_mode],
                    "n_tasks": len(group_df),
                    "all_sr": {m: round(v, 4) for m, v in group_sr.items()},
                }
        feature_oracle_sr = _safe_ratio(weighted_successes, n_tasks)

    perfect_headroom = union_sr - best_single
    feature_headroom = feature_oracle_sr - best_single
    feature_gap = union_sr - feature_oracle_sr

    mode_n_tested: Dict[str, int] = {}
    for m in modes:
        col = f"{m}{success_suffix}"
        if col in pivot.columns:
            mode_n_tested[m] = int(pivot[col].notna().sum())
        else:
            mode_n_tested[m] = 0

    return {
        "n_tasks": n_tasks,
        "modes": modes,
        "per_mode_sr": {m: round(v, 4) for m, v in mode_sr.items()},
        "per_mode_success_count": {m: len(s) for m, s in mode_sets.items()},
        "per_mode_n_tested": mode_n_tested,
        "per_mode_sr_tested": {
            m: round(_safe_ratio(len(mode_sets[m]), mode_n_tested[m]), 4)
            for m in modes
        },
        "union_sr": round(union_sr, 4),
        "union_count": len(union_set),
        "intersection_sr": round(intersection_sr, 4),
        "intersection_count": len(intersection_set),
        "best_single_mode": best_single_mode,
        "best_single_sr": round(best_single, 4),
        "perfect_oracle_ceiling": round(union_sr, 4),
        "feature_oracle_ceiling": round(feature_oracle_sr, 4),
        "perfect_headroom": round(perfect_headroom, 4),
        "feature_headroom": round(feature_headroom, 4),
        "feature_gap": round(feature_gap, 4),
        "feature_oracle_choices": feature_oracle_choices,
        "_mode_sets": mode_sets,
        "_union_set": union_set,
        "_intersection_set": intersection_set,
    }


def a2_set_analysis(
    pivot: pd.DataFrame, modes: List[str], dirs: OutputDirs,
    na_fp_count: Optional[Dict[str, int]] = None,
    eval_fp_count: Optional[Dict[str, int]] = None,
) -> Dict:
    """A2: Set analysis + dual-layer oracle ceiling (raw + adjusted)."""
    if len(pivot) == 0:
        return {}

    # --- Raw metrics ---
    raw = _compute_set_metrics(pivot, modes, "_success")
    union_set = raw.pop("_union_set")
    intersection_set = raw.pop("_intersection_set")
    raw.pop("_mode_sets")

    # --- Adjusted metrics (na_fp + eval_fp; visual_fp removed in §95) ---
    has_adj = any(f"{m}_success_adj" in pivot.columns for m in modes)
    if has_adj:
        adj = _compute_set_metrics(pivot, modes, "_success_adj")
        adj.pop("_union_set", None)
        adj.pop("_intersection_set", None)
        adj.pop("_mode_sets", None)
        raw["per_mode_sr_adjusted"] = adj["per_mode_sr"]
        raw["per_mode_success_count_adjusted"] = adj["per_mode_success_count"]
        raw["per_mode_sr_tested_adjusted"] = adj["per_mode_sr_tested"]
        raw["union_sr_adjusted"] = adj["union_sr"]
        raw["union_count_adjusted"] = adj["union_count"]
        raw["intersection_sr_adjusted"] = adj["intersection_sr"]
        raw["intersection_count_adjusted"] = adj["intersection_count"]
        raw["best_single_mode_adjusted"] = adj["best_single_mode"]
        raw["best_single_sr_adjusted"] = adj["best_single_sr"]
        raw["perfect_oracle_ceiling_adjusted"] = adj["perfect_oracle_ceiling"]
        raw["feature_oracle_ceiling_adjusted"] = adj["feature_oracle_ceiling"]
        raw["perfect_headroom_adjusted"] = adj["perfect_headroom"]
        raw["feature_headroom_adjusted"] = adj["feature_headroom"]
        raw["feature_gap_adjusted"] = adj["feature_gap"]
        raw["feature_oracle_choices_adjusted"] = adj["feature_oracle_choices"]
    if na_fp_count is not None:
        raw["na_fp_count"] = na_fp_count
    if eval_fp_count is not None:
        raw["eval_fp_count"] = eval_fp_count

    summary = raw

    # Per-task annotation
    pivot_out = pivot.copy()
    pivot_out["in_union"] = pivot_out.apply(
        lambda r: (r["site"], r["task_id"]) in union_set, axis=1
    )
    pivot_out["in_intersection"] = pivot_out.apply(
        lambda r: (r["site"], r["task_id"]) in intersection_set, axis=1
    )

    _write_json(summary, dirs.base / "A2_set_analysis_summary.json")
    pivot_out.to_csv(dirs.tables / "A2_set_analysis.csv", index=False)

    union_sr = summary["union_sr"]
    intersection_sr = summary["intersection_sr"]
    best_single_mode = summary["best_single_mode"]
    best_single = summary["best_single_sr"]
    perfect_headroom = summary["perfect_headroom"]
    feature_gap = summary["feature_gap"]
    print(
        f"  A2: union={_pct(union_sr)}, intersection={_pct(intersection_sr)}, "
        f"best_single={best_single_mode}@{_pct(best_single)}, "
        f"perfect_headroom={_pct(perfect_headroom)}, feature_gap={_pct(feature_gap)}"
    )
    if has_adj:
        print(
            f"  A2 (adj): union={_pct(summary['union_sr_adjusted'])}, "
            f"best_single={summary['best_single_mode_adjusted']}@"
            f"{_pct(summary['best_single_sr_adjusted'])}, "
            f"headroom={_pct(summary['perfect_headroom_adjusted'])}"
        )
    return summary


def _compute_exclusive_sets(
    pivot: pd.DataFrame, modes: List[str], success_suffix: str = "_success",
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Compute exclusive set summary + detail using the given success suffix.

    Returns (summary_df, pivot_with_exclusive_set_col, set_col_name).

    Each task is classified into a tri-state vector per mode:
      True  → tested and succeeded
      False → tested and failed
      None  → not tested (column missing or NaN value)
    Untested modes are reported in the set name explicitly so they are not
    silently conflated with failure (which would inflate "only_X" buckets
    when modes have asymmetric task coverage).
    """
    def _success_vector(row):
        out = []
        for m in modes:
            col = f"{m}{success_suffix}"
            if col not in row.index:
                out.append(None)
                continue
            v = row[col]
            if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
                out.append(None)
            else:
                out.append(bool(v))
        return tuple(out)

    pivot_c = pivot.copy()
    pivot_c["_svec"] = pivot_c.apply(_success_vector, axis=1)

    def _set_name(svec: tuple) -> str:
        successes = [modes[i] for i, v in enumerate(svec) if v is True]
        failures = [modes[i] for i, v in enumerate(svec) if v is False]
        untested = [modes[i] for i, v in enumerate(svec) if v is None]
        # Build base classification over tested modes only.
        if not successes and not failures:
            return "all_untested"
        if not successes:
            base = "all_tested_fail"
        elif not failures:
            base = "all_tested_success" if untested else "all_success"
        elif len(successes) == 1:
            base = f"only_{successes[0]}"
        else:
            base = "_and_".join(successes) + "_not_" + "_".join(failures)
        if untested:
            return base + "_untested_" + "_".join(untested)
        # No untested modes → restore the legacy "all_fail" / "all_success"
        # labels for backward-compat with downstream consumers.
        if base == "all_tested_fail":
            return "all_fail"
        return base

    set_col = "exclusive_set" if success_suffix == "_success" else "exclusive_set_adj"
    pivot_c[set_col] = pivot_c["_svec"].apply(_set_name)

    summary_rows = []
    for sname, grp in pivot_c.groupby(set_col):
        tt_dist = dict(Counter(grp.get("task_type", pd.Series(dtype=str)).dropna()))
        summary_rows.append({
            "exclusive_set": sname,
            "count": len(grp),
            "pct": round(len(grp) / len(pivot_c), 4),
            "task_type_distribution": json.dumps(tt_dist),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("count", ascending=False)
    return summary_df, pivot_c, set_col


def a3_exclusive_sets(pivot: pd.DataFrame, modes: List[str], dirs: OutputDirs) -> None:
    """A3: Enumerate exclusive success/failure sets with task_type distribution."""
    n_modes = len(modes)
    if n_modes == 0:
        return

    # --- Raw ---
    summary_df, pivot_c, set_col = _compute_exclusive_sets(pivot, modes, "_success")
    summary_df.to_csv(dirs.tables / "A3_exclusive_sets_summary.csv", index=False)

    detail_cols = ["site", "task_id", set_col]
    if "task_type" in pivot_c.columns:
        detail_cols.append("task_type")
    if "task_intent" in pivot_c.columns:
        detail_cols.append("task_intent")
    for m in modes:
        detail_cols.extend([f"{m}_success", f"{m}_reason_bucket"])
    detail_cols = [c for c in detail_cols if c in pivot_c.columns]
    pivot_c[detail_cols].to_csv(dirs.tables / "A3_exclusive_sets_detail.csv", index=False)

    print(f"  A3: {len(summary_df)} exclusive sets")
    for _, r in summary_df.iterrows():
        print(f"      {r['exclusive_set']}: {r['count']} ({_pct(r['pct'])})")

    # --- Adjusted (if columns exist) ---
    has_adj = any(f"{m}_success_adj" in pivot.columns for m in modes)
    if has_adj:
        adj_summary_df, adj_pivot_c, adj_set_col = _compute_exclusive_sets(
            pivot, modes, "_success_adj",
        )
        adj_summary_df.to_csv(
            dirs.tables / "A3_exclusive_sets_summary_adjusted.csv", index=False,
        )

        adj_detail_cols = ["site", "task_id", adj_set_col]
        if "task_type" in adj_pivot_c.columns:
            adj_detail_cols.append("task_type")
        if "task_intent" in adj_pivot_c.columns:
            adj_detail_cols.append("task_intent")
        for m in modes:
            adj_detail_cols.extend([f"{m}_success_adj", f"{m}_reason_bucket"])
        adj_detail_cols = [c for c in adj_detail_cols if c in adj_pivot_c.columns]
        adj_pivot_c[adj_detail_cols].to_csv(
            dirs.tables / "A3_exclusive_sets_detail_adjusted.csv", index=False,
        )

        print(f"  A3 (adj): {len(adj_summary_df)} exclusive sets")
        for _, r in adj_summary_df.iterrows():
            print(f"      {r['exclusive_set']}: {r['count']} ({_pct(r['pct'])})")


# ---------------------------------------------------------------------------
# P1: Deep analysis
# ---------------------------------------------------------------------------


def a4_cost_at_success(
    pivot: pd.DataFrame, modes: List[str], ep_summaries: pd.DataFrame,
    cond_mode: Dict[str, str], dirs: OutputDirs,
) -> None:
    """A4: Cost comparison for tasks that succeed in ALL present modes."""
    if ep_summaries.empty:
        print("  A4: skipped (no episode summaries)")
        return

    # Filter to intersection tasks (must be True, not NaN, in ALL modes)
    success_cols = [
        f"{m}_success_adj" if f"{m}_success_adj" in pivot.columns else f"{m}_success"
        for m in modes
    ]
    success_cols = [c for c in success_cols if c in pivot.columns]
    mask = pivot[success_cols].apply(lambda row: all(v == True for v in row), axis=1)
    inter_tasks = set(pivot.loc[mask, ["site", "task_id"]].apply(tuple, axis=1))
    if not inter_tasks:
        print("  A4: skipped (no tasks succeed in all modes)")
        return

    # Add mode to summaries
    mode_col = "condition_id"
    if mode_col not in ep_summaries.columns:
        print("  A4: skipped (no condition_id in summaries)")
        return
    es = ep_summaries.copy()
    es["mode"] = es["condition_id"].map(cond_mode)
    es = es.dropna(subset=["mode"])
    # Resolve site per-row: prefer benchmark_site, fall back to site
    # only when benchmark_site is missing on that row.
    if "benchmark_site" in es.columns and "site" in es.columns:
        es["_site_resolved"] = es["benchmark_site"].fillna(es["site"])
    elif "benchmark_site" in es.columns:
        es["_site_resolved"] = es["benchmark_site"]
    elif "site" in es.columns:
        es["_site_resolved"] = es["site"]
    else:
        es["_site_resolved"] = ""
    es["_key"] = list(zip(es["_site_resolved"], es["task_id"]))
    es = es[es["_key"].isin(inter_tasks)]

    cost_fields = [
        "total_cost_usd", "total_latency_ms", "total_tokens",
        "total_energy_kwh", "steps",
    ]
    available_fields = [f for f in cost_fields if f in es.columns]

    # Pivot to per-task, per-mode
    rows = []
    for key, grp in es.groupby("_key"):
        site_val = key[0] if isinstance(key, tuple) else key
        tid_val = key[1] if isinstance(key, tuple) else grp.iloc[0]["task_id"]
        row = {"site": site_val, "task_id": tid_val}
        for _, er in grp.iterrows():
            m = er["mode"]
            for f in available_fields:
                row[f"{m}_{f}"] = er.get(f)
        # Cheapest mode
        mode_costs = {m: row.get(f"{m}_total_cost_usd") for m in modes}
        mode_costs = {m: v for m, v in mode_costs.items() if v is not None}
        if mode_costs:
            row["cheapest_mode"] = min(mode_costs, key=mode_costs.get)
        rows.append(row)

    if not rows:
        print("  A4: skipped (no matched rows)")
        return

    cost_df = pd.DataFrame(rows)
    cost_df.to_csv(dirs.tables / "A4_cost_at_success.csv", index=False)

    # Summary stats
    summary = {"n_tasks": len(cost_df)}
    for m in modes:
        m_stats = {}
        for f in available_fields:
            col = f"{m}_{f}"
            if col in cost_df.columns:
                vals = cost_df[col].dropna()
                if len(vals):
                    m_stats[f] = {"mean": round(vals.mean(), 6), "median": round(vals.median(), 6)}
        summary[m] = m_stats

    # Cheapest mode distribution
    if "cheapest_mode" in cost_df.columns:
        summary["cheapest_mode_distribution"] = dict(Counter(cost_df["cheapest_mode"]))

    # Wilcoxon paired test on cost (if scipy available)
    if HAS_SCIPY and len(modes) >= 2:
        stat_tests = {}
        for i in range(len(modes)):
            for j in range(i + 1, len(modes)):
                ma, mb = modes[i], modes[j]
                ca = cost_df.get(f"{ma}_total_cost_usd")
                cb = cost_df.get(f"{mb}_total_cost_usd")
                if ca is not None and cb is not None:
                    valid = ca.notna() & cb.notna()
                    if valid.sum() >= 5:
                        try:
                            stat, p = scipy_stats.wilcoxon(ca[valid], cb[valid])
                            stat_tests[f"{ma}_vs_{mb}"] = {
                                "wilcoxon_stat": round(float(stat), 4),
                                "p_value": round(float(p), 6),
                                "n": int(valid.sum()),
                            }
                        except Exception:
                            pass
        if stat_tests:
            summary["wilcoxon_tests"] = stat_tests

    _write_json(summary, dirs.base / "A4_cost_at_success_summary.json")
    print(f"  A4: {len(cost_df)} intersection tasks, cheapest_mode dist: "
          f"{summary.get('cheapest_mode_distribution', {})}")


def a5_task_type_success_rate(
    pivot: pd.DataFrame, modes: List[str], dirs: OutputDirs, skip_plots: bool,
) -> None:
    """A5: Task type × mode success rate breakdown (raw + adjusted)."""
    if "task_type" not in pivot.columns:
        print("  A5: skipped (no task_type)")
        return

    has_adj = any(f"{m}_success_adj" in pivot.columns for m in modes)

    rows = []
    for tt, grp in pivot.groupby("task_type"):
        row = {"task_type": tt, "n_tasks": len(grp)}
        for m in modes:
            col = f"{m}_success"
            if col in grp.columns:
                n_success = grp[col].sum()
                n_present = grp[col].notna().sum()
                row[f"{m}_success_count"] = int(n_success)
                row[f"{m}_n_present"] = int(n_present)
                row[f"{m}_sr"] = round(_safe_ratio(n_success, n_present), 4)
            # Adjusted
            adj_col = f"{m}_success_adj"
            if has_adj and adj_col in grp.columns:
                n_adj = grp[adj_col].sum()
                n_present_adj = grp[adj_col].notna().sum()
                row[f"{m}_success_count_adj"] = int(n_adj)
                row[f"{m}_sr_adj"] = round(_safe_ratio(n_adj, n_present_adj), 4)
        rows.append(row)

    tt_df = pd.DataFrame(rows).sort_values("n_tasks", ascending=False)
    tt_df.to_csv(dirs.tables / "A5_task_type_success_rate.csv", index=False)

    # Plot
    if not skip_plots and HAS_MPL and len(tt_df) > 0:
        fig, ax = plt.subplots(figsize=(max(8, len(tt_df) * 1.5), 5))
        x = np.arange(len(tt_df))
        width = 0.8 / max(len(modes), 1)
        for i, m in enumerate(modes):
            sr_col = f"{m}_sr_adj" if f"{m}_sr_adj" in tt_df.columns else f"{m}_sr"
            if sr_col in tt_df.columns:
                vals = tt_df[sr_col].fillna(0)
                ax.bar(x + i * width, vals, width, label=m)
        ax.set_xticks(x + width * (len(modes) - 1) / 2)
        ax.set_xticklabels(tt_df["task_type"], rotation=30, ha="right")
        ax.set_ylabel("Success Rate")
        ax.set_title("Success Rate by Task Type × Observation Mode (adjusted)")
        ax.legend()
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(dirs.plots / "A5_task_type_success_rate.png", dpi=150)
        plt.close(fig)

    print(f"  A5: {len(tt_df)} task types")


def a6_venn_diagram(pivot: pd.DataFrame, modes: List[str], dirs: OutputDirs, skip_plots: bool) -> None:
    """A6: Venn diagram of success sets."""
    mode_sets = {}
    for m in modes:
        col = f"{m}_success_adj" if f"{m}_success_adj" in pivot.columns else f"{m}_success"
        if col in pivot.columns:
            keys = set(pivot.loc[pivot[col] == True, ["site", "task_id"]].apply(tuple, axis=1))
            mode_sets[m] = keys

    present_modes = [m for m in modes if m in mode_sets and len(mode_sets[m]) > 0]
    if len(present_modes) < 2:
        print("  A6: skipped (need >=2 modes with successes)")
        return

    # Compute all region sizes for the table
    table_rows = []
    if len(present_modes) == 2:
        a, b = present_modes
        sa, sb = mode_sets[a], mode_sets[b]
        table_rows = [
            {"region": f"only_{a}", "count": len(sa - sb)},
            {"region": f"only_{b}", "count": len(sb - sa)},
            {"region": f"{a}_and_{b}", "count": len(sa & sb)},
        ]
    elif len(present_modes) >= 3:
        a, b, c = present_modes[:3]
        sa, sb, sc = mode_sets[a], mode_sets[b], mode_sets[c]
        table_rows = [
            {"region": f"only_{a}", "count": len(sa - sb - sc)},
            {"region": f"only_{b}", "count": len(sb - sa - sc)},
            {"region": f"only_{c}", "count": len(sc - sa - sb)},
            {"region": f"{a}_and_{b}_only", "count": len((sa & sb) - sc)},
            {"region": f"{a}_and_{c}_only", "count": len((sa & sc) - sb)},
            {"region": f"{b}_and_{c}_only", "count": len((sb & sc) - sa)},
            {"region": f"all_three", "count": len(sa & sb & sc)},
        ]

    pd.DataFrame(table_rows).to_csv(dirs.tables / "A6_venn_table.csv", index=False)

    # Plot
    if not skip_plots and HAS_MPL:
        fig, ax = plt.subplots(figsize=(7, 7))
        if len(present_modes) == 2 and HAS_VENN:
            a, b = present_modes
            venn2(
                [mode_sets[a], mode_sets[b]],
                set_labels=(a, b),
                ax=ax,
            )
        elif len(present_modes) >= 3 and HAS_VENN:
            a, b, c = present_modes[:3]
            venn3(
                [mode_sets[a], mode_sets[b], mode_sets[c]],
                set_labels=(a, b, c),
                ax=ax,
            )
        else:
            # Fallback: text summary
            ax.text(0.5, 0.5, "Venn data in A6_venn_table.csv\n(matplotlib_venn not installed)",
                    ha="center", va="center", fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.set_title("Success Set Overlap (adjusted)")
        fig.tight_layout()
        fig.savefig(dirs.plots / "A6_venn_diagram.png", dpi=150)
        plt.close(fig)

    print(f"  A6: {len(present_modes)}-way Venn, regions: {len(table_rows)}")


def b1_reason_transition_matrix(
    pivot: pd.DataFrame, modes: List[str], dirs: OutputDirs, skip_plots: bool,
) -> None:
    """B1: Reason bucket transition matrix between mode pairs."""
    pairs = []
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            ma, mb = modes[i], modes[j]
            ca, cb = f"{ma}_reason_bucket", f"{mb}_reason_bucket"
            if ca in pivot.columns and cb in pivot.columns:
                pairs.append((ma, mb, ca, cb))

    if not pairs:
        print("  B1: skipped (need >=2 modes with reason data)")
        return

    for ma, mb, ca, cb in pairs:
        valid = pivot[[ca, cb]].dropna()
        if valid.empty:
            continue
        ct = pd.crosstab(valid[ca], valid[cb], margins=True)
        ct.to_csv(dirs.tables / f"B1_transition_{ma}_to_{mb}.csv")

        # Heatmap (without margins)
        if not skip_plots and HAS_MPL:
            ct_no_margin = pd.crosstab(valid[ca], valid[cb])
            if ct_no_margin.size > 0:
                fig, ax = plt.subplots(figsize=(
                    max(8, len(ct_no_margin.columns) * 0.8),
                    max(6, len(ct_no_margin.index) * 0.5),
                ))
                im = ax.imshow(ct_no_margin.values, aspect="auto", cmap="YlOrRd")
                ax.set_xticks(range(len(ct_no_margin.columns)))
                ax.set_xticklabels(ct_no_margin.columns, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(len(ct_no_margin.index)))
                ax.set_yticklabels(ct_no_margin.index, fontsize=7)
                # Annotate cells
                for yi in range(len(ct_no_margin.index)):
                    for xi in range(len(ct_no_margin.columns)):
                        v = ct_no_margin.values[yi, xi]
                        if v > 0:
                            ax.text(xi, yi, str(v), ha="center", va="center", fontsize=7)
                ax.set_xlabel(mb)
                ax.set_ylabel(ma)
                ax.set_title(f"Reason Bucket: {ma} → {mb}")
                fig.colorbar(im, ax=ax, shrink=0.6)
                fig.tight_layout()
                fig.savefig(dirs.plots / f"B1_transition_{ma}_to_{mb}.png", dpi=150)
                plt.close(fig)

    print(f"  B1: {len(pairs)} transition pairs")


def b2_reason_stability(
    pivot: pd.DataFrame, modes: List[str], dirs: OutputDirs, skip_plots: bool,
) -> None:
    """B2: Per-task reason stability score across modes."""
    bucket_cols = [f"{m}_reason_bucket" for m in modes if f"{m}_reason_bucket" in pivot.columns]
    if len(bucket_cols) < 2:
        print("  B2: skipped (need >=2 modes with reason data)")
        return

    rows = []
    for _, r in pivot.iterrows():
        buckets = [r[c] for c in bucket_cols if pd.notna(r[c])]
        n_present = len(buckets)
        if n_present < 2:
            stability = float("nan")
        else:
            n_unique = len(set(buckets))
            stability = 1.0 - (n_unique - 1) / (n_present - 1)
        rows.append({
            "site": r["site"],
            "task_id": r["task_id"],
            "n_modes_present": n_present,
            "n_unique_buckets": len(set(buckets)) if buckets else 0,
            "stability": round(stability, 4) if not np.isnan(stability) else None,
            "buckets": "|".join(str(b) for b in buckets) if buckets else "",
        })

    stab_df = pd.DataFrame(rows)
    stab_df.to_csv(dirs.tables / "B2_reason_stability.csv", index=False)

    valid = stab_df["stability"].dropna()
    mean_stab = valid.mean() if len(valid) else float("nan")

    # Histogram
    if not skip_plots and HAS_MPL and len(valid) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(valid, bins=np.linspace(0, 1, 11), edgecolor="black", alpha=0.7)
        ax.axvline(mean_stab, color="red", linestyle="--", label=f"mean={mean_stab:.2f}")
        ax.set_xlabel("Stability Score")
        ax.set_ylabel("Count")
        ax.set_title("Reason Bucket Stability Across Modes")
        ax.legend()
        fig.tight_layout()
        fig.savefig(dirs.plots / "B2_reason_stability_histogram.png", dpi=150)
        plt.close(fig)

    print(f"  B2: mean stability={mean_stab:.3f} (n={len(valid)} tasks with >=2 modes)")


# ---------------------------------------------------------------------------
# P1 extras: A4b (cost by reason) + B3 (subtype breakdown)
# ---------------------------------------------------------------------------


def a4b_fail_reason_cost_stats(
    reason_df: pd.DataFrame,
    ep_summaries: pd.DataFrame,
    cond_mode: Dict[str, str],
    dirs: OutputDirs,
    skip_plots: bool,
) -> None:
    """A4b: Cost / latency / steps breakdown by failure reason bucket."""
    if ep_summaries.empty:
        print("  A4b: skipped (no episode summaries)")
        return

    # Normalise site column name
    es = ep_summaries.copy()
    site_col = "benchmark_site" if "benchmark_site" in es.columns else "site"
    if site_col not in es.columns:
        print("  A4b: skipped (no site column in summaries)")
        return
    es = es.rename(columns={site_col: "_site"})

    cost_cols = ["total_cost_usd", "total_latency_ms", "steps"]
    available_cost_cols = [c for c in cost_cols if c in es.columns]
    if "total_cost_usd" not in available_cost_cols:
        print("  A4b: skipped (no total_cost_usd in summaries)")
        return

    rd = reason_df[["condition_id", "site", "task_id", "success", "reason_bucket"]].copy()
    merge_es = es[["condition_id", "_site", "task_id"] + available_cost_cols].rename(
        columns={"_site": "site"}
    )
    merged = rd.merge(merge_es, on=["condition_id", "site", "task_id"], how="left")

    rows = []
    for bucket, grp in merged.groupby("reason_bucket"):
        row: Dict[str, Any] = {"reason_bucket": bucket, "count": len(grp)}
        if "steps" in grp.columns:
            row["avg_steps"] = round(float(grp["steps"].mean()), 2)
        if "total_cost_usd" in grp.columns:
            valid_cost = grp["total_cost_usd"].dropna()
            if len(valid_cost):
                row["avg_cost_usd"] = round(float(valid_cost.mean()), 6)
                row["p50_cost_usd"] = round(float(valid_cost.median()), 6)
        if "total_latency_ms" in grp.columns:
            valid_lat = grp["total_latency_ms"].dropna()
            if len(valid_lat):
                row["p95_latency_ms"] = round(float(valid_lat.quantile(0.95)), 1)
        rows.append(row)

    if not rows:
        print("  A4b: skipped (no rows after join)")
        return

    stats_df = pd.DataFrame(rows)
    if "avg_cost_usd" in stats_df.columns:
        stats_df = stats_df.sort_values("avg_cost_usd", ascending=False)
    stats_df.to_csv(dirs.tables / "A4b_fail_reason_cost_stats.csv", index=False)

    if not skip_plots and HAS_MPL and "avg_cost_usd" in stats_df.columns:
        n = len(stats_df)
        fig, ax = plt.subplots(figsize=(8, max(3, n * 0.4)))
        y = np.arange(n)
        ax.barh(y, stats_df["avg_cost_usd"].fillna(0).values)
        ax.set_yticks(y)
        ax.set_yticklabels(stats_df["reason_bucket"].values, fontsize=8)
        ax.set_xlabel("Avg Cost (USD)")
        ax.set_title("Average Cost by Failure Reason Bucket")
        fig.tight_layout()
        fig.savefig(dirs.plots / "A4b_fail_reason_cost_heatmap.png", dpi=150)
        plt.close(fig)

    print(f"  A4b: {len(stats_df)} reason buckets with cost stats")


def b3_subtype_breakdown(
    reason_df: pd.DataFrame,
    dirs: OutputDirs,
    skip_plots: bool,
) -> None:
    """B3: Failure subtype distribution within reason buckets."""
    subtype_cols = [c for c in ["unreachable_subtype", "stuck_subtype"] if c in reason_df.columns]
    if not subtype_cols:
        print("  B3: skipped (no subtype columns in reason data)")
        return

    df = reason_df.copy()
    # Collapse all subtype columns into one 'subtype' column (first non-null wins)
    df["subtype"] = df.apply(
        lambda r: next(
            (str(r[c]) for c in subtype_cols if pd.notna(r.get(c)) and str(r.get(c, "")).strip()),
            "(none)",
        ),
        axis=1,
    )
    df = df.dropna(subset=["reason_bucket"])

    # Summary: (reason_bucket, subtype) → count + pct_of_bucket
    rows = []
    for bucket, bgrp in df.groupby("reason_bucket"):
        bucket_total = len(bgrp)
        for subtype, sgrp in bgrp.groupby("subtype"):
            rows.append({
                "reason_bucket": bucket,
                "subtype": subtype,
                "count": len(sgrp),
                "pct_of_bucket": round(len(sgrp) / max(bucket_total, 1), 4),
            })

    if not rows:
        print("  B3: skipped (no subtype data)")
        return

    summary_df = pd.DataFrame(rows).sort_values(
        ["reason_bucket", "count"], ascending=[True, False]
    )
    summary_df.to_csv(dirs.tables / "B3_subtype_breakdown.csv", index=False)

    # Per-task detail CSV
    detail_cols = ["site", "task_id", "condition_id", "reason_bucket", "success", "subtype"]
    extra_cols = [c for c in ["observation_mode", "steps"] if c in df.columns]
    detail_cols = [c for c in detail_cols + extra_cols if c in df.columns]
    df[detail_cols].to_csv(dirs.tables / "B3_subtype_detail.csv", index=False)

    # Stacked bar plot
    if not skip_plots and HAS_MPL and len(summary_df) > 0:
        buckets = list(summary_df["reason_bucket"].unique())
        subtypes = [s for s in summary_df["subtype"].unique() if s != "(none)"]
        if not subtypes:
            subtypes = list(summary_df["subtype"].unique())

        fig, ax = plt.subplots(figsize=(max(8, len(buckets) * 0.9), 5))
        x = np.arange(len(buckets))
        bottom = np.zeros(len(buckets))
        for st in subtypes:
            vals = []
            for b in buckets:
                row = summary_df[(summary_df["reason_bucket"] == b) & (summary_df["subtype"] == st)]
                vals.append(int(row["count"].sum()) if len(row) else 0)
            ax.bar(x, vals, bottom=bottom, label=st)
            bottom += np.array(vals, dtype=float)
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Count")
        ax.set_title("Failure Subtype Breakdown by Reason Bucket")
        if subtypes:
            ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(dirs.plots / "B3_subtype_breakdown.png", dpi=150)
        plt.close(fig)

    print(f"  B3: {len(summary_df)} (bucket, subtype) combinations")


# ---------------------------------------------------------------------------
# P2: Router design support
# ---------------------------------------------------------------------------


def r1_task_features(
    pivot: pd.DataFrame, modes: List[str],
    task_configs: Dict[Tuple[str, int], Dict], dirs: OutputDirs,
) -> None:
    """R1: Extract per-task features for router design."""
    rows = []
    for _, r in pivot.iterrows():
        site, tid = r["site"], int(r["task_id"])
        intent = r.get("task_intent", "") or ""
        cfg = task_configs.get((site, tid), {})

        # From intent
        intent_lower = intent.lower()
        has_color_word = any(w in intent_lower.split() for w in COLOR_WORDS)
        has_visual_desc = bool(VISUAL_PATTERNS.search(intent))
        has_numeric = bool(NUMERIC_PATTERNS.search(intent))
        has_nav_verb = bool(NAV_VERB_PATTERNS.search(intent))

        # From task config
        eval_cfg = cfg.get("eval", {})
        eval_types = eval_cfg.get("eval_types", [])
        eval_type_str = "|".join(eval_types) if eval_types else r.get("eval_type", "")
        visual_diff = cfg.get("visual_difficulty", "")
        reasoning_diff = cfg.get("reasoning_difficulty", "")
        overall_diff = cfg.get("overall_difficulty", "")
        has_image = cfg.get("image") is not None

        # From episode data: succeeded_in_modes, best_mode (raw + adjusted)
        def _collect(success_suffix: str):
            succ, best, ms = [], "", float("inf")
            for m in modes:
                scol = f"{m}{success_suffix}"
                if scol in pivot.columns and r.get(scol) == True:
                    succ.append(m)
                    st = r.get(f"{m}_steps")
                    # Skip NaN/None steps when picking best.
                    if st is not None and not pd.isna(st) and st < ms:
                        ms = st
                        best = m
            # If no mode had usable steps but at least one succeeded,
            # fall back to the first succeeded mode rather than "".
            if not best and succ:
                best = succ[0]
            return succ, best

        succeeded_modes, best_mode = _collect("_success")
        succeeded_modes_adj, best_mode_adj = _collect("_success_adj")

        row = {
            "site": site,
            "task_id": tid,
            "task_type": r.get("task_type", ""),
            "eval_type": eval_type_str,
            "visual_difficulty": visual_diff,
            "reasoning_difficulty": reasoning_diff,
            "overall_difficulty": overall_diff,
            "has_image": has_image,
            "has_color_word": has_color_word,
            "has_visual_description": has_visual_desc,
            "has_numeric_comparison": has_numeric,
            "has_navigation_verb": has_nav_verb,
            "intent_length": len(intent),
            "succeeded_in_modes": "|".join(succeeded_modes),
            "n_modes_succeeded": len(succeeded_modes),
            "best_mode": best_mode,
            "succeeded_in_modes_adj": "|".join(succeeded_modes_adj),
            "n_modes_succeeded_adj": len(succeeded_modes_adj),
            "best_mode_adj": best_mode_adj,
        }
        rows.append(row)

    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(dirs.tables / "R1_task_features.csv", index=False)
    print(f"  R1: {len(feat_df)} tasks with features")


def r2_escalation_signals(
    reason_df: pd.DataFrame, pivot: pd.DataFrame, modes: List[str],
    cond_mode: Dict[str, str], dirs: OutputDirs, skip_plots: bool,
) -> None:
    """R2: Step-level escalation signals + counterfactual analysis."""
    df = reason_df.copy()
    df["mode"] = df["condition_id"].map(cond_mode)
    df = df.dropna(subset=["mode"])

    # Fields of interest
    signal_cols = ["stuck_first_step", "page_unchanged_streak_max_pos", "page_unchanged_streak_max_len"]
    available_signals = [c for c in signal_cols if c in df.columns]

    rows = []
    for _, r in df.iterrows():
        site, tid, mode = r["site"], int(r["task_id"]), r["mode"]
        success = bool(r.get("success", False))
        row = {
            "site": site,
            "task_id": tid,
            "mode": mode,
            "success": success,
            "reason_bucket": r.get("reason_bucket", ""),
            "steps": r.get("steps", 0),
        }
        for sc in available_signals:
            row[sc] = r.get(sc)

        # Divergence step = earliest of stuck_first_step / page_unchanged_streak_max_pos.
        div_candidates = []
        sfs = r.get("stuck_first_step")
        if pd.notna(sfs) and int(sfs) >= 0:
            div_candidates.append(int(sfs))
        pump = r.get("page_unchanged_streak_max_pos")
        if pd.notna(pump) and int(pump) >= 0:
            div_candidates.append(int(pump))
        div_step = min(div_candidates) if div_candidates else None
        row["divergence_step"] = div_step

        # Counterfactual: did other modes succeed for this task?
        # Compute both raw and adjusted variants — raw can over-count
        # "escalation_would_help" if the other mode's success is itself
        # an FP (na_fp/eval_fp). Adjusted uses the FP-filtered column
        # when available and is the load-bearing number for routing
        # headroom estimates.
        def _other_success(success_suffix: str) -> bool:
            for m in modes:
                if m == mode:
                    continue
                scol = f"{m}{success_suffix}"
                task_row = pivot[
                    (pivot["site"] == site) & (pivot["task_id"] == tid)
                ]
                if not task_row.empty and scol in task_row.columns:
                    if task_row.iloc[0].get(scol) == True:
                        return True
            return False

        if not success and div_step is not None:
            row["escalation_would_help"] = _other_success("_success")
            # Only emit adj column if any mode has it (avoid dense
            # all-False column when adjusted wasn't computed).
            if any(f"{m}_success_adj" in pivot.columns for m in modes):
                row["escalation_would_help_adj"] = _other_success("_success_adj")
            else:
                row["escalation_would_help_adj"] = None
        else:
            row["escalation_would_help"] = None
            row["escalation_would_help_adj"] = None

        rows.append(row)

    esc_df = pd.DataFrame(rows)
    esc_df.to_csv(dirs.tables / "R2_escalation_signals.csv", index=False)

    # Summary stats (raw + adjusted)
    failed_with_div = esc_df[(esc_df["success"] == False) & esc_df["divergence_step"].notna()]
    n_would_help = failed_with_div["escalation_would_help"].sum() if len(failed_with_div) else 0
    help_rate = _safe_ratio(n_would_help, len(failed_with_div))
    n_would_help_adj = (
        failed_with_div["escalation_would_help_adj"].fillna(False).sum()
        if "escalation_would_help_adj" in failed_with_div.columns and len(failed_with_div)
        else 0
    )
    help_rate_adj = _safe_ratio(n_would_help_adj, len(failed_with_div))

    # Divergence step distribution plot
    if not skip_plots and HAS_MPL:
        div_steps = failed_with_div["divergence_step"].dropna()
        if len(div_steps) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            for m in modes:
                m_div = failed_with_div[failed_with_div["mode"] == m]["divergence_step"].dropna()
                if len(m_div) > 0:
                    ax.hist(m_div, bins=range(0, int(m_div.max()) + 2), alpha=0.5, label=m)
            ax.set_xlabel("Divergence Step")
            ax.set_ylabel("Count")
            ax.set_title("Divergence Step Distribution (Failed Episodes)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(dirs.plots / "R2_divergence_step_distribution.png", dpi=150)
            plt.close(fig)

    print(f"  R2: {len(failed_with_div)} failed episodes with divergence, "
          f"escalation_would_help={_pct(help_rate)} ({n_would_help}/{len(failed_with_div)})")
    if "escalation_would_help_adj" in failed_with_div.columns:
        print(f"  R2 (adj): escalation_would_help={_pct(help_rate_adj)} "
              f"({int(n_would_help_adj)}/{len(failed_with_div)})")


def _build_oracle_rows(
    pivot_subset: pd.DataFrame, modes: List[str],
    cost_lookup: Dict[Tuple[str, int, str], float],
    task_configs: Dict[Tuple[str, int], Dict],
    success_suffix: str = "_success",
) -> List[Dict]:
    """Build oracle decomposition rows for given success column suffix."""
    rows = []
    for _, r in pivot_subset.iterrows():
        _tid_raw = r.get("task_id")
        if _tid_raw is None or pd.isna(_tid_raw):
            continue
        site, tid = r["site"], int(_tid_raw)
        succeeded = []
        for m in modes:
            scol = f"{m}{success_suffix}"
            if scol in r.index and r[scol] == True:
                cost = cost_lookup.get((site, tid, m))
                # Treat NaN cost as missing — min() on NaN is undefined.
                if cost is not None and pd.isna(cost):
                    cost = None
                steps_val = r.get(f"{m}_steps")
                # Same for steps: NaN ruins min().
                if steps_val is not None and pd.isna(steps_val):
                    steps_val = None
                succeeded.append({"mode": m, "cost": cost, "steps": steps_val})
        if not succeeded:
            continue
        with_cost = [s for s in succeeded if s["cost"] is not None]
        if with_cost:
            oracle = min(with_cost, key=lambda x: x["cost"])
        else:
            with_steps = [s for s in succeeded if s["steps"] is not None]
            if with_steps:
                # Use explicit None guard rather than `x or inf` —
                # 0 is a valid step count and would otherwise be coerced to inf.
                oracle = min(with_steps, key=lambda x: x["steps"])
            else:
                # No cost, no steps — pick first deterministically.
                oracle = succeeded[0]
        cfg = task_configs.get((site, tid), {})
        rows.append({
            "site": site,
            "task_id": tid,
            "task_type": r.get("task_type", ""),
            "eval_type": r.get("eval_type", ""),
            "visual_difficulty": cfg.get("visual_difficulty", ""),
            "reasoning_difficulty": cfg.get("reasoning_difficulty", ""),
            "oracle_choice": oracle["mode"],
            "oracle_cost": oracle["cost"],
            "oracle_steps": oracle["steps"],
            "n_modes_succeeded": len(succeeded),
            "succeeded_modes": "|".join(s["mode"] for s in succeeded),
        })
    return rows


def r3_oracle_decomposition(
    pivot: pd.DataFrame, modes: List[str],
    ep_summaries: pd.DataFrame, cond_mode: Dict[str, str],
    task_configs: Dict[Tuple[str, int], Dict], dirs: OutputDirs,
) -> None:
    """R3: Oracle router decomposition -- for each union-success task, pick cheapest mode."""
    # Build cost lookup from episode summaries
    cost_lookup: Dict[Tuple[str, int, str], float] = {}
    if not ep_summaries.empty and "condition_id" in ep_summaries.columns:
        es = ep_summaries.copy()
        es["mode"] = es["condition_id"].map(cond_mode)
        # Resolve site per-row: prefer benchmark_site, fall back to site
        # only when benchmark_site is missing on that row (DataFrame.get
        # is per-column, not per-row, so this needs explicit fillna).
        if "benchmark_site" in es.columns and "site" in es.columns:
            es["_site_resolved"] = es["benchmark_site"].fillna(es["site"])
        elif "benchmark_site" in es.columns:
            es["_site_resolved"] = es["benchmark_site"]
        elif "site" in es.columns:
            es["_site_resolved"] = es["site"]
        else:
            es["_site_resolved"] = ""
        for _, r in es.iterrows():
            site = r.get("_site_resolved", "")
            if site is None or (isinstance(site, float) and pd.isna(site)):
                continue
            _tid_raw = r.get("task_id")
            if _tid_raw is None or pd.isna(_tid_raw):
                continue
            tid = int(_tid_raw)
            mode = r.get("mode", "")
            cost = r.get("total_cost_usd")
            # Skip NaN cost — would corrupt min() in oracle selection.
            if mode and cost is not None and not pd.isna(cost):
                cost_lookup[(site, tid, mode)] = cost

    # --- Raw ---
    success_cols = [f"{m}_success" for m in modes if f"{m}_success" in pivot.columns]
    mask = pivot[success_cols].any(axis=1)
    union_tasks = pivot[mask].copy()

    if union_tasks.empty:
        print("  R3: skipped (no successful tasks)")
        return

    rows = _build_oracle_rows(union_tasks, modes, cost_lookup, task_configs, "_success")
    oracle_df = pd.DataFrame(rows)
    oracle_df.to_csv(dirs.tables / "R3_oracle_decomposition.csv", index=False)

    summary: Dict[str, Any] = {
        "n_union_tasks": len(oracle_df),
        "oracle_choice_distribution": dict(Counter(oracle_df["oracle_choice"])) if len(oracle_df) else {},
    }
    if "task_type" in oracle_df.columns and len(oracle_df):
        tt_choice = {}
        for tt, grp in oracle_df.groupby("task_type"):
            tt_choice[tt] = dict(Counter(grp["oracle_choice"]))
        summary["oracle_choice_by_task_type"] = tt_choice

    print(f"  R3: {len(oracle_df)} union tasks, oracle choice: "
          f"{summary['oracle_choice_distribution']}")

    # --- Adjusted ---
    has_adj = any(f"{m}_success_adj" in pivot.columns for m in modes)
    if has_adj:
        adj_cols = [f"{m}_success_adj" for m in modes if f"{m}_success_adj" in pivot.columns]
        adj_mask = pivot[adj_cols].any(axis=1)
        adj_union = pivot[adj_mask].copy()
        adj_rows = _build_oracle_rows(adj_union, modes, cost_lookup, task_configs, "_success_adj")
        adj_df = pd.DataFrame(adj_rows)
        adj_df.to_csv(dirs.tables / "R3_oracle_decomposition_adjusted.csv", index=False)
        summary["n_union_tasks_adjusted"] = len(adj_df)
        summary["oracle_choice_distribution_adjusted"] = (
            dict(Counter(adj_df["oracle_choice"])) if len(adj_df) else {}
        )
        if "task_type" in adj_df.columns and len(adj_df):
            tt_choice_adj = {}
            for tt, grp in adj_df.groupby("task_type"):
                tt_choice_adj[tt] = dict(Counter(grp["oracle_choice"]))
            summary["oracle_choice_by_task_type_adjusted"] = tt_choice_adj
        print(f"  R3 (adj): {len(adj_df)} union tasks, oracle choice: "
              f"{summary['oracle_choice_distribution_adjusted']}")

    _write_json(summary, dirs.base / "R3_oracle_decomposition.json")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def write_summary(
    a2_summary: Dict, modes: List[str], priority: str, dirs: OutputDirs,
) -> None:
    """Write cross_representation_summary.json."""
    produced = dirs.all_outputs()
    summary = {
        "priority": priority,
        "modes_analyzed": modes,
        "outputs": produced,
    }
    if a2_summary:
        summary["oracle_ceiling"] = a2_summary.get("perfect_oracle_ceiling")
        summary["routing_headroom"] = a2_summary.get("perfect_headroom")
        summary["feature_oracle_ceiling"] = a2_summary.get("feature_oracle_ceiling")
        summary["feature_gap"] = a2_summary.get("feature_gap")
        summary["per_mode_sr"] = a2_summary.get("per_mode_sr")
        # Adjusted (visual FP filtered)
        if "per_mode_sr_adjusted" in a2_summary:
            summary["per_mode_sr_adjusted"] = a2_summary["per_mode_sr_adjusted"]
            summary["oracle_ceiling_adjusted"] = a2_summary.get("perfect_oracle_ceiling_adjusted")
            summary["routing_headroom_adjusted"] = a2_summary.get("perfect_headroom_adjusted")
            summary["na_fp_count"] = a2_summary.get("na_fp_count")
            summary["eval_fp_count"] = a2_summary.get("eval_fp_count")
    _write_json(summary, dirs.base / "cross_representation_summary.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cross-representation task-level analysis for Phase 1"
    )
    parser.add_argument("--run-dir", required=True, help="Run directory path")
    parser.add_argument("--reason-diag-dir", default=None,
                        help="Path to reason diagnostics dir or CSV")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <run_dir>/analysis/cross_representation/)")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip PNG generation")
    parser.add_argument("--priority", choices=["p0", "p1", "p2", "all"], default="p0",
                        help="Analysis scope (default: p0)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"[ERROR] Run directory not found: {run_dir}")
        sys.exit(1)

    out_root = Path(args.output_dir) if args.output_dir else (
        run_dir / "analysis" / "results" / "cross_representation"
    )

    run_p1 = args.priority in ("p1", "all")
    run_p2 = args.priority in ("p2", "all")

    # --- Load data ---
    print("[1/5] Loading data...")
    csv_path = resolve_reason_diagnostics(run_dir, args.reason_diag_dir)
    print(f"  reason_rows: {csv_path}")
    reason_df = load_reason_rows(csv_path)
    print(f"  {len(reason_df)} episode rows loaded")

    cond_metas = load_condition_meta(run_dir)
    ep_summaries = load_episode_summaries(run_dir) if (run_p1 or run_p2) else pd.DataFrame()
    task_configs = load_task_configs(run_dir) if run_p2 else {}

    # --- Split by site (never cross-site) ---
    sites = sorted(reason_df["site"].dropna().unique())
    print(f"[2/5] Sites detected: {sites}")

    if len(sites) == 0:
        print("[ERROR] No site info in data — nothing to analyze")
        sys.exit(1)

    all_site_summaries: Dict[str, Dict] = {}
    for site in sites:
        site_reason_df = reason_df[reason_df["site"] == site]

        # Output dir: use site subdir only when multi-site
        if len(sites) == 1:
            site_out = out_root
        else:
            site_out = out_root / site
        dirs = OutputDirs(site_out)
        dirs.ensure()

        print(f"\n{'='*60}")
        print(f"[3/5] Analyzing site: {site} ({len(site_reason_df)} episodes)")
        print(f"{'='*60}")

        site_ep_summaries = pd.DataFrame()
        if not ep_summaries.empty:
            site_col = "benchmark_site" if "benchmark_site" in ep_summaries.columns else "site"
            if site_col in ep_summaries.columns:
                site_ep_summaries = ep_summaries[ep_summaries[site_col] == site]
            else:
                site_ep_summaries = ep_summaries

        site_task_configs = {
            k: v for k, v in task_configs.items() if k[0] == site
        }

        a2_summary = _run_site_analysis(
            site_reason_df, cond_metas, site_ep_summaries, site_task_configs,
            dirs, args.priority, run_p1, run_p2, args.skip_plots,
            run_dir=run_dir,
        )
        all_site_summaries[site] = a2_summary

    # --- Global summary ---
    print(f"\n[5/5] Writing global summary...")
    out_root.mkdir(parents=True, exist_ok=True)
    global_summary = {
        "priority": args.priority,
        "sites": sites,
        "per_site": {},
    }
    # Transpose every per-site A2 summary verbatim (minus the leading-
    # underscore "private" set fields used internally), so the global
    # summary doesn't silently drop feature_oracle_ceiling / feature_gap /
    # FP counts. This also keeps single-site and multi-site output
    # structures aligned (single-site runs previously had dirs.base ==
    # out_root, causing the per-site write_summary file to be overwritten
    # by this global write with a thinner structure).
    for site, a2s in all_site_summaries.items():
        if not a2s:
            global_summary["per_site"][site] = None
            continue
        site_info = {k: v for k, v in a2s.items() if not k.startswith("_")}
        # Preserve the legacy top-level field names callers may rely on.
        site_info.setdefault("oracle_ceiling", a2s.get("perfect_oracle_ceiling"))
        site_info.setdefault("routing_headroom", a2s.get("perfect_headroom"))
        if "per_mode_sr_adjusted" in a2s:
            site_info.setdefault(
                "oracle_ceiling_adjusted", a2s.get("perfect_oracle_ceiling_adjusted")
            )
            site_info.setdefault(
                "routing_headroom_adjusted", a2s.get("perfect_headroom_adjusted")
            )
        global_summary["per_site"][site] = site_info
    _write_json(global_summary, out_root / "cross_representation_summary.json")
    print(f"\nDone! Outputs in: {out_root}")


def _run_site_analysis(
    reason_df: pd.DataFrame,
    cond_metas: Dict[str, Dict],
    ep_summaries: pd.DataFrame,
    task_configs: Dict[Tuple[str, int], Dict],
    dirs: OutputDirs,
    priority: str,
    run_p1: bool,
    run_p2: bool,
    skip_plots: bool,
    *,
    run_dir: Optional[Path] = None,
) -> Dict:
    """Run all analyses for a single site. Returns A2 summary dict."""

    # --- Build pivot ---
    pivot, modes = build_task_pivot(reason_df, cond_metas)
    print(f"  {len(pivot)} tasks x {len(modes)} modes: {modes}")

    if len(pivot) == 0:
        print("  [WARN] No tasks in pivot — skipping")
        return {}

    if len(modes) < 2:
        print(f"  [WARN] Only {len(modes)} mode(s) — cross-representation needs >=2, skipping")
        return {}

    cond_mode = _condition_to_mode(cond_metas, reason_df)

    # --- False-positive detection (visual + N/A + eval) ---
    pivot, na_fp_count, eval_fp_count = _mark_false_positives(pivot, modes, run_dir=run_dir)

    # --- P0: Core ---
    print("  --- P0: Core cross-comparison ---")
    a1_task_result_matrix(pivot, modes, dirs)
    a2_summary = a2_set_analysis(pivot, modes, dirs, na_fp_count=na_fp_count, eval_fp_count=eval_fp_count)
    a3_exclusive_sets(pivot, modes, dirs)

    # --- P1: Deep ---
    if run_p1:
        print("  --- P1: Deep analysis ---")
        a4_cost_at_success(pivot, modes, ep_summaries, cond_mode, dirs)
        a4b_fail_reason_cost_stats(reason_df, ep_summaries, cond_mode, dirs, skip_plots)
        a5_task_type_success_rate(pivot, modes, dirs, skip_plots)
        a6_venn_diagram(pivot, modes, dirs, skip_plots)
        b1_reason_transition_matrix(pivot, modes, dirs, skip_plots)
        b2_reason_stability(pivot, modes, dirs, skip_plots)
        b3_subtype_breakdown(reason_df, dirs, skip_plots)

    # --- P2: Router support ---
    if run_p2:
        print("  --- P2: Router design support ---")
        r1_task_features(pivot, modes, task_configs, dirs)
        r3_oracle_decomposition(pivot, modes, ep_summaries, cond_mode, task_configs, dirs)
        r2_escalation_signals(reason_df, pivot, modes, cond_mode, dirs, skip_plots)

    # --- Per-site summary ---
    write_summary(a2_summary, modes, priority, dirs)
    print(f"  tables/ : {len(list(dirs.tables.iterdir()))} files")
    print(f"  plots/  : {len(list(dirs.plots.iterdir()))} files")

    return a2_summary


if __name__ == "__main__":
    main()
