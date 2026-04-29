#!/usr/bin/env python3
"""Per-run diagnostic; not part of the 4-dimension evidence framework.

Confidence calibration & signal analysis for Phase 2 routing go/no-go decision.

Analyses:
  C0 – Coverage report per condition (token-level + verbalized)
  C1 – Success vs failure distribution + Wilcoxon rank-sum
  C2 – Reliability diagram + ECE/MCE/Brier/AUROC (token-level + verbalized + behavioral)
  C3 – Per-step trajectory + position heatmap (logprob + entropy + verbalized)
  C4 – Per-mode comparison (dom/som/vision)
  C5 – Mode × outcome cross-analysis
  C6 – Behavioral signals (URL revisit, action diversity) + AUROC comparison
  C7 – Cross-mode AUROC grouped bar chart (DOM vs SoM per signal)
  C8 – Behavioral signal accumulation curve (earliest routing step)
  C9 – Token-level vs verbalized confidence comparison
  C10 – Composite signal exploration (correlation matrix + weighted AUROC grid search)

Usage:
  python scripts/analysis/analyze_confidence_calibration.py \
      --run-dir results/visualwebarena/phase1/B1_3mode_classifieds_20260404_141103
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ── Data loading ──────────────────────────────────────────────────────────

def _load_step_records(run_dir: Path) -> List[Dict[str, Any]]:
    """Load all *_steps_v2.jsonl with restart dedup.

    Delegates to analysis._collect_step_records which handles watchdog
    restart artifacts (stale lines from earlier runs in append-mode JSONL).
    """
    try:
        from p79.experiment.io_utils import read_jsonl_dedup
    except ImportError:
        from p79.experiment.analysis import _collect_step_records
        return _collect_step_records(run_dir)
    rows: List[Dict[str, Any]] = []
    for path in run_dir.glob("*/episodes/*_steps_v2.jsonl"):
        rows.extend(read_jsonl_dedup(path))
    return rows


def _load_episode_summaries(run_dir: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Load *_summary_v2.json, keyed by (condition_id, task_id).

    Skips summaries with missing/invalid task_id rather than coalescing
    them all to (-1) which silently overwrites entries.
    """
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for path in run_dir.glob("*/episodes/*_summary_v2.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as exc:
            print(f"  [WARN] cannot read {path.name}: {exc}")
            continue
        tid_raw = d.get("task_id")
        if tid_raw is None:
            print(f"  [WARN] missing task_id in {path.name}, skipping")
            continue
        try:
            tid = int(tid_raw)
        except (TypeError, ValueError):
            print(f"  [WARN] invalid task_id={tid_raw!r} in {path.name}, skipping")
            continue
        key = (d.get("condition_id", ""), tid)
        out[key] = d
    return out


# ── Episode-level aggregation ─────────────────────────────────────────────

CONF_KEYS = ("mean_logprob", "min_logprob", "mean_margin", "min_margin",
             "mean_entropy", "max_entropy", "verbalized")
EP_AGG_COLS = (
    "ep_mean_logprob", "ep_min_logprob", "ep_mean_margin", "ep_min_margin",
    "ep_mean_entropy", "ep_max_entropy",
    "ep_last3_mean_logprob", "ep_prob",
    "ep_mean_verbalized", "ep_min_verbalized",
)
# Behavioral signal columns (computed from step records, zero cost)
BEHAVIORAL_COLS = (
    "url_revisit_count", "url_revisit_max", "url_unique_count",
    "action_diversity", "action_unique_types", "max_repeat_streak",
)


def _build_episode_df(
    step_records: List[Dict[str, Any]],
    summaries: Dict[Tuple[str, int], Dict[str, Any]],
    *,
    run_dir: "Path | None" = None,
) -> pd.DataFrame:
    """Aggregate step-level confidence → episode rows with success labels."""

    # Group steps by (condition_id, task_id)
    episodes: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for rec in step_records:
        key = (rec.get("condition_id", ""), int(rec.get("task_id", -1)))
        episodes[key].append(rec)

    rows: List[Dict[str, Any]] = []
    for (cond, tid), steps in episodes.items():
        steps_sorted = sorted(steps, key=lambda s: s.get("step_idx", 0))

        # Observation mode from first step
        obs_mode = steps_sorted[0].get("observation_mode", "unknown")

        # Confidence steps only
        conf_steps = [s for s in steps_sorted if s.get("confidence")]
        total_steps = len(steps_sorted)
        conf_step_count = len(conf_steps)

        # Success label: prefer summary, fallback to last step reward
        summary = summaries.get((cond, tid), {})
        if "success" in summary and summary["success"] is not None:
            success = bool(summary["success"])
        else:
            # Fallback: last step reward > 0
            last = steps_sorted[-1]
            success = float(last.get("reward", 0)) > 0

        row: Dict[str, Any] = {
            "condition_id": cond,
            "task_id": tid,
            "observation_mode": obs_mode,
            "success": success,
            "total_steps": total_steps,
            "conf_step_count": conf_step_count,
            "conf_step_coverage": conf_step_count / total_steps if total_steps else 0.0,
        }

        if conf_step_count > 0:
            # Token-level logprobs (absent for API-based runs like B0)
            means = [s["confidence"]["mean_logprob"] for s in conf_steps
                     if "mean_logprob" in s.get("confidence", {})]
            mins = [s["confidence"]["min_logprob"] for s in conf_steps
                    if "min_logprob" in s.get("confidence", {})]
            margins_mean = [s["confidence"]["mean_margin"] for s in conf_steps
                            if "mean_margin" in s.get("confidence", {})]
            margins_min = [s["confidence"]["min_margin"] for s in conf_steps
                           if "min_margin" in s.get("confidence", {})]

            if means:
                row["ep_mean_logprob"] = float(np.mean(means))
                row["ep_min_logprob"] = float(np.min(mins))
                row["ep_mean_margin"] = float(np.mean(margins_mean))
                row["ep_min_margin"] = float(np.min(margins_min))
            else:
                row["ep_mean_logprob"] = np.nan
                row["ep_min_logprob"] = np.nan
                row["ep_mean_margin"] = np.nan
                row["ep_min_margin"] = np.nan

            # Entropy (may be absent for older episodes or API runs)
            ent_means = [s["confidence"]["mean_entropy"] for s in conf_steps
                         if "mean_entropy" in s.get("confidence", {})]
            ent_maxes = [s["confidence"]["max_entropy"] for s in conf_steps
                         if "max_entropy" in s.get("confidence", {})]
            if ent_means:
                row["ep_mean_entropy"] = float(np.mean(ent_means))
                row["ep_max_entropy"] = float(np.max(ent_maxes)) if ent_maxes else np.nan
            else:
                row["ep_mean_entropy"] = np.nan
                row["ep_max_entropy"] = np.nan

            # Verbalized confidence (may be absent for older episodes)
            verb_vals = [s["confidence"]["verbalized"] for s in conf_steps
                         if "verbalized" in s.get("confidence", {})]
            if verb_vals:
                row["ep_mean_verbalized"] = float(np.mean(verb_vals))
                row["ep_min_verbalized"] = float(np.min(verb_vals))
            else:
                row["ep_mean_verbalized"] = np.nan
                row["ep_min_verbalized"] = np.nan

            # Last-3 steps (token-level)
            if means:
                last3 = means[-3:] if len(means) >= 3 else means
                row["ep_last3_mean_logprob"] = float(np.mean(last3))
                row["ep_prob"] = float(np.exp(row["ep_mean_logprob"]))
            else:
                row["ep_last3_mean_logprob"] = np.nan
                row["ep_prob"] = np.nan
        else:
            for col in EP_AGG_COLS:
                row[col] = np.nan

        # ── Behavioral signals (always available, no logprobs needed) ──
        url_counts: Dict[str, int] = defaultdict(int)
        action_types_list: List[str] = []
        max_repeat_streak = 0
        streak = 0
        last_sig: str = ""
        for s in steps_sorted:
            digest = s.get("state_digest") or {}
            url = str(s.get("obs_url", "") or digest.get("url_after", "") or "").strip()
            if url:
                url_counts[url] += 1
            act = s.get("action") or {}
            atype = str(act.get("action_type", "") or "").lower()
            if atype:
                action_types_list.append(atype)
            sig = "|".join([
                atype,
                str(act.get("element_id", "")),
                str(act.get("text", ""))[:80],
                str(act.get("coordinate", "")),
                str(act.get("delta", "")),
            ])
            if sig == last_sig:
                streak += 1
            else:
                streak = 1
                last_sig = sig
            if streak > max_repeat_streak:
                max_repeat_streak = streak

        row["url_revisit_count"] = sum(v - 1 for v in url_counts.values() if v > 1)
        row["url_revisit_max"] = max(url_counts.values()) if url_counts else 0
        row["url_unique_count"] = len(url_counts)
        n_actions = len(action_types_list)
        row["action_diversity"] = len(set(action_types_list)) / n_actions if n_actions else 0.0
        row["action_unique_types"] = len(set(action_types_list))
        row["max_repeat_streak"] = max_repeat_streak

        rows.append(row)

    return pd.DataFrame(rows)


def _build_step_df(step_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build step-level DataFrame with confidence fields flattened."""
    rows = []
    for rec in step_records:
        conf = rec.get("confidence")
        if not conf:
            continue
        rows.append({
            "condition_id": rec.get("condition_id", ""),
            "task_id": int(rec.get("task_id", -1)),
            "step_idx": int(rec.get("step_idx", 0)),
            "observation_mode": rec.get("observation_mode", "unknown"),
            "mean_logprob": conf.get("mean_logprob"),
            "min_logprob": conf.get("min_logprob"),
            "mean_margin": conf.get("mean_margin"),
            "min_margin": conf.get("min_margin"),
            "mean_entropy": conf.get("mean_entropy"),
            "max_entropy": conf.get("max_entropy"),
            "verbalized": conf.get("verbalized"),
        })
    return pd.DataFrame(rows)


# ── C0: Coverage ──────────────────────────────────────────────────────────

def c0_coverage(ep_df: pd.DataFrame, tables_dir: Path,
                step_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-condition confidence coverage report (token-level + verbalized)."""
    groups = ep_df.groupby("condition_id")
    rows = []
    for cond, grp in groups:
        n_ep = len(grp)
        n_with = int((grp["conf_step_count"] > 0).sum())
        ep_cov = n_with / n_ep if n_ep else 0.0
        total_steps = int(grp["total_steps"].sum())
        total_conf_steps = int(grp["conf_step_count"].sum())
        step_cov = total_conf_steps / total_steps if total_steps else 0.0

        # Verbalized coverage
        n_with_verb = int(grp["ep_mean_verbalized"].notna().sum())
        ep_verb_cov = n_with_verb / n_ep if n_ep else 0.0
        # Step-level verbalized coverage (from step_df)
        verb_step_count = 0
        if step_df is not None and not step_df.empty:
            cond_steps = step_df[step_df["condition_id"] == cond]
            verb_step_count = int(cond_steps["verbalized"].notna().sum())
        verb_step_cov = verb_step_count / total_steps if total_steps else 0.0

        rows.append({
            "condition_id": cond,
            "episodes": n_ep,
            "episodes_with_confidence": n_with,
            "episode_coverage": round(ep_cov, 4),
            "total_steps": total_steps,
            "confidence_steps": total_conf_steps,
            "step_coverage": round(step_cov, 4),
            "episodes_with_verbalized": n_with_verb,
            "verbalized_episode_coverage": round(ep_verb_cov, 4),
            "verbalized_steps": verb_step_count,
            "verbalized_step_coverage": round(verb_step_cov, 4),
        })
    cov_df = pd.DataFrame(rows)
    cov_df.to_csv(tables_dir / "confidence_coverage.csv", index=False)
    print(f"  C0: coverage table → {tables_dir / 'confidence_coverage.csv'}")
    return cov_df


# ── C1: Success vs Failure Distribution ───────────────────────────────────

METRICS_CORE = ["ep_mean_logprob", "ep_min_logprob", "ep_mean_margin", "ep_min_margin"]
METRICS_ENTROPY = ["ep_mean_entropy", "ep_max_entropy"]
METRICS_VERBALIZED = ["ep_mean_verbalized", "ep_min_verbalized"]
METRICS_ALL = METRICS_CORE + METRICS_ENTROPY + METRICS_VERBALIZED
METRIC_LABELS = {
    "ep_mean_logprob": "Mean Log-Prob",
    "ep_min_logprob": "Min Log-Prob",
    "ep_mean_margin": "Mean Margin",
    "ep_min_margin": "Min Margin",
    "ep_mean_entropy": "Mean Entropy",
    "ep_max_entropy": "Max Entropy",
    "ep_mean_verbalized": "Mean Verbalized Conf",
    "ep_min_verbalized": "Min Verbalized Conf",
}


def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from Mann-Whitney U.

    Standard definition: rb = 2 * AUROC - 1 = 2*U/(n1*n2) - 1, where U is U1
    (the U statistic for the FIRST sample in mannwhitneyu(x, y)). With x =
    success_vals, positive rb means success > failure on the metric.

    Previous version (`1 - 2*U/(n1*n2)`) was sign-reversed; downstream
    `_routing_readiness` uses |rb| so the verdict was unaffected, but the
    `mannwhitney_test.csv` rb column had the wrong sign.
    """
    return (2.0 * float(u_stat)) / (n1 * n2) - 1.0


def c1_distribution(ep_df: pd.DataFrame, tables_dir: Path, plots_dir: Path):
    """Success vs failure violin plots + Wilcoxon rank-sum tests."""
    # Determine which metrics have enough data (>=4 non-NaN episodes)
    active_metrics: list = []
    for m in METRICS_CORE + METRICS_ENTROPY + METRICS_VERBALIZED:
        n_valid = ep_df[m].notna().sum() if m in ep_df.columns else 0
        if n_valid >= 4:
            active_metrics.append(m)

    if not active_metrics:
        print("  C1: skipped – no metric has >=4 episodes with data")
        return

    # Use episodes that have at least one active metric
    df = ep_df.dropna(subset=active_metrics, how="all")
    if len(df) < 4:
        print("  C1: skipped – too few episodes with any confidence metric")
        return

    succ = df[df["success"]]
    fail = df[~df["success"]]

    # Stats table
    stat_rows = []
    for m in active_metrics:
        for label, grp in [("success", succ), ("failure", fail)]:
            vals = grp[m].dropna()
            stat_rows.append({
                "metric": m, "outcome": label,
                "n": len(vals),
                "mean": round(float(vals.mean()), 6) if len(vals) else np.nan,
                "median": round(float(vals.median()), 6) if len(vals) else np.nan,
                "std": round(float(vals.std()), 6) if len(vals) else np.nan,
            })
    pd.DataFrame(stat_rows).to_csv(tables_dir / "confidence_by_outcome.csv", index=False)

    # Wilcoxon rank-sum tests
    test_rows = []
    for m in active_metrics:
        s_vals = succ[m].dropna().values
        f_vals = fail[m].dropna().values
        if len(s_vals) < 2 or len(f_vals) < 2:
            test_rows.append({"metric": m, "U": np.nan, "p_value": np.nan,
                              "rank_biserial": np.nan, "n_success": len(s_vals),
                              "n_failure": len(f_vals)})
            continue
        u, p = sp_stats.mannwhitneyu(s_vals, f_vals, alternative="two-sided")
        rb = _rank_biserial(u, len(s_vals), len(f_vals))
        test_rows.append({
            "metric": m, "U": round(float(u), 2),
            "p_value": round(float(p), 6),
            "rank_biserial": round(float(rb), 4),
            "n_success": len(s_vals), "n_failure": len(f_vals),
        })
    pd.DataFrame(test_rows).to_csv(tables_dir / "mannwhitney_test.csv", index=False)
    print(f"  C1: tables → confidence_by_outcome.csv, mannwhitney_test.csv")

    # Violin plot – dynamic grid based on active metrics
    n_metrics = len(active_metrics)
    ncols = 3 if n_metrics > 4 else 2
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for idx, m in enumerate(active_metrics):
        ax = axes_flat[idx]
        data = [succ[m].dropna().values, fail[m].dropna().values]
        if any(len(d) == 0 for d in data):
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(METRIC_LABELS.get(m, m))
            continue
        parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)
        # Color: success=green, failure=red
        for i, color in enumerate(["#2ca02c", "#d62728"]):
            if i < len(parts["bodies"]):
                parts["bodies"][i].set_facecolor(color)
                parts["bodies"][i].set_alpha(0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Success", "Failure"])
        ax.set_title(METRIC_LABELS.get(m, m))
        ax.grid(alpha=0.3)
    # Hide unused subplot slots
    for idx in range(n_metrics, nrows * ncols):
        axes_flat[idx].set_visible(False)
    fig.suptitle("C1: Confidence by Episode Outcome", fontsize=14)
    fig.tight_layout()
    fig.savefig(plots_dir / "C1_confidence_violin.png", dpi=150)
    plt.close(fig)
    print(f"  C1: plot → C1_confidence_violin.png")


# ── C2: Reliability Diagram + Calibration Metrics ─────────────────────────

def _compute_ece_mce_brier(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10,
) -> Tuple[float, float, float]:
    """Compute ECE, MCE, and Brier score from probabilities and binary labels."""
    if len(probs) == 0:
        return (float("nan"), float("nan"), float("nan"))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_sum = 0.0
    mce = 0.0
    total_n = len(probs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        diff = abs(float(probs[mask].mean()) - float(labels[mask].mean()))
        ece_sum += n_bin * diff
        mce = max(mce, diff)
    ece = ece_sum / total_n if total_n else float("nan")
    brier = float(np.mean((probs - labels) ** 2))
    return (ece, mce, brier)


def _auroc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC via Mann-Whitney U (no sklearn needed). NaN if single class."""
    unique = np.unique(y_true)
    if len(unique) < 2:
        return float("nan")
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u, _ = sp_stats.mannwhitneyu(pos, neg, alternative="greater")
    return float(u / (len(pos) * len(neg)))


def _c2_auroc_table(ep_df: pd.DataFrame, tables_dir: Path) -> Dict[str, Any]:
    """Compute AUROC for all signal types and optimal thresholds.

    Called unconditionally (does not require token-level ep_prob).
    Returns a metrics dict consumed by downstream routing readiness.
    """
    auroc_rows = []
    for m in METRICS_ALL:
        sig_type = "verbalized" if m in METRICS_VERBALIZED else "token_level"
        mdf = ep_df.dropna(subset=[m])
        if len(mdf) < 4 or mdf["success"].nunique() < 2:
            auroc_rows.append({"metric": m, "AUROC": np.nan, "AUROC_ci_lower": np.nan,
                               "AUROC_ci_upper": np.nan, "n": len(mdf),
                               "signal_type": sig_type})
            continue
        y = mdf["success"].astype(int).values
        scores = mdf[m].values
        if "entropy" in m:
            scores = -scores
        a, ci_lo, ci_hi = _auroc_bootstrap_ci(y, scores)
        auroc_rows.append({
            "metric": m,
            "AUROC": round(a, 6) if not math.isnan(a) else None,
            "AUROC_ci_lower": round(ci_lo, 6) if not math.isnan(ci_lo) else None,
            "AUROC_ci_upper": round(ci_hi, 6) if not math.isnan(ci_hi) else None,
            "n": len(mdf), "signal_type": sig_type,
        })

    behavioral_auroc_config = {
        "url_revisit_count": True, "url_revisit_max": True,
        "url_unique_count": False, "action_diversity": False,
        "action_unique_types": False, "max_repeat_streak": True,
    }
    for m, negate in behavioral_auroc_config.items():
        if m not in ep_df.columns:
            continue
        mdf = ep_df.dropna(subset=[m])
        if len(mdf) < 4 or mdf["success"].nunique() < 2:
            auroc_rows.append({"metric": m, "AUROC": np.nan, "AUROC_ci_lower": np.nan,
                               "AUROC_ci_upper": np.nan, "n": len(mdf),
                               "signal_type": "behavioral"})
            continue
        y = mdf["success"].astype(int).values
        scores = mdf[m].values
        if negate:
            scores = -scores
        a, ci_lo, ci_hi = _auroc_bootstrap_ci(y, scores)
        auroc_rows.append({
            "metric": m,
            "AUROC": round(a, 6) if not math.isnan(a) else None,
            "AUROC_ci_lower": round(ci_lo, 6) if not math.isnan(ci_lo) else None,
            "AUROC_ci_upper": round(ci_hi, 6) if not math.isnan(ci_hi) else None,
            "n": len(mdf), "signal_type": "behavioral",
        })

    auroc_df = pd.DataFrame(auroc_rows)
    auroc_df.to_csv(tables_dir / "auroc_all_metrics.csv", index=False)
    print(f"  C2: auroc_all_metrics.csv ({len(auroc_rows)} signals)")

    # Optimal threshold (Youden's J) per signal
    threshold_rows = []
    threshold_negate = {
        "ep_mean_entropy": True, "ep_max_entropy": True,
        "url_revisit_count": True, "url_revisit_max": True,
        "url_unique_count": False, "action_unique_types": False,
        "action_diversity": False, "max_repeat_streak": True,
    }
    all_threshold_signals = list(METRICS_ALL) + list(BEHAVIORAL_LABELS.keys())
    for m in all_threshold_signals:
        if m not in ep_df.columns:
            continue
        mdf = ep_df.dropna(subset=[m])
        if len(mdf) < 10 or mdf["success"].nunique() < 2:
            continue
        y = mdf["success"].astype(int).values
        scores = mdf[m].values
        if threshold_negate.get(m, False):
            scores = -scores
        opt = _optimal_threshold(y, scores)
        opt["metric"] = m
        if threshold_negate.get(m, False) and not math.isnan(opt["threshold"]):
            opt["threshold"] = -opt["threshold"]
        threshold_rows.append(opt)
    if threshold_rows:
        pd.DataFrame(threshold_rows).to_csv(tables_dir / "optimal_thresholds.csv", index=False)
        print(f"  C2: optimal_thresholds.csv ({len(threshold_rows)} signals)")

    return {}


def c2_calibration(
    ep_df: pd.DataFrame, tables_dir: Path, plots_dir: Path, n_bins: int = 10,
) -> Dict[str, Any]:
    """Reliability diagram + ECE/MCE/Brier/AUROC.

    AUROC table (auroc_all_metrics.csv) is always computed for all available
    signals (token-level, verbalized, behavioral).  The reliability diagram
    and ECE/MCE/Brier require token-level ep_prob and are skipped when absent.
    """
    # ── Always compute multi-metric AUROC (independent of ep_prob) ──
    metrics = _c2_auroc_table(ep_df, tables_dir)

    df = ep_df.dropna(subset=["ep_prob"]).copy()
    if len(df) < 4:
        print("  C2: reliability diagram skipped – too few episodes with ep_prob")
        return metrics

    probs = df["ep_prob"].values
    labels = df["success"].astype(int).values

    # Bin edges 0..1
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_rows = []
    ece_sum = 0.0
    mce = 0.0
    total_n = len(probs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            bin_rows.append({
                "bin": i, "lo": round(lo, 2), "hi": round(hi, 2),
                "n": 0, "mean_predicted": np.nan, "mean_actual": np.nan,
                "abs_diff": np.nan,
            })
            continue
        mean_pred = float(probs[mask].mean())
        mean_actual = float(labels[mask].mean())
        diff = abs(mean_pred - mean_actual)
        ece_sum += n_bin * diff
        mce = max(mce, diff)
        bin_rows.append({
            "bin": i, "lo": round(lo, 2), "hi": round(hi, 2),
            "n": n_bin,
            "mean_predicted": round(mean_pred, 6),
            "mean_actual": round(mean_actual, 6),
            "abs_diff": round(diff, 6),
        })

    ece = ece_sum / total_n if total_n else float("nan")
    brier = float(np.mean((probs - labels) ** 2))
    auroc = _auroc_safe(labels, probs)

    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(tables_dir / "calibration_bins.csv", index=False)

    metrics = {
        "ECE": round(ece, 6), "MCE": round(mce, 6),
        "Brier": round(brier, 6), "AUROC": round(auroc, 6) if not math.isnan(auroc) else None,
        "n_episodes": total_n,
    }
    pd.DataFrame([metrics]).to_csv(tables_dir / "calibration_metrics.csv", index=False)

    print(f"  C2: tables → calibration_bins.csv, calibration_metrics.csv, auroc_all_metrics.csv")

    # Plot
    valid = bin_df.dropna(subset=["mean_predicted", "mean_actual"])
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    if len(valid):
        ax1.plot(valid["mean_predicted"], valid["mean_actual"], "o-", color="#1f77b4",
                 label="Model")
    ax1.set_xlabel("Mean Predicted Probability (ep_prob)")
    ax1.set_ylabel("Observed Success Rate")
    ax1.set_title(f"C2: Reliability Diagram  (ECE={ece:.3f}, AUROC={auroc:.3f})")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)

    # Bin counts on secondary axis
    ax2 = ax1.twinx()
    mid = [(r["lo"] + r["hi"]) / 2 for _, r in bin_df.iterrows()]
    ax2.bar(mid, bin_df["n"], width=1.0 / n_bins * 0.8, alpha=0.2, color="gray",
            label="Bin count")
    ax2.set_ylabel("Bin count")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(plots_dir / "C2_reliability_diagram.png", dpi=150)
    plt.close(fig)
    print(f"  C2: plot → C2_reliability_diagram.png")

    # ── Verbalized reliability diagram (ep_mean_verbalized is already in [0,1]) ──
    verb_df = ep_df.dropna(subset=["ep_mean_verbalized"]).copy()
    if len(verb_df) >= 4 and verb_df["success"].nunique() >= 2:
        v_probs = verb_df["ep_mean_verbalized"].values
        v_labels = verb_df["success"].astype(int).values
        v_bin_rows = []
        v_ece_sum = 0.0
        v_mce = 0.0
        v_total = len(v_probs)
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (v_probs >= lo) & (v_probs < hi) if i < n_bins - 1 else (v_probs >= lo) & (v_probs <= hi)
            n_bin = int(mask.sum())
            if n_bin == 0:
                v_bin_rows.append({"bin": i, "lo": round(lo, 2), "hi": round(hi, 2),
                                   "n": 0, "mean_predicted": np.nan, "mean_actual": np.nan,
                                   "abs_diff": np.nan})
                continue
            mp = float(v_probs[mask].mean())
            ma = float(v_labels[mask].mean())
            diff = abs(mp - ma)
            v_ece_sum += n_bin * diff
            v_mce = max(v_mce, diff)
            v_bin_rows.append({"bin": i, "lo": round(lo, 2), "hi": round(hi, 2),
                               "n": n_bin, "mean_predicted": round(mp, 6),
                               "mean_actual": round(ma, 6), "abs_diff": round(diff, 6)})

        v_ece = v_ece_sum / v_total if v_total else float("nan")
        v_brier = float(np.mean((v_probs - v_labels) ** 2))
        v_auroc = _auroc_safe(v_labels, v_probs)

        v_bin_df = pd.DataFrame(v_bin_rows)
        v_bin_df.to_csv(tables_dir / "verbalized_calibration_bins.csv", index=False)

        v_metrics = {
            "ECE": round(v_ece, 6), "MCE": round(v_mce, 6),
            "Brier": round(v_brier, 6),
            "AUROC": round(v_auroc, 6) if not math.isnan(v_auroc) else None,
            "n_episodes": v_total,
        }
        pd.DataFrame([v_metrics]).to_csv(
            tables_dir / "verbalized_calibration_metrics.csv", index=False)
        metrics["verbalized"] = v_metrics

        # Plot
        v_valid = v_bin_df.dropna(subset=["mean_predicted", "mean_actual"])
        fig_v, ax_v1 = plt.subplots(figsize=(7, 5))
        ax_v1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        if len(v_valid):
            ax_v1.plot(v_valid["mean_predicted"], v_valid["mean_actual"], "s-",
                       color="#ff7f0e", label="Verbalized")
        ax_v1.set_xlabel("Mean Verbalized Confidence")
        ax_v1.set_ylabel("Observed Success Rate")
        ax_v1.set_title(f"C2: Verbalized Reliability Diagram  "
                        f"(ECE={v_ece:.3f}, AUROC={v_auroc:.3f})")
        ax_v1.legend(loc="upper left")
        ax_v1.grid(alpha=0.3)
        ax_v1.set_xlim(-0.05, 1.05)
        ax_v1.set_ylim(-0.05, 1.05)
        ax_v2 = ax_v1.twinx()
        v_mid = [(r["lo"] + r["hi"]) / 2 for _, r in v_bin_df.iterrows()]
        ax_v2.bar(v_mid, v_bin_df["n"], width=1.0 / n_bins * 0.8, alpha=0.2,
                  color="gray", label="Bin count")
        ax_v2.set_ylabel("Bin count")
        ax_v2.legend(loc="lower right")
        fig_v.tight_layout()
        fig_v.savefig(plots_dir / "C2_verbalized_reliability_diagram.png", dpi=150)
        plt.close(fig_v)
        print(f"  C2: verbalized → verbalized_calibration_bins.csv, "
              f"verbalized_calibration_metrics.csv, C2_verbalized_reliability_diagram.png")
    else:
        print(f"  C2: verbalized reliability skipped – {len(verb_df)} episodes with data")

    return metrics


# ── C3: Per-Step Trajectory ───────────────────────────────────────────────

def c3_trajectory(
    step_df: pd.DataFrame,
    ep_df: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
):
    """Mean confidence trajectory along step index + position heatmap."""
    if step_df.empty:
        print("  C3: skipped – no step-level confidence data")
        return

    # Merge success label into step_df
    success_map = dict(zip(
        zip(ep_df["condition_id"], ep_df["task_id"]),
        ep_df["success"],
    ))
    sdf = step_df.copy()
    sdf["success"] = sdf.apply(
        lambda r: success_map.get((r["condition_id"], r["task_id"]), None), axis=1,
    )
    sdf = sdf.dropna(subset=["success"])
    sdf["success"] = sdf["success"].astype(bool)

    # ── Trajectory plot ──
    max_step = int(sdf["step_idx"].max()) if len(sdf) else 0
    step_range = range(0, min(max_step + 1, 30))  # cap at 30 for readability

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, grp, color in [
        ("Success", sdf[sdf["success"]], "#2ca02c"),
        ("Failure", sdf[~sdf["success"]], "#d62728"),
    ]:
        means = []
        stds = []
        xs = []
        for si in step_range:
            vals = grp.loc[grp["step_idx"] == si, "mean_logprob"]
            if len(vals) >= 1:
                means.append(float(vals.mean()))
                stds.append(float(vals.std()) if len(vals) > 1 else 0.0)
                xs.append(si)
        if xs:
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            ax.plot(xs, means_arr, "o-", color=color, label=label, markersize=4)
            ax.fill_between(xs, means_arr - stds_arr, means_arr + stds_arr,
                            alpha=0.15, color=color)
    ax.set_xlabel("Step Index")
    ax.set_ylabel("Mean Log-Prob")
    ax.set_title("C3: Confidence Trajectory by Outcome")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "C3_confidence_trajectory.png", dpi=150)
    plt.close(fig)
    print(f"  C3: plot → C3_confidence_trajectory.png")

    # ── Entropy trajectory (if available) ──
    sdf_ent = sdf.dropna(subset=["mean_entropy"])
    if len(sdf_ent) >= 10:
        fig_e, ax_e = plt.subplots(figsize=(7, 5))
        for label, grp, color in [
            ("Success", sdf_ent[sdf_ent["success"]], "#2ca02c"),
            ("Failure", sdf_ent[~sdf_ent["success"]], "#d62728"),
        ]:
            ent_means, ent_stds, xs = [], [], []
            for si in step_range:
                vals = grp.loc[grp["step_idx"] == si, "mean_entropy"]
                if len(vals) >= 1:
                    ent_means.append(float(vals.mean()))
                    ent_stds.append(float(vals.std()) if len(vals) > 1 else 0.0)
                    xs.append(si)
            if xs:
                m_arr = np.array(ent_means)
                s_arr = np.array(ent_stds)
                ax_e.plot(xs, m_arr, "o-", color=color, label=label, markersize=4)
                ax_e.fill_between(xs, m_arr - s_arr, m_arr + s_arr,
                                  alpha=0.15, color=color)
        ax_e.set_xlabel("Step Index")
        ax_e.set_ylabel("Mean Entropy")
        ax_e.set_title("C3: Entropy Trajectory by Outcome")
        ax_e.legend()
        ax_e.grid(alpha=0.3)
        fig_e.tight_layout()
        fig_e.savefig(plots_dir / "C3_entropy_trajectory.png", dpi=150)
        plt.close(fig_e)
        print(f"  C3: plot → C3_entropy_trajectory.png")
    else:
        print(f"  C3: entropy trajectory skipped – only {len(sdf_ent)} steps with entropy")

    # ── Verbalized trajectory (if available) ──
    sdf_verb = sdf.dropna(subset=["verbalized"])
    if len(sdf_verb) >= 10:
        fig_v, ax_v = plt.subplots(figsize=(7, 5))
        for label, grp, color in [
            ("Success", sdf_verb[sdf_verb["success"]], "#2ca02c"),
            ("Failure", sdf_verb[~sdf_verb["success"]], "#d62728"),
        ]:
            v_means, v_stds, xs = [], [], []
            for si in step_range:
                vals = grp.loc[grp["step_idx"] == si, "verbalized"]
                if len(vals) >= 1:
                    v_means.append(float(vals.mean()))
                    v_stds.append(float(vals.std()) if len(vals) > 1 else 0.0)
                    xs.append(si)
            if xs:
                m_arr = np.array(v_means)
                s_arr = np.array(v_stds)
                ax_v.plot(xs, m_arr, "o-", color=color, label=label, markersize=4)
                ax_v.fill_between(xs, m_arr - s_arr, m_arr + s_arr,
                                  alpha=0.15, color=color)
        ax_v.set_xlabel("Step Index")
        ax_v.set_ylabel("Verbalized Confidence")
        ax_v.set_title("C3: Verbalized Confidence Trajectory by Outcome")
        ax_v.set_ylim(-0.05, 1.05)
        ax_v.legend()
        ax_v.grid(alpha=0.3)
        fig_v.tight_layout()
        fig_v.savefig(plots_dir / "C3_verbalized_trajectory.png", dpi=150)
        plt.close(fig_v)
        print(f"  C3: plot → C3_verbalized_trajectory.png")
    else:
        print(f"  C3: verbalized trajectory skipped – only {len(sdf_verb)} steps with data")

    # ── Position heatmap (requires token-level logprobs) ──
    sdf_lp = sdf.dropna(subset=["mean_logprob"])
    if len(sdf_lp) >= 10:
        sdf_lp = sdf_lp.copy()
        sdf_lp["logprob_bin"] = pd.cut(sdf_lp["mean_logprob"], bins=5, labels=False)
        sdf_lp["step_bin"] = pd.cut(sdf_lp["step_idx"], bins=min(6, max_step + 1), labels=False)

        pos_stats = []
        for (sb, lb), grp in sdf_lp.groupby(["step_bin", "logprob_bin"], observed=True):
            if len(grp) == 0:
                continue
            pos_stats.append({
                "step_bin": int(sb) if pd.notna(sb) else -1,
                "logprob_bin": int(lb) if pd.notna(lb) else -1,
                "n": len(grp),
                "success_rate": round(float(grp["success"].mean()), 4),
                "mean_logprob": round(float(grp["mean_logprob"].mean()), 6),
            })
        pos_df = pd.DataFrame(pos_stats)
        pos_df.to_csv(tables_dir / "step_position_stats.csv", index=False)
    else:
        pos_df = pd.DataFrame()
        print("  C3: position heatmap skipped – no token-level logprobs")

    if len(pos_df) > 1:
        pivot = pos_df.pivot_table(
            index="logprob_bin", columns="step_bin",
            values="success_rate", aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                        origin="lower")
        ax.set_xlabel("Step Position Bin")
        ax.set_ylabel("Log-Prob Bin (low → high)")
        ax.set_title("C3: Step Position × Confidence → Success Rate")
        fig.colorbar(im, ax=ax, label="Success Rate")
        fig.tight_layout()
        fig.savefig(plots_dir / "C3_step_position_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"  C3: plot → C3_step_position_heatmap.png")
    print(f"  C3: table → step_position_stats.csv")


# ── C4: Per-Mode Comparison ───────────────────────────────────────────────

def c4_per_mode(ep_df: pd.DataFrame, tables_dir: Path, plots_dir: Path) -> List[Dict[str, Any]]:
    """Violin + reliability diagram per observation_mode. Returns per-mode calibration list."""
    df = ep_df.dropna(subset=["ep_mean_logprob"])
    modes = sorted(df["observation_mode"].unique())
    if not modes:
        print("  C4: skipped – no data with confidence")
        return []

    # Summary table (token-level + verbalized + calibration metrics)
    summary_rows = []
    for mode in modes:
        grp = df[df["observation_mode"] == mode]
        row = {
            "observation_mode": mode,
            "n": len(grp),
            "success_rate": round(float(grp["success"].mean()), 4),
            "mean_logprob_mean": round(float(grp["ep_mean_logprob"].mean()), 6),
            "mean_logprob_std": round(float(grp["ep_mean_logprob"].std()), 6),
            "ep_prob_mean": round(float(grp["ep_prob"].mean()), 6),
        }
        # Token-level calibration per mode
        prob_vals = grp["ep_prob"].dropna().values
        labels_arr = grp["success"].astype(int).values
        if len(prob_vals) >= 4:
            ece, mce, brier = _compute_ece_mce_brier(prob_vals, labels_arr)
            row["ECE"] = round(ece, 6)
            row["MCE"] = round(mce, 6)
            row["Brier"] = round(brier, 6)
        else:
            row["ECE"] = np.nan
            row["MCE"] = np.nan
            row["Brier"] = np.nan

        # Verbalized calibration per mode
        verb_vals = grp["ep_mean_verbalized"].dropna()
        row["n_verbalized"] = len(verb_vals)
        row["mean_verbalized_mean"] = round(float(verb_vals.mean()), 6) if len(verb_vals) else np.nan
        row["mean_verbalized_std"] = round(float(verb_vals.std()), 6) if len(verb_vals) else np.nan
        if len(verb_vals) >= 4:
            v_labels = grp.loc[verb_vals.index, "success"].astype(int).values
            v_ece, v_mce, v_brier = _compute_ece_mce_brier(verb_vals.values, v_labels)
            row["verbalized_ECE"] = round(v_ece, 6)
            row["verbalized_MCE"] = round(v_mce, 6)
            row["verbalized_Brier"] = round(v_brier, 6)
        else:
            row["verbalized_ECE"] = np.nan
            row["verbalized_MCE"] = np.nan
            row["verbalized_Brier"] = np.nan

        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(tables_dir / "per_mode_summary.csv", index=False)
    print(f"  C4: table → per_mode_summary.csv")

    # Violin per mode
    fig, ax = plt.subplots(figsize=(7, 5))
    mode_data = [df.loc[df["observation_mode"] == m, "ep_mean_logprob"].dropna().values
                 for m in modes]
    non_empty = [(m, d) for m, d in zip(modes, mode_data) if len(d) > 0]
    if non_empty:
        parts = ax.violinplot([d for _, d in non_empty],
                              positions=range(len(non_empty)),
                              showmeans=True, showmedians=True)
        ax.set_xticks(range(len(non_empty)))
        ax.set_xticklabels([m for m, _ in non_empty])
    ax.set_ylabel("Episode Mean Log-Prob")
    ax.set_title("C4: Confidence Distribution per Mode")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "C4_per_mode_violin.png", dpi=150)
    plt.close(fig)

    # Reliability per mode
    n_modes = len(non_empty)
    if n_modes == 0:
        print("  C4: plots skipped – no data")
        return
    fig, axes = plt.subplots(1, max(n_modes, 1), figsize=(6 * max(n_modes, 1), 5),
                              squeeze=False)
    for idx, (mode, _) in enumerate(non_empty):
        ax = axes[0][idx]
        mdf = df[df["observation_mode"] == mode].dropna(subset=["ep_prob"])
        probs = mdf["ep_prob"].values
        labels_arr = mdf["success"].astype(int).values
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        pred_means, act_means = [], []
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
            if mask.sum() == 0:
                continue
            pred_means.append(float(probs[mask].mean()))
            act_means.append(float(labels_arr[mask].mean()))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        if pred_means:
            ax.plot(pred_means, act_means, "o-", color="#1f77b4")
        ax.set_title(f"{mode} (n={len(mdf)})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    fig.suptitle("C4: Reliability Diagram per Mode (Token-Level)", fontsize=14)
    fig.tight_layout()
    fig.savefig(plots_dir / "C4_per_mode_reliability.png", dpi=150)
    plt.close(fig)
    print(f"  C4: plots → C4_per_mode_violin.png, C4_per_mode_reliability.png")

    # ── Verbalized per mode (if available) ──
    verb_modes = [(m, df.loc[df["observation_mode"] == m, "ep_mean_verbalized"].dropna().values)
                  for m in modes]
    verb_non_empty = [(m, d) for m, d in verb_modes if len(d) > 0]
    if len(verb_non_empty) >= 1:
        # Violin
        fig, ax = plt.subplots(figsize=(7, 5))
        parts = ax.violinplot([d for _, d in verb_non_empty],
                              positions=range(len(verb_non_empty)),
                              showmeans=True, showmedians=True)
        ax.set_xticks(range(len(verb_non_empty)))
        ax.set_xticklabels([f"{m} (n={len(d)})" for m, d in verb_non_empty])
        ax.set_ylabel("Episode Mean Verbalized Confidence")
        ax.set_title("C4: Verbalized Confidence Distribution per Mode")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "C4_per_mode_verbalized_violin.png", dpi=150)
        plt.close(fig)

        # Reliability per mode (verbalized)
        fig, axes = plt.subplots(1, max(len(verb_non_empty), 1),
                                  figsize=(6 * max(len(verb_non_empty), 1), 5),
                                  squeeze=False)
        for idx, (mode, vdata) in enumerate(verb_non_empty):
            ax = axes[0][idx]
            mdf = df[df["observation_mode"] == mode].dropna(subset=["ep_mean_verbalized"])
            v_probs = mdf["ep_mean_verbalized"].values
            v_labels = mdf["success"].astype(int).values
            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            pred_means, act_means = [], []
            for i in range(n_bins):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                mask = (v_probs >= lo) & (v_probs < hi) if i < n_bins - 1 else (v_probs >= lo) & (v_probs <= hi)
                if mask.sum() == 0:
                    continue
                pred_means.append(float(v_probs[mask].mean()))
                act_means.append(float(v_labels[mask].mean()))
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            if pred_means:
                ax.plot(pred_means, act_means, "s-", color="#9467bd")
            ax.set_title(f"{mode} (n={len(mdf)})")
            ax.set_xlabel("Verbalized Confidence")
            ax.set_ylabel("Observed SR")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
        fig.suptitle("C4: Verbalized Reliability Diagram per Mode", fontsize=14)
        fig.tight_layout()
        fig.savefig(plots_dir / "C4_per_mode_verbalized_reliability.png", dpi=150)
        plt.close(fig)
        print(f"  C4: verbalized → C4_per_mode_verbalized_violin.png, "
              f"C4_per_mode_verbalized_reliability.png")
    else:
        print(f"  C4: verbalized per-mode skipped – no modes with verbalized data")

    return summary_rows


# ── Holm-Bonferroni correction ────────────────────────────────────────────

def _holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Return list of booleans indicating significance after Holm-Bonferroni correction."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        if p <= adjusted_alpha:
            significant[orig_idx] = True
        else:
            break  # all remaining are not significant
    return significant


# ── Bootstrap CI for AUROC ────────────────────────────────────────────────

def _auroc_bootstrap_ci(
    y_true: np.ndarray, y_score: np.ndarray,
    n_boot: int = 2000, ci: float = 0.95, seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (auroc_point, ci_lower, ci_upper) via bootstrap."""
    point = _auroc_safe(y_true, y_score)
    if math.isnan(point) or len(y_true) < 10:
        return (point, float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_aurocs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bt = y_true[idx]
        bs = y_score[idx]
        if len(np.unique(bt)) < 2:
            continue
        boot_aurocs.append(_auroc_safe(bt, bs))

    boot_aurocs = [a for a in boot_aurocs if not math.isnan(a)]
    if len(boot_aurocs) < 100:
        return (point, float("nan"), float("nan"))

    lo_pct = (1.0 - ci) / 2 * 100
    hi_pct = (1.0 - (1.0 - ci) / 2) * 100
    return (point, float(np.percentile(boot_aurocs, lo_pct)),
            float(np.percentile(boot_aurocs, hi_pct)))


# ── Optimal threshold (Youden's J) ───────────────────────────────────────

def _youden_j_search(
    y_true: np.ndarray, y_score: np.ndarray, n_thresholds: int = 200,
) -> Dict[str, Any]:
    """Inner Youden's J grid search — used both in-sample and inside LOO."""
    if len(np.unique(y_true)) < 2:
        return {"threshold": float("nan"), "sensitivity": float("nan"),
                "specificity": float("nan"), "f1": float("nan"),
                "youden_j": float("nan")}
    lo, hi = float(y_score.min()), float(y_score.max())
    if lo == hi:
        return {"threshold": lo, "sensitivity": 1.0, "specificity": 0.0,
                "f1": float("nan"), "youden_j": 0.0}
    thresholds = np.linspace(lo, hi, n_thresholds)
    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos = pos.sum()
    n_neg = neg.sum()
    best = {"threshold": float("nan"), "sensitivity": 0.0,
            "specificity": 0.0, "f1": 0.0, "youden_j": -1.0}
    for t in thresholds:
        pred_pos = y_score >= t
        tp = (pred_pos & pos).sum()
        fp = (pred_pos & neg).sum()
        tn = (~pred_pos & neg).sum()
        sens = tp / n_pos if n_pos else 0.0
        spec = tn / n_neg if n_neg else 0.0
        j = sens + spec - 1.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
        if j > best["youden_j"]:
            best = {"threshold": float(t), "sensitivity": float(sens),
                    "specificity": float(spec), "f1": float(f1),
                    "youden_j": float(j)}
    return best


def _optimal_threshold(
    y_true: np.ndarray, y_score: np.ndarray, n_thresholds: int = 200,
    *, loo: bool = True, n_bootstrap: int = 500, seed: int = 42,
) -> Dict[str, Any]:
    """Youden's J optimal threshold with cross-validated point estimate.

    Returns the same in-sample {threshold, sensitivity, specificity, f1,
    youden_j} keys for backward compatibility, plus optional CV diagnostics:
      - threshold_loo_mean / threshold_loo_std: LOO-CV averaged threshold +
        spread (out-of-sample)
      - sensitivity_loo / specificity_loo / youden_j_loo: averaged held-out
        performance (the honest estimate of how a deployed router will do)
      - threshold_ci_lower / threshold_ci_upper: bootstrap 95% CI of in-sample
        threshold (non-cv variability)
      - validation: "in_sample" if too few samples for CV, "loo_cv" otherwise

    The in-sample fields are kept because they were the previous output schema,
    but **callers should prefer `*_loo` for any go/no-go threshold deployment**
    — in-sample Youden's J is overfit to the data it was selected on.
    """
    in_sample = _youden_j_search(y_true, y_score, n_thresholds)
    n = len(y_true)
    if n < 20 or len(np.unique(y_true)) < 2 or not loo:
        in_sample.update({
            "threshold_loo_mean": float("nan"),
            "threshold_loo_std": float("nan"),
            "sensitivity_loo": float("nan"),
            "specificity_loo": float("nan"),
            "youden_j_loo": float("nan"),
            "threshold_ci_lower": float("nan"),
            "threshold_ci_upper": float("nan"),
            "validation": "in_sample",
        })
        return in_sample

    # LOO-CV: for each held-out i, fit threshold on rest, evaluate on i.
    rng = np.random.default_rng(seed)
    loo_thresholds = []
    loo_correct_pos = []  # per-fold sensitivity numerators
    loo_correct_neg = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        sub = _youden_j_search(y_true[mask], y_score[mask], n_thresholds)
        t = sub["threshold"]
        if math.isnan(t):
            continue
        loo_thresholds.append(t)
        # Apply held-out fold
        held_label = int(y_true[i])
        pred_pos = y_score[i] >= t
        if held_label == 1:
            loo_correct_pos.append(int(pred_pos))
        else:
            loo_correct_neg.append(int(not pred_pos))

    sens_loo = float(np.mean(loo_correct_pos)) if loo_correct_pos else float("nan")
    spec_loo = float(np.mean(loo_correct_neg)) if loo_correct_neg else float("nan")

    # Bootstrap CI of in-sample threshold (variability under resampling).
    boot_ts = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sub = _youden_j_search(y_true[idx], y_score[idx], n_thresholds)
        if not math.isnan(sub["threshold"]):
            boot_ts.append(sub["threshold"])
    if len(boot_ts) >= 100:
        ci_lo = float(np.percentile(boot_ts, 2.5))
        ci_hi = float(np.percentile(boot_ts, 97.5))
    else:
        ci_lo = ci_hi = float("nan")

    in_sample.update({
        "threshold_loo_mean": float(np.mean(loo_thresholds)) if loo_thresholds else float("nan"),
        "threshold_loo_std": float(np.std(loo_thresholds)) if loo_thresholds else float("nan"),
        "sensitivity_loo": sens_loo,
        "specificity_loo": spec_loo,
        "youden_j_loo": (
            sens_loo + spec_loo - 1.0
            if not (math.isnan(sens_loo) or math.isnan(spec_loo))
            else float("nan")
        ),
        "threshold_ci_lower": ci_lo,
        "threshold_ci_upper": ci_hi,
        "validation": "loo_cv",
    })
    return in_sample


# ── C5: Mode × Outcome Cross-Analysis ────────────────────────────────────

def c5_mode_outcome(
    ep_df: pd.DataFrame, tables_dir: Path, plots_dir: Path,
) -> Dict[str, Any]:
    """Mode × outcome cross-distribution + Kruskal-Wallis / Mann-Whitney."""
    df = ep_df.dropna(subset=["ep_mean_logprob"]).copy()
    df["group"] = df["observation_mode"] + "_" + df["success"].map({True: "success", False: "failure"})
    groups = sorted(df["group"].unique())

    if len(groups) < 2:
        print("  C5: skipped – fewer than 2 mode×outcome groups")
        return {"signal_mode_invariant": None}

    # Cross table
    cross_rows = []
    for g in groups:
        grp = df[df["group"] == g]
        cross_rows.append({
            "group": g,
            "n": len(grp),
            "mean": round(float(grp["ep_mean_logprob"].mean()), 6),
            "median": round(float(grp["ep_mean_logprob"].median()), 6),
            "std": round(float(grp["ep_mean_logprob"].std()), 6),
        })
    pd.DataFrame(cross_rows).to_csv(tables_dir / "mode_outcome_cross.csv", index=False)

    # Statistical tests — expanded to all signals
    all_c5_signals = list(METRICS_ALL) + list(BEHAVIORAL_LABELS.keys())
    negate_map_c5 = {
        "ep_mean_entropy": True, "ep_max_entropy": True,
        "url_revisit_count": True, "url_revisit_max": True,
        "url_unique_count": False, "action_unique_types": False,
        "action_diversity": False, "max_repeat_streak": True,
    }

    test_rows = []
    # Kruskal-Wallis across all groups (logprob only, for backward compat)
    group_vals = [df.loc[df["group"] == g, "ep_mean_logprob"].values for g in groups]
    group_vals_nonempty = [v for v in group_vals if len(v) >= 2]
    if len(group_vals_nonempty) >= 2:
        try:
            h_stat, h_p = sp_stats.kruskal(*group_vals_nonempty)
            test_rows.append({
                "test": "Kruskal-Wallis (all groups)",
                "signal": "ep_mean_logprob",
                "comparison": " vs ".join(groups),
                "statistic": round(float(h_stat), 4),
                "p_value": round(float(h_p), 6),
                "p_adjusted": np.nan,
                "significant_holm": None,
            })
        except ValueError:
            pass

    # Pairwise: same-outcome across modes, all signals
    modes = sorted(df["observation_mode"].unique())
    pairwise_p_values: List[float] = []
    pairwise_indices: List[int] = []

    for signal in all_c5_signals:
        if signal not in df.columns:
            continue
        for outcome in ["success", "failure"]:
            pairs = [(m1, m2) for i, m1 in enumerate(modes) for m2 in modes[i + 1:]]
            for m1, m2 in pairs:
                g1 = f"{m1}_{outcome}"
                g2 = f"{m2}_{outcome}"
                v1 = df.loc[df["group"] == g1, signal].dropna().values
                v2 = df.loc[df["group"] == g2, signal].dropna().values
                row_idx = len(test_rows)
                if len(v1) < 2 or len(v2) < 2:
                    test_rows.append({
                        "test": "Mann-Whitney",
                        "signal": signal,
                        "comparison": f"{g1} vs {g2}",
                        "statistic": np.nan, "p_value": np.nan,
                        "p_adjusted": np.nan, "significant_holm": None,
                    })
                    continue
                u, p = sp_stats.mannwhitneyu(v1, v2, alternative="two-sided")
                test_rows.append({
                    "test": "Mann-Whitney",
                    "signal": signal,
                    "comparison": f"{g1} vs {g2}",
                    "statistic": round(float(u), 2),
                    "p_value": round(float(p), 6),
                    "p_adjusted": np.nan,
                    "significant_holm": None,
                })
                pairwise_p_values.append(float(p))
                pairwise_indices.append(row_idx)

    # Holm-Bonferroni correction
    if pairwise_p_values:
        holm_sig = _holm_bonferroni(pairwise_p_values, alpha=0.05)
        # Compute adjusted p-values
        n_tests = len(pairwise_p_values)
        indexed_p = sorted(enumerate(pairwise_p_values), key=lambda x: x[1])
        adjusted_p = [0.0] * n_tests
        for rank, (orig_idx, p) in enumerate(indexed_p):
            adjusted_p[orig_idx] = min(p * (n_tests - rank), 1.0)
        # Write back
        for i, row_idx in enumerate(pairwise_indices):
            test_rows[row_idx]["p_adjusted"] = round(adjusted_p[i], 6)
            test_rows[row_idx]["significant_holm"] = holm_sig[i]

    mode_invariant = not any(
        r.get("significant_holm") is True
        for r in test_rows
        if r.get("test") == "Mann-Whitney" and r.get("signal") == "ep_mean_logprob"
    )

    pd.DataFrame(test_rows).to_csv(tables_dir / "mode_outcome_tests.csv", index=False)
    print(f"  C5: tables → mode_outcome_cross.csv, mode_outcome_tests.csv ({len(test_rows)} tests)")

    # ── Violin 2×2: rows=mode, cols=outcome ──
    outcomes = ["success", "failure"]
    fig, axes = plt.subplots(len(modes), len(outcomes),
                              figsize=(10, 4 * max(len(modes), 1)),
                              squeeze=False)
    for ri, mode in enumerate(modes):
        for ci, outcome in enumerate(outcomes):
            ax = axes[ri][ci]
            g = f"{mode}_{outcome}"
            vals = df.loc[df["group"] == g, "ep_mean_logprob"].dropna().values
            if len(vals) > 0:
                ax.violinplot([vals], positions=[0], showmeans=True, showmedians=True)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_title(f"{mode} / {outcome} (n={len(vals)})")
            ax.grid(alpha=0.3)
            if ri == len(modes) - 1:
                ax.set_xlabel("Mean Log-Prob")
    fig.suptitle("C5: Mode × Outcome Confidence Distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(plots_dir / "C5_mode_outcome_violin.png", dpi=150)
    plt.close(fig)

    # ── Ridge plot (overlaid KDEs) ──
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"success": "#2ca02c", "failure": "#d62728"}
    linestyles = {m: ls for m, ls in zip(modes, ["-", "--", ":", "-."])}
    for g in groups:
        vals = df.loc[df["group"] == g, "ep_mean_logprob"].dropna().values
        if len(vals) < 2:
            continue
        mode_part = g.rsplit("_", 1)[0]
        outcome_part = g.rsplit("_", 1)[1]
        try:
            kde = sp_stats.gaussian_kde(vals)
            x_range = np.linspace(vals.min() - 0.1, vals.max() + 0.1, 200)
            ax.plot(x_range, kde(x_range), color=colors.get(outcome_part, "gray"),
                    linestyle=linestyles.get(mode_part, "-"), linewidth=1.5,
                    label=g)
        except np.linalg.LinAlgError:
            continue
    ax.set_xlabel("Episode Mean Log-Prob")
    ax.set_ylabel("Density")
    ax.set_title("C5: Mode × Outcome Ridge Comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "C5_mode_outcome_ridge.png", dpi=150)
    plt.close(fig)
    print(f"  C5: plots → C5_mode_outcome_violin.png, C5_mode_outcome_ridge.png")

    return {"signal_mode_invariant": mode_invariant}


# ── C6: Behavioral Signals ───────────────────────────────────────────────

BEHAVIORAL_LABELS = {
    "url_revisit_count": "URL Revisit Count",
    "url_revisit_max": "Max Visits to Same URL",
    "url_unique_count": "Unique URLs Visited",
    "action_diversity": "Action Type Diversity",
    "action_unique_types": "Unique Action Types",
    "max_repeat_streak": "Max Repeat Streak",
}


def c6_behavioral(ep_df: pd.DataFrame, tables_dir: Path, plots_dir: Path):
    """Behavioral signal analysis: violin plots + AUROC bar chart vs token-level."""
    df = ep_df.copy()
    if len(df) < 4 or df["success"].nunique() < 2:
        print("  C6: skipped – too few episodes or single class")
        return

    succ = df[df["success"]]
    fail = df[~df["success"]]

    active_behavioral = [m for m in BEHAVIORAL_LABELS if m in df.columns and df[m].notna().sum() >= 4]
    if not active_behavioral:
        print("  C6: skipped – no behavioral signals with enough data")
        return

    # ── Stats table ──
    stat_rows = []
    for m in active_behavioral:
        for label, grp in [("success", succ), ("failure", fail)]:
            vals = grp[m].dropna()
            stat_rows.append({
                "metric": m, "outcome": label,
                "n": len(vals),
                "mean": round(float(vals.mean()), 4) if len(vals) else np.nan,
                "median": round(float(vals.median()), 4) if len(vals) else np.nan,
                "std": round(float(vals.std()), 4) if len(vals) else np.nan,
            })
    pd.DataFrame(stat_rows).to_csv(tables_dir / "behavioral_by_outcome.csv", index=False)

    # ── Wilcoxon for behavioral ──
    test_rows = []
    for m in active_behavioral:
        s_vals = succ[m].dropna().values
        f_vals = fail[m].dropna().values
        if len(s_vals) < 2 or len(f_vals) < 2:
            test_rows.append({"metric": m, "U": np.nan, "p_value": np.nan,
                              "rank_biserial": np.nan, "n_success": len(s_vals),
                              "n_failure": len(f_vals)})
            continue
        u, p = sp_stats.mannwhitneyu(s_vals, f_vals, alternative="two-sided")
        rb = _rank_biserial(u, len(s_vals), len(f_vals))
        test_rows.append({
            "metric": m, "U": round(float(u), 2),
            "p_value": round(float(p), 6),
            "rank_biserial": round(float(rb), 4),
            "n_success": len(s_vals), "n_failure": len(f_vals),
        })
    pd.DataFrame(test_rows).to_csv(tables_dir / "behavioral_wilcoxon.csv", index=False)
    print(f"  C6: tables → behavioral_by_outcome.csv, behavioral_wilcoxon.csv")

    # ── Violin plots ──
    n_metrics = len(active_behavioral)
    ncols = min(n_metrics, 2)
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_flat = np.atleast_1d(axes).flat if hasattr(np.atleast_1d(axes), "flat") else [axes]
    for idx, m in enumerate(active_behavioral):
        ax = axes_flat[idx]
        data = [succ[m].dropna().values, fail[m].dropna().values]
        if any(len(d) == 0 for d in data):
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(BEHAVIORAL_LABELS.get(m, m))
            continue
        parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)
        for i, color in enumerate(["#2ca02c", "#d62728"]):
            if i < len(parts["bodies"]):
                parts["bodies"][i].set_facecolor(color)
                parts["bodies"][i].set_alpha(0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Success", "Failure"])
        ax.set_title(BEHAVIORAL_LABELS.get(m, m))
        ax.grid(alpha=0.3)
    for idx in range(n_metrics, nrows * ncols):
        axes_flat[idx].set_visible(False)
    fig.suptitle("C6: Behavioral Signals by Outcome", fontsize=14)
    fig.tight_layout()
    fig.savefig(plots_dir / "C6_behavioral_violin.png", dpi=150)
    plt.close(fig)

    # ── AUROC bar chart: token-level vs behavioral ──
    auroc_path = tables_dir / "auroc_all_metrics.csv"
    if auroc_path.exists():
        auroc_df = pd.read_csv(auroc_path)
        auroc_valid = auroc_df.dropna(subset=["AUROC"])
        if len(auroc_valid) >= 2:
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = {"token_level": "#1f77b4", "behavioral": "#ff7f0e",
                      "verbalized": "#9467bd"}
            bars_x = range(len(auroc_valid))
            bar_colors = [colors.get(t, "gray") for t in auroc_valid["signal_type"]]
            ax.bar(bars_x, auroc_valid["AUROC"].values, color=bar_colors, alpha=0.8,
                   edgecolor="black")
            ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random (0.5)")
            ax.set_xticks(list(bars_x))
            ax.set_xticklabels(auroc_valid["metric"].values, rotation=30, ha="right",
                               fontsize=9)
            ax.set_ylabel("AUROC")
            ax.set_title("C6: AUROC Comparison — Token-Level vs Behavioral Signals")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3, axis="y")
            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor=colors["token_level"], label="Token-level"),
                Patch(facecolor=colors["behavioral"], label="Behavioral"),
            ]
            if "verbalized" in auroc_valid["signal_type"].values:
                legend_handles.append(
                    Patch(facecolor=colors["verbalized"], label="Verbalized"))
            legend_handles.append(
                plt.Line2D([0], [0], color="red", linestyle="--", label="Random baseline"))
            ax.legend(handles=legend_handles, loc="upper left")
            fig.tight_layout()
            fig.savefig(plots_dir / "C6_auroc_comparison.png", dpi=150)
            plt.close(fig)
            print(f"  C6: plots → C6_behavioral_violin.png, C6_auroc_comparison.png")
        else:
            print(f"  C6: AUROC bar chart skipped – fewer than 2 valid AUROC metrics")
    else:
        print(f"  C6: AUROC bar chart skipped – auroc_all_metrics.csv not found")


# ── C7: Cross-Mode AUROC Comparison ──────────────────────────────────────

def c7_cross_mode_auroc(ep_df: pd.DataFrame, tables_dir: Path, plots_dir: Path):
    """Grouped bar chart: AUROC per signal, grouped by observation mode."""
    modes = sorted(ep_df["observation_mode"].unique())
    if len(modes) < 2:
        print("  C7: skipped – need >= 2 modes for cross-mode comparison")
        return

    all_signals = list(METRICS_CORE) + list(METRICS_ENTROPY) + list(METRICS_VERBALIZED) + list(BEHAVIORAL_LABELS.keys())
    # Negate config: True = lower is better (negate for AUROC)
    negate_map = {
        "ep_mean_logprob": False, "ep_min_logprob": False,
        "ep_mean_margin": False, "ep_min_margin": False,
        "ep_mean_entropy": True, "ep_max_entropy": True,
        "ep_mean_verbalized": False, "ep_min_verbalized": False,
        "url_revisit_count": True, "url_revisit_max": True,
        "url_unique_count": False, "action_unique_types": False,
        "action_diversity": False, "max_repeat_streak": True,
    }
    signal_types = {}
    for s in METRICS_CORE:
        signal_types[s] = "token_level"
    for s in METRICS_ENTROPY:
        signal_types[s] = "token_level"
    for s in METRICS_VERBALIZED:
        signal_types[s] = "verbalized"
    for s in BEHAVIORAL_LABELS:
        signal_types[s] = "behavioral"

    rows = []
    for mode in modes:
        mdf = ep_df[ep_df["observation_mode"] == mode]
        if mdf["success"].nunique() < 2:
            continue
        y = mdf["success"].astype(int).values
        for sig in all_signals:
            if sig not in mdf.columns:
                continue
            valid = mdf[sig].dropna()
            if len(valid) < 4:
                continue
            scores = valid.values
            y_valid = mdf.loc[valid.index, "success"].astype(int).values
            if len(np.unique(y_valid)) < 2:
                continue
            if negate_map.get(sig, False):
                scores = -scores
            a, ci_lo, ci_hi = _auroc_bootstrap_ci(y_valid, scores)
            rows.append({
                "mode": mode, "signal": sig, "signal_type": signal_types.get(sig, ""),
                "AUROC": round(a, 4) if not math.isnan(a) else None,
                "AUROC_ci_lower": round(ci_lo, 4) if not math.isnan(ci_lo) else None,
                "AUROC_ci_upper": round(ci_hi, 4) if not math.isnan(ci_hi) else None,
                "n": len(valid),
            })

    if not rows:
        print("  C7: no valid AUROC data across modes")
        return

    cross_df = pd.DataFrame(rows).dropna(subset=["AUROC"])
    cross_df.to_csv(tables_dir / "cross_mode_auroc.csv", index=False)

    # Grouped bar chart
    signals_present = [s for s in all_signals if s in cross_df["signal"].values]
    x = np.arange(len(signals_present))
    width = 0.8 / len(modes)
    mode_colors = {"dom": "#1f77b4", "som": "#ff7f0e", "vision": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, mode in enumerate(modes):
        aurocs = []
        for sig in signals_present:
            row = cross_df[(cross_df["mode"] == mode) & (cross_df["signal"] == sig)]
            aurocs.append(float(row["AUROC"].iloc[0]) if len(row) else 0)
        ax.bar(x + i * width, aurocs, width, label=mode,
               color=mode_colors.get(mode, "gray"), alpha=0.8, edgecolor="black")

    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels(signals_present, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_title("C7: AUROC by Signal × Observation Mode")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # Dividers between signal type regions
    n_token = sum(1 for s in signals_present if signal_types.get(s) == "token_level")
    n_verb = sum(1 for s in signals_present if signal_types.get(s) == "verbalized")
    n_beh = sum(1 for s in signals_present if signal_types.get(s) == "behavioral")
    cursor = 0
    if n_token > 0:
        ax.text(cursor + n_token / 2 - 0.5, 0.95, "Token-level", ha="center",
                fontsize=8, color="gray", transform=ax.get_xaxis_transform())
        cursor += n_token
        if n_verb + n_beh > 0:
            ax.axvline(x=cursor - 0.5, color="gray", linestyle=":", alpha=0.5)
    if n_verb > 0:
        ax.text(cursor + n_verb / 2 - 0.5, 0.95, "Verbalized", ha="center",
                fontsize=8, color="gray", transform=ax.get_xaxis_transform())
        cursor += n_verb
        if n_beh > 0:
            ax.axvline(x=cursor - 0.5, color="gray", linestyle=":", alpha=0.5)
    if n_beh > 0:
        ax.text(cursor + n_beh / 2 - 0.5, 0.95, "Behavioral", ha="center",
                fontsize=8, color="gray", transform=ax.get_xaxis_transform())

    fig.tight_layout()
    fig.savefig(plots_dir / "C7_cross_mode_auroc.png", dpi=150)
    plt.close(fig)
    print(f"  C7: table → cross_mode_auroc.csv, plot → C7_cross_mode_auroc.png")


# ── C8: Behavioral Signal Accumulation (Earliest Routing Step) ───────────

def c8_signal_accumulation(
    step_records: List[Dict[str, Any]],
    summaries: Dict[Tuple[str, int], Dict[str, Any]],
    tables_dir: Path,
    plots_dir: Path,
):
    """AUROC of cumulative behavioral signals at each step position.

    Shows how early behavioral signals become discriminative,
    answering: 'at which step can we first reliably route?'
    """
    # Group steps by episode
    episodes: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for rec in step_records:
        key = (rec.get("condition_id", ""), int(rec.get("task_id", -1)))
        episodes[key].append(rec)

    # Get success labels
    ep_labels: Dict[Tuple[str, int], bool] = {}
    for key, steps in episodes.items():
        summary = summaries.get(key, {})
        if "success" in summary and summary["success"] is not None:
            ep_labels[key] = bool(summary["success"])
        else:
            last = sorted(steps, key=lambda s: s.get("step_idx", 0))[-1]
            ep_labels[key] = float(last.get("reward", 0)) > 0

    # Compute cumulative signals at each step cutoff
    max_step = 30
    signal_funcs = {
        "url_revisit_count": lambda urls: sum(v - 1 for v in urls.values() if v > 1),
        "url_revisit_max": lambda urls: max(urls.values()) if urls else 0,
        "action_diversity": lambda acts: len(set(acts)) / len(acts) if acts else 0,
        "max_repeat_streak": None,  # computed separately
        "mean_logprob_at_k": None,  # computed separately
        "max_entropy_at_k": None,   # computed separately
        "mean_verbalized_at_k": None,  # computed separately
    }

    cutoff_rows = []
    for cutoff in range(1, max_step + 1):
        labels = []
        signals: Dict[str, List[float]] = {s: [] for s in signal_funcs}

        for key, steps in episodes.items():
            if key not in ep_labels:
                continue
            sorted_steps = sorted(steps, key=lambda s: s.get("step_idx", 0))
            subset = [s for s in sorted_steps if s.get("step_idx", 0) < cutoff]
            if not subset:
                continue

            labels.append(int(ep_labels[key]))

            # URL revisits
            url_counts: Dict[str, int] = defaultdict(int)
            action_types: List[str] = []
            streak, max_str, last_sig = 0, 0, ""
            logprob_vals: List[float] = []
            entropy_vals: List[float] = []
            verbalized_vals: List[float] = []
            for s in subset:
                digest = s.get("state_digest") or {}
                url = str(s.get("obs_url", "") or digest.get("url_after", "") or "").strip()
                if url:
                    url_counts[url] += 1
                act = s.get("action") or {}
                atype = str(act.get("action_type", "") or "").lower()
                if atype:
                    action_types.append(atype)
                # 5-field signature (consistent with _build_episode_df)
                sig = "|".join([
                    atype,
                    str(act.get("element_id", "")),
                    str(act.get("text", ""))[:80],
                    str(act.get("coordinate", "")),
                    str(act.get("delta", "")),
                ])
                if sig == last_sig:
                    streak += 1
                else:
                    streak = 1
                    last_sig = sig
                max_str = max(max_str, streak)
                # Token/entropy/verbalized signals
                conf = s.get("confidence") or {}
                if "mean_logprob" in conf:
                    logprob_vals.append(conf["mean_logprob"])
                if "max_entropy" in conf:
                    entropy_vals.append(conf["max_entropy"])
                if "verbalized" in conf:
                    verbalized_vals.append(conf["verbalized"])

            signals["url_revisit_count"].append(
                sum(v - 1 for v in url_counts.values() if v > 1))
            signals["url_revisit_max"].append(
                max(url_counts.values()) if url_counts else 0)
            signals["action_diversity"].append(
                len(set(action_types)) / len(action_types) if action_types else 0)
            signals["max_repeat_streak"].append(max_str)
            # Token/entropy/verbalized cumulative signals
            signals["mean_logprob_at_k"].append(
                float(np.mean(logprob_vals)) if logprob_vals else np.nan)
            signals["max_entropy_at_k"].append(
                float(np.max(entropy_vals)) if entropy_vals else np.nan)
            signals["mean_verbalized_at_k"].append(
                float(np.mean(verbalized_vals)) if verbalized_vals else np.nan)

        if len(labels) < 10 or len(set(labels)) < 2:
            continue

        y = np.array(labels)
        negate_sigs = {"url_revisit_count", "url_revisit_max",
                       "max_repeat_streak", "max_entropy_at_k"}
        for sig_name, vals in signals.items():
            scores = np.array(vals, dtype=float)
            # Filter NaN for token/entropy/verbalized signals
            valid_mask = ~np.isnan(scores)
            if valid_mask.sum() < 10 or len(np.unique(y[valid_mask])) < 2:
                continue
            y_valid = y[valid_mask]
            scores_valid = scores[valid_mask]
            if sig_name in negate_sigs:
                scores_valid = -scores_valid
            a = _auroc_safe(y_valid, scores_valid)
            cutoff_rows.append({
                "step_cutoff": cutoff, "signal": sig_name,
                "AUROC": round(a, 4) if not math.isnan(a) else None,
                "n_episodes": int(valid_mask.sum()),
            })

    if not cutoff_rows:
        print("  C8: no data for accumulation analysis")
        return

    acc_df = pd.DataFrame(cutoff_rows).dropna(subset=["AUROC"])
    acc_df.to_csv(tables_dir / "signal_accumulation.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"url_revisit_count": "#1f77b4", "url_revisit_max": "#ff7f0e",
              "action_diversity": "#2ca02c", "max_repeat_streak": "#d62728",
              "mean_logprob_at_k": "#9467bd", "max_entropy_at_k": "#8c564b",
              "mean_verbalized_at_k": "#e377c2"}
    for sig in acc_df["signal"].unique():
        sdf = acc_df[acc_df["signal"] == sig].sort_values("step_cutoff")
        ax.plot(sdf["step_cutoff"], sdf["AUROC"], "o-", markersize=3,
                color=colors.get(sig, "gray"), label=sig, linewidth=1.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=0.6, color="red", linestyle=":", alpha=0.4, label="Routing threshold (0.6)")
    ax.set_xlabel("Step Cutoff (signals computed from steps 0..N-1)")
    ax.set_ylabel("AUROC")
    ax.set_title("C8: Behavioral Signal AUROC Accumulation — When Can We Route?")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0.3, 0.85)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "C8_signal_accumulation.png", dpi=150)
    plt.close(fig)
    print(f"  C8: table → signal_accumulation.csv, plot → C8_signal_accumulation.png")


# ── C9: Token vs Verbalized Comparison ───────────────────────────────────

def c9_token_vs_verbalized(
    ep_df: pd.DataFrame, tables_dir: Path, plots_dir: Path,
):
    """Compare token-level (ep_prob) vs verbalized confidence as predictors.

    Generates:
      - Scatter plot: ep_prob vs ep_mean_verbalized colored by success
      - Correlation table
      - AUROC comparison bar chart (token vs verbalized)
    """
    df = ep_df.dropna(subset=["ep_prob", "ep_mean_verbalized"]).copy()
    if len(df) < 10:
        print(f"  C9: skipped – only {len(df)} episodes with both token + verbalized")
        return

    probs = df["ep_prob"].values
    verb = df["ep_mean_verbalized"].values
    labels = df["success"].astype(int).values

    # ── Correlation table ──
    corr_rows = []
    # Spearman correlation between signals
    rho, rho_p = sp_stats.spearmanr(probs, verb)
    corr_rows.append({
        "comparison": "ep_prob vs ep_mean_verbalized",
        "spearman_rho": round(float(rho), 4),
        "p_value": round(float(rho_p), 6),
        "n": len(df),
    })
    # Per-outcome correlations
    for outcome, label_val in [("success", True), ("failure", False)]:
        sub = df[df["success"] == label_val]
        if len(sub) >= 4:
            r, p = sp_stats.spearmanr(sub["ep_prob"].values, sub["ep_mean_verbalized"].values)
            corr_rows.append({
                "comparison": f"ep_prob vs ep_mean_verbalized ({outcome})",
                "spearman_rho": round(float(r), 4),
                "p_value": round(float(p), 6),
                "n": len(sub),
            })
    pd.DataFrame(corr_rows).to_csv(tables_dir / "token_vs_verbalized_corr.csv", index=False)

    # ── AUROC comparison ──
    auroc_token = _auroc_safe(labels, probs)
    auroc_verb = _auroc_safe(labels, verb)
    auroc_rows = [
        {"signal": "ep_prob (token)", "AUROC": round(auroc_token, 4)
         if not math.isnan(auroc_token) else None, "n": len(df)},
        {"signal": "ep_mean_verbalized", "AUROC": round(auroc_verb, 4)
         if not math.isnan(auroc_verb) else None, "n": len(df)},
    ]
    # Also compare min_verbalized
    df_minv = ep_df.dropna(subset=["ep_prob", "ep_min_verbalized"])
    if len(df_minv) >= 10 and df_minv["success"].nunique() >= 2:
        a = _auroc_safe(df_minv["success"].astype(int).values,
                        df_minv["ep_min_verbalized"].values)
        auroc_rows.append({
            "signal": "ep_min_verbalized", "AUROC": round(a, 4)
            if not math.isnan(a) else None, "n": len(df_minv),
        })
    pd.DataFrame(auroc_rows).to_csv(tables_dir / "token_vs_verbalized_auroc.csv", index=False)
    print(f"  C9: tables → token_vs_verbalized_corr.csv, token_vs_verbalized_auroc.csv")

    # ── Scatter plot ──
    fig, ax = plt.subplots(figsize=(7, 6))
    succ_mask = labels == 1
    ax.scatter(probs[succ_mask], verb[succ_mask], c="#2ca02c", alpha=0.5,
               s=30, label="Success", edgecolors="none")
    ax.scatter(probs[~succ_mask], verb[~succ_mask], c="#d62728", alpha=0.5,
               s=30, label="Failure", edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("Token-Level Confidence (ep_prob)")
    ax.set_ylabel("Verbalized Confidence (ep_mean_verbalized)")
    ax.set_title(f"C9: Token vs Verbalized Confidence  "
                 f"(ρ={rho:.3f}, n={len(df)})")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "C9_token_vs_verbalized_scatter.png", dpi=150)
    plt.close(fig)

    # ── AUROC comparison bar chart ──
    auroc_vals = pd.DataFrame(auroc_rows).dropna(subset=["AUROC"])
    if len(auroc_vals) >= 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        bar_colors = ["#1f77b4", "#9467bd", "#9467bd"][:len(auroc_vals)]
        ax.bar(range(len(auroc_vals)), auroc_vals["AUROC"].values,
               color=bar_colors, alpha=0.8, edgecolor="black")
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random (0.5)")
        ax.set_xticks(range(len(auroc_vals)))
        ax.set_xticklabels(auroc_vals["signal"].values, fontsize=9)
        ax.set_ylabel("AUROC")
        ax.set_title("C9: AUROC — Token-Level vs Verbalized Confidence")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(plots_dir / "C9_token_vs_verbalized_auroc.png", dpi=150)
        plt.close(fig)
        print(f"  C9: plots → C9_token_vs_verbalized_scatter.png, "
              f"C9_token_vs_verbalized_auroc.png")
    else:
        print(f"  C9: plot → C9_token_vs_verbalized_scatter.png")


# ── C10: Composite Signal Exploration ─────────────────────────────────────

def c10_composite_signals(
    ep_df: pd.DataFrame, tables_dir: Path, plots_dir: Path,
):
    """Signal correlation matrix + composite AUROC grid search.

    1. Spearman correlation matrix across all signals → heatmap
    2. Top-2/Top-3 z-score weighted combinations → composite_auroc.csv
    """
    # Collect all numeric signal columns
    all_signals = [m for m in (list(METRICS_ALL) + list(BEHAVIORAL_LABELS.keys()))
                   if m in ep_df.columns and ep_df[m].notna().sum() >= 10]
    if len(all_signals) < 3 or ep_df["success"].nunique() < 2:
        print("  C10: skipped – too few signals or single class")
        return

    # ── 1. Spearman correlation matrix ──
    sig_df = ep_df[all_signals].copy()
    n_sig = len(all_signals)
    corr_matrix = np.full((n_sig, n_sig), np.nan)
    for i in range(n_sig):
        for j in range(i, n_sig):
            valid = sig_df[[all_signals[i], all_signals[j]]].dropna()
            if len(valid) < 5:
                continue
            rho, _ = sp_stats.spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    corr_df = pd.DataFrame(corr_matrix, index=all_signals, columns=all_signals)
    corr_df.to_csv(tables_dir / "signal_correlation_matrix.csv")
    print(f"  C10: signal_correlation_matrix.csv ({n_sig} signals)")

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(8, n_sig * 0.8), max(6, n_sig * 0.7)))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_sig))
    ax.set_yticks(range(n_sig))
    ax.set_xticklabels(all_signals, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(all_signals, fontsize=8)
    for i in range(n_sig):
        for j in range(n_sig):
            v = corr_matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("C10: Signal Correlation Matrix")
    fig.tight_layout()
    fig.savefig(plots_dir / "C10_signal_correlation.png", dpi=150)
    plt.close(fig)

    # ── 2. Composite AUROC grid search ──
    # Negate map: entropy and revisit signals need negation for AUROC
    negate_composite = {
        "ep_mean_entropy": True, "ep_max_entropy": True,
        "url_revisit_count": True, "url_revisit_max": True,
        "max_repeat_streak": True,
    }

    # Z-score standardize all signals (directionally aligned)
    z_scores: Dict[str, np.ndarray] = {}
    valid_mask = ep_df["success"].notna()
    for sig in all_signals:
        vals = ep_df[sig].values.copy().astype(float)
        if negate_composite.get(sig, False):
            vals = -vals
        mu = np.nanmean(vals)
        sd = np.nanstd(vals)
        if sd > 0:
            z_scores[sig] = (vals - mu) / sd
        else:
            z_scores[sig] = np.zeros_like(vals)

    y_all = ep_df["success"].astype(int).values
    weights = [0.0, 0.25, 0.5, 0.75, 1.0]

    composite_rows = []
    # Top-2 combinations. WARNING: this is an in-sample grid search across
    # C(n,2) * len(weights) combinations — the reported "best AUROC" is
    # overfit. For honest reporting we attach (a) the number of combinations
    # tested so a Bonferroni-style adjustment is possible, and (b) per-row
    # `validation = "in_sample"` so consumers can't accidentally take it as
    # a CV-validated number. For Phase 2 router design, perform a separate
    # holdout / k-fold validation on the top combinations.
    for i, s1 in enumerate(all_signals):
        for j, s2 in enumerate(all_signals):
            if j <= i:
                continue
            z1 = z_scores[s1]
            z2 = z_scores[s2]
            for w in weights:
                combined = w * z1 + (1 - w) * z2
                mask = ~(np.isnan(z1) | np.isnan(z2))
                if mask.sum() < 10 or len(np.unique(y_all[mask])) < 2:
                    continue
                a = _auroc_safe(y_all[mask], combined[mask])
                if not math.isnan(a):
                    composite_rows.append({
                        "combination": f"{s1}+{s2}",
                        "signals": f"{s1},{s2}",
                        "weights": f"{w:.2f},{1-w:.2f}",
                        "AUROC": round(a, 4),
                        "n": int(mask.sum()),
                        "validation": "in_sample",
                    })

    if composite_rows:
        n_combinations = len(composite_rows)
        comp_df = pd.DataFrame(composite_rows)
        comp_df = comp_df.sort_values("AUROC", ascending=False)
        # Annotate effective rank: top-K out of N combinations searched.
        # Useful for "best AUROC" interpretation (it's a max over many trials).
        comp_df["rank"] = range(1, len(comp_df) + 1)
        comp_df["n_combinations_searched"] = n_combinations
        comp_df.to_csv(tables_dir / "composite_auroc.csv", index=False)
        print(f"  C10: composite_auroc.csv ({n_combinations} combinations, "
              f"best={comp_df.iloc[0]['AUROC']:.4f} [{comp_df.iloc[0]['combination']}], "
              f"in-sample only — validate with holdout before Phase 2 deployment)")
    else:
        print("  C10: no valid composite combinations")

    print(f"  C10: plot → C10_signal_correlation.png")


# ── Routing Readiness Verdict ─────────────────────────────────────────────

def _routing_readiness(
    cov_df: pd.DataFrame,
    wilcoxon_path: Path,
    cal_metrics: Dict[str, Any],
    c5_result: Dict[str, Any],
    tables_dir: Path | None = None,
) -> Dict[str, Any]:
    """Compute routing readiness verdict (token-level + entropy + behavioral + verbalized)."""
    # Coverage: at least one condition with > 50% episode coverage
    max_cov = float(cov_df["episode_coverage"].max()) if len(cov_df) else 0.0
    sufficient_coverage = max_cov > 0.5

    # Discrimination (token-level, logprob/margin only): Wilcoxon p < 0.05 and |rank_biserial| > 0.2
    token_discriminative = False
    token_metrics = {"ep_mean_logprob", "ep_min_logprob", "ep_mean_margin", "ep_min_margin"}
    if wilcoxon_path.exists():
        wdf = pd.read_csv(wilcoxon_path)
        for _, row in wdf.iterrows():
            m = row.get("metric", "")
            if m not in token_metrics:
                continue
            p = row.get("p_value")
            rb = row.get("rank_biserial")
            if pd.notna(p) and pd.notna(rb) and p < 0.05 and abs(rb) > 0.2:
                token_discriminative = True
                break

    # Entropy discrimination (separate from token_discriminative)
    entropy_discriminative = False
    best_entropy_auroc = None
    best_entropy_metric = None

    # Behavioral discrimination: any behavioral AUROC > 0.6
    behavioral_discriminative = False
    best_behavioral_auroc = None
    best_behavioral_metric = None
    # Verbalized discrimination: any verbalized AUROC > 0.6
    verbalized_discriminative = False
    best_verbalized_auroc = None
    best_verbalized_metric = None

    auroc_path = (tables_dir / "auroc_all_metrics.csv") if tables_dir else None
    if auroc_path and auroc_path.exists():
        adf = pd.read_csv(auroc_path)

        # Entropy: check entropy-specific metrics
        entropy_metrics = {"ep_mean_entropy", "ep_max_entropy"}
        ent = adf[adf["metric"].isin(entropy_metrics)].dropna(subset=["AUROC"])
        if len(ent):
            best_idx = ent["AUROC"].idxmax()
            best_entropy_auroc = round(float(ent.loc[best_idx, "AUROC"]), 4)
            best_entropy_metric = str(ent.loc[best_idx, "metric"])
            if best_entropy_auroc > 0.6:
                entropy_discriminative = True

        beh = adf[adf["signal_type"] == "behavioral"].dropna(subset=["AUROC"])
        if len(beh):
            best_idx = beh["AUROC"].idxmax()
            best_behavioral_auroc = round(float(beh.loc[best_idx, "AUROC"]), 4)
            best_behavioral_metric = str(beh.loc[best_idx, "metric"])
            if best_behavioral_auroc > 0.6:
                behavioral_discriminative = True

        verb = adf[adf["signal_type"] == "verbalized"].dropna(subset=["AUROC"])
        if len(verb):
            best_idx = verb["AUROC"].idxmax()
            best_verbalized_auroc = round(float(verb.loc[best_idx, "AUROC"]), 4)
            best_verbalized_metric = str(verb.loc[best_idx, "metric"])
            if best_verbalized_auroc > 0.6:
                verbalized_discriminative = True

    # Calibration: ECE < 0.15
    ece = cal_metrics.get("ECE")
    calibrated = (ece is not None and not math.isnan(ece) and ece < 0.15)

    mode_invariant = c5_result.get("signal_mode_invariant")

    # Overall: (token OR entropy OR behavioral OR verbalized) AND coverage
    # AND (mode_invariant OR single-mode-run). Phase 2 router uses one
    # threshold across modes — if signal is mode-dependent, a single
    # threshold won't generalize. mode_invariant=None (single mode / not
    # tested) does NOT block readiness; only an explicit False does.
    discriminative = (token_discriminative or entropy_discriminative
                      or behavioral_discriminative or verbalized_discriminative)
    mode_ok = mode_invariant is not False  # True or None both pass
    overall = discriminative and sufficient_coverage and mode_ok

    return {
        "token_discriminative": token_discriminative,
        "entropy_discriminative": entropy_discriminative,
        "best_entropy_metric": best_entropy_metric,
        "best_entropy_auroc": best_entropy_auroc,
        "behavioral_discriminative": behavioral_discriminative,
        "best_behavioral_metric": best_behavioral_metric,
        "best_behavioral_auroc": best_behavioral_auroc,
        "verbalized_discriminative": verbalized_discriminative,
        "best_verbalized_metric": best_verbalized_metric,
        "best_verbalized_auroc": best_verbalized_auroc,
        "signal_discriminative": discriminative,
        "signal_calibrated": calibrated,
        "signal_sufficient_coverage": sufficient_coverage,
        "signal_mode_invariant": mode_invariant,
        "overall_usable": overall,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confidence calibration analysis for Phase 2 routing go/no-go",
    )
    parser.add_argument("--run-dir", required=True, help="Path to a phase1 run directory")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: <run_dir>/analysis/confidence)")
    parser.add_argument("--mode", default=None,
                        help="Filter to a single observation mode (dom/som/vision)")
    parser.add_argument("--no-adjust", action="store_true",
                        help="Disable adjusted labels (keep raw success as-is)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    mode_name = args.mode if args.mode else "combined"
    output_dir = (Path(args.output_dir) if args.output_dir
                  else run_dir / "analysis" / "signals" / mode_name)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    # Clean previous outputs to avoid stale files from prior runs.
    # Skip directories defensively (don't recurse into / delete subdirs).
    if tables_dir.exists():
        for f in tables_dir.iterdir():
            if f.is_file():
                f.unlink()
    if plots_dir.exists():
        for f in plots_dir.iterdir():
            if f.is_file():
                f.unlink()
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir : {run_dir}")
    print(f"Output  : {output_dir}")

    # Load data
    print("Loading step records …")
    step_records = _load_step_records(run_dir)
    summaries = _load_episode_summaries(run_dir)
    print(f"  {len(step_records)} step records, {len(summaries)} episode summaries")

    ep_df = _build_episode_df(step_records, summaries, run_dir=run_dir)
    step_df = _build_step_df(step_records)
    print(f"  {len(ep_df)} episodes, {len(step_df)} steps with confidence")

    # ── Adjusted labels ──
    use_adjusted = not args.no_adjust
    label_mode = "adjusted" if use_adjusted else "raw"
    if use_adjusted and not ep_df.empty:
        from p79.experiment.analysis import compute_adjusted_success_batch
        # Detect benchmark from run_dir path (visualwebarena vs webarena).
        # Previously hardcoded "visualwebarena" → wrong na_task_ids loaded for WA.
        benchmark = "webarena" if any(p == "webarena" for p in run_dir.parts) else "visualwebarena"
        # Detect benchmark_site from summaries
        sites = set()
        for (_, _), s in summaries.items():
            bs = s.get("benchmark_site") or s.get("site") or ""
            if bs:
                sites.add(bs)

        if not sites:
            # No site info — abort adjusting rather than silently using a
            # wrong default site (was: hardcoded fallback to 'classifieds').
            print("  ⚠ Cannot detect benchmark_site from summaries; skipping adjustment.")
            ep_df["raw_success"] = ep_df["success"]
            ep_df["adjusted_success"] = ep_df["success"]
            ep_df["fp_reason"] = ""
            label_mode = "raw_no_site_info"
        elif len(sites) == 1:
            bsite = next(iter(sites))
            ep_df["raw_success"] = ep_df["success"]
            compute_adjusted_success_batch(ep_df, bsite, benchmark)
            n_adjusted = int((ep_df["raw_success"] != ep_df["adjusted_success"]).sum())
            ep_df["success"] = ep_df["adjusted_success"]
            print(f"  Adjusted labels ({bsite}, benchmark={benchmark}): {n_adjusted} episodes changed")
        else:
            # Multi-site: per-site batch (was: hardcoded fallback to 'classifieds').
            print(f"  ⚠ Multi-site run detected: {sorted(sites)}. Adjusting per-site.")
            ep_df["raw_success"] = ep_df["success"]
            bs_map = {
                (s.get("condition_id", ""), int(s.get("task_id", -1))):
                (s.get("benchmark_site") or s.get("site") or "")
                for (_, _), s in summaries.items()
            }
            ep_df["_bsite"] = ep_df.apply(
                lambda r: bs_map.get((r["condition_id"], int(r["task_id"])), ""), axis=1
            )
            adj_parts = []
            for site in sorted(sites):
                site_ep = ep_df[ep_df["_bsite"] == site].copy()
                if site_ep.empty:
                    continue
                compute_adjusted_success_batch(site_ep, site, benchmark)
                adj_parts.append(site_ep[["adjusted_success", "fp_reason"]])
            if adj_parts:
                import pandas as _pd
                adj_combined = _pd.concat(adj_parts)
                ep_df["adjusted_success"] = adj_combined["adjusted_success"]
                ep_df["fp_reason"] = adj_combined["fp_reason"]
            n_adjusted = int((ep_df["raw_success"] != ep_df["adjusted_success"]).sum())
            ep_df["success"] = ep_df["adjusted_success"]
            ep_df.drop(columns=["_bsite"], inplace=True)
            print(f"  Adjusted labels (multi-site, benchmark={benchmark}): {n_adjusted} episodes changed")
    else:
        ep_df["raw_success"] = ep_df["success"]
        ep_df["adjusted_success"] = ep_df["success"]
        ep_df["fp_reason"] = ""

    if args.mode:
        print(f"  Filtering to mode: {args.mode}")
        ep_df = ep_df[ep_df["observation_mode"] == args.mode].copy()
        step_df = step_df[step_df["observation_mode"] == args.mode].copy()
        print(f"  After filter: {len(ep_df)} episodes, {len(step_df)} steps")

    # ── C0 ──
    print("\n── C0: Coverage ──")
    cov_df = c0_coverage(ep_df, tables_dir, step_df=step_df)
    low_cov = cov_df[cov_df["episode_coverage"] < 0.10]
    if len(low_cov):
        for _, r in low_cov.iterrows():
            print(f"  ⚠ {r['condition_id']}: coverage {r['episode_coverage']:.1%} < 10% – "
                  "excluded from downstream analysis")

    # Filter to conditions with >= 10% coverage for downstream
    good_conds = set(cov_df.loc[cov_df["episode_coverage"] >= 0.10, "condition_id"])
    ep_df_filt = ep_df[ep_df["condition_id"].isin(good_conds)].copy()
    step_df_filt = step_df[step_df["condition_id"].isin(good_conds)].copy()
    print(f"  Conditions passing 10% threshold: {sorted(good_conds) if good_conds else 'none'}")

    has_token_level = not ep_df_filt.dropna(subset=["ep_mean_logprob"]).empty
    has_verbalized = not ep_df_filt.dropna(subset=["ep_mean_verbalized"]).empty
    if not has_token_level and not has_verbalized:
        print("\n⚠ No episodes with any confidence data pass coverage threshold.")
        print("  Writing minimal summary and exiting.")
        summary = {
            "coverage": cov_df.to_dict(orient="records"),
            "discrimination": {},
            "calibration": {},
            "routing_readiness": {
                "signal_discriminative": False,
                "signal_calibrated": False,
                "signal_sufficient_coverage": False,
                "signal_mode_invariant": None,
                "overall_usable": False,
            },
        }
        with open(output_dir / "confidence_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nDone → {output_dir / 'confidence_summary.json'}")
        return
    if not has_token_level:
        print("\n  ℹ No token-level logprobs (API-based run); analyses will use verbalized + behavioral signals only.")

    # ── C1 ──
    print("\n── C1: Success vs Failure ──")
    c1_distribution(ep_df_filt, tables_dir, plots_dir)

    # ── C2 ──
    print("\n── C2: Reliability Diagram ──")
    cal_metrics = c2_calibration(ep_df_filt, tables_dir, plots_dir)

    # ── C3 ──
    print("\n── C3: Per-Step Trajectory ──")
    c3_trajectory(step_df_filt, ep_df_filt, tables_dir, plots_dir)

    # ── C4 ── (skip in single-mode runs)
    n_modes = ep_df_filt["observation_mode"].nunique()
    per_mode_calibration: List[Dict[str, Any]] = []
    if n_modes >= 2:
        print("\n── C4: Per-Mode Comparison ──")
        per_mode_calibration = c4_per_mode(ep_df_filt, tables_dir, plots_dir) or []
    else:
        print(f"\n── C4: skipped (single mode: {ep_df_filt['observation_mode'].unique()[0]})")

    # ── C5 ── (skip in single-mode runs)
    if n_modes >= 2:
        print("\n── C5: Mode × Outcome Cross-Analysis ──")
        c5_result = c5_mode_outcome(ep_df_filt, tables_dir, plots_dir)
    else:
        print(f"── C5: skipped (single mode)")
        c5_result = {"signal_mode_invariant": None}

    # ── C6 ──
    print("\n── C6: Behavioral Signals ──")
    c6_behavioral(ep_df_filt, tables_dir, plots_dir)

    # ── C7 ── (cross-mode comparison, needs >= 2 modes)
    if n_modes >= 2:
        print("\n── C7: Cross-Mode AUROC Comparison ──")
        c7_cross_mode_auroc(ep_df_filt, tables_dir, plots_dir)
    else:
        print("\n── C7: skipped (single mode)")

    # ── C8 ──
    print("\n── C8: Signal Accumulation (Earliest Routing Step) ──")
    # Use mode-filtered step records for C8
    if args.mode:
        filtered_steps = [r for r in step_records
                          if r.get("observation_mode") == args.mode]
    else:
        filtered_steps = step_records
    c8_signal_accumulation(filtered_steps, summaries, tables_dir, plots_dir)

    # ── C9 ──
    print("\n── C9: Token vs Verbalized Comparison ──")
    c9_token_vs_verbalized(ep_df_filt, tables_dir, plots_dir)

    # ── C10 ──
    print("\n── C10: Composite Signal Exploration ──")
    c10_composite_signals(ep_df_filt, tables_dir, plots_dir)

    # ── Routing Readiness ──
    print("\n── Routing Readiness Verdict ──")
    readiness = _routing_readiness(
        cov_df, tables_dir / "mannwhitney_test.csv", cal_metrics, c5_result,
        tables_dir=tables_dir,
    )
    for k, v in readiness.items():
        print(f"  {k}: {v}")

    # ── Write summary JSON ──
    summary: Dict[str, Any] = {
        "mode_filter": args.mode,
        "label_mode": label_mode,
        "n_adjusted": int((ep_df_filt["fp_reason"] != "").sum()) if "fp_reason" in ep_df_filt.columns else 0,
        "n_success_raw": int(ep_df_filt["raw_success"].sum()) if "raw_success" in ep_df_filt.columns else None,
        "n_success_adjusted": int(ep_df_filt["success"].sum()),
        "coverage": cov_df.to_dict(orient="records"),
        "calibration": cal_metrics,
        "routing_readiness": readiness,
        "per_mode_calibration": per_mode_calibration,
    }

    # Token-level discrimination (wilcoxon)
    wilcoxon_path = tables_dir / "mannwhitney_test.csv"
    if wilcoxon_path.exists():
        wdf = pd.read_csv(wilcoxon_path)
        summary["discrimination_token"] = wdf.to_dict(orient="records")

    # Behavioral discrimination (wilcoxon)
    beh_wilcoxon_path = tables_dir / "behavioral_wilcoxon.csv"
    if beh_wilcoxon_path.exists():
        bdf = pd.read_csv(beh_wilcoxon_path)
        summary["discrimination_behavioral"] = bdf.to_dict(orient="records")

    # Full AUROC table (token + behavioral)
    auroc_path = tables_dir / "auroc_all_metrics.csv"
    if auroc_path.exists():
        adf = pd.read_csv(auroc_path)
        summary["auroc_all"] = adf.to_dict(orient="records")

    # Cross-mode AUROC (combined mode only)
    cross_auroc_path = tables_dir / "cross_mode_auroc.csv"
    if cross_auroc_path.exists():
        cdf = pd.read_csv(cross_auroc_path)
        summary["cross_mode_auroc"] = cdf.to_dict(orient="records")

    # Token vs verbalized comparison
    tv_auroc_path = tables_dir / "token_vs_verbalized_auroc.csv"
    if tv_auroc_path.exists():
        tvdf = pd.read_csv(tv_auroc_path)
        summary["token_vs_verbalized"] = tvdf.to_dict(orient="records")
    tv_corr_path = tables_dir / "token_vs_verbalized_corr.csv"
    if tv_corr_path.exists():
        tcdf = pd.read_csv(tv_corr_path)
        summary["token_vs_verbalized_correlation"] = tcdf.to_dict(orient="records")

    with open(output_dir / "confidence_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nDone → {output_dir / 'confidence_summary.json'}")


if __name__ == "__main__":
    main()
