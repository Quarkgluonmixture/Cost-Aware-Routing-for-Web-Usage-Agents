#!/usr/bin/env python3
"""Confidence calibration analysis for Phase 2 routing go/no-go decision.

Analyses:
  C0 – Coverage report per condition
  C1 – Success vs failure distribution + Wilcoxon rank-sum
  C2 – Reliability diagram + ECE/MCE/Brier/AUROC
  C3 – Per-step trajectory + position heatmap
  C4 – Per-mode comparison (dom/som/vision)
  C5 – Mode × outcome cross-analysis

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
    """Load *_summary_v2.json, keyed by (condition_id, task_id)."""
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for path in run_dir.glob("*/episodes/*_summary_v2.json"):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        key = (d.get("condition_id", ""), int(d.get("task_id", -1)))
        out[key] = d
    return out


# ── Episode-level aggregation ─────────────────────────────────────────────

CONF_KEYS = ("mean_logprob", "min_logprob", "mean_margin", "min_margin")
EP_AGG_COLS = (
    "ep_mean_logprob", "ep_min_logprob", "ep_mean_margin", "ep_min_margin",
    "ep_last3_mean_logprob", "ep_prob",
)


def _build_episode_df(
    step_records: List[Dict[str, Any]],
    summaries: Dict[Tuple[str, int], Dict[str, Any]],
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
            means = [s["confidence"]["mean_logprob"] for s in conf_steps]
            mins = [s["confidence"]["min_logprob"] for s in conf_steps]
            margins_mean = [s["confidence"]["mean_margin"] for s in conf_steps]
            margins_min = [s["confidence"]["min_margin"] for s in conf_steps]

            row["ep_mean_logprob"] = float(np.mean(means))
            row["ep_min_logprob"] = float(np.min(mins))
            row["ep_mean_margin"] = float(np.mean(margins_mean))
            row["ep_min_margin"] = float(np.min(margins_min))

            # Last-3 steps
            last3 = means[-3:] if len(means) >= 3 else means
            row["ep_last3_mean_logprob"] = float(np.mean(last3))

            # Prob for reliability diagram
            row["ep_prob"] = float(np.exp(row["ep_mean_logprob"]))
        else:
            for col in EP_AGG_COLS:
                row[col] = np.nan

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
        })
    return pd.DataFrame(rows)


# ── C0: Coverage ──────────────────────────────────────────────────────────

def c0_coverage(ep_df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """Per-condition confidence coverage report."""
    groups = ep_df.groupby("condition_id")
    rows = []
    for cond, grp in groups:
        n_ep = len(grp)
        n_with = int((grp["conf_step_count"] > 0).sum())
        ep_cov = n_with / n_ep if n_ep else 0.0
        total_steps = int(grp["total_steps"].sum())
        total_conf_steps = int(grp["conf_step_count"].sum())
        step_cov = total_conf_steps / total_steps if total_steps else 0.0
        rows.append({
            "condition_id": cond,
            "episodes": n_ep,
            "episodes_with_confidence": n_with,
            "episode_coverage": round(ep_cov, 4),
            "total_steps": total_steps,
            "confidence_steps": total_conf_steps,
            "step_coverage": round(step_cov, 4),
        })
    cov_df = pd.DataFrame(rows)
    cov_df.to_csv(tables_dir / "confidence_coverage.csv", index=False)
    print(f"  C0: coverage table → {tables_dir / 'confidence_coverage.csv'}")
    return cov_df


# ── C1: Success vs Failure Distribution ───────────────────────────────────

METRICS_4 = ["ep_mean_logprob", "ep_min_logprob", "ep_mean_margin", "ep_min_margin"]
METRIC_LABELS = {
    "ep_mean_logprob": "Mean Log-Prob",
    "ep_min_logprob": "Min Log-Prob",
    "ep_mean_margin": "Mean Margin",
    "ep_min_margin": "Min Margin",
}


def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from Mann-Whitney U."""
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


def c1_distribution(ep_df: pd.DataFrame, tables_dir: Path, plots_dir: Path):
    """Success vs failure violin plots + Wilcoxon rank-sum tests."""
    df = ep_df.dropna(subset=METRICS_4)
    if len(df) < 4:
        print("  C1: skipped – too few episodes with confidence")
        return

    succ = df[df["success"]]
    fail = df[~df["success"]]

    # Stats table
    stat_rows = []
    for m in METRICS_4:
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
    for m in METRICS_4:
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
    pd.DataFrame(test_rows).to_csv(tables_dir / "wilcoxon_test.csv", index=False)
    print(f"  C1: tables → confidence_by_outcome.csv, wilcoxon_test.csv")

    # Violin plot 2×2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, m in zip(axes.flat, METRICS_4):
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
    fig.suptitle("C1: Confidence by Episode Outcome", fontsize=14)
    fig.tight_layout()
    fig.savefig(plots_dir / "C1_confidence_violin.png", dpi=150)
    plt.close(fig)
    print(f"  C1: plot → C1_confidence_violin.png")


# ── C2: Reliability Diagram + Calibration Metrics ─────────────────────────

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


def c2_calibration(
    ep_df: pd.DataFrame, tables_dir: Path, plots_dir: Path, n_bins: int = 10,
) -> Dict[str, Any]:
    """Reliability diagram + ECE/MCE/Brier/AUROC."""
    df = ep_df.dropna(subset=["ep_prob"]).copy()
    if len(df) < 4:
        print("  C2: skipped – too few episodes with ep_prob")
        return {}

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
    print(f"  C2: tables → calibration_bins.csv, calibration_metrics.csv")

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

    # ── Position heatmap ──
    # Bin step positions and mean_logprob → success rate
    sdf["logprob_bin"] = pd.cut(sdf["mean_logprob"], bins=5, labels=False)
    sdf["step_bin"] = pd.cut(sdf["step_idx"], bins=min(6, max_step + 1), labels=False)

    pos_stats = []
    for (sb, lb), grp in sdf.groupby(["step_bin", "logprob_bin"], observed=True):
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

def c4_per_mode(ep_df: pd.DataFrame, tables_dir: Path, plots_dir: Path):
    """Violin + reliability diagram per observation_mode."""
    df = ep_df.dropna(subset=["ep_mean_logprob"])
    modes = sorted(df["observation_mode"].unique())
    if not modes:
        print("  C4: skipped – no data with confidence")
        return

    # Summary table
    summary_rows = []
    for mode in modes:
        grp = df[df["observation_mode"] == mode]
        summary_rows.append({
            "observation_mode": mode,
            "n": len(grp),
            "success_rate": round(float(grp["success"].mean()), 4),
            "mean_logprob_mean": round(float(grp["ep_mean_logprob"].mean()), 6),
            "mean_logprob_std": round(float(grp["ep_mean_logprob"].std()), 6),
            "ep_prob_mean": round(float(grp["ep_prob"].mean()), 6),
        })
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
    fig.suptitle("C4: Reliability Diagram per Mode", fontsize=14)
    fig.tight_layout()
    fig.savefig(plots_dir / "C4_per_mode_reliability.png", dpi=150)
    plt.close(fig)
    print(f"  C4: plots → C4_per_mode_violin.png, C4_per_mode_reliability.png")


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

    # Statistical tests
    test_rows = []
    # Kruskal-Wallis across all groups
    group_vals = [df.loc[df["group"] == g, "ep_mean_logprob"].values for g in groups]
    group_vals_nonempty = [v for v in group_vals if len(v) >= 2]
    if len(group_vals_nonempty) >= 2:
        try:
            h_stat, h_p = sp_stats.kruskal(*group_vals_nonempty)
            test_rows.append({
                "test": "Kruskal-Wallis (all groups)",
                "comparison": " vs ".join(groups),
                "statistic": round(float(h_stat), 4),
                "p_value": round(float(h_p), 6),
            })
        except ValueError:
            pass

    # Pairwise: same-outcome across modes
    modes = sorted(df["observation_mode"].unique())
    mode_invariant = True  # assume invariant until proven otherwise
    for outcome in ["success", "failure"]:
        pairs = [(m1, m2) for i, m1 in enumerate(modes) for m2 in modes[i + 1:]]
        for m1, m2 in pairs:
            g1 = f"{m1}_{outcome}"
            g2 = f"{m2}_{outcome}"
            v1 = df.loc[df["group"] == g1, "ep_mean_logprob"].values
            v2 = df.loc[df["group"] == g2, "ep_mean_logprob"].values
            if len(v1) < 2 or len(v2) < 2:
                test_rows.append({
                    "test": "Mann-Whitney",
                    "comparison": f"{g1} vs {g2}",
                    "statistic": np.nan, "p_value": np.nan,
                })
                continue
            u, p = sp_stats.mannwhitneyu(v1, v2, alternative="two-sided")
            test_rows.append({
                "test": "Mann-Whitney",
                "comparison": f"{g1} vs {g2}",
                "statistic": round(float(u), 2),
                "p_value": round(float(p), 6),
            })
            if p < 0.05:
                mode_invariant = False

    pd.DataFrame(test_rows).to_csv(tables_dir / "mode_outcome_tests.csv", index=False)
    print(f"  C5: tables → mode_outcome_cross.csv, mode_outcome_tests.csv")

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


# ── Routing Readiness Verdict ─────────────────────────────────────────────

def _routing_readiness(
    cov_df: pd.DataFrame,
    wilcoxon_path: Path,
    cal_metrics: Dict[str, Any],
    c5_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute routing readiness verdict."""
    # Coverage: at least one condition with > 50% episode coverage
    max_cov = float(cov_df["episode_coverage"].max()) if len(cov_df) else 0.0
    sufficient_coverage = max_cov > 0.5

    # Discrimination: Wilcoxon p < 0.05 and |rank_biserial| > 0.2
    discriminative = False
    if wilcoxon_path.exists():
        wdf = pd.read_csv(wilcoxon_path)
        for _, row in wdf.iterrows():
            p = row.get("p_value")
            rb = row.get("rank_biserial")
            if pd.notna(p) and pd.notna(rb) and p < 0.05 and abs(rb) > 0.2:
                discriminative = True
                break

    # Calibration: ECE < 0.15
    ece = cal_metrics.get("ECE")
    calibrated = (ece is not None and not math.isnan(ece) and ece < 0.15)

    mode_invariant = c5_result.get("signal_mode_invariant")

    overall = discriminative and calibrated and sufficient_coverage

    return {
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
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis" / "confidence"
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir : {run_dir}")
    print(f"Output  : {output_dir}")

    # Load data
    print("Loading step records …")
    step_records = _load_step_records(run_dir)
    summaries = _load_episode_summaries(run_dir)
    print(f"  {len(step_records)} step records, {len(summaries)} episode summaries")

    ep_df = _build_episode_df(step_records, summaries)
    step_df = _build_step_df(step_records)
    print(f"  {len(ep_df)} episodes, {len(step_df)} steps with confidence")

    # ── C0 ──
    print("\n── C0: Coverage ──")
    cov_df = c0_coverage(ep_df, tables_dir)
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

    if ep_df_filt.dropna(subset=["ep_mean_logprob"]).empty:
        print("\n⚠ No episodes with confidence data pass coverage threshold.")
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

    # ── C1 ──
    print("\n── C1: Success vs Failure ──")
    c1_distribution(ep_df_filt, tables_dir, plots_dir)

    # ── C2 ──
    print("\n── C2: Reliability Diagram ──")
    cal_metrics = c2_calibration(ep_df_filt, tables_dir, plots_dir)

    # ── C3 ──
    print("\n── C3: Per-Step Trajectory ──")
    c3_trajectory(step_df_filt, ep_df_filt, tables_dir, plots_dir)

    # ── C4 ──
    print("\n── C4: Per-Mode Comparison ──")
    c4_per_mode(ep_df_filt, tables_dir, plots_dir)

    # ── C5 ──
    print("\n── C5: Mode × Outcome Cross-Analysis ──")
    c5_result = c5_mode_outcome(ep_df_filt, tables_dir, plots_dir)

    # ── Routing Readiness ──
    print("\n── Routing Readiness Verdict ──")
    readiness = _routing_readiness(
        cov_df, tables_dir / "wilcoxon_test.csv", cal_metrics, c5_result,
    )
    for k, v in readiness.items():
        print(f"  {k}: {v}")

    # ── Write summary JSON ──
    summary = {
        "coverage": cov_df.to_dict(orient="records"),
        "discrimination": {},
        "calibration": cal_metrics,
        "routing_readiness": readiness,
    }
    # Add discrimination details if wilcoxon exists
    wilcoxon_path = tables_dir / "wilcoxon_test.csv"
    if wilcoxon_path.exists():
        wdf = pd.read_csv(wilcoxon_path)
        summary["discrimination"] = wdf.to_dict(orient="records")

    with open(output_dir / "confidence_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nDone → {output_dir / 'confidence_summary.json'}")


if __name__ == "__main__":
    main()
