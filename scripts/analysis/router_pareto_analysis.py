#!/usr/bin/env python3
"""Post-hoc cost--success Pareto analysis of the 2026-07-15 router replay.

The input is an OFFLINE/NON-GATE replay artifact.  This producer recomputes
success rates and mean billed costs from counts/sums (or task records for routed
policies), reports standard Pareto dominance with ties preserved, and optionally
fits a separate six-head OOF binary-success LR variant for a cost-aware threshold
curve.  Nothing emitted here is H10-eligible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analysis.router_offline_replay import (  # noqa: E402
    DISPLAY_MODES,
    MODE_LABELS,
    build_offline_raw_features,
    collect_cell_outcomes,
    load_paper_grade_entries,
    policy_metrics,
    sha256_file,
)


DEFAULT_REPLAY = (
    REPO
    / "results/phantom_paper/l1_router_offline_20260715/router_offline_replay.json"
)
DEFAULT_OUT_DIR = DEFAULT_REPLAY.parent / "pareto"
SCHEMA_VERSION = "2026-07-15-router-pareto-posthoc-v1"
PROBABILITY_SCHEMA_VERSION = "2026-07-15-cost-aware-success-oof-v1"
ARTIFACT_STATUS = "OFFLINE/NON-GATE — POST-HOC EXPLORATORY"
DISCLAIMER = (
    "OFFLINE/NON-GATE POST-HOC EXPLORATORY ANALYSIS — reuses Pass-1 trajectories; "
    "not the preregistered live H10 gate"
)
DEFAULT_BOOTSTRAP_REPLICATES = 5_000
DEFAULT_BOOTSTRAP_SEED = 20260715
SUCCESS_HEAD_SEED = 42
ATOL = 1e-12


@dataclass(frozen=True)
class PolicyPoint:
    """One policy operating point; lower cost and higher SR are preferred."""

    policy_id: str
    label: str
    category: str
    mean_cost_usd: float
    success_rate_pct: float
    n_tasks: int
    n_success: int


DominanceRelation = Literal[
    "a_strictly_dominates_b",
    "a_weakly_dominates_b",
    "b_strictly_dominates_a",
    "b_weakly_dominates_a",
    "equivalent",
    "incomparable",
]


def dominance_relation(
    a: PolicyPoint,
    b: PolicyPoint,
    *,
    atol: float = ATOL,
) -> DominanceRelation:
    """Classify two points under lower-cost/higher-SR Pareto preference.

    ``strictly`` means strictly better on both axes.  ``weakly`` means no worse
    on both and strictly better on exactly one axis.  Exact (within ``atol``)
    two-axis ties are equivalent and neither point dominates the other.
    """

    cost_equal = math.isclose(a.mean_cost_usd, b.mean_cost_usd, abs_tol=atol, rel_tol=0.0)
    sr_equal = math.isclose(a.success_rate_pct, b.success_rate_pct, abs_tol=atol, rel_tol=0.0)
    if cost_equal and sr_equal:
        return "equivalent"

    a_cost_better = a.mean_cost_usd < b.mean_cost_usd - atol
    a_sr_better = a.success_rate_pct > b.success_rate_pct + atol
    b_cost_better = b.mean_cost_usd < a.mean_cost_usd - atol
    b_sr_better = b.success_rate_pct > a.success_rate_pct + atol

    a_no_worse = (a_cost_better or cost_equal) and (a_sr_better or sr_equal)
    b_no_worse = (b_cost_better or cost_equal) and (b_sr_better or sr_equal)
    if a_no_worse:
        return (
            "a_strictly_dominates_b"
            if a_cost_better and a_sr_better
            else "a_weakly_dominates_b"
        )
    if b_no_worse:
        return (
            "b_strictly_dominates_a"
            if b_cost_better and b_sr_better
            else "b_weakly_dominates_a"
        )
    return "incomparable"


def pareto_dominates(a: PolicyPoint, b: PolicyPoint, *, atol: float = ATOL) -> bool:
    """Return whether ``a`` Pareto-dominates ``b`` (one or two strict axes)."""

    return dominance_relation(a, b, atol=atol) in {
        "a_strictly_dominates_b",
        "a_weakly_dominates_b",
    }


def pareto_frontier(
    points: Iterable[PolicyPoint], *, atol: float = ATOL
) -> list[PolicyPoint]:
    """Return all non-dominated points, preserving exact two-axis ties."""

    rows = list(points)
    front = [
        point
        for idx, point in enumerate(rows)
        if not any(
            pareto_dominates(other, point, atol=atol)
            for jdx, other in enumerate(rows)
            if idx != jdx
        )
    ]
    return sorted(front, key=lambda point: (point.mean_cost_usd, -point.success_rate_pct, point.policy_id))


def select_cheapest_eligible_mode(
    probabilities: dict[str, float],
    mean_train_costs: dict[str, float],
    threshold: float,
    fallback_mode: str,
) -> tuple[str, bool]:
    """Pure threshold decision: cheapest mode with P(success)>=threshold."""

    order = {mode: idx for idx, mode in enumerate(DISPLAY_MODES)}
    eligible = [mode for mode in DISPLAY_MODES if probabilities[mode] >= threshold]
    if not eligible:
        return fallback_mode, True
    return min(eligible, key=lambda mode: (mean_train_costs[mode], order[mode])), False


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _metric_point(
    policy_id: str,
    label: str,
    category: str,
    metric: dict[str, Any],
) -> PolicyPoint:
    """Recompute a point from primitive counts and cost sum, checking cached fields."""

    n_tasks = int(metric["n_tasks"])
    n_success = int(metric["n_success"])
    if n_tasks <= 0 or not 0 <= n_success <= n_tasks:
        raise ValueError(f"Invalid metric counts for {policy_id}: {n_success}/{n_tasks}")
    sum_cost = float(metric["sum_total_billed_cost_usd"])
    if not math.isfinite(sum_cost) or sum_cost < 0:
        raise ValueError(f"Invalid cost sum for {policy_id}: {sum_cost!r}")
    sr_pct = 100.0 * n_success / n_tasks
    mean_cost = sum_cost / n_tasks
    cached_sr = metric.get("success_rate_pct")
    cached_cost = metric.get("mean_total_billed_cost_usd")
    if cached_sr is not None and not math.isclose(
        sr_pct, float(cached_sr), abs_tol=ATOL, rel_tol=0.0
    ):
        raise ValueError(f"Cached SR drift for {policy_id}: {cached_sr} != {sr_pct}")
    if cached_cost is not None and not math.isclose(
        mean_cost, float(cached_cost), abs_tol=ATOL, rel_tol=0.0
    ):
        raise ValueError(f"Cached mean-cost drift for {policy_id}: {cached_cost} != {mean_cost}")
    return PolicyPoint(policy_id, label, category, mean_cost, sr_pct, n_tasks, n_success)


def _task_record_point(
    policy_id: str,
    label: str,
    records: list[dict[str, Any]],
    expected_metric: dict[str, Any],
) -> PolicyPoint:
    oof = [row for row in records if row.get("prediction_status") == "oof"]
    if not oof:
        raise ValueError(f"No OOF task records for {policy_id}")
    task_ids = [int(row["task_id"]) for row in oof]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"Duplicate OOF task IDs for {policy_id}")
    for row in oof:
        if not isinstance(row.get("success"), bool):
            raise ValueError(f"Non-boolean success in {policy_id}/task {row.get('task_id')}")
    metric = {
        "n_tasks": len(oof),
        "n_success": sum(row["success"] is True for row in oof),
        "sum_total_billed_cost_usd": sum(float(row["total_billed_cost_usd"]) for row in oof),
    }
    point = _metric_point(policy_id, label, "router", metric)
    expected = _metric_point("expected_router", label, "router", expected_metric)
    if (
        point.n_tasks != expected.n_tasks
        or point.n_success != expected.n_success
        or not math.isclose(point.mean_cost_usd, expected.mean_cost_usd, abs_tol=ATOL, rel_tol=0.0)
    ):
        raise ValueError(f"Task-record aggregate drift for {policy_id}")
    return point


def _pairwise_relations(points: list[PolicyPoint]) -> list[dict[str, Any]]:
    return [
        {
            "policy_a": a.policy_id,
            "policy_b": b.policy_id,
            "relation": dominance_relation(a, b),
        }
        for a, b in combinations(points, 2)
    ]


def _dominators(points: list[PolicyPoint], target_id: str) -> list[dict[str, str]]:
    target = next(point for point in points if point.policy_id == target_id)
    rows = []
    for point in points:
        if point.policy_id == target_id:
            continue
        relation = dominance_relation(point, target)
        if relation == "a_strictly_dominates_b":
            rows.append({"policy_id": point.policy_id, "type": "strict"})
        elif relation == "a_weakly_dominates_b":
            rows.append({"policy_id": point.policy_id, "type": "weak"})
    return rows


def _analyze_point_set(
    refs: dict[str, Any],
    *,
    router_point: PolicyPoint | None = None,
) -> dict[str, Any]:
    fixed = [
        _metric_point(
            f"fixed_{mode}",
            f"Always {MODE_LABELS[mode]}",
            "fixed",
            refs["single_modes"][mode],
        )
        for mode in DISPLAY_MODES
    ]
    best = min(
        fixed,
        key=lambda point: (
            -point.success_rate_pct,
            point.mean_cost_usd,
            DISPLAY_MODES.index(point.policy_id.removeprefix("fixed_")),
        ),
    )
    stored_best = refs["best_single_mode"]["mode"]
    if best.policy_id != f"fixed_{stored_best}":
        raise ValueError(f"Best-single recomputation drift: {best.policy_id} != {stored_best}")
    oracle = _metric_point(
        "oracle", "Six-mode oracle", "hindsight_oracle", refs["six_mode_oracle_ceiling"]
    )
    deployable = fixed + ([router_point] if router_point is not None else [])
    augmented = deployable + [oracle]
    oracle_vs_fixed = []
    for point in fixed:
        relation = dominance_relation(oracle, point)
        oracle_vs_fixed.append({"fixed_policy_id": point.policy_id, "relation": relation})
    output = {
        "points": [asdict(point) for point in augmented],
        "best_single_policy_id": best.policy_id,
        "fixed_policy_frontier": [point.policy_id for point in pareto_frontier(fixed)],
        "deployable_frontier": [point.policy_id for point in pareto_frontier(deployable)],
        "hindsight_augmented_frontier": [point.policy_id for point in pareto_frontier(augmented)],
        "pairwise_relations": _pairwise_relations(augmented),
        "oracle_vs_fixed": oracle_vs_fixed,
        "oracle_strictly_dominates_all_fixed": all(
            row["relation"] == "a_strictly_dominates_b" for row in oracle_vs_fixed
        ),
        "oracle_pareto_dominates_all_fixed": all(
            row["relation"] in {"a_strictly_dominates_b", "a_weakly_dominates_b"}
            for row in oracle_vs_fixed
        ),
        "vision_vs_dom": dominance_relation(
            next(point for point in fixed if point.policy_id == "fixed_vision"),
            next(point for point in fixed if point.policy_id == "fixed_dom"),
        ),
    }
    if router_point is not None:
        output["router_dominators"] = _dominators(deployable, router_point.policy_id)
        output["router_on_deployable_frontier"] = router_point.policy_id in output["deployable_frontier"]
        output["oracle_vs_router"] = dominance_relation(oracle, router_point)
    else:
        output["router_dominators"] = []
        output["router_on_deployable_frontier"] = None
        output["oracle_vs_router"] = None
    return output


def _load_mode_task_rows(
    replay_cell: dict[str, Any], mode: str, task_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Load one fixed mode's task-aligned outcomes using replay provenance paths."""

    cell_id = str(replay_cell["cell_id"])
    site = cell_id.split("_", 1)[1]
    condition_dir = REPO / replay_cell["outcome_provenance"]["source_by_mode"][mode]["condition_dir"]
    successes: list[float] = []
    costs: list[float] = []
    for task_id in task_ids:
        path = condition_dir / "episodes" / f"{site}_task_{task_id}_summary_v2.json"
        record = json.loads(path.read_text())
        if int(record.get("task_id")) != task_id or not isinstance(record.get("success"), bool):
            raise ValueError(f"Invalid paired-bootstrap row: {path}")
        cost = float(record["total_billed_cost_usd"])
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(f"Invalid paired-bootstrap cost: {path}")
        successes.append(float(record["success"] is True))
        costs.append(cost)
    return np.asarray(successes), np.asarray(costs)


def paired_bootstrap_router_vs_best(
    replay_cell: dict[str, Any],
    refs: dict[str, Any],
    router_metric: dict[str, Any],
    *,
    n_replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Task-paired bootstrap of router minus subset-matched best single."""

    records = [
        row for row in replay_cell["task_records"] if row.get("prediction_status") == "oof"
    ]
    records.sort(key=lambda row: int(row["task_id"]))
    task_ids = [int(row["task_id"]) for row in records]
    if len(task_ids) != int(router_metric["n_tasks"]):
        raise ValueError("Bootstrap router task coverage does not match metric")
    router_success = np.asarray([float(row["success"] is True) for row in records])
    router_cost = np.asarray([float(row["total_billed_cost_usd"]) for row in records])
    best_mode = str(refs["best_single_mode"]["mode"])
    best_success, best_cost = _load_mode_task_rows(replay_cell, best_mode, task_ids)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(task_ids), size=(n_replicates, len(task_ids)))
    delta_sr_pp = 100.0 * (
        router_success[indices].mean(axis=1) - best_success[indices].mean(axis=1)
    )
    delta_cost = router_cost[indices].mean(axis=1) - best_cost[indices].mean(axis=1)
    best_no_worse = (delta_cost >= -ATOL) & (delta_sr_pp <= ATOL)
    best_at_least_one_strict = (delta_cost > ATOL) | (delta_sr_pp < -ATOL)
    best_strict_both = (delta_cost > ATOL) & (delta_sr_pp < -ATOL)
    router_no_worse = (delta_cost <= ATOL) & (delta_sr_pp >= -ATOL)
    router_at_least_one_strict = (delta_cost < -ATOL) | (delta_sr_pp > ATOL)
    router_strict_both = (delta_cost < -ATOL) & (delta_sr_pp > ATOL)

    def ci(values: np.ndarray) -> list[float]:
        return [float(x) for x in np.percentile(values, [2.5, 97.5])]

    point_delta_sr = 100.0 * float((router_success - best_success).mean())
    point_delta_cost = float((router_cost - best_cost).mean())
    return {
        "n_tasks": len(task_ids),
        "n_replicates": n_replicates,
        "seed": seed,
        "best_single_mode": best_mode,
        "delta_definition": "router minus best single",
        "delta_sr_pp": point_delta_sr,
        "delta_sr_pp_percentile_95_ci": ci(delta_sr_pp),
        "delta_mean_cost_usd": point_delta_cost,
        "delta_mean_cost_usd_percentile_95_ci": ci(delta_cost),
        "best_single_pareto_dominates_router_fraction": float(
            np.mean(best_no_worse & best_at_least_one_strict)
        ),
        "best_single_strictly_dominates_router_fraction": float(np.mean(best_strict_both)),
        "router_pareto_dominates_best_single_fraction": float(
            np.mean(router_no_worse & router_at_least_one_strict)
        ),
        "router_strictly_dominates_best_single_fraction": float(np.mean(router_strict_both)),
    }


def _reconstruct_entries(replay: dict[str, Any], cell_id: str) -> dict[str, dict[str, Any]]:
    manifest = REPO / replay["inputs"]["run_manifest"]
    return load_paper_grade_entries(manifest, [cell_id])[cell_id]


def build_cost_aware_success_curve(
    replay: dict[str, Any],
    replay_path: Path,
    *,
    cell_id: str = "B0_classifieds",
) -> dict[str, Any]:
    """Fit six OOF binary success heads and sweep a cheapest-eligible threshold.

    This is deliberately distinct from the locked multiclass router, whose
    probabilities estimate the cheapest-success *label* conditional on a task
    having at least one success.  Those probabilities are not P(success).
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from p79.policies.learned_router import (
        build_runtime_feature_vector,
        load_selected_idx_fold,
        load_vectorizer_fold,
    )
    from scripts.analysis.train_l1_router import LR_C, LR_MAX_ITER

    source_dir = replay_path.parent
    entries = _reconstruct_entries(replay, cell_id)
    phase1_root = REPO / replay["inputs"]["phase1_root"]
    outcomes, outcome_provenance = collect_cell_outcomes(cell_id, entries, phase1_root)
    fold_path = source_dir / f"{cell_id}_fold_assignment.json"
    fold_payload = json.loads(fold_path.read_text())
    fold_map = {int(task_id): int(fold) for task_id, fold in fold_payload["fold_assignment"].items()}
    if set(fold_map) != set(outcomes):
        raise ValueError("Cost-aware OOF fold map does not cover the full cell")

    site = cell_id.split("_", 1)[1]
    raw_by_task = {
        task_id: build_offline_raw_features(entries, phase1_root, site, task_id)[0]
        for task_id in sorted(outcomes)
    }
    probabilities: dict[int, dict[str, float]] = {task_id: {} for task_id in outcomes}
    decision_context: dict[int, dict[str, Any]] = {}
    fit_rows: list[dict[str, Any]] = []
    order = {mode: idx for idx, mode in enumerate(DISPLAY_MODES)}

    for fold_k in range(5):
        train_ids = sorted(task_id for task_id in outcomes if fold_map[task_id] != fold_k)
        holdout_ids = sorted(task_id for task_id in outcomes if fold_map[task_id] == fold_k)
        vectorizer = load_vectorizer_fold(source_dir, fold_k)
        selected_mask, _ = load_selected_idx_fold(source_dir, fold_k)
        if vectorizer is None or selected_mask is None:
            raise FileNotFoundError(f"Missing fold-local feature artifacts for fold {fold_k}")
        X_train = np.vstack(
            [build_runtime_feature_vector(raw_by_task[task_id], vectorizer, selected_mask) for task_id in train_ids]
        )
        X_holdout = np.vstack(
            [build_runtime_feature_vector(raw_by_task[task_id], vectorizer, selected_mask) for task_id in holdout_ids]
        )
        mean_train_costs = {
            mode: float(np.mean([outcomes[task_id][mode]["cost_usd"] for task_id in train_ids]))
            for mode in DISPLAY_MODES
        }
        train_srs = {
            mode: float(np.mean([outcomes[task_id][mode]["success"] for task_id in train_ids]))
            for mode in DISPLAY_MODES
        }
        fallback_mode = min(
            DISPLAY_MODES,
            key=lambda mode: (-train_srs[mode], mean_train_costs[mode], order[mode]),
        )
        decision_context[fold_k] = {
            "n_train": len(train_ids),
            "n_holdout": len(holdout_ids),
            "mean_train_cost_usd_by_mode": mean_train_costs,
            "train_success_rate_by_mode": train_srs,
            "fallback_best_single_mode": fallback_mode,
        }
        for mode in DISPLAY_MODES:
            y_train = np.asarray(
                [int(outcomes[task_id][mode]["success"] is True) for task_id in train_ids],
                dtype=int,
            )
            if len(np.unique(y_train)) == 1:
                success_prob = np.full(len(holdout_ids), float(y_train[0]))
                fit_kind = "constant_single_class"
            else:
                estimator = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                class_weight=None,
                                max_iter=LR_MAX_ITER,
                                C=LR_C,
                                solver="lbfgs",
                                random_state=SUCCESS_HEAD_SEED,
                            ),
                        ),
                    ]
                )
                estimator.fit(X_train, y_train)
                positive_idx = int(np.where(estimator.classes_ == 1)[0][0])
                success_prob = estimator.predict_proba(X_holdout)[:, positive_idx]
                fit_kind = "binary_logistic_regression"
            if not np.isfinite(success_prob).all() or np.any((success_prob < 0) | (success_prob > 1)):
                raise ValueError(f"Invalid OOF success probability for fold={fold_k}, mode={mode}")
            for task_id, probability in zip(holdout_ids, success_prob.tolist()):
                probabilities[task_id][mode] = float(probability)
            fit_rows.append(
                {
                    "fold_k": fold_k,
                    "mode": mode,
                    "fit_kind": fit_kind,
                    "n_train": len(train_ids),
                    "n_positive_train": int(y_train.sum()),
                    "n_holdout": len(holdout_ids),
                }
            )

    if any(set(row) != set(DISPLAY_MODES) for row in probabilities.values()):
        raise AssertionError("Incomplete six-mode OOF success probability matrix")
    thresholds = [round(float(value), 2) for value in np.linspace(0.0, 1.0, 21)]
    curve: list[dict[str, Any]] = []
    selections_by_threshold: dict[str, dict[str, str]] = {}
    for threshold in thresholds:
        selected: dict[int, str] = {}
        n_fallback = 0
        for task_id in sorted(outcomes):
            fold_k = fold_map[task_id]
            context = decision_context[fold_k]
            mode, fallback = select_cheapest_eligible_mode(
                probabilities[task_id],
                context["mean_train_cost_usd_by_mode"],
                threshold,
                context["fallback_best_single_mode"],
            )
            selected[task_id] = mode
            n_fallback += int(fallback)
        metric = policy_metrics(outcomes, selected)
        point = _metric_point(
            f"cost_aware_tau_{threshold:.2f}",
            f"Cost-aware τ={threshold:.2f}",
            "cost_aware_threshold",
            metric,
        )
        curve.append(
            {
                **asdict(point),
                "threshold": threshold,
                "fallback_count": n_fallback,
                "fallback_rate": n_fallback / len(outcomes),
                "selected_mode_counts": metric["selected_mode_counts"],
            }
        )
        selections_by_threshold[f"{threshold:.2f}"] = {
            str(task_id): mode for task_id, mode in sorted(selected.items())
        }
    curve_points = [PolicyPoint(**{key: row[key] for key in PolicyPoint.__dataclass_fields__}) for row in curve]
    frontier_ids = [point.policy_id for point in pareto_frontier(curve_points)]
    replay_cell = replay["cells"][cell_id]
    locked_router = _task_record_point(
        "router_oof",
        "OOF learned router",
        replay_cell["task_records"],
        replay_cell["offline_routed"],
    )
    base_analysis = _analyze_point_set(replay_cell["reference_points"], router_point=locked_router)
    base_points = [
        PolicyPoint(**point)
        for point in base_analysis["points"]
    ]
    deployable_points = [point for point in base_points if point.category != "hindsight_oracle"]
    oracle_point = next(point for point in base_points if point.category == "hindsight_oracle")
    combined_frontier_ids = [
        point.policy_id for point in pareto_frontier(deployable_points + curve_points)
    ]
    combined_hindsight_frontier_ids = [
        point.policy_id
        for point in pareto_frontier(deployable_points + curve_points + [oracle_point])
    ]

    return {
        "schema_version": PROBABILITY_SCHEMA_VERSION,
        "artifact_status": ARTIFACT_STATUS,
        "gate_eligible": False,
        "h10_eligible": False,
        "post_hoc_exploratory": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cell_id": cell_id,
        "probability_semantics": (
            "Six separate fold-held-out binary LogisticRegression estimates of "
            "P(Pass-1 success for mode | static task features); not the locked "
            "multiclass oracle-label probability and not calibrated live success."
        ),
        "decision_rule": (
            "Within each held-out task, choose the mode with lowest fold-training "
            "mean billed cost among modes with P(success)>=tau; if none qualify, "
            "use the fold-training best single mode."
        ),
        "protocol": {
            "outer_folds": 5,
            "fold_map_reused": str(fold_path.relative_to(REPO)),
            "fold_map_sha256": _sha256_json(fold_payload["fold_assignment"]),
            "feature_assets_reused": "fold-local vectorizer + MI-18 selector from locked router",
            "binary_heads": "StandardScaler + LogisticRegression, one head per mode per fold",
            "lr_C": LR_C,
            "lr_max_iter": LR_MAX_ITER,
            "solver": "lbfgs",
            "random_seed": SUCCESS_HEAD_SEED,
            "class_weight": None,
            "thresholds": thresholds,
            "threshold_comparator": ">=",
            "cost_for_decision": "fold-training mean total_billed_cost_usd by mode",
            "fallback": "fold-training best single by SR, then lower mean cost",
            "outcome_replay_cost": "selected held-out Pass-1 total_billed_cost_usd",
        },
        "inputs": {
            "replay_json": str(replay_path.relative_to(REPO)),
            "replay_json_sha256": sha256_file(replay_path),
            "outcome_provenance": outcome_provenance,
        },
        "fit_rows": fit_rows,
        "fold_decision_context": {str(key): value for key, value in decision_context.items()},
        "curve": curve,
        "curve_pareto_frontier": frontier_ids,
        "combined_deployable_frontier": combined_frontier_ids,
        "combined_hindsight_frontier": combined_hindsight_frontier_ids,
        "task_probability_records": [
            {
                "task_id": task_id,
                "fold_k": fold_map[task_id],
                "probability_success_by_mode": probabilities[task_id],
            }
            for task_id in sorted(probabilities)
        ],
        "selections_by_threshold": selections_by_threshold,
    }


def analyze_replay(
    replay: dict[str, Any],
    replay_path: Path,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    include_cost_aware: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if replay.get("artifact_status") != "OFFLINE/NON-GATE" or replay.get("gate_eligible") is not False:
        raise ValueError("Input must be the OFFLINE/NON-GATE, gate_eligible=false replay")
    cells: dict[str, Any] = {}
    for offset, (cell_id, replay_cell) in enumerate(replay["cells"].items()):
        refs = replay_cell["reference_points"]
        full_router_metric = replay_cell.get("offline_routed")
        router_point = None
        if full_router_metric is not None:
            router_point = _task_record_point(
                "router_oof", "OOF learned router", replay_cell["task_records"], full_router_metric
            )
        analysis = _analyze_point_set(refs, router_point=router_point)
        analysis.update(
            {
                "cell_id": cell_id,
                "training_status": replay_cell["training_status"],
                "cost_unit_basis": replay_cell["outcome_provenance"]["cost_unit_basis"],
                "primary_frontier_scope": (
                    "fixed_plus_full_oof_router" if router_point is not None else "fixed_only_no_full_oof_router"
                ),
                "primary_frontier": (
                    analysis["deployable_frontier"] if router_point is not None else analysis["fixed_policy_frontier"]
                ),
            }
        )
        if router_point is not None:
            analysis["paired_bootstrap_router_vs_best_single"] = paired_bootstrap_router_vs_best(
                replay_cell,
                refs,
                full_router_metric,
                n_replicates=bootstrap_replicates,
                seed=bootstrap_seed + offset,
            )
        else:
            analysis["paired_bootstrap_router_vs_best_single"] = None

        partial_metric = replay_cell.get("partial_oof_diagnostic")
        partial_refs = replay_cell.get("partial_oof_reference_points")
        if partial_metric is not None and partial_refs is not None:
            partial_router = _task_record_point(
                "router_partial_oof",
                "Partial OOF router (diagnostic)",
                replay_cell["task_records"],
                partial_metric,
            )
            partial = _analyze_point_set(partial_refs, router_point=partial_router)
            partial.update(
                {
                    "coverage_fraction": float(partial_metric["coverage_fraction"]),
                    "folds_ok": list(replay_cell["folds_ok"]),
                    "caveat": "4/5-fold partial OOF diagnostic only; not a full-cell estimate or gate result",
                    "paired_bootstrap_router_vs_best_single": paired_bootstrap_router_vs_best(
                        replay_cell,
                        partial_refs,
                        partial_metric,
                        n_replicates=bootstrap_replicates,
                        seed=bootstrap_seed + 100 + offset,
                    ),
                }
            )
            analysis["partial_oof_diagnostic"] = partial
        else:
            analysis["partial_oof_diagnostic"] = None
        cells[cell_id] = analysis

    cost_aware = (
        build_cost_aware_success_curve(replay, replay_path) if include_cost_aware else None
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_status": ARTIFACT_STATUS,
        "disclaimer": DISCLAIMER,
        "gate_eligible": False,
        "h10_eligible": False,
        "post_hoc_exploratory": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "definitions": {
            "cost": replay["cost_definition"],
            "strict_dominance": "lower cost AND higher SR",
            "weak_dominance": "no worse on both axes and strictly better on exactly one",
            "tie": "equal cost and SR; neither dominates, both remain on the frontier",
            "frontier": "points not strictly or weakly dominated by any other point in scope",
            "oracle_scope": "hindsight ceiling; excluded from deployable frontier",
        },
        "inputs": {
            "replay_json": str(replay_path.relative_to(REPO)),
            "replay_json_sha256": sha256_file(replay_path),
            "replay_schema_version": replay["schema_version"],
        },
        "bootstrap": {
            "task_paired": True,
            "n_replicates": bootstrap_replicates,
            "base_seed": bootstrap_seed,
        },
        "cells": cells,
        "cost_aware_variant": (
            None
            if cost_aware is None
            else {
                key: cost_aware[key]
                for key in (
                    "schema_version",
                    "artifact_status",
                    "cell_id",
                    "probability_semantics",
                    "decision_rule",
                    "protocol",
                    "curve",
                    "curve_pareto_frontier",
                    "combined_deployable_frontier",
                    "combined_hindsight_frontier",
                )
            }
        ),
    }
    return payload, cost_aware


def _front_labels(cell: dict[str, Any], key: str) -> str:
    labels = {point["policy_id"]: point["label"] for point in cell["points"]}
    return ", ".join(labels[policy_id] for policy_id in cell[key])


def _relation_text(relation: str, a_label: str, b_label: str) -> str:
    mapping = {
        "a_strictly_dominates_b": f"{a_label} strictly dominates {b_label}",
        "a_weakly_dominates_b": f"{a_label} weakly dominates {b_label}",
        "b_strictly_dominates_a": f"{b_label} strictly dominates {a_label}",
        "b_weakly_dominates_a": f"{b_label} weakly dominates {a_label}",
        "equivalent": f"{a_label} ties {b_label}",
        "incomparable": "incomparable",
    }
    return mapping[relation]


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {DISCLAIMER}",
        "",
        f"**Artifact status:** `{ARTIFACT_STATUS}` · `gate_eligible=false` · `h10_eligible=false`",
        "",
        "All SR and mean-cost values below were recomputed from primitive JSON counts/sums; routed points were additionally recomputed from per-task OOF records. Costs are never pooled across cost bases.",
        "",
        "## Frontier and headline checks",
        "",
        "| Cell | Primary frontier | Router conclusion | Oracle vs fixed | Vision vs DOM |",
        "|---|---|---|---|---|",
    ]
    for cell_id, cell in payload["cells"].items():
        router = "no full-cell router point"
        if cell["router_on_deployable_frontier"] is not None:
            dominators = ", ".join(
                f"{row['policy_id']} ({row['type']})" for row in cell["router_dominators"]
            ) or "none"
            router = (
                f"on frontier={str(cell['router_on_deployable_frontier']).lower()}; "
                f"dominators: {dominators}"
            )
        oracle = (
            "strictly dominates all six"
            if cell["oracle_strictly_dominates_all_fixed"]
            else "does not dominate every fixed mode"
        )
        lines.append(
            f"| {cell_id} | {_front_labels(cell, 'primary_frontier')} | {router} | "
            f"{oracle} | {cell['vision_vs_dom']} |"
        )

    for cell_id, cell in payload["cells"].items():
        labels = {point["policy_id"]: point["label"] for point in cell["points"]}
        lines.extend(
            [
                "",
                f"## {cell_id}",
                "",
                f"Training status: **{cell['training_status']}** · cost basis: `{cell['cost_unit_basis']}` · primary scope: `{cell['primary_frontier_scope']}`.",
                "",
                "| Policy | Type | n | Success | SR | Mean billed cost |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for point in cell["points"]:
            lines.append(
                f"| {point['label']} | {point['category']} | {point['n_tasks']} | "
                f"{point['n_success']} | {point['success_rate_pct']:.2f}% | "
                f"{point['mean_cost_usd']:.8f} |"
            )
        lines.extend(
            [
                "",
                f"- Fixed-policy frontier: **{_front_labels(cell, 'fixed_policy_frontier')}**.",
                f"- Deployable frontier: **{_front_labels(cell, 'deployable_frontier')}**.",
                f"- Hindsight-augmented frontier: **{_front_labels(cell, 'hindsight_augmented_frontier')}** (oracle is not deployable).",
                "",
                "Pairwise dominance table (incomparable pairs omitted):",
                "",
                "| Policy A | Policy B | Relation |",
                "|---|---|---|",
            ]
        )
        displayed = 0
        for relation in cell["pairwise_relations"]:
            if relation["relation"] == "incomparable":
                continue
            displayed += 1
            a_label = labels[relation["policy_a"]]
            b_label = labels[relation["policy_b"]]
            lines.append(
                f"| {a_label} | {b_label} | {_relation_text(relation['relation'], a_label, b_label)} |"
            )
        if displayed == 0:
            lines.append("| — | — | no dominance or exact ties |")

        boot = cell.get("paired_bootstrap_router_vs_best_single")
        if boot is not None:
            lines.extend(
                [
                    "",
                    f"Paired bootstrap ({boot['n_replicates']} task-level resamples, seed {boot['seed']}): router − best-single ΔSR = **{boot['delta_sr_pp']:+.2f} pp** (95% percentile CI {boot['delta_sr_pp_percentile_95_ci'][0]:+.2f}, {boot['delta_sr_pp_percentile_95_ci'][1]:+.2f}); Δcost = **{boot['delta_mean_cost_usd']:+.6f} USD/task** (CI {boot['delta_mean_cost_usd_percentile_95_ci'][0]:+.6f}, {boot['delta_mean_cost_usd_percentile_95_ci'][1]:+.6f}). Best-single Pareto-dominates router in **{boot['best_single_pareto_dominates_router_fraction']:.1%}** of resamples (strict on both axes: {boot['best_single_strictly_dominates_router_fraction']:.1%}).",
                ]
            )

        partial = cell.get("partial_oof_diagnostic")
        if partial is not None:
            partial_labels = {point["policy_id"]: point["label"] for point in partial["points"]}
            pboot = partial["paired_bootstrap_router_vs_best_single"]
            lines.extend(
                [
                    "",
                    "### Partial OOF diagnostic (not a cell estimate)",
                    "",
                    f"**Caveat:** {partial['caveat']}; folds={partial['folds_ok']}, coverage={partial['coverage_fraction']:.1%}.",
                    "",
                    "Deployable subset frontier: **"
                    + ", ".join(partial_labels[item] for item in partial["deployable_frontier"])
                    + "**.",
                    f"Router − subset-best ΔSR={pboot['delta_sr_pp']:+.2f} pp, Δcost={pboot['delta_mean_cost_usd']:+.6f}; router Pareto-dominates subset-best in {pboot['router_pareto_dominates_best_single_fraction']:.1%} of resamples.",
                ]
            )

    variant = payload.get("cost_aware_variant")
    lines.extend(["", "## Cost-aware OOF success-probability variant", ""])
    if variant is None:
        lines.append("Skipped by explicit CLI option; no threshold curve was produced.")
    else:
        lines.extend(
            [
                f"**POST-HOC only.** {variant['probability_semantics']}",
                "",
                variant["decision_rule"],
                "",
                "Combined fixed + locked-router + threshold-curve deployable frontier: **"
                + ", ".join(variant["combined_deployable_frontier"])
                + "**. Adding the hindsight oracle leaves **"
                + ", ".join(variant["combined_hindsight_frontier"])
                + "**.",
                "",
                "| τ | SR | Mean billed cost | Fallback | Selected modes |",
                "|---:|---:|---:|---:|---|",
            ]
        )
        for row in variant["curve"]:
            counts = ", ".join(f"{MODE_LABELS[mode]}={count}" for mode, count in row["selected_mode_counts"].items())
            marker = " **(curve frontier)**" if row["policy_id"] in variant["curve_pareto_frontier"] else ""
            lines.append(
                f"| {row['threshold']:.2f}{marker} | {row['success_rate_pct']:.2f}% | "
                f"{row['mean_cost_usd']:.8f} | {row['fallback_rate']:.1%} | {counts} |"
            )

    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- Oracle selection uses hindsight and is a ceiling; its location is useful for headroom, not deployability.",
            "- Router and threshold variants replay one realized Pass-1 trajectory per selected mode and omit serving overhead, fresh-state interaction, and trajectory stochasticity.",
            "- B1_classifieds partial OOF uses only four of five folds and is diagnostic-only. B2_classifieds and B1_reddit have fixed-policy frontiers only because no full router was trainable.",
            "- B0 is API USD; B1/B2 are electricity-derived USD. No cross-cell cost aggregation is valid.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-json", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--skip-cost-aware", action="store_true")
    args = parser.parse_args()
    replay_path = args.replay_json.resolve()
    out_dir = args.out_dir.resolve()
    if args.bootstrap_replicates < 2_000:
        parser.error("bootstrap-replicates must be >=2000 for this analysis package")
    if not replay_path.is_file():
        parser.error(f"Replay JSON not found: {replay_path}")
    replay = json.loads(replay_path.read_text())
    payload, probability_payload = analyze_replay(
        replay,
        replay_path,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        include_cost_aware=not args.skip_cost_aware,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "router_pareto_analysis.json"
    md_path = out_dir / "router_pareto_analysis.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    md_path.write_text(render_markdown(payload))
    if probability_payload is not None:
        probability_path = out_dir / "cost_aware_success_oof.json"
        probability_path.write_text(
            json.dumps(probability_payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        print(f"Wrote: {probability_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
