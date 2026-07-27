#!/usr/bin/env python3
"""Replay fold-held-out learned-router choices against canonical Pass-1 outcomes.

This producer is deliberately descriptive.  It does not serve a router, measure
router overhead/latency, or implement the preregistered H10 live Pass-2 gate.  The
reported routed cost is the mean Pass-1 ``total_billed_cost_usd`` of the episode
selected for each task by the OOF policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from p79.policies.learned_router import (
    LearnedRouterArtifactError,
    extract_raw_features,
    predict_mode_fold_aware,
)
from p79.policies.router_features import (
    derive_oracle_label,
    difficulty_to_int,
    estimate_input_tokens,
)
try:
    from scripts.analysis.lib.canonical_task_universe import (
        expected_scored_ids,
        protocol_excluded_in_universe,
    )
except ModuleNotFoundError:  # Direct ``python scripts/analysis/...`` execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.analysis.lib.canonical_task_universe import (
        expected_scored_ids,
        protocol_excluded_in_universe,
    )


REPO = Path(__file__).resolve().parents[2]
PHASE1_ROOT = REPO / "results/visualwebarena/phase1"
DEFAULT_OUT_DIR = REPO / "results/phantom_paper/l1_router_offline_20260715"
DEFAULT_RUN_MANIFEST = REPO / "results/phantom_paper/run_manifest.yaml"
FORBIDDEN_CANONICAL_OUT = (REPO / "results/phantom_paper/l1_router").resolve()

SCHEMA_VERSION = "2026-07-15-router-offline-replay-v1"
ARTIFACT_STATUS = "OFFLINE/NON-GATE"
DISCLAIMER = (
    "OFFLINE REPLAY — NOT the preregistered H10 operational gate "
    "(no live serving cost/latency; simulated from Pass-1 outcomes)"
)
FALLBACK_MODE = "phantom_som"
DISPLAY_MODES = [
    "dom",
    "som",
    "vision",
    "phantom_text",
    "phantom_prompt",
    "phantom_som",
]
MODE_LABELS = {
    "dom": "DOM",
    "som": "SoM",
    "vision": "Vision",
    "phantom_text": "P-text",
    "phantom_prompt": "P-prompt",
    "phantom_som": "P-SoM",
}
DEFAULT_CELLS = ["B0_classifieds", "B1_classifieds", "B2_classifieds", "B1_reddit"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_map_sha256(fold_assignment: dict[str | int, int]) -> str:
    """Hash only task->fold content, independent of cell metadata/JSON whitespace."""
    canonical = {str(int(tid)): int(fold) for tid, fold in fold_assignment.items()}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_mode(raw: str) -> str:
    token = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "dom": "dom",
        "som": "som",
        "vision": "vision",
        "p_text": "phantom_text",
        "phantom_text": "phantom_text",
        "phantom_dom": "phantom_text",
        "p_prompt": "phantom_prompt",
        "phantom_prompt": "phantom_prompt",
        "p_som": "phantom_som",
        "phantom_som": "phantom_som",
    }
    if token not in aliases:
        raise ValueError(f"Unknown observation mode in run manifest: {raw!r}")
    return aliases[token]


def _load_manifest_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
    else:
        payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Run manifest must be a mapping: {path}")
    return payload


def load_paper_grade_entries(
    manifest_path: Path,
    requested_cells: Iterable[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return exact paper-grade run/condition entries, one per cell and mode."""
    payload = _load_manifest_payload(manifest_path)
    requested = list(requested_cells)
    wanted = set(requested)
    result: dict[str, dict[str, dict[str, Any]]] = {cell: {} for cell in requested}

    if "cells" in payload:
        rows = payload["cells"]
        if not isinstance(rows, list):
            raise ValueError("run_manifest.yaml 'cells' must be a list")
        for row in rows:
            if not isinstance(row, dict) or row.get("grade") != "paper-grade":
                continue
            cell_id = f"{row.get('baseline')}_{row.get('site')}"
            if cell_id not in wanted:
                continue
            mode = normalize_mode(row.get("mode", ""))
            if mode in result[cell_id]:
                raise ValueError(f"Duplicate paper-grade {cell_id}/{mode} manifest entry")
            if not row.get("run_dir") or not row.get("condition_subdir"):
                raise ValueError(f"Incomplete paper-grade manifest entry: {cell_id}/{mode}")
            result[cell_id][mode] = {
                "run_dir": str(row["run_dir"]),
                "condition_subdir": str(row["condition_subdir"]),
                "manifest_mode": str(row.get("mode")),
                "grade": "paper-grade",
            }
    elif "pass1" in payload:
        raise ValueError(
            "The compact pass1 JSON whitelist lacks condition_subdir values; "
            "use results/phantom_paper/run_manifest.yaml for offline replay."
        )
    else:
        raise ValueError(f"Unsupported run-manifest schema: {manifest_path}")

    for cell_id, modes in result.items():
        missing = [mode for mode in DISPLAY_MODES if mode not in modes]
        extra = sorted(set(modes) - set(DISPLAY_MODES))
        if missing or extra:
            raise ValueError(
                f"{cell_id} does not have exactly six paper-grade modes; "
                f"missing={missing}, extra={extra}"
            )
    return result


def collect_cell_outcomes(
    cell_id: str,
    entries: dict[str, dict[str, Any]],
    phase1_root: Path,
) -> tuple[dict[int, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Read exact run-manifest condition dirs and fail closed on universe/cost drift."""
    _baseline, site = cell_id.split("_", 1)
    expected_ids, expected_sha = expected_scored_ids(site)
    outcomes: dict[int, dict[str, dict[str, Any]]] = {}
    source_rows: dict[str, Any] = {}
    basis_counts: Counter[str] = Counter()

    for mode in DISPLAY_MODES:
        entry = entries[mode]
        condition_dir = phase1_root / entry["run_dir"] / entry["condition_subdir"]
        episode_dir = condition_dir / "episodes"
        if not episode_dir.is_dir():
            raise FileNotFoundError(f"Missing paper-grade episode dir: {episode_dir}")
        seen: set[int] = set()
        for summary_path in sorted(episode_dir.glob(f"{site}_task_*_summary_v2.json")):
            record = json.loads(summary_path.read_text())
            if "success" not in record:
                raise ValueError(f"Missing success in {summary_path}")
            if record.get("total_billed_cost_usd") is None:
                raise ValueError(f"Missing canonical total_billed_cost_usd in {summary_path}")
            task_id = int(record["task_id"])
            if task_id in seen:
                raise ValueError(f"Duplicate task summary for {cell_id}/{mode}/task {task_id}")
            seen.add(task_id)
            # B-1911 (/stress Mode B codex follow-up, 2026-07-27): AMENDMENT_08
            # keeps the runner COLLECTING the protocol-excluded tasks, so a
            # landed reddit condition holds 205 episodes against a 203-task
            # scored set. Skip those rows here rather than at the check below:
            # they must not enter `outcomes` (a router replay over an unscored
            # task is not a scored outcome), while `seen` still records them so
            # the check can tell "expected-but-unscored" from "contamination".
            if task_id not in expected_ids:
                continue
            cost = float(record["total_billed_cost_usd"])
            if not math.isfinite(cost) or cost < 0:
                raise ValueError(f"Invalid billed cost {cost!r} in {summary_path}")
            basis = str(record.get("cost_unit_basis") or "unknown")
            basis_counts[basis] += 1
            outcomes.setdefault(task_id, {})[mode] = {
                "success": bool(record["success"]),
                "cost_usd": cost,
                "cost_unit_basis": basis,
                "summary_path": str(summary_path.relative_to(REPO)),
            }
        missing = sorted(expected_ids - seen)
        # Only IDs that are neither scored NOR protocol-excluded are drift.
        protocol_excluded = protocol_excluded_in_universe(site)
        excluded_seen = sorted(seen & protocol_excluded)
        extra = sorted(seen - expected_ids - protocol_excluded)
        if missing or extra:
            raise ValueError(
                f"Canonical task universe mismatch for {cell_id}/{mode}: "
                f"n_seen={len(seen)}, missing={missing[:10]}, extra={extra[:10]}"
            )
        entry = {**entry, "protocol_excluded_observed": excluded_seen}
        source_rows[mode] = {
            **entry,
            "condition_dir": str(condition_dir.relative_to(REPO)),
            "n_tasks": len(seen & expected_ids),
            "n_collected": len(seen),
        }

    if set(outcomes) != set(expected_ids):
        raise AssertionError(f"Internal universe assembly mismatch for {cell_id}")
    for task_id, modes in outcomes.items():
        if set(modes) != set(DISPLAY_MODES):
            raise AssertionError(f"Task {task_id} in {cell_id} lacks a six-mode row")

    basis_homogeneous = len(basis_counts) == 1
    if not basis_homogeneous:
        raise ValueError(f"Mixed cost_unit_basis within {cell_id}: {dict(basis_counts)}")
    provenance = {
        "canonical_task_universe_sha256": expected_sha,
        "source_by_mode": source_rows,
        "cost_unit_basis_counts": dict(sorted(basis_counts.items())),
        "cost_unit_basis_homogeneous": basis_homogeneous,
        "cost_unit_basis": next(iter(basis_counts)),
    }
    return outcomes, provenance


def policy_metrics(
    outcomes: dict[int, dict[str, dict[str, Any]]],
    selected_mode_by_task: dict[int, str],
) -> dict[str, Any]:
    """Evaluate a task->mode policy against the already-loaded Pass-1 matrix."""
    successes: list[bool] = []
    costs: list[float] = []
    selected_counts: Counter[str] = Counter()
    for task_id in sorted(selected_mode_by_task):
        mode = selected_mode_by_task[task_id]
        if mode not in DISPLAY_MODES:
            raise ValueError(f"Policy selected invalid mode {mode!r} for task {task_id}")
        try:
            row = outcomes[task_id][mode]
        except KeyError as exc:
            raise KeyError(f"No Pass-1 outcome for task={task_id}, mode={mode}") from exc
        successes.append(bool(row["success"]))
        costs.append(float(row["cost_usd"]))
        selected_counts[mode] += 1
    n_tasks = len(successes)
    n_success = sum(successes)
    return {
        "n_tasks": n_tasks,
        "n_success": n_success,
        "success_rate": n_success / n_tasks if n_tasks else None,
        "success_rate_pct": (100.0 * n_success / n_tasks) if n_tasks else None,
        "mean_total_billed_cost_usd": sum(costs) / n_tasks if n_tasks else None,
        "sum_total_billed_cost_usd": sum(costs) if n_tasks else None,
        "selected_mode_counts": dict(sorted(selected_counts.items())),
    }


def reference_points(
    outcomes: dict[int, dict[str, dict[str, Any]]],
    task_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Six fixed modes, best single, six-mode oracle, and always-P-SoM."""
    ids = sorted(outcomes if task_ids is None else {int(t) for t in task_ids})
    per_mode: dict[str, Any] = {}
    for mode in DISPLAY_MODES:
        metric = policy_metrics(outcomes, {task_id: mode for task_id in ids})
        metric["mode"] = mode
        per_mode[mode] = metric

    # Best-single is selected by SR, then lower realized mean billed cost, then
    # the fixed display order.  The cost tie-break is descriptive, not a label change.
    order = {mode: idx for idx, mode in enumerate(DISPLAY_MODES)}
    best_mode = min(
        DISPLAY_MODES,
        key=lambda mode: (
            -float(per_mode[mode]["success_rate"]),
            float(per_mode[mode]["mean_total_billed_cost_usd"]),
            order[mode],
        ),
    )
    best_single = {**per_mode[best_mode], "mode": best_mode}

    oracle_policy: dict[int, str] = {}
    n_no_success = 0
    for task_id in ids:
        label = derive_oracle_label(
            {mode: bool(outcomes[task_id][mode]["success"]) for mode in DISPLAY_MODES}
        )
        if label is None:
            n_no_success += 1
            label = FALLBACK_MODE
        oracle_policy[task_id] = label
    oracle = policy_metrics(outcomes, oracle_policy)
    oracle.update(
        {
            "definition": (
                "six-mode success union; cost replays the locked cheapest-prior "
                "successful label and uses P-SoM on no-success tasks"
            ),
            "uses_hindsight": True,
            "n_no_success_tasks": n_no_success,
        }
    )
    return {
        "single_modes": per_mode,
        "best_single_mode": best_single,
        "six_mode_oracle_ceiling": oracle,
        "always_p_som": dict(per_mode[FALLBACK_MODE]),
        "oracle_gap_from_best_single_pp": (
            oracle["success_rate_pct"] - best_single["success_rate_pct"]
        ),
    }


def _read_task_config(site: str, task_id: int) -> dict[str, Any]:
    path = REPO / "external/visualwebarena/config_files/vwa" / f"test_{site}" / f"{task_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing VWA task config: {path}")
    cfg = json.loads(path.read_text())
    image = cfg.get("image")
    return {
        "path": path,
        "intent": cfg.get("intent", "") or "",
        "has_reference_image": image not in (None, "None", "", []),
        "reasoning_difficulty": difficulty_to_int(cfg.get("reasoning_difficulty")),
    }


def _read_step0_features(
    entries: dict[str, dict[str, Any]],
    phase1_root: Path,
    site: str,
    task_id: int,
) -> tuple[dict[str, int], str | None]:
    """Mirror Stage 1: try all canonical runs, then explicitly zero-fill."""
    for mode in DISPLAY_MODES:
        entry = entries[mode]
        step_path = (
            phase1_root
            / entry["run_dir"]
            / entry["condition_subdir"]
            / "episodes"
            / f"{site}_task_{task_id}_steps_v2.jsonl"
        )
        if not step_path.is_file():
            continue
        try:
            with step_path.open() as handle:
                first_line = handle.readline()
            step0 = json.loads(first_line)
        except (OSError, json.JSONDecodeError):
            continue
        digest = step0.get("state_digest") or {}
        text_length = int(digest.get("text_length", 0) or 0)
        return {
            "dom_complexity": int(digest.get("dom_complexity", 0) or 0),
            "text_length": text_length,
            "tokens_input_text": estimate_input_tokens(text_length),
        }, str(step_path.relative_to(REPO))
    return {
        "dom_complexity": 0,
        "text_length": 0,
        "tokens_input_text": 0,
    }, None


def build_offline_raw_features(
    entries: dict[str, dict[str, Any]],
    phase1_root: Path,
    site: str,
    task_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _read_task_config(site, task_id)
    step0, step0_source = _read_step0_features(entries, phase1_root, site, task_id)
    raw = extract_raw_features(
        intent=cfg["intent"],
        has_reference_image=cfg["has_reference_image"],
        dom_complexity=step0["dom_complexity"],
        text_length=step0["text_length"],
        tokens_input_text=step0["tokens_input_text"],
        reasoning_difficulty=cfg["reasoning_difficulty"],
    )
    return raw, {
        "task_config_path": str(cfg["path"].relative_to(REPO)),
        "step0_path": step0_source,
        "step0_zero_filled": step0_source is None,
    }


def replay_cell(
    cell_id: str,
    entries: dict[str, dict[str, Any]],
    outcomes: dict[int, dict[str, dict[str, Any]]],
    outcome_provenance: dict[str, Any],
    artifacts_dir: Path,
    phase1_root: Path,
) -> dict[str, Any]:
    meta_path = artifacts_dir / f"{cell_id}_lr_meta.json"
    fold_path = artifacts_dir / f"{cell_id}_fold_assignment.json"
    if not meta_path.is_file() or not fold_path.is_file():
        raise FileNotFoundError(f"Missing Stage 2/3 metadata for {cell_id}")
    meta = json.loads(meta_path.read_text())
    fold_payload = json.loads(fold_path.read_text())
    fold_assignment = {
        int(task_id): int(fold) for task_id, fold in fold_payload["fold_assignment"].items()
    }
    if set(fold_assignment) != set(outcomes):
        missing = sorted(set(outcomes) - set(fold_assignment))
        extra = sorted(set(fold_assignment) - set(outcomes))
        # B-1904 (/stress Mode B codex, 2026-07-27): the cached fold maps were
        # built pre-AMENDMENT_08 over the 205-task COLLECTED reddit set and
        # carry no canonical-universe SHA, so they cannot be validated against
        # the scored set they are now being replayed on.  Deliberately NOT
        # tolerated: silently dropping the two extra fold entries would reuse a
        # split whose stratification was computed over a different universe,
        # which is a quieter version of the same defect. Regenerate instead.
        hint = ""
        if extra and not missing:
            excluded = sorted(protocol_excluded_in_universe(cell_id.split("_", 1)[1]))
            if set(extra) <= set(excluded):
                hint = (
                    f" — these are exactly the AMENDMENT_08 protocol-excluded IDs "
                    f"{excluded}, i.e. this fold map predates the amendment "
                    f"(B-1904). Regenerate the Stage 2/3 artifacts against the "
                    f"scored universe; do not drop the entries in place."
                )
        raise ValueError(
            f"Fold-map universe mismatch for {cell_id}: "
            f"missing={missing[:10]}, extra={extra[:10]}{hint}"
        )

    folds_ok = {int(fold) for fold in meta.get("folds_ok", [])}
    cell_complete = bool(meta.get("cell_complete")) and folds_ok == set(range(5))
    training_status = (
        "TRAINED_COMPLETE"
        if cell_complete
        else ("UNTRAINABLE" if not folds_ok else "INCOMPLETE")
    )
    full_refs = reference_points(outcomes)

    cache: dict[str, Any] = {}
    selected: dict[int, str] = {}
    task_records: list[dict[str, Any]] = []
    n_step0_zero_filled = 0
    n_signal_fallback = 0
    for task_id in sorted(outcomes):
        fold_k = fold_assignment[task_id]
        if fold_k not in folds_ok:
            task_records.append(
                {
                    "task_id": task_id,
                    "fold_k": fold_k,
                    "prediction_status": "missing_fold_model",
                    "selected_mode": None,
                }
            )
            continue
        raw_features, feature_provenance = build_offline_raw_features(
            entries, phase1_root, cell_id.split("_", 1)[1], task_id
        )
        n_step0_zero_filled += int(feature_provenance["step0_zero_filled"])
        try:
            predicted_mode, diag = predict_mode_fold_aware(
                cell_id=cell_id,
                task_id=task_id,
                artifacts_dir=artifacts_dir,
                cache=cache,
                raw_features=raw_features,
                fallback_mode=FALLBACK_MODE,
            )
        except LearnedRouterArtifactError:
            raise
        if predicted_mode not in DISPLAY_MODES:
            raise ValueError(f"{cell_id}/task {task_id}: model predicted {predicted_mode!r}")
        selected[task_id] = predicted_mode
        selected_outcome = outcomes[task_id][predicted_mode]
        n_signal_fallback += int(bool(diag.get("fallback_fired")))
        task_records.append(
            {
                "task_id": task_id,
                "fold_k": fold_k,
                "prediction_status": "oof",
                "selected_mode": predicted_mode,
                "argmax_mode": diag.get("argmax_mode"),
                "tau": diag.get("tau_used"),
                "max_probability": diag.get("max_prob"),
                "signal_strength_fallback_fired": bool(diag.get("fallback_fired")),
                "success": bool(selected_outcome["success"]),
                "total_billed_cost_usd": float(selected_outcome["cost_usd"]),
                "cost_unit_basis": selected_outcome["cost_unit_basis"],
                "outcome_summary_path": selected_outcome["summary_path"],
                **feature_provenance,
            }
        )

    partial_metrics = policy_metrics(outcomes, selected) if selected else None
    partial_refs = reference_points(outcomes, selected) if selected else None
    if partial_metrics is not None:
        partial_metrics.update(
            {
                "coverage_fraction": len(selected) / len(outcomes),
                "fallback_count": n_signal_fallback,
                "fallback_rate": n_signal_fallback / len(selected),
                "delta_vs_subset_best_single_pp": (
                    partial_metrics["success_rate_pct"]
                    - partial_refs["best_single_mode"]["success_rate_pct"]
                ),
                "oracle_gap_pp": (
                    partial_refs["six_mode_oracle_ceiling"]["success_rate_pct"]
                    - partial_metrics["success_rate_pct"]
                ),
            }
        )

    offline_routed = partial_metrics if cell_complete and len(selected) == len(outcomes) else None
    if offline_routed is not None:
        offline_routed = dict(offline_routed)
        offline_routed.update(
            {
                "delta_vs_best_single_pp": (
                    offline_routed["success_rate_pct"]
                    - full_refs["best_single_mode"]["success_rate_pct"]
                ),
                "oracle_gap_pp": (
                    full_refs["six_mode_oracle_ceiling"]["success_rate_pct"]
                    - offline_routed["success_rate_pct"]
                ),
            }
        )

    raw_meta = json.loads((artifacts_dir / "raw_features_phase1a.json").read_text())
    raw_cell = raw_meta.get("per_cell_summary", {}).get(cell_id, {})
    return {
        "cell_id": cell_id,
        "training_status": training_status,
        "cell_complete": cell_complete,
        "folds_ok": sorted(folds_ok),
        "incomplete_folds": meta.get("incomplete_folds", {}),
        "n_labeled_tasks": raw_cell.get("n_kept"),
        "n_tasks": len(outcomes),
        "thresholds_per_fold": meta.get("thresholds_per_fold", {}),
        "safe_fallback_mode": meta.get("safe_fallback_mode", FALLBACK_MODE),
        "fold_map_sha256": fold_map_sha256(fold_payload["fold_assignment"]),
        "fold_map_file_sha256": sha256_file(fold_path),
        "fold_map_sha_definition": "sha256(canonical sorted task_id->fold JSON mapping)",
        "n_oof_predictions": len(selected),
        "n_missing_fold_predictions": len(outcomes) - len(selected),
        "n_step0_features_zero_filled": n_step0_zero_filled,
        "offline_routed": offline_routed,
        "partial_oof_diagnostic": (
            partial_metrics if not cell_complete and partial_metrics is not None else None
        ),
        "partial_oof_reference_points": (
            partial_refs if not cell_complete and partial_refs is not None else None
        ),
        "reference_points": full_refs,
        "fallback_reference_for_untrainable_cell": (
            full_refs["always_p_som"] if not cell_complete else None
        ),
        "oracle_gap_reference_pp": (
            full_refs["six_mode_oracle_ceiling"]["success_rate_pct"]
            - (
                offline_routed["success_rate_pct"]
                if offline_routed is not None
                else full_refs["always_p_som"]["success_rate_pct"]
            )
        ),
        "outcome_provenance": outcome_provenance,
        "training_artifacts": {
            "lr_meta": str(meta_path.relative_to(REPO)),
            "lr_meta_sha256": sha256_file(meta_path),
            "fold_assignment": str(fold_path.relative_to(REPO)),
        },
        "task_records": task_records,
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}%"


def _pp(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f} pp"


def _cost(value: float | None) -> str:
    return "—" if value is None else f"{value:.8g}"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {DISCLAIMER}",
        "",
        f"**Artifact status:** `{ARTIFACT_STATUS}` · **Gate eligible:** `false`",
        "",
        (
            "Routed cost below is the mean selected Pass-1 episode "
            "`total_billed_cost_usd`; it excludes live router overhead and must not be "
            "pooled across cells with different `cost_unit_basis`."
        ),
        "",
        "## Summary",
        "",
        "| Cell | Training | n | OOF n | Routed SR | Best single | Δ vs best | Oracle | Oracle gap | Always P-SoM | Mean routed billed cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell_id, cell in payload["cells"].items():
        routed = cell["offline_routed"]
        best = cell["reference_points"]["best_single_mode"]
        oracle = cell["reference_points"]["six_mode_oracle_ceiling"]
        psom = cell["reference_points"]["always_p_som"]
        lines.append(
            "| {cell} | {status} | {n} | {oof} | {routed_sr} | {best_sr} ({best_mode}) | "
            "{delta} | {oracle} | {gap} | {psom} | {cost} |".format(
                cell=cell_id,
                status=cell["training_status"],
                n=cell["n_tasks"],
                oof=cell["n_oof_predictions"],
                routed_sr=_pct(routed and routed["success_rate_pct"]),
                best_sr=_pct(best["success_rate_pct"]),
                best_mode=MODE_LABELS[best["mode"]],
                delta=_pp(routed and routed["delta_vs_best_single_pp"]),
                oracle=_pct(oracle["success_rate_pct"]),
                gap=_pp(routed and routed["oracle_gap_pp"]),
                psom=_pct(psom["success_rate_pct"]),
                cost=_cost(routed and routed["mean_total_billed_cost_usd"]),
            )
        )

    for cell_id, cell in payload["cells"].items():
        refs = cell["reference_points"]
        lines.extend(
            [
                "",
                f"## {cell_id}",
                "",
                (
                    f"Training: **{cell['training_status']}**; labeled={cell['n_labeled_tasks']}; "
                    f"OOF coverage={cell['n_oof_predictions']}/{cell['n_tasks']}; "
                    f"folds_ok={cell['folds_ok']}; τ={cell['thresholds_per_fold']}."
                ),
                "",
                f"Fold map SHA-256: `{cell['fold_map_sha256']}`. Cost basis: "
                f"`{cell['outcome_provenance']['cost_unit_basis']}`.",
                "",
                "| Policy/reference | Coverage | SR | Δ vs full-cell best single | Mean billed cost | Status |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        best_sr = refs["best_single_mode"]["success_rate_pct"]
        for mode in DISPLAY_MODES:
            metric = refs["single_modes"][mode]
            lines.append(
                f"| Always {MODE_LABELS[mode]} | {metric['n_tasks']} | "
                f"{_pct(metric['success_rate_pct'])} | "
                f"{_pp(metric['success_rate_pct'] - best_sr)} | "
                f"{_cost(metric['mean_total_billed_cost_usd'])} | single mode |"
            )
        best = refs["best_single_mode"]
        oracle = refs["six_mode_oracle_ceiling"]
        psom = refs["always_p_som"]
        lines.extend(
            [
                f"| **Best single ({MODE_LABELS[best['mode']]})** | {best['n_tasks']} | "
                f"**{_pct(best['success_rate_pct'])}** | +0.00 pp | "
                f"{_cost(best['mean_total_billed_cost_usd'])} | reference |",
                f"| **Six-mode oracle ceiling** | {oracle['n_tasks']} | "
                f"**{_pct(oracle['success_rate_pct'])}** | "
                f"{_pp(oracle['success_rate_pct'] - best_sr)} | "
                f"{_cost(oracle['mean_total_billed_cost_usd'])} | hindsight ceiling |",
                f"| **Always P-SoM fallback** | {psom['n_tasks']} | "
                f"**{_pct(psom['success_rate_pct'])}** | "
                f"{_pp(psom['success_rate_pct'] - best_sr)} | "
                f"{_cost(psom['mean_total_billed_cost_usd'])} | fallback reference |",
            ]
        )
        if cell["offline_routed"] is not None:
            routed = cell["offline_routed"]
            lines.append(
                f"| **OOF offline router** | {routed['n_tasks']} | "
                f"**{_pct(routed['success_rate_pct'])}** | "
                f"**{_pp(routed['delta_vs_best_single_pp'])}** | "
                f"{_cost(routed['mean_total_billed_cost_usd'])} | OFFLINE/NON-GATE |"
            )
            lines.extend(
                [
                    "",
                    f"Oracle gap after OOF routing: **{routed['oracle_gap_pp']:.2f} pp**; "
                    f"signal-threshold fallback rate: {routed['fallback_rate']:.1%}.",
                ]
            )
        else:
            lines.append(
                f"| **OOF offline router** | — | — | — | — | "
                f"{cell['training_status']}: no full-cell routed estimate |"
            )
            lines.extend(
                [
                    "",
                    f"Full-cell OOF replay is unavailable: `{cell['incomplete_folds']}`. "
                    f"The full-cell reference therefore remains always P-SoM at "
                    f"**{psom['success_rate_pct']:.2f}%**, with oracle gap "
                    f"**{cell['oracle_gap_reference_pp']:.2f} pp**.",
                ]
            )
            partial = cell.get("partial_oof_diagnostic")
            if partial is not None:
                lines.extend(
                    [
                        "",
                        f"Diagnostic partial OOF only (not a cell estimate): "
                        f"n={partial['n_tasks']}, SR={partial['success_rate_pct']:.2f}%, "
                        f"subset-best delta={partial['delta_vs_subset_best_single_pp']:+.2f} pp, "
                        f"coverage={partial['coverage_fraction']:.1%}.",
                    ]
                )

    lines.extend(
        [
            "",
            "## Provenance and limits",
            "",
            f"- Source manifest: `{payload['inputs']['run_manifest']}` "
            f"(SHA-256 `{payload['inputs']['run_manifest_sha256']}`).",
            f"- Router artifacts: `{payload['inputs']['artifacts_dir']}`; Stage 2/3 "
            "uses per-site shared task-held-out folds, fold-local TF-IDF/MI top-18, "
            "and per-cell LR heads with per-fold τ.",
            "- Outcomes and selected costs are reused from one Pass-1 realization per "
            "mode. This is realized-ish offline replay, not independent live evaluation; "
            "it omits router serving cost, latency, state interaction, and a fresh trajectory.",
            "- The oracle uses hindsight and is a ceiling, not a deployable policy. Its "
            "cost column follows the locked cheapest-prior successful label, with P-SoM "
            "on no-success tasks, solely to make the replay accounting explicit.",
            "- B0 uses API USD while B1/B2 use electricity-derived USD. No cross-cell "
            "cost mean or pooled cost comparison is reported.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_replay(
    artifacts_dir: Path,
    run_manifest: Path,
    phase1_root: Path,
    cells: list[str],
) -> dict[str, Any]:
    entries_by_cell = load_paper_grade_entries(run_manifest, cells)
    cell_results: dict[str, Any] = {}
    for cell_id in cells:
        outcomes, provenance = collect_cell_outcomes(
            cell_id, entries_by_cell[cell_id], phase1_root
        )
        cell_results[cell_id] = replay_cell(
            cell_id,
            entries_by_cell[cell_id],
            outcomes,
            provenance,
            artifacts_dir,
            phase1_root,
        )
    n_complete = sum(c["training_status"] == "TRAINED_COMPLETE" for c in cell_results.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_status": ARTIFACT_STATUS,
        "gate_eligible": False,
        "disclaimer": DISCLAIMER,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cost_definition": (
            "mean Pass-1 total_billed_cost_usd selected task-by-task; excludes live "
            "router overhead and latency"
        ),
        "inputs": {
            "run_manifest": str(run_manifest.relative_to(REPO)),
            "run_manifest_sha256": sha256_file(run_manifest),
            "artifacts_dir": str(artifacts_dir.relative_to(REPO)),
            "phase1_root": str(phase1_root.relative_to(REPO)),
            "requested_cells": cells,
        },
        "summary": {
            "n_cells_requested": len(cells),
            "n_cells_trained_complete": n_complete,
            "n_cells_incomplete_or_untrainable": len(cells) - n_complete,
            "cross_cell_pooled_sr": None,
            "cross_cell_pooled_cost": None,
            "note": "No pooled result: only complete cells have a full OOF policy, and cost bases differ.",
        },
        "cells": cell_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-manifest", type=Path, default=DEFAULT_RUN_MANIFEST)
    parser.add_argument("--phase1-root", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir.resolve()
    run_manifest = args.run_manifest.resolve()
    phase1_root = args.phase1_root.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir == FORBIDDEN_CANONICAL_OUT:
        parser.error("Refusing to write OFFLINE/NON-GATE artifacts into canonical l1_router/")
    if not artifacts_dir.is_dir():
        parser.error(f"Artifacts directory does not exist: {artifacts_dir}")
    if not run_manifest.is_file():
        parser.error(f"Run manifest does not exist: {run_manifest}")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = run_replay(artifacts_dir, run_manifest, phase1_root, list(args.cells))
    json_path = out_dir / "router_offline_replay.json"
    md_path = out_dir / "router_offline_replay.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    md_path.write_text(render_markdown(payload))
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(
        f"{DISCLAIMER}: {payload['summary']['n_cells_trained_complete']}/"
        f"{payload['summary']['n_cells_requested']} cells have full OOF replay"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
