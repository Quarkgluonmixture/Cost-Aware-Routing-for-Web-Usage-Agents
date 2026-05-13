#!/usr/bin/env python3
"""Per-run diagnostic; not part of the 4-dimension evidence framework.

Validate a P79 experiment run directory for data integrity and completeness.

Runs 27 checks across 10 groups and outputs a structured report.

Usage:
    .venv/bin/python3 scripts/analysis/validate_run.py \
        --run-dir results/visualwebarena/phase1/B0_3mode_classifieds_20260413

    .venv/bin/python3 scripts/analysis/validate_run.py \
        --run-dir results/visualwebarena/phase1/B0_3mode_reddit_20260422 \
        --compare-dir results/visualwebarena/phase1/B1_3mode_reddit_20260413

    .venv/bin/python3 scripts/analysis/validate_run.py \
        --run-dir results/visualwebarena/phase1/B0_3mode_classifieds_20260413 \
        --output /tmp/validation_report.json --strict
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# --- Add project root to path for imports ---
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from p79.experiment.io_utils import read_jsonl_dedup
from p79.experiment.types import REQUIRED_STEP_FIELDS_V2, validate_step_record_v2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXCLUDED_DIRS = {"analysis", "task_configs", "_vwa"}
_EXPECTED_CONDITIONS = [
    "phase1_dom_router_0",
    "phase1_som_router_0",
    "phase1_vision_router_0",
    # Phantom variants added 2026-05-13 (codex audit P1):
    # validate_run was gating phantom cell paper-grade promotion at the
    # condition-ID level. Without these, any phantom_* run failed validation.
    "phase1_phantom_dom_router_0",
    "phase1_phantom_som_router_0",
    "phase1_phantom_text_router_0",
    "phase1_phantom_prompt_router_0",
]
_MODE_FROM_CONDITION = {
    "phase1_dom_router_0": "dom",
    "phase1_som_router_0": "som",
    "phase1_vision_router_0": "vision",
    "phase1_phantom_dom_router_0": "phantom_dom",
    "phase1_phantom_som_router_0": "phantom_som",
    "phase1_phantom_text_router_0": "phantom_text",
    "phase1_phantom_prompt_router_0": "phantom_prompt",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    check_id: str
    name: str
    status: str  # "pass" | "warn" | "fail" | "skip"
    detail: str
    items: Optional[List[Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _condition_dirs(run_dir: Path) -> List[Path]:
    """Return condition directories inside a run directory."""
    return sorted(
        p
        for p in run_dir.iterdir()
        if p.is_dir() and p.name not in _EXCLUDED_DIRS and not p.name.startswith(".")
    )


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _list_summaries(cond_dir: Path) -> List[Path]:
    ep_dir = cond_dir / "episodes"
    if not ep_dir.exists():
        return []
    return sorted(ep_dir.glob("*_summary_v2.json"))


def _list_steps_files(cond_dir: Path) -> List[Path]:
    ep_dir = cond_dir / "episodes"
    if not ep_dir.exists():
        return []
    return sorted(ep_dir.glob("*_steps_v2.jsonl"))


def _task_id_from_filename(name: str) -> Optional[int]:
    m = re.search(r"task_(\d+)", name)
    return int(m.group(1)) if m else None


def _sample_items(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    return random.sample(items, n)


def _infer_benchmark(run_dir: Path) -> str:
    """Infer benchmark from run directory path."""
    parts = run_dir.parts
    for i, p in enumerate(parts):
        if p in ("visualwebarena", "webarena"):
            return p
    return "unknown"


def _infer_site(run_dir: Path) -> str:
    """Infer site from run_id."""
    name = run_dir.name
    for site in ("classifieds", "reddit", "shopping"):
        if site in name:
            return site
    return "unknown"


# ---------------------------------------------------------------------------
# Group 1: File existence
# ---------------------------------------------------------------------------


def check_required_files(run_dir: Path) -> CheckResult:
    """C01: Check that required analysis/condition files exist."""
    missing = []
    analysis = run_dir / "analysis"

    # analysis_summary.json
    if not (analysis / "analysis_summary.json").exists():
        missing.append("analysis/analysis_summary.json")

    # condition_metrics.csv
    csv_path = analysis / "results" / "_overview" / "tables" / "condition_metrics.csv"
    if not csv_path.exists():
        missing.append("analysis/results/_overview/tables/condition_metrics.csv")

    # Each condition's condition_summary_v2.json
    for cond_dir in _condition_dirs(run_dir):
        cs = cond_dir / "condition_summary_v2.json"
        if not cs.exists():
            missing.append(f"{cond_dir.name}/condition_summary_v2.json")

    total_expected = 2 + len(list(_condition_dirs(run_dir)))
    found = total_expected - len(missing)

    if missing:
        return CheckResult(
            "C01", "Required files", "fail",
            f"{found}/{total_expected} required files exist; missing: {', '.join(missing)}",
            missing,
        )
    return CheckResult(
        "C01", "Required files", "pass",
        f"{found}/{total_expected} required files exist",
    )


def check_optional_files(run_dir: Path) -> CheckResult:
    """C02: List missing optional files (INFO level)."""
    analysis = run_dir / "analysis"
    optional_paths = [
        "digest/",
        "results/cross_representation/",
        "results/_overview/reports/statistical_tests.json",
        "reason_diagnostics/condition_overview.csv",
        "reason_diagnostics/condition_reason_summary.csv",
        "reason_diagnostics/episode_reason_rows.csv",
        "benchmark_noise/visual_lucky_hits.csv",
        "benchmark_noise/na_reference_tasks.csv",
        "signals/combined/tables/",
        "reason_diagnostics/thought_trajectories.jsonl",
        "reason_diagnostics/bucket_thought_samples.txt",
    ]
    missing = []
    for p in optional_paths:
        target = analysis / p
        if not target.exists():
            missing.append(p.rstrip("/"))

    found = len(optional_paths) - len(missing)
    if missing:
        return CheckResult(
            "C02", "Optional files", "info",
            f"{found}/{len(optional_paths)} optional paths present; missing: {', '.join(missing)}",
            missing,
        )
    return CheckResult(
        "C02", "Optional files", "pass",
        f"{found}/{len(optional_paths)} optional paths present",
    )


# ---------------------------------------------------------------------------
# Group 2: Run structure consistency
# ---------------------------------------------------------------------------


def check_condition_completeness(run_dir: Path) -> CheckResult:
    """C03: Check that all expected conditions exist."""
    present = {d.name for d in _condition_dirs(run_dir)}
    missing = [c for c in _EXPECTED_CONDITIONS if c not in present]
    if missing:
        return CheckResult(
            "C03", "Condition completeness", "warn",
            f"Missing conditions: {', '.join(missing)}",
            missing,
        )
    return CheckResult(
        "C03", "Condition completeness", "pass",
        f"All {len(_EXPECTED_CONDITIONS)} expected conditions present",
    )


def check_observation_mode_mapping(run_dir: Path) -> CheckResult:
    """C04: Verify observation_mode matches condition directory name."""
    errors = []
    for cond_name, expected_mode in _MODE_FROM_CONDITION.items():
        cond_dir = run_dir / cond_name
        if not cond_dir.exists():
            continue
        # Check condition_meta.json
        meta = _load_json(cond_dir / "condition_meta.json")
        if meta and meta.get("observation_mode") != expected_mode:
            errors.append(f"{cond_name}/condition_meta.json: observation_mode={meta.get('observation_mode')} (expected {expected_mode})")
        # Check condition_summary_v2.json
        summary = _load_json(cond_dir / "condition_summary_v2.json")
        if summary and summary.get("observation_mode") != expected_mode:
            errors.append(f"{cond_name}/condition_summary_v2.json: observation_mode={summary.get('observation_mode')} (expected {expected_mode})")

    if errors:
        return CheckResult(
            "C04", "Observation mode mapping", "fail",
            f"Mismatches: {'; '.join(errors)}",
            errors,
        )
    return CheckResult(
        "C04", "Observation mode mapping", "pass",
        "All observation_mode fields match condition directory names",
    )


def check_module_flags(run_dir: Path) -> CheckResult:
    """C05: Phase 1 module_flags should all be false."""
    issues = []
    for cond_dir in _condition_dirs(run_dir):
        meta = _load_json(cond_dir / "condition_meta.json")
        if not meta:
            continue
        modules = meta.get("modules", {})
        enabled = [k for k, v in modules.items() if v]
        if enabled:
            issues.append(f"{cond_dir.name}: {', '.join(enabled)} are True")

    if issues:
        return CheckResult(
            "C05", "Module flags consistency", "warn",
            f"Expected all false in Phase 1: {'; '.join(issues)}",
            issues,
        )
    return CheckResult(
        "C05", "Module flags consistency", "pass",
        "All module_flags are false (correct for Phase 1)",
    )


def check_benchmark_field(run_dir: Path, sample_size: int) -> CheckResult:
    """C06: Sampled step/summary benchmark field matches run directory path."""
    expected_benchmark = _infer_benchmark(run_dir)
    if expected_benchmark == "unknown":
        return CheckResult(
            "C06", "Benchmark field consistency", "skip",
            "Cannot infer benchmark from run directory path",
        )

    errors = []
    for cond_dir in _condition_dirs(run_dir):
        summaries = _list_summaries(cond_dir)
        for sf in _sample_items(summaries, sample_size):
            data = _load_json(sf)
            if data and data.get("benchmark") != expected_benchmark:
                errors.append(f"{sf.name}: benchmark={data.get('benchmark')}")

        steps_files = _list_steps_files(cond_dir)
        for stf in _sample_items(steps_files, sample_size):
            lines = read_jsonl_dedup(stf)
            if lines and lines[0].get("benchmark") != expected_benchmark:
                errors.append(f"{stf.name}: benchmark={lines[0].get('benchmark')}")

    if errors:
        return CheckResult(
            "C06", "Benchmark field consistency", "fail",
            f"Mismatches (expected {expected_benchmark}): {'; '.join(errors[:10])}",
            errors,
        )
    return CheckResult(
        "C06", "Benchmark field consistency", "pass",
        f"All sampled records have benchmark={expected_benchmark}",
    )


# ---------------------------------------------------------------------------
# Group 3: Episode coverage
# ---------------------------------------------------------------------------


def check_task_coverage(run_dir: Path) -> CheckResult:
    """C07: Episode count vs task_configs count per condition."""
    task_configs_dir = run_dir / "task_configs"
    if not task_configs_dir.exists():
        return CheckResult(
            "C07", "Task coverage", "skip",
            "task_configs/ directory not found",
        )

    total_tasks = len(list(task_configs_dir.glob("*.json")))
    if total_tasks == 0:
        return CheckResult(
            "C07", "Task coverage", "skip",
            "No task config files found",
        )

    details = []
    worst_pct = 100.0
    for cond_dir in _condition_dirs(run_dir):
        summaries = _list_summaries(cond_dir)
        n = len(summaries)
        pct = (n / total_tasks) * 100 if total_tasks > 0 else 0
        details.append(f"{cond_dir.name.split('_')[1]} {n}/{total_tasks} ({pct:.1f}%)")
        worst_pct = min(worst_pct, pct)

    detail_str = ", ".join(details)
    if worst_pct < 80:
        status = "fail"
    elif worst_pct < 100:
        status = "warn"
    else:
        status = "pass"

    return CheckResult(
        "C07", "Task coverage", status,
        f"Task coverage: {detail_str}",
    )


def check_cross_mode_task_set(run_dir: Path) -> CheckResult:
    """C08: Task ID sets should be consistent across modes."""
    mode_tasks: Dict[str, Set[int]] = {}
    for cond_dir in _condition_dirs(run_dir):
        summaries = _list_summaries(cond_dir)
        task_ids = set()
        for sf in summaries:
            tid = _task_id_from_filename(sf.name)
            if tid is not None:
                task_ids.add(tid)
        mode_tasks[cond_dir.name] = task_ids

    if len(mode_tasks) < 2:
        return CheckResult(
            "C08", "Cross-mode task set", "skip",
            "Fewer than 2 conditions, nothing to compare",
        )

    # Find differences
    all_tasks = set()
    for ts in mode_tasks.values():
        all_tasks |= ts

    diffs = []
    for cond, tasks in mode_tasks.items():
        missing = all_tasks - tasks
        if missing:
            mode_label = cond.split("_")[1]
            diffs.append(f"{mode_label} missing {len(missing)} tasks: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")

    if diffs:
        return CheckResult(
            "C08", "Cross-mode task set", "warn",
            "; ".join(diffs),
            diffs,
        )
    return CheckResult(
        "C08", "Cross-mode task set", "pass",
        f"All {len(mode_tasks)} conditions share the same {len(all_tasks)} task IDs",
    )


# ---------------------------------------------------------------------------
# Group 4: Episode completeness
# ---------------------------------------------------------------------------


def check_summary_null_fields(run_dir: Path) -> CheckResult:
    """C09: Detect summaries with null/0 steps or null score."""
    issues = []
    for cond_dir in _condition_dirs(run_dir):
        for sf in _list_summaries(cond_dir):
            data = _load_json(sf)
            if not data:
                continue
            # Skip benchmark_noise episodes
            if data.get("benchmark_noise"):
                continue
            steps = data.get("steps")
            score = data.get("score")
            if steps is None or steps == 0:
                issues.append(f"{cond_dir.name}/{sf.name}: steps={steps}")
            if score is None:
                issues.append(f"{cond_dir.name}/{sf.name}: score=null")

    if issues:
        return CheckResult(
            "C09", "Summary null fields", "warn",
            f"{len(issues)} episodes with null/0 steps or null score",
            issues[:20],
        )
    return CheckResult(
        "C09", "Summary null fields", "pass",
        "No null/0 steps or null score detected",
    )


def check_score_success_match(run_dir: Path) -> CheckResult:
    """C10: success=True & score==0 or success=False & score>0 should not occur."""
    issues = []
    for cond_dir in _condition_dirs(run_dir):
        for sf in _list_summaries(cond_dir):
            data = _load_json(sf)
            if not data:
                continue
            success = data.get("success")
            score = data.get("score")
            if score is None:
                continue
            if success and score == 0:
                issues.append(f"{cond_dir.name}/task_{data.get('task_id')}: success=True but score=0")
            elif not success and score > 0:
                issues.append(f"{cond_dir.name}/task_{data.get('task_id')}: success=False but score={score}")

    if issues:
        return CheckResult(
            "C10", "Score/success match", "warn",
            f"{len(issues)} score/success mismatches",
            issues[:20],
        )
    return CheckResult(
        "C10", "Score/success match", "pass",
        "All score/success values are consistent",
    )


def check_agent_finished_consistency(run_dir: Path, sample_size: int) -> CheckResult:
    """C11: summary.agent_finished vs last step action_type + fallback_finish."""
    issues = []
    for cond_dir in _condition_dirs(run_dir):
        summaries = _list_summaries(cond_dir)
        for sf in _sample_items(summaries, sample_size):
            data = _load_json(sf)
            if not data:
                continue
            summary_af = data.get("agent_finished")
            if summary_af is None:
                continue

            # Find corresponding steps file
            steps_name = sf.name.replace("_summary_v2.json", "_steps_v2.jsonl")
            steps_path = sf.parent / steps_name
            if not steps_path.exists():
                continue

            lines = read_jsonl_dedup(steps_path)
            if not lines:
                continue

            last_step = lines[-1]
            last_action = last_step.get("action_type", "")
            fallback = last_step.get("fallback_finish", False)

            # agent_finished should be True if last action is finish/stop and no fallback
            step_af = (last_action in ("finish", "stop")) and not fallback
            if summary_af != step_af:
                issues.append(
                    f"{cond_dir.name}/task_{data.get('task_id')}: "
                    f"summary.agent_finished={summary_af} but last_action={last_action}, fallback={fallback}"
                )

    if issues:
        return CheckResult(
            "C11", "agent_finished consistency", "warn",
            f"{len(issues)} agent_finished mismatches (sampled)",
            issues,
        )
    return CheckResult(
        "C11", "agent_finished consistency", "pass",
        "agent_finished consistent in sampled episodes",
    )


# ---------------------------------------------------------------------------
# Group 5: Step-level completeness
# ---------------------------------------------------------------------------


def check_jsonl_corrupt_lines(run_dir: Path) -> CheckResult:
    """C12: Full scan for corrupt JSONL lines."""
    total_files = 0
    corrupt_files = 0
    total_corrupt_lines = 0
    corrupt_details = []

    for cond_dir in _condition_dirs(run_dir):
        for stf in _list_steps_files(cond_dir):
            total_files += 1
            n_corrupt = 0
            with open(stf, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        n_corrupt += 1
            if n_corrupt > 0:
                corrupt_files += 1
                total_corrupt_lines += n_corrupt
                corrupt_details.append(f"{stf.relative_to(run_dir)}: {n_corrupt} corrupt lines")

    if total_corrupt_lines > 0:
        return CheckResult(
            "C12", "JSONL integrity", "warn",
            f"{total_corrupt_lines} corrupt lines in {corrupt_files}/{total_files} files",
            corrupt_details,
        )
    return CheckResult(
        "C12", "JSONL integrity", "pass",
        f"0 corrupt lines in {total_files} files",
    )


def check_restart_dedup(run_dir: Path) -> CheckResult:
    """C13: Full scan for step_idx=0 appearing more than once (restart artifacts)."""
    total_files = 0
    restart_files = 0
    restart_details = []

    for cond_dir in _condition_dirs(run_dir):
        for stf in _list_steps_files(cond_dir):
            total_files += 1
            step0_count = 0
            with open(stf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("step_idx") == 0:
                            step0_count += 1
                    except json.JSONDecodeError:
                        continue
            if step0_count > 1:
                restart_files += 1
                restart_details.append(f"{stf.relative_to(run_dir)}: step_idx=0 appears {step0_count} times")

    if restart_files > 0:
        return CheckResult(
            "C13", "Restart dedup", "warn",
            f"{restart_files}/{total_files} files have restart artifacts (step_idx=0 > 1)",
            restart_details,
        )
    return CheckResult(
        "C13", "Restart dedup", "pass",
        f"No restart artifacts in {total_files} files",
    )


def check_required_step_fields(run_dir: Path, sample_size: int) -> CheckResult:
    """C14: Sampled step validation against schema v2."""
    errors = []
    for cond_dir in _condition_dirs(run_dir):
        steps_files = _list_steps_files(cond_dir)
        for stf in _sample_items(steps_files, sample_size):
            lines = read_jsonl_dedup(stf)
            if not lines:
                continue
            # Check first and last step
            for step in [lines[0], lines[-1]]:
                try:
                    validate_step_record_v2(step)
                except ValueError as e:
                    errors.append(f"{stf.relative_to(run_dir)}: {e}")

    if errors:
        return CheckResult(
            "C14", "Required step fields", "fail",
            f"{len(errors)} step schema violations (sampled)",
            errors[:20],
        )
    return CheckResult(
        "C14", "Required step fields", "pass",
        "All sampled steps pass schema validation",
    )


def check_parse_valid_distribution(run_dir: Path) -> CheckResult:
    """C15: Full scan for parse_valid=false ratio per condition."""
    results = []
    worst_pct = 0.0

    for cond_dir in _condition_dirs(run_dir):
        total_steps = 0
        parse_invalid = 0
        for stf in _list_steps_files(cond_dir):
            with open(stf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        total_steps += 1
                        if rec.get("parse_valid") is False:
                            parse_invalid += 1
                    except json.JSONDecodeError:
                        continue

        if total_steps > 0:
            pct = (parse_invalid / total_steps) * 100
            mode_label = cond_dir.name.split("_")[1]
            results.append(f"{mode_label}: {parse_invalid}/{total_steps} ({pct:.1f}%)")
            worst_pct = max(worst_pct, pct)

    detail = ", ".join(results) if results else "no steps found"
    if worst_pct > 5:
        return CheckResult(
            "C15", "parse_valid distribution", "warn",
            f"parse_valid=false: {detail}",
            results,
        )
    return CheckResult(
        "C15", "parse_valid distribution", "pass",
        f"parse_valid=false: {detail}",
    )


# ---------------------------------------------------------------------------
# Group 6: Scaffold safety
# ---------------------------------------------------------------------------


def check_fallback_finish(run_dir: Path) -> CheckResult:
    """C16: Full scan for fallback_finish=true in last step of episodes."""
    affected = []

    for cond_dir in _condition_dirs(run_dir):
        mode_label = cond_dir.name.split("_")[1]
        for stf in _list_steps_files(cond_dir):
            lines = read_jsonl_dedup(stf)
            if not lines:
                continue
            last = lines[-1]
            if last.get("fallback_finish") is True:
                tid = last.get("task_id", _task_id_from_filename(stf.name))
                affected.append(f"{mode_label}:task_{tid}")

    if affected:
        # Group by mode
        by_mode: Dict[str, list] = {}
        for item in affected:
            mode = item.split(":")[0]
            by_mode.setdefault(mode, []).append(item)
        summary_parts = [f"{m}:{len(ts)}" for m, ts in by_mode.items()]
        return CheckResult(
            "C16", "fallback_finish detection", "fail",
            f"{len(affected)} episodes with fallback_finish ({', '.join(summary_parts)})",
            affected,
        )
    return CheckResult(
        "C16", "fallback_finish detection", "pass",
        "No fallback_finish detected",
    )


def check_benchmark_noise_rate(run_dir: Path) -> CheckResult:
    """C17: benchmark_noise_rate from condition_summary_v2.json."""
    results = []
    worst_pct = 0.0

    for cond_dir in _condition_dirs(run_dir):
        summary = _load_json(cond_dir / "condition_summary_v2.json")
        if not summary:
            continue
        rate = summary.get("benchmark_noise_rate", 0) or 0
        pct = rate * 100
        mode_label = cond_dir.name.split("_")[1]
        results.append(f"{mode_label}: {pct:.1f}%")
        worst_pct = max(worst_pct, pct)

    detail = ", ".join(results) if results else "no data"
    if worst_pct > 5:
        return CheckResult(
            "C17", "benchmark_noise rate", "warn",
            f"benchmark_noise_rate: {detail}",
            results,
        )
    return CheckResult(
        "C17", "benchmark_noise rate", "pass",
        f"benchmark_noise_rate: {detail}",
    )


def check_cross_baseline_symmetry(
    run_dir: Path, compare_dir: Optional[Path]
) -> CheckResult:
    """C18: Cross-baseline scaffold symmetry check (parse_error rate diff)."""
    if compare_dir is None:
        return CheckResult(
            "C18", "Cross-baseline symmetry", "skip",
            "No --compare-dir provided",
        )

    if not compare_dir.exists():
        return CheckResult(
            "C18", "Cross-baseline symmetry", "skip",
            f"Compare dir does not exist: {compare_dir}",
        )

    def _parse_error_rates(rd: Path) -> Dict[str, float]:
        rates = {}
        for cond_dir in _condition_dirs(rd):
            total_steps = 0
            parse_errors = 0
            for stf in _list_steps_files(cond_dir):
                with open(stf, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            total_steps += 1
                            if rec.get("parse_valid") is False:
                                parse_errors += 1
                        except json.JSONDecodeError:
                            continue
            if total_steps > 0:
                rates[cond_dir.name] = (parse_errors / total_steps) * 100
        return rates

    rates_a = _parse_error_rates(run_dir)
    rates_b = _parse_error_rates(compare_dir)

    issues = []
    for cond in sorted(set(rates_a) & set(rates_b)):
        diff = abs(rates_a[cond] - rates_b[cond])
        if diff > 2.0:
            mode_label = cond.split("_")[1]
            issues.append(
                f"{mode_label}: {run_dir.name} {rates_a[cond]:.1f}% vs "
                f"{compare_dir.name} {rates_b[cond]:.1f}% (diff {diff:.1f}pp)"
            )

    if issues:
        return CheckResult(
            "C18", "Cross-baseline symmetry", "warn",
            f"parse_error rate diff >2pp: {'; '.join(issues)}",
            issues,
        )
    return CheckResult(
        "C18", "Cross-baseline symmetry", "pass",
        "parse_error rate diff <=2pp across all conditions",
    )


# ---------------------------------------------------------------------------
# Group 7: Artifact completeness
# ---------------------------------------------------------------------------


def check_screenshot_validity(run_dir: Path, sample_size: int) -> CheckResult:
    """C19: Sampled screenshot files should be >5KB for som/vision modes."""
    issues = []

    for cond_dir in _condition_dirs(run_dir):
        mode_label = cond_dir.name.split("_")[1]
        if mode_label not in ("som", "vision"):
            continue

        art_dir = cond_dir / "artifacts"
        if not art_dir.exists():
            continue

        task_dirs = sorted(d for d in art_dir.iterdir() if d.is_dir())
        for td in _sample_items(task_dirs, sample_size):
            screenshots = list(td.rglob("screenshot.png"))
            if not screenshots:
                issues.append(f"{mode_label}/{td.name}: no screenshots found")
                continue
            for ss in screenshots[:2]:  # Check up to 2 per task
                sz = ss.stat().st_size
                if sz < 5 * 1024:
                    issues.append(f"{mode_label}/{td.name}/{ss.parent.name}: {sz} bytes (<5KB)")

    if issues:
        return CheckResult(
            "C19", "Screenshot validity", "warn",
            f"{len(issues)} screenshot issues (sampled)",
            issues,
        )
    return CheckResult(
        "C19", "Screenshot validity", "pass",
        "All sampled screenshots are valid (>5KB)",
    )


def check_orphan_artifacts(run_dir: Path) -> CheckResult:
    """C20: Artifact dirs with no corresponding summary (excluding recent)."""
    cutoff = time.time() - 10 * 60  # 10 minutes
    orphans = []

    for cond_dir in _condition_dirs(run_dir):
        art_dir = cond_dir / "artifacts"
        ep_dir = cond_dir / "episodes"
        if not art_dir.exists():
            continue

        for artifact in sorted(art_dir.iterdir()):
            if not artifact.is_dir():
                continue
            summary_path = ep_dir / f"{artifact.name}_summary_v2.json"
            if summary_path.exists():
                continue
            if artifact.stat().st_mtime > cutoff:
                continue
            orphans.append(f"{cond_dir.name}/artifacts/{artifact.name}")

    if orphans:
        return CheckResult(
            "C20", "Orphan artifacts", "warn",
            f"{len(orphans)} orphan artifact dirs (no summary, >10min old)",
            orphans[:20],
        )
    return CheckResult(
        "C20", "Orphan artifacts", "pass",
        "No orphan artifacts detected",
    )


# ---------------------------------------------------------------------------
# Group 8: Analysis freshness & digest
# ---------------------------------------------------------------------------


def check_analysis_freshness(run_dir: Path) -> CheckResult:
    """C21: analysis_summary.json should be newer than latest condition_summary."""
    analysis_summary = run_dir / "analysis" / "analysis_summary.json"
    if not analysis_summary.exists():
        return CheckResult(
            "C21", "Analysis freshness", "skip",
            "analysis_summary.json not found",
        )

    analysis_mtime = analysis_summary.stat().st_mtime
    latest_condition_mtime = 0.0

    for cond_dir in _condition_dirs(run_dir):
        cs = cond_dir / "condition_summary_v2.json"
        if cs.exists():
            latest_condition_mtime = max(latest_condition_mtime, cs.stat().st_mtime)

    if latest_condition_mtime == 0:
        return CheckResult(
            "C21", "Analysis freshness", "skip",
            "No condition_summary_v2.json files found",
        )

    if analysis_mtime < latest_condition_mtime:
        diff_mins = (latest_condition_mtime - analysis_mtime) / 60
        return CheckResult(
            "C21", "Analysis freshness", "warn",
            f"analysis_summary.json is {diff_mins:.0f} min older than latest condition_summary",
        )
    return CheckResult(
        "C21", "Analysis freshness", "pass",
        "analysis_summary.json is up to date",
    )


def check_glm_digest_coverage(run_dir: Path) -> CheckResult:
    """C22: GLM digest line count vs failed episode count + duplicate detection."""
    digest_dir = run_dir / "analysis" / "digest"
    if not digest_dir.exists():
        return CheckResult(
            "C22", "GLM digest coverage", "skip",
            "digest/ directory not found",
        )

    results = []
    worst_pct = 100.0
    has_excess = False

    for cond_dir in _condition_dirs(run_dir):
        mode_label = cond_dir.name.split("_")[1]
        digest_file = digest_dir / f"digest_{mode_label}.jsonl"
        if not digest_file.exists():
            results.append(f"{mode_label}: digest file missing")
            worst_pct = 0.0
            continue

        # Count failed episodes
        failed_count = 0
        for sf in _list_summaries(cond_dir):
            data = _load_json(sf)
            if data and not data.get("success", False):
                failed_count += 1

        if failed_count == 0:
            results.append(f"{mode_label}: 0 failed episodes, skip")
            continue

        # Count digest lines and unique task_ids
        digest_lines = 0
        digest_task_ids: Set[Any] = set()
        with open(digest_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                digest_lines += 1
                try:
                    rec = json.loads(line)
                    tid = rec.get("task_id")
                    if tid is not None:
                        digest_task_ids.add(tid)
                except json.JSONDecodeError:
                    continue

        pct = (digest_lines / failed_count) * 100 if failed_count > 0 else 100
        n_unique = len(digest_task_ids)
        n_dup = digest_lines - n_unique

        label = f"{mode_label}: {digest_lines}/{failed_count} ({pct:.1f}%)"
        if n_dup > 0:
            label += f" [{n_dup} dup task_ids]"
            has_excess = True
        results.append(label)
        worst_pct = min(worst_pct, pct)

    detail = ", ".join(results)
    if worst_pct < 95 or has_excess:
        return CheckResult(
            "C22", "GLM digest coverage", "warn",
            f"Digest coverage: {detail}",
            results,
        )
    return CheckResult(
        "C22", "GLM digest coverage", "pass",
        f"Digest coverage: {detail}",
    )


# ---------------------------------------------------------------------------
# Group 9: Temporal analysis
# ---------------------------------------------------------------------------


def check_temporal_sr_degradation(run_dir: Path) -> CheckResult:
    """C23: Detect success rate drop in later episodes."""
    warnings = []

    for cond_dir in _condition_dirs(run_dir):
        summaries = _list_summaries(cond_dir)
        if len(summaries) < 9:
            continue  # need at least 3 per segment

        # Sort by task_id
        sorted_sums = sorted(summaries, key=lambda p: _task_id_from_filename(p.name) or 0)

        # Split into 3 equal groups
        n = len(sorted_sums)
        seg_size = n // 3
        segments = [
            sorted_sums[:seg_size],
            sorted_sums[seg_size:2 * seg_size],
            sorted_sums[2 * seg_size:],
        ]

        seg_srs = []
        for seg in segments:
            successes = 0
            for sf in seg:
                data = _load_json(sf)
                if data and data.get("success"):
                    successes += 1
            sr = (successes / len(seg)) * 100 if seg else 0.0
            seg_srs.append(sr)

        early_sr, mid_sr, late_sr = seg_srs
        mode_label = cond_dir.name.split("_")[1]

        # Count early successes for minimum threshold
        early_successes = sum(
            1 for sf in segments[0]
            if (_load_json(sf) or {}).get("success")
        )

        if early_successes >= 3 and late_sr < early_sr * 0.65:
            warnings.append(f"{mode_label}: {early_sr:.1f}% → {mid_sr:.1f}% → {late_sr:.1f}% (early→late)")

    if warnings:
        return CheckResult(
            "C23", "Temporal SR degradation", "warn",
            f"{len(warnings)} conditions show late SR drop: {'; '.join(warnings)}",
            warnings,
        )
    return CheckResult(
        "C23", "Temporal SR degradation", "pass",
        "No significant SR degradation across segments",
    )


def check_auth_drift(run_dir: Path, sample_size: int = 30) -> CheckResult:
    """C24: Detect increasing login-redirect rate in later episodes."""
    site = _infer_site(run_dir)

    login_patterns = {
        "classifieds": "page=login",
        "reddit": "/login",
        "shopping": "/customer/account/login",
    }

    pattern = login_patterns.get(site)
    if not pattern:
        return CheckResult(
            "C24", "Auth drift", "skip",
            f"Cannot infer login pattern for site={site}",
        )

    warnings = []

    for cond_dir in _condition_dirs(run_dir):
        steps_files = _list_steps_files(cond_dir)
        if len(steps_files) < 2 * sample_size:
            continue

        # Sort by task_id
        sorted_files = sorted(steps_files, key=lambda p: _task_id_from_filename(p.name) or 0)
        early_files = sorted_files[:sample_size]
        late_files = sorted_files[-sample_size:]

        def _login_rate(files: List[Path]) -> float:
            episodes_with_login = 0
            for stf in files:
                lines = read_jsonl_dedup(stf)
                has_login = False
                for rec in lines:
                    obs_url = rec.get("obs_url", "") or ""
                    if pattern in obs_url:
                        has_login = True
                        break
                if has_login:
                    episodes_with_login += 1
            return episodes_with_login / len(files) if files else 0.0

        early_rate = _login_rate(early_files)
        late_rate = _login_rate(late_files)

        if late_rate - early_rate > 0.10:
            mode_label = cond_dir.name.split("_")[1]
            warnings.append(
                f"{mode_label}: early={early_rate:.1%} → late={late_rate:.1%} "
                f"(+{(late_rate - early_rate):.1%})"
            )

    if warnings:
        return CheckResult(
            "C24", "Auth drift", "warn",
            f"Login-redirect rate increased in later episodes: {'; '.join(warnings)}",
            warnings,
        )
    return CheckResult(
        "C24", "Auth drift", "pass",
        f"No auth drift detected (site={site})",
    )


def check_reset_pollution(run_dir: Path) -> CheckResult:
    """C25: Detect SR drop after failed require_reset tasks."""
    task_configs_dir = run_dir / "task_configs"
    if not task_configs_dir.exists():
        return CheckResult(
            "C25", "Reset pollution", "skip",
            "task_configs/ directory not found",
        )

    # Find task_ids where require_reset=True
    reset_task_ids: Set[int] = set()
    for cfg_path in task_configs_dir.glob("*.json"):
        cfg = _load_json(cfg_path)
        if cfg and cfg.get("require_reset"):
            tid = cfg.get("task_id")
            if tid is None:
                tid = _task_id_from_filename(cfg_path.name)
            if tid is not None:
                reset_task_ids.add(int(tid))

    if not reset_task_ids:
        return CheckResult(
            "C25", "Reset pollution", "skip",
            "No require_reset tasks found in task_configs/",
        )

    warnings = []

    for cond_dir in _condition_dirs(run_dir):
        summaries = _list_summaries(cond_dir)
        if not summaries:
            continue

        # Build task_id -> summary data map, sorted by task_id
        task_data: List[Tuple[int, Dict[str, Any]]] = []
        for sf in summaries:
            tid = _task_id_from_filename(sf.name)
            data = _load_json(sf)
            if tid is not None and data is not None:
                task_data.append((tid, data))
        task_data.sort(key=lambda x: x[0])

        # Overall SR
        total = len(task_data)
        total_success = sum(1 for _, d in task_data if d.get("success"))
        overall_sr = (total_success / total) if total > 0 else 0.0

        # Find failed reset tasks and check next task's success
        post_reset_total = 0
        post_reset_success = 0
        for i, (tid, data) in enumerate(task_data):
            if tid in reset_task_ids and not data.get("success"):
                # Check next task
                if i + 1 < len(task_data):
                    post_reset_total += 1
                    if task_data[i + 1][1].get("success"):
                        post_reset_success += 1

        if post_reset_total < 5:
            continue

        post_reset_sr = post_reset_success / post_reset_total
        mode_label = cond_dir.name.split("_")[1]

        if post_reset_sr < overall_sr * 0.5:
            warnings.append(
                f"{mode_label}: post-reset-failure SR={post_reset_sr:.1%} "
                f"vs overall SR={overall_sr:.1%} (n={post_reset_total})"
            )

    if warnings:
        return CheckResult(
            "C25", "Reset pollution", "warn",
            f"SR drop after failed reset tasks: {'; '.join(warnings)}",
            warnings,
        )
    return CheckResult(
        "C25", "Reset pollution", "pass",
        "No significant SR drop after failed reset tasks",
    )


# ---------------------------------------------------------------------------
# Group 10: Data consistency
# ---------------------------------------------------------------------------


def check_summary_steps_vs_jsonl(run_dir: Path, sample_size: int) -> CheckResult:
    """C26: Sampled summary.steps should match JSONL line count after dedup."""
    mismatches = []
    checked = 0

    for cond_dir in _condition_dirs(run_dir):
        summaries = _list_summaries(cond_dir)
        for sf in _sample_items(summaries, sample_size):
            data = _load_json(sf)
            if not data:
                continue
            summary_steps = data.get("steps")
            if summary_steps is None:
                continue

            steps_name = sf.name.replace("_summary_v2.json", "_steps_v2.jsonl")
            steps_path = sf.parent / steps_name
            if not steps_path.exists():
                continue

            lines = read_jsonl_dedup(steps_path)
            checked += 1
            if int(summary_steps) != len(lines):
                mode_label = cond_dir.name.split("_")[1]
                tid = _task_id_from_filename(sf.name)
                mismatches.append(
                    f"{mode_label}/task_{tid}: summary.steps={summary_steps} vs jsonl={len(lines)}"
                )

    if mismatches:
        return CheckResult(
            "C26", "Summary steps vs JSONL", "warn",
            f"{len(mismatches)}/{checked} sampled episodes have step count mismatch",
            mismatches,
        )
    return CheckResult(
        "C26", "Summary steps vs JSONL", "pass",
        f"All {checked} sampled episodes have consistent step counts",
    )


def check_zero_cost_episodes(run_dir: Path) -> CheckResult:
    """C27: Non-noise episodes with zero cost (likely missing token accounting)."""
    issues = []
    total_checked = 0

    for cond_dir in _condition_dirs(run_dir):
        mode_label = cond_dir.name.split("_")[1]
        zero_count = 0
        cond_total = 0
        for sf in _list_summaries(cond_dir):
            data = _load_json(sf)
            if not data:
                continue
            if data.get("benchmark_noise"):
                continue
            cond_total += 1
            total_checked += 1
            cost = float(data.get("total_cost_usd", 0) or 0)
            if cost == 0:
                zero_count += 1
        if zero_count > 0:
            issues.append(f"{mode_label}: {zero_count}/{cond_total} zero-cost episodes")

    if issues:
        return CheckResult(
            "C27", "Zero-cost episodes", "warn",
            f"Non-noise episodes with zero cost: {'; '.join(issues)}",
            issues,
        )
    return CheckResult(
        "C27", "Zero-cost episodes", "pass",
        f"All {total_checked} non-noise episodes have cost > 0",
    )


def check_analysis_content_freshness(run_dir: Path) -> CheckResult:
    """C28: Check analysis sub-directories have content and are up-to-date."""
    analysis = run_dir / "analysis"
    issues = []

    # Check key analysis output directories for empty or stale content
    checks = {
        "cross_representation/plots": "cross-rep plots",
        "cross_representation/tables": "cross-rep tables",
    }

    # Find latest condition_summary mtime as reference
    latest_cond_mtime = 0.0
    for cond_dir in _condition_dirs(run_dir):
        cs = cond_dir / "condition_summary_v2.json"
        if cs.exists():
            latest_cond_mtime = max(latest_cond_mtime, cs.stat().st_mtime)

    for rel_path, label in checks.items():
        target = analysis / "results" / rel_path
        if not target.exists():
            continue  # Missing dirs are caught by C02
        files = [f for f in target.iterdir() if f.is_file()]
        if len(files) == 0:
            issues.append(f"{label}: directory exists but is empty")
            continue
        # Check freshness: all files should be newer than latest condition_summary
        if latest_cond_mtime > 0:
            newest_file = max(f.stat().st_mtime for f in files)
            if newest_file < latest_cond_mtime:
                diff_mins = (latest_cond_mtime - newest_file) / 60
                issues.append(f"{label}: {len(files)} files, but {diff_mins:.0f}min older than latest condition_summary")

    if issues:
        return CheckResult(
            "C28", "Analysis content freshness", "warn",
            f"{len(issues)} issue(s): {'; '.join(issues)}",
            issues,
        )
    return CheckResult(
        "C28", "Analysis content freshness", "pass",
        "Analysis sub-directories have content and are up-to-date",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all_checks(
    run_dir: Path,
    compare_dir: Optional[Path] = None,
    sample_size: int = 5,
) -> List[CheckResult]:
    """Execute all 28 checks and return results."""
    results: List[CheckResult] = []

    # Group 1: File existence
    results.append(check_required_files(run_dir))
    results.append(check_optional_files(run_dir))

    # Group 2: Run structure consistency
    results.append(check_condition_completeness(run_dir))
    results.append(check_observation_mode_mapping(run_dir))
    results.append(check_module_flags(run_dir))
    results.append(check_benchmark_field(run_dir, sample_size))

    # Group 3: Episode coverage
    results.append(check_task_coverage(run_dir))
    results.append(check_cross_mode_task_set(run_dir))

    # Group 4: Episode completeness
    results.append(check_summary_null_fields(run_dir))
    results.append(check_score_success_match(run_dir))
    results.append(check_agent_finished_consistency(run_dir, sample_size))

    # Group 5: Step-level completeness
    results.append(check_jsonl_corrupt_lines(run_dir))
    results.append(check_restart_dedup(run_dir))
    results.append(check_required_step_fields(run_dir, sample_size))
    results.append(check_parse_valid_distribution(run_dir))

    # Group 6: Scaffold safety
    results.append(check_fallback_finish(run_dir))
    results.append(check_benchmark_noise_rate(run_dir))
    results.append(check_cross_baseline_symmetry(run_dir, compare_dir))

    # Group 7: Artifact completeness
    results.append(check_screenshot_validity(run_dir, sample_size))
    results.append(check_orphan_artifacts(run_dir))

    # Group 8: Analysis freshness & digest
    results.append(check_analysis_freshness(run_dir))
    results.append(check_glm_digest_coverage(run_dir))

    # Group 9: Temporal analysis
    results.append(check_temporal_sr_degradation(run_dir))
    results.append(check_auth_drift(run_dir))
    results.append(check_reset_pollution(run_dir))

    # Group 10: Data consistency
    results.append(check_summary_steps_vs_jsonl(run_dir, sample_size))
    results.append(check_zero_cost_episodes(run_dir))
    results.append(check_analysis_content_freshness(run_dir))

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# ANSI color codes
_COLORS = {
    "pass": "\033[32m",  # green
    "warn": "\033[33m",  # yellow
    "fail": "\033[31m",  # red
    "skip": "\033[90m",  # gray
    "info": "\033[36m",  # cyan
    "reset": "\033[0m",
}


def _status_tag(status: str, use_color: bool) -> str:
    label = status.upper().center(4)
    if use_color:
        return f"{_COLORS.get(status, '')}{label}{_COLORS['reset']}"
    return label


def print_report(results: List[CheckResult], run_dir: Path) -> None:
    """Print colored report to terminal."""
    use_color = sys.stdout.isatty()

    print()
    header = f"=== P79 Run Validation: {run_dir.name} ==="
    print(header)
    print()

    for r in results:
        tag = _status_tag(r.status, use_color)
        print(f"[{tag}]  {r.check_id}  {r.name}: {r.detail}")

    # Summary
    counts = {"pass": 0, "warn": 0, "fail": 0, "skip": 0, "info": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    print()
    parts = []
    for status in ("pass", "warn", "fail", "skip", "info"):
        if counts.get(status, 0) > 0:
            if use_color:
                parts.append(f"{_COLORS.get(status, '')}{counts[status]} {status.upper()}{_COLORS['reset']}")
            else:
                parts.append(f"{counts[status]} {status.upper()}")
    print(f"Summary: {', '.join(parts)}")
    print()


def write_json_report(
    results: List[CheckResult],
    run_dir: Path,
    output_path: Path,
) -> None:
    """Write JSON report to file."""
    checks = {}
    for r in results:
        entry: Dict[str, Any] = {
            "status": r.status,
            "detail": r.detail,
        }
        if r.items:
            entry["items"] = r.items
        checks[r.check_id] = entry

    counts = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    report = {
        "run_dir": str(run_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "summary": counts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"JSON report written to: {output_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a P79 experiment run directory for data integrity",
    )
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to the run directory (e.g. results/visualwebarena/phase1/B0_3mode_classifieds_20260413)",
    )
    parser.add_argument(
        "--compare-dir", default=None,
        help="Optional second run dir for cross-baseline symmetry check (C18)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write JSON report",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat warnings as failures (exit code 2)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=5,
        help="Number of episodes to sample for sampled checks (default: 5)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        return 1

    compare_dir = Path(args.compare_dir).resolve() if args.compare_dir else None

    # Seed random for reproducible sampling
    random.seed(42)

    results = run_all_checks(run_dir, compare_dir, args.sample_size)

    print_report(results, run_dir)

    if args.output:
        write_json_report(results, run_dir, Path(args.output))

    # Determine exit code
    has_fail = any(r.status == "fail" for r in results)
    has_warn = any(r.status == "warn" for r in results)

    if has_fail:
        return 1
    if has_warn and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
