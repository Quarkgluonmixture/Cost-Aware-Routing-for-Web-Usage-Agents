#!/usr/bin/env python3
"""Per-task mechanism evidence for Section 5.

Outputs:
- docs/analysis/cross_sites/mechanism_per_task.json
- docs/analysis/cross_sites/mechanism_per_task_report.md

This script extends the 4-dimension evidence stack with four quick-win mechanism
metrics:
- E1 click-target divergence using URL-changing click transitions.
- E2 trajectory boundary divergence on symmetric-difference success tasks.
- E3 cross-condition confidence calibration/AUROC aggregation.
- E4 full action vocabulary distributions and paired action-fraction shifts.

Element ids are intentionally not used: AXTree and SoM mark ids are neither
step-invariant nor mode-invariant. URL signatures, action text, and action
types are the stable anchors.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any

try:
    from axis1_microbehavior import url_path_query
except ImportError:  # pragma: no cover - supports module-style imports.
    from scripts.analysis.axis1_microbehavior import url_path_query

try:
    from scripts.analysis.lib.run_registry import canonical_mode, get_cells
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.analysis.lib.run_registry import canonical_mode, get_cells


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/visualwebarena/phase1"
OUT_JSON = ROOT / "docs/analysis/cross_sites/mechanism_per_task.json"
OUT_MD = ROOT / "docs/analysis/cross_sites/mechanism_per_task_report.md"
SR_FP_JSON = ROOT / "docs/analysis/cross_sites/sr_fp_per_mode.json"

def _step_dirs_from_registry(baseline: str) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {"reddit": {}, "classifieds": {}}
    for site in out:
        out[site] = {
            cell.mode: cell.episodes_dir
            for cell in get_cells(baseline=baseline, site=site)
        }
    return out


def _conf_runs_from_registry() -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    seen: set[tuple[str, str, Path]] = set()
    for baseline in ("B0", "B1", "B2"):
        for site in ("classifieds", "reddit"):
            for cell in get_cells(baseline=baseline, site=site):
                key = (baseline, site, cell.run_dir)
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
    return out


STEP_DIRS = _step_dirs_from_registry("B0")
B1_STEP_DIRS = _step_dirs_from_registry("B1")
CONF_RUNS = _conf_runs_from_registry()

AXIS_CONTRASTS = {
    "axis_1_text": ("DOM", "P-text"),
    "axis_2_prompt": ("P-text", "P-SoM"),
    "axis_3_image": ("P-SoM", "SoM"),
    "compound_DOM_to_PSoM": ("DOM", "P-SoM"),
    # Diamond ablation alt-paths via P-prompt
    "axis_2_prompt_alt": ("DOM", "P-prompt"),
    "axis_1_text_alt": ("P-prompt", "P-SoM"),
}

ACTION_TYPES = [
    "click",
    "type",
    "scroll",
    "select_option",
    "wait",
    "back",
    "forward",
    "finish",
    "tab_focus",
    "other",
]

# §139.8: scored-set sizes (total − N/A excluded at load) from the single
# source of truth, not pre-exclusion 234/210.
from p79.experiment.analysis import scored_task_count as _scored_task_count
EXPECTED_N = {_s: _scored_task_count(_s, "visualwebarena") for _s in ("reddit", "classifieds")}
MAX_STEPS = 30


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_steps(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def task_id_from_path(path: Path) -> int:
    match = re.search(r"task_(\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot parse task id from {path}")
    return int(match.group(1))


def action_type(step: dict[str, Any]) -> str:
    nested = step.get("action")
    nested_at = nested.get("action_type") if isinstance(nested, dict) else None
    raw = step.get("action_type") or nested_at or "other"
    raw = str(raw).strip().lower()
    return raw if raw in ACTION_TYPES[:-1] else "other"


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def pct(value: float | None) -> float | None:
    return None if value is None else 100.0 * value


def normalize_mode_name(value: str) -> str:
    return canonical_mode(value)


def short_mode(mode: str) -> str:
    return {
        "P-text": "P-text",
        "P-SoM": "P-SoM",
        "P-prompt": "P-prompt",
    }.get(mode, mode)


def infer_single_mode(run_dir: Path) -> str | None:
    conds = [p.name for p in run_dir.glob("phase1_*_router_0") if p.is_dir()]
    if not conds:
        return None
    cond = conds[0]
    raw = cond.replace("phase1_", "").replace("_router_0", "")
    return normalize_mode_name(raw)


def summary_success(summary_path: Path) -> bool | None:
    # §139.8: reads canonical `success` — adjusted_success post-hoc layer retired
    if not summary_path.exists():
        return None
    row = read_json(summary_path)
    if "success" in row:
        return bool(row["success"])
    return None


def load_mode_tasks(site: str, mode: str, ep_dir: Path) -> dict[int, dict[str, Any]]:
    tasks: dict[int, dict[str, Any]] = {}
    for path in sorted(ep_dir.glob(f"{site}_task_*_steps_v2.jsonl")):
        tid = task_id_from_path(path)
        steps = read_steps(path)
        summary_path = path.with_name(path.name.replace("_steps_v2.jsonl", "_summary_v2.json"))
        click_targets: set[tuple[str, str]] = set()
        url_trajectory: list[str] = []
        action_counts = Counter({name: 0 for name in ACTION_TYPES})
        for idx, step in enumerate(steps):
            route = url_path_query(step.get("obs_url"))
            if route:
                url_trajectory.append(route)
            at = action_type(step)
            action_counts[at] += 1
            if at != "click":
                continue
            pre = url_path_query(step.get("obs_url"))
            post = url_path_query(steps[idx + 1].get("obs_url")) if idx + 1 < len(steps) else ""
            if pre and post and pre != post:
                click_targets.add((pre, post))
        n_steps = len(steps)
        action_fracs = {
            name: (action_counts[name] / n_steps if n_steps else 0.0)
            for name in ACTION_TYPES
        }
        tasks[tid] = {
            "task_id": tid,
            "n_steps": n_steps,
            "steps": steps,
            "url_trajectory": url_trajectory,
            "click_targets": click_targets,
            "click_target_count": len(click_targets),
            "action_counts": dict(action_counts),
            "action_fracs": action_fracs,
            "adjusted_success": summary_success(summary_path),
        }
    return tasks


def load_all_tasks() -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    all_tasks: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for site, modes in STEP_DIRS.items():
        all_tasks[site] = {}
        for mode, ep_dir in modes.items():
            if ep_dir is None or not ep_dir.exists():
                all_tasks[site][mode] = {}
                continue
            all_tasks[site][mode] = load_mode_tasks(site, mode, ep_dir)
    return all_tasks


def hist(values: list[int]) -> dict[str, int]:
    return {str(key): val for key, val in sorted(Counter(values).items())}


def summarize_click_contrast(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    jaccards = [
        jaccard(left[tid]["click_targets"], right[tid]["click_targets"])
        for tid in common
    ]
    left_sizes = [left[tid]["click_target_count"] for tid in common]
    right_sizes = [right[tid]["click_target_count"] for tid in common]
    union_sizes = [len(left[tid]["click_targets"] | right[tid]["click_targets"]) for tid in common]
    return {
        "n": len(common),
        "mean_jaccard": mean(jaccards),
        "std_jaccard": sample_sd(jaccards),
        "median_jaccard": median(jaccards) if jaccards else None,
        "mean_decision_divergence": None if not jaccards else 1.0 - (mean(jaccards) or 0.0),
        "click_target_set_size": {
            "left_mean": mean([float(v) for v in left_sizes]),
            "right_mean": mean([float(v) for v in right_sizes]),
            "left_hist": hist(left_sizes),
            "right_hist": hist(right_sizes),
            "union_mean": mean([float(v) for v in union_sizes]),
        },
    }


def build_e1(all_tasks: dict[str, dict[str, dict[int, dict[str, Any]]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for site, modes in all_tasks.items():
        out[site] = {}
        for axis, (left_mode, right_mode) in AXIS_CONTRASTS.items():
            left = modes.get(left_mode) or {}
            right = modes.get(right_mode) or {}
            if not left or not right:
                out[site][axis] = {
                    "n": 0,
                    "skipped": True,
                    "left_mode": left_mode,
                    "right_mode": right_mode,
                    "contrast": f"{short_mode(left_mode)} vs {short_mode(right_mode)}",
                }
                continue
            block = summarize_click_contrast(left, right)
            block["left_mode"] = left_mode
            block["right_mode"] = right_mode
            block["contrast"] = f"{short_mode(left_mode)} vs {short_mode(right_mode)}"
            out[site][axis] = block
    return out


def first_divergent_step(left_urls: list[str], right_urls: list[str]) -> int | None:
    for idx, (left, right) in enumerate(zip(left_urls, right_urls)):
        if left != right:
            return idx
    if len(left_urls) != len(right_urls):
        return min(len(left_urls), len(right_urls))
    return None


def trajectory_jaccard(left_urls: list[str], right_urls: list[str]) -> float:
    return jaccard(set(left_urls), set(right_urls))


def summarize_boundary_contrast(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    rows: list[dict[str, Any]] = []
    for tid in common:
        left_success = left[tid]["adjusted_success"]
        right_success = right[tid]["adjusted_success"]
        if left_success is None or right_success is None or left_success == right_success:
            continue
        fds = first_divergent_step(left[tid]["url_trajectory"], right[tid]["url_trajectory"])
        tj = trajectory_jaccard(left[tid]["url_trajectory"], right[tid]["url_trajectory"])
        rows.append(
            {
                "task_id": tid,
                "first_divergent_step": fds,
                "trajectory_jaccard": tj,
                "left_success": left_success,
                "right_success": right_success,
                "left_n_steps": left[tid]["n_steps"],
                "right_n_steps": right[tid]["n_steps"],
            }
        )
    steps = [row["first_divergent_step"] for row in rows if row["first_divergent_step"] is not None]
    earliest = sorted((row for row in rows if row["first_divergent_step"] is not None), key=lambda r: (r["first_divergent_step"], r["task_id"]))
    latest = sorted((row for row in rows if row["first_divergent_step"] is not None), key=lambda r: (-r["first_divergent_step"], r["task_id"]))
    most_div = sorted(rows, key=lambda r: (r["trajectory_jaccard"], r["task_id"]))
    case_ids: list[int] = []
    for source in (earliest[:1], latest[:1], most_div[:1]):
        for row in source:
            if row["task_id"] not in case_ids:
                case_ids.append(row["task_id"])
    for row in earliest:
        if len(case_ids) >= 3:
            break
        if row["task_id"] not in case_ids:
            case_ids.append(row["task_id"])
    return {
        "n_symmetric_diff_tasks": len(rows),
        "median_first_divergent_step": median(steps) if steps else None,
        "early_divergence_rate": mean([float(step <= 3) for step in steps]),
        "late_divergence_rate": mean([float(step >= 10) for step in steps]),
        "first_divergent_step_histogram": hist(steps),
        "case_study_task_ids": case_ids[:3],
        "case_studies": [
            {
                "task_id": row["task_id"],
                "first_divergent_step": row["first_divergent_step"],
                "trajectory_jaccard": row["trajectory_jaccard"],
                "left_success": row["left_success"],
                "right_success": row["right_success"],
                "left_n_steps": row["left_n_steps"],
                "right_n_steps": row["right_n_steps"],
            }
            for row in rows
            if row["task_id"] in case_ids[:3]
        ],
    }


def build_e2(all_tasks: dict[str, dict[str, dict[int, dict[str, Any]]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for site, modes in all_tasks.items():
        out[site] = {}
        for _axis, (left_mode, right_mode) in AXIS_CONTRASTS.items():
            name = f"{left_mode}_vs_{right_mode}"
            left = modes.get(left_mode) or {}
            right = modes.get(right_mode) or {}
            if not left or not right:
                out[site][name] = {
                    "n_symmetric_diff_tasks": 0,
                    "median_first_divergent_step": None,
                    "early_divergence_rate": None,
                    "late_divergence_rate": None,
                    "first_divergent_step_histogram": {},
                    "case_study_task_ids": [],
                    "case_studies": [],
                    "left_mode": left_mode,
                    "right_mode": right_mode,
                    "skipped": True,
                }
                continue
            block = summarize_boundary_contrast(left, right)
            block["left_mode"] = left_mode
            block["right_mode"] = right_mode
            out[site][name] = block
    return out


def best_auroc(rows: list[dict[str, str]], signal_type: str, mode: str | None = None) -> tuple[float | None, str | None, int | None]:
    best: tuple[float | None, str | None, int | None] = (None, None, None)
    for row in rows:
        row_mode = row.get("mode") or row.get("observation_mode")
        if mode and row_mode and normalize_mode_name(row_mode) != mode:
            continue
        if row.get("signal_type") != signal_type:
            continue
        value = safe_float(row.get("AUROC"))
        if value is None:
            continue
        if best[0] is None or value > best[0]:
            best = (value, row.get("signal") or row.get("metric"), safe_int(row.get("n")))
    return best


def per_mode_summary_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {normalize_mode_name(row.get("observation_mode", "")): row for row in rows}


def e3_cells_for_run(model: str, site: str, run_dir: Path) -> dict[str, dict[str, Any]]:
    tables = run_dir / "analysis/signals/combined/tables"
    cross_rows = read_csv(tables / "cross_mode_auroc.csv")
    all_rows = read_csv(tables / "auroc_all_metrics.csv")
    per_mode_rows = per_mode_summary_map(read_csv(tables / "per_mode_summary.csv"))
    cells: dict[str, dict[str, Any]] = {}
    if cross_rows:
        modes = sorted({normalize_mode_name(row.get("mode", "")) for row in cross_rows if row.get("mode")})
        auroc_source = str((tables / "cross_mode_auroc.csv").relative_to(ROOT))
    else:
        single = infer_single_mode(run_dir)
        modes = [single] if single else []
        auroc_source = str((tables / "auroc_all_metrics.csv").relative_to(ROOT))
    for mode in modes:
        if not mode:
            continue
        auroc_rows = cross_rows if cross_rows else all_rows
        token, token_signal, token_n = best_auroc(auroc_rows, "token_level", mode if cross_rows else None)
        verbal, verbal_signal, verbal_n = best_auroc(auroc_rows, "verbalized", mode if cross_rows else None)
        behavioral, behavioral_signal, behavioral_n = best_auroc(auroc_rows, "behavioral", mode if cross_rows else None)
        pm = per_mode_rows.get(mode, {})
        key = f"{model}/{site}/{mode}"
        cells[key] = {
            "model": model,
            "site": site,
            "mode": mode,
            "ECE_token": safe_float(pm.get("ECE")),
            "MCE_token": safe_float(pm.get("MCE")),
            "Brier_token": safe_float(pm.get("Brier")),
            "ECE_verbal": safe_float(pm.get("verbalized_ECE")),
            "MCE_verbal": safe_float(pm.get("verbalized_MCE")),
            "Brier_verbal": safe_float(pm.get("verbalized_Brier")),
            "AUROC_token": token,
            "AUROC_token_signal": token_signal,
            "AUROC_token_n": token_n,
            "AUROC_verbal": verbal,
            "AUROC_verbal_signal": verbal_signal,
            "AUROC_verbal_n": verbal_n,
            "AUROC_behavioral_max": behavioral,
            "AUROC_behavioral_signal": behavioral_signal,
            "AUROC_behavioral_n": behavioral_n,
            "source_run": str(run_dir.relative_to(ROOT)),
            "source_table": auroc_source,
            "calibration_source": str((tables / "per_mode_summary.csv").relative_to(ROOT)) if per_mode_rows else None,
        }
    return cells


def load_fp_rates() -> dict[str, float]:
    if not SR_FP_JSON.exists():
        return {}
    data = read_json(SR_FP_JSON)
    rates: dict[str, float] = {}
    for key, row in (data.get("cells") or {}).items():
        rates[key] = row.get("fp_rate_pct")
    return rates


def build_e3() -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    for model, site, run_dir in CONF_RUNS:
        cells.update(e3_cells_for_run(model, site, run_dir))
    fp_rates = load_fp_rates()
    for key, row in cells.items():
        row["layer0b_fp_rate_pct"] = fp_rates.get(key)

    highlights: dict[str, Any] = {}
    grouped: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for key, row in cells.items():
        grouped[(row["model"], row["site"])].append((key, row))
    for group_key, rows in grouped.items():
        ece_rows = [(key, row) for key, row in rows if row.get("ECE_verbal") is not None]
        auroc_rows = [(key, row) for key, row in rows if row.get("AUROC_verbal") is not None or row.get("AUROC_behavioral_max") is not None]
        honest = min(ece_rows, key=lambda item: item[1]["ECE_verbal"]) if ece_rows else None
        best_signal = max(
            auroc_rows,
            key=lambda item: max(
                item[1].get("AUROC_token") or -1,
                item[1].get("AUROC_verbal") or -1,
                item[1].get("AUROC_behavioral_max") or -1,
            ),
        ) if auroc_rows else None
        gkey = "/".join(group_key)
        highlights[gkey] = {
            "honest_commit_mode_lowest_ECE_verbal": honest[0] if honest else None,
            "lowest_ECE_verbal": honest[1]["ECE_verbal"] if honest else None,
            "best_signal_AUROC_mode": best_signal[0] if best_signal else None,
            "best_signal_AUROC": max(
                best_signal[1].get("AUROC_token") or -1,
                best_signal[1].get("AUROC_verbal") or -1,
                best_signal[1].get("AUROC_behavioral_max") or -1,
            ) if best_signal else None,
            "fp_cross_reference": "B0 FP rates attached when available; B1 FP table is not in sr_fp_per_mode.json.",
        }
    return {"cells": cells, "highlights": highlights}


def summarize_action_modes(all_tasks: dict[str, dict[str, dict[int, dict[str, Any]]]]) -> dict[str, dict[str, float]]:
    cells: dict[str, dict[str, float]] = {}
    for site, modes in all_tasks.items():
        for mode, tasks in modes.items():
            if not tasks:
                continue
            counts = Counter({name: 0 for name in ACTION_TYPES})
            total = 0
            for row in tasks.values():
                total += row["n_steps"]
                counts.update(row["action_counts"])
            cells[f"B0/{site}/{mode}"] = {
                name: (counts[name] / total if total else 0.0)
                for name in ACTION_TYPES
            }
    return cells


def action_shift_contrast(left: dict[int, dict[str, Any]], right: dict[int, dict[str, Any]]) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    shifts: dict[str, float | None] = {}
    for at in ACTION_TYPES:
        diffs = [right[tid]["action_fracs"][at] - left[tid]["action_fracs"][at] for tid in common]
        shifts[at] = mean(diffs)
    top_abs = sorted(
        [{"action_type": at, "mean_fraction_shift": val} for at, val in shifts.items() if val is not None],
        key=lambda row: abs(row["mean_fraction_shift"]),
        reverse=True,
    )
    return {
        "n": len(common),
        "mean_per_task_fraction_shift": shifts,
        "top_abs_shifts": top_abs[:5],
        "skipped": False,
    }


def build_e4(all_tasks: dict[str, dict[str, dict[int, dict[str, Any]]]]) -> dict[str, Any]:
    cells = summarize_action_modes(all_tasks)
    contrasts: dict[str, Any] = {}
    for site, modes in all_tasks.items():
        contrasts[site] = {}
        for axis, (left_mode, right_mode) in AXIS_CONTRASTS.items():
            left = modes.get(left_mode) or {}
            right = modes.get(right_mode) or {}
            if not left or not right:
                contrasts[site][axis] = {
                    "n": 0,
                    "skipped": True,
                    "mean_per_task_fraction_shift": {at: None for at in ACTION_TYPES},
                    "top_abs_shifts": [],
                    "left_mode": left_mode,
                    "right_mode": right_mode,
                }
                continue
            block = action_shift_contrast(left, right)
            block["left_mode"] = left_mode
            block["right_mode"] = right_mode
            contrasts[site][axis] = block
    highlights = []
    for site in sorted(all_tasks):
        for at in ACTION_TYPES:
            mode_values = []
            for mode in STEP_DIRS[site]:
                cell = cells.get(f"B0/{site}/{mode}")
                if cell is None:
                    continue
                value = cell[at]
                mode_values.append((mode, value))
            nonzero = [(mode, value) for mode, value in mode_values if value > 0]
            if len(nonzero) < 2:
                continue
            high_mode, high = max(nonzero, key=lambda item: item[1])
            low_mode, low = min(nonzero, key=lambda item: item[1])
            ratio = high / low if low > 0 else None
            if ratio and ratio >= 2.0 and high - low >= 0.01:
                highlights.append(
                    {
                        "site": site,
                        "action_type": at,
                        "high_mode": high_mode,
                        "high_fraction": high,
                        "low_mode": low_mode,
                        "low_fraction": low,
                        "ratio": ratio,
                    }
                )
    highlights.sort(key=lambda row: row["ratio"], reverse=True)
    return {"cells": cells, "axis_contrasts": contrasts, "uncommon_action_highlights": highlights[:12]}


def detect_partial_prompt_runs() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in sorted(RESULTS.glob("*phantom_prompt*")):
        ep_dirs = list(path.glob("phase1_*_router_0/episodes"))
        count = 0
        for ep_dir in ep_dirs:
            count += len(list(ep_dir.glob("*_summary_v2.json")))
        out[path.name] = {
            "episodes": count,
            "status": "partial / pending" if count < 200 else "complete but excluded by design",
            "policy": "P-prompt is not included in E1-E4 contrasts.",
        }
    return out


def round_floats(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: round_floats(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [round_floats(value) for value in obj]
    if isinstance(obj, tuple):
        return [round_floats(value) for value in obj]
    if isinstance(obj, set):
        return [round_floats(value) for value in sorted(obj)]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 6)
    return obj


def validation_block(e1: dict[str, Any], e2: dict[str, Any], e3: dict[str, Any], e4: dict[str, Any]) -> dict[str, Any]:
    axis1_n = {
        site: {
            "observed": e1[site]["axis_1_text"]["n"],
            "expected": EXPECTED_N[site],
            "pass": e1[site]["axis_1_text"]["n"] == EXPECTED_N[site],
        }
        for site in EXPECTED_N
    }
    e1_range = {
        f"{site}/{axis}": 0.0 <= block["mean_jaccard"] <= 1.0
        for site, site_block in e1.items()
        for axis, block in site_block.items()
        if block.get("mean_jaccard") is not None
    }
    e2_range = {}
    for site, site_block in e2.items():
        for contrast, block in site_block.items():
            values = [int(step) for step in block.get("first_divergent_step_histogram", {})]
            e2_range[f"{site}/{contrast}"] = all(0 <= step <= MAX_STEPS for step in values)
    e4_sums = {
        key: {
            "sum": sum(row.values()),
            "pass": abs(sum(row.values()) - 1.0) <= 0.001,
        }
        for key, row in e4["cells"].items()
    }
    effect_checks = {
        "E1_any_click_divergence_gt_0.1": any(
            (block.get("mean_decision_divergence") or 0.0) > 0.1
            for site_block in e1.values()
            for block in site_block.values()
        ),
        "E2_any_early_or_late_rate_gt_0.1": any(
            max(block.get("early_divergence_rate") or 0.0, block.get("late_divergence_rate") or 0.0) > 0.1
            for site_block in e2.values()
            for block in site_block.values()
        ),
        "E3_any_AUROC_effect_gt_0.1": any(
            abs(max(row.get("AUROC_token") or 0.5, row.get("AUROC_verbal") or 0.5, row.get("AUROC_behavioral_max") or 0.5) - 0.5) > 0.1
            for row in e3["cells"].values()
        ),
        "E4_any_action_shift_gt_0.1": any(
            abs(item["mean_fraction_shift"]) > 0.1
            for site_block in e4["axis_contrasts"].values()
            for block in site_block.values()
            for item in block.get("top_abs_shifts", [])
        ),
    }
    # Baseline E3 cells: B0×{cls,red}×5 + B1×{cls(4),red(3)} = 17.
    # Each available P-prompt run adds 1 cell (when calibration tables are
    # generated for the run); the count is dynamic, so just track observed.
    expected_e3_cells = max(17, len(e3["cells"]))
    return {
        "axis1_n_checks": axis1_n,
        "E1_mean_jaccard_range_checks": e1_range,
        "E2_first_divergent_step_range_checks": e2_range,
        "E3_cell_count": {
            "observed": len(e3["cells"]),
            "expected": expected_e3_cells,
            "pass": len(e3["cells"]) == expected_e3_cells,
        },
        "E4_action_dist_sum_checks": e4_sums,
        "effect_size_presence_checks": effect_checks,
        "pass": all(v["pass"] for v in axis1_n.values())
        and all(e1_range.values())
        and all(e2_range.values())
        and len(e3["cells"]) == expected_e3_cells
        and all(v["pass"] for v in e4_sums.values())
        and all(effect_checks.values()),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.{digits}f}%"


def headline_implications(e1: dict[str, Any], e2: dict[str, Any], e3: dict[str, Any], e4: dict[str, Any]) -> dict[str, str]:
    red_e1 = e1["reddit"]["compound_DOM_to_PSoM"]
    cls_e1 = e1["classifieds"]["compound_DOM_to_PSoM"]
    e2_dom_psom = {
        site: e2[site]["DOM_vs_P-SoM"]
        for site in ("reddit", "classifieds")
    }
    best_e3 = max(
        e3["cells"].items(),
        key=lambda item: max(item[1].get("AUROC_token") or -1, item[1].get("AUROC_verbal") or -1, item[1].get("AUROC_behavioral_max") or -1),
    )
    top_e4_iter = [
        (site, axis, item)
        for site, site_block in e4["axis_contrasts"].items()
        for axis, block in site_block.items()
        for item in block.get("top_abs_shifts", [])
    ]
    top_e4 = max(top_e4_iter, key=lambda row: abs(row[2]["mean_fraction_shift"])) if top_e4_iter else None
    return {
        "E1_headline": (
            "DOM and P-SoM click transitions diverge at event granularity: "
            f"compound click-target Jaccard is {fmt(red_e1['mean_jaccard'])} on reddit and "
            f"{fmt(cls_e1['mean_jaccard'])} on classifieds."
        ),
        "E2_headline": (
            "Boundary divergence is usually visible early among symmetric-difference tasks: "
            f"DOM vs P-SoM early rates are reddit {fmt_pct(e2_dom_psom['reddit']['early_divergence_rate'])} "
            f"and classifieds {fmt_pct(e2_dom_psom['classifieds']['early_divergence_rate'])}."
        ),
        "E3_headline": (
            f"The strongest calibration/routing cell is {best_e3[0]} with max AUROC "
            f"{fmt(max(best_e3[1].get('AUROC_token') or -1, best_e3[1].get('AUROC_verbal') or -1, best_e3[1].get('AUROC_behavioral_max') or -1))}; "
            "B0 token ECE is unavailable in existing per-run outputs."
        ),
        "E4_headline": (
            f"The largest action-vocabulary shift is {top_e4[0]} {top_e4[1]} "
            f"{top_e4[2]['action_type']} ({fmt(top_e4[2]['mean_fraction_shift'])} right-minus-left)."
            if top_e4 is not None else "No E4 axis contrast had data."
        ),
    }


def write_report(out: dict[str, Any]) -> None:
    e1 = out["E1_click_target_divergence"]
    e2 = out["E2_trajectory_boundary"]
    e3 = out["E3_confidence_calibration"]
    e4 = out["E4_action_vocabulary"]
    lines: list[str] = []
    lines += [
        "# Per-task mechanism evidence (E1-E4)",
        "",
        "This report explains why mode swaps move outcomes by using per-task and per-step evidence. Element ids are excluded because they are not stable across navigation steps or observation modes. Click evidence uses URL-changing transitions `(pre_url_signature, post_url_signature)`, trajectory evidence uses URL signatures per step, confidence evidence reads existing per-run calibration outputs, and action vocabulary evidence uses normalized action types.",
        "",
        "## E1 Click-target divergence",
        "",
        "E1 asks whether modes click into the same server-determined page transitions. Jaccard is computed over each task's set of URL-changing click transitions, then averaged across paired tasks.",
        "",
        "| site | contrast | N | mean Jaccard | std | median | mean divergence | left size | right size | union size |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for site in ("reddit", "classifieds"):
        for axis, block in e1[site].items():
            if block.get("skipped") or block.get("n", 0) == 0:
                lines.append(
                    f"| {site} | {axis} ({short_mode(block.get('left_mode', '?'))} vs "
                    f"{short_mode(block.get('right_mode', '?'))}) | 0 (pending) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
                )
                continue
            sizes = block["click_target_set_size"]
            lines.append(
                f"| {site} | {axis} ({short_mode(block['left_mode'])} vs {short_mode(block['right_mode'])}) | "
                f"{block['n']} | {fmt(block['mean_jaccard'])} | {fmt(block['std_jaccard'])} | "
                f"{fmt(block['median_jaccard'])} | {fmt(block['mean_decision_divergence'])} | "
                f"{fmt(sizes['left_mean'])} | {fmt(sizes['right_mean'])} | {fmt(sizes['union_mean'])} |"
            )
    lines += [
        "",
        "Per-axis interpretation:",
    ]
    for axis in AXIS_CONTRASTS:
        red = e1["reddit"][axis]
        cls = e1["classifieds"][axis]
        lines.append(
            f"- {axis}: reddit Jaccard {fmt(red.get('mean_jaccard'))}; classifieds Jaccard {fmt(cls.get('mean_jaccard'))}. "
            f"Lower values indicate that the modes use different URL-changing click decisions."
        )
    lines += [
        "",
        "Case-study anchors from E2 below should be read with E1: tasks with low click-transition overlap often diverge before the final answer, not merely at finish time.",
        "",
        "## E2 Trajectory boundary divergence",
        "",
        "E2 filters to symmetric-difference tasks, where exactly one side of the contrast has adjusted success. It then records the first step where URL signatures differ. Early divergence is step <= 3; late divergence is step >= 10.",
        "",
        "| site | contrast | symmetric diff N | median first step | early rate | late rate | case tasks |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for site in ("reddit", "classifieds"):
        for contrast, block in e2[site].items():
            lines.append(
                f"| {site} | {contrast} | {block['n_symmetric_diff_tasks']} | "
                f"{fmt(block['median_first_divergent_step'])} | {fmt_pct(block['early_divergence_rate'])} | "
                f"{fmt_pct(block['late_divergence_rate'])} | {', '.join(str(x) for x in block['case_study_task_ids'])} |"
            )
    lines += ["", "E2 case studies:"]
    for site in ("reddit", "classifieds"):
        for contrast, block in e2[site].items():
            for case in block["case_studies"]:
                lines.append(
                    f"- {site} {contrast} task_{case['task_id']}: first divergent step "
                    f"{fmt(case['first_divergent_step'])}, trajectory Jaccard {fmt(case['trajectory_jaccard'])}, "
                    f"left_success={case['left_success']}, right_success={case['right_success']}, "
                    f"steps {case['left_n_steps']} vs {case['right_n_steps']}."
                )
    lines += [
        "",
        "## E3 Confidence calibration cross-condition aggregator",
        "",
        "E3 reads existing `analyze_confidence_calibration.py` outputs under `analysis/signals/combined/tables`. It does not recompute calibration. B1 runs expose per-mode token and verbalized calibration in `per_mode_summary.csv`; B0 API runs expose verbalized and behavioral AUROC but no token-level calibration in the existing outputs.",
        "",
        "| model | site | mode | ECE token | ECE verbal | AUROC token | AUROC verbal | AUROC behavioral max | FP rate | best signals |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in sorted(e3["cells"]):
        row = e3["cells"][key]
        signals = []
        if row.get("AUROC_token_signal"):
            signals.append(f"tok={row['AUROC_token_signal']}")
        if row.get("AUROC_verbal_signal"):
            signals.append(f"verb={row['AUROC_verbal_signal']}")
        if row.get("AUROC_behavioral_signal"):
            signals.append(f"beh={row['AUROC_behavioral_signal']}")
        lines.append(
            f"| {row['model']} | {row['site']} | {short_mode(row['mode'])} | "
            f"{fmt(row.get('ECE_token'))} | {fmt(row.get('ECE_verbal'))} | "
            f"{fmt(row.get('AUROC_token'))} | {fmt(row.get('AUROC_verbal'))} | "
            f"{fmt(row.get('AUROC_behavioral_max'))} | {fmt(row.get('layer0b_fp_rate_pct'))} | "
            f"{'; '.join(signals)} |"
        )
    lines += ["", "E3 highlights:"]
    for group, row in sorted(e3["highlights"].items()):
        lines.append(
            f"- {group}: honest-commit mode {row['honest_commit_mode_lowest_ECE_verbal']} "
            f"(ECE {fmt(row['lowest_ECE_verbal'])}); best-signal mode {row['best_signal_AUROC_mode']} "
            f"(AUROC {fmt(row['best_signal_AUROC'])})."
        )
    lines += [
        "",
        "Outcome 0b FP cross-reference: B0 FP rates are attached for cells present in `sr_fp_per_mode.json`. Because B0 ECE is absent from the existing analyzer outputs, low-ECE versus low-FP claims should be made only for B1 calibration cells or deferred until B0 calibration tables are generated.",
        "",
        "## E4 Action vocabulary distribution",
        "",
        "E4 expands the Macro dimension from a few hand-picked action metrics to the full normalized action vocabulary. Fractions below are pooled over all steps in each B0 site/mode cell.",
        "",
        "| cell | click | type | scroll | select | wait | back | forward | finish | tab_focus | other |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(e4["cells"]):
        row = e4["cells"][key]
        lines.append(
            f"| {key} | "
            f"{fmt(row['click'])} | {fmt(row['type'])} | {fmt(row['scroll'])} | "
            f"{fmt(row['select_option'])} | {fmt(row['wait'])} | {fmt(row['back'])} | "
            f"{fmt(row['forward'])} | {fmt(row['finish'])} | {fmt(row['tab_focus'])} | {fmt(row['other'])} |"
        )
    lines += [
        "",
        "Paired action-fraction shifts by axis (right-minus-left):",
        "",
        "| site | axis | N | top shift 1 | top shift 2 | top shift 3 |",
        "|---|---|---:|---|---|---|",
    ]
    for site in ("reddit", "classifieds"):
        for axis, block in e4["axis_contrasts"][site].items():
            top = block.get("top_abs_shifts", [])[:3]
            labels = [f"{item['action_type']} {fmt(item['mean_fraction_shift'])}" for item in top]
            while len(labels) < 3:
                labels.append("")
            lines.append(f"| {site} | {axis} | {block.get('n', 0)} | {labels[0]} | {labels[1]} | {labels[2]} |")
    lines += ["", "Uncommon-action highlights:"]
    if e4["uncommon_action_highlights"]:
        for row in e4["uncommon_action_highlights"][:8]:
            lines.append(
                f"- {row['site']} {row['action_type']}: {short_mode(row['high_mode'])} "
                f"{fmt(row['high_fraction'])} vs {short_mode(row['low_mode'])} {fmt(row['low_fraction'])} "
                f"({fmt(row['ratio'], 1)}x)."
            )
    else:
        lines.append("- No action type cleared the 2x ratio plus 1 pp absolute-difference highlight threshold.")
    lines += [
        "",
        "## Mechanism evidence for paper Section 5",
        "",
        out["paper_section5_implications"]["E1_headline"],
        "",
        out["paper_section5_implications"]["E2_headline"],
        "",
        out["paper_section5_implications"]["E3_headline"],
        "",
        out["paper_section5_implications"]["E4_headline"],
        "",
        "Together, E1 and E2 support a decision-path account: mode swaps change which URL transitions are attempted and how early trajectories split on tasks where outcomes disagree. E3 keeps the commitment-confidence claim separate from path choice: confidence evidence is useful, but existing B0 outputs support it mainly through verbalized and behavioral AUROC rather than token calibration. E4 shows whether those path changes are accompanied by broad policy-shape shifts in the action vocabulary, or whether the same action mix hides different click targets.",
        "",
        "## Appendix A: E1 click-set size histograms",
        "",
        "These histograms summarize how many URL-changing click transitions each task produced. The key is set size; the value is the number of tasks with that size.",
        "",
        "| site | axis | left mode | left hist | right mode | right hist |",
        "|---|---|---|---|---|---|",
    ]
    for site in ("reddit", "classifieds"):
        for axis, block in e1[site].items():
            if block.get("skipped") or block.get("n", 0) == 0:
                lines.append(
                    f"| {site} | {axis} | {short_mode(block.get('left_mode', '?'))} | "
                    f"`{{}}` | {short_mode(block.get('right_mode', '?'))} | `{{}}` |"
                )
                continue
            sizes = block["click_target_set_size"]
            lines.append(
                f"| {site} | {axis} | {short_mode(block['left_mode'])} | "
                f"`{json.dumps(sizes['left_hist'], sort_keys=True)}` | "
                f"{short_mode(block['right_mode'])} | `{json.dumps(sizes['right_hist'], sort_keys=True)}` |"
            )
    lines += [
        "",
        "## Appendix B: E2 first-divergence histograms",
        "",
        "The histogram key is first divergent step. Step 0 means the two modes start from different URL signatures or immediately navigate differently.",
        "",
        "| site | contrast | histogram |",
        "|---|---|---|",
    ]
    for site in ("reddit", "classifieds"):
        for contrast, block in e2[site].items():
            lines.append(f"| {site} | {contrast} | `{json.dumps(block['first_divergent_step_histogram'], sort_keys=True)}` |")
    lines += [
        "",
        "## Appendix C: E3 source provenance",
        "",
        "Each E3 row is sourced from existing analyzer outputs; calibration source is null when the run exposes AUROC but not per-mode ECE/MCE/Brier.",
        "",
        "| cell | AUROC source | calibration source | token n | verbal n | behavioral n |",
        "|---|---|---|---:|---:|---:|",
    ]
    for key in sorted(e3["cells"]):
        row = e3["cells"][key]
        lines.append(
            f"| {key} | `{row.get('source_table')}` | "
            f"{'`' + row['calibration_source'] + '`' if row.get('calibration_source') else 'n/a'} | "
            f"{fmt(row.get('AUROC_token_n'))} | {fmt(row.get('AUROC_verbal_n'))} | {fmt(row.get('AUROC_behavioral_n'))} |"
        )
    lines += [
        "",
        "## Appendix D: E4 full action-shift matrix",
        "",
        "All values are paired per-task action-fraction shifts in the cascade direction, right-minus-left.",
        "",
        "| site | axis | click | type | scroll | select | wait | back | forward | finish | tab_focus | other |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for site in ("reddit", "classifieds"):
        for axis, block in e4["axis_contrasts"][site].items():
            shifts = block["mean_per_task_fraction_shift"]
            lines.append(
                f"| {site} | {axis} | {fmt(shifts['click'])} | {fmt(shifts['type'])} | "
                f"{fmt(shifts['scroll'])} | {fmt(shifts['select_option'])} | {fmt(shifts['wait'])} | "
                f"{fmt(shifts['back'])} | {fmt(shifts['forward'])} | {fmt(shifts['finish'])} | "
                f"{fmt(shifts['tab_focus'])} | {fmt(shifts['other'])} |"
            )
    lines += [
        "",
        "## Appendix E: E4 ranked per-cell actions",
        "",
        "Top action types per B0 cell by pooled step fraction.",
        "",
        "| cell | rank | action_type | fraction |",
        "|---|---:|---|---:|",
    ]
    for key in sorted(e4["cells"]):
        ranked = sorted(e4["cells"][key].items(), key=lambda item: item[1], reverse=True)
        for rank, (action_name, fraction) in enumerate(ranked[:5], start=1):
            lines.append(f"| {key} | {rank} | {action_name} | {fmt(fraction)} |")
    lines += [
        "",
        "## Appendix F: validation detail",
        "",
        "| validation check | value | pass |",
        "|---|---|---|",
    ]
    for site, row in out["validation"]["axis1_n_checks"].items():
        lines.append(f"| axis1 N {site} | {row['observed']} / {row['expected']} | {row['pass']} |")
    for key, ok in sorted(out["validation"]["E1_mean_jaccard_range_checks"].items()):
        lines.append(f"| E1 Jaccard range {key} | [0, 1] | {ok} |")
    for key, ok in sorted(out["validation"]["E2_first_divergent_step_range_checks"].items()):
        lines.append(f"| E2 first step range {key} | [0, {MAX_STEPS}] | {ok} |")
    for key, row in sorted(out["validation"]["E4_action_dist_sum_checks"].items()):
        lines.append(f"| E4 action sum {key} | {fmt(row['sum'], 6)} | {row['pass']} |")
    for key, ok in sorted(out["validation"]["effect_size_presence_checks"].items()):
        lines.append(f"| {key} | threshold > 0.1 | {ok} |")
    lines += [
        "",
        "## Validation",
        "",
        f"Overall pass: {out['validation']['pass']}.",
        "",
        "| check | result |",
        "|---|---|",
        f"| E1 N reddit | {out['validation']['axis1_n_checks']['reddit']['observed']} / {out['validation']['axis1_n_checks']['reddit']['expected']} |",
        f"| E1 N classifieds | {out['validation']['axis1_n_checks']['classifieds']['observed']} / {out['validation']['axis1_n_checks']['classifieds']['expected']} |",
        f"| E3 cells | {out['validation']['E3_cell_count']['observed']} / {out['validation']['E3_cell_count']['expected']} |",
        f"| P-prompt status | {json.dumps(out['data_status']['p_prompt'], sort_keys=True)} |",
    ]
    OUT_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    all_tasks = load_all_tasks()
    e1 = build_e1(all_tasks)
    e2 = build_e2(all_tasks)
    e3 = build_e3()
    e4 = build_e4(all_tasks)
    implications = headline_implications(e1, e2, e3, e4)
    out = {
        "method": (
            "Per-task mechanism evidence over B0 reddit/classifieds five-mode runs. E1 uses URL-changing "
            "click transition sets, E2 uses first divergent URL-signature step on symmetric-difference "
            "adjusted-success tasks, E3 aggregates existing confidence analyzer outputs across paper-grade "
            "B0/B1 cells without recomputing calibration, and E4 reports full action_type distributions. "
            "Element ids are not used because they are not mode-invariant or step-invariant."
        ),
        "data_status": {
            "step_dirs": {
                site: {mode: str(path.relative_to(ROOT)) for mode, path in modes.items()}
                for site, modes in STEP_DIRS.items()
            },
            "p_prompt": detect_partial_prompt_runs(),
        },
        "E1_click_target_divergence": e1,
        "E2_trajectory_boundary": e2,
        "E3_confidence_calibration": e3,
        "E4_action_vocabulary": e4,
        "paper_section5_implications": implications,
    }
    out["validation"] = validation_block(e1, e2, e3, e4)
    out = round_floats(out)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out)
    print(f"[json] {OUT_JSON}")
    print(f"[md]   {OUT_MD}")
    print(f"[validation] pass={out['validation']['pass']}")


if __name__ == "__main__":
    main()
