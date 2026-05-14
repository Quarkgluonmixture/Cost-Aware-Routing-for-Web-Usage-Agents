#!/usr/bin/env python3
"""[Micro 2a-2e] Micro dimension — per-step decision quality.

Outputs:
- docs/analysis/cross_sites/axis1_microbehavior.json  (machine-readable)
- docs/analysis/cross_sites/axis1_microbehavior_report.md  (paper-ready)

Micro 2a URL signature, 2b target-hit, 2c keyword reuse, 2d first-action,
and 2e cross-site validity.

See docs/checkpoints/paper_planning.md §3 Micro dimension framework.

Axis-1 micro-behavior decomposition for reddit and classifieds.

This script tests whether the text-payload axis changes per-step decision
quality more than macro action frequencies. Element ids are intentionally not
compared across modes; the mode-invariant anchors are URL trajectories, target
page hits, typed keywords, and first action transitions.

Outputs:
- docs/analysis/cross_sites/axis1_microbehavior.json
- docs/analysis/cross_sites/axis1_microbehavior_report.md
"""
from __future__ import annotations

from collections import Counter
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/visualwebarena/phase1"
OUT_JSON = ROOT / "docs/analysis/cross_sites/axis1_microbehavior.json"
OUT_MD = ROOT / "docs/analysis/cross_sites/axis1_microbehavior_report.md"
MACRO_JSON = ROOT / "docs/analysis/cross_sites/axis_effect_size.json"

TASK_CONFIGS = {
    "reddit": ROOT / "external/visualwebarena/config_files/vwa/test_reddit.json",
    "classifieds": ROOT / "external/visualwebarena/config_files/vwa/test_classifieds.json",
}

def _phantom_prompt_dir(baseline: str, site: str) -> Path | None:
    candidates = sorted(RESULTS.glob(f"{baseline}_phantom_prompt_{site}_*/phase1_phantom_prompt_router_0/episodes"))
    return candidates[-1] if candidates else None


STEP_DIRS: dict[str, dict[str, dict[str, Path]]] = {
    "B0": {
        "reddit": {
            "DOM": RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
            "Vision": RESULTS / "B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
            "SoM": RESULTS / "B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
            "Phantom-SoM": RESULTS / "B0_phantom_som_reddit_20260428/phase1_phantom_som_router_0/episodes",
            "P-text": RESULTS / "B0_phantom_text_reddit_20260427/phase1_phantom_dom_router_0/episodes",
            "Phantom-prompt": _phantom_prompt_dir("B0", "reddit"),
        },
        "classifieds": {
            "DOM": RESULTS / "B0_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
            "Vision": RESULTS / "B0_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
            "SoM": RESULTS / "B0_3mode_classifieds_20260413/phase1_som_router_0/episodes",
            "Phantom-SoM": RESULTS / "B0_phantom_som_classifieds_20260426/phase1_phantom_som_router_0/episodes",
            "P-text": RESULTS / "B0_phantom_text_classifieds_20260427/phase1_phantom_dom_router_0/episodes",
            "Phantom-prompt": _phantom_prompt_dir("B0", "classifieds"),
        },
    },
    "B1": {
        "reddit": {
            "DOM": RESULTS / "B1_3mode_reddit_20260413/phase1_dom_router_0/episodes",
            "Vision": RESULTS / "B1_3mode_reddit_20260413/phase1_vision_router_0/episodes",
            "SoM": RESULTS / "B1_3mode_reddit_20260413/phase1_som_router_0/episodes",
            "Phantom-prompt": _phantom_prompt_dir("B1", "reddit"),
        },
        "classifieds": {
            "DOM": RESULTS / "B1_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
            "Vision": RESULTS / "B1_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
            "SoM": RESULTS / "B1_3mode_classifieds_20260413/phase1_som_router_0/episodes",
            "Phantom-SoM": RESULTS / "B1_phantom_som_classifieds_20260428/phase1_phantom_som_router_0/episodes",
            "Phantom-prompt": _phantom_prompt_dir("B1", "classifieds"),
        },
    },
}
BASELINES = ["B0", "B1"]
SITES_LIST = ["reddit", "classifieds"]

MODE_LABELS = {
    "DOM": "DOM",
    "P-text": "P-text",
    "Phantom-SoM": "P-SoM",
    "Phantom-prompt": "P-prompt",
    "SoM": "SoM",
    "Vision": "Vision",
}

AXIS_CONTRASTS = {
    # Tier 2 mechanism cascade — each axis isolated via P-text intermediate
    "axis_1_text": ("DOM", "P-text"),
    "axis_2_prompt": ("P-text", "Phantom-SoM"),
    "axis_3_image": ("Phantom-SoM", "SoM"),
    # Tier 1 hook — compound DOM->P-SoM (text+prompt simultaneously swapped, image still no)
    # Validates that P-SoM's per-step decisions diverge from DOM at the micro level even
    # when aggregate macro action frequencies converge (especially relevant for cls).
    "compound_dom_to_psom": ("DOM", "Phantom-SoM"),
    # Diamond ablation alt-paths via P-prompt
    # axis_2_prompt_alt: prompt-only effect on AXTree text (DOM->P-prompt)
    "axis_2_prompt_alt": ("DOM", "Phantom-prompt"),
    # axis_1_text_alt: text-only effect on SoM prompt (P-prompt->P-SoM)
    "axis_1_text_alt": ("Phantom-prompt", "Phantom-SoM"),
    # Endpoint sanity — full mode swap for reference
    "endpoint_dom_to_som": ("DOM", "SoM"),
}

# §139.8: scored-set sizes (total − N/A excluded at load) from the single
# source of truth, not pre-exclusion 234/210.
from p79.experiment.analysis import scored_task_count as _scored_task_count
EXPECTED_N = {_s: _scored_task_count(_s, "visualwebarena") for _s in ("reddit", "classifieds")}
REPORT_CASE_REDDIT = [23, 30, 4]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_steps(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def task_id_from_path(path: Path) -> int:
    match = re.search(r"task_(\d+)_steps", path.name)
    if not match:
        raise ValueError(f"Cannot parse task id from {path}")
    return int(match.group(1))


def action_type(step: dict[str, Any]) -> str | None:
    action = step.get("action")
    nested = action.get("action_type") if isinstance(action, dict) else None
    return step.get("action_type") or nested


def action_text(step: dict[str, Any]) -> str | None:
    action = step.get("action")
    if not isinstance(action, dict):
        return None
    text = action.get("text")
    if text is None:
        return None
    normalized = str(text).strip().lower()
    return normalized or None


def url_path(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    path = parsed.path or "/"
    return path.rstrip("/") or "/"


def url_path_query(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    path = (parsed.path or "/").rstrip("/") or "/"
    if parsed.query:
        return f"{path}?{parsed.query}"
    return path


def target_match_parts(url: str) -> tuple[str, str]:
    clean = url.split("#", 1)[0].rstrip("/")
    parsed = urlparse(clean)
    path_query = parsed.path or "/"
    if parsed.query:
        path_query += "?" + parsed.query
    return clean, path_query


def first_non_empty_url(value: Any) -> str | None:
    if isinstance(value, dict):
        url = value.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
        for child in value.values():
            found = first_non_empty_url(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = first_non_empty_url(child)
            if found:
                return found
    return None


def load_task_configs(site: str) -> dict[int, dict[str, Any]]:
    configs = {}
    for row in read_json(TASK_CONFIGS[site]):
        task_id = int(row["task_id"])
        eval_block = row.get("eval") or {}
        target = eval_block.get("reference_url")
        if not target:
            target = first_non_empty_url(eval_block.get("program_html") or [])
        configs[task_id] = {
            "intent": row.get("intent", ""),
            "target_url": target.strip() if isinstance(target, str) and target.strip() else None,
        }
    return configs


def target_url_visited(urls: set[str], target_url: str | None) -> bool | None:
    if not target_url:
        return None
    clean, path_query = target_match_parts(target_url)
    for url in urls:
        candidate = url.split("#", 1)[0].rstrip("/")
        if candidate.startswith(clean) or clean in candidate:
            return True
        if path_query and path_query in candidate:
            return True
    return False


def consecutive_unique_routes(steps: list[dict[str, Any]], *, include_query: bool) -> list[str]:
    trajectory: list[str] = []
    for step in steps:
        route = url_path_query(step.get("obs_url")) if include_query else url_path(step.get("obs_url"))
        if not route:
            continue
        if not trajectory or trajectory[-1] != route:
            trajectory.append(route)
    return trajectory


def per_task_mode_metrics(baseline: str, site: str, mode: str, task_configs: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    ep_dir = STEP_DIRS.get(baseline, {}).get(site, {}).get(mode)
    out: dict[int, dict[str, Any]] = {}
    if ep_dir is None or not ep_dir.exists():
        return out
    for path in sorted(ep_dir.glob(f"{site}_task_*_steps_v2.jsonl")):
        task_id = task_id_from_path(path)
        steps = read_steps(path)
        urls = {str(step.get("obs_url") or "") for step in steps if step.get("obs_url")}
        paths = {url_path(url) for url in urls if url}
        keywords = [text for step in steps if action_type(step) == "type" for text in [action_text(step)] if text]
        keyword_counts = Counter(keywords)
        first_type = action_type(steps[0]) if steps else None
        first_target_path = url_path(steps[1].get("obs_url")) if len(steps) > 1 else None
        last_type = action_type(steps[-1]) if steps else None
        target = task_configs.get(task_id, {}).get("target_url")
        out[task_id] = {
            "task_id": task_id,
            "url_set": urls,
            "url_path_set": paths,
            "url_path_trajectory": consecutive_unique_routes(steps, include_query=False),
            "url_route_trajectory": consecutive_unique_routes(steps, include_query=True),
            "n_url_visits": len(urls),
            "target_url_visited": target_url_visited(urls, target),
            "search_keywords": keywords,
            "n_type_actions": len(keywords),
            "max_keyword_repeat": max(keyword_counts.values()) if keyword_counts else 0,
            "distinct_keywords": len(keyword_counts),
            "first_action_type": first_type,
            "first_action_target_url_path": first_target_path,
            "n_steps": len(steps),
            "reached_finish": last_type == "finish",
            "reward": steps[-1].get("reward") if steps else None,
        }
    return out


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def pct(value: float | None) -> float | None:
    return None if value is None else 100.0 * value


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def paired_binary_p(discordant_left: int, discordant_right: int) -> float:
    """Two-sided exact McNemar/binomial p-value for discordant pairs."""
    n = discordant_left + discordant_right
    if n == 0:
        return 1.0
    k = min(discordant_left, discordant_right)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def summarize_mode(tasks: dict[int, dict[str, Any]]) -> dict[str, Any]:
    rows = list(tasks.values())
    target_rows = [row for row in rows if row["target_url_visited"] is not None]
    return {
        "n_tasks": len(rows),
        "mean_url_set_size": safe_mean([row["n_url_visits"] for row in rows]),
        "mean_url_path_set_size": safe_mean([len(row["url_path_set"]) for row in rows]),
        "mean_n_steps": safe_mean([row["n_steps"] for row in rows]),
        "finish_rate": safe_mean([float(row["reached_finish"]) for row in rows]),
        "target_hit_n": len(target_rows),
        "target_hit_rate": safe_mean([float(row["target_url_visited"]) for row in target_rows]),
        "mean_n_type_actions": safe_mean([row["n_type_actions"] for row in rows]),
        "mean_max_keyword_repeat": safe_mean([row["max_keyword_repeat"] for row in rows]),
        "mean_distinct_keywords": safe_mean([row["distinct_keywords"] for row in rows]),
        "first_action_type_counts": dict(sorted(Counter(row["first_action_type"] or "none" for row in rows).items())),
    }


def contrast_metrics(left: dict[int, dict[str, Any]], right: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Compute right-minus-left paired contrasts except Jaccard/divergence."""
    common = sorted(set(left) & set(right))
    jaccards = [jaccard(left[task_id]["url_path_set"], right[task_id]["url_path_set"]) for task_id in common]
    repeat_diffs = [right[task_id]["max_keyword_repeat"] - left[task_id]["max_keyword_repeat"] for task_id in common]
    distinct_diffs = [right[task_id]["distinct_keywords"] - left[task_id]["distinct_keywords"] for task_id in common]
    type_diffs = [right[task_id]["n_type_actions"] - left[task_id]["n_type_actions"] for task_id in common]
    step_diffs = [right[task_id]["n_steps"] - left[task_id]["n_steps"] for task_id in common]
    finish_diffs = [float(right[task_id]["reached_finish"]) - float(left[task_id]["reached_finish"]) for task_id in common]
    first_action_divergence = [
        float(left[task_id]["first_action_type"] != right[task_id]["first_action_type"]) for task_id in common
    ]
    first_target_path_divergence = [
        float(left[task_id]["first_action_target_url_path"] != right[task_id]["first_action_target_url_path"])
        for task_id in common
    ]

    target_task_ids = [
        task_id
        for task_id in common
        if left[task_id]["target_url_visited"] is not None and right[task_id]["target_url_visited"] is not None
    ]
    left_hits = [float(left[task_id]["target_url_visited"]) for task_id in target_task_ids]
    right_hits = [float(right[task_id]["target_url_visited"]) for task_id in target_task_ids]
    target_diffs = [right_value - left_value for left_value, right_value in zip(left_hits, right_hits)]
    left_only = sum(
        1
        for task_id in target_task_ids
        if left[task_id]["target_url_visited"] and not right[task_id]["target_url_visited"]
    )
    right_only = sum(
        1
        for task_id in target_task_ids
        if right[task_id]["target_url_visited"] and not left[task_id]["target_url_visited"]
    )

    return {
        "n": len(common),
        "url_jaccard_mean": safe_mean(jaccards),
        "url_decision_divergence": None if not jaccards else 1.0 - mean(jaccards),
        "url_jaccard_min": min(jaccards) if jaccards else None,
        "url_jaccard_max": max(jaccards) if jaccards else None,
        "target_hit_n": len(target_task_ids),
        "target_hit_rate_left": safe_mean(left_hits),
        "target_hit_rate_right": safe_mean(right_hits),
        "target_hit_rate_diff": safe_mean(target_diffs),
        "target_hit_rate_diff_pct_pts": pct(safe_mean(target_diffs)),
        "target_hit_mcnemar": {
            "left_only": left_only,
            "right_only": right_only,
            "p_exact": paired_binary_p(left_only, right_only),
        },
        "max_keyword_repeat_diff": safe_mean(repeat_diffs),
        "distinct_keywords_diff": safe_mean(distinct_diffs),
        "n_type_actions_diff": safe_mean(type_diffs),
        "n_steps_diff": safe_mean(step_diffs),
        "finish_rate_diff": safe_mean(finish_diffs),
        "first_action_divergence_rate": safe_mean(first_action_divergence),
        "first_action_target_url_path_divergence_rate": safe_mean(first_target_path_divergence),
    }


def macro_axis1_effects(baseline: str, site: str) -> dict[str, float]:
    if not MACRO_JSON.exists():
        return {}
    data = read_json(MACRO_JSON)
    # New schema: results[baseline][site][metric]; fall back to legacy results[site][metric] if needed.
    results = data.get("results", {})
    if baseline in results and site in results[baseline]:
        site_data = results[baseline][site]
    else:
        site_data = results.get(site, {}) if baseline == "B0" else {}
    effects: dict[str, float] = {}
    for metric, block in site_data.items():
        text = block.get("text", {})
        if not text or text.get("n", 0) == 0:
            continue
        key = "cohen_h" if "cohen_h" in text else "cohen_d_z"
        value = text.get(key)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            effects[metric] = float(value)
    return effects


def decision_axis1_effect(axis1: dict[str, Any]) -> dict[str, float]:
    values = {
        "url_decision_divergence": float(axis1.get("url_decision_divergence") or 0.0),
        "target_hit_abs_diff": abs(float(axis1.get("target_hit_rate_diff") or 0.0)),
        "first_action_divergence_rate": float(axis1.get("first_action_divergence_rate") or 0.0),
    }
    values["mean_abs_decision_effect"] = mean(values.values())
    return values


def trajectory_summary(row: dict[str, Any]) -> dict[str, Any]:
    keywords = row["search_keywords"]
    return {
        "n_steps": row["n_steps"],
        "n_url_visits": row["n_url_visits"],
        "target_url_visited": row["target_url_visited"],
        "reward": row["reward"],
        "first_action_type": row["first_action_type"],
        "first_action_target_url_path": row["first_action_target_url_path"],
        "search_keywords": keywords[:8],
        "max_keyword_repeat": row["max_keyword_repeat"],
        "distinct_keywords": row["distinct_keywords"],
        "url_path_trajectory": row["url_path_trajectory"][:12],
        "url_route_trajectory": row["url_route_trajectory"][:12],
    }


def select_classified_case_tasks(metrics: dict[str, dict[str, dict[int, dict[str, Any]]]]) -> list[int]:
    left = metrics["classifieds"].get("DOM", {})
    right = metrics["classifieds"].get("P-text", {})
    if not left or not right:
        return []
    scored = []
    for task_id in sorted(set(left) & set(right)):
        left_hit = left[task_id]["target_url_visited"]
        right_hit = right[task_id]["target_url_visited"]
        hit_gap = 1 if left_hit != right_hit else 0
        jac = jaccard(left[task_id]["url_path_set"], right[task_id]["url_path_set"])
        first_gap = 1 if left[task_id]["first_action_type"] != right[task_id]["first_action_type"] else 0
        scored.append((hit_gap, 1.0 - jac, first_gap, task_id))
    scored.sort(reverse=True)
    return [item[-1] for item in scored[:2]]


def build_case_studies(
    metrics: dict[str, dict[str, dict[int, dict[str, Any]]]],
    task_configs: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    requested = {"reddit": REPORT_CASE_REDDIT, "classifieds": select_classified_case_tasks(metrics)}
    for site, task_ids in requested.items():
        site_modes = metrics.get(site, {})
        if not site_modes.get("DOM") or not site_modes.get("P-text"):
            continue
        for task_id in task_ids:
            dom = site_modes["DOM"].get(task_id)
            pdom = site_modes["P-text"].get(task_id)
            if not dom or not pdom:
                continue
            cases.append(
                {
                    "site": site,
                    "task_id": task_id,
                    "intent": task_configs[site].get(task_id, {}).get("intent", ""),
                    "target_url": task_configs[site].get(task_id, {}).get("target_url"),
                    "url_path_jaccard": jaccard(dom["url_path_set"], pdom["url_path_set"]),
                    "DOM": trajectory_summary(dom),
                    "P-text": trajectory_summary(pdom),
                }
            )
    return cases[:5]


def round_floats(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: round_floats(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [round_floats(value) for value in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return obj
        return round(obj, 6)
    return obj


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_paths(paths: list[str]) -> str:
    if not paths:
        return "none"
    return ", ".join(paths)


def main() -> None:
    task_configs = {site: load_task_configs(site) for site in SITES_LIST}
    # metrics_by_baseline[baseline][site][mode] -> dict task_id -> per-task metric
    metrics_by_baseline: dict[str, dict[str, dict[str, dict[int, dict[str, Any]]]]] = {}
    for baseline in BASELINES:
        metrics_by_baseline[baseline] = {}
        for site in SITES_LIST:
            metrics_by_baseline[baseline][site] = {}
            for mode in ["DOM", "Vision", "SoM", "Phantom-SoM", "P-text", "Phantom-prompt"]:
                metrics_by_baseline[baseline][site][mode] = per_task_mode_metrics(
                    baseline, site, mode, task_configs[site]
                )

    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for baseline, sites in metrics_by_baseline.items():
        summary[baseline] = {}
        for site, site_modes in sites.items():
            summary[baseline][site] = {
                mode: summarize_mode(mode_data) for mode, mode_data in site_modes.items() if mode_data
            }

    axis_contrasts: dict[str, dict[str, dict[str, Any]]] = {}
    validation: dict[str, Any] = {
        "expected_axis1_n": EXPECTED_N,
        "axis1_n_checks": {},
        "target_extraction": {},
        "url_jaccard_range_checks": {},
    }
    for baseline in BASELINES:
        axis_contrasts[baseline] = {}
        for site in SITES_LIST:
            axis_contrasts[baseline][site] = {}
            site_metrics = metrics_by_baseline[baseline][site]
            if baseline == "B0":
                target_available = sum(1 for cfg in task_configs[site].values() if cfg.get("target_url"))
                validation["target_extraction"][site] = {
                    "task_config_n": len(task_configs[site]),
                    "target_url_extracted_n": target_available,
                }
            for axis_name, (left_mode, right_mode) in AXIS_CONTRASTS.items():
                left = site_metrics.get(left_mode, {})
                right = site_metrics.get(right_mode, {})
                if not left or not right:
                    # Skip contrasts that need missing modes (B1 cls P-text / B1 red phantom)
                    axis_contrasts[baseline][site][axis_name] = {
                        "n": 0,
                        "skipped": True,
                        "contrast": f"{MODE_LABELS[right_mode]} minus {MODE_LABELS[left_mode]}",
                        "left_mode": MODE_LABELS[left_mode],
                        "right_mode": MODE_LABELS[right_mode],
                    }
                    continue
                contrast = contrast_metrics(left, right)
                contrast["contrast"] = f"{MODE_LABELS[right_mode]} minus {MODE_LABELS[left_mode]}"
                contrast["left_mode"] = MODE_LABELS[left_mode]
                contrast["right_mode"] = MODE_LABELS[right_mode]
                axis_contrasts[baseline][site][axis_name] = contrast
                if axis_name == "axis_1_text":
                    validation["axis1_n_checks"][f"{baseline}/{site}"] = {
                        "observed": contrast["n"],
                        "expected": EXPECTED_N[site],
                        "pass": contrast["n"] == EXPECTED_N[site],
                    }
                value = contrast.get("url_jaccard_mean")
                validation["url_jaccard_range_checks"][f"{baseline}/{site}/{axis_name}"] = {
                    "value": value,
                    "pass": value is not None and 0.0 <= value <= 1.0,
                }

    ratio_block: dict[str, Any] = {
        "claim": "axis 1 decision-quality effect > axis 1 macro-action-freq effect",
    }
    site_ratios: dict[str, float | None] = {}
    for baseline in BASELINES:
        for site in SITES_LIST:
            axis1 = axis_contrasts[baseline][site].get("axis_1_text", {})
            if axis1.get("skipped") or axis1.get("n", 0) == 0:
                site_ratios[f"{baseline}/{site}"] = None
                continue
            decision_effects = decision_axis1_effect(axis1)
            macro_effects = macro_axis1_effects(baseline, site)
            macro_mean = mean(abs(value) for value in macro_effects.values()) if macro_effects else None
            decision_mean = decision_effects["mean_abs_decision_effect"]
            ratio = decision_mean / macro_mean if macro_mean and macro_mean > 0 else None
            site_ratios[f"{baseline}/{site}"] = ratio
            prefix = f"{baseline}_{site}"
            ratio_block[f"{prefix}_decision_effects"] = decision_effects
            ratio_block[f"{prefix}_macro_axis1_effects"] = macro_effects
            ratio_block[f"{prefix}_macro_mean_abs_effect"] = macro_mean
            ratio_block[f"{prefix}_ratio"] = ratio

    # Verdict computed on B0 only (B1 axis-1 contrasts unavailable until P-text data lands).
    reddit_ok = bool(site_ratios.get("B0/reddit") is not None and site_ratios["B0/reddit"] > 1.0)
    classifieds_ok = bool(site_ratios.get("B0/classifieds") is not None and site_ratios["B0/classifieds"] > 1.0)
    if reddit_ok and classifieds_ok:
        verdict = "generalizes"
        narrative = (
            "Both sites show a larger axis-1 shift in mode-invariant decision anchors than in macro action "
            "frequencies, so the claim generalizes beyond the reddit search-loop failure mode."
        )
    elif reddit_ok or classifieds_ok:
        verdict = "site-specific"
        narrative = (
            "Only one site clears the decision-over-macro ratio threshold, so the claim should be framed as "
            "site-specific rather than a cross-site mechanism."
        )
    else:
        verdict = "not supported"
        narrative = (
            "Neither site clears the decision-over-macro ratio threshold, so the paper should not claim that "
            "axis 1 primarily changes decision quality."
        )
    ratio_block["verdict"] = verdict
    ratio_block["narrative"] = narrative

    # Case studies use B0 metrics (need P-text intermediate, only B0 has full set).
    case_studies = build_case_studies(metrics_by_baseline["B0"], task_configs)
    out = {
        "method": (
            "Mode-invariant micro-behavior analysis over reddit and classifieds. Per-task/per-mode metrics "
            "extract URL sets, URL path sets, target-page hits from task-config URLs, typed keywords, first "
            "action type, first post-action URL path, step count, and finish status. Cascade contrasts are "
            "right-minus-left: P-text minus DOM (axis 1 text), P-SoM minus P-text (axis 2 prompt), and SoM "
            "minus P-SoM (axis 3 image). URL path Jaccard is symmetric; lower values indicate stronger "
            "decision divergence. The cross-site claim ratio compares bounded decision effects "
            "(1 - URL-path Jaccard, absolute target-hit diff, first-action divergence) with the mean absolute "
            "axis-1 macro effect from axis_effect_size.json."
        ),
        "metrics_per_task_per_mode": summary,
        "axis_contrasts": axis_contrasts,
        "case_studies": case_studies,
        "cross_site_validity": ratio_block,
        "validation": validation,
    }

    out = round_floats(out)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")

    lines: list[str] = []
    lines.append("## Headline finding")
    lines.append("")
    lines.append(
        f"Axis 1 decision-quality vs macro-frequency test (B0 only — B1 P-text pending): reddit ratio "
        f"{fmt(out['cross_site_validity'].get('B0_reddit_ratio'), 2)}, classifieds ratio "
        f"{fmt(out['cross_site_validity'].get('B0_classifieds_ratio'), 2)}; verdict: "
        f"**{out['cross_site_validity']['verdict']}**."
    )
    lines.append("")
    lines.append("## Per-(baseline, site) Axis 1 Table")
    lines.append("")
    lines.append("| baseline | site | N | URL-path Jaccard | URL divergence | target-hit diff | target N | keyword repeat diff | distinct keyword diff | first-action divergence | macro mean | ratio |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for baseline in BASELINES:
        for site in SITES_LIST:
            axis1 = out["axis_contrasts"].get(baseline, {}).get(site, {}).get("axis_1_text", {})
            if not axis1 or axis1.get("skipped") or axis1.get("n", 0) == 0:
                lines.append(f"| {baseline} | {site} | — | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            prefix = f"{baseline}_{site}"
            lines.append(
                f"| {baseline} | {site} | {axis1['n']} | {fmt(axis1['url_jaccard_mean'])} | "
                f"{fmt(axis1['url_decision_divergence'])} | {fmt(axis1['target_hit_rate_diff_pct_pts'], 2)} pp | "
                f"{axis1['target_hit_n']} | {fmt(axis1['max_keyword_repeat_diff'])} | "
                f"{fmt(axis1['distinct_keywords_diff'])} | {fmt(axis1['first_action_divergence_rate'])} | "
                f"{fmt(out['cross_site_validity'].get(f'{prefix}_macro_mean_abs_effect'))} | "
                f"{fmt(out['cross_site_validity'].get(f'{prefix}_ratio'), 2)} |"
            )
    lines.append("")
    lines.append(
        "All signed differences are cascade-direction right-minus-left, so axis 1 is P-text minus DOM. "
        "Classifieds search-keyword levels should be read by axis differential, because OSClass tasks normally use search pages."
    )

    lines.append("")
    lines.append("## Tier 1 Hook — Compound DOM ↔ P-SoM micro contrast")
    lines.append("")
    lines.append(
        "Direct test of the hook claim: even when aggregate macro action frequencies converge "
        "toward DOM (especially on classifieds, see Tier 1 macro), per-step decisions still "
        "diverge meaningfully. Lower URL-path Jaccard ⇒ more pages visited that DOM does not "
        "visit (or vice versa) ⇒ task-pool divergence at the decision-trace level."
    )
    lines.append("")
    lines.append("| baseline | site | N | URL-path Jaccard (compound) | URL divergence | target-hit diff | first-action divergence |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for baseline in BASELINES:
        for site in SITES_LIST:
            c = out["axis_contrasts"].get(baseline, {}).get(site, {}).get("compound_dom_to_psom", {})
            if not c or c.get("skipped") or c.get("n", 0) == 0:
                lines.append(f"| {baseline} | {site} | — | n/a | n/a | n/a | n/a |")
                continue
            lines.append(
                f"| {baseline} | {site} | {c.get('n', 'n/a')} | {fmt(c.get('url_jaccard_mean'))} | "
                f"{fmt(c.get('url_decision_divergence'))} | {fmt(c.get('target_hit_rate_diff_pct_pts'), 2)} pp | "
                f"{fmt(c.get('first_action_divergence_rate'))} |"
            )
    lines.append("")
    lines.append("## Cross-site Validity")
    lines.append("")
    lines.append(out["cross_site_validity"]["narrative"])
    lines.append("")
    b0_red_n = out["validation"]["axis1_n_checks"].get("B0/reddit", {}).get("observed", 0)
    b0_cls_n = out["validation"]["axis1_n_checks"].get("B0/classifieds", {}).get("observed", 0)
    lines.append("Validation checks: B0 axis-1 N is "
                 f"reddit {b0_red_n}/210 and "
                 f"classifieds {b0_cls_n}/234. "
                 f"Target URLs were extracted for reddit "
                 f"{out['validation']['target_extraction']['reddit']['target_url_extracted_n']}/210 and classifieds "
                 f"{out['validation']['target_extraction']['classifieds']['target_url_extracted_n']}/234 tasks. "
                 "B1 axis-1 (P-text minus DOM) cannot be computed yet because B1 P-text data is pending; "
                 "B1 compound (DOM ↔ P-SoM) is computed for cls only.")
    lines.append("")
    lines.append("## Case Studies")
    for case in out["case_studies"]:
        lines.append("")
        lines.append(f"### {case['site']} task_{case['task_id']}")
        lines.append("")
        lines.append(f"Intent: {case['intent']}")
        lines.append(f"Target: {case['target_url']}")
        lines.append(f"URL-path Jaccard: {fmt(case['url_path_jaccard'])}")
        for mode in ["DOM", "P-text"]:
            row = case[mode]
            lines.append(
                f"- {mode}: steps={row['n_steps']}, target_hit={row['target_url_visited']}, "
                f"reward={row['reward']}, first={row['first_action_type']} -> "
                f"{row['first_action_target_url_path']}, keywords={row['search_keywords']}, "
                f"trajectory={format_paths(row['url_route_trajectory'])}"
            )
    lines.append("")
    lines.append("## Paper Section 5 Implication")
    lines.append("")
    if verdict == "generalizes":
        lines.append(
            "The paper can state that axis 1 is first-order at the task-success level because it changes "
            "where the agent goes and which target pages it reaches, even when macro action frequencies barely move. "
            "The classifieds result is important because it separates this from a reddit-only search-loop explanation; "
            "axis 3 can still be emphasized as stronger for image-heavy OSClass listing inspection."
        )
    elif verdict == "site-specific":
        lines.append(
            "The paper should limit the axis-1 decision-quality claim to the site that clears the ratio test and "
            "avoid presenting reddit search-loop behavior as a general mechanism."
        )
    else:
        lines.append(
            "The paper should rewrite the axis-1 claim: the micro-behavior evidence does not show a larger "
            "decision-quality effect than macro-frequency effect on either site."
        )

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"[json] {OUT_JSON}")
    print(f"[md]   {OUT_MD}")
    print(f"[verdict] {verdict}")


if __name__ == "__main__":
    main()
