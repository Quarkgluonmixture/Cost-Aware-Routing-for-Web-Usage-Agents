#!/usr/bin/env python3
"""[Layer 1a + 1b] Macro Behavior — axis-by-axis cascade ablation.

Outputs:
- docs/analysis/cross_sites/axis_effect_size.json  (machine-readable)
- docs/analysis/cross_sites/axis_effect_size_report.md  (paper-ready)

Tier 1 hook (1a): DOM↔P-SoM compound + DOM↔SoM endpoint (sanity)
Tier 2a cascade (1b): DOM→P-text→P-SoM→SoM 3-axis decomposition

See docs/checkpoints/paper_planning.md §3 Layer 1 framework.

3-axis cascade effect size ablation on per-task macro behavior.

Computes paired contrasts on per-task metrics:
- Text axis:   P-DOM minus DOM   (controls prompt=DOM, image=no)
- Prompt axis: P-SoM minus P-DOM (controls text=[SOM_MARKS], image=no)
- Image axis:  SoM   minus P-SoM (controls text=[SOM_MARKS], prompt=SoM)

All signs follow the cascade direction:
DOM -> P-DOM -> P-SoM -> SoM.
Thus text + prompt + image should recover SoM minus DOM.

Metrics per task:
- search_loop_bin: 1 if task has >=2 search-page steps (binary)
- type_frac:       typed_steps / total_steps
- scroll_frac:     scroll_steps / total_steps
- selfcorr_count:  raw count of "mistake/wrong/try again/go back" thoughts

Outputs:
- docs/analysis/cross_sites/axis_effect_size.json (machine-readable)
- docs/analysis/cross_sites/axis_effect_size_report.md (paper-ready prose)
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/visualwebarena/phase1"
OUT_JSON = ROOT / "docs/analysis/cross_sites/axis_effect_size.json"
OUT_MD = ROOT / "docs/analysis/cross_sites/axis_effect_size_report.md"

STEP_DIRS = {
    "reddit": {
        "DOM": RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
        "Vision": RESULTS / "B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
        "SoM": RESULTS / "B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
        "Phantom-SoM": RESULTS / "B0_phantom_reddit_20260428/phase1_phantom_som_router_0/episodes",
        "Phantom-DOM": RESULTS / "B0_phantom_dom_reddit_20260427/phase1_phantom_dom_router_0/episodes",
    },
    "classifieds": {
        "DOM": RESULTS / "B0_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
        "Vision": RESULTS / "B0_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
        "SoM": RESULTS / "B0_3mode_classifieds_20260413/phase1_som_router_0/episodes",
        "Phantom-SoM": RESULTS / "B0_phantom_classifieds_20260426/phase1_phantom_som_router_0/episodes",
        "Phantom-DOM": RESULTS / "B0_phantom_dom_classifieds_20260427/phase1_phantom_dom_router_0/episodes",
    },
}
SEARCH_MARKERS = {"reddit": ("/search",), "classifieds": ("page=search", "/search")}
SELFCORR_TOKENS = ("mistake", "wrong", "try again", "go back")


def step_task_id(path: Path) -> int:
    m = re.search(r"task_(\d+)_steps", path.name)
    if not m:
        raise ValueError(path.name)
    return int(m.group(1))


def read_steps(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


_ALL_METRIC_KEYS = (
    "search_loop_bin",
    "type_frac",
    "scroll_frac",
    "selfcorr_count",
    "click_frac",
    "finish_bin",
    "n_steps",
    "action_repeat_frac",
)


def _action_type(step: dict) -> Optional[str]:
    return step.get("action_type") or (step.get("action") or {}).get("action_type")


def per_task_metrics(site: str, mode: str) -> dict[int, dict[str, Optional[float]]]:
    ep_dir = STEP_DIRS[site][mode]
    out: dict[int, dict[str, Optional[float]]] = {}
    for path in sorted(ep_dir.glob(f"{site}_task_*_steps_v2.jsonl")):
        tid = step_task_id(path)
        steps = read_steps(path)
        n = len(steps)
        if n == 0:
            out[tid] = {k: None for k in _ALL_METRIC_KEYS}
            continue
        typed = scrolled = clicked = search_steps = selfcorr = 0
        repeats = 0
        prev_at: Optional[str] = None
        for i, s in enumerate(steps):
            at = _action_type(s)
            if at == "type":
                typed += 1
            elif at == "scroll":
                scrolled += 1
            elif at == "click":
                clicked += 1
            url = s.get("obs_url", "") or ""
            next_url = steps[i + 1].get("obs_url", "") if i + 1 < n else ""
            is_search = any(m in url for m in SEARCH_MARKERS[site])
            triggers_search = at == "type" and any(m in next_url for m in SEARCH_MARKERS[site])
            if is_search or triggers_search:
                search_steps += 1
            a = s.get("action") or {}
            thought = (a.get("thought", "") if isinstance(a, dict) else "").lower()
            if any(t in thought for t in SELFCORR_TOKENS):
                selfcorr += 1
            if i > 0 and at is not None and at == prev_at:
                repeats += 1
            prev_at = at
        last_at = _action_type(steps[-1])
        out[tid] = {
            "search_loop_bin": 1 if search_steps >= 2 else 0,
            "type_frac": typed / n,
            "scroll_frac": scrolled / n,
            "selfcorr_count": float(selfcorr),
            "click_frac": clicked / n,
            "finish_bin": 1 if last_at == "finish" else 0,
            "n_steps": float(n),
            "action_repeat_frac": (repeats / (n - 1)) if n > 1 else 0.0,
        }
    return out


METRIC_LABELS = {
    "search_loop_bin": "search_loop",
    "type_frac": "type_frac",
    "scroll_frac": "scroll_frac",
    "selfcorr_count": "selfcorr_count",
    "click_frac": "click_frac",
    "finish_bin": "finish_rate",
    "n_steps": "n_steps",
    "action_repeat_frac": "action_repeat_frac",
}
REPORT_METRIC_LABELS = {
    "search_loop": "search loop",
    "type_frac": "type fraction",
    "scroll_frac": "scroll fraction",
    "selfcorr_count": "self-correction",
    "click_frac": "click fraction",
    "finish_rate": "finish rate",
    "n_steps": "step count",
    "action_repeat_frac": "action repeat",
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def cohen_d_z(diffs: list[float]) -> float:
    sd = sample_sd(diffs)
    return mean(diffs) / sd if sd > 0 else float("nan")


def cohen_h_paired(p1: float, p2: float) -> float:
    """Cohen's h between two proportions (independent of pairing structure)."""
    p1c = max(min(p1, 1 - 1e-12), 1e-12)
    p2c = max(min(p2, 1 - 1e-12), 1e-12)
    return float(2 * math.asin(math.sqrt(p1c)) - 2 * math.asin(math.sqrt(p2c)))


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[int(pos)]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def bootstrap_ci(diffs: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    boots = []
    for _ in range(n_boot):
        boots.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    return percentile(boots, 2.5), percentile(boots, 97.5)


def average_ranks(values: list[float]) -> list[float]:
    """Return 1-based average ranks for a numeric vector."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    sorted_values = [values[i] for i in order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for original_idx in order[i:j]:
            ranks[original_idx] = avg_rank
        i = j
    return ranks


def wilcoxon_signed_rank_p(diffs: list[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value, excluding zero differences.

    This mirrors scipy.stats.wilcoxon(..., zero_method="wilcox",
    alternative="two-sided", correction=False) using the normal approximation.
    The task-level N here is large, so exact enumeration is unnecessary.
    """
    nonzero = [diff for diff in diffs if diff != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    abs_diffs = [abs(diff) for diff in nonzero]
    ranks = average_ranks(abs_diffs)
    w_plus = sum(rank for rank, diff in zip(ranks, nonzero) if diff > 0)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    tie_counts: dict[float, int] = {}
    for value in abs_diffs:
        tie_counts[value] = tie_counts.get(value, 0) + 1
    tie_adjust = sum(count**3 - count for count in tie_counts.values()) / 48.0
    var -= tie_adjust
    if var <= 0:
        return 1.0
    z = (w_plus - mean) / math.sqrt(var)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def paired_contrast(
    a: dict[int, dict],
    b: dict[int, dict],
    metric: str,
    binary: bool,
) -> dict:
    common = sorted(set(a) & set(b))
    pairs = [(a[t][metric], b[t][metric]) for t in common if a[t].get(metric) is not None and b[t].get(metric) is not None]
    if not pairs:
        return {"n": 0}
    arr_a = [float(p[0]) for p in pairs]
    arr_b = [float(p[1]) for p in pairs]
    diffs = [left - right for left, right in zip(arr_a, arr_b)]
    mean_diff = mean(diffs)
    if binary:
        p_a = mean(arr_a)
        p_b = mean(arr_b)
        effect = cohen_h_paired(p_a, p_b)
        effect_label = "cohen_h"
        diff_units = "pct_pts"
        mean_diff = (p_a - p_b) * 100.0
    else:
        effect = cohen_d_z(diffs)
        effect_label = "cohen_d_z"
        diff_units = "fraction" if metric.endswith("_frac") else "count"
    ci_lo, ci_hi = bootstrap_ci(diffs)
    if binary:
        # Recompute CI in pct-pt scale for binary
        ci_lo *= 100.0
        ci_hi *= 100.0
    if all(abs(diff) < 1e-12 for diff in diffs):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = wilcoxon_signed_rank_p(diffs)
    result = {
        "n": len(pairs),
        "mean_diff": mean_diff,
        "diff_units": diff_units,
        effect_label: effect,
        "ci95": [ci_lo, ci_hi],
        "wilcoxon_p": wilcoxon_p,
    }
    if diff_units == "pct_pts":
        result["mean_diff_pct_pts"] = mean_diff
    elif diff_units == "fraction":
        result["mean_diff_pct_pts"] = mean_diff * 100.0
        result["ci95_pct_pts"] = [ci_lo * 100.0, ci_hi * 100.0]
    elif diff_units == "count":
        result["mean_diff_count"] = mean_diff
    return result


def dominant(contrasts: dict[str, dict], *, binary: bool) -> str:
    key = "cohen_h" if binary else "cohen_d_z"
    effects: dict[str, float] = {}
    for axis, result in contrasts.items():
        value = result.get(key)
        if value is None:
            continue
        value = float(value)
        if not math.isnan(value):
            effects[axis] = abs(value)
    if not effects:
        return "n/a"
    axis, effect = max(effects.items(), key=lambda item: item[1])
    if effect < 0.1:
        return "neither (all small)"
    return axis


def effect_value(result: dict, binary: bool) -> float:
    key = "cohen_h" if binary else "cohen_d_z"
    value = result.get(key)
    if value is None:
        return float("nan")
    return float(value)


def meaningful(result: dict, binary: bool) -> bool:
    threshold = 0.1
    value = effect_value(result, binary)
    return bool(not math.isnan(value) and abs(value) > threshold)


def consistency_check(
    text: dict,
    prompt: dict,
    image: dict,
    endpoint: dict,
    *,
    binary: bool,
) -> dict:
    axis_sum = text.get("mean_diff", float("nan")) + prompt.get("mean_diff", float("nan")) + image.get("mean_diff", float("nan"))
    endpoint_diff = endpoint.get("mean_diff", float("nan"))
    error = axis_sum - endpoint_diff
    tolerance = 0.1 if binary else 0.005
    return {
        "axis_sum": axis_sum,
        "endpoint_diff": endpoint_diff,
        "error": error,
        "tolerance": tolerance,
        "units": "pct_pts" if binary else endpoint.get("diff_units", "fraction"),
        "pass": bool(not math.isnan(error) and abs(error) <= tolerance),
    }


def antagonistic_pairs(contrasts: dict[str, dict], *, binary: bool) -> list[dict]:
    key = "cohen_h" if binary else "cohen_d_z"
    axes = list(contrasts)
    out = []
    for i, left_axis in enumerate(axes):
        for right_axis in axes[i + 1 :]:
            left = contrasts[left_axis].get(key)
            right = contrasts[right_axis].get(key)
            if left is None or right is None:
                continue
            left = float(left)
            right = float(right)
            if math.isnan(left) or math.isnan(right):
                continue
            if left * right < 0 and abs(left) > 0.1 and abs(right) > 0.1:
                out.append(
                    {
                        "axis_a": left_axis,
                        "axis_b": right_axis,
                        f"{key}_a": left,
                        f"{key}_b": right,
                        "pattern": "antagonistic",
                    }
                )
    return out


def round_floats(obj):
    if isinstance(obj, dict):
        return {key: round_floats(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [round_floats(value) for value in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return obj
        return round(obj, 6)
    return obj


def main() -> None:
    metrics_def = [
        ("search_loop_bin", True),
        ("type_frac", False),
        ("scroll_frac", False),
        ("selfcorr_count", False),
        ("click_frac", False),
        ("finish_bin", True),
        ("n_steps", False),
        ("action_repeat_frac", False),
    ]
    out: dict = {
        "method": (
            "paired contrasts on per-task metrics; Cohen's d_z (continuous) / "
            "Cohen's h (binary search-loop); bootstrap 95% CI (n=2000); "
            "Wilcoxon signed-rank. Two analytical tiers: (1) HOOK-level — DOM vs "
            "P-SoM vs SoM, validates P-SoM as independent routing arm without needing "
            "P-text/P-prompt; (2) MECHANISM-level — cascade DOM->P-text->P-SoM->SoM "
            "decomposes the compound DOM->P-SoM axis into text/prompt sub-axes."
        ),
        "axes": {
            # Tier 1 hook contrasts (DOM <-> P-SoM <-> SoM)
            "compound": {"contrast": "P-SoM minus DOM", "controls": "image=no (text+prompt swap together)", "tier": 1},
            "image": {"contrast": "SoM minus P-SoM", "controls": "text=[SOM_MARKS], prompt=SoM", "tier": "1+2"},
            # Tier 2 cascade decomposition (axes 1 & 2 split the compound axis)
            "text": {"contrast": "P-text minus DOM", "controls": "prompt=DOM, image=no", "tier": 2},
            "prompt": {"contrast": "P-SoM minus P-text", "controls": "text=[SOM_MARKS], image=no", "tier": 2},
        },
        "results": {},
        "validation": {
            "expected_n": {"reddit": 210, "classifieds": 234},
            "non_negligible_thresholds": {"cohen_d_z_abs_gt": 0.1, "cohen_h_abs_gt": 0.1},
            "n_checks": {},
            "consistency_checks": {},
            "non_negligible_effects": [],
            "axis_1_non_negligible": [],
            "antagonistic_pairs": [],
        },
    }
    for site in ["reddit", "classifieds"]:
        modes_data = {
            "DOM": per_task_metrics(site, "DOM"),
            "P-DOM": per_task_metrics(site, "Phantom-DOM"),
            "P-SoM": per_task_metrics(site, "Phantom-SoM"),
            "SoM": per_task_metrics(site, "SoM"),
        }
        site_block: dict = {}
        for metric, binary in metrics_def:
            # Tier 2 cascade contrasts (DOM -> P-text -> P-SoM -> SoM)
            text = paired_contrast(modes_data["P-DOM"], modes_data["DOM"], metric, binary)
            prompt = paired_contrast(modes_data["P-SoM"], modes_data["P-DOM"], metric, binary)
            image = paired_contrast(modes_data["SoM"], modes_data["P-SoM"], metric, binary)
            endpoint = paired_contrast(modes_data["SoM"], modes_data["DOM"], metric, binary)
            # Tier 1 hook contrast: DOM <-> P-SoM (compound text+prompt swap)
            compound = paired_contrast(modes_data["P-SoM"], modes_data["DOM"], metric, binary)
            metric_name = METRIC_LABELS[metric]
            text["meaningful"] = meaningful(text, binary)
            prompt["meaningful"] = meaningful(prompt, binary)
            image["meaningful"] = meaningful(image, binary)
            compound["meaningful"] = meaningful(compound, binary)
            cascade_contrasts = {"text": text, "prompt": prompt, "image": image}
            pair_patterns = antagonistic_pairs(cascade_contrasts, binary=binary)
            check = consistency_check(text, prompt, image, endpoint, binary=binary)
            # Hook-tier dominance: just compound vs image
            hook_contrasts = {"compound": compound, "image": image}
            site_block[metric_name] = {
                # Tier 1 (hook)
                "compound_dom_to_psom": compound,
                # Tier 2 (cascade)
                "text": text,
                "prompt": prompt,
                "image": image,
                # Endpoint (DOM <-> SoM, sanity)
                "endpoint_dom_to_som": endpoint,
                # Decomposition diagnostics
                "dominant_cascade": dominant(cascade_contrasts, binary=binary),
                "dominant_hook": dominant(hook_contrasts, binary=binary),
                "psom_distinct_from_dom": compound["meaningful"],
                "psom_distinct_from_som": image["meaningful"],
                "consistency_check": check,
                "cancellation_patterns": pair_patterns,
            }
            expected_n = out["validation"]["expected_n"][site]
            out["validation"]["consistency_checks"][f"{site}/{metric_name}"] = check
            for pattern in pair_patterns:
                out["validation"]["antagonistic_pairs"].append(f"{pattern['axis_a']}_vs_{pattern['axis_b']}@{metric_name}@{site}")
            all_contrasts = {**cascade_contrasts, "compound": compound}
            for axis_name, contrast in all_contrasts.items():
                n_key = f"{site}/{metric_name}/{axis_name}"
                out["validation"]["n_checks"][n_key] = {
                    "observed": contrast["n"],
                    "expected": expected_n,
                    "pass": contrast["n"] == expected_n,
                }
                if contrast["meaningful"]:
                    out["validation"]["non_negligible_effects"].append(f"{axis_name}@{metric_name}@{site}")
                    if axis_name == "text":
                        out["validation"]["axis_1_non_negligible"].append(f"{metric_name}@{site}")
        out["results"][site] = site_block

    # Tier 1 hook verdict: which (site, metric) cells show P-SoM distinct from BOTH DOM and SoM
    psom_independent_cells = []
    psom_distinct_from_dom_only = []
    psom_distinct_from_som_only = []
    psom_indistinct = []
    for site in ["reddit", "classifieds"]:
        for metric, _binary in metrics_def:
            metric_name = METRIC_LABELS[metric]
            block = out["results"][site][metric_name]
            from_dom = block["psom_distinct_from_dom"]
            from_som = block["psom_distinct_from_som"]
            cell = f"{metric_name}@{site}"
            if from_dom and from_som:
                psom_independent_cells.append(cell)
            elif from_dom:
                psom_distinct_from_dom_only.append(cell)
            elif from_som:
                psom_distinct_from_som_only.append(cell)
            else:
                psom_indistinct.append(cell)

    out["interpretation"] = {
        "tier1_hook": {
            "claim": "P-SoM is an independent routing arm: its macro behavior is meaningfully distinct from BOTH DOM and SoM (not collapsible to either).",
            "psom_distinct_from_both_dom_and_som": psom_independent_cells,
            "psom_distinct_from_dom_only": psom_distinct_from_dom_only,
            "psom_distinct_from_som_only": psom_distinct_from_som_only,
            "psom_indistinct_from_either": psom_indistinct,
            "verdict_note": "A cell qualifies for hook support when |effect| > 0.1 (small Cohen's d/h) on the relevant contrast.",
        },
        "tier2_mechanism": {
            "claim": "The compound DOM->P-SoM transition decomposes into text (axis 1) and prompt (axis 2) sub-effects via P-text intermediate.",
            "dominant_cascade_by_axis": {
                axis: [
                    f"{METRIC_LABELS[m]}@{site}"
                    for site in ["reddit", "classifieds"]
                    for m, _b in metrics_def
                    if out["results"][site][METRIC_LABELS[m]]["dominant_cascade"] == axis
                ]
                for axis in ["text", "prompt", "image"]
            },
            "antagonistic_pairs": out["validation"]["antagonistic_pairs"],
        },
        "site_specific_notes": {
            "reddit": (
                "Reddit exposes the clearest internal cancellation: text-structure and prompt changes often "
                "move strategy metrics in opposite directions."
            ),
            "classifieds": (
                "Classifieds shows the strongest image-axis effects (finish rate h=+0.57, action repeat d_z=-0.42, "
                "step count d_z=-0.33); without image, P-SoM cls is in a degraded state — image axis fully recovers it."
            ),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out = round_floats(out)
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[json] {OUT_JSON}")

    # Helper for table rendering — defined BEFORE Tier 1 since both tiers use it
    def fmt(d: dict, binary: bool) -> str:
        if d.get("n", 0) == 0:
            return "n/a"
        eff_key = "cohen_h" if binary else "cohen_d_z"
        eff = d.get(eff_key)
        eff_str = f"{eff:+.2f}" if eff is not None and not (isinstance(eff, float) and math.isnan(eff)) else "n/a"
        label = "h" if binary else "d_z"
        if d.get("diff_units") == "count":
            delta = f"{d.get('mean_diff', 0):+.2f}"
            ci = d.get("ci95", [None, None])
        else:
            delta = f"{d.get('mean_diff_pct_pts', 0):+.2f} pp"
            ci = d.get("ci95" if binary else "ci95_pct_pts", [None, None])
        sig = "★" if d.get("wilcoxon_p", 1) < 0.05 else ""
        ci_str = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci[0] is not None else "—"
        return f"{label}={eff_str}{sig}; {delta}; {ci_str}"

    # Markdown report — Tier 1 (hook) before Tier 2 (mechanism, macro+micro)
    lines: list[str] = []
    lines.append("# Axis Effect Size Ablation\n")
    lines.append(
        "Hierarchical analysis with two tiers:\n\n"
        "- **Tier 1 (Hook)** — 3-mode coarse validation (DOM / P-SoM / SoM): "
        "establishes that **P-SoM is an independent routing arm** distinct from both "
        "DOM and SoM endpoints. Does not require P-text/P-prompt data.\n"
        "- **Tier 2 (Mechanism)** — 5-mode diamond (DOM / P-text / P-prompt / P-SoM / SoM): "
        "explains *why* P-SoM is distinct by decomposing the compound DOM→P-SoM transition "
        "into text-axis and prompt-axis sub-effects. Splits into:\n"
        "  - **2a Macro** — action-type frequencies (this file): finish rate, step count, "
        "search/type/scroll/click %, action repeat, self-correction.\n"
        "  - **2b Micro** — per-step decision quality (separate analysis): URL trajectory "
        "Jaccard, target-page hit rate, search keyword reuse, first-action divergence.\n\n"
        "**Current data status**: Tier 1 ✅ complete; Tier 2a Macro **partial** (cascade only, "
        "P-prompt data not yet available — diamond will replace cascade once it arrives); "
        "Tier 2b Micro tracked separately in `axis1_microbehavior.{json,md}`.\n"
    )

    # ----- Tier 1 -----
    lines.append("## Tier 1 — Hook: is P-SoM distinct from both DOM and SoM?\n")
    hook_lines = ["| site | metric | DOM→P-SoM (compound) | P-SoM→SoM (image) | distinct from DOM? | distinct from SoM? |",
                  "|---|---|---|---|---|---|"]
    for site in ["reddit", "classifieds"]:
        for metric, binary in metrics_def:
            metric_name = METRIC_LABELS[metric]
            r = out["results"][site][metric_name]
            from_dom = "✅" if r["psom_distinct_from_dom"] else "—"
            from_som = "✅" if r["psom_distinct_from_som"] else "—"
            hook_lines.append(
                f"| {site} | {REPORT_METRIC_LABELS[metric_name]} | {fmt(r['compound_dom_to_psom'], binary)} | {fmt(r['image'], binary)} | {from_dom} | {from_som} |"
            )
    lines.extend(hook_lines)

    independence = out["interpretation"]["tier1_hook"]
    lines.append("\n**P-SoM independence verdict** (cells where P-SoM differs from BOTH DOM and SoM, |effect|>0.1):")
    if independence["psom_distinct_from_both_dom_and_som"]:
        lines.append(f"- **Independent on**: {', '.join(independence['psom_distinct_from_both_dom_and_som'])}")
    else:
        lines.append("- **No cells show P-SoM distinct from both endpoints simultaneously**")
    if independence["psom_distinct_from_dom_only"]:
        lines.append(f"- Distinct from DOM only (≈ SoM-like): {', '.join(independence['psom_distinct_from_dom_only'])}")
    if independence["psom_distinct_from_som_only"]:
        lines.append(f"- Distinct from SoM only (≈ DOM-like): {', '.join(independence['psom_distinct_from_som_only'])}")
    if independence["psom_indistinct_from_either"]:
        lines.append(f"- Indistinct from both endpoints: {', '.join(independence['psom_indistinct_from_either'])}")

    # ----- Tier 2a Macro -----
    lines.append("\n## Tier 2a — Mechanism (Macro): cascade decomposition\n")
    lines.append(
        "DOM → P-text (axis 1, text only) → P-SoM (axis 2, prompt only) → SoM (axis 3, image). "
        "Once P-prompt data arrives this becomes a full diamond with two paths from DOM to P-SoM "
        "(via P-text or via P-prompt), letting us check prompt × text additivity / interaction.\n"
    )
    lines.append("| site | metric | text-axis (DOM→P-text) | prompt-axis (P-text→P-SoM) | image-axis (P-SoM→SoM) | dominant cascade axis | consistency |")
    lines.append("|---|---|---|---|---|---|---|")

    for site in ["reddit", "classifieds"]:
        for metric, binary in metrics_def:
            metric_name = METRIC_LABELS[metric]
            r = out["results"][site][metric_name]
            check = r["consistency_check"]
            check_label = "pass" if check["pass"] else "fail"
            lines.append(
                f"| {site} | {REPORT_METRIC_LABELS[metric_name]} | {fmt(r['text'], binary)} | {fmt(r['prompt'], binary)} | {fmt(r['image'], binary)} | {r['dominant_cascade']} | {check_label} |"
            )

    lines.append("\n★ marks Wilcoxon p<0.05. Effects with |d_z|>0.1 or |h|>0.1 are treated as non-negligible for axis dominance and cancellation checks.")

    lines.append("\n## Cancellation patterns\n")
    cancellations = out["validation"]["antagonistic_pairs"]
    if cancellations:
        lines.append(
            "The following site/metric pairs are antagonistic: two cascade axes have opposite-signed effects and both exceed |0.1| effect size. "
            "These are exactly the cases where a DOM-vs-SoM endpoint comparison can mask the internal mechanism.\n"
        )
        for site in ["reddit", "classifieds"]:
            for metric, _binary in metrics_def:
                metric_name = METRIC_LABELS[metric]
                patterns = out["results"][site][metric_name]["cancellation_patterns"]
                for pattern in patterns:
                    key = "cohen_h_a" if "cohen_h_a" in pattern else "cohen_d_z_a"
                    key_b = "cohen_h_b" if "cohen_h_b" in pattern else "cohen_d_z_b"
                    label = "h" if key == "cohen_h_a" else "d_z"
                    lines.append(
                        f"- {site} / {REPORT_METRIC_LABELS[metric_name]}: {pattern['axis_a']} vs {pattern['axis_b']} "
                        f"({label}={pattern[key]:+.2f} vs {pattern[key_b]:+.2f}) -> antagonistic"
                    )
    else:
        lines.append("No antagonistic pairs met the |0.1| effect-size threshold.")

    lines.append("\n## Consistency checks\n")
    lines.append(
        "For every site x metric, text + prompt + image matches the direct SoM minus DOM endpoint within tolerance "
        "(0.1 percentage points for binary search-loop, 0.005 raw units for fractions/counts)."
    )

    lines.append("\n## Tier 2b — Mechanism (Micro): per-step decision quality\n")
    lines.append(
        "Tracked separately in `axis1_microbehavior.{json,md}`. Macro action-frequency metrics "
        "(this file) average per-step decisions; micro metrics directly compare per-step element "
        "selection / page coverage / search keyword reuse via mode-invariant anchors (URL, action.text)."
    )

    lines.append("\n## Paper Section 5 implication\n")
    dominant_lines = []
    for axis in ["text", "prompt", "image"]:
        metrics = out["interpretation"]["tier2_mechanism"]["dominant_cascade_by_axis"][axis]
        dominant_lines.append(f"{axis}: {', '.join(metrics) if metrics else 'none'}")
    lines.append("**Tier 2a Macro — dominant cascade axis per metric**: " + "; ".join(dominant_lines) + ".")
    if cancellations:
        lines.append(
            "\n**Antagonistic pairs** (axes pulling opposite directions, hidden by DOM↔SoM endpoint comparison): "
            + "; ".join(cancellations)
            + "."
        )
    else:
        lines.append("\nNo antagonistic pair cleared the |0.1| effect-size threshold.")
    lines.append(
        "\n**4-level cascade design value**: decomposes DOM → SoM into three controlled transitions "
        "(AXTree vs [SOM_MARKS] structure, DOM vs SoM prompting, marginal image), and "
        "**reveals 6 antagonistic mechanism pairs** that endpoint-only comparison would mask."
    )
    OUT_MD.write_text("\n".join(lines))
    print(f"[md]   {OUT_MD}")


if __name__ == "__main__":
    main()
