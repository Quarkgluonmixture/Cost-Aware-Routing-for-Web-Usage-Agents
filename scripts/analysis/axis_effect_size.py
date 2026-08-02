#!/usr/bin/env python3
"""[Macro 1a + 1b] Macro dimension — axis-by-axis cascade ablation.

Outputs:
- docs/analysis/cross_sites/axis_effect_size.json  (machine-readable)
- docs/analysis/cross_sites/axis_effect_size_report.md  (paper-ready)

Tier 1 hook (1a): DOM↔P-SoM compound + DOM↔SoM endpoint (sanity)
Tier 2a cascade (1b): DOM→P-text→P-SoM→SoM 3-axis decomposition

See docs/checkpoints/paper_planning.md §3 Macro dimension framework.

3-axis cascade effect size ablation on per-task macro behavior.

Computes paired contrasts on per-task metrics:
- Text axis:   P-text minus DOM   (controls prompt=DOM, image=no)
- Prompt axis: P-SoM minus P-text (controls text=[SOM_MARKS], image=no)
- Image axis:  SoM   minus P-SoM (controls text=[SOM_MARKS], prompt=SoM)

All signs follow the cascade direction:
DOM -> P-text -> P-SoM -> SoM.
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
import os
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

from p79.experiment.analysis import paper_scored_task_count
from p79.experiment.io_utils import read_jsonl_dedup

ROOT = Path(__file__).resolve().parents[2]
# Run as `python3 scripts/analysis/axis_effect_size.py` and sys.path[0] is scripts/analysis/,
# so `import scripts.analysis.lib.run_registry` below raises ModuleNotFoundError. It used to be
# caught and downgraded to empty STEP_DIRS, which produced a full report with n=0 everywhere and
# exit 0 — see the §F audit note in _build_step_dirs().
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402

RESULTS = ROOT / "results/visualwebarena/phase1"
OUT_JSON = ROOT / "docs/analysis/cross_sites/axis_effect_size.json"
OUT_MD = ROOT / "docs/analysis/cross_sites/axis_effect_size_report.md"

# ---------------------------------------------------------------------------
# Registry-driven STEP_DIRS construction (A-fix: replaces hardcoded run_ids)
# ---------------------------------------------------------------------------
# Maps registry PAPER_MODES → axis-script mode key names used in downstream
# contrasts.  Axis-script uses "Phantom-SoM" (legacy key) for P-SoM and
# "Phantom-prompt" for P-prompt to keep downstream contrast lookups unchanged.
_REGISTRY_MODE_TO_AXIS_KEY: dict[str, str] = {
    "DOM": "DOM",
    "SoM": "SoM",
    "Vision": "Vision",
    "P-text": "P-text",
    "P-SoM": "Phantom-SoM",
    "P-prompt": "Phantom-prompt",
}

# Grade preference order: try paper-grade first, fall back to archived so the
# script can run pre-fire for validation.  Override via env AXIS_GRADE=archived.
_AXIS_GRADE = os.environ.get("AXIS_GRADE", "paper-grade")


def _build_step_dirs() -> dict[str, dict[str, dict[str, Path | None]]]:
    """Build STEP_DIRS from run_registry, replacing hardcoded archive run_ids.

    Returns nested dict: {baseline: {site: {axis_key: episodes_dir | None}}}.
    episodes_dir is None when a cell is absent from the registry at the
    requested grade (downstream per_task_metrics() already handles None/missing
    gracefully).  B2 cells present in registry but with no episode summaries on
    disk emit a stderr warning (not silent skip).
    """
    # DO NOT restore the old `except Exception: warn(); return {}` here. It ran for weeks:
    # every invocation of this script from the command line hit ModuleNotFoundError, returned
    # no directories, and wrote a complete-looking report in which every contrast had n=0 and
    # every negative finding ("no cells show P-SoM distinct from both endpoints", "for every
    # site x metric the axes sum to the endpoint") was vacuously true over an empty set. The
    # warning was emitted and lost in normal output; exit status was 0. An analysis that cannot
    # find its inputs must fail, not describe an empty world. (§F audit, 2026-08-02)
    from scripts.analysis.lib.run_registry import get_cells, BASELINES as REG_BASELINES, SITES as REG_SITES

    # Collect all grades that should be considered, in preference order.
    if _AXIS_GRADE == "paper-grade":
        grade_pref = ["paper-grade", "archived"]
    else:
        grade_pref = [_AXIS_GRADE]

    # Load cells across all relevant grades at once, then pick best per (b,s,m).
    all_cells: dict[tuple[str, str, str], object] = {}
    for grade in reversed(grade_pref):  # lower priority first so higher overwrites
        try:
            cells = get_cells(grade=grade)
        except Exception:
            cells = []
        for cell in cells:
            all_cells[(cell.baseline, cell.site, cell.mode)] = cell

    result: dict[str, dict[str, dict[str, Path | None]]] = {}
    for baseline in ["B0", "B1", "B2"]:
        result[baseline] = {}
        for site in ["reddit", "classifieds"]:
            site_dict: dict[str, Path | None] = {}
            for reg_mode, axis_key in _REGISTRY_MODE_TO_AXIS_KEY.items():
                cell = all_cells.get((baseline, site, reg_mode))
                if cell is None:
                    # Cell not in registry at any requested grade tier — silently absent.
                    site_dict[axis_key] = None
                    continue
                ep_dir = cell.episodes_dir
                if baseline == "B2" and not ep_dir.exists():
                    print(
                        f"[axis_effect_size] WARN: B2 cell {site}/{reg_mode} found in registry "
                        f"but episodes_dir missing on disk: {ep_dir}",
                        file=sys.stderr,
                    )
                site_dict[axis_key] = ep_dir
            result[baseline][site] = site_dict
    return result


STEP_DIRS: dict[str, dict[str, dict[str, Path | None]]] = _build_step_dirs()
BASELINES = ["B0", "B1", "B2"]
SITES = ["reddit", "classifieds"]
SEARCH_MARKERS = {"reddit": ("/search",), "classifieds": ("page=search", "/search"),
                  "wa_reddit": ("/search",)}

# --- WebArena as a seventh cell (--with-wa) -------------------------------------------------
# §G2 found this dimension had no WA cell while the other four step-reading products got one on
# 2026-08-02. The reason is instructive: this product looked ✅ complete in the coverage matrix
# while every contrast in it was n=0, so nobody asked whether it covered WA. A bug concealed a
# hole. Output is split to *_with_wa.* because appending a cell rewrites the /6 consistency
# denominators and is not a superset of the six-cell result.
WA_ROOT = ROOT / "results/webarena/phase1"
WA_GLOBS = {
    "DOM": "B1_dom_wa_reddit_2026*_R*", "SoM": "B1_som_wa_reddit_2026*_R*",
    "Vision": "B1_vision_wa_reddit_2026*_R*", "P-text": "B1_phantom_text_wa_reddit_2026*_R*",
    "P-prompt": "B1_phantom_prompt_wa_reddit_2026*_R*",
    "Phantom-SoM": "B1_phantom_som_wa_reddit_2026*_R*",
}
WA_UNIVERSE: Optional[set] = None      # set by attach_wa(); WA has no AMENDMENT_08 list


def attach_wa() -> int:
    """Add B1 x WA-reddit to STEP_DIRS and SITES. Raises rather than degrading silently."""
    global WA_UNIVERSE
    import glob as _glob
    modes: dict[str, Path] = {}
    for disp, pat in WA_GLOBS.items():
        hits = sorted(d for d in _glob.glob(str(WA_ROOT / pat)) if Path(d).is_dir())
        if not hits:
            raise SystemExit(f"attach_wa: no run dir for {disp} ({pat})")
        ep = next(Path(hits[-1]).glob("*/episodes"), None)
        if ep is None or not ep.is_dir():
            raise SystemExit(f"attach_wa: no episodes dir under {hits[-1]}")
        modes[disp] = ep
    uni = None
    for ep in modes.values():
        ids = {int(f.name.split("_task_")[1].split("_")[0])
               for f in ep.glob("reddit_task_*_summary_v2.json")}
        uni = ids if uni is None else (uni & ids)
    if not uni:
        raise SystemExit("attach_wa: empty task intersection across the six modes")
    WA_UNIVERSE = uni
    STEP_DIRS.setdefault("B1", {})["wa_reddit"] = {
        # STEP_DIRS keys are the axis-script mode names; Vision/P-text carry through unchanged
        k: modes[k] for k in modes}
    SITES.append("wa_reddit")
    return len(uni)


def _scored_for(site: str) -> set:
    """Canonical scored set. WA carries no AMENDMENT_08 exclusion, so its universe is the task
    set common to all six modes rather than a curated list."""
    if site == "wa_reddit":
        if WA_UNIVERSE is None:
            raise SystemExit("wa_reddit requested without attach_wa()")
        return WA_UNIVERSE
    return set(expected_scored_ids(site)[0])
SELFCORR_TOKENS = ("mistake", "wrong", "try again", "go back")


def step_task_id(path: Path) -> int:
    m = re.search(r"task_(\d+)_steps", path.name)
    if not m:
        raise ValueError(path.name)
    return int(m.group(1))


def _summary_path_for_steps(steps_path: Path) -> Path | None:
    """Derive sibling *_summary_v2.json path from a *_steps_v2.jsonl path."""
    name = steps_path.name  # e.g. classifieds_task_3_steps_v2.jsonl
    summary_name = name.replace("_steps_v2.jsonl", "_summary_v2.json")
    if summary_name == name:
        return None  # name didn't match expected pattern
    candidate = steps_path.parent / summary_name
    return candidate


def read_steps(path: Path) -> list[dict]:
    """Read step records with restart dedup via read_jsonl_dedup.

    Uses sibling *_summary_v2.json for identity validation (strict_identity=True)
    when available; falls back to dedup-only when summary is absent.
    This replaces the previous bare read_text loop which let watchdog restart
    tail segments pollute §4 metrics.
    """
    summary = _summary_path_for_steps(path)
    if summary is not None and summary.exists():
        return read_jsonl_dedup(path, summary_path=summary, strict_identity=True)
    return read_jsonl_dedup(path)


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


# (baseline, site, mode) -> task ids dropped because their step file does not belong to their
# summary. Two such episodes exist across all 7,686 (audit: steps_summary_identity_audit.json),
# both in B0_reddit/Phantom-SoM. They are inside the scored universe, so dropping them shrinks
# every contrast touching that arm from 203 to 201 pairs — reported, not swallowed.
IDENTITY_SKIPS: dict[tuple[str, str, str], list[int]] = {}


def per_task_metrics(baseline: str, site: str, mode: str) -> dict[int, dict[str, Optional[float]]]:
    ep_dir = STEP_DIRS.get(baseline, {}).get(site, {}).get(mode)
    out: dict[int, dict[str, Optional[float]]] = {}
    if ep_dir is None or not ep_dir.exists():
        return out
    # Restrict to the canonical scored universe, as every other cross-site product does. Without
    # this, reddit contributed 205 tasks against a scored set of 203 — the two AMENDMENT_08
    # exclusions (58, 160) were inside every effect size. On the P-SoM arm the count even read
    # as correct: two identity-dropped episodes cancelled the two extra ones, giving n=203 and a
    # passing check over the wrong 203 tasks. Compare sets, not counts. (§F audit, 2026-08-02)
    scored = _scored_for(site)
    prefix = "reddit" if site == "wa_reddit" else site      # WA files are reddit_task_*
    for path in sorted(ep_dir.glob(f"{prefix}_task_*_steps_v2.jsonl")):
        tid = step_task_id(path)
        if tid not in scored:
            continue
        try:
            steps = read_steps(path)
        except ValueError:
            # strict_identity refused the pair. Skipping one episode is right; letting it abort
            # the whole 36-cell analysis is not, and neither is dropping it silently.
            IDENTITY_SKIPS.setdefault((baseline, site, mode), []).append(tid)
            continue
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
        # For a binary metric only DISCORDANT pairs carry information; `n` is the pairing count
        # and overstates it badly (finish_rate on B2/classifieds: n=224, 26 discordant).
        # Wilcoxon here is fine — it has a tie correction and agrees with McNemar exact to
        # within a factor — but a reader sizing the claim needs the smaller number.
        "n_discordant": sum(1 for x in diffs if x != 0) if binary else None,
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


# Cohen's conventional "small" boundary. It is a convention, not a property of this data, and
# it is applied to two different quantities: d_z for continuous metrics and h for the binary
# one. They are not on the same scale (h is an arcsine-transformed proportion difference), so a
# single number cannot mean the same thing for both. Kept because changing it post hoc would be
# worse than an arbitrary-but-declared threshold, and because every verdict that depends on it
# is now also reported under multiplicity control, where the p-values do the discriminating.
# (§H stress P2-3, 2026-08-02.)
EFFECT_THRESHOLD = {"cohen_d_z": 0.1, "cohen_h": 0.1}


def meaningful(result: dict, binary: bool) -> bool:
    threshold = EFFECT_THRESHOLD["cohen_h" if binary else "cohen_d_z"]
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


def multiplicity_filtered_independence(out: dict, metrics_def: list) -> dict:
    """Re-run the Tier 1 independence verdict with multiplicity control on both legs.

    The headline count asks only |effect| > 0.1, over 48 (cell, metric) combinations and 96
    Wilcoxon tests. A count built that way is the first thing a reviewer attacks, so this
    reports what survives when each leg must also clear Benjamini-Hochberg (FDR) and Holm
    (FWER) across ALL legs at once. Effect size stays a requirement — significance alone on
    n=201..224 would let trivial differences in.
    """
    legs, keyed, offseg = [], [], {}
    for baseline in BASELINES:
        for site in SITES:
            for metric, _b in metrics_def:
                mn = METRIC_LABELS[metric]
                blk = out["results"][baseline].get(site, {}).get(mn)
                if not blk or not blk["compound_dom_to_psom"].get("n"):
                    continue
                c, i = blk["compound_dom_to_psom"], blk["image"]
                ec = c.get("cohen_h", c.get("cohen_d_z"))
                ei = i.get("cohen_h", i.get("cohen_d_z"))
                if ec is None or ei is None or c.get("wilcoxon_p") is None \
                        or i.get("wilcoxon_p") is None:
                    continue
                key = f"{mn}@{baseline}/{site}"
                keyed.append((key, len(legs), len(legs) + 1,
                              abs(ec) > 0.1 and abs(ei) > 0.1))
                # opposite-signed legs => P-SoM lies outside the DOM..SoM segment
                offseg[key] = (c.get("mean_diff", 0) > 0) != (i.get("mean_diff", 0) > 0)
                legs += [c["wilcoxon_p"], i["wilcoxon_p"]]

    def bh(p, q=0.05):
        order = sorted(range(len(p)), key=lambda i: p[i])
        cut = 0
        for rank, i in enumerate(order, 1):
            if p[i] <= rank / len(p) * q:
                cut = rank
        return set(order[:cut])

    def holm(p, a=0.05):
        order = sorted(range(len(p)), key=lambda i: p[i])
        keep = set()
        for rank, i in enumerate(order, 1):
            if p[i] <= a / (len(p) - rank + 1):
                keep.add(i)
            else:
                break
        return keep

    if not legs:
        return {"n_legs": 0}
    kb, kh = bh(legs), holm(legs)
    eff = [k for k, *_ , ok in keyed if ok]
    both_bh = [k for k, a, b, ok in keyed if ok and a in kb and b in kb]
    both_holm = [k for k, a, b, ok in keyed if ok and a in kh and b in kh]
    off = [k for k in both_bh if offseg.get(k)]
    return {
        "n_legs": len(legs), "n_combinations": len(keyed),
        "effect_only": len(eff),
        "effect_and_bh_fdr_0.05": len(both_bh), "cells_bh": both_bh,
        "effect_and_holm_fwer_0.05": len(both_holm), "cells_holm": both_holm,
        # Differing from both endpoints is not the same as being an independent direction: a
        # mode that INTERPOLATES between DOM and SoM also differs from both. P-SoM is off the
        # segment when the two legs have opposite signs, i.e. it is an extremum rather than a
        # midpoint. (§H stress P1-3 + gemini F2, 2026-08-02.)
        "bh_and_off_segment": len(off), "cells_bh_off_segment": off,
        "cells_bh_interpolating": [k for k in both_bh if not offseg.get(k)],
        "note": "both legs (compound and image) must clear the correction for the combination "
                "to count; corrections are applied across all legs jointly, not per cell. "
                "`off_segment` additionally requires the two legs to disagree in sign.",
    }


def diamond_additivity(block: dict, *, binary: bool) -> dict:
    """Do the two routes from DOM to P-SoM recover the direct contrast?

    Path A goes DOM --text--> P-text --prompt--> P-SoM.
    Path B goes DOM --prompt--> P-prompt --text--> P-SoM.

    READ THIS BEFORE READING THE NUMBERS. On `mean_diff` this is an ALGEBRAIC IDENTITY, not an
    empirical test: mean(P-text − DOM) + mean(P-SoM − P-text) = mean(P-SoM − DOM) whenever the
    three contrasts are averaged over the same tasks. So a zero residual is arithmetic and says
    nothing about text x prompt interaction, and a NON-zero residual means the legs were summed
    over DIFFERENT task sets — here, the P-SoM arm losing its two identity-mismatched episodes,
    which makes `prompt` and `compound` 201-task means while `text` is a 203-task mean.

    Which makes this a useful canary and a worthless finding. It is reported as a base-set
    consistency check. An interaction test would have to compare effect sizes or fit an
    interaction term, and this function does not do that. (§F audit, 2026-08-02)
    """
    def md(key: str) -> float:
        return block.get(key, {}).get("mean_diff", float("nan"))

    comp = md("compound_dom_to_psom")
    a = md("text") + md("prompt")
    b = md("axis_2_prompt_alt") + md("axis_1_text_alt")
    tol = 0.1 if binary else 0.005
    ns = {k: block.get(k, {}).get("n") for k in
          ("text", "prompt", "axis_2_prompt_alt", "axis_1_text_alt", "compound_dom_to_psom")}
    return {
        "path_a_sum": a, "path_b_sum": b, "compound": comp,
        "residual_a": a - comp, "residual_b": b - comp,
        "tolerance": tol,
        "identity_holds_a": bool(not math.isnan(a - comp) and abs(a - comp) <= tol),
        "identity_holds_b": bool(not math.isnan(b - comp) and abs(b - comp) <= tol),
        "is_algebraic_identity": True,
        # Legs touching the P-SoM arm lose the two identity-mismatched episodes, so the paths
        # are not summed over an identical task set. Residuals below are read against this.
        "n_per_leg": ns,
        "n_mismatched_legs": len({v for v in ns.values() if v is not None}) > 1,
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
    # A report where every contrast has n=0 reads as a set of negative findings; it is actually
    # a report about an empty input. Refuse to write one.
    _live = sum(1 for b in STEP_DIRS.values() for s in b.values()
                for d in s.values() if d is not None and d.exists())
    if _live == 0:
        raise SystemExit(
            "[axis_effect_size] no episode directories resolved from the run registry. "
            "Every contrast would be n=0 and every negative finding vacuous; refusing to write. "
            f"STEP_DIRS covers {len(STEP_DIRS)} baselines.")

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
            # Diamond ablation alt-paths via P-prompt (text=AXTree + prompt=SoM)
            # axis_2_prompt_alt: prompt-only effect on AXTree text (DOM->P-prompt)
            "axis_2_prompt_alt": {"contrast": "P-prompt minus DOM", "controls": "text=AXTree, image=no", "tier": 2},
            # axis_1_text_alt: text-only effect on SoM prompt (P-prompt->P-SoM)
            "axis_1_text_alt": {"contrast": "P-SoM minus P-prompt", "controls": "prompt=SoM, image=no", "tier": 2},
        },
        "results": {},
        "validation": {
            # §139.8: scored-set sizes (total − N/A excluded at load) from the
            # single source of truth, not pre-exclusion 234/210.
            "expected_n": {_s: (len(WA_UNIVERSE) if _s == "wa_reddit"
                                else paper_scored_task_count(_s, "visualwebarena", strict=True))
                           for _s in SITES},
            "non_negligible_thresholds": {"cohen_d_z_abs_gt": 0.1, "cohen_h_abs_gt": 0.1},
            "n_checks": {},
            "consistency_checks": {},
            "non_negligible_effects": [],
            "axis_1_non_negligible": [],
            "antagonistic_pairs": [],
            "identity_skipped_episodes": {},
        },
    }
    def empty_contrast() -> dict:
        return {"n": 0, "meaningful": False}

    for baseline in BASELINES:
        out["results"].setdefault(baseline, {})
        for site in SITES:
            modes_data = {
                "DOM": per_task_metrics(baseline, site, "DOM"),
                "P-text": per_task_metrics(baseline, site, "P-text"),
                "P-SoM": per_task_metrics(baseline, site, "Phantom-SoM"),
                "SoM": per_task_metrics(baseline, site, "SoM"),
                "P-prompt": per_task_metrics(baseline, site, "Phantom-prompt"),
            }
            available = {k: bool(v) for k, v in modes_data.items()}
            site_block: dict = {}
            for metric, binary in metrics_def:
                metric_name = METRIC_LABELS[metric]

                def maybe_contrast(left_mode: str, right_mode: str) -> dict:
                    if not (available[left_mode] and available[right_mode]):
                        return empty_contrast()
                    return paired_contrast(modes_data[left_mode], modes_data[right_mode], metric, binary)

                # Tier 2 cascade contrasts (DOM -> P-text -> P-SoM -> SoM)
                text = maybe_contrast("P-text", "DOM")
                prompt = maybe_contrast("P-SoM", "P-text")
                image = maybe_contrast("SoM", "P-SoM")
                endpoint = maybe_contrast("SoM", "DOM")
                # Tier 1 hook contrast: DOM <-> P-SoM (compound text+prompt swap)
                compound = maybe_contrast("P-SoM", "DOM")
                # Diamond alt-path contrasts via P-prompt
                # axis_2_prompt_alt = P-prompt minus DOM (prompt-only on AXTree)
                # axis_1_text_alt = P-SoM minus P-prompt (text-only on SoM prompt)
                axis_2_prompt_alt = maybe_contrast("P-prompt", "DOM")
                axis_1_text_alt = maybe_contrast("P-SoM", "P-prompt")
                text["meaningful"] = meaningful(text, binary) if text.get("n", 0) else False
                prompt["meaningful"] = meaningful(prompt, binary) if prompt.get("n", 0) else False
                image["meaningful"] = meaningful(image, binary) if image.get("n", 0) else False
                compound["meaningful"] = meaningful(compound, binary) if compound.get("n", 0) else False
                axis_2_prompt_alt["meaningful"] = meaningful(axis_2_prompt_alt, binary) if axis_2_prompt_alt.get("n", 0) else False
                axis_1_text_alt["meaningful"] = meaningful(axis_1_text_alt, binary) if axis_1_text_alt.get("n", 0) else False
                cascade_contrasts = {"text": text, "prompt": prompt, "image": image}
                # Antagonism / consistency only when all 3 cascade legs are present
                if text.get("n", 0) and prompt.get("n", 0) and image.get("n", 0):
                    pair_patterns = antagonistic_pairs(cascade_contrasts, binary=binary)
                else:
                    pair_patterns = []
                if all(c.get("n", 0) for c in (text, prompt, image, endpoint)):
                    check = consistency_check(text, prompt, image, endpoint, binary=binary)
                else:
                    check = {"axis_sum": float("nan"), "endpoint_diff": float("nan"),
                             "error": float("nan"), "tolerance": 0.0,
                             "units": "n/a", "pass": False, "skipped": True}
                # Hook-tier dominance: just compound vs image
                hook_contrasts = {"compound": compound, "image": image}
                site_block[metric_name] = {
                    # Tier 1 (hook)
                    "compound_dom_to_psom": compound,
                    # Tier 2 (cascade)
                    "text": text,
                    "prompt": prompt,
                    "image": image,
                    # Diamond alt-path (via P-prompt)
                    "axis_2_prompt_alt": axis_2_prompt_alt,
                    "axis_1_text_alt": axis_1_text_alt,
                    # Endpoint (DOM <-> SoM, sanity)
                    "endpoint_dom_to_som": endpoint,
                    # Decomposition diagnostics
                    "dominant_cascade": dominant(cascade_contrasts, binary=binary),
                    "dominant_hook": dominant(hook_contrasts, binary=binary),
                    "psom_distinct_from_dom": compound.get("meaningful", False),
                    "psom_distinct_from_som": image.get("meaningful", False),
                    "consistency_check": check,
                    "cancellation_patterns": pair_patterns,
                }
                site_block[metric_name]["diamond_additivity"] = diamond_additivity(
                    site_block[metric_name], binary=binary)
                expected_n = out["validation"]["expected_n"][site]
                out["validation"]["consistency_checks"][f"{baseline}/{site}/{metric_name}"] = check
                for pattern in pair_patterns:
                    out["validation"]["antagonistic_pairs"].append(
                        f"{pattern['axis_a']}_vs_{pattern['axis_b']}@{metric_name}@{baseline}/{site}"
                    )
                all_contrasts = {**cascade_contrasts, "compound": compound}
                for axis_name, contrast in all_contrasts.items():
                    n_key = f"{baseline}/{site}/{metric_name}/{axis_name}"
                    # An arm whose episodes were dropped for identity mismatch can never reach
                    # expected_n; the pair count is the intersection, so it loses those tasks on
                    # both legs. Allow exactly that shortfall and name it, rather than emitting a
                    # permanent pass=False nobody can act on.
                    dropped = len({t for (b_, s_, _m), ts in IDENTITY_SKIPS.items()
                                   if b_ == baseline and s_ == site for t in ts})
                    obs = contrast.get("n", 0)
                    out["validation"]["n_checks"][n_key] = {
                        "observed": obs,
                        "expected": expected_n,
                        "identity_dropped": dropped,
                        "pass": obs == expected_n or (dropped and obs == expected_n - dropped),
                    }
                    if contrast.get("meaningful", False):
                        out["validation"]["non_negligible_effects"].append(
                            f"{axis_name}@{metric_name}@{baseline}/{site}"
                        )
                        if axis_name == "text":
                            out["validation"]["axis_1_non_negligible"].append(f"{metric_name}@{baseline}/{site}")
            out["results"][baseline][site] = site_block

    # Tier 1 hook verdict: which (baseline, site, metric) cells show P-SoM distinct from BOTH DOM and SoM
    psom_independent_cells = []
    psom_distinct_from_dom_only = []
    psom_distinct_from_som_only = []
    psom_indistinct = []
    for baseline in BASELINES:
        for site in SITES:
            site_block = out["results"][baseline].get(site, {})
            for metric, _binary in metrics_def:
                metric_name = METRIC_LABELS[metric]
                block = site_block.get(metric_name)
                if not block:
                    continue
                # Skip cells where P-SoM is missing (no compound/image contrast)
                if block["compound_dom_to_psom"].get("n", 0) == 0:
                    continue
                from_dom = block["psom_distinct_from_dom"]
                from_som = block["psom_distinct_from_som"]
                cell = f"{metric_name}@{baseline}/{site}"
                if from_dom and from_som:
                    psom_independent_cells.append(cell)
                elif from_dom:
                    psom_distinct_from_dom_only.append(cell)
                elif from_som:
                    psom_distinct_from_som_only.append(cell)
                else:
                    psom_indistinct.append(cell)

    # The count above uses |effect| > 0.1 and no p at all, over 48 (cell, metric) combinations
    # and 96 individual contrasts. Reporting it bare invites the obvious attack, so the same
    # verdict is recomputed requiring BOTH legs to survive multiplicity control. (§G3, 08-02)
    survivors = multiplicity_filtered_independence(out, metrics_def)

    out["interpretation"] = {
        "tier1_hook": {
            "claim": "P-SoM is an independent routing arm: its macro behavior is meaningfully distinct from BOTH DOM and SoM (not collapsible to either).",
            "psom_distinct_from_both_dom_and_som": psom_independent_cells,
            "psom_distinct_from_dom_only": psom_distinct_from_dom_only,
            "psom_distinct_from_som_only": psom_distinct_from_som_only,
            "psom_indistinct_from_either": psom_indistinct,
            "verdict_note": "A cell qualifies for hook support when |effect| > 0.1 (small Cohen's d/h) on the relevant contrast.",
            "multiplicity": survivors,
        },
        "tier2_mechanism": {
            "claim": "The compound DOM->P-SoM transition decomposes into text (axis 1) and prompt (axis 2) sub-effects via P-text intermediate.",
            "dominant_cascade_by_axis": {
                axis: [
                    f"{METRIC_LABELS[m]}@{baseline}/{site}"
                    for baseline in BASELINES
                    for site in SITES
                    for m, _b in metrics_def
                    if out["results"].get(baseline, {}).get(site, {}).get(METRIC_LABELS[m], {}).get("dominant_cascade") == axis
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
    out["validation"]["identity_skipped_episodes"] = {
        "/".join(k): sorted(v) for k, v in sorted(IDENTITY_SKIPS.items())}
    out = round_floats(out)
    _oj = (OUT_JSON.with_name(OUT_JSON.stem + "_with_wa" + OUT_JSON.suffix)
           if WA_UNIVERSE is not None else OUT_JSON)
    _oj.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[json] {_oj}")

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
        "**Current data status**: computed from the run registry at grade "
        f"`{_AXIS_GRADE}`; Tier 2b Micro is tracked separately in "
        "`axis1_microbehavior.{json,md}`.\n"
    )

    # Data-integrity banner. This used to live only in the JSON's validation block, where a
    # `pass: False` on all 288 n-checks sat unread while this report described an empty world
    # as a set of negative findings (§F audit, 2026-08-02). A reader of the prose must see it.
    _nc = out["validation"]["n_checks"]
    _failed = [k for k, v in _nc.items() if not v["pass"]]
    _observed_any = any(v["observed"] for v in _nc.values())
    if not _observed_any:
        lines.append(
            "> 🚨 **Every contrast below has n = 0.** This report is about an empty input, not "
            "about the world: each 'no effect' and each 'consistent' verdict is vacuously true "
            "over an empty set. Do not read anything below as a finding.\n")
    elif _failed:
        lines.append(
            f"> ⚠️ **{len(_failed)} of {len(_nc)} pair-count checks did not reach the expected "
            "scored-set size** beyond the identity-mismatch allowance. Counts per contrast are "
            "in `axis_effect_size.json` → `validation.n_checks`.\n")
    if IDENTITY_SKIPS:
        _det = "; ".join(f"{b}/{s}/{m}: tasks {sorted(t)}"
                         for (b, s, m), t in sorted(IDENTITY_SKIPS.items()))
        lines.append(
            f"> **Episodes dropped for steps↔summary identity mismatch**: {_det}. Contrasts "
            "touching those arms pair on the intersection and so lose the task on both legs.\n")

    # ----- Tier 1 -----
    lines.append("## Tier 1 — Hook: is P-SoM distinct from both DOM and SoM?\n")
    hook_lines = ["| baseline | site | metric | DOM→P-SoM (compound) | P-SoM→SoM (image) | distinct from DOM? | distinct from SoM? |",
                  "|---|---|---|---|---|---|---|"]
    for baseline in BASELINES:
        for site in SITES:
            site_block = out["results"].get(baseline, {}).get(site, {})
            for metric, binary in metrics_def:
                metric_name = METRIC_LABELS[metric]
                r = site_block.get(metric_name)
                if not r or r["compound_dom_to_psom"].get("n", 0) == 0:
                    continue
                from_dom = "✅" if r["psom_distinct_from_dom"] else "—"
                from_som = "✅" if r["psom_distinct_from_som"] else "—"
                hook_lines.append(
                    f"| {baseline} | {site} | {REPORT_METRIC_LABELS[metric_name]} | {fmt(r['compound_dom_to_psom'], binary)} | {fmt(r['image'], binary)} | {from_dom} | {from_som} |"
                )
    lines.extend(hook_lines)

    independence = out["interpretation"]["tier1_hook"]
    lines.append("\n**P-SoM independence verdict** (cells where P-SoM differs from BOTH DOM and SoM, |effect|>0.1):")
    if independence["psom_distinct_from_both_dom_and_som"]:
        lines.append(f"- **Independent on**: {', '.join(independence['psom_distinct_from_both_dom_and_som'])}")
    _mp = independence.get("multiplicity") or {}
    if _mp.get("n_legs"):
        lines.append(
            f"\n> **Multiplicity.** That count asks only |effect| > 0.1, across "
            f"{_mp['n_combinations']} (cell, metric) combinations and {_mp['n_legs']} Wilcoxon "
            f"tests. Requiring **both** legs to also clear a correction applied jointly over all "
            f"legs: **{_mp['effect_and_bh_fdr_0.05']} survive Benjamini-Hochberg** (FDR 0.05) and "
            f"**{_mp['effect_and_holm_fwer_0.05']} survive Holm** (FWER 0.05), against "
            f"{_mp['effect_only']} on effect size alone. The BH set spans "
            f"{len({c.split('@')[1] for c in _mp['cells_bh']})} of the six cells, so it is not "
            f"one cell's accident: {', '.join(_mp['cells_bh']) or '—'}. Report the corrected "
            f"count, not the bare one.")
        lines.append(
            f"\n> **Distinct from both endpoints is not the same as independent.** A mode that "
            f"*interpolates* between DOM and SoM also differs from both. P-SoM is off the "
            f"DOM–SoM segment — an extremum rather than a midpoint — exactly when the two legs "
            f"disagree in sign. Of the {_mp['effect_and_bh_fdr_0.05']} BH survivors, "
            f"**{_mp['bh_and_off_segment']} are off the segment** and "
            f"{len(_mp['cells_bh_interpolating'])} interpolate"
            + (f" ({', '.join(_mp['cells_bh_interpolating'])})"
               if _mp['cells_bh_interpolating'] else "")
            + ". The off-segment count is the one that supports an independent arm; on "
            f"`finish_rate@B1/reddit` P-SoM sits about 9pp below *both* endpoints while the "
            f"endpoints differ from each other by 0.5pp.")
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
        "The second route, through P-prompt, closes this into a full diamond; the two paths and "
        "their additivity are reported in Tier 2b below.\n"
    )
    lines.append("| baseline | site | metric | text-axis (DOM→P-text) | prompt-axis (P-text→P-SoM) | image-axis (P-SoM→SoM) | dominant cascade axis | consistency |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for baseline in BASELINES:
        for site in SITES:
            site_block = out["results"].get(baseline, {}).get(site, {})
            for metric, binary in metrics_def:
                metric_name = METRIC_LABELS[metric]
                r = site_block.get(metric_name)
                if not r:
                    continue
                # Skip cells where the cascade text-axis is missing (no P-text) — full cascade not computable
                if r["text"].get("n", 0) == 0 and r["prompt"].get("n", 0) == 0:
                    continue
                check = r["consistency_check"]
                if check.get("skipped"):
                    check_label = "skipped"
                else:
                    check_label = "pass" if check.get("pass") else "fail"
                lines.append(
                    f"| {baseline} | {site} | {REPORT_METRIC_LABELS[metric_name]} | {fmt(r['text'], binary)} | {fmt(r['prompt'], binary)} | {fmt(r['image'], binary)} | {r['dominant_cascade']} | {check_label} |"
                )

    lines.append("\n★ marks Wilcoxon p<0.05. Effects with |d_z|>0.1 or |h|>0.1 are treated as non-negligible for axis dominance and cancellation checks.")

    # ----- Tier 2b: the diamond's second path -----
    lines.append("\n## Tier 2b — Diamond: base-set consistency across the two routes\n")
    lines.append(
        "Two routes lead from DOM to P-SoM:\n\n"
        "- **path A** DOM →(text)→ P-text →(prompt)→ P-SoM\n"
        "- **path B** DOM →(prompt)→ P-prompt →(text)→ P-SoM\n\n"
        "⚠️ **This is not an interaction test.** On mean differences the agreement is an "
        "algebraic identity — mean(P-text−DOM) + mean(P-SoM−P-text) = mean(P-SoM−DOM) — whenever "
        "the legs are averaged over the same tasks. A zero residual is therefore arithmetic and "
        "carries no evidence about text × prompt interaction. What a **non**-zero residual does "
        "carry is that the legs were averaged over *different* task sets, which is why the table "
        "is kept: it is a base-set consistency check that fires automatically. Testing for an "
        "interaction would require comparing effect sizes or fitting an interaction term, and "
        "nothing on this page does that.\n")
    dia = ["| baseline | site | metric | path A | path B | compound | A−comp | B−comp | same base set? |",
           "|---|---|---|---|---|---|---|---|---|"]
    n_add = n_tot_d = 0
    mismatched_legs = False
    for baseline in BASELINES:
        for site in SITES:
            for _metric, _binary in metrics_def:
                mn = METRIC_LABELS[_metric]
                d = out["results"][baseline][site].get(mn, {}).get("diamond_additivity")
                if not d or math.isnan(d["compound"]):
                    continue
                n_tot_d += 1
                ok = d["identity_holds_a"] and d["identity_holds_b"]
                n_add += ok
                mismatched_legs |= d["n_mismatched_legs"]
                verdict = "✅" if ok else (
                    "⚠️ n differs across legs" if d["n_mismatched_legs"] else "**✗ unexplained**")
                dia.append(
                    f"| {baseline} | {site} | {REPORT_METRIC_LABELS.get(mn, mn)} | "
                    f"{d['path_a_sum']:+.2f} | {d['path_b_sum']:+.2f} | {d['compound']:+.2f} | "
                    f"{d['residual_a']:+.3f} | {d['residual_b']:+.3f} | {verdict} |")
    lines.extend(dia)
    lines.append(
        f"\n**The identity holds in {n_add} of {n_tot_d} (cell × metric) combinations.** Where it "
        "does not, the legs were averaged over different task sets, not over a world containing "
        "an interaction.")
    if mismatched_legs:
        lines.append(
            "The rows that miss are exactly the ones on the B0·reddit P-SoM arm, whose legs are "
            "summed over 201 tasks against 203 on the others (the two identity-mismatched "
            "episodes). The residual there is the base-set difference and nothing else.")

    lines.append("\n## Cancellation patterns\n")
    cancellations = out["validation"]["antagonistic_pairs"]
    if cancellations:
        lines.append(
            "The following site/metric pairs are antagonistic: two cascade axes have opposite-signed effects and both exceed |0.1| effect size. "
            "These are exactly the cases where a DOM-vs-SoM endpoint comparison can mask the internal mechanism.\n"
        )
        for baseline in BASELINES:
            for site in SITES:
                site_block = out["results"].get(baseline, {}).get(site, {})
                for metric, _binary in metrics_def:
                    metric_name = METRIC_LABELS[metric]
                    block = site_block.get(metric_name)
                    if not block:
                        continue
                    patterns = block.get("cancellation_patterns", [])
                    for pattern in patterns:
                        key = "cohen_h_a" if "cohen_h_a" in pattern else "cohen_d_z_a"
                        key_b = "cohen_h_b" if "cohen_h_b" in pattern else "cohen_d_z_b"
                        label = "h" if key == "cohen_h_a" else "d_z"
                        lines.append(
                            f"- {baseline} {site} / {REPORT_METRIC_LABELS[metric_name]}: {pattern['axis_a']} vs {pattern['axis_b']} "
                            f"({label}={pattern[key]:+.2f} vs {pattern[key_b]:+.2f}) -> antagonistic"
                        )
    else:
        lines.append("No antagonistic pairs met the |0.1| effect-size threshold.")

    lines.append("\n## Consistency checks\n")
    _cc = out["validation"]["consistency_checks"]
    _cc_live = {k: v for k, v in _cc.items() if not v.get("skipped")}
    _cc_pass = sum(1 for v in _cc_live.values() if v.get("pass"))
    lines.append(
        f"text + prompt + image recovers the direct SoM − DOM endpoint in **{_cc_pass} of "
        f"{len(_cc_live)}** (site × metric) combinations, at tolerance 0.1 pp for the binary "
        "metric and 0.005 raw units for fractions and counts.\n\n"
        "Read this the same way as Tier 2b: on mean differences the three axes summing to the "
        "endpoint is an **algebraic identity**, so passing is arithmetic rather than evidence "
        "that the cascade decomposes cleanly. A failure means the legs were averaged over "
        "different task sets. This sentence used to be a fixed claim that every combination "
        "passed, which was true of the empty table it was printed under. (§F audit, 2026-08-02)"
    )

    lines.append("\n## Tier 2c — Mechanism (Micro): per-step decision quality\n")
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
    # The count here was hardcoded at 6 and sat two lines under "No antagonistic pair cleared
    # the |0.1| threshold", contradicting it. Count what this run actually found.
    _n_anta = len(cancellations)
    lines.append(
        "\n**4-level cascade design value**: decomposes DOM → SoM into three controlled transitions "
        "(AXTree vs [SOM_MARKS] structure, DOM vs SoM prompting, marginal image). This run finds "
        f"**{_n_anta} antagonistic mechanism pair(s)** that endpoint-only comparison would mask."
    )
    _om = (OUT_MD.with_name(OUT_MD.stem + "_with_wa" + OUT_MD.suffix)
           if WA_UNIVERSE is not None else OUT_MD)
    _om.write_text("\n".join(lines))
    print(f"[md]   {_om}")


if __name__ == "__main__":
    if "--with-wa" in sys.argv:
        _n = attach_wa()
        print(f"[axis_effect_size] WA cell attached: B1 x wa_reddit, n={_n} "
              f"(task set common to all six modes; WA has no AMENDMENT_08 list)",
              file=sys.stderr)
    main()
