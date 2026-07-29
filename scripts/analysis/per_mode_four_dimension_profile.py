#!/usr/bin/env python3
"""The four evidence dimensions, computed per MODE — 2026-07-28, post_hoc_exploratory

笔记 §108 defined the evidence layer as 4 measurement types × 4 comparison axes
= 16 sub-cells, and marked cross-mode the paper-headline axis
(`paper_section2_framework.canvas`). `INDEX.md §7` records that the framework was
never actually computed per mode: the 2×2 existed to disentangle two effects
(§103) and stopped once attribution was done, and **Vision is structurally off
the 2×2 grid** (no AXTree text) so it never even got computed incidentally. Only
the Macro dimension has ever been run per mode, once, on 2026-07-28.

This producer computes all four, for all six modes, on all six cells.

  Outcome     did the task succeed — SR, unique solves, per-mode solve sets
  Macro       how the agent acts on average — action mix, trajectory length
  Micro       per-step decision quality — parse / execution / no-op / fallback
  Efficiency  what it costs — billed cost, latency, tokens, steps

Two data layers, two different exclusion rules, deliberately not merged:

  * Outcome and Efficiency read `*_summary_v2.json` and use EVERY scored task.
  * Macro and Micro read `*_steps_v2.jsonl` and must exclude the episodes whose
    step file does not belong to their summary. `audit_steps_summary_identity.py`
    (2026-07-28) put that at exactly 2 of 7722 episodes — reddit tasks 87 and 149
    in B0·reddit·P-SoM, where the quarantine → resume-rerun path wrote a new
    summary (clean, 13 and 11 steps) but left the original 503-interrupted step
    file (18 and 14 steps) in place. The exclusion is reported per condition
    rather than silently applied.

⚠️ The canvas cells are a 2026-05-03 snapshot and several of their numbers have
since been retracted — `Efficiency × mode` still reads "(d) drop-one 1.7-3.8pp"
and `Outcome × task` still reads "B0 red P-text +3.81 ... CI sig", both from the
k=4/5 era. At k=6 H1 FAILED (θ_FE 0.7897, p=0.807, §395.6). Nothing here is
copied from the canvas; every number below is recomputed from the landed data.

Cost is comparable WITHIN a cell only (B0 bills a proxy API; B1/B2 are
electricity-derived), so the Efficiency dimension reports each mode relative to
that cell's DOM as well as in absolute terms.

post_hoc_exploratory=True, h10_eligible=False. Touches no gating producer.

Usage:
  .venv/bin/python3 scripts/analysis/per_mode_four_dimension_profile.py \
      --out docs/analysis/cross_sites/per_mode_four_dimension_profile.md \
      --json-out docs/analysis/cross_sites/per_mode_four_dimension_profile.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.analysis.axis_effect_size as A  # noqa: E402
from scripts.analysis.aggregate_phantom_lift import CELLS  # noqa: E402
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402
from scripts.analysis.lib.episode_rows import load_cell_task_rows  # noqa: E402

SCHEMA_VERSION = "2026-07-28-per-mode-four-dimension-profile-v1"

DISPLAY_MODES = ("DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM")
# axis_effect_size keeps two legacy key names; map at the boundary.
DISPLAY_TO_AXIS = {
    "DOM": "DOM", "SoM": "SoM", "Vision": "Vision", "P-text": "P-text",
    "P-SoM": "Phantom-SoM", "P-prompt": "Phantom-prompt",
}
COST_FIELD = "total_billed_cost_usd"

# Which dimension each metric belongs to, and whether bigger is notable.
DIMENSIONS: dict[str, list[tuple[str, str]]] = {
    "Outcome": [
        ("sr_pct", "success rate %"),
        ("n_success", "solves"),
        ("n_unique_solves", "unique solves (no other mode got it)"),
    ],
    "Macro": [
        ("n_steps", "steps / episode"),
        ("click_frac", "click fraction"),
        ("type_frac", "type fraction"),
        ("scroll_frac", "scroll fraction"),
        ("search_loop_rate", "search-loop rate"),
    ],
    "Micro": [
        ("parse_fail_rate", "parse-invalid step rate"),
        ("action_fail_rate", "action-execution failure rate"),
        ("no_change_rate", "page-unchanged (no-op) step rate"),
        ("locator_fallback_rate", "locator fallback rate"),
        ("action_repeat_frac", "consecutive same-action rate"),
        ("finish_rate", "episodes ending in finish"),
    ],
    "Efficiency": [
        ("mean_cost_usd", "billed cost / episode"),
        ("cost_rel_dom", "cost relative to DOM (within cell)"),
        ("mean_latency_s", "latency / episode (s)"),
        ("mean_tokens", "tokens / episode"),
    ],
}
LOWER_IS_BETTER = {
    "parse_fail_rate", "action_fail_rate", "no_change_rate",
    "locator_fallback_rate", "action_repeat_frac", "mean_cost_usd",
    "cost_rel_dom", "mean_latency_s", "mean_tokens", "n_steps",
}

# Metrics on which a mode's extreme position follows from how the mode is BUILT,
# not from how the agent behaved. Reporting these next to the empirical ones
# without a label would let an architectural tautology read as a finding — the
# same failure mode paper §3 already guards against by calling the phantom cost
# property "by construction" rather than a result.
BY_CONSTRUCTION: dict[str, str] = {
    "mean_tokens": "Vision carries no AXTree text, so its token count is lower "
                   "by construction",
    "mean_cost_usd": "billed cost is dominated by input tokens, so this inherits "
                     "the token property above",
    "cost_rel_dom": "same quantity as mean_cost_usd, expressed against DOM",
    "locator_fallback_rate": "Vision emits coordinates (`coordinate_type: "
                             "qwen_0_1000`) and has zero element ids, so it "
                             "barely enters the element-id locator path at all "
                             "— the residual 0.002-0.011 is not a lower "
                             "fallback rate on the same mechanism",
}

# A third category, added 2026-07-29 after Gemini (Mode C) attacked the binary
# split. These are not tautologies the way BY_CONSTRUCTION rows are — nothing in
# the design forces a particular magnitude — but they are the *expected mechanical
# consequence* of the same architectural choice, in a single causal chain:
#
#     coordinate-only addressing → clicks land off-target more often
#         → the page does not change → the agent scrolls to re-orient
#
# Calling them "empirical findings" without that chain stated implies the ordering
# was a surprise. It was not. Promoting any of them to a behavioural finding needs
# a baseline for what a coordinate-addressed agent *should* score, and then a
# demonstration that Vision exceeds it — which this profile does not provide.
ARCH_DOWNSTREAM: dict[str, str] = {
    "action_fail_rate": "coordinate addressing has no element-identity guarantee, "
                        "so a higher miss rate is the expected consequence, not a "
                        "discovery about behaviour",
    "no_change_rate": "a missed click leaves the page unchanged — this is "
                      "downstream of the action-failure row above, not independent "
                      "evidence",
    "scroll_frac": "re-orienting after a no-op, plus viewport-only observation with "
                   "no AXTree to enumerate off-screen targets, both push toward "
                   "scrolling; the 1.2-6.8x magnitude is real but its DIRECTION was "
                   "predictable from the design",
}


def _num(v: Any) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def summary_layer(spec: dict) -> dict[str, dict[int, dict]]:
    """Per-mode task -> summary row over the cell's canonical scored universe."""
    universe, _ = expected_scored_ids(spec["site"])
    rows_by_mode = load_cell_task_rows(spec, modes=DISPLAY_MODES)
    out: dict[str, dict[int, dict]] = {}
    for m in DISPLAY_MODES:
        rows = rows_by_mode.get(m) or {}
        out[m] = {t: rows[t] for t in sorted(universe) if t in rows}
    return out


def steps_layer(baseline: str, site: str, mode: str) -> tuple[dict[int, dict], list[int]]:
    """Per-task step-derived metrics, plus the task ids that had to be skipped.

    Skips are identity mismatches (steps file does not belong to its summary).
    They are returned rather than logged so the caller can disclose the count.
    """
    ep_dir = A.STEP_DIRS.get(baseline, {}).get(site, {}).get(DISPLAY_TO_AXIS[mode])
    if ep_dir is None or not ep_dir.exists():
        return {}, []
    # A landed reddit condition holds 205 step files against a 203-task SCORED
    # set: AMENDMENT_08 keeps the runner COLLECTING the protocol-excluded tasks
    # (58, 160). Globbing the directory would fold them into every Macro/Micro
    # mean. `summary_layer` already restricts to the scored universe; this path
    # must use the same gate or the two data layers silently describe different
    # task sets (test_universe_consumption_lint guards the summary side only).
    universe, _ = expected_scored_ids(site)
    per_task: dict[int, dict] = {}
    skipped: list[int] = []
    for path in sorted(ep_dir.glob(f"{site}_task_*_steps_v2.jsonl")):
        tid = A.step_task_id(path)
        if tid not in universe:
            continue
        try:
            steps = A.read_steps(path)
        except Exception:  # noqa: BLE001 — identity mismatch is the expected case
            skipped.append(tid)
            continue
        n = len(steps)
        if n == 0:
            continue
        acts = [A._action_type(s) for s in steps]
        parse_fail = sum(1 for s in steps if s.get("parse_valid") is False)
        act_fail = sum(1 for s in steps if s.get("action_success") is False)
        no_change = sum(1 for s in steps if s.get("page_changed") is False)
        loc_fb = sum(1 for s in steps
                     if isinstance(s.get("locator_route_meta"), dict)
                     and s["locator_route_meta"].get("fallback_used") is True)
        repeats = sum(1 for i in range(1, n)
                      if acts[i] is not None and acts[i] == acts[i - 1])
        search_markers = A.SEARCH_MARKERS.get(site, ())
        search_steps = 0
        for i, s in enumerate(steps):
            url = s.get("obs_url", "") or ""
            nxt = steps[i + 1].get("obs_url", "") if i + 1 < n else ""
            if any(mk in url for mk in search_markers) or (
                    acts[i] == "type" and any(mk in nxt for mk in search_markers)):
                search_steps += 1
        per_task[tid] = {
            "n_steps": float(n),
            "click_frac": acts.count("click") / n,
            "type_frac": acts.count("type") / n,
            "scroll_frac": acts.count("scroll") / n,
            "search_loop_rate": 1.0 if search_steps >= 2 else 0.0,
            "parse_fail_rate": parse_fail / n,
            "action_fail_rate": act_fail / n,
            "no_change_rate": no_change / n,
            "locator_fallback_rate": loc_fb / n,
            "action_repeat_frac": (repeats / (n - 1)) if n > 1 else 0.0,
            "finish_rate": 1.0 if acts[-1] == "finish" else 0.0,
            # Raw numerator/denominator kept alongside the per-episode rate so the
            # POOLED-STEP estimand is computable downstream. The two answer
            # different questions and can differ substantially (codex Mode B,
            # 2026-07-29: B1·cls Vision action-failure is 0.4540 task-macro vs
            # 0.6386 pooled-step) — a rate reported without naming its estimand
            # is not reproducible, so both are now carried and both are reported.
            "_n_steps": float(n),
            "_click": float(acts.count("click")), "_type": float(acts.count("type")),
            "_scroll": float(acts.count("scroll")), "_parse_fail": float(parse_fail),
            "_act_fail": float(act_fail), "_no_change": float(no_change),
            "_loc_fb": float(loc_fb), "_repeats": float(repeats),
            "_repeat_denom": float(n - 1) if n > 1 else 0.0,
        }
    return per_task, sorted(skipped)


# metric key -> (numerator field, denominator field) for the pooled-step estimand.
POOLED_SPEC = {
    "click_frac": ("_click", "_n_steps"),
    "type_frac": ("_type", "_n_steps"),
    "scroll_frac": ("_scroll", "_n_steps"),
    "parse_fail_rate": ("_parse_fail", "_n_steps"),
    "action_fail_rate": ("_act_fail", "_n_steps"),
    "no_change_rate": ("_no_change", "_n_steps"),
    "locator_fallback_rate": ("_loc_fb", "_n_steps"),
    "action_repeat_frac": ("_repeats", "_repeat_denom"),
}


def profile_cell(spec: dict) -> dict[str, Any]:
    baseline, site = spec["baseline"], spec["site"]
    summ = summary_layer(spec)
    solve_sets = {m: {t for t, r in summ[m].items() if r.get("success") is True}
                  for m in DISPLAY_MODES}

    # ── trajectory layer, loaded once so the modes can be matched on a common set
    steps_by_mode: dict[str, dict[int, dict]] = {}
    skipped_by_mode: dict[str, list[int]] = {}
    for m in DISPLAY_MODES:
        steps_by_mode[m], skipped_by_mode[m] = steps_layer(baseline, site, m)

    # (codex Mode B finding, 2026-07-29) Cross-mode comparison must be PAIRED.
    # Excluding the two identity-mismatched episodes removes them from P-SoM
    # only, so B0·reddit compared P-SoM on 201 tasks against five modes on 203 —
    # a per-mode difference then partly reflects a different task set. Macro and
    # Micro are therefore computed on the intersection of tasks for which EVERY
    # mode has a usable trajectory. The dropped count is reported.
    common_steps: set[int] = set.intersection(
        *[set(steps_by_mode[m]) for m in DISPLAY_MODES]) if all(
        steps_by_mode[m] for m in DISPLAY_MODES) else set()

    per_mode: dict[str, dict[str, Any]] = {}
    for m in DISPLAY_MODES:
        rows = summ[m]
        n = len(rows)
        others = set().union(*[solve_sets[o] for o in DISPLAY_MODES if o != m]) \
            if len(DISPLAY_MODES) > 1 else set()
        metrics: dict[str, float | None] = {
            "sr_pct": 100.0 * len(solve_sets[m]) / n if n else None,
            "n_success": float(len(solve_sets[m])),
            "n_unique_solves": float(len(solve_sets[m] - others)),
            "mean_cost_usd": statistics.fmean(
                [_num(r.get(COST_FIELD)) for r in rows.values()]) if n else None,
            "mean_latency_s": statistics.fmean(
                [_num(r.get("total_latency_ms")) / 1000.0
                 for r in rows.values()]) if n else None,
            "mean_tokens": statistics.fmean(
                [_num(r.get("total_tokens")) for r in rows.values()]) if n else None,
            "n_tasks_summary": float(n),
        }
        st = {t: v for t, v in steps_by_mode[m].items() if t in common_steps}
        if st:
            for k in next(iter(st.values())):
                if k.startswith("_"):
                    continue
                # task-macro: mean over episodes of a within-episode rate
                metrics[k] = statistics.fmean([v[k] for v in st.values()])
            # pooled-step: sum(numerator) / sum(denominator) over all steps
            for k, (num, den) in POOLED_SPEC.items():
                d = sum(v[den] for v in st.values())
                metrics[f"{k}__pooled"] = (
                    sum(v[num] for v in st.values()) / d) if d else None
        metrics["n_tasks_steps"] = float(len(st))
        metrics["n_tasks_steps_excluded"] = float(len(skipped_by_mode[m]))
        per_mode[m] = metrics

    dom_cost = per_mode["DOM"].get("mean_cost_usd")
    for m in DISPLAY_MODES:
        c = per_mode[m].get("mean_cost_usd")
        per_mode[m]["cost_rel_dom"] = (c / dom_cost) if (c and dom_cost) else None

    return {
        "cell_id": f"{baseline}_{site}", "baseline": baseline, "site": site,
        "per_mode": per_mode,
        "n_common_trajectory_tasks": len(common_steps),
        "n_dropped_for_pairing": {
            m: len(steps_by_mode[m]) - len(common_steps) for m in DISPLAY_MODES},
        "steps_excluded_tasks": {m: v for m, v in skipped_by_mode.items() if v},
    }


def rank_consistency(cells: list[dict]) -> dict[str, Any]:
    """For each metric, which mode is the extreme, and in how many of the 6 cells.

    This is the shape the one existing per-mode finding takes ("Vision scroll_frac
    highest in 6/6 cells, 2.6-10x"), so the whole profile is reported in it: a
    per-mode difference only means something if it survives across cells.
    """
    out: dict[str, Any] = {}
    for dim, metrics in DIMENSIONS.items():
        for key, label in metrics:
            top: Counter = Counter()
            bot: Counter = Counter()
            ratios: list[float] = []
            n_cells = ties_high = ties_low = 0
            for c in cells:
                vals = {m: c["per_mode"][m].get(key) for m in DISPLAY_MODES}
                vals = {m: v for m, v in vals.items() if v is not None}
                if len(vals) < 2:
                    continue
                n_cells += 1
                # (codex Mode B finding, 2026-07-29) `max`/`min` silently award a
                # tie to whichever mode comes first in DISPLAY_MODES. On
                # B2·classifieds SoM and Vision both score 2.2321% with 5 solves,
                # and the table reported SoM alone as the maximum — an artifact of
                # list order, not a measurement. Co-extrema now share the count and
                # ties are recorded, so a "6/6" can never be manufactured by
                # ordering.
                hv, lv = max(vals.values()), min(vals.values())
                hi_modes = [m for m, v in vals.items() if v == hv]
                lo_modes = [m for m, v in vals.items() if v == lv]
                for m in hi_modes:
                    top[m] += 1.0 / len(hi_modes)
                for m in lo_modes:
                    bot[m] += 1.0 / len(lo_modes)
                if len(hi_modes) > 1:
                    ties_high += 1
                if len(lo_modes) > 1:
                    ties_low += 1
                rest = [v for m, v in vals.items() if m not in hi_modes]
                second = max(rest) if rest else 0.0
                if second > 0:
                    ratios.append(hv / second)
            if not n_cells:
                continue
            hi_mode, hi_n = top.most_common(1)[0]
            lo_mode, lo_n = bot.most_common(1)[0]
            out[key] = {
                "dimension": dim, "label": label, "n_cells": n_cells,
                "highest_mode": hi_mode, "highest_in_n_cells": hi_n,
                "lowest_mode": lo_mode, "lowest_in_n_cells": lo_n,
                "top_vs_second_ratio_min": min(ratios) if ratios else None,
                "top_vs_second_ratio_max": max(ratios) if ratios else None,
                "unanimous_high": abs(hi_n - n_cells) < 1e-9,
                "unanimous_low": abs(lo_n - n_cells) < 1e-9,
                "n_cells_with_tie_at_top": ties_high,
                "n_cells_with_tie_at_bottom": ties_low,
                "lower_is_better": key in LOWER_IS_BETTER,
                "by_construction": BY_CONSTRUCTION.get(key),
                "arch_downstream": ARCH_DOWNSTREAM.get(key),
            }
    return out


def _fmt(v: float | None, key: str) -> str:
    if v is None:
        return "—"
    if key in ("n_success", "n_unique_solves"):
        return f"{v:.0f}"
    if key in ("mean_cost_usd",):
        return f"{v:.5f}"
    if key in ("mean_tokens",):
        return f"{v:.0f}"
    if key in ("sr_pct", "mean_latency_s", "n_steps"):
        return f"{v:.2f}"
    return f"{v:.4f}"


def render(payload: dict) -> str:
    L: list[str] = []
    L.append("# Per-mode four-dimension evidence profile")
    L.append("")
    L.append(f"- generated: `{payload['generated_utc']}`")
    L.append(f"- schema: `{payload['schema_version']}`")
    L.append("- **post_hoc_exploratory=True / h10_eligible=False**")
    L.append("- 笔记 §108 evidence layer, cross-mode axis (the paper-headline axis "
             "per `paper_section2_framework.canvas`). `INDEX.md §7`: the framework "
             "was defined but only the **Macro** dimension had ever been computed "
             "per mode. This is the first run of all four.")
    L.append("- ⚠️ No number here is taken from the canvas — its cells are a "
             "2026-05-03 snapshot and several were later retracted (it still shows "
             "`drop-one 1.7-3.8pp`; at k=6 **H1 FAILED**, θ_FE 0.7897, p=0.807, §395.6).")
    L.append("- cost comparable **within a cell only** (B0 = proxy API bill; "
             "B1/B2 = electricity-derived), hence the `cost_rel_dom` column.")
    L.append("")

    exc = payload["exclusions"]
    L.append("## Data layers and what was excluded")
    L.append("")
    L.append("Outcome + Efficiency read episode summaries and use **every** scored "
             "task. Macro + Micro read step JSONL and must drop episodes whose step "
             "file does not belong to their summary.")
    L.append("")
    if exc:
        for cid, modes in exc.items():
            for m, tasks in modes.items():
                L.append(f"- **{cid} / {m}**: excluded from Macro+Micro — tasks "
                         f"`{tasks}` ({len(tasks)} episodes). Cause: quarantine → "
                         "resume-rerun wrote a new summary but left the original "
                         "interrupted step file in place.")
    else:
        L.append("- none")
    L.append("")
    L.append(f"Blast radius measured by `audit_steps_summary_identity.py`: "
             f"**{payload['audit']['n_mismatch']} of {payload['audit']['n_episodes']} "
             f"episodes** across all {payload['audit']['n_combinations']} combinations.")
    L.append("")

    L.append("## Cross-cell consistency — which per-mode differences survive")
    L.append("")
    L.append("A per-mode difference only counts if it holds across cells. "
             "`unanimous` = the same mode is the extreme in all 6 cells.")
    L.append("")
    L.append("⚙️ = the extreme follows from how the mode is **built** (tautology). "
             "◆ = the magnitude is real but its **direction was predictable** from "
             "the design. Neither may be cited as a behavioural finding. "
             "`tie` counts cells where two or more modes share the extreme — those "
             "cells contribute a fractional count, so ordering can never manufacture "
             "a 6/6.")
    L.append("")
    L.append("| dim | metric | highest | in | lowest | in | tie | top÷2nd (min–max) | unanimous |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for key, r in payload["consistency"].items():
        ratio = "—"
        if r["top_vs_second_ratio_min"] is not None:
            ratio = (f"{r['top_vs_second_ratio_min']:.2f}–"
                     f"{r['top_vs_second_ratio_max']:.2f}×")
        mark = ""
        if r["unanimous_high"]:
            mark = f"**high: {r['highest_mode']} 6/6**"
        elif r["unanimous_low"]:
            mark = f"**low: {r['lowest_mode']} 6/6**"
        if mark and r.get("by_construction"):
            mark = "⚙️ " + mark + " (by construction)"
        elif mark and r.get("arch_downstream"):
            mark = "◆ " + mark + " (arch. downstream)"
        ties = r.get("n_cells_with_tie_at_top", 0) + r.get("n_cells_with_tie_at_bottom", 0)
        L.append(f"| {r['dimension']} | {r['label']} | {r['highest_mode']} | "
                 f"{r['highest_in_n_cells']:.3g}/{r['n_cells']} | {r['lowest_mode']} | "
                 f"{r['lowest_in_n_cells']:.3g}/{r['n_cells']} | "
                 f"{ties if ties else '—'} | {ratio} | {mark} |")
    L.append("")
    dwn = {k: r for k, r in payload["consistency"].items()
           if r.get("arch_downstream") and (r["unanimous_high"] or r["unanimous_low"])}
    if dwn:
        L.append("Why each ◆ row is architecturally downstream:")
        L.append("")
        for k, r in dwn.items():
            L.append(f"- `{r['label']}` — {r['arch_downstream']}.")
        L.append("")
    bc = {k: r for k, r in payload["consistency"].items()
          if r.get("by_construction") and (r["unanimous_high"] or r["unanimous_low"])}
    if bc:
        L.append("Why each ⚙️ row is architectural:")
        L.append("")
        for k, r in bc.items():
            L.append(f"- `{r['label']}` — {r['by_construction']}.")
        L.append("")

    for dim, metrics in DIMENSIONS.items():
        L.append(f"## {dim}")
        L.append("")
        for cell in payload["cells"]:
            L.append(f"### {cell['cell_id']}")
            L.append("")
            if dim in ("Macro", "Micro"):
                L.append(f"trajectory metrics on the **{cell['n_common_trajectory_tasks']} "
                         "tasks every mode has a usable trajectory for** (paired); "
                         f"dropped for pairing: {cell['n_dropped_for_pairing']}")
                L.append("")
            L.append("| metric | " + " | ".join(DISPLAY_MODES) + " |")
            L.append("|---" * (len(DISPLAY_MODES) + 1) + "|")
            for key, label in metrics:
                vals = [_fmt(cell["per_mode"][m].get(key), key) for m in DISPLAY_MODES]
                suffix = " *(task-macro)*" if key in POOLED_SPEC else ""
                L.append(f"| {label}{suffix} | " + " | ".join(vals) + " |")
                if key in POOLED_SPEC:
                    pv = [_fmt(cell["per_mode"][m].get(f"{key}__pooled"), key)
                          for m in DISPLAY_MODES]
                    L.append(f"| {label} *(pooled-step)* | " + " | ".join(pv) + " |")
            L.append("")
    L.append("## Reading notes")
    L.append("")
    for line in payload["reading"]:
        L.append(line)
    L.append("")
    return "\n".join(L)


def reading(cells: list[dict], cons: dict) -> list[str]:
    out = []
    unan = [(k, r) for k, r in cons.items() if r["unanimous_high"] or r["unanimous_low"]]
    tau = [(k, r) for k, r in unan if r.get("by_construction")]
    dwn = [(k, r) for k, r in unan if r.get("arch_downstream")]
    emp = [(k, r) for k, r in unan
           if not r.get("by_construction") and not r.get("arch_downstream")]

    def line(k, r):
        side = "highest" if r["unanimous_high"] else "lowest"
        mode = r["highest_mode"] if r["unanimous_high"] else r["lowest_mode"]
        ratio = ""
        if r["unanimous_high"] and r["top_vs_second_ratio_min"] is not None:
            ratio = (f", {r['top_vs_second_ratio_min']:.1f}–"
                     f"{r['top_vs_second_ratio_max']:.1f}× the next mode")
        return f"   - `{r['label']}` ({r['dimension']}): **{mode}** {side} in 6/6{ratio}"

    out.append(
        f"1. **{len(unan)} of {len(cons)} metrics have a unanimous extreme mode — "
        f"but they fall into three classes, not two: {len(emp)} empirical, "
        f"{len(dwn)} architecturally downstream, {len(tau)} tautological.** "
        "The earlier revision of this file used a binary split and put the "
        "downstream group on the empirical side; Gemini (cross-AI Mode C, "
        "2026-07-29) attacked that, correctly.")
    if emp:
        out.append("   **Empirical — not predictable from the design:**")
        out.extend(line(k, r) for k, r in emp)
    if dwn:
        out.append("   **◆ Architecturally downstream — real magnitudes, predictable "
                   "direction.** One causal chain explains all of them: coordinate-only "
                   "addressing → more off-target clicks → page unchanged → scroll to "
                   "re-orient. Citing these as behavioural discoveries overstates them; "
                   "promoting one requires a baseline for what a coordinate-addressed "
                   "agent *should* score, which this profile does not provide.")
        out.extend(line(k, r) for k, r in dwn)
    if tau:
        out.append("   **⚙️ By construction — do NOT cite as findings:** "
                   + ", ".join(f"`{r['label']}`" for _, r in tau))
    out.append("2. **Mechanism claims are not established here.** These are Evidence-layer "
               "observations. Reading `scroll_frac` as \"viewport-only forces scrolling\" "
               "is an Explanation-layer hypothesis and the canvas's own reviewer caveat "
               "(\"Evidence ≠ Explanation\") applies: the two must be written separately "
               "and linked explicitly, not merged.")
    out.append("3. **Vision is structurally off the 2×2 grid** (no AXTree text), which is "
               "why it never appeared in the earlier per-axis analyses (§103). Any Vision "
               "row here is the first time that mode has been profiled on this dimension.")
    out.append("4. **Two estimands are reported for every step-level rate.** `task-macro` "
               "is the mean over episodes of a within-episode rate; `pooled-step` is "
               "total numerator over total denominator. They weight long and short "
               "episodes differently and can diverge substantially, so neither is "
               "reported alone.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs/analysis/cross_sites/per_mode_four_dimension_profile.md")
    ap.add_argument("--json-out", type=Path,
                    default=REPO / "docs/analysis/cross_sites/per_mode_four_dimension_profile.json")
    ap.add_argument("--audit-json", type=Path,
                    default=REPO / "docs/analysis/cross_sites/steps_summary_identity_audit.json")
    a = ap.parse_args()

    cells = [profile_cell(spec) for spec in CELLS]
    cons = rank_consistency(cells)
    exclusions = {c["cell_id"]: c["steps_excluded_tasks"]
                  for c in cells if c["steps_excluded_tasks"]}

    audit = {"n_mismatch": None, "n_episodes": None, "n_combinations": None}
    if a.audit_json.is_file():
        ad = json.loads(a.audit_json.read_text())
        audit = {k: ad.get(k) for k in ("n_mismatch", "n_episodes", "n_combinations")}

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "post_hoc_exploratory": True,
        "h10_eligible": False,
        "framework": "笔记 §108 evidence layer — 4 measurement types × cross-mode axis",
        "cost_field": COST_FIELD,
        "cells": cells,
        "consistency": cons,
        "exclusions": exclusions,
        "audit": audit,
    }
    payload["reading"] = reading(cells, cons)

    a.json_out.parent.mkdir(parents=True, exist_ok=True)
    a.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render(payload), encoding="utf-8")
    print(render(payload))
    print(f"\nwrote {a.out}\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
