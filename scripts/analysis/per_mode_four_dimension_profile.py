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

SCHEMA_VERSION = "2026-08-02-per-mode-four-dimension-profile-v2"

# Why the WA cell is opt-in and writes elsewhere. Consistency is counted as "the same mode is
# the extreme in k of the N cells", so appending a seventh cell rewrites every denominator and
# is not a superset of the six-cell result. Empirically it moves things: SoM's three-metric
# "terminates sooner" signature (fewest steps, least budget exhaustion, most explicit finishes)
# is 5/6 on the VWA grid and 5/7 with WA, because WA is the one cell where SoM is not the
# strongest mode. The load-bearing negative is unaffected: the four image-free modes reach the
# bar on NOTHING under either denominator, which is what licenses grouping them.

# Episode step budget (configs/exp_v2_base.yaml `max_steps`, B-700 2026-05-17: 40 -> 30).
# An episode at the cap did not decide to stop; it ran out.
MAX_STEPS = 30

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
        ("cap_hit_rate", "episodes exhausting the step budget"),
        ("click_frac", "click fraction"),
        ("type_frac", "type fraction"),
        ("scroll_frac", "scroll fraction"),
        ("search_loop_rate", "search-loop rate"),
        ("url_revisit_rate", "URL-revisit step rate"),
    ],
    "Micro": [
        ("parse_fail_rate", "parse-invalid step rate"),
        ("action_fail_rate", "action-execution failure rate"),
        ("click_fail_rate", "action failure | action was a click"),
        ("type_fail_rate", "action failure | action was a type"),
        ("no_change_rate", "page-unchanged (no-op) step rate"),
        ("scroll_inert_rate", "scroll action that did not move the viewport"),
        ("noop_inert_rate", "no-op despite a SUCCEEDING action"),
        ("visibility_gap_rate", "page changed but channel did not show it"),
        ("locator_fallback_rate", "locator fallback rate"),
        ("action_repeat_frac", "consecutive same-action rate"),
        ("finish_rate", "episodes ending in finish"),
    ],
    "Efficiency": [
        ("mean_cost_usd", "billed cost / episode"),
        ("cost_rel_dom", "cost relative to DOM (within cell)"),
        ("mean_latency_s", "latency / episode (s)"),
        ("mean_latency_canonical_s", "latency canonical / episode (s)"),
        ("mean_tokens", "tokens / episode"),
    ],
}
LOWER_IS_BETTER = {
    "parse_fail_rate", "action_fail_rate", "no_change_rate", "scroll_inert_rate",
    "locator_fallback_rate", "action_repeat_frac", "mean_cost_usd",
    "cost_rel_dom", "mean_latency_s", "mean_latency_canonical_s", "mean_tokens", "n_steps",
    "cap_hit_rate", "url_revisit_rate", "noop_inert_rate",
    "visibility_gap_rate", "click_fail_rate", "type_fail_rate",
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
#
# ⚠️ 2026-08-02 AUDIT. Each entry below deletes a whole row of evidence, so each one
# needs its own support, and until now none had any: they were author-written causal
# assertions. All three were tested. **None survives as originally written.** The
# strings have been rewritten to what the data supports; the ◆ marking is kept where
# a surviving mechanism still makes the direction predictable, and the tests are
# recorded here so the next reader does not have to rediscover them.
ARCH_DOWNSTREAM: dict[str, str] = {
    "action_fail_rate": "coordinate addressing has no element-identity guarantee, so a "
                        "higher miss rate is the expected direction. ⚠️ WEAKENED: if that "
                        "mechanism drove the total, the excess should concentrate in "
                        "spatially-targeted actions. It does not. Vision leads on "
                        "click-conditional failure in only 3/6 cells and is the LOWEST "
                        "mode on type-conditional failure in 4/6. The direction of the "
                        "total is still predictable; the stated decomposition is not what "
                        "produces it",
    "no_change_rate": "largely downstream of the action-failure row above. ⚠️ REFINED: "
                      "measured, `action_success = False` implies `page_changed = False` "
                      "in 100% of steps across all six B0 combinations checked, so action "
                      "failure is a strict SUBSET of no-op and the metric decomposes "
                      "exactly into it plus a successful-but-inert residual. Vision's high "
                      "rate is dominated by the first term (85-95% of its no-ops), but the "
                      "residual is NOT Vision-led — see the separate `noop_inert_rate` row, "
                      "where SoM is the extreme in 5/6 cells",
    "scroll_frac": "viewport-only observation with no AXTree to enumerate off-screen "
                   "targets pushes toward scrolling; the 1.2-6.8x magnitude is real but "
                   "its DIRECTION was predictable from the design. ⚠️ HALF REFUTED: this "
                   "entry previously gave a second mechanism, re-orienting after a no-op. "
                   "Measured on B0 x {cls, red} x {dom, som, vision}, the share of scroll "
                   "steps whose predecessor was a no-op sits AT OR BELOW the base rate of "
                   "no-ops in the same run in all six combinations (Vision on classifieds: "
                   "18.5% against a 36.4% base, 17.9 points below chance). Scrolls are not "
                   "preferentially preceded by no-ops anywhere. Only the viewport-"
                   "enumeration mechanism survives and the marking now rests on it alone",
}

# Metrics added 2026-08-02 that are NOT yet classified either way. Their absence from
# the two registries above means "nobody has adjudicated this", not "verified clean".
# Saying so explicitly stops an unflagged row from reading as an endorsed finding.
UNADJUDICATED: dict[str, str] = {
    "url_revisit_rate": "Vision is the extreme in every cell, the only unflagged unanimous row "
                        "in the grid. A plausible architectural story exists (a channel "
                        "that cannot enumerate off-screen targets navigates more "
                        "exploratorily and so returns to pages it has seen), and it has "
                        "not been tested. Do not cite as a behavioural finding until it is",
    "cap_hit_rate": "SoM is the extreme (lowest) in 5/6. Reads with `n_steps` (lowest, "
                    "5/6) and `finish_rate` (highest, 5/6) as one signature rather than "
                    "three findings: the fused mode terminates sooner and more often by "
                    "choice. Count it once",
    "noop_inert_rate": "SoM highest in 5/6. This is the residual of `no_change_rate` after "
                       "action failures are removed, so it is the part of that metric the "
                       "◆ marking above does NOT cover",
    "visibility_gap_rate": "no signal: Vision is the extreme at both ends (highest in 2/6, "
                           "lowest in 4/6). A two-cell probe on 2026-08-02 read Vision as "
                           "uniformly highest and that reading did not survive the full "
                           "grid. Reported so the absence is on the record",
    "click_fail_rate": "diagnostic for the `action_fail_rate` marking above, not a "
                       "standalone finding",
    "type_fail_rate": "diagnostic for the `action_fail_rate` marking above, not a "
                      "standalone finding",
}


# WebArena as a seventh cell. Kept behind --with-wa and appended AFTER the six, so with the
# flag off every byte of the VWA output is unchanged. WA carries no AMENDMENT_08 exclusion, so
# its universe is the task set common to all six modes rather than a canonical scored list.
#
# The step records for these runs live on the paper-grade host and were missing from the local
# mirror until 2026-08-02, which briefly read as a structural limit and was a sync gap (笔记
# §407.25). If this builder raises on a missing steps dir, re-pull rather than concluding the
# layer is impossible.
WA_ROOT = REPO / "results/webarena/phase1"
# Parameterised on backbone 2026-08-03: B0 x WA landed 07:23 that morning, three days
# before the estimate in HANDOFF_frame_rethink §1 ("no new data before the deadline").
# WA is no longer a single cell, so the glob templates carry a {b} slot rather than a
# hardcoded B1. Both WA cells are appended under --with-wa; the consistency denominator
# is computed from len(cells), so it follows automatically (/6 → /8).
WA_GLOB_TMPL = {
    "DOM": "{b}_dom_wa_reddit_2026*_R*", "SoM": "{b}_som_wa_reddit_2026*_R*",
    "Vision": "{b}_vision_wa_reddit_2026*_R*", "P-text": "{b}_phantom_text_wa_reddit_2026*_R*",
    "P-prompt": "{b}_phantom_prompt_wa_reddit_2026*_R*",
    "P-SoM": "{b}_phantom_som_wa_reddit_2026*_R*",
}
WA_BASELINES = ("B1", "B0")


def wa_spec(baseline: str = "B1") -> dict:
    """<baseline> x WebArena-reddit as a profile cell. Raises rather than degrading silently."""
    import glob as _glob
    modes: dict[str, Path] = {}
    for disp, tmpl in WA_GLOB_TMPL.items():
        pat = tmpl.format(b=baseline)
        hits = sorted(d for d in _glob.glob(str(WA_ROOT / pat))
                      if Path(d).is_dir() and "ABORTED" not in d)
        if not hits:
            raise SystemExit(f"wa_spec[{baseline}]: no run dir for {disp} ({pat})")
        ep = next(Path(hits[-1]).glob("*/episodes"), None)
        if ep is None or not ep.is_dir():
            raise SystemExit(f"wa_spec: no episodes dir under {hits[-1]}")
        modes[disp] = ep
    universe = None
    for ep in modes.values():
        ids = {int(f.name.split("_task_")[1].split("_")[0])
               for f in ep.glob("reddit_task_*_summary_v2.json")}
        universe = ids if universe is None else (universe & ids)
    if not universe:
        raise SystemExit(f"wa_spec[{baseline}]: empty task intersection across the six modes")
    return {"baseline": baseline, "site": "wa_reddit", "n_expected": len(universe),
            "modes": modes, "universe": universe,
            "steps_glob": "reddit_task_*_steps_v2.jsonl"}


def _num(v: Any) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def summary_layer(spec: dict) -> dict[str, dict[int, dict]]:
    """Per-mode task -> summary row over the cell's canonical scored universe."""
    universe = spec.get("universe")           # WA supplies its own; VWA uses the canonical set
    if universe is None:
        universe, _ = expected_scored_ids(spec["site"])
    rows_by_mode = load_cell_task_rows(spec, modes=DISPLAY_MODES)
    out: dict[str, dict[int, dict]] = {}
    for m in DISPLAY_MODES:
        rows = rows_by_mode.get(m) or {}
        out[m] = {t: rows[t] for t in sorted(universe) if t in rows}
    return out


def steps_layer(baseline: str, site: str, mode: str, *, spec: dict | None = None
                ) -> tuple[dict[int, dict], list[int]]:
    """Per-task step-derived metrics, plus the task ids that had to be skipped.

    Skips are identity mismatches (steps file does not belong to its summary).
    They are returned rather than logged so the caller can disclose the count.
    """
    if spec is not None and spec.get("universe") is not None:
        ep_dir = spec["modes"].get(mode)          # WA: dirs come from the spec, not STEP_DIRS
    else:
        ep_dir = A.STEP_DIRS.get(baseline, {}).get(site, {}).get(DISPLAY_TO_AXIS[mode])
    if ep_dir is None or not ep_dir.exists():
        return {}, []
    # A landed reddit condition holds 205 step files against a 203-task SCORED
    # set: AMENDMENT_08 keeps the runner COLLECTING the protocol-excluded tasks
    # (58, 160). Globbing the directory would fold them into every Macro/Micro
    # mean. `summary_layer` already restricts to the scored universe; this path
    # must use the same gate or the two data layers silently describe different
    # task sets (test_universe_consumption_lint guards the summary side only).
    universe = spec.get("universe") if spec is not None else None
    if universe is None:
        universe, _ = expected_scored_ids(site)
    glob_pat = (spec.get("steps_glob") if spec is not None else None) \
        or f"{site}_task_*_steps_v2.jsonl"
    per_task: dict[int, dict] = {}
    skipped: list[int] = []
    for path in sorted(ep_dir.glob(glob_pat)):
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
        # scroll_inert: a scroll action after which `state_digest.scroll_y` is unchanged —
        # the agent asked to scroll and the viewport did not move. Added 2026-08-02 from the
        # §G1 unread-field inventory; `state_digest.scroll_y_before/after` were populated on
        # every step and read by nothing. Most of that inventory turned out to be dead schema
        # (retry_count, screenshot_timeout_recovered, destructive_action_count and six others
        # are 0% or never populated), which is itself the answer to "the metric pool is only
        # as wide as what we chose".
        _scroll_steps = [s for s in steps if A._action_type(s) == "scroll"]
        scroll_inert = sum(
            1 for s in _scroll_steps
            if (s.get("state_digest") or {}).get("scroll_y_after") is not None
            and (s["state_digest"]["scroll_y_after"]
                 == (s["state_digest"] or {}).get("scroll_y_before")))
        scroll_denom = sum(
            1 for s in _scroll_steps
            if (s.get("state_digest") or {}).get("scroll_y_after") is not None)
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
        # --- metrics added 2026-08-02. Each answers a question the original eighteen
        # could not, and two of them exist to test markings the profile itself makes.
        #
        # visibility_gap: the page moved and the channel did not show it. This is the one
        #   quantity that is directly about what an observation channel can see, and it was
        #   absent from a profile whose whole subject is observation channels.
        # cap_hit: the episode exhausted its 30-step budget rather than terminating. Step
        #   count is the best escalation signal in the cascade experiment; this asks whether
        #   what that signal detects is "stuck".
        # url_revisit: the agent returned to a URL it had already loaded this episode. The
        #   existing search_loop_rate is one special case of going in circles; this is the
        #   general one.
        # noop_inert: a no-op whose action SUCCEEDED. Measured, action failure implies no
        #   change, so no_change_rate decomposes exactly into action failure plus this
        #   residual, and the profile's "no-op is downstream of action failure" marking
        #   applies only to the first term.
        # {click,type}_fail_rate: failures within one action type. Coordinate addressing
        #   should hurt spatially-targeted actions specifically; a uniform excess would mean
        #   the "no element-identity guarantee" story is not what drives the total.
        vis_denom = sum(1 for s in steps if s.get("page_changed") is True)
        vis_gap = sum(1 for s in steps if s.get("page_changed") is True
                      and s.get("agent_visible_changed") is False)
        inert = sum(1 for s in steps if s.get("page_changed") is False
                    and s.get("action_success") is True)
        seen: set[str] = set()
        revisit = 0
        for s in steps:
            after = ((s.get("state_digest") or {}).get("url_after") or "")
            if after:
                if after in seen:
                    revisit += 1
                seen.add(after)
        by_type: dict[str, list[int]] = {}
        for s, a in zip(steps, acts):
            if a:
                by_type.setdefault(a, []).append(1 if s.get("action_success") is False else 0)
        cap_hit = 1.0 if n >= MAX_STEPS else 0.0

        per_task[tid] = {
            "visibility_gap_rate": (vis_gap / vis_denom) if vis_denom else 0.0,
            "cap_hit_rate": cap_hit,
            "url_revisit_rate": revisit / n,
            "noop_inert_rate": inert / n,
            "click_fail_rate": (sum(by_type["click"]) / len(by_type["click"])
                                if by_type.get("click") else 0.0),
            "type_fail_rate": (sum(by_type["type"]) / len(by_type["type"])
                               if by_type.get("type") else 0.0),
            "n_steps": float(n),
            "click_frac": acts.count("click") / n,
            "type_frac": acts.count("type") / n,
            "scroll_frac": acts.count("scroll") / n,
            "search_loop_rate": 1.0 if search_steps >= 2 else 0.0,
            "parse_fail_rate": parse_fail / n,
            "action_fail_rate": act_fail / n,
            "no_change_rate": no_change / n,
            "scroll_inert_rate": (scroll_inert / scroll_denom) if scroll_denom else 0.0,
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
            "_scroll_inert": float(scroll_inert), "_scroll_denom": float(scroll_denom),
            "_loc_fb": float(loc_fb), "_repeats": float(repeats),
            "_repeat_denom": float(n - 1) if n > 1 else 0.0,
            "_vis_gap": float(vis_gap), "_vis_denom": float(vis_denom),
            "_inert": float(inert), "_revisit": float(revisit),
            "_cap_hit": cap_hit, "_episode": 1.0,
            "_click_fail": float(sum(by_type.get("click", []))),
            "_click_denom": float(len(by_type.get("click", []))),
            "_type_fail": float(sum(by_type.get("type", []))),
            "_type_denom": float(len(by_type.get("type", []))),
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
    "scroll_inert_rate": ("_scroll_inert", "_scroll_denom"),
    "locator_fallback_rate": ("_loc_fb", "_n_steps"),
    "action_repeat_frac": ("_repeats", "_repeat_denom"),
    "visibility_gap_rate": ("_vis_gap", "_vis_denom"),
    "noop_inert_rate": ("_inert", "_n_steps"),
    "url_revisit_rate": ("_revisit", "_n_steps"),
    "cap_hit_rate": ("_cap_hit", "_episode"),
    "click_fail_rate": ("_click_fail", "_click_denom"),
    "type_fail_rate": ("_type_fail", "_type_denom"),
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
        steps_by_mode[m], skipped_by_mode[m] = steps_layer(baseline, site, m, spec=spec)

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
            # total_latency_canonical_ms = minus_retry − busy_wait − recovered_screenshot
            # (B-1402 / B-1669 / B-1780). types.py:446 says the two are meant to be reported
            # side by side; only the raw one ever was. It matters on the API-served arm, where
            # the correction is 11% on B0/reddit P-text and P-prompt against 2-4% on DOM/SoM —
            # uneven enough across modes to reorder them. (§G1 unconsumed-field sweep, 08-02)
            "mean_latency_canonical_s": statistics.fmean(
                [_num(r.get("total_latency_canonical_ms", r.get("total_latency_ms"))) / 1000.0
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
             "a unanimous row.")
    L.append("")
    L.append("| dim | metric | highest | in | lowest | in | tie | top÷2nd (min–max) | unanimous |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for key, r in payload["consistency"].items():
        ratio = "—"
        if r["top_vs_second_ratio_min"] is not None:
            ratio = (f"{r['top_vs_second_ratio_min']:.2f}–"
                     f"{r['top_vs_second_ratio_max']:.2f}×")
        # The denominator is derived, not literal: this said "6/6" until 2026-08-03 and kept
        # saying it after the seventh and eighth cells landed, contradicting the count column
        # two cells to its left in the same row. Anything a reader compares must move together.
        mark = ""
        n = r["n_cells"]
        if r["unanimous_high"]:
            mark = f"**high: {r['highest_mode']} {n}/{n}**"
        elif r["unanimous_low"]:
            mark = f"**low: {r['lowest_mode']} {n}/{n}**"
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
        L.append("Why each ◆ row is architecturally downstream. **These are author-written "
                 "causal assertions, and each one deletes a whole row of evidence, so each "
                 "needs its own support.** All three were tested on 2026-08-02 and none "
                 "survived as originally written; the entries below are the rewritten "
                 "versions and carry the test that forced the rewrite:")
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
    # The three registries below are hand-written adjudications, recorded when the grid had six
    # cells. Their embedded "n/6" counts are frozen prose, unlike every table in this document,
    # which is derived. Say so once rather than letting a reader compare a stale count against a
    # live one two lines away — that mismatch was a real defect elsewhere (2026-08-03).
    n_cells_now = next(iter(payload["consistency"].values()))["n_cells"]
    if n_cells_now != 6:
        L.append(f"⚠️ **The `n/6` counts inside the adjudications above and below are frozen "
                 f"prose from the six-cell grid; this run has {n_cells_now} cells.** They record "
                 "*why a metric was flagged*, not a current tally — the tables in this document "
                 "are the live counts. The flags themselves were not re-adjudicated.")
        L.append("")
    if UNADJUDICATED:
        L.append("Metrics added 2026-08-02 and **not yet adjudicated** either way. Absence "
                 "from the ⚙️ and ◆ lists above means *nobody has ruled on this*, not "
                 "*verified clean*, and an unflagged unanimous row must not be read as an "
                 "endorsed behavioural finding:")
        L.append("")
        for k, why in UNADJUDICATED.items():
            lbl = payload["consistency"].get(k, {}).get("label", k)
            L.append(f"- `{lbl}` — {why}.")
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
        n = r["n_cells"]
        return f"   - `{r['label']}` ({r['dimension']}): **{mode}** {side} in {n}/{n}{ratio}"

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
    ap.add_argument("--with-wa", action="store_true",
                    help="append every WebArena-reddit cell (B1, then B0 — landed 2026-08-03) "
                         "after the six VWA cells. Writes to *_with_wa.* so the six-cell grid "
                         "the paper cites is never overwritten: adding cells changes every "
                         "consistency denominator from /6 to /8")
    a = ap.parse_args()
    if a.with_wa:
        a.out = a.out.with_name(a.out.stem + "_with_wa" + a.out.suffix)
        a.json_out = a.json_out.with_name(a.json_out.stem + "_with_wa" + a.json_out.suffix)

    cells = [profile_cell(spec) for spec in CELLS]
    if a.with_wa:
        for wb in WA_BASELINES:
            cells.append(profile_cell(wa_spec(wb)))
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
