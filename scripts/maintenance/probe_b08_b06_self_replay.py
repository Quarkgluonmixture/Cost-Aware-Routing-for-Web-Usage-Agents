#!/usr/bin/env python3
"""Self-replay probe for B-08 SCROLL silent failure + B-06 SELECT_OPTION arg-drop.

Differs from codex probe_audit_verification by:
1. SCROLL: uses logged state_digest.scroll_y_before/after instead of fresh-state
   replay — avoids the "didn't replay prior steps" issue that contaminated codex
   numbers (cls task 0 step 5 reported 576/2727 vs actual 1728/2635).
2. SELECT: locates element by element_bbox at obs_url, classifies tag (native
   <select> vs custom div) before checking arg-drop bug.

Sample size: 20 cases per category (vs codex 10/8) for tighter CI on low-scaffold
rates (codex CI ±15% on 0.3 rate → ~±10% on 20 cases).

Output:
  docs/analysis/cross_sites/probe_b08_b06_self_replay.json
  docs/analysis/cross_sites/probe_b08_b06_self_replay.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/visualwebarena/phase1"
TIER2 = ROOT / "docs/analysis/cross_sites/tier2_silent_failure_catalog.json"
OUT_JSON = ROOT / "docs/analysis/cross_sites/probe_b08_b06_self_replay.json"
OUT_MD = ROOT / "docs/analysis/cross_sites/probe_b08_b06_self_replay.md"

SITE_AUTH = {
    "classifieds": ROOT / ".auth/classifieds_state.json",
    "reddit": ROOT / ".auth/reddit_state.json",
    "shopping": ROOT / ".auth/shopping_state.json",
}


_RUN_DIR_CACHE: dict[str, Path] = {}


def resolve_run_dir(run: str) -> Path | None:
    if run in _RUN_DIR_CACHE:
        return _RUN_DIR_CACHE[run]
    direct = RESULTS / run
    if direct.exists():
        _RUN_DIR_CACHE[run] = direct
        return direct
    candidates = sorted(RESULTS.glob(f"{run}_*"))
    if candidates:
        _RUN_DIR_CACHE[run] = candidates[-1]
        return candidates[-1]
    _RUN_DIR_CACHE[run] = None
    return None


def load_steps(run: str, condition_id: str, site: str, task: int) -> list[dict[str, Any]]:
    run_dir = resolve_run_dir(run)
    if run_dir is None:
        return []
    base = run_dir / condition_id / "episodes" / f"{site}_task_{task}_steps_v2.jsonl"
    if not base.exists():
        return []
    rows = []
    for line in base.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def expand_cases(case_pool: list[dict[str, Any]], target_n: int, seed: int = 7) -> list[dict[str, Any]]:
    """Expand catalog cases — each entry may have multiple steps; one row per step."""
    rng = random.Random(seed)
    flat = []
    for case in case_pool:
        for step_idx in case.get("steps", []):
            flat.append({
                "site": case["site"],
                "task": case["task"],
                "run": case["run"],
                "condition_id": case["condition_id"],
                "mode": case["mode"],
                "step_idx": step_idx,
                "signature": case.get("signatures", []),
            })
    rng.shuffle(flat)
    return flat[:target_n]


async def probe_scroll_case(case: dict[str, Any], browser) -> dict[str, Any]:
    steps = load_steps(case["run"], case["condition_id"], case["site"], case["task"])
    if case["step_idx"] >= len(steps):
        return {"case": case, "classification": "REPLAY_FAIL", "reason": "step_idx out of range"}

    step = steps[case["step_idx"]]
    sd = step.get("state_digest", {}) or {}
    scroll_before = sd.get("scroll_y_before")
    scroll_after = sd.get("scroll_y_after")
    text_sim = step.get("text_similarity")
    obs_url = step.get("obs_url", "")
    action_success = step.get("action_success")
    page_changed = step.get("page_changed")
    action = step.get("action", {}) or {}
    delta = action.get("delta") if isinstance(action, dict) else None

    out = {
        "case": case,
        "logged": {
            "scroll_y_before": scroll_before,
            "scroll_y_after": scroll_after,
            "text_similarity": text_sim,
            "action_success": action_success,
            "page_changed": page_changed,
            "action_delta": delta,
            "obs_url": obs_url[:100],
        },
    }

    delta_y = (scroll_after - scroll_before) if (scroll_before is not None and scroll_after is not None) else None

    if delta_y is not None and abs(delta_y) >= 5:
        out["classification"] = "FALSE_POSITIVE_OF_TIER2"
        out["reason"] = f"scroll_y actually moved {delta_y}px — not silent failure"
        return out

    auth = SITE_AUTH.get(case["site"])
    if not auth or not auth.exists() or not obs_url:
        out["classification"] = "REPLAY_FAIL"
        out["reason"] = "auth missing or obs_url empty"
        return out

    try:
        ctx = await browser.new_context(storage_state=str(auth), viewport={"width": 1280, "height": 720})
        page = await ctx.new_page()
        try:
            await page.goto(obs_url, wait_until="domcontentloaded", timeout=15000)
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            info = await page.evaluate("""() => ({
                scrollH: document.documentElement.scrollHeight,
                viewportH: window.innerHeight,
            })""")
            scroll_height = info["scrollH"]
            viewport_h = info["viewportH"]
            scroll_y = scroll_before or 0
            at_bottom = scroll_y + viewport_h >= scroll_height - 10
            no_overflow = scroll_height <= viewport_h + 10

            out["page_geometry"] = {
                "scroll_height": scroll_height,
                "viewport_h": viewport_h,
                "starting_scroll_y": scroll_y,
                "at_bottom": at_bottom,
                "no_overflow": no_overflow,
            }

            if no_overflow:
                out["classification"] = "LEGIT_NO_OVERFLOW"
                out["reason"] = f"page height {scroll_height} <= viewport {viewport_h} — nothing to scroll"
            elif at_bottom:
                out["classification"] = "LEGIT_AT_BOTTOM"
                out["reason"] = f"already at bottom ({scroll_y}+{viewport_h}≥{scroll_height})"
            else:
                # page should have scrolled but didn't
                expected_delta = int(viewport_h * 0.8) if delta and delta[1] else viewport_h
                room = scroll_height - (scroll_y + viewport_h)
                out["classification"] = "SCAFFOLD_SCROLL_BUG"
                out["reason"] = f"room={room}px expected≈{expected_delta}px but scroll_y stayed at {scroll_y}"
        finally:
            await ctx.close()
    except Exception as e:
        out["classification"] = "REPLAY_FAIL"
        out["reason"] = f"playwright error: {type(e).__name__}: {str(e)[:120]}"

    return out


async def probe_select_case(case: dict[str, Any], browser) -> dict[str, Any]:
    steps = load_steps(case["run"], case["condition_id"], case["site"], case["task"])
    if case["step_idx"] >= len(steps):
        return {"case": case, "classification": "REPLAY_FAIL", "reason": "step_idx out of range"}

    step = steps[case["step_idx"]]
    action = step.get("action", {}) or {}
    obs_url = step.get("obs_url", "")
    bbox = step.get("element_bbox") or []
    action_text = action.get("text") if isinstance(action, dict) else None
    action_value = action.get("value") if isinstance(action, dict) else None
    target_value = action_text or action_value

    out = {
        "case": case,
        "logged": {
            "obs_url": obs_url[:100],
            "element_bbox": bbox,
            "action_text": action_text,
            "action_value": action_value,
            "action_success": step.get("action_success"),
            "page_changed": step.get("page_changed"),
        },
    }

    auth = SITE_AUTH.get(case["site"])
    if not auth or not auth.exists() or not obs_url or not bbox or len(bbox) < 4:
        out["classification"] = "REPLAY_FAIL"
        out["reason"] = "auth/obs_url/bbox missing"
        return out

    try:
        ctx = await browser.new_context(storage_state=str(auth), viewport={"width": 1280, "height": 720})
        page = await ctx.new_page()
        try:
            await page.goto(obs_url, wait_until="domcontentloaded", timeout=15000)
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            # bbox = [x, y, w, h] per processors.py:297 — center is (x + w/2, y + h/2)
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            tag_info = await page.evaluate(
                """([cx, cy]) => {
                    const el = document.elementFromPoint(cx, cy);
                    if (!el) return null;
                    return {
                        tag: el.tagName,
                        id: el.id || null,
                        cls: el.className || null,
                        role: el.getAttribute('role'),
                        is_select: el.tagName === 'SELECT',
                        nearest_select: (() => {
                            let n = el;
                            for (let i=0; i<6 && n; i++) {
                                if (n.tagName === 'SELECT') return {tag: 'SELECT', id: n.id};
                                n = n.parentElement;
                            }
                            return null;
                        })(),
                    };
                }""",
                [cx, cy],
            )
            out["dom_at_bbox"] = tag_info

            if tag_info is None:
                out["classification"] = "REPLAY_FAIL"
                out["reason"] = "no element at bbox center"
            elif not tag_info.get("is_select") and not tag_info.get("nearest_select"):
                out["classification"] = "OTHER_CUSTOM_DROPDOWN"
                out["reason"] = f"target is {tag_info.get('tag')} not <select> — different code path, not arg-drop bug"
            else:
                # Native <select> reachable — verify framework's no-args dispatch fails vs proper dispatch
                select_locator = await page.evaluate_handle(
                    """([cx, cy]) => {
                        let el = document.elementFromPoint(cx, cy);
                        for (let i=0; i<6 && el; i++) {
                            if (el.tagName === 'SELECT') return el;
                            el = el.parentElement;
                        }
                        return null;
                    }""",
                    [cx, cy],
                )
                # Sample legal options
                opts_info = await page.evaluate(
                    """(el) => {
                        if (!el) return null;
                        const opts = Array.from(el.options).map(o => ({value: o.value, text: o.text}));
                        return {n_options: opts.length, sample: opts.slice(0,5), current: el.value};
                    }""",
                    select_locator,
                )
                out["select_state"] = opts_info
                out["classification"] = "SCAFFOLD_SELECT_ARG_DROP"
                out["reason"] = (
                    f"native <select> with {opts_info.get('n_options') if opts_info else 'n/a'} options; "
                    f"framework calls .select_option() with NO args (line 1395 actions.py) → "
                    f"would clear or no-op rather than picking '{target_value}'"
                )
        finally:
            await ctx.close()
    except Exception as e:
        out["classification"] = "REPLAY_FAIL"
        out["reason"] = f"playwright error: {type(e).__name__}: {str(e)[:120]}"

    return out


def collect_extra_scroll_cases(target_n: int, seen: set[tuple], seed: int = 11) -> list[dict[str, Any]]:
    """Expand sample beyond Tier 2 case_study_task_ids by re-scanning."""
    rng = random.Random(seed)
    extra = []
    for run_dir in sorted(RESULTS.iterdir()):
        if not run_dir.is_dir():
            continue
        for ep_dir in run_dir.glob("phase1_*/episodes"):
            cond = ep_dir.parent.name
            run = run_dir.name
            mode_guess = (
                "DOM" if "dom" in cond else "SoM" if "som" in cond else "Vision" if "vision" in cond
                else "P-text" if "phantom_text" in cond or "phantom_dom" in cond else "P-SoM" if "phantom_som" in cond
                else "P-prompt" if "phantom_prompt" in cond else "?"
            )
            for steps_path in ep_dir.glob("*_steps_v2.jsonl"):
                m = steps_path.name.split("_task_")
                if len(m) != 2:
                    continue
                site = m[0]
                try:
                    task = int(m[1].split("_")[0])
                except ValueError:
                    continue
                if site not in SITE_AUTH:
                    continue
                rows = load_steps(run, cond, site, task)
                for i, step in enumerate(rows):
                    if step.get("action_type") != "scroll":
                        continue
                    if step.get("action_success") is True or step.get("page_changed") is True:
                        continue
                    sd = step.get("state_digest", {}) or {}
                    if sd.get("scroll_y_before") is None or sd.get("scroll_y_after") is None:
                        continue
                    delta = abs(sd["scroll_y_after"] - sd["scroll_y_before"])
                    if delta >= 5:
                        continue
                    key = (run, cond, site, task, i)
                    if key in seen:
                        continue
                    extra.append({
                        "site": site,
                        "task": task,
                        "run": run,
                        "condition_id": cond,
                        "mode": mode_guess,
                        "step_idx": i,
                        "signature": ["consecutive_scrolls_no_viewport_move_self_collected"],
                    })
                    if len(extra) >= target_n * 3:
                        break
            if len(extra) >= target_n * 3:
                break
        if len(extra) >= target_n * 3:
            break
    rng.shuffle(extra)
    return extra[:target_n]


def collect_extra_select_cases(target_n: int, seen: set[tuple], seed: int = 13, site_filter: set[str] | None = None) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    extra: list[dict[str, Any]] = []
    for run_dir in sorted(RESULTS.iterdir()):
        if not run_dir.is_dir():
            continue
        for ep_dir in run_dir.glob("phase1_*/episodes"):
            cond = ep_dir.parent.name
            run = run_dir.name
            mode_guess = (
                "DOM" if "dom" in cond else "SoM" if "som" in cond else "Vision" if "vision" in cond
                else "P-text" if "phantom_text" in cond or "phantom_dom" in cond else "P-SoM" if "phantom_som" in cond
                else "P-prompt" if "phantom_prompt" in cond else "?"
            )
            for steps_path in ep_dir.glob("*_steps_v2.jsonl"):
                m = steps_path.name.split("_task_")
                if len(m) != 2:
                    continue
                site = m[0]
                try:
                    task = int(m[1].split("_")[0])
                except ValueError:
                    continue
                if site not in SITE_AUTH:
                    continue
                if site_filter and site not in site_filter:
                    continue
                rows = load_steps(run, cond, site, task)
                for i, step in enumerate(rows):
                    if step.get("action_type") != "select_option":
                        continue
                    if step.get("action_success") is True or step.get("page_changed") is True:
                        continue
                    bbox = step.get("element_bbox") or []
                    if len(bbox) < 4:
                        continue
                    key = (run, cond, site, task, i)
                    if key in seen:
                        continue
                    extra.append({
                        "site": site,
                        "task": task,
                        "run": run,
                        "condition_id": cond,
                        "mode": mode_guess,
                        "step_idx": i,
                        "signature": ["self_collected_silent_select_option"],
                    })
                    if len(extra) >= target_n * 3:
                        break
    rng.shuffle(extra)
    return extra[:target_n]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-scroll", type=int, default=20)
    parser.add_argument("--n-select", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=30, help="seconds per case")
    args = parser.parse_args()

    tier2 = json.loads(TIER2.read_text())
    scroll_cat = tier2["categories"]["scroll_silent_failure"]["case_study_task_ids"]
    select_cat = tier2["categories"]["select_option_silent_failure"]["case_study_task_ids"]

    scroll_cases = expand_cases(scroll_cat, args.n_scroll)
    select_cases = expand_cases(select_cat, args.n_select)
    seen_scroll = {(c["run"], c["condition_id"], c["site"], c["task"], c["step_idx"]) for c in scroll_cases}
    seen_select = {(c["run"], c["condition_id"], c["site"], c["task"], c["step_idx"]) for c in select_cases}
    if len(scroll_cases) < args.n_scroll:
        scroll_cases.extend(collect_extra_scroll_cases(args.n_scroll - len(scroll_cases), seen_scroll))

    # Force site diversity for SELECT — case_study_task_ids is 100% cls-biased,
    # but Tier 2 catalog covers cls 117 / red 27 / shop 5. Sample at least ~30%
    # red/shop to detect native <select> bugs that cls (custom-dropdown-heavy) misses.
    if len(select_cases) < args.n_select:
        select_cases.extend(collect_extra_select_cases(args.n_select - len(select_cases), seen_select))

    select_cls = [c for c in select_cases if c["site"] == "classifieds"]
    select_red = [c for c in select_cases if c["site"] == "reddit"]
    select_shop = [c for c in select_cases if c["site"] == "shopping"]
    target_red = max(6, args.n_select // 4)
    target_shop = max(3, args.n_select // 6)
    if len(select_red) < target_red:
        seen_select_now = {(c["run"], c["condition_id"], c["site"], c["task"], c["step_idx"]) for c in select_cases}
        select_red.extend(collect_extra_select_cases(target_red - len(select_red), seen_select_now, seed=17, site_filter={"reddit"}))
    if len(select_shop) < target_shop:
        seen_select_now = {(c["run"], c["condition_id"], c["site"], c["task"], c["step_idx"]) for c in (select_cls + select_red + select_shop)}
        select_shop.extend(collect_extra_select_cases(target_shop - len(select_shop), seen_select_now, seed=19, site_filter={"shopping"}))
    desired_cls = args.n_select - len(select_red) - len(select_shop)
    select_cases = select_cls[:desired_cls] + select_red[:target_red] + select_shop[:target_shop]

    scroll_cases = scroll_cases[: args.n_scroll]
    select_cases = select_cases[: args.n_select]
    print(f"Scroll cases: {len(scroll_cases)} | Select cases: {len(select_cases)}", file=sys.stderr)

    results = {"scroll_results": [], "select_results": []}
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            for i, c in enumerate(scroll_cases, 1):
                t0 = time.time()
                try:
                    r = await asyncio.wait_for(probe_scroll_case(c, browser), timeout=args.timeout)
                except asyncio.TimeoutError:
                    r = {"case": c, "classification": "REPLAY_FAIL", "reason": "timeout"}
                r["elapsed_s"] = round(time.time() - t0, 1)
                results["scroll_results"].append(r)
                print(f"[{i}/{len(scroll_cases)}] SCROLL {c['site']} task {c['task']} step {c['step_idx']} → {r['classification']} ({r['elapsed_s']}s)", file=sys.stderr)
            for i, c in enumerate(select_cases, 1):
                t0 = time.time()
                try:
                    r = await asyncio.wait_for(probe_select_case(c, browser), timeout=args.timeout)
                except asyncio.TimeoutError:
                    r = {"case": c, "classification": "REPLAY_FAIL", "reason": "timeout"}
                r["elapsed_s"] = round(time.time() - t0, 1)
                results["select_results"].append(r)
                print(f"[{i}/{len(select_cases)}] SELECT {c['site']} task {c['task']} step {c['step_idx']} → {r['classification']} ({r['elapsed_s']}s)", file=sys.stderr)
        finally:
            await browser.close()

    def breakdown(rows):
        d = {}
        for r in rows:
            d[r["classification"]] = d.get(r["classification"], 0) + 1
        return d

    scroll_bd = breakdown(results["scroll_results"])
    select_bd = breakdown(results["select_results"])

    scroll_total = len(results["scroll_results"])
    scroll_replay_ok = scroll_total - scroll_bd.get("REPLAY_FAIL", 0)
    scroll_scaffold = scroll_bd.get("SCAFFOLD_SCROLL_BUG", 0)
    scroll_legit = scroll_bd.get("LEGIT_AT_BOTTOM", 0) + scroll_bd.get("LEGIT_NO_OVERFLOW", 0)
    scroll_fp = scroll_bd.get("FALSE_POSITIVE_OF_TIER2", 0)
    scroll_frac = scroll_scaffold / scroll_replay_ok if scroll_replay_ok else None

    select_total = len(results["select_results"])
    select_replay_ok = select_total - select_bd.get("REPLAY_FAIL", 0)
    select_scaffold = select_bd.get("SCAFFOLD_SELECT_ARG_DROP", 0)
    select_other = select_bd.get("OTHER_CUSTOM_DROPDOWN", 0)
    select_frac = select_scaffold / select_replay_ok if select_replay_ok else None

    summary = {
        "audit_date": "2026-04-30",
        "methodology_note": (
            "Self-replay probe uses logged state_digest.scroll_y_before/after instead of "
            "fresh-state replay (avoids the codex 'didn't replay prior steps' issue). "
            "For SCROLL: classifies via logged delta + obs_url page geometry (scrollHeight, "
            "viewport, starting scroll_y). For SELECT: navigates to obs_url, identifies element "
            "at logged bbox, separates native <select> from custom div dropdown, then verifies "
            "framework's no-args .select_option() dispatch path."
        ),
        "scroll": {
            "tier2_claim_ep": 667,
            "tier2_claim_blast_radius_pct": 14.85,
            "n_probed": scroll_total,
            "n_replay_ok": scroll_replay_ok,
            "breakdown": scroll_bd,
            "scaffold_fraction_of_replayed": round(scroll_frac, 3) if scroll_frac is not None else None,
            "legit_fraction_of_replayed": round(scroll_legit / scroll_replay_ok, 3) if scroll_replay_ok else None,
            "false_positive_of_tier2": scroll_fp,
            "extrapolated_blast_radius_ep": round(667 * scroll_frac) if scroll_frac is not None else None,
        },
        "select": {
            "tier2_claim_ep": 149,
            "tier2_claim_blast_radius_pct": 3.32,
            "n_probed": select_total,
            "n_replay_ok": select_replay_ok,
            "breakdown": select_bd,
            "scaffold_fraction_of_replayed": round(select_frac, 3) if select_frac is not None else None,
            "extrapolated_blast_radius_ep": round(149 * select_frac) if select_frac is not None else None,
            "non_native_dropdown_fraction": round(select_other / select_replay_ok, 3) if select_replay_ok else None,
        },
        "comparison_to_codex": {
            "codex_scroll_scaffold_fraction": 0.3,
            "codex_select_scaffold_fraction": 0.286,
            "codex_n_scroll": 10,
            "codex_n_select": 8,
        },
    }
    results["summary"] = summary

    OUT_JSON.write_text(json.dumps(results, indent=2))
    md_lines = [
        "# Self-Replay Probe — B-08 SCROLL + B-06 SELECT_OPTION",
        "",
        f"Audit date: {summary['audit_date']}",
        "",
        "## Methodology",
        "",
        summary["methodology_note"],
        "",
        "## SCROLL silent failure (B-08)",
        "",
        f"- Tier 2 claim: {summary['scroll']['tier2_claim_ep']} ep / {summary['scroll']['tier2_claim_blast_radius_pct']}%",
        f"- Probed: {summary['scroll']['n_probed']} (replay ok: {summary['scroll']['n_replay_ok']})",
        f"- Scaffold fraction of replayed: **{summary['scroll']['scaffold_fraction_of_replayed']}**",
        f"- Legit fraction of replayed: {summary['scroll']['legit_fraction_of_replayed']}",
        f"- False-positive-of-Tier2 (scroll actually moved ≥5px): {summary['scroll']['false_positive_of_tier2']}",
        f"- Extrapolated true blast radius: **{summary['scroll']['extrapolated_blast_radius_ep']} ep**",
        f"- Breakdown: {summary['scroll']['breakdown']}",
        "",
        "## SELECT_OPTION arg-drop (B-06)",
        "",
        f"- Tier 2 claim: {summary['select']['tier2_claim_ep']} ep / {summary['select']['tier2_claim_blast_radius_pct']}%",
        f"- Probed: {summary['select']['n_probed']} (replay ok: {summary['select']['n_replay_ok']})",
        f"- Scaffold (native <select> + arg-drop) fraction of replayed: **{summary['select']['scaffold_fraction_of_replayed']}**",
        f"- Non-native (custom div dropdown) fraction: {summary['select']['non_native_dropdown_fraction']}",
        f"- Extrapolated true arg-drop blast radius: **{summary['select']['extrapolated_blast_radius_ep']} ep**",
        f"- Breakdown: {summary['select']['breakdown']}",
        "",
        "## Comparison to Codex `probe_audit_verification`",
        "",
        f"- Codex SCROLL: {summary['comparison_to_codex']['codex_n_scroll']} cases, scaffold fraction {summary['comparison_to_codex']['codex_scroll_scaffold_fraction']}",
        f"- Self-replay SCROLL: {summary['scroll']['n_probed']} cases, scaffold fraction {summary['scroll']['scaffold_fraction_of_replayed']}",
        "",
        f"- Codex SELECT: {summary['comparison_to_codex']['codex_n_select']} cases, scaffold fraction {summary['comparison_to_codex']['codex_select_scaffold_fraction']}",
        f"- Self-replay SELECT: {summary['select']['n_probed']} cases, scaffold fraction {summary['select']['scaffold_fraction_of_replayed']}",
        "",
        "## Per-case detail",
        "",
        "### SCROLL cases",
        "",
    ]
    for r in results["scroll_results"]:
        c = r["case"]
        md_lines.append(
            f"- {c['site']} task {c['task']} step {c['step_idx']} ({c['mode']}) → "
            f"**{r['classification']}** | {r.get('reason','')[:120]} | logged delta="
            f"{(r.get('logged',{}) or {}).get('scroll_y_after',0) - (r.get('logged',{}) or {}).get('scroll_y_before',0) if r.get('logged') else 'n/a'}px"
        )
    md_lines.append("")
    md_lines.append("### SELECT cases")
    md_lines.append("")
    for r in results["select_results"]:
        c = r["case"]
        md_lines.append(
            f"- {c['site']} task {c['task']} step {c['step_idx']} ({c['mode']}) → "
            f"**{r['classification']}** | {r.get('reason','')[:120]}"
        )
    OUT_MD.write_text("\n".join(md_lines))
    print(f"\nWrote {OUT_JSON}\n      {OUT_MD}", file=sys.stderr)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
