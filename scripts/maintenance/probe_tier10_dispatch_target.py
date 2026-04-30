#!/usr/bin/env python3
"""Tier 10 audit: dispatch-effective-target mapping across ALL action types.

Existing 5-tier audit only covered Tier 1 "AXTree element_id assignment" (Q2)
which checks how AX node IDs are assigned, NOT how union_bound maps to physical
click targets. The B-33 discovery (AXTREE_MAPPING_ERROR 55.9% of click-loops via
artifact) showed this gap. This probe extends to all dispatch action types
(click/type/hover/select_option/clear/check) to find similar mapping bugs.

Methodology:
1. For each action type, sample 10-15 FAILED steps (action_success=False) from
   paper-grade data, site-diverse.
2. Navigate to obs_url, find element at logged bbox center using CORRECT formula
   (x + w/2, y + h/2 per processors.py:297).
3. Compare element_at_center.tag against expected target for that action:
   - click/hover: should hit <a>/<button>/[role=link]/[role=button]
   - type: should hit <input>/<textarea>/[contenteditable]
   - select_option: should hit <select> or be inside a select-group
   - clear: should hit <input>/<textarea>
4. Categorize sub-patterns of mapping mismatch:
   - LISTING_CARD_CHILD: hit span.date/.location/.price/.desc inside listing card
   - HEADING_ELEMENT: hit h1/h2/h3/h4 (agent thought heading was link)
   - BUTTON_LABEL_SPAN: hit span inside button (subscribe-button child)
   - INLINE_GAP: hit <li>/<div> parent of inline-multi-line <a> (§106 classic)
   - INPUT_AT_CENTER_NO_FOLLOWUP: hit input/textarea but action wasn't type
   - ICON_OR_IMG_INSIDE: hit <svg>/<img>/<i> child
   - ON_TARGET: bbox center actually matches expected target
   - OTHER

Output: docs/analysis/cross_sites/probe_tier10_dispatch_target.{json,md}
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
OUT_JSON = ROOT / "docs/analysis/cross_sites/probe_tier10_dispatch_target.json"
OUT_MD = ROOT / "docs/analysis/cross_sites/probe_tier10_dispatch_target.md"

SITE_AUTH = {
    "classifieds": ROOT / ".auth/classifieds_state.json",
    "reddit": ROOT / ".auth/reddit_state.json",
    "shopping": ROOT / ".auth/shopping_state.json",
}

ACTION_TARGET_TAGS = {
    "click": {"A", "BUTTON", "INPUT"},  # input also clickable (button/submit/checkbox)
    "hover": {"A", "BUTTON", "DIV", "SPAN"},  # hover targets are looser
    "type": {"INPUT", "TEXTAREA"},
    "clear": {"INPUT", "TEXTAREA"},
    "select_option": {"SELECT"},
    "check": {"INPUT"},  # checkbox/radio
}

_RUN_DIR_CACHE: dict[str, Path | None] = {}


def resolve_run_dir(run: str) -> Path | None:
    if run in _RUN_DIR_CACHE:
        return _RUN_DIR_CACHE[run]
    direct = RESULTS / run
    if direct.exists():
        _RUN_DIR_CACHE[run] = direct
        return direct
    cands = sorted(RESULTS.glob(f"{run}_*"))
    _RUN_DIR_CACHE[run] = cands[-1] if cands else None
    return _RUN_DIR_CACHE[run]


def collect_cases_per_action(action_type: str, target_n: int, seed: int = 31) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_site: dict[str, list] = {"classifieds": [], "reddit": [], "shopping": []}
    target_each = max(2, target_n // 3)
    # Scan ALL runs/conditions/episodes; only break when EACH site has enough
    for run_dir in sorted(RESULTS.iterdir()):
        if not run_dir.is_dir():
            continue
        for ep_dir in run_dir.glob("phase1_*/episodes"):
            cond = ep_dir.parent.name
            run = run_dir.name
            mode = (
                "DOM" if "dom" in cond else "SoM" if "som" in cond else "Vision" if "vision" in cond
                else "P-text" if "phantom_text" in cond or "phantom_dom" in cond else "P-SoM" if "phantom_som" in cond
                else "P-prompt" if "phantom_prompt" in cond else "?"
            )
            for sp in ep_dir.glob("*_steps_v2.jsonl"):
                m = sp.name.split("_task_")
                if len(m) != 2:
                    continue
                site = m[0]
                if site not in SITE_AUTH:
                    continue
                # Skip site if already at quota
                if len(by_site[site]) >= target_each * 5:
                    continue
                try:
                    task = int(m[1].split("_")[0])
                except ValueError:
                    continue
                try:
                    rows = [json.loads(l) for l in sp.read_text().splitlines() if l.strip()]
                except Exception:
                    continue
                for i, step in enumerate(rows):
                    if step.get("action_type") != action_type:
                        continue
                    if step.get("action_success") is True:
                        continue
                    bbox = step.get("element_bbox") or []
                    if len(bbox) < 4:
                        continue
                    if not step.get("obs_url"):
                        continue
                    by_site[site].append({
                        "site": site, "task": task, "run": run, "condition_id": cond,
                        "mode": mode, "step_idx": i, "action_type": action_type,
                    })
            # Break only when all 3 sites have at least target_each * 3 (oversampled for site diversity)
            if all(len(by_site[s]) >= target_each * 3 for s in by_site):
                break
        if all(len(by_site[s]) >= target_each * 3 for s in by_site):
            break

    for s in by_site:
        rng.shuffle(by_site[s])
    selected = []
    for s in ["classifieds", "reddit", "shopping"]:
        selected.extend(by_site.get(s, [])[:target_each])
    # Fill any shortfall from any site
    if len(selected) < target_n:
        seen_keys = {(c["run"], c["condition_id"], c["site"], c["task"], c["step_idx"]) for c in selected}
        all_remaining = []
        for s in by_site.values():
            all_remaining.extend([c for c in s if (c["run"], c["condition_id"], c["site"], c["task"], c["step_idx"]) not in seen_keys])
        selected.extend(all_remaining[: target_n - len(selected)])
    return selected[:target_n]


def load_step(case: dict[str, Any]) -> dict[str, Any] | None:
    rd = resolve_run_dir(case["run"])
    if rd is None:
        return None
    p = rd / case["condition_id"] / "episodes" / f"{case['site']}_task_{case['task']}_steps_v2.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if case["step_idx"] >= len(rows):
        return None
    return rows[case["step_idx"]]


def categorize_mismatch(elem: dict[str, Any], action_type: str) -> str:
    """Sub-pattern categorization given element_at_bbox_center info."""
    if elem is None:
        return "REPLAY_FAIL_NO_ELEMENT"
    tag = (elem.get("tag") or "").upper()
    cls = (elem.get("cls") or "").lower()

    # ON_TARGET first
    expected_tags = ACTION_TARGET_TAGS.get(action_type, set())
    if tag in expected_tags:
        # For type action, bbox hits input → could be either ON_TARGET or no-followup pattern
        if action_type == "click" and tag == "INPUT":
            # Click on input: technically on-target but agent likely meant search-no-type pattern
            # if input.type is search/text and no type follow-up
            return "INPUT_AT_CENTER_AGENT_PATTERN"
        return "ON_TARGET"

    # Off-target sub-patterns
    if tag in {"H1", "H2", "H3", "H4", "H5", "H6"}:
        return "HEADING_ELEMENT"
    if tag == "SPAN":
        if any(k in cls for k in ["date", "location", "price", "desc", "subtitle", "info"]):
            return "LISTING_CARD_CHILD_SPAN"
        if any(k in cls for k in ["button", "btn", "label"]):
            return "BUTTON_LABEL_SPAN"
        return "OTHER_SPAN"
    if tag in {"SVG", "IMG", "I", "PATH", "USE"}:
        return "ICON_OR_IMG_INSIDE"
    if tag in {"LI", "UL", "DIV", "MAIN", "ARTICLE", "SECTION", "P", "HEADER", "FOOTER"}:
        return "BLOCK_PARENT"
    if tag == "TEXTAREA" and action_type != "type":
        return "TEXTAREA_AT_CENTER_NO_FOLLOWUP"
    return "OTHER"


async def probe_one(case: dict[str, Any], browser) -> dict[str, Any]:
    step = load_step(case)
    if step is None:
        return {"case": case, "category": "REPLAY_FAIL", "reason": "step missing"}
    bbox = step.get("element_bbox") or []
    obs_url = step.get("obs_url", "")
    auth = SITE_AUTH.get(case["site"])
    if not auth or not auth.exists() or len(bbox) < 4:
        return {"case": case, "category": "REPLAY_FAIL", "reason": "auth/bbox"}

    out = {
        "case": case,
        "logged": {"bbox": bbox, "obs_url": obs_url[:100]},
    }
    try:
        ctx = await browser.new_context(storage_state=str(auth), viewport={"width": 1280, "height": 720})
        page = await ctx.new_page()
        try:
            await page.goto(obs_url, wait_until="domcontentloaded", timeout=15000)
            try:
                await page.wait_for_load_state("networkidle", timeout=4000)
            except Exception:
                pass
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            info = await page.evaluate(
                """([cx, cy]) => {
                    const el = document.elementFromPoint(cx, cy);
                    if (!el) return null;
                    const a_anc = el.closest('a');
                    const btn_anc = el.closest('button');
                    const inp_anc = el.closest('input,textarea,[contenteditable=true]');
                    return {
                        tag: el.tagName,
                        cls: el.className || null,
                        type: el.type || null,
                        role: el.getAttribute('role'),
                        nearest_a: a_anc ? {href: a_anc.href, text: (a_anc.innerText||'').slice(0,40)} : null,
                        nearest_button: btn_anc ? {text: (btn_anc.innerText||'').slice(0,40)} : null,
                        nearest_input: inp_anc ? {tag: inp_anc.tagName, type: inp_anc.type || null} : null,
                    };
                }""",
                [cx, cy],
            )
            out["dom_at_bbox"] = info
            out["category"] = categorize_mismatch(info, case["action_type"])
        finally:
            await ctx.close()
    except Exception as e:
        out["category"] = "REPLAY_FAIL"
        out["reason"] = f"{type(e).__name__}: {str(e)[:120]}"
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-action", type=int, default=12)
    ap.add_argument("--actions", nargs="+", default=["click", "type", "hover", "select_option", "clear"])
    args = ap.parse_args()

    all_cases: dict[str, list] = {}
    for action in args.actions:
        all_cases[action] = collect_cases_per_action(action, args.n_per_action)
        print(f"  {action}: {len(all_cases[action])} cases", file=sys.stderr)

    results: dict[str, list] = {a: [] for a in args.actions}
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            for action, cases in all_cases.items():
                for i, c in enumerate(cases, 1):
                    t0 = time.time()
                    try:
                        r = await asyncio.wait_for(probe_one(c, browser), timeout=25)
                    except asyncio.TimeoutError:
                        r = {"case": c, "category": "REPLAY_FAIL", "reason": "timeout"}
                    r["elapsed_s"] = round(time.time() - t0, 1)
                    results[action].append(r)
                    print(
                        f"[{action} {i}/{len(cases)}] {c['site']} task {c['task']} step {c['step_idx']} ({c['mode']}) → {r['category']}",
                        file=sys.stderr,
                    )
        finally:
            await browser.close()

    def breakdown(rows):
        d: dict[str, int] = {}
        for r in rows:
            d[r["category"]] = d.get(r["category"], 0) + 1
        return d

    summary = {
        "audit_date": "2026-04-30",
        "purpose": "Tier 10 dispatch-effective-target mapping audit across all action types.",
        "per_action": {},
    }
    for action, rows in results.items():
        bd = breakdown(rows)
        total = len(rows)
        replay_ok = total - bd.get("REPLAY_FAIL", 0) - bd.get("REPLAY_FAIL_NO_ELEMENT", 0)
        on_target = bd.get("ON_TARGET", 0)
        summary["per_action"][action] = {
            "n_total": total,
            "n_replay_ok": replay_ok,
            "breakdown": bd,
            "on_target_fraction_of_replayed": round(on_target / replay_ok, 3) if replay_ok else None,
            "off_target_fraction_of_replayed": round((replay_ok - on_target) / replay_ok, 3) if replay_ok else None,
        }
    results["summary"] = summary

    OUT_JSON.write_text(json.dumps(results, indent=2))

    md = [
        "# Tier 10 Dispatch-Effective-Target Audit",
        "",
        f"Audit date: {summary['audit_date']}",
        "",
        "## Purpose",
        summary["purpose"],
        "",
        "## Per-action mapping accuracy",
        "",
        "| Action | Probed | Replay OK | ON_TARGET | Off-target % | Top off-target patterns |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for action, s in summary["per_action"].items():
        bd = s["breakdown"]
        off = sorted(((k, v) for k, v in bd.items() if k != "ON_TARGET" and "REPLAY_FAIL" not in k), key=lambda x: -x[1])[:3]
        off_str = ", ".join(f"{k}={v}" for k, v in off)
        md.append(
            f"| {action} | {s['n_total']} | {s['n_replay_ok']} | {bd.get('ON_TARGET', 0)} | "
            f"{(s['off_target_fraction_of_replayed'] or 0) * 100:.1f}% | {off_str} |"
        )
    md.append("")
    md.append("## Per-case detail")
    md.append("")
    for action, rows in results.items():
        if action == "summary":
            continue
        md.append(f"### {action}")
        md.append("")
        for r in rows:
            c = r["case"]
            elem = r.get("dom_at_bbox") or {}
            md.append(
                f"- {c['site']} task {c['task']} step {c['step_idx']} ({c['mode']}) → "
                f"**{r['category']}** | hit `{elem.get('tag','?')}.{(elem.get('cls') or '')[:30]}` | "
                f"nearest_a={bool(elem.get('nearest_a'))} nearest_button={bool(elem.get('nearest_button'))} "
                f"nearest_input={bool(elem.get('nearest_input'))}"
            )
        md.append("")

    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_JSON}\n      {OUT_MD}", file=sys.stderr)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
