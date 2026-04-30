#!/usr/bin/env python3
"""Self-verify probe for B-01 TYPE 100% scaffold + B-13 NOT_A_BUG (action_fail_page_changed).

Both classifications come from codex `probe_audit_verification` and may suffer
from the same methodology bugs that contaminated SCROLL/SELECT (no faithful
prior-step replay + wrong bbox center formula `(x+w)/2,(y+h)/2`).

Methodology:
- B-01 TYPE: navigate to step's obs_url, find DOM element at logged bbox using
  CORRECT center formula (x + w/2, y + h/2). If element is INPUT/TEXTAREA/
  contenteditable, framework's center-click + Meta+A path would actually focus
  the right input — NOT a scaffold bug. If it's a non-editable, the agent's
  Meta+A goes to wrong target → scaffold confirmed.
- B-13 action_fail/page_changed: load Tier 4 I2 case study examples and check
  whether the logged page_changed=True is actually justified (URL/title/scroll
  delta from state_digest), independent of codex's REPLAY_FAIL bias.

Output:
  docs/analysis/cross_sites/probe_b01_b13_self_verify.{json,md}
"""
from __future__ import annotations

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
TIER4 = ROOT / "docs/analysis/cross_sites/tier4_invariant_audit.json"
OUT_JSON = ROOT / "docs/analysis/cross_sites/probe_b01_b13_self_verify.json"
OUT_MD = ROOT / "docs/analysis/cross_sites/probe_b01_b13_self_verify.md"

SITE_AUTH = {
    "classifieds": ROOT / ".auth/classifieds_state.json",
    "reddit": ROOT / ".auth/reddit_state.json",
    "shopping": ROOT / ".auth/shopping_state.json",
}

_RUN_DIR_CACHE: dict[str, Path | None] = {}


def resolve_run_dir(run: str) -> Path | None:
    if run in _RUN_DIR_CACHE:
        return _RUN_DIR_CACHE[run]
    direct = RESULTS / run
    if direct.exists():
        _RUN_DIR_CACHE[run] = direct
        return direct
    candidates = sorted(RESULTS.glob(f"{run}_*"))
    _RUN_DIR_CACHE[run] = candidates[-1] if candidates else None
    return _RUN_DIR_CACHE[run]


def load_steps(run: str, condition_id: str, site: str, task: int) -> list[dict[str, Any]]:
    rd = resolve_run_dir(run)
    if rd is None:
        return []
    p = rd / condition_id / "episodes" / f"{site}_task_{task}_steps_v2.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def collect_b01_cases(target_n: int, seed: int = 23) -> list[dict[str, Any]]:
    """Sample TYPE silent failure cases, site-diverse."""
    tier2 = json.loads(TIER2.read_text())
    pool: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    # First pass — Tier 2 case_study_task_ids
    for case in tier2["categories"]["type_silent_failure"]["case_study_task_ids"]:
        for step_idx in case.get("steps", []):
            key = (case["run"], case["condition_id"], case["site"], case["task"], step_idx)
            if key in seen:
                continue
            seen.add(key)
            pool.append({
                "site": case["site"], "task": case["task"], "run": case["run"],
                "condition_id": case["condition_id"], "mode": case.get("mode", "?"),
                "step_idx": step_idx,
                "signature": case.get("signatures", []),
            })

    # Second pass — re-scan for diversity
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
            for sp in ep_dir.glob("*_steps_v2.jsonl"):
                m = sp.name.split("_task_")
                if len(m) != 2:
                    continue
                site = m[0]
                if site not in SITE_AUTH:
                    continue
                try:
                    task = int(m[1].split("_")[0])
                except ValueError:
                    continue
                rows = load_steps(run, cond, site, task)
                for i, step in enumerate(rows):
                    if step.get("action_type") != "type":
                        continue
                    if step.get("action_success") is True:
                        continue
                    bbox = step.get("element_bbox") or []
                    if len(bbox) < 4:
                        continue
                    key = (run, cond, site, task, i)
                    if key in seen:
                        continue
                    seen.add(key)
                    pool.append({
                        "site": site, "task": task, "run": run, "condition_id": cond,
                        "mode": mode_guess, "step_idx": i,
                        "signature": ["self_collected_silent_type"],
                    })
                    if len(pool) >= target_n * 5:
                        break
            if len(pool) >= target_n * 5:
                break
        if len(pool) >= target_n * 5:
            break

    rng = random.Random(seed)
    rng.shuffle(pool)
    # Site-balanced: at least 3 from each site if possible
    by_site: dict[str, list] = {"classifieds": [], "reddit": [], "shopping": []}
    for c in pool:
        by_site.setdefault(c["site"], []).append(c)
    out = []
    target_each = max(3, target_n // 3)
    for s in ["classifieds", "reddit", "shopping"]:
        out.extend(by_site.get(s, [])[:target_each])
    # Fill remainder
    remaining = [c for c in pool if c not in out]
    out.extend(remaining[: max(0, target_n - len(out))])
    return out[:target_n]


def collect_b13_cases(target_n: int) -> list[dict[str, Any]]:
    """Tier 4 I2 case studies."""
    tier4 = json.loads(TIER4.read_text())
    inv = next((i for i in tier4["invariants"] if i["id"] == "I2"), None)
    if not inv:
        return []
    pool: list[dict[str, Any]] = []
    for ex in inv.get("case_study_examples", []):
        # Format may vary; common keys: site, task, step, run, condition_id, mode
        run = ex.get("run") or ex.get("run_root") or ""
        site = ex.get("site")
        task = ex.get("task") or ex.get("task_id")
        step = ex.get("step") or ex.get("step_idx")
        cond = ex.get("condition_id") or ex.get("condition")
        if not (site and task is not None and step is not None and run and cond):
            continue
        pool.append({
            "site": site, "task": int(task), "run": run, "condition_id": cond,
            "mode": ex.get("mode", "?"), "step_idx": int(step),
            "snippet": ex.get("snippet", ""),
        })
    return pool[:target_n]


async def probe_b01_case(case: dict[str, Any], browser) -> dict[str, Any]:
    steps = load_steps(case["run"], case["condition_id"], case["site"], case["task"])
    if case["step_idx"] >= len(steps):
        return {"case": case, "classification": "REPLAY_FAIL", "reason": "step_idx out of range"}
    step = steps[case["step_idx"]]
    bbox = step.get("element_bbox") or []
    obs_url = step.get("obs_url", "")
    action = step.get("action") or {}
    text_to_type = action.get("text") if isinstance(action, dict) else None

    out = {
        "case": case,
        "logged": {
            "obs_url": obs_url[:100],
            "element_bbox": bbox,
            "action_text": text_to_type,
            "action_success": step.get("action_success"),
            "page_changed": step.get("page_changed"),
        },
    }
    auth = SITE_AUTH.get(case["site"])
    if not auth or not auth.exists() or not obs_url or len(bbox) < 4:
        out["classification"] = "REPLAY_FAIL"
        out["reason"] = "auth/obs_url/bbox missing"
        return out

    try:
        ctx = await browser.new_context(storage_state=str(auth), viewport={"width": 1280, "height": 720})
        page = await ctx.new_page()
        try:
            await page.goto(obs_url, wait_until="domcontentloaded", timeout=15000)
            try:
                await page.wait_for_load_state("networkidle", timeout=4000)
            except Exception:
                pass
            cx = bbox[0] + bbox[2] / 2  # CORRECT formula
            cy = bbox[1] + bbox[3] / 2
            info = await page.evaluate(
                """([cx, cy]) => {
                    const el = document.elementFromPoint(cx, cy);
                    if (!el) return null;
                    return {
                        tag: el.tagName,
                        type: el.type || null,
                        editable: (
                            el.tagName === 'INPUT' ||
                            el.tagName === 'TEXTAREA' ||
                            el.isContentEditable === true ||
                            el.contentEditable === 'true'
                        ),
                        nearest_input: (() => {
                            let n = el;
                            for (let i = 0; i < 6 && n; i++) {
                                if (n.tagName === 'INPUT' || n.tagName === 'TEXTAREA' || n.isContentEditable) {
                                    return {tag: n.tagName, type: n.type || null};
                                }
                                n = n.parentElement;
                            }
                            return null;
                        })(),
                    };
                }""",
                [cx, cy],
            )
            out["dom_at_bbox"] = info
            if info is None:
                out["classification"] = "REPLAY_FAIL"
                out["reason"] = "no element at bbox center"
            elif info.get("editable"):
                out["classification"] = "EDITABLE_AT_CENTER"  # framework path actually OK
                out["reason"] = f"center hits {info.get('tag')} type={info.get('type')} — editable, no Meta+A leak"
            elif info.get("nearest_input"):
                out["classification"] = "NEAR_INPUT_BUT_OFFSET"  # likely 部分 scaffold (clicked label/wrapper, but adjacent input present)
                out["reason"] = f"center hits {info.get('tag')} ({info.get('nearest_input')} nearby) — Meta+A may leak before locator finds input"
            else:
                out["classification"] = "SCAFFOLD_TYPE_BUG"  # confirmed: Meta+A on non-editable
                out["reason"] = f"center hits {info.get('tag')} (no nearby input within 6 ancestors) — Meta+A would select page text"
        finally:
            await ctx.close()
    except Exception as e:
        out["classification"] = "REPLAY_FAIL"
        out["reason"] = f"playwright error: {type(e).__name__}: {str(e)[:120]}"
    return out


async def probe_b13_case(case: dict[str, Any], browser) -> dict[str, Any]:
    steps = load_steps(case["run"], case["condition_id"], case["site"], case["task"])
    if case["step_idx"] >= len(steps):
        return {"case": case, "classification": "REPLAY_FAIL", "reason": "step_idx out of range"}
    step = steps[case["step_idx"]]
    sd = step.get("state_digest", {}) or {}
    out = {
        "case": case,
        "logged": {
            "action_type": step.get("action_type"),
            "action_success": step.get("action_success"),
            "page_changed": step.get("page_changed"),
            "page_change_reasons": step.get("page_change_reasons", []),
            "url_before": sd.get("url_before"),
            "url_after": sd.get("url_after"),
            "title_before": sd.get("title_before"),
            "title_after": sd.get("title_after"),
            "scroll_y_before": sd.get("scroll_y_before"),
            "scroll_y_after": sd.get("scroll_y_after"),
            "text_similarity": step.get("text_similarity"),
        },
    }
    # Logger consistency check based on logged state_digest only — no replay needed
    url_changed = sd.get("url_before") != sd.get("url_after")
    title_changed = sd.get("title_before") != sd.get("title_after")
    scroll_delta = (
        abs((sd.get("scroll_y_after") or 0) - (sd.get("scroll_y_before") or 0))
        if (sd.get("scroll_y_before") is not None and sd.get("scroll_y_after") is not None)
        else 0
    )
    text_sim = step.get("text_similarity")
    real_change_signals = []
    if url_changed:
        real_change_signals.append("url")
    if title_changed:
        real_change_signals.append("title")
    if scroll_delta >= 5:
        real_change_signals.append(f"scroll({scroll_delta}px)")
    if text_sim is not None and text_sim < 0.95:
        real_change_signals.append(f"text_sim<{text_sim:.3f}")

    out["state_digest_check"] = {
        "url_changed": url_changed,
        "title_changed": title_changed,
        "scroll_delta": scroll_delta,
        "text_similarity": text_sim,
        "real_change_signals": real_change_signals,
    }
    if not real_change_signals:
        out["classification"] = "PAGE_CHANGED_FALSE_TRIGGER"  # I2 + I10 hybrid — runner says page changed but state_digest disagrees
        out["reason"] = "logged page_changed=True but no url/title/scroll/text_similarity evidence — false trigger"
    elif step.get("action_success") is False and real_change_signals:
        out["classification"] = "RUNNER_FALSE_NEGATIVE"
        out["reason"] = f"action_success=False but real change signals: {real_change_signals} — runner missed success"
    else:
        out["classification"] = "OTHER"
        out["reason"] = f"action_success={step.get('action_success')} signals={real_change_signals}"
    return out


async def main():
    n_b01 = 12
    n_b13 = 8

    b01_cases = collect_b01_cases(n_b01)
    b13_cases = collect_b13_cases(n_b13)
    print(f"B-01 TYPE: {len(b01_cases)} cases | B-13 I2: {len(b13_cases)} cases", file=sys.stderr)

    results = {"b01_results": [], "b13_results": []}
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            for i, c in enumerate(b01_cases, 1):
                t0 = time.time()
                try:
                    r = await asyncio.wait_for(probe_b01_case(c, browser), timeout=30)
                except asyncio.TimeoutError:
                    r = {"case": c, "classification": "REPLAY_FAIL", "reason": "timeout"}
                r["elapsed_s"] = round(time.time() - t0, 1)
                results["b01_results"].append(r)
                print(f"[{i}/{len(b01_cases)}] B-01 {c['site']} task {c['task']} step {c['step_idx']} ({c['mode']}) → {r['classification']}", file=sys.stderr)
            for i, c in enumerate(b13_cases, 1):
                t0 = time.time()
                # B-13 needs no Playwright — pure log analysis. But keeping infra symmetric.
                r = await probe_b13_case(c, browser)
                r["elapsed_s"] = round(time.time() - t0, 1)
                results["b13_results"].append(r)
                print(f"[{i}/{len(b13_cases)}] B-13 {c['site']} task {c['task']} step {c['step_idx']} → {r['classification']}", file=sys.stderr)
        finally:
            await browser.close()

    def breakdown(rows):
        d: dict[str, int] = {}
        for r in rows:
            d[r["classification"]] = d.get(r["classification"], 0) + 1
        return d

    b01_bd = breakdown(results["b01_results"])
    b13_bd = breakdown(results["b13_results"])
    b01_total = len(results["b01_results"])
    b01_replay_ok = b01_total - b01_bd.get("REPLAY_FAIL", 0)
    b01_scaffold = b01_bd.get("SCAFFOLD_TYPE_BUG", 0)
    b01_partial = b01_bd.get("NEAR_INPUT_BUT_OFFSET", 0)
    b01_editable = b01_bd.get("EDITABLE_AT_CENTER", 0)

    b13_total = len(results["b13_results"])
    b13_false_trigger = b13_bd.get("PAGE_CHANGED_FALSE_TRIGGER", 0)
    b13_runner_fn = b13_bd.get("RUNNER_FALSE_NEGATIVE", 0)
    b13_other = b13_bd.get("OTHER", 0)

    summary = {
        "audit_date": "2026-04-30",
        "purpose": "Self-verify B-01 TYPE 100% scaffold (codex claim) + B-13 NOT_A_BUG (codex 0/5 with 3 REPLAY_FAIL).",
        "b01_type_silent_failure": {
            "codex_claim_scaffold_fraction": 1.0,
            "codex_n_cases": 15,
            "self_verify_n_probed": b01_total,
            "self_verify_n_replay_ok": b01_replay_ok,
            "self_verify_breakdown": b01_bd,
            "scaffold_fraction_strict": round(b01_scaffold / b01_replay_ok, 3) if b01_replay_ok else None,
            "scaffold_fraction_lenient_with_partial": round((b01_scaffold + b01_partial) / b01_replay_ok, 3) if b01_replay_ok else None,
            "editable_at_center_fraction": round(b01_editable / b01_replay_ok, 3) if b01_replay_ok else None,
        },
        "b13_action_fail_page_changed": {
            "tier4_claim_violations": 25,
            "codex_claim_scaffold_fraction": 0.0,
            "codex_n_cases": 5,
            "self_verify_n_probed": b13_total,
            "self_verify_breakdown": b13_bd,
            "page_changed_false_trigger_count": b13_false_trigger,
            "runner_false_negative_count": b13_runner_fn,
            "verdict": (
                "NOT_A_BUG_CONFIRMED" if b13_runner_fn == 0
                else f"PARTIAL_BUG ({b13_runner_fn} runner-false-negatives)"
            ),
        },
    }
    results["summary"] = summary

    OUT_JSON.write_text(json.dumps(results, indent=2))
    md = [
        "# Self-Verify Probe — B-01 TYPE 100% + B-13 NOT_A_BUG",
        "",
        f"Audit date: {summary['audit_date']}",
        "",
        "## Purpose",
        "",
        summary["purpose"],
        "",
        "## B-01 TYPE Silent Failure",
        "",
        f"- Codex probe_audit_verification claim: scaffold fraction **{summary['b01_type_silent_failure']['codex_claim_scaffold_fraction']}** (15/15)",
        f"- Self-verify probed: {b01_total} cases, replay ok: {b01_replay_ok}",
        f"- Breakdown: {b01_bd}",
        f"- Strict scaffold fraction (only SCAFFOLD_TYPE_BUG): **{summary['b01_type_silent_failure']['scaffold_fraction_strict']}**",
        f"- Lenient (incl. NEAR_INPUT_BUT_OFFSET): **{summary['b01_type_silent_failure']['scaffold_fraction_lenient_with_partial']}**",
        f"- EDITABLE_AT_CENTER fraction: **{summary['b01_type_silent_failure']['editable_at_center_fraction']}** (these are NOT bugs — agent's center actually hits an input)",
        "",
        "## B-13 action_fail_but_page_changed",
        "",
        f"- Tier 4 I2 violations: 25",
        f"- Codex probe_audit_verification claim: 0/5 scaffold (3 REPLAY_FAIL + 2 REPLAY_DID_NOT_CHANGE)",
        f"- Self-verify probed: {b13_total} cases via state_digest log analysis (no Playwright replay — independent of codex's REPLAY_FAIL artifacts)",
        f"- Breakdown: {b13_bd}",
        f"- Runner false negative count: **{b13_runner_fn}**",
        f"- Verdict: **{summary['b13_action_fail_page_changed']['verdict']}**",
        "",
        "## Per-case detail",
        "",
        "### B-01 TYPE",
    ]
    for r in results["b01_results"]:
        c = r["case"]
        md.append(f"- {c['site']} task {c['task']} step {c['step_idx']} ({c['mode']}) → **{r['classification']}** | {r.get('reason','')[:140]}")
    md.append("")
    md.append("### B-13 I2")
    for r in results["b13_results"]:
        c = r["case"]
        md.append(f"- {c['site']} task {c['task']} step {c['step_idx']} ({c.get('mode','?')}) → **{r['classification']}** | {r.get('reason','')[:140]}")
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_JSON}\n      {OUT_MD}", file=sys.stderr)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
