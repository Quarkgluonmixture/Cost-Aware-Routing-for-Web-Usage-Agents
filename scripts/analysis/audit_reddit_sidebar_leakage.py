#!/usr/bin/env python3
"""Which reddit sidebar-task successes were earned, and which leaked in?

Seven reddit tasks are scored by reading one selector — `#sidebar > section > ul`,
the list of forums the logged-in user is subscribed to. Six check `must_include`
(a forum that should have been subscribed); task 160 checks `must_exclude` only.

`require_reset` is a no-op on reddit: `external/visualwebarena/browser_env/envs.py:172`
gates the reset POST on `"classifieds" in instance_config["sites"]`, with a
`TODO(jykoh)` for the other two sites. Condition-level reset does happen (the
docker stack is rebuilt per condition), but WITHIN a run the 205 episodes share
one Postmill instance and one logged-in account — so a subscription made in
episode k is still there in episode k+n.

Consequence: an episode can satisfy a `must_include` sidebar check without ever
visiting the forum in question, because an earlier episode in the same run
subscribed to it. Verified by hand on B2·dom (tasks 178/188/189 all scored 1.0
having visited neither the target forum nor any subscribe control for it).

This script applies the same test to every scored success on these tasks:

    earned   = the episode visited the required forum at least once
    leaked   = scored success, never visited it
    passive  = task 160 only — the eval is satisfiable by doing nothing

The verdict is per-episode and mechanical. It does NOT decide what to do about
it: excluding tasks from the scored universe is a preregistration-level action
(tasks 58 and 160 went through AMENDMENT_08), so this produces the evidence and
stops there.

Usage:
  .venv/bin/python3 scripts/analysis/audit_reddit_sidebar_leakage.py \
      --out docs/analysis/cross_sites/reddit_sidebar_leakage_audit.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.analysis.axis_effect_size as A  # noqa: E402
from scripts.analysis.lib.canonical_task_universe import (  # noqa: E402
    expected_scored_ids, protocol_excluded_in_universe)

CONFIG_DIR = REPO / "external/visualwebarena/config_files/vwa/test_reddit"
SIDEBAR_SELECTOR = "#sidebar > section > ul"
FORUM_RE = re.compile(r"/f/([^/?#]+)")


def sidebar_tasks() -> dict[int, dict]:
    """Every reddit task whose eval reads the subscription sidebar."""
    out: dict[int, dict] = {}
    for cfg_path in sorted(CONFIG_DIR.glob("*.json")):
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:  # noqa: BLE001
            continue
        for block in (cfg.get("eval", {}).get("program_html") or []):
            if SIDEBAR_SELECTOR not in str(block.get("locator", "")):
                continue
            rc = block.get("required_contents") or {}
            inc = rc.get("must_include") or []
            exc = rc.get("must_exclude") or []
            # "a |OR| b" means any one of them satisfies the check
            wanted = {w.strip().lower()
                      for clause in inc for w in str(clause).split("|OR|")}
            out[int(cfg_path.stem)] = {
                "task_id": int(cfg_path.stem),
                "intent": cfg.get("intent", ""),
                "must_include": inc, "must_exclude": exc,
                "wanted_forums": sorted(wanted),
                "passive_satisfiable": bool(exc) and not inc,
            }
    return out


WA_RAW = REPO / "external/visualwebarena/config_files/wa/test_reddit.raw.json"
WA_SIDEBAR_SELECTOR = "#sidebar > section"


def wa_sidebar_tasks() -> dict[int, dict]:
    """WebArena's subscription-sidebar tasks — the same defect surface, never audited.

    Added 2026-08-03 after a coverage sweep asked which evidence still has no WA cell.
    `leakage_sensitivity` covered `red_B0/B1/B2` only, and `EVIDENCE_LAYER_SUMMARY` §8b
    had hand-traced exactly two WA episodes while saying in the same breath that two
    episodes are not an audit.

    The mechanism transfers verbatim: WA reddit is the same Postmill image, the
    `require_reset` gate is the same `"classifieds" in sites` test, and five of its 106
    tasks (595-599, "subscribe to forum X") are scored by reading `#sidebar > section`
    for a forum name. VWA's selector carries a trailing `> ul`; the enclosing section is
    the same subscription list. WA ships one raw config array rather than per-task files,
    so the task list is read differently — that is the only difference.
    """
    out: dict[int, dict] = {}
    if not WA_RAW.is_file():
        return out
    for cfg in json.loads(WA_RAW.read_text()):
        for block in (cfg.get("eval", {}).get("program_html") or []):
            if WA_SIDEBAR_SELECTOR not in str(block.get("locator", "")):
                continue
            rc = block.get("required_contents") or {}
            inc = rc.get("must_include") or []
            exc = rc.get("must_exclude") or []
            wanted = {w.strip().lower()
                      for clause in inc for w in str(clause).split("|OR|")}
            out[int(cfg["task_id"])] = {
                "task_id": int(cfg["task_id"]),
                "intent": cfg.get("intent", ""),
                "must_include": inc, "must_exclude": exc,
                "wanted_forums": sorted(wanted),
                "passive_satisfiable": bool(exc) and not inc,
            }
    return out


def forums_visited(steps_path: Path) -> set[str]:
    seen: set[str] = set()
    if not steps_path.is_file():
        return seen
    for line in steps_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        for field in ("obs_url", "url_before", "url_after"):
            val = rec.get(field)
            if isinstance(val, str):
                seen.update(m.lower() for m in FORUM_RE.findall(val))
        sd = rec.get("state_digest")
        if isinstance(sd, dict):
            for v in sd.values():
                if isinstance(v, str):
                    seen.update(m.lower() for m in FORUM_RE.findall(v))
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs/analysis/cross_sites/reddit_sidebar_leakage_audit.md")
    ap.add_argument("--json-out", type=Path,
                    default=REPO / "docs/analysis/cross_sites/reddit_sidebar_leakage_audit.json")
    ap.add_argument("--with-wa", action="store_true",
                    help="also audit WebArena reddit (tasks 595-599, same Postmill defect); "
                         "writes to *_with_wa.* so the VWA-only product stays byte-stable")
    a = ap.parse_args()

    if a.with_wa:
        for suffix in ("out", "json_out"):
            p = getattr(a, suffix)
            setattr(a, suffix, p.with_name(p.name.replace(".", "_with_wa.", 1)))

    tasks = sidebar_tasks()
    scored, _ = expected_scored_ids("reddit")
    excluded = protocol_excluded_in_universe("reddit")

    rows: list[dict] = []
    for baseline in ("B0", "B1", "B2"):
        for mode, axis_key in A._REGISTRY_MODE_TO_AXIS_KEY.items():
            ep = A.STEP_DIRS.get(baseline, {}).get("reddit", {}).get(axis_key)
            if ep is None or not ep.exists():
                continue
            for tid, spec in sorted(tasks.items()):
                summ = ep / f"reddit_task_{tid}_summary_v2.json"
                if not summ.is_file():
                    continue
                success = json.loads(summ.read_text()).get("success") is True
                visited = forums_visited(ep / f"reddit_task_{tid}_steps_v2.jsonl")
                hit = sorted(set(spec["wanted_forums"]) & visited)
                if spec["passive_satisfiable"]:
                    verdict = "passive-satisfiable" if success else "failed"
                elif not success:
                    verdict = "failed"
                elif hit:
                    verdict = "earned"
                else:
                    verdict = "LEAKED"
                rows.append({
                    "cell": f"{baseline}_reddit", "baseline": baseline, "mode": mode,
                    "task_id": tid, "success": success, "verdict": verdict,
                    "wanted": spec["wanted_forums"], "visited_target": hit,
                    "n_forums_visited": len(visited),
                    "in_scored_universe": tid in scored,
                    "protocol_excluded": tid in excluded,
                })

    wa_tasks: dict[int, dict] = {}
    if a.with_wa:
        wa_tasks = wa_sidebar_tasks()
        if not wa_tasks:
            raise SystemExit(f"--with-wa: no sidebar tasks found in {WA_RAW}")
        for baseline in A.WA_BASELINES:
            # attach_wa raises rather than degrading, and sets A.WA_UNIVERSE to the task
            # set common to all six modes — WA carries no AMENDMENT_08 exclusion list.
            A.attach_wa(baseline)
            for mode, axis_key in A._REGISTRY_MODE_TO_AXIS_KEY.items():
                ep = A.STEP_DIRS.get(baseline, {}).get("wa_reddit", {}).get(axis_key)
                if ep is None or not ep.exists():
                    continue
                for tid, spec in sorted(wa_tasks.items()):
                    summ = ep / f"reddit_task_{tid}_summary_v2.json"
                    if not summ.is_file():
                        continue
                    success = json.loads(summ.read_text()).get("success") is True
                    visited = forums_visited(ep / f"reddit_task_{tid}_steps_v2.jsonl")
                    hit = sorted(set(spec["wanted_forums"]) & visited)
                    if spec["passive_satisfiable"]:
                        verdict = "passive-satisfiable" if success else "failed"
                    elif not success:
                        verdict = "failed"
                    elif hit:
                        verdict = "earned"
                    else:
                        verdict = "LEAKED"
                    rows.append({
                        "cell": f"{baseline}_wa_reddit", "baseline": baseline, "mode": mode,
                        "task_id": tid, "success": success, "verdict": verdict,
                        "wanted": spec["wanted_forums"], "visited_target": hit,
                        "n_forums_visited": len(visited),
                        "in_scored_universe": tid in (A.WA_UNIVERSE or set()),
                        "protocol_excluded": False,
                        "benchmark": "webarena",
                    })

    scored_rows = [r for r in rows if r["in_scored_universe"]]
    leaked = [r for r in scored_rows if r["verdict"] == "LEAKED"]
    earned = [r for r in scored_rows if r["verdict"] == "earned"]
    passive = [r for r in rows if r["verdict"] == "passive-satisfiable"]

    L: list[str] = []
    L.append("# reddit sidebar-task leakage audit")
    L.append("")
    L.append(f"- **{len(tasks)} tasks** are scored by reading `{SIDEBAR_SELECTOR}` "
             "(the subscribed-forum list)")
    L.append("- `require_reset` is a **no-op on reddit** (`envs.py:172` gates it on "
             "`\"classifieds\" in sites`, `TODO(jykoh)` for the rest), so "
             "subscriptions accumulate across the 205 episodes of a run")
    L.append("- **earned** = the episode visited the required forum · **LEAKED** = "
             "scored success without ever visiting it · **passive-satisfiable** = "
             "`must_exclude`-only eval, satisfied by doing nothing")
    L.append("- verdicts are mechanical and per-episode. Whether to exclude these "
             "tasks is a **preregistration-level decision** and is not made here.")
    L.append("")
    L.append("## Verdict counts (scored universe only)")
    L.append("")
    L.append(f"| verdict | n |")
    L.append("|---|---|")
    L.append(f"| **LEAKED** | **{len(leaked)}** |")
    L.append(f"| earned | {len(earned)} |")
    L.append(f"| failed | {sum(1 for r in scored_rows if r['verdict'] == 'failed')} |")
    L.append("")
    if passive:
        L.append(f"Plus **{len(passive)}** passive-satisfiable successes on "
                 f"protocol-excluded task(s) "
                 f"{sorted({r['task_id'] for r in passive})} — already outside the "
                 "scored universe via AMENDMENT_08, listed for completeness.")
        L.append("")

    L.append("## Per-cell impact on the scored success count")
    L.append("")
    L.append("| cell · mode | scored successes | of which LEAKED | leaked share |")
    L.append("|---|---|---|---|")
    for baseline in ("B0", "B1", "B2"):
        for mode in A._REGISTRY_MODE_TO_AXIS_KEY:
            ep = A.STEP_DIRS.get(baseline, {}).get("reddit", {}).get(
                A._REGISTRY_MODE_TO_AXIS_KEY[mode])
            if ep is None or not ep.exists():
                continue
            n_succ = sum(1 for f in ep.glob("reddit_task_*_summary_v2.json")
                         if (lambda d: d.get("success") is True
                             and int(d["task_id"]) in scored)(json.loads(f.read_text())))
            lk = [r for r in leaked if r["baseline"] == baseline and r["mode"] == mode]
            share = f"{100.0*len(lk)/n_succ:.1f}%" if n_succ else "—"
            L.append(f"| {baseline} · {mode} | {n_succ} | {len(lk)} | {share} |")
    L.append("")

    L.append("## Every leaked success")
    L.append("")
    L.append("| cell | mode | task | eval wants sidebar to contain | forums visited |")
    L.append("|---|---|---|---|---|")
    for r in sorted(leaked, key=lambda r: (r["baseline"], r["mode"], r["task_id"])):
        L.append(f"| {r['cell']} | {r['mode']} | {r['task_id']} | "
                 f"`{' | '.join(r['wanted'])}` | {r['n_forums_visited']} forums, "
                 f"**none of them the target** |")
    L.append("")
    L.append("## Earned successes (kept)")
    L.append("")
    for r in sorted(earned, key=lambda r: (r["baseline"], r["mode"], r["task_id"])):
        L.append(f"- {r['cell']} · {r['mode']} · task {r['task_id']} — visited "
                 f"`{', '.join(r['visited_target'])}`")
    L.append("")

    if a.with_wa:
        wa_rows = [r for r in rows if r.get("benchmark") == "webarena"]
        wa_scored = [r for r in wa_rows if r["in_scored_universe"]]
        wa_leak = [r for r in wa_scored if r["verdict"] == "LEAKED"]
        wa_earn = [r for r in wa_scored if r["verdict"] == "earned"]
        L.append("")
        L.append("## WebArena reddit (tasks 595-599) — first audit, 2026-08-03")
        L.append("")
        L.append(f"- **{len(wa_tasks)} tasks** scored by `{WA_SIDEBAR_SELECTOR}`, same Postmill "
                 "image and the same `require_reset` no-op as VWA reddit")
        L.append(f"- {len(wa_scored)} scored episodes across both WA backbones x 6 modes: "
                 f"**{len(wa_leak)} LEAKED**, {len(wa_earn)} earned, "
                 f"{sum(1 for r in wa_scored if r['verdict'] == 'failed')} failed")
        L.append("")
        L.append("⚠️ **What this zero does and does not mean.** The `earned` test is "
                 "*did the episode visit the required forum* — the same test VWA uses, so the "
                 "two are comparable, and it is a **lower bound on leakage**. Visiting is not "
                 "subscribing: an episode can arrive at a forum an *earlier* episode already "
                 "subscribed to, read `Unsubscribe` on the button, and finish without acting. "
                 "That is a leak this test scores as earned.")
        L.append("")
        L.append("**The leakage window is open on WA.** Within one run the target forums are "
                 "reached by many non-target tasks (on `B1`/DOM: `books` by 13 other tasks, "
                 "`machinelearning` by 8, `pittsburgh` by 4, `consoles` by 2; only `space` is "
                 "touched by its own task alone). So the mechanism is available here; what the "
                 "audit establishes is that **no scored success was obtained without ever "
                 "reaching the forum**, not that no success inherited a subscription.")
        L.append("")
        L.append("**One arrival-already-subscribed case is confirmed by hand**: `B1`/DOM task "
                 "597, whose final step reads *\"a visible \'Unsubscribe 1 subscriber\' "
                 "button, indicating the user is already subscribed ... so the task is "
                 "complete\"*. It is scored `earned` here because it did visit `consoles`.")
        L.append("")
        L.append("**A text heuristic was tried and rejected.** Flagging episodes whose "
                 "reasoning says *already subscribed* without saying *clicked subscribe* "
                 "returns 6 of 37 — but it misses the hand-confirmed 597 above, because that "
                 "episode says both (it deliberates: *I need to click the 'Unsubscribe'* and "
                 "*I will click the 'Subscribe'*). Model self-report cannot separate "
                 "deliberation from action, so no count is published from it. Deciding this "
                 "mechanically needs the subscription state before and after each click, which "
                 "`state_digest` does not carry.")
        L.append("")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L), encoding="utf-8")
    a.json_out.write_text(json.dumps(
        {"selector": SIDEBAR_SELECTOR, "tasks": tasks,
         "wa_selector": WA_SIDEBAR_SELECTOR if a.with_wa else None,
         "wa_tasks": wa_tasks or None,
         "n_leaked": len(leaked), "n_earned": len(earned),
         "n_passive_satisfiable": len(passive), "rows": rows},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {a.out}\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
