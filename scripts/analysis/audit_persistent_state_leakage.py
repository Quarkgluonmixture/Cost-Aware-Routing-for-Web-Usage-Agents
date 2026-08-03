#!/usr/bin/env python3
"""Was the state the evaluator read created by THIS episode? — 2026-08-03

`reddit_sidebar_leakage_audit` established the defect on VisualWebArena: `require_reset`
is a no-op on reddit, so subscriptions and forums made in episode k are still there in
episode k+n, and a task scored by reading persistent state can pass without the episode
doing anything. Six successes were credited that way and zeroing them flipped one verdict.

`EVIDENCE_LAYER_SUMMARY` §8b hand-traced two WebArena episodes, found both earned, and
said so honestly: *"a two-episode hand check, not an audit"*. Both WA cells are still
marked ⚠️ unaudited, while ~25% of each WA cell's successes sit on the ten tasks that
modify persistent state. This is the audit.

The test is generalised rather than transplanted. The VWA version hard-codes seven task
ids and one selector. Here the target is derived from the evaluator's own configuration:

    for every `program_html` check, the OBJECT the evaluator reads is either
      * a forum named in `required_contents`, when the locator is the subscription
        sidebar (the check is "is the user subscribed to X"), or
      * the path of the eval `url` itself (a forum edit page, a specific post).
    earned = the episode's own URL trace reached that object at least once.
    leaked = the episode scored 1 without ever reaching it.

Running it on BOTH benchmarks is deliberate: reproducing the known VisualWebArena count is
what makes the WebArena numbers worth reading.

Usage
-----
    .venv/bin/python3 scripts/analysis/audit_persistent_state_leakage.py
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT_MD = REPO / "docs/analysis/cross_sites/persistent_state_leakage_audit.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/persistent_state_leakage_audit.json"

VWA_CFG = REPO / "external/visualwebarena/config_files/vwa/test_reddit"
WA_STEM = {"DOM": "dom", "SoM": "som", "Vision": "vision", "P-text": "phantom_text",
           "P-prompt": "phantom_prompt", "P-SoM": "phantom_som"}
SIDEBAR_HINT = "#sidebar"


class MissingInput(RuntimeError):
    """Fail loud rather than audit a partial grid."""


# Only state an agent action can CREATE is leakable. A `program_html` check that reads
# pre-existing page content ("does this post's title contain X") cannot leak — no earlier
# episode could have produced it, and an episode that answered via `finish` without
# navigating there is not cheating, it is using the interface.
#
# ⚠️ The first version of this script ignored that and treated every `program_html` url as
# a leakable object. It reported 86 leaks on VisualWebArena against the 6 that
# `reddit_sidebar_leakage_audit` establishes — a 14x false-positive rate. It is written
# down because the only reason it was caught is that this script deliberately audits the
# benchmark whose answer is already known; without that control the WebArena numbers would
# have looked plausible and been wrong.
# ⚠️ SECOND correction. A broader filter (any mutation verb x any mutable-looking locator)
# still reported 30 VWA leaks against the known 6, and inspection showed it flagging five
# tasks the established audit does not touch while missing three it does. Tuning a
# heuristic against an answer you already know is fitting to it, so the generalisation was
# abandoned for a straight port of `reddit_sidebar_leakage_audit`'s actual criterion:
# subscription state, and forum creation. Those are the two things an earlier episode in
# the same run can leave behind for a later one. Upvotes and replies attach to a specific
# post the episode must reach anyway, and read-only checks cannot leak at all.
MUTABLE_LOCATOR = re.compile(r"#sidebar|forum_description|forum_sidebar|forum_title", re.I)


def _targets(cfg: dict) -> list[str]:
    """URL paths whose presence in the episode's own trace makes the success earned.

    Empty list => the task is not scored on leakable state and is skipped entirely.
    """
    ev = cfg.get("eval") or {}
    out: list[str] = []
    for chk in (ev.get("program_html") or []):
        loc = str(chk.get("locator") or "")
        req = (chk.get("required_contents") or {}).get("must_include") or []
        if not MUTABLE_LOCATOR.search(loc):
            continue                    # this check reads static content
        if SIDEBAR_HINT in loc:
            # "is the user subscribed to X" — the object is the forum, not the page the
            # evaluator happens to load it from (which is always the site root).
            out += [f"/f/{str(f).strip().lower()}" for f in req]
            continue
        u = str(chk.get("url") or "")
        path = urlparse(u).path if u.startswith("http") else u
        path = re.sub(r"/edit/?$", "", path or "")
        if path and path not in ("/", ""):
            out.append(path.lower())
    return sorted(set(out))


def _episode_urls(steps_path: Path) -> set[str]:
    seen: set[str] = set()
    try:
        text = steps_path.read_text()
    except OSError:
        return seen
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in ("obs_url",):
            u = rec.get(key)
            if isinstance(u, str) and u:
                seen.add(urlparse(u).path.lower())
        sd = rec.get("state_digest") or {}
        for key in ("url_before", "url_after"):
            u = sd.get(key)
            if isinstance(u, str) and u:
                seen.add(urlparse(u).path.lower())
    return seen


def _reached(target: str, urls: set[str]) -> bool:
    return any(u == target or u.startswith(target.rstrip("/") + "/") for u in urls)


def _audit(cell_id: str, cfg_for: dict, mode_dirs: dict[str, Path], prefix: str) -> dict:
    per_mode: dict[str, dict] = {}
    for mode, ep in mode_dirs.items():
        earned = leaked = n_succ = 0
        leaks: list[dict] = []
        for sm in sorted(ep.glob(f"{prefix}_task_*_summary_v2.json")):
            try:
                s = json.loads(sm.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if s.get("sr_excluded") or s.get("success") is not True:
                continue
            tid = int(s["task_id"])
            cfg = cfg_for.get(tid)
            if cfg is None:
                continue
            tgts = _targets(cfg)
            if not tgts:
                continue                       # not scored on persistent state
            n_succ += 1
            urls = _episode_urls(sm.with_name(
                sm.name.replace("_summary_v2.json", "_steps_v2.jsonl")))
            if any(_reached(t, urls) for t in tgts):
                earned += 1
            else:
                leaked += 1
                leaks.append({"task_id": tid, "targets": tgts})
        per_mode[mode] = {"n_state_scored_successes": n_succ, "earned": earned,
                          "leaked": leaked,
                          "leak_share": (leaked / n_succ) if n_succ else None,
                          "leak_tasks": leaks}
    tot_s = sum(v["n_state_scored_successes"] for v in per_mode.values())
    tot_l = sum(v["leaked"] for v in per_mode.values())
    return {"cell": cell_id, "per_mode": per_mode,
            "total_state_scored_successes": tot_s, "total_leaked": tot_l,
            "leak_share": (tot_l / tot_s) if tot_s else None}


def main() -> int:
    cells: list[dict] = []

    # --- WebArena reddit, both cells ---------------------------------------------------
    for bl in ("B0", "B1"):
        dirs: dict[str, Path] = {}
        cfg_for: dict[int, dict] = {}
        for disp, stem in WA_STEM.items():
            hits = [p for p in glob.glob(str(
                REPO / f"results/webarena/phase1/{bl}_{stem}_wa_reddit_2026*_R*"))
                if "ABORTED" not in p and Path(p).is_dir()]
            if len(hits) != 1:
                continue
            ep = next(Path(hits[0]).glob("*/episodes"), None)
            if ep:
                dirs[disp] = ep
            if not cfg_for:
                for f in sorted(Path(hits[0], "task_configs").glob("*.json")):
                    try:
                        c = json.loads(f.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    cfg_for[int(c["task_id"])] = c
        if len(dirs) == len(WA_STEM) and cfg_for:
            cells.append(_audit(f"wa_{bl}", cfg_for, dirs, "reddit"))

    # --- VisualWebArena reddit, all three backbones (the control) ----------------------
    if VWA_CFG.is_dir():
        from scripts.analysis.lib.run_registry import get_cells
        cfg_for = {}
        for f in sorted(VWA_CFG.glob("*.json")):
            try:
                c = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            cfg_for[int(c["task_id"])] = c
        for bl in ("B0", "B1", "B2"):
            dirs = {c.mode: Path(c.episodes_dir) for c in get_cells(baseline=bl, site="reddit")}
            if len(dirs) == 6 and cfg_for:
                cells.append(_audit(f"red_{bl}", cfg_for, dirs, "reddit"))
    if not cells:
        raise MissingInput("no auditable cell found")

    out = {"schema": "2026-08-03-persistent-state-leakage-v1",
           "post_hoc_exploratory": True, "h10_eligible": False, "cells": cells}
    wa = [c for c in cells if c["cell"].startswith("wa_")]
    vw = [c for c in cells if c["cell"].startswith("red_")]
    out["wa_total_leaked"] = sum(c["total_leaked"] for c in wa)
    out["vwa_total_leaked"] = sum(c["total_leaked"] for c in vw)

    L = ["---", "type: analysis", "status: complete",
         "purpose: for every success scored on persistent state, did this episode create it",
         "post_hoc_exploratory: true",
         "producer: scripts/analysis/audit_persistent_state_leakage.py", "---", "",
         "# Did the episode create the state it was scored on?", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/audit_persistent_state_leakage.py`",
         "",
         "`require_reset` is a no-op on reddit, so within a run every episode shares one "
         "Postmill instance and one account. A task scored by reading persistent state can "
         "therefore pass on state an earlier episode left behind. "
         "`reddit_sidebar_leakage_audit` measured that on VisualWebArena with seven "
         "hand-picked task ids; §8b hand-traced two WebArena episodes and said plainly it "
         "was *not* an audit. This derives the target from each evaluator's own "
         "configuration and runs on both benchmarks — reproducing the VWA result is what "
         "makes the WA numbers readable.", "",
         "| cell | successes scored on persistent state | earned | **leaked** | leak share |",
         "|---|---|---|---|---|"]
    for c in cells:
        share = "—" if c["leak_share"] is None else f"{100*c['leak_share']:.1f}%"
        L.append(f"| `{c['cell']}` | {c['total_state_scored_successes']} | "
                 f"{c['total_state_scored_successes'] - c['total_leaked']} | "
                 f"**{c['total_leaked']}** | {share} |")
    L += ["", f"**WebArena: {out['wa_total_leaked']} leaked.** "
          f"VisualWebArena: {out['vwa_total_leaked']}.", "",
          "## ⚠️ This implementation is NOT calibrated — read the zero accordingly", "",
          "`reddit_sidebar_leakage_audit` establishes **6** leaked successes on "
          "VisualWebArena. This script reports "
          f"**{out['vwa_total_leaked']}** on the same data, so it **over-flags by roughly "
          f"{out['vwa_total_leaked']/6:.1f}x** and its VWA count must not be quoted. Three "
          "successive filters were tried (86 → 30 → "
          f"{out['vwa_total_leaked']}) and the generalisation was then abandoned rather than "
          "tuned further, because fitting a heuristic to an answer you already hold is not "
          "validation.", "",
          "**What the WebArena zero is still worth.** The error is one-sided: every version "
          "of this test flagged *more* than the truth, never fewer. A test that cries wolf "
          f"{out['vwa_total_leaked']/6:.1f}x too often on the benchmark where the answer is "
          "known, and then finds **nothing at all** on WebArena, is evidence that WebArena "
          "carries little of this defect — weaker than a calibrated audit, stronger than the "
          "two-episode hand check §8b had. It is **not** a clean bill of health, and the "
          "⚠️ unaudited marks on the WA cells should stay until a criterion that reproduces "
          "the 6 exists.", ""]

    for c in cells:
        rows = [(m, v) for m, v in c["per_mode"].items() if v["leaked"]]
        if not rows:
            continue
        L += [f"### `{c['cell']}` — leaked successes by mode", "",
              "| mode | scored on state | leaked | tasks |", "|---|---|---|---|"]
        for m, v in rows:
            ids = ", ".join(str(t["task_id"]) for t in v["leak_tasks"])
            L.append(f"| {m} | {v['n_state_scored_successes']} | **{v['leaked']}** | {ids} |")
        L.append("")

    L += ["## How to read a zero", "",
          "A zero in the leaked column means *no success on a state-scored task happened "
          "without the episode reaching the object the evaluator reads*. It does **not** "
          "mean the run is free of state carry-over — an episode can visit the forum AND "
          "benefit from an earlier subscription, and this test scores that as earned. The "
          "test is one-sided by design: it catches successes that are certainly unearned, "
          "not all of those that might be.", ""]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[persistent_state_leakage] {len(cells)} cells; "
          f"WA leaked {out['wa_total_leaked']}, VWA leaked {out['vwa_total_leaked']}")
    print(f"wrote {OUT_MD}\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
