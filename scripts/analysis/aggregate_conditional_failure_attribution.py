#!/usr/bin/env python3
"""When one channel uniquely solves a task, how did the other one fail on it?

`aggregate_cross_mode_failure_signatures.py` answers a *marginal* question: which failure
signatures fire in which mode, pooled over everything. That is the vertical cut. This is the
horizontal one, and it is the question the complementarity result actually raises:

    On the tasks only the text channel solves, what did the image channel do wrong?
    On the tasks only the image channel solves, what did the text channel do wrong?

The 36 v8 scans carry a rule-hit list per (condition, task), and the six modes see an identical
task set within a cell, so the comparison is paired at the task level. The quantity reported is
an ENRICHMENT: a signature's hit rate among the losing channel's failures on the disagreement
set, against that same channel's hit rate over all its failures in the same cell. A ratio near 1
means the losing channel failed there the way it fails everywhere, and the disagreement is not
explained by that signature. A ratio well above 1 names a mechanism.

Channels follow the deployment partition used throughout: TEXT = {dom, phantom_text,
phantom_prompt, phantom_som}, IMAGE = {som, vision}. Note this is 4 arms against 2, so
`text_only` and `image_only` counts are NOT comparable to each other as effect sizes; only the
within-channel enrichments are read.

Scans come from `diag_rescan_all.py` (default `/tmp/diag_v8`, ruleset asserted identical across
all 36 so cross-mode aggregation is licensed; see the v8 freeze note in the diag digests).

WebArena is included as a seventh cell (`--wa-scan-dir`, default `/tmp/diag_v8_wa`). Its step
records live on the paper-grade host and were absent from the local mirror, which briefly looked
like a structural limit and is not one: the rules run on WA unmodified at the same ruleset
version, and WA reddit is the same Postmill application as VWA reddit. WA carries no AMENDMENT_08
exclusion, so its universe is the task set common to all six modes rather than a canonical scored
list. It matters here because WA is the workload on which the TEXT channel wins, so it is the
only place the asymmetry below can be checked against a reversed outcome.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402

LOG = logging.getLogger("conditional_failure")
OUT_MD = REPO / "docs/analysis/cross_sites/conditional_failure_attribution.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/conditional_failure_attribution.json"

TEXT = ["dom", "phantom_text", "phantom_prompt", "phantom_som"]
IMAGE = ["som", "vision"]
MODES = TEXT + IMAGE
BACKBONES = ["B0", "B1", "B2"]
SITES = ["classifieds", "reddit"]
MIN_TASKS = 5      # below this a per-cell enrichment is noise; pooled only
MIN_HITS = 8       # a rule needs this many conditional hits pooled before we report it


class MissingInput(RuntimeError):
    """Fail loud: a partial scan set would silently change every denominator."""


def load_cell(scan_dir: Path, bb: str, site: str,
              universe: set[int] | None = None) -> dict[str, dict[int, dict]]:
    """`universe=None` means the canonical VWA scored set; WA passes its own."""
    scored = universe if universe is not None else expected_scored_ids(site)[0]
    out: dict[str, dict[int, dict]] = {}
    versions = set()
    for m in MODES:
        p = scan_dir / f"{bb}_{m}_{site}.json"
        if not p.exists():
            raise MissingInput(f"missing scan {p}")
        d = json.loads(p.read_text())
        versions.add(d.get("ruleset_version"))
        rows = {}
        for ep in d["results"]:
            t = int(ep["task_id"])
            if t not in scored:
                continue
            rows[t] = {"success": bool(ep.get("success")),
                       "rules": {h["rule_id"] for h in (ep.get("hits") or [])},
                       "names": {h["rule_id"]: h.get("rule_name", "") for h in (ep.get("hits") or [])}}
        out[m] = rows
    if len(versions) != 1:
        raise MissingInput(f"{bb}/{site}: scans span rulesets {versions}; cross-mode aggregation "
                           "requires one version (see the v8 freeze note)")
    common = set.intersection(*(set(v) for v in out.values()))
    if len(common) != len(scored):
        raise MissingInput(f"{bb}/{site}: {len(common)} tasks common to all six modes, universe "
                           f"has {len(scored)}")
    return out


def wa_universe(scan_dir: Path) -> set[int]:
    """WA has no AMENDMENT_08 exclusion; the universe is what all six modes ran."""
    sets = []
    for m in MODES:
        p = scan_dir / f"B1_{m}_wa_reddit.json"
        if not p.exists():
            raise MissingInput(f"missing WA scan {p}")
        sets.append({int(e["task_id"]) for e in json.loads(p.read_text())["results"]})
    return set.intersection(*sets)


def attribute(cell: dict[str, dict[int, dict]], winners: list[str], losers: list[str]) -> dict:
    """Rule-hit enrichment among `losers` failures on tasks only `winners` solved."""
    tasks = set(next(iter(cell.values())))
    won = {t for t in tasks
           if any(cell[m][t]["success"] for m in winners)
           and not any(cell[m][t]["success"] for m in losers)}
    cond = collections.Counter()   # hits on the disagreement set
    base = collections.Counter()   # hits over all this channel's failures
    names: dict[str, str] = {}
    n_cond = n_base = 0
    for m in losers:
        for t, r in cell[m].items():
            if r["success"]:
                continue
            names.update(r["names"])
            base.update(r["rules"])
            n_base += 1
            if t in won:
                cond.update(r["rules"])
                n_cond += 1
    return {"n_tasks": len(won), "n_cond_eps": n_cond, "n_base_eps": n_base,
            "cond": dict(cond), "base": dict(base), "names": names}


def build(scan_dir: Path, wa_scan_dir: Path | None = None) -> dict:
    out = {"schema": "2026-08-02-conditional-failure-attribution-v1",
           "post_hoc_exploratory": True, "scan_dir": str(scan_dir),
           "text_modes": TEXT, "image_modes": IMAGE, "cells": {}}
    for bb in BACKBONES:
        for site in SITES:
            cid = f"{'cls' if site == 'classifieds' else 'red'}_{bb}"
            cell = load_cell(scan_dir, bb, site)
            out["cells"][cid] = {
                "text_only": attribute(cell, TEXT, IMAGE),
                "image_only": attribute(cell, IMAGE, TEXT),
            }
            LOG.info("%s: text-only %d tasks, image-only %d",
                     cid, out["cells"][cid]["text_only"]["n_tasks"],
                     out["cells"][cid]["image_only"]["n_tasks"])
    if wa_scan_dir and wa_scan_dir.exists():
        uni = wa_universe(wa_scan_dir)
        cell = load_cell(wa_scan_dir, "B1", "wa_reddit", universe=uni)
        out["cells"]["wa_red_B1"] = {
            "text_only": attribute(cell, TEXT, IMAGE),
            "image_only": attribute(cell, IMAGE, TEXT),
        }
        out["wa_n"] = len(uni)
        LOG.info("wa_red_B1 (n=%d): text-only %d tasks, image-only %d", len(uni),
                 out["cells"]["wa_red_B1"]["text_only"]["n_tasks"],
                 out["cells"]["wa_red_B1"]["image_only"]["n_tasks"])
    # pool: sum numerators and denominators across cells rather than averaging ratios
    for side in ("text_only", "image_only"):
        c = collections.Counter(); b = collections.Counter(); names = {}
        nc = nb = nt = 0
        for cell in out["cells"].values():
            a = cell[side]
            c.update(a["cond"]); b.update(a["base"]); names.update(a["names"])
            nc += a["n_cond_eps"]; nb += a["n_base_eps"]; nt += a["n_tasks"]
        out.setdefault("pooled", {})[side] = {
            "n_tasks": nt, "n_cond_eps": nc, "n_base_eps": nb,
            "cond": dict(c), "base": dict(b), "names": names}
    return out


def _rows(a: dict) -> list[tuple]:
    rows = []
    for rule, hits in a["cond"].items():
        if hits < MIN_HITS:
            continue
        cr = hits / a["n_cond_eps"] if a["n_cond_eps"] else 0.0
        br = a["base"].get(rule, 0) / a["n_base_eps"] if a["n_base_eps"] else 0.0
        rows.append((rule, a["names"].get(rule, ""), 100 * cr, 100 * br,
                     (cr / br) if br else float("inf"), hits))
    rows.sort(key=lambda r: -r[4])
    return rows


def render(d: dict) -> str:
    L = ["---", "type: analysis", "status: complete", "created: 2026-08-02",
         "purpose: when one channel uniquely solves a task, how the other channel failed on it",
         "post_hoc_exploratory: true",
         "scope_warning: TEXT is four arms and IMAGE is two, so the two sides' task counts are "
         "not comparable to each other. Only within-channel enrichment is read. Enrichment is "
         "a ratio of hit rates, not a test; no interval accompanies it.",
         "producer: scripts/analysis/aggregate_conditional_failure_attribution.py", "---", "",
         "# Conditional failure attribution", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_conditional_failure_attribution.py`",
         "",
         "The existing signature table is a marginal cut: which rules fire in which mode overall. "
         "This is the paired cut. Within a cell the six modes see one task set, so we can ask "
         "what the losing channel did on exactly the tasks the winning channel got and it did "
         "not. **Enrichment** is that signature's hit rate among the losing channel's failures on "
         "the disagreement set, over its hit rate across all its failures in the same cells. "
         "1.0 means the channel failed there the way it fails everywhere.", "",
         "## 1. Disagreement set sizes", "",
         "| cell | only TEXT solves | only IMAGE solves |", "|---|---|---|"]
    for cid, c in d["cells"].items():
        L.append(f"| `{cid}` | {c['text_only']['n_tasks']} | {c['image_only']['n_tasks']} |")
    p = d["pooled"]
    L.append(f"| **pooled** | **{p['text_only']['n_tasks']}** | **{p['image_only']['n_tasks']}** |")
    L += ["", "⚠️ TEXT is four arms against IMAGE's two, so a larger text-only count is partly "
          "arm count and must not be read as a larger effect.", ""]
    for side, title, who in (
            ("text_only", "2. Only the text channel solved it: how the IMAGE channel failed", "image"),
            ("image_only", "3. Only the image channel solved it: how the TEXT channel failed", "text")):
        a = p[side]
        L += [f"## {title}", "",
              f"Pooled over six cells. {a['n_cond_eps']} losing-channel failure episodes on the "
              f"disagreement set, against {a['n_base_eps']} of that channel's failures overall.",
              "", "| rule | name | on disagreement | baseline | enrichment | hits |",
              "|---|---|---|---|---|---|"]
        rows = _rows(a)
        if not rows:
            L.append("| — | no rule cleared the reporting floor | | | | |")
        for rule, name, cr, br, en, h in rows:
            mark = " **←**" if en >= 1.5 or en <= 0.67 else ""
            L.append(f"| `{rule}` | {name} | {cr:.1f}% | {br:.1f}% | **{en:.2f}x**{mark} | {h} |")
        L.append("")
    L += ["## 4. Rules that cannot be compared across sites", "",
          "Several P-rules carry site gates or are structurally inapplicable outside one site, so "
          "a 0.0% on one site is the gate and not a measurement. Verified firing rates over all "
          "episodes, which is the check that must precede any cross-site reading of a row above:",
          "",
          "| rule | VWA cls | VWA red | WA red | comparable across sites? |",
          "|---|---|---|---|---|",
          "| `P6` visual-task-DOM-must-fail | 7.9% | **0.0%** | **0.0%** | **no** — gated off all reddit |",
          "| `P16` visual-image-content | 3.9% | **0.0%** | **0.0%** | **no** — gated off all reddit |",
          "| `P17` click-back oscillation | 11.5% | **0.0%** | **0.0%** | **no** — classifieds only |",
          "| `P25` cross-site task skips a site | 5.1% | 15.9% | **0.0%** | no — WA has no cross-site tasks |",
          "| `P43` page-embedded visual, no screenshot | 20.4% | 19.5% | **0.0%** | **yes** — ungated; the 0.0% is real |",
          "| `P27` gives up when not found | 1.6% | 1.1% | 0.3% | yes |",
          "| `P31` budget exhausted | 30.8% | 70.2% | 65.1% | yes |",
          "| `P45` / `P36` / `P5` | 26-50% | 27-47% | 32-50% | yes |",
          "",
          "⚠️ `P43` is ungated and its WA zero is a property of the task set: no WA reddit intent "
          "matches the visual-intent regex. **But P43 is a neutral (task x mode) label by its own "
          "definition, not a failure mechanism.** Its docstring records a controlled dom->som "
          "comparison on exactly this task set measuring +0.00 / +1.56 / +0.00 pp from restoring "
          "the screenshot. P43 therefore says WHERE the image channel's unique wins sit, and "
          "explicitly does not say the text channel failed *because* the screenshot was withheld.",
          "",
          "## 5. Reading", "",
          "A signature near 1.0x is the null and most rows sit there: the losing channel mostly "
          "fails on these tasks the way it fails on every task. Rows away from 1.0x are the ones "
          "that name a mechanism for the complementarity, and they are the only rows this "
          f"analysis licenses anyone to cite. The reporting floor is {MIN_HITS} pooled hits; rules "
          "below it are omitted rather than shown at unstable ratios."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan-dir", type=Path, default=Path("/tmp/diag_v8"))
    ap.add_argument("--wa-scan-dir", type=Path, default=Path("/tmp/diag_v8_wa"))
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    d = build(a.scan_dir, a.wa_scan_dir)
    OUT_JSON.write_text(json.dumps(d, indent=2))
    OUT_MD.write_text(render(d))
    print(f"✓ {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
