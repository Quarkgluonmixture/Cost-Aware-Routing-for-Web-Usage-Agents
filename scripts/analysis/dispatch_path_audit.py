#!/usr/bin/env python3
"""How much of a mode difference is the scaffold's click implementation? — 2026-08-03

`action_executed.dispatch_path` records HOW each action was actually delivered to the
browser. It is populated on 70% of steps and, until today, was read by one script
(`aggregate_select_option_dispatch.py`, a narrow select-option study). Found by
`audit_field_consumption.py`.

Three delivery paths exist and their action success rates differ by a factor of five:

    element_id_locator_route   the emitted element id resolved to a Playwright locator
    element_id_framework       it did not, and the framework's own dispatch was used
    coord_*                    a coordinate click; the only path available without ids

That matters because the MIX is not constant across the grid:

  * `Vision` is on the coordinate path by construction — it has no element ids to emit —
    so its action success is bounded by whatever coordinate clicking achieves here.
  * The fallback share on the text arms rises as the backbone weakens, so a mode or model
    comparison silently carries a dispatch-mix difference alongside the representation
    difference it means to measure.

What is a model property and what is a scaffold property has to be kept apart, and this
script does not merge them:

    HOW OFTEN a run falls back is downstream of the model — it emitted an id the locator
    could not resolve. That is a legitimate part of what the representation buys.
    HOW WELL the fallback then works is a property of this scaffold. A better coordinate
    implementation or a better fallback would move these numbers without changing a single
    thing about the representations under study.

So the finding is an external-validity limitation, not a correction to any success rate:
the per-mode gaps reported elsewhere are partly mediated by an implementation that a
different harness would implement differently.

Usage
-----
    .venv/bin/python3 scripts/analysis/dispatch_path_audit.py
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.lib.run_registry import get_cells  # noqa: E402

OUT_MD = REPO / "docs/analysis/cross_sites/dispatch_path_audit.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/dispatch_path_audit.json"
MAX_EP = 30
FAMILIES = ("id_locator", "id_framework", "coord", "other")
PRETTY = {"id_locator": "element-id → locator",
          "id_framework": "element-id → framework fallback",
          "coord": "coordinate click", "other": "other"}


def _family(dp: str) -> str:
    if dp.startswith("coord"):
        return "coord"
    if dp == "element_id_locator_route":
        return "id_locator"
    if dp == "element_id_framework":
        return "id_framework"
    return "other"


def main() -> int:
    cells: dict[str, dict] = {}
    overall_ok: dict[str, int] = Counter()
    overall_n: dict[str, int] = Counter()
    raw_names: Counter = Counter()

    for bl in ("B0", "B1", "B2"):
        for site in ("classifieds", "reddit"):
            for cell in get_cells(baseline=bl, site=site):
                n_fam, ok_fam = Counter(), Counter()
                for f in sorted(Path(cell.episodes_dir).glob("*steps_v2.jsonl"))[:MAX_EP]:
                    for line in f.read_text().splitlines():
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        dp = (rec.get("action_executed") or {}).get("dispatch_path")
                        if not dp:
                            continue
                        raw_names[dp] += 1
                        fam = _family(dp)
                        n_fam[fam] += 1
                        good = rec.get("action_success") is True
                        ok_fam[fam] += good
                        overall_n[fam] += 1
                        overall_ok[fam] += good
                total = sum(n_fam.values())
                if total < 30:
                    continue
                cells[f"{bl}·{site}·{cell.mode}"] = {
                    "backbone": bl, "site": site, "mode": cell.mode, "n_actions": total,
                    "share": {k: n_fam[k] / total for k in FAMILIES if n_fam[k]},
                    "success": {k: (ok_fam[k] / n_fam[k]) for k in FAMILIES if n_fam[k]},
                    "action_success_overall": sum(ok_fam.values()) / total,
                }

    ov = {k: {"n": overall_n[k], "success": overall_ok[k] / overall_n[k]}
          for k in FAMILIES if overall_n[k]}
    # fallback share on the text arms only (Vision is coord by construction)
    fb = defaultdict(list)
    for v in cells.values():
        if v["mode"] != "Vision":
            fb[v["backbone"]].append(v["share"].get("id_framework", 0.0))
    out = {
        "schema": "2026-08-03-dispatch-path-audit-v1",
        "post_hoc_exploratory": True, "h10_eligible": False,
        "path_families": ov,
        "raw_dispatch_names": dict(raw_names.most_common()),
        "fallback_share_by_backbone": {b: {"min": min(v), "max": max(v),
                                           "mean": statistics.fmean(v)}
                                       for b, v in sorted(fb.items())},
        "cells": cells,
    }

    L = ["---", "type: analysis", "status: complete",
         "purpose: how much of a per-mode gap is carried by the scaffold's click delivery path",
         "post_hoc_exploratory: true",
         "scope_warning: this corrects no success rate. It identifies a mediator that a "
         "different harness would implement differently, i.e. an external-validity limit.",
         "producer: scripts/analysis/dispatch_path_audit.py", "---", "",
         "# What actually delivered the click", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/dispatch_path_audit.py`", "",
         "`action_executed.dispatch_path` says how each action reached the browser. It is on "
         "70% of steps and had one narrow consumer before 2026-08-03.", "",
         "## 1. Three paths, five-fold different action success", "",
         "| delivery path | actions | action success |", "|---|---|---|"]
    for k, v in sorted(ov.items(), key=lambda kv: -kv[1]["success"]):
        L.append(f"| {PRETTY[k]} | {v['n']:,} | **{100*v['success']:.1f}%** |")
    L += ["", "An action that does not succeed still consumes a step from the budget, so a "
          "mode spending more of its actions on a weak path is spending its step budget at a "
          "discount — which is a mechanism by which representation and outcome are connected "
          "that has nothing to do with what the model saw.", "",
          "## 2. The mix is not constant", "",
          "| cell · mode | actions | locator | fallback | coordinate | action success |",
          "|---|---|---|---|---|---|"]
    for cid, v in cells.items():
        s, u = v["share"], v["success"]
        g = lambda d, k: (f"{100*d[k]:.0f}%" if k in d else "—")   # noqa: E731
        L.append(f"| `{cid}` | {v['n_actions']:,} | {g(s,'id_locator')} | "
                 f"{g(s,'id_framework')} | {g(s,'coord')} | "
                 f"{100*v['action_success_overall']:.0f}% |")
    fbb = out["fallback_share_by_backbone"]
    L += ["", "Two structural facts in that table:", "",
          "1. **`Vision` is on the coordinate path by construction.** It emits no element "
          "ids, so it cannot use the path that succeeds "
          f"{100*ov['id_locator']['success']:.0f}% of the time. Its action success is capped "
          f"by whatever coordinate clicking achieves in this harness "
          f"({100*ov['coord']['success']:.0f}% overall). **This is not a confound to remove "
          "— it is part of what screenshot-only *is*** — but it does mean the Vision arm "
          "measures this scaffold's coordinate implementation as much as it measures the "
          "representation, and a harness with better grounding would report a different "
          "Vision.",
          "2. **The fallback share rises as the backbone weakens**: mean "
          + " · ".join(f"{b} {100*v['mean']:.0f}%" for b, v in fbb.items())
          + " on the text arms. Falling back is downstream of the model — it emitted an id "
          "the locator could not resolve — so that part is a legitimate capability "
          f"difference. What is **not** a capability difference is the "
          f"{100*ov['id_framework']['success']:.0f}% success of the fallback itself: that is "
          "this harness's fallback, and a better one would narrow every backbone gap that "
          "runs through it.", "",
          "## 3. What this licenses", "",
          "Nothing here changes a success rate, and no number elsewhere should be adjusted by "
          "it. What it establishes is that **the per-mode and per-backbone gaps reported in "
          "this project are partly mediated by two implementation choices** — how a "
          "coordinate click is issued, and what happens when an element id fails to resolve. "
          "Both are properties of this harness. A paper claiming a representation effect has "
          "to say so, because a reader's first alternative explanation for \"screenshot-only "
          "does worst\" is \"their coordinate clicking is bad\", and on this evidence that "
          "explanation is *partly correct*.", "",
          "## 4. Raw dispatch names", "",
          "Grouped above; listed here so the grouping can be checked rather than trusted.", "",
          "| dispatch_path | actions |", "|---|---|"]
    for name, n in out["raw_dispatch_names"].items():
        L.append(f"| `{name}` | {n:,} |")
    L.append("")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("[dispatch_path_audit] " + " · ".join(
        f"{PRETTY[k]} {100*v['success']:.1f}% (n={v['n']:,})" for k, v in ov.items()))
    print(f"wrote {OUT_MD}\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
