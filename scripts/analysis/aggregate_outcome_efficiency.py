#!/usr/bin/env python3
"""Cost and latency per SUCCESS, not per attempt — and what changes when you switch.

Every efficiency number in this project is per attempt: `total_billed_cost_usd` and
`total_latency_ms` averaged over episodes. A deployment does not buy attempts. It buys completed
tasks, and the two orderings are not the same ordering.

The screenshot-only channel is the cheapest per attempt in 6 of 6 cells — by construction, since
it carries no accessibility-tree text. Divide by the success rate and that lead survives only on
the visual site. On the text site each success costs 1.4-1.9x what the matched text channel
costs, because the cheaper attempts have to be made more often. The fused channel, which is the
dearest per attempt in 5 of 6 cells and has no detectable accuracy premium over the matched
single channel, is the FASTEST per success in four of the four cells where success rates are high
enough for the ratio to mean anything.

Estimand. `sum(cost) / sum(success)` over the cell's scored tasks — the total spent divided by
the tasks completed — not `mean(cost) / SR`, which is the same number but invites reading it as
a per-episode quantity. CIs are a paired bootstrap over tasks, so they carry the success rate's
own sampling noise, which is what makes these ratios fragile at low SR.

⚠️ Read the CI, not the point estimate, on any cell whose success count is small. B2's cells
land 3-5 successes out of 224; a ratio with that denominator is a direction at best. The four
B0/B1 cells are where this analysis has content.

post_hoc_exploratory=True. Touches no gating producer.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402
from scripts.analysis.lib.run_registry import get_cells  # noqa: E402

LOG = logging.getLogger("outcome_efficiency")
OUT_MD = REPO / "docs/analysis/cross_sites/outcome_efficiency.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/outcome_efficiency.json"

MODES = ["DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM"]
COST_FIELD = "total_billed_cost_usd"
N_BOOT = 10000
SEED = 20260802
MIN_SUCC_FOR_CONTENT = 10       # below this the ratio is a direction, not a measurement


class MissingInput(RuntimeError):
    """Fail loud rather than divide by a success count from a partial read."""


def load_cell(baseline: str, site: str, cells: dict) -> dict[str, list[tuple[int, float, float]]]:
    """mode -> [(success, cost, latency_s)] over the canonical scored universe."""
    scored = set(expected_scored_ids(site)[0])
    out: dict[str, list] = {}
    for mode in MODES:
        c = cells.get((baseline, site, mode))
        if c is None:
            raise MissingInput(f"{baseline}/{site}/{mode}: absent from the registry")
        rows = []
        for p in c.episodes_dir.glob(f"{site}_task_*_summary_v2.json"):
            tid = int(re.search(r"task_(\d+)_", p.name).group(1))
            if tid not in scored:
                continue
            s = json.loads(p.read_text())
            if s.get("sr_excluded"):
                continue
            if COST_FIELD not in s or s.get("total_latency_ms") is None:
                raise MissingInput(f"{p}: missing {COST_FIELD} or total_latency_ms")
            rows.append((1 if s.get("success") else 0, float(s[COST_FIELD]),
                         float(s["total_latency_ms"]) / 1000.0))
        if len(rows) != len(scored):
            raise MissingInput(f"{baseline}/{site}/{mode}: {len(rows)} scored episodes, "
                               f"universe has {len(scored)}")
        out[mode] = rows
    return out


def per_success(rows: list[tuple[int, float, float]]) -> dict:
    """Total spent / tasks completed, with a paired bootstrap CI over tasks."""
    n = len(rows)
    succ = sum(r[0] for r in rows)
    if succ == 0:
        return {"n": n, "n_success": 0, "cost_per_success": None, "latency_per_success": None,
                "cost_ci": None, "latency_ci": None, "sr_pct": 0.0}
    point_c = sum(r[1] for r in rows) / succ
    point_l = sum(r[2] for r in rows) / succ
    rng = random.Random(SEED)
    bc, bl = [], []
    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        s = sum(rows[i][0] for i in idx)
        if s == 0:
            continue                      # resample with no successes: ratio undefined, drop
        bc.append(sum(rows[i][1] for i in idx) / s)
        bl.append(sum(rows[i][2] for i in idx) / s)
    bc.sort(); bl.sort()

    def ci(b):
        return [b[int(0.025 * len(b))], b[int(0.975 * len(b)) - 1]] if b else None

    return {"n": n, "n_success": succ, "sr_pct": 100 * succ / n,
            "cost_per_success": point_c, "latency_per_success": point_l,
            "cost_ci": ci(bc), "latency_ci": ci(bl),
            "n_boot_undefined": N_BOOT - len(bc)}


def build() -> dict:
    reg = {(c.baseline, c.site, c.mode): c for c in get_cells(grade="paper-grade")}
    out = {"schema": "2026-08-02-outcome-efficiency-v1", "post_hoc_exploratory": True,
           "estimand": "sum(cost) / sum(success) over the cell's scored tasks",
           "n_boot": N_BOOT, "seed": SEED, "cells": {}}
    for baseline in ("B0", "B1", "B2"):
        for site in ("classifieds", "reddit"):
            cid = f"{'cls' if site == 'classifieds' else 'red'}_{baseline}"
            data = load_cell(baseline, site, reg)
            per_mode = {m: per_success(rows) for m, rows in data.items()}
            att = {m: sum(r[1] for r in rows) / len(rows) for m, rows in data.items()}
            lat_att = {m: sum(r[2] for r in rows) / len(rows) for m, rows in data.items()}
            live = [m for m in MODES if per_mode[m]["cost_per_success"] is not None]
            out["cells"][cid] = {
                "per_mode": per_mode,
                "cost_per_attempt": att, "latency_per_attempt": lat_att,
                "cheapest_per_attempt": min(MODES, key=lambda m: att[m]),
                "cheapest_per_success": (min(live, key=lambda m: per_mode[m]["cost_per_success"])
                                         if live else None),
                "fastest_per_attempt": min(MODES, key=lambda m: lat_att[m]),
                "fastest_per_success": (min(live, key=lambda m: per_mode[m]["latency_per_success"])
                                        if live else None),
                "max_n_success": max(per_mode[m]["n_success"] for m in MODES),
                "has_content": max(per_mode[m]["n_success"] for m in MODES) >= MIN_SUCC_FOR_CONTENT,
            }
            LOG.info("%s: cheapest/attempt=%s cheapest/success=%s fastest/success=%s", cid,
                     out["cells"][cid]["cheapest_per_attempt"],
                     out["cells"][cid]["cheapest_per_success"],
                     out["cells"][cid]["fastest_per_success"])
    return out


def render(d: dict) -> str:
    cells = d["cells"]
    live = {k: v for k, v in cells.items() if v["has_content"]}
    L = ["---", "type: analysis", "status: complete", "created: 2026-08-02",
         "purpose: does the efficiency ordering survive switching from per-attempt to per-success",
         "post_hoc_exploratory: true",
         "scope_warning: within-cell only (B0 bills an API, B1/B2 are electricity-derived). "
         "Ratios at low success counts are directions, not measurements — read the CI.",
         "producer: scripts/analysis/aggregate_outcome_efficiency.py", "---", "",
         "# Per attempt is not per success", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_outcome_efficiency.py`", "",
         "Every efficiency figure elsewhere in this project is **per attempt**. A deployment buys "
         "completed tasks, not attempts, and the two orderings differ. Estimand: "
         "`sum(cost) / sum(success)` over the cell's scored tasks, with a paired bootstrap over "
         "tasks so the CI carries the success rate's own sampling noise.", "",
         "## 1. Who wins, under each denominator", "",
         "| cell | max successes | cheapest / attempt | cheapest / **success** | fastest / attempt | fastest / **success** |",
         "|---|---|---|---|---|---|"]
    flips_c = flips_l = 0
    for cid, c in cells.items():
        fc = c["cheapest_per_attempt"] != c["cheapest_per_success"]
        fl = c["fastest_per_attempt"] != c["fastest_per_success"]
        flips_c += fc and c["has_content"]
        flips_l += fl and c["has_content"]
        mark = "" if c["has_content"] else " ⚠️"
        L.append(f"| `{cid}`{mark} | {c['max_n_success']} | {c['cheapest_per_attempt']} | "
                 f"**{c['cheapest_per_success']}**{' ←flips' if fc else ''} | "
                 f"{c['fastest_per_attempt']} | **{c['fastest_per_success']}**"
                 f"{' ←flips' if fl else ''} |")
    n_live = len(live)
    L += ["", f"⚠️ marks cells whose best mode has fewer than {MIN_SUCC_FOR_CONTENT} successes; "
          f"their ratios are directions at best. The {n_live} unmarked cells are where this has "
          "content.", "",
          f"Among those {n_live}: the cheapest-per-attempt mode stops being cheapest-per-success "
          f"in **{flips_c}**, and the fastest-per-attempt mode stops being fastest-per-success in "
          f"**{flips_l}**.", "",
          "## 2. The three channels, side by side", "",
          "| cell | mode | cost/attempt | SR% | **cost/success** | 95% CI | **latency/success (s)** | 95% CI |",
          "|---|---|---|---|---|---|---|---|"]
    for cid, c in cells.items():
        for m in ("DOM", "SoM", "Vision"):
            p = c["per_mode"][m]
            if p["cost_per_success"] is None:
                L.append(f"| `{cid}` | {m} | {c['cost_per_attempt'][m]:.4f} | 0.00 | — | — | — | — |")
                continue
            cc, lc = p["cost_ci"], p["latency_ci"]
            L.append(f"| `{cid}` | {m} | {c['cost_per_attempt'][m]:.4f} | {p['sr_pct']:.2f} | "
                     f"**{p['cost_per_success']:.3f}** | [{cc[0]:.3f}, {cc[1]:.3f}] | "
                     f"**{p['latency_per_success']:.0f}** | [{lc[0]:.0f}, {lc[1]:.0f}] |")
    L += ["", "## 3. What this does and does not license", "",
          "**Licensed.** The efficiency ordering is denominator-dependent, and the denominator "
          "the field reports is not the one a deployment pays. The screenshot channel's "
          "per-attempt lead is universal and by construction; its per-success lead is not "
          "universal. The fused channel's per-attempt cost penalty is real and its per-success "
          "latency position is much better than that penalty suggests.", "",
          "**Not licensed.** Any statement of the form \"mode X is more efficient\" without a "
          "denominator. Also any cross-cell comparison of these ratios: B0 bills an API and "
          "B1/B2 are electricity-derived, so only within-cell ordering is meaningful.", "",
          "⚠️ These ratios inherit the success rate's noise twice over — once in the estimate and "
          "once in the fact that success itself moves 0.89–2.23pp between identical reruns "
          "(`noise_floor_inventory`). The CIs above capture the first, not the second."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    d = build()
    OUT_JSON.write_text(json.dumps(d, indent=2))
    OUT_MD.write_text(render(d))
    print(f"✓ {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
