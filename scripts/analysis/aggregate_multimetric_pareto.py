#!/usr/bin/env python3
"""Does the Pareto verdict survive adding latency as a third axis?

`_status/audit/audit_s8_multimetric_pareto.md` has carried this as open since section 8 was
planned. §5.3 reports non-dominance on (success, cost) only, and a cost-aware paper that
reports one cost axis invites the obvious question.

Latency is a genuinely independent axis here, not a restatement of cost. Across modes within a
cell it spans 1.12x to 1.40x against cost's 1.12x to 1.63x, and the two disagree in direction:
on classifieds B0 the cheapest mode is also the slowest. Tokens are NOT independent, being what
the bill is computed from, and are reported only to show that.

An earlier note in `next_steps` excluded B0 from this analysis on the grounds that its latency
runs through a shared API proxy whose queueing might dominate. That was asserted, then measured,
then withdrawn: B0's per-step latency has coefficient of variation 0.15-0.22 against the locally
served B1's 0.11-0.19, and it tracks tokens per step monotonically. All six cells are used.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("multimetric_pareto")
SRC = REPO / "docs/analysis/cross_sites/per_mode_four_dimension_profile.md"
OUT_MD = REPO / "docs/analysis/cross_sites/multimetric_pareto.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/multimetric_pareto.json"

MODES = ["DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM"]
# (index into the tuple, +1 if higher is better)
AXES = {
    "success x cost":            [(0, +1), (1, -1)],
    "+ latency":                 [(0, +1), (1, -1), (2, -1)],
    "+ tokens (not independent)": [(0, +1), (1, -1), (2, -1), (3, -1)],
}


class MissingInput(RuntimeError):
    """Fail loud rather than emit a table built on a partial parse."""


def parse_profile() -> dict[str, dict[str, tuple]]:
    """Read per-cell (SR, cost, latency, tokens) out of the four-dimension profile.

    Single-sourced from that document rather than recomputed, so the two can never disagree;
    if its layout changes this raises instead of silently returning fewer cells.
    """
    if not SRC.exists():
        raise MissingInput(f"{SRC} not found; run aggregate_per_mode_four_dimension_profile first")
    cells: dict[str, dict[str, list]] = {}
    section, cell = None, None
    for line in SRC.read_text().splitlines():
        s = line.strip()
        if s.startswith("## "):
            section = s[3:].strip()
        elif s.startswith("### ") and section in ("Outcome", "Efficiency"):
            cell = s[4:].strip()
            cells.setdefault(cell, {})
        elif s.startswith("|") and cell and section in ("Outcome", "Efficiency"):
            parts = [c.strip() for c in s.strip("|").split("|")]
            if len(parts) != 7 or parts[0] in ("metric", "---"):
                continue
            key = parts[0]
            wanted = {"success rate %": "sr", "billed cost / episode": "cost",
                      "latency / episode (s)": "lat", "tokens / episode": "tok"}
            if key in wanted:
                try:
                    cells[cell][wanted[key]] = [float(x) for x in parts[1:]]
                except ValueError:
                    raise MissingInput(f"{cell}/{key}: unparsable row {parts[1:]}")
    out = {}
    for cell, d in cells.items():
        missing = {"sr", "cost", "lat", "tok"} - set(d)
        if missing:
            raise MissingInput(f"{cell}: missing {sorted(missing)} in the profile")
        out[cell] = {m: (d["sr"][i], d["cost"][i], d["lat"][i], d["tok"][i])
                     for i, m in enumerate(MODES)}
    if len(out) != 6:
        raise MissingInput(f"expected 6 cells in the profile, parsed {len(out)}")
    return out


def frontier(cell: dict[str, tuple], axes) -> list[str]:
    nd = []
    for a in cell:
        if not any(a != b
                   and all((cell[b][i] - cell[a][i]) * d >= 0 for i, d in axes)
                   and any((cell[b][i] - cell[a][i]) * d > 0 for i, d in axes)
                   for b in cell):
            nd.append(a)
    return nd


def build() -> dict:
    data = parse_profile()
    res = {"schema": "2026-08-02-multimetric-pareto-v1", "cells": {}}
    for cell, modes in data.items():
        spans = {
            "cost": max(v[1] for v in modes.values()) / min(v[1] for v in modes.values()),
            "latency": max(v[2] for v in modes.values()) / min(v[2] for v in modes.values()),
        }
        cheapest = min(modes, key=lambda m: modes[m][1])
        fastest = min(modes, key=lambda m: modes[m][2])
        res["cells"][cell] = {
            "spans": spans,
            "cheapest": cheapest, "fastest": fastest,
            "cheapest_is_fastest": cheapest == fastest,
            "frontiers": {name: frontier(modes, ax) for name, ax in AXES.items()},
        }
        LOG.info("%s: cost %.2fx lat %.2fx | %s -> %s", cell, spans["cost"], spans["latency"],
                 res["cells"][cell]["frontiers"]["success x cost"],
                 res["cells"][cell]["frontiers"]["+ latency"])
    return res


def render(d: dict) -> str:
    L = ["---", "type: analysis", "status: complete", "created: 2026-08-02",
         "purpose: does the (success, cost) Pareto verdict survive adding latency as a third axis",
         "scope_warning: within-cell only. B0 reports an API bill and B1/B2 an electricity-derived "
         "figure, so no quantity here is comparable across backbones.",
         "producer: scripts/analysis/aggregate_multimetric_pareto.py "
         "(single-sources per_mode_four_dimension_profile.md)", "---", "",
         "# Multi-metric Pareto", "",
         "Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_multimetric_pareto.py`", "",
         "## 1. Is latency an independent axis?", "",
         "| cell | cost span | latency span | cheapest mode | fastest mode | same? |",
         "|---|---|---|---|---|---|"]
    for cell, c in d["cells"].items():
        L.append(f"| `{cell}` | {c['spans']['cost']:.2f}x | {c['spans']['latency']:.2f}x | "
                 f"{c['cheapest']} | {c['fastest']} | {'yes' if c['cheapest_is_fastest'] else '**no**'} |")
    n_diff = sum(1 for c in d["cells"].values() if not c["cheapest_is_fastest"])
    L += ["", f"Latency spans the same order of magnitude as cost, and in **{n_diff} of "
          f"{len(d['cells'])}** cells the cheapest mode is not the fastest. It is a second axis, "
          "not a restatement of the first.", "",
          "## 2. What adding it does to the frontier", "",
          "| cell | success x cost | + latency | + tokens |", "|---|---|---|---|"]
    grew = 0
    for cell, c in d["cells"].items():
        f = c["frontiers"]
        a, b = f["success x cost"], f["+ latency"]
        grew += len(b) > len(a)
        L.append(f"| `{cell}` | {', '.join(a)} ({len(a)}) | **{', '.join(b)} ({len(b)})** | "
                 f"{len(f['+ tokens (not independent)'])} |")
    L += ["", f"The frontier grows in **{grew} of {len(d['cells'])}** cells. This cuts both ways "
          "and the paper should say so. Pareto *dominance* becomes strictly harder to achieve "
          "with a third axis, so §5.3's negative result (no learned policy dominates a fixed one) "
          "holds a fortiori. But non-dominance becomes correspondingly cheaper to satisfy, so "
          "wherever the paper reports non-dominance as informative it should be read against a "
          "frontier this wide.", "",
          "Adding tokens changes nothing beyond what latency already changed, which is expected: "
          "the bill is computed from tokens, so that column is a check rather than an axis."]
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
