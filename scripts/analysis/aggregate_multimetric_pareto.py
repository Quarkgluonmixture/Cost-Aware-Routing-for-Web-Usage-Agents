#!/usr/bin/env python3
"""Does the Pareto verdict survive adding latency as a third axis?

`_status/audit/audit_s8_multimetric_pareto.md` has carried this as open since section 8 was
planned. §5.3 reports non-dominance on (success, cost) only, and a cost-aware paper that
reports one cost axis invites the obvious question.

Carbon was considered as a fifth axis and rejected on measurement, not on principle. The four
locally-served cells do log `co2e_kg` on every step, but the recorded `source` is
`psutil_profile` rather than NVML, so the figure is a host-level estimate: it averages 66W at a
coefficient of variation of 0.03 on a device rated several times that, and it correlates with
wall-clock latency at r = 0.9999 in all four cells. A near-constant power estimate times elapsed
time is latency in other units. The two API-served cells have no local draw at all
(`source: disabled`), so a carbon axis would also exist on only four of six cells.

Latency carries information the cost axis does not, but the evidence for that is NOT frontier
growth. Adding an axis can only weakly enlarge a Pareto frontier; permuting latency across the
six modes widens it with probability 0.75-0.83 per cell, so the observed 3-of-6 is below chance
(§2). The evidence is the ordering: mean Spearman rho(cost, latency) = -0.095, negative on all
three classifieds cells, and the cheapest mode differs from the fastest in exactly the three
classifieds cells - a split that follows the site rather than the backbone. Tokens are NOT
independent, being what the bill is computed from, and are reported only to show that.

An earlier note in `next_steps` excluded B0 from this analysis on the grounds that its latency
runs through a shared API proxy whose queueing might dominate. That was asserted, then measured,
then withdrawn: B0's per-step latency has coefficient of variation 0.15-0.22 against the locally
served B1's 0.11-0.19, and it tracks tokens per step monotonically. All six cells are used.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import statistics as st
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


def parse_profile(src: Path = SRC, n_expected: int = 6) -> dict[str, dict[str, tuple]]:
    """Read per-cell (SR, cost, latency, tokens) out of the four-dimension profile.

    Single-sourced from that document rather than recomputed, so the two can never disagree;
    if its layout changes this raises instead of silently returning fewer cells.
    """
    if not src.exists():
        raise MissingInput(f"{src} not found; run per_mode_four_dimension_profile.py first "
                           "(with --with-wa if you asked for the seven-cell variant)")
    cells: dict[str, dict[str, list]] = {}
    section, cell = None, None
    for line in src.read_text().splitlines():
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
                      "latency / episode (s)": "lat",
                      # P1-4 (§H stress): the canonical estimand was tested by hand and the
                      # result written into prose, but no producer read the column, so the
                      # robustness statement could not be regenerated. It is read here now.
                      "latency canonical / episode (s)": "lat_canon",
                      "tokens / episode": "tok"}
            if key in wanted:
                try:
                    cells[cell][wanted[key]] = [float(x) for x in parts[1:]]
                except ValueError:
                    raise MissingInput(f"{cell}/{key}: unparsable row {parts[1:]}")
    out = {}
    for cell, d in cells.items():
        missing = {"sr", "cost", "lat", "tok", "lat_canon"} - set(d)
        if missing:
            raise MissingInput(f"{cell}: missing {sorted(missing)} in the profile")
        out[cell] = {m: (d["sr"][i], d["cost"][i], d["lat"][i], d["tok"][i], d["lat_canon"][i])
                     for i, m in enumerate(MODES)}
    if len(out) != n_expected:
        raise MissingInput(f"expected {n_expected} cells in {src.name}, parsed {len(out)}")
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


def _rank(v: list[float]) -> list[int]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for pos, i in enumerate(order):
        r[i] = pos
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
    return 1 - 6 * d2 / (n * (n * n - 1))


def latency_independence(modes: dict[str, tuple]) -> dict:
    """Is latency a second axis, or cost restated? Two quantities, one of them a trap.

    The trap is frontier growth. Adding an axis can only weakly ENLARGE a Pareto frontier, and
    six modes give five chances for a dominated mode to escape. Permuting latency across the six
    modes within a cell — all 6! = 720 assignments, the exact null — widens the frontier with
    probability 0.75-0.83 per cell. So "the frontier grew in 3 of 6 cells" is BELOW what chance
    produces (expected 4.70/6, P(>=3 of 6) = 0.978) and is not evidence of anything.
    (codex Mode B, §H stress 2026-08-02. The permutation control was listed as an open cheap
    check in EVIDENCE_LAYER_SUMMARY §6 and then run.)

    What does carry information is whether the cost ordering and the latency ordering agree.
    """
    names = list(modes)
    cost = [modes[m][1] for m in names]
    lat = [modes[m][2] for m in names]
    canon = [modes[m][4] for m in names]
    rho = spearman(cost, lat)
    ge = sum(1 for p in itertools.permutations(lat) if spearman(cost, list(p)) >= rho)
    return {"spearman_rho_cost_latency": rho,
            "exact_p_one_sided": ge / math.factorial(len(names)),
            # Same quantity on the canonical estimand (retry / busy-wait / recovered-screenshot
            # subtracted). Identical to the raw figure on the locally-served cells by
            # construction, so only the API-served cells actually test it.
            "spearman_rho_cost_latency_canonical": spearman(cost, canon),
            # Tolerance, not equality: the profile prints raw latency at 2 decimals and the
            # canonical column at 4 (308.93 vs 308.9298), so an exact test calls every cell
            # different and hides the fact that four of them are identical by construction.
            "canonical_equals_raw": all(abs(a - b) <= 0.011 for a, b in zip(lat, canon)),
            "n_modes": len(names)}


def frontier_growth_null(modes: dict[str, tuple]) -> dict:
    """P(the +latency frontier is larger than the success x cost frontier) under permuted latency."""
    names = list(modes)
    base = len(frontier(modes, AXES["success x cost"]))
    lat = [modes[m][2] for m in names]
    hits = 0
    for perm in itertools.permutations(lat):
        mm = {m: (modes[m][0], modes[m][1], perm[i], modes[m][3], modes[m][4])
              for i, m in enumerate(names)}
        hits += len(frontier(mm, AXES["+ latency"])) > base
    n = math.factorial(len(names))
    return {"p_widen_under_null": hits / n, "n_permutations": n,
            "observed_widen": len(frontier(modes, AXES["+ latency"])) > base}


def build(src: Path = SRC, n_expected: int = 6) -> dict:
    data = parse_profile(src, n_expected)
    res = {"schema": "2026-08-03-multimetric-pareto-v2", "cells": {}}
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
            "independence": latency_independence(modes),
            "frontier_null": frontier_growth_null(modes),
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
         "Two claims are separable here and only one needs our data. That a deployment may care "
         "about wall-clock rather than tokens is an industry fact, not something to demonstrate. "
         "What needs evidence is narrower: **in these runs, is the latency ordering something "
         "other than the cost ordering restated?**", "",
         "| cell | cost span | latency span | cheapest | fastest | same? | ρ(cost, latency) |",
         "|---|---|---|---|---|---|---|"]
    for cell, c in d["cells"].items():
        ind = c["independence"]
        L.append(f"| `{cell}` | {c['spans']['cost']:.2f}x | {c['spans']['latency']:.2f}x | "
                 f"{c['cheapest']} | {c['fastest']} | "
                 f"{'yes' if c['cheapest_is_fastest'] else '**no**'} | "
                 f"{ind['spearman_rho_cost_latency']:+.3f} |")
    n_diff = sum(1 for c in d["cells"].values() if not c["cheapest_is_fastest"])
    rhos = [c["independence"]["spearman_rho_cost_latency"] for c in d["cells"].values()]
    mean_rho = st.fmean(rhos)
    splits = sorted(cell for cell, c in d["cells"].items() if not c["cheapest_is_fastest"])
    L += ["", f"Mean **ρ = {mean_rho:+.3f}** over {len(rhos)} cells — the two orderings are close "
          f"to uncorrelated, and on the classifieds cells they run *opposite*: the cheapest mode "
          f"is the slowest. The cheapest mode is not the fastest in **{n_diff} of "
          f"{len(d['cells'])}** cells, and those cells are `{'`, `'.join(splits)}` — the split "
          "follows the **site**, not the backbone.", "",
          f"**Under the canonical latency estimand** (retry, busy-wait and recovered-screenshot "
          f"subtracted) the mean is ρ = "
          f"{st.fmean(c['independence']['spearman_rho_cost_latency_canonical'] for c in d['cells'].values()):+.3f}. "
          f"⚠️ That agreement is weaker evidence than it looks: the two estimands are "
          f"**identical by construction** on "
          f"{sum(1 for c in d['cells'].values() if c['independence']['canonical_equals_raw'])} of "
          f"{len(d['cells'])} cells (the locally-served ones have no retry, busy-wait or "
          f"screenshot-timeout to subtract), so only the API-served cells test it at all.", "",
          "⚠️ Per-cell exact permutation p-values on ρ are not significant (six modes give a "
          "Spearman test almost no power), so this is a descriptive structure and not a test. "
          "The cross-cell regularity is what carries it: three classifieds cells all put "
          "`Vision` cheapest and `SoM` fastest, three reddit cells all put `Vision` at both.", "",
          "## 2. Why the frontier count is NOT the evidence for §1", "",
          "An earlier version of this document argued §1 from frontier growth — the frontier "
          "widened in 3 of 6 cells when latency was added. **That argument is void**, and the "
          "control that kills it is exact rather than approximate. Adding an axis can only "
          "weakly enlarge a Pareto frontier, and six modes give five chances for a dominated "
          "mode to escape. Permuting latency across the modes within each cell (all "
          f"{d['cells'][next(iter(d['cells']))]['frontier_null']['n_permutations']} assignments) "
          "gives:", "",
          "| cell | frontier widened? | P(widen) under permuted latency |", "|---|---|---|"]
    ps = []
    for cell, c in d["cells"].items():
        fn = c["frontier_null"]
        ps.append(fn["p_widen_under_null"])
        L.append(f"| `{cell}` | {'yes' if fn['observed_widen'] else 'no'} | "
                 f"{fn['p_widen_under_null']:.3f} |")
    # Poisson-binomial tail: P(at least k of n widen) under the null
    dp = [1.0]
    for p in ps:
        nd = [0.0] * (len(dp) + 1)
        for i, v in enumerate(dp):
            nd[i] += v * (1 - p)
            nd[i + 1] += v * p
        dp = nd
    obs = sum(1 for c in d["cells"].values() if c["frontier_null"]["observed_widen"])
    L += ["", f"Expected widened cells under the null: **{sum(ps):.2f} of {len(ps)}**. Observed: "
          f"**{obs}**. P(at least {obs} widen | null) = **{sum(dp[obs:]):.3f}**. The observed "
          "count is *below* chance, so frontier growth carries no information about whether "
          "latency is independent. It is reported here only so nobody reconstructs the "
          "retracted argument.", "",
          "## 3. What the frontier count still legitimately says", "",
          "| cell | success x cost | + latency | + tokens |", "|---|---|---|---|"]
    tok_changed = 0
    for cell, c in d["cells"].items():
        f = c["frontiers"]
        a, b, t = f["success x cost"], f["+ latency"], f["+ tokens (not independent)"]
        tok_changed += len(t) > len(b)
        L.append(f"| `{cell}` | {', '.join(a)} ({len(a)}) | **{', '.join(b)} ({len(b)})** | "
                 f"{len(t)} |")
    L += ["", "Read as *width*, not as evidence: Pareto dominance is strictly harder to achieve "
          "against three axes, so §5.3's negative result (no learned policy dominates a fixed "
          "one) holds a fortiori. Non-dominance becomes correspondingly cheaper to satisfy, so "
          "wherever the paper treats non-dominance as informative it must be read against a "
          "frontier this wide.", ""]
    if tok_changed:
        L.append(f"Adding tokens enlarges the frontier further in **{tok_changed}** cell(s). "
                 "The bill is computed from tokens, so this column is a consistency check rather "
                 "than an axis — but the earlier claim that it 'changes nothing beyond latency' "
                 "was false against this producer's own table.")
    else:
        L.append("Adding tokens changes nothing beyond what latency already changed, which is "
                 "expected: the bill is computed from tokens, so that column is a check rather "
                 "than an axis.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-wa", action="store_true",
                    help="read the WA-inclusive profile (6 VWA + every WA cell) and write to "
                         "*_with_wa.*. WA has no AMENDMENT_08; B1xWA cost is electricity-derived "
                         "like the other local cells while B0xWA is API-billed, so within-cell "
                         "comparison holds and cross-cell does not, same as everywhere")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    src, n, om, oj = SRC, 6, OUT_MD, OUT_JSON
    if a.with_wa:
        src = SRC.with_name(SRC.stem + "_with_wa" + SRC.suffix)
        # Expected cell count is read from the profile's own JSON rather than written as a
        # literal here: WA went from one cell to two on 2026-08-03 (B0xWA landed), and a
        # hardcoded 7 would have failed loudly — which it did. The check still fails loud,
        # it just compares the Markdown against its own producer instead of against a guess.
        pj = src.with_suffix(".json")
        if not pj.is_file():
            raise MissingInput(f"profile JSON absent: {pj}")
        n = len(json.loads(pj.read_text())["cells"])
        om = OUT_MD.with_name(OUT_MD.stem + "_with_wa" + OUT_MD.suffix)
        oj = OUT_JSON.with_name(OUT_JSON.stem + "_with_wa" + OUT_JSON.suffix)
    d = build(src, n)
    oj.write_text(json.dumps(d, indent=2))
    om.write_text(render(d))
    print(f"✓ {om.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
