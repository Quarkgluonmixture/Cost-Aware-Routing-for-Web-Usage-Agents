#!/usr/bin/env python3
"""Whether the supply of routing labels and the value of routing move together.

The lower bound (paper section 5) dies of label supply; the upper bound (section 4)
is sized by the routable set. This asks whether those two are one quantity wearing
two hats, because if they are, the paper's negative result is indexed to the current
capability regime rather than being a permanent property of routing.

Reads `routing_ceiling.json` only. Adds no estimand: every input column already
exists there and is already rendered in Table 42.

Emits `supply_value_coupling.{json,md}` in the cross-site analysis directory.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CROSS = REPO / "docs" / "analysis" / "cross_sites"
CEILING = CROSS / "routing_ceiling.json"


class MissingInput(RuntimeError):
    pass


def _rank(values):
    """Average-tie ranks, so a tie cannot silently pick an ordering."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def _spearman(xs, ys):
    return _pearson(_rank(xs), _rank(ys))


def _permutation_p(xs, ys, stat_fn):
    """Exact two-sided permutation p over all 8! pairings.

    8! = 40320, so the exact null is cheap and there is no sampling error to
    report. The null is 'the pairing of the two columns across cells is
    arbitrary' -- it does NOT model the cells as independent draws, which they
    are not (cells share sites and share backbones).
    """
    observed = abs(stat_fn(xs, ys))
    n_total = n_extreme = 0
    for perm in itertools.permutations(ys):
        n_total += 1
        if abs(stat_fn(xs, list(perm))) >= observed - 1e-12:
            n_extreme += 1
    return n_extreme / n_total, n_total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ceiling", type=Path, default=CEILING)
    ap.add_argument("--out-dir", type=Path, default=CROSS)
    args = ap.parse_args()

    if not args.ceiling.exists():
        raise MissingInput(f"{args.ceiling} missing; run aggregate_routing_ceiling.py first")
    doc = json.loads(args.ceiling.read_text())

    # MUST match whatever the ceiling table renders, or the same cell carries two
    # success rates in one paper (red_B2 is 3.94% kept and 2.46% zeroed). The paper's
    # main analysis keeps the leaked successes and puts the zeroing in a sensitivity
    # table, because the detection criterion is a lower bound and a lower bound cannot
    # license a corrected point estimate.
    policy = "leak_kept"
    rows = []
    for cell in doc["cells"]:
        block = cell.get(policy)
        if block is None:
            raise MissingInput(f"{cell['cell']}: no '{policy}' block in routing_ceiling.json")
        for field in ("best_sr_pct", "multi_solver_share_pct", "oracle_sr_pct", "n_multi_solver"):
            if field not in block:
                raise MissingInput(f"{cell['cell']}: routing_ceiling.json lost '{field}'")
        rows.append(
            {
                "cell": cell["cell"],
                "n_tasks": block["n_tasks"],
                "best_mode": block["best_mode"],
                "best_sr_pct": block["best_sr_pct"],
                "routable_share_pct": block["multi_solver_share_pct"],
                "n_routable": block["n_multi_solver"],
                "solvable_share_pct": block["oracle_sr_pct"],
                # the share of the solvable set that carries a routing choice at all
                "routable_over_solvable": (
                    block["multi_solver_share_pct"] / block["oracle_sr_pct"]
                    if block["oracle_sr_pct"]
                    else float("nan")
                ),
            }
        )
    rows.sort(key=lambda r: -r["best_sr_pct"])

    best = [r["best_sr_pct"] for r in rows]
    routable = [r["routable_share_pct"] for r in rows]
    ratios = [r["routable_over_solvable"] for r in rows]

    above_floor = [r["routable_over_solvable"] for r in rows if r["n_routable"] >= 5]
    rho = _spearman(best, routable)
    r_pearson = _pearson(best, routable)
    p_rho, n_perm = _permutation_p(best, routable, _spearman)
    mean_abs_gap = sum(abs(a - b) for a, b in zip(best, routable)) / len(rows)

    out = {
        "generated_from": str(args.ceiling.relative_to(REPO)),
        "leak_policy": policy,
        "n_cells": len(rows),
        "cells": rows,
        "spearman_rho": rho,
        "pearson_r": r_pearson,
        "permutation_p_two_sided": p_rho,
        "permutation_n": n_perm,
        "mean_abs_gap_pp": mean_abs_gap,
        "routable_over_solvable_min": min(ratios),
        "routable_over_solvable_max": max(ratios),
        # The two near-floor cells carry 3 and 4 routable tasks, so their ratio is
        # one or two tasks wide. Report the rest separately rather than quoting a
        # single range that a 3-task cell dominates.
        "n_routable_floor_threshold": 5,
        "routable_over_solvable_above_floor": [
            r["routable_over_solvable"] for r in rows if r["n_routable"] >= 5
        ],
        "cells_at_floor": [r["cell"] for r in rows if r["n_routable"] < 5],
        "caveats": [
            "The 8 cells are not independent: they share 3 sites and 3 backbones, so "
            "the permutation null tests whether the pairing is arbitrary, not whether "
            "8 independent systems were sampled.",
            "Both columns are functionals of the same per-task solve matrix, and both "
            "are subsets of the solvable set. Part of the association is therefore "
            "structural, not empirical. The non-structural content is the SIZE of the "
            "ratio routable/solvable, which is 0.53-0.71 above the floor and is NOT "
            "constant -- it halves in the two near-floor cells.",
            "Under mode independence the routable share would be a convex (near-"
            "quadratic at low success) function of the per-mode rate, not the near-"
            "identity observed here; the observed near-linearity is a statement about "
            "how strongly task difficulty dominates mode-task matching.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "supply_value_coupling.json").write_text(json.dumps(out, indent=2))

    lines = [
        "---",
        "type: analysis",
        "status: complete",
        "purpose: whether the supply of routing labels and the value of routing are one quantity",
        "post_hoc_exploratory: true",
        f"producer: scripts/analysis/{Path(__file__).name}",
        "---",
        "",
        "# Supply and value are the same set",
        "",
        f"Regenerate: `.venv/bin/python3 scripts/analysis/{Path(__file__).name}`",
        "",
        "Both quantities below are already in Table 42. This asks only whether they move together.",
        "",
        "| cell | n | best single mode | best SR | routable set (>1 solver) | solvable (oracle) | routable / solvable |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['cell']}` | {r['n_tasks']} | {r['best_mode']} | {r['best_sr_pct']:.2f}% | "
            f"{r['routable_share_pct']:.1f}% ({r['n_routable']}) | {r['solvable_share_pct']:.1f}% | "
            f"{r['routable_over_solvable']:.2f} |"
        )
    lines += [
        "",
        f"**Spearman rho = {rho:.3f}** (exact permutation p = {p_rho:.4f} over {n_perm:,} pairings), "
        f"Pearson r = {r_pearson:.3f}, mean |best SR - routable share| = **{mean_abs_gap:.2f}pp**.",
        "",
        f"The routable set is **{min(above_floor):.2f}-{max(above_floor):.2f}** of the solvable set "
        f"in the {len(above_floor)} cells carrying more than 4 routable tasks. The two near-floor "
        f"cells ({', '.join('`'+c+'`' for c in out['cells_at_floor'])}) sit at "
        f"{min(r['routable_over_solvable'] for r in rows if r['n_routable'] < 5):.2f} on **3 and 4 "
        f"tasks** -- a ratio one task wide, quoted separately rather than widening the range.",
        "",
        "## What this licenses, and what it does not",
        "",
        "**Licensed.** The set a router can learn from and the set where routing can pay are the "
        "same set. So the two obstructions the lower bound reports -- too few labels, too few "
        "contested tasks -- are not two independent walls; they are one wall, whose height is set "
        "by how many tasks are solvable at all.",
        "",
        "**Not licensed: reading this as a surprising empirical law.** Both columns are functionals "
        "of one solve matrix and both are subsets of the solvable set, so a positive association is "
        "partly structural. The empirical content is the *near-linearity*: under mode independence "
        "the routable share would grow near-quadratically in the per-mode rate at these success "
        "levels, and it does not.",
        "",
        "**Not licensed: a constant conversion factor.** The ratio is not flat -- it falls to "
        "0.25 in the two near-floor cells. Whether that is a floor effect on 3-4 tasks or a real "
        "bend at low capability cannot be told apart here, and the direction matters: if real, the "
        "routable set shrinks *faster* than the solvable set as capability drops.",
        "",
        "**Not licensed: a forecast.** That both quantities rise with capability is what the "
        "coupling implies; how fast, and whether the ratio holds outside 2-36% success, is not "
        "measured here. It is stated in the paper as the condition under which the negative result "
        "would be overturned, i.e. as a falsifiable prediction, not as a projection.",
        "",
        "## Caveats",
        "",
    ]
    lines += [f"- {c}" for c in out["caveats"]]
    (args.out_dir / "supply_value_coupling.md").write_text("\n".join(lines) + "\n")

    print(f"rho={rho:.4f} (p={p_rho:.4f})  r={r_pearson:.4f}  mean_gap={mean_abs_gap:.2f}pp")
    print(f"routable/solvable in [{min(ratios):.2f}, {max(ratios):.2f}]")
    print(f"wrote {args.out_dir/'supply_value_coupling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
