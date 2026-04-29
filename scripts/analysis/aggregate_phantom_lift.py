#!/usr/bin/env python3
"""Aggregate phantom routing lift across (baseline, site) cells.

For each cell with all 5 modes (DOM / SoM / Vision / P-DOM / P-SoM) present:
  - Compute 3-mode oracle ceiling (DOM ∪ SoM ∪ Vision)
  - Compute 5-mode oracle ceiling (+ P-DOM + P-SoM)
  - Routing lift = 5-mode - 3-mode oracle SR (pp)
  - 95% bootstrap CI on lift (n=1000 task resamples)
  - Decomposition: P-DOM-only, P-SoM-only, both-add-same contributions

Outputs (results/phantom_paper/):
  - phantom_lift.csv          (one row per (baseline, site, decomposition))
  - phantom_lift_summary.md   (paper-ready table for Section 1/4 hook)

Usage:
    python3 scripts/analysis/aggregate_phantom_lift.py [--cells <baseline:site:...>]

Default cells = paper-grade clean B0 cls/red. B1 cells included if all 5 modes
have data (>= 50 ep each). Partial cells use the common observed-task universe.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results/visualwebarena/phase1"


# Cell registry: (baseline, site, expected_N, run_paths_per_mode)
CELLS = [
    {
        "baseline": "B0", "site": "classifieds", "n_expected": 234,
        "modes": {
            "DOM":   RES/"B0_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
            "SoM":   RES/"B0_3mode_classifieds_20260413/phase1_som_router_0/episodes",
            "Vision":RES/"B0_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
            "P-DOM": RES/"B0_phantom_dom_classifieds_20260427/phase1_phantom_dom_router_0/episodes",
            "P-SoM": RES/"B0_phantom_classifieds_20260426/phase1_phantom_som_router_0/episodes",
        },
    },
    {
        "baseline": "B0", "site": "reddit", "n_expected": 210,
        "modes": {
            "DOM":   RES/"B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
            "SoM":   RES/"B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
            "Vision":RES/"B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
            "P-DOM": RES/"B0_phantom_dom_reddit_20260427/phase1_phantom_dom_router_0/episodes",
            "P-SoM": RES/"B0_phantom_reddit_20260428/phase1_phantom_som_router_0/episodes",
        },
    },
    # B1 cells — partial / pending (chain in flight)
    {
        "baseline": "B1", "site": "classifieds", "n_expected": 234,
        "modes": {
            "DOM":   RES/"B1_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
            "SoM":   RES/"B1_3mode_classifieds_20260413/phase1_som_router_0/episodes",
            "Vision":RES/"B1_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
            "P-DOM": RES/"B1_phantom_dom_classifieds_20260428/phase1_phantom_dom_router_0/episodes",
            "P-SoM": RES/"B1_phantom_classifieds_20260428/phase1_phantom_som_router_0/episodes",
        },
    },
    {
        "baseline": "B1", "site": "reddit", "n_expected": 210,
        "modes": {
            "DOM":   RES/"B1_3mode_reddit_20260413/phase1_dom_router_0/episodes",
            "SoM":   RES/"B1_3mode_reddit_20260413/phase1_som_router_0/episodes",
            "Vision":RES/"B1_3mode_reddit_20260413/phase1_vision_router_0/episodes",
            "P-DOM": RES/"B1_phantom_dom_reddit_20260428/phase1_phantom_dom_router_0/episodes",
            "P-SoM": RES/"B1_phantom_reddit_20260428/phase1_phantom_som_router_0/episodes",
        },
    },
]

MIN_EP_FOR_CELL = 50  # skip cells where any mode has < 50 ep (too partial)


def load(d: Path) -> tuple[set[int], set[int]]:
    """Returns (succ_set, observed_set)."""
    s, o = set(), set()
    if not d.exists():
        return s, o
    for p in sorted(d.glob("*_summary_v2.json")):
        m = re.search(r"task_(\d+)", p.name)
        if not m:
            continue
        tid = int(m.group(1))
        o.add(tid)
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        if rec.get("adjusted_success", rec.get("success", False)):
            s.add(tid)
    return s, o


def bootstrap_lift_ci(in_3: np.ndarray, in_5: np.ndarray, B: int = 1000, seed: int = 42
                      ) -> tuple[float, float]:
    """Bootstrap 95% CI on (5-mode oracle SR - 3-mode oracle SR)."""
    n = len(in_3)
    rng = np.random.default_rng(seed)
    lifts = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        lifts[b] = 100 * (int(in_5[idx].sum()) - int(in_3[idx].sum())) / n
    return float(np.quantile(lifts, 0.025)), float(np.quantile(lifts, 0.975))


def analyze_cell(cell: dict) -> Optional[dict]:
    """Compute phantom lift for a single (baseline, site) cell.

    Returns None if cell incomplete (any mode missing or <MIN_EP_FOR_CELL ep).
    """
    succ, obs = {}, {}
    for mode, ep_dir in cell["modes"].items():
        s, o = load(ep_dir)
        if len(o) < MIN_EP_FOR_CELL:
            return None
        succ[mode] = s
        obs[mode] = o

    # Common observed universe (intersection across all 5 modes)
    common = set.intersection(*obs.values())
    n = len(common)
    if n < MIN_EP_FOR_CELL:
        return None

    # Restrict each mode's success set to common universe
    succ_r = {m: s & common for m, s in succ.items()}

    # 3-mode and 5-mode oracle unions on common universe
    union_3 = succ_r["DOM"] | succ_r["SoM"] | succ_r["Vision"]
    union_5 = union_3 | succ_r["P-DOM"] | succ_r["P-SoM"]
    sr_3 = 100 * len(union_3) / n
    sr_5 = 100 * len(union_5) / n
    lift_pp = sr_5 - sr_3

    # Bootstrap CI
    universe = sorted(common)
    in_3 = np.array([t in union_3 for t in universe], dtype=bool)
    in_5 = np.array([t in union_5 for t in universe], dtype=bool)
    ci_lo, ci_hi = bootstrap_lift_ci(in_3, in_5)

    # Decomposition
    pdom_adds = succ_r["P-DOM"] - union_3
    psom_adds = succ_r["P-SoM"] - union_3
    both_add = pdom_adds & psom_adds
    pdom_only = pdom_adds - psom_adds
    psom_only = psom_adds - pdom_adds

    is_partial = any(len(o) < cell["n_expected"] for o in obs.values())

    return {
        "baseline": cell["baseline"],
        "site": cell["site"],
        "n_common": n,
        "n_expected": cell["n_expected"],
        "is_partial": is_partial,
        "sr_dom":     round(100 * len(succ_r["DOM"]) / n, 4),
        "sr_som":     round(100 * len(succ_r["SoM"]) / n, 4),
        "sr_vision":  round(100 * len(succ_r["Vision"]) / n, 4),
        "sr_pdom":    round(100 * len(succ_r["P-DOM"]) / n, 4),
        "sr_psom":    round(100 * len(succ_r["P-SoM"]) / n, 4),
        "oracle_3mode_pp":  round(sr_3, 4),
        "oracle_5mode_pp":  round(sr_5, 4),
        "lift_pp":          round(lift_pp, 4),
        "lift_ci95_lo_pp":  round(ci_lo, 4),
        "lift_ci95_hi_pp":  round(ci_hi, 4),
        "pdom_adds_count":      len(pdom_adds),
        "psom_adds_count":      len(psom_adds),
        "pdom_only_count":      len(pdom_only),
        "psom_only_count":      len(psom_only),
        "both_phantom_overlap_count": len(both_add),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(REPO / "results/phantom_paper/phantom_lift.csv"))
    args = ap.parse_args()

    rows = []
    skipped = []
    for cell in CELLS:
        r = analyze_cell(cell)
        if r is None:
            skipped.append(f"{cell['baseline']} {cell['site']}")
            continue
        rows.append(r)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} cells)")
    if skipped:
        print(f"skipped (incomplete): {', '.join(skipped)}")

    # Markdown summary
    md = out.with_suffix(".md")
    lines = [
        "# Phantom routing lift — paper Section 1/4 hook evidence",
        "",
        "Routing lift = (5-mode oracle ceiling) - (3-mode oracle ceiling), where",
        "5-mode = DOM ∪ SoM ∪ Vision ∪ Phantom-DOM ∪ Phantom-SoM. CI from",
        "1000-resample task-level bootstrap.",
        "",
        "| Baseline | Site | N | 3-mode oracle | 5-mode oracle | **Lift (pp)** | 95% CI | Significant? |",
        "|---|---|---:|---:|---:|---:|---|:---:|",
    ]
    for r in rows:
        n_label = (f"{r['n_common']}/{r['n_expected']}†" if r["is_partial"]
                   else f"{r['n_common']}")
        sig = "✅" if r["lift_ci95_lo_pp"] > 0 else ("🟡" if r["lift_ci95_hi_pp"] > 0 else "❌")
        lines.append(
            f"| {r['baseline']} | {r['site']} | {n_label} | "
            f"{r['oracle_3mode_pp']:.2f}% | {r['oracle_5mode_pp']:.2f}% | "
            f"**+{r['lift_pp']:.2f}** | "
            f"[{r['lift_ci95_lo_pp']:.2f}, {r['lift_ci95_hi_pp']:.2f}] | {sig} |"
        )
    if any(r["is_partial"] for r in rows):
        lines.append("")
        lines.append("† = partial (any mode < expected N); using intersection of observed tasks.")
    lines += [
        "",
        "## Decomposition: which phantom contributes which tasks",
        "",
        "| Baseline | Site | P-DOM adds | P-SoM adds | P-DOM only | P-SoM only | Both phantoms overlap |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        n = max(r["n_common"], 1)
        lines.append(
            f"| {r['baseline']} | {r['site']} | "
            f"{r['pdom_adds_count']} ({100*r['pdom_adds_count']/n:.2f}pp) | "
            f"{r['psom_adds_count']} ({100*r['psom_adds_count']/n:.2f}pp) | "
            f"{r['pdom_only_count']} ({100*r['pdom_only_count']/n:.2f}pp) | "
            f"{r['psom_only_count']} ({100*r['psom_only_count']/n:.2f}pp) | "
            f"{r['both_phantom_overlap_count']} ({100*r['both_phantom_overlap_count']/n:.2f}pp) |"
        )
    if skipped:
        lines += ["", f"_Cells pending data (skipped): {', '.join(skipped)}_"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
