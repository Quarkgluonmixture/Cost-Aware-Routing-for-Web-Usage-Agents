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
import math
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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


def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size between two proportions p1, p2 ∈ [0, 1].

    h = 2 * (arcsin(√p1) - arcsin(√p2))

    Interpretation: |h|<0.2 small, 0.2-0.5 medium, 0.5-0.8 large, >0.8 huge.
    Sign indicates direction (p1 > p2 → h > 0).
    """
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def cohen_h_label(h: float) -> str:
    a = abs(h)
    if a < 0.2:
        return "small"
    if a < 0.5:
        return "medium"
    if a < 0.8:
        return "large"
    return "huge"


def wilcoxon_signed_rank(in_a: np.ndarray, in_b: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """Wilcoxon signed-rank test on paired binary task outcomes (a vs b).

    For binary outcomes diff ∈ {-1, 0, +1}; scipy drops zero diffs. When set b
    ⊇ set a (e.g. 5-mode oracle ⊇ 3-mode oracle), all non-zero diffs are
    positive (b solves task that a doesn't) → test reduces to one-sided
    binomial (sign test). Returns (statistic, p_two_sided) or (None, None) if
    scipy unavailable / undefined.
    """
    if not HAS_SCIPY:
        return None, None
    diffs = in_b.astype(int) - in_a.astype(int)
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return None, 1.0  # no difference, p = 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = sp_stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return None, None


def mcnemar_exact_one_sided(in_a: np.ndarray, in_b: np.ndarray) -> Optional[float]:
    """McNemar exact one-sided p-value: H1 = b > a (b adds tasks a misses).

    For monotonic case (b ⊇ a), discordant b-only count = sum(b - a > 0),
    a-only count = 0. Exact binomial: p = 0.5^(b_only).
    """
    if not HAS_SCIPY:
        return None
    a = in_a.astype(int); b = in_b.astype(int)
    a_only = int(((a > b)).sum())
    b_only = int(((b > a)).sum())
    n_disc = a_only + b_only
    if n_disc == 0:
        return 1.0
    # one-sided: H1 = b > a
    return float(sp_stats.binom.cdf(a_only, n_disc, 0.5))


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

    # 3-mode, 4-mode (each phantom alone), 5-mode oracle unions
    union_3 = succ_r["DOM"] | succ_r["SoM"] | succ_r["Vision"]
    union_4_pdom = union_3 | succ_r["P-DOM"]
    union_4_psom = union_3 | succ_r["P-SoM"]
    union_5 = union_3 | succ_r["P-DOM"] | succ_r["P-SoM"]
    sr_3 = 100 * len(union_3) / n
    sr_4_pdom = 100 * len(union_4_pdom) / n
    sr_4_psom = 100 * len(union_4_psom) / n
    sr_5 = 100 * len(union_5) / n

    universe = sorted(common)
    in_3        = np.array([t in union_3       for t in universe], dtype=bool)
    in_4_pdom   = np.array([t in union_4_pdom  for t in universe], dtype=bool)
    in_4_psom   = np.array([t in union_4_psom  for t in universe], dtype=bool)
    in_5        = np.array([t in union_5       for t in universe], dtype=bool)

    # Bootstrap CI on lift (5-mode vs 3-mode)
    ci_lo, ci_hi = bootstrap_lift_ci(in_3, in_5)
    # Single-phantom lift CIs
    ci_lo_pdom, ci_hi_pdom = bootstrap_lift_ci(in_3, in_4_pdom)
    ci_lo_psom, ci_hi_psom = bootstrap_lift_ci(in_3, in_4_psom)

    # Cohen's h effect sizes (proportion difference, dimensionless)
    h_5_vs_3        = cohen_h(sr_5 / 100, sr_3 / 100)
    h_4pdom_vs_3    = cohen_h(sr_4_pdom / 100, sr_3 / 100)
    h_4psom_vs_3    = cohen_h(sr_4_psom / 100, sr_3 / 100)

    # Wilcoxon signed-rank (paired binary) — degenerate to sign test for monotonic
    wstat_5, wp_5 = wilcoxon_signed_rank(in_3, in_5)
    wstat_pdom, wp_pdom = wilcoxon_signed_rank(in_3, in_4_pdom)
    wstat_psom, wp_psom = wilcoxon_signed_rank(in_3, in_4_psom)

    # McNemar exact one-sided (b > a; trivially significant if any new tasks added)
    mc_p_5    = mcnemar_exact_one_sided(in_3, in_5)
    mc_p_pdom = mcnemar_exact_one_sided(in_3, in_4_pdom)
    mc_p_psom = mcnemar_exact_one_sided(in_3, in_4_psom)

    # Decomposition
    pdom_adds = succ_r["P-DOM"] - union_3
    psom_adds = succ_r["P-SoM"] - union_3
    both_add = pdom_adds & psom_adds
    pdom_only = pdom_adds - psom_adds
    psom_only = psom_adds - pdom_adds

    # Jaccard P-SoM ↔ P-DOM (Scenario C sentinel — paper Section 5 axis 2 evidence)
    inter = succ_r["P-SoM"] & succ_r["P-DOM"]
    union = succ_r["P-SoM"] | succ_r["P-DOM"]
    jaccard = (len(inter) / len(union)) if union else 0.0
    # Threshold: > 0.7 → P-SoM ≈ P-DOM redundant, paper claim weakens (Scenario C)
    jaccard_warn = jaccard > 0.7

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
        "oracle_4mode_pdom_pp": round(sr_4_pdom, 4),
        "oracle_4mode_psom_pp": round(sr_4_psom, 4),
        "oracle_5mode_pp":  round(sr_5, 4),
        "lift_5_vs_3_pp":   round(sr_5 - sr_3, 4),
        "lift_5_vs_3_ci95_lo_pp":  round(ci_lo, 4),
        "lift_5_vs_3_ci95_hi_pp":  round(ci_hi, 4),
        "lift_4pdom_vs_3_pp":   round(sr_4_pdom - sr_3, 4),
        "lift_4pdom_vs_3_ci95_lo_pp": round(ci_lo_pdom, 4),
        "lift_4pdom_vs_3_ci95_hi_pp": round(ci_hi_pdom, 4),
        "lift_4psom_vs_3_pp":   round(sr_4_psom - sr_3, 4),
        "lift_4psom_vs_3_ci95_lo_pp": round(ci_lo_psom, 4),
        "lift_4psom_vs_3_ci95_hi_pp": round(ci_hi_psom, 4),
        # Effect sizes (Cohen's h on oracle proportions)
        "cohen_h_5_vs_3":     round(h_5_vs_3, 4),
        "cohen_h_5_vs_3_label": cohen_h_label(h_5_vs_3),
        "cohen_h_4pdom_vs_3": round(h_4pdom_vs_3, 4),
        "cohen_h_4pdom_vs_3_label": cohen_h_label(h_4pdom_vs_3),
        "cohen_h_4psom_vs_3": round(h_4psom_vs_3, 4),
        "cohen_h_4psom_vs_3_label": cohen_h_label(h_4psom_vs_3),
        # Wilcoxon (paired sign on binary)
        "wilcoxon_5_vs_3_p":     wp_5,
        "wilcoxon_4pdom_vs_3_p": wp_pdom,
        "wilcoxon_4psom_vs_3_p": wp_psom,
        # McNemar exact 1-sided
        "mcnemar_5_vs_3_p":     mc_p_5,
        "mcnemar_4pdom_vs_3_p": mc_p_pdom,
        "mcnemar_4psom_vs_3_p": mc_p_psom,
        # Decomposition
        "pdom_adds_count":      len(pdom_adds),
        "psom_adds_count":      len(psom_adds),
        "pdom_only_count":      len(pdom_only),
        "psom_only_count":      len(psom_only),
        "both_phantom_overlap_count": len(both_add),
        # Scenario C sentinel: P-SoM ↔ P-DOM Jaccard
        "phantom_pair_jaccard": round(jaccard, 4),
        "phantom_pair_jaccard_warn": jaccard_warn,
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
        "Routing lift = (X-mode oracle ceiling) - (3-mode oracle ceiling), where",
        "3-mode = DOM ∪ SoM ∪ Vision (baseline). 95% CI from 1000-resample",
        "task-level bootstrap. Cohen's h effect size (small <0.2, medium 0.2-0.5,",
        "large 0.5-0.8). Wilcoxon paired (binary, equiv to sign test). McNemar",
        "exact 1-sided (H1: extra mode adds tasks).",
        "",
        "## Routing lift summary (5-mode vs 3-mode + each single phantom)",
        "",
        "| Baseline | Site | N | 3→5-mode lift | 95% CI | Cohen's h | Wilcoxon p | McNemar p | sig? |",
        "|---|---|---:|---:|---|---:|---:|---:|:---:|",
    ]
    for r in rows:
        n_label = (f"{r['n_common']}/{r['n_expected']}†" if r["is_partial"]
                   else f"{r['n_common']}")
        sig = "✅" if r["lift_5_vs_3_ci95_lo_pp"] > 0 else ("🟡" if r["lift_5_vs_3_ci95_hi_pp"] > 0 else "❌")
        wp = f"{r['wilcoxon_5_vs_3_p']:.4f}" if r['wilcoxon_5_vs_3_p'] is not None else "—"
        mp = f"{r['mcnemar_5_vs_3_p']:.4f}" if r['mcnemar_5_vs_3_p'] is not None else "—"
        lines.append(
            f"| {r['baseline']} | {r['site']} | {n_label} | "
            f"+{r['lift_5_vs_3_pp']:.2f}pp | "
            f"[{r['lift_5_vs_3_ci95_lo_pp']:.2f}, {r['lift_5_vs_3_ci95_hi_pp']:.2f}] | "
            f"{r['cohen_h_5_vs_3']:.3f} ({r['cohen_h_5_vs_3_label']}) | "
            f"{wp} | {mp} | {sig} |"
        )

    lines += [
        "",
        "## Single-phantom upgrade lifts (4-mode vs 3-mode)",
        "",
        "| Baseline | Site | +P-DOM lift | CI | h | +P-SoM lift | CI | h |",
        "|---|---|---:|---|---:|---:|---|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['baseline']} | {r['site']} | "
            f"+{r['lift_4pdom_vs_3_pp']:.2f}pp | "
            f"[{r['lift_4pdom_vs_3_ci95_lo_pp']:.2f}, {r['lift_4pdom_vs_3_ci95_hi_pp']:.2f}] | "
            f"{r['cohen_h_4pdom_vs_3']:.3f} | "
            f"+{r['lift_4psom_vs_3_pp']:.2f}pp | "
            f"[{r['lift_4psom_vs_3_ci95_lo_pp']:.2f}, {r['lift_4psom_vs_3_ci95_hi_pp']:.2f}] | "
            f"{r['cohen_h_4psom_vs_3']:.3f} |"
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

    # Scenario C sentinel: P-SoM ↔ P-DOM Jaccard
    lines += [
        "",
        "## Scenario C sentinel — P-SoM ↔ P-DOM task-pool Jaccard",
        "",
        "Threshold: Jaccard > 0.7 → phantoms become routing-redundant; paper",
        "Section 5 axis-2 prompt-effect claim weakens. Current paper claim",
        "(\"prompt creates task-pool divergence without uniform SR change\")",
        "requires Jaccard ≤ 0.7 across cells.",
        "",
        "| Baseline | Site | Jaccard | Status |",
        "|---|---|---:|:---:|",
    ]
    for r in rows:
        if r["phantom_pair_jaccard_warn"]:
            status = "🔴 > 0.7 (WARN: redundant)"
        elif r["phantom_pair_jaccard"] > 0.6:
            status = "🟡 > 0.6 (watch)"
        else:
            status = "✅ ≤ 0.6 (safe)"
        lines.append(
            f"| {r['baseline']} | {r['site']} | {r['phantom_pair_jaccard']:.3f} | {status} |"
        )

    if skipped:
        lines += ["", f"_Cells pending data (skipped): {', '.join(skipped)}_"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
