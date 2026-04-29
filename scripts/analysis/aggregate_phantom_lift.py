#!/usr/bin/env python3
"""[Outcome 0c + 0d] Outcome dimension — routing oracle lift and task-pool Jaccard.

Outputs:
- results/phantom_paper/phantom_lift.csv
- results/phantom_paper/phantom_lift.md

Outcome 0c: 3-mode to 4/5-mode oracle lift and significance tests.
Outcome 0d: P-text↔P-SoM task-pool Jaccard Scenario C sentinel.

See docs/checkpoints/paper_planning.md §3 Outcome dimension framework.

Aggregate phantom routing lift across (baseline, site) cells.

For each cell with all 5 modes (DOM / SoM / Vision / P-text / P-SoM) present:
  - Compute 3-mode oracle ceiling (DOM ∪ SoM ∪ Vision)
  - Compute 5-mode oracle ceiling (+ P-text + P-SoM)
  - Routing lift = 5-mode - 3-mode oracle SR (pp)
  - 95% bootstrap CI on lift (n=1000 task resamples)
  - Decomposition: P-text-only, P-SoM-only, both-add-same contributions

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


def _phantom_prompt_dir(baseline: str, site: str) -> Path | None:
    candidates = sorted(RES.glob(f"{baseline}_phantom_prompt_{site}_*/phase1_phantom_prompt_router_0/episodes"))
    return candidates[-1] if candidates else None


# Cell registry: (baseline, site, expected_N, run_paths_per_mode)
def _build_cell(baseline: str, site: str, expected: int, base_modes: dict) -> dict:
    """Augment a cell's mode dict with a P-prompt entry only when its run dir exists.

    Avoids polluting analyze_cell with a missing-mode bypass for P-prompt; the
    function itself handles modes that are present-but-undersized via MIN_EP_FOR_CELL.
    """
    pp = _phantom_prompt_dir(baseline, site)
    if pp is not None and pp.exists():
        # only attach if run dir actually exists (cell partial otherwise)
        modes = dict(base_modes)
        modes["P-prompt"] = pp
    else:
        modes = base_modes
    return {"baseline": baseline, "site": site, "n_expected": expected, "modes": modes}


CELLS = [
    _build_cell("B0", "classifieds", 234, {
        "DOM":   RES/"B0_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
        "SoM":   RES/"B0_3mode_classifieds_20260413/phase1_som_router_0/episodes",
        "Vision":RES/"B0_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
        "P-text": RES/"B0_phantom_text_classifieds_20260427/phase1_phantom_dom_router_0/episodes",
        "P-SoM": RES/"B0_phantom_som_classifieds_20260426/phase1_phantom_som_router_0/episodes",
    }),
    _build_cell("B0", "reddit", 210, {
        "DOM":   RES/"B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
        "SoM":   RES/"B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
        "Vision":RES/"B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
        "P-text": RES/"B0_phantom_text_reddit_20260427/phase1_phantom_dom_router_0/episodes",
        "P-SoM": RES/"B0_phantom_som_reddit_20260428/phase1_phantom_som_router_0/episodes",
    }),
    # B1 cells — partial. B1 cls Phantom-SoM is paper-grade (234 ep); P-text is
    # not yet available, so this cell is treated as 4-mode-no-P-text. B1 reddit
    # phantom data not started, so the reddit cell is intentionally absent.
    _build_cell("B1", "classifieds", 234, {
        "DOM":   RES/"B1_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
        "SoM":   RES/"B1_3mode_classifieds_20260413/phase1_som_router_0/episodes",
        "Vision":RES/"B1_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
        # P-text intentionally omitted — no paper-grade B1 phantom-DOM data yet
        "P-SoM": RES/"B1_phantom_som_classifieds_20260428/phase1_phantom_som_router_0/episodes",
    }),
]

MIN_EP_FOR_CELL = 50  # skip cells where any present mode has < 50 ep (too partial)


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

    Required modes: DOM, SoM, Vision, P-SoM. P-text and P-prompt are optional —
    when absent, dependent oracle/lift columns are emitted as None (CSV blank).

    Returns None if any required mode missing or below MIN_EP_FOR_CELL.
    """
    succ, obs = {}, {}
    for mode, ep_dir in cell["modes"].items():
        s, o = load(ep_dir)
        if len(o) < MIN_EP_FOR_CELL:
            # Skip undersized modes silently; allow rest of cell to still build.
            # P-prompt cell may be partial (e.g. 127/210); keep it if it meets MIN.
            continue
        succ[mode] = s
        obs[mode] = o

    required = ("DOM", "SoM", "Vision", "P-SoM")
    if any(m not in succ for m in required):
        return None
    has_pdom = "P-text" in succ
    has_pprompt = "P-prompt" in succ

    # Common observed universe (intersection across all present modes)
    common = set.intersection(*obs.values())
    n = len(common)
    if n < MIN_EP_FOR_CELL:
        return None

    # Restrict each mode's success set to common universe
    succ_r = {m: s & common for m, s in succ.items()}

    # 3-mode + 4-mode (P-SoM alone). 4-mode P-text and 5-mode only when P-text present.
    union_3 = succ_r["DOM"] | succ_r["SoM"] | succ_r["Vision"]
    union_4_psom = union_3 | succ_r["P-SoM"]
    sr_3 = 100 * len(union_3) / n
    sr_4_psom = 100 * len(union_4_psom) / n

    universe = sorted(common)
    in_3 = np.array([t in union_3 for t in universe], dtype=bool)
    in_4_psom = np.array([t in union_4_psom for t in universe], dtype=bool)

    # Single-P-SoM lift CI
    ci_lo_psom, ci_hi_psom = bootstrap_lift_ci(in_3, in_4_psom)
    h_4psom_vs_3 = cohen_h(sr_4_psom / 100, sr_3 / 100)
    wstat_psom, wp_psom = wilcoxon_signed_rank(in_3, in_4_psom)
    mc_p_psom = mcnemar_exact_one_sided(in_3, in_4_psom)

    psom_adds = succ_r["P-SoM"] - union_3

    if has_pdom:
        union_4_pdom = union_3 | succ_r["P-text"]
        union_5 = union_3 | succ_r["P-text"] | succ_r["P-SoM"]
        sr_4_pdom = 100 * len(union_4_pdom) / n
        sr_5 = 100 * len(union_5) / n
        in_4_pdom = np.array([t in union_4_pdom for t in universe], dtype=bool)
        in_5 = np.array([t in union_5 for t in universe], dtype=bool)
        ci_lo, ci_hi = bootstrap_lift_ci(in_3, in_5)
        ci_lo_pdom, ci_hi_pdom = bootstrap_lift_ci(in_3, in_4_pdom)
        h_5_vs_3 = cohen_h(sr_5 / 100, sr_3 / 100)
        h_4pdom_vs_3 = cohen_h(sr_4_pdom / 100, sr_3 / 100)
        wstat_5, wp_5 = wilcoxon_signed_rank(in_3, in_5)
        wstat_pdom, wp_pdom = wilcoxon_signed_rank(in_3, in_4_pdom)
        mc_p_5 = mcnemar_exact_one_sided(in_3, in_5)
        mc_p_pdom = mcnemar_exact_one_sided(in_3, in_4_pdom)
        pdom_adds = succ_r["P-text"] - union_3
        both_add = pdom_adds & psom_adds
        pdom_only = pdom_adds - psom_adds
        psom_only = psom_adds - pdom_adds
        inter = succ_r["P-SoM"] & succ_r["P-text"]
        unionj = succ_r["P-SoM"] | succ_r["P-text"]
        jaccard = (len(inter) / len(unionj)) if unionj else 0.0
        jaccard_warn = jaccard > 0.7
    else:
        sr_4_pdom = None
        sr_5 = None
        ci_lo = ci_hi = None
        ci_lo_pdom = ci_hi_pdom = None
        h_5_vs_3 = None
        h_4pdom_vs_3 = None
        wp_5 = wp_pdom = None
        mc_p_5 = mc_p_pdom = None
        pdom_adds = both_add = pdom_only = set()
        psom_only = psom_adds  # no overlap with absent P-text
        jaccard = None
        jaccard_warn = False

    # P-prompt 4-mode lift + 6-mode oracle (when present)
    if has_pprompt:
        union_4_pprompt = union_3 | succ_r["P-prompt"]
        sr_4_pprompt = 100 * len(union_4_pprompt) / n
        in_4_pprompt = np.array([t in union_4_pprompt for t in universe], dtype=bool)
        ci_lo_pprompt, ci_hi_pprompt = bootstrap_lift_ci(in_3, in_4_pprompt)
        h_4pprompt_vs_3 = cohen_h(sr_4_pprompt / 100, sr_3 / 100)
        wstat_pprompt, wp_pprompt = wilcoxon_signed_rank(in_3, in_4_pprompt)
        mc_p_pprompt = mcnemar_exact_one_sided(in_3, in_4_pprompt)
        pprompt_adds = succ_r["P-prompt"] - union_3
        if has_pdom:
            union_6 = union_5 | succ_r["P-prompt"]
            sr_6 = 100 * len(union_6) / n
            in_6 = np.array([t in union_6 for t in universe], dtype=bool)
            ci_lo_6, ci_hi_6 = bootstrap_lift_ci(in_3, in_6)
            ci_lo_6v5, ci_hi_6v5 = bootstrap_lift_ci(in_5, in_6)
            h_6_vs_3 = cohen_h(sr_6 / 100, sr_3 / 100)
            h_6_vs_5 = cohen_h(sr_6 / 100, sr_5 / 100)
            _, wp_6 = wilcoxon_signed_rank(in_3, in_6)
            _, wp_6v5 = wilcoxon_signed_rank(in_5, in_6)
            mc_p_6 = mcnemar_exact_one_sided(in_3, in_6)
            mc_p_6v5 = mcnemar_exact_one_sided(in_5, in_6)
        else:
            sr_6 = None
            ci_lo_6 = ci_hi_6 = ci_lo_6v5 = ci_hi_6v5 = None
            h_6_vs_3 = h_6_vs_5 = None
            wp_6 = wp_6v5 = None
            mc_p_6 = mc_p_6v5 = None
    else:
        sr_4_pprompt = None
        ci_lo_pprompt = ci_hi_pprompt = None
        h_4pprompt_vs_3 = None
        wp_pprompt = None
        mc_p_pprompt = None
        pprompt_adds = set()
        sr_6 = None
        ci_lo_6 = ci_hi_6 = ci_lo_6v5 = ci_hi_6v5 = None
        h_6_vs_3 = h_6_vs_5 = None
        wp_6 = wp_6v5 = None
        mc_p_6 = mc_p_6v5 = None

    is_partial = (any(len(o) < cell["n_expected"] for o in obs.values()) or not has_pdom
                  or not has_pprompt)

    def maybe_round(value, ndigits=4):
        return None if value is None else round(value, ndigits)

    return {
        "baseline": cell["baseline"],
        "site": cell["site"],
        "n_common": n,
        "n_expected": cell["n_expected"],
        "is_partial": is_partial,
        "has_pdom": has_pdom,
        "has_pprompt": has_pprompt,
        "sr_dom":     round(100 * len(succ_r["DOM"]) / n, 4),
        "sr_som":     round(100 * len(succ_r["SoM"]) / n, 4),
        "sr_vision":  round(100 * len(succ_r["Vision"]) / n, 4),
        "sr_pdom":    (round(100 * len(succ_r["P-text"]) / n, 4) if has_pdom else None),
        "sr_psom":    round(100 * len(succ_r["P-SoM"]) / n, 4),
        "sr_pprompt": (round(100 * len(succ_r["P-prompt"]) / n, 4) if has_pprompt else None),
        "oracle_3mode_pp":  round(sr_3, 4),
        "oracle_4mode_pdom_pp": maybe_round(sr_4_pdom),
        "oracle_4mode_psom_pp": round(sr_4_psom, 4),
        "oracle_4mode_pprompt_pp": maybe_round(sr_4_pprompt),
        "oracle_5mode_pp":  maybe_round(sr_5),
        "oracle_6mode_pp":  maybe_round(sr_6),
        "lift_5_vs_3_pp":   (round(sr_5 - sr_3, 4) if sr_5 is not None else None),
        "lift_5_vs_3_ci95_lo_pp":  maybe_round(ci_lo),
        "lift_5_vs_3_ci95_hi_pp":  maybe_round(ci_hi),
        "lift_4pdom_vs_3_pp":   (round(sr_4_pdom - sr_3, 4) if sr_4_pdom is not None else None),
        "lift_4pdom_vs_3_ci95_lo_pp": maybe_round(ci_lo_pdom),
        "lift_4pdom_vs_3_ci95_hi_pp": maybe_round(ci_hi_pdom),
        "lift_4psom_vs_3_pp":   round(sr_4_psom - sr_3, 4),
        "lift_4psom_vs_3_ci95_lo_pp": round(ci_lo_psom, 4),
        "lift_4psom_vs_3_ci95_hi_pp": round(ci_hi_psom, 4),
        "lift_4pprompt_vs_3_pp": (round(sr_4_pprompt - sr_3, 4) if sr_4_pprompt is not None else None),
        "lift_4pprompt_vs_3_ci95_lo_pp": maybe_round(ci_lo_pprompt),
        "lift_4pprompt_vs_3_ci95_hi_pp": maybe_round(ci_hi_pprompt),
        "lift_6_vs_3_pp": (round(sr_6 - sr_3, 4) if sr_6 is not None else None),
        "lift_6_vs_3_ci95_lo_pp": maybe_round(ci_lo_6),
        "lift_6_vs_3_ci95_hi_pp": maybe_round(ci_hi_6),
        "lift_6_vs_5_pp": (round(sr_6 - sr_5, 4) if (sr_6 is not None and sr_5 is not None) else None),
        "lift_6_vs_5_ci95_lo_pp": maybe_round(ci_lo_6v5),
        "lift_6_vs_5_ci95_hi_pp": maybe_round(ci_hi_6v5),
        # Effect sizes (Cohen's h on oracle proportions)
        "cohen_h_5_vs_3":     maybe_round(h_5_vs_3),
        "cohen_h_5_vs_3_label": (cohen_h_label(h_5_vs_3) if h_5_vs_3 is not None else None),
        "cohen_h_4pdom_vs_3": maybe_round(h_4pdom_vs_3),
        "cohen_h_4pdom_vs_3_label": (cohen_h_label(h_4pdom_vs_3) if h_4pdom_vs_3 is not None else None),
        "cohen_h_4psom_vs_3": round(h_4psom_vs_3, 4),
        "cohen_h_4psom_vs_3_label": cohen_h_label(h_4psom_vs_3),
        "cohen_h_4pprompt_vs_3": maybe_round(h_4pprompt_vs_3),
        "cohen_h_4pprompt_vs_3_label": (cohen_h_label(h_4pprompt_vs_3) if h_4pprompt_vs_3 is not None else None),
        "cohen_h_6_vs_3": maybe_round(h_6_vs_3),
        "cohen_h_6_vs_3_label": (cohen_h_label(h_6_vs_3) if h_6_vs_3 is not None else None),
        "cohen_h_6_vs_5": maybe_round(h_6_vs_5),
        "cohen_h_6_vs_5_label": (cohen_h_label(h_6_vs_5) if h_6_vs_5 is not None else None),
        # Wilcoxon (paired sign on binary)
        "wilcoxon_5_vs_3_p":     wp_5,
        "wilcoxon_4pdom_vs_3_p": wp_pdom,
        "wilcoxon_4psom_vs_3_p": wp_psom,
        "wilcoxon_4pprompt_vs_3_p": wp_pprompt,
        "wilcoxon_6_vs_3_p": wp_6,
        "wilcoxon_6_vs_5_p": wp_6v5,
        # McNemar exact 1-sided
        "mcnemar_5_vs_3_p":     mc_p_5,
        "mcnemar_4pdom_vs_3_p": mc_p_pdom,
        "mcnemar_4psom_vs_3_p": mc_p_psom,
        "mcnemar_4pprompt_vs_3_p": mc_p_pprompt,
        "mcnemar_6_vs_3_p": mc_p_6,
        "mcnemar_6_vs_5_p": mc_p_6v5,
        # Decomposition
        "pdom_adds_count":      (len(pdom_adds) if has_pdom else None),
        "psom_adds_count":      len(psom_adds),
        "pprompt_adds_count":   (len(pprompt_adds) if has_pprompt else None),
        "pdom_only_count":      (len(pdom_only) if has_pdom else None),
        "psom_only_count":      (len(psom_only) if has_pdom else None),
        "both_phantom_overlap_count": (len(both_add) if has_pdom else None),
        # Scenario C sentinel: P-SoM ↔ P-text Jaccard
        "phantom_pair_jaccard": (round(jaccard, 4) if jaccard is not None else None),
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
        if r.get("lift_5_vs_3_pp") is None:
            lines.append(
                f"| {r['baseline']} | {r['site']} | {n_label} | n/a (P-text pending) | — | — | — | — | — |"
            )
            continue
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
        "| Baseline | Site | +P-text lift | CI | h | +P-SoM lift | CI | h | +P-prompt lift | CI | h |",
        "|---|---|---:|---|---:|---:|---|---:|---:|---|---:|",
    ]
    for r in rows:
        if r.get("lift_4pdom_vs_3_pp") is None:
            pdom_cell = "n/a"
            pdom_ci = "—"
            pdom_h = "—"
        else:
            pdom_cell = f"+{r['lift_4pdom_vs_3_pp']:.2f}pp"
            pdom_ci = f"[{r['lift_4pdom_vs_3_ci95_lo_pp']:.2f}, {r['lift_4pdom_vs_3_ci95_hi_pp']:.2f}]"
            pdom_h = f"{r['cohen_h_4pdom_vs_3']:.3f}"
        if r.get("lift_4pprompt_vs_3_pp") is None:
            pprompt_cell = "n/a (pending)"
            pprompt_ci = "—"
            pprompt_h = "—"
        else:
            pprompt_cell = f"+{r['lift_4pprompt_vs_3_pp']:.2f}pp"
            pprompt_ci = f"[{r['lift_4pprompt_vs_3_ci95_lo_pp']:.2f}, {r['lift_4pprompt_vs_3_ci95_hi_pp']:.2f}]"
            pprompt_h = f"{r['cohen_h_4pprompt_vs_3']:.3f}"
        lines.append(
            f"| {r['baseline']} | {r['site']} | "
            f"{pdom_cell} | {pdom_ci} | {pdom_h} | "
            f"+{r['lift_4psom_vs_3_pp']:.2f}pp | "
            f"[{r['lift_4psom_vs_3_ci95_lo_pp']:.2f}, {r['lift_4psom_vs_3_ci95_hi_pp']:.2f}] | "
            f"{r['cohen_h_4psom_vs_3']:.3f} | "
            f"{pprompt_cell} | {pprompt_ci} | {pprompt_h} |"
        )

    # 6-mode oracle (when P-prompt + P-text both present)
    lines += [
        "",
        "## 6-mode oracle (5-mode + P-prompt)",
        "",
        "| Baseline | Site | 6-mode SR | 6 vs 3 lift | CI | h | 6 vs 5 lift | CI | h |",
        "|---|---|---:|---:|---|---:|---:|---|---:|",
    ]
    for r in rows:
        if r.get("oracle_6mode_pp") is None:
            lines.append(
                f"| {r['baseline']} | {r['site']} | n/a (pending) | — | — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {r['baseline']} | {r['site']} | {r['oracle_6mode_pp']:.2f}% | "
            f"+{r['lift_6_vs_3_pp']:.2f}pp | "
            f"[{r['lift_6_vs_3_ci95_lo_pp']:.2f}, {r['lift_6_vs_3_ci95_hi_pp']:.2f}] | "
            f"{r['cohen_h_6_vs_3']:.3f} | "
            f"+{r['lift_6_vs_5_pp']:.2f}pp | "
            f"[{r['lift_6_vs_5_ci95_lo_pp']:.2f}, {r['lift_6_vs_5_ci95_hi_pp']:.2f}] | "
            f"{r['cohen_h_6_vs_5']:.3f} |"
        )

    if any(r["is_partial"] for r in rows):
        lines.append("")
        lines.append("† = partial (any mode < expected N); using intersection of observed tasks.")
    lines += [
        "",
        "## Decomposition: which phantom contributes which tasks",
        "",
        "| Baseline | Site | P-text adds | P-SoM adds | P-prompt adds | P-text only | P-SoM only | Both phantoms overlap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        n = max(r["n_common"], 1)
        def cell(val):
            return "n/a" if val is None else f"{val} ({100*val/n:.2f}pp)"
        lines.append(
            f"| {r['baseline']} | {r['site']} | "
            f"{cell(r['pdom_adds_count'])} | "
            f"{cell(r['psom_adds_count'])} | "
            f"{cell(r.get('pprompt_adds_count'))} | "
            f"{cell(r['pdom_only_count'])} | "
            f"{cell(r['psom_only_count'])} | "
            f"{cell(r['both_phantom_overlap_count'])} |"
        )

    # Scenario C sentinel: P-SoM ↔ P-text Jaccard
    lines += [
        "",
        "## Scenario C sentinel — P-SoM ↔ P-text task-pool Jaccard",
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
        if r.get("phantom_pair_jaccard") is None:
            lines.append(
                f"| {r['baseline']} | {r['site']} | n/a | ⏳ P-text pending |"
            )
            continue
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
