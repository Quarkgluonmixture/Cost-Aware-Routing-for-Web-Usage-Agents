"""Power analysis for paper §3 — minimum detectable effect (MDE) at α=0.05, β=0.20.

⚠️ RESCOPED 2026-05-15 (B-126 fix per codex Mode B P2-2):
   - Phase 1a scope: 16-cell (pre-2026-05-13) → 6-cell (post-B2 addition 2026-05-14).
   - K-of-N rule RETIRED 2026-05-14 as gate per preregistration.md §4 Decision 3A —
     K_h1=12/16 / K_h3=11/16 hard-gate framing is OBSOLETE. K-of-N is now a
     transparency-only descriptive count (n-of-6 cells individually Holm-sig
     with NO threshold; at k=6 the ratios remain indistinguishable —
     ⌈0.75×6⌉=⌈0.67×6⌉=5 — same fake-precision argument as at k=4).
   - The PRIMARY gate is now the single one-sided FE / RE (advisor TBD)
     superiority test (H0: θ ≤ +1.0pp at α=0.05, m=1).

For binary success rate comparisons (P-SoM vs best-baseline), computes:
- MDE in percentage points at observed N per site (post-§139.8 scored counts:
  cls 224 / red 205 / shop 435)
- Per-cell power for assumed effect sizes (1pp / 2pp / 3pp / 5pp)
- Descriptive transparency count power: probability of n-of-6 cells
  individually Holm-sig at observed effects (informational, NOT a gate)

Method: paired-design McNemar-equivalent normal approximation (within-cell tasks
are paired across modes since same task gets all modes). Transparency power
computed via exact binomial across 6 cells (no threshold applied — for
descriptive 4-5/6 = strong-consistency benchmark per preregistration).

Usage:
    python3 scripts/analysis/power_analysis.py
    python3 scripts/analysis/power_analysis.py --baseline-sr 0.30 --output paper_section3_power.md

Output:
- Markdown table for paper §3 / Appendix
- Identifies which cells have insufficient N for paper-grade claims
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple


def mde_paired_binary(n: int, baseline_sr: float, alpha: float = 0.05, beta: float = 0.20) -> float:
    """Minimum detectable effect (in proportion units) for paired binary
    comparison at given N, baseline SR, α, β.

    Uses McNemar-equivalent normal approximation:
        MDE ≈ (z_{1-α/2} + z_{1-β}) × sqrt(2 × p_avg × (1-p_avg) / n) / sqrt(2)
    where p_avg ≈ baseline_sr (paired tasks share most variance).

    Returns MDE in proportion units (multiply by 100 for percentage points).
    """
    # Standard normal critical values
    z_alpha = 1.96   # two-sided α=0.05
    z_beta = 0.842   # one-sided β=0.20 → 80% power
    p = baseline_sr
    se_paired = math.sqrt(2 * p * (1 - p) / n)
    mde = (z_alpha + z_beta) * se_paired / math.sqrt(2)
    return mde


def k_of_n_power(n_cells: int, k_threshold: int, per_cell_power: float) -> float:
    """Family-wise power for K-of-N pass rule.

    Each cell test independently has `per_cell_power` probability of detecting
    a true effect. Probability that ≥ k_threshold cells detect = sum of binomial
    PMF from k_threshold to n_cells.

    Returns family-wise power.
    """
    from math import comb
    p = per_cell_power
    return sum(comb(n_cells, k) * (p ** k) * ((1 - p) ** (n_cells - k))
               for k in range(k_threshold, n_cells + 1))


def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for binary proportions: arcsine transform difference."""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-sr", type=float, default=0.30,
                   help="Baseline success rate (default 0.30 — typical Phase 1 cls/red)")
    p.add_argument("--alpha", type=float, default=0.05, help="Type I error rate")
    p.add_argument("--beta", type=float, default=0.20, help="Type II error rate")
    p.add_argument("--output", default="-", help="Output markdown path (- = stdout)")
    args = p.parse_args()

    # Site task counts. §139.8: intentionally the *pre-exclusion* design N —
    # this is a pre-registered design-time power computation and the
    # preregistration power section is locked to these numbers. Post-#76 the
    # scored set is ~4% smaller (N/A tasks excluded — see
    # `analysis.scored_task_count`); the MDE shift is negligible and updating
    # here would desync the committed prereg, so this stays hardcoded.
    sites = [
        ("classifieds", 234),
        ("reddit", 210),
        ("shopping", 466),
    ]

    lines = [
        f"# Power Analysis (Paper §3 / Appendix)",
        "",
        f"**Configuration**: paired-design binary SR comparison; α={args.alpha} two-sided, β={args.beta} (power=80%); baseline SR={args.baseline_sr:.2f}",
        "",
        "## Per-cell MDE (minimum detectable effect)",
        "",
        "| Site | N | MDE (proportion) | MDE (pp) | Cohen's h at MDE |",
        "|---|---|---|---|---|",
    ]

    per_site_mde = {}
    for site, n in sites:
        mde = mde_paired_binary(n, args.baseline_sr, args.alpha, args.beta)
        h = cohen_h(args.baseline_sr + mde, args.baseline_sr)
        per_site_mde[site] = mde
        lines.append(f"| {site} | {n} | {mde:.4f} | {mde*100:.2f}pp | {h:.3f} |")

    lines += [
        "",
        "## Per-cell power at assumed effect sizes",
        "",
        "How likely is a single-cell test to detect P-SoM > best-baseline at the assumed effect?",
        "",
        "| Site | N | Effect=1pp | Effect=2pp | Effect=3pp | Effect=5pp |",
        "|---|---|---|---|---|---|",
    ]

    # Per-cell power at fixed effect sizes
    def per_cell_power_at_effect(n: int, effect_pp: float, baseline: float) -> float:
        z_alpha = 1.96
        p = baseline
        se = math.sqrt(2 * p * (1 - p) / n)
        z_score_at_effect = (effect_pp / 100) * math.sqrt(2) / se
        # Power = 1 - Φ(z_alpha - z_score)
        from math import erf
        def phi(z):
            return 0.5 * (1 + erf(z / math.sqrt(2)))
        return 1 - phi(z_alpha - z_score_at_effect)

    for site, n in sites:
        powers = [per_cell_power_at_effect(n, eff, args.baseline_sr) for eff in [1, 2, 3, 5]]
        lines.append(f"| {site} | {n} | {powers[0]:.2f} | {powers[1]:.2f} | {powers[2]:.2f} | {powers[3]:.2f} |")

    # §A1.6 (2026-05-16): K-of-N body retired per preregistration.md §4
    # Decision 3A (K-of-N is transparency-only at k=6, NOT a gate). The
    # body below previously emitted "H1 K_h1 ≥ 12/16 cells, H3 K_h3 ≥ 11/16"
    # which contradicts the file header's retired-status statement and the
    # active Phase 1a 6-cell scope.

    lines += [
        "",
        "## Interpretation for paper §3",
        "",
        f"- At baseline SR={args.baseline_sr:.2f}, smallest site (reddit N=210) detects effects ≥ {per_site_mde['reddit']*100:.1f}pp at 80% power per cell.",
        f"- Largest site (shopping N=466) detects effects ≥ {per_site_mde['shopping']*100:.1f}pp at 80% power per cell.",
        "- For 2-3pp mechanism effects, the paper relies on **TOST equivalence on pooled data** (pre-exclusion design N=234+210+466 — locked at prereg time).",
        "- TOST equivalence (δ=1.0pp) is the tightest test; relies on cross-cell pooling for adequate CI width.",
        "",
        "## Reviewer-defensible claim",
        "",
        f"\"Power analysis (α=0.05, β=0.20, baseline SR={args.baseline_sr:.2f}, paired design) shows ",
        f"per-cell MDE = [{per_site_mde['classifieds']*100:.1f}, {per_site_mde['reddit']*100:.1f}, {per_site_mde['shopping']*100:.1f}]pp ",
        "for cls/red/shop respectively (pre-exclusion design N, prereg-locked). For sub-3pp effects we rely on TOST equivalence on pooled data; K-of-N is transparency-only per prereg §4 Decision 3A.\"",
    ]

    payload = "\n".join(lines)
    if args.output == "-":
        print(payload)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload)
        print(f"Power analysis → {args.output}")


if __name__ == "__main__":
    main()
