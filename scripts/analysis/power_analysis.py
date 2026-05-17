"""Power analysis for paper §3 — minimum detectable effect (MDE) at α=0.05, β=0.20.

⚠️ HISTORY:
   - 2026-05-15 (B-126 fix per codex Mode B P2-2): Phase 1a scope 16-cell → 6-cell;
     K-of-N rule RETIRED 2026-05-14 as gate per preregistration.md §4 Decision 3A.
   - 2026-05-17 (/stress A2.3a Mode A + B + C cross-AI cycle, B-941~B-946): empirical-
     grounded rewrite. Theoretical 1-sample SE replaced with explicit paired-McNemar SE
     accepting πD assumption. z_α defaults to one-sided (matches prereg §2.5 single-sided
     FE gate). Added fe_pool_power() for the PRIMARY H1 gate that was previously
     uncomputed. Deleted orphan k_of_n_power. Empirical SE from archive
     `meta_phantom_lift.csv` P-SoM row (k=3 cells, SE_FE=0.529pp) shows realistic per-cell
     SE_i ≈ 0.916pp — about 2.2× smaller than theoretical 1-sample SE bound, because
     paired-test gains from real between-mode correlation ρ. The "Mode A theoretical
     48% power crisis" was a bound artifact: empirical archive at k=3 already gives
     81% power at observed +2.34pp; projected k=6 = 97%.

For binary success rate comparisons (P-SoM vs best-baseline), computes:
- MDE in percentage points at observed N per site (post-§139.8 scored counts:
  cls 224 / red 205 / shop 435)
- Per-cell power for assumed effect sizes (1pp / 2pp / 3pp / 5pp)
- FE-pool gate power at observed +2.34pp pooled drop-one (PRIMARY H1 gate)
- TOST equivalence power (defensive computation per /stress A2.3a P1-1)

Method: paired McNemar normal approximation with explicit πD (discordant-pair
proportion) assumption. Default conservative πD = 2*min(p, 1-p) corresponds to
ρ=0 (zero between-mode correlation, paired test worst case). Realistic ρ > 0
reduces πD and tightens SE — the empirical archive πD ≈ 0.019 (= 1.88%) is ~10×
smaller than the default conservative bound at p=0.10.

Usage:
    python3 scripts/analysis/power_analysis.py
    python3 scripts/analysis/power_analysis.py --baseline-sr 0.10 --output docs/analysis/cross_sites/power_analysis.md

Output:
- Markdown table for paper §3 / Appendix
- Empirical FE-pool power section (replaces retired k=4 vintage TODO)
- TOST equivalence row (defensive)
"""

from __future__ import annotations

import argparse
import math
from math import erf
from pathlib import Path


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + erf(z / math.sqrt(2)))


def paired_mcnemar_se(n: int, p: float, pi_D: float | None = None) -> float:
    """Paired McNemar SE under explicit πD assumption.

    Default πD = 2*min(p, 1-p) is the conservative worst-case (zero between-mode
    correlation ρ=0). Realistic ρ > 0 reduces πD and tightens SE → boosts paired
    power. Archive empirical πD ≈ 0.019 at observed SR ≈ 0.10 → ~10× smaller than
    this conservative default, reflecting high between-mode task-difficulty
    correlation.

    Args:
        n: paired sample size (number of tasks both modes ran)
        p: baseline (or marginal) success rate
        pi_D: discordant-pair proportion P(π10 + π01); if None, uses
              conservative 2*min(p, 1-p) bound
    """
    if pi_D is None:
        pi_D = 2 * min(p, 1 - p)
    return math.sqrt(pi_D / n)


def mde_paired_binary(n: int, baseline_sr: float, alpha: float = 0.05,
                      beta: float = 0.20, one_sided: bool = True,
                      pi_D: float | None = None) -> float:
    """MDE (proportion units) for paired-binary McNemar at given N, baseline SR.

    One-sided by default (matches prereg §2.5 single-sided FE superiority gate).
    Two-sided available for legacy / two-tail comparisons.
    """
    z_alpha = 1.645 if one_sided else 1.96
    z_beta = 0.842   # 80% power
    se = paired_mcnemar_se(n, baseline_sr, pi_D=pi_D)
    return (z_alpha + z_beta) * se


def per_cell_power(n: int, baseline_sr: float, effect_pp: float,
                   alpha: float = 0.05, one_sided: bool = True,
                   pi_D: float | None = None) -> float:
    """Per-cell power for paired-binary McNemar at given effect size."""
    z_alpha = 1.645 if one_sided else 1.96
    se = paired_mcnemar_se(n, baseline_sr, pi_D=pi_D)
    z_score = (effect_pp / 100) / se
    return 1 - _phi(z_alpha - z_score)


def fe_pool_power(per_cell_ses_pp: list[float], theta_fe_pp: float,
                  h0_threshold_pp: float = 1.0, alpha: float = 0.05,
                  one_sided: bool = True) -> dict:
    """Power of the PRIMARY H1 FE-pool gate (one-sided H0: θ_FE ≤ δ_pp).

    This is the gate prereg §2.5 L336 locks as paper-1 hero gate. Previously
    uncomputed — /stress A2.3a Mode A F2 (RETRACTED after empirical re-check)
    initially mis-computed via theoretical 1-sample SE, getting 48% at
    observed +2.34pp. Empirical SE from archive `meta_phantom_lift.csv` shows
    actual paired SE is ~2.2× smaller → power actually 81% (k=3 archive) /
    97% (k=6 projected Phase 1a).

    Args:
        per_cell_ses_pp: per-cell SE in percentage points (e.g. [0.92, 0.92, ...])
        theta_fe_pp: observed (or hypothesized) FE-pool lift in pp
        h0_threshold_pp: superiority threshold δ (prereg §4 row "H1 δ" = 1.0pp)
        alpha: type-I rate
        one_sided: prereg §2.5 = True

    Returns dict with se_fe_pp, z_obs, p_one_sided, power, mde_80_pp.
    """
    z_alpha = 1.645 if one_sided else 1.96
    z_beta = 0.842
    # FE inverse-variance pooling
    sum_w = sum(1.0 / (se_pp / 100) ** 2 for se_pp in per_cell_ses_pp)
    se_fe_pp = math.sqrt(1.0 / sum_w) * 100
    z_obs = (theta_fe_pp - h0_threshold_pp) / se_fe_pp
    power = 1 - _phi(z_alpha - z_obs)
    p_one_sided = 1 - _phi(z_obs) if one_sided else 2 * (1 - _phi(abs(z_obs)))
    mde_80 = h0_threshold_pp + (z_alpha + z_beta) * se_fe_pp
    return {
        "k_cells": len(per_cell_ses_pp),
        "se_fe_pp": se_fe_pp,
        "z_obs": z_obs,
        "p_one_sided": p_one_sided,
        "power": power,
        "mde_80_pp": mde_80,
    }


def tost_paired_binary(n_pooled: int, baseline_sr: float, delta_pp: float,
                       observed_pp: float = 0.0, alpha: float = 0.05,
                       pi_D: float | None = None) -> dict:
    """TOST equivalence power for paired-binary at pooled N (Schuirmann 1987).

    Defensive computation per /stress A2.3a P1-1. Prereg §4 L419 / §2.4 L327
    call TOST "the tightest test" / "the mitigation" but never compute its
    power. This function provides the missing power so the prereg can honestly
    disclose whether TOST is a power-validated fallback or informational only.

    TOST = two one-sided tests at α each. Power to declare equivalence requires
    the observed CI to fit entirely within (−δ, +δ). At paired SE_pooled,
    approximate power = 2 × Φ((δ − |observed|) / SE − z_α) − 1 if (δ − |obs|) > z_α × SE.

    Args:
        n_pooled: pooled paired sample size (Σ N_cell)
        baseline_sr: baseline (or marginal) SR
        delta_pp: equivalence margin (prereg = 1.0pp)
        observed_pp: hypothesized true effect for power calc (default 0 = pure equivalence)
    """
    z_alpha = 1.645  # one-sided α=0.05 per TOST end
    se = paired_mcnemar_se(n_pooled, baseline_sr, pi_D=pi_D)  # SE in proportion units
    delta_prop = delta_pp / 100
    observed_prop = observed_pp / 100
    margin = delta_prop - abs(observed_prop)
    if margin <= z_alpha * se:
        power = 0.0  # CI cannot fit within (-δ, +δ) at any prob
    else:
        # Two one-sided: P((obs - δ)/se < -z_α  AND  (obs + δ)/se > z_α) under H_alt: |θ| < δ
        # Approximation (Schuirmann normal): power ≈ 2 × Φ((δ - |obs|)/se - z_α) - 1
        power = 2 * _phi(margin / se - z_alpha) - 1
        power = max(0.0, power)
    return {
        "n_pooled": n_pooled,
        "delta_pp": delta_pp,
        "observed_pp": observed_pp,
        "se_pooled_pp": se * 100,
        "tost_power": power,
    }


def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for binary proportions: arcsine transform difference."""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-sr", type=float, default=0.10,
                   help="Baseline success rate (default 0.10 — matches prereg §2.4 "
                        "observed adjusted-SR 8-15% lower bound; old default 0.30 "
                        "did not match prereg-cited reality)")
    p.add_argument("--alpha", type=float, default=0.05, help="Type I error rate")
    p.add_argument("--beta", type=float, default=0.20, help="Type II error rate")
    p.add_argument("--two-sided", action="store_true",
                   help="Use two-sided z_α=1.96 (default: one-sided 1.645 per prereg §2.5)")
    p.add_argument("--pi-d", type=float, default=None,
                   help="Override paired McNemar πD. Default conservative 2*min(p,1-p) "
                        "(ρ=0 worst case). Empirical archive ≈ 0.019 at p ≈ 0.10.")
    p.add_argument("--output", default="-", help="Output markdown path (- = stdout)")
    args = p.parse_args()

    one_sided = not args.two_sided
    z_label = "1.645 one-sided" if one_sided else "1.96 two-sided"

    sites = [
        ("classifieds", 234),
        ("reddit", 210),
        ("shopping", 466),
    ]

    lines = [
        "# Power Analysis (Paper §3 / Appendix)",
        "",
        f"**Configuration**: paired-design McNemar normal approximation; "
        f"α={args.alpha} ({z_label}), β={args.beta} (power=80%); "
        f"baseline SR={args.baseline_sr:.2f}; "
        f"πD = {'conservative 2×min(p,1-p)' if args.pi_d is None else f'{args.pi_d:.4f} (override)'}.",
        "",
        "> ⚠️ **Empirical reality** (from `results/phantom_paper/meta_phantom_lift.csv` "
        "P-SoM `4psom_vs_3` row, k=3 archive cells): SE_FE = 0.529pp, implied per-cell "
        "SE_i ≈ 0.916pp, implied empirical πD ≈ 0.019 at p ≈ 0.10 — **~10× smaller "
        "than conservative πD bound** because real between-mode correlation ρ is high "
        "(modes share task-difficulty variance). Theoretical numbers below are "
        "**conservative bounds**; empirical paired SE is substantially smaller.",
        "",
        "## Per-cell MDE (minimum detectable effect)",
        "",
        "| Site | N (pre-exclusion) | MDE (proportion) | MDE (pp) | Cohen's h at MDE |",
        "|---|---|---|---|---|",
    ]

    per_site_mde = {}
    for site, n in sites:
        mde = mde_paired_binary(n, args.baseline_sr, args.alpha, args.beta,
                                one_sided=one_sided, pi_D=args.pi_d)
        h = cohen_h(args.baseline_sr + mde, args.baseline_sr)
        per_site_mde[site] = mde
        lines.append(f"| {site} | {n} | {mde:.4f} | {mde*100:.2f}pp | {h:.3f} |")

    lines += [
        "",
        "> Post-§139.8 scored N (after N/A task-load exclusion): cls 224 / red 205 / shop 435. "
        "MDE shift vs pre-exclusion N is ~2.2% (negligible) but operational denominators "
        "are the scored count.",
        "",
        "## Per-cell power at assumed effect sizes",
        "",
        "How likely is a single-cell test to detect P-SoM > best-baseline at the assumed effect?",
        "",
        "| Site | N | Effect=1pp | Effect=2pp | Effect=3pp | Effect=5pp |",
        "|---|---|---|---|---|---|",
    ]

    for site, n in sites:
        powers = [per_cell_power(n, args.baseline_sr, eff, alpha=args.alpha,
                                  one_sided=one_sided, pi_D=args.pi_d)
                  for eff in [1, 2, 3, 5]]
        lines.append(f"| {site} | {n} | {powers[0]:.2f} | {powers[1]:.2f} | {powers[2]:.2f} | {powers[3]:.2f} |")

    # ---------------------------------------------------------------
    # PRIMARY H1 FE-pool gate (added 2026-05-17, /stress A2.3a P1-new)
    # ---------------------------------------------------------------
    lines += [
        "",
        "## PRIMARY H1 gate power — FE-pool one-sided superiority (prereg §2.5)",
        "",
        "Empirical numbers from archive `meta_phantom_lift.csv` P-SoM row (k=3 cells:",
        "B0 cls / B0 red / B1 cls — pre-2026-05-14 archive; missing B1 red + B2 cells",
        "which Phase 1a clean rerun will add for k=6 total).",
        "",
        "**Archive (k=3) empirical (SE_FE = 0.529pp directly from aggregator bootstrap)**:",
        "",
    ]
    # Empirical archive
    arch = fe_pool_power([0.916, 0.916, 0.916], theta_fe_pp=2.336, h0_threshold_pp=1.0,
                         alpha=args.alpha, one_sided=one_sided)
    lines += [
        f"- θ_FE = +2.336pp (observed P-SoM drop-in pooled FE lift)",
        f"- SE_FE = {arch['se_fe_pp']:.3f}pp (= 0.916pp / sqrt(3); matches aggregator 0.529pp ±)",
        f"- z_obs vs H0=1.0pp = {arch['z_obs']:.3f}",
        f"- p_one_sided = {arch['p_one_sided']:.4f} → **STRONGLY rejects H0**",
        f"- **Power at observed +2.34pp = {arch['power']:.3f}** ← {'**' + str(round(arch['power']*100)) + '%' if arch['power']>0.5 else str(round(arch['power']*100))+'%'}",
        f"- MDE @ 80% power = {arch['mde_80_pp']:.3f}pp",
        "",
        "**Projection (k=6 Phase 1a clean rerun, per-cell SE ≈ 0.916pp from archive)**:",
        "",
    ]
    proj = fe_pool_power([0.916] * 6, theta_fe_pp=2.336, h0_threshold_pp=1.0,
                         alpha=args.alpha, one_sided=one_sided)
    lines += [
        f"- SE_FE projected = {proj['se_fe_pp']:.3f}pp (= 0.916 / sqrt(6))",
        f"- **Power at θ_FE = +2.336pp = {proj['power']:.3f}** ← {round(proj['power']*100)}%",
        f"- MDE @ 80% power = {proj['mde_80_pp']:.3f}pp",
        "",
        "**Sensitivity** (k=6 projection, varying observed effect)**:",
        "",
        "| θ_FE assumed | Power |",
        "|---|---|",
    ]
    for theta in [1.0, 1.5, 2.0, 2.336, 3.0]:
        sens = fe_pool_power([0.916] * 6, theta_fe_pp=theta, h0_threshold_pp=1.0,
                             alpha=args.alpha, one_sided=one_sided)
        lines.append(f"| +{theta:.2f}pp | {sens['power']:.3f} |")

    # ---------------------------------------------------------------
    # TOST equivalence power (added 2026-05-17, /stress A2.3a P1-1)
    # ---------------------------------------------------------------
    lines += [
        "",
        "## TOST equivalence power (defensive, prereg §4 L419 / §2.4 L327)",
        "",
        "Prereg calls TOST equivalence δ=1.0pp \"the tightest test\". Computed here to",
        "show whether it is a power-validated fallback or informational only.",
        "",
        "| n_pooled | δ (pp) | Observed θ (pp) | SE_pooled (pp) | TOST power |",
        "|---|---|---|---|---|",
    ]
    for n_pooled_label, n_pooled in [("cls (n=224)", 224), ("red (n=205)", 205),
                                       ("cls+red (n=429)", 429)]:
        for obs in [0.0, 0.5]:
            t = tost_paired_binary(n_pooled, args.baseline_sr, delta_pp=1.0,
                                   observed_pp=obs, alpha=args.alpha, pi_D=args.pi_d)
            lines.append(f"| {n_pooled_label} | {t['delta_pp']:.1f} | {t['observed_pp']:.1f} | "
                         f"{t['se_pooled_pp']:.3f} | {t['tost_power']:.3f} |")

    lines += [
        "",
        "**TOST status**: see numbers above — at conservative πD bound + cls/red N alone, TOST",
        "equivalence at δ=1.0pp is power-limited; pooled N=429 helps but does not reach 80%.",
        "If empirical paired SE (πD ≈ 0.019) holds in Phase 1a, TOST power improves substantially.",
        "Paper §3 prose should disclose TOST as **complementary evidence** not as a power-validated",
        "fallback for H1 (the H1 FE gate is the substantive test).",
        "",
        "## Interpretation for paper §3",
        "",
        f"- At baseline SR={args.baseline_sr:.2f} (conservative πD bound), smallest site (reddit "
        f"N=210) detects per-cell effects ≥ {per_site_mde['reddit']*100:.1f}pp at 80% power.",
        f"- **PRIMARY H1 FE-pool gate** (the gate paper-1 actually hinges on): empirical archive "
        f"k=3 → 81% power at observed +2.34pp; projected k=6 Phase 1a → 97% power. **Well-powered**.",
        f"- Empirical paired SE (from aggregator paired-bootstrap) is ~2.2× smaller than the "
        f"theoretical 1-sample bound used above for per-cell MDE — paired test benefits from real "
        f"between-mode correlation. Per-cell MDE numbers in table above are **conservative upper "
        f"bounds**, not empirically tight.",
        "",
        "## Reviewer-defensible claim",
        "",
        f"\"Per-cell paired McNemar MDE (conservative bound at πD = 2·min(p,1-p)) = "
        f"[{per_site_mde['classifieds']*100:.1f}, {per_site_mde['reddit']*100:.1f}, "
        f"{per_site_mde['shopping']*100:.1f}]pp for cls/red/shop. **PRIMARY H1 FE-pool gate "
        f"empirical power = 81% (k=3 archive) / 97% (k=6 Phase 1a projection) at observed +2.34pp "
        f"pooled drop-one** — strongly powered. The conservative per-cell MDE bound reflects the "
        f"paired-test worst case (ρ=0); empirical πD ≈ 0.019 (≈10× smaller than the bound) "
        f"indicates substantial paired-test benefit from real between-mode correlation. K-of-N "
        f"is transparency-only per prereg §4 Decision 3A.\"",
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
