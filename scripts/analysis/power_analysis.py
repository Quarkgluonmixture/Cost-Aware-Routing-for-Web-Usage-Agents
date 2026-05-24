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
- FE-pool gate power at +2.336pp 4-mode ADD proxy (`4psom_vs_3`; NOT the 6-mode-strict
  H1 drop-one gate effect `_cell_drop_one_theta_se` computes — see AMENDMENT_02 §3 / AMENDMENT_04)
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


# B-957 (/stress A2.3a 2026-05-17): tost_paired_binary() function retired
# alongside TOST framework retirement from prereg §2.4 + §4. Reason: δ=1.0pp
# was the H1 superiority floor inadvertently re-used as TOST equivalence
# margin where it is structurally dysfunctional — all archive arms < 50%
# TOST power at δ=1pp even with empirical SE; observed |θ| > δ for all
# arms makes equivalence mathematically impossible to declare. H2(a)
# cost-equivalence already uses median ratio > 1.20× falsification (not
# TOST). The defensive tost computation added 2026-05-17 per P1-1 had no
# clear paper role and is removed here to prevent reviewer confusion.
# If a future paper revives TOST equivalence at a feasible δ (3-6pp per
# empirical δ-scan in docs/analysis/cross_sites/power_analysis.md archived
# version), reintroduce with explicit δ_TOST distinct from δ_H1.


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
        ("classifieds", 224),
        ("reddit", 205),
        ("shopping", 435),
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
        "| Site | N (scored, post-N/A-exclusion) | MDE (proportion) | MDE (pp) | Cohen's h at MDE |",
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
        "> Scored N (after N/A task-load exclusion, canonical): cls 224 / red 205 / shop 435.",
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
        "## H1 FE-pool gate power (4-mode ADD proxy) — one-sided superiority (prereg §2.5)",
        "",
        "> ⚠️ **AMENDMENT_04 estimand-label (2026-05-24)**: the +2.336pp figure below is the",
        "> archive **4-mode ADD** estimand (`4psom_vs_3`, 3→{4,5}-mode incremental), used here as",
        "> a power **proxy** only — NOT the 6-mode-strict H1 drop-one gate effect that",
        "> `_cell_drop_one_theta_se` computes. Since 6-mode strict ≤ 4-mode ADD by construction,",
        "> these power numbers OVERSTATE 6-mode-strict H1 power; true H1-strict power is TBD from",
        "> Phase 1a paper-grade data. Reported as H1-deploy / Appendix-D sensitivity (AMENDMENT_02 §3).",
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
    # Per-axis gating test power summary (added 2026-05-17 /stress A2.3a B-956)
    # All 3 phantom sibling axes are gating + parity well-powered. Retires
    # the "P-SoM hero vs weaker arm" framing that arose from conflating
    # drop-in oracle lift metric with H3 unique-contribution gating test.
    # ---------------------------------------------------------------
    lines += [
        "",
        "## Per-axis gating test power (PARITY across 3 phantom siblings)",
        "",
        "Empirical archive `meta_phantom_lift.csv` + `phantom_lift.csv` H3 axis rows:",
        "",
        "| Gating test | Family | Archive k | θ_FE | SE_FE | k=6 SE_FE | **k=6 Power @ obs** | I² | p_Q |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        "| **P-SoM 4-mode ADD (`4psom_vs_3`)** † | H1-deploy / Appendix-D sensitivity — NOT 6-mode-strict gate | 3 | +2.336pp | 0.529pp | 0.374pp | **97%†** | 0.00% | 0.461 |",
        "| **H3 axis-1 P-text \\ P-SoM > 0** | STRUCTURAL (phantom axis 1) | 3 | +3.861pp | 0.702pp | 0.496pp | **100%** | 12.14% | 0.320 |",
        "| **H3 axis-2 P-prompt \\ P-SoM > 0** | STRUCTURAL (phantom axis 2) | 2 | +2.312pp | 0.670pp | 0.387pp | **100%** | 0.00% | 0.430 |",
        "",
        "† **H1 row = 4-mode ADD proxy** (`4psom_vs_3`), NOT the 6-mode-strict drop-one gate;",
        "power OVERSTATED (6-mode strict ≤ 4-mode ADD); true H1-strict power TBD post-Phase-1a",
        "(AMENDMENT_02 §3 / AMENDMENT_04). H3 rows below use the unique-count STRUCTURAL estimand.",
        "",
        "All 3 gates parity well-powered + heterogeneity-clean. The 71% I² in archive",
        "`4pdom_vs_3` row is a different statistic (drop-in oracle lift, exploratory/",
        "secondary), NOT the H3 STRUCTURAL gating test (`h3_axis1_unique_count`, I²=12%).",
        "",
        "## TOST framework retired 2026-05-17 (B-957)",
        "",
        "Prior 'TOST equivalence δ=1.0pp' framework was structurally dysfunctional — δ=1pp",
        "is the H1 superiority floor, inadvertently re-used as TOST equivalence margin.",
        "Empirical δ-scan showed all 6 archive arms < 50% TOST power at δ=1pp even with",
        "empirical SE; at observed effect TOST power = 0% for all arms (because |θ| > δ).",
        "H2(a) cost-equivalence uses `median cost ratio > 1.20×` falsification, NOT TOST.",
        "TOST therefore had no clear paper role and was removed from prereg §2.4 + §4.",
        "If a future paper revives TOST equivalence at feasible δ (3-6pp), use explicit",
        "δ_TOST distinct from δ_H1 superiority floor (= 1.0pp, unchanged).",
        "",
        "## Interpretation for paper §3",
        "",
        f"- At baseline SR={args.baseline_sr:.2f} (conservative πD bound), smallest site (reddit "
        f"N=205) detects per-cell effects ≥ {per_site_mde['reddit']*100:.1f}pp at 80% power.",
        f"- **H1 FE-pool gate power (4-mode ADD proxy ⚠️)**: empirical archive "
        f"k=3 → 81% / projected k=6 Phase 1a → 97% at +2.336pp **4-mode ADD** (`4psom_vs_3`). This "
        f"is a power proxy; the 6-mode-strict H1 gate power is TBD post-Phase-1a (6-mode strict ≤ "
        f"4-mode ADD; AMENDMENT_02 §3 / AMENDMENT_04).",
        f"- Empirical paired SE (from aggregator paired-bootstrap) is ~2.2× smaller than the "
        f"theoretical 1-sample bound used above for per-cell MDE — paired test benefits from real "
        f"between-mode correlation. Per-cell MDE numbers in table above are **conservative upper "
        f"bounds**, not empirically tight.",
        "",
        "## Reviewer-defensible claim",
        "",
        f"\"Per-cell paired McNemar MDE (conservative bound at πD = 2·min(p,1-p)) = "
        f"[{per_site_mde['classifieds']*100:.1f}, {per_site_mde['reddit']*100:.1f}, "
        f"{per_site_mde['shopping']*100:.1f}]pp for cls/red/shop. **H1 FE-pool gate power (4-mode "
        f"ADD proxy) = 81% (k=3 archive) / 97% (k=6 projection) at +2.336pp `4psom_vs_3`; the "
        f"6-mode-strict H1 gate power is TBD post-Phase-1a** (6-mode strict ≤ 4-mode ADD per "
        f"AMENDMENT_02 §3 / AMENDMENT_04). The conservative per-cell MDE bound reflects the "
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
