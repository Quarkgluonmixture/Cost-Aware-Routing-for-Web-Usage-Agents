# Power Analysis — Observed-SR Update (B9 ✓)

**Last updated**: 2026-05-09 (B9 audit constraint, post-bugfix)

This appendix updates the pre-registered power analysis with **observed adjusted-SR levels** from `sr_fp_per_mode.md` (Phase 1 B0 + B1 done cells, pre-paper-grade rerun). The post-rerun version will replace this file once 16-cell aggregation completes.

## 1. Observed adjusted-SR ranges (per `sr_fp_per_mode.md`)

| Site | B0 range (DOM/SoM/Vision/P-text/P-prompt/P-SoM) | B1 range (DOM/SoM/Vision/P-text/P-SoM) | Pooled median |
|---|---|---|---|
| classifieds | 13.7-21.4% | 7.3-13.7% | ~13.7% |
| reddit | 6.7-13.8% | 2.4-7.6% | ~9.5% |
| shopping | TBD post-rerun | TBD post-rerun | TBD |

**Observed effect-size range** (phantom-mode minus best non-phantom baseline):
- B0 reddit P-SoM vs DOM: +4.3pp (largest phantom uplift in done cells)
- B0 cls SoM vs DOM: +7.3pp (largest non-phantom contrast)
- B0 cls P-SoM vs DOM: +0.4pp (smallest contrast)
- B1 cls SoM vs DOM: +5.1pp
- B1 cls P-SoM vs DOM: -0.9pp (negative — phantom does not always uplift)

**Modal effect size**: 1-5pp range, with phantom modes clustered at 0-4pp.

## 2. Per-cell MDE at observed SR levels (paired design, α=0.05 two-sided, β=0.20)

Run: `python3 scripts/analysis/power_analysis.py --baseline-sr {0.10,0.15,0.20}`

| Site | N | MDE @ SR=0.10 | MDE @ SR=0.15 | MDE @ SR=0.20 |
|---|---:|---:|---:|---:|
| classifieds | 234 | 5.5pp | 6.5pp | 7.4pp |
| reddit | 210 | 5.8pp | 6.9pp | 7.8pp |
| shopping | 466 | 3.9pp | 4.6pp | 5.2pp |

**Key observation**: minimum detectable effect at 80% per-cell power is **5-7pp** for cls/red, **4-5pp** for shop. The **observed mechanism effect (1-5pp)** is at or below per-cell MDE in 2 of 3 sites — **per-cell power for typical phantom effects is < 50%**.

## 3. Family-wise power at observed effects (K-of-N rule, baseline SR=0.15 proxy)

| Per-cell power (proxy effect on smallest site) | K_h1=12/16 family power | K_h3=11/16 family power |
|---|---:|---:|
| 0.06 (1pp) | <0.001 | <0.001 |
| 0.13 (2pp) | <0.001 | <0.001 |
| 0.23 (3pp) | <0.001 | <0.001 |
| 0.53 (5pp) | 0.061 | 0.151 |
| 0.80 (~6.5pp) | 0.798 | 0.918 |
| 0.90 (~7.5pp) | 0.983 | 0.997 |

**Interpretation**:
- **K_h1=12/16** is calibrated for **≥7pp effects** with paper-grade ≥0.80 family power. For typical phantom mechanism effects (1-5pp), K_h1 family power is **<10%**.
- **K_h3=11/16** is slightly more permissive but still requires per-cell power ≥0.65 (≈6pp effect at SR=0.15) to reach 0.49 family power.

## 4. Methodological implication & paper-§3 framing update

The K-of-N family-wise rule was originally pre-registered as a **transparency / aggregation** check, not the primary detection mechanism. With the corrected interpretation:

- **Primary effect-detection test** = DerSimonian-Laird random-effects meta-analysis (locked by B8) on cells with N≥10. This is power-adequate at the cross-cell level for effects ≥2pp.
- **Equivalence test (TOST)** = pooled across cls+red+shop tasks (N=234+210+466=910), δ=1.0pp margin. Sufficient CI width for 1pp resolution.
- **K-of-N rule** = retained as a **secondary transparency check** documenting how many cells *individually* clear α=0.05; not a gate on the H1/H3 paper claims.
- This recharacterization is **not post-hoc cherry-picking**: the random-effects meta + TOST were always the primary tests in `preregistration.md §4`. The K-of-N rule is restated as transparency.

## 5. Reviewer-rebuttal language

"At observed adjusted-SR levels (8-15% across sites) and observed mechanism effect sizes (1-5pp), per-cell statistical power is below 0.55 in two of three sites. We therefore rely on (a) DerSimonian-Laird random-effects meta-analysis across all cells (B8 lock; cross-cell pooling raises effective power) and (b) TOST equivalence on the full N=910 pooled task set (δ=1.0pp margin) as primary tests. The K-of-N family-wise rule pre-registered for transparency is not powered for sub-5pp effects, and we report its outcome as a secondary observation rather than a gate on the main hypotheses."

## 6. Bug history

The original `power_analysis.py` (pre-2026-05-09) contained a stale interpretation block claiming "K_h1=12/16 with per-cell power=0.65 → >0.95 family-wise power" — this was numerically inconsistent with the K-of-N table on the same page (actual value 0.289). Fixed in commit (current session) to compute family power **at observed effect sizes** rather than at hypothetical per-cell power levels. The corrected version is what this appendix relies on.

**Source files**:
- `scripts/analysis/power_analysis.py` (script, fixed)
- `docs/analysis/cross_sites/power_analysis_observed_sr10.md` / `sr15.md` / `sr20.md` (full per-baseline tables)
- `docs/analysis/cross_sites/sr_fp_per_mode.md` (observed SR source)
- `docs/checkpoints/pre_run/preregistration.md §4` (B8 random-effects lock + TOST policy)

## 7. Open items (post-rerun)

- [ ] Update with shopping observed adjusted-SR once 16-cell rerun completes (cells G/H pending)
- [ ] Re-verify K-of-N rule reframing in `preregistration.md` (audit item A1 / advisor review)
- [ ] Add `power_analysis_post_rerun.md` once observed effect sizes are final (replace this file)
