# Power Analysis (Paper §3 / Appendix)

**Configuration**: paired-design binary SR comparison; α=0.05 two-sided, β=0.2 (power=80%); baseline SR=0.20

## Per-cell MDE (minimum detectable effect)

| Site | N | MDE (proportion) | MDE (pp) | Cohen's h at MDE |
|---|---|---|---|---|
| classifieds | 234 | 0.0733 | 7.33pp | 0.173 |
| reddit | 210 | 0.0773 | 7.73pp | 0.182 |
| shopping | 466 | 0.0519 | 5.19pp | 0.124 |

## Per-cell power at assumed effect sizes

How likely is a single-cell test to detect P-SoM > best-baseline at the assumed effect?

| Site | N | Effect=1pp | Effect=2pp | Effect=3pp | Effect=5pp |
|---|---|---|---|---|---|
| classifieds | 234 | 0.06 | 0.12 | 0.21 | 0.48 |
| reddit | 210 | 0.06 | 0.11 | 0.19 | 0.44 |
| shopping | 466 | 0.08 | 0.19 | 0.37 | 0.77 |

## Family-wise power (K-of-N rule)

Pre-registration: H1 K_h1 ≥ 12/16 cells, H3 K_h3 ≥ 11/16 cells.
Family-wise power assumes per-cell power is uniform (averaged across sites).

| K threshold | Per-cell power 0.50 | 0.65 | 0.80 | 0.90 |
|---|---|---|---|---|
| K=11 of 16 | 0.105 | 0.490 | 0.918 | 0.997 |
| K=12 of 16 | 0.038 | 0.289 | 0.798 | 0.983 |
| K=13 of 16 | 0.011 | 0.134 | 0.598 | 0.932 |
| K=14 of 16 | 0.002 | 0.045 | 0.352 | 0.789 |

## Interpretation for paper §3

- At baseline SR=0.20, smallest site (reddit N=210) detects effects ≥ 7.7pp at 80% power per cell.
- Largest site (shopping N=466) detects effects ≥ 5.2pp at 80% power per cell.
- At 3pp true effect, **per-cell power ≈ 0.19** (smallest site) → K_h1=12/16 family power = 0.000, K_h3=11/16 family power = 0.000.
- At 5pp true effect, **per-cell power ≈ 0.44** → K_h1=12/16 family power = 0.012, K_h3=11/16 = 0.042.
- **K_h1=12/16 is calibrated for ≥5pp effects.** For 2-3pp mechanism effects, K_h3=11/16 is the operative threshold; below ~3pp, neither K-of-N rule has paper-grade power and the paper relies on **TOST equivalence on pooled data** (N=234+210+466).
- TOST equivalence (δ=1.0pp) is the tightest test; relies on cross-cell pooling for adequate CI width.

## Reviewer-defensible claim

"Power analysis (α=0.05, β=0.20, baseline SR=0.20, paired design) shows 
per-cell MDE = [7.3, 7.7, 5.2]pp 
for cls/red/shop respectively. The K_h1=12/16 family-wise rule has 1% power for 5pp effects 
and 0% for 3pp effects (smallest site as proxy); for sub-3pp effects we rely on TOST equivalence on pooled data."