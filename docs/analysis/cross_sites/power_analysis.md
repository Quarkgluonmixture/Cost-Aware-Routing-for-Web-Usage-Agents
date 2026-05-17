# Power Analysis (Paper §3 / Appendix)

**Configuration**: paired-design McNemar normal approximation; α=0.05 (1.645 one-sided), β=0.2 (power=80%); baseline SR=0.10; πD = conservative 2×min(p,1-p).

> ⚠️ **Empirical reality** (from `results/phantom_paper/meta_phantom_lift.csv` P-SoM `4psom_vs_3` row, k=3 archive cells): SE_FE = 0.529pp, implied per-cell SE_i ≈ 0.916pp, implied empirical πD ≈ 0.019 at p ≈ 0.10 — **~10× smaller than conservative πD bound** because real between-mode correlation ρ is high (modes share task-difficulty variance). Theoretical numbers below are **conservative bounds**; empirical paired SE is substantially smaller.

## Per-cell MDE (minimum detectable effect)

| Site | N (pre-exclusion) | MDE (proportion) | MDE (pp) | Cohen's h at MDE |
|---|---|---|---|---|
| classifieds | 234 | 0.0727 | 7.27pp | 0.214 |
| reddit | 210 | 0.0768 | 7.68pp | 0.224 |
| shopping | 466 | 0.0515 | 5.15pp | 0.156 |

> Post-§139.8 scored N (after N/A task-load exclusion): cls 224 / red 205 / shop 435. MDE shift vs pre-exclusion N is ~2.2% (negligible) but operational denominators are the scored count.

## Per-cell power at assumed effect sizes

How likely is a single-cell test to detect P-SoM > best-baseline at the assumed effect?

| Site | N | Effect=1pp | Effect=2pp | Effect=3pp | Effect=5pp |
|---|---|---|---|---|---|
| classifieds | 234 | 0.10 | 0.17 | 0.27 | 0.53 |
| reddit | 210 | 0.09 | 0.16 | 0.25 | 0.49 |
| shopping | 466 | 0.12 | 0.25 | 0.42 | 0.78 |

## PRIMARY H1 gate power — FE-pool one-sided superiority (prereg §2.5)

Empirical numbers from archive `meta_phantom_lift.csv` P-SoM row (k=3 cells:
B0 cls / B0 red / B1 cls — pre-2026-05-14 archive; missing B1 red + B2 cells
which Phase 1a clean rerun will add for k=6 total).

**Archive (k=3) empirical (SE_FE = 0.529pp directly from aggregator bootstrap)**:

- θ_FE = +2.336pp (observed P-SoM drop-in pooled FE lift)
- SE_FE = 0.529pp (= 0.916pp / sqrt(3); matches aggregator 0.529pp ±)
- z_obs vs H0=1.0pp = 2.526
- p_one_sided = 0.0058 → **STRONGLY rejects H0**
- **Power at observed +2.34pp = 0.811** ← **81%
- MDE @ 80% power = 2.315pp

**Projection (k=6 Phase 1a clean rerun, per-cell SE ≈ 0.916pp from archive)**:

- SE_FE projected = 0.374pp (= 0.916 / sqrt(6))
- **Power at θ_FE = +2.336pp = 0.973** ← 97%
- MDE @ 80% power = 1.930pp

**Sensitivity** (k=6 projection, varying observed effect)**:

| θ_FE assumed | Power |
|---|---|
| +1.00pp | 0.050 |
| +1.50pp | 0.379 |
| +2.00pp | 0.848 |
| +2.34pp | 0.973 |
| +3.00pp | 1.000 |

## Per-axis gating test power (PARITY across 3 phantom siblings)

Empirical archive `meta_phantom_lift.csv` + `phantom_lift.csv` H3 axis rows:

| Gating test | Family | Archive k | θ_FE | SE_FE | k=6 SE_FE | **k=6 Power @ obs** | I² | p_Q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **H1 P-SoM drop-one > 1.0pp** | PRIMARY (deployment hero) | 3 | +2.336pp | 0.529pp | 0.374pp | **97%** | 0.00% | 0.461 |
| **H3 axis-1 P-text \ P-SoM > 0** | STRUCTURAL (phantom axis 1) | 3 | +3.861pp | 0.702pp | 0.496pp | **100%** | 12.14% | 0.320 |
| **H3 axis-2 P-prompt \ P-SoM > 0** | STRUCTURAL (phantom axis 2) | 2 | +2.312pp | 0.670pp | 0.387pp | **100%** | 0.00% | 0.430 |

All 3 gates parity well-powered + heterogeneity-clean. The 71% I² in archive
`4pdom_vs_3` row is a different statistic (drop-in oracle lift, exploratory/
secondary), NOT the H3 STRUCTURAL gating test (`h3_axis1_unique_count`, I²=12%).

## TOST framework retired 2026-05-17 (B-957)

Prior 'TOST equivalence δ=1.0pp' framework was structurally dysfunctional — δ=1pp
is the H1 superiority floor, inadvertently re-used as TOST equivalence margin.
Empirical δ-scan showed all 6 archive arms < 50% TOST power at δ=1pp even with
empirical SE; at observed effect TOST power = 0% for all arms (because |θ| > δ).
H2(a) cost-equivalence uses `median cost ratio > 1.20×` falsification, NOT TOST.
TOST therefore had no clear paper role and was removed from prereg §2.4 + §4.
If a future paper revives TOST equivalence at feasible δ (3-6pp), use explicit
δ_TOST distinct from δ_H1 superiority floor (= 1.0pp, unchanged).

## Interpretation for paper §3

- At baseline SR=0.10 (conservative πD bound), smallest site (reddit N=210) detects per-cell effects ≥ 7.7pp at 80% power.
- **PRIMARY H1 FE-pool gate** (the gate paper-1 actually hinges on): empirical archive k=3 → 81% power at observed +2.34pp; projected k=6 Phase 1a → 97% power. **Well-powered**.
- Empirical paired SE (from aggregator paired-bootstrap) is ~2.2× smaller than the theoretical 1-sample bound used above for per-cell MDE — paired test benefits from real between-mode correlation. Per-cell MDE numbers in table above are **conservative upper bounds**, not empirically tight.

## Reviewer-defensible claim

"Per-cell paired McNemar MDE (conservative bound at πD = 2·min(p,1-p)) = [7.3, 7.7, 5.2]pp for cls/red/shop. **PRIMARY H1 FE-pool gate empirical power = 81% (k=3 archive) / 97% (k=6 Phase 1a projection) at observed +2.34pp pooled drop-one** — strongly powered. The conservative per-cell MDE bound reflects the paired-test worst case (ρ=0); empirical πD ≈ 0.019 (≈10× smaller than the bound) indicates substantial paired-test benefit from real between-mode correlation. K-of-N is transparency-only per prereg §4 Decision 3A."