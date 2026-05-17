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

## TOST equivalence power (defensive, prereg §4 L419 / §2.4 L327)

Prereg calls TOST equivalence δ=1.0pp "the tightest test". Computed here to
show whether it is a power-validated fallback or informational only.

| n_pooled | δ (pp) | Observed θ (pp) | SE_pooled (pp) | TOST power |
|---|---|---|---|---|
| cls (n=224) | 1.0 | 0.0 | 2.988 | 0.000 |
| cls (n=224) | 1.0 | 0.5 | 2.988 | 0.000 |
| red (n=205) | 1.0 | 0.0 | 3.123 | 0.000 |
| red (n=205) | 1.0 | 0.5 | 3.123 | 0.000 |
| cls+red (n=429) | 1.0 | 0.0 | 2.159 | 0.000 |
| cls+red (n=429) | 1.0 | 0.5 | 2.159 | 0.000 |

**TOST status**: see numbers above — at conservative πD bound + cls/red N alone, TOST
equivalence at δ=1.0pp is power-limited; pooled N=429 helps but does not reach 80%.
If empirical paired SE (πD ≈ 0.019) holds in Phase 1a, TOST power improves substantially.
Paper §3 prose should disclose TOST as **complementary evidence** not as a power-validated
fallback for H1 (the H1 FE gate is the substantive test).

## Interpretation for paper §3

- At baseline SR=0.10 (conservative πD bound), smallest site (reddit N=210) detects per-cell effects ≥ 7.7pp at 80% power.
- **PRIMARY H1 FE-pool gate** (the gate paper-1 actually hinges on): empirical archive k=3 → 81% power at observed +2.34pp; projected k=6 Phase 1a → 97% power. **Well-powered**.
- Empirical paired SE (from aggregator paired-bootstrap) is ~2.2× smaller than the theoretical 1-sample bound used above for per-cell MDE — paired test benefits from real between-mode correlation. Per-cell MDE numbers in table above are **conservative upper bounds**, not empirically tight.

## Reviewer-defensible claim

"Per-cell paired McNemar MDE (conservative bound at πD = 2·min(p,1-p)) = [7.3, 7.7, 5.2]pp for cls/red/shop. **PRIMARY H1 FE-pool gate empirical power = 81% (k=3 archive) / 97% (k=6 Phase 1a projection) at observed +2.34pp pooled drop-one** — strongly powered. The conservative per-cell MDE bound reflects the paired-test worst case (ρ=0); empirical πD ≈ 0.019 (≈10× smaller than the bound) indicates substantial paired-test benefit from real between-mode correlation. K-of-N is transparency-only per prereg §4 Decision 3A."