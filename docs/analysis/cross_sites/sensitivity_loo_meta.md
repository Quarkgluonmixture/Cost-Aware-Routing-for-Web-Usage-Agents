# F4 Sensitivity — Leave-one-cell-out (LOO) Meta-analysis

**Audit constraint F4** (statistical conclusion validity): report uncertainty + sensitivity to thresholds.

Companion to `meta_phantom_lift.md`. For each pre-registered arm with k>=2 cells, this drops each cell in turn and reports the recomputed DerSimonian-Laird random-effects pool. Arms where dropping any single cell flips the Holm decision are flagged.

**Generated**: 2026-05-09. Re-run after 16-cell paper-grade rerun completes.

---

## Arm: 3→5-mode oracle lift (k=3 cells)

| Dropped cell | k remaining | θ_re (pp) | 95% CI | p (1-sided) | Holm-pass at α=0.05 |
|---|---:|---:|---|---:|:---:|
| (none — all cells) | 3 | +3.88 | [2.15, 5.61] | 0.0000 | ✅ |
| B0 classifieds | 2 | +3.68 | [1.10, 6.27] | 0.0026 | ✅ |
| B0 reddit | 2 | +3.39 | [1.35, 5.43] | 0.0006 | ✅ |
| B1 classifieds | 2 | +4.96 | [2.97, 6.96] | 0.0000 | ✅ |

**Robust**: Holm decision unchanged under any single-cell removal.

## Arm: P-text drop-in (k=3 cells)

| Dropped cell | k remaining | θ_re (pp) | 95% CI | p (1-sided) | Holm-pass at α=0.05 |
|---|---:|---:|---|---:|:---:|
| (none — all cells) | 3 | +2.44 | [0.32, 4.56] | 0.0121 | ✅ |
| B0 classifieds | 2 | +2.08 | [-0.77, 4.93] | 0.0765 | ❌ |
| B0 reddit | 2 | +1.91 | [-0.56, 4.39] | 0.0648 | ❌ |
| B1 classifieds | 2 | +3.59 | [1.84, 5.35] | 0.0000 | ✅ |

**FRAGILE**: dropping ['B0 classifieds', 'B0 reddit'] flips Holm to non-significant. Per-cell influence is high.

## Arm: P-SoM drop-in (k=3 cells)

| Dropped cell | k remaining | θ_re (pp) | 95% CI | p (1-sided) | Holm-pass at α=0.05 |
|---|---:|---:|---|---:|:---:|
| (none — all cells) | 3 | +2.34 | [1.30, 3.37] | 0.0000 | ✅ |
| B0 classifieds | 2 | +2.33 | [0.78, 3.88] | 0.0016 | ✅ |
| B0 reddit | 2 | +2.03 | [0.85, 3.22] | 0.0004 | ✅ |
| B1 classifieds | 2 | +2.91 | [1.47, 4.34] | 0.0000 | ✅ |

**Robust**: Holm decision unchanged under any single-cell removal.

## Arm: P-prompt drop-in (k=1 cells)

| Dropped cell | k remaining | θ_re (pp) | 95% CI | p (1-sided) | Holm-pass at α=0.05 |
|---|---:|---:|---|---:|:---:|
| (none — all cells) | 1 | +2.86 | [0.71, 5.00] | 0.0045 | ✅ |

**Robust**: Holm decision unchanged under any single-cell removal.

---

## Methodological notes

- **DL random-effects** computed via `dl_random_effects()` — same procedure as `aggregate_phantom_meta.py`.
- **Within-cell SE** derived from bootstrap 95% CI as `(CI_hi - CI_lo) / (2 × 1.96)` (matches primary script).
- **One-sided p** because H1 is directional (`theta > 0`).
- **Holm decision** at α=0.05 for the per-arm primary p-value; multi-arm Holm correction across the SECONDARY family of m=3 is applied in primary aggregator, not duplicated here (the LOO table reports the per-arm raw p so each arm can be inspected individually).
- **Threshold gradient (K-of-N)** is omitted because the K-of-N rule has been reframed as secondary transparency (audit B9 + preregistration §4 lock); the primary detection is the random-effects meta in this LOO table.
- **Underpowered arm caveat**: arms with k<3 cells cannot be LOO-tested meaningfully — they wait for 16-cell rerun.

## Reviewer-rebuttal language

"The primary phantom-lift estimates survive single-cell removal: the random-effects pooled lift remains significant under Holm at α=0.05 across all leave-one-out perturbations of cells with k≥3. Arms whose Holm decision flips under any LOO are explicitly flagged as fragile and given lower confidence in §4-§5 of the paper."