# Stage 4 Method 4.4 v2: Layer × α Sweep

**Config**: tier=strong, max_new_tokens=15
**Split**: train/eval (seed=20260513, n_train=16)
- Train task_ids (direction fit on these): `[1, 19, 33, 40, 60, 69, 82, 99, 108, 109, 122, 161, 181, 214, 215, 228]`
- Eval task_ids (held-out, headline numbers from these): `[9, 20, 32, 37, 61, 73, 116, 227]`
**Direction norms per layer (train-fit only)**: L11=2.15, L17=3.32, L23=9.36, L29=25.79, L33=45.35, L34=52.27
**N eval cells (task × step)**: 15
**N in-sample cells (task × step)**: 30

## Hero summary — held-out vs in-sample peak HDMI

- **Held-out best**: L33, α=20.0, H-mean=0.12
- **In-sample best**: L11, α=20.0, H-mean=0.29
- **Generalization gap (in_sample − held_out)**: +0.16 (different cell)

> ⚠️  **Reviewer-3 flag**: gap > 0.10 suggests direction may be over-fit to training cohort. Paper §5.3 should report held-out as headline.

## HDMI Reliability — harmonic mean (completeness × selectivity)

Following Khorasani et al. 2026 (arXiv:2605.07631): reliability = 2·c·s/(c+s). Penalizes "shift target but break envelope" failure mode. Higher = better.

### Held-out (paper-grade headline)

| Layer \ α | α=1.0 | α=2.0 | α=5.0 | α=10.0 | α=20.0 |
|---|---|---|---|---|---|
| L11 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| L17 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| L23 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| L29 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| L33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 |
| L34 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### In-sample (training cohort — for reviewer comparison only)

| Layer \ α | α=1.0 | α=2.0 | α=5.0 | α=10.0 | α=20.0 |
|---|---|---|---|---|---|
| L11 | 0.06 | 0.06 | 0.06 | 0.24 | 0.29 |
| L17 | 0.12 | 0.00 | 0.12 | 0.18 | 0.06 |
| L23 | 0.12 | 0.12 | 0.00 | 0.00 | 0.05 |
| L29 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 |
| L33 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 |
| L34 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Completeness (shifted-toward-P-SoM rate: overlap_psom > overlap_dom)

### Held-out (paper-grade headline)

| Layer \ α | α=1.0 | α=2.0 | α=5.0 | α=10.0 | α=20.0 |
|---|---|---|---|---|---|
| L11 | 0% | 0% | 0% | 0% | 0% |
| L17 | 0% | 0% | 0% | 0% | 0% |
| L23 | 0% | 0% | 0% | 0% | 0% |
| L29 | 0% | 0% | 0% | 0% | 0% |
| L33 | 0% | 0% | 0% | 0% | 7% |
| L34 | 0% | 0% | 0% | 0% | 7% |

### In-sample (training cohort — for reviewer comparison only)

| Layer \ α | α=1.0 | α=2.0 | α=5.0 | α=10.0 | α=20.0 |
|---|---|---|---|---|---|
| L11 | 3% | 3% | 3% | 13% | 17% |
| L17 | 7% | 0% | 7% | 10% | 3% |
| L23 | 7% | 7% | 0% | 0% | 13% |
| L29 | 7% | 7% | 0% | 0% | 10% |
| L33 | 0% | 0% | 7% | 10% | 0% |
| L34 | 0% | 0% | 0% | 10% | 10% |

## Selectivity (JSON envelope valid rate: steered output still starts with `{`)

### Held-out (paper-grade headline)

| Layer \ α | α=1.0 | α=2.0 | α=5.0 | α=10.0 | α=20.0 |
|---|---|---|---|---|---|
| L11 | 100% | 100% | 100% | 100% | 100% |
| L17 | 100% | 100% | 100% | 100% | 100% |
| L23 | 100% | 100% | 100% | 100% | 0% |
| L29 | 100% | 100% | 100% | 100% | 0% |
| L33 | 100% | 100% | 100% | 0% | 100% |
| L34 | 100% | 100% | 100% | 0% | 0% |

### In-sample (training cohort — for reviewer comparison only)

| Layer \ α | α=1.0 | α=2.0 | α=5.0 | α=10.0 | α=20.0 |
|---|---|---|---|---|---|
| L11 | 100% | 100% | 100% | 100% | 100% |
| L17 | 100% | 100% | 100% | 100% | 100% |
| L23 | 100% | 100% | 100% | 100% | 3% |
| L29 | 100% | 100% | 100% | 100% | 0% |
| L33 | 100% | 100% | 100% | 0% | 100% |
| L34 | 100% | 100% | 100% | 0% | 0% |

## Token overlap to DOM baseline (1.0 = identical, 0 = different)

### Held-out (paper-grade headline)

| Layer \ α | α=1.0 | α=2.0 | α=5.0 | α=10.0 | α=20.0 |
|---|---|---|---|---|---|
| L11 | 1.00 | 0.97 | 1.00 | 0.93 | 0.91 |
| L17 | 1.00 | 1.00 | 0.89 | 0.89 | 0.89 |
| L23 | 1.00 | 1.00 | 0.97 | 0.97 | 0.64 |
| L29 | 1.00 | 1.00 | 1.00 | 1.00 | 0.74 |
| L33 | 1.00 | 1.00 | 1.00 | 0.64 | 0.81 |
| L34 | 1.00 | 1.00 | 1.00 | 0.65 | 0.23 |

### In-sample (training cohort — for reviewer comparison only)

| Layer \ α | α=1.0 | α=2.0 | α=5.0 | α=10.0 | α=20.0 |
|---|---|---|---|---|---|
| L11 | 0.90 | 0.91 | 0.92 | 0.87 | 0.80 |
| L17 | 0.86 | 0.89 | 0.84 | 0.82 | 0.83 |
| L23 | 0.88 | 0.88 | 0.89 | 0.89 | 0.55 |
| L29 | 0.89 | 0.89 | 0.94 | 0.96 | 0.71 |
| L33 | 0.97 | 0.97 | 0.92 | 0.62 | 0.84 |
| L34 | 1.00 | 1.00 | 0.97 | 0.67 | 0.23 |

## Token overlap to P-SoM baseline

### Held-out (paper-grade headline)

| Layer \ α | α=1.0 | α=2.0 | α=5.0 | α=10.0 | α=20.0 |
|---|---|---|---|---|---|
| L11 | 0.69 | 0.69 | 0.69 | 0.67 | 0.68 |
| L17 | 0.69 | 0.69 | 0.66 | 0.66 | 0.66 |
| L23 | 0.69 | 0.69 | 0.69 | 0.69 | 0.47 |
| L29 | 0.69 | 0.69 | 0.69 | 0.69 | 0.57 |
| L33 | 0.69 | 0.69 | 0.69 | 0.51 | 0.66 |
| L34 | 0.69 | 0.69 | 0.69 | 0.51 | 0.23 |

### In-sample (training cohort — for reviewer comparison only)

| Layer \ α | α=1.0 | α=2.0 | α=5.0 | α=10.0 | α=20.0 |
|---|---|---|---|---|---|
| L11 | 0.64 | 0.64 | 0.65 | 0.70 | 0.66 |
| L17 | 0.64 | 0.62 | 0.59 | 0.60 | 0.57 |
| L23 | 0.65 | 0.66 | 0.60 | 0.63 | 0.45 |
| L29 | 0.65 | 0.65 | 0.63 | 0.66 | 0.57 |
| L33 | 0.66 | 0.67 | 0.66 | 0.51 | 0.60 |
| L34 | 0.68 | 0.68 | 0.66 | 0.55 | 0.23 |

