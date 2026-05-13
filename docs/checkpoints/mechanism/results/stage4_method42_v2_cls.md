# Stage 4 Method 4.2: PCA Cosine Gap Analysis

**Data**: 144 examples × 37 layers × 6 modes (Qwen3-VL-4B B1 cls)
**Per-mode n**: DOM=24, P-text=24, P-prompt=24, P-SoM=24, SoM=24, Vision=24

**AUROC protocol** (Bug 3 fix, codex methodology audit 2026-05-12): paper-grade metric is `auroc_lototask` = leave-one-task-out cross-validation (fit mode-mean direction on training tasks, score held-out task). `auroc_in_sample` (fit + score on same examples) is reported for descriptive comparison only; treat any in-sample ≥0.95 as expected algebraic separability, NOT held-out linear-readability.

## Peak disruption layer per mode pair

Sorted by cosine gap magnitude (= geometric distance between mode means in hidden space):

| Mode pair | Peak layer | Cosine gap | AUROC (in-sample) | AUROC (lototask) |
|---|---|---|---|---|
| DOM vs Vision | L04 | 0.0670 | 1.000 | 1.000 |
| P-prompt vs Vision | L04 | 0.0664 | 1.000 | 1.000 |
| P-text vs Vision | L04 | 0.0602 | 1.000 | 1.000 |
| P-SoM vs Vision | L04 | 0.0599 | 1.000 | 1.000 |
| DOM vs SoM | L36 | 0.0496 | 1.000 | 1.000 |
| P-text vs SoM | L36 | 0.0488 | 1.000 | 1.000 |
| P-prompt vs SoM | L36 | 0.0439 | 1.000 | 1.000 |
| P-SoM vs SoM | L36 | 0.0416 | 1.000 | 1.000 |
| SoM vs Vision | L36 | 0.0255 | 1.000 | 1.000 |
| DOM vs P-SoM | L36 | 0.0152 | 1.000 | 1.000 |
| P-text vs P-SoM | L36 | 0.0088 | 1.000 | 1.000 |
| P-text vs P-prompt | L36 | 0.0081 | 1.000 | 1.000 |
| DOM vs P-prompt | L36 | 0.0068 | 0.998 | 1.000 |
| P-prompt vs P-SoM | L36 | 0.0048 | 1.000 | 1.000 |
| DOM vs P-text | L36 | 0.0047 | 1.000 | 1.000 |

## L17 cosine gap snapshot (paper §5 disruption locus)

| Mode pair | L17 cosine gap | L17 AUROC in-sample | L17 AUROC lototask |
|---|---|---|---|
| DOM vs P-text | 0.0021 | 1.000 | 1.000 |
| DOM vs P-prompt | 0.0013 | 1.000 | 1.000 |
| DOM vs P-SoM | 0.0029 | 1.000 | 1.000 |
| DOM vs SoM | 0.0452 | 1.000 | 1.000 |
| DOM vs Vision | 0.0571 | 1.000 | 1.000 |
| P-text vs P-prompt | 0.0031 | 1.000 | 1.000 |
| P-text vs P-SoM | 0.0019 | 1.000 | 1.000 |
| P-text vs SoM | 0.0436 | 1.000 | 1.000 |
| P-text vs Vision | 0.0550 | 1.000 | 1.000 |
| P-prompt vs P-SoM | 0.0017 | 1.000 | 1.000 |
| P-prompt vs SoM | 0.0421 | 1.000 | 1.000 |
| P-prompt vs Vision | 0.0550 | 1.000 | 1.000 |
| P-SoM vs SoM | 0.0386 | 1.000 | 1.000 |
| P-SoM vs Vision | 0.0508 | 1.000 | 1.000 |
| SoM vs Vision | 0.0170 | 1.000 | 1.000 |

## P-SoM vs baseline modes (paper §5 HERO arm)

P-SoM identity test: is P-SoM closer to SoM (prompt-axis sibling) or DOM (text-axis sibling)?

| L | P-SoM↔DOM | P-SoM↔SoM | P-SoM↔Vision | P-SoM↔P-text | P-SoM↔P-prompt |
|---|---|---|---|---|---|
| L00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| L08 | 0.0016 | 0.0233 | 0.0398 | 0.0004 | 0.0014 |
| L11 | 0.0021 | 0.0314 | 0.0413 | 0.0010 | 0.0014 |
| L17 | 0.0029 | 0.0386 | 0.0508 | 0.0019 | 0.0017 |
| L24 | 0.0065 | 0.0212 | 0.0292 | 0.0051 | 0.0016 |
| L30 | 0.0069 | 0.0194 | 0.0226 | 0.0046 | 0.0014 |
| L36 | 0.0152 | 0.0416 | 0.0497 | 0.0088 | 0.0048 |
