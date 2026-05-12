# Stage 4 Method 4.2: PCA Cosine Gap Analysis

**Data**: 144 examples × 37 layers × 6 modes (Qwen3-VL-4B B1 cls)
**Per-mode n**: DOM=24, P-text=24, P-prompt=24, P-SoM=24, SoM=24, Vision=24

**AUROC protocol** (Bug 3 fix, codex methodology audit 2026-05-12): paper-grade metric is `auroc_lototask` = leave-one-task-out cross-validation (fit mode-mean direction on training tasks, score held-out task). `auroc_in_sample` (fit + score on same examples) is reported for descriptive comparison only; treat any in-sample ≥0.95 as expected algebraic separability, NOT held-out linear-readability.

## Peak disruption layer per mode pair

Sorted by cosine gap magnitude (= geometric distance between mode means in hidden space):

| Mode pair | Peak layer | Cosine gap | AUROC (in-sample) | AUROC (lototask) |
|---|---|---|---|---|
| DOM vs Vision | L04 | 0.0658 | 1.000 | 1.000 |
| P-prompt vs Vision | L04 | 0.0634 | 1.000 | 1.000 |
| P-text vs Vision | L04 | 0.0590 | 1.000 | 1.000 |
| P-SoM vs Vision | L04 | 0.0586 | 1.000 | 1.000 |
| DOM vs SoM | L04 | 0.0455 | 1.000 | 1.000 |
| P-prompt vs SoM | L04 | 0.0434 | 1.000 | 1.000 |
| P-text vs SoM | L17 | 0.0433 | 1.000 | 1.000 |
| P-SoM vs SoM | L04 | 0.0386 | 1.000 | 1.000 |
| SoM vs Vision | L36 | 0.0193 | 1.000 | 1.000 |
| DOM vs P-SoM | L36 | 0.0122 | 1.000 | 1.000 |
| P-text vs P-prompt | L36 | 0.0074 | 1.000 | 1.000 |
| P-text vs P-SoM | L36 | 0.0069 | 1.000 | 1.000 |
| DOM vs P-prompt | L36 | 0.0063 | 1.000 | 1.000 |
| P-prompt vs P-SoM | L36 | 0.0042 | 1.000 | 1.000 |
| DOM vs P-text | L36 | 0.0037 | 1.000 | 1.000 |

## L17 cosine gap snapshot (paper §5 disruption locus)

| Mode pair | L17 cosine gap | L17 AUROC in-sample | L17 AUROC lototask |
|---|---|---|---|
| DOM vs P-text | 0.0019 | 1.000 | 1.000 |
| DOM vs P-prompt | 0.0015 | 1.000 | 1.000 |
| DOM vs P-SoM | 0.0031 | 1.000 | 1.000 |
| DOM vs SoM | 0.0450 | 1.000 | 1.000 |
| DOM vs Vision | 0.0537 | 1.000 | 1.000 |
| P-text vs P-prompt | 0.0032 | 1.000 | 1.000 |
| P-text vs P-SoM | 0.0020 | 1.000 | 1.000 |
| P-text vs SoM | 0.0433 | 1.000 | 1.000 |
| P-text vs Vision | 0.0513 | 1.000 | 1.000 |
| P-prompt vs P-SoM | 0.0016 | 1.000 | 1.000 |
| P-prompt vs SoM | 0.0392 | 1.000 | 1.000 |
| P-prompt vs Vision | 0.0492 | 1.000 | 1.000 |
| P-SoM vs SoM | 0.0367 | 1.000 | 1.000 |
| P-SoM vs Vision | 0.0468 | 1.000 | 1.000 |
| SoM vs Vision | 0.0130 | 1.000 | 1.000 |

## P-SoM vs baseline modes (paper §5 HERO arm)

P-SoM identity test: is P-SoM closer to SoM (prompt-axis sibling) or DOM (text-axis sibling)?

| L | P-SoM↔DOM | P-SoM↔SoM | P-SoM↔Vision | P-SoM↔P-text | P-SoM↔P-prompt |
|---|---|---|---|---|---|
| L00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| L08 | 0.0016 | 0.0241 | 0.0387 | 0.0003 | 0.0014 |
| L11 | 0.0023 | 0.0323 | 0.0407 | 0.0011 | 0.0014 |
| L17 | 0.0031 | 0.0367 | 0.0468 | 0.0020 | 0.0016 |
| L24 | 0.0055 | 0.0190 | 0.0257 | 0.0041 | 0.0016 |
| L30 | 0.0051 | 0.0147 | 0.0193 | 0.0035 | 0.0012 |
| L36 | 0.0122 | 0.0316 | 0.0429 | 0.0069 | 0.0042 |
