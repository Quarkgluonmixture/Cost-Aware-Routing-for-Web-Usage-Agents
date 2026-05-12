# Stage 4 Method 4.2: PCA Cosine Gap Analysis

**Data**: 288 examples × 37 layers × 6 modes (Qwen3-VL-4B B1 cls)
**Per-mode n**: DOM=48, P-text=48, P-prompt=48, P-SoM=48, SoM=48, Vision=48

## Peak disruption layer per mode pair

Sorted by cosine gap magnitude (= geometric distance between mode means in hidden space):

| Mode pair | Peak layer | Cosine gap | AUROC at peak |
|---|---|---|---|
| DOM vs Vision | L04 | 0.0687 | 1.000 |
| P-prompt vs Vision | L04 | 0.0671 | 1.000 |
| DOM vs SoM | L04 | 0.0643 | 1.000 |
| P-prompt vs SoM | L04 | 0.0626 | 1.000 |
| P-text vs Vision | L36 | 0.0618 | 1.000 |
| P-SoM vs Vision | L36 | 0.0614 | 1.000 |
| P-text vs SoM | L22 | 0.0492 | 1.000 |
| P-SoM vs SoM | L04 | 0.0434 | 1.000 |
| DOM vs P-SoM | L23 | 0.0272 | 1.000 |
| P-text vs P-prompt | L23 | 0.0249 | 1.000 |
| P-prompt vs P-SoM | L23 | 0.0240 | 1.000 |
| SoM vs Vision | L36 | 0.0217 | 1.000 |
| DOM vs P-text | L23 | 0.0217 | 1.000 |
| P-text vs P-SoM | L23 | 0.0098 | 1.000 |
| DOM vs P-prompt | L36 | 0.0059 | 0.994 |

## L17 cosine gap snapshot (paper §5 disruption locus)

| Mode pair | L17 cosine gap | L17 AUROC |
|---|---|---|
| DOM vs P-text | 0.0092 | 1.000 |
| DOM vs P-prompt | 0.0012 | 1.000 |
| DOM vs P-SoM | 0.0098 | 1.000 |
| DOM vs SoM | 0.0543 | 1.000 |
| DOM vs Vision | 0.0537 | 1.000 |
| P-text vs P-prompt | 0.0102 | 1.000 |
| P-text vs P-SoM | 0.0027 | 1.000 |
| P-text vs SoM | 0.0479 | 1.000 |
| P-text vs Vision | 0.0476 | 1.000 |
| P-prompt vs P-SoM | 0.0086 | 1.000 |
| P-prompt vs SoM | 0.0510 | 1.000 |
| P-prompt vs Vision | 0.0515 | 1.000 |
| P-SoM vs SoM | 0.0423 | 1.000 |
| P-SoM vs Vision | 0.0457 | 1.000 |
| SoM vs Vision | 0.0091 | 1.000 |

## P-SoM vs baseline modes (paper §5 HERO arm)

P-SoM identity test: is P-SoM closer to SoM (prompt-axis sibling) or DOM (text-axis sibling)?

| L | P-SoM↔DOM | P-SoM↔SoM | P-SoM↔Vision | P-SoM↔P-text | P-SoM↔P-prompt |
|---|---|---|---|---|---|
| L00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| L08 | 0.0098 | 0.0307 | 0.0321 | 0.0006 | 0.0096 |
| L11 | 0.0086 | 0.0379 | 0.0385 | 0.0011 | 0.0087 |
| L17 | 0.0098 | 0.0423 | 0.0457 | 0.0027 | 0.0086 |
| L24 | 0.0203 | 0.0303 | 0.0452 | 0.0071 | 0.0170 |
| L30 | 0.0127 | 0.0219 | 0.0302 | 0.0036 | 0.0088 |
| L36 | 0.0252 | 0.0434 | 0.0614 | 0.0080 | 0.0176 |
