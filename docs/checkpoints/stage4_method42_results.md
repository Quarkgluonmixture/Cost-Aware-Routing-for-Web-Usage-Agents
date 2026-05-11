# Stage 4 Method 4.2: PCA Cosine Gap Analysis

**Data**: 288 examples × 37 layers × 6 modes (Qwen3-VL-4B B1 cls)
**Per-mode n**: DOM=48, P-text=48, P-prompt=48, P-SoM=48, SoM=48, Vision=48

## Peak disruption layer per mode pair

Sorted by cosine gap magnitude (= geometric distance between mode means in hidden space):

| Mode pair | Peak layer | Cosine gap | AUROC at peak |
|---|---|---|---|
| DOM vs Vision | L04 | 0.0653 | 1.000 |
| P-prompt vs Vision | L04 | 0.0649 | 1.000 |
| P-text vs Vision | L36 | 0.0614 | 1.000 |
| P-SoM vs Vision | L36 | 0.0613 | 1.000 |
| DOM vs SoM | L04 | 0.0604 | 1.000 |
| P-prompt vs SoM | L04 | 0.0600 | 1.000 |
| P-text vs SoM | L20 | 0.0494 | 1.000 |
| P-SoM vs SoM | L17 | 0.0412 | 1.000 |
| DOM vs P-SoM | L23 | 0.0321 | 1.000 |
| P-prompt vs P-SoM | L23 | 0.0292 | 1.000 |
| P-text vs P-prompt | L23 | 0.0288 | 1.000 |
| DOM vs P-text | L23 | 0.0254 | 1.000 |
| SoM vs Vision | L22 | 0.0238 | 1.000 |
| P-text vs P-SoM | L23 | 0.0114 | 1.000 |
| DOM vs P-prompt | L36 | 0.0067 | 0.998 |

## L17 cosine gap snapshot (paper §5 disruption locus)

| Mode pair | L17 cosine gap | L17 AUROC |
|---|---|---|
| DOM vs P-text | 0.0120 | 1.000 |
| DOM vs P-prompt | 0.0013 | 1.000 |
| DOM vs P-SoM | 0.0124 | 1.000 |
| DOM vs SoM | 0.0557 | 1.000 |
| DOM vs Vision | 0.0545 | 1.000 |
| P-text vs P-prompt | 0.0132 | 1.000 |
| P-text vs P-SoM | 0.0028 | 1.000 |
| P-text vs SoM | 0.0466 | 1.000 |
| P-text vs Vision | 0.0458 | 1.000 |
| P-prompt vs P-SoM | 0.0113 | 1.000 |
| P-prompt vs SoM | 0.0529 | 1.000 |
| P-prompt vs Vision | 0.0526 | 1.000 |
| P-SoM vs SoM | 0.0412 | 1.000 |
| P-SoM vs Vision | 0.0440 | 1.000 |
| SoM vs Vision | 0.0090 | 1.000 |

## P-SoM vs baseline modes (paper §5 HERO arm)

P-SoM identity test: is P-SoM closer to SoM (prompt-axis sibling) or DOM (text-axis sibling)?

| L | P-SoM↔DOM | P-SoM↔SoM | P-SoM↔Vision | P-SoM↔P-text | P-SoM↔P-prompt |
|---|---|---|---|---|---|
| L00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| L08 | 0.0103 | 0.0250 | 0.0268 | 0.0006 | 0.0101 |
| L11 | 0.0096 | 0.0329 | 0.0339 | 0.0011 | 0.0092 |
| L17 | 0.0124 | 0.0412 | 0.0440 | 0.0028 | 0.0113 |
| L24 | 0.0234 | 0.0291 | 0.0455 | 0.0082 | 0.0206 |
| L30 | 0.0150 | 0.0215 | 0.0309 | 0.0041 | 0.0109 |
| L36 | 0.0278 | 0.0411 | 0.0613 | 0.0089 | 0.0201 |
