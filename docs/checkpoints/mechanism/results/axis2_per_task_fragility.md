# Axis-2 per-task fragility check

Per-task cosine gap distribution at L23 (axis-2 peak per §5.7 / Exp 1).
Each task averaged across its 2 steps; cosine gap computed between mode pairs.

**Defuse target**: /stress W2 attack — axis-2 mean 0.0114 might be dominated by 2-3 outlier tasks.

## Classifieds (24 tasks)

| Pair | Axis | Mean | Median | IQR | min | max | % > 0.005 | % > 0.010 | % > 0.020 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P-text ↔ P-SoM   (axis-2 flat-text) | axis-2 | 0.0132 | 0.0131 | [0.0124, 0.0142] | 0.0107 | 0.0174 | 100% | 100% | 0% |
| DOM ↔ P-prompt  (axis-2 hierarchical) | axis-2 | 0.0048 | 0.0047 | [0.0044, 0.0052] | 0.0039 | 0.0065 | 33% | 0% | 0% |
| DOM ↔ P-text     (axis-1 reference) | axis-1 | 0.0287 | 0.0280 | [0.0250, 0.0312] | 0.0186 | 0.0456 | 100% | 100% | 92% |
| P-SoM ↔ SoM     (axis-3 image ref) | axis-3 | 0.0407 | 0.0415 | [0.0353, 0.0438] | 0.0308 | 0.0597 | 100% | 100% | 100% |

## Reddit (24 tasks)

| Pair | Axis | Mean | Median | IQR | min | max | % > 0.005 | % > 0.010 | % > 0.020 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P-text ↔ P-SoM   (axis-2 flat-text) | axis-2 | 0.0121 | 0.0120 | [0.0113, 0.0127] | 0.0102 | 0.0152 | 100% | 100% | 0% |
| DOM ↔ P-prompt  (axis-2 hierarchical) | axis-2 | 0.0052 | 0.0051 | [0.0047, 0.0055] | 0.0039 | 0.0067 | 50% | 0% | 0% |
| DOM ↔ P-text     (axis-1 reference) | axis-1 | 0.0260 | 0.0263 | [0.0226, 0.0305] | 0.0174 | 0.0344 | 100% | 100% | 83% |
| P-SoM ↔ SoM     (axis-3 image ref) | axis-3 | 0.0436 | 0.0439 | [0.0409, 0.0453] | 0.0382 | 0.0535 | 100% | 100% | 100% |

## Top 5 axis-2 tasks (classifieds, P-text ↔ P-SoM @ L23)

| Task ID | Cosine gap |
|---|---:|
| 109 | 0.0174 |
| 211 | 0.0151 |
| 181 | 0.0146 |
| 108 | 0.0146 |
| 191 | 0.0143 |

## Bottom 5 axis-2 tasks (classifieds, P-text ↔ P-SoM @ L23)

| Task ID | Cosine gap |
|---|---:|
| 10 | 0.0121 |
| 228 | 0.0118 |
| 116 | 0.0117 |
| 32 | 0.0108 |
| 161 | 0.0107 |

## Top 5 axis-2 tasks (reddit, P-text ↔ P-SoM @ L23)

| Task ID | Cosine gap |
|---|---:|
| 133 | 0.0152 |
| 132 | 0.0145 |
| 142 | 0.0135 |
| 122 | 0.0135 |
| 148 | 0.0131 |

## Bottom 5 axis-2 tasks (reddit, P-text ↔ P-SoM @ L23)

| Task ID | Cosine gap |
|---|---:|
| 137 | 0.0111 |
| 107 | 0.0109 |
| 115 | 0.0104 |
| 116 | 0.0103 |
| 135 | 0.0102 |

## Verdict

Read the `% > 0.010` column for the axis-2 P-text↔P-SoM pair:
- cls: **100%** of 24 tasks above the L23 axis-2 mean magnitude
- reddit: **100%** of 24 tasks above

Interpretation tree:
- If both ≥ 50% → axis-2 signal **broad**, /stress W2 attack defused, §5.7 framing OK
- If both 25-50% → axis-2 signal **modest but present**, §5.7 needs to add 'task-conditional sparse' qualifier
- If both < 25% → axis-2 signal **aggregate artifact**, §5.7 three-axis claim must downgrade to 'axis-1 + image-axis with axis-2 weak per-task'

Median values: cls=0.0131, reddit=0.0120.
Compare to mean: cls=0.0132, reddit=0.0121.
If median << mean, the distribution is right-skewed → outlier-driven (consistent with /stress W2 attack).
