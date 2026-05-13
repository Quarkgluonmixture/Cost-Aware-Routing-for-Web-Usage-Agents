# Axis-2 per-task fragility check

Per-task cosine gap distribution at L23 (axis-2 peak per §5.7 / Exp 1).
Each task averaged across its 2 steps; cosine gap computed between mode pairs.

**Defuse target**: /stress W2 attack — axis-2 mean 0.0114 might be dominated by 2-3 outlier tasks.

## Classifieds (24 tasks)

| Pair | Axis | Mean | Median | IQR | min | max | % > 0.005 | % > 0.010 | % > 0.020 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P-text ↔ P-SoM   (axis-2 flat-text) | axis-2 | 0.0073 | 0.0075 | [0.0066, 0.0081] | 0.0051 | 0.0093 | 100% | 0% | 0% |
| DOM ↔ P-prompt  (axis-2 hierarchical) | axis-2 | 0.0055 | 0.0055 | [0.0052, 0.0058] | 0.0045 | 0.0063 | 83% | 0% | 0% |
| DOM ↔ P-text     (axis-1 reference) | axis-1 | 0.0039 | 0.0036 | [0.0032, 0.0042] | 0.0026 | 0.0085 | 8% | 0% | 0% |
| P-SoM ↔ SoM     (axis-3 image ref) | axis-3 | 0.0324 | 0.0316 | [0.0292, 0.0354] | 0.0237 | 0.0466 | 100% | 100% | 100% |

## Reddit (24 tasks)

| Pair | Axis | Mean | Median | IQR | min | max | % > 0.005 | % > 0.010 | % > 0.020 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P-text ↔ P-SoM   (axis-2 flat-text) | axis-2 | 0.0070 | 0.0071 | [0.0062, 0.0075] | 0.0056 | 0.0085 | 100% | 0% | 0% |
| DOM ↔ P-prompt  (axis-2 hierarchical) | axis-2 | 0.0057 | 0.0056 | [0.0052, 0.0061] | 0.0045 | 0.0071 | 92% | 0% | 0% |
| DOM ↔ P-text     (axis-1 reference) | axis-1 | 0.0055 | 0.0049 | [0.0042, 0.0055] | 0.0023 | 0.0141 | 42% | 12% | 0% |
| P-SoM ↔ SoM     (axis-3 image ref) | axis-3 | 0.0332 | 0.0327 | [0.0291, 0.0353] | 0.0246 | 0.0502 | 100% | 100% | 100% |

## Top 5 axis-2 tasks (classifieds, P-text ↔ P-SoM @ L23)

| Task ID | Cosine gap |
|---|---:|
| 99 | 0.0093 |
| 214 | 0.0085 |
| 73 | 0.0085 |
| 109 | 0.0083 |
| 60 | 0.0082 |

## Bottom 5 axis-2 tasks (classifieds, P-text ↔ P-SoM @ L23)

| Task ID | Cosine gap |
|---|---:|
| 161 | 0.0065 |
| 32 | 0.0061 |
| 9 | 0.0061 |
| 40 | 0.0054 |
| 228 | 0.0051 |

## Top 5 axis-2 tasks (reddit, P-text ↔ P-SoM @ L23)

| Task ID | Cosine gap |
|---|---:|
| 113 | 0.0085 |
| 133 | 0.0081 |
| 125 | 0.0080 |
| 156 | 0.0078 |
| 147 | 0.0078 |

## Bottom 5 axis-2 tasks (reddit, P-text ↔ P-SoM @ L23)

| Task ID | Cosine gap |
|---|---:|
| 189 | 0.0060 |
| 162 | 0.0057 |
| 151 | 0.0057 |
| 190 | 0.0057 |
| 137 | 0.0056 |

## Verdict

Read the `% > 0.010` column for the axis-2 P-text↔P-SoM pair:
- cls: **0%** of 24 tasks above the L23 axis-2 mean magnitude
- reddit: **0%** of 24 tasks above

Interpretation tree:
- If both ≥ 50% → axis-2 signal **broad**, /stress W2 attack defused, §5.7 framing OK
- If both 25-50% → axis-2 signal **modest but present**, §5.7 needs to add 'task-conditional sparse' qualifier
- If both < 25% → axis-2 signal **aggregate artifact**, §5.7 three-axis claim must downgrade to 'axis-1 + image-axis with axis-2 weak per-task'

Median values: cls=0.0075, reddit=0.0071.
Compare to mean: cls=0.0073, reddit=0.0070.
If median << mean, the distribution is right-skewed → outlier-driven (consistent with /stress W2 attack).
