# Axis-2 per-task fragility check

Per-task cosine gap distribution at L17 (axis-2 peak per §5.7 / Exp 1).
Each task averaged across its 2 steps; cosine gap computed between mode pairs.

**Defuse target**: /stress W2 attack — axis-2 mean 0.0114 might be dominated by 2-3 outlier tasks.

## Classifieds (24 tasks)

| Pair | Axis | Mean | Median | IQR | min | max | % > 0.005 | % > 0.010 | % > 0.020 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P-text ↔ P-SoM   (axis-2 flat-text) | axis-2 | 0.0035 | 0.0032 | [0.0028, 0.0038] | 0.0025 | 0.0059 | 12% | 0% | 0% |
| DOM ↔ P-prompt  (axis-2 hierarchical) | axis-2 | 0.0028 | 0.0028 | [0.0024, 0.0032] | 0.0021 | 0.0042 | 0% | 0% | 0% |
| DOM ↔ P-text     (axis-1 reference) | axis-1 | 0.0037 | 0.0035 | [0.0031, 0.0044] | 0.0026 | 0.0053 | 8% | 0% | 0% |
| P-SoM ↔ SoM     (axis-3 image ref) | axis-3 | 0.0406 | 0.0403 | [0.0387, 0.0426] | 0.0364 | 0.0466 | 100% | 100% | 100% |

## Reddit (24 tasks)

| Pair | Axis | Mean | Median | IQR | min | max | % > 0.005 | % > 0.010 | % > 0.020 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P-text ↔ P-SoM   (axis-2 flat-text) | axis-2 | 0.0037 | 0.0037 | [0.0033, 0.0040] | 0.0028 | 0.0050 | 0% | 0% | 0% |
| DOM ↔ P-prompt  (axis-2 hierarchical) | axis-2 | 0.0028 | 0.0028 | [0.0025, 0.0031] | 0.0022 | 0.0036 | 0% | 0% | 0% |
| DOM ↔ P-text     (axis-1 reference) | axis-1 | 0.0035 | 0.0034 | [0.0030, 0.0038] | 0.0025 | 0.0061 | 4% | 0% | 0% |
| P-SoM ↔ SoM     (axis-3 image ref) | axis-3 | 0.0404 | 0.0405 | [0.0386, 0.0421] | 0.0352 | 0.0443 | 100% | 100% | 100% |

## Top 5 axis-2 tasks (classifieds, P-text ↔ P-SoM @ L23)

| Task ID | Cosine gap |
|---|---:|
| 73 | 0.0059 |
| 1 | 0.0058 |
| 116 | 0.0054 |
| 215 | 0.0043 |
| 161 | 0.0041 |

## Bottom 5 axis-2 tasks (classifieds, P-text ↔ P-SoM @ L23)

| Task ID | Cosine gap |
|---|---:|
| 61 | 0.0028 |
| 40 | 0.0028 |
| 33 | 0.0027 |
| 32 | 0.0026 |
| 228 | 0.0025 |

## Top 5 axis-2 tasks (reddit, P-text ↔ P-SoM @ L23)

| Task ID | Cosine gap |
|---|---:|
| 147 | 0.0050 |
| 188 | 0.0047 |
| 115 | 0.0043 |
| 156 | 0.0041 |
| 124 | 0.0041 |

## Bottom 5 axis-2 tasks (reddit, P-text ↔ P-SoM @ L23)

| Task ID | Cosine gap |
|---|---:|
| 189 | 0.0031 |
| 190 | 0.0031 |
| 174 | 0.0030 |
| 194 | 0.0030 |
| 137 | 0.0028 |

## Verdict

Read the `% > 0.010` column for the axis-2 P-text↔P-SoM pair:
- cls: **0%** of 24 tasks above the L23 axis-2 mean magnitude
- reddit: **0%** of 24 tasks above

Interpretation tree:
- If both ≥ 50% → axis-2 signal **broad**, /stress W2 attack defused, §5.7 framing OK
- If both 25-50% → axis-2 signal **modest but present**, §5.7 needs to add 'task-conditional sparse' qualifier
- If both < 25% → axis-2 signal **aggregate artifact**, §5.7 three-axis claim must downgrade to 'axis-1 + image-axis with axis-2 weak per-task'

Median values: cls=0.0032, reddit=0.0037.
Compare to mean: cls=0.0035, reddit=0.0037.
If median << mean, the distribution is right-skewed → outlier-driven (consistent with /stress W2 attack).
