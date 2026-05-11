# Stage 4 Method 4.4: Counterfactual Activation Steering

**Config**: layer L17, steering direction = mean(P-SoM) − mean(DOM) at L17, ‖v‖=5.3753
**Tier**: strong cls × steps {2, 5} × n_tasks variable
**Max new tokens**: 15

## Aggregate per α (does adding α·v to DOM forward shift toward P-SoM?)

| α | n | mean overlap_DOM | mean overlap_P-SoM | shifted-toward-P-SoM rate | first-token P-SoM match |
|---|---|---|---|---|---|
| 0.5 | 4 | 1.000 | 0.689 | 0% | 100% |
| 1.0 | 4 | 0.833 | 0.576 | 25% | 100% |
| 2.0 | 4 | 0.833 | 0.576 | 25% | 100% |
| 5.0 | 4 | 0.800 | 0.543 | 25% | 100% |

Interpretation: if α=0 baseline overlap_DOM = 1.0 + overlap_P-SoM = some baseline,
then as α↑ overlap_P-SoM should rise + overlap_DOM should fall, monotonically.
Tool Calling paper (Anonymous 2026 ACL) reports 80-93% tool-switch rate at α=2-3.

