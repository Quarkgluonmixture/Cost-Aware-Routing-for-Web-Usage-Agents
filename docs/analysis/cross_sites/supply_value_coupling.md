---
type: analysis
status: complete
purpose: whether the supply of routing labels and the value of routing are one quantity
post_hoc_exploratory: true
producer: scripts/analysis/aggregate_supply_value_coupling.py
---

# Supply and value are the same set

Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_supply_value_coupling.py`

Both quantities below are already in Table 42. This asks only whether they move together.

| cell | n | best single mode | best SR | routable set (>1 solver) | solvable (oracle) | routable / solvable |
|---|---:|---|---:|---:|---:|---:|
| `wa_red_B0` | 104 | P-text | 35.58% | 34.6% (36) | 51.9% | 0.67 |
| `cls_B0` | 224 | SoM | 27.23% | 30.4% (68) | 43.3% | 0.70 |
| `wa_red_B1` | 104 | DOM | 16.35% | 17.3% (18) | 30.8% | 0.56 |
| `red_B0` | 203 | SoM | 14.78% | 17.7% (36) | 26.1% | 0.68 |
| `cls_B1` | 224 | SoM | 14.29% | 12.9% (29) | 24.6% | 0.53 |
| `red_B1` | 203 | SoM | 7.39% | 8.4% (17) | 11.8% | 0.71 |
| `red_B2` | 203 | DOM | 3.94% | 1.5% (3) | 7.4% | 0.20 |
| `cls_B2` | 224 | SoM | 2.23% | 1.8% (4) | 7.1% | 0.25 |

**Spearman rho = 0.952** (exact permutation p = 0.0011 over 40,320 pairings), Pearson r = 0.987, mean |best SR - routable share| = **1.65pp**.

The routable set is **0.53-0.71** of the solvable set in the 6 cells carrying more than 4 routable tasks. The two near-floor cells (`red_B2`, `cls_B2`) sit at 0.20 on **3 and 4 tasks** -- a ratio one task wide, quoted separately rather than widening the range.

## What this licenses, and what it does not

**Licensed.** The set a router can learn from and the set where routing can pay are the same set. So the two obstructions the lower bound reports -- too few labels, too few contested tasks -- are not two independent walls; they are one wall, whose height is set by how many tasks are solvable at all.

**Not licensed: reading this as a surprising empirical law.** Both columns are functionals of one solve matrix and both are subsets of the solvable set, so a positive association is partly structural. The empirical content is the *near-linearity*: under mode independence the routable share would grow near-quadratically in the per-mode rate at these success levels, and it does not.

**Not licensed: a constant conversion factor.** The ratio is not flat -- it falls to 0.25 in the two near-floor cells. Whether that is a floor effect on 3-4 tasks or a real bend at low capability cannot be told apart here, and the direction matters: if real, the routable set shrinks *faster* than the solvable set as capability drops.

**Not licensed: a forecast.** That both quantities rise with capability is what the coupling implies; how fast, and whether the ratio holds outside 2-36% success, is not measured here. It is stated in the paper as the condition under which the negative result would be overturned, i.e. as a falsifiable prediction, not as a projection.

## Caveats

- The 8 cells are not independent: they share 3 sites and 3 backbones, so the permutation null tests whether the pairing is arbitrary, not whether 8 independent systems were sampled.
- Both columns are functionals of the same per-task solve matrix, and both are subsets of the solvable set. Part of the association is therefore structural, not empirical. The non-structural content is the SIZE of the ratio routable/solvable, which is 0.53-0.71 above the floor and is NOT constant -- it halves in the two near-floor cells.
- Under mode independence the routable share would be a convex (near-quadratic at low success) function of the per-mode rate, not the near-identity observed here; the observed near-linearity is a statement about how strongly task difficulty dominates mode-task matching.
