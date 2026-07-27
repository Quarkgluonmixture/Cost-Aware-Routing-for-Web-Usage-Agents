## 3. Results

### 3.1 H1 fails, and the failure is precise rather than noisy

| quantity | value |
|---|---|
| pooled θ_FE | **0.7897 pp** |
| paired-bootstrap pooled median | 0.7490 pp |
| bootstrap 95% CI | [0.2858, 1.4471] |
| preregistered margin δ | 1.0 pp |
| one-sided bootstrap p | **0.807** |
| verdict | **FAIL** |
| cells contributing | 6 of 6 |
| cells hitting the SE floor | 4 |
| Cochran Q (df = 5) | 1.4265 |
| **I²** | **0.0%** |

*Table 2: H1, the preregistered superiority test on P-SoM's drop-one oracle contribution.
The verdict is FAIL, and I² = 0.0% locates the failure in the effect size rather than in the
precision.*

P-SoM's drop-one oracle contribution pools below the preregistered margin, and the test
does not come close: p = 0.807 against α = 0.05.

The heterogeneity statistics [@higgins2002quantifying] are what make this a usable negative
result rather than an inconclusive one. I² = 0.0% with Q = 1.43 on 5 degrees of freedom says the six cells are
consistent with a common small effect. Two readings are therefore excluded. It is not one
outlying cell dragging the pool, and it is not six noisy cells whose intervals happen to
straddle the margin. Collecting more cells from the same design would narrow the interval
around a value beneath δ, not carry it across.

Four cells hit the degenerate-cell SE floor, which merits a note because the direction is
counter-intuitive. Flooring a cell's standard error at 1.0 pp *reduces* its inverse-variance
weight. The floored cells are the ones with the smallest effects and the smallest standard
errors, so the floor moved θ_FE **upward**. A rule adopted to stop zero-information cells
from hijacking the pool ends up flattering the hypothesis it constrains. Under the
superseded `SE = 0 exactly` trigger the same data pools to 0.6533 pp, further from the
margin. The preregistered value is the one reported above; the alternative appears in
Appendix D as a labelled robustness row, and both fail.

### 3.2 H3 passes on both axes: the compound arm absorbs neither single-axis arm

| axis | estimand | θ_FE | bootstrap 95% CI | Wald p (Holm, m=2) | cells above noise floor | individually Holm-significant |
|---|---|---|---|---|---|---|
| axis-1 | \|P-text \\ P-SoM\| | **1.3528 pp** | **[0.799, 2.026]** | 1.19 × 10⁻⁵ | 5 of 6 | 3 of 6 |
| axis-2 | \|P-prompt \\ P-SoM\| | **2.0877 pp** | **[1.399, 2.919]** | 7.52 × 10⁻⁷ | 5 of 6 | 4 of 6 |

*Table 3: H3, the preregistered decomposition. Each axis counts per cell the tasks a
single-axis arm solves and the compound arm does not. Both pool clear of zero after Holm
correction over the two-axis family.*

Each axis counts, per cell, the tasks a single-axis arm solves and the compound arm does
not. Both pool well clear of zero and both survive Holm correction over the two-axis family.
The preregistered noise floor of two unique tasks is exceeded in five of six cells on each
axis.

This is the paper's positive result. Moving only
the text payload leaves behind tasks that moving both payload and prompt does not recover.
Moving only the prompt family does the same, more strongly. The two knobs are not two names
for one intervention, and the compound configuration is not a superset of its parts. That
is what licenses calling this region a space with axes.

The asymmetry between the axes is itself informative: axis-2 (prompt family alone) pools
1.5× higher than axis-1 (text format alone). The arm that keeps the accessibility tree and
merely changes how the prompt describes it carries more irreplaceable coverage than the arm
that changes the text. We do not have a mechanism for this; §4 reports the behavioural
correlates we can measure.

### 3.3 Single-arm success rates, and what they do not show

| cell | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| classifieds · B0 | 17.41 | **27.23** | 25.00 | 15.62 | 19.64 | 15.62 |
| classifieds · B1 | 6.25 | **14.29** | 12.50 | 7.59 | 6.70 | 6.70 |
| classifieds · B2 | 1.34 | **2.23** | **2.23** | 0.45 | 1.79 | 0.89 |
| reddit · B0 | 14.29 | **14.78** | 7.39 | 13.30 | 12.32 | 10.84 |
| reddit · B1 | 5.91 | **7.39** | 2.46 | 5.91 | 5.42 | 5.91 |
| reddit · B2 | **3.94** | 0.99 | 1.97 | 1.97 | 0.00 | 0.49 |

*Table 4: Success rate (%) per mode per cell, bold for the best arm in each row. Full SoM
leads in five of six cells. Single-arm rates are reported for context; the paper's claims
rest on the per-task set differences reported in §3.1 and §3.2, not on these means.*

Full SoM is the strongest single arm in five of six cells, and on classifieds the margin is
large: 27.23% against P-SoM's 15.62% at B0. **We do not claim that any phantom arm replaces
SoM.** Where the marked screenshot resolves elements the text does not, it earns its cost.

Two rows deserve comment. On **reddit · B0** the six arms fall within 7.4 percentage points
of each other and DOM is within 0.5 pp of SoM, which is the regime where the image adds
least. On **reddit · B2** the ordering inverts: DOM is strongest at 3.94% and P-prompt scores
zero. That cell's single success under P-prompt was a task later excluded from the scored
set by protocol amendment, so the arm's scored rate is exactly 0 of 203. We report it
rather than smoothing it; a 4B cross-family model at 0.5–4% success is near the floor where
mode comparisons stop discriminating, and §5 treats that as a scope limit.

Complementarity, in contrast, is a per-task property and survives these rates: **P-SoM
uniquely solves 6 classifieds and 3 reddit tasks** that none of the other five modes solves.
A mode can be mediocre on average and still be the only arm that reaches those tasks.

### 3.4 Cost stays in the DOM band, by construction

H2(a) is not falsified: the per-task median cost ratio of P-SoM to DOM lies within the
preregistered ±20% band in **all six cells**, over **1,281 paired tasks**, with no cell
excluded for a zero-cost DOM denominator.

We flag the epistemic status deliberately. This is a consequence of how the mark legend is
built (a regex pass over text the DOM agent already receives, then a renumber) rather than a
discovery about model behaviour. A reader should treat it as a check that the construction
does what it claims, and should not read it as evidence that phantom arms are a cheaper
route to SoM's accuracy. §3.3 shows they are not.

Cost is never pooled across backbones. B0 reports commercial API pricing; B1 and B2 report
electricity-derived cost. The two bases differ by roughly three orders of magnitude per
token, so only within-backbone ratios are compared anywhere in this paper.

### 3.5 Where this leaves the preregistered framing

The decision rule fixed before data collection maps a failed H1 to tier **R5** with a
structural pivot. That is the outcome. The paper claims:

- the phantom routing space is **structured**, decomposable along a text axis and a prompt
  axis, neither of which the compound arm absorbs (§3.2);
- P-SoM is **not** a superior deployment arm, by the preregistered test (§3.1);
- P-SoM's coverage is **complementary but small**, at 6 and 3 unique tasks (§3.3);
- the cost profile is **inherited from DOM by construction**, not measured as a saving
  (§3.4).

Earlier drafts of this work led with a drop-one oracle contribution of 3.33 pp on reddit and
2.56 pp on classifieds, computed over a four-mode archive universe. Under the six-mode
universe with all six cells the same estimand measures 0.0–1.3 pp per cell and fails its
gate. Those archive figures are reported in Appendix D as a sensitivity row and should not
be cited as this paper's effect size.
