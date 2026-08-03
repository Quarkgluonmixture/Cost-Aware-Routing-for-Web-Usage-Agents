<!--
PARKED. Deliberately NOT in convert.sh's SECTIONS list for paperB.

This is branch B of the 2026-08-03 decision "does ② go in the body?". Branch A is the
two-sentence pointer that currently sits at the end of §3.3. Exactly one of the two should
be live:

  keep the pointer  -> do nothing; this file stays parked
  promote to a §    -> insert "section3b_structure_PARKED.md" into SECTIONS in convert.sh
                       between section3_ceiling.md and section4_supply.md, rename the file
                       without the _PARKED suffix, delete the pointer paragraph in §3.3,
                       renumber the tables after it (+1), and bump EXPECTED_TABLES.

Cost of promoting: roughly one page, on a body already 2 pages over the 8-page limit.

Recommendation on file (2026-08-01): keep the pointer. This is the one step of the four
that does not survive its own noise floor, and a page spent developing a result we then
withdraw is the least defensible page in the paper. The decision is the advisor's.
-->

## 3b. The routing region has axes, and they are smaller than a rerun

§3.3 established what one extra arm is worth when the arm carries no new information. This
section asks the finer question the pre-registration posed: within the three phantom arms, do
the two knobs (text payload and prompt family) each carry coverage the compound arm does not?

The pre-registered decomposition is two set differences, both against the compound arm P-SoM.
Axis-1 varies the text payload with the prompt family held fixed; axis-2 varies the prompt
family with the text payload held fixed.

| axis | estimand | θ_FE | bootstrap 95% CI | cells meeting the ≥2-task pre-registered floor |
|---|---|---|---|---|
| axis-1 | \|P-text \\ P-SoM\| | **1.3528 pp** | [0.799, 2.026] | 5 of 6 |
| axis-2 | \|P-prompt \\ P-SoM\| | **2.0877 pp** | [1.399, 2.919] | 5 of 6 |

*Table 3b: The pre-registered two-axis decomposition. Each axis counts, per cell, the tasks a
single-knob arm solves and the compound arm does not. Both intervals clear zero, and both
clear it after Holm correction over the two-axis family. The last column reports the
pre-registered admissibility floor of two uniquely solved tasks per cell, which is a
different quantity from the rerun floor of §3.3 and should not be read as one.*

On its own terms the decomposition passes: both axes pool clear of zero by four orders of
magnitude, so the two knobs are not two names for one intervention and the compound
configuration is not a superset of its parts. Axis-2 pools 1.5× higher than axis-1, which says
the arm that keeps the accessibility tree and changes only how the prompt describes it carries
more irreplaceable coverage than the arm that changes the text. We have no mechanism for that
asymmetry.

**We nonetheless do not report this as a result, because the effects are smaller than the
reference §3.3 measured.** Both axes are the same functional as the rerun drop, so the two are
directly comparable, and the comparison is unfavourable in every direction available:

| quantity | value |
|---|---|
| axis-1 pooled | 1.35 pp |
| axis-2 pooled | 2.09 pp |
| rerun drop, locally served backbone (the *lowest* floor we measured) | 2.00 – 4.00 pp |
| rerun drop, API-served backbone | 4.91 – 7.59 pp |

*Table 3c: The axes against the rerun references. Axis-1 falls below even the most permissive
floor; axis-2 sits inside its lower half.*

Passing a gate that tests "different from zero" is weak evidence here, because two runs of one
fixed configuration are also different from zero. The pre-registered gate was specified before
any rerun floor existed to compare against; having since measured one, we report the
decomposition as a measurement and withdraw the structural reading it was meant to license.

Two things this does *not* do. It does not overturn the sign, which is positive on both axes in
five of six cells. And it does not apply to §3's cost half, which is accuracy-neutral by
construction and involves no set difference. What it removes is the inference from "the axes
are non-zero" to "the region has structure a router could exploit" — an inference §§4–6 then
independently fail to cash in.
