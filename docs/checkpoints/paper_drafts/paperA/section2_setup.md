## 2. The phantom routing space

### 2.1 Two knobs on the image-off boundary

A DOM agent receives the accessibility tree as text under a prompt that describes it as
such. A Set-of-Mark agent [@yang2023som] receives a marked screenshot plus a legend mapping
mark numbers to elements, under a prompt that describes that pairing. Between them sit
configurations that nobody deploys deliberately, and they are what we study.

Holding the screenshot off, two properties of the DOM baseline can be varied
independently:

| arm | text payload | prompt family | image |
|---|---|---|---|
| DOM (origin) | accessibility tree | DOM | off |
| **P-text** | mark legend | DOM | off |
| **P-prompt** | accessibility tree | SoM | off |
| **P-SoM** | mark legend | SoM | off |
| SoM (reference) | mark legend | SoM | **on** |
| Vision (reference) | none | vision | **on** |

*Table 1: The six modes as coordinates in text payload, prompt family, and image. The three
phantom arms are the image-off cells that differ from the DOM origin; the two image-bearing
modes are references, not members.*

DOM is the origin, not a member: the axes are defined as displacements from it, so
including it as an arm would make the comparison tautological. The two image-bearing modes
are references that bound the space from outside.

The mark legend deserves precision, because it is the object the text axis moves. Under
VisualWebArena's accessibility-tree serialization [@koh2024visualwebarena], elements already
carry numeric ids. The
legend is produced by a regex filter over that same text, followed by a deterministic
renumber to a compact 1..K sequence, with a map back to native ids kept only for action
dispatch. No bounding boxes are computed and no image is touched. This is why the cost
consequence in §3.4 is a property of the construction rather than a measurement.

### 2.2 Why the compound arm is not obviously redundant

P-SoM applies both displacements at once. If the two knobs were two descriptions of one
intervention, P-SoM would dominate or duplicate the single-axis arms and the space would
collapse to a line. The preregistered structural hypothesis (§3.2) is the test of that
collapse: it asks whether each single-axis arm solves tasks the compound arm misses.

The prompt-family axis is the stranger of the two. P-prompt gives the model a prompt that
announces an annotated screenshot, and then supplies the accessibility tree with no image.
This is a deliberate mismatch. It is not a bug in the harness: the mode routes to the SoM
system prompt and the native accessibility-tree text by design, with element ids identical
to those the DOM agent uses.

### 2.3 Design

Six preregistered cells: two VisualWebArena sites [@koh2024visualwebarena] (classifieds,
reddit) crossed with three
backbones. **B0** is Qwen3-VL-235B-A22B served through an API. **B1** is Qwen3-VL-4B served
locally. **B2** is Gemma3-VL-4B, also local, from a different model family. B0 against B1
spans capability within one family; B1 against B2 spans families at matched parameter
count. Neither contrast is a clean single-variable ablation, and §5 records the asymmetries
we did not control.

Each cell runs all six modes over the same task set: 224 classifieds tasks and 203 reddit
tasks. Success is the benchmark's own binary judgement.

The reddit denominator requires disclosure. It is 205 as collected and 203 as scored, because
two reddit tasks were removed from the scored set by a protocol amendment that we wrote
**after all six modes had run in all six cells and after we had inspected the outcomes**. It
is a post-hoc, outcome-visible amendment and we do not present it as preregistered. One
removed task has an evaluator that scores inaction as success; the other has a reference
answer recoverable without visiting the second site its own configuration declares. Both
criteria were applied as uniform rules over the whole reddit task pool rather than
task-by-task, and each rule selects exactly one task. The amendment moves the pooled reddit
success rate from 6.94% to 6.40%, and it removes drop-one credit from P-SoM in the one cell
where it touches the H1 estimand at all, so it works against this paper's own hypothesis.
Sensitivity to including both tasks is reported in Appendix D.

### 2.4 What was preregistered

Three hypotheses, fixed before Phase 1a data collection, with their decision rules:

**H1 (superiority).** P-SoM's drop-one oracle contribution, pooled by inverse-variance
fixed effects over all six cells, exceeds a margin of δ = 1.0 pp. One-sided, α = 0.05,
tested by paired-bootstrap percentile. This is a single pooled test, not a family of
per-cell tests.

**H2(a) (cost).** The per-task median cost ratio of P-SoM to DOM falls within ±20% of 1.0
in every cell. Any cell outside the band falsifies it.

**H3 (structure).** Two axes, each a fixed-effects pool of per-cell unique-task counts:
axis-1 counts tasks solved by P-text and not P-SoM; axis-2 counts tasks solved by P-prompt
and not P-SoM. The gate is a bootstrap percentile CI whose lower bound excludes zero,
with Holm correction [@holm1979sequentially] over the two-axis family.

A degenerate-cell floor of 1.0 pp applies to any cell whose paired-bootstrap standard error
falls below the Agresti-Coull threshold [@agresti1998approximate] of 0.68 pp, so that a
zero-information cell cannot receive unbounded weight in the pool. Four cells triggered it
in H1.

The preregistered framing rule maps outcomes to claim tiers. H1 failing sends the paper to
tier R5 with a structural pivot, which is the situation we are in and which §3 reports as
such.
