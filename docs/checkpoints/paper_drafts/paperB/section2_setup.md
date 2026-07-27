## 2. Setup

### 2.1 Modes, backbones, sites

We evaluate on VisualWebArena [@koh2024visualwebarena], on the classifieds and reddit sites.
Every task is run under six observation modes that differ in what the agent receives at each
step:

DOM sends the accessibility tree under a prompt that names it an accessibility tree. SoM sends
a screenshot with
numbered marks plus a legend naming them. Vision sends an unannotated screenshot and no text
payload. The three P-modes keep the mark legend, the SoM prompt, or both, while removing the
per-step screenshot. Appendix A.1 gives the full grid.

The modes matter here only through what they cost, and the measured costs do not follow the
obvious rule. **Vision is the cheapest mode in all six cells**, below every text-only mode,
because dropping the accessibility tree saves more input tokens than the screenshot adds. Cost
tracks a mode's token count per step, not the presence of an image, so any tier built on
"image versus text" is a modality partition rather than a cost one.

Three backbones span roughly two orders of magnitude of capability and cross a family
boundary: a 235B mixture-of-experts model served through an API (B0), a 4B dense model served
locally (B1), and a 4B model from a different family (B2). Crossing sites with backbones gives
six cells, each running all six modes over an identical task universe: 224 tasks on
classifieds and 203 on reddit.

The reddit figure is 205 as collected and 203 as scored. Two reddit tasks were removed from
the scored set by an amendment written **after all thirty-six conditions had run and been
analysed**: it is post-hoc and outcome-visible, not preregistered, and we label it that way
wherever the denominator appears.
One removed task scores inaction as success; the other is answerable from parametric
knowledge without loading the second site its configuration declares. Each criterion was
applied as a uniform rule over the whole reddit pool and each selects one task. Pooled reddit
success moves from 6.94% to 6.40%.

### 2.2 What a routing label is

The router we set out to build is offline and per-task: from what is known before the
agent acts, predict which mode to use. Two supervision targets are natural.

**Which-mode.** The label is meant to be the identity of the cheapest mode that solved the
task, which is what an oracle would have chosen. Our pipeline produces it by walking a fixed
mode list in assumed-ascending-cost order and taking the first success; §4.3 measures how
often that disagrees with the measured cost, and the disagreement is large enough to be a
finding in its own right. The label is undefined when no mode solved the task, and such tasks
are the majority everywhere. This target has six classes.

**Triage.** The label is binary: did any mode solve this task. It is defined for every
task in the universe. A triage router does not choose among modes; it decides whether to
spend, sending predicted-hopeless tasks to the cheapest mode and the rest to the
strongest.

The distinction matters for the rest of the paper because the two targets fail for
opposite reasons: the first for want of labels, the second in spite of having them.

### 2.3 Features

Features are deliberately restricted to what a deployment would have before committing
to a mode: the task intent text, properties derivable from the task configuration, and
the first observation. Concretely, five numeric features (DOM complexity, text length,
input token estimate, intent token count, a task-difficulty field) and fifteen binary
features (whether the task supplies a reference image, plus fourteen intent-category
regex matches). The which-mode pipeline additionally fits a fold-local TF-IDF vectoriser over
the raw intent text. The triage results in §5 do **not**: that pipeline omits the text
features deliberately, so every triage number we report comes from the twenty numeric and
binary features alone.

One property of this feature set drives §6: **it contains no model identity**. Two
backbones facing the same task produce the same feature vector. This is the right choice
for a deployment that selects a mode for a fixed backbone, and it is exactly what makes
cross-backbone pooling incoherent.

### 2.4 Evaluation protocol

Success is the benchmark's own binary judgement. Cost is the billed cost per episode,
measured within a backbone: the API-served model reports commercial pricing and the
locally-served models report electricity-derived cost [@strubell2019energy], so absolute cost
is never pooled across backbones. Only ratios within a backbone are compared.

All routing results are computed by offline replay over completed episodes. We never
serve a router live, which means the numbers here exclude router inference overhead and
therefore *flatter* the router relative to a real deployment. This is deliberate: we are
establishing an upper bound on what the router could have achieved, and the result is
negative even at that bound.

Cross-validation is five-fold throughout. Where a threshold or a mode choice has to be
selected, §5 reports a fully nested design in which every such choice is re-derived
inside each outer fold; the difference from the naive design is quantified there, because
it is large enough to change how the result should be read.

### 2.5 Baselines that matter

Two baselines appear throughout, and the choice between them determines whether the
router looks useful.

**Best single mode** is the mode with the highest success rate in the cell. Comparing a
router against it asks "did routing preserve accuracy more cheaply".

**Always-cheapest** is the mode with the lowest mean cost, applied to every task. It
requires no model, no features, and no inference. Comparing a router against it asks
"was the router worth building".

Most of the routing literature reports the first comparison. We report both, and the
second is where our negative result lives.
