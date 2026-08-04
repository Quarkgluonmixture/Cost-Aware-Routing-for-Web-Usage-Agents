# Setup

**Benchmarks and cells.** We evaluate on VisualWebArena \citep{koh2024visualwebarena}
and WebArena \citep{zhou2024webarena}. A *cell* is one (site, backbone) pair. Six
cells come from VisualWebArena classifieds and reddit crossed with three backbones
--- Qwen3-VL-235B-A22B (`B0`), Qwen3-VL-4B (`B1`) and Gemma-3-4B (`B2`) --- and two
from WebArena reddit crossed with `B0` and `B1`. **`WA` $\times$ `B2` does not
exist**: the cross-family backbone never ran WebArena, so no statement in this paper
holds cross-benchmark and cross-family at the same time.

The two benchmarks also share one application on the reddit axis: WebArena reddit
*is* the same Postmill deployment as VisualWebArena reddit, with a different task
set (104 against 203 tasks, success rates 2--3$\times$ apart). Results that hold on
both are therefore robust across task sets, not across applications.

**Observation modes.** Each cell runs six modes, which differ on two axes: whether
the screenshot is provided, and how the page text is serialised and prompted for.
`DOM` gives the accessibility tree; `Vision` gives the screenshot alone; `SoM`
gives an annotated screenshot together with mark-indexed text
\citep{yang2023som}. The remaining three vary the text payload and the prompt
style without the image --- `DOM+somtext`, `DOM+somprompt`, and `SoM-image`, the
last being `SoM` with the screenshot withheld. They group into the three shapes a
deployment actually ships: **no-image**, **vision-only** and **hybrid**.

**Scored universes.** classifieds 224 tasks, reddit 203, WebArena reddit 104. The
reddit figure excludes two tasks whose evaluators cannot separate a completed task
from an untouched one; the WebArena figure is the six-mode intersection.

**Estimands, stated because products disagree without them.** Cost is
`total_billed_cost_usd`. Latency is canonical --- retry, busy-wait and recovered
screenshot time removed --- and we report throughout that the model call is only
22--67\% of it. Success is binary: the evaluator emits exactly two distinct values
over all 7,686 scored episodes, so there is no graded target to regress on.

**The rerun band.** Running the same condition twice on `cls`$\cdot$`B0` moves the
success rate by 0.89--2.23pp, and the union of tasks solved moves by 2.0--7.6pp.
Every effect in this paper is reported against that band rather than against zero.
Three arms carry a measured replicate; the band is two draws, not a bound, and the
exchangeability null behind those draws puts one standard deviation at
2.32--2.53pp.
