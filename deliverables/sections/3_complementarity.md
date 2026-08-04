# Representations fail in different ways

\todo{Prose. The numbers below are settled and quotable; the argument connecting
them is the part to write.}

**The claim.** The image-bearing and image-free channels do not fail the same way,
and the difference is one-sided: when the image channel wins, the text channel's
failures have names; when the text channel wins, the image channel's failures do
not.

**What licenses grouping the four image-free modes.** Over 26 behavioural metrics
$\times$ 8 cells, the four image-free modes are the extreme in $\geq 7$ cells on
**zero** metrics; `Vision` reaches that bar on 9 and `SoM` on 5. So the difference
this section is about is produced by *one* cut --- whether the screenshot is
present --- and not by how the text is serialised or how the prompt is worded.
That negative is load-bearing and is reported at a threshold of $\geq 7$/8 = 87.5\%,
the same proportion the six-cell version meant by $\geq 5$/6.

**The asymmetry.** On tasks only one channel solved, the losing channel's failures
are enriched relative to how it fails everywhere. Where the image channel wins,
four rules clear 1.6$\times$ --- giving up when a target is not found
(2.31$\times$), click-back oscillation (2.25$\times$), a visual-content task the
accessibility tree cannot express (2.24$\times$), and page-embedded visual content
with no screenshot (1.65$\times$). Where the text channel wins, one rule clears
1.2$\times$; it rests on 8 hits, exactly the reporting floor, and all of them fall
in the two WebArena cells. Every other rule on that side sits at or below the
everywhere-baseline.

**The vocabulary objection is closed.** The ruleset was developed on
VisualWebArena, so an absent signature could be a property of the vocabulary. Six
probes computed from raw step fields with no rule hits at all find the same thing:
largest enrichment 1.15$\times$, five of six below 1. On the tasks the text channel
uniquely solves, the image channel fails *more blandly* than it fails elsewhere ---
it did not arrive, rather than breaking somewhere nameable.

\todo{Decide whether the dispatch-path limitation goes here or in threats. It is
the reviewer's first alternative explanation and it is partly correct: three
delivery paths have 88.9\% / 38.6\% / 16.1\% action success, and Vision is on the
coordinate path by construction, so its arm measures our grounding code as well as
the representation.}

**Tables.** Non-separability of the image-free modes: Table~\ref{tab:nonsep}. The paired failure cut, both sides: Table~\ref{tab:failmode}.
