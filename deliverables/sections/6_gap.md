# The gap, and why it is not a tuning problem

\todo{Prose. This is the contribution. Write it as a problem statement handed to
the community, not as a limitation of our methods.}

**The gap.** A perfect per-task chooser reaches the ceiling of Section 4; every
policy we could construct sits at the fixed-mode baseline of Section 5. The
distance between them is not closed by a better classifier, and the reason is
measurable.

**Supervision is scarce where it is needed.** The which-mode label exists only on
tasks some mode solves: 3--68 rows per cell.

**Supervision is unstable exactly where it is needed.** Rerunning one condition
flips 49 of 224 task outcomes. The flips are not spread evenly --- on the rows a
which-mode router would train on, the flip rate is 48--52\% against 2.94\% on the
complement, an enrichment of 16.5--17.6$\times$. \todo{Report the range 3.9--17.4$\times$
and say why: ``contested'' defined over all six arms is correct for the claim, but
the flips are produced by rerunning two of those six, so the same arms decide both
membership and outcome; rebuilding the proxy from the other four gives 3.95$\times$.}

**The target is binary.** The evaluator emits two distinct values over 7,686
scored episodes, so there is nothing graded to regress on.

**What would close it.** \todo{This is the paper's ask. Candidates: a graded
target; supervision that does not require running every arm; a ceiling defined on
cost rather than success, where the label is defined on every task.}

**Table.** Per-task label instability on the contested rows: Table~\ref{tab:instability}.
