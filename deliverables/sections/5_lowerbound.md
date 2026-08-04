# What routing actually achieves

\todo{Prose. The point of this section is that the negative is not one method
failing but five, and that they fail for a stated reason.}

**Five policies, none of which beats a fixed single mode on success rate.**

1. **Rule-based, zero-token.** A regex over the task intent plus ``carries no
   reference image'' flags 71/224 classifieds tasks, where the screenshot is worth
   **+22.54pp** [+9.86, +33.80] against +0.65pp on the rest --- a large, free,
   *ex-ante* signal. Turned into a policy it yields 24.55\% on `cls`$\cdot$`B0`,
   **below always-Vision's 25.00\%**, because the screenshot does not hurt on the
   unflagged tasks either.
2. **Confidence-triggered cascade.** No operating point beats always-rich on
   success rate in any of the six comparable cells.
3. **Pooled tier router.** Dominates always-cheapest in 0 of 6 cells.
4. **Learned triage, out-of-fold.** AUROC 0.651--0.717, clearing the
   best-single-feature baseline in 5 of 6 cells. One cell reaches **+0.00pp success
   at $-4$.5\% cost** --- a real win, not an oracle. The other three carrying cells
   are negative.
5. **Benchmark-annotated difficulty.** Adding the benchmark's own
   `visual_difficulty` label moves out-of-fold AUROC by a mean of **+0.008**,
   improving three of six cells --- inside fold-split noise.

**The signal is informative and still not enough.** Against the same escalation
budget spent at random, 64.4\% of 222 (signal $\times$ cell $\times$ fraction)
points are positive with a median of +0.327pp, and the best signal recovers
0--57\% of the gap to a per-task oracle. \todo{Keep the unselected statement, not
the maximum: quoting the per-cell max makes positivity likely before any
information is involved.}
