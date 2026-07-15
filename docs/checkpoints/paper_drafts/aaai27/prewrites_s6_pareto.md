# §6 Pareto prewrites — DRAFT-PENDING-REVIEW

> **DRAFT-PENDING-REVIEW · OFFLINE/NON-GATE.** Candidate equal-length replacements only. Do not splice into `aaai27_main.md` before review and canonical Pass-2/H10 disposition.

## Version A — without the exploratory threshold curve (111 words)

In the complete B0-Classifieds offline replay, the cost–success plane sharpens both conclusions. The six-mode hindsight oracle is the sole frontier point after adding oracle references: 43.30% SR at $0.0623/task, strictly dominating every fixed policy and the learned router. Thus the menu contains substantial joint accuracy–cost headroom, not merely an SR ceiling. The OOF learned router does not realize it: 25.45% at $0.0756 is strictly dominated by always-SoM (27.23% at $0.0724), so the negative router result survives the Pareto criterion. Billed cost is trajectory-level rather than a fixed mode charge; successful trajectories often terminate earlier, so an effective router should raise SR while reducing expected cost. These are offline, non-gate operating points.

## Version B — with the exploratory threshold curve (119 words)

In the complete B0-Classifieds offline replay, the six-mode hindsight oracle is the sole frontier point after adding oracle references: 43.30% SR at $0.0623/task, strictly dominating every fixed policy and the learned router. The locked OOF router remains off-frontier—25.45% at $0.0756 versus always-SoM’s 27.23% at $0.0724—so its negative result survives a cost–success evaluation. A post-hoc cost-aware variant, using six fold-held-out binary success heads and choosing the cheapest mode above threshold, reaches 29.91% at $0.0705 for τ=0.10 and dominates both locked points; because τ was swept on the same replay, this is exploratory rather than H10 evidence. Billed cost is trajectory-level: successful trajectories often stop earlier, so better routing can jointly increase SR and reduce expected cost. All points are offline/non-gate.

### §3 Prior-work baselines (offline) — DRAFT-PENDING-REVIEW

On B0-Classifieds, the RouteLLM-style kNN baseline peaks at 26.79% SR and $0.0707/task (PGR −0.028), strictly dominating the locked LR router (25.45%, $0.0756) but remaining below the post-hoc cost-aware OOF curve. The FrugalGPT-style observed-confidence cascade never exceeds the best single policy: its highest-SR point only matches 27.23% while accumulating $0.2690/task. Thus prior-work-style adaptations do not recover the oracle headroom, although kNN improves on locked LR. These are OFFLINE/NON-GATE, post-hoc exploratory adaptations, not faithful reproductions.
