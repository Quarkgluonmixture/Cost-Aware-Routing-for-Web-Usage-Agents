# Negative Results Registry

> Open-science discipline (Nosek et al. 2015; Lipton & Steinhardt 2018):
> failed pilots, retracted framings, and abandoned hypotheses are documented
> here so the final paper narrative is **constrained by what we tried**, not
> just by what worked. Reviewers who diff archived branches against published
> claims should find every framing pivot recorded.
>
> This addresses audit constraints **H2** (negative-result registry) and
> **D8/H5** ("controlled characterization" framing — see paper_planning §21-§22).

## Entry format

| # | Date | Claim / framing (retracted) | Replaced by | Why retracted | Paper action |
|---|------|------|------|------|------|

## Entries

| # | Date | Claim / framing (retracted) | Replaced by | Why retracted | Paper action |
|---|------|------|------|------|------|
| **1** | 2026-04-28 | **Phantom-DOM 18 modes scope** (full factorial expansion across image / SoM / prompt axes) | 5-mode scope (DOM / SoM / Vision / phantom_som / phantom_dom) | 18 modes exceeded paper-grade focus; 13 ablation modes added marginal information at high compute cost | §3 paper-grade scope explicit; pre-rerun audit §1.1 enforces |
| **2** | 2026-05-01 | **Phantom-SoM is hidden 4th routing arm** | **Phantom routing space (3 arms: P-text / P-prompt / P-SoM) sharing 4-fold drop-in property** | B0 reddit 6-mode oracle vs 3-mode +7.14pp [3.81, 10.48] sig + 3 arms drop-one all sig → "1 arm" framing literally inaccurate; "1 routing dimension" stronger venue claim | Paper hook §1 reframe; provisional pending data confirm; advisor sync Q3 |
| **3** | 2026-05-01 | **8-corner 2x2x2 cube factorial design** as paper §2 axis | **M1/M2 mechanism activation 2x2** (LLM internal state level, not prompt structure level) | Prompt-text coupling ≠ mechanism activation coupling; 8-corner conflated levels | Memory `project_paper_hook.md` retract list; paper §2 rewrite |
| **4** | 2026-05-01 | **6-corner asymmetric grid** (a/b × c/¬c × 1/2) | M1/M2 2x2 (4-corner) | Same level-confusion as #3 | Same retract list |
| **5** | 2026-05-01 | **(a)(c) prompt decomposition** as paper axis | Evidence/Explanation 双层 + Zoom 1-4 | Decomposition was prompt-structure thinking, not mechanism thinking | Same retract list |
| **6** | 2026-05-01 | **"Three-layer mechanism argument" (Layer 1/2/3)** naming | Evidence/Explanation 双层 + Zoom 1-4 hierarchy | Naming overlap with neural-network "layer" caused reader confusion | Same retract list |
| **7** | 2026-05-01 | **"Approach 1 vs Approach 2" dichotomy** | Approach 2 = Zoom 1 (architectural completeness); "Approach 1" was not a single thing | Dichotomy was strawman | Same retract list |
| **8** | 2026-05-01 | **"First inference-time substitution / first deployment of text-only or marked observations"** novelty claim | "Controlled behavioral characterization of phantom configurations" | Industry artifacts (yang2023som SoM-Mark, zheng2024seeact, yang2025magma) precede our deployment claim; honesty matters for venue review | Paper §1 + related-work rewrite (audit D8/H5 — codex-delegated) |
| **9** | 2026-05-01 | **SteerMoE-style expert routing self-probe** for B0 | Zoom 4 future work direction (paper §8), no self-probe | B0 is proxy API → model internals invisible; local 235B deploy budget exceeds RunPod $200 allocation | Paper §8 future work |
| **10** | 2026-05-06 | **§111 task-0 single-task "L11 flips 93% match"** as paper §5 representative finding | 24-task aggregate Stage 2B L17 Holm-significant (p_Holm=0.011 \*\*) | Task-0 was distribution outlier (some tasks fully flip, some don't disrupt); single-task evidence cherry-pick | 笔记 §117.4 + paper §5 cite aggregate not task-0 |
| **11** | 2026-05-06 | **§111.5b "reverse direction null at all layers" as asymmetric encoding evidence** | 15-task aggregate Stage 2C reverse shows L11+L17 Holm-significant (p_Holm=0.044 / 0.033 \*); reverse magnitude **identical** on strong-tier (Δ=-0.193) and reverse-tier (Δ=-0.193, Welch p=1.000) | §111.5b was N=1 (task 0 reverse) theoretical extrapolation, not measured aggregate; 15-task scaled-up overturned the asymmetry hypothesis | 笔记 §117.2 reframe + paper §5 mechanism = "bidirectional mid-layer L11-L17 disruption" pending cross-site confirmation cells F/G |
| **12** | 2026-05-09 | **"Cell E random-injection should produce mid-layer L17 dip if mechanism is generic"** (null hypothesis for specificity check) | Cell E shows random Gaussian destroys output uniformly at all layers (overlap 1.00→0.03), specificity ratio random-LD/real-LD = 5-19× across layers | Mechanism IS content-specific — random control didn't produce mid-layer pattern | 笔记 §117 update pending; paper §5 control PASSED audit G6 ✓ |

## Pivots that did NOT retract (data confirmed framing)

These are kept here so reviewers can verify the registry is symmetric (we
report data-confirmed framings as well as data-broken ones):

| # | Date | Original framing | Confirmed by | Paper action |
|---|---|---|---|---|
| C1 | 2026-04-26 | Phantom-SoM 4-fold drop-in property (cost ≈ DOM / latency ~50% / signal AUROC ≥ baseline / drop-one 1.7-3.8pp) | B0 reddit Phase A archived data + drop-one sig | Paper hook §1 (provisional pending 16-cell rerun) |
| C2 | 2026-05-09 | Mid-layer L11-L17 mechanism Holm-significant disruption | 4 cells: A (Holm L17 p=0.011), B (Holm L11+L17), D (Holm L11+L17 p=0.006/0.008) — 3/4 cells Holm-confirmed | Paper §5 mechanism evidence |

## Paper § action items derived from this registry

1. **Paper §1 + Related Work**: rewrite to "controlled characterization" framing, NOT "first inference-time substitution" (entry #8). Acknowledge yang2023som / zheng2024seeact / yang2025magma industry artifacts as context. (Codex-delegated, audit D8/H5.)
2. **Paper §2 Background**: confirm M1/M2 2x2 framework + Evidence/Explanation 双层 + Zoom 1-4. Retract earlier 8-corner cube / 6-corner / (a)(c) decomposition / Three-Layer / Approach 1-2 framings (entries #3-#7). (Already done in `paper_planning.md §2` reframe; final prose confirms.)
3. **Paper §5 Mechanism**: cite 24-task aggregate L17, NOT §111 task-0 single-case (entry #10). Show forward+reverse symmetry from 4-cell 2x2 (entry #11). Cite cell E random-injection control specificity ratio (entry #12).
4. **Paper §8 Discussion**: SteerMoE-style probe is future work, not self-conducted (entry #9).
5. **Paper hook (§1)**: phantom routing space (3 arms) — retracted from "4th arm" (entry #2). State "provisional pending 16-cell rerun" until R1-R5 framing rules (preregistration.md §2) trigger.

## Future entries (placeholder)

When framings shift in upcoming work, append below. Common triggers:

- 16-cell rerun outcome inconsistent with archived data → entry for "phase A pre-fix data was over-optimistic" (audit F1)
- Cells F/G reddit cross-site shows null mid-layer disruption → entry for "phantom mechanism is cls-specific, not universal" + paper §5 scope reduction
- Advisor sync 5/X feedback retracts a hypothesis → entry for "H_X retracted per advisor"
- Reviewer pre-print feedback identifies a new failed assumption

## Caveat

This registry is honest but incomplete — early ad-hoc explorations (pre-2026-04-28)
are not all logged. The discipline started with Phase A 4-cluster bug fix
wave (笔记 §107) when paper-grade re-run was first scoped. Entries before
that date are reconstructed from chronicle notes / paper_planning decision
log / memory `project_paper_hook.md` retract list, not from contemporaneous
record. Future paper revisions log all pivots from this point forward.
