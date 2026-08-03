## Abstract

<!-- NOT an abstract. This slot holds the evidence inventory so that opening the document
     shows the measurements before any story about them. Generated — rerun
     scripts/analysis/export_ablation_tables.py rather than editing. Replace with a real
     abstract once the claim is chosen. -->

**Evidence inventory — 8 cells = (site x backbone), 6 observation modes.**
`cls`/`red` are VisualWebArena classifieds/reddit \citep{koh2024visualwebarena};
`WA` is WebArena reddit \citep{zhou2024webarena}. `WA-B2` does not exist, so no
statement holds cross-benchmark and cross-family simultaneously.

1. **Success rate per mode**, 6 modes × 8 cells (Table 1). The best mode is SoM in 5 cells, DOM in 2, P-text in 1.
2. **Best arm per deployment class** (Table 2). Sole best: hybrid 4/8, no-image 3/8; 1 tied cell. `vision-only` is never a sole best. On `WA·B0` the no-image class leads the hybrid class by 13.46pp (35.58 vs 22.12).
3. **Class ablation** (Table 3). Unmatched, dropping the no-image class costs the most in every cell — **but it has four arms and the others one each**. Arm-matched, the largest single-arm gain lands on no-image 4×, vision-only 3×. The matched panel is the one that compares like with like.
4. **Behavioural non-separability** (Table 4). Over 26 metrics × 8 cells, the four image-free modes are the extreme in ≥7 cells on **zero** metrics (Vision 9, SoM 5).
5. **Fusion premium** (Table 5). The fused mode does not beat the workload-matched single channel in any of the 8 cells. The comparison is against a measured rerun band of **0.89–2.23pp**, not against zero; `cls_B0`'s +2.23pp equals the band's upper edge exactly (both are 5/224).
6. **A 0-token ex-ante partition** (Table 6). A regex over the task intent flags 71/224 classifieds tasks; on them the screenshot is worth **+22.54pp** [+9.86, +33.80] against **+0.65pp** [-5.88, +7.84] on the other 153.
7. **New representation versus a rerun** (Table 7). Adding one distinct arm and adding one rerun are the same functional at the same arm count. Only 2 of 8 cells carry a measured floor, and neither measures it on the arm being added.
8. **Routing policies on the (success, cost, latency) frontier** (Table 8). A policy built on the partition in (6) survives undominated in 5 of 6 cells — where *undominated* means nothing beats it on all three axes, not that it is preferable. On `cls·B0` all three rule policies sit between always-SoM and always-Vision, and `always-P-prompt` at 19.64% is equally undominated.

### What is known to be wrong with it

- **6 leaked successes on VWA reddit** (`require_reset` is a no-op there, so subscriptions accumulate). Zeroing them flips `red_B2`'s SoM−DOM interval across zero — the only cell that showed fusion significantly beaten. The WA cells are **unaudited** for the same defect.
- **1.05–2.13% of reddit steps run on the public internet** (Postmill is a link aggregator); classifieds is 0.00–0.16%. Those steps are *faster*, not slower. Separately, reddit's container is **1.69×** slower than classifieds' before any agent behaviour enters, so no between-site latency number is quotable bare.
- **The `vision` column of any diag per-rule table is not co-tabulable** with `dom`/`som`: `P2`/`P4` read `element_bbox`, which vision's clicks do not carry, so those cells are structural zeros rather than measurements.
- **Per-rule frequencies are symptom distributions, not cause distributions.** `P36` (51%) and `P31` (50%) are risk markers; causal verification exists for `P49` and not for them.
- **Six conclusions were found hardcoded in their producers** on 2026-08-03, one wrong on the fact and not only on the denominator. The sweep that found them covered one textual shape (`n/6`); mode names, ratios and directions were not swept.

### What cannot be answered with this data

- Whether the reversal in (2) turns on modality, task set, or benchmark — two workloads cannot identify a moderator.
- Anything requiring a third workload: `shopping` has zero landed directories.
- What a *real* cascade does: every escalation number is an offline splice.
- Whether a learned router could do better than the rule in (8): the which-mode label exists only where some mode succeeded (15–97 rows per cell, 260 total).

**Known defects in the above.** Six scored successes on VWA reddit were credited by
accumulated site state, and zeroing them flips the one cell that showed fusion
significantly beaten. 1.05-2.13% of reddit steps run on the public internet against
0.00-0.16% on classifieds, and reddit's container is 1.69x slower before any agent
behaviour enters. Per-rule failure frequencies are symptom distributions, not cause
distributions. Six conclusions were found hardcoded in their producers on 2026-08-03.

**Out of reach with this data.** Whether the reversal turns on modality, task set or
benchmark; anything needing a third workload; what a real cascade does; whether a learned
router beats the rule.

## 1. Introduction

<!-- EMPTY BY DESIGN. Write this LAST. Three frames died between 08-01 and 08-03 because
     they were written before being checked against coverage; a fourth was killed by a
     selection bias in the single number it rested on. Choose the claim against the
     inventory above, with the advisor, then write this. -->
