---
type: issue
category: action
status: blocked
priority: medium
action: launch on A100 (post-SSH verify) — Stage 2B curated 10-20 task + Llama-4 cross-arch + position-resolved patching
created: 2026-05-06
---

# Mechanistic Stage 2 scale-up — paper §5 quantitative evidence

5/6 mechanistic pilot (笔记 §111) found L11 mirage causal layer (case study) + forward-vs-reverse asymmetry (paper-grade mechanism). Three followup directions queued for A100 dedicated execution.

## Three followups (priority order)

### (a) Stage 2B curated scale-up — 10-20 tasks ⭐⭐⭐

**Why**: 笔记 §111.4 Stage 2B 现 N=3 task pilot, only task 0 (cls blue kayak) showed clean L11 signal. Tasks 1, 2 were null because source/target diverged for non-mirage reasons. Paper §5 mechanism quotable claim needs N=10-20 curated mirage cases for cross-task aggregate (mean ± std AUROC curve, robust L11 peak).

**Curation criterion**: pick cls task pairs where source (SoM with image) outputs **image-grounded ground truth** vs target (P-SoM no image) outputs **image-hallucinated mirage** at step_002+. Manual review of archived B1 phantom_som artifacts to identify candidates.

**Scope**: 10-20 curated cls task pairs × max_new_tokens=15 × forward direction × all 36 layers. ETA ~3-4h on A100 dedicated.

**Deliverable**: `results/mechanistic/stage2b_curated_b1_cls_n15/` + paper-grade plot (token_overlap_to_source mean ± std with L11 peak), updated paper §5 quotable claim with N≥10 robust signal.

### (b) Llama-4 cross-arch validation ⭐⭐ (paper claim power upgrade)

**Why**: advisor 5/5 sync explicit push "如果是 cross-model 的话, 那这个价值就很大了, 这就是 golden feature" (see 笔记 §110.2). 5/6 A100 + Llama-4 multimodal local 现 affordable (was trade-off pre-A100).

**Setup**: same Stage 2B continuation patching protocol on Llama-4 multimodal (vision-capable). Compare L*-region asymmetry pattern to Qwen3-VL-4B finding. If pattern hold cross-arch → paper §1 hook upgrade to "**golden feature** of mid-layer image-content aggregation".

**Scope**: Llama-4 multimodal load + 10-20 curated tasks × forward + reverse × 全 layer. ETA ~6-8h on A100 dedicated (Llama-4 likely larger than 4B).

**Deliverable**: cross-arch comparison fig + universal claim quote.

### (c) Position-resolved patching (verify distributed encoding)

**Why**: §111.5b interpretation says "image content distributed across image embedding tokens (positions 0-256+) and aggregated at last token through L11". Direct evidence: patch image-token position k (k=0..255) at L11 instead of last token, see which positions carry causal info.

**Setup**: per-task per-position-k patching grid. Forward direction (SoM → P-SoM with [SOM_MARKS] only, but inject SoM's position-k hidden into corresponding position in P-SoM run). Note: P-SoM doesn't have image positions, so this requires either (i) padding P-SoM input with placeholder image tokens OR (ii) patching SoM run's image positions with zero-noise as contrastive.

**Scope**: 1-2 task × 256 positions × 1 layer (L11) = ~512 forwards. ETA ~30-60 min on A100. Mostly engineering complexity not compute.

**Deliverable**: position-causality heatmap, validates §111.5b mechanistic story directly.

## Compute path

⭐ **UCL Condense A100 dedicated** (allocated 5/6, pending Steve SSH info verify, 笔记 §112). 80GB VRAM 余量 8× for B1 4B / 2-3× for Llama-4 ~10B class. Cell-parallel feasible.

## Blocks

- A100 SSH verify (Steve admin info pending)

## Unblocks (after delivery)

- Paper §5 mechanism prose (codex #13) — currently stuck at N=3 case-study weak; scale-up gives paper-grade aggregate
- Paper §1 hook upgrade — cross-arch validation enables "golden feature" claim per advisor 5/5 push
- Stage 3 SAE feature steering scoping decision (Stage 3 deferred unless mechanistic validates strong, then 1-2 week SAE training investment justified)

## Refs

- `docs/checkpoints/实验笔记.md §111` (Stage 1+2 pilot results)
- `docs/checkpoints/实验笔记.md §111.5b` (forward-reverse asymmetry, paper-grade mechanism)
- `docs/checkpoints/实验笔记.md §112` (A100 unblock + Llama-4 affordable)
- `docs/checkpoints/advisor_sync_5_5_followup.md` Q3 + Part 3 #3 (mechanistic scope advisor lock pending — likely auto-resolved by A100)
- `p79/mechanistic/activation_patching.py` (`patching_grid_continuation` ready, accepts task batches)
- `scripts/mechanistic/run_stage2b_continuation_pilot.py` (driver — `--reverse` flag supports reverse direction; new `--n-tasks` scale-up natively supported)
