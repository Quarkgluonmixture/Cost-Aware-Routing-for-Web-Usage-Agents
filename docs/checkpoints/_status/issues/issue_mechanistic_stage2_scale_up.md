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

### (a) Stage 2B curated scale-up — 24 tasks (dataset ready) ⭐⭐⭐

**Why**: 笔记 §111.4 Stage 2B 现 N=3 pilot, 1/3 task clean. Paper §5 needs cross-task aggregate.

**Dataset (5/6 evening curated 笔记 §113, commit `4425fa6` script)**:
- ✅ 209/234 cls task scored via `scripts/mechanistic/curate_mirage_tasks.py`
- ✅ **24 strong mirage candidates** (composite ≥ 1.0 ∧ overlap < 0.5) ready for forward-direction patching
- ✅ **Top 7 cluster (composite +4.00-+4.20)** shares identical paper-grade mirage signature:
  - Source (SoM with image): "do not show any X" — image-grounded absence detection
  - Target (P-SoM no image): "show items/listings related to X" — mirage hallucination
  - Tasks: 0 (blue kayak) / 81 (hurricane book) / 112 (basketball) / 113 (football) / 127 (MCAT) / 201 (snare drum) / 224 (wall rack)
- Reference: `results/mechanistic/curate_mirage_b1_classifieds/candidates.md` (gitignore whitelisted)

**Scope**: 24 task × forward direction × 36 layer × max_new_tokens=15 = 864 patched-generate calls + 48 baseline gen + 24 source cache. ETA ~2-3h on A100 dedicated.

**Deliverable**: `results/mechanistic/stage2b_curated_b1_cls/` + per-task per-layer 4-metric grid + cross-24-task aggregate mean ± std curve. Replace §111 N=3 placeholder.

**Quotable claim post-completion**: "Patching SoM hidden state at L11 into P-SoM run recovers source's continuation token-by-token across **N=24 curated mirage cases** (B1 Qwen3-VL-4B, mean overlap_to_source X.XX ± Y.YY, p < 0.001 vs envelope baseline 0.5)."

### (a-bis) Stage 2C reverse curated — 11 tasks (dataset ready)

**Why**: §111.5b reverse direction null at all layers (single-task asymmetry); paper §5 needs cross-task aggregate to confirm pattern.

**Dataset (curated 5/6, 笔记 §113)**: 11 reverse-direction candidates (composite ≤ −1.5):
- task 10, 123, 130, 151, 155, 156, 157, 160, 188, 191, 200
- Pattern: source (SoM with image) describes specific page element / target (P-SoM no image) abstract task statement

**Scope**: 11 task × reverse direction × 36 layer × max_new_tokens=15 = ~400 patched-gen. ETA ~1.5h on A100.

**Deliverable**: confirm reverse direction null effect across N=11 (vs §111.5b N=3 / 1 task case study). Strengthen paper §5 asymmetry-as-mechanism evidence.

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

⭐ **UCL Condenser A100 dedicated** — ✅ operational 2026-05-14 (VM `a100-jiaming-test` @ `10.134.51.2`, PyTorch smoke test passed). GPU is A100-PCIE-**40GB** (NOT 80GB): B1 Qwen3-VL-4B ~10GB fits comfortably; Llama-3.2-11B-Vision fits tight; Llama-4 Scout ~17B borderline (4-bit). See memory `reference_compute_resources.md` + `docs/reference/COMPUTE_INFRASTRUCTURE.md`.

## Blocks

- ~~A100 SSH verify~~ ✅ resolved 2026-05-14 — A100 operational. (Note: paper §5 mechanism scope itself 暂搁 per advisor discussion 2026-05-14 — this issue's followups frozen, not compute-blocked.)

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
