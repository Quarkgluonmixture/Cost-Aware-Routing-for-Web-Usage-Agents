# Phase A baseline pre-run methodology audit

## Why this audit exists

User caught 4 high-severity bugs in Stage 4 mechanism pipeline this week (SOM_MARKS regex dropped 71/72 marks producing buggy NPZ; tier filter missing; model revision unpinned; cosine-gap script ran on wrong NPZ). Caught 1 watcher dispatch bug 30 min ago (prefix shortest-match instead of longest → 2 cellhprompt result dirs corrupted on local). Phantom_prompt classifieds prefix bug caught last week (commit 3d41953).

Each bug was "everything looked fine until we looked carefully." Each one independently invalidated a section of paper claims.

**Phase A is the foundation of paper §1 + §4 hero claims**. If Phase A has the same kind of hidden bug, the entire paper retracts. User now paranoid (correctly).

## What Phase A is

Phase A = "表征筛选: per-site per-baseline per-mode JSONL log → episode summary → condition aggregate". Outputs paper §1 hero figures (SR per mode, drop-one CI, FP rate per mode, cost+latency per mode) and §4 empirical SR tables.

Five sites × two baselines × seven modes × ~210-466 tasks per site. 7 modes are dom / som / vision / phantom_dom / phantom_som / phantom_text / phantom_prompt.

Modes that paper §1 hero claims depend on:
- P-SoM (phantom_som) drop-one CI strict-positive on reddit + classifieds = main hero
- P-text 12.38 SR (canonicalized 2026-05-12, was 11.90 before)
- DOM vs P-text vs P-SoM vs SoM 4-fold property (cost ≈ DOM / latency ~50% / signal AUROC / drop-one)

## Your job

Read the Phase A pipeline + analysis scripts + a sample of raw data + the FP detection rules. Find bugs that would change paper-grade numbers by ≥0.5pp. Set your own attack vectors based on what you see in the code; do not restrict to typical Web agent pitfalls.

## Scope (read these)

Pipeline:
- `p79/experiment/runner/main.py` — main orchestrator condition → seed → task → step
- `p79/experiment/runner/helpers.py` — cycle detection, diagnostic control, ntfy
- `p79/experiment/logger_v2.py` — JSONL writes (with fsync)
- `p79/experiment/io_utils.py` — JSONL reads + restart dedup
- `p79/experiment/analysis.py` — adjusted_success canonical, Pareto, analyze_run
- `p79/experiment/metrics.py` — cost / latency / energy aggregation
- `p79/experiment/types.py` — EpisodeSummaryV2 / StepRecordV2 dataclasses
- `p79/experiment/conditions.py` — condition generation per phase

Mode-specific paths:
- `p79/agents/qwen3vl_agent.py` — B1 local model agent
- `p79/agents/proxy_api_agent.py` — B0 proxy API agent
- `p79/backends/local_qwen.py` + `p79/backends/api_proxy.py` — backend implementations
- `p79/experiment/som.py` — SoM annotation + mark extraction (this was the source of Stage 4 Bug 2)
- `p79/envs/vwa_wrapper.py` — viewport filtering (current_viewport_only=True)

Analysis layer:
- `scripts/analysis/aggregate_phase1_v2.py` (or whatever the cross-condition aggregator is named — find it)
- `scripts/analysis/analyze_experiment.py`
- `scripts/analysis/validate_run.py`
- `scripts/analysis/figures/fig0a_sr_per_mode_heatmap.py` — paper §1 main hook figure
- `scripts/analysis/figures/fig0b_fp_rate_per_mode.py`
- `scripts/analysis/figures/fig0b_extra_confidence_calibration.py`
- `scripts/analysis/figures/fig3b_image_token_gap.py`
- `scripts/analysis/figures/fig_meta_forest.py` — drop-one forest plot
- `scripts/analysis/bootstrap_ci_dropone.py` (or similar — W1 task #47)

FP / hygiene logic (current rules in 实验笔记 §95 for eval_fp/visual_fp and §78a for na_fp — read those sections):
- Wherever the FP detection lives in code (grep for `na_fp`, `eval_fp`, `visual_fp`)
- `docs/analysis/cross_sites/vwa_manual_non_visual_task_ids.py` — manual visual/non-visual audit
- VWA benchmark task counts: classifieds 234 / reddit 210 / shopping 466 (do these match the JSONL episode counts?)

Sample data:
- Pick one recent Phase 1 run dir (`ls -lt results/visualwebarena/phase1/ | head -3`) and trace 1 condition end-to-end: raw JSONL → episode summary → condition_summary_v2.json → cross-condition aggregator output → fig0a heatmap data
- Spot-check the SR number for one cell agrees across all 4 representations

## Output format

Open with one-sentence current-trust verdict ("Phase A is paper-grade trustworthy" / "Phase A has 1+ paper-grade bug" / "Phase A has structural concerns but no proven bug").

Then sections:

### Confirmed bugs (would change paper numbers ≥0.5pp)
For each: file:line, what the code does wrong, what number it changes by, defuse effort. Include grep / file evidence.

### Probable bugs (suspicious but couldn't fully verify)
Same format but mark "needs further check."

### Methodology concerns (not bugs, but reviewer ammo)
Things like "stratification missing", "unit of analysis unclear", "FP detection rule depends on N/A label that might be wrong for K tasks."

### Cross-representation inconsistencies
If the same SR number can be computed from JSONL or from condition_summary or from cross-condition aggregator and they disagree, that's the highest-priority finding.

### What you read and what you didn't
Brief enumeration so user can decide if a follow-up audit is needed.

## What this is NOT

- Not a code review for style or efficiency
- Not a paper prose audit (that's /codex-stress)
- Not a request to propose fixes — just identify the suspect locations
- Not paranoid for paranoid's sake — calibrate severity by paper-grade impact, not aesthetic ugliness

## Stop conditions

If you exceed 25 min reading without finding anything ≥0.5pp suspect, write the trust verdict and stop. Negative result (no bug found) is also a valid paper-grade contribution to the user's confidence calibration.

If the codebase clearly has a layer of indirection you haven't time to trace, say so and don't fabricate a verdict.
