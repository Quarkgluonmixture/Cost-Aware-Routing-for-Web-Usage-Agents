# Full pre-run methodology + design + code + data audit

## Why this audit exists

User has spent months building a paper-1 (毕设论文) claim stack on Phase 1 baseline data (Phase A in project taxonomy). Four high-severity bugs surfaced in Stage 4 mechanism pipeline this week, each "everything looked fine until we looked carefully", each independently invalidating a section of paper claims. Watcher dispatch bug 30 min ago corrupted 2 local data dirs. Phantom_prompt prefix bug last week silently affected classifieds runs.

The bug pattern: hidden correctness defect in upstream pipeline → all downstream analyses computed on subtly-wrong data → paper claims built on top → retracts cascade. User now justifiably paranoid that Phase 1 baseline (which feeds §1 hero + §4 hero tables + drop-one CI + per-mode forest plot + fp_rate + cost+latency + bootstrap CI) has the same kind of hidden defect.

**Your job: full-stack pre-run audit. Methodology → experiment design → code → data → existing-bug-fix records.** Find anything that would change paper-grade numbers by ≥0.5pp OR invalidate a paper-grade qualitative claim.

## Scope

This is paper-1 backbone. Five layers, audit each:

### Layer 1 — Methodology (pre-registered claims)

Read everything in `docs/checkpoints/pre_run/`:
- `preregistration.md` — pre-registered hypotheses and analysis plans
- `topvenue_constraints.md` — top-venue acceptance constraints / methodology gates
- `pre_rerun_audit.md` — existing audit findings (CRITICAL: previous audit notes, bug fix records)
- `dataset_card.md`, `model_card.md`, `locked_versions.md` — formal dataset / model / version cards
- `evaluator_change_protocol.md`, `reeval_audit_protocol.md` — evaluation rule version control
- `negative_results_registry.md` — what was tried and didn't work (and is it actually negative?)
- `osf_lock_manifest.md` — what's pre-registered with OSF (if any)
- `ethics_license_coi_statements.md`, `release_redaction_checklist.md` — paper-meta hygiene

Question to answer: **Does what was pre-registered match what was actually run?** If preregistration.md says "Phase 1 reports A/B/C", does the current code path emit A/B/C with the agreed definitions? Any drift = paper claim risk.

### Layer 2 — Experiment design

Read these to understand the design intent:
- `CLAUDE.md` project root — section "三阶段实验设计" + "关键变量" + "实验启动 hard rules"
- `docs/checkpoints/paper_planning.md` — strategy notebook (theory framework, findings, decision log)
- `scripts/queues/queue_baseline.sh` + `queue_phantom_*.sh` — the only authorized launch paths
- `scripts/preflight_v2.sh` — preflight check what
- `docs/reference/launch_checklist.md` — 16-cell paper-grade rerun protocol
- `docs/reference/condition_map.md` — condition_id → benchmark mapping

Questions: Same-site B0/B1 serialization protocol. RESET_BEFORE protocol. How is "reset" actually enforced — is there proof the docker container state was reset before each condition? Could one condition's state (cart, posted listing, subscribed forum) leak into the next condition's tasks? Watchdog auto-clean protocol — does it actually purge contaminated episodes, or just mark them?

### Layer 3 — Code (pipeline)

The Phase 1 runner pipeline:
- `p79/experiment/runner/main.py` — orchestrator
- `p79/experiment/runner/helpers.py` — cycle detection, ntfy
- `p79/experiment/logger_v2.py` — JSONL writes (fsync)
- `p79/experiment/io_utils.py` — JSONL reads + dedup
- `p79/experiment/analysis.py` — adjusted_success canonical, Pareto
- `p79/experiment/metrics.py` — cost/latency/energy aggregation
- `p79/experiment/types.py` — V2 dataclasses (EpisodeSummaryV2, StepRecordV2)
- `p79/experiment/conditions.py` — condition generation
- `p79/experiment/som.py` — SoM annotation (Stage 4 Bug 2 source)
- `p79/envs/vwa_wrapper.py` — viewport filtering
- `p79/agents/qwen3vl_agent.py` (B1) + `p79/agents/proxy_api_agent.py` (B0)
- `p79/backends/local_qwen.py` + `p79/backends/api_proxy.py`

The Phase 1 analysis layer:
- `scripts/analysis/aggregate_phase1_v2.py` (or find the canonical aggregator)
- `scripts/analysis/analyze_experiment.py`
- `scripts/analysis/validate_run.py`
- `scripts/analysis/figures/fig0a_sr_per_mode_heatmap.py` — §1 main hero
- `scripts/analysis/figures/fig0b_fp_rate_per_mode.py`
- `scripts/analysis/figures/fig_meta_forest.py` — drop-one forest
- `scripts/analysis/bootstrap_ci_dropone.py` (or whatever exists)

FP rules (current spec lives in 实验笔记 §95 for eval_fp/visual_fp + §78a for na_fp):
- Grep `na_fp`, `eval_fp`, `visual_fp` in code
- `docs/analysis/cross_sites/vwa_manual_non_visual_task_ids.py` — manual audit

Questions: Are JSONL log → episode summary → condition aggregate → cross-condition aggregator → figure data the same number through all 4 representations? If they disagree, which is right? Restart deduplication: a task that crashed mid-run and was restarted — does it get counted once, twice, or partial-twice? FP rule scope: na_fp applies to N/A label tasks — what's the trust on the N/A label itself? If wrong, FP gets misattributed.

### Layer 4 — Baseline data integrity

Pick one recent paper-grade run dir from `results/visualwebarena/phase1/` (e.g., `B1_3mode_classifieds_20260413` or `B0_phantom_som_classifieds_20260426`).

Trace 1 condition end-to-end:
1. Raw episode JSONL count vs `condition_summary_v2.json` episode count — do they match?
2. SR formula: count "success=True" episodes in JSONL → does this match `condition_summary_v2.json` success_rate × episode count?
3. Cost: cumulative tokens × per-token price vs `condition_summary_v2.json` mean_cost — same formula across all conditions?
4. Latency: from JSONL step deltas vs condition_summary aggregate — agree to ±5%?
5. The cross-condition aggregator output (find it: `docs/analysis/cross_sites/*.json` or similar) — does the SR for this cell match condition_summary?
6. fig0a heatmap data source — does its SR for this cell match condition_summary?

If any layer disagrees, **that is a confirmed bug** and likely affects all paper claims.

### Layer 5 — Existing bug fix records

Read these to understand what's already been audited and fixed:
- `docs/reference/master_bug_catalog.md` — bug taxonomy + fix history
- `docs/reference/EVIDENCE_LAYER_AUDIT.md` — outcome-supporting figure registry
- `docs/checkpoints/pre_run/pre_rerun_audit.md` — previous audit findings
- `docs/checkpoints/实验笔记.md` — append-only chronicle, grep for `[bug]` tag
- Git log: `git log --oneline | grep -iE "fix|bug" | head -30`

For each existing fix in these records, verify the fix is still in place in current code (not regressed via subsequent edits). Especially: phantom_prompt prefix fix (commit 3d41953), Stage 4 Bug 1+2+5, watcher silent-miss, watcher dispatch (af2299f tonight). Are there any fixes that were noted as "done" but the code path the fix targets has since been refactored away?

## Output format

Open with one-sentence current-trust verdict on Phase 1 baseline:
- "Phase 1 is paper-grade trustworthy"
- "Phase 1 has 1+ paper-grade bug affecting §N"
- "Phase 1 has methodological concerns but no proven bug"
- "Insufficient time to verify — partial audit only"

Then:

### Confirmed bugs (would change paper numbers ≥0.5pp or invalidate a qualitative claim)
For each: layer (methodology / design / code / data / fix-regression), file:line evidence, what number/claim is affected, severity (HIGH/MED/LOW), defuse effort.

### Probable bugs (suspicious but couldn't fully verify)
Same format, mark "needs further check."

### Methodology concerns (not bugs, reviewer ammo)
Stratification missing / unit of analysis ambiguous / FP rule depends on something fragile / preregistration drift / etc. Quote preregistration.md lines vs current code where applicable.

### Cross-representation inconsistencies (highest priority class)
If SR for one cell differs across JSONL / condition_summary / aggregator / figure data: report all 4 numbers + which is right + which downstream claims use which.

### Fix-regression check
For each previously-documented bug fix in master_bug_catalog / pre_rerun_audit, verify it's still in place. If regressed, that's a HIGH severity finding.

### What you read and what you didn't
Brief enumeration. If a section was time-constrained out, say so.

### Verdict on next steps
If Phase 1 baseline holds: tell user they can proceed with paper §1 + §4 + §5 prose-fix tonight with confidence.
If Phase 1 has bugs: prioritized list of which to fix before next rerun. The "one thing to fix tonight" if user has 1-3h energy.

## Calibration

- This is paper-grade audit, not code review. Don't flag style or efficiency.
- Don't propose fixes. Identify the suspect, the impact, the defuse cost. User decides whether to fix.
- Calibrate severity by paper-grade impact, not aesthetic ugliness.
- If you find nothing ≥0.5pp suspect after 60 min reading, write the trust verdict and stop. Negative result is valid.
- Don't fabricate. If a file path doesn't exist, note it. If you can't trace a chain, say so.

## What this is NOT

- Not /codex-stress (that's paper prose reviewer)
- Not a methodology blessing — your role is suspicion-mode, not approval-mode
- Not bound by typical Web agent or interpretability subfield pitfalls — set your own attack vectors based on what the code and docs show
- Not a 5-minute scan. Take the time you need within 60 min budget. Better one deep verified finding than 20 shallow flags.

## Time budget

Up to 60 min. Codex foreground PID-based monitor will fire when done.
