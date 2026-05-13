---
type: issue
id: phase1-audit-2026-05-13
created: 2026-05-13
severity: HIGH
status: open
paper_section: "§1 + §4"
audit_source: codex_full_prerun + claude_parallel
defuse_eta: 2026-05-13 (~3h prose+code work)
related: [stage4-bug-pattern, codex-stress-wrap-2026-05-13]
---

# Phase 1 baseline pre-run audit — 6 paper-grade bugs surfaced

Codex full pre-run audit (PID 1811370, 10 min runtime) + Claude parallel verify found **4 HIGH codex + 1 HIGH Claude (fixed tonight) + 1 MED Claude** paper-grade issues affecting paper §1 + §4.

User paranoia (after 4 Stage 4 bugs caught this week + watcher dispatch bug tonight) is justified. Phase 1 baseline currently cannot generate §1 hero figure from canonical registry, plus 4 defuse-low fixes needed before paper-grade rerun.

## Codex findings (Claude verified file:line)

### HIGH-1 — Draft preregistration can launch paper-grade
- Evidence: [preregistration.md:3](../../pre_run/preregistration.md) `status: draft` + `registered_at: <pending advisor sync lock>`
- Launcher: [queue_16cell_paper_grade.sh:63](../../../../scripts/queues/queue_16cell_paper_grade.sh) only greps unresolved `TBD` thresholds, no prereg status check
- Affected: any "pre-registered" H1/H2/H3 claim until advisor lock
- Defuse: add `grep -q "^status: locked" docs/checkpoints/pre_run/preregistration.md || exit 2` to queue script

### HIGH-2 — GLM BLOCK downgraded to WARN at runtime
- Evidence: [glm_pre_launch_check.py:15](../../../../scripts/maintenance/glm/glm_pre_launch_check.py) documents exit 2 = BLOCK
- Implementation: [glm_pre_launch_check.py:150](../../../../scripts/maintenance/glm/glm_pre_launch_check.py) returns only OK/WARN, exits rc=1 on non-OK
- [launch.sh:130](../../../../scripts/maintenance/launch.sh) treats rc=1 as prompt-overridable, only rc=2 aborts
- Affected: same-site B0/B1 serialization + RESET_BEFORE hard-rule enforcement
- Defuse: differentiate BLOCK (exit 2) from WARN (exit 1) in glm_pre_launch_check.py

### MED-HIGH — TOST δ drift in §4 forest figures
- Evidence: prereg locks `TOST equivalence margin δ = 1.0pp` at [preregistration.md:192](../../pre_run/preregistration.md)
- Aggregator OK: [aggregate_phantom_lift.py:249](../../../../scripts/analysis/aggregate_phantom_lift.py) documents F03 fix to 1.0pp ✓
- Figure bug: [fig_meta_forest.py:56](../../../../scripts/analysis/figures/fig_meta_forest.py) `TOST_DELTA_PP = 0.5` hardcoded
- Figure bug: [fig_forest_drop_one.py:39](../../../../scripts/analysis/figures/fig_forest_drop_one.py) same hardcode
- Affected: §4 forest practical-equivalence band 2x too narrow per prereg
- Defuse: change `TOST_DELTA_PP = 1.0` in both figure files (10 min)

### HIGH-BLOCKER — Paper registry currently has zero paper-grade cells
- Evidence: [run_registry.py:37](../../../../scripts/analysis/lib/run_registry.py) defaults to `paper-grade` filter
- Manifest: [run_manifest.yaml:23](../../../../results/phantom_paper/run_manifest.yaml) all entries `grade: archived`
- Output: [sr_fp_per_mode.json:1](../../../../docs/analysis/cross_sites/sr_fp_per_mode.json) `{"cells": {}, "summary_table": []}` — empty
- Affected: §1 hero heatmap + FP plot cannot be generated from canonical registry
- Defuse: promote relevant cells in run_manifest.yaml from `archived` → `paper-grade`, then `make analysis`

### MED — validate_run gate doesn't accept phantom modes
- Evidence: [validate_run.py:48 _EXPECTED_CONDITIONS](../../../../scripts/analysis/validate_run.py) only `phase1_dom_router_0` / `phase1_som_router_0` / `phase1_vision_router_0`
- Affected: any phantom_* cell FAILS validation; paper-grade promotion via [Makefile:120](../../../../Makefile) gates on validation
- Defuse: extend `_EXPECTED_CONDITIONS` to include phantom_{dom,som,text,prompt} (10 min)

## Claude parallel findings (codex missed)

### CL1 — Stage 4 v2 NPZ default regression risk → FIXED tonight ✅
- 5 stage4 analysis scripts had `DEFAULT_NPZ = hidden_states.npz` (v1 buggy):
  - `stage4_layer_axis_emergence.py:33`
  - `stage4_axis2_layer_profile.py:36-37`
  - `stage4_axis2_per_task_fragility.py:35-36`
  - `stage4_logit_lens_axis2.py:38-39`
  - `stage4_pca_cosine_gap.py:30`
- Any rerun without explicit `--cls-npz hidden_states_v2_fixed.npz` silently uses v1 buggy data with Bug 2 (71/72 SOM_MARKS dropped)
- **FIX (2026-05-13 ~02:25 BST)**: all 5 defaults patched to `hidden_states_v2_fixed.npz`. Will commit with this issue doc.

### CL2 — §5.4 prose "axis-2 patching 0.20-0.30 displacement" vs cellhprm overlap=0.188
- cellhprm cls original L14 peak overlap→src = 0.188 (only 0.02 above chance ~0.16)
- §5.4 prose may quote a different metric (overlap→tgt drop, LD→tgt rise, etc.) — needs verification
- Affected: §5.4 magnitude claim. If wrong metric quoted, the "L11-L17 effect 0.20-0.30" is mis-cited
- Defuse: read section5_mechanism.md §5.4, map each cited number to results.json field
- Status: open, requires §5 prose re-read

## Agreement (highest-confidence weakest, lower-priority pile-on)

- Phantom-prompt prefix fix [som.py:178](../../../../p79/experiment/som.py) ✓ in place (commit 3d41953)
- FP rule uniform across modes [analysis.py:148](../../../../p79/experiment/analysis.py) `compute_adjusted_success_batch` no per-mode branch ✓
- Watcher longest-prefix dispatch [myriad_watcher.py:197](../../../../scripts/maintenance/glm/myriad_watcher.py) ✓ verified production (af2299f tonight + cls_tsh 359768 dispatched correctly)
- Composite eval `|` handling [analysis.py:80](../../../../p79/experiment/analysis.py) F20 fix 2026-05-09 ✓

## Verdict

**Codex**: "Do not use current Phase 1 chain as paper-grade tonight."
**Claude**: agree. §1 hero heatmap literally cannot be generated from canonical paper registry (sr_fp_per_mode.json empty). Prereg lock + GLM BLOCK + TOST δ + validate phantom gates all need defuse first.

## Recommended fix order (before next paper-grade rerun)

1. (10 min) HIGH-3 fix `TOST_DELTA_PP = 1.0` in fig_meta_forest.py + fig_forest_drop_one.py
2. (10 min) MED fix `_EXPECTED_CONDITIONS` in validate_run.py (add phantom IDs)
3. (15 min) HIGH-2 fix glm_pre_launch_check.py exit codes (rc=2 for BLOCK)
4. (15 min) HIGH-1 fix queue_16cell_paper_grade.sh prereg status grep
5. (30+ min) HIGH-BLOCKER: promote relevant cells in run_manifest.yaml → `make analysis` regenerates sr_fp_per_mode.json
6. (~1h) CL2 read §5.4 prose, verify cited cellhprm magnitudes vs results.json field

Total **~2h defuse + 1h §5.4 prose verify = 3h before paper-grade rerun is safe**.

## Sources

- Codex audit output: `docs/checkpoints/codex_outputs/codex_full_prerun_audit_2026-05-13.md` (19886 lines)
- Codex prompt: `docs/checkpoints/codex_prompts/codex_full_prerun_audit_2026-05-13.md`
- Claude session: 2026-05-13 02:00-02:30 BST
