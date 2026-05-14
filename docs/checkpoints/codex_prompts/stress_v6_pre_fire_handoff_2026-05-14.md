# Audit scope handoff (Claude → codex Mode B) — /stress v6 pre-fire 2026-05-14

## Claude scope (already covered, do NOT re-read)

**Files read by Claude /stress this round**:
- `scripts/analysis/preregistration_decision_test.py` (deep — Round 3 rewrite: drop-one + DL meta + superiority + framing rule mapper L90-580)
- `scripts/analysis/aggregate_phantom_lift.py:420-470` (denominator inconsistency section)
- `scripts/analysis/aggregate_phantom_meta.py:1-100` (function signatures only)
- `scripts/queues/queue_baseline.sh:90-240` (env loading + config check + reset logic)
- `scripts/queues/queue_phantom_som.sh:1-90` (header + validation)
- `configs/exp_v2_base.yaml:1-120` (defaults + backends)
- `configs/exp_v2_B0_dom_classifieds.yaml` (one new baseline config sample)
- `docs/checkpoints/pre_run/preregistration.md` (§2 H1/H2/H3, §3 family, §4 locked choices, R1-R5)

**Claude's 7 findings filed** (use these as cross-validate targets):
- F1 [P0 OOB]: DL τ² biased at k=4 → REML/PM recommended (Veroniki et al. 2016)
- F2 [P0 OOB]: Wald 1.96 CI on k=4 RE meta anti-conservative → Hartung-Knapp t_{k-1}
- F3 [P0 OOB]: Heterogeneity branch only checks H1 I², ignores H3 axes
- F4 [P0]: aggregate_phantom_lift.py denominator inconsistency NOT yet fixed (sr_3 vs sr_3_psom_only)
- F5 [P1 OOB]: TWO DL meta implementations coexist (`derslong_laird_meta` vs `dersimonian_laird_meta`)
- F6 [P1]: H1(i) two-sided p redundant with H1(ii) superiority
- F7 [P2 OOB]: bootstrap seed=42 shared across all calls → cross-statistic CI correlated

**Claude's top OOB attack**: DL τ² estimator at k=4 is biased toward 0; Wald critical inflates Type I → both reviewer-3 mode kills. Advisor sync question.

**Claude scope gaps (NOT covered, your job)**:
- queue_phase1_paper_grade.sh post-rename + chain logic (renamed in commit e9ddbe3)
- queue_chain.sh (wrapper invoked by build_*_chain)
- scripts/preflight_v2.sh (pre-launch validation)
- p79/experiment/som.py production `_extract_text_marks` (deployment-vs-experiment boundary)
- p79/envs/vwa_wrapper.py viewport filtering (CLAUDE.md memory mentions known operator-precedence bug in_viewport_ratio)
- p79/experiment/runner/main.py orchestrator
- p79/experiment/logger_v2.py JSONL fsync semantics
- p79/experiment/io_utils.py restart dedup
- p79/experiment/config.py config merge + defaults propagation
- `aggregate_phantom_meta.py:71-160` derslong_laird_meta full body (verify cross-implementation drift)
- Multiple new B0+B1 baseline configs (12 total) — sample-only checked by Claude
- scripts/run_experiment.py entry point

## Codex scope (assigned, complementary)

**Persona** = **reproducibility auditor + ML systems engineer** (NOT mechinterp implementer — Claude already covered statistical methodology).

**Files to read** (deep, 8-12 total):
1. `scripts/queues/queue_phase1_paper_grade.sh` (full)
2. `scripts/queues/queue_chain.sh`
3. `scripts/preflight_v2.sh`
4. `p79/experiment/som.py` — production `_extract_text_marks` regex
5. `p79/envs/vwa_wrapper.py` — viewport filtering (CLAUDE.md mem: `processors.py:218 in_viewport_ratio` has operator-precedence bug, check if propagated)
6. `p79/experiment/runner/main.py` — orchestrator
7. `p79/experiment/logger_v2.py` — JSONL fsync
8. `p79/experiment/io_utils.py` — restart dedup
9. `p79/experiment/config.py` — config merge + DEFAULT_CONFIG
10. `scripts/analysis/aggregate_phantom_meta.py` (full body — verify derslong_laird_meta vs Claude's dersimonian_laird_meta)
11. At least 4-6 NEW baseline configs (Claude only checked 1 sample) — verify all 12 are paper-grade clean

**Do NOT re-read**: files in Claude scope list above. Don't waste budget on redundant statistical methodology audit (Claude covered DL meta + superiority + bootstrap + framing rule).

## Cross-validate targets

1. **Sibling propagation check** (v6 mandate): for each fix in commit dccd11f, grep for siblings still using v1 logic. Specifically:
   - T1 (TOST→superiority): did any script still call `tost_equivalence` as primary gate? Grep all paper-related analysis scripts.
   - T3 (heterogeneity branch): does ANY OTHER script implement R1-R5 framing rule with stale I² check? Cross-check `aggregate_phantom_meta.py` + figure scripts.
   - A4 (12 baseline configs): does `queue_baseline.sh` accept all 12 cleanly? Run `bash scripts/queues/queue_baseline.sh B1 vision reddit` dry-grep (don't actually launch) and verify config path + backend choice resolve.

2. **Production-experiment boundary**: `p79/experiment/som.py::_extract_text_marks` is the production [SOM_MARKS] extractor. Does the new Phase 1a launch path use the SAME regex as paper §3 prose claims? If `phantom_som` mode invokes a different code path than `som` mode, deployment claim "P-SoM = regex filter of AXTree" may not hold in current code.

3. **Config merge silent failures**: `p79/experiment/config.py` should merge `exp_v2_base.yaml` + `exp_v2_<baseline>_<mode>_<site>.yaml`. Verify the 12 NEW baseline configs (`exp_v2_B0_dom_classifieds.yaml` etc.) inherit all required fields correctly — especially:
   - `runtime.max_steps` (configs say 30, base says 40 — which wins?)
   - `metrics.energy.enabled` (B0 configs say `false`, base says `true` — which wins?)
   - `experiment.seed` (base says 42; configs don't override — verify propagates)
   - `task.include_sites` (base says all 3 sites; configs say one site — does merge correctly subset?)

4. **JSONL restart dedup correctness**: `p79/experiment/io_utils.py::read_jsonl_dedup` handles restart artifacts. If a Phase 1a condition crashes + restarts, does the dedup preserve evaluator verdict correctly? Verify the dedup key includes task_id + step_idx + episode_id (not just task_id which would collide across restarts).

5. **Reproducibility provenance**: do the new baseline configs capture HF model SHA + git commit + env_snapshot? CLAUDE.md memory says B1 needs `model_revision` pin for paper-grade. Audit one new B1 config (`exp_v2_B1_dom_classifieds.yaml`) for revision pinning.

## Calibration

- Pre-fire scope (≥7 findings, ≥3 OOB)
- Persona = reproducibility auditor / ML systems engineer (NOT methodologist)
- Bilingual output per v6 FAIL CHECK (中文 attacks + English code quotes)
- Bypass any finding Claude already filed (F1-F7 above)
- If you find Claude's gap was actually fine, say so — that's valid negative result
- Time budget: 45-60min
