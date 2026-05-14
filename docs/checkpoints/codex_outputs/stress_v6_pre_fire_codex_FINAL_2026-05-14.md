Phase 1a has 9 system / reproducibility flaw(s) — must fix before fire

**Findings**
1. **[OOB] Layer: chain/config resume**
   Evidence: `scripts/queues/queue_baseline.sh:133-140`, `scripts/queues/queue_phantom_text.sh:152-163`, `p79/experiment/runner/main.py:333-342`
   English quote: `EXISTING="$(ls -dt "${PHASE_DIR}/${CFG_NAME}_"[0-9]* ...)"`, then runner skips existing summaries when `has_steps`.
   Failure mode: silent corruption. Phase 1a will reuse old archived/pre-fix dirs for several phantom cells instead of making fresh 2026-05-14 runs. I found existing matches for B0 P-text cls/red, B0 P-prompt cls/red, B1 P-text cls, B1 P-prompt cls, B1 P-SoM red.
   Severity: **P0**
   Defuse: add `FORCE_NEW=1` default for paper-grade master chain or make run IDs timestamped (`${TS_FULL}`) unless explicit `RESUME_EXISTING=1`. 30-60 min.

2. **[OOB] Layer: preflight**
   Evidence: `scripts/queues/queue_phase1_paper_grade.sh:112-115`
   English quote: `bash scripts/preflight_v2.sh --no-strict-ports ... | tail -5 | sed ...`
   Failure mode: launch gate is non-blocking. I ran preflight here and it returned `PRECHECK_RC=1` for missing VWA env/imports, but master script would only print the last 5 lines and continue because it never increments `errors`.
   Severity: **P0**
   Defuse: capture preflight exit status and `errors+=1`; use strict ports for actual fire. 15-30 min.

3. **[OOB] Layer: chain/watchdog**
   Evidence: `scripts/queues/queue_chain.sh:67-80`, `scripts/queues/queue_chain.sh:140`
   English quote: `while pgrep -f "run_experiment.py.*${pattern}" ...; ... log "runner done"`
   Failure mode: runner crash is treated as completion; chain advances without checking `condition_summary_v2.json`.
   Severity: **P0/P1**
   Defuse: after process exit, require `results/.../<run_id>/<COND_ID>/condition_summary_v2.json`; abort chain if absent. 30-45 min.

4. **[OOB] Layer: logging**
   Evidence: `scripts/queues/queue_phantom_som.sh:175-181`, `scripts/queues/queue_phantom_text.sh:215-221`, `scripts/queues/queue_phantom_prompt.sh:171-177`; `p79/cli/run_experiment.py:36-37`
   English quote: `--log_path "${RUNNER_LOG}" > /dev/null 2>&1`
   Failure mode: phantom runner logs are discarded. `--log_path` is only stored in config metadata; Python logging still goes to stderr, which is redirected to `/dev/null`.
   Severity: **P1**
   Defuse: redirect runner stdout/stderr to `${RUNNER_LOG}` like `queue_baseline.sh` does. 10 min.

5. **[OOB] Layer: watchdog startup**
   Evidence: `scripts/queues/queue_baseline.sh:239-243`
   English quote: `echo "[baseline][warn] watchdog failed to start"`
   Failure mode: baseline cells continue without watchdog if watchdog spawn fails; combined with finding 3, mid-run crashes become silent missing data.
   Severity: **P1**
   Defuse: make watchdog failure fatal for paper-grade launch. 10 min.

6. **Layer: active-run protection**
   Evidence: `scripts/queues/queue_phase1_paper_grade.sh:124-130`, `scripts/queues/queue_baseline.sh:147-156`
   English quote: master only `WARN: Existing runs detected`; queue scripts reset after checking only exact `RUN_ID`.
   Failure mode: if any other same-site run is active, `RESET_BEFORE=1` can reset site state underneath it.
   Severity: **P1**
   Defuse: make active run detection fatal unless explicit `ALLOW_ACTIVE_RUNS=1`; check same site, not only exact run id. 30 min.

7. **Layer: propagation/config naming**
   Evidence: `scripts/queues/queue_phantom_text.sh:78-87`; missing `configs/exp_v2_B1_phantom_text_classifieds.yaml`; legacy present at `configs/exp_v2_B1_phantom_dom_classifieds.yaml:1-27`
   Failure mode: launch works via legacy fallback, but run ID/config name remains `phantom_dom`, not canonical `phantom_text`; easy reviewer/provenance confusion.
   Severity: **P2**
   Defuse: add canonical B1 phantom_text classifieds config or explicitly document manifest alias. 15 min.

8. **Layer: reproducibility provenance**
   Evidence: B1 configs such as `configs/exp_v2_B1_dom_classifieds.yaml:23-27`; code default at `p79/agents/qwen3vl_agent.py:53-57`; snapshot at `scripts/provenance/snapshot_env.py:114-125`
   Failure mode: configs do not carry `revision`; code pins default SHA, and current HF HEAD matches, but run metadata config does not prove the loaded SHA.
   Severity: **P2**
   Defuse: put `revision: ebb281ec70b05090aa6165b016eac8ec08e71b17` in `backends.local_4b` and snapshot the configured revision. 20 min.

9. **Layer: phase chain completeness**
   Evidence: `scripts/queues/queue_phase1_paper_grade.sh:178-195`; dry resolution found missing `configs/exp_v2_B0_phantom_prompt_shopping.yaml`, `exp_v2_B1_dom_shopping.yaml`, `exp_v2_B1_som_shopping.yaml`, `exp_v2_B1_vision_shopping.yaml`, `exp_v2_B1_phantom_prompt_shopping.yaml`
   Failure mode: Phase 1b shop chain advertised by this launcher will fail mid-chain. Not Phase 1a, but same orchestration surface.
   Severity: **P2**
   Defuse: hide Phase 1b until configs exist or fail dry-run on missing configs. 20 min.

**Negative Checks**
- Viewport operator-precedence bug is fixed: `external/visualwebarena/browser_env/processors.py:218` now has `ratio = (overlap_width * overlap_height) / (width * height)`.
- SoM/P-SoM text path is shared: `_extract_text_marks` at `p79/experiment/som.py:24-35`; `som`, `phantom_som`, and `phantom_text` all route through `_build_som_result` at `p79/experiment/som.py:165-189`.
- Config merge works for 12 new baseline configs: `runtime.max_steps=30`, B0 `energy.enabled=false`, B1 inherits `energy.enabled=true`, `seed=42`, and site subsets load correctly.
- JSONL dedup is weak (`p79/experiment/io_utils.py:12-30` only detects `step_idx==0`), but runner unlinks stale step JSONL before rerun at `p79/experiment/runner/main.py:697-701`; no proven Phase 1a blocker from this path.

**Cross-Validate Claude F1-F7**
- F1: N/A, stats estimator scope.
- F2: N/A, stats CI scope.
- F3: N/A for systems; no other R1-R5 mapper found outside `preregistration_decision_test.py`.
- F4: N/A, Claude-owned denominator section.
- F5: CONFIRM/EXTEND. Two DL implementations exist; they agree numerically only if `aggregate_phantom_meta` gets SEs and `preregistration_decision_test` gets variances.
- F6: EXTEND. `aggregate_phantom_lift.py` still has `bootstrap_tost_p = bootstrap_tost_equivalence_p` and old TOST wording/call sites at lines `278`, `452`, `471`, `662`, `823`, `901`.
- F7: CONFIRM in aggregate helpers: shared `seed=42` defaults remain in bootstrap functions.

**Sibling-Propagation Report**
- `grep -l "tost_equivalence\|equivalence_rejected" scripts/analysis/`: only `aggregate_phantom_lift.py` and `preregistration_decision_test.py` excluding pycache. No `equivalence_rejected`, but `aggregate_phantom_lift.py` still has v1 alias/call wording.
- 12 new baseline configs are accepted by `queue_baseline.sh` dry path: all `B0/B1 × dom/som/vision × classifieds/reddit` resolve to existing config files.
- DL implementation agreement: numeric agreement to floating precision after converting SE→variance for `preregistration_decision_test`; API contract drift remains.
- T2 H2 cost-only scope appears propagated in decision script; energy figures remain descriptive, not H2 gate.
- T3 R1-R5 branch appears isolated to `preregistration_decision_test.py`; no sibling R1-R5 implementation found.

**Pre-Fire Actions**
P0 before fire: fix forced-new run IDs, make preflight fatal, and make `queue_chain` require completion sentinel.

P1 can launch dirty only with disclosure: phantom logs to `/dev/null`, baseline watchdog warning-only, active-run reset warning-only.

P2 defer: canonical B1 P-text config naming, explicit HF revision in YAML, Phase 1b shop config gaps.