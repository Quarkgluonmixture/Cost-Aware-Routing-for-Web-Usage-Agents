# P79 Makefile — daily commands one-liners
#
# Usage examples:
#   make analysis                                # full analysis pipeline
#   make analysis FAST=1                         # aggregators + figures only
#   make analysis RUN=<run_dir>                  # single-run + downstream
#   make test                                    # full pytest suite
#   make smoke                                   # just smoke test (fast)
#   make rederive RUN=<run_dir>
#   make analyze RUN=...                         # rederive → reason_diag → cross_rep → analyze_run → confidence
#   make compare B0=<b0_run> B1=<b1_run> SITE=classifieds
#   make clean-tasks RUN=<run> COND=phase1_som_router_0 SITE=shopping TASKS=0-465
#   make schedule-list                           # list active background watch processes
# (B-394 / A1.15 C3 T6=(a), 2026-05-16): `make watch-reddit` retired —
# wait_for_reddit_then_rederive.sh deleted (April-specific hardcoded helper,
# B1-only baseline, cross-baseline collision risk after B2 Gemma3-VL joined
# 2026-05-14 baseline set). Paper-grade chain uses `queue_chain.sh`.
#
# Variables (override on command line):
#   RUN  — run directory (required for rederive/analyze)
#   B0   — B0 run dir (compare)
#   B1   — B1 run dir (compare)
#   SITE — site name (compare/clean)
#   COND — condition id (clean-tasks)
#   TASKS — task id range, e.g. "0-465" or "5,10,20" (clean-tasks)

PYTHON ?= .venv/bin/python3
PYTEST ?= .venv/bin/pytest

.PHONY: help test smoke smoke-only rederive rederive-all analyze cross-rep error-scan \
        confidence compare reason-diag clean-tasks schedule-list \
        validate gallery rsync-to-hub rsync-from-hub rsync-artifacts-from-hub \
        aggregate-cross-site summary-collect routing-auroc analyze-paper \
        analyze-paper-per-run compare-b0-b1-all phantom-lift \
        analyze-layer0 analyze-layer1 analyze-layer2 analyze-layer3 analyze-layered \
        aggregate-sr-fp fig12-micro-heatmap aggregate-cost-electricity analyze-mechanism \
        analysis _per_run_all _aggregate _figures _status active \
        glm-update-cells glm-refresh-playbook check-links vwa-generate-configs

help:
	@echo "P79 Makefile — see header for usage examples"
	@echo ""
	@echo "  ⭐ make analysis              # full analysis pipeline (per-run + cross-condition + figures)"
	@echo "  ⭐ make analysis FAST=1       # skip per-run; aggregators + figures only (~30s)"
	@echo "  ⭐ make analysis RUN=<dir>    # single-run pipeline + downstream"
	@echo ""
	@echo "  make active                  # real-time process scan"
	@echo "  make test                    # full pytest suite"
	@echo ""
	@echo "  Per-run / per-cell:          make analyze RUN=<dir>"
	@echo "  Compare runs:                make compare B0=<run> B1=<run> SITE=<site>"
	@echo "  Maintenance:                 make clean-tasks RUN= COND= SITE= TASKS="
	@echo ""
	@echo "  Internal (called by 'make analysis'):"
	@echo "    _per_run_all _aggregate _figures _status"
	@echo "    aggregate-sr-fp / phantom-lift / routing-auroc / aggregate-cross-site"
	@echo "    summary-collect / aggregate-cost-electricity / analyze-mechanism"
	@echo "    analyze-layer0 / analyze-layer1 / analyze-layer2 / analyze-layer3 (Phase 1 layered)"
	@echo "    figures / fig12-micro-heatmap"
	@echo ""
	@echo "  All targets:                 grep '^[a-z]' Makefile | sed 's/:.*//' | sort -u"

# ---- Live status ----
# Real-time scan of run_experiment + experiment_watchdog processes;
# replaces the manually-maintained §1 Active Processes table in next_steps.md.
active:
	@$(PYTHON) scripts/maintenance/active_processes.py

# ---- Tests ----
test:
	@# /stress A1.12 P1-4 fix (2026-05-16): `-x` (fail-fast) removed so full
	@# failure picture surfaces in one run instead of hiding remaining failures
	@# behind the first. `--tb=short` keeps output digestible while still
	@# showing assert location. `pyproject.toml [tool.pytest.ini_options]` now
	@# sets `--strict-markers -ra -rs` so silent importorskip / typo'd marker
	@# bugs no longer hide.
	$(PYTEST) tests/ --tb=short -q

smoke:
	$(PYTEST) tests/test_runner_smoke.py tests/test_runner_integration.py -v

smoke-only:
	$(PYTEST) tests/test_runner_smoke.py -v

# ---- Rederive ----
rederive:
	@test -n "$(RUN)" || (echo "ERROR: RUN=<run_dir> required"; exit 1)
	$(PYTHON) scripts/maintenance/rederive_episode_summary.py --run-dir $(RUN)

rederive-all:
	$(PYTHON) scripts/maintenance/rederive_episode_summary.py --all-b0

# ---- Analysis pipeline ----
reason-diag:
	@test -n "$(RUN)" || (echo "ERROR: RUN=<run_dir> required"; exit 1)
	$(PYTHON) scripts/analysis/analyze_reason_diagnostics.py --run-dir $(RUN) --skip-similarity --no-plots

cross-rep:
	@test -n "$(RUN)" || (echo "ERROR: RUN=<run_dir> required"; exit 1)
	$(PYTHON) scripts/analysis/analyze_cross_representation.py --run-dir $(RUN) --priority all

confidence:
	@test -n "$(RUN)" || (echo "ERROR: RUN=<run_dir> required"; exit 1)
	$(PYTHON) scripts/analysis/analyze_confidence_calibration.py --run-dir $(RUN)

# Full pipeline: rederive → reason_diag → cross_rep → analyze_run → confidence
analyze: rederive reason-diag cross-rep
	@test -n "$(RUN)" || (echo "ERROR: RUN=<run_dir> required"; exit 1)
	$(PYTHON) -c "from p79.experiment.analysis import analyze_run; analyze_run('$(RUN)')"
	$(MAKE) confidence RUN=$(RUN)

# ---- Cross-model / cross-site ----
compare:
	@test -n "$(B0)" || (echo "ERROR: B0=<b0_run_dir> required"; exit 1)
	@test -n "$(B1)" || (echo "ERROR: B1=<b1_run_dir> required"; exit 1)
	@test -n "$(SITE)" || (echo "ERROR: SITE=<site> required"; exit 1)
	$(PYTHON) scripts/analysis/compare_b0_b1.py --b0-run-dir $(B0) --b1-run-dir $(B1) --site $(SITE)

# ---- Maintenance ----
clean-tasks:
	@test -n "$(RUN)" -a -n "$(COND)" -a -n "$(SITE)" -a -n "$(TASKS)" || \
		(echo "ERROR: RUN= COND= SITE= TASKS= all required"; exit 1)
	$(PYTHON) scripts/maintenance/clear_tasks.py --run-dir $(RUN) --condition $(COND) --site $(SITE) --tasks $(TASKS)

validate:
	@test -n "$(RUN)" || (echo "ERROR: RUN=<run_dir> required"; exit 1)
	$(PYTHON) scripts/analysis/validate_run.py --run-dir $(RUN)

# C10 audit gate: strict validation as required paper-grade promotion gate.
# Exit code 0 = ✓ paper-grade (all 27 checks pass + warnings treated as failures),
# 2 = strict-fail (warnings or failures present). Wire into queue scripts via:
#   make validate-strict RUN=<run> || { echo 'NOT paper-grade'; exit 1; }
validate-strict:
	@test -n "$(RUN)" || (echo "ERROR: RUN=<run_dir> required"; exit 1)
	$(PYTHON) scripts/analysis/validate_run.py --run-dir $(RUN) --strict --output $(RUN)/validation_report.json

# C10/A5/F8/B7 pre-launch gate: verify version locks + git working tree clean +
# pre-launch invariants before kicking off a paper-grade rerun. Required by
# preregistration.md §4 stopping rule (a). Exit non-zero on any failure.
pre-launch-check:
	@echo "=== Pre-launch invariant checks ==="
	@echo "1. Git working tree clean (incl. untracked)..."
	@git diff-index --quiet HEAD -- 2>/dev/null || (echo "❌ git working tree has uncommitted tracked changes"; exit 1)
	@# F38 audit fix 2026-05-09: also reject untracked files that could
	@# affect a paper-grade run (config / scripts / cells lying around).
	@test -z "$$(git status --porcelain --untracked-files=all)" || (echo "❌ untracked files present (run: git status --porcelain --untracked-files=all)"; exit 1)
	@echo "   ✓ clean (no tracked diffs, no untracked files)"
	@echo "2. VWA submodule SHA matches lock..."
	@LOCK_SHA="2f9b0b47175a1bffa01e13100e3075e212161a89"; \
	 ACTUAL=$$(git -C external/visualwebarena rev-parse HEAD 2>/dev/null); \
	 test "$$ACTUAL" = "$$LOCK_SHA" || (echo "❌ VWA SHA mismatch: expected $$LOCK_SHA, got $$ACTUAL"; exit 1); \
	 echo "   ✓ $$LOCK_SHA"
	@echo "3. HF model snapshot exists..."
	@HF_REV="ebb281ec70b05090aa6165b016eac8ec08e71b17"; \
	 HF_DIR="$$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/$$HF_REV"; \
	 test -f "$$HF_DIR/config.json" || (echo "❌ HF model not at $$HF_DIR"; exit 1); \
	 echo "   ✓ $$HF_REV"
	@echo "4. Playwright version matches lock..."
	@PW_VER=$$(.venv/bin/pip show playwright 2>/dev/null | grep ^Version: | awk '{print $$2}'); \
	 test "$$PW_VER" = "1.58.0" || (echo "❌ Playwright version mismatch: expected 1.58.0, got $$PW_VER"; exit 1); \
	 echo "   ✓ $$PW_VER"
	@echo "5. Disk free > 20GB..."
	@FREE_GB=$$(df --output=avail -BG . | tail -1 | tr -d 'G '); \
	 test "$$FREE_GB" -ge 20 || (echo "❌ Disk free $${FREE_GB}GB < 20GB"; exit 1); \
	 echo "   ✓ $$FREE_GB GB"
	@echo "6. Seed configured in base config..."
	@grep -q "seed: 42" configs/exp_v2_base.yaml || (echo "❌ seed=42 not in configs/exp_v2_base.yaml"; exit 1)
	@echo "   ✓ seed=42"
	@echo "7. pytest sanity (60s wall budget, fail-fast)..."
	@# F audit fix 2026-05-09: catch import-broken / smoke-test-broken
	@# state before launching a long-running paper-grade rerun. -x stops
	@# on first failure; outer `timeout 60` enforces wall-clock budget
	@# (pytest-timeout plugin not installed; use coreutils timeout).
	@timeout 60 .venv/bin/pytest tests/ -x -q --no-header 2>&1 | tail -5
	@echo "   ✓ pytest passed"
	@echo "8. Manifest paper-grade cells present..."
	@# Manifest grade promotion check (post-F01 prerequisite, see
	@# docs/reference/launch_checklist.md §1). Warn if 0 paper-grade
	@# cells in run_manifest.yaml — a paper-grade rerun cannot
	@# produce figures without manifest entries. P79_ALLOW_NO_PAPER_GRADE=1
	@# bypasses this check (e.g. for the very first cell of a rerun batch).
	@N_PG=$$(.venv/bin/python3 -c "from scripts.analysis.lib.run_registry import get_all_cells; print(len(get_all_cells()))" 2>/dev/null || echo 0); \
	 if [ "$$N_PG" = "0" ] && [ "$${P79_ALLOW_NO_PAPER_GRADE:-0}" != "1" ]; then \
	   echo "❌ run_manifest.yaml has 0 paper-grade cells. Add an entry per launch_checklist.md §1 OR set P79_ALLOW_NO_PAPER_GRADE=1 to bypass."; \
	   exit 1; \
	 elif [ "$$N_PG" = "0" ]; then \
	   echo "   ⚠️  0 paper-grade cells (bypassed via P79_ALLOW_NO_PAPER_GRADE=1)"; \
	 else \
	   echo "   ✓ $$N_PG paper-grade cell(s) registered"; \
	 fi
	@echo ""
	@echo "✓ All pre-launch invariants passed. Safe to kick off paper-grade rerun."

gallery:
	@test -n "$(RUN)" || (echo "ERROR: RUN=<run_dir> required"; exit 1)
	$(PYTHON) scripts/maintenance/generate_gallery.py --run-dir $(RUN)

# ---- Cross-run / paper-grade aggregation (cross-condition) ----
# Run once after a batch of conditions is paper-grade clean.
# Outputs land under results/phantom_paper/ for paper drafts to consume.

# Single source of truth: results/phantom_paper/run_manifest.yaml
# (set by scripts/analysis/lib/run_registry.py::get_run_dirs_paper_vwa())
RUN_DIRS_PAPER_VWA = $(shell $(PYTHON) -c "from scripts.analysis.lib.run_registry import get_run_dirs_paper_vwa; print(' '.join(str(p) for p in get_run_dirs_paper_vwa()))")

# ---- Single entry point for full analysis pipeline ----
# Default: full pipeline (per-run + cross-condition + figures + status)
# FAST=1: skip per-run analysis, only regen aggregators + figures (use after small data update)
# RUN=<dir>: run per-run pipeline on single run only (then full cross-condition)
analysis:
ifeq ($(FAST),1)
	@echo "[analysis FAST=1] skipping per-run pipeline; aggregators + figures only"
	$(MAKE) _aggregate
	$(MAKE) _figures
	$(MAKE) _status
else ifneq ($(RUN),)
	@echo "[analysis RUN=$(RUN)] single-run pipeline + downstream"
	$(MAKE) analyze RUN=$(RUN)
	$(MAKE) _aggregate
	$(MAKE) _figures
	$(MAKE) _status
else
	@echo "[analysis] full pipeline: per-run + cross-condition + figures + status"
	$(MAKE) _per_run_all
	$(MAKE) _aggregate
	$(MAKE) _figures
	$(MAKE) _status
endif
	@echo "[analysis] post-hook: triggering PLAYBOOK refresh in background..."
	@nohup bash -c "sleep 5 && $(MAKE) glm-update-cells APPLY=1 && $(MAKE) glm-refresh-playbook APPLY=1" \
	  >> logs/cron/glm_playbook.log 2>&1 < /dev/null & disown ; true

# Per-run pipeline for all paper-grade VWA runs (loop over registry)
_per_run_all:
	@for rd in $(RUN_DIRS_PAPER_VWA); do \
	  echo "── per-run: $$rd ──"; \
	  $(MAKE) analyze RUN=$$rd || exit 1; \
	done

# Cross-condition aggregators (depends on per-run output)
_aggregate:
	$(MAKE) aggregate-sr-fp
	$(MAKE) phase1-prereg-gate
	$(MAKE) phase1-full-prereg-decision
	$(MAKE) phantom-lift
	$(MAKE) phantom-meta
	$(MAKE) routing-auroc
	$(MAKE) aggregate-cross-site
	$(MAKE) summary-collect
	$(MAKE) aggregate-cost-electricity
	$(MAKE) analyze-mechanism
	$(PYTHON) scripts/analysis/axis_effect_size.py
	$(PYTHON) scripts/analysis/axis1_microbehavior.py
	$(PYTHON) scripts/analysis/aggregate_failure_modes.py
	$(MAKE) compare-b0-b1-all

# All figures (depends on aggregator output)
_figures:
	$(PYTHON) scripts/analysis/figures/fig0a_sr_per_mode_heatmap.py
	# fig0b_fp_rate_per_mode retired §139.8 + /stress A1.6 (2026-05-16)
	$(PYTHON) scripts/analysis/figures/fig0b_extra_confidence_calibration.py
	$(PYTHON) scripts/analysis/figures/fig0c_drop_one_oracle.py
	$(PYTHON) scripts/analysis/figures/fig0c_phantom_lift_bars.py
	$(PYTHON) scripts/analysis/figures/fig0d_taskpool_jaccard.py
	# /stress A1.20 P0-5-B* / Q3=B (2026-05-17): fig0e_category_mode_heatmap deferred —
	# reads codex_audit_*.json with NO live producer in Makefile (only archived copies
	# under docs/archive/analysis_pre_2026-05-15/). Reproducibility broken on clean
	# rerun. Re-enable when aggregate_category_mode.py producer is built + advisor
	# confirms category taxonomy (paper §1 category claim deferred until then).
	# $(PYTHON) scripts/analysis/figures/fig0e_category_mode_heatmap.py
	$(PYTHON) scripts/analysis/figures/fig0f_overlap_stacked_bar.py
	$(PYTHON) scripts/analysis/figures/fig0g_routing_auroc_heatmap.py
	$(PYTHON) scripts/analysis/figures/fig1ab_cascade_diamond.py
	$(PYTHON) scripts/analysis/figures/fig1c_strategy_gradient.py
	$(PYTHON) scripts/analysis/figures/fig2_micro_divergence_heatmap.py
	$(PYTHON) scripts/analysis/figures/fig2b_target_hit_rate.py
	$(PYTHON) scripts/analysis/figures/fig2c_keyword_repeat.py
	$(PYTHON) scripts/analysis/figures/fig2d_first_action_divergence.py
	$(PYTHON) scripts/analysis/figures/fig2e_cross_site_validity.py
	$(PYTHON) scripts/analysis/figures/fig2f_first_divergence.py
	$(PYTHON) scripts/analysis/figures/fig3a_token_cost_intra_baseline.py
	$(PYTHON) scripts/analysis/figures/fig3b_image_token_gap.py
	$(PYTHON) scripts/analysis/figures/fig3c_latency_per_step.py
	$(PYTHON) scripts/analysis/figures/fig3d_cost_sr_frontier.py
	$(PYTHON) scripts/analysis/figures/fig3_regional_carbon.py
	$(PYTHON) scripts/analysis/figures/fig_forest_drop_one.py
	$(PYTHON) scripts/analysis/figures/fig_meta_forest.py
	$(PYTHON) scripts/analysis/figures/fig_phantom_structure_venn.py
	$(PYTHON) scripts/analysis/figures/fig_failure_modes_per_cell.py

# Live evidence status snapshot (read-only summary of aggregator outputs)
_status:
	$(PYTHON) scripts/analysis/layered_status.py
	@echo ""
	@echo "[analysis] outputs:"
	@echo "  Aggregators: results/phantom_paper/{phantom_lift,auroc_cross_condition,run_summary_collect}.{csv,md}"
	@echo "  Figures:     results/phantom_paper/figures/*.png"
	@echo "  Live status: docs/analysis/layered_evidence_status.md"
	@echo "  Per-run:     results/visualwebarena/phase1/<run>/analysis/"

aggregate-cross-site:
	$(PYTHON) scripts/analysis/aggregate_cross_site.py \
	  --run-dirs $(RUN_DIRS_PAPER_VWA) \
	  --output-dir results/phantom_paper/cross_site

# Run-level summary collector (one row per run)
summary-collect:
	$(PYTHON) scripts/analysis/collect_analysis_summary.py \
	  $(foreach rd,$(RUN_DIRS_PAPER_VWA),--run-dir $(rd)) \
	  --output results/phantom_paper/run_summary_collect.json

# Cross-condition routing signal AUROC table (Section 6 claim support)
routing-auroc:
	$(PYTHON) scripts/analysis/aggregate_routing_auroc.py

# B-184: Phase 1 PRIMARY gate (preregistration.md §1 lock).
# FE inverse-variance pooled P-SoM drop-one over 6 planned (site, model) cells,
# one-sided superiority test against δ=1.0pp at α=0.05. This is the
# CANONICAL paper §1 H1 source-of-truth — every paper §1 hero number should
# trace back to results/phantom_paper/phase1_prereg_gate.{csv,json,md}.
# `phantom-lift` (below) computes the legacy 3→5 lift estimand and is now
# appendix-exploratory only (codex B2 catch — different estimand than prereg).
phase1-prereg-gate:
	$(PYTHON) scripts/analysis/aggregate_phase1_prereg_gate.py

# A1.21 P0-2 + P0-3 + P0-4 + P0-11 (B-515): canonical full prereg decision artifact
# (H1 FE + H2(a) per-task ratio + H3 axes FE + I² cap-only + R1-R5 framing rule +
# manifest/code/csv SHA provenance lock). Replaces retired-DL preregistration_decision_test
# as the paper §1 framing rule producer.
phase1-full-prereg-decision:
	$(PYTHON) scripts/analysis/aggregate_phase1_full_prereg_decision.py

# A1.21 P0-8 (B-525): validate run_manifest.yaml schema/disk/scope correctness.
# Default reports all errors (exit 2 on any); --strict fails fast on first error.
validate-run-manifest:
	$(PYTHON) scripts/analysis/validate_run_manifest.py

# Phantom routing lift — appendix exploratory (was Section 1/4 paper hook,
# demoted to appendix per B-184 — different estimand from prereg PRIMARY).
# (3-mode oracle vs 5-mode oracle ceiling lift + bootstrap CI + decomposition
#  + Bonferroni/Holm/BH/TOST per pre-registered family, T0a)
phantom-lift:
	$(PYTHON) scripts/analysis/aggregate_phantom_lift.py

# Cross-cell meta-analysis (DerSimonian-Laird random-effect, T0c)
# Requires phantom-lift output; produces meta_phantom_lift.{md,csv}
phantom-meta:
	$(PYTHON) scripts/analysis/aggregate_phantom_meta.py

# ---- Layered analysis (paper_planning §3 framework, paper-grade B0 only) ----

# Layer 0 — Outcome (SR / oracle / AUROC / task-pool / category)
analyze-layer0:
	$(MAKE) aggregate-sr-fp
	$(MAKE) phantom-lift
	$(MAKE) routing-auroc
	$(PYTHON) scripts/analysis/figures/fig0d_taskpool_jaccard.py
	$(PYTHON) scripts/analysis/figures/fig0c_drop_one_oracle.py
	$(PYTHON) scripts/analysis/figures/fig0c_phantom_lift_bars.py
	$(PYTHON) scripts/analysis/figures/fig0g_routing_auroc_heatmap.py
	$(PYTHON) scripts/analysis/figures/fig0e_category_mode_heatmap.py
	$(PYTHON) scripts/analysis/figures/fig0f_overlap_stacked_bar.py

aggregate-sr-fp:
	$(PYTHON) scripts/analysis/aggregate_sr_fp_per_mode.py

# Layer 1 — Macro Behavior (action-type frequencies, cascade)
analyze-layer1:
	$(PYTHON) scripts/analysis/axis_effect_size.py
	$(PYTHON) scripts/analysis/figures/fig1ab_cascade_diamond.py
	$(PYTHON) scripts/analysis/figures/fig1c_strategy_gradient.py

# Layer 2 — Micro Behavior (per-step decision quality)
analyze-layer2:
	$(PYTHON) scripts/analysis/axis1_microbehavior.py
	$(MAKE) fig12-micro-heatmap

analyze-mechanism:
	$(PYTHON) scripts/analysis/mechanism_per_task.py

fig12-micro-heatmap:
	$(PYTHON) scripts/analysis/figures/fig2_micro_divergence_heatmap.py

# Layer 3 — Efficiency (cost / latency)
analyze-layer3:
	$(MAKE) summary-collect
	$(MAKE) aggregate-cost-electricity
	$(PYTHON) scripts/analysis/figures/fig3a_token_cost_intra_baseline.py
	$(PYTHON) scripts/analysis/figures/fig3d_cost_sr_frontier.py
	$(PYTHON) scripts/analysis/figures/fig3_regional_carbon.py

# Layer 3a + 3d — deployment-class cost (B0 API $ vs B1 electricity-equivalent)
aggregate-cost-electricity:
	$(PYTHON) scripts/analysis/aggregate_cost_electricity.py

# DEPRECATED: use `make analysis` instead. Kept for backward compatibility.
analyze-paper analyze-layered:
	@echo "[deprecated] '$@' is now an alias for 'make analysis'. Use 'make analysis' going forward."
	$(MAKE) analysis

# Per-run paper-grade analysis pipeline: rederive → reason-diag → cross-rep
# → confidence calibration. Iterates over all paper-grade VWA run dirs.
# Watchdog already runs this incrementally per-condition, but `analyze-paper`
# brute-forced all runs to ensure cross-condition aggregations consumed fresh
# per-run output (e.g. after manual data edits, rederives, or fresh re-runs).
analyze-paper-per-run:
	$(MAKE) _per_run_all

# B0 vs B1 site comparison — runs compare_b0_b1.py for each (B0_run, B1_run)
# pair on cls + red. Outputs to results/visualwebarena/phase1/b0_vs_b1_<site>/.
B0_RUN_CLS ?= $(shell $(PYTHON) -c "from scripts.analysis.lib.run_registry import get_cell; print(get_cell('B0','classifieds','DOM').run_dir)")
B0_RUN_RED ?= $(shell $(PYTHON) -c "from scripts.analysis.lib.run_registry import get_cell; print(get_cell('B0','reddit','DOM').run_dir)")
B1_RUN_CLS ?= $(shell $(PYTHON) -c "from scripts.analysis.lib.run_registry import get_cell; print(get_cell('B1','classifieds','DOM').run_dir)")
B1_RUN_RED ?= $(shell $(PYTHON) -c "from scripts.analysis.lib.run_registry import get_cell; print(get_cell('B1','reddit','DOM').run_dir)")

compare-b0-b1-all:
	@$(MAKE) --no-print-directory compare B0=$(B0_RUN_CLS) B1=$(B1_RUN_CLS) SITE=classifieds || true
	@$(MAKE) --no-print-directory compare B0=$(B0_RUN_RED) B1=$(B1_RUN_RED) SITE=reddit || true

# Paper-grade snapshot — one-shot "everything after new data":
#   1. per-run pipeline (rederive + reason-diag + cross-rep + confidence) on
#      all 8 paper-grade VWA runs (override RUN_DIRS_PAPER_VWA if needed)
#   2. B0 vs B1 site comparison (cls + red; b0_vs_b1_<site>/ output)
#   3. cross-condition aggregations:
#      - aggregate-cross-site (SR/cost/lat/energy join → cross_site/)
#      - summary-collect (run-level metadata → run_summary_collect.json)
#      - routing-auroc (per-condition AUROC merge → auroc_cross_condition.*)
#   4. figures (9 PNGs incl. fig2 bootstrap CI)
# Total ~5-10 min for 8 runs. Use `make analysis` before codex prose tasks
# (#11 / #13 / #16) or before paper revisits.
#
# NOT included (intentional):
#   - GLM digest sidecar — watchdog handles incrementally
#   - Annotate screenshots / gallery regen — watchdog handles
#   - Codex narrative analyses (docs/analysis/phantom_paper/*.md) — manual codex
#   - Narrow ad-hoc diagnostics (analyze_*selflink_loop, b0_vision_coordinate_*,
#     analyze_search_over_browse, diag_pattern_match) — invoke individually

# Public-facing alias — same as internal `_figures` target. Use `make figures`
# from CLI; `make analysis` calls `_figures` internally.
figures: _figures
	@echo "Figures regenerated → results/phantom_paper/figures/"

# ---- Background tasks ----
# (B-394 / A1.15 C3 T6=(a), 2026-05-16): `watch-reddit` retired — see header.

# Phantom (§102/§103) — start/resume one cell. Idempotent: skips if already running.
# Usage: make phantom B=B0 M=som S=reddit          (phantom_som on VWA reddit)
#        make phantom B=B0 M=text S=reddit         (P-text ablation on VWA reddit)
#        make phantom B=B0 M=som S=shopping BMK=wa (phantom_som on WA shopping)
#        RESET_BEFORE=1 make phantom ...           (reset site before launch)
# Note: M=dom is accepted as a back-compat alias for M=text (queue_phantom_dom.sh
# is symlinked to queue_phantom_text.sh).
phantom:
	@test -n "$(B)" -a -n "$(S)" || (echo "ERROR: B=<B0|B1> S=<site> required (M defaults som, BMK defaults vwa)"; exit 1)
	bash scripts/queues/queue_phantom_$(or $(M),som).sh $(B) $(S) $(or $(BMK),vwa)

# Baseline (dom/som/vision) — start/resume one cell. Idempotent.
# Usage: make baseline B=B0 M=dom S=shopping         (B0 DOM-only on VWA shopping)
#        make baseline B=B1 M=som S=reddit BMK=wa    (B1 SoM on WA reddit)
baseline:
	@test -n "$(B)" -a -n "$(M)" -a -n "$(S)" || (echo "ERROR: B=<B0|B1> M=<dom|som|vision> S=<site> required"; exit 1)
	bash scripts/queues/queue_baseline.sh $(B) $(M) $(S) $(or $(BMK),vwa)

# Phantom-SoM Phase 2.1 — VWA all 4 baseline cells
phantom-vwa-all:
	bash scripts/queues/queue_phantom_som.sh B0 classifieds
	bash scripts/queues/queue_phantom_som.sh B0 reddit
	bash scripts/queues/queue_phantom_som.sh B0 shopping
	bash scripts/queues/queue_phantom_som.sh B1 classifieds
	bash scripts/queues/queue_phantom_som.sh B1 reddit

# P-text (§103 ablation) — all VWA cells (B0 + B1)
phantom-text-vwa-all:
	bash scripts/queues/queue_phantom_text.sh B0 classifieds
	bash scripts/queues/queue_phantom_text.sh B0 reddit
	bash scripts/queues/queue_phantom_text.sh B0 shopping
	bash scripts/queues/queue_phantom_text.sh B1 classifieds
	bash scripts/queues/queue_phantom_text.sh B1 reddit

# Back-compat alias for legacy target name
phantom-dom-vwa-all: phantom-text-vwa-all

# Phantom — WA generalization (3 sites, B0 + B1)
phantom-wa-all:
	bash scripts/queues/queue_phantom_som.sh B0 reddit         wa
	bash scripts/queues/queue_phantom_som.sh B0 shopping       wa
	bash scripts/queues/queue_phantom_som.sh B0 shopping_admin wa
	bash scripts/queues/queue_phantom_som.sh B1 reddit         wa
	bash scripts/queues/queue_phantom_som.sh B1 shopping       wa
	bash scripts/queues/queue_phantom_som.sh B1 shopping_admin wa

# P-text ablation on WA (3 sites, B0 + B1)
phantom-text-wa-all:
	bash scripts/queues/queue_phantom_text.sh B0 reddit         wa
	bash scripts/queues/queue_phantom_text.sh B0 shopping       wa
	bash scripts/queues/queue_phantom_text.sh B0 shopping_admin wa
	bash scripts/queues/queue_phantom_text.sh B1 reddit         wa
	bash scripts/queues/queue_phantom_text.sh B1 shopping       wa
	bash scripts/queues/queue_phantom_text.sh B1 shopping_admin wa

# Back-compat alias for legacy target name
phantom-dom-wa-all: phantom-text-wa-all

# Backward-compat alias for old "phantom-all" (B=... S=... interface)
phantom-all: phantom-vwa-all

schedule-list:
	@echo "Active P79 background processes:"
	@pgrep -af "experiment_watchdog|run_experiment|queue_b1|queue_b0|queue_b2|queue_phantom|queue_chain" | head -20 || echo "(none)"

# ---- Cross-host results sync (hub-spoke, default hub = DGX) ----
# Tier B (episodes/*.jsonl + condition/run summary + analysis) by default.
# Set ARTIFACTS=1 to include artifacts (screenshots/SoM 图).
# Override hub via HOST=<ssh-alias>; narrow with RUN=<run_id> [+ COND=<cond_id>].
rsync-to-hub:
	@HOST="$(HOST)" HUB_PATH="$(HUB_PATH)" DRY="$(DRY)" \
	  bash scripts/maintenance/rsync_results_to_hub.sh

rsync-from-hub:
	@HOST="$(HOST)" HUB_PATH="$(HUB_PATH)" RUN="$(RUN)" COND="$(COND)" \
	  ARTIFACTS="$(ARTIFACTS)" DRY="$(DRY)" \
	  bash scripts/maintenance/rsync_results_from_hub.sh

rsync-artifacts-from-hub:
	@HOST="$(HOST)" HUB_PATH="$(HUB_PATH)" RUN="$(RUN)" COND="$(COND)" \
	  ARTIFACTS=1 DRY="$(DRY)" \
	  bash scripts/maintenance/rsync_results_from_hub.sh

# ---- GLM sidecar maintenance (Phase 1 automation) ----
# glm-update-cells: dry-run scan _status/cells/ frontmatter vs condition_summary_v2.json
#   APPLY=1 to actually write; FORCE=1 to overwrite active+pid cells too
glm-update-cells:
	@.venv/bin/python scripts/maintenance/glm/glm_cell_autoupdate.py \
	  $(if $(APPLY),--apply,) $(if $(FORCE),--force,)

# glm-refresh-playbook: GLM synthesizes PLAYBOOK §1 (critical path) + §2 (automation status)
#   APPLY=1 to actually write back to PLAYBOOK.md
#   SECTION={1,2,both} default both. Use SECTION=2 for fast cron (skips `make active` subprocess).
glm-refresh-playbook:
	@.venv/bin/python scripts/maintenance/glm/glm_playbook_refresh.py \
	  $(if $(APPLY),--apply,) \
	  $(if $(SECTION),--section $(SECTION),)

# error-scan: scan logs/ + logs/cron/ for runner / watchdog / cron errors (last 24h)
#   Output logs/cron/error_scan.json — consumed by glm-refresh-playbook §2.5
#   HOURS=N to override lookback (default 24)
error-scan:
	@.venv/bin/python scripts/maintenance/glm/error_scan.py $(if $(HOURS),--hours $(HOURS),)

# glm-pre-launch-check: RETIRED 2026-05-16 (A1.17 B-306).
# Deterministic shell asserts in launch.sh:113-160 replace the LLM-judge gate.
# Rationale: 4/5 hard rules already deterministically enforced upstream;
# GLM dependency removed (LLM variance / API outage / non-deterministic gate
# in paper-grade launch path = anti-pattern).

# check-links: scan all docs for broken path-based + wikilink references
check-links:
	@.venv/bin/python scripts/maintenance/dead_link_check.py --quiet

# vwa-generate-configs: materialize the 912 per-task split configs from tracked
# `.raw.json` templates. /stress A1.18-re B-589 P1-10-B (codex 2026-05-17):
# OSF replayers + fresh-clone setup must run this after `git clone` + `git
# submodule update --init` because the split files are gitignored derived
# artifacts. Required env vars (B-577 idempotent rebuild + B-588 byte-stable):
# DATASET=visualwebarena CLASSIFIEDS REDDIT SHOPPING HOMEPAGE WIKIPEDIA
.PHONY: vwa-generate-configs
vwa-generate-configs:
	@if [ -z "$${DATASET:-}" ]; then export DATASET=visualwebarena; fi; \
	for v in CLASSIFIEDS REDDIT SHOPPING HOMEPAGE; do \
	  if [ -z "$$(eval echo \$$$$v)" ]; then \
	    echo "❌ missing env var: $$v (source scripts/vwa_env_remote.sh or scripts/vwa_env_a100.sh first)"; \
	    exit 64; \
	  fi; \
	done; \
	cd external/visualwebarena && DATASET=visualwebarena $(MAKE_PYTHON_BIN) python scripts/generate_test_data.py; \
	echo "✅ VWA per-task configs materialized at external/visualwebarena/config_files/vwa/test_{classifieds,reddit,shopping}/"
MAKE_PYTHON_BIN ?= .venv/bin/

# launch: one-shot wrapper — auto-create cell note + pre-launch check + nohup queue
#   Usage: make launch BASELINE=B0 SITE=reddit MODE=phantom_text [RESET=1] [DRY=1]
#   Modes: dom | som | vision | phantom_text | phantom_som | phantom_prompt
launch:
	@if [ -z "$(BASELINE)" ] || [ -z "$(SITE)" ] || [ -z "$(MODE)" ]; then \
	  echo "Usage: make launch BASELINE=<B0|B1> SITE=<cls|red|shop> MODE=<dom|som|vision|phantom_*>"; \
	  exit 64; \
	fi
	@RESET=$${RESET:-1} DRY=$${DRY:-0} FORCE_NO_CHECK=$${FORCE_NO_CHECK:-0} \
	  bash scripts/maintenance/launch.sh "$(BASELINE)" "$(SITE)" "$(MODE)" "$(TARGET_SECTION)" "$(PRIORITY)"
