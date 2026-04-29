# P79 Makefile — daily commands one-liners
#
# Usage examples:
#   make test                                    # full pytest suite
#   make smoke                                   # just smoke test (fast)
#   make rederive RUN=results/visualwebarena/phase1/B0_3mode_classifieds_20260413
#   make analyze RUN=...                         # rederive → reason_diag → cross_rep → analyze_run → confidence
#   make compare B0=<b0_run> B1=<b1_run> SITE=classifieds
#   make clean-tasks RUN=<run> COND=phase1_som_router_0 SITE=shopping TASKS=0-465
#   make watch-reddit                            # launch the reddit-then-rederive watch script
#   make schedule-list                           # list active background watch processes
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

.PHONY: help test smoke smoke-only rederive rederive-all analyze cross-rep \
        confidence compare reason-diag clean-tasks watch-reddit schedule-list \
        validate gallery rsync-to-hub rsync-from-hub rsync-artifacts-from-hub \
        aggregate-cross-site summary-collect routing-auroc analyze-paper \
        analyze-paper-per-run compare-b0-b1-all phantom-lift \
        analyze-layer0 analyze-layer1 analyze-layer2 analyze-layer3 analyze-layered \
        aggregate-sr-fp fig12-micro-heatmap aggregate-cost-electricity

help:
	@echo "P79 Makefile — see header for usage examples"
	@echo "Layered analysis: make analyze-layered"
	@echo "Layer 0 SR/FP: make aggregate-sr-fp"
	@echo "Layer 2 figure: make fig12-micro-heatmap"
	@grep -E '^[a-z0-9-]+:' Makefile | grep -v '^.PHONY' | sed 's/:.*//' | sort | sed 's/^/  /'

# ---- Tests ----
test:
	$(PYTEST) tests/ -x -q

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

gallery:
	@test -n "$(RUN)" || (echo "ERROR: RUN=<run_dir> required"; exit 1)
	$(PYTHON) scripts/maintenance/generate_gallery.py --run-dir $(RUN)

# ---- Cross-run / paper-grade aggregation (cross-condition) ----
# Run once after a batch of conditions is paper-grade clean.
# Outputs land under results/phantom_paper/ for paper drafts to consume.

# Cross-site SR/cost/lat/energy aggregator (per benchmark).
# Default RUN_DIRS = paper-grade clean B0+B1 cls+red runs (override via RUN_DIRS=...).
RUN_DIRS_PAPER_VWA ?= \
  results/visualwebarena/phase1/B0_3mode_classifieds_20260413 \
  results/visualwebarena/phase1/B0_3mode_reddit_20260422 \
  results/visualwebarena/phase1/B0_phantom_classifieds_20260426 \
  results/visualwebarena/phase1/B0_phantom_reddit_20260428 \
  results/visualwebarena/phase1/B0_phantom_dom_classifieds_20260427 \
  results/visualwebarena/phase1/B0_phantom_dom_reddit_20260427 \
  results/visualwebarena/phase1/B1_3mode_classifieds_20260413 \
  results/visualwebarena/phase1/B1_3mode_reddit_20260413

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

# Phantom routing lift — Section 1/4 paper hook evidence
# (3-mode oracle vs 5-mode oracle ceiling lift + bootstrap CI + decomposition)
phantom-lift:
	$(PYTHON) scripts/analysis/aggregate_phantom_lift.py

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

fig12-micro-heatmap:
	$(PYTHON) scripts/analysis/figures/fig2_micro_divergence_heatmap.py

# Layer 3 — Efficiency (cost / latency)
analyze-layer3:
	$(MAKE) summary-collect
	$(MAKE) aggregate-cost-electricity
	$(PYTHON) scripts/analysis/figures/fig3d_cost_sr_frontier.py
	$(PYTHON) scripts/analysis/figures/fig3_regional_carbon.py

# Layer 3a + 3d — deployment-class cost (B0 API $ vs B1 electricity-equivalent)
aggregate-cost-electricity:
	$(PYTHON) scripts/analysis/aggregate_cost_electricity.py

# Run all 4 layers
analyze-layered: analyze-layer0 analyze-layer1 analyze-layer2 analyze-layer3
	$(PYTHON) scripts/analysis/figures/fig_capability_b0_b1.py
	$(PYTHON) scripts/analysis/layered_status.py

# Per-run paper-grade analysis pipeline: rederive → reason-diag → cross-rep
# → confidence calibration. Iterates over all paper-grade VWA run dirs.
# Watchdog already runs this incrementally per-condition, but `analyze-paper`
# brute-forces all runs to ensure cross-condition aggregations consume fresh
# per-run output (e.g. after manual data edits, rederives, or fresh re-runs).
analyze-paper-per-run:
	@for rd in $(RUN_DIRS_PAPER_VWA); do \
	  echo ""; \
	  echo "=== [analyze-paper-per-run] $$rd ==="; \
	  $(MAKE) --no-print-directory analyze RUN=$$rd || echo "  [warn] analyze failed for $$rd"; \
	  $(MAKE) --no-print-directory confidence RUN=$$rd || echo "  [warn] confidence failed for $$rd"; \
	done

# B0 vs B1 site comparison — runs compare_b0_b1.py for each (B0_run, B1_run)
# pair on cls + red. Outputs to results/visualwebarena/phase1/b0_vs_b1_<site>/.
B0_RUN_CLS ?= results/visualwebarena/phase1/B0_3mode_classifieds_20260413
B0_RUN_RED ?= results/visualwebarena/phase1/B0_3mode_reddit_20260422
B1_RUN_CLS ?= results/visualwebarena/phase1/B1_3mode_classifieds_20260413
B1_RUN_RED ?= results/visualwebarena/phase1/B1_3mode_reddit_20260413

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
# Total ~5-10 min for 8 runs. Run before codex prose tasks (#11 / #13 / #16)
# or before paper revisits.
#
# NOT included (intentional):
#   - GLM digest sidecar — watchdog handles incrementally
#   - Annotate screenshots / gallery regen — watchdog handles
#   - Codex narrative analyses (docs/analysis/phantom_paper/*.md) — manual codex
#   - Narrow ad-hoc diagnostics (analyze_*selflink_loop, b0_vision_coordinate_*,
#     analyze_search_over_browse, diag_pattern_match) — invoke individually
analyze-paper: analyze-paper-per-run compare-b0-b1-all aggregate-cross-site summary-collect routing-auroc phantom-lift figures
	@echo ""
	@echo "[analyze-paper] cross-condition outputs in results/phantom_paper/:"
	@ls results/phantom_paper/*.csv results/phantom_paper/*.md 2>/dev/null || true
	@ls results/phantom_paper/cross_site/*.csv 2>/dev/null || true
	@echo "[analyze-paper] B0 vs B1 site comparisons:"
	@ls -d results/visualwebarena/phase1/b0_vs_b1_* 2>/dev/null || true
	@echo "[analyze-paper] figures in results/phantom_paper/figures/:"
	@ls results/phantom_paper/figures/*.png 2>/dev/null || true

# Regenerate paper figures (12 PNGs in results/phantom_paper/figures/).
# fig0c_phantom_lift_bars/fig0g_routing_auroc_heatmap depend on phantom_lift.csv / auroc_cross_condition.csv —
# automatically regenerated upstream by `make analyze-paper`.
figures:
	$(PYTHON) scripts/analysis/figures/fig0d_taskpool_jaccard.py
	$(PYTHON) scripts/analysis/figures/fig0c_drop_one_oracle.py
	$(PYTHON) scripts/analysis/figures/fig1c_strategy_gradient.py
	$(PYTHON) scripts/analysis/figures/fig1ab_cascade_diamond.py
	$(PYTHON) scripts/analysis/figures/fig0e_category_mode_heatmap.py
	$(PYTHON) scripts/analysis/figures/fig_capability_b0_b1.py
	$(PYTHON) scripts/analysis/figures/fig3d_cost_sr_frontier.py
	$(PYTHON) scripts/analysis/figures/fig0f_overlap_stacked_bar.py
	$(PYTHON) scripts/analysis/figures/fig3_regional_carbon.py
	$(PYTHON) scripts/analysis/figures/fig0c_phantom_lift_bars.py
	$(PYTHON) scripts/analysis/figures/fig0g_routing_auroc_heatmap.py
	$(PYTHON) scripts/analysis/figures/fig2_micro_divergence_heatmap.py
	@echo "Figures regenerated → results/phantom_paper/figures/"

# ---- Background tasks ----
watch-reddit:
	setsid nohup bash scripts/maintenance/wait_for_reddit_then_rederive.sh \
	    > logs/wait_reddit_followup.log 2>&1 < /dev/null &
	@sleep 1 && pgrep -af wait_for_reddit_then_rederive | head -1 || echo "(watch may have exited)"

# Phantom (§102/§103) — start/resume one cell. Idempotent: skips if already running.
# Usage: make phantom B=B0 M=som S=reddit          (phantom_som on VWA reddit)
#        make phantom B=B0 M=dom S=reddit          (phantom_dom ablation on VWA reddit)
#        make phantom B=B0 M=som S=shopping BMK=wa (phantom_som on WA shopping)
#        RESET_BEFORE=1 make phantom ...           (reset site before launch)
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

# Phantom-DOM (§103 ablation) — all VWA cells (B0 + B1)
phantom-dom-vwa-all:
	bash scripts/queues/queue_phantom_dom.sh B0 classifieds
	bash scripts/queues/queue_phantom_dom.sh B0 reddit
	bash scripts/queues/queue_phantom_dom.sh B0 shopping
	bash scripts/queues/queue_phantom_dom.sh B1 classifieds
	bash scripts/queues/queue_phantom_dom.sh B1 reddit

# Phantom — WA generalization (3 sites, B0 + B1)
phantom-wa-all:
	bash scripts/queues/queue_phantom_som.sh B0 reddit         wa
	bash scripts/queues/queue_phantom_som.sh B0 shopping       wa
	bash scripts/queues/queue_phantom_som.sh B0 shopping_admin wa
	bash scripts/queues/queue_phantom_som.sh B1 reddit         wa
	bash scripts/queues/queue_phantom_som.sh B1 shopping       wa
	bash scripts/queues/queue_phantom_som.sh B1 shopping_admin wa

# Phantom-DOM ablation on WA (3 sites, B0 + B1)
phantom-dom-wa-all:
	bash scripts/queues/queue_phantom_dom.sh B0 reddit         wa
	bash scripts/queues/queue_phantom_dom.sh B0 shopping       wa
	bash scripts/queues/queue_phantom_dom.sh B0 shopping_admin wa
	bash scripts/queues/queue_phantom_dom.sh B1 reddit         wa
	bash scripts/queues/queue_phantom_dom.sh B1 shopping       wa
	bash scripts/queues/queue_phantom_dom.sh B1 shopping_admin wa

# Backward-compat alias for old "phantom-all" (B=... S=... interface)
phantom-all: phantom-vwa-all

schedule-list:
	@echo "Active P79 background processes:"
	@pgrep -af "wait_for_reddit|experiment_watchdog|run_experiment|queue_b1|queue_b0|queue_phantom" | head -20 || echo "(none)"

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
