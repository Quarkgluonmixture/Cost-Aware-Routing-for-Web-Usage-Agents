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
        analyze-paper-per-run

help:
	@echo "P79 Makefile — see header for usage examples"
	@grep -E '^[a-z-]+:' Makefile | grep -v '^.PHONY' | sed 's/:.*//' | sort | sed 's/^/  /'

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

# Paper-grade snapshot — one-shot "everything after new data":
#   1. per-run pipeline (rederive + reason-diag + cross-rep + confidence) on
#      all 8 paper-grade VWA runs (override RUN_DIRS_PAPER_VWA if needed)
#   2. cross-condition aggregations (cross-site SR/cost/lat/energy + run
#      summary + routing AUROC merge)
#   3. figures (9 PNGs incl. fig2 bootstrap CI)
# Total ~5-10 min for 8 runs. Run before codex prose tasks (#11 / #13 / #16)
# or before paper revisits.
analyze-paper: analyze-paper-per-run aggregate-cross-site summary-collect routing-auroc figures
	@echo ""
	@echo "[analyze-paper] outputs in results/phantom_paper/:"
	@ls results/phantom_paper/*.csv results/phantom_paper/*.md 2>/dev/null || true
	@ls results/phantom_paper/cross_site/*.csv 2>/dev/null || true
	@echo "[analyze-paper] figures in results/phantom_paper/figures/:"
	@ls results/phantom_paper/figures/*.png 2>/dev/null || true

# Regenerate paper figures (9 PNGs in results/phantom_paper/figures/)
figures:
	$(PYTHON) scripts/analysis/figures/fig1_4mode_venn.py
	$(PYTHON) scripts/analysis/figures/fig2_drop_one_oracle.py
	$(PYTHON) scripts/analysis/figures/fig3_strategy_gradient.py
	$(PYTHON) scripts/analysis/figures/fig4_two_knob_diagram.py
	$(PYTHON) scripts/analysis/figures/fig5_category_mode_heatmap.py
	$(PYTHON) scripts/analysis/figures/fig6_capability_contrast.py
	$(PYTHON) scripts/analysis/figures/fig7_cost_sr_frontier.py
	$(PYTHON) scripts/analysis/figures/fig8_overlap_stacked_bar.py
	$(PYTHON) scripts/analysis/figures/fig9_regional_carbon_sensitivity.py
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
