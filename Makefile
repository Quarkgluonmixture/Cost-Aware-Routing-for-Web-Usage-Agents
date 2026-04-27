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
        validate gallery

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

# Regenerate paper figures (7 PNGs in results/phantom_paper/figures/)
figures:
	$(PYTHON) scripts/analysis/figures/fig1_4mode_venn.py
	$(PYTHON) scripts/analysis/figures/fig2_drop_one_oracle.py
	$(PYTHON) scripts/analysis/figures/fig3_strategy_gradient.py
	$(PYTHON) scripts/analysis/figures/fig4_two_knob_diagram.py
	$(PYTHON) scripts/analysis/figures/fig5_category_mode_heatmap.py
	$(PYTHON) scripts/analysis/figures/fig6_capability_contrast.py
	$(PYTHON) scripts/analysis/figures/fig7_cost_sr_frontier.py
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
phantom:
	@test -n "$(B)" -a -n "$(S)" || (echo "ERROR: B=<B0|B1> S=<site> required (M defaults som, BMK defaults vwa)"; exit 1)
	bash scripts/queues/queue_phantom.sh $(B) $(or $(M),som) $(S) $(or $(BMK),vwa)

# Phantom-SoM Phase 2.1 — VWA all 4 baseline cells
phantom-vwa-all:
	bash scripts/queues/queue_phantom.sh B0 som classifieds
	bash scripts/queues/queue_phantom.sh B0 som reddit
	bash scripts/queues/queue_phantom.sh B0 som shopping
	bash scripts/queues/queue_phantom.sh B1 som classifieds
	bash scripts/queues/queue_phantom.sh B1 som reddit

# Phantom-DOM (§103 ablation) — all VWA cells (B0 + B1)
phantom-dom-vwa-all:
	bash scripts/queues/queue_phantom.sh B0 dom classifieds
	bash scripts/queues/queue_phantom.sh B0 dom reddit
	bash scripts/queues/queue_phantom.sh B0 dom shopping
	bash scripts/queues/queue_phantom.sh B1 dom classifieds
	bash scripts/queues/queue_phantom.sh B1 dom reddit

# Phantom — WA generalization (3 sites, B0 + B1)
phantom-wa-all:
	bash scripts/queues/queue_phantom.sh B0 som reddit         wa
	bash scripts/queues/queue_phantom.sh B0 som shopping       wa
	bash scripts/queues/queue_phantom.sh B0 som shopping_admin wa
	bash scripts/queues/queue_phantom.sh B1 som reddit         wa
	bash scripts/queues/queue_phantom.sh B1 som shopping       wa
	bash scripts/queues/queue_phantom.sh B1 som shopping_admin wa

# Phantom-DOM ablation on WA (3 sites, B0 + B1)
phantom-dom-wa-all:
	bash scripts/queues/queue_phantom.sh B0 dom reddit         wa
	bash scripts/queues/queue_phantom.sh B0 dom shopping       wa
	bash scripts/queues/queue_phantom.sh B0 dom shopping_admin wa
	bash scripts/queues/queue_phantom.sh B1 dom reddit         wa
	bash scripts/queues/queue_phantom.sh B1 dom shopping       wa
	bash scripts/queues/queue_phantom.sh B1 dom shopping_admin wa

# Backward-compat alias for old "phantom-all" (B=... S=... interface)
phantom-all: phantom-vwa-all

schedule-list:
	@echo "Active P79 background processes:"
	@pgrep -af "wait_for_reddit|experiment_watchdog|run_experiment|queue_b1|queue_b0|queue_phantom" | head -20 || echo "(none)"
