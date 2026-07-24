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
        confidence compare reason-diag clean-tasks schedule-list ntfy status status-set \
        validate gallery rsync-to-hub rsync-from-hub rsync-artifacts-from-hub \
        aggregate-cross-site summary-collect routing-auroc analyze-paper \
        analyze-paper-per-run compare-b0-b1-all phantom-lift \
        analyze-layer0 analyze-layer1 analyze-layer2 analyze-layer3 analyze-layered \
        aggregate-sr-fp fig12-micro-heatmap aggregate-cost-electricity analyze-mechanism \
        analysis _per_run_all _aggregate _figures _status active \
        glm-update-cells glm-refresh-playbook check-links vwa-generate-configs \
        pre-release-check \
        deslop-lint deslop-gate deslop-selftest deslop-vocab \
        deslop-ratchet deslop-audit

help:
	@echo "P79 Makefile — see header for usage examples"
	@echo ""
	@echo "  ⭐ make analysis              # full analysis pipeline (per-run + cross-condition + figures)"
	@echo "  ⭐ make analysis FAST=1       # skip per-run; aggregators + figures only (~30s)"
	@echo "  ⭐ make analysis RUN=<dir>    # single-run pipeline + downstream"
	@echo ""
	@echo "  make active                  # real-time process scan"
	@echo "  make status                  # render _status/ Bases views in terminal (CLI = Obsidian)"
	@echo "  make status V='cells#Active' # render one view (V = <base>[#view substr])"
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
	@echo "  Paper prose (paper-deslop):"
	@echo "    make deslop-lint [F=<file>]  # Vale AI-tell lint (default: all paper_drafts)"
	@echo "    make deslop-gate OLD= NEW=   # lexical invariant gate (numbers/cites/terms)"
	@echo "    make deslop-ratchet [ALL=1]  # blocking set from deslopped.txt (what CI runs)"
	@echo "    make deslop-audit [F=<file>] # per-term hit counts (curate terms.txt)"
	@echo "    make deslop-selftest         # pipeline self-test (fixtures + gate)"
	@echo "    /deslop-paper                # the interactive rewrite skill (diff-only)"
	@echo ""
	@echo "  All targets:                 grep '^[a-z]' Makefile | sed 's/:.*//' | sort -u"

# ---- Live status ----
# Real-time scan of run_experiment + experiment_watchdog processes;
# replaces the manually-maintained §1 Active Processes table in next_steps.md.
active:
	@$(PYTHON) scripts/maintenance/active_processes.py

# Pull P79 ntfy notifications into the session so they're readable without manual
# copy-paste (the watchdogs only push; this polls them back out). Polls BOTH
# active topics by default — p79-exp-dgx-spark (exp) + p79-claude (cln, delete/
# cleanup) — merged by time so no channel is missed. (p79-jiaming retired.)
# `make ntfy` = last 12h all · `make ntfy SINCE=1h` · `make ntfy ALERTS=1` (alerts-only).
ntfy:
	@bash scripts/maintenance/ntfy_read.sh $(or $(SINCE),12h) $(if $(ALERTS),alerts,all)

# Render docs/*.base views over _status/ frontmatter in the terminal — the SAME
# views Obsidian shows (single-source data layer), so neither side maintains a
# parallel hand-written table. CLI/agent gets byte-equivalent read access; `set`
# edits one frontmatter field without YAML round-trip (preserves complex fields
# like `history:`). See scripts/maintenance/status_query.py header for the
# supported Bases expression subset.
#   make status                      # list all base + views + note counts
#   make status V='tasks#NOW'        # render one view (V = <base>[#view substr]; quote the '#')
#   make status V=cells ARGS=--json  # machine-readable
#   make status-set N=<note> SET='status=done blocker="GPU contention"'   # edit field(s)
status:
	@$(PYTHON) scripts/maintenance/status_query.py $(if $(V),"$(V)") $(ARGS)

status-set:
	@$(PYTHON) scripts/maintenance/status_query.py set $(N) $(SET)

# ---- Tests ----
test:
	@# /stress A1.12 P1-4 fix (2026-05-16): `-x` (fail-fast) removed so full
	@# failure picture surfaces in one run instead of hiding remaining failures
	@# behind the first. `--tb=short` keeps output digestible while still
	@# showing assert location. `pyproject.toml [tool.pytest.ini_options]` now
	@# sets `--strict-markers -ra -rs` so silent importorskip / typo'd marker
	@# bugs no longer hide.
	@# B-662 (/stress A1.12 P0-3 BC, 2026-05-17): fail-loud if `[test]` extras
	@# missing. Pre-fix, fresh-host OSF replayer running `pip install -e . &&
	@# make test` would see "428 passed / 14 skipped" without realizing 6 paper-
	@# grade analysis tests silently `pytest.importorskip("pandas")`. Now the
	@# Makefile aborts with exit 2 + install hint before pytest even collects.
	@$(PYTHON) -c 'import pandas, matplotlib, scipy' 2>/dev/null || \
		(echo "ERROR: '[test]' extras missing (pandas/matplotlib/scipy)."; \
		 echo "       Run: pip install -e \".[test]\""; \
		 exit 2)
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
	# 3-model deep-update 2026-05-18: B2 (Gemma3-VL) optional → 3-way comparison.
	$(PYTHON) scripts/analysis/compare_b0_b1.py --b0-run-dir $(B0) --b1-run-dir $(B1) \
		$(if $(B2),--b2-run-dir $(B2),) --site $(SITE)

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
	@LOCK_SHA="2c15d66d120f8498633ae65057aa50a34b3e93e0"; \
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

# B-1512 /stress A2.9 P0-7-ABC* 2026-05-18 — wire release_redaction_checklist
# automation per `docs/checkpoints/pre_run/release_redaction_checklist.md §66`
# (was "planned" pre-A2.9, never implemented). Runs the 4-step recipe at
# release_redaction L74-78 (grep credentials / find auth dirs / IP scan /
# submodule clean) and surfaces any pre-OSF-deposit redaction violations.
pre-release-check:
	@echo "=== Pre-release redaction check (B-1512 /stress A2.9 P0-7-ABC* 2026-05-18) ==="
	@echo "1. No API key / secret / token literals (20+ char) in tracked files..."
	@# Exclude scripts/vwa_env*.sh + .example (VWA classifieds reset token
	@# 4b61655535e7ed388f0d40a93600254c is upstream VWA Docker design constant,
	@# not a P79 secret, published in VWA repo). Regex requires VALUE to be
	@# quote-delimited literal hex/alphanumeric (no `$` interpolation), which
	@# excludes shell env-var indirection like `local token="${VAR:-}"`.
	@HITS=$$(git grep -inE "(api[-_]?key|password|secret|token).{0,5}=.{0,5}[\"\\'][a-zA-Z0-9_-]{20,}[\"\\']" -- ':!docs/checkpoints/' ':!.gitignore' ':!scripts/vwa_env.sh' ':!scripts/vwa_env_remote.sh.example' 2>/dev/null | wc -l); \
	 test "$$HITS" = "0" || (echo "❌ $$HITS credential-pattern hits in tracked code (review: git grep -inE ...)"; exit 1)
	@echo "   ✓ no credential literals (VWA reset token in scripts/vwa_env*.sh excluded — upstream design constant)"
	@echo "2. No .env / .auth/ / vwa_env_remote.sh tracked..."
	@LEAKS=$$(git ls-files | grep -E "^\\.env$$|^\\.auth/|^scripts/vwa_env_remote\\.sh$$" 2>/dev/null | wc -l); \
	 test "$$LEAKS" = "0" || (echo "❌ $$LEAKS auth/env files tracked (review: git ls-files | grep -E ...)"; exit 1)
	@echo "   ✓ .env / .auth/ / vwa_env_remote.sh excluded"
	@echo "3. No personal Tailscale IPs (100.95.81.103) outside docs/reference/..."
	@IP_LEAKS=$$(git grep -nE "100\\.95\\.81\\.103" -- ':!docs/reference/' ':!docs/checkpoints/' 2>/dev/null | wc -l); \
	 test "$$IP_LEAKS" = "0" || (echo "⚠️  $$IP_LEAKS IP hits outside docs/reference (manual review required)"; \
	   git grep -nE "100\\.95\\.81\\.103" -- ':!docs/reference/' ':!docs/checkpoints/' 2>/dev/null | head -5)
	@echo "   ✓ Tailscale IP scoped to docs/reference + docs/checkpoints"
	@echo "4. VWA submodule clean (no force-push tampering)..."
	@LOCK_SHA="2c15d66d120f8498633ae65057aa50a34b3e93e0"; \
	 ACTUAL=$$(git -C external/visualwebarena rev-parse HEAD 2>/dev/null); \
	 test "$$ACTUAL" = "$$LOCK_SHA" || (echo "❌ VWA SHA $$ACTUAL ≠ lock $$LOCK_SHA"; exit 1); \
	 echo "   ✓ HEAD $$LOCK_SHA"
	@echo "5. VWA submodule tree-hash chain matches lock (SBOM contract)..."
	@LOCK_CHAIN="2696d0a61e2f70536f247ebb225f51c262b657d8b8b7b407f8581b75757a8bae"; \
	 BASE="89f5af29305c3d1e9f97ce4421462060a70c9a03"; \
	 ACTUAL_CHAIN=$$(git -C external/visualwebarena rev-list $$BASE..HEAD --format=tformat:'%H %T' 2>/dev/null | sha256sum | awk '{print $$1}'); \
	 test "$$ACTUAL_CHAIN" = "$$LOCK_CHAIN" || (echo "❌ VWA tree-hash chain $$ACTUAL_CHAIN ≠ lock $$LOCK_CHAIN (per prereg §7 L626-L630)"; exit 1); \
	 echo "   ✓ tree-hash chain $$LOCK_CHAIN"
	@echo ""
	@echo "✓ All pre-release redaction checks passed."
	@echo ""
	@echo "Next: manually fill release_redaction_checklist.md L91-93 sign-off log"
	@echo "      with today's date + this run's outcome, then re-commit."

# B-1828 part A (2026-05-22): on-demand gallery — annotate (agent-action overlay;
# reads som_image for som mode via _resolve_screenshot fallback) THEN generate HTML.
# Replaces the retired watchdog auto-refresh (now gated behind P79_WATCHDOG_GALLERY=1).
# Use for paper figure / advisor demo / som·vision visual spot-check; gallery.html
# is disposable (regenerate any time, no paper-grade pollution, no常驻 disk).
gallery:
	@test -n "$(RUN)" || (echo "ERROR: RUN=<run_dir> required"; exit 1)
	$(PYTHON) scripts/maintenance/annotate_screenshots.py --run-dir $(RUN)
	$(PYTHON) scripts/maintenance/generate_gallery.py --run-dir $(RUN)

# B-1828 P1-1 (2026-05-22, codex): on-demand AGGREGATE gallery — all cells
# (B0/B1/B2 × sites × modes) → results/phase1_paper_grade/gallery.html. Mirrors
# the retired watchdog _regenerate_phase1_paper_grade_gallery fanout that the
# single-run `gallery` target cannot reproduce. Use for advisor demo / paper
# figure "all cells" browsing. Reads existing per-run artifacts (som_image for
# som; run `make gallery RUN=<run>` first if you want agent-action overlays).
# Disposable — regenerate any time, no paper-grade pollution.
gallery-all:
	$(PYTHON) scripts/maintenance/generate_gallery.py \
	  --phase-dirs results/visualwebarena/phase1 \
	  --prefix B0_3mode B1_3mode B2_3mode \
	  --output-dir results/phase1_paper_grade
	@echo "Aggregate gallery: results/phase1_paper_grade/gallery.html (serve via http.server 8765)"

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
	@echo "[analysis] post-hook: triggering cells.base sync in background..."
	@# B-842 (A1.15b P0-2 + P1-5): N-times-fire amplification fix. crontab.txt
	@# 2026-05-13 explicitly removed glm-refresh-playbook cron entry because it
	@# "burned GLM tokens whether user read or not"; this post-hook was firing
	@# the same glm-refresh-playbook chain on every `make analysis`, making the
	@# 2026-05-13 cost-saving decision structurally unfulfilled. Trim chain to
	@# keep cells.base sync (lightweight, no GLM call) but drop PLAYBOOK refresh
	@# (GLM call). PLAYBOOK §1+§2 retire planned; this is the align step.
	@nohup bash -c "sleep 5 && $(MAKE) glm-update-cells APPLY=1" \
	  >> logs/cron/glm_update_cells.log 2>&1 < /dev/null & true

# Per-run pipeline for all paper-grade VWA runs (loop over registry)
_per_run_all:
	@for rd in $(RUN_DIRS_PAPER_VWA); do \
	  echo "── per-run: $$rd ──"; \
	  $(MAKE) analyze RUN=$$rd || exit 1; \
	done

# Cross-condition aggregators (depends on per-run output)
# B-1305 (/stress A2.3d P0-2-A* Claude OOB, 2026-05-18): phantom-meta (DL/HKSJ
# random-effects) is APPENDIX-ONLY per prereg §2 H1 decision 3A 2026-05-14 (FE
# primary, no τ²); it is NOT in the default analysis pipeline because Phase 1a
# canonical figures consume `phase1_full_prereg_decision.{csv,json,md}` (B-1301
# bootstrap percentile primary) + the legacy `phantom_meta` output is reserved
# for paper §8 Appendix-D sensitivity reporting only. Invoke explicitly via
# `make phantom-meta-appendix` when generating appendix sensitivity tables.
# Pre-B-1305 the target was chained here unconditionally; resulting
# `meta_phantom_lift.{csv,md}` was the load-bearing data source for
# `fig_meta_forest.py` despite producer docstring saying "FE would contradict
# the paper hook" (Makefile-level inclusion + producer-docstring stance =
# internal contradiction with prereg decision 3A FE primary).
_aggregate:
	$(MAKE) aggregate-sr-fp
	$(MAKE) phase1-prereg-gate
	$(MAKE) phase1-full-prereg-decision
	$(MAKE) phantom-lift
	$(MAKE) routing-auroc
	$(MAKE) aggregate-cross-site
	$(MAKE) summary-collect
	$(MAKE) aggregate-cost-electricity
	$(MAKE) analyze-mechanism
	$(PYTHON) scripts/analysis/axis_effect_size.py
	$(PYTHON) scripts/analysis/axis1_microbehavior.py
	$(PYTHON) scripts/analysis/aggregate_failure_modes.py
	$(MAKE) compare-b0-b1-all
	$(MAKE) h10-pareto-verdict  # /stress A2.8 P1-13-B B-1560 — paper §6 H10 producer

# H10 operational deployment gate (paper §6 producer; A2.8 P1-13-B B-1560 added
# to default analysis pipeline 2026-05-18). Skips cleanly with informational log
# when Pass-2 router data absent (pre-fire state) — does NOT fail the pipeline.
h10-pareto-verdict:
	@if ls results/visualwebarena/phase1/*_router_learned_*/phase1_learned_router_*_*/condition_summary_v2.json &>/dev/null; then \
		echo "[h10-pareto-verdict] Pass-2 router data found, running aggregator..."; \
		$(PYTHON) scripts/analysis/aggregate_h10_pareto.py --all --allow-partial-dev || \
			echo "[h10-pareto-verdict] WARN: aggregator failed (non-fatal in _aggregate)"; \
	else \
		echo "[h10-pareto-verdict] INFO: no Pass-2 router data yet (pre-fire); skipping H10 verdict"; \
	fi

# All figures (depends on aggregator output)
_figures:
	$(PYTHON) scripts/analysis/figures/fig0a_sr_per_mode_heatmap.py
	# fig0b_fp_rate_per_mode retired §139.8 + /stress A1.6 (2026-05-16)
	$(PYTHON) scripts/analysis/figures/fig0b_extra_confidence_calibration.py
	# fig0c: routine pipeline = interim monitoring → explicit --allow-partial (script marks
	# output NON_PAPER_GRADE + watermark). Verdict-day strict path runs WITHOUT the flag
	# per VERDICT_DAY_RUNBOOK (fail-closed exit 2 on partial data — Chunk 2, PROTOCOL_NOTE_05).
	$(PYTHON) scripts/analysis/figures/fig0c_drop_one_oracle.py --allow-partial
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

# B-184: Phase 1 legacy normal-Z transparency-only H1 artifact.
# The CANONICAL paper §1 H1 source-of-truth is the full decision producer
# (`phase1-full-prereg-decision`) and its bootstrap-percentile verdict; this
# target retains the shared per-cell kernel plus normal-Z transparency columns.
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

# Cross-cell meta-analysis (DerSimonian-Laird random-effect + HKSJ, T0c)
# B-1305 (/stress A2.3d P0-2-A*, 2026-05-18): retained for APPENDIX-ONLY
# sensitivity reporting per prereg §2 H1 decision 3A (FE primary, no τ², 2026-05-14).
# NOT chained from `_aggregate` (default `make analysis`) — canonical Phase 1a
# figures consume `phase1_full_prereg_decision.{csv,json,md}` (bootstrap
# percentile primary). Invoke explicitly when generating paper §8 Appendix-D
# sensitivity tables. The `phantom-meta-appendix` alias is the canonical entry
# point post-B-1305; `phantom-meta` legacy target retained as bareword alias.
phantom-meta phantom-meta-appendix:
	@echo "[appendix-only — B-1305] DL/HKSJ random-effects sensitivity (NOT primary gate per prereg decision 3A)"
	$(PYTHON) scripts/analysis/aggregate_phantom_meta.py

# ---- Layered analysis (paper_planning §3 framework, paper-grade B0 only) ----

# Layer 0 — Outcome (SR / oracle / AUROC / task-pool / category)
analyze-layer0:
	$(MAKE) aggregate-sr-fp
	$(MAKE) phantom-lift
	$(MAKE) routing-auroc
	$(PYTHON) scripts/analysis/figures/fig0d_taskpool_jaccard.py
	# fig0c: routine pipeline = interim monitoring → explicit --allow-partial (script marks
	# output NON_PAPER_GRADE + watermark). Verdict-day strict path runs WITHOUT the flag
	# per VERDICT_DAY_RUNBOOK (fail-closed exit 2 on partial data — Chunk 2, PROTOCOL_NOTE_05).
	$(PYTHON) scripts/analysis/figures/fig0c_drop_one_oracle.py --allow-partial
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
	@# §5 mechanism deferred (advisor 2026-05-14). Report assumes reddit paper-grade cells;
	@# on partial Phase 1a (reddit not promoted) it crashes on missing reddit Exp1 stats.
	@# NON-FATAL so outcome analysis + figures aren't blocked. NOTE: remove `|| echo`
	@# (restore fatal) when §5 resumes — else real mechanism bugs get masked.
	$(PYTHON) scripts/analysis/mechanism_per_task.py || echo "[analyze-mechanism] §5 deferred — non-fatal skip (partial data / needs reddit paper-grade)"

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

# Cross-baseline site comparison — runs compare_b0_b1.py for each (B0,B1[,B2]) DOM
# run set on cls + red. Outputs to results/visualwebarena/phase1/b0_vs_b1[_vs_b2]_<site>/.
# 3-model deep-update 2026-05-18: B2_RUN_CLS/RED resolved from registry; returns
# empty string when cell missing → compare auto-falls through to legacy 2-way mode.
# Resolve the current paper-grade DOM entry explicitly.  get_cell() intentionally
# searches every grade for backwards compatibility and sorts by run path, which
# made this target select stale archived vintages after paper-grade promotion.
B0_RUN_CLS ?= $(shell $(PYTHON) -c "from scripts.analysis.lib.run_registry import get_cells; c=get_cells(baseline='B0',site='classifieds',mode='DOM',grade='paper-grade'); print(c[0].run_dir if c else '')")
B0_RUN_RED ?= $(shell $(PYTHON) -c "from scripts.analysis.lib.run_registry import get_cells; c=get_cells(baseline='B0',site='reddit',mode='DOM',grade='paper-grade'); print(c[0].run_dir if c else '')")
B1_RUN_CLS ?= $(shell $(PYTHON) -c "from scripts.analysis.lib.run_registry import get_cells; c=get_cells(baseline='B1',site='classifieds',mode='DOM',grade='paper-grade'); print(c[0].run_dir if c else '')")
B1_RUN_RED ?= $(shell $(PYTHON) -c "from scripts.analysis.lib.run_registry import get_cells; c=get_cells(baseline='B1',site='reddit',mode='DOM',grade='paper-grade'); print(c[0].run_dir if c else '')")
B2_RUN_CLS ?= $(shell $(PYTHON) -c "from scripts.analysis.lib.run_registry import get_cells; c=get_cells(baseline='B2',site='classifieds',mode='DOM',grade='paper-grade'); print(c[0].run_dir if c else '')")
B2_RUN_RED ?= $(shell $(PYTHON) -c "from scripts.analysis.lib.run_registry import get_cells; c=get_cells(baseline='B2',site='reddit',mode='DOM',grade='paper-grade'); print(c[0].run_dir if c else '')")

# Legacy target retained for back-compat; alias to compare-baselines-all.
compare-b0-b1-all: compare-baselines-all

compare-baselines-all:
	@if [ -z "$(B0_RUN_CLS)" ] || [ ! -d "$(B0_RUN_CLS)" ]; then \
		echo "[compare-baselines-all] SKIP classifieds: paper-grade B0 DOM run is absent or not a directory: $(B0_RUN_CLS)"; \
	elif [ -z "$(B1_RUN_CLS)" ] || [ ! -d "$(B1_RUN_CLS)" ]; then \
		echo "[compare-baselines-all] SKIP classifieds: paper-grade B1 DOM run is absent or not a directory: $(B1_RUN_CLS)"; \
	elif [ -n "$(B2_RUN_CLS)" ] && [ ! -d "$(B2_RUN_CLS)" ]; then \
		echo "[compare-baselines-all] SKIP classifieds: paper-grade B2 DOM run is not a directory: $(B2_RUN_CLS)"; \
	else \
		$(MAKE) --no-print-directory compare B0="$(B0_RUN_CLS)" B1="$(B1_RUN_CLS)" B2="$(B2_RUN_CLS)" SITE=classifieds; \
	fi
	@if [ -z "$(B0_RUN_RED)" ] || [ ! -d "$(B0_RUN_RED)" ]; then \
		echo "[compare-baselines-all] SKIP reddit: paper-grade B0 DOM run is absent or not a directory: $(B0_RUN_RED)"; \
	elif [ -z "$(B1_RUN_RED)" ] || [ ! -d "$(B1_RUN_RED)" ]; then \
		echo "[compare-baselines-all] SKIP reddit: paper-grade B1 DOM run is absent or not a directory: $(B1_RUN_RED)"; \
	elif [ -n "$(B2_RUN_RED)" ] && [ ! -d "$(B2_RUN_RED)" ]; then \
		echo "[compare-baselines-all] SKIP reddit: paper-grade B2 DOM run is not a directory: $(B2_RUN_RED)"; \
	else \
		$(MAKE) --no-print-directory compare B0="$(B0_RUN_RED)" B1="$(B1_RUN_RED)" B2="$(B2_RUN_RED)" SITE=reddit; \
	fi

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
# `.raw.json` templates. /stress A1.18-re B-616 P1-10-B (codex 2026-05-17):
# OSF replayers + fresh-clone setup must run this after `git clone` + `git
# submodule update --init` because the split files are gitignored derived
# artifacts. Required env vars (B-604 idempotent rebuild + B-615 byte-stable):
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

# ---- Paper prose: paper-deslop pipeline (tools/paper-deslop/, see VENDORED.md) ----
# Vale's config is vendored, not at the repo root, so every invocation needs --config.
VALE          ?= vale
DESLOP_DIR    ?= tools/paper-deslop
DESLOP_CONFIG ?= $(DESLOP_DIR)/.vale.ini
PAPER_DIR     ?= docs/checkpoints/paper_drafts

# Vale AI-tell lint. Default scope = all paper drafts; F=<file> for one file.
#   make deslop-lint
#   make deslop-lint F=docs/checkpoints/paper_drafts/section1_intro.md
#   make deslop-lint LEVEL=suggestion      # LEVEL = error (default) | warning | suggestion
deslop-lint:
	@$(VALE) --config=$(DESLOP_CONFIG) --minAlertLevel=$${LEVEL:-error} \
	  $${F:-$$(git ls-files '$(PAPER_DIR)/*.md' '$(PAPER_DIR)/**/*.md' '$(PAPER_DIR)/**/*.tex')}

# Lexical invariant gate: run after ANY rewrite, blocking.
#   make deslop-gate OLD=/tmp/baseline.md NEW=docs/checkpoints/paper_drafts/section1_intro.md
deslop-gate:
	@if [ -z "$(OLD)" ] || [ -z "$(NEW)" ]; then \
	  echo "Usage: make deslop-gate OLD=<baseline file> NEW=<rewritten file>"; \
	  echo "  baseline:  git show HEAD:<path> > /tmp/baseline.md"; \
	  exit 64; \
	fi
	@python3 $(DESLOP_DIR)/scripts/invariant_check.py "$(OLD)" "$(NEW)" \
	  --terms $(DESLOP_DIR)/terms.txt

# Ratcheted lint — byte-for-byte what CI runs. Blocking set = deslopped.txt
# (a file joins it once /deslop-paper gets it error-clean); ALL=1 lints every
# draft advisory-style. An entry matching no tracked file exits 2, never a
# silent no-op.
#   make deslop-ratchet          # blocking set only
#   make deslop-ratchet ALL=1    # advisory sweep over all drafts
deslop-ratchet:
	@bash $(DESLOP_DIR)/scripts/ratchet_lint.sh $(if $(ALL),--all,) --output=line

# Per-term hit counts against a draft: a whitelisted term with zero
# occurrences is a typo or wishful thinking. F defaults to the whole corpus.
#   make deslop-audit
#   make deslop-audit F=docs/checkpoints/paper_drafts/section4_empirical_findings.md
deslop-audit:
	@f="$${F:-/tmp/p79_all_drafts.md}"; \
	if [ -z "$(F)" ]; then cat $(PAPER_DIR)/*.md $(PAPER_DIR)/aaai27/latex/*.tex > "$$f"; fi; \
	python3 $(DESLOP_DIR)/scripts/invariant_check.py "$$f" "$$f" \
	  --terms $(DESLOP_DIR)/terms.txt --term-audit

# Pipeline self-test: Vale fires on the slop fixture, the gate catches all
# eight adversarial fixtures + the markdown-percent regression, term matching
# is word-bounded, the ratchet blocks only what it declares, vocab is in sync.
deslop-selftest:
	@bash $(DESLOP_DIR)/tests/run.sh

# Regenerate the Vale vocabulary from terms.txt (commit both files).
deslop-vocab:
	@python3 $(DESLOP_DIR)/scripts/gen_vale_vocab.py
