# Phase 2 — `make analysis` consolidation: single entry point for analysis pipeline

**Date**: 2026-05-01
**Scope**: Phase 2 of analysis pipeline refactor (P1 — solves "user doesn't know which target to run")
**Prerequisites**: Phase 1 (`run_registry.py` + `run_manifest.yaml`) already shipped (commit `ce05366`)
**Out of scope**: Phase 3 (new figures for Micro 2b/2c/2d/2e/2f + Efficiency 3c)
**Style**: Minimal change. Don't introduce new abstractions or build systems beyond what Makefile supports.

---

## Goal

Currently `Makefile` has **15+ overlapping analysis targets** that do partially-overlapping work:

```
analyze RUN=...                      # per-run: rederive + reason_diag + cross_rep + analyze_run + confidence
analyze-paper                        # = analyze-paper-per-run + cross-condition + figures
analyze-paper-per-run                # loop over RUN_DIRS_PAPER_VWA running per-run pipeline
analyze-layered                      # = analyze-layer0/1/2/3 + analyze-mechanism + figures
analyze-layer0 / analyze-layer1 / analyze-layer2 / analyze-layer3
analyze-mechanism
aggregate-cross-site / aggregate-cost-electricity / aggregate-sr-fp / phantom-lift / routing-auroc
compare-b0-b1-all / compare B0= B1= SITE=
figures / fig12-micro-heatmap
summary-collect
```

**User confusion**: which target to run? `make analyze-paper` and `make analyze-layered` overlap; user has to remember which subset of evidence layer each covers.

**Goal**: single `make analysis` entry point that runs the complete pipeline. Sub-targets remain accessible (backward-compat) but hidden from `make help` default output. Add `FAST=1` mode for skip-per-run (figures+aggregators only, ~10s vs ~5min).

---

## Deliverables

### 1. Single `make analysis` consolidated target

```make
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
```

### 2. Internal sub-targets (prefixed `_` to mark hidden)

```make
.PHONY: _per_run_all _aggregate _figures _status

# Per-run pipeline for all paper-grade VWA runs (loop over registry)
_per_run_all:
	@for rd in $(RUN_DIRS_PAPER_VWA); do \
	  echo "── per-run: $$rd ──"; \
	  $(MAKE) analyze RUN=$$rd || exit 1; \
	done

# Cross-condition aggregators (depends on per-run output)
_aggregate:
	$(MAKE) aggregate-sr-fp
	$(MAKE) phantom-lift
	$(MAKE) routing-auroc
	$(MAKE) aggregate-cross-site
	$(MAKE) summary-collect
	$(MAKE) aggregate-cost-electricity
	$(MAKE) analyze-mechanism
	$(PYTHON) scripts/analysis/axis_effect_size.py
	$(PYTHON) scripts/analysis/axis1_microbehavior.py
	$(MAKE) compare-b0-b1-all

# All figures (depends on aggregator output)
_figures:
	$(PYTHON) scripts/analysis/figures/fig0c_drop_one_oracle.py
	$(PYTHON) scripts/analysis/figures/fig0c_phantom_lift_bars.py
	$(PYTHON) scripts/analysis/figures/fig0d_taskpool_jaccard.py
	$(PYTHON) scripts/analysis/figures/fig0e_category_mode_heatmap.py
	$(PYTHON) scripts/analysis/figures/fig0f_overlap_stacked_bar.py
	$(PYTHON) scripts/analysis/figures/fig0g_routing_auroc_heatmap.py
	$(PYTHON) scripts/analysis/figures/fig1ab_cascade_diamond.py
	$(PYTHON) scripts/analysis/figures/fig1c_strategy_gradient.py
	$(PYTHON) scripts/analysis/figures/fig2_micro_divergence_heatmap.py
	$(PYTHON) scripts/analysis/figures/fig3a_token_cost_intra_baseline.py
	$(PYTHON) scripts/analysis/figures/fig3d_cost_sr_frontier.py
	$(PYTHON) scripts/analysis/figures/fig3_regional_carbon.py
	$(PYTHON) scripts/analysis/figures/fig_capability_b0_b1.py

# Live evidence status snapshot (read-only summary of aggregator outputs)
_status:
	$(PYTHON) scripts/analysis/layered_status.py
	@echo ""
	@echo "[analysis] outputs:"
	@echo "  Aggregators: results/phantom_paper/{phantom_lift,auroc_cross_condition,run_summary_collect}.{csv,md}"
	@echo "  Figures:     results/phantom_paper/figures/*.png"
	@echo "  Live status: docs/analysis/layered_evidence_status.md"
	@echo "  Per-run:     results/visualwebarena/phase1/<run>/analysis/"
```

### 3. Backward-compat aliases (deprecated but kept)

These targets become **aliases for `make analysis`** with deprecation warnings; existing scripts/docs that reference them keep working:

```make
# DEPRECATED: use `make analysis` instead. Kept for backward compatibility.
analyze-paper analyze-layered: 
	@echo "[deprecated] '$@' is now an alias for 'make analysis'. Use 'make analysis' going forward."
	$(MAKE) analysis

# Layer-specific targets remain (callable from analysis pipeline + ad-hoc)
analyze-layer0 / analyze-layer1 / analyze-layer2 / analyze-layer3: # unchanged from Phase 1
analyze-paper-per-run: # alias for _per_run_all
	$(MAKE) _per_run_all
```

**Don't delete** any existing target — just add `analysis` as new entry point and convert the umbrella targets (`analyze-paper`, `analyze-layered`) to thin aliases. This guarantees nothing breaks for users with `make analyze-paper` muscle memory.

### 4. Update `make help` to feature `make analysis`

```make
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
```

---

## Acceptance criteria

1. **Smoke**: `make analysis` succeeds end-to-end from clean state, produces same output files as `make analyze-paper && make analyze-layered` did before.
2. **Output equivalence**: After `make analysis`, all artifacts under `results/phantom_paper/` and `docs/analysis/` exist + `phantom_lift.csv` / `auroc_cross_condition.csv` byte-identical to pre-refactor (re-run baseline before refactor as gold copy).
3. **FAST mode**: `make analysis FAST=1` succeeds without running any per-run analysis (rederive/reason-diag/cross-rep/analyze_run/confidence skipped) — confirms by `time make analysis FAST=1 > /tmp/fast.log` is < 60 seconds.
4. **RUN mode**: `make analysis RUN=results/visualwebarena/phase1/B1_phantom_dom_classifieds_20260429` runs per-run on that single dir + full downstream cross-condition.
5. **Backward-compat**: `make analyze-paper` and `make analyze-layered` still work (with deprecation warning), produce same outputs as `make analysis`.
6. **Help readability**: `make help` clearly shows `analysis` as primary entry point at top, internal `_*` targets listed but not prominent.
7. **Docs**: Update `Makefile` header comment + `CLAUDE.md` `## 构建与运行` section to reference `make analysis` instead of `make analyze-paper` as primary command.

---

## Constraints + risks

- **Don't break existing scripts/CI**: `scripts/queues/*.sh`, `scripts/maintenance/experiment_watchdog.py`, and similar may invoke `make analyze RUN=...` or `make analyze-paper` — those must continue to work.
- **Don't change figure script invocation order**: figures may have implicit dependencies (e.g. fig0c_phantom_lift_bars reads phantom_lift.csv from aggregator). The `_aggregate` → `_figures` order in pipeline is critical.
- **Don't merge `analyze-layer0/1/2/3` into `_aggregate`**: they're useful as standalone targets for layer-specific debugging. Keep both: layered targets remain, but `_aggregate` doesn't call them (it directly calls aggregator scripts to avoid double-execution).
- **Don't introduce shell-level conditional logic**: use Makefile `ifeq`/`ifneq` (already in template), not bash `if`. This keeps Makefile portable.
- **Don't add new aggregators or figures** in this phase — just consolidate existing.
- **Backward-compat for `RUN_DIRS_PAPER_VWA`**: this env var (set by `run_registry.get_run_dirs_paper_vwa()` shim) is consumed by aggregators; don't break it.

---

## Reference docs

- `Makefile` (current state, Phase 1 refactor commit ce05366)
- `docs/checkpoints/paper_planning.md` §3 4-dim Evidence framework + §3 顶部 evidence/explanation separation (2026-05-01 update)
- `docs/checkpoints/实验笔记.md` §108 chronicle (Phase 2 motivation: "make 很多很乱")
- `scripts/analysis/lib/run_registry.py` (Phase 1, provides `get_run_dirs_paper_vwa()`)
- `results/phantom_paper/run_manifest.yaml` (Phase 1 source of truth)

---

## Suggested implementation order

1. Read current `Makefile` end-to-end. List all analysis-related targets in scope.
2. Run baseline: `make analyze-paper` and `make analyze-layered`. Save output file list to `/tmp/before.txt`.
3. Add `_per_run_all`, `_aggregate`, `_figures`, `_status` internal targets to `Makefile` (don't remove anything yet).
4. Add new `analysis` target with `FAST=1` / `RUN=` / default branches.
5. Run `make analysis` end-to-end. Save output file list to `/tmp/after.txt`. Diff `/tmp/before.txt` vs `/tmp/after.txt` to confirm equivalence.
6. Convert `analyze-paper` and `analyze-layered` to thin aliases with deprecation warning (keep backward-compat).
7. Update `make help` text to feature `analysis` prominently.
8. Run all 5 acceptance smoke tests:
   - `make analysis` (full)
   - `make analysis FAST=1` (skip per-run)
   - `make analysis RUN=results/visualwebarena/phase1/B1_phantom_dom_classifieds_20260429`
   - `make analyze-paper` (alias works)
   - `make analyze-layered` (alias works)
9. Update `Makefile` header comment to reference `make analysis` as primary command.
10. Update `.claude/CLAUDE.md` `## 构建与运行` section — replace `make analyze RUN=...` and `make analyze-paper` examples with `make analysis` as primary; keep one `make analyze RUN=` example for single-run case.

Total estimated changes: 1 file (Makefile, ~50 LOC added), 1 file (CLAUDE.md, ~5 lines edited). ~5K tokens of code output.

---

## Out of scope (later phases)

- **Phase 3** (new figures for Micro 2b/2c/2d/2e/2f + Efficiency 3c latency): not this phase.
- Don't refactor analysis logic itself.
- Don't remove `analyze RUN=...` (per-run target stays — it's the building block).
- Don't change `aggregate-*` script implementations.
- Don't touch `experiment_watchdog.py` or runner code.
