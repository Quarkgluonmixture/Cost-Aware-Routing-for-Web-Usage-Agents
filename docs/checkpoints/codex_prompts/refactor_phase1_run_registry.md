# Phase 1 — Run Registry: replace hardcoded paths with single source of truth

**Date**: 2026-05-01
**Scope**: Phase 1 of analysis pipeline refactor (P0 — solves "add cell = modify 7 files" maintenance burden)
**Out of scope**: `make analysis` consolidation (Phase 2), new figures for Micro 2b-2f / Efficiency 3c (Phase 3)
**Style**: Minimal change. Don't introduce plugin system, config DSL, or new abstractions beyond `run_registry.py` + `run_manifest.yaml`.

---

## Goal

Currently 13+ scripts hardcode run directory paths to discover cells (B0/B1 × cls/red × DOM/SoM/Vision/P-text/P-prompt/P-SoM). Adding a new cell (e.g. today's `B1_phantom_dom_classifieds_20260429`) requires modifying 7 figure scripts + 1 aggregator + 1 Makefile var. This is unsustainable and error-prone (today's 8 manual edits).

**Single source of truth**: `results/phantom_paper/run_manifest.yaml` declares all paper-grade + in-flight runs. All scripts read via `scripts/analysis/lib/run_registry.py` typed API. Adding a cell = update manifest only.

---

## Deliverables

### 1. NEW module: `scripts/analysis/lib/run_registry.py`

**API design** (typed dataclasses, no magic):

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Static constants (paper-canonical mode names)
PAPER_MODES = ["DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM"]
LEGACY_MODE_ALIAS = {
    # legacy mode-id-in-condition-summary → paper-canonical name
    "dom": "DOM",
    "som": "SoM",
    "vision": "Vision",
    "phantom_dom": "P-text",   # legacy alias preserved (existing JSONL)
    "phantom_text": "P-text",  # current code path
    "phantom_prompt": "P-prompt",
    "phantom_som": "P-SoM",
}
SITES = ["classifieds", "reddit", "shopping"]
BASELINES = ["B0", "B1"]
EXPECTED_N = {"classifieds": 234, "reddit": 210, "shopping": 466}

Grade = Literal["paper-grade", "paper-grade-pre-bug", "in-flight", "archived"]

@dataclass(frozen=True)
class CellSpec:
    """One (baseline × site × mode) cell, identified by full run + condition path."""
    baseline: str          # "B0" / "B1" / future "Claude"
    site: str              # "classifieds" / "reddit" / "shopping"
    mode: str              # paper-canonical: "DOM" / "P-text" / "P-SoM" etc.
    run_dir: Path          # absolute or repo-relative: results/visualwebarena/phase1/B0_3mode_classifieds_20260413
    condition_subdir: str  # e.g. "phase1_dom_router_0"
    expected_n: int        # 234 / 210 / 466
    grade: Grade
    notes: str = ""        # free-form, e.g. "Phase A pre-bug, kept as reference"

    @property
    def episodes_dir(self) -> Path:
        return self.run_dir / self.condition_subdir / "episodes"

    @property
    def condition_summary_path(self) -> Path:
        return self.run_dir / self.condition_subdir / "condition_summary_v2.json"

    @property
    def actual_n(self) -> int:
        """Live count of episode summary files."""
        if not self.episodes_dir.exists():
            return 0
        return len(list(self.episodes_dir.glob("*_summary_v2.json")))

    @property
    def is_complete(self) -> bool:
        return self.actual_n >= self.expected_n


def load_manifest(path: Path | None = None) -> dict:
    """Load run_manifest.yaml. Default path: results/phantom_paper/run_manifest.yaml."""

def get_all_cells(grade_filter: list[Grade] | None = None) -> list[CellSpec]:
    """All cells in manifest. grade_filter defaults to ['paper-grade', 'paper-grade-pre-bug']."""

def get_cells(
    *,
    baseline: str | list[str] | None = None,
    site: str | list[str] | None = None,
    mode: str | list[str] | None = None,
    grade: Grade | list[Grade] | None = None,
) -> list[CellSpec]:
    """Filter cells by any combination of fields. Returns sorted list."""

def get_cell(baseline: str, site: str, mode: str) -> CellSpec | None:
    """Get exactly one cell or None if not in manifest."""

def get_run_dirs_paper_vwa() -> list[Path]:
    """Compatibility shim: return paper-grade VWA run dirs (deduped) for legacy Makefile var.
    Replaces RUN_DIRS_PAPER_VWA hardcoded list."""
```

**Implementation notes**:
- YAML loading: use `yaml.safe_load`. Add `pyyaml` to `pyproject.toml` `[analysis]` extras if not already present (verify first).
- Path resolution: relative paths in manifest are relative to `results/visualwebarena/phase1/`. Absolute paths allowed.
- Validation: on load, error if grade ∉ Grade union or mode ∉ paper-canonical. Issue warning if `actual_n == 0` for paper-grade cells (run dir missing).
- No caching beyond per-process (function-level memoization OK; don't introduce file-watch).

### 2. NEW manifest: `results/phantom_paper/run_manifest.yaml`

Schema:

```yaml
# results/phantom_paper/run_manifest.yaml
# Single source of truth for analysis pipeline run discovery.
# Edit this file when adding/removing runs; no code changes needed.

# Each entry maps one (baseline, site, mode) to its run dir + condition subdir.
# Multiple modes from one run share a `run_dir` and differ in `condition_subdir`.

cells:
  # ── B0 classifieds ──
  - baseline: B0
    site: classifieds
    mode: DOM
    run_dir: B0_3mode_classifieds_20260413
    condition_subdir: phase1_dom_router_0
    expected_n: 234
    grade: paper-grade-pre-bug   # Phase A pre-bug-fix, kept as reference until 14-cell rerun
    notes: "Phase A pre-bug; will be replaced after RunPod rerun"

  - baseline: B0
    site: classifieds
    mode: SoM
    run_dir: B0_3mode_classifieds_20260413
    condition_subdir: phase1_som_router_0
    expected_n: 234
    grade: paper-grade-pre-bug

  - baseline: B0
    site: classifieds
    mode: Vision
    run_dir: B0_3mode_classifieds_20260413
    condition_subdir: phase1_vision_router_0
    expected_n: 234
    grade: paper-grade-pre-bug

  - baseline: B0
    site: classifieds
    mode: P-text
    run_dir: B0_phantom_text_classifieds_20260427
    condition_subdir: phase1_phantom_dom_router_0
    expected_n: 234
    grade: paper-grade-pre-bug

  - baseline: B0
    site: classifieds
    mode: P-SoM
    run_dir: B0_phantom_som_classifieds_20260426
    condition_subdir: phase1_phantom_som_router_0
    expected_n: 234
    grade: paper-grade-pre-bug

  # ── B0 reddit ── (5 cells: DOM/SoM/Vision/P-text/P-prompt/P-SoM)
  # ... (full list to populate by codex from existing RUN_DIRS_PAPER_VWA + new cells)

  # ── B1 classifieds ──
  # ... (DOM/SoM/Vision/P-text/P-SoM, P-prompt is in_flight)

  # ── B1 reddit ── (3-mode: DOM/SoM/Vision)
  # ...

in_flight:
  - baseline: B1
    site: classifieds
    mode: P-prompt
    run_dir: B1_phantom_prompt_classifieds_20260501
    condition_subdir: phase1_phantom_prompt_router_0
    expected_n: 234
    grade: in-flight
    notes: "Tier 2 chain, ~24h ETA; pre-Phase-A data, will be discarded after 14-cell rerun"

archived:
  # Archived runs (not used in analysis, kept for git history):
  # - B1_3mode_shopping_20260413_pre_magento_bug
```

**Codex action**: Populate the manifest by:
1. Reading current `Makefile` `RUN_DIRS_PAPER_VWA` list (10 entries).
2. Adding all currently-existing run dirs visible in `results/visualwebarena/phase1/` matching `B[01]_*classifieds*` and `B[01]_*reddit*` patterns.
3. For each run dir, scan `phase1_*_router_0/` subdirs to discover modes.
4. Use `LEGACY_MODE_ALIAS` to map condition subdir mode-id back to paper-canonical mode name.
5. Mark all current cells as `paper-grade-pre-bug` (since they were collected before Phase A 4-cluster fix on 2026-04-30).
6. Move `B1_phantom_prompt_classifieds_20260501` to `in_flight:` section.
7. Add `archived:` section with any `_archive/` or `_pre_magento_bug` dirs found.

### 3. Refactor 13+ scripts to use registry

**Aggregator scripts**:
- `scripts/analysis/aggregate_phantom_lift.py` — replace `_build_cell` hardcoded modes dict with `get_cells(baseline=, site=)` lookup
- `scripts/analysis/aggregate_routing_auroc.py` — same
- `scripts/analysis/aggregate_cross_site.py` — replace `RUN_DIRS_PAPER_VWA` arg consumption with `get_run_dirs_paper_vwa()`
- `scripts/analysis/aggregate_cost_electricity.py` — same pattern
- `scripts/analysis/aggregate_sr_fp_per_mode.py` — same pattern
- `scripts/analysis/mechanism_per_task.py` — same pattern
- `scripts/analysis/collect_analysis_summary.py` (called by `summary-collect`) — same pattern

**Figure scripts (8 files)**:
- `scripts/analysis/figures/fig0c_drop_one_oracle.py`
- `scripts/analysis/figures/fig0d_taskpool_jaccard.py`
- `scripts/analysis/figures/fig0e_category_mode_heatmap.py`
- `scripts/analysis/figures/fig0f_overlap_stacked_bar.py`
- `scripts/analysis/figures/fig1c_strategy_gradient.py`
- `scripts/analysis/figures/fig3a_token_cost_intra_baseline.py`
- `scripts/analysis/figures/fig3d_cost_sr_frontier.py`
- `scripts/analysis/figures/fig0c_phantom_lift_bars.py` (reads phantom_lift.csv — should still work, verify)

**Refactor pattern (per file)**:
```python
# OLD (hardcoded):
"DOM": RESULTS / "B1_3mode_classifieds_20260413/phase1_dom_router_0/episodes",

# NEW (registry-driven):
from scripts.analysis.lib.run_registry import get_cells

cells = get_cells(baseline="B1", site="classifieds")
modes = {cell.mode: cell.episodes_dir for cell in cells}
```

For figure scripts that need a fixed display order, use `PAPER_MODES` list to sort.

### 4. Makefile cleanup

**Replace** `RUN_DIRS_PAPER_VWA ?= ...` (hardcoded 10 entries) with:

```make
# Single source of truth: results/phantom_paper/run_manifest.yaml
# (set by scripts/analysis/lib/run_registry.py::get_run_dirs_paper_vwa())
RUN_DIRS_PAPER_VWA = $(shell $(PYTHON) -c "from scripts.analysis.lib.run_registry import get_run_dirs_paper_vwa; print(' '.join(str(p) for p in get_run_dirs_paper_vwa()))")
```

Test: `make -n analyze-paper-per-run` should still print correct run loop.

---

## Acceptance criteria

1. **Smoke**: `make analyze-paper` passes after refactor with same output as before (no hardcoded paths anywhere except `run_manifest.yaml`).
2. **Add-cell test**: Adding a new entry to `run_manifest.yaml` (e.g. mock B1 reddit P-text) and re-running `make figures` picks it up automatically — no Python file edits needed.
3. **Backward-compat**: All existing make targets (`make analyze RUN=`, `make analyze-paper`, `make analyze-layered`, `make figures`, `make phantom-lift`) work without changes from user perspective.
4. **Validation**: `python3 -c "from scripts.analysis.lib.run_registry import get_all_cells; cells=get_all_cells(); print(len(cells), 'cells loaded')"` returns ≥10 cells (current paper-grade count).
5. **Tests**: Add `tests/analysis/test_run_registry.py` with:
   - `test_load_manifest_succeeds()`
   - `test_get_cells_by_baseline()`
   - `test_get_cell_returns_none_when_missing()`
   - `test_episodes_dir_exists()` (smoke check on at least one cell)
   - `test_legacy_mode_alias_resolves()` (e.g. `phantom_dom` → `P-text`)
6. **Coverage report**: Print summary at end of refactor (e.g. `python3 -m scripts.analysis.lib.run_registry --report`):
   ```
   Paper-grade cells: 10 (5 B0 cls + 5 B0 red + ...)
   Paper-grade-pre-bug: 11 (...)
   In-flight: 1 (B1 P-prompt cls)
   Archived: 1 (B1 shopping pre-Magento-bug)
   ```

---

## Out of scope (later phases)

- **Phase 2** (`make analysis` single entry point): leave as-is, just refactor data layer.
- **Phase 3** (new figures for Micro 2b/2c/2d/2e/2f + Efficiency 3c): no figure additions.
- Don't refactor analysis logic itself (per-task computations, statistical methods stay identical).
- Don't add cross-baseline / cross-model new aggregators.
- Don't touch `experiment_watchdog.py` or runner code.

---

## Constraints + risks

- **Don't break paper-strategic data**: `phantom_lift.csv` / `auroc_cross_condition.csv` outputs must be **byte-identical or equivalent** before/after refactor (verify with `make analyze-paper && diff`).
- **Don't change figure visual content**: PNG bytes might differ due to matplotlib randomness, but visible content (which cells / which colors / labels) must be equivalent.
- **YAML ordering**: don't depend on dict insertion order in manifest cell ordering; sort canonically `(baseline, site, paper_mode_index)` in `get_cells()` output.
- **PYTHONPATH**: `scripts/analysis/lib/` may need `__init__.py`. Verify import path works from Makefile $(PYTHON) shell-out.
- **Symlinks / archived dirs**: gracefully skip if `run_dir` doesn't exist (warn for paper-grade, silent for archived).

---

## Reference docs

- `docs/checkpoints/paper_planning.md` §3 (4-dim Evidence framework + sub-codes 0a-3d, authoritative spec for what aggregators produce)
- `docs/checkpoints/实验笔记.md` §108 (2026-05-01 framework refinement chronicle, evidence/explanation separation)
- `Makefile` lines containing `RUN_DIRS_PAPER_VWA` and `analyze-paper-per-run` (current pipeline structure)
- `scripts/analysis/aggregate_phantom_lift.py` (canonical example of hardcoded `_build_cell` pattern to be replaced)
- `scripts/analysis/figures/fig0c_drop_one_oracle.py` (canonical example of figure-script hardcoded paths)

---

## Suggested implementation order

1. Create `scripts/analysis/lib/__init__.py` + `run_registry.py` (skeleton + dataclasses + load_manifest, no scripts using it yet).
2. Create `results/phantom_paper/run_manifest.yaml` populated from current state.
3. Add tests + run them, verify registry loads.
4. Refactor `aggregate_phantom_lift.py` first (smallest aggregator, well-tested via phantom_lift.md output).
5. Run `make phantom-lift` and verify output unchanged. Diff `phantom_lift.csv` before/after.
6. Refactor remaining aggregators one-by-one with same diff-verify cadence.
7. Refactor figure scripts in parallel (each independent).
8. Update Makefile `RUN_DIRS_PAPER_VWA` to use shell-out.
9. Run `make analyze-paper` end-to-end. Verify all outputs unchanged.
10. Print coverage report. Confirm cell count matches expected.

Total estimated changes: ~24 files modified or created, ~15K tokens of code output.
