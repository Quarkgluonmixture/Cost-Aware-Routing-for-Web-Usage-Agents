# Phase 3 — Evidence layer coverage figures (Micro 2b/2c/2d/2e/2f + Efficiency 3c)

**Date**: 2026-05-01
**Scope**: Phase 3 of analysis pipeline refactor — fill 6 evidence-layer coverage gaps with new figure scripts
**Prerequisites**: Phase 1 (run_registry, commit `ce05366`) + Phase 2 (`make analysis`, commit `d47bb68`) shipped
**Out of scope**: New aggregator scripts, data analysis logic changes, new sub-codes beyond existing §3 framework
**Style**: Match existing figure script conventions (matplotlib + `from scripts.analysis.lib.run_registry import get_cells, PAPER_MODES` + `try/except ModuleNotFoundError` import shim + paper_planning §3 sub-code in docstring)

---

## Goal

Paper §3 4-dim Evidence framework has 16 sub-codes (4 dim × 4 cross-X). Of these, **6 sub-codes have data but no dedicated figure**. This Phase fills the visualization gap so paper §5 mechanism prose can cite figures by name (currently `section5_mechanism_reddit.md` cites "Micro 2f, median first divergent step 0" but **no figure exists** — paper-writing blocker).

**Before Phase 3 (current state)**:
| Sub-code | Live data | Figure |
|---|---|---|
| Micro 2a URL signature divergence | ✅ axis1_microbehavior.json | ✅ fig2_micro_divergence_heatmap.png |
| **Micro 2b Target-page hit rate** | ✅ axis1_microbehavior.json | ❌ no figure |
| **Micro 2c Search-keyword reuse** | ✅ axis1_microbehavior.json | ❌ no figure |
| **Micro 2d First-action divergence** | ✅ axis1_microbehavior.json | ❌ no figure |
| **Micro 2e Cross-site validity ratio** | ✅ axis1_microbehavior.json (`cross_site_validity` field) | ❌ no figure |
| **Micro 2f First-divergence step** | ✅ axis1_microbehavior.json (or computed live from steps JSONL) | ❌ no figure |
| Efficiency 3a Token cost | ✅ condition_summary_v2.json | ✅ fig3a_token_cost_intra_baseline.png (bundled with 3c) |
| **Efficiency 3c Latency per step** | ✅ condition_summary_v2.json (`avg_total_latency_ms` / `avg_steps`) | ❌ no dedicated figure (bundled in 3a) |
| Efficiency 3d Cost-SR Pareto | ✅ paper_cost_usd | ✅ fig3d_cost_sr_frontier.png |

**After Phase 3 (target state)**: 6 NEW figures fill the gap. `make analysis` `_figures` target invokes them automatically.

---

## Deliverables — 6 NEW figure scripts

### 1. `scripts/analysis/figures/fig2b_target_hit_rate.py`

**Sub-code**: Micro 2b — Target-page hit rate per mode × site (paired comparison)

**Data source**: `docs/analysis/cross_sites/axis1_microbehavior.json` (read pre-aggregated values; do NOT re-compute from raw episode summaries)
- Field: `target_hit_rate_per_mode_per_site` (or equivalent — confirm exact field name from current JSON)
- Fallback: live compute from `condition_summary_v2.json` per-cell if pre-aggregate missing

**Visualization**: 4-panel bar chart (B0 cls / B0 red / B1 cls / B1 red), x-axis = mode (DOM/P-text/P-prompt/P-SoM/SoM/Vision), y-axis = target-hit-rate %. Annotate axis-1 effect (DOM → P-text delta) and axis-2 effect (P-text → P-SoM delta) on top of bars.

**Output**: `results/phantom_paper/figures/fig2b_target_hit_rate.png`

---

### 2. `scripts/analysis/figures/fig2c_keyword_repeat.py`

**Sub-code**: Micro 2c — Search-keyword reuse / max-keyword-repeat per trajectory

**Data source**: `docs/analysis/cross_sites/axis1_microbehavior.json` (`max_keyword_repeat` field per mode per site, distribution stats)

**Visualization**: 4-panel box plot (B0/B1 × cls/red), x-axis = mode, y-axis = max keyword repeat count per task. Show median + IQR + outliers. Optional overlay: paired delta arrows (DOM → P-text, P-text → P-SoM) for axis effect direction.

**Output**: `results/phantom_paper/figures/fig2c_keyword_repeat.png`

---

### 3. `scripts/analysis/figures/fig2d_first_action_divergence.py`

**Sub-code**: Micro 2d — First-action divergence between mode pairs (% tasks where mode A and mode B chose different first action_type)

**Data source**: `docs/analysis/cross_sites/axis1_microbehavior.json` (`first_action_divergence_pct` per mode pair per site)
- If pre-aggregate missing, compute live from steps JSONL: read step 0 of each task per mode, compare action_type field across modes pairwise.

**Visualization**: 4-panel grouped bar chart (B0 cls / B0 red / B1 cls / B1 red). x-axis = mode pair (DOM↔P-text axis 1 alone, DOM↔P-prompt axis 2 alone, P-text↔P-SoM axis 2 with text fixed, P-prompt↔P-SoM axis 1 with prompt fixed, DOM↔P-SoM compound). y-axis = % tasks with divergent first action. Color-code by axis effect (axis 1 = blue, axis 2 = orange, compound = purple).

**Output**: `results/phantom_paper/figures/fig2d_first_action_divergence.png`

---

### 4. `scripts/analysis/figures/fig2e_cross_site_validity.py`

**Sub-code**: Micro 2e — Cross-site validity ratio (does axis effect generalize cls → red?)

**Data source**: `docs/analysis/cross_sites/axis1_microbehavior.json` `cross_site_validity` field (ratio of effect magnitude red / cls per axis)

**Visualization**: Single panel (or 2-panel B0/B1 if scaling is asymmetric). x-axis = axis (axis 1 / axis 2 / compound). y-axis = effect ratio (red effect / cls effect on Micro 2a-2d metric). Horizontal target line at 1.0 (perfect generalization). Annotate with which Micro sub-metric is being aggregated. **If ratio > 1.0, effect is reddit-amplified; if < 1.0, cls-amplified; if ≈ 1.0, generalize-symmetric.**

**Output**: `results/phantom_paper/figures/fig2e_cross_site_validity.png`

---

### 5. ⭐ `scripts/analysis/figures/fig2f_first_divergence.py`

**Sub-code**: Micro 2f — First-divergence step distribution per mode pair × site (paper §5 prose 已 cite, missing figure is paper-writing blocker)

**Data source**: live compute from `episodes/*_steps_v2.jsonl` per cell (or pre-aggregate if `axis1_microbehavior.json` has `first_divergent_step_distribution` field). For each task in mode pair (A, B), compute first step `i` where action_type or element_id differ; if all steps identical, use truncation length; if one mode terminates earlier, use earlier termination step.

**Visualization**: 4-panel for B0 cls / B0 red / B1 cls / B1 red. Each panel: stacked horizontal bar chart by mode pair (5 pairs as in fig2d). Stack segments = % tasks with divergence at step ranges (step 0 = "early", 1-3 = "mid-early", 4-10 = "mid", 11+ = "late", "no divergence in observed trajectory" = grey). Annotate median first-divergence step on right side of each bar.

**This is paper-cited figure** (`section5_mechanism_reddit.md` line 27/39/43/51 references "Micro 2f"). Section 5 prose currently reads "median first divergent step 0 and all divergent cases are early (Micro 2f, N=15, median first divergent step 0, early divergence 100%)" — this figure must visualize that.

**Output**: `results/phantom_paper/figures/fig2f_first_divergence.png`

---

### 6. ⭐ `scripts/analysis/figures/fig3c_latency_per_step.py`

**Sub-code**: Efficiency 3c — Per-step latency separated from cost (paper §1 hook 4-fold drop-in property (b))

**Data source**: `condition_summary_v2.json` per cell. Compute `avg_total_latency_ms / avg_steps` for per-step latency. Use `p95_step_latency_ms` field if available for tail latency.

**Visualization**: 4-panel B0/B1 × cls/red. x-axis = mode (paper-canonical order: DOM/P-text/P-prompt/P-SoM/SoM/Vision). y-axis (left) = mean per-step latency (ms). Optional y-axis (right) = p95 per-step latency. Bars + error bars (sd or IQR if available). Annotate paper §1 hook claim: "(b) ~50% lower" — show DOM/P-SoM ratio for B0 cls + B0 red.

**This is paper §1 hook visualization** — currently latency only shown bundled with cost in fig3a. Dedicated figure single-message-clear for advisor sync + paper writing.

**Output**: `results/phantom_paper/figures/fig3c_latency_per_step.png`

---

## Makefile updates

In `Makefile`, the `_figures` internal target (Phase 2 added) needs 6 new figure script invocations appended. Insert after existing fig3d_cost_sr_frontier line:

```make
_figures:
	# ... existing fig0c/0d/0e/0f/0g/1ab/1c/2/3a/3d/3_regional_carbon/_capability_b0_b1 ...
	$(PYTHON) scripts/analysis/figures/fig2b_target_hit_rate.py
	$(PYTHON) scripts/analysis/figures/fig2c_keyword_repeat.py
	$(PYTHON) scripts/analysis/figures/fig2d_first_action_divergence.py
	$(PYTHON) scripts/analysis/figures/fig2e_cross_site_validity.py
	$(PYTHON) scripts/analysis/figures/fig2f_first_divergence.py
	$(PYTHON) scripts/analysis/figures/fig3c_latency_per_step.py
```

(Order: 2b/2c/2d/2e/2f sequentially after fig2_micro_divergence_heatmap; 3c after fig3a / before fig3d.)

---

## Style guide (match existing scripts)

Each figure script must follow these conventions (see `fig0c_drop_one_oracle.py` / `fig3a_token_cost_intra_baseline.py` as templates):

1. **Module docstring**: `"""[Micro 2X] <dim> dimension — <description>.\n\nOutput:\n- results/phantom_paper/figures/figXX.png\n\n<sub-code description>.\n\nSee docs/checkpoints/paper_planning.md §3 <dim> dimension framework.\n"""`

2. **Imports**:
   ```python
   from __future__ import annotations
   import json, sys
   from pathlib import Path
   import matplotlib.pyplot as plt
   import numpy as np

   try:
       from scripts.analysis.lib.run_registry import get_cells, PAPER_MODES
   except ModuleNotFoundError:  # pragma: no cover
       sys.path.append(str(Path(__file__).resolve().parents[3]))
       from scripts.analysis.lib.run_registry import get_cells, PAPER_MODES
   ```

3. **Path convention**:
   ```python
   ROOT = Path(__file__).resolve().parents[3]
   OUT = ROOT / "results/phantom_paper/figures/figXX_name.png"
   ```

4. **Color palette** (consistent across figures):
   ```python
   MODE_COLORS = {
       "DOM": "#4c78a8", "SoM": "#f58518", "Vision": "#54a24b",
       "P-text": "#e45756", "P-prompt": "#9467bd", "P-SoM": "#b279a2",
   }
   ```

5. **Mode display order**: use `PAPER_MODES` from registry (DOM/SoM/Vision/P-text/P-prompt/P-SoM canonical order).

6. **Figure size**: 4-panel layouts use `figsize=(13.5, 9.5)` or `(22, 5.8)` (match neighboring figures).

7. **DPI + savefig**: `plt.rcParams.update({"figure.dpi": 150}); fig.savefig(OUT, bbox_inches="tight")`.

8. **Print final path** at end: `print(OUT)`.

9. **Graceful missing data**: if a cell missing data, print `[warn] <cell> missing <field>` to stderr, render empty/dashed bar with "N/A pending" annotation. Don't crash.

10. **Direct script execution**: must work both as `python3 -m scripts.analysis.figures.figXX_name` AND `python3 scripts/analysis/figures/figXX_name.py` (try/except ModuleNotFoundError import shim handles this).

---

## Acceptance criteria

1. **Smoke**: `make analysis FAST=1` runs to completion, all 6 new figures generated under `results/phantom_paper/figures/`.
2. **Each figure script runs standalone**: `python3 scripts/analysis/figures/figXX.py` succeeds (smoke each independently).
3. **Each figure visually populated** for B0 cls + B0 red (paper-headline cells); B1 cls partial OK with "N/A pending" annotation for missing modes.
4. **No data layer regression**: `phantom_lift.csv` / `auroc_cross_condition.csv` byte-identical to pre-Phase-3 (no aggregator scripts modified, only new figure scripts added).
5. **fig2f matches paper §5 prose claim** (`section5_mechanism_reddit.md` cites "Micro 2f, N=15, median first divergent step 0, early divergence 100%" for B0 reddit P-text↔P-SoM pair) — figure must show this finding for that pair specifically.
6. **fig3c shows latency drop**: B0 cls + B0 red P-SoM latency should be visibly < SoM latency (paper hook (b) ~50% claim). If data shows otherwise, flag as "anomaly to investigate" but still render figure.

---

## Reference docs

- `docs/checkpoints/paper_planning.md` §3 Evidence framework (Micro 2a-2f / Efficiency 3a-3d sub-code definitions)
- `docs/checkpoints/paper_drafts/section5_mechanism_reddit.md` (paper-cited Micro 2f references)
- `scripts/analysis/figures/fig0c_drop_one_oracle.py` (canonical 4-panel paired figure template)
- `scripts/analysis/figures/fig3a_token_cost_intra_baseline.py` (canonical Efficiency dimension figure template)
- `scripts/analysis/lib/run_registry.py` (Phase 1 API)
- `Makefile` `_figures` target (Phase 2 list to append to)
- `docs/analysis/cross_sites/axis1_microbehavior.{json,md}` (Micro data source)
- `results/visualwebarena/phase1/<run>/<cond>/condition_summary_v2.json` (Efficiency data source)

---

## Suggested implementation order

1. Inventory `axis1_microbehavior.json` field names — confirm what's pre-aggregated vs what needs live compute.
2. Implement fig2f_first_divergence.py FIRST (highest leverage, paper §5 prose blocker).
3. Smoke test fig2f with `python3 scripts/analysis/figures/fig2f_first_divergence.py`. Verify visual matches paper §5 prose claim.
4. Implement fig3c_latency_per_step.py (paper §1 hook visualization).
5. Smoke test fig3c.
6. Implement fig2b/2c/2d/2e in parallel (similar template).
7. Update `Makefile` `_figures` target with 6 new invocations.
8. Run `make analysis FAST=1` end-to-end, verify all 14+6 = 20 figures generated.
9. Visual sanity check: all 6 new PNGs render with no blank panels (or N/A pending where partial data).

Total estimated changes: 6 new files (~150-250 LOC each), 1 Makefile diff (+6 lines). ~20-25K tokens of code output.

---

## Out of scope

- Don't refactor existing figures (fig0c/0d/0e/etc).
- Don't add new aggregator scripts (use existing `axis1_microbehavior.json` + `condition_summary_v2.json` data).
- Don't modify `axis1_microbehavior.py` aggregator (read existing fields only).
- Don't change `_aggregate` target.
- Don't change `_per_run_all` target.
- Don't add Phase 3 sub-codes beyond §3 framework (no NEW evidence layer — just visualize existing).
- Don't touch advisor sync / next_steps / paper_planning docs (manual update post-codex).
