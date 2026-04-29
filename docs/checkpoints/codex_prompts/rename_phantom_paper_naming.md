# Codex prompt: Rename phantom run dirs + paper-facing display labels

## 用途

After P-prompt mode added (§106), "phantom" naming became ambiguous:
- Legacy `phantom` alone defaulted to phantom_som (P-SoM) — now ambiguous vs phantom_prompt
- Legacy `phantom_dom` paper-facing label "P-DOM" → updated paper convention is "P-text"
  (text-payload swap, not DOM-prompt swap; clearer per paper_planning §3 4-Layer)

User wants paper-facing rename to disambiguate. Internal mode_id stays unchanged
(yaml `observation_mode: phantom_dom`, agent `phantom_dom` key) for backward compat
with already-recorded step JSONL data.

## Scope

### 1. Filesystem rename (DONE dirs ONLY, skip in-flight)

```
results/visualwebarena/phase1/B0_phantom_classifieds_20260426    → B0_phantom_som_classifieds_20260426
results/visualwebarena/phase1/B0_phantom_reddit_20260428         → B0_phantom_som_reddit_20260428
results/visualwebarena/phase1/B1_phantom_classifieds_20260428    → B1_phantom_som_classifieds_20260428
results/visualwebarena/phase1/B0_phantom_dom_classifieds_20260427 → B0_phantom_text_classifieds_20260427
results/visualwebarena/phase1/B0_phantom_dom_reddit_20260427     → B0_phantom_text_reddit_20260427
```

Use `git mv` if tracked (results/ is gitignored, so just `mv`).

**SKIP these in-flight dirs (active runners writing to them)**:
- `B1_phantom_dom_classifieds_20260429` (active runner PID 2280869, queue_phantom_dom.sh just launched 18:31 today)
- `B0_phantom_prompt_reddit_20260429` (active runner PID 2075552, B0 P-prompt reddit ~52/210)
- `B0_dom_shopping_20260428` (active runner PID 1106560, ~465/466)

After in-flight runs finish, user will trigger a follow-up rename pass.
Document this caveat in completion report.

### 2. Update path references in Python scripts

For every renamed dir above, update path strings in:
- `scripts/analysis/axis_effect_size.py` (STEP_DIRS dict)
- `scripts/analysis/axis1_microbehavior.py` (STEP_DIRS dict)
- `scripts/analysis/mechanism_per_task.py` (STEP_DIRS dict)
- `scripts/analysis/aggregate_cost_electricity.py` (RUNS dict)
- `scripts/analysis/aggregate_sr_fp_per_mode.py` (B0_RUNS / B1_RUNS or similar)
- `scripts/analysis/aggregate_phantom_lift.py` (run dir lookup)
- `scripts/analysis/aggregate_routing_auroc.py`
- `scripts/analysis/aggregate_cross_site.py`
- `scripts/analysis/layered_status.py` (MODE_SPECS dict)
- `scripts/analysis/figures/fig0c_drop_one_oracle.py`
- `scripts/analysis/figures/fig0c_phantom_lift_bars.py`
- `scripts/analysis/figures/fig0d_taskpool_jaccard.py`
- `scripts/analysis/figures/fig0e_category_mode_heatmap.py`
- `scripts/analysis/figures/fig0f_overlap_stacked_bar.py`
- `scripts/analysis/figures/fig0g_routing_auroc_heatmap.py`
- `scripts/analysis/figures/fig1ab_cascade_diamond.py`
- `scripts/analysis/figures/fig1c_strategy_gradient.py`
- `scripts/analysis/figures/fig2_micro_divergence_heatmap.py`
- `scripts/analysis/figures/fig3a_token_cost_intra_baseline.py`
- `scripts/analysis/figures/fig3d_cost_sr_frontier.py`
- `scripts/analysis/figures/fig_capability_b0_b1.py`
- `scripts/analysis/figures/fig3_regional_carbon.py`
- `Makefile` `RUN_DIRS_PAPER_VWA ?=` list

For DONE dirs (renamed): update path string to new name.
For in-flight dirs (skipped): leave path strings as-is (`B1_phantom_dom_classifieds_20260429` etc.).

### 3. Paper-facing display labels (replace literal strings)

In **docs/** (any .md), **scripts/analysis/** (any .py legend/title strings):

Replace:
- `"Phantom-DOM"` → `"P-text"` (literal display label)
- `"P-DOM"` → `"P-text"` (in markdown text + figure legends)
- `"Phantom DOM"` → `"P-text"` (with space)
- `"phantom_dom"` in **markdown prose** → `"phantom_text"` only when context is paper-display
  - **DO NOT replace** in code paths or yaml config keys (those stay as internal mode_id)
  - **DO NOT replace** in `internal mode_id` references (e.g., `_system_prompts["phantom_dom"]`, agent.py)
  - DO replace in figure legend dicts like `MODE_DISPLAY = {"Phantom-DOM": "P-text"}` — actually those are already P-text, just verify

For "Phantom-SoM" / "P-SoM" — keep as-is (already paper-canonical).
For "Phantom" alone (ambiguous) — only replace where context is unambiguously P-SoM (e.g., "Phantom-SoM is hidden 4th routing arm" stays).

### 4. Aggregate JSON keys — regenerate after rename

After dir renames + py path updates, run:
```bash
make analyze-layered
```

This regenerates:
- `docs/analysis/cross_sites/cost_per_mode.{json,md}` with new dir names in cells
- `docs/analysis/cross_sites/sr_fp_per_mode.{json,md}`
- `docs/analysis/cross_sites/axis_effect_size.{json,md}`
- `docs/analysis/cross_sites/axis1_microbehavior.{json,md}`
- `docs/analysis/cross_sites/mechanism_per_task.{json,md}`
- `docs/analysis/layered_evidence_status.md`
- All figs in `results/phantom_paper/figures/`

### 5. Don't change (preserve backward compat)

- Internal mode_id strings (`phantom_dom`, `phantom_som`, `phantom_prompt`) in:
  - `p79/agents/qwen3vl_agent.py` `_system_prompts` keys
  - `p79/agents/proxy_api_agent.py` same
  - `p79/experiment/som.py` mode dispatch
  - All yaml configs `observation_mode:` field
- Queue script names (`queue_phantom_dom.sh`, `queue_phantom_som.sh`, `queue_phantom_prompt.sh`)
- Yaml config filenames (`exp_v2_B0_phantom_dom_*.yaml`)
- Step JSONL records' `observation_mode` field (immutable historical data)

This deferral lets in-flight runs continue writing using the old internal mode_id while new paper writeup uses paper-facing labels P-text/P-SoM/P-prompt.

## Verification

After rename:
- [ ] `ls results/visualwebarena/phase1/` shows 5 renamed dirs + 3 unchanged in-flight dirs
- [ ] `make analyze-layered` passes (path strings all updated)
- [ ] `python -m py_compile scripts/analysis/**/*.py` passes
- [ ] In-flight runners still alive (PIDs 2075552, 2280869, 1106560)
- [ ] No "Phantom-DOM" / "P-DOM" left in docs/ except in:
  - quoted historical context (e.g., "§103 N=48 narrative said 'P-DOM = P-SoM'")
  - codex_prompts archive (legacy, intentional)
- [ ] `docs/analysis/cross_sites/site_mechanism_dictionary.{json,md}` regenerated or in-place updated to use P-text label
- [ ] `paper_planning.md` Legacy index 04-29 entry mentions both old + new names for traceability

## Output

- Renamed run dirs (filesystem)
- Updated path strings in ~25 .py files
- Updated display labels in ~15-20 .md files + figure scripts
- Regenerated 8+ aggregate JSON/MD outputs
- Brief completion report listing in-flight dirs deferred for later rename

## Token budget

~70-100K (read 25 py + 20 md + execute mv + grep+replace + run aggregates + verify)

## Trigger command

```bash
~/.npm-global/bin/codex exec --skip-git-repo-check \
  -C /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents \
  -s danger-full-access \
  --output-last-message logs/codex_rename_paper_naming.last.md \
  - < docs/checkpoints/codex_prompts/rename_phantom_paper_naming.md \
  > logs/codex_rename_paper_naming.run.log 2>&1 &
```
