**范围 / Verdict**
Scope: `p79/envs/vwa_wrapper.py`, `p79/experiment/analysis.py`, with narrow peeks at `io_utils.py`, VWA `processors.py`, and runner fields feeding `adjusted_success`. Verdict: primary adjusted SR path is mostly uniform after episode summaries exist, but there are pre-fire risks in “raw AXTree” provenance, parser-dependent FP filtering, and 6-mode reporting.

**交叉验证答案**
1. Claude F1: `vwa_wrapper.py` does **not** truncate `obs.text` or cap element count. It **does** force viewport-only AXTree through `current_viewport_only=True`; F1 should be restated as “untruncated viewport AXTree” vs agent-side `max_obs_chars=12000`, not full-page AXTree.
2. F2-style dead config in `analysis.py`: I did not find YAML FP/cost thresholds read by `analysis.py`. It reads VWA task configs for N/A IDs, not experiment YAML. Cost-per-1k is runner-side, not analyzer-side.
3. `analysis.py` SR **does** depend on `agent_finished` / parse-fallback metadata. This interacts directly with B0-only GLM rescue and B1 token-cap parse failures.

**发现 1（OOB）**
Claim: “raw AXTree” is not raw; wrapper injects live-DOM option text before any mode consumes `obs.text`.

代码现实: `p79/envs/vwa_wrapper.py:725-778` (`_to_p79_obs`) reads `obs["text"]`, then calls `_inject_select_options` and `_inject_css_dropdown_options`. `p79/envs/vwa_wrapper.py:887-939` (`_inject_select_options`) queries `document.querySelectorAll('select')`. `p79/envs/vwa_wrapper.py:795-873` (`_inject_css_dropdown_options`) scans hidden `ul` menus and appends `[DROPDOWN OPTIONS]`.

攻击: 这不是格式转换，而是观测语义增强。DOM/SOM/phantom text branches receive extra option labels not necessarily present in AXTree text or screenshot pixels. It also changes text length before downstream truncation/mark extraction, so Claude F1 is compounded by a hidden pre-agent mutation.

Defuse: 把它显式命名为 `augmented_axtree`，保存 `raw_obs_text_chars` / `augmented_obs_text_chars`，并确保 six modes either all use the same augmented source or none do. Marks should be extracted from the exact canonical string delivered to text modes.

Effort: M. Confidence: High.

**发现 2**
Claim: Wrapper has no truncation/cap, but viewport filtering is hardwired and not experiment-configurable.

代码现实: `p79/envs/vwa_wrapper.py:72-77` (`VWAWrapper.__init__`) defaults `current_viewport_only=True`; `p79/envs/vwa_wrapper.py:135-140` (`_lazy_init`) passes it into `ScriptBrowserEnv`. `p79/experiment/environment.py:254-262` (`create_environment`) never forwards an `env.current_viewport_only` key. VWA then removes out-of-viewport AXTree nodes at `external/visualwebarena/browser_env/processors.py:449-508` (`fetch_page_accessibility_tree`).

攻击: 这改变 F1 的量级：SoM marks are not from the full page, but from viewport-only, untruncated AXTree. 但更危险的是这个控制面是隐式的；YAML 不能关掉 viewport filtering, so “raw AXTree” prose can drift from actual delivered input without metadata proving the setting.

Defuse: Add `env.current_viewport_only` to config normalization and `create_environment`, write it into `condition_meta.json`, and log per-step `obs_text_chars` / `obs_node_count`.

Effort: S. Confidence: High.

**发现 3**
Claim: Adjusted SR is parser-state dependent, so B0-only GLM fallback and B1 `max_new_tokens=384` can change SR even with identical browser state.

代码现实: `p79/experiment/analysis.py:76-94` (`compute_adjusted_success`) keeps N/A raw success only if `agent_finished`, and downgrades selected successes when `agent_finished=False`. Runner marks fallback finish at `p79/experiment/runner/main.py:1153-1158`, then computes `_agent_finished` at `p79/experiment/runner/main.py:1561-1565`, and feeds it to `compute_adjusted_success` at `p79/experiment/runner/main.py:1583-1589`.

攻击: FP filtering is not just evaluator correction; it imports model output-parse quality into SR. If B0 GLM rescues a malformed finish into a valid finish, B0 can keep credit where B1’s parse-limited malformed finish is marked fallback/non-finish and downgraded. That is a B0/B1 symmetry break on the reported adjusted SR.

Defuse: For paper-grade fire, either disable B0 GLM rescue or give B1 equivalent rescue. At minimum report raw SR, adjusted SR, `parse_valid`, `fallback_finish`, and `glm_fallback_used` stratified by backend × mode.

Effort: M. Confidence: High.

**发现 4**
Claim: `program_html` FP filter defines “effective action” too narrowly.

代码现实: `p79/experiment/analysis.py:91-94` (`compute_adjusted_success`) downgrades `program_html` success when `has_effective_action=False`. Runner sets `_has_eff` only for `("type", "select_option")` at `p79/experiment/runner/main.py:1578-1580`.

攻击: Many VWA `program_html` tasks are click-causal: add to cart, wishlist, delete, submit. A click-only successful episode that fails to emit/parse a final finish can be downgraded as “no meaningful action.” This will bias modes/backends that complete browser work but more often miss final JSON/finish formatting.

Defuse: Define effective action as any non-navigation action with `agent_visible_changed=True`, or include `click` when evaluator state changed. Then rerun FP deltas before launch.

Effort: S-M. Confidence: Medium-High.

**发现 5**
Claim: Phase-1 analysis plot silently drops phantom modes.

代码现实: `p79/experiment/analysis.py:1351-1360` (`_plot_phase1`) writes a table with all available rows, but `p79/experiment/analysis.py:1365-1369` hardcodes `mode_order = ["dom", "som", "vision"]`.

攻击: A 24-condition / 6-mode fire can produce a headline “Phase 1 Representation Screening” plot that excludes `phantom_som`, `phantom_text`, `phantom_prompt`. CSV is safer, but paper figures or quick decisions from PNGs become selection-biased.

Defuse: Use explicit six-mode order and assert every expected mode appears, or title the legacy plot “3-mode only.”

Effort: S. Confidence: High.

**发现 6**
Claim: Some success-derived cost summaries remain raw while SR is adjusted.

代码现实: `p79/experiment/analysis.py:1210-1227` overwrites `cond_df["success_rate"]` with adjusted SR. But `p79/experiment/metrics.py:381-388` (`aggregate_condition_metrics`) documents `cost_efficiency_ratio` as raw-success based and says analysis does not recompute it.

攻击: Primary Pareto uses adjusted `success_rate`, so this is not fatal for SR. But any table using `cost_efficiency_ratio` mixes raw success economics with adjusted success conclusions.

Defuse: Add `cost_efficiency_ratio_adjusted` from `ep_df["adjusted_success"]`, keep raw separately.

Effort: S. Confidence: High.

**诚实缺口**
I did not run a full episode replay or inspect Claude’s already-covered agent/config files. I sampled task configs only to sanity-check `program_html` semantics. I did not verify downstream paper scripts outside `analysis.py` that may consume generated CSV/PNG outputs.