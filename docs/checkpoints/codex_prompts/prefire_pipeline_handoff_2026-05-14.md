# Audit scope handoff (Claude → codex Mode B) — Pre-fire pipeline audit 2026-05-14

Context: Phase 1a paper-grade rerun (24 conditions = cls+red × B0+B1 × 6 modes)
about to fire via `queue_phase1_paper_grade.sh launch`. Goal: catch B-82-class
silent defects (declared-mode ≠ actually-delivered-input) BEFORE the fire.

## Claude scope (already read — do NOT re-read)
- `p79/experiment/som.py` — mode dispatch + canonical SoM builder
- `p79/agents/qwen3vl_agent.py` (B1) + `p79/agents/proxy_api_agent.py` (B0)
- `p79/backends/local_qwen.py` + `p79/backends/api_proxy.py` — config remap wrappers
- `p79/experiment/config.py` + `configs/exp_v2_base.yaml` + 4 Phase-1a configs
- `p79/experiment/conditions.py` + `p79/experiment/modules.py`
- `p79/experiment/runner/main.py` (partial: obs-clone + agent-construction paths)
- `scripts/queues/queue_phase1_paper_grade.sh` + `queue_phantom_prompt.sh` + `queue_chain.sh` (grep)

## Claude findings filed (6)
- F1 [P0 OOB] **axis-1 truncation/cap asymmetry**: for som/phantom_som/phantom_text,
  `[SOM_MARKS]` is built by `_extract_text_marks(obs_text, max_marks=200)` from the
  FULL untruncated AXTree (runner main.py:861 → som.py). For dom/phantom_prompt the
  agent then truncates `obs.text` at `max_obs_chars=12000` chars INSIDE `agent.step()`.
  → AXTree modes see `rawAXTree[:12000]`; marks modes see first-200-marks-from-whole-page.
  Not "same information, different format" — differential page coverage on the paper's
  main axis-1.
- F2 [P0 OOB] **revision dead-config**: `local_qwen.py` builds `agent_cfg["model"]`
  WITHOUT a `revision` key → `qwen3vl_agent.py` falls back to hard-coded SHA. The
  `model.revision` block added to `exp_v2_base.yaml` (codex C8 fix, 2026-05-14) never
  reaches the agent — merged config decoupled from actually-loaded SHA.
- F3 [P1] **GLM-fallback scaffold asymmetry**: `use_glm_fallback: true` on every B0
  config, no B1 equivalent → B0 parse failures get GLM rescue, B1's don't (SR-affecting,
  not just cost).
- F4 [P1] B1 `max_new_tokens: 384` sits below agent's own documented parse-safe
  envelope (~400-1500 tok).
- F5 [P2] reference-images still injected into phantom "no-image" modes.
- F6 [P2 gap] B0 token count from API `usage`, B1 from local tokenizer — cross-model
  cost basis differs.

## Codex scope (assigned — complementary, data-pipeline side)
Persona: **reproducibility auditor + stats methodologist**.
Read and audit:
- `p79/envs/vwa_wrapper.py` (~939 lines) — how `obs.text` (the raw AXTree) is actually
  PRODUCED + viewport filtering. Known prior issue: §80 `in_viewport_ratio` operator-
  precedence bug at `processors.py:218`. Is there ANY truncation / element-count cap /
  viewport filter in the wrapper that COMPOUNDS Claude's F1 axis-1 asymmetry?
- `p79/experiment/analysis.py` (~1675 lines) — `adjusted_success` canonical definition,
  FP filtering (na_fp / eval_fp / visual_fp), Pareto, `analyze_run`. Does the SR/FP
  computation treat the 6 modes symmetrically? Any mode-specific or model-specific
  branch in how an episode becomes a paper-grade SR number?

## Cross-validate targets (explicit asks)
1. Claude's F1: does `vwa_wrapper.py` apply its own truncation or element cap to
   `obs.text` before the runner ever sees it? If yes, F1's magnitude changes.
2. Claude's F2 pattern (config key set in YAML but never read by consumer because of a
   remap layer): grep `analysis.py` for any config field it reads — is there a similar
   dead-config (e.g. cost-per-1k, FP thresholds) set in YAML under a path the analyzer
   doesn't actually read?
3. Whole-pipeline coherence: does `analysis.py` SR depend on `max_steps` / parse-error
   handling in a way that interacts with F3 (B0-only GLM rescue) or F4 (B1 384-token cap)?

Output: claim-by-claim, quote `file:line` + function name. Bilingual (headers中文,
code English, attack/defuse中文 prose). ≤2400 words. Write final review via `-o` flag.
