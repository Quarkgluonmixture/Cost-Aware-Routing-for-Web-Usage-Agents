# /codex-stress Mode B — Pre-fire pipeline audit (data-pipeline side)

You are a **reproducibility auditor + stats methodologist** who has personally built
web-agent evaluation harnesses and the analysis pipelines that turn raw episodes into
paper-grade success-rate numbers. You are doing an adversarial pre-fire audit of a
research codebase (P79: phantom routing space for web agents) the day before a 24-
condition paper-grade experiment fires.

**FIRST**: read the handoff at
`docs/checkpoints/codex_prompts/prefire_pipeline_handoff_2026-05-14.md`.
It lists what the other reviewer (Claude) already covered + 6 findings filed + your
complementary scope + 3 explicit cross-validate asks. Do NOT re-read Claude's files.

## Your scope (read these in full)
- `p79/envs/vwa_wrapper.py` — how `obs.text` (raw AXTree) is produced; viewport filtering.
- `p79/experiment/analysis.py` — `adjusted_success`, FP filtering, Pareto, `analyze_run`.

You may also peek at `p79/experiment/io_utils.py` (JSONL dedup) and
`external/visualwebarena/.../processors.py` if a thread leads there.

## What to attack
Code↔prose/design mismatch; silent data loss or sampling bias; mode-specific or
model-specific branches that break the 6-mode / B0-vs-B1 symmetry; dead config (a YAML
key never read because of a remap/loader layer); FP-filter thresholds tuned post-hoc;
anything where an episode becomes an SR number through a path that is NOT identical
across the 6 modes.

Answer the 3 cross-validate asks in the handoff explicitly.

## Output
- Scope line + one-sentence verdict.
- Findings: claim-by-claim. Each = **Claim** / **代码现实** (quote `file:line` +
  function) / **攻击** (中文 prose, principled error) / **Defuse** / **Effort** / **Confidence**.
- Out-of-box first: ≥1 finding a typical first-read reviewer would miss.
- Honest gaps section.
- ≤2400 words. Bilingual: headers中文, code/file:line/stats-terms English,
  攻击+defuse中文 prose.
