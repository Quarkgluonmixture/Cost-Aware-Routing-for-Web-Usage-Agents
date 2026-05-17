# Project context for AI CLI agents (codex / others)

**Single source of truth: [.claude/CLAUDE.md](.claude/CLAUDE.md)** — read it first.

That file documents:
- P79 project scope (Cost-Aware Routing for Web Usage Agents, paper-1 = phantom routing space on VWA classifieds + reddit + shopping)
- 3-baseline matrix (B0 Qwen3-VL-235B / B1 Qwen3-VL-4B / B2 Gemma3-VL-4B), 6-mode observation grid (dom / som / vision / phantom_{som,dom,text,prompt})
- Code structure (`p79/{agents,backends,envs,experiment,mechanistic,policies,...}`)
- Run conventions (DGX Spark `spark-9ea3` aarch64 + GB10, `.venv/bin/python3`, `setsid nohup` background)
- Experiment launch hard rules (single-baseline-per-site lock, `RESET_BEFORE=1`, queue scripts only — no bare `run_experiment.py`)
- /stress audit protocol (Mode A Claude / Mode B codex / Mode C gemini), v7.4 bilingual + 推荐修改 spec, scope calibration table

For audit / stress-test invocations specifically, also read the per-task **handoff doc** under `docs/checkpoints/codex_prompts/*_handoff_*.md` if one is referenced by the prompt.

For project state at any time (active runs, today's blockers, GLM-managed live status), see `make active` and `docs/checkpoints/PLAYBOOK.md`.

Master bug catalog (cross-AI shared taxonomy + B-number reservation): `docs/reference/master_bug_catalog.md`.

## Current fire state (refresh per invocation)

**Phase 1a has NOT yet fired on the Condenser A100 paper-grade target host as of 2026-05-17.** Paper §1 SR / paper §4 covariate-adjusted claims / any hero number in prose are **NOT yet pinned to published values** — the Phase 1a output will *be* the first published number, not a revision of an existing one.

**Implication for /codex-stress audit framing**:
- Treat A1.x audit findings as **pre-fire correctness fixes** by default. Do NOT frame fixes as "post-publication SR revision blockers" or "requires codex round to re-write paper §1 prose with new SR number" unless explicit prose with a fixed SR number can be cited.
- "Distance to top-tier" / "submission-today probability" should assume current paper-prose-stub state (numbers not in prose), not assume already-public state.
- Fixes that change SR semantics (e.g. exception-path `success=False` hardcode → `needs_reevaluation` + force re-run) just become the canonical published mechanism on Phase 1a fire — no separate paper-revision dependency.

Fire-state authoritative source: `docs/checkpoints/phase1_plan.md` + `docs/checkpoints/PLAYBOOK.md` §1-§2 (GLM-managed live status). If those say Phase 1a `[x]` fired, override above with their content.
