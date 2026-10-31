# Project context for AI CLI agents (codex / others)

**Entry authority:** read [README.md](README.md) for the research scope and final claim boundaries,
[the experiment ledger](docs/checkpoints/台账.md) and [experiment notes](docs/checkpoints/实验笔记.md) for dated operational evidence,
and [GOTCHAS](docs/reference/GOTCHAS.md) before operational work. The former `.claude/CLAUDE.md` entry is absent from this tree.

The following inventory describes the former entry and is retained as historical context, not a replacement launch authorization:
- P79 project scope (Cost-Aware Routing for Web Usage Agents, paper-1 = phantom routing space on VWA classifieds + reddit + shopping)
- 3-baseline matrix (B0 Qwen3-VL-235B / B1 Qwen3-VL-4B / B2 Gemma3-VL-4B), 6-mode observation grid (dom / som / vision / phantom_{som,dom,text,prompt})
- Code structure (`p79/{agents,backends,envs,experiment,mechanistic,policies,...}`)
- Run conventions (DGX Spark `spark-9ea3` aarch64 + GB10, `.venv/bin/python3`, `setsid nohup` background)
- Experiment launch hard rules (single-baseline-per-site lock, `RESET_BEFORE=1`, queue scripts only — no bare `run_experiment.py`)
- /stress audit protocol (Mode A Claude / Mode B codex / Mode C gemini), v7.4 bilingual + 推荐修改 spec, scope calibration table

For audit / stress-test invocations specifically, also read the per-task **handoff doc** under `docs/checkpoints/codex_prompts/*_handoff_*.md` if one is referenced by the prompt.

For current work, inspect `make active` and the dated checkpoint evidence rather than assuming an old run cursor is live. The former PLAYBOOK is retained as [a retired historical document](docs/archive/PLAYBOOK_retired_20260523.md).

Master bug catalog (cross-AI shared taxonomy + B-number reservation): `docs/reference/master_bug_catalog.md`.

## Historical fire-state note — 2026-05-17

The following note records the pre-fire audit framing as of 2026-05-17; it is not the current run or publication state. Current research claims follow README and its final-dissertation evidence pointers.

**Phase 1a has NOT yet fired on the Condenser A100 paper-grade target host as of 2026-05-17.** Paper §1 SR / paper §4 covariate-adjusted claims / any hero number in prose are **NOT yet pinned to published values** — the Phase 1a output will *be* the first published number, not a revision of an existing one.

**Implication for /codex-stress audit framing**:
- Treat A1.x audit findings as **pre-fire correctness fixes** by default. Do NOT frame fixes as "post-publication SR revision blockers" or "requires codex round to re-write paper §1 prose with new SR number" unless explicit prose with a fixed SR number can be cited.
- "Distance to top-tier" / "submission-today probability" should assume current paper-prose-stub state (numbers not in prose), not assume already-public state.
- Fixes that change SR semantics (e.g. exception-path `success=False` hardcode → `needs_reevaluation` + force re-run) just become the canonical published mechanism on Phase 1a fire — no separate paper-revision dependency.

The [Phase 1 plan](docs/checkpoints/phase1_plan.md) remains as dated planning evidence; its former `PLAYBOOK.md` live-status companion is retired. Do not use this historical note to override the current dissertation evidence or infer a new run authorization.
