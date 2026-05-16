# Project context for Gemini CLI

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

## Gemini-specific tools

- You have `--yolo` full-shell access when invoked with `gemini --yolo -p "<prompt>"`. Use Bash freely for file reads, `wc -c`, grep, etc. Don't ask permission for read-only ops.
- For long file reads (e.g. `master_bug_catalog.md` ~50KB+), prefer `tail -200` or `grep -A 30 "§147"` over dumping the whole file (wastes token budget).
- When invoked as `/gemini-stress` Mode C, always end output with `=== GEMINI_<task>_DONE ===` marker for Tier 1 file-marker monitor compatibility.
