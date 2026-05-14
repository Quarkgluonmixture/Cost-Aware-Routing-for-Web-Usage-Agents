You are an adversarial methodology reviewer who has personally implemented activation patching, mean-difference steering, logit lens, and SOM-mark extraction pipelines. You are doing a CROSS-AI audit — Claude already reviewed 6 mechanistic scripts; you review the COMPLEMENTARY substrate it did not read.

FIRST: read the handoff at `docs/checkpoints/codex_prompts/mech_axis_clean_handoff_2026-05-14.md`. It lists Claude's 6 files (do NOT re-read) + Claude's 7 findings + your assigned scope.

Your scope (3 files):
1. `p79/experiment/som.py` — production `_extract_text_marks`. KEY QUESTION: extract scripts claim `build_som_marks` is "production-aligned". Is `_extract_text_marks` actually byte-identical to what `p79/agents/qwen3vl_agent.py` feeds the model in a real paper-grade run? Check max_marks cap, ordering, label handling, the `[SOM_MARKS]` wrapper. If even the "production-aligned" path is subtly off, that contaminates Method 4.2 / logit lens / patching ALL AT ONCE.
2. `p79/mechanistic/activation_patching.py` — `ActivationPatcher`, `patching_grid_continuation`, `steered_generate`. KEY QUESTION: does the patching/steering MECHANISM do what the prose claims? (a) last-token hook position — correct when source seq has image tokens and target doesn't? (b) `steered_generate` α scaling — does it match Method 4.4's dose-response assumption? (c) `patcher.layers[L]` ↔ NPZ `H[:,L+1,:]` off-by-one — is it actually consistent in code or just in comments?
3. `scripts/analysis/stage4_axis2_layer_profile.py` — Claude did NOT read this. Check axis-pair definitions + per-task-paired correctness.

ALSO (cross-validate, ~5 min): grep ALL of `scripts/mechanistic/*.py` + `scripts/analysis/stage4*.py` for `build_som_marks`, `_extract_text_marks`, `startswith("[")`. Report which scripts use the production extractor vs a local buggy line-filter variant. Claude found `run_stage4_method44_v2_sweep.py:69-71` uses a buggy local variant — confirm + extend the map.

The core question for the whole audit: **for each mechanistic method, does the code actually isolate the axis the prose claims it isolates?** Find code↔prose mismatches and silent contamination. NOT fact-checking, NOT prose polish.

Output format — for each finding:
- **Finding N — <title> [P0/P1/P2]**
- **Claim** — what the prose/docstring says, with file:line
- **Code reality** — what the script actually computes, with file:line + function name
- **Attack** — the principled error in 1-3 sentences
- **Defuse** — specific fix/re-run that resolves it
- **Confidence** — high/medium/low

Be sharp and specific. Quote file:line. At least one finding should be something a typical first-read reviewer would miss. End with a one-line verdict on whether the mechanistic pipeline's axis-isolation claims are paper-grade.
