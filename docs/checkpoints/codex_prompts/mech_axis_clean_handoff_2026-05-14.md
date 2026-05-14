# Audit scope handoff (Claude → codex Mode B) — mechanistic axis-cleanliness

## Claude scope (already read)
- `scripts/analysis/stage4_pca_cosine_gap.py` (Method 4.2)
- `scripts/mechanistic/run_stage2b_continuation_pilot.py` (patching)
- `scripts/mechanistic/run_stage4_multimode_extract.py` (v2 NPZ extraction)
- `p79/mechanistic/extract_hidden_states.py` (extraction substrate)
- `scripts/analysis/stage4_logit_lens_axis2.py` (logit lens)
- `scripts/mechanistic/run_stage4_method44_v2_sweep.py` (Method 4.4 steering)

## Claude findings filed
- **F1 [P0 OOB]** `run_stage4_method44_v2_sweep.py:69-71` `build_som_marks` is a crude AXTree line-grep (`startswith("[") and "]" in s[:6]`), NOT the production `_extract_text_marks`. The steering DIRECTION is computed from the v2-fixed NPZ (clean), but the P-SoM EVAL BASELINE (`psom_inputs`) is generated from this buggy formatter → "v2" is a half-fix, direction-clean / baseline-dirty.
- **F2 [P0 OOB]** H-d cells = `som→dom` patching = flips image+text+prompt (3 axes), but §5 prose calls it "axis-1 (text-format) patching".
- **F3 [P0 OOB]** `vision` mode = vision-prompt + EMPTY text + image → every `X vs Vision` cosine pair confounds 3 axes; "image axis 0.04-0.07" leans on Vision pairs; clean image pair `som↔phantom_som` is only ~0.04.
- F4 [P1] `stage4_pca_cosine_gap.py:201` hardcodes "cls" in the MD f-string.
- F5 [P1] patching script has no axis-isolation guard / provenance doesn't record axis-count.
- F6 [P1] logit lens docstring says input `hidden_states.npz` (v1) but default is `_v2_fixed.npz`.
- F7 [P2] Method 4.4 completeness = binary `overlap_psom > overlap_dom`.

## Codex scope (assigned, complementary — do NOT re-read Claude's 6 files)
Persona: **reproducibility auditor + mechinterp implementer**.
1. `p79/experiment/som.py` — the production `_extract_text_marks`. **Key question**: the extract scripts claim `build_som_marks` is "production-aligned" — is it actually byte-identical to what `qwen3vl_agent.py` feeds the model in a real run? Or is even the "production-aligned" version subtly off (max_marks cap, ordering, label truncation)?
2. `p79/mechanistic/activation_patching.py` — `patching_grid_continuation` + `steered_generate` + `ActivationPatcher`. **Key question**: does the patching/steering MECHANISM do what's claimed? Last-token hook position correctness; `steered_generate` — does α scale the direction the way Method 4.4 prose assumes; layer-index `patcher.layers[L]` vs NPZ `H[:,L+1,:]` off-by-one actually consistent in code.
3. `scripts/analysis/stage4_axis2_layer_profile.py` — Claude did NOT read this. Check axis-pair definitions + whether it shares the per-task-paired helper or re-implements.

## Cross-validate targets
- Claude flagged F1 (method44 build_som_marks is v1-buggy). Please grep ALL of `scripts/mechanistic/` + `scripts/analysis/` for `build_som_marks` / `_extract_text_marks` / `startswith("[")` and report which scripts use production vs which use a local buggy variant — confirm or extend the sibling-propagation map.
- Claude could not verify whether `_extract_text_marks` itself matches production agent input. That's your #1.
