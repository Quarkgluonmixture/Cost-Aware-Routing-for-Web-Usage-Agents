**Finding 1 — “Production-Aligned” SoM Text Is Not Production-Identical [P0]**

**Claim** — `scripts/mechanistic/run_stage4_multimode_extract.py:46-56` says `build_som_marks` is “production-aligned” and equivalent to patching extraction.

**Code reality** — Production SoM input is not just `_extract_text_marks`. Real runs call `prepare_observation_for_mode` in `p79/experiment/runner/main.py:840-870`, clone `obs_prep.som_text` into the observation at `p79/experiment/runner/main.py:529-531`, and `Qwen3VLAgent.step` passes that text through unchanged for SoM modes at `p79/agents/qwen3vl_agent.py:431-436`. The production builder adds `[OPTIONS]` / `[DROPDOWN OPTIONS]` lines via `_options_map` at `p79/experiment/som.py:222-247`, then wraps with `[SOM_MARKS]...[/SOM_MARKS]` at `p79/experiment/som.py:249`.

**Attack** — The mechanistic “fixed” formatter uses `_extract_text_marks` and reconstructs `[id=N] label`, but omits production dropdown option recovery. That means the hidden-state NPZ is not byte-identical to paper-grade agent input on pages with select/dropdown annotations, contaminating Method 4.2, logit lens, patching, and steering wherever SoM/P-SoM text is involved.

**Defuse** — Export a public `build_som_text_from_obs_text(obs_text, max_marks=200)` from `p79.experiment.som` that includes `_options_map`, wrapper, cap, empty case, and exact line format. Re-run all v2 hidden-state extraction and downstream analyses from that substrate.

**Confidence** — high

**Finding 2 — Buggy Local SoM Line-Filter Propagates Beyond Method 4.4 [P0]**

**Claim** — `scripts/mechanistic/run_stage4_format_variation_extract.py:83` says its baseline `[SOM_MARKS]` is “same as run_stage2b build_som_marks.”

**Code reality** — It actually keeps raw AXTree lines with `line.strip().startswith("[") and "]" in line.strip()[:6]` at `scripts/mechanistic/run_stage4_format_variation_extract.py:82-85`. Same local filter appears in `scripts/mechanistic/run_stage4_method44_v2_sweep.py:69-71`, `scripts/mechanistic/run_stage4_method44_steering.py:63-69`, and `scripts/mechanistic/diag_stage4_method44_layer_check.py:58-64`.

**Attack** — This is not production SoM text: no `[SOM_MARKS]` wrapper, no `[id=N]` label syntax, no empty-block behavior, no max-mark cap, no dropdown options. It is AXTree-shaped text selected by a bracket prefix heuristic, so any “P-SoM baseline” built from it is off-axis.

**Defuse** — Delete local formatters and import the single production SoM text builder. Mark existing Method 4.4 / format-variation outputs from these scripts as invalid or legacy.

**Confidence** — high

**Finding 3 — Steering α Is Not a Unit Dose [P1]**

**Claim** — `p79/mechanistic/activation_patching.py:141-142` says `direction` is a vector and `alpha` is scalar magnitude, with `1 = unit direction`.

**Code reality** — `steered_generate` casts the raw vector and adds `alpha * dir_vec` directly at `p79/mechanistic/activation_patching.py:149-160`. The v2 sweep computes raw mean differences from NPZ as `H[psom][:, L + 1, :].mean(0) - H[dom][:, L + 1, :].mean(0)` at `scripts/mechanistic/run_stage4_method44_v2_sweep.py:172`.

**Attack** — α is a multiplier on an unnormalized mean-difference vector, not a unit-norm dose. Layer-to-layer and site-to-site dose-response curves are therefore confounded by direction norm unless the prose defines α as “multiples of the empirical mean difference.”

**Defuse** — Either normalize `direction / ||direction||` before steering and report α as activation-norm dose, or keep raw mean-diff steering but rewrite Method 4.4 and plots as “k × empirical MD vector” with norm reported per layer.

**Confidence** — high

**Finding 4 — Legacy Steering Is Still Off By One [P1]**

**Claim** — The corrected convention is `patcher.layers[L] ↔ H[:, L+1, :]`, stated in `scripts/mechanistic/run_stage4_method44_v2_sweep.py:20-21` and `:125-130`.

**Code reality** — The v2 sweep is consistent: it uses `H[:, L + 1, :]` at `scripts/mechanistic/run_stage4_method44_v2_sweep.py:172` and calls `steered_generate(layer_idx=L, ...)` at `:256-257`. But legacy `scripts/mechanistic/run_stage4_method44_steering.py` computes `H[:, layer, :]` at `:83-85` and applies it to `patcher.layers[args.layer]` at `:163-164`.

**Attack** — The current v2 path fixed the off-by-one; the legacy steering path did not. Any result from `run_stage4_method44_steering.py` claiming “L17” is actually mixing NPZ hidden index 17, i.e. block 16 output, into hook block 17.

**Defuse** — Retire the legacy script or patch it to use `H[:, layer + 1, :]`, `hidden_states_v2_fixed.npz`, and the shared production SoM builder.

**Confidence** — high

**Finding 5 — Continuation Patching Does Not Fully Propagate Through the Hooked Layer Cache [P1]**

**Claim** — `patched_generate` says the patched first-token hidden state “propagates through KV cache” at `p79/mechanistic/activation_patching.py:85-89`; `patching_grid_continuation` treats layer L as causal if multi-token output moves sourceward at `:361-369`.

**Code reality** — The hook fires only once at `p79/mechanistic/activation_patching.py:98-104` and returns `(hs_patched,) + layer_output[1:]` at `:105-106`, preserving the original target `past_key_values` for the hooked layer. `steered_generate` has the same pattern at `:154-162`.

**Attack** — For cached generation, the post-block residual is patched, but the hooked layer’s own K/V cache remains target-derived. Downstream layer caches can reflect the patch, but the hooked layer cache itself does not, so continuation effects are not a clean “source state at layer L for the whole generation” intervention.

**Defuse** — For continuation claims, either patch layer input/pre-attention so K/V are recomputed from the intervention, explicitly patch `past_key_values`, or disable cache and reapply the hook on each decode step. Otherwise describe this as a one-shot final-prompt-state intervention.

**Confidence** — medium-high

**Finding 6 — Axis-2 Layer Profile Is Row-Paired, Not Per-Task Averaged [P1]**

**Claim** — `scripts/analysis/stage4_axis2_layer_profile.py:58-66` and `:137-142` claim per-task paired cosine gap with task-level bootstrap.

**Code reality** — It inner-joins correctly via `paired_rows` at `scripts/analysis/stage4_axis2_layer_profile.py:76`, but `paired_cosine_gap_per_layer` averages over paired rows at `scripts/analysis/_paired_npz_helpers.py:87-91`. The bootstrap resamples task IDs, then concatenates all rows for selected tasks and again averages rows at `scripts/analysis/_paired_npz_helpers.py:117-119`.

**Attack** — If tasks have unequal surviving step counts, longer trajectories get more weight. That is not the prose’s per-task estimand and can silently shift layer peaks toward tasks with more extracted steps.

**Defuse** — Compute cosine gaps per `(task, step)`, average within each task first, then average/bootstrap over task means. Or assert equal step counts per task before using the current row estimator.

**Confidence** — high

**Cross-Validation Map**

Production extractor wrapper, but not full production byte identity because dropdown options are omitted: `curate_mirage_tasks.py:94-99`, `run_stage1_pilot.py:89-100`, `run_stage2_patching_pilot.py:78-83`, `run_stage2b_continuation_pilot.py:121-126`, `run_stage4_multimode_extract.py:46-63`.

Extractor-based variants, not production wrapper: `run_stage4_h1_qwen2vl.py:58-88`, `run_stage4_h1_phi35.py:63-94`.

Buggy local bracket line-filter: `run_stage4_method44_v2_sweep.py:69-71`, `run_stage4_method44_steering.py:63-69`, `diag_stage4_method44_layer_check.py:58-64`, `run_stage4_format_variation_extract.py:82-85`.

`stage4_axis2_layer_profile.py` pair definitions are axis-clean: DOM↔P-prompt and P-text↔P-SoM isolate prompt family; DOM↔P-text and P-prompt↔P-SoM isolate text format; P-SoM↔SoM isolates image presence. The estimator, not the pair definition, is the problem.

Verdict: not paper-grade yet; the axis-isolation story is plausible in places, but the SoM substrate drift, legacy bracket filters, raw-dose steering, and row-weighted “per-task” layer profile leave silent contamination in core mechanistic claims.