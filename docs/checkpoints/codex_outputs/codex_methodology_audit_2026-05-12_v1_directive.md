Reading prompt from stdin...
OpenAI Codex v0.128.0 (research preview)
--------
workdir: /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: high
reasoning summaries: none
session id: 019e1e21-f253-73d3-beb0-299d95b78bed
--------
user
# Codex methodology audit — paper-grade protocol soundness check

You are a senior mechanistic-interpretability researcher (think: David Bau, Neel Nanda, or Adam Lopez peer reviewing for ICLR mechinterp workshop). You have implemented activation patching, mean-difference steering, logit lens, and PCA-based representation probing many times. You are **annoyed** by mechinterp papers that:

- Hide layer-indexing off-by-one bugs behind table-formatted L0-L36 conventions
- Compute mean-difference vectors over imbalanced N without bias correction
- Patch at one token position and call it a "circuit" claim
- Apply Holm-Bonferroni with inconsistent family-wise m parameters across cells
- Extract hidden states without specifying token position / step / pre-vs-post-layernorm
- Use sign-ambiguous PCA components without sign-flip discipline
- Mix steering layer-index conventions between papers and codebases

**Your job today**: read the **methodology** of this paper-1 mechanism work, not its claims. Find protocol bugs, hidden assumptions, statistical errors, multiple-comparison violations, multimodal-specific gotchas (Qwen3-VL has vision projector + cross-attention; non-trivial interactions with residual-stream extraction). Be technical. Be specific. Cite line numbers from code + plan + paper.

**This is DIFFERENT from a claim-level reviewer audit**. Do not focus on "is this claim supported by enough evidence". Focus on "is the evidence correctly computed / extracted / tested".

## 🚫 Independence requirement

Do NOT read these files (they contain prior Claude or codex audit output and would anchor your view):
- `docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md` (prior codex claim-level audit)
- `.claude/skills/stress/SKILL.md` / `.claude/skills/codex-stress/SKILL.md`
- `docs/checkpoints/process/stress_skill_replica.md` / `docs/checkpoints/process/codex_stress_skill_replica.md`
- Any conversation context / session memory

You are doing an **independent methodology audit** from a methodology-expert perspective.

## Read order

### Code (most important — methodology lives in code, not prose)

1. `scripts/mechanistic/run_stage2b_continuation_pilot.py` — activation patching script (Stage 2/3, Exp 5)
   - How are source hidden states extracted? At which token position? Which step?
   - How is injection performed? Pre or post LayerNorm? Single-token or full-stream?
   - How is cache invalidation handled? Are downstream layers re-computed correctly after injection?
   - Is `--random-inject` matched to source variance per layer or globally?
   - Is final-token-replacement at the input-token position, or at a later generation step?

2. `scripts/analysis/stage4_pca_cosine_gap.py` — Method 4.2 cosine gap
   - How is the cosine gap computed? Numerator and denominator?
   - Is PCA fit per-layer or once over all layers? Sign-flip discipline?
   - How are AUROC values computed? On per-task means or per-example?
   - Per-task vs aggregate normalization?

3. `scripts/analysis/stage4_axis2_layer_profile.py` — Exp 1 layer profile (axis-2 hierarchy)
4. `scripts/analysis/stage4_logit_lens_axis2.py` — Exp 3 logit lens KL
   - Which RMSNorm and lm_head are applied? `model.model.language_model.norm` vs `model.norm`?
   - Are per-layer mean hidden states a sufficient summary, or does this collapse task-level structure?
   - KL is between which two distributions? Mode-mean vs mode-mean?

5. `scripts/analysis/stage2_layer_significance.py` (if exists) — Holm-Bonferroni correction
   - Family-wise m parameter — what is m? Set of layers in grid (36 or 37)? Or smaller subset?
   - Is the same m used across cells (cell A/B/C/D/E vs F/G vs Cr/Dr/Er vs H-t/H-p/H-d)?

6. `scripts/analysis/hero_claim_bootstrap.py` — bootstrap CI (claim-level not methodology, but check seeding)

7. `p79/mechanistic/activation_patching.py` (if exists) — patching primitive
8. Any hooks or layer indexing code in `p79/mechanistic/`

### Documentation (cross-reference against code)

1. `docs/checkpoints/mechanism/plan.md` §3-§5 — method descriptions
2. `docs/checkpoints/paper_drafts/section5_mechanism.md` §5.2, §5.3, §5.4, §5.7 — method prose
3. `docs/checkpoints/mechanism/results/*.md` — spot-check claimed methodology details against code

## Specific things to check (illustrative, not checklist)

### Layer indexing
- Is layer 0 the embedding-block output or the post-embedding-layernorm output or the first decoder block?
- Does prose say "L0-L36, where L0 is the embedding-block output and L1-L36 are the 36 transformer decoder block outputs" (per section5_mechanism.md:7)?
- Does code agree? Does `hidden_states[0]` correspond to the embedding output, and is there an off-by-one between code indexing and paper L-index?

### Token position in patching
- For activation patching of `phantom_som → phantom_text`: at which token position is the cached hidden state injected? Final input token? Last assistant prefix token? First generation token?
- Does the injection persist through generation, or only at the first forward pass?
- If only at first forward, is the downstream cache (post-injection forward) actually used? Or do KV-cache updates re-overwrite the injection?

### Hidden state extraction for cosine gap / PCA
- Which token position is averaged over for the per-step mode-mean? Single final input token? Mean over all input tokens? Mean over assistant prefix?
- Does Qwen3-VL place image tokens before, after, or interleaved with text? Are image-token hidden states included or excluded?
- For `--source-mode phantom_som` (claim: no image): is the image actually removed from `inputs_embeds` and from positional embeddings, or only masked? What does extracted hidden state include if image tokens still occupy positions?

### Random injection control specificity
- Is `--random-inject` Gaussian matched to source mean+variance per-layer per-task, or globally?
- Is the same random tensor reused across the 36-layer grid, or fresh per layer? (Both have failure modes.)
- Does seeding with `--random-seed 42` actually make the run byte-reproducible?

### Multiple-comparison correction
- Holm-Bonferroni m = number of layers tested = 36? 37? Per-task family or per-cell family?
- If reported "best layer" is taken from same data used for Holm correction, is this a post-hoc selection bias? Was best-layer pre-registered or selected after seeing per-layer p-values?

### PCA + cosine gap
- For Method 4.2 cosine gap: is gap = `1 - cos(mean_a, mean_b)` or `||mean_a - mean_b|| / ||mean_a + mean_b||`?
- Is per-task de-meaning applied before computing the mode-mean? (If not, individual-task variance dominates.)
- AUROC on what? Per-example projection score? Per-task mean projection?

### Steering (Method 4.4)
- Mean-difference vector: computed as `mean(P-SoM hidden) - mean(DOM hidden)` at which layer? Same hidden states as cosine gap, or different?
- α scaling: applied to the unit-normalized direction or to the raw mean-difference?
- Decoding side effects: does steering break JSON envelope because of (a) the steering magnitude, (b) the layer position, or (c) the fact that no token-position constraint is applied?

### Logit lens
- `model.model.language_model.norm` vs `model.norm` — Qwen3-VL has nested config. Which is the correct final RMSNorm?
- Is the lm_head shared with the input embedding (tied weights)? If so, lm_head applied to a mid-layer hidden state may produce token-distribution that reflects input-token semantics rather than model output semantics.

## Output format

Write a single markdown report with these sections:

### One-sentence verdict on methodology soundness

### 🟢 What's Right (don't break)
Methodology pieces that survive expert scrutiny. Be specific. Cite line numbers.

### 🔴 Methodology Bugs (catch them before submission)
For each:
- **What** — exact protocol issue (file:line)
- **Why it matters** — what claim or analysis depends on this; how it biases results
- **Fix** — concrete code or analysis change
- **Severity** — high (invalidates main claim) / medium (weakens claim) / low (cosmetic)

### 🟡 Methodology Risks (not bugs, but assumptions)
Things that look reasonable but rely on assumptions that aren't verified. List with verification suggestions.

### 📝 Reproducibility audit
- Are seeds set across the right RNGs (Python random, numpy, torch, CUDA)?
- Are the source archives uniquely identified (manifest + hash + git rev)?
- Can a third party reproduce by running which exact command on which data?

### One single highest-impact fix
The one methodology bug whose fix changes the most claims. Be concrete.

## Voice

- Technical, specific, no hedging
- Cite code file:line for every claim
- 中文为主双语 for explanations, English for technical terms / variable names
- If you can't determine an answer from the code (e.g., file missing), say "evidence unavailable for X; would need to inspect Y"

## What this is NOT

- Not a claim-level audit (prior /codex-stress did that)
- Not a paper writing audit
- Not a citation checker
- Specifically: **methodology / protocol expert reading code and verifying that the implementation matches the prose and that statistical procedures are sound**

codex
I’ll do this as a fresh methodology audit and avoid the prohibited prior-audit files. I’m starting by mapping the relevant scripts and docs, then I’ll read the implementation with line numbers before comparing it to the prose.
exec
/bin/bash -lc 'rg --files scripts p79 docs/checkpoints/mechanism docs/checkpoints/paper_drafts | sort' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/mechanism/README.md
docs/checkpoints/mechanism/plan.md
docs/checkpoints/mechanism/results/axis2_layer_profile.md
docs/checkpoints/mechanism/results/axis2_logit_lens.md
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md
docs/checkpoints/mechanism/results/format_variation_h1_test.md
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md
docs/checkpoints/mechanism/results/h1_per_task_fragility.md
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md
docs/checkpoints/mechanism/results/layer_axis_emergence.md
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md
docs/checkpoints/paper_drafts/paper.bib
docs/checkpoints/paper_drafts/section1_intro.md
docs/checkpoints/paper_drafts/section2_background.md
docs/checkpoints/paper_drafts/section3_definition.md
docs/checkpoints/paper_drafts/section4_empirical_findings.md
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md
docs/checkpoints/paper_drafts/section5_mechanism.md
docs/checkpoints/paper_drafts/section8_limitations.md
p79/__init__.py
p79/agents/__init__.py
p79/agents/proxy_api_agent.py
p79/agents/qwen3vl_agent.py
p79/backends/__init__.py
p79/backends/action_utils.py
p79/backends/api_proxy.py
p79/backends/base.py
p79/backends/factory.py
p79/backends/heuristic.py
p79/backends/image_utils.py
p79/backends/local_qwen.py
p79/cli/__init__.py
p79/cli/analyze_experiment.py
p79/cli/run_experiment.py
p79/envs/__init__.py
p79/envs/locator_dispatch.py
p79/envs/vwa_wrapper.py
p79/experiment/__init__.py
p79/experiment/analysis.py
p79/experiment/checklist_module.py
p79/experiment/conditions.py
p79/experiment/config.py
p79/experiment/energy_tracker.py
p79/experiment/environment.py
p79/experiment/io_utils.py
p79/experiment/logger_v2.py
p79/experiment/metrics.py
p79/experiment/modules.py
p79/experiment/router.py
p79/experiment/runner/__init__.py
p79/experiment/runner/helpers.py
p79/experiment/runner/main.py
p79/experiment/schema_migrations/__init__.py
p79/experiment/schema_migrations/v2.py
p79/experiment/som.py
p79/experiment/state_change.py
p79/experiment/tasks.py
p79/experiment/types.py
p79/logging/__init__.py
p79/mechanistic/__init__.py
p79/mechanistic/activation_patching.py
p79/mechanistic/extract_hidden_states.py
p79/mechanistic/linear_probe.py
p79/policies/__init__.py
p79/utils/__init__.py
p79/utils/asyncio_workarounds.py
p79/utils/auth_refresh.py
p79/utils/log_cleanup.py
p79/utils/torch_cuda_workarounds.py
scripts/README.md
scripts/analysis/README.md
scripts/analysis/aggregate_cost_electricity.py
scripts/analysis/aggregate_cross_site.py
scripts/analysis/aggregate_failure_modes.py
scripts/analysis/aggregate_phantom_lift.py
scripts/analysis/aggregate_phantom_meta.py
scripts/analysis/aggregate_routing_auroc.py
scripts/analysis/aggregate_sr_fp_per_mode.py
scripts/analysis/analyze_comment_selflink_loop.py
scripts/analysis/analyze_comment_selflink_loop_v2.py
scripts/analysis/analyze_confidence_calibration.py
scripts/analysis/analyze_cross_representation.py
scripts/analysis/analyze_experiment.py
scripts/analysis/analyze_noninteractive_click_earlystop.py
scripts/analysis/analyze_reason_diagnostics.py
scripts/analysis/analyze_reddit_selflink_cycle.py
scripts/analysis/analyze_search_over_browse.py
scripts/analysis/axis1_microbehavior.py
scripts/analysis/axis_effect_size.py
scripts/analysis/b0_vision_coordinate_errors.py
scripts/analysis/collect_analysis_summary.py
scripts/analysis/compare_b0_b1.py
scripts/analysis/compare_pilot_t0_vs_paper_grade.py
scripts/analysis/diag_pattern_match.py
scripts/analysis/figures/fig0a_sr_per_mode_heatmap.py
scripts/analysis/figures/fig0b_extra_confidence_calibration.py
scripts/analysis/figures/fig0b_fp_rate_per_mode.py
scripts/analysis/figures/fig0c_drop_one_oracle.py
scripts/analysis/figures/fig0c_phantom_lift_bars.py
scripts/analysis/figures/fig0d_taskpool_jaccard.py
scripts/analysis/figures/fig0e_category_mode_heatmap.py
scripts/analysis/figures/fig0f_overlap_stacked_bar.py
scripts/analysis/figures/fig0g_routing_auroc_heatmap.py
scripts/analysis/figures/fig1ab_cascade_diamond.py
scripts/analysis/figures/fig1c_strategy_gradient.py
scripts/analysis/figures/fig2_micro_divergence_heatmap.py
scripts/analysis/figures/fig2b_target_hit_rate.py
scripts/analysis/figures/fig2c_keyword_repeat.py
scripts/analysis/figures/fig2d_first_action_divergence.py
scripts/analysis/figures/fig2e_cross_site_validity.py
scripts/analysis/figures/fig2f_first_divergence.py
scripts/analysis/figures/fig3_regional_carbon.py
scripts/analysis/figures/fig3a_token_cost_intra_baseline.py
scripts/analysis/figures/fig3b_image_token_gap.py
scripts/analysis/figures/fig3c_latency_per_step.py
scripts/analysis/figures/fig3d_cost_sr_frontier.py
scripts/analysis/figures/fig_failure_modes_per_cell.py
scripts/analysis/figures/fig_forest_drop_one.py
scripts/analysis/figures/fig_mechanism_pilot.py
scripts/analysis/figures/fig_meta_forest.py
scripts/analysis/figures/fig_phantom_structure_venn.py
scripts/analysis/hero_claim_bootstrap.py
scripts/analysis/layered_status.py
scripts/analysis/lib/__init__.py
scripts/analysis/lib/run_registry.py
scripts/analysis/mechanism_per_task.py
scripts/analysis/power_analysis.py
scripts/analysis/preregistration_decision_test.py
scripts/analysis/reaggregate_method44_v2_hmean.py
scripts/analysis/replot_stage2_figure.py
scripts/analysis/sensitivity_loo_meta.py
scripts/analysis/stage2_heterogeneity_figure.py
scripts/analysis/stage2_layer_significance.py
scripts/analysis/stage2_transfer_correlation.py
scripts/analysis/stage4_axis2_layer_profile.py
scripts/analysis/stage4_axis2_per_task_fragility.py
scripts/analysis/stage4_format_variation_analyze.py
scripts/analysis/stage4_h1_per_task_fragility.py
scripts/analysis/stage4_layer_axis_emergence.py
scripts/analysis/stage4_logit_lens_axis2.py
scripts/analysis/stage4_pca_cosine_gap.py
scripts/analysis/stage4_robustness.py
scripts/analysis/stage4_w6_l04_tokenization.py
scripts/analysis/validate_run.py
scripts/maintenance/README.md
scripts/maintenance/active_processes.py
scripts/maintenance/annotate_screenshots.py
scripts/maintenance/auto_pull_myriad_cell.sh
scripts/maintenance/check_disk_usage.sh
scripts/maintenance/cleanup_logs.py
scripts/maintenance/clear_tasks.py
scripts/maintenance/create_b1_classifieds_stub.py
scripts/maintenance/crontab.txt
scripts/maintenance/dead_link_check.py
scripts/maintenance/digest_enrich.py
scripts/maintenance/experiment_watchdog.py
scripts/maintenance/generate_gallery.py
scripts/maintenance/glm/error_scan.py
scripts/maintenance/glm/glm_batch_digest.py
scripts/maintenance/glm/glm_cell_autoupdate.py
scripts/maintenance/glm/glm_diagnosis_sidecar.py
scripts/maintenance/glm/glm_playbook_refresh.py
scripts/maintenance/glm/glm_pre_launch_check.py
scripts/maintenance/glm/myriad_watcher.py
scripts/maintenance/launch.sh
scripts/maintenance/notify_on_fail.sh
scripts/maintenance/probe_b01_b13_self_verify.py
scripts/maintenance/probe_b08_b06_self_replay.py
scripts/maintenance/probe_b37_api_determinism.py
scripts/maintenance/probe_som_occlusion.py
scripts/maintenance/probe_tier10_dispatch_target.py
scripts/maintenance/rederive_episode_summary.py
scripts/maintenance/reeval_phase1.py
scripts/maintenance/refresh_gallery.sh
scripts/maintenance/reset_vwa_sites.sh
scripts/maintenance/restart_watchdog.sh
scripts/maintenance/retry_b1_single_task.sh
scripts/maintenance/rsync_results_from_hub.sh
scripts/maintenance/rsync_results_to_hub.sh
scripts/maintenance/run_one_vwa_episode.py
scripts/maintenance/smoke_test_vwa.py
scripts/maintenance/split_wa_tasks.py
scripts/maintenance/trigger_watchdog_status.sh
scripts/maintenance/wait_for_reddit_then_rederive.sh
scripts/mechanistic/curate_mirage_tasks.py
scripts/mechanistic/diag_stage4_method44_layer_check.py
scripts/mechanistic/extract_archive_subset.py
scripts/mechanistic/run_stage1_pilot.py
scripts/mechanistic/run_stage2_patching_pilot.py
scripts/mechanistic/run_stage2b_continuation_pilot.py
scripts/mechanistic/run_stage4_format_variation_extract.py
scripts/mechanistic/run_stage4_h1_phi35.py
scripts/mechanistic/run_stage4_h1_qwen2vl.py
scripts/mechanistic/run_stage4_method44_steering.py
scripts/mechanistic/run_stage4_method44_v2_sweep.py
scripts/mechanistic/run_stage4_multimode_extract.py
scripts/myriad/smoke_compute.qsub
scripts/myriad/smoke_login.sh
scripts/preflight_v2.sh
scripts/provenance/numerical_determinism_check.py
scripts/provenance/snapshot_env.py
scripts/provenance/snapshot_vwa.sh
scripts/queues/_deprecated/queue_b0_wa_with_reset.sh
scripts/queues/_deprecated/queue_b0_with_reset.sh
scripts/queues/_deprecated/queue_b1_wa_with_reset.sh
scripts/queues/_deprecated/queue_b1_with_reset.sh
scripts/queues/_deprecated/queue_phantom_pair.sh
scripts/queues/qsub_2x2_fwd_revtasks_myriad.sh
scripts/queues/qsub_2x2_rev_strongtasks_myriad.sh
scripts/queues/qsub_celle_random_cls_strong_myriad.sh
scripts/queues/qsub_cellf_fwd_reddit_strong_myriad.sh
scripts/queues/qsub_cellg_rev_reddit_reverse_myriad.sh
scripts/queues/qsub_curate_reddit_myriad.sh
scripts/queues/qsub_stage2_cellcr_reddit_fwd_revtier.sh
scripts/queues/qsub_stage2_celldr_reddit_rev_strongtier.sh
scripts/queues/qsub_stage2_celler_reddit_fwd_random.sh
scripts/queues/qsub_stage2b_myriad.sh
scripts/queues/qsub_stage2c_myriad.sh
scripts/queues/qsub_stage3_cellhd_cls_fwd_dom.sh
scripts/queues/qsub_stage3_cellhd_red_fwd_dom.sh
scripts/queues/qsub_stage3_cellhp_cls_fwd_prompt.sh
scripts/queues/qsub_stage3_cellhp_red_fwd_prompt.sh
scripts/queues/qsub_stage3_cellhprompt_cls.sh
scripts/queues/qsub_stage3_cellhprompt_cls_rand.sh
scripts/queues/qsub_stage3_cellhprompt_red.sh
scripts/queues/qsub_stage3_cellhprompt_red_rand.sh
scripts/queues/qsub_stage3_cellht_cls_fwd_text.sh
scripts/queues/qsub_stage3_cellht_red_fwd_text.sh
scripts/queues/qsub_stage4_format_variation_cls.sh
scripts/queues/qsub_stage4_h1_cls_reverse.sh
scripts/queues/qsub_stage4_h1_reddit.sh
scripts/queues/qsub_stage4_multimode_extract_cls.sh
scripts/queues/qsub_stage4mm_reddit.sh
scripts/queues/queue_16cell_paper_grade.sh
scripts/queues/queue_baseline.sh
scripts/queues/queue_chain.sh
scripts/queues/queue_phantom_prompt.sh
scripts/queues/queue_phantom_som.sh
scripts/queues/queue_phantom_text.sh
scripts/queues/queue_pilot_t0.sh
scripts/queues/run_scroll_comparison.sh
scripts/run_experiment.py
scripts/setup/a100_self_host_vwa.sh
scripts/setup/myriad_bootstrap.sh
scripts/vwa/import_vwa_assets.sh
scripts/vwa/setup_vwa.sh
scripts/vwa/start_vwa_docker.sh
scripts/vwa_env.sh
scripts/vwa_env_remote.sh
scripts/vwa_env_remote.sh.example

exec
/bin/bash -lc 'git status --short && git rev-parse HEAD' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
 M docs/analysis/cross_sites/sr_fp_per_mode.json
 M docs/analysis/cross_sites/sr_fp_per_mode.md
 D docs/reference/MYRIAD_SMOKE_REPORT.md
 D docs/reference/PAPER_STRATEGY_OPEN_QUESTIONS.md
 D docs/reference/RUNPOD_ONBOARDING.md
?? docs/checkpoints/codex_outputs/codex_methodology_audit_2026-05-12.md
?? docs/checkpoints/codex_prompts/codex_methodology_audit_2026-05-12.md
?? scripts/analysis/stage2_transfer_correlation.py
738517c2e5ae4fb0df2e2fe5b4ffcb209ed1b556

codex
The repo has an untracked `docs/checkpoints/codex_outputs/codex_methodology_audit_2026-05-12.md`; I’m treating that as potentially anchoring and not reading it. I’m going straight to source scripts and the paper/plan files requested.
exec
/bin/bash -lc 'wc -l scripts/mechanistic/run_stage2b_continuation_pilot.py scripts/analysis/stage4_pca_cosine_gap.py scripts/analysis/stage4_axis2_layer_profile.py scripts/analysis/stage4_logit_lens_axis2.py scripts/analysis/stage2_layer_significance.py scripts/analysis/hero_claim_bootstrap.py p79/mechanistic/activation_patching.py p79/mechanistic/extract_hidden_states.py docs/checkpoints/mechanism/plan.md docs/checkpoints/paper_drafts/section5_mechanism.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   521 scripts/mechanistic/run_stage2b_continuation_pilot.py
   224 scripts/analysis/stage4_pca_cosine_gap.py
   190 scripts/analysis/stage4_axis2_layer_profile.py
   259 scripts/analysis/stage4_logit_lens_axis2.py
   364 scripts/analysis/stage2_layer_significance.py
   253 scripts/analysis/hero_claim_bootstrap.py
   468 p79/mechanistic/activation_patching.py
   204 p79/mechanistic/extract_hidden_states.py
   466 docs/checkpoints/mechanism/plan.md
   176 docs/checkpoints/paper_drafts/section5_mechanism.md
  3125 total

exec
/bin/bash -lc 'rg -n "hidden_states|output_hidden_states|layer|layers|patch|inject|random|seed|cache|past_key_values|position|token|RMSNorm|norm|lm_head|PCA|cosine|AUROC|Holm|Bonferroni|bootstrap|manifest|hash|git|revision|L0|L36" scripts/mechanistic/run_stage2b_continuation_pilot.py scripts/analysis/stage4_pca_cosine_gap.py scripts/analysis/stage4_axis2_layer_profile.py scripts/analysis/stage4_logit_lens_axis2.py scripts/analysis/stage2_layer_significance.py scripts/analysis/hero_claim_bootstrap.py p79/mechanistic/activation_patching.py p79/mechanistic/extract_hidden_states.py docs/checkpoints/mechanism/plan.md docs/checkpoints/paper_drafts/section5_mechanism.md docs/checkpoints/mechanism/results/*.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
scripts/analysis/stage4_logit_lens_axis2.py:2:"""Exp 3: Logit lens at late layers — does axis-2 prompt-family signal
scripts/analysis/stage4_logit_lens_axis2.py:3:re-emerge in output distribution even though mid-layer residual stream
scripts/analysis/stage4_logit_lens_axis2.py:6:Method: Apply Qwen3-VL-4B's lm_head + final_norm to each per-layer hidden
scripts/analysis/stage4_logit_lens_axis2.py:7:state mean, get a token distribution per (mode, layer). For each axis-2
scripts/analysis/stage4_logit_lens_axis2.py:9:  - top-1 token disagreement rate per layer
scripts/analysis/stage4_logit_lens_axis2.py:10:  - KL divergence (P-text || P-SoM) per layer
scripts/analysis/stage4_logit_lens_axis2.py:11:  - log-prob gap on canonical SoM-prompt vs DOM-prompt action tokens
scripts/analysis/stage4_logit_lens_axis2.py:15:axis-2 cosine gap is 0.011 at L23 but output KL is large at L30-L36,
scripts/analysis/stage4_logit_lens_axis2.py:16:prompt prior is amplified by late-layer decoding into different output.
scripts/analysis/stage4_logit_lens_axis2.py:19:  results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz
scripts/analysis/stage4_logit_lens_axis2.py:20:  results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz
scripts/analysis/stage4_logit_lens_axis2.py:23:  docs/checkpoints/mechanism/results/axis2_logit_lens.md
scripts/analysis/stage4_logit_lens_axis2.py:24:  results/phantom_paper/figures/fig_axis2_logit_lens.png
scripts/analysis/stage4_logit_lens_axis2.py:38:DEFAULT_CLS_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz"
scripts/analysis/stage4_logit_lens_axis2.py:39:DEFAULT_RED_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz"
scripts/analysis/stage4_logit_lens_axis2.py:40:DEFAULT_MD = ROOT / "docs/checkpoints/mechanism/results/axis2_logit_lens.md"
scripts/analysis/stage4_logit_lens_axis2.py:41:DEFAULT_FIG = ROOT / "results/phantom_paper/figures/fig_axis2_logit_lens.png"
scripts/analysis/stage4_logit_lens_axis2.py:54:def load_lm_head_and_norm(device="cuda"):
scripts/analysis/stage4_logit_lens_axis2.py:55:    """Load Qwen3-VL-4B lm_head + final_norm from HF cache (offline)."""
scripts/analysis/stage4_logit_lens_axis2.py:59:    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
scripts/analysis/stage4_logit_lens_axis2.py:60:    print(f"  loading Qwen3VLForConditionalGeneration (lm_head + norm only)")
scripts/analysis/stage4_logit_lens_axis2.py:64:    # Qwen3-VL structure (verified via p79/mechanistic/activation_patching.py):
scripts/analysis/stage4_logit_lens_axis2.py:65:    #   model.model.language_model.layers  (36 decoder layers, no embedding included)
scripts/analysis/stage4_logit_lens_axis2.py:66:    #   model.model.language_model.norm    (final RMSNorm, sibling of layers)
scripts/analysis/stage4_logit_lens_axis2.py:67:    #   model.lm_head                       (top-level projection)
scripts/analysis/stage4_logit_lens_axis2.py:68:    norm = model.model.language_model.norm
scripts/analysis/stage4_logit_lens_axis2.py:69:    lm_head = model.lm_head
scripts/analysis/stage4_logit_lens_axis2.py:70:    print(f"  norm: {type(norm).__name__}, lm_head: {type(lm_head).__name__}")
scripts/analysis/stage4_logit_lens_axis2.py:71:    return tokenizer, lm_head, norm, model
scripts/analysis/stage4_logit_lens_axis2.py:75:def logits_at_layer(hidden: torch.Tensor, lm_head, norm) -> torch.Tensor:
scripts/analysis/stage4_logit_lens_axis2.py:76:    """hidden: (D,) → logits (V,) after final_norm + lm_head."""
scripts/analysis/stage4_logit_lens_axis2.py:77:    h = hidden.unsqueeze(0).to(lm_head.weight.device).to(lm_head.weight.dtype)
scripts/analysis/stage4_logit_lens_axis2.py:78:    h = norm(h)
scripts/analysis/stage4_logit_lens_axis2.py:79:    logits = lm_head(h).squeeze(0)
scripts/analysis/stage4_logit_lens_axis2.py:80:    return logits
scripts/analysis/stage4_logit_lens_axis2.py:83:def kl_divergence(p_logits, q_logits) -> float:
scripts/analysis/stage4_logit_lens_axis2.py:84:    """KL(P || Q) with softmax on logits."""
scripts/analysis/stage4_logit_lens_axis2.py:85:    log_p = torch.log_softmax(p_logits, dim=-1)
scripts/analysis/stage4_logit_lens_axis2.py:86:    log_q = torch.log_softmax(q_logits, dim=-1)
scripts/analysis/stage4_logit_lens_axis2.py:92:def top1_agree(p_logits, q_logits) -> bool:
scripts/analysis/stage4_logit_lens_axis2.py:93:    return torch.argmax(p_logits).item() == torch.argmax(q_logits).item()
scripts/analysis/stage4_logit_lens_axis2.py:96:def compute_pair_logit_lens(npz: Path, pair_pairs: list, lm_head, norm, n_layers_use: int):
scripts/analysis/stage4_logit_lens_axis2.py:98:    H = d["hidden_states"]  # (N, L, D)
scripts/analysis/stage4_logit_lens_axis2.py:111:        layer_kl = []
scripts/analysis/stage4_logit_lens_axis2.py:112:        layer_disagree = []
scripts/analysis/stage4_logit_lens_axis2.py:113:        for L in range(n_layers_use):
scripts/analysis/stage4_logit_lens_axis2.py:116:            l_a = logits_at_layer(h_a, lm_head, norm)
scripts/analysis/stage4_logit_lens_axis2.py:117:            l_b = logits_at_layer(h_b, lm_head, norm)
scripts/analysis/stage4_logit_lens_axis2.py:118:            layer_kl.append(kl_divergence(l_a, l_b))
scripts/analysis/stage4_logit_lens_axis2.py:119:            layer_disagree.append(0.0 if top1_agree(l_a, l_b) else 1.0)
scripts/analysis/stage4_logit_lens_axis2.py:121:            "kl": np.array(layer_kl),
scripts/analysis/stage4_logit_lens_axis2.py:122:            "disagree": np.array(layer_disagree),
scripts/analysis/stage4_logit_lens_axis2.py:124:            "peak_kl_L": int(np.argmax(layer_kl)),
scripts/analysis/stage4_logit_lens_axis2.py:125:            "peak_kl": float(np.max(layer_kl)),
scripts/analysis/stage4_logit_lens_axis2.py:127:    return result, n_layers_use
scripts/analysis/stage4_logit_lens_axis2.py:130:def write_md(cls_axis2, cls_axis1, red_axis2, red_axis1, n_layers, out: Path):
scripts/analysis/stage4_logit_lens_axis2.py:133:        "# Exp 3 — Logit lens at late layers (axis-2 vs axis-1)",
scripts/analysis/stage4_logit_lens_axis2.py:135:        "Apply Qwen3-VL-4B's final_norm + lm_head to per-layer per-mode mean hidden states.",
scripts/analysis/stage4_logit_lens_axis2.py:136:        "For each axis-isolated pair, compute KL(mode_a || mode_b) and top-1 token disagreement",
scripts/analysis/stage4_logit_lens_axis2.py:137:        "across all 37 layers. This probes whether axis-2 cosine signal (L23 peak 0.011) gets",
scripts/analysis/stage4_logit_lens_axis2.py:138:        "amplified into output distribution divergence by late-layer decoding.",
scripts/analysis/stage4_logit_lens_axis2.py:189:        "- **H_A (axis-2 absent from output)**: axis-2 KL flat <0.1 at all layers → prompt-family",
scripts/analysis/stage4_logit_lens_axis2.py:190:        "  effect bypasses logit lens, only visible via attention heads or runtime decoding.",
scripts/analysis/stage4_logit_lens_axis2.py:191:        "- **H_B (axis-2 amplified at output)**: axis-2 KL peak at L30+ ≫ cosine 0.011 magnitude →",
scripts/analysis/stage4_logit_lens_axis2.py:192:        "  late-layer decoding amplifies prompt prior into output divergence (Wu et al. tool calling",
scripts/analysis/stage4_logit_lens_axis2.py:194:        "- **H_C (axis-2 tracks residual stream)**: axis-2 KL peak at L23 same as cosine peak →",
scripts/analysis/stage4_logit_lens_axis2.py:195:        "  prompt prior signal proportional to mid-layer geometry, no amplification.",
scripts/analysis/stage4_logit_lens_axis2.py:198:        "axis-1 KL magnitudes to see whether 3-4x ratio in cosine space persists at output level.",
scripts/analysis/stage4_logit_lens_axis2.py:204:def plot(cls_a2, cls_a1, red_a2, red_a1, n_layers, out: Path):
scripts/analysis/stage4_logit_lens_axis2.py:207:    layers = np.arange(n_layers)
scripts/analysis/stage4_logit_lens_axis2.py:211:            ax.plot(layers, info["kl"], color="#d62728" if "DOM" in label else "#ff7f0e",
scripts/analysis/stage4_logit_lens_axis2.py:214:            ax.plot(layers, info["kl"], color="#1f77b4" if "DOM" in label else "#2ca02c",
scripts/analysis/stage4_logit_lens_axis2.py:222:    axes[0].set_ylabel("KL divergence (logit lens)")
scripts/analysis/stage4_logit_lens_axis2.py:223:    fig.suptitle("Exp 3: Output distribution divergence per layer via logit lens", fontsize=11)
scripts/analysis/stage4_logit_lens_axis2.py:239:    tokenizer, lm_head, norm, _ = load_lm_head_and_norm(device="cuda")
scripts/analysis/stage4_logit_lens_axis2.py:242:    cls_a2, n_L = compute_pair_logit_lens(args.cls_npz, AXIS_2_PAIRS, lm_head, norm, 37)
scripts/analysis/stage4_logit_lens_axis2.py:246:    cls_a1, _ = compute_pair_logit_lens(args.cls_npz, AXIS_1_PAIRS, lm_head, norm, 37)
scripts/analysis/stage4_logit_lens_axis2.py:250:    red_a2, _ = compute_pair_logit_lens(args.red_npz, AXIS_2_PAIRS, lm_head, norm, 37)
scripts/analysis/stage4_logit_lens_axis2.py:252:    red_a1, _ = compute_pair_logit_lens(args.red_npz, AXIS_1_PAIRS, lm_head, norm, 37)
scripts/analysis/stage2_layer_significance.py:2:"""Stage 2B/2C layer-resolved significance test.
scripts/analysis/stage2_layer_significance.py:4:Question: Is the apparent mid-layer (L11-L17) disruption in
scripts/analysis/stage2_layer_significance.py:5:patching_continuation_results.json statistically significant, or could
scripts/analysis/stage2_layer_significance.py:9:1. Load per-task per-layer results from forward (2B) and reverse (2C) runs.
scripts/analysis/stage2_layer_significance.py:10:2. For each layer L_n in [0, 5, 11, 17, 23, 29], paired t-test (and Wilcoxon
scripts/analysis/stage2_layer_significance.py:15:3. Holm-Bonferroni correction across the 6 tested layers (per direction).
scripts/analysis/stage2_layer_significance.py:19:Output: markdown summary written to results/mechanistic/layer_significance_<date>.md
scripts/analysis/stage2_layer_significance.py:31:# F12 audit fix 2026-05-09: L0-L35 are **transformer block outputs**, NOT
scripts/analysis/stage2_layer_significance.py:32:# embedding output. The patching hooks (`activation_patching.py`
scripts/analysis/stage2_layer_significance.py:35:#   L0  = output of block 0 (first decoder layer post-norm). Often
scripts/analysis/stage2_layer_significance.py:39:#         construction since only final_norm + lm_head remain to produce
scripts/analysis/stage2_layer_significance.py:40:#         target tokens.
scripts/analysis/stage2_layer_significance.py:41:# If you need true embedding output (n_layers+1 indexing), add a hook to
scripts/analysis/stage2_layer_significance.py:42:# `model.model.language_model.embed_tokens` instead.
scripts/analysis/stage2_layer_significance.py:53:    """Return shape (n_tasks, n_layers) ndarray for a given metric.
scripts/analysis/stage2_layer_significance.py:54:    metric ∈ {token_overlap_to_source, token_overlap_to_target, ld_to_source, ld_to_target}.
scripts/analysis/stage2_layer_significance.py:58:        rows.append([pl[metric] for pl in t["per_layer"]])
scripts/analysis/stage2_layer_significance.py:63:    """Holm-Bonferroni step-down. Input p-values, output (adj_p, reject@0.05)."""
scripts/analysis/stage2_layer_significance.py:83:def per_direction_layer_test(
scripts/analysis/stage2_layer_significance.py:84:    label: str, per_task: list[dict], metric: str = "token_overlap_to_target"
scripts/analysis/stage2_layer_significance.py:86:    """Paired t-test + Wilcoxon for each test-layer vs L35 baseline.
scripts/analysis/stage2_layer_significance.py:88:    Direction convention: H1 = layer disrupts vs baseline. For overlap metrics
scripts/analysis/stage2_layer_significance.py:89:    higher means closer to target → disruption is layer < baseline (alternative='less').
scripts/analysis/stage2_layer_significance.py:91:    layer > baseline (alternative='greater'). We auto-detect from metric name.
scripts/analysis/stage2_layer_significance.py:93:    Returns dict with per-layer rows + Holm-adjusted p-values.
scripts/analysis/stage2_layer_significance.py:103:    rng = np.random.default_rng(seed=42)
scripts/analysis/stage2_layer_significance.py:109:        layer_vals = grid[:, L]
scripts/analysis/stage2_layer_significance.py:110:        diff = layer_vals - baseline
scripts/analysis/stage2_layer_significance.py:112:        # C9 fix: handle constant-column edge case (e.g. cell D L0 has all 1.0
scripts/analysis/stage2_layer_significance.py:120:            t_stat, t_p_one = stats.ttest_rel(layer_vals, baseline, alternative=alt_direction)
scripts/analysis/stage2_layer_significance.py:124:                w_stat, w_p_one = stats.wilcoxon(layer_vals, baseline,
scripts/analysis/stage2_layer_significance.py:129:            # C3 fix: bootstrap percentile 95% CI on mean diff (resample tasks)
scripts/analysis/stage2_layer_significance.py:137:            "layer": L,
scripts/analysis/stage2_layer_significance.py:138:            "mean_layer": float(layer_vals.mean()),
scripts/analysis/stage2_layer_significance.py:159:        "baseline_layer": BASELINE_LAYER,
scripts/analysis/stage2_layer_significance.py:167:    layer: int = 17,
scripts/analysis/stage2_layer_significance.py:168:    metric: str = "token_overlap_to_target",
scripts/analysis/stage2_layer_significance.py:170:    """Welch's t-test: are forward and reverse disruption magnitudes equal at layer L?
scripts/analysis/stage2_layer_significance.py:176:    fwd_diff = fwd_grid[:, layer] - fwd_grid[:, BASELINE_LAYER]
scripts/analysis/stage2_layer_significance.py:177:    rev_diff = rev_grid[:, layer] - rev_grid[:, BASELINE_LAYER]
scripts/analysis/stage2_layer_significance.py:182:        "layer": layer,
scripts/analysis/stage2_layer_significance.py:214:    out.append("Tests whether mid-layer disruption (L11-L17) in continuation patching")
scripts/analysis/stage2_layer_significance.py:216:    out.append("paired t-test of layer L_n vs L35 baseline, one-sided H1: disruption.")
scripts/analysis/stage2_layer_significance.py:217:    out.append("Holm-Bonferroni correction across 6 tested layers per direction.")
scripts/analysis/stage2_layer_significance.py:219:    out.append("Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001 (Holm-adjusted).")
scripts/analysis/stage2_layer_significance.py:224:        out.append(f"N tasks: {direction_results['n_tasks']}, baseline layer: L{direction_results['baseline_layer']}")
scripts/analysis/stage2_layer_significance.py:226:        out.append("| Layer | mean(L) | mean(L35) | Δ mean | Δ std | 95% CI (boot) | t-stat | p (raw) | p (Holm) | reject H0 |")
scripts/analysis/stage2_layer_significance.py:231:                f"| L{r['layer']:>2} | {r['mean_layer']:.3f} | {r['mean_baseline']:.3f} | "
scripts/analysis/stage2_layer_significance.py:243:        out.append(f"**Metric**: `{cross['metric']}` @ L{cross['layer']}")
scripts/analysis/stage2_layer_significance.py:251:    out.append("- **Holm-rejected layers** in forward direction = layers where mid-layer disruption survives multiple-comparison correction.")
scripts/analysis/stage2_layer_significance.py:260:    out.append("  with patching effect size — strong-tier may be effectively the high-")
scripts/analysis/stage2_layer_significance.py:269:    p.add_argument("--fwd-results", default="results/mechanistic/stage2b_curated_b1_cls_myriad/patching_continuation_results.json")
scripts/analysis/stage2_layer_significance.py:270:    p.add_argument("--rev-results", default="results/mechanistic/stage2c_reverse_curated_b1_cls_myriad/patching_continuation_results.json")
scripts/analysis/stage2_layer_significance.py:271:    p.add_argument("--cellc-results", default="results/mechanistic/stage2b_2x2_fwd_revtasks_myriad/patching_continuation_results.json",
scripts/analysis/stage2_layer_significance.py:273:    p.add_argument("--celld-results", default="results/mechanistic/stage2c_2x2_rev_strongtasks_myriad/patching_continuation_results.json",
scripts/analysis/stage2_layer_significance.py:276:                   help="Output markdown path. Default: results/mechanistic/layer_significance_<date>.md")
scripts/analysis/stage2_layer_significance.py:304:        cell_results_overlap.append(per_direction_layer_test(
scripts/analysis/stage2_layer_significance.py:305:            f"{label} — overlap_to_target", ptasks, metric="token_overlap_to_target",
scripts/analysis/stage2_layer_significance.py:307:        cell_results_ld.append(per_direction_layer_test(
scripts/analysis/stage2_layer_significance.py:322:        fwd_per_task, rev_per_task, layer=17,
scripts/analysis/stage2_layer_significance.py:323:        metric="token_overlap_to_target",
scripts/analysis/stage2_layer_significance.py:326:        fwd_per_task, rev_per_task, layer=17,
scripts/analysis/stage2_layer_significance.py:336:        extra.append(f"N={r['n_tasks']}, baseline L{r['baseline_layer']}")
scripts/analysis/stage2_layer_significance.py:338:        extra.append("| Layer | mean(L) | mean(L35) | Δ mean | Δ std | 95% CI (boot) | t-stat | p (raw) | p (Holm) | reject H0 |")
scripts/analysis/stage2_layer_significance.py:343:                f"| L{row['layer']:>2} | {row['mean_layer']:.3f} | {row['mean_baseline']:.3f} | "
scripts/analysis/stage2_layer_significance.py:355:        out_path = Path(f"results/mechanistic/layer_significance_{today}.md")
docs/checkpoints/mechanism/results/axis2_logit_lens.md:1:# Exp 3 — Logit lens at late layers (axis-2 vs axis-1)
docs/checkpoints/mechanism/results/axis2_logit_lens.md:3:Apply Qwen3-VL-4B's final_norm + lm_head to per-layer per-mode mean hidden states.
docs/checkpoints/mechanism/results/axis2_logit_lens.md:4:For each axis-isolated pair, compute KL(mode_a || mode_b) and top-1 token disagreement
docs/checkpoints/mechanism/results/axis2_logit_lens.md:5:across all 37 layers. This probes whether axis-2 cosine signal (L23 peak 0.011) gets
docs/checkpoints/mechanism/results/axis2_logit_lens.md:6:amplified into output distribution divergence by late-layer decoding.
docs/checkpoints/mechanism/results/axis2_logit_lens.md:44:- **H_A (axis-2 absent from output)**: axis-2 KL flat <0.1 at all layers → prompt-family
docs/checkpoints/mechanism/results/axis2_logit_lens.md:45:  effect bypasses logit lens, only visible via attention heads or runtime decoding.
docs/checkpoints/mechanism/results/axis2_logit_lens.md:46:- **H_B (axis-2 amplified at output)**: axis-2 KL peak at L30+ ≫ cosine 0.011 magnitude →
docs/checkpoints/mechanism/results/axis2_logit_lens.md:47:  late-layer decoding amplifies prompt prior into output divergence (Wu et al. tool calling
docs/checkpoints/mechanism/results/axis2_logit_lens.md:49:- **H_C (axis-2 tracks residual stream)**: axis-2 KL peak at L23 same as cosine peak →
docs/checkpoints/mechanism/results/axis2_logit_lens.md:50:  prompt prior signal proportional to mid-layer geometry, no amplification.
docs/checkpoints/mechanism/results/axis2_logit_lens.md:53:axis-1 KL magnitudes to see whether 3-4x ratio in cosine space persists at output level.
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:3:Per-task cosine gap distribution at L23 (axis-2 peak per §5.7 / Exp 1).
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:4:Each task averaged across its 2 steps; cosine gap computed between mode pairs.
docs/checkpoints/mechanism/results/layer_axis_emergence.md:1:# Stage 4: image-axis peak-layer split — Mirage Effect signature
docs/checkpoints/mechanism/results/layer_axis_emergence.md:3:Eight mode pairs differ in image presence (one side has image, the other doesn't). Peak cosine-gap layer reveals **when** image-axis mechanism emerges:
docs/checkpoints/mechanism/results/layer_axis_emergence.md:5:| no-image side | image side | no-img text | peak layer | peak cosine gap |
docs/checkpoints/mechanism/results/layer_axis_emergence.md:7:| DOM | SoM | AXTree | **L04** | 0.0604 |
docs/checkpoints/mechanism/results/layer_axis_emergence.md:8:| DOM | Vision | AXTree | **L04** | 0.0653 |
docs/checkpoints/mechanism/results/layer_axis_emergence.md:9:| P-prompt | SoM | AXTree | **L04** | 0.0600 |
docs/checkpoints/mechanism/results/layer_axis_emergence.md:10:| P-prompt | Vision | AXTree | **L04** | 0.0649 |
docs/checkpoints/mechanism/results/layer_axis_emergence.md:13:| P-text | Vision | [SOM_MARKS] | **L36** | 0.0614 |
docs/checkpoints/mechanism/results/layer_axis_emergence.md:14:| P-SoM | Vision | [SOM_MARKS] | **L36** | 0.0613 |
docs/checkpoints/mechanism/results/layer_axis_emergence.md:20:- DOM ↔ SoM: peak **L04** = 0.0604
docs/checkpoints/mechanism/results/layer_axis_emergence.md:21:- DOM ↔ Vision: peak **L04** = 0.0653
docs/checkpoints/mechanism/results/layer_axis_emergence.md:22:- P-prompt ↔ SoM: peak **L04** = 0.0600
docs/checkpoints/mechanism/results/layer_axis_emergence.md:23:- P-prompt ↔ Vision: peak **L04** = 0.0649
docs/checkpoints/mechanism/results/layer_axis_emergence.md:28:- P-text ↔ Vision: peak **L36** = 0.0614
docs/checkpoints/mechanism/results/layer_axis_emergence.md:30:- P-SoM ↔ Vision: peak **L36** = 0.0613
docs/checkpoints/mechanism/results/layer_axis_emergence.md:34:When the no-image side carries `AXTree` text (DOM, P-prompt), the image-axis cosine gap peaks at **L04** — early-layer fresh image-presence detection (vision encoder + cross-modal fusion).
docs/checkpoints/mechanism/results/layer_axis_emergence.md:36:When the no-image side carries `[SOM_MARKS]` text (P-text, P-SoM), the image-axis cosine gap peak shifts to **L17–L36** — image yes/no divergence is deferred to mid/output layers.
docs/checkpoints/mechanism/results/layer_axis_emergence.md:38:**Mechanism story**: `[SOM_MARKS]` text in input primes an indexed-parsing pathway through mid-layer computation. The model processes marks structurally regardless of whether image is provided, producing image-axis divergence only at late integration stages. This is the direct Method 4.2 empirical anchor for the **Mirage Effect** (Asadi et al. 2026, VLM ~70-80% no-image accuracy) and **Cross-modal flow** (Kaduri et al., middle-layer cross-modal flows store image info in query tokens): the marks-primed mid-layer computation runs *as if image were available*, with image grounding contributed only late.
docs/checkpoints/mechanism/results/layer_axis_emergence.md:42:> *Method 4.2 reveals a peak-layer shift signature for the Mirage Effect: image-axis cosine-gap peak transitions from L04 (when no-image side carries AXTree text) to L17–L36 (when no-image side carries [SOM_MARKS] text). The peak-layer shift quantifies how text-payload format primes mid-layer computation pathways — [SOM_MARKS] format triggers indexed-parsing through mid-layers regardless of image presence, with image-axis divergence deferred to late integration. This identifies [SOM_MARKS] as the mechanism trigger for the Mirage Effect, anchoring Asadi et al. 2026's behavioral finding (~70-80% no-image VLM accuracy) and Kaduri et al.'s middle-layer cross-modal flow hypothesis with layer-resolved empirical evidence.*
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:1:# Hero-claim bootstrap CI (W1 defuse)
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:3:Per-seed bootstrap 95% percentile CI on paired adjusted-SR diffs and drop-one oracle. B=10000, seed=42. Tasks resampled with replacement at task level.
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:18:**Pairwise SR difference, bootstrap 95% CI:**
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:31:**Drop-one oracle on 4-mode set (dom, som, vision, phantom_som), bootstrap 95% CI:**
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:54:**Pairwise SR difference, bootstrap 95% CI:**
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:67:**Drop-one oracle on 4-mode set (dom, som, vision, phantom_som), bootstrap 95% CI:**
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:84:- If both CIs are strict-positive (ci_lo > 0) AND P(diff > 0) > 0.95 → **W1 attack defused**,   §1 hero claim is bootstrap-supported. Remove the '2σ hedge' from line 5, lead with the magnitude.
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:1:# W6 feature attribution — H1 reddit 2/6 marks-like L04 peak
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:3:**Setup**: Qwen3-VL-4B tokenizer (Qwen/Qwen3-VL-4B-Instruct). Each marks-like format variant tokenized on a canonical single-element example (N=1, role=button, label=Submit). First-token character class + marker-fingerprint token count compared between L04-peak and L17-peak subgroups.
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:5:## Per-variant tokenization
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:7:| Variant | Peak | Example | n_tok | First token | First char class | Marker fp |
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:9:| appagent_id | L04 | `id_1: Submit` | 5 | `id` | alphanumeric | 4 (`id·_·1·:`) |
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:10:| plain_numbered | L04 | `1. Submit` | 3 | `1` | alphanumeric | 2 (`1·.`) |
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:15:| hash_id_control | L04 | `#a3f7 Submit` | 5 | `#a` | markup-sigil | 4 (`#a·3·f·7`) |
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:17:| dom | L04 | `button: Submit (AXTree)` | 7 | `button` | alphanumeric | 2 (`button·:`) |
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:24:| L04-peak (2) | 2 | 0 | 0 | 0 | 0 |
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:29:✅ **Hypothesis supported (clean split)**: L04-peak variants both start with alphanumeric tokens (2/2); L17-peak variants start with markup-sigil tokens (4/4).
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:33:- L04-peak mean marker-fp tokens: 3.00
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:34:- L17-peak mean marker-fp tokens: 4.00
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:35:- Δ (L17 − L04): +1.00
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:37:## Full token sequence per variant (marks-like 6)
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:39:- **appagent_id** (L04, `id_1: Submit`): 5 tokens: `id` · `_` · `1` · `:` · `ĠSubmit`
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:40:- **plain_numbered** (L04, `1. Submit`): 3 tokens: `1` · `.` · `ĠSubmit`
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:41:- **som_standard** (L17, `[1] button 'Submit'`): 7 tokens: `[` · `1` · `]` · `Ġbutton` · `Ġ'` · `Submit` · `'`
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:42:- **browser_use_at** (L17, `@1 Submit`): 3 tokens: `@` · `1` · `ĠSubmit`
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:43:- **tarsier_typed** (L17, `[B1:button:Submit]`): 7 tokens: `[B` · `1` · `:` · `button` · `:` · `Submit` · `]`
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:44:- **xml_tagged** (L17, `<el_1 role='button'>Submit</el_1>`): 14 tokens: `<` · `el` · `_` · `1` · `Ġrole` · `='` · `button` · `'>` · `Submit` · `</` · `el` · `_` · `1` · `>`
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:48:Within the 6 marks-like variants, the L17 vs L04 split corresponds to whether the variant's first tokens are **markup-sigil tokens** (`[`, `<`, `@`) — which co-occur with HTML / web-agent traces in pretraining and trigger the visual-grounding shortcut at mid layers — versus **plain alphanumeric tokens** (`id`, `1`) — which are common in prose / dictionary listings and behave like AXTree-DOM, peaking early at L04 where the image-axis divergence is freshly observable but not yet routed through the shortcut path.
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:51:- `hash_id_control` (`#a3f7 Submit`): markup-sigil first token but L04 peak. The `#` sigil alone is not sufficient — the marker must contain an **integer index** (which `#a3f7` does not). This is consistent with prior H2 "integer is the trigger token" framing.
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:52:- `plain_sentence` (`Submit`): alphanumeric first token but L17 peak. With no list/marker structure at all, the divergence path differs — possibly because the text observation drops to bare labels with no positional anchors, which the model handles via a different late-layer routing (likely commitment without grounding).
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:54:Together these say: the L17 mid-layer shortcut requires **(a) integer-indexed marker + (b) markup-sigil-leading delimiter**. Either alone fails to trigger it.
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:56:**Paper §5 implication**: H1's binary 'marks-like vs not' prediction is too coarse. The mechanism trigger is the **conjunction** of integer marker + markup-sigil first token, not the abstract concept of 'indexed list'. Variants like `id_N:` and `N.` are nominally indexed but lack the sigil; `hash_id_control` has the sigil but lacks an integer. Both fail to peak at L17. This refines H1 to **'integer marker + markup-sigil delimiter → triggers shortcut at L17'**, which is testable on additional variants and on a `bare_N` falsifier (drop the bracket from `[N]` and re-extract).
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md:58:**Falsifier (concrete next experiment)**: variant `bare_N` = `N button 'Submit'` (no brackets), which has integer + no sigil. Hypothesis predicts L04 peak. If it peaks L17, hypothesis fails.
scripts/analysis/stage4_axis2_layer_profile.py:2:"""Exp 1: Axis-2 (prompt-family) layer profile.
scripts/analysis/stage4_axis2_layer_profile.py:4:Method 4.2 cosine gap at L17 places (DOM, P-prompt) and (P-text, P-SoM) into
scripts/analysis/stage4_axis2_layer_profile.py:6:contribution to that single layer. The four-fold drop-in property + forest
scripts/analysis/stage4_axis2_layer_profile.py:10:This script asks: across ALL 37 layers, where does prompt-family contribute
scripts/analysis/stage4_axis2_layer_profile.py:24:  - docs/checkpoints/mechanism/results/axis2_layer_profile.md
scripts/analysis/stage4_axis2_layer_profile.py:25:  - results/phantom_paper/figures/fig_axis2_prompt_layer_profile.png
scripts/analysis/stage4_axis2_layer_profile.py:36:DEFAULT_NPZ_CLS = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz"
scripts/analysis/stage4_axis2_layer_profile.py:37:DEFAULT_NPZ_RED = ROOT / "results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz"
scripts/analysis/stage4_axis2_layer_profile.py:38:DEFAULT_OUT_MD = ROOT / "docs/checkpoints/mechanism/results/axis2_layer_profile.md"
scripts/analysis/stage4_axis2_layer_profile.py:39:DEFAULT_OUT_FIG = ROOT / "results/phantom_paper/figures/fig_axis2_prompt_layer_profile.png"
scripts/analysis/stage4_axis2_layer_profile.py:51:def cosine_gap(a: np.ndarray, b: np.ndarray) -> float:
scripts/analysis/stage4_axis2_layer_profile.py:52:    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
scripts/analysis/stage4_axis2_layer_profile.py:57:    H = d["hidden_states"]
scripts/analysis/stage4_axis2_layer_profile.py:59:    n_layers = H.shape[1]
scripts/analysis/stage4_axis2_layer_profile.py:71:        curve = np.array([cosine_gap(means[a][L], means[b][L]) for L in range(n_layers)])
scripts/analysis/stage4_axis2_layer_profile.py:81:            "L17": float(curve[17]) if n_layers > 17 else None,
scripts/analysis/stage4_axis2_layer_profile.py:82:            "L4": float(curve[4]) if n_layers > 4 else None,
scripts/analysis/stage4_axis2_layer_profile.py:83:            "L0": float(curve[0]),
scripts/analysis/stage4_axis2_layer_profile.py:86:    return curves, n_layers, means
scripts/analysis/stage4_axis2_layer_profile.py:89:def write_md(curves_cls: dict, curves_red: dict, n_layers: int, out: Path):
scripts/analysis/stage4_axis2_layer_profile.py:92:        "# Exp 1 — Axis-2 (prompt-family) layer profile",
scripts/analysis/stage4_axis2_layer_profile.py:98:        "**Method**: For each prompt-only pair (text format fixed, prompt swap), compute full 37-layer cosine gap.",
scripts/analysis/stage4_axis2_layer_profile.py:103:        f"| Pair | Group | L0 | L4 | L17 | L{n_layers-1} | Peak L | Peak gap |",
scripts/analysis/stage4_axis2_layer_profile.py:108:            f"| {label} | {info['group']} | {info['L0']:.4f} | {info['L4']:.4f} | {info['L17']:.4f} | "
scripts/analysis/stage4_axis2_layer_profile.py:116:        f"| Pair | Group | L0 | L4 | L17 | L{n_layers-1} | Peak L | Peak gap |",
scripts/analysis/stage4_axis2_layer_profile.py:121:            f"| {label} | {info['group']} | {info['L0']:.4f} | {info['L4']:.4f} | {info['L17']:.4f} | "
scripts/analysis/stage4_axis2_layer_profile.py:129:        "Three hypotheses about axis-2 mechanism layer:",
scripts/analysis/stage4_axis2_layer_profile.py:131:        "1. **Truly null geometry** — axis-2 pair curves flat <0.01 at all layers. Prompt-family bypasses residual stream entirely (acts at attention pattern or output head). → Next: Exp 3 logit lens or Exp 4 attention probe.",
scripts/analysis/stage4_axis2_layer_profile.py:132:        "2. **Late-layer spike** — axis-2 pair curves spike at L25+ but flat at mid-layer. Prompt prior re-emerges at output decoding. → Next: Exp 5 late-layer patching.",
scripts/analysis/stage4_axis2_layer_profile.py:133:        "3. **Early-layer spike absorbed** — axis-2 pair curves spike at L0-L5 then collapse to ~0. Prompt embedding effect absorbed by mid-layer fusion. → Next: Exp 3 logit lens to verify if it re-emerges in output distribution.",
scripts/analysis/stage4_axis2_layer_profile.py:135:        "Compare peak layers above against axis-1 (text-format) pairs (the established mechanism with L17 peak) and image-axis reference (~0.04 magnitude). If axis-2 pair peak < 0.01 at all layers, hypothesis 1 holds.",
scripts/analysis/stage4_axis2_layer_profile.py:141:def plot(curves_cls: dict, curves_red: dict, n_layers: int, out: Path):
scripts/analysis/stage4_axis2_layer_profile.py:146:        layers = np.arange(n_layers)
scripts/analysis/stage4_axis2_layer_profile.py:150:            ax.plot(layers, info["curve"], color=info["color"], linestyle=info["linestyle"],
scripts/analysis/stage4_axis2_layer_profile.py:154:        ax.set_xlabel("Layer index (L0 = embedding, L36 = final block)")
scripts/analysis/stage4_axis2_layer_profile.py:160:    fig.suptitle("Exp 1: Axis-2 (prompt-family) layer profile — where does the prompt act?", fontsize=11)
scripts/analysis/stage4_axis2_layer_profile.py:176:    curves_cls, n_layers_cls, _ = compute_pair_curves(args.cls_npz)
scripts/analysis/stage4_axis2_layer_profile.py:177:    print(f"  {len(curves_cls)} pairs, {n_layers_cls} layers")
scripts/analysis/stage4_axis2_layer_profile.py:180:    curves_red, n_layers_red, _ = compute_pair_curves(args.red_npz)
scripts/analysis/stage4_axis2_layer_profile.py:181:    print(f"  {len(curves_red)} pairs, {n_layers_red} layers")
scripts/analysis/stage4_axis2_layer_profile.py:183:    assert n_layers_cls == n_layers_red, f"layer count mismatch cls={n_layers_cls} red={n_layers_red}"
scripts/analysis/stage4_axis2_layer_profile.py:185:    write_md(curves_cls, curves_red, n_layers_cls, args.output_md)
scripts/analysis/stage4_axis2_layer_profile.py:186:    plot(curves_cls, curves_red, n_layers_cls, args.output_fig)
docs/checkpoints/mechanism/results/axis2_layer_profile.md:1:# Exp 1 — Axis-2 (prompt-family) layer profile
docs/checkpoints/mechanism/results/axis2_layer_profile.md:7:**Method**: For each prompt-only pair (text format fixed, prompt swap), compute full 37-layer cosine gap.
docs/checkpoints/mechanism/results/axis2_layer_profile.md:12:| Pair | Group | L0 | L4 | L17 | L36 | Peak L | Peak gap |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:14:| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0013 | 0.0067 | **L36** | 0.0067 |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:22:| Pair | Group | L0 | L4 | L17 | L36 | Peak L | Peak gap |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:24:| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0012 | 0.0059 | **L36** | 0.0059 |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:32:Three hypotheses about axis-2 mechanism layer:
docs/checkpoints/mechanism/results/axis2_layer_profile.md:34:1. **Truly null geometry** — axis-2 pair curves flat <0.01 at all layers. Prompt-family bypasses residual stream entirely (acts at attention pattern or output head). → Next: Exp 3 logit lens or Exp 4 attention probe.
docs/checkpoints/mechanism/results/axis2_layer_profile.md:35:2. **Late-layer spike** — axis-2 pair curves spike at L25+ but flat at mid-layer. Prompt prior re-emerges at output decoding. → Next: Exp 5 late-layer patching.
docs/checkpoints/mechanism/results/axis2_layer_profile.md:36:3. **Early-layer spike absorbed** — axis-2 pair curves spike at L0-L5 then collapse to ~0. Prompt embedding effect absorbed by mid-layer fusion. → Next: Exp 3 logit lens to verify if it re-emerges in output distribution.
docs/checkpoints/mechanism/results/axis2_layer_profile.md:38:Compare peak layers above against axis-1 (text-format) pairs (the established mechanism with L17 peak) and image-axis reference (~0.04 magnitude). If axis-2 pair peak < 0.01 at all layers, hypothesis 1 holds.
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:11:## Per-task peak-layer distribution
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:13:AXTree-DOM peak layer: mean = **27.9**, std = 13.1, range L04-L36
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:14:Marks-like (avg across 7) peak layer: mean = **31.9**, std = 8.0
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:15:**Separation** = marks - dom = **+4.0 layers**
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:25:| 214 | 5 | L04 | L36.0 | **+32.0** |
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:26:| 228 | 2 | L04 | L36.0 | **+32.0** |
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:27:| 32 | 5 | L04 | L31.4 | **+27.4** |
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:28:| 228 | 5 | L04 | L29.4 | **+25.4** |
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:29:| 9 | 2 | L04 | L24.6 | **+20.6** |
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:36:| 20 | 2 | L36 | L33.3 | -2.7 |
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:37:| 122 | 2 | L36 | L33.3 | -2.7 |
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:39:| 37 | 2 | L36 | L28.0 | -8.0 |
scripts/analysis/stage4_pca_cosine_gap.py:2:"""Stage 4 Method 4.2: PCA cosine gap analysis of phantom routing space.
scripts/analysis/stage4_pca_cosine_gap.py:6:mechanistically distinct in hidden state geometry layer-by-layer.
scripts/analysis/stage4_pca_cosine_gap.py:8:Three analyses per (mode pair, layer):
scripts/analysis/stage4_pca_cosine_gap.py:10:  B. AUROC: project hidden states onto (mean_A - mean_B) direction, predict mode
scripts/analysis/stage4_pca_cosine_gap.py:11:  C. Per-(mode, layer) PCA top-10 variance explained
scripts/analysis/stage4_pca_cosine_gap.py:16:  - results/phantom_paper/figures/fig_stage4_pca_cosine_gap.png
scripts/analysis/stage4_pca_cosine_gap.py:26:from sklearn.decomposition import PCA
scripts/analysis/stage4_pca_cosine_gap.py:30:DEFAULT_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz"
scripts/analysis/stage4_pca_cosine_gap.py:33:DEFAULT_OUT_FIG = ROOT / "results/phantom_paper/figures/fig_stage4_pca_cosine_gap.png"
scripts/analysis/stage4_pca_cosine_gap.py:40:def cosine_gap(v1: np.ndarray, v2: np.ndarray) -> float:
scripts/analysis/stage4_pca_cosine_gap.py:41:    return float(1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
scripts/analysis/stage4_pca_cosine_gap.py:64:    H = d["hidden_states"]
scripts/analysis/stage4_pca_cosine_gap.py:66:    n_layers = H.shape[1]
scripts/analysis/stage4_pca_cosine_gap.py:67:    print(f"[stage4] loaded {H.shape[0]} examples × {n_layers} layers × {H.shape[2]} dim")
scripts/analysis/stage4_pca_cosine_gap.py:74:    cos_gap = np.zeros((len(pairs), n_layers))
scripts/analysis/stage4_pca_cosine_gap.py:75:    auroc = np.zeros((len(pairs), n_layers))
scripts/analysis/stage4_pca_cosine_gap.py:77:        for L in range(n_layers):
scripts/analysis/stage4_pca_cosine_gap.py:79:            cos_gap[pi, L] = cosine_gap(c1, c2)
scripts/analysis/stage4_pca_cosine_gap.py:80:            direction = (c1 - c2) / (np.linalg.norm(c1 - c2) + 1e-9)
scripts/analysis/stage4_pca_cosine_gap.py:90:    pca_var = np.zeros((len(MODES), n_layers))
scripts/analysis/stage4_pca_cosine_gap.py:93:        for L in range(n_layers):
scripts/analysis/stage4_pca_cosine_gap.py:96:                pca_var[mi, L] = PCA(n_components=n_comp).fit(X[:, L, :]).explained_variance_ratio_.sum()
scripts/analysis/stage4_pca_cosine_gap.py:101:        peak[f"{m1}_vs_{m2}"] = {"layer": L, "gap": float(cos_gap[pi, L]),
scripts/analysis/stage4_pca_cosine_gap.py:105:        "n_examples": int(H.shape[0]), "n_layers": int(n_layers), "n_modes": len(MODES),
scripts/analysis/stage4_pca_cosine_gap.py:107:        "pairwise_cosine_gap": {f"{m1}_vs_{m2}": cos_gap[pi].tolist()
scripts/analysis/stage4_pca_cosine_gap.py:112:        "peak_disruption_layers": peak,
scripts/analysis/stage4_pca_cosine_gap.py:123:    sorted_pairs = sorted(m["peak_disruption_layers"].items(),
scripts/analysis/stage4_pca_cosine_gap.py:126:        "# Stage 4 Method 4.2: PCA Cosine Gap Analysis",
scripts/analysis/stage4_pca_cosine_gap.py:128:        f"**Data**: {m['n_examples']} examples × {m['n_layers']} layers × {m['n_modes']} modes (Qwen3-VL-4B B1 cls)",
scripts/analysis/stage4_pca_cosine_gap.py:131:        "## Peak disruption layer per mode pair",
scripts/analysis/stage4_pca_cosine_gap.py:133:        "Sorted by cosine gap magnitude (= geometric distance between mode means in hidden space):",
scripts/analysis/stage4_pca_cosine_gap.py:135:        "| Mode pair | Peak layer | Cosine gap | AUROC at peak |",
scripts/analysis/stage4_pca_cosine_gap.py:140:        lines.append(f"| {DISPLAY[m1]} vs {DISPLAY[m2]} | L{v['layer']:02d} | {v['gap']:.4f} | {v['auroc_at_peak']:.3f} |")
scripts/analysis/stage4_pca_cosine_gap.py:142:    # Mid-layer (L17) snapshot — paper §5 disruption locus
scripts/analysis/stage4_pca_cosine_gap.py:143:    L17_section = ["", "## L17 cosine gap snapshot (paper §5 disruption locus)", ""]
scripts/analysis/stage4_pca_cosine_gap.py:144:    L17_section.append("| Mode pair | L17 cosine gap | L17 AUROC |")
scripts/analysis/stage4_pca_cosine_gap.py:148:        gap = m["pairwise_cosine_gap"][f"{m1}_vs_{m2}"][17]
scripts/analysis/stage4_pca_cosine_gap.py:153:    # Phantom-arm specific anchor — P-SoM cosine to each baseline mode at L17
scripts/analysis/stage4_pca_cosine_gap.py:163:            row.append(f"{m['pairwise_cosine_gap'][key][L]:.4f}")
scripts/analysis/stage4_pca_cosine_gap.py:183:    ax.set_title("(a) Pairwise cosine gap (geometric distance between mode means)")
scripts/analysis/stage4_pca_cosine_gap.py:191:    ax.set_title("(b) Pairwise AUROC (project onto Δ-mean direction, classify)")
scripts/analysis/stage4_pca_cosine_gap.py:198:    ax.set_ylabel("Top-10 PCA cumulative variance explained")
scripts/analysis/stage4_pca_cosine_gap.py:211:    ax.set_title("(d) P-SoM identity — closest sibling per layer")
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:6:**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:7:- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:10:## Result table (sorted by peak layer)
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:12:| Variant | Format example | H1 class | Peak layer | Peak cosine gap |
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:14:| appagent_id | `id_N: label (AppAgent)` | marks-like | **L04** | 0.0488 |
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:15:| plain_numbered | `N. label (numbered)` | marks-like | **L04** | 0.0505 |
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:16:| hash_id_control | `#hash label (no integer)` | control (no integer) | **L04** | 0.0508 |
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:17:| dom | `AXTree (baseline DOM)` | AXTree-baseline | **L04** | 0.0495 |
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:30:- `id_N: label (AppAgent)`: peak **L04** = 0.0488
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:32:- `N. label (numbered)`: peak **L04** = 0.0505
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:37:- `#hash label (no integer)`: peak **L04** = 0.0508
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:45:- `AXTree (baseline DOM)`: peak **L04** = 0.0495
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:49:- **6 marks-like variants**: mean peak layer = 13, range L04-L17
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:50:- **2 control variants** (no integer / no list): mean peak layer = 10, range L04-L17
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md:51:- **AXTree-DOM baseline**: peak L04
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:6:**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:7:- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:10:## Result table (sorted by peak layer)
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:12:| Variant | Format example | H1 class | Peak layer | Peak cosine gap |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:14:| dom | `AXTree (baseline DOM)` | AXTree-baseline | **L04** | 0.0434 |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:16:| som_standard | `[N] role 'label' (SoM)` | marks-like | **L36** | 0.0429 |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:17:| browser_use_at | `@N label (Browser Use)` | marks-like | **L36** | 0.0520 |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:18:| appagent_id | `id_N: label (AppAgent)` | marks-like | **L36** | 0.0526 |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:19:| tarsier_typed | `[BN:role:label] (Tarsier)` | marks-like | **L36** | 0.0475 |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:20:| plain_numbered | `N. label (numbered)` | marks-like | **L36** | 0.0518 |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:21:| xml_tagged | `<el_N role='..'>label</el_N> (XML)` | marks-like | **L36** | 0.0439 |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:22:| hash_id_control | `#hash label (no integer)` | control (no integer) | **L36** | 0.0516 |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:26:### marks-like  (mean peak L36)
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:28:- `[N] role 'label' (SoM)`: peak **L36** = 0.0429
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:29:- `@N label (Browser Use)`: peak **L36** = 0.0520
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:30:- `id_N: label (AppAgent)`: peak **L36** = 0.0526
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:31:- `[BN:role:label] (Tarsier)`: peak **L36** = 0.0475
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:32:- `N. label (numbered)`: peak **L36** = 0.0518
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:33:- `<el_N role='..'>label</el_N> (XML)`: peak **L36** = 0.0439
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:35:### control (no integer)  (mean peak L36)
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:37:- `#hash label (no integer)`: peak **L36** = 0.0516
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:45:- `AXTree (baseline DOM)`: peak **L04** = 0.0434
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:49:- **6 marks-like variants**: mean peak layer = 36, range L36-L36
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:50:- **2 control variants** (no integer / no list): mean peak layer = 29, range L22-L36
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:51:- **AXTree-DOM baseline**: peak L04
docs/checkpoints/mechanism/results/format_variation_h1_test.md:6:**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:
docs/checkpoints/mechanism/results/format_variation_h1_test.md:7:- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)
docs/checkpoints/mechanism/results/format_variation_h1_test.md:10:## Result table (sorted by peak layer)
docs/checkpoints/mechanism/results/format_variation_h1_test.md:12:| Variant | Format example | H1 class | Peak layer | Peak cosine gap |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:14:| dom | `AXTree (baseline DOM)` | AXTree-baseline | **L04** | 0.0462 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:16:| som_standard | `[N] role 'label' (SoM)` | marks-like | **L36** | 0.0434 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:17:| browser_use_at | `@N label (Browser Use)` | marks-like | **L36** | 0.0502 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:18:| appagent_id | `id_N: label (AppAgent)` | marks-like | **L36** | 0.0506 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:19:| tarsier_typed | `[BN:role:label] (Tarsier)` | marks-like | **L36** | 0.0468 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:20:| plain_numbered | `N. label (numbered)` | marks-like | **L36** | 0.0502 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:21:| xml_tagged | `<el_N role='..'>label</el_N> (XML)` | marks-like | **L36** | 0.0426 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:22:| hash_id_control | `#hash label (no integer)` | control (no integer) | **L36** | 0.0514 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:26:### marks-like  (mean peak L36)
docs/checkpoints/mechanism/results/format_variation_h1_test.md:28:- `[N] role 'label' (SoM)`: peak **L36** = 0.0434
docs/checkpoints/mechanism/results/format_variation_h1_test.md:29:- `@N label (Browser Use)`: peak **L36** = 0.0502
docs/checkpoints/mechanism/results/format_variation_h1_test.md:30:- `id_N: label (AppAgent)`: peak **L36** = 0.0506
docs/checkpoints/mechanism/results/format_variation_h1_test.md:31:- `[BN:role:label] (Tarsier)`: peak **L36** = 0.0468
docs/checkpoints/mechanism/results/format_variation_h1_test.md:32:- `N. label (numbered)`: peak **L36** = 0.0502
docs/checkpoints/mechanism/results/format_variation_h1_test.md:33:- `<el_N role='..'>label</el_N> (XML)`: peak **L36** = 0.0426
docs/checkpoints/mechanism/results/format_variation_h1_test.md:35:### control (no integer)  (mean peak L36)
docs/checkpoints/mechanism/results/format_variation_h1_test.md:37:- `#hash label (no integer)`: peak **L36** = 0.0514
docs/checkpoints/mechanism/results/format_variation_h1_test.md:45:- `AXTree (baseline DOM)`: peak **L04** = 0.0462
docs/checkpoints/mechanism/results/format_variation_h1_test.md:49:- **6 marks-like variants**: mean peak layer = 36, range L36-L36
docs/checkpoints/mechanism/results/format_variation_h1_test.md:50:- **2 control variants** (no integer / no list): mean peak layer = 26, range L17-L36
docs/checkpoints/mechanism/results/format_variation_h1_test.md:51:- **AXTree-DOM baseline**: peak L04
p79/mechanistic/extract_hidden_states.py:27:    """Extract last-input-token hidden states from Qwen3-VL-4B forward pass.
p79/mechanistic/extract_hidden_states.py:40:        # Paper-grade: pin HF revision SHA — DGX baseline lock 2026-05-07 (笔记 §114)
p79/mechanistic/extract_hidden_states.py:41:        model_revision: str = "ebb281ec70b05090aa6165b016eac8ec08e71b17",
p79/mechanistic/extract_hidden_states.py:56:        logger.info(f"Loading {model_path} (revision={model_revision[:12]}...) for hidden state extraction (dtype={dtype})")
p79/mechanistic/extract_hidden_states.py:59:            revision=model_revision,
p79/mechanistic/extract_hidden_states.py:66:            model_path, revision=model_revision, trust_remote_code=True
p79/mechanistic/extract_hidden_states.py:69:        self.model_revision = model_revision
p79/mechanistic/extract_hidden_states.py:119:        """Forward pass with output_hidden_states=True. Return last-token hidden states.
p79/mechanistic/extract_hidden_states.py:129:            Tensor of shape (n_layers + 1, hidden_dim). Layer 0 is embedding output;
p79/mechanistic/extract_hidden_states.py:130:            layer L for L >= 1 is post-transformer-block-L hidden state.
p79/mechanistic/extract_hidden_states.py:143:            messages, tokenize=False, add_generation_prompt=True
p79/mechanistic/extract_hidden_states.py:163:            output_hidden_states=True,
p79/mechanistic/extract_hidden_states.py:164:            use_cache=False,
p79/mechanistic/extract_hidden_states.py:168:        # outputs.hidden_states is tuple of (n_layers + 1) tensors of shape
p79/mechanistic/extract_hidden_states.py:169:        # (batch=1, seq_len, hidden_dim). Extract last token of each.
p79/mechanistic/extract_hidden_states.py:170:        hidden_states = torch.stack(
p79/mechanistic/extract_hidden_states.py:171:            [h[0, -1, :].detach().float().cpu() for h in outputs.hidden_states],
p79/mechanistic/extract_hidden_states.py:173:        )  # (n_layers + 1, hidden_dim)
p79/mechanistic/extract_hidden_states.py:174:        return hidden_states
p79/mechanistic/extract_hidden_states.py:186:            (hidden_states, mode_labels)
p79/mechanistic/extract_hidden_states.py:187:            - hidden_states: Tensor (n_items, n_layers + 1, hidden_dim)
docs/checkpoints/paper_drafts/section5_mechanism.md:7:This section is the Zoom-4 layer of the paper's four-level account. Zoom 1 defines the architectural intervention, "skip the annotated image"; Zoom 2 measures the behavioral axes of text payload, prompt family, and image presence; Zoom 3 links the observed behavior to Mirage-style no-image visual reasoning and prompt-format sensitivity; Zoom 4 asks where the resulting mode is represented and whether it is causally used by the model. We index layers L0-L36, where L0 is the embedding-block output and L1-L36 are the 36 transformer decoder block outputs.
docs/checkpoints/paper_drafts/section5_mechanism.md:9:The analysis builds on the linear-readable and steerable circuit framework of Wu et al., which uses mode means, PCA geometry, and mean-difference activation steering to study tool selection, and on work showing middle-layer cross-modal information flow in VLMs \citep{wu2026toolcalling,kaduri2024whatsintheimage}. Our setting is not a replication of those papers. It is a multimodal web-agent application of the same representation-level question: whether a behaviorally useful routing arm is linearly readable, partially steerable, and causally active inside the model.
docs/checkpoints/paper_drafts/section5_mechanism.md:11:Four mechanism claims organize the evidence. First, observation modes are geometrically separable in the residual stream: the PCA cosine-gap analysis reaches AUROC 1.0 on the main mode contrasts. Second, Phantom-SoM is a mid-layer text-axis sibling of DOM/P-text, not an image-axis sibling of full SoM. Third, SoM-to-no-image activation patching displaces target continuations at mid layers L11-L17 with consistent magnitude across classifieds and reddit. Fourth, the shortcut trigger is **better described as the conjunction of integer-indexed marker and markup-sigil leading delimiter** than as an abstract "flat element list" — AXTree hierarchy preserves the early L04 image-axis peak, but so do indexed variants that lack either the integer (e.g., `hash_id_control`) or the sigil (e.g., `appagent_id`, `plain_numbered`); only the conjunction triggers the late shift. This refinement is **exploratory after W6** and awaits held-out falsifiers (`bare_N`, `bracket_no_int`).
docs/checkpoints/paper_drafts/section5_mechanism.md:15:| Evidence layer | Method | Status |
docs/checkpoints/paper_drafts/section5_mechanism.md:17:| Geometry (cosine PCA per layer) | Method 4.2 (§5.2) | **Strong** — AUROC 1.0 on key contrasts, 5/5 robustness, 100% per-task above 0.010 threshold |
docs/checkpoints/paper_drafts/section5_mechanism.md:18:| Causal continuation patching (SoM → no-image arms) | Stage 2/3 (§5.4) | **Causal** — mid-layer L12-L18 transfers across cls + reddit, additive across DOM/P-text/P-prompt targets, Gaussian-random negative controls at ~0 |
docs/checkpoints/paper_drafts/section5_mechanism.md:19:| Causal axis-2 prompt-only patching | Exp 5 cellhprompt (this section) | **Causal continuation evidence on 2 sites N=24 each**; bootstrap CI and content-matched (task-shuffled / random-source) controls **PENDING** (cellhprm_*_rand 359719/359720 in flight) |
docs/checkpoints/paper_drafts/section5_mechanism.md:20:| Steering (mean-diff activation) | Method 4.4 (§5.3) | **Weak / partial** — best H-mean 0.33 at L33 α=10, layer-α tradeoff prevents single sweet spot, treated as evidence ceiling not validation |
docs/checkpoints/paper_drafts/section5_mechanism.md:21:| Output divergence | Exp 3 logit lens (§5.7) | **Geometry-amplification at L23-L25**, NOT a deployed classifier; deployment claim deferred |
docs/checkpoints/paper_drafts/section5_mechanism.md:22:| Trigger attribution (which formats trigger shortcut) | W6 tokenization (§5.5) | **Exploratory** — 6 marks-like variants split 2-vs-4 on first-token sigil; held-out falsifier `bare_N` (integer no sigil) and `bracket_no_int` (sigil no integer) pending |
docs/checkpoints/paper_drafts/section5_mechanism.md:24:The cross-site evidence stack is deliberately defensive. Per-task H1 fragility shows the dichotomy is an aggregate mechanism rather than a deterministic per-task law. Reverse-tier classifieds runs defend against strong-tier selection bias. Reddit format variation replicates the shortcut direction with cleaner mid-layer peaks. Reddit Method 4.2 replicates the Mirage signature: Phantom-SoM remains close to DOM on the text axis while separating from SoM on the image axis. Paper 1 uses these results for mechanism interpretation only; routing implementation is deferred to paper 2, consistent with the paper-planning scope split.
docs/checkpoints/paper_drafts/section5_mechanism.md:26:## 5.2 Method 4.2: PCA Cosine Gap
docs/checkpoints/paper_drafts/section5_mechanism.md:28:Method 4.2 extracts hidden states from Qwen3-VL-4B B1 runs and compares observation modes by layer. For each mode pair and layer, we compute the cosine gap between hidden-state means, evaluate AUROC by projecting examples onto the mean-difference direction, and summarize per-mode geometry through PCA top-10 variance. The classifieds baseline contains 288 examples, formed from 24 strong-tier tasks, two archived steps, and six modes, over 37 indexed layers.
docs/checkpoints/paper_drafts/section5_mechanism.md:30:The robustness suite passes all five checks in the plan. Label permutation leaves the real AUROC 9.8 standard deviations above the permuted baseline. Per-task analysis is positive for all 24 tasks. Step 2 and step 5 curves are invariant at the mechanism level. The L23 silhouette score is at least 0.5, showing nontrivial clustering. Bootstrap 95% confidence intervals are tight, with widths of roughly 4-15% of the corresponding means.
docs/checkpoints/paper_drafts/section5_mechanism.md:32:The key classifieds snapshot is the L17 cosine-gap table:
docs/checkpoints/paper_drafts/section5_mechanism.md:34:| Pair at L17 | Cosine gap | 95% CI | AUROC |
docs/checkpoints/paper_drafts/section5_mechanism.md:41:The reddit replication lands the same qualitative geometry. At L17, P-SoM is close to DOM with cosine gap 0.0098 and close to P-text with gap 0.0027, while P-SoM-to-SoM remains much larger at 0.0423 and P-SoM-to-Vision at 0.0457. The DOM-to-Vision image-axis peak is L04 with cosine gap 0.0687 and AUROC 1.0.
docs/checkpoints/paper_drafts/section5_mechanism.md:43:This is the Mirage signature in geometric form. Phantom-SoM is not represented as a weakened image mode. At the mid-layer disruption locus, it is a text-axis sibling of DOM/P-text, while the image-axis distance to full SoM remains large.
docs/checkpoints/paper_drafts/section5_mechanism.md:47:Method 4.4 tests whether the readable mode direction can be used as a steering direction. For each layer, we form a mean-difference vector between Phantom-SoM-like and DOM-like hidden states, add it to each input at generation time with scaling factor $\alpha$, and evaluate whether the continuation moves toward the target mode while preserving the JSON action envelope. Following HDMI's evaluation vocabulary, reliability is the harmonic mean of completeness and selectivity, not a raw shift rate \citep{khorasani2026hdmi}.
docs/checkpoints/paper_drafts/section5_mechanism.md:49:The v2 sweep covers layers [11, 17, 23, 29, 33, 34] and $\alpha \in [1,2,5,10,20]$, for 45 completed cells in the plan summary. The original L17, $\alpha=5$ smoke result reported H-mean 0.44, but the full sweep lowers that cell to 0.16. The plan records this as a smoke-variance artifact from notes 126/127: a 4-cell smoke was too small to support a sweet-spot claim.
docs/checkpoints/paper_drafts/section5_mechanism.md:51:The strongest full-sweep cell is L33, $\alpha=10$, with H-mean 0.33. Its completeness is 38% and its selectivity is 29%. The layer profile is the important result: mid layers L11-L23 preserve the JSON envelope with 100% selectivity but have low completeness, while late-layer L33 produces the largest shifts but frequently over-steers the continuation out of the expected JSON format.
docs/checkpoints/paper_drafts/section5_mechanism.md:53:This creates a probe-causal dissociation. The mid-layer geometry is cleanly readable and causally implicated by patching, but fixed mean-difference steering is only partially reliable. The 0.33 H-mean is therefore an evidence ceiling for Method 4.4, not a final control result. Section 8 should treat LA-HDMI and SAE feature steering as future work motivated by this ceiling, without claiming that either method has already improved it.
docs/checkpoints/paper_drafts/section5_mechanism.md:57:Activation patching provides the causal test. For each task, the clean/source run and corrupt/target run use the same archived browser step and deterministic 50-token continuation. In the core SoM-to-Phantom-SoM setup, the source prompt is `som`: task instruction, SoM prompt family, flat `[SOM_MARKS]` text, and annotated screenshot. The target prompt is `phantom_som`: the same instruction, same prompt family, and same `[SOM_MARKS]` text, but no image. Source hidden states are cached by layer, injected into the final input-token position of the target on the first forward pass, and subsequent decoding proceeds normally through the model cache.
docs/checkpoints/paper_drafts/section5_mechanism.md:59:Each patched continuation is scored against the unpatched source and target continuations. The main disruption statistic is the drop in `token_overlap_to_target`; Levenshtein distance to target is the paired backup. Layer-wise tests compare each grid layer to the final-layer reference using task-paired differences and Holm-Bonferroni correction across the canonical grid. Random-injection controls replace source hidden states with Gaussian tensors matched to source activation mean and standard deviation.
docs/checkpoints/paper_drafts/section5_mechanism.md:61:The Stage 2 P-SoM<->SoM dashboard now contains ten completed cells, including reddit F/G and the reddit random control:
docs/checkpoints/paper_drafts/section5_mechanism.md:63:| Cell | Site | Direction | Mid-layer target-overlap drop | Holm status |
docs/checkpoints/paper_drafts/section5_mechanism.md:69:| E | cls | random injection | -0.03 uniform | negative control |
docs/checkpoints/paper_drafts/section5_mechanism.md:73:| Er | reddit | random injection | approximately 0 uniform | negative control |
docs/checkpoints/paper_drafts/section5_mechanism.md:75:Stage 3 extends this from P-SoM to the three no-image arms, testing whether the image-feature axis is shared across DOM, P-text, and P-prompt targets. The table below reports per-task-paired Δoverlap-to-target from the patching_continuation_results.json under each cell directory, with the layer at which the disruption peaks.
docs/checkpoints/paper_drafts/section5_mechanism.md:82:All six Stage 3 cells are now closed. Two observations carry the cross-site claim. First, every cell's best layer falls inside the L12-L18 mid-layer window, and every cell's best Δoverlap-to-target is between -0.27 and -0.35. The mid-layer fusion locus is therefore not a single layer index but a tight 7-layer window that transfers across cls and reddit. Second, the interpretation is additive rather than arm-specific: a SoM source state displaces DOM, P-text, and P-prompt targets toward the source with similar magnitude, implying a shared image-feature substrate across all three no-image arms. The negative controls, Cell E at -0.03 and Cell Er near zero, rule out a generic nonzero-injection explanation.
docs/checkpoints/paper_drafts/section5_mechanism.md:86:The cleanest single-pair signature is the image-axis peak-layer dichotomy. Across eight image-presence contrasts, the no-image side's text format predicts the peak layer with zero overlap. If the no-image side is AXTree text, the image-axis cosine gap peaks at L04 in all four pairs: DOM<->Vision, DOM<->SoM, P-prompt<->Vision, and P-prompt<->SoM. If the no-image side is `[SOM_MARKS]` or another flat marks text, the peak shifts to L17-L36 in all four pairs: P-text<->Vision, P-text<->SoM, P-SoM<->Vision, and P-SoM<->SoM.
docs/checkpoints/paper_drafts/section5_mechanism.md:88:The refined H1 is a pretraining co-occurrence shortcut: when the input contains a marker token sequence that pretraining data associates with HTML / agent-trace visual grounding (specifically the conjunction of integer index and markup-sigil leading delimiter such as `[`, `<`, `@`), the model activates a visual-grounding pathway even if the image is absent. Flat element-list form alone is **not sufficient** — `appagent_id` (`id_N: label`) and `plain_numbered` (`N. label`) are nominally flat indexed lists but lack the markup-sigil and behave like AXTree-DOM (W6 evidence, exploratory). Prompt-format sensitivity makes this plausible at the input level \citep{sclar2024promptformat}; Method 4.2 shows it as a layer-resolved internal signature.
docs/checkpoints/paper_drafts/section5_mechanism.md:90:The format-variation grid contains ten modes: six marks-like variants, two controls, and DOM/SoM baselines. In the classifieds strong-tier baseline, all six marks-like variants peak at L36, the hash-ID control also peaks at L36, the plain-sentence control peaks at L17, and the DOM baseline preserves the L04 peak. Because L36 is the boundary layer, this is best read as a strong late/monotonic signature rather than as a precise late-layer mechanism.
docs/checkpoints/paper_drafts/section5_mechanism.md:92:The classifieds reverse-tier run reproduces the strong-tier shape. The six marks-like variants and hash-ID control again peak at L36, the plain-sentence control moves to L22, and DOM remains at L04. This defends H1 against the selection-bias concern that strong-tier curation alone created the pattern.
docs/checkpoints/paper_drafts/section5_mechanism.md:94:The reddit format run is cleaner for the mid-layer interpretation. Four of six marks-like variants peak at L17, the plain-sentence control peaks at L17, hash-ID control returns to L04, and DOM remains at L04. **W6 attribution** (`docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md`, exploratory) further finds that the two L04 marks-like variants (`appagent_id`, `plain_numbered`) share a feature with the L04 DOM baseline: their first tokens are alphanumeric, while the four L17-peaking marks-like variants all start with markup-sigil tokens (`[`, `<`, `@`). The hash-ID control (`#a3f7`) starts with a sigil but lacks integer-marker structure and also peaks at L04, suggesting the trigger conjunction is integer-marker + markup-sigil rather than either alone. This is a post-hoc feature-attribution on a small (N=6 marks-like) format set; held-out falsifiers (`bare_N` = integer without sigil, `bracket_no_int` = sigil without integer) are not yet run. Cross-site, the safe claim is directional: marker formats that combine integer indexing with markup-sigil leading delimiters tend to delay image-axis separation into mid/late layers, while AXTree hierarchy and indexed-list variants lacking either feature preserve the early L04 image-axis peak. The reddit curve reveals the true L11-L17 fusion locus more clearly than the classifieds L36 boundary artifact.
docs/checkpoints/paper_drafts/section5_mechanism.md:100:The second defense is selection-bias robustness. The classifieds reverse-tier run replicates the strong-tier H1 pattern, including L36 marks-like peaks and L04 DOM baseline. The shortcut signature is therefore not an artifact of selecting tasks where SoM beats DOM.
docs/checkpoints/paper_drafts/section5_mechanism.md:102:The third defense is cross-site H1. Reddit does not reproduce the exact boundary-layer shape, but it reproduces the direction of the indexed-list shortcut with a cleaner L17 mid-layer peak for four of six marks-like formats. The site changes the curve shape, not the basic interpretation.
docs/checkpoints/paper_drafts/section5_mechanism.md:104:The fourth defense is cross-site Mirage geometry. Reddit Method 4.2 reproduces the central relation: P-SoM is close to DOM/P-text at L17 and far from SoM/Vision on the image axis, with AUROC 1.0 on the key contrasts. This supports cross-site generalization of the mechanism claim, not B0/B1 capability scaling.
docs/checkpoints/paper_drafts/section5_mechanism.md:110:A naive reading of Method 4.2's L17 snapshot suggests the four phantom-boundary modes split into two text-format clusters with prompt-family making no geometric contribution. That reading is incomplete: it inspects the wrong layer. Computing full 37-layer cosine-gap profiles for axis-isolated pairs reveals a layered three-axis hierarchy in the residual stream.
docs/checkpoints/paper_drafts/section5_mechanism.md:112:The pairs are constructed to isolate each axis. Axis-1 (text-format swap, prompt fixed) is measured by DOM<->P-text (both DOM prompts) and P-prompt<->P-SoM (both SoM prompts). Axis-2 (prompt-family swap, text fixed) is measured by DOM<->P-prompt (both hierarchical AXTree) and P-text<->P-SoM (both flat indexed list). Axis-3 (image-feature swap, mode otherwise fixed) is measured by the P-SoM<->SoM reference pair. All five curves are computed on `stage4_multimode_b1_cls/hidden_states.npz` (288 examples, 37 layers) and replicated cross-site on the matching reddit run.
docs/checkpoints/paper_drafts/section5_mechanism.md:114:The peak-layer and magnitude table (cls site, reddit columns omitted but qualitatively identical):
docs/checkpoints/paper_drafts/section5_mechanism.md:116:| Axis | Pair | L17 | L23 | L36 | Peak L | Peak gap |
docs/checkpoints/paper_drafts/section5_mechanism.md:122:| Axis-2 prompt-family | DOM <-> P-prompt | 0.0013 | 0.0050 | 0.0067 | **L36** | 0.0067 |
docs/checkpoints/paper_drafts/section5_mechanism.md:124:Three regularities organize the table. First, the three axes have distinct peak layers: image-axis at L17 (fast, sharp), text-format at L23 (slower late-mid build), prompt-family at L23 or L36 (same timing as text-format on the flat-text pair, boundary peak on hierarchical). Second, the three axes have distinct magnitudes: image axis approximately 0.04, text-format approximately 0.03, prompt-family approximately 0.01. Prompt-family is roughly 3 to 4 times smaller than text-format and 4 to 8 times smaller than image. Third, the magnitude rank holds cross-site: the reddit P-text<->P-SoM axis-2 peak is 0.0098 at L23 (versus cls 0.0114), the same rank-order and the same peak layer.
docs/checkpoints/paper_drafts/section5_mechanism.md:126:The L17 snapshot exclusion of axis-2 is therefore a layer-selection artifact rather than a structural absence. Prompt-family is geometrically present in the residual stream; it simply emerges at L23 rather than L17, and at one-third the magnitude of text-format. The mid-layer fusion locus identified in Sections 5.2-5.5 is specifically the image-axis fusion locus (Mirage signature). The text-format and prompt-family axes share a separate late-mid build at L23 that runs in parallel.
docs/checkpoints/paper_drafts/section5_mechanism.md:128:Per-task fragility check confirms the axis-2 signal is broad rather than outlier-driven. At L23 on the P-text<->P-SoM pair, the per-task cosine gap distribution is tight on both sites: classifieds median 0.0131 with IQR [0.0124, 0.0142] and reddit median 0.0120 with IQR [0.0113, 0.0127], with mean approximately equal to median in both cases (not right-skewed). 100% of 24 classifieds tasks and 100% of 24 reddit tasks exceed both the 0.005 noise threshold and the 0.010 cohort-mean magnitude. None of the 48 tasks falls outside the 0.010-0.018 band on either site. This contrasts with the per-task fragility of the binary H1 dichotomy in Section 5.5 (5 of 45 task-step pairs strict, 11%); the H1 dichotomy is fragile because it asks a per-task layer-comparison question, while the axis-2 cosine gap measures a continuous mode-pair distance that has low per-task variance even when the aggregate magnitude is small. The axis-2 claim is therefore not a 2-3 outlier mean artifact but a uniform 0.011-0.013 cosine-gap signature across every task in both sites.
docs/checkpoints/paper_drafts/section5_mechanism.md:130:This layered hierarchy partially resolves the Phantom-SoM hero puzzle without requiring a non-mechanistic explanation. Phantom-SoM occupies three distinct axis positions: **large image-axis separation from SoM** at L17 (0.0412 cosine gap, the largest single signal — note this is a *separation* from SoM, not a proximity; the no-image marks-text reshapes how image-axis divergence accumulates rather than recovering an image-like internal state), **small prompt-family separation from P-text** at L23 (0.011 signal, consistent across cls and reddit), and **medium text-format separation from P-prompt** at L23 (0.029 signal). P-text occupies only the text-format separation; P-prompt occupies only the prompt-family separation against P-SoM; DOM occupies none of the three. The drop-one hero status of Phantom-SoM in `fig_meta_forest.png` therefore corresponds to the only mode that simultaneously expresses all three axis positions. The connection from "three-axis positional uniqueness" to "drop-one task-success contribution" is currently associational, not yet causally bridged — the patching evidence shows residual-stream displacement on 50-token continuation, not direct displacement of task-level SR / drop-one oracle. Closing that behavioral causal bridge is open work.
docs/checkpoints/paper_drafts/section5_mechanism.md:132:This reframing is itself paper-grade contribution and not a downgrade of the original mechanism story. It strengthens Section 5 from "mid-layer image-feature axis explains text-format cluster" to "the residual stream carries three quantitatively distinct axes with image-axis dominant at L17 and text-format + prompt-family at L23". Section 8 inherits a sharper instruction for future single-axis steering: LA-HDMI and SAE residual-stream interventions can target the L23 prompt-family direction directly, but the signal-to-noise ratio is approximately 3-4 times worse than the image-axis intervention, so steering experiments on axis-2 require correspondingly more samples or alternative attribution methods.
docs/checkpoints/paper_drafts/section5_mechanism.md:134:The three-axis hierarchy persists when we move from residual-stream geometry to output distribution. A logit lens test (Exp 3) applies Qwen3-VL-4B's final RMSNorm and lm_head to each per-layer per-mode mean hidden state, then computes KL divergence between mode pairs across all 37 layers. On classifieds the axis-2 prompt-family pair (P-text vs P-SoM) reaches peak KL 0.162 at L23, the axis-1 text-format pair (P-prompt vs P-SoM) reaches peak KL 0.695 at L23, and the reddit replication holds the same rank-order with peak KL 0.126 at L24 and 0.617 at L23 respectively. The output-level axis-1 to axis-2 magnitude ratio is therefore approximately 4.3 on classifieds and 4.9 on reddit, preserving the 3-4x residual-stream rank from Exp 1. The lm_head amplifies the residual-stream cosine signal into output divergence by roughly 14x on the axis-2 flat-text pair (cosine 0.011 to KL 0.16) and 24x on the axis-1 SoM-prompt pair (cosine 0.029 to KL 0.69), but it is axis-agnostic in the ratio it preserves.
docs/checkpoints/paper_drafts/section5_mechanism.md:136:Two corollaries follow. First, the KL trajectory drops to approximately zero at L36 even though L23 KL is substantial. The mean hidden state at the final layer collapses to the shared JSON action-header tokens that every mode emits, so mode-distinct output signal is concentrated in the L23-L25 decoding window rather than at the final embedding. Second, this output-amplification observation is **mechanistic, not a deployment-time classifier claim**: the lm_head acts as an axis-agnostic ratio-preserving projection that scales residual-stream geometry into output-space KL — the L23-L25 KL magnitude is a property of the mean hidden state, not a per-task discriminator. Whether the L23-L25 hidden representation can be used as a held-out mode classifier — with per-task AUROC, random-direction baseline, and competitive comparison to surface-token classifiers — is open work. Routing exploitation, deferred to paper 2, will need to make this case explicitly rather than inheriting it from §5.7.
docs/checkpoints/paper_drafts/section5_mechanism.md:140:The main limit is the Method 4.4 ceiling. The cosine-gap and patching evidence point to L11-L17 as the readable and causally active fusion region, while the best fixed mean-difference steering cell is late, L33 with $\alpha=10$, and has H-mean 0.33 because completeness and selectivity trade off. This supports a mechanism interpretation but not a strong deployment-time steering claim.
docs/checkpoints/paper_drafts/section5_mechanism.md:142:The second limit is layer precision. Classifieds H1 peaks often hit L36, while reddit reveals cleaner L17 peaks. The robust claim is therefore an effect-direction claim: AXTree hierarchy preserves early image-axis separation, and flat element-list formats delay that separation into mid/late computation. We should not claim that every site or task has an identical peak layer.
docs/checkpoints/paper_drafts/section5_mechanism.md:144:Literature positioning should stay modest. Section 5 applies the linear-readable, steerable, and mid/late-layer circuit framework to multimodal web-agent observation modes \citep{wu2026toolcalling,kaduri2024whatsintheimage,khorasani2026hdmi,fayyaz2026steermoe}. It should not claim novelty as the first such circuit or the first use of marked text. The contribution is controlled scientific characterization of the phantom boundary.
docs/checkpoints/paper_drafts/section5_mechanism.md:146:Finally, AXTree hierarchy is the unique defeating format in the aggregate, but the reason hierarchy defeats the shortcut remains open. The plan records one attribution-pending hypothesis: hierarchy or indentation tokens may redirect cross-modal attention before the flat-list shortcut activates. That should be treated as a supplement question, not as a Section 5 finding.
docs/checkpoints/paper_drafts/section5_mechanism.md:150:Bibkeys audit (2026-05-12 21:18): all 5 core mechanism anchors verified present in `paper.bib` — `wu2026toolcalling`, `khorasani2026hdmi`, `kaduri2024whatsintheimage`, `sclar2024promptformat`, `fayyaz2026steermoe`. Plus 5 method/protocol references added: `wang2023interpretability` (IOI patching), `zhang2024patching` (patching survey, NEEDS_VERIFY exact paper), `holm1979sequentially` (multiple-comparison correction), `lipton2018troubling` (ML scholarship critique), `neurips2024checklist` (reproducibility standard). paper.bib total 67 entries / 638 lines.
docs/checkpoints/paper_drafts/section5_mechanism.md:152:Behavioral content to relocate from current `section5_mechanism_reddit.md`: lines 17-75 should move to Section 4 or a new behavioral-routing subsection. Specifically, lines 17-23 are reddit substrate framing; lines 25-35 are Axis 1 text-payload behavior; lines 37-47 are Axis 2 prompt behavior; lines 49-59 are Axis 3 image behavior; lines 61-67 are compound P-SoM versus DOM behavior; lines 69-75 are scope/noise limitations. Lines 1-15 are method material that was retained conceptually but must use the new L0-L36 layer convention. Line 77 should be deleted or replaced because routing implementation is now paper-2, not paper-1 Section 6.
docs/checkpoints/paper_drafts/section5_mechanism.md:154:Stage 3 numbers verified 2026-05-12 from full per-task paired-test computation on `patching_continuation_results.json` (each cell, 24 tasks × 36 layers). H-d-cls best L18 Δ=-0.352, H-d-red best L14 Δ=-0.338, H-t-cls best L12 -0.270, H-t-red best L15 -0.330, H-p-cls best L13 -0.273, H-p-red best L14 -0.322. All 6 cells' best layer lands in L12-L18 mid-layer window, Δ range [-0.27, -0.35]. The L17-only column previously cited in plan §5.2 reads -0.309/-0.255/-0.223 (cls) and -0.255/-0.236/-0.191 (reddit); plan §5.2 has been updated to record best-layer Δ instead of L17-only Δ.
docs/checkpoints/paper_drafts/section5_mechanism.md:156:Pending items (post 2026-05-12 audit): (a) Method 4.4 sweep description should be "45 completed cells out of a 6x5 layer-alpha grid plus 3 placeholder cells that did not finish", not "45/48-cell sweep" (the 48-cell wording in plan §5.3 implies a 48-cell denominator that was never executed). (b) Bibkey `zhang2024patching` is marked NEEDS_VERIFY in `paper.bib` because the intended reference may be Heimersheim & Nanda 2024 [arXiv:2404.15255] rather than Zhang & Nanda 2024 [arXiv:2309.16042]; verify before submission. (c) Bibkey `fayyaz2026steermoe` is marked NEEDS_VERIFY pending deanon of the ICLR 2026 submission.
docs/checkpoints/paper_drafts/section5_mechanism.md:164:3. ✅ §5.7 corollary 2 — "deployment-time mode classifier on output logprobs has strictly more signal" + "Section 6 routing should treat L23-L25 logit-lens features as the cheapest mode-axis discriminator" → softened to "mechanistic observation, not deployment-time classifier claim; held-out classifier with random-direction baseline is open work"
docs/checkpoints/paper_drafts/section5_mechanism.md:165:4. ✅ Evidence status table added at end of §5.1 — geometry strong / patching causal-continuation / Exp 5 axis-2 CI pending / steering weak / output divergence not classifier / W6 trigger exploratory
docs/checkpoints/paper_drafts/section5_mechanism.md:169:- **§4 P-text adjusted SR inconsistency**: §4 table line 37 says 11.90, prose line 106 says 12.38, hero_claim_bootstrap_ci.md says 12.38. Need to canonicalize one number from episode-level adjusted-success and update every occurrence. (1h)
docs/checkpoints/paper_drafts/section5_mechanism.md:170:- **plan.md:125-135 stale "L17 planning site"**: plan still asserts L17 singular planning site, while new evidence shows cosine peak L23 + patching peak L11-L17 + steering best L33; replace with "patch-sensitive continuation window L11-L17 under final-token replacement patching". (1-2h)
docs/checkpoints/paper_drafts/section5_mechanism.md:172:- **Exp 5 cellhprompt bootstrap CI + content-matched control**: Gaussian random injection control 359719/359720 in flight; codex notes Gaussian alone is weak — also need task-shuffled (source from different task) and per-task bootstrap CIs.
docs/checkpoints/paper_drafts/section5_mechanism.md:173:- **Behavioral causal bridge gap**: patching displaces 50-token continuation, not SR / drop-one. No experiment currently bridges mid-layer patching effect to task-success outcome. Open work.
docs/checkpoints/paper_drafts/section5_mechanism.md:176:**Codex verdict**: 0.10-0.20 NeurIPS/ICML/ACL main accept probability; would reject @ reviewer-3 4/10. **Workshop / borderline mid-tier conference today**. Codex agrees the drop-one oracle CI + cross-site asymmetry + axis-2 cosine signal breadth are strong; §5 mechanism prose was over-claiming relative to evidence.
docs/checkpoints/mechanism/plan.md:19:| **4** | Model-internal | L17 mid-layer is BOTH discrimination locus (probe AUROC 1.0) AND causally active planning site (Stage 2/3 patching + Method 4.4 v2 reliability) |
docs/checkpoints/mechanism/plan.md:21:### 1.2 Three-axis hierarchy quantified (Method 4.2 PCA cosine gap, Qwen3-VL-4B B1 cls)
docs/checkpoints/mechanism/plan.md:23:| Axis | Peak cosine gap | Peak layer | Magnitude ratio |
docs/checkpoints/mechanism/plan.md:27:| Prompt-axis (SoM-prompt vs DOM-prompt alone) | 0.007 | L36 | **1×** |
docs/checkpoints/mechanism/plan.md:29:→ Mechanism magnitude image >> text > prompt. Validates `project_phantom_space_axes_format_not_information.md` memory: P-SoM closest mode at every layer is **P-text** (text-axis sibling, L17 cosine 0.0028 vs P-SoM↔SoM 0.0412 = 14.7× more distant).
docs/checkpoints/mechanism/plan.md:31:### 1.3 Image-axis peak-layer dichotomy (Mirage mechanism signature)
docs/checkpoints/mechanism/plan.md:33:Method 4.2 reveals image-axis cosine-gap peak shifts based on text format of the no-image side. Clean dichotomy, zero overlap across 8 image-axis pairs:
docs/checkpoints/mechanism/plan.md:35:| No-image side text | Peak layer | Pairs |
docs/checkpoints/mechanism/plan.md:37:| AXTree (hierarchical) | **L04** | DOM↔Vision, DOM↔SoM, P-prompt↔Vision, P-prompt↔SoM |
docs/checkpoints/mechanism/plan.md:38:| [SOM_MARKS] / flat | **L17–L36** | P-text↔Vision, P-text↔SoM, P-SoM↔Vision, P-SoM↔SoM |
docs/checkpoints/mechanism/plan.md:42:Format variation extraction (Myriad job 352998, `stage4_format_variation_b1_cls`, 450 hidden states = 45 task-step × 10 modes). For each text format V, compute image-axis cosine gap V↔som per layer; peak layer reveals shortcut activation:
docs/checkpoints/mechanism/plan.md:44:| Format | Peak layer | Verdict |
docs/checkpoints/mechanism/plan.md:46:| **AXTree hierarchical (DOM)** | **L04** | **SOLE format defeating shortcut** |
docs/checkpoints/mechanism/plan.md:48:| `[N] role 'label'` (SoM standard) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:49:| `@N label` (Browser Use) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:50:| `id_N: label` (AppAgent) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:51:| `[BN:r:l]` (Tarsier) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:52:| `N. label` (numbered) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:53:| `<el_N>label</el_N>` (XML) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:54:| `#hash label` (control: no integer) | L36 | **still triggers!** |
docs/checkpoints/mechanism/plan.md:56:**Refined H1 verdict**: trigger is **flat element listing**, not "indexed list pattern". Even integer-free hash IDs and pure-sentence variants engage the shortcut. AXTree hierarchical depth is the **unique format** that defeats shortcut activation.
docs/checkpoints/mechanism/plan.md:64:| **Wu et al. 2026** (UCL lab, our advisors) | Method backbone | `wu2026toolcalling` (2605.07990) | Mean-difference activation steering at second-to-last layer, 77–100% switch on tool selection (93–100% at 4B+). Our Method 4.2/4.4 port to multimodal Qwen3-VL-4B web agent |
docs/checkpoints/mechanism/plan.md:65:| **Ma & Rui 2026** | Probe-vs-causal vocabulary | `maRui2026planning` (2605.07984) | "Planning-compatible representation" vs "causally active planning site". Qwen3-family pattern: probe works, causal patching weak (1% rhyme newline causal vs Gemma 67%). Our Method 4.4 v2 50% reliability is consistent with this family pattern |
docs/checkpoints/mechanism/plan.md:67:| **Lin & Liu 2026 Position paper** | Methodology protocol | `linLiu2026disclosure` (2605.08012) | 5-step identification disclosure norm: state claim / name strategy / enumerate assumptions / stress-test / separate validation. Paper §5 adopts as identification subsection structure |
docs/checkpoints/mechanism/plan.md:68:| **Peale et al. 2026** | §6 routing theory | `peale2026flexibleRouting` (2605.07805) | Uncertainty decomposition (reducible + irreducible) with regret bound. Paper §6 theoretical anchor; 4-fold drop-in maps onto predict/route/abstain trichotomy |
docs/checkpoints/mechanism/plan.md:74:`scripts/mechanistic/curate_mirage_tasks.py` builds a strong-tier task manifest from a B1 production run:
docs/checkpoints/mechanism/plan.md:75:- Filter to tasks where Mirage Effect is well-defined: agent succeeds in `som` mode but fails in `dom` mode (real image-feature dependence, not random failure)
docs/checkpoints/mechanism/plan.md:77:- Archive raw observations (AXTree, screenshots, SoM-marks) per (task, step) → `archive_subset_b1_<site>/<site>_task_<tid>/step_<step>/` for offline patching/extraction (Stages 2-4 read from this archive, no live env required)
docs/checkpoints/mechanism/plan.md:80:- `results/mechanistic/curate_mirage_b1_classifieds/manifest.json` — cls strong/reverse tier task list
docs/checkpoints/mechanism/plan.md:81:- `results/mechanistic/curate_mirage_b1_reddit/manifest.json` — reddit strong tier
docs/checkpoints/mechanism/plan.md:85:### 3.1 Method 4.2 — PCA cosine gap (DONE)
docs/checkpoints/mechanism/plan.md:87:`scripts/analysis/stage4_pca_cosine_gap.py` + `stage4_robustness.py`. Three metrics per (mode_pair, layer):
docs/checkpoints/mechanism/plan.md:89:- B. AUROC via (mean_A − mean_B) projection
docs/checkpoints/mechanism/plan.md:90:- C. Per-(mode, layer) PCA top-10 variance explained
docs/checkpoints/mechanism/plan.md:97:- Test E bootstrap 95% CI tight (4-15% of mean)
docs/checkpoints/mechanism/plan.md:116:- **LA-HDMI**: probe-free gradient steering (Khorasani 2026 method). Per-input optimization replaces fixed mean-diff direction. May overcome Qwen3-family causal patching weakness
docs/checkpoints/mechanism/plan.md:121:## 4. Identification protocol (Lin & Liu 2026 disclosure norm)
docs/checkpoints/mechanism/plan.md:127:> Mid-layer L17 hidden state at last-token position is the causally active planning site for phantom routing space mode selection in Qwen3-VL-4B web agents.
docs/checkpoints/mechanism/plan.md:132:1. **Probe-level** (Method 4.2 PCA cosine gap, AUROC 1.000 across 540 tests)
docs/checkpoints/mechanism/plan.md:133:2. **Replacement patching** (Stage 2/3 Cell A-H, L17 disruption peak, 8/8 Holm-sig)
docs/checkpoints/mechanism/plan.md:134:3. **Additive steering** (Method 4.4 v2, mid-layer L17 α=5 H-mean reliability 0.44)
docs/checkpoints/mechanism/plan.md:140:| A1 | L17 last-token hidden state mediates action selection (not earlier obs token positions) | Stage 2/3 swept all layers, L17 is peak |
docs/checkpoints/mechanism/plan.md:144:| A5 | Replacement patching faithfully simulates "natural" model read of the representation | Cell E random-injection control rules out non-specific disruption — content-specific causation confirmed |
docs/checkpoints/mechanism/plan.md:148:Cell E random-injection control: replacing source hidden with Gaussian noise (same μ, σ) yields **null L17 disruption effect**. Confirms our patching effect is source-content-specific, not noise-driven. Most directly stresses A5.
docs/checkpoints/mechanism/plan.md:152:- Method 4.2 AUROC 1.000 = validation (decodability)
docs/checkpoints/mechanism/plan.md:158:### 5.1 Stage 4 Method 4.2 (Qwen3-VL-4B B1 cls, 288 examples × 37 layers)
docs/checkpoints/mechanism/plan.md:160:| Pair @L17 | Cosine gap | 95% CI | AUROC |
docs/checkpoints/mechanism/plan.md:167:### 5.2 Stage 2/3 patching disruption (14 cells, B1 cls + reddit)
docs/checkpoints/mechanism/plan.md:169:**Stage 2 — P-SoM ↔ SoM patching (10 cells):**
docs/checkpoints/mechanism/plan.md:171:| Cell | Site | Direction | L17 Δoverlap | Holm-sig |
docs/checkpoints/mechanism/plan.md:177:| E | cls | random injection | -0.03 (uniform) | ✓ (negative control) |
docs/checkpoints/mechanism/plan.md:181:| Er | reddit | random injection | ~0 (uniform) | ✓ |
docs/checkpoints/mechanism/plan.md:191:| H-t-red | reddit | SoM → P-text | L01 (0.194) | -0.24 | `stage3_cellht_red_fwd_text_myriad/` |
docs/checkpoints/mechanism/plan.md:194:**Stage 3 interpretation (6/6 cells complete 2026-05-12)**: All forward SoM→{no-image-arm} patching cells show mid-layer L11-L17 disruption -0.19 to -0.33 Δoverlap→tgt. Magnitude > random injection control (Cell E -0.03) at all 6. **Mechanism additivity confirmed**: image-feature axis is shared substrate across DOM / P-text / P-prompt arms — single SoM→{any-no-image-arm} patching displaces target prediction toward source. Cross-site cls + reddit both replicate (paper §5 universal mid-layer fusion locus); reddit fusion locus slightly earlier (L11 vs cls L17), magnitude identical.
docs/checkpoints/mechanism/plan.md:196:Stage 3 cross-site DOM-axis additivity table (paired-test Δoverlap-to-target from `patching_continuation_results.json`):
docs/checkpoints/mechanism/plan.md:203:All 6 cells best layer 落在 **L12-L18 mid-layer 窗口** (tight 7-layer band), Δ range [-0.27, -0.35]. Cross-site / cross-arm 一致, mid-layer fusion locus 不是 single layer index 而是稳定窗口.
docs/checkpoints/mechanism/plan.md:207:H-mean reliability (HDMI framework) per (layer, α). **L17 α=5 smoke claim REFUTED by full sweep**; actual sweet spot at L33 α=10:
docs/checkpoints/mechanism/plan.md:219:- Mid-layer (L11-L23): **selectivity 100%** at all α (JSON envelope preserved), but completeness 0-11% (modest shift)
docs/checkpoints/mechanism/plan.md:220:- Late-layer (L33): completeness 38% (highest), but selectivity drops to 29% (over-steers JSON)
docs/checkpoints/mechanism/plan.md:225:### 5.4 Image-axis peak-layer dichotomy (Method 4.2, 8 pairs)
docs/checkpoints/mechanism/plan.md:227:`docs/checkpoints/mechanism/results/layer_axis_emergence.md`. AXTree-no-image side → L04 peak (4/4); [SOM_MARKS]-no-image side → L17–L36 peak (4/4). Zero overlap. Mirage Effect mechanism signature.
docs/checkpoints/mechanism/plan.md:231:`docs/checkpoints/mechanism/results/format_variation_h1_test.md`. 8 industry-relevant text formats + 2 controls. AXTree hierarchical (DOM) is **unique format** preserving L04 image-axis peak; all 8 flat-list variants (SoM standard, Browser Use @, AppAgent id_, Tarsier typed, plain numbered, XML tagged, hash-ID control, plain-sentence control) shift peak to L17–L36. Trigger is flat element listing, not specific token pattern.
docs/checkpoints/mechanism/plan.md:238:| ✅ H1 test: do all flat-list formats trigger shortcut? | **Closed 2026-05-12 00:00**: YES, including hash-ID + plain-sentence controls. AXTree-DOM is sole defeating format | — |
docs/checkpoints/mechanism/plan.md:240:| ✅ Cross-site Method 4.2 — does cls finding replicate on reddit? | **Closed 2026-05-12 16:30**: P-SoM↔DOM L17=0.0098 + P-SoM↔SoM L17=0.0423, AUROC 1.0 → Mirage signature replicated. See §7.3.1 | — |
docs/checkpoints/mechanism/plan.md:245:| AXTree-defeats-shortcut mechanism — *why* hierarchy beats flat? Cross-modal attention specific to indentation tokens? | High (paper §5 supplement) | Activation patching at L4 with hierarchical-text vs flat-text → see which attention heads pre-disrupt image embedding |
docs/checkpoints/mechanism/plan.md:251:- 2026-04-09 笔记 §19: I first grok the paper (then "Anonymous 2026 ACL"), record cosine gap method + L23+ steering 80-93%
docs/checkpoints/mechanism/plan.md:261:Updated after v2 full sweep + H1 test. Key revisions from §125.10 draft:
docs/checkpoints/mechanism/plan.md:262:- ❌ Removed: "L17 α=5 H-mean 0.44 mid-layer sweet spot" (smoke variance artifact, full data refutes)
docs/checkpoints/mechanism/plan.md:263:- ✓ Added: **L33 α=10 H-mean 0.33** = matches your second-to-last-layer choice; multi-step JSON selectivity drop explains 38% vs your 93% gap
docs/checkpoints/mechanism/plan.md:265:- ✓ Three asks: (a) attribution co-author vs cite + independent; (b) your ablation on mid- vs late-layer (we see selectivity tradeoff); (c) SAE direction priority given mean-diff ceiling
docs/checkpoints/mechanism/plan.md:274:> # 1. Method 4.2 PCA cosine gap port 到 6 modes
docs/checkpoints/mechanism/plan.md:275:> 24 cls strong-tier × 2 step × 6 mode = 288 hidden states, 37 layer × 2560 dim。全 540 pair × layer AUROC = 1.000 (perm baseline 0.629, real 9.8σ above). 你方法在 multimodal Qwen 上 readable transfer 干净。
docs/checkpoints/mechanism/plan.md:278:> 45 task-step × 6 layer × 5 α full sweep. 用 HDMI completeness×selectivity → H-mean 评估:
docs/checkpoints/mechanism/plan.md:280:>   - **L33 α=10 H-mean 0.33** (sweet spot, c=38% s=29%) ← matches 你 paper second-to-last-layer
docs/checkpoints/mechanism/plan.md:281:>   - Mid-layer (L11-L23) selectivity 100% 但 completeness 0-11% — readable but not effectively steerable
docs/checkpoints/mechanism/plan.md:282:>   - 你 paper Qwen 3 4B 93% switch vs 我 38% — 我猜原因是 multi-step JSON gen 的 selectivity 是真约束 (你 single-token tool decision selectivity 自动 1.0)
docs/checkpoints/mechanism/plan.md:285:> 测了 8 个 industry-relevant text format (Browser Use @, AppAgent id_, Tarsier typed, numbered, XML, hash-ID, plain-sentence + SoM baseline) vs AXTree-DOM:
docs/checkpoints/mechanism/plan.md:287:>   - 全 8 flat variants peak L17/L36 (= 都触发 shortcut)
docs/checkpoints/mechanism/plan.md:288:>   - **AXTree hierarchical 是唯一保留 L04 peak 的 format**
docs/checkpoints/mechanism/plan.md:289:>   - 包括 hash-ID (no integer) + plain-sentence (no list) 都触发
docs/checkpoints/mechanism/plan.md:295:> (2) 你 ablation 里有跑过 mid- vs late-layer 对比吗? 我 mid-layer selectivity 100% 但 shift 弱, late-layer shift 强但 envelope 破 — 不知道你 tool calling 上是不是也有这种 tradeoff。
docs/checkpoints/mechanism/plan.md:310:| **P4** | cls reverse-tier (selection-bias defense) | Myriad 353763 | ✅ **done 18:50:46** — shape (260, 37, 2560), 10 modes, 46 MB pulled. Same pattern as cls strong-tier (L36 marks-like + L04 dom). Selection-bias defended | `stage4_format_variation_b1_cls_reverse/hidden_states.npz` |
docs/checkpoints/mechanism/plan.md:311:| **P5a** | reddit format variation (cross-site H1) | Myriad **354382** (3rd attempt) | ✅ **done 08:09:38** — shape (430, 37, 2560), 10 modes, 76 MB pulled | `stage4_format_variation_b1_reddit/hidden_states.npz` |
docs/checkpoints/mechanism/plan.md:312:| **P5b** | reddit Method 4.2 multimode (cross-site Mirage) | Myriad 353890 | ✅ **done 07:31:14** — 288 examples, 6 modes, 51 MB pulled | `stage4_multimode_b1_reddit/hidden_states.npz` |
docs/checkpoints/mechanism/plan.md:336:### 7.3.0 Exp 1 axis-2 layer profile (2026-05-12 21:00 — three-axis hierarchy)
docs/checkpoints/mechanism/plan.md:338:`axis2_layer_profile.md` + `fig_axis2_prompt_layer_profile.png`. Re-examine residual stream geometry per axis-isolated pair, full 37-layer cosine curves on `stage4_multimode_b1_{cls,reddit}` (288 ex each).
docs/checkpoints/mechanism/plan.md:340:Cls site peak layers + magnitudes:
docs/checkpoints/mechanism/plan.md:342:| Pair | Group | L17 | L23 | L36 | Peak L | Peak gap |
docs/checkpoints/mechanism/plan.md:348:| DOM↔P-prompt (prompt fam, hier) | axis-2 | 0.0013 | 0.0050 | 0.0067 | L36 | 0.0067 |
docs/checkpoints/mechanism/plan.md:350:Reddit cross-site replicates: P-text↔P-SoM L23 = 0.0098 (vs cls 0.0114), same rank-order, same peak layer.
docs/checkpoints/mechanism/plan.md:353:1. **Distinct peak layers**: image L17 (fast sharp), text-format L23 (slower late-mid), prompt-family L23 (same timing as text-format on flat-text)
docs/checkpoints/mechanism/plan.md:357:**Reframe**: Axis-2 prompt-family is NOT null at residual stream. It's 3-4x weaker than axis-1 + peaks at L23 not L17. Method 4.2 plan §5.1 L17 snapshot 错失它. New paper §5 framing: layered three-axis hierarchy, image-axis dominant at L17 Mirage locus, text-format + prompt-family late-mid build at L23 parallel.
docs/checkpoints/mechanism/plan.md:363:`axis2_per_task_fragility.md` + `fig_axis2_per_task_fragility.png`. /stress reviewer 第一次 invocation W2 attack: 怀疑 axis-2 cosine 0.0114 mean 由 2-3 outlier 主导, 类比 h1_per_task_fragility 11% strict per-task. Defuse 实验:
docs/checkpoints/mechanism/plan.md:378:**/stress W2 attack defused completely**: axis-2 cosine gap 是 uniform per-task signature, 不是 aggregate artifact. 这与 H1 binary dichotomy 11% strict per-task fragile 形成对比 — H1 因为问 layer-comparison 离散问题易 fragile, axis-2 cosine 是 continuous mode-pair distance 即使 magnitude 小也 robust per-task.
docs/checkpoints/mechanism/plan.md:382:### 7.3.0a Exp 3 logit lens 输出层 amplification (2026-05-12 21:02)
docs/checkpoints/mechanism/plan.md:384:`axis2_logit_lens.md` + `fig_axis2_logit_lens.png`. 应用 Qwen3-VL-4B `model.model.language_model.norm` + `model.lm_head` to per-layer per-mode mean hidden states, 算 KL across 37 层.
docs/checkpoints/mechanism/plan.md:386:| Pair | Site | Peak L (KL) | Peak KL | Exp 1 cosine peak | 放大倍数 |
docs/checkpoints/mechanism/plan.md:395:1. Axis-2 prompt-family **IS in output distribution** — KL 0.16 at L23, NOT null. Exp 1 cosine 0.011 is not the end of the story.
docs/checkpoints/mechanism/plan.md:396:2. **lm_head 10-25x amplification of cosine → KL** but axis-agnostic ratio preserved (axis-1/axis-2 ratio ~4.3 cls, ~4.9 reddit, vs cosine ratio ~3 — slight amplification of stronger axis but not breaking 3-4x rank).
docs/checkpoints/mechanism/plan.md:397:3. **KL @ L36 ≈ 0 paradox**: 因 mean hidden state at last layer collapse to common JSON format header. Mode-distinct signal concentrated in **L23-L25 decoding window** (not final embedding). This is the "knows but says differently" structural mirror of Wu et al. tool calling.
docs/checkpoints/mechanism/plan.md:399:**Paper §5.7 follow-up paragraph** added: 三轴 hierarchy persists at output distribution with same rank-order. Deployment routing (paper-2) should treat L23-L25 logit-lens features as cheapest highest-signal mode-axis discriminator.
docs/checkpoints/mechanism/plan.md:407:| som_standard / browser_use_at / tarsier_typed / xml_tagged | **L17** | L36 (last) |
docs/checkpoints/mechanism/plan.md:408:| appagent_id / plain_numbered | **L04** | L36 |
docs/checkpoints/mechanism/plan.md:409:| hash_id_control | **L04** ✓ (acts as control) | L36 (control failed) |
docs/checkpoints/mechanism/plan.md:411:| dom (baseline) | **L04** ✓ | L04 ✓ |
docs/checkpoints/mechanism/plan.md:413:**Reddit nuance — cleaner mid-layer fusion**: Reddit 上 marks-like 4/6 真 peak 在 L17 (mid-layer), cls 上 L36 是 monotonic increasing artifact (peak hit boundary). Reddit hash_id_control L04 acts as proper "no integer" control (cls 上失败). Reddit data supports Q5 mid-layer fusion hypothesis better than cls.
docs/checkpoints/mechanism/plan.md:415:Caveats: small n (24×2=48/mode) makes 2/6 marks-like falling to L04 (appagent_id, plain_numbered) plausible as sampling noise; plain_sentence triggering L17 on reddit (not cls) suggests reddit narrative comments may pattern-match list semantics.
docs/checkpoints/mechanism/plan.md:424:| DOM ↔ Vision peak | L04 = 0.0687 (AUROC=1.0) | L04 similar |
docs/checkpoints/mechanism/plan.md:429:1. P-SoM mid-layer mechanism (4-fold drop-one) — cls + reddit replicated ✓
docs/checkpoints/mechanism/plan.md:437:| 6 marks-like | L36 monotonic | **L36 monotonic** ✓ same | L17 (4/6 真 peak) |
docs/checkpoints/mechanism/plan.md:438:| hash_id_control | L36 (failed control) | **L36** ✓ same | L04 ✓ proper control |
docs/checkpoints/mechanism/plan.md:440:| dom baseline | L04 ✓ | **L04** ✓ | L04 ✓ |
docs/checkpoints/mechanism/plan.md:442:H1 mechanism in cls is **not tier selection artifact** (strong vs reverse both replicate). Reddit data paradoxically cleaner reveal of true L17 mid-layer fusion locus (cls L36 is monotonic-boundary artifact).
docs/checkpoints/mechanism/plan.md:464:- **§6 cost-aware routing** — Peale et al. 2026 uncertainty decomposition anchor adds theoretical layer to phantom routing space's empirical AUROC. Method 4.2 cosine gap could serve as "reducible uncertainty" signal in deployment
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:13:| Layers | 37 (L0-L36, Qwen3-VL-4B language decoder) | matching |
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:15:**Test logic**: Holding both `image` and `text-format` constant (off + flat) and patching source hidden states from `phantom_som` into a `phantom_text` run isolates whether the residual-stream prompt-family signature has *causal* effect on token continuation, not just *geometric* magnitude (which Exp 1 already showed is small at 0.011 cosine gap @ L23).
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:17:## Result — mid-layer (L11-L17) patching causal effect
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:26:(Baseline `overlap→tgt = 1.00` at L35 = full target preservation, no patching effect.)
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:28:### Causal weight decomposition
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:30:- Axis-2 (prompt) **alone** displaces target output by **0.20-0.30 overlap** units, mid-layer L11-L17 peak.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:31:- Combined image+prompt (H-text) displaces by **0.24-0.26** at same layers.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:36:Both cls + red show the same mid-layer L11-L17 peak. Reddit shows *stronger* axis-2 effect at L17 than cls (overlap→tgt 0.70 vs 0.79).
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:38:## Geometric ⫨ causal disjoint (two disjoints: magnitude AND layer)
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:42:Compared with Exp 1 cosine geometry, using best-layer values:
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:44:| Axis | Cosine gap (best layer) | Patching displacement (best causal layer) |
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:50:**4:3:1 cosine geometry ratio does NOT translate to 4:3:1 causal patching ratio.** Prompt-family has the **smallest** geometric magnitude but the **largest** causal patching weight.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:52:### Disjoint 2 — layer
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:54:Critically, the **layer at which cosine peaks ≠ the layer at which patching has maximal effect** for prompt-family:
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:58:| L0  | 0.86 | 0.92 | early, signal not yet routed |
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:61:| **L23** | 0.96 | 0.89 | **cosine geometry peak, but patching weak** — representation stabilized |
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:65:At **L23** (the cosine peak), patching displaces target output by only **0.04-0.11 overlap units** — much smaller than the **0.20-0.30** displacement at L11-L17.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:71:- **L23 is the prompt-family "signature layer"**: representation has stabilized to its most discriminable form (highest cosine separation between P-SoM and P-text). It reflects *what prompt was given* — a state variable.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:72:- **L11-L17 is the prompt-family "decision routing layer"**: patching here changes upstream signal that downstream layers consume to drive token continuation. It reflects *how the model uses the prompt* — a causal variable.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:74:Activation patching is path-dependent: an upstream patch propagates into all downstream computations, while a downstream patch leaves upstream inputs unchanged so subsequent layers can re-encode the same signal. This is consistent with standard mechanistic-interpretability findings (cf. \citep{wang2023interpretability} IOI circuit: feature *encoded* ≠ feature *used*).
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:78:1. Residual-stream cosine separation is a **necessary but not sufficient** signal of causal mechanism.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:79:2. Prompt-family information is **dispatchable** — small geometric perturbation at the decision layer produces large output displacement when patched.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:80:3. **Where a feature is most readable (L23) and where it is most consequential (L11-L17) are different layers** — paper-grade mechanism claims must report both, not collapse them.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:85:- Axis-1 (text-format): Exp 1 cosine 0.029 + H-d cells causal patching (prior)
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:86:- Axis-2 (prompt-family): Exp 1 cosine 0.011 + **Exp 5 cellhprompt causal patching (this)**
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:87:- Axis-image: Exp 1 cosine 0.041 + indirect (H-text − cellhprompt residual ~0.04-0.05)
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:91:- 2-site cross-replication (cls + red), N=24 each, paired L0-L36 sweep
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:94:- Behavioral: exploration rate axis-1 dependent (Exp 1 cosine sigma + §4.5 reddit behavioral)
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:95:- Causal mechanism: prompt-family mid-layer L11-L17 patching produces output displacement comparable to image-axis flip
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:99:- N=24 per cell — bootstrap CI on per-layer overlap means would tighten interpretation.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:100:- "phantom_som" archive vs "som" archive — the codebase uses same hidden-state extraction infrastructure but `--source-mode phantom_som` extracts without image. The pilot_summary template label "(with image — clean)" is a hardcoded artifact, not a runtime check. Manual verification of the archive contents (hidden state norms) would close a soundness gap.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:101:- Patching displacement is a token-level metric; doesn't directly translate to SR / drop-one oracle. Behavioral consequence (which paper §1 hero is about) operates on top of this causal signal.
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:106:- `patching_continuation_results.json`: per-layer per-task continuation strings + metrics (~1.3 MB each)
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:107:- `patching_continuation_curves.png`: visual layer profile
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:112:- Watcher missed GONE events due to silent-miss bug (PR same commit) — auto_pull dispatched manually
scripts/mechanistic/run_stage2b_continuation_pilot.py:1:"""Stage 2B — Multi-token continuation activation patching for B1 mirage analysis.
scripts/mechanistic/run_stage2b_continuation_pilot.py:3:Addresses Stage 2A trivial first-token-agree problem (source argmax == target argmax
scripts/mechanistic/run_stage2b_continuation_pilot.py:4:== JSON `{` opener forced by chat template). Generate 10-15 tokens past `{` so action_type
scripts/mechanistic/run_stage2b_continuation_pilot.py:5:+ element_id divergence between source/target emerges, then measure how patching at each
scripts/mechanistic/run_stage2b_continuation_pilot.py:6:layer pulls patched continuation toward source.
scripts/mechanistic/run_stage2b_continuation_pilot.py:11:- Per (task) × per (layer L = 0..35): patch source's L-th hidden state into target,
scripts/mechanistic/run_stage2b_continuation_pilot.py:12:  greedy-generate max_new_tokens, compare full token sequences.
scripts/mechanistic/run_stage2b_continuation_pilot.py:15:- token_overlap_to_source: ratio of positions where patched matches source (1=identical)
scripts/mechanistic/run_stage2b_continuation_pilot.py:16:- token_overlap_to_target: same vs target baseline
scripts/mechanistic/run_stage2b_continuation_pilot.py:17:- ld_to_source: Levenshtein edit distance to source token sequence
scripts/mechanistic/run_stage2b_continuation_pilot.py:18:- ld_to_target: Levenshtein edit distance to target token sequence
scripts/mechanistic/run_stage2b_continuation_pilot.py:22:      patching_continuation_results.json
scripts/mechanistic/run_stage2b_continuation_pilot.py:23:      patching_continuation_curves.png
scripts/mechanistic/run_stage2b_continuation_pilot.py:28:      --site classifieds --n-tasks 3 --step 2 --max-new-tokens 15 \
scripts/mechanistic/run_stage2b_continuation_pilot.py:31:ETA: 3 task × (1 source-gen + 1 target-gen + 1 source-cache + 36 patched-gen)
scripts/mechanistic/run_stage2b_continuation_pilot.py:48:# (sm_75) Myriad nodes, SDPA dispatcher raises "cutlassF: no kernel found to
scripts/mechanistic/run_stage2b_continuation_pilot.py:64:from p79.mechanistic.activation_patching import ActivationPatcher, patching_grid_continuation
scripts/mechanistic/run_stage2b_continuation_pilot.py:65:from p79.mechanistic.extract_hidden_states import HiddenStateExtractor
scripts/mechanistic/run_stage2b_continuation_pilot.py:94:def load_intents_from_subset_manifest(manifest_path: Path, tier: str, n_tasks: int) -> list[tuple[int, str]]:
scripts/mechanistic/run_stage2b_continuation_pilot.py:95:    """Load intents from archive_subset manifest.json (cross-machine paper-grade
scripts/mechanistic/run_stage2b_continuation_pilot.py:97:    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
scripts/mechanistic/run_stage2b_continuation_pilot.py:98:    entries = manifest.get(tier, [])[:n_tasks]
scripts/mechanistic/run_stage2b_continuation_pilot.py:137:    text = extractor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
scripts/mechanistic/run_stage2b_continuation_pilot.py:155:    p.add_argument("--max-new-tokens", type=int, default=15, help="Continuation length (15 covers JSON envelope start)")
scripts/mechanistic/run_stage2b_continuation_pilot.py:164:        help="Swap source ↔ target: patch target's hidden state into source run "
scripts/mechanistic/run_stage2b_continuation_pilot.py:169:        help="Override manifest tier subset (strong=24 forward-mirage / reverse=15 reverse-mirage). "
scripts/mechanistic/run_stage2b_continuation_pilot.py:175:        "--random-inject", action="store_true",
scripts/mechanistic/run_stage2b_continuation_pilot.py:176:        help="Random-injection control (paper §5 reviewer Q): replace cached source "
scripts/mechanistic/run_stage2b_continuation_pilot.py:177:             "hidden state with Gaussian noise matched to per-layer mean/std. Tests "
scripts/mechanistic/run_stage2b_continuation_pilot.py:178:             "whether mid-layer disruption depends on source-content specificity vs "
scripts/mechanistic/run_stage2b_continuation_pilot.py:179:             "any non-zero injection. Expected null at all layers if mechanism is "
scripts/mechanistic/run_stage2b_continuation_pilot.py:183:        "--random-seed", type=int, default=42,
scripts/mechanistic/run_stage2b_continuation_pilot.py:184:        help="Seed for --random-inject Gaussian noise (paper-grade reproducibility). "
scripts/mechanistic/run_stage2b_continuation_pilot.py:185:             "Same seed + same input = same noise = byte-identical re-runs. Default 42.",
scripts/mechanistic/run_stage2b_continuation_pilot.py:189:    # C8 fix: seed all RNGs when random-inject is on, for paper-grade
scripts/mechanistic/run_stage2b_continuation_pilot.py:190:    # reproducibility. Affects torch.randn_like in patching_grid_continuation.
scripts/mechanistic/run_stage2b_continuation_pilot.py:191:    # Default seed=42 means re-running with same data + code produces
scripts/mechanistic/run_stage2b_continuation_pilot.py:192:    # byte-identical noise + byte-identical patched outputs.
scripts/mechanistic/run_stage2b_continuation_pilot.py:193:    if args.random_inject:
scripts/mechanistic/run_stage2b_continuation_pilot.py:194:        import random as _rnd
scripts/mechanistic/run_stage2b_continuation_pilot.py:197:        _rnd.seed(args.random_seed)
scripts/mechanistic/run_stage2b_continuation_pilot.py:198:        _np.random.seed(args.random_seed)
scripts/mechanistic/run_stage2b_continuation_pilot.py:199:        _t.manual_seed(args.random_seed)
scripts/mechanistic/run_stage2b_continuation_pilot.py:201:            _t.cuda.manual_seed_all(args.random_seed)
scripts/mechanistic/run_stage2b_continuation_pilot.py:202:        # Defense-in-depth visibility (防忘): log prominently + this seed
scripts/mechanistic/run_stage2b_continuation_pilot.py:204:        print(f"\n{'=' * 60}\n[RANDOM-INJECT SEED] {args.random_seed} "
scripts/mechanistic/run_stage2b_continuation_pilot.py:207:        # Note: cell E (job 335404) is currently running with NO seed (commit
scripts/mechanistic/run_stage2b_continuation_pilot.py:209:        # grade re-runs will be byte-reproducible with --random-seed 42.
scripts/mechanistic/run_stage2b_continuation_pilot.py:226:                "random_inject": args.random_inject,
scripts/mechanistic/run_stage2b_continuation_pilot.py:227:                "random_seed": args.random_seed,
scripts/mechanistic/run_stage2b_continuation_pilot.py:230:                "max_new_tokens": args.max_new_tokens,
scripts/mechanistic/run_stage2b_continuation_pilot.py:240:    # Auto-detect: if archived_run_dir contains manifest.json, it's a subset
scripts/mechanistic/run_stage2b_continuation_pilot.py:241:    # (extract_archive_subset.py output). Use intents from manifest, support
scripts/mechanistic/run_stage2b_continuation_pilot.py:244:    subset_manifest = archived_dir / "manifest.json"
scripts/mechanistic/run_stage2b_continuation_pilot.py:245:    if subset_manifest.exists():
scripts/mechanistic/run_stage2b_continuation_pilot.py:247:        intents = load_intents_from_subset_manifest(subset_manifest, tier=tier, n_tasks=args.n_tasks)
scripts/mechanistic/run_stage2b_continuation_pilot.py:249:            f"Subset mode: loaded {len(intents)} intents from manifest "
scripts/mechanistic/run_stage2b_continuation_pilot.py:261:    patcher = ActivationPatcher(extractor.model, extractor.processor)
scripts/mechanistic/run_stage2b_continuation_pilot.py:262:    logger.info(f"Model loaded; n_layers={patcher.n_layers}")
scripts/mechanistic/run_stage2b_continuation_pilot.py:278:        # H-prompt-red 24/24 patched_text identical, confirming bug).
scripts/mechanistic/run_stage2b_continuation_pilot.py:291:        # --reverse: swap roles. patch target's hidden into source run = "remove image content"
scripts/mechanistic/run_stage2b_continuation_pilot.py:294:            logger.info(f"task {task_id}: REVERSE direction (patching {args.target_mode} → {args.source_mode})")
scripts/mechanistic/run_stage2b_continuation_pilot.py:297:            logger.info(f"task {task_id}: forward direction (patching {args.source_mode} → {args.target_mode})")
scripts/mechanistic/run_stage2b_continuation_pilot.py:299:        logger.info(f"task {task_id}: running continuation patching grid (max_new_tokens={args.max_new_tokens})...")
scripts/mechanistic/run_stage2b_continuation_pilot.py:300:        result = patching_grid_continuation(
scripts/mechanistic/run_stage2b_continuation_pilot.py:301:            patcher, source_inputs, target_inputs,
scripts/mechanistic/run_stage2b_continuation_pilot.py:302:            max_new_tokens=args.max_new_tokens,
scripts/mechanistic/run_stage2b_continuation_pilot.py:303:            randomize_source_hidden=args.random_inject,
scripts/mechanistic/run_stage2b_continuation_pilot.py:310:        # F18 audit fix 2026-05-09: include reverse / tier / random_inject /
scripts/mechanistic/run_stage2b_continuation_pilot.py:311:        # random_seed in incremental JSON so downstream stage2 stat scripts
scripts/mechanistic/run_stage2b_continuation_pilot.py:317:        # patched into what), matching the per-task INFO log line above.
scripts/mechanistic/run_stage2b_continuation_pilot.py:325:        with (out_dir / "patching_continuation_results.json").open("w") as f:
scripts/mechanistic/run_stage2b_continuation_pilot.py:329:                    "max_new_tokens": args.max_new_tokens,
scripts/mechanistic/run_stage2b_continuation_pilot.py:336:                    "random_inject": args.random_inject,
scripts/mechanistic/run_stage2b_continuation_pilot.py:337:                    "random_seed": args.random_seed,
scripts/mechanistic/run_stage2b_continuation_pilot.py:340:                    "n_layers": patcher.n_layers,
scripts/mechanistic/run_stage2b_continuation_pilot.py:349:    # Aggregate per-layer mean ± std across tasks
scripts/mechanistic/run_stage2b_continuation_pilot.py:350:    n_layers = patcher.n_layers
scripts/mechanistic/run_stage2b_continuation_pilot.py:351:    metric_names = ["token_overlap_to_source", "token_overlap_to_target", "ld_to_source", "ld_to_target"]
scripts/mechanistic/run_stage2b_continuation_pilot.py:355:            [layer_r[m] for layer_r in t["per_layer"]]
scripts/mechanistic/run_stage2b_continuation_pilot.py:357:        ])  # (n_tasks, n_layers)
scripts/mechanistic/run_stage2b_continuation_pilot.py:367:    layers_x = np.arange(n_layers)
scripts/mechanistic/run_stage2b_continuation_pilot.py:369:        "token_overlap_to_source": "Token overlap → source\n(1=patched matches source position-by-position)",
scripts/mechanistic/run_stage2b_continuation_pilot.py:370:        "token_overlap_to_target": "Token overlap → target\n(higher = patch had no effect)",
scripts/mechanistic/run_stage2b_continuation_pilot.py:371:        "ld_to_source": f"Levenshtein dist → source\n(0=identical, max~{args.max_new_tokens})",
scripts/mechanistic/run_stage2b_continuation_pilot.py:372:        "ld_to_target": f"Levenshtein dist → target\n(higher = patch pulled away from target)",
scripts/mechanistic/run_stage2b_continuation_pilot.py:377:        ax.plot(layers_x, mean, marker="o", lw=1.5, label=f"mean (N={len(per_task_results)})")
scripts/mechanistic/run_stage2b_continuation_pilot.py:378:        ax.fill_between(layers_x, mean - std, mean + std, alpha=0.25, label="±1 std")
scripts/mechanistic/run_stage2b_continuation_pilot.py:391:        f"max_new_tokens={args.max_new_tokens})",
scripts/mechanistic/run_stage2b_continuation_pilot.py:395:    fig.savefig(out_dir / "patching_continuation_curves.png", dpi=150)
scripts/mechanistic/run_stage2b_continuation_pilot.py:397:    logger.info("Saved patching_continuation_curves.png")
scripts/mechanistic/run_stage2b_continuation_pilot.py:400:    overlap_src = np.array(agg["token_overlap_to_source_mean"])
scripts/mechanistic/run_stage2b_continuation_pilot.py:401:    overlap_tgt = np.array(agg["token_overlap_to_target_mean"])
scripts/mechanistic/run_stage2b_continuation_pilot.py:405:    best_overlap_layer = int(overlap_src.argmax())
scripts/mechanistic/run_stage2b_continuation_pilot.py:406:    best_ld_layer = int(ld_src.argmin())
scripts/mechanistic/run_stage2b_continuation_pilot.py:408:    # Sample some patched outputs for qualitative check
scripts/mechanistic/run_stage2b_continuation_pilot.py:415:            r = t["per_layer"][L]
scripts/mechanistic/run_stage2b_continuation_pilot.py:416:            qualitative.append(f"  L{L:2d} patched: {r['patched_text']!r}  (overlap→src={r['token_overlap_to_source']:.2f}, LD→src={r['ld_to_source']})")
scripts/mechanistic/run_stage2b_continuation_pilot.py:427:- max_new_tokens: {args.max_new_tokens} (greedy continuation, deterministic)
scripts/mechanistic/run_stage2b_continuation_pilot.py:428:- Random injection: {"YES, seed=" + str(args.random_seed) + " (paper-grade reproducible)" if args.random_inject else "NO (real source hidden injected)"}
scripts/mechanistic/run_stage2b_continuation_pilot.py:431:## Result (per-layer mean across tasks)
scripts/mechanistic/run_stage2b_continuation_pilot.py:432:- Best layer for **token overlap → source**: L{best_overlap_layer} (overlap {overlap_src[best_overlap_layer]:.3f})
scripts/mechanistic/run_stage2b_continuation_pilot.py:433:- Best layer for **min Levenshtein → source**: L{best_ld_layer} (LD {ld_src[best_ld_layer]:.2f})
scripts/mechanistic/run_stage2b_continuation_pilot.py:438:| L0  | {overlap_src[0]:.2f} | {overlap_tgt[0]:.2f} | {ld_src[0]:.1f} | {ld_tgt[0]:.1f} |
scripts/mechanistic/run_stage2b_continuation_pilot.py:447:- overlap→src curve climbs monotonically with depth → mirage info accumulates layer-by-layer (deep layer wins)
scripts/mechanistic/run_stage2b_continuation_pilot.py:448:- overlap→src peaks at middle layer then decays → "computed feature" emerges mid then abstracts
scripts/mechanistic/run_stage2b_continuation_pilot.py:449:- overlap→src flat ~0 → patching has no causal effect (mirage info distributed elsewhere)
scripts/mechanistic/run_stage2b_continuation_pilot.py:450:- LD→src minimum identifies "most source-like patched output" layer — opposite signal of overlap→src
scripts/mechanistic/run_stage2b_continuation_pilot.py:456:- If mid-layer peak emerges (e.g. L17-L25) → consistent with Stage 2A logit_shift L17 finding ✓
scripts/mechanistic/run_stage2b_continuation_pilot.py:457:- If late-layer monotone climb → mirage signature is residual-stream cumulative, no single causal layer
scripts/mechanistic/run_stage2b_continuation_pilot.py:458:- Scale up: 5 task × max_new_tokens=20 (~75 min) for tighter mean ± std
scripts/mechanistic/run_stage2b_continuation_pilot.py:459:- Then Stage 2C: reverse direction (target→source patching) for asymmetry check
scripts/mechanistic/run_stage2b_continuation_pilot.py:463:    # Paper-grade run manifest (Gap 3, 笔记 §114) — single-file roll-up of
scripts/mechanistic/run_stage2b_continuation_pilot.py:464:    # patch config + per-task outcomes for OSF DOI lock + cross-machine compare.
scripts/mechanistic/run_stage2b_continuation_pilot.py:465:    run_manifest = {
scripts/mechanistic/run_stage2b_continuation_pilot.py:469:        "patch_config": {
scripts/mechanistic/run_stage2b_continuation_pilot.py:473:            "max_new_tokens": args.max_new_tokens,
scripts/mechanistic/run_stage2b_continuation_pilot.py:474:            "n_layers_swept": int(patcher.n_layers),
scripts/mechanistic/run_stage2b_continuation_pilot.py:475:            "hook_position": "last_token",
scripts/mechanistic/run_stage2b_continuation_pilot.py:481:            "n_layers": int(patcher.n_layers),
scripts/mechanistic/run_stage2b_continuation_pilot.py:493:                "best_layer_overlap_src": int(np.argmax([r["token_overlap_to_source"] for r in t["per_layer"]])),
scripts/mechanistic/run_stage2b_continuation_pilot.py:494:                "best_overlap_src": float(max(r["token_overlap_to_source"] for r in t["per_layer"])),
scripts/mechanistic/run_stage2b_continuation_pilot.py:495:                "L11_overlap_src": float(t["per_layer"][11]["token_overlap_to_source"]) if patcher.n_layers > 11 else None,
scripts/mechanistic/run_stage2b_continuation_pilot.py:496:                "L17_overlap_src": float(t["per_layer"][17]["token_overlap_to_source"]) if patcher.n_layers > 17 else None,
scripts/mechanistic/run_stage2b_continuation_pilot.py:501:            "best_layer_overlap_src_mean": int(best_overlap_layer),
scripts/mechanistic/run_stage2b_continuation_pilot.py:502:            "best_overlap_src_mean": float(overlap_src[best_overlap_layer]),
scripts/mechanistic/run_stage2b_continuation_pilot.py:503:            "best_layer_ld_src_mean": int(best_ld_layer),
scripts/mechanistic/run_stage2b_continuation_pilot.py:504:            "L11_overlap_src_mean": float(overlap_src[11]) if patcher.n_layers > 11 else None,
scripts/mechanistic/run_stage2b_continuation_pilot.py:505:            "L17_overlap_src_mean": float(overlap_src[17]) if patcher.n_layers > 17 else None,
scripts/mechanistic/run_stage2b_continuation_pilot.py:509:            "per_task_jsonl": "patching_continuation_results.json",
scripts/mechanistic/run_stage2b_continuation_pilot.py:510:            "curves_plot": "patching_continuation_curves.png",
scripts/mechanistic/run_stage2b_continuation_pilot.py:514:    with (out_dir / "run_manifest.json").open("w") as f:
scripts/mechanistic/run_stage2b_continuation_pilot.py:515:        json.dump(run_manifest, f, indent=2)
scripts/mechanistic/run_stage2b_continuation_pilot.py:516:    logger.info(f"run_manifest.json emitted ({len(per_task_results)} tasks)")
scripts/mechanistic/run_stage2b_continuation_pilot.py:517:    logger.info(f"Stage 2B continuation patching pilot DONE → {out_dir}")
scripts/analysis/hero_claim_bootstrap.py:2:"""W1 hero-claim bootstrap CI (response to /stress W1 attack).
scripts/analysis/hero_claim_bootstrap.py:10:The reviewer demands per-seed bootstrap 95% CI on the pairwise comparison
scripts/analysis/hero_claim_bootstrap.py:15:modes (DOM, SoM, Vision, P-SoM, P-text, P-prompt), bootstraps 10000 task
scripts/analysis/hero_claim_bootstrap.py:19:  - P(diff > 0) — strict-positive bootstrap probability
scripts/analysis/hero_claim_bootstrap.py:20:  - P(diff > 1pp) — practical-significance bootstrap probability
scripts/analysis/hero_claim_bootstrap.py:26:  docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md
scripts/analysis/hero_claim_bootstrap.py:38:DEFAULT_OUT_MD = ROOT / "docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md"
scripts/analysis/hero_claim_bootstrap.py:100:def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, B: int = 10000, seed: int = 42):
scripts/analysis/hero_claim_bootstrap.py:103:    rng = np.random.default_rng(seed)
scripts/analysis/hero_claim_bootstrap.py:118:def bootstrap_drop_one_ci(M: np.ndarray, drop_mode_idx: int, modes: list[str],
scripts/analysis/hero_claim_bootstrap.py:119:                          B: int = 10000, seed: int = 42):
scripts/analysis/hero_claim_bootstrap.py:122:    rng = np.random.default_rng(seed)
scripts/analysis/hero_claim_bootstrap.py:144:    p.add_argument("--bootstraps", type=int, default=10000)
scripts/analysis/hero_claim_bootstrap.py:145:    p.add_argument("--seed", type=int, default=42)
scripts/analysis/hero_claim_bootstrap.py:149:    np.random.seed(args.seed)
scripts/analysis/hero_claim_bootstrap.py:151:        "# Hero-claim bootstrap CI (W1 defuse)",
scripts/analysis/hero_claim_bootstrap.py:153:        f"Per-seed bootstrap 95% percentile CI on paired adjusted-SR diffs and drop-one oracle. "
scripts/analysis/hero_claim_bootstrap.py:154:        f"B={args.bootstraps}, seed={args.seed}. Tasks resampled with replacement at task level.",
scripts/analysis/hero_claim_bootstrap.py:194:            "**Pairwise SR difference, bootstrap 95% CI:**",
scripts/analysis/hero_claim_bootstrap.py:200:            r = bootstrap_diff_ci(M[:, mi[a]], M[:, mi[b]], B=args.bootstraps, seed=args.seed)
scripts/analysis/hero_claim_bootstrap.py:215:                f"**Drop-one oracle on {len(core_modes)}-mode set ({', '.join(core_modes)}), bootstrap 95% CI:**",
scripts/analysis/hero_claim_bootstrap.py:221:                r = bootstrap_drop_one_ci(Msub, j, core_modes,
scripts/analysis/hero_claim_bootstrap.py:222:                                           B=args.bootstraps, seed=args.seed + j)
scripts/analysis/hero_claim_bootstrap.py:238:        "  §1 hero claim is bootstrap-supported. Remove the '2σ hedge' from line 5, lead with the magnitude.",
p79/mechanistic/activation_patching.py:1:"""Activation patching via PyTorch forward hooks (Stage 2 — causal mechanism analysis).
p79/mechanistic/activation_patching.py:3:Advisor 5/5 instruction: "patch 到哪一层的时候, 它的结果就切换了" — find the layer
p79/mechanistic/activation_patching.py:4:where source-into-target hidden-state injection causes output to flip. That's
p79/mechanistic/activation_patching.py:5:the mirage critical layer.
p79/mechanistic/activation_patching.py:13:- Patch source's layer-L last-token hidden state into target run at same layer.
p79/mechanistic/activation_patching.py:14:- If patched output flips toward source's behavior → layer L carries source info.
p79/mechanistic/activation_patching.py:27:def get_transformer_layers(model) -> torch.nn.ModuleList:
p79/mechanistic/activation_patching.py:28:    """Locate transformer decoder layer ModuleList in Qwen3-VL.
p79/mechanistic/activation_patching.py:31:        model.model.language_model.layers — 36 × Qwen3VLTextDecoderLayer
p79/mechanistic/activation_patching.py:33:    return model.model.language_model.layers
p79/mechanistic/activation_patching.py:37:    """Cache + patch interface for transformer layer outputs.
p79/mechanistic/activation_patching.py:39:    F16 audit clarification 2026-05-09: layers L0..L_{n-1} are transformer
p79/mechanistic/activation_patching.py:40:    BLOCK OUTPUTS (post-attention + ffn + residual + post-norm). L0 is the
p79/mechanistic/activation_patching.py:41:    output of the first decoder layer, NOT the embedding output. To get
p79/mechanistic/activation_patching.py:42:    embedding output, hook `model.model.language_model.embed_tokens`
p79/mechanistic/activation_patching.py:50:        self.layers = get_transformer_layers(model)
p79/mechanistic/activation_patching.py:51:        self.n_layers = len(self.layers)
p79/mechanistic/activation_patching.py:54:    def cache_hidden_states(self, **inputs) -> list[torch.Tensor]:
p79/mechanistic/activation_patching.py:55:        """Forward inputs and return per-layer post-block hidden states.
p79/mechanistic/activation_patching.py:58:            list of (batch, seq_len, hidden_dim) tensors, length = n_layers.
p79/mechanistic/activation_patching.py:61:        cached: list[Optional[torch.Tensor]] = [None] * self.n_layers
p79/mechanistic/activation_patching.py:63:        for i, layer in enumerate(self.layers):
p79/mechanistic/activation_patching.py:64:            def hook(module, layer_input, layer_output, idx=i):
p79/mechanistic/activation_patching.py:65:                hs = layer_output[0] if isinstance(layer_output, tuple) else layer_output
p79/mechanistic/activation_patching.py:66:                cached[idx] = hs.detach().clone()
p79/mechanistic/activation_patching.py:67:            hooks.append(layer.register_forward_hook(hook))
p79/mechanistic/activation_patching.py:69:            self.model(**inputs, use_cache=False, return_dict=True)
p79/mechanistic/activation_patching.py:73:        return cached
p79/mechanistic/activation_patching.py:76:    def patched_generate(
p79/mechanistic/activation_patching.py:78:        layer_idx: int,
p79/mechanistic/activation_patching.py:80:        max_new_tokens: int = 30,
p79/mechanistic/activation_patching.py:83:        """Patch last-token hidden state at layer_idx on FIRST forward, then greedy-generate.
p79/mechanistic/activation_patching.py:85:        With use_cache=True, the first forward processes full input (seq_len = N input
p79/mechanistic/activation_patching.py:86:        tokens). The hook only fires for this first forward — subsequent forwards
p79/mechanistic/activation_patching.py:87:        process 1-token-at-a-time and shouldn't be patched (they're new generated content,
p79/mechanistic/activation_patching.py:88:        not source's input). Patched first-token hidden state propagates through KV cache
p79/mechanistic/activation_patching.py:92:            Generated token IDs (1D tensor, only generated portion not input).
p79/mechanistic/activation_patching.py:94:        layer = self.layers[layer_idx]
p79/mechanistic/activation_patching.py:98:        def hook(module, layer_input, layer_output):
p79/mechanistic/activation_patching.py:102:            hs = layer_output[0] if isinstance(layer_output, tuple) else layer_output
p79/mechanistic/activation_patching.py:103:            hs_patched = hs.clone()
p79/mechanistic/activation_patching.py:104:            hs_patched[:, -1, :] = src[:, -1, :]
p79/mechanistic/activation_patching.py:105:            if isinstance(layer_output, tuple):
p79/mechanistic/activation_patching.py:106:                return (hs_patched,) + layer_output[1:]
p79/mechanistic/activation_patching.py:107:            return hs_patched
p79/mechanistic/activation_patching.py:109:        h = layer.register_forward_hook(hook)
p79/mechanistic/activation_patching.py:113:                max_new_tokens=max_new_tokens,
p79/mechanistic/activation_patching.py:116:                use_cache=True,
p79/mechanistic/activation_patching.py:127:        layer_idx: int,
p79/mechanistic/activation_patching.py:130:        max_new_tokens: int = 15,
p79/mechanistic/activation_patching.py:133:        """Add alpha * direction to last-token hidden at layer_idx on first forward, generate.
p79/mechanistic/activation_patching.py:135:        Variant of patched_generate: instead of substituting hidden with source's, we
p79/mechanistic/activation_patching.py:140:            layer_idx: which transformer block
p79/mechanistic/activation_patching.py:141:            direction: (hidden_dim,) vector — typically (mean_A - mean_B) at layer_idx
p79/mechanistic/activation_patching.py:143:            max_new_tokens: generation length
p79/mechanistic/activation_patching.py:147:            Generated token IDs (1D tensor, only generated portion)
p79/mechanistic/activation_patching.py:149:        layer = self.layers[layer_idx]
p79/mechanistic/activation_patching.py:154:        def hook(module, layer_input, layer_output):
p79/mechanistic/activation_patching.py:158:            hs = layer_output[0] if isinstance(layer_output, tuple) else layer_output
p79/mechanistic/activation_patching.py:161:            if isinstance(layer_output, tuple):
p79/mechanistic/activation_patching.py:162:                return (hs_steered,) + layer_output[1:]
p79/mechanistic/activation_patching.py:165:        h = layer.register_forward_hook(hook)
p79/mechanistic/activation_patching.py:169:                max_new_tokens=max_new_tokens,
p79/mechanistic/activation_patching.py:172:                use_cache=True,
p79/mechanistic/activation_patching.py:181:    def patched_forward(
p79/mechanistic/activation_patching.py:183:        layer_idx: int,
p79/mechanistic/activation_patching.py:185:        position: str = "last",
p79/mechanistic/activation_patching.py:188:        """Forward inputs with hook on layer_idx that swaps hidden state.
p79/mechanistic/activation_patching.py:191:            layer_idx: which transformer block to patch at (0 .. n_layers-1)
p79/mechanistic/activation_patching.py:192:            source_hidden: source-run cached hidden state at same layer
p79/mechanistic/activation_patching.py:194:            position: 'last' = patch only last-token position (works across
p79/mechanistic/activation_patching.py:200:            model output dict (.logits at last position used for downstream metric)
p79/mechanistic/activation_patching.py:202:        layer = self.layers[layer_idx]
p79/mechanistic/activation_patching.py:205:        def hook(module, layer_input, layer_output):
p79/mechanistic/activation_patching.py:206:            hs = layer_output[0] if isinstance(layer_output, tuple) else layer_output
p79/mechanistic/activation_patching.py:207:            hs_patched = hs.clone()
p79/mechanistic/activation_patching.py:208:            if position == "last":
p79/mechanistic/activation_patching.py:209:                hs_patched[:, -1, :] = src[:, -1, :]
p79/mechanistic/activation_patching.py:210:            elif position == "all":
p79/mechanistic/activation_patching.py:213:                        f"shape mismatch for position='all': target {hs.shape} vs source {src.shape}"
p79/mechanistic/activation_patching.py:215:                hs_patched = src
p79/mechanistic/activation_patching.py:217:                raise ValueError(f"unknown position={position!r}")
p79/mechanistic/activation_patching.py:218:            if isinstance(layer_output, tuple):
p79/mechanistic/activation_patching.py:219:                return (hs_patched,) + layer_output[1:]
p79/mechanistic/activation_patching.py:220:            return hs_patched
p79/mechanistic/activation_patching.py:222:        h = layer.register_forward_hook(hook)
p79/mechanistic/activation_patching.py:224:            output = self.model(**inputs, use_cache=False, return_dict=True)
p79/mechanistic/activation_patching.py:231:def patching_grid(
p79/mechanistic/activation_patching.py:232:    patcher: ActivationPatcher,
p79/mechanistic/activation_patching.py:235:    layers: Optional[list[int]] = None,
p79/mechanistic/activation_patching.py:237:    """Per-layer source-into-target patching, return causal-effect metrics.
p79/mechanistic/activation_patching.py:239:    Standard activation patching protocol (Meng et al. 2022 ROME-style):
p79/mechanistic/activation_patching.py:240:    1. Cache source's per-layer hidden states (source_cache[L])
p79/mechanistic/activation_patching.py:243:    4. For each L: forward target with hook injecting source_cache[L] at last
p79/mechanistic/activation_patching.py:244:       token → patched distribution. Compare to source/target baselines.
p79/mechanistic/activation_patching.py:247:        argmax_match_source: 1 if patched_argmax == source_argmax else 0
p79/mechanistic/activation_patching.py:248:        logit_shift_to_source: (patched_logit_src - target_logit_src) /
p79/mechanistic/activation_patching.py:249:                              (source_logit_src - target_logit_src). 1.0 = full
p79/mechanistic/activation_patching.py:251:        kl_patched_to_source: KL(patched || source). Lower = closer to source.
p79/mechanistic/activation_patching.py:252:        kl_patched_to_target: KL(patched || target). Higher = further from target.
p79/mechanistic/activation_patching.py:254:    Returns dict with above keys (each mapped to list[float] of length n_layers).
p79/mechanistic/activation_patching.py:256:    if layers is None:
p79/mechanistic/activation_patching.py:257:        layers = list(range(patcher.n_layers))
p79/mechanistic/activation_patching.py:260:    source_cache = patcher.cache_hidden_states(**source_inputs)
p79/mechanistic/activation_patching.py:263:    source_out = patcher.model(**source_inputs, use_cache=False, return_dict=True)
p79/mechanistic/activation_patching.py:264:    source_logits = source_out.logits[0, -1, :].float().cpu()  # (vocab,)
p79/mechanistic/activation_patching.py:265:    source_probs = torch.softmax(source_logits, dim=-1)
p79/mechanistic/activation_patching.py:268:    target_out = patcher.model(**target_inputs, use_cache=False, return_dict=True)
p79/mechanistic/activation_patching.py:269:    target_logits = target_out.logits[0, -1, :].float().cpu()
p79/mechanistic/activation_patching.py:270:    target_probs = torch.softmax(target_logits, dim=-1)
p79/mechanistic/activation_patching.py:275:    logit_shift_to_source = []
p79/mechanistic/activation_patching.py:276:    kl_patched_to_source = []
p79/mechanistic/activation_patching.py:277:    kl_patched_to_target = []
p79/mechanistic/activation_patching.py:279:    denom_src_logit = source_logits[source_argmax].item() - target_logits[source_argmax].item()
p79/mechanistic/activation_patching.py:281:    for L in layers:
p79/mechanistic/activation_patching.py:282:        patched_out = patcher.patched_forward(
p79/mechanistic/activation_patching.py:283:            layer_idx=L,
p79/mechanistic/activation_patching.py:284:            source_hidden=source_cache[L],
p79/mechanistic/activation_patching.py:285:            position="last",
p79/mechanistic/activation_patching.py:288:        patched_logits = patched_out.logits[0, -1, :].float().cpu()
p79/mechanistic/activation_patching.py:289:        patched_probs = torch.softmax(patched_logits, dim=-1)
p79/mechanistic/activation_patching.py:290:        patched_argmax = int(patched_probs.argmax())
p79/mechanistic/activation_patching.py:292:        argmax_match_source.append(1.0 if patched_argmax == source_argmax else 0.0)
p79/mechanistic/activation_patching.py:294:        if abs(denom_src_logit) > 1e-6:
p79/mechanistic/activation_patching.py:296:                (patched_logits[source_argmax].item() - target_logits[source_argmax].item())
p79/mechanistic/activation_patching.py:297:                / denom_src_logit
p79/mechanistic/activation_patching.py:301:        logit_shift_to_source.append(shift)
p79/mechanistic/activation_patching.py:303:        kl_ps = float(torch.sum(patched_probs * (torch.log(patched_probs + eps) - torch.log(source_probs + eps))))
p79/mechanistic/activation_patching.py:304:        kl_pt = float(torch.sum(patched_probs * (torch.log(patched_probs + eps) - torch.log(target_probs + eps))))
p79/mechanistic/activation_patching.py:305:        kl_patched_to_source.append(kl_ps)
p79/mechanistic/activation_patching.py:306:        kl_patched_to_target.append(kl_pt)
p79/mechanistic/activation_patching.py:310:                f"  patched L{L}: argmax_match={argmax_match_source[-1]}, "
p79/mechanistic/activation_patching.py:311:                f"logit_shift={shift:.3f}, KL→src={kl_ps:.3f}, KL→tgt={kl_pt:.3f}"
p79/mechanistic/activation_patching.py:315:        "layers": layers,
p79/mechanistic/activation_patching.py:317:        "logit_shift_to_source": logit_shift_to_source,
p79/mechanistic/activation_patching.py:318:        "kl_patched_to_source": kl_patched_to_source,
p79/mechanistic/activation_patching.py:319:        "kl_patched_to_target": kl_patched_to_target,
p79/mechanistic/activation_patching.py:320:        "source_argmax_token_id": source_argmax,
p79/mechanistic/activation_patching.py:321:        "target_argmax_token_id": target_argmax,
p79/mechanistic/activation_patching.py:322:        "source_logit_at_argmax": float(source_logits[source_argmax].item()),
p79/mechanistic/activation_patching.py:323:        "target_logit_at_argmax": float(target_logits[target_argmax].item()),
p79/mechanistic/activation_patching.py:327:def _token_seq_overlap(seq_a, seq_b) -> float:
p79/mechanistic/activation_patching.py:328:    """Ratio of positions where seq_a[i] == seq_b[i] (prefix-aligned). 1.0 = identical."""
p79/mechanistic/activation_patching.py:335:def _levenshtein_token(a, b) -> int:
p79/mechanistic/activation_patching.py:353:def patching_grid_continuation(
p79/mechanistic/activation_patching.py:354:    patcher: ActivationPatcher,
p79/mechanistic/activation_patching.py:357:    max_new_tokens: int = 15,
p79/mechanistic/activation_patching.py:358:    layers: Optional[list[int]] = None,
p79/mechanistic/activation_patching.py:359:    randomize_source_hidden: bool = False,
p79/mechanistic/activation_patching.py:361:    """Multi-token continuation patching.
p79/mechanistic/activation_patching.py:363:    Per-layer patch source's last-token hidden into target run, then greedy-generate
p79/mechanistic/activation_patching.py:364:    `max_new_tokens` tokens. Compare patched output sequence to source/target baselines.
p79/mechanistic/activation_patching.py:366:    This addresses the first-token-trivial-agree problem of patching_grid: by
p79/mechanistic/activation_patching.py:367:    generating 10+ tokens, divergence between source/target output sequences emerges
p79/mechanistic/activation_patching.py:368:    (e.g. action_type / element_id values vary). Layer L is causal if patching at L
p79/mechanistic/activation_patching.py:369:    pulls patched output toward source's full sequence.
p79/mechanistic/activation_patching.py:373:        - "source_tokens": list[int] (source's greedy sequence)
p79/mechanistic/activation_patching.py:374:        - "target_tokens": list[int]
p79/mechanistic/activation_patching.py:377:        - "per_layer": list of {layer, patched_tokens, patched_text,
p79/mechanistic/activation_patching.py:378:                                token_overlap_to_source, token_overlap_to_target,
p79/mechanistic/activation_patching.py:381:    if layers is None:
p79/mechanistic/activation_patching.py:382:        layers = list(range(patcher.n_layers))
p79/mechanistic/activation_patching.py:385:    source_gen = patcher.model.generate(
p79/mechanistic/activation_patching.py:386:        **source_inputs, max_new_tokens=max_new_tokens, do_sample=False,
p79/mechanistic/activation_patching.py:387:        return_dict_in_generate=True, use_cache=True,
p79/mechanistic/activation_patching.py:390:    source_tokens = source_gen.sequences[0, src_input_len:].cpu().tolist()
p79/mechanistic/activation_patching.py:391:    source_text = patcher.processor.tokenizer.decode(source_tokens, skip_special_tokens=True)
p79/mechanistic/activation_patching.py:394:    target_gen = patcher.model.generate(
p79/mechanistic/activation_patching.py:395:        **target_inputs, max_new_tokens=max_new_tokens, do_sample=False,
p79/mechanistic/activation_patching.py:396:        return_dict_in_generate=True, use_cache=True,
p79/mechanistic/activation_patching.py:399:    target_tokens = target_gen.sequences[0, tgt_input_len:].cpu().tolist()
p79/mechanistic/activation_patching.py:400:    target_text = patcher.processor.tokenizer.decode(target_tokens, skip_special_tokens=True)
p79/mechanistic/activation_patching.py:405:    # 3. Cache source's per-layer hidden states (full forward)
p79/mechanistic/activation_patching.py:406:    source_cache = patcher.cache_hidden_states(**source_inputs)
p79/mechanistic/activation_patching.py:408:    # Random-injection control (paper §5 reviewer Q "is L17 disruption from
p79/mechanistic/activation_patching.py:409:    # specific source content or any non-zero injection?"): replace each
p79/mechanistic/activation_patching.py:410:    # layer's cached source hidden with Gaussian noise matched to that
p79/mechanistic/activation_patching.py:411:    # layer's mean+std. Preserves activation magnitude while destroying
p79/mechanistic/activation_patching.py:412:    # task-specific structure. If L17 disruption persists with random
p79/mechanistic/activation_patching.py:413:    # injection → mechanism is non-specific (any patch disrupts). If it
p79/mechanistic/activation_patching.py:415:    if randomize_source_hidden:
p79/mechanistic/activation_patching.py:416:        import torch as _torch_for_random
p79/mechanistic/activation_patching.py:417:        randomized = []
p79/mechanistic/activation_patching.py:418:        for L_idx, h in enumerate(source_cache):
p79/mechanistic/activation_patching.py:421:            noise = _torch_for_random.randn_like(h) * std + mean
p79/mechanistic/activation_patching.py:422:            randomized.append(noise)
p79/mechanistic/activation_patching.py:423:        source_cache = randomized
p79/mechanistic/activation_patching.py:425:            "  RANDOMIZED source hidden: replaced cached activations with "
p79/mechanistic/activation_patching.py:426:            "Gaussian noise matched to per-layer mean/std"
p79/mechanistic/activation_patching.py:429:    # 4. Per-layer patched generate
p79/mechanistic/activation_patching.py:430:    per_layer = []
p79/mechanistic/activation_patching.py:431:    for L in layers:
p79/mechanistic/activation_patching.py:432:        patched_token_tensor = patcher.patched_generate(
p79/mechanistic/activation_patching.py:433:            layer_idx=L,
p79/mechanistic/activation_patching.py:434:            source_hidden=source_cache[L],
p79/mechanistic/activation_patching.py:435:            max_new_tokens=max_new_tokens,
p79/mechanistic/activation_patching.py:438:        patched_tokens = patched_token_tensor.cpu().tolist()
p79/mechanistic/activation_patching.py:439:        patched_text = patcher.processor.tokenizer.decode(patched_tokens, skip_special_tokens=True)
p79/mechanistic/activation_patching.py:441:        per_layer.append({
p79/mechanistic/activation_patching.py:442:            "layer": L,
p79/mechanistic/activation_patching.py:443:            "patched_tokens": patched_tokens,
p79/mechanistic/activation_patching.py:444:            "patched_text": patched_text,
p79/mechanistic/activation_patching.py:445:            "token_overlap_to_source": _token_seq_overlap(patched_tokens, source_tokens),
p79/mechanistic/activation_patching.py:446:            "token_overlap_to_target": _token_seq_overlap(patched_tokens, target_tokens),
p79/mechanistic/activation_patching.py:447:            "ld_to_source": _levenshtein_token(patched_tokens, source_tokens),
p79/mechanistic/activation_patching.py:448:            "ld_to_target": _levenshtein_token(patched_tokens, target_tokens),
p79/mechanistic/activation_patching.py:449:            "exact_match_source": patched_tokens == source_tokens,
p79/mechanistic/activation_patching.py:450:            "exact_match_target": patched_tokens == target_tokens,
p79/mechanistic/activation_patching.py:454:            r = per_layer[-1]
p79/mechanistic/activation_patching.py:456:                f"  L{L}: overlap→src={r['token_overlap_to_source']:.2f}, "
p79/mechanistic/activation_patching.py:457:                f"overlap→tgt={r['token_overlap_to_target']:.2f}, "
p79/mechanistic/activation_patching.py:462:        "source_tokens": source_tokens,
p79/mechanistic/activation_patching.py:463:        "target_tokens": target_tokens,
p79/mechanistic/activation_patching.py:466:        "per_layer": per_layer,
p79/mechanistic/activation_patching.py:467:        "max_new_tokens": max_new_tokens,

exec
/bin/bash -lc "nl -ba scripts/mechanistic/run_stage2b_continuation_pilot.py | sed -n '1,260p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	"""Stage 2B — Multi-token continuation activation patching for B1 mirage analysis.
     2	
     3	Addresses Stage 2A trivial first-token-agree problem (source argmax == target argmax
     4	== JSON `{` opener forced by chat template). Generate 10-15 tokens past `{` so action_type
     5	+ element_id divergence between source/target emerges, then measure how patching at each
     6	layer pulls patched continuation toward source.
     7	
     8	Setup (same as Stage 2A):
     9	- Source = SoM (with screenshot_annotated.png, [SOM_MARKS], SoM prompt)
    10	- Target = P-SoM (no image, [SOM_MARKS], SoM prompt) — mirage condition
    11	- Per (task) × per (layer L = 0..35): patch source's L-th hidden state into target,
    12	  greedy-generate max_new_tokens, compare full token sequences.
    13	
    14	Metrics:
    15	- token_overlap_to_source: ratio of positions where patched matches source (1=identical)
    16	- token_overlap_to_target: same vs target baseline
    17	- ld_to_source: Levenshtein edit distance to source token sequence
    18	- ld_to_target: Levenshtein edit distance to target token sequence
    19	
    20	Output:
    21	    results/mechanistic/stage2b_continuation_b1_cls_pilot/
    22	      patching_continuation_results.json
    23	      patching_continuation_curves.png
    24	      pilot_summary.md
    25	
    26	Usage:
    27	    python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    28	      --site classifieds --n-tasks 3 --step 2 --max-new-tokens 15 \
    29	      --archived-run-dir results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428
    30	
    31	ETA: 3 task × (1 source-gen + 1 target-gen + 1 source-cache + 36 patched-gen)
    32	   = 3 × 39 generation passes × ~15 forward each = ~1750 forwards × 1.5s
    33	   ≈ 45 min compute + 2 min model load = ~50 min total.
    34	"""
    35	
    36	from __future__ import annotations
    37	
    38	import argparse
    39	import json
    40	import logging
    41	import os
    42	import sys
    43	from pathlib import Path
    44	
    45	# B-81h workaround (笔记 §117): force SDPA math backend so the script runs on
    46	# any GPU architecture. PyTorch's flash + memory-efficient SDPA backends only
    47	# have bf16 cutlass kernels for sm_80+ (A100/H100). On V100 (sm_70) or T4
    48	# (sm_75) Myriad nodes, SDPA dispatcher raises "cutlassF: no kernel found to
    49	# launch!" instead of falling back. The math backend always works (slower
    50	# ~2-3x but correct on any GPU). Disabled via FORCE_MATH_SDP=0 to opt back in.
    51	if os.environ.get("FORCE_MATH_SDP", "1") != "0":
    52	    try:
    53	        import torch as _torch_for_sdp_setup
    54	        _torch_for_sdp_setup.backends.cuda.enable_flash_sdp(False)
    55	        _torch_for_sdp_setup.backends.cuda.enable_mem_efficient_sdp(False)
    56	        _torch_for_sdp_setup.backends.cuda.enable_math_sdp(True)
    57	    except Exception:
    58	        pass
    59	
    60	import numpy as np
    61	
    62	sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    63	
    64	from p79.mechanistic.activation_patching import ActivationPatcher, patching_grid_continuation
    65	from p79.mechanistic.extract_hidden_states import HiddenStateExtractor
    66	
    67	logging.basicConfig(
    68	    level=logging.INFO,
    69	    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    70	    datefmt="%H:%M:%S",
    71	)
    72	logger = logging.getLogger("stage2b-continuation")
    73	
    74	REPO_ROOT = Path(__file__).resolve().parents[2]
    75	SITE_TO_CONFIG_DIR = {
    76	    "classifieds": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_classifieds",
    77	    "reddit": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_reddit",
    78	    "shopping": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_shopping",
    79	}
    80	
    81	
    82	def load_intents(site: str, n_tasks: int) -> list[tuple[int, str]]:
    83	    """Load intents from VWA config_files (full repo with submodule init)."""
    84	    config_dir = SITE_TO_CONFIG_DIR[site]
    85	    json_files = sorted(config_dir.glob("*.json"), key=lambda p: int(p.stem))
    86	    intents = []
    87	    for jf in json_files[:n_tasks]:
    88	        d = json.loads(jf.read_text(encoding="utf-8"))
    89	        if d.get("intent"):
    90	            intents.append((int(jf.stem), d["intent"]))
    91	    return intents
    92	
    93	
    94	def load_intents_from_subset_manifest(manifest_path: Path, tier: str, n_tasks: int) -> list[tuple[int, str]]:
    95	    """Load intents from archive_subset manifest.json (cross-machine paper-grade
    96	    dataset). Used on Myriad / A100 where VWA submodule isn't init'd."""
    97	    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    98	    entries = manifest.get(tier, [])[:n_tasks]
    99	    return [(int(e["task_id"]), e["intent"]) for e in entries]
   100	
   101	
   102	def find_artifacts_dir(run_dir: Path) -> Path:
   103	    """Find artifacts directory; supports two layouts:
   104	    (a) nested:  <run>/<condition>/artifacts/<site>_task_X/step_NNN/
   105	    (b) flat:    <subset>/<site>_task_X/step_NNN/  (extract_archive_subset.py output)
   106	    """
   107	    # Layout (a): nested condition/artifacts
   108	    for child in run_dir.iterdir():
   109	        if child.is_dir() and (child / "artifacts").is_dir():
   110	            return child / "artifacts"
   111	    # Layout (b): flat subset (run_dir IS the artifacts dir)
   112	    for child in run_dir.iterdir():
   113	        if child.is_dir() and any(
   114	            child.name.startswith(prefix)
   115	            for prefix in ("classifieds_task_", "reddit_task_", "shopping_task_")
   116	        ):
   117	            return run_dir
   118	    raise FileNotFoundError(f"No artifacts in {run_dir} (tried nested + flat layouts)")
   119	
   120	
   121	def build_som_marks(obs_text: str, max_marks: int = 200) -> str:
   122	    from p79.experiment.som import _extract_text_marks
   123	    marks = _extract_text_marks(obs_text, max_marks=max_marks)
   124	    if not marks:
   125	        return "[SOM_MARKS]\n[/SOM_MARKS]"
   126	    return "\n".join(["[SOM_MARKS]"] + [f"[id={m['id']}] {m['label']}" for m in marks] + ["[/SOM_MARKS]"])
   127	
   128	
   129	def build_inputs(extractor: HiddenStateExtractor, intent: str, mode: str, obs_text: str, image_path):
   130	    user_text = extractor._build_user_text(intent, mode, obs_text)
   131	    content = []
   132	    if image_path is not None:
   133	        img = HiddenStateExtractor._load_resize_image(image_path)
   134	        content.append({"type": "image", "image": img})
   135	    content.append({"type": "text", "text": user_text})
   136	    messages = [{"role": "user", "content": content}]
   137	    text = extractor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   138	    if image_path is not None:
   139	        from qwen_vl_utils import process_vision_info
   140	        image_inputs, video_inputs = process_vision_info(messages)
   141	        inputs = extractor.processor(
   142	            text=[text], images=image_inputs, videos=video_inputs,
   143	            padding=True, return_tensors="pt",
   144	        )
   145	    else:
   146	        inputs = extractor.processor(text=[text], padding=True, return_tensors="pt")
   147	    return {k: v.to(extractor.model.device) for k, v in inputs.items()}
   148	
   149	
   150	def main():
   151	    p = argparse.ArgumentParser()
   152	    p.add_argument("--site", default="classifieds", choices=list(SITE_TO_CONFIG_DIR))
   153	    p.add_argument("--n-tasks", type=int, default=3, help="Default 3 for fast pilot, scale to 5+ for paper")
   154	    p.add_argument("--step", type=int, default=2)
   155	    p.add_argument("--max-new-tokens", type=int, default=15, help="Continuation length (15 covers JSON envelope start)")
   156	    p.add_argument("--archived-run-dir", required=True)
   157	    p.add_argument("--output-dir", default=None)
   158	    p.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
   159	    p.add_argument("--source-mode", default="som")
   160	    p.add_argument("--target-mode", default="phantom_som")
   161	    p.add_argument("--min-free-vram-gb", type=float, default=0.0)
   162	    p.add_argument(
   163	        "--reverse", action="store_true",
   164	        help="Swap source ↔ target: patch target's hidden state into source run "
   165	             "(asymmetry control test). Output dir gets _reverse suffix.",
   166	    )
   167	    p.add_argument(
   168	        "--tier", default=None, choices=["strong", "reverse"],
   169	        help="Override manifest tier subset (strong=24 forward-mirage / reverse=15 reverse-mirage). "
   170	             "If unset, auto-derives from --reverse flag (default behavior). "
   171	             "Set explicitly to enable 2x2 cross-subset control: e.g. --reverse --tier strong "
   172	             "tests reverse direction on forward-easy tasks (selection-bias control).",
   173	    )
   174	    p.add_argument(
   175	        "--random-inject", action="store_true",
   176	        help="Random-injection control (paper §5 reviewer Q): replace cached source "
   177	             "hidden state with Gaussian noise matched to per-layer mean/std. Tests "
   178	             "whether mid-layer disruption depends on source-content specificity vs "
   179	             "any non-zero injection. Expected null at all layers if mechanism is "
   180	             "source-content-specific.",
   181	    )
   182	    p.add_argument(
   183	        "--random-seed", type=int, default=42,
   184	        help="Seed for --random-inject Gaussian noise (paper-grade reproducibility). "
   185	             "Same seed + same input = same noise = byte-identical re-runs. Default 42.",
   186	    )
   187	    args = p.parse_args()
   188	
   189	    # C8 fix: seed all RNGs when random-inject is on, for paper-grade
   190	    # reproducibility. Affects torch.randn_like in patching_grid_continuation.
   191	    # Default seed=42 means re-running with same data + code produces
   192	    # byte-identical noise + byte-identical patched outputs.
   193	    if args.random_inject:
   194	        import random as _rnd
   195	        import numpy as _np
   196	        import torch as _t
   197	        _rnd.seed(args.random_seed)
   198	        _np.random.seed(args.random_seed)
   199	        _t.manual_seed(args.random_seed)
   200	        if _t.cuda.is_available():
   201	            _t.cuda.manual_seed_all(args.random_seed)
   202	        # Defense-in-depth visibility (防忘): log prominently + this seed
   203	        # value also flows to env_snapshot.json + pilot_summary.md (below).
   204	        print(f"\n{'=' * 60}\n[RANDOM-INJECT SEED] {args.random_seed} "
   205	              f"(reproducibility — see pilot_summary.md + env_snapshot.json)\n"
   206	              f"{'=' * 60}\n", flush=True)
   207	        # Note: cell E (job 335404) is currently running with NO seed (commit
   208	        # before this fix). That run is one valid realization; future paper-
   209	        # grade re-runs will be byte-reproducible with --random-seed 42.
   210	
   211	    suffix = "_reverse" if args.reverse else ""
   212	    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / f"results/mechanistic/stage2b_continuation_b1_{args.site}_pilot{suffix}"
   213	    out_dir.mkdir(parents=True, exist_ok=True)
   214	    logger.info(f"Output dir: {out_dir}")
   215	
   216	    # Paper-grade provenance: dump env snapshot at run start (Gap 1+3, 笔记 §114)
   217	    try:
   218	        from scripts.provenance.snapshot_env import capture_env_snapshot
   219	        capture_env_snapshot(
   220	            out_dir / "env_snapshot.json",
   221	            extra={
   222	                "stage": "stage2b_curated",
   223	                "site": args.site,
   224	                "reverse": args.reverse,
   225	                "tier": args.tier or ("reverse" if args.reverse else "strong"),
   226	                "random_inject": args.random_inject,
   227	                "random_seed": args.random_seed,
   228	                "n_tasks_requested": args.n_tasks,
   229	                "step": args.step,
   230	                "max_new_tokens": args.max_new_tokens,
   231	                "source_mode": args.source_mode,
   232	                "target_mode": args.target_mode,
   233	            },
   234	        )
   235	    except Exception as e:
   236	        logger.warning(f"Env snapshot failed (non-fatal): {e}")
   237	
   238	    archived_dir = Path(args.archived_run_dir)
   239	
   240	    # Auto-detect: if archived_run_dir contains manifest.json, it's a subset
   241	    # (extract_archive_subset.py output). Use intents from manifest, support
   242	    # flat layout. This enables cross-machine paper-grade workflow (Myriad / A100)
   243	    # without needing the full B1_phantom_som_classifieds_20260428 archive (~1.8GB).
   244	    subset_manifest = archived_dir / "manifest.json"
   245	    if subset_manifest.exists():
   246	        tier = args.tier if args.tier else ("reverse" if args.reverse else "strong")
   247	        intents = load_intents_from_subset_manifest(subset_manifest, tier=tier, n_tasks=args.n_tasks)
   248	        logger.info(
   249	            f"Subset mode: loaded {len(intents)} intents from manifest "
   250	            f"(tier={tier}, reverse_dir={args.reverse}, "
   251	            f"cross_subset={'YES' if args.tier and args.tier != ('reverse' if args.reverse else 'strong') else 'NO'})"
   252	        )
   253	    else:
   254	        intents = load_intents(args.site, args.n_tasks)
   255	        logger.info(f"Full archive mode: loaded {len(intents)} intents from VWA config_files")
   256	
   257	    artifacts_dir = find_artifacts_dir(archived_dir)
   258	    logger.info(f"Archived artifacts: {artifacts_dir}")
   259	
   260	    extractor = HiddenStateExtractor(model_path=args.model_path, min_free_vram_gb=args.min_free_vram_gb)

exec
/bin/bash -lc "nl -ba scripts/mechanistic/run_stage2b_continuation_pilot.py | sed -n '261,560p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   261	    patcher = ActivationPatcher(extractor.model, extractor.processor)
   262	    logger.info(f"Model loaded; n_layers={patcher.n_layers}")
   263	
   264	    per_task_results = []
   265	    for task_id, intent in intents:
   266	        step_dir = artifacts_dir / f"{args.site}_task_{task_id}" / f"step_{args.step:03d}"
   267	        obs_file = step_dir / "observation_dom.txt"
   268	        screenshot_annotated = step_dir / "screenshot_annotated.png"
   269	        if not obs_file.exists() or not screenshot_annotated.exists():
   270	            logger.warning(f"task {task_id}: missing artifacts, skip")
   271	            continue
   272	        obs_text = obs_file.read_text(encoding="utf-8")
   273	        som_marks_text = build_som_marks(obs_text)
   274	
   275	        # 2026-05-10 bug fix: text payload depends on observation mode.
   276	        # phantom_prompt = SoM prompt + AXTree (no marks); previously hardcoded som_marks_text
   277	        # for all modes, which made phantom_prompt byte-identical to phantom_som (Cell F vs
   278	        # H-prompt-red 24/24 patched_text identical, confirming bug).
   279	        text_payload_for = lambda mode: (
   280	            som_marks_text if mode in ("som", "phantom_som", "phantom_text")
   281	            else obs_text  if mode in ("phantom_prompt", "dom", "phantom_dom")
   282	            else ""        if mode == "vision"
   283	            else som_marks_text
   284	        )
   285	        source_text = text_payload_for(args.source_mode)
   286	        target_text = text_payload_for(args.target_mode)
   287	
   288	        source_inputs_orig = build_inputs(extractor, intent, args.source_mode, source_text, str(screenshot_annotated))
   289	        target_inputs_orig = build_inputs(extractor, intent, args.target_mode, target_text, None)
   290	
   291	        # --reverse: swap roles. patch target's hidden into source run = "remove image content"
   292	        if args.reverse:
   293	            source_inputs, target_inputs = target_inputs_orig, source_inputs_orig
   294	            logger.info(f"task {task_id}: REVERSE direction (patching {args.target_mode} → {args.source_mode})")
   295	        else:
   296	            source_inputs, target_inputs = source_inputs_orig, target_inputs_orig
   297	            logger.info(f"task {task_id}: forward direction (patching {args.source_mode} → {args.target_mode})")
   298	
   299	        logger.info(f"task {task_id}: running continuation patching grid (max_new_tokens={args.max_new_tokens})...")
   300	        result = patching_grid_continuation(
   301	            patcher, source_inputs, target_inputs,
   302	            max_new_tokens=args.max_new_tokens,
   303	            randomize_source_hidden=args.random_inject,
   304	        )
   305	        result["task_id"] = task_id
   306	        result["step_idx"] = args.step
   307	        result["intent"] = intent
   308	        per_task_results.append(result)
   309	
   310	        # F18 audit fix 2026-05-09: include reverse / tier / random_inject /
   311	        # random_seed in incremental JSON so downstream stage2 stat scripts
   312	        # can reconstruct which causal/control cell produced these numbers.
   313	        # (Previously only env_snapshot.json carried these; analysis scripts
   314	        # consume the results JSON directly.)
   315	        # F19 audit fix 2026-05-09: source_mode/target_mode reported here
   316	        # are the role labels AFTER the reverse swap (i.e. what was actually
   317	        # patched into what), matching the per-task INFO log line above.
   318	        if args.reverse:
   319	            logged_source_mode = args.target_mode
   320	            logged_target_mode = args.source_mode
   321	        else:
   322	            logged_source_mode = args.source_mode
   323	            logged_target_mode = args.target_mode
   324	
   325	        with (out_dir / "patching_continuation_results.json").open("w") as f:
   326	            json.dump({
   327	                "config": {
   328	                    "site": args.site, "n_tasks": args.n_tasks, "step": args.step,
   329	                    "max_new_tokens": args.max_new_tokens,
   330	                    "source_mode": logged_source_mode,
   331	                    "target_mode": logged_target_mode,
   332	                    "source_mode_raw": args.source_mode,
   333	                    "target_mode_raw": args.target_mode,
   334	                    "reverse": args.reverse,
   335	                    "tier": args.tier or ("reverse" if args.reverse else "strong"),
   336	                    "random_inject": args.random_inject,
   337	                    "random_seed": args.random_seed,
   338	                    "archived_run_dir": str(archived_dir),
   339	                    "model_path": args.model_path,
   340	                    "n_layers": patcher.n_layers,
   341	                },
   342	                "per_task": per_task_results,
   343	            }, f, indent=2)
   344	
   345	    if not per_task_results:
   346	        logger.error("No tasks completed; aborting plot")
   347	        return
   348	
   349	    # Aggregate per-layer mean ± std across tasks
   350	    n_layers = patcher.n_layers
   351	    metric_names = ["token_overlap_to_source", "token_overlap_to_target", "ld_to_source", "ld_to_target"]
   352	    agg = {}
   353	    for m in metric_names:
   354	        arr = np.array([
   355	            [layer_r[m] for layer_r in t["per_layer"]]
   356	            for t in per_task_results
   357	        ])  # (n_tasks, n_layers)
   358	        agg[f"{m}_mean"] = arr.mean(axis=0).tolist()
   359	        agg[f"{m}_std"] = arr.std(axis=0).tolist()
   360	
   361	    # Plot
   362	    import matplotlib
   363	    matplotlib.use("Agg")
   364	    import matplotlib.pyplot as plt
   365	
   366	    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
   367	    layers_x = np.arange(n_layers)
   368	    titles = {
   369	        "token_overlap_to_source": "Token overlap → source\n(1=patched matches source position-by-position)",
   370	        "token_overlap_to_target": "Token overlap → target\n(higher = patch had no effect)",
   371	        "ld_to_source": f"Levenshtein dist → source\n(0=identical, max~{args.max_new_tokens})",
   372	        "ld_to_target": f"Levenshtein dist → target\n(higher = patch pulled away from target)",
   373	    }
   374	    for ax, m in zip(axes.flat, metric_names):
   375	        mean = np.array(agg[f"{m}_mean"])
   376	        std = np.array(agg[f"{m}_std"])
   377	        ax.plot(layers_x, mean, marker="o", lw=1.5, label=f"mean (N={len(per_task_results)})")
   378	        ax.fill_between(layers_x, mean - std, mean + std, alpha=0.25, label="±1 std")
   379	        ax.set_xlabel("Layer index (0=embedding, ≥1=post-block)")
   380	        ax.set_title(titles[m], fontsize=10)
   381	        ax.grid(alpha=0.3)
   382	        ax.legend(fontsize=8, loc="best")
   383	    if args.reverse:
   384	        direction_label = f"{args.target_mode}→{args.source_mode} (reverse)"
   385	    else:
   386	        direction_label = f"{args.source_mode}→{args.target_mode} (forward)"
   387	    tier_label = args.tier if args.tier else ("reverse" if args.reverse else "strong")
   388	    fig.suptitle(
   389	        f"Stage 2B Continuation Activation Patching — {direction_label} "
   390	        f"({args.site} N={len(per_task_results)} {tier_label}-tier task × step_{args.step:03d}, "
   391	        f"max_new_tokens={args.max_new_tokens})",
   392	        fontsize=12,
   393	    )
   394	    fig.tight_layout()
   395	    fig.savefig(out_dir / "patching_continuation_curves.png", dpi=150)
   396	    plt.close(fig)
   397	    logger.info("Saved patching_continuation_curves.png")
   398	
   399	    # Summary
   400	    overlap_src = np.array(agg["token_overlap_to_source_mean"])
   401	    overlap_tgt = np.array(agg["token_overlap_to_target_mean"])
   402	    ld_src = np.array(agg["ld_to_source_mean"])
   403	    ld_tgt = np.array(agg["ld_to_target_mean"])
   404	
   405	    best_overlap_layer = int(overlap_src.argmax())
   406	    best_ld_layer = int(ld_src.argmin())
   407	
   408	    # Sample some patched outputs for qualitative check
   409	    qualitative = []
   410	    for t in per_task_results[:1]:  # first task only for brevity
   411	        qualitative.append(f"\n### Task {t['task_id']} (intent: {t['intent'][:80]})")
   412	        qualitative.append(f"  source: {t['source_text']!r}")
   413	        qualitative.append(f"  target: {t['target_text']!r}")
   414	        for L in [0, 5, 11, 17, 23, 29, 35]:
   415	            r = t["per_layer"][L]
   416	            qualitative.append(f"  L{L:2d} patched: {r['patched_text']!r}  (overlap→src={r['token_overlap_to_source']:.2f}, LD→src={r['ld_to_source']})")
   417	    qual_block = "\n".join(qualitative)
   418	
   419	    summary = f"""# Stage 2B Continuation Activation Patching — Summary
   420	
   421	## Setup
   422	- Model: {args.model_path}
   423	- Site: {args.site}, N task: {len(per_task_results)} × step_{args.step:03d}
   424	- Source: `{args.source_mode}` (with image — clean) / Target: `{args.target_mode}` (no image — mirage)
   425	- Direction: {"reverse (target→source)" if args.reverse else "forward (source→target)"}
   426	- Tier: {args.tier or ("reverse" if args.reverse else "strong")}
   427	- max_new_tokens: {args.max_new_tokens} (greedy continuation, deterministic)
   428	- Random injection: {"YES, seed=" + str(args.random_seed) + " (paper-grade reproducible)" if args.random_inject else "NO (real source hidden injected)"}
   429	- Archived: {args.archived_run_dir}
   430	
   431	## Result (per-layer mean across tasks)
   432	- Best layer for **token overlap → source**: L{best_overlap_layer} (overlap {overlap_src[best_overlap_layer]:.3f})
   433	- Best layer for **min Levenshtein → source**: L{best_ld_layer} (LD {ld_src[best_ld_layer]:.2f})
   434	
   435	## Layer-resolved curves (source-side metrics):
   436	| Layer | overlap→src | overlap→tgt | LD→src | LD→tgt |
   437	|---|---|---|---|---|
   438	| L0  | {overlap_src[0]:.2f} | {overlap_tgt[0]:.2f} | {ld_src[0]:.1f} | {ld_tgt[0]:.1f} |
   439	| L5  | {overlap_src[5]:.2f} | {overlap_tgt[5]:.2f} | {ld_src[5]:.1f} | {ld_tgt[5]:.1f} |
   440	| L11 | {overlap_src[11]:.2f} | {overlap_tgt[11]:.2f} | {ld_src[11]:.1f} | {ld_tgt[11]:.1f} |
   441	| L17 | {overlap_src[17]:.2f} | {overlap_tgt[17]:.2f} | {ld_src[17]:.1f} | {ld_tgt[17]:.1f} |
   442	| L23 | {overlap_src[23]:.2f} | {overlap_tgt[23]:.2f} | {ld_src[23]:.1f} | {ld_tgt[23]:.1f} |
   443	| L29 | {overlap_src[29]:.2f} | {overlap_tgt[29]:.2f} | {ld_src[29]:.1f} | {ld_tgt[29]:.1f} |
   444	| L35 | {overlap_src[35]:.2f} | {overlap_tgt[35]:.2f} | {ld_src[35]:.1f} | {ld_tgt[35]:.1f} |
   445	
   446	## Interpretation
   447	- overlap→src curve climbs monotonically with depth → mirage info accumulates layer-by-layer (deep layer wins)
   448	- overlap→src peaks at middle layer then decays → "computed feature" emerges mid then abstracts
   449	- overlap→src flat ~0 → patching has no causal effect (mirage info distributed elsewhere)
   450	- LD→src minimum identifies "most source-like patched output" layer — opposite signal of overlap→src
   451	
   452	## Qualitative samples (first task)
   453	{qual_block}
   454	
   455	## Next steps
   456	- If mid-layer peak emerges (e.g. L17-L25) → consistent with Stage 2A logit_shift L17 finding ✓
   457	- If late-layer monotone climb → mirage signature is residual-stream cumulative, no single causal layer
   458	- Scale up: 5 task × max_new_tokens=20 (~75 min) for tighter mean ± std
   459	- Then Stage 2C: reverse direction (target→source patching) for asymmetry check
   460	"""
   461	    (out_dir / "pilot_summary.md").write_text(summary)
   462	
   463	    # Paper-grade run manifest (Gap 3, 笔记 §114) — single-file roll-up of
   464	    # patch config + per-task outcomes for OSF DOI lock + cross-machine compare.
   465	    run_manifest = {
   466	        "stage": "stage2b_continuation_curated" if args.n_tasks > 5 else "stage2b_continuation_pilot",
   467	        "direction": "reverse" if args.reverse else "forward",
   468	        "site": args.site,
   469	        "patch_config": {
   470	            "source_mode": args.source_mode,
   471	            "target_mode": args.target_mode,
   472	            "step_idx": args.step,
   473	            "max_new_tokens": args.max_new_tokens,
   474	            "n_layers_swept": int(patcher.n_layers),
   475	            "hook_position": "last_token",
   476	            "first_forward_only": True,
   477	            "min_free_vram_gb": args.min_free_vram_gb,
   478	        },
   479	        "model": {
   480	            "path": args.model_path,
   481	            "n_layers": int(patcher.n_layers),
   482	        },
   483	        "input_dataset": {
   484	            "archived_run_dir": str(archived_dir),
   485	            "n_tasks_requested": args.n_tasks,
   486	            "n_tasks_completed": len(per_task_results),
   487	            "task_ids": [int(t["task_id"]) for t in per_task_results],
   488	        },
   489	        "outcomes_per_task": [
   490	            {
   491	                "task_id": int(t["task_id"]),
   492	                "step_idx": int(t["step_idx"]),
   493	                "best_layer_overlap_src": int(np.argmax([r["token_overlap_to_source"] for r in t["per_layer"]])),
   494	                "best_overlap_src": float(max(r["token_overlap_to_source"] for r in t["per_layer"])),
   495	                "L11_overlap_src": float(t["per_layer"][11]["token_overlap_to_source"]) if patcher.n_layers > 11 else None,
   496	                "L17_overlap_src": float(t["per_layer"][17]["token_overlap_to_source"]) if patcher.n_layers > 17 else None,
   497	            }
   498	            for t in per_task_results
   499	        ],
   500	        "aggregate": {
   501	            "best_layer_overlap_src_mean": int(best_overlap_layer),
   502	            "best_overlap_src_mean": float(overlap_src[best_overlap_layer]),
   503	            "best_layer_ld_src_mean": int(best_ld_layer),
   504	            "L11_overlap_src_mean": float(overlap_src[11]) if patcher.n_layers > 11 else None,
   505	            "L17_overlap_src_mean": float(overlap_src[17]) if patcher.n_layers > 17 else None,
   506	        },
   507	        "env_snapshot_ref": "env_snapshot.json",
   508	        "results_files": {
   509	            "per_task_jsonl": "patching_continuation_results.json",
   510	            "curves_plot": "patching_continuation_curves.png",
   511	            "summary_md": "pilot_summary.md",
   512	        },
   513	    }
   514	    with (out_dir / "run_manifest.json").open("w") as f:
   515	        json.dump(run_manifest, f, indent=2)
   516	    logger.info(f"run_manifest.json emitted ({len(per_task_results)} tasks)")
   517	    logger.info(f"Stage 2B continuation patching pilot DONE → {out_dir}")
   518	
   519	
   520	if __name__ == "__main__":
   521	    main()

exec
/bin/bash -lc "nl -ba p79/mechanistic/activation_patching.py | sed -n '1,240p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	"""Activation patching via PyTorch forward hooks (Stage 2 — causal mechanism analysis).
     2	
     3	Advisor 5/5 instruction: "patch 到哪一层的时候, 它的结果就切换了" — find the layer
     4	where source-into-target hidden-state injection causes output to flip. That's
     5	the mirage critical layer.
     6	
     7	Hand-rolled (not nnsight) — nnsight wheel build failed on aarch64/GB10. PyTorch
     8	register_forward_hook gives equivalent control with no extra dependency.
     9	
    10	Convention:
    11	- "source" run = ground-truth condition (e.g. SoM with image — no mirage)
    12	- "target" run = perturbed condition (e.g. P-SoM no image — mirage induced)
    13	- Patch source's layer-L last-token hidden state into target run at same layer.
    14	- If patched output flips toward source's behavior → layer L carries source info.
    15	"""
    16	
    17	from __future__ import annotations
    18	
    19	import logging
    20	from typing import Optional
    21	
    22	import torch
    23	
    24	logger = logging.getLogger(__name__)
    25	
    26	
    27	def get_transformer_layers(model) -> torch.nn.ModuleList:
    28	    """Locate transformer decoder layer ModuleList in Qwen3-VL.
    29	
    30	    Tested on Qwen3VLForConditionalGeneration (verified 2026-05-06):
    31	        model.model.language_model.layers — 36 × Qwen3VLTextDecoderLayer
    32	    """
    33	    return model.model.language_model.layers
    34	
    35	
    36	class ActivationPatcher:
    37	    """Cache + patch interface for transformer layer outputs.
    38	
    39	    F16 audit clarification 2026-05-09: layers L0..L_{n-1} are transformer
    40	    BLOCK OUTPUTS (post-attention + ffn + residual + post-norm). L0 is the
    41	    output of the first decoder layer, NOT the embedding output. To get
    42	    embedding output, hook `model.model.language_model.embed_tokens`
    43	    separately. Paper §5 prose should call these "block indices" or "B0..B35"
    44	    to avoid confusion with the embedding.
    45	    """
    46	
    47	    def __init__(self, model, processor):
    48	        self.model = model
    49	        self.processor = processor
    50	        self.layers = get_transformer_layers(model)
    51	        self.n_layers = len(self.layers)
    52	
    53	    @torch.no_grad()
    54	    def cache_hidden_states(self, **inputs) -> list[torch.Tensor]:
    55	        """Forward inputs and return per-layer post-block hidden states.
    56	
    57	        Returns:
    58	            list of (batch, seq_len, hidden_dim) tensors, length = n_layers.
    59	            Tensors are detached + cloned (CPU or device, matches model device).
    60	        """
    61	        cached: list[Optional[torch.Tensor]] = [None] * self.n_layers
    62	        hooks = []
    63	        for i, layer in enumerate(self.layers):
    64	            def hook(module, layer_input, layer_output, idx=i):
    65	                hs = layer_output[0] if isinstance(layer_output, tuple) else layer_output
    66	                cached[idx] = hs.detach().clone()
    67	            hooks.append(layer.register_forward_hook(hook))
    68	        try:
    69	            self.model(**inputs, use_cache=False, return_dict=True)
    70	        finally:
    71	            for h in hooks:
    72	                h.remove()
    73	        return cached
    74	
    75	    @torch.no_grad()
    76	    def patched_generate(
    77	        self,
    78	        layer_idx: int,
    79	        source_hidden: torch.Tensor,
    80	        max_new_tokens: int = 30,
    81	        **inputs,
    82	    ) -> torch.Tensor:
    83	        """Patch last-token hidden state at layer_idx on FIRST forward, then greedy-generate.
    84	
    85	        With use_cache=True, the first forward processes full input (seq_len = N input
    86	        tokens). The hook only fires for this first forward — subsequent forwards
    87	        process 1-token-at-a-time and shouldn't be patched (they're new generated content,
    88	        not source's input). Patched first-token hidden state propagates through KV cache
    89	        so subsequent generations attend to it.
    90	
    91	        Returns:
    92	            Generated token IDs (1D tensor, only generated portion not input).
    93	        """
    94	        layer = self.layers[layer_idx]
    95	        src = source_hidden.to(self.model.device)
    96	        fire_count = [0]
    97	
    98	        def hook(module, layer_input, layer_output):
    99	            fire_count[0] += 1
   100	            if fire_count[0] > 1:
   101	                return None  # subsequent forwards: pass through unchanged
   102	            hs = layer_output[0] if isinstance(layer_output, tuple) else layer_output
   103	            hs_patched = hs.clone()
   104	            hs_patched[:, -1, :] = src[:, -1, :]
   105	            if isinstance(layer_output, tuple):
   106	                return (hs_patched,) + layer_output[1:]
   107	            return hs_patched
   108	
   109	        h = layer.register_forward_hook(hook)
   110	        try:
   111	            out = self.model.generate(
   112	                **inputs,
   113	                max_new_tokens=max_new_tokens,
   114	                do_sample=False,
   115	                return_dict_in_generate=True,
   116	                use_cache=True,
   117	            )
   118	        finally:
   119	            h.remove()
   120	
   121	        input_len = inputs["input_ids"].shape[1]
   122	        return out.sequences[0, input_len:]
   123	
   124	    @torch.no_grad()
   125	    def steered_generate(
   126	        self,
   127	        layer_idx: int,
   128	        direction: torch.Tensor,
   129	        alpha: float,
   130	        max_new_tokens: int = 15,
   131	        **inputs,
   132	    ) -> torch.Tensor:
   133	        """Add alpha * direction to last-token hidden at layer_idx on first forward, generate.
   134	
   135	        Variant of patched_generate: instead of substituting hidden with source's, we
   136	        ADD a direction vector with magnitude alpha. Used for Method 4.4 counterfactual
   137	        activation steering (Tool Calling Linear Steerable Circuit, Anonymous 2026 ACL).
   138	
   139	        Args:
   140	            layer_idx: which transformer block
   141	            direction: (hidden_dim,) vector — typically (mean_A - mean_B) at layer_idx
   142	            alpha: scalar magnitude. 0 = no steering. 1 = unit direction. Larger = stronger push.
   143	            max_new_tokens: generation length
   144	            **inputs: target run inputs
   145	
   146	        Returns:
   147	            Generated token IDs (1D tensor, only generated portion)
   148	        """
   149	        layer = self.layers[layer_idx]
   150	        dir_vec = direction.to(self.model.device).to(self.model.dtype if hasattr(self.model, "dtype") else torch.bfloat16)
   151	        scaled = (alpha * dir_vec)
   152	        fire_count = [0]
   153	
   154	        def hook(module, layer_input, layer_output):
   155	            fire_count[0] += 1
   156	            if fire_count[0] > 1:
   157	                return None
   158	            hs = layer_output[0] if isinstance(layer_output, tuple) else layer_output
   159	            hs_steered = hs.clone()
   160	            hs_steered[:, -1, :] = hs_steered[:, -1, :] + scaled.to(hs.dtype)
   161	            if isinstance(layer_output, tuple):
   162	                return (hs_steered,) + layer_output[1:]
   163	            return hs_steered
   164	
   165	        h = layer.register_forward_hook(hook)
   166	        try:
   167	            out = self.model.generate(
   168	                **inputs,
   169	                max_new_tokens=max_new_tokens,
   170	                do_sample=False,
   171	                return_dict_in_generate=True,
   172	                use_cache=True,
   173	            )
   174	        finally:
   175	            h.remove()
   176	
   177	        input_len = inputs["input_ids"].shape[1]
   178	        return out.sequences[0, input_len:]
   179	
   180	    @torch.no_grad()
   181	    def patched_forward(
   182	        self,
   183	        layer_idx: int,
   184	        source_hidden: torch.Tensor,
   185	        position: str = "last",
   186	        **inputs,
   187	    ):
   188	        """Forward inputs with hook on layer_idx that swaps hidden state.
   189	
   190	        Args:
   191	            layer_idx: which transformer block to patch at (0 .. n_layers-1)
   192	            source_hidden: source-run cached hidden state at same layer
   193	                (batch, source_seq_len, hidden_dim)
   194	            position: 'last' = patch only last-token position (works across
   195	                different seq_len between source/target); 'all' = swap entire
   196	                sequence (requires matching seq_len)
   197	            **inputs: target run inputs (model kwargs)
   198	
   199	        Returns:
   200	            model output dict (.logits at last position used for downstream metric)
   201	        """
   202	        layer = self.layers[layer_idx]
   203	        src = source_hidden.to(self.model.device)
   204	
   205	        def hook(module, layer_input, layer_output):
   206	            hs = layer_output[0] if isinstance(layer_output, tuple) else layer_output
   207	            hs_patched = hs.clone()
   208	            if position == "last":
   209	                hs_patched[:, -1, :] = src[:, -1, :]
   210	            elif position == "all":
   211	                if hs.shape != src.shape:
   212	                    raise ValueError(
   213	                        f"shape mismatch for position='all': target {hs.shape} vs source {src.shape}"
   214	                    )
   215	                hs_patched = src
   216	            else:
   217	                raise ValueError(f"unknown position={position!r}")
   218	            if isinstance(layer_output, tuple):
   219	                return (hs_patched,) + layer_output[1:]
   220	            return hs_patched
   221	
   222	        h = layer.register_forward_hook(hook)
   223	        try:
   224	            output = self.model(**inputs, use_cache=False, return_dict=True)
   225	        finally:
   226	            h.remove()
   227	        return output
   228	
   229	
   230	@torch.no_grad()
   231	def patching_grid(
   232	    patcher: ActivationPatcher,
   233	    source_inputs: dict,
   234	    target_inputs: dict,
   235	    layers: Optional[list[int]] = None,
   236	) -> dict:
   237	    """Per-layer source-into-target patching, return causal-effect metrics.
   238	
   239	    Standard activation patching protocol (Meng et al. 2022 ROME-style):
   240	    1. Cache source's per-layer hidden states (source_cache[L])

exec
/bin/bash -lc "nl -ba p79/mechanistic/activation_patching.py | sed -n '241,520p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   241	    2. Run target unperturbed → baseline target distribution
   242	    3. Run source unperturbed → baseline source distribution
   243	    4. For each L: forward target with hook injecting source_cache[L] at last
   244	       token → patched distribution. Compare to source/target baselines.
   245	
   246	    Metrics:
   247	        argmax_match_source: 1 if patched_argmax == source_argmax else 0
   248	        logit_shift_to_source: (patched_logit_src - target_logit_src) /
   249	                              (source_logit_src - target_logit_src). 1.0 = full
   250	                              shift to source, 0.0 = no shift.
   251	        kl_patched_to_source: KL(patched || source). Lower = closer to source.
   252	        kl_patched_to_target: KL(patched || target). Higher = further from target.
   253	
   254	    Returns dict with above keys (each mapped to list[float] of length n_layers).
   255	    """
   256	    if layers is None:
   257	        layers = list(range(patcher.n_layers))
   258	
   259	    # 1. Cache source hidden states
   260	    source_cache = patcher.cache_hidden_states(**source_inputs)
   261	
   262	    # 2. Baseline source + target output dist
   263	    source_out = patcher.model(**source_inputs, use_cache=False, return_dict=True)
   264	    source_logits = source_out.logits[0, -1, :].float().cpu()  # (vocab,)
   265	    source_probs = torch.softmax(source_logits, dim=-1)
   266	    source_argmax = int(source_probs.argmax())
   267	
   268	    target_out = patcher.model(**target_inputs, use_cache=False, return_dict=True)
   269	    target_logits = target_out.logits[0, -1, :].float().cpu()
   270	    target_probs = torch.softmax(target_logits, dim=-1)
   271	    target_argmax = int(target_probs.argmax())
   272	
   273	    eps = 1e-12
   274	    argmax_match_source = []
   275	    logit_shift_to_source = []
   276	    kl_patched_to_source = []
   277	    kl_patched_to_target = []
   278	
   279	    denom_src_logit = source_logits[source_argmax].item() - target_logits[source_argmax].item()
   280	
   281	    for L in layers:
   282	        patched_out = patcher.patched_forward(
   283	            layer_idx=L,
   284	            source_hidden=source_cache[L],
   285	            position="last",
   286	            **target_inputs,
   287	        )
   288	        patched_logits = patched_out.logits[0, -1, :].float().cpu()
   289	        patched_probs = torch.softmax(patched_logits, dim=-1)
   290	        patched_argmax = int(patched_probs.argmax())
   291	
   292	        argmax_match_source.append(1.0 if patched_argmax == source_argmax else 0.0)
   293	
   294	        if abs(denom_src_logit) > 1e-6:
   295	            shift = (
   296	                (patched_logits[source_argmax].item() - target_logits[source_argmax].item())
   297	                / denom_src_logit
   298	            )
   299	        else:
   300	            shift = 0.0
   301	        logit_shift_to_source.append(shift)
   302	
   303	        kl_ps = float(torch.sum(patched_probs * (torch.log(patched_probs + eps) - torch.log(source_probs + eps))))
   304	        kl_pt = float(torch.sum(patched_probs * (torch.log(patched_probs + eps) - torch.log(target_probs + eps))))
   305	        kl_patched_to_source.append(kl_ps)
   306	        kl_patched_to_target.append(kl_pt)
   307	
   308	        if (L + 1) % 6 == 0:
   309	            logger.info(
   310	                f"  patched L{L}: argmax_match={argmax_match_source[-1]}, "
   311	                f"logit_shift={shift:.3f}, KL→src={kl_ps:.3f}, KL→tgt={kl_pt:.3f}"
   312	            )
   313	
   314	    return {
   315	        "layers": layers,
   316	        "argmax_match_source": argmax_match_source,
   317	        "logit_shift_to_source": logit_shift_to_source,
   318	        "kl_patched_to_source": kl_patched_to_source,
   319	        "kl_patched_to_target": kl_patched_to_target,
   320	        "source_argmax_token_id": source_argmax,
   321	        "target_argmax_token_id": target_argmax,
   322	        "source_logit_at_argmax": float(source_logits[source_argmax].item()),
   323	        "target_logit_at_argmax": float(target_logits[target_argmax].item()),
   324	    }
   325	
   326	
   327	def _token_seq_overlap(seq_a, seq_b) -> float:
   328	    """Ratio of positions where seq_a[i] == seq_b[i] (prefix-aligned). 1.0 = identical."""
   329	    n = min(len(seq_a), len(seq_b))
   330	    if n == 0:
   331	        return 0.0
   332	    return sum(int(seq_a[i] == seq_b[i]) for i in range(n)) / n
   333	
   334	
   335	def _levenshtein_token(a, b) -> int:
   336	    """Token-level edit distance between two integer sequences (DP, no extra dep)."""
   337	    if not a:
   338	        return len(b)
   339	    if not b:
   340	        return len(a)
   341	    m, n = len(a), len(b)
   342	    prev = list(range(n + 1))
   343	    for i in range(1, m + 1):
   344	        curr = [i] + [0] * n
   345	        for j in range(1, n + 1):
   346	            cost = 0 if a[i - 1] == b[j - 1] else 1
   347	            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
   348	        prev = curr
   349	    return prev[n]
   350	
   351	
   352	@torch.no_grad()
   353	def patching_grid_continuation(
   354	    patcher: ActivationPatcher,
   355	    source_inputs: dict,
   356	    target_inputs: dict,
   357	    max_new_tokens: int = 15,
   358	    layers: Optional[list[int]] = None,
   359	    randomize_source_hidden: bool = False,
   360	) -> dict:
   361	    """Multi-token continuation patching.
   362	
   363	    Per-layer patch source's last-token hidden into target run, then greedy-generate
   364	    `max_new_tokens` tokens. Compare patched output sequence to source/target baselines.
   365	
   366	    This addresses the first-token-trivial-agree problem of patching_grid: by
   367	    generating 10+ tokens, divergence between source/target output sequences emerges
   368	    (e.g. action_type / element_id values vary). Layer L is causal if patching at L
   369	    pulls patched output toward source's full sequence.
   370	
   371	    Returns:
   372	        dict with:
   373	        - "source_tokens": list[int] (source's greedy sequence)
   374	        - "target_tokens": list[int]
   375	        - "source_text": decoded
   376	        - "target_text": decoded
   377	        - "per_layer": list of {layer, patched_tokens, patched_text,
   378	                                token_overlap_to_source, token_overlap_to_target,
   379	                                ld_to_source, ld_to_target, exact_match_source}
   380	    """
   381	    if layers is None:
   382	        layers = list(range(patcher.n_layers))
   383	
   384	    # 1. Source baseline generation
   385	    source_gen = patcher.model.generate(
   386	        **source_inputs, max_new_tokens=max_new_tokens, do_sample=False,
   387	        return_dict_in_generate=True, use_cache=True,
   388	    )
   389	    src_input_len = source_inputs["input_ids"].shape[1]
   390	    source_tokens = source_gen.sequences[0, src_input_len:].cpu().tolist()
   391	    source_text = patcher.processor.tokenizer.decode(source_tokens, skip_special_tokens=True)
   392	
   393	    # 2. Target baseline generation
   394	    target_gen = patcher.model.generate(
   395	        **target_inputs, max_new_tokens=max_new_tokens, do_sample=False,
   396	        return_dict_in_generate=True, use_cache=True,
   397	    )
   398	    tgt_input_len = target_inputs["input_ids"].shape[1]
   399	    target_tokens = target_gen.sequences[0, tgt_input_len:].cpu().tolist()
   400	    target_text = patcher.processor.tokenizer.decode(target_tokens, skip_special_tokens=True)
   401	
   402	    logger.info(f"  source generated: {source_text!r}")
   403	    logger.info(f"  target generated: {target_text!r}")
   404	
   405	    # 3. Cache source's per-layer hidden states (full forward)
   406	    source_cache = patcher.cache_hidden_states(**source_inputs)
   407	
   408	    # Random-injection control (paper §5 reviewer Q "is L17 disruption from
   409	    # specific source content or any non-zero injection?"): replace each
   410	    # layer's cached source hidden with Gaussian noise matched to that
   411	    # layer's mean+std. Preserves activation magnitude while destroying
   412	    # task-specific structure. If L17 disruption persists with random
   413	    # injection → mechanism is non-specific (any patch disrupts). If it
   414	    # vanishes → source-content-specific causal claim valid.
   415	    if randomize_source_hidden:
   416	        import torch as _torch_for_random
   417	        randomized = []
   418	        for L_idx, h in enumerate(source_cache):
   419	            mean = h.mean()
   420	            std = h.std()
   421	            noise = _torch_for_random.randn_like(h) * std + mean
   422	            randomized.append(noise)
   423	        source_cache = randomized
   424	        logger.info(
   425	            "  RANDOMIZED source hidden: replaced cached activations with "
   426	            "Gaussian noise matched to per-layer mean/std"
   427	        )
   428	
   429	    # 4. Per-layer patched generate
   430	    per_layer = []
   431	    for L in layers:
   432	        patched_token_tensor = patcher.patched_generate(
   433	            layer_idx=L,
   434	            source_hidden=source_cache[L],
   435	            max_new_tokens=max_new_tokens,
   436	            **target_inputs,
   437	        )
   438	        patched_tokens = patched_token_tensor.cpu().tolist()
   439	        patched_text = patcher.processor.tokenizer.decode(patched_tokens, skip_special_tokens=True)
   440	
   441	        per_layer.append({
   442	            "layer": L,
   443	            "patched_tokens": patched_tokens,
   444	            "patched_text": patched_text,
   445	            "token_overlap_to_source": _token_seq_overlap(patched_tokens, source_tokens),
   446	            "token_overlap_to_target": _token_seq_overlap(patched_tokens, target_tokens),
   447	            "ld_to_source": _levenshtein_token(patched_tokens, source_tokens),
   448	            "ld_to_target": _levenshtein_token(patched_tokens, target_tokens),
   449	            "exact_match_source": patched_tokens == source_tokens,
   450	            "exact_match_target": patched_tokens == target_tokens,
   451	        })
   452	
   453	        if (L + 1) % 6 == 0:
   454	            r = per_layer[-1]
   455	            logger.info(
   456	                f"  L{L}: overlap→src={r['token_overlap_to_source']:.2f}, "
   457	                f"overlap→tgt={r['token_overlap_to_target']:.2f}, "
   458	                f"LD→src={r['ld_to_source']}, LD→tgt={r['ld_to_target']}"
   459	            )
   460	
   461	    return {
   462	        "source_tokens": source_tokens,
   463	        "target_tokens": target_tokens,
   464	        "source_text": source_text,
   465	        "target_text": target_text,
   466	        "per_layer": per_layer,
   467	        "max_new_tokens": max_new_tokens,
   468	    }

codex
The patching path already exposes one hard issue: the hook layer convention is block-output L0..L35 in `activation_patching.py`, while one plot label still says `0=embedding`; I’ll verify whether the paper prose repeats the older L0..L36 convention and whether other analyses use extractor hidden-state indexing instead.
exec
/bin/bash -lc "nl -ba p79/mechanistic/extract_hidden_states.py | sed -n '1,260p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	"""Forward-pass-only hidden state extractor for Qwen3-VL-4B (B1) contrastive set.
     2	
     3	Reuses Qwen3VLAgent prompt construction (system_prompt inlined into user content
     4	per agent line 436) so hidden states reflect identical prompt structure as the
     5	agent saw during paper-grade runs.
     6	"""
     7	
     8	from __future__ import annotations
     9	
    10	import logging
    11	from pathlib import Path
    12	from typing import Optional, Union
    13	
    14	import torch
    15	from PIL import Image
    16	from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    17	
    18	from p79.utils.torch_cuda_workarounds import apply_nvrtc_prod_fallback_if_needed
    19	
    20	logger = logging.getLogger(__name__)
    21	
    22	# Match qwen3vl_agent.py default image_max_size for production parity
    23	IMAGE_MAX_SIZE_DEFAULT = 1024
    24	
    25	
    26	class HiddenStateExtractor:
    27	    """Extract last-input-token hidden states from Qwen3-VL-4B forward pass.
    28	
    29	    Stage 1 pilot scope: empty observation, system prompt + task intent only.
    30	    Later stages: load actual archived observations (DOM AXTree / SoM marks).
    31	    """
    32	
    33	    # System prompts — copied from p79.agents.qwen3vl_agent (must keep in sync; if
    34	    # the agent prompts drift, hidden states will not reflect production conditions).
    35	    # Source: qwen3vl_agent.py::_make_dom_prompt / _make_som_prompt as of 2026-05-05.
    36	
    37	    def __init__(
    38	        self,
    39	        model_path: str = "Qwen/Qwen3-VL-4B-Instruct",
    40	        # Paper-grade: pin HF revision SHA — DGX baseline lock 2026-05-07 (笔记 §114)
    41	        model_revision: str = "ebb281ec70b05090aa6165b016eac8ec08e71b17",
    42	        device: str = "cuda",
    43	        dtype: torch.dtype = torch.bfloat16,
    44	        min_free_vram_gb: float = 12.0,
    45	    ):
    46	        apply_nvrtc_prod_fallback_if_needed()
    47	
    48	        if min_free_vram_gb > 0 and torch.cuda.is_available():
    49	            free_gb = torch.cuda.mem_get_info(0)[0] / (1024**3)
    50	            if free_gb < min_free_vram_gb:
    51	                raise RuntimeError(
    52	                    f"Insufficient VRAM: {free_gb:.1f} GB free < {min_free_vram_gb:.1f} GB required. "
    53	                    f"Wait for other GPU jobs to finish or set min_free_vram_gb=0 to skip check."
    54	                )
    55	
    56	        logger.info(f"Loading {model_path} (revision={model_revision[:12]}...) for hidden state extraction (dtype={dtype})")
    57	        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
    58	            model_path,
    59	            revision=model_revision,
    60	            torch_dtype=dtype,
    61	            device_map=device,
    62	            trust_remote_code=True,
    63	        )
    64	        self.model.eval()
    65	        self.processor = AutoProcessor.from_pretrained(
    66	            model_path, revision=model_revision, trust_remote_code=True
    67	        )
    68	        self.device = device
    69	        self.model_revision = model_revision
    70	
    71	        # Load system prompts from the agent — single source of truth.
    72	        from p79.agents.qwen3vl_agent import Qwen3VLAgent
    73	        self._dom_prompt = Qwen3VLAgent._make_dom_prompt(self)
    74	        self._som_prompt = Qwen3VLAgent._make_som_prompt(self)
    75	        self._mode_to_prompt = {
    76	            "dom": self._dom_prompt,
    77	            "som": self._som_prompt,
    78	            "phantom_som": self._som_prompt,
    79	            "phantom_text": self._dom_prompt,
    80	            "phantom_dom": self._dom_prompt,
    81	            "phantom_prompt": self._som_prompt,
    82	            "vision": Qwen3VLAgent._make_vision_prompt(self),
    83	        }
    84	
    85	    def _build_user_text(
    86	        self,
    87	        intent: str,
    88	        mode: str,
    89	        observation_text: str = "",
    90	    ) -> str:
    91	        """Replicate agent's user content text format (qwen3vl_agent.py:436).
    92	
    93	        Format: f"Task: {instruction}\\nSystem: {system_prompt}\\n[observation if any]"
    94	        """
    95	        system_prompt = self._mode_to_prompt.get(mode, self._dom_prompt)
    96	        text = f"Task: {intent}\nSystem: {system_prompt}\n"
    97	        if observation_text:
    98	            text += observation_text
    99	        return text
   100	
   101	    @staticmethod
   102	    def _load_resize_image(image_path: Union[str, Path], max_size: int = IMAGE_MAX_SIZE_DEFAULT) -> Image.Image:
   103	        """Load + LANCZOS-resize image to max_size (matches qwen3vl_agent.py:447-450)."""
   104	        img = Image.open(image_path).convert("RGB")
   105	        if max(img.size) > max_size:
   106	            ratio = max_size / max(img.size)
   107	            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
   108	            img = img.resize(new_size, Image.Resampling.LANCZOS)
   109	        return img
   110	
   111	    @torch.no_grad()
   112	    def extract(
   113	        self,
   114	        intent: str,
   115	        mode: str,
   116	        observation_text: str = "",
   117	        image_path: Optional[Union[str, Path]] = None,
   118	    ) -> torch.Tensor:
   119	        """Forward pass with output_hidden_states=True. Return last-token hidden states.
   120	
   121	        Args:
   122	            intent: task instruction
   123	            mode: observation mode (dom / som / phantom_som / phantom_text / phantom_prompt / vision)
   124	            observation_text: full AXTree or [SOM_MARKS] text (mode-conditional)
   125	            image_path: if provided, load image and add to messages content
   126	                (multimodal forward pass; for SoM / Vision modes)
   127	
   128	        Returns:
   129	            Tensor of shape (n_layers + 1, hidden_dim). Layer 0 is embedding output;
   130	            layer L for L >= 1 is post-transformer-block-L hidden state.
   131	        """
   132	        user_text = self._build_user_text(intent, mode, observation_text)
   133	
   134	        # Build content. For multimodal: image first, then text (matches agent line 471).
   135	        content = []
   136	        if image_path is not None:
   137	            img = self._load_resize_image(image_path)
   138	            content.append({"type": "image", "image": img})
   139	        content.append({"type": "text", "text": user_text})
   140	        messages = [{"role": "user", "content": content}]
   141	
   142	        text = self.processor.apply_chat_template(
   143	            messages, tokenize=False, add_generation_prompt=True
   144	        )
   145	
   146	        if image_path is not None:
   147	            from qwen_vl_utils import process_vision_info
   148	            image_inputs, video_inputs = process_vision_info(messages)
   149	            inputs = self.processor(
   150	                text=[text],
   151	                images=image_inputs,
   152	                videos=video_inputs,
   153	                padding=True,
   154	                return_tensors="pt",
   155	            )
   156	        else:
   157	            inputs = self.processor(text=[text], padding=True, return_tensors="pt")
   158	
   159	        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
   160	
   161	        outputs = self.model(
   162	            **inputs,
   163	            output_hidden_states=True,
   164	            use_cache=False,
   165	            return_dict=True,
   166	        )
   167	
   168	        # outputs.hidden_states is tuple of (n_layers + 1) tensors of shape
   169	        # (batch=1, seq_len, hidden_dim). Extract last token of each.
   170	        hidden_states = torch.stack(
   171	            [h[0, -1, :].detach().float().cpu() for h in outputs.hidden_states],
   172	            dim=0,
   173	        )  # (n_layers + 1, hidden_dim)
   174	        return hidden_states
   175	
   176	    def extract_batch(
   177	        self,
   178	        items: list[tuple[str, str, Optional[str], Optional[Union[str, Path]]]],
   179	    ) -> tuple[torch.Tensor, list[str]]:
   180	        """Sequential extraction over (intent, mode, observation_text, image_path) tuples.
   181	
   182	        Args:
   183	            items: list of (intent, mode, observation_text or None, image_path or None)
   184	
   185	        Returns:
   186	            (hidden_states, mode_labels)
   187	            - hidden_states: Tensor (n_items, n_layers + 1, hidden_dim)
   188	            - mode_labels: list of mode strings (for label encoding downstream)
   189	        """
   190	        hs_list = []
   191	        labels = []
   192	        for i, item in enumerate(items):
   193	            # Backward-compat: support 3-tuple (without image_path)
   194	            if len(item) == 3:
   195	                intent, mode, obs = item
   196	                image_path = None
   197	            else:
   198	                intent, mode, obs, image_path = item
   199	            hs = self.extract(intent, mode, obs or "", image_path=image_path)
   200	            hs_list.append(hs)
   201	            labels.append(mode)
   202	            if (i + 1) % 20 == 0:
   203	                logger.info(f"Extracted {i + 1}/{len(items)} hidden states")
   204	        return torch.stack(hs_list, dim=0), labels

exec
/bin/bash -lc "nl -ba scripts/analysis/stage4_pca_cosine_gap.py | sed -n '1,260p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Stage 4 Method 4.2: PCA cosine gap analysis of phantom routing space.
     3	
     4	Ports Tool Calling Linear Steerable Circuit method (Anonymous 2026 ACL, validated
     5	on Qwen3-4B) to Qwen3-VL-4B (B1). Tests whether phantom routing space modes are
     6	mechanistically distinct in hidden state geometry layer-by-layer.
     7	
     8	Three analyses per (mode pair, layer):
     9	  A. Cosine gap between mean hidden states
    10	  B. AUROC: project hidden states onto (mean_A - mean_B) direction, predict mode
    11	  C. Per-(mode, layer) PCA top-10 variance explained
    12	
    13	Outputs:
    14	  - results/mechanistic/stage4_multimode_b1_cls/method42_metrics.json
    15	  - docs/checkpoints/stage4_method42_results.md
    16	  - results/phantom_paper/figures/fig_stage4_pca_cosine_gap.png
    17	"""
    18	from __future__ import annotations
    19	
    20	import json
    21	from itertools import combinations
    22	from pathlib import Path
    23	
    24	import matplotlib.pyplot as plt
    25	import numpy as np
    26	from sklearn.decomposition import PCA
    27	from sklearn.metrics import roc_auc_score
    28	
    29	ROOT = Path(__file__).resolve().parents[2]
    30	DEFAULT_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz"
    31	DEFAULT_OUT_JSON = ROOT / "results/mechanistic/stage4_multimode_b1_cls/method42_metrics.json"
    32	DEFAULT_OUT_MD = ROOT / "docs/checkpoints/stage4_method42_results.md"
    33	DEFAULT_OUT_FIG = ROOT / "results/phantom_paper/figures/fig_stage4_pca_cosine_gap.png"
    34	
    35	MODES = ["dom", "phantom_text", "phantom_prompt", "phantom_som", "som", "vision"]
    36	DISPLAY = {"dom": "DOM", "phantom_text": "P-text", "phantom_prompt": "P-prompt",
    37	           "phantom_som": "P-SoM", "som": "SoM", "vision": "Vision"}
    38	
    39	
    40	def cosine_gap(v1: np.ndarray, v2: np.ndarray) -> float:
    41	    return float(1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    42	
    43	
    44	def pair_key(a: str, b: str) -> str:
    45	    """Canonical pair key using MODES index order (matches itertools.combinations output)."""
    46	    i, j = MODES.index(a), MODES.index(b)
    47	    return f"{MODES[min(i, j)]}_vs_{MODES[max(i, j)]}"
    48	
    49	
    50	def main() -> None:
    51	    import argparse
    52	    parser = argparse.ArgumentParser()
    53	    parser.add_argument("--input", type=Path, default=DEFAULT_NPZ)
    54	    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUT_JSON)
    55	    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUT_MD)
    56	    parser.add_argument("--output-fig", type=Path, default=DEFAULT_OUT_FIG)
    57	    args = parser.parse_args()
    58	    NPZ = args.input
    59	    OUT_JSON = args.output_json
    60	    OUT_MD = args.output_md
    61	    OUT_FIG = args.output_fig
    62	
    63	    d = np.load(NPZ, allow_pickle=True)
    64	    H = d["hidden_states"]
    65	    mode_labels = d["mode_labels_str"]
    66	    n_layers = H.shape[1]
    67	    print(f"[stage4] loaded {H.shape[0]} examples × {n_layers} layers × {H.shape[2]} dim")
    68	
    69	    states = {m: H[mode_labels == m] for m in MODES}
    70	    means = {m: states[m].mean(axis=0) for m in MODES}  # each (37, 2560)
    71	    print(f"[stage4] per-mode counts: " + ", ".join(f"{m}={len(states[m])}" for m in MODES))
    72	
    73	    pairs = list(combinations(MODES, 2))
    74	    cos_gap = np.zeros((len(pairs), n_layers))
    75	    auroc = np.zeros((len(pairs), n_layers))
    76	    for pi, (m1, m2) in enumerate(pairs):
    77	        for L in range(n_layers):
    78	            c1, c2 = means[m1][L], means[m2][L]
    79	            cos_gap[pi, L] = cosine_gap(c1, c2)
    80	            direction = (c1 - c2) / (np.linalg.norm(c1 - c2) + 1e-9)
    81	            s1 = states[m1][:, L, :] @ direction
    82	            s2 = states[m2][:, L, :] @ direction
    83	            y = np.concatenate([np.ones(len(s1)), np.zeros(len(s2))])
    84	            scores = np.concatenate([s1, s2])
    85	            try:
    86	                auroc[pi, L] = roc_auc_score(y, scores)
    87	            except Exception:
    88	                auroc[pi, L] = 0.5
    89	
    90	    pca_var = np.zeros((len(MODES), n_layers))
    91	    for mi, mode in enumerate(MODES):
    92	        X = states[mode]  # (n, 37, 2560)
    93	        for L in range(n_layers):
    94	            if X.shape[0] >= 11:
    95	                n_comp = min(10, X.shape[0] - 1)
    96	                pca_var[mi, L] = PCA(n_components=n_comp).fit(X[:, L, :]).explained_variance_ratio_.sum()
    97	
    98	    peak = {}
    99	    for pi, (m1, m2) in enumerate(pairs):
   100	        L = int(np.argmax(cos_gap[pi]))
   101	        peak[f"{m1}_vs_{m2}"] = {"layer": L, "gap": float(cos_gap[pi, L]),
   102	                                  "auroc_at_peak": float(auroc[pi, L])}
   103	
   104	    metrics = {
   105	        "n_examples": int(H.shape[0]), "n_layers": int(n_layers), "n_modes": len(MODES),
   106	        "modes": MODES, "n_per_mode": {m: int(len(states[m])) for m in MODES},
   107	        "pairwise_cosine_gap": {f"{m1}_vs_{m2}": cos_gap[pi].tolist()
   108	                                  for pi, (m1, m2) in enumerate(pairs)},
   109	        "pairwise_auroc": {f"{m1}_vs_{m2}": auroc[pi].tolist()
   110	                             for pi, (m1, m2) in enumerate(pairs)},
   111	        "pca_top10_var_ratio": {m: pca_var[mi].tolist() for mi, m in enumerate(MODES)},
   112	        "peak_disruption_layers": peak,
   113	    }
   114	    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
   115	    OUT_JSON.write_text(json.dumps(metrics, indent=2))
   116	    print(f"[stage4] metrics → {OUT_JSON}")
   117	
   118	    write_summary(metrics, OUT_MD)
   119	    plot(cos_gap, auroc, pairs, pca_var, OUT_FIG)
   120	
   121	
   122	def write_summary(m: dict, out: Path) -> None:
   123	    sorted_pairs = sorted(m["peak_disruption_layers"].items(),
   124	                           key=lambda x: -x[1]["gap"])
   125	    lines = [
   126	        "# Stage 4 Method 4.2: PCA Cosine Gap Analysis",
   127	        "",
   128	        f"**Data**: {m['n_examples']} examples × {m['n_layers']} layers × {m['n_modes']} modes (Qwen3-VL-4B B1 cls)",
   129	        f"**Per-mode n**: " + ", ".join(f"{DISPLAY[k]}={v}" for k, v in m['n_per_mode'].items()),
   130	        "",
   131	        "## Peak disruption layer per mode pair",
   132	        "",
   133	        "Sorted by cosine gap magnitude (= geometric distance between mode means in hidden space):",
   134	        "",
   135	        "| Mode pair | Peak layer | Cosine gap | AUROC at peak |",
   136	        "|---|---|---|---|",
   137	    ]
   138	    for k, v in sorted_pairs:
   139	        m1, m2 = k.split("_vs_")
   140	        lines.append(f"| {DISPLAY[m1]} vs {DISPLAY[m2]} | L{v['layer']:02d} | {v['gap']:.4f} | {v['auroc_at_peak']:.3f} |")
   141	
   142	    # Mid-layer (L17) snapshot — paper §5 disruption locus
   143	    L17_section = ["", "## L17 cosine gap snapshot (paper §5 disruption locus)", ""]
   144	    L17_section.append("| Mode pair | L17 cosine gap | L17 AUROC |")
   145	    L17_section.append("|---|---|---|")
   146	    pairs = list(combinations(MODES, 2))
   147	    for pi, (m1, m2) in enumerate(pairs):
   148	        gap = m["pairwise_cosine_gap"][f"{m1}_vs_{m2}"][17]
   149	        a = m["pairwise_auroc"][f"{m1}_vs_{m2}"][17]
   150	        L17_section.append(f"| {DISPLAY[m1]} vs {DISPLAY[m2]} | {gap:.4f} | {a:.3f} |")
   151	    lines.extend(L17_section)
   152	
   153	    # Phantom-arm specific anchor — P-SoM cosine to each baseline mode at L17
   154	    psom_section = ["", "## P-SoM vs baseline modes (paper §5 HERO arm)", "",
   155	                     "P-SoM identity test: is P-SoM closer to SoM (prompt-axis sibling) or DOM (text-axis sibling)?",
   156	                     ""]
   157	    psom_section.append("| L | P-SoM↔DOM | P-SoM↔SoM | P-SoM↔Vision | P-SoM↔P-text | P-SoM↔P-prompt |")
   158	    psom_section.append("|---|---|---|---|---|---|")
   159	    for L in [0, 8, 11, 17, 24, 30, 36]:
   160	        row = [f"L{L:02d}"]
   161	        for other in ["dom", "som", "vision", "phantom_text", "phantom_prompt"]:
   162	            key = pair_key("phantom_som", other)
   163	            row.append(f"{m['pairwise_cosine_gap'][key][L]:.4f}")
   164	        psom_section.append("| " + " | ".join(row) + " |")
   165	    lines.extend(psom_section)
   166	
   167	    out.parent.mkdir(parents=True, exist_ok=True)
   168	    out.write_text("\n".join(lines) + "\n")
   169	    print(f"[stage4] summary → {out}")
   170	
   171	
   172	def plot(cos_gap, auroc, pairs, pca_var, out):
   173	    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
   174	    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
   175	
   176	    pair_labels = [f"{DISPLAY[m1]}↔{DISPLAY[m2]}" for m1, m2 in pairs]
   177	
   178	    ax = axes[0, 0]
   179	    im = ax.imshow(cos_gap, aspect="auto", cmap="viridis")
   180	    ax.set_yticks(range(len(pairs)))
   181	    ax.set_yticklabels(pair_labels, fontsize=7)
   182	    ax.set_xlabel("Layer index")
   183	    ax.set_title("(a) Pairwise cosine gap (geometric distance between mode means)")
   184	    plt.colorbar(im, ax=ax)
   185	
   186	    ax = axes[0, 1]
   187	    im = ax.imshow(auroc, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
   188	    ax.set_yticks(range(len(pairs)))
   189	    ax.set_yticklabels(pair_labels, fontsize=7)
   190	    ax.set_xlabel("Layer index")
   191	    ax.set_title("(b) Pairwise AUROC (project onto Δ-mean direction, classify)")
   192	    plt.colorbar(im, ax=ax)
   193	
   194	    ax = axes[1, 0]
   195	    for mi, mode in enumerate(MODES):
   196	        ax.plot(pca_var[mi], label=DISPLAY[mode], linewidth=1.5)
   197	    ax.set_xlabel("Layer index")
   198	    ax.set_ylabel("Top-10 PCA cumulative variance explained")
   199	    ax.set_title("(c) Per-mode within-cluster dimensionality")
   200	    ax.legend(fontsize=8, loc="lower right")
   201	    ax.grid(alpha=0.3)
   202	
   203	    ax = axes[1, 1]
   204	    psom_idx = {f"{m1}_vs_{m2}": i for i, (m1, m2) in enumerate(pairs)}
   205	    for other in ["dom", "som", "vision", "phantom_text", "phantom_prompt"]:
   206	        key = pair_key("phantom_som", other)
   207	        ax.plot(cos_gap[psom_idx[key]], label=f"P-SoM ↔ {DISPLAY[other]}", linewidth=1.5)
   208	    ax.axvline(17, color="red", linestyle=":", alpha=0.5, label="L17 (Stage 2 disruption locus)")
   209	    ax.set_xlabel("Layer index")
   210	    ax.set_ylabel("Cosine gap to P-SoM")
   211	    ax.set_title("(d) P-SoM identity — closest sibling per layer")
   212	    ax.legend(fontsize=7, loc="upper left")
   213	    ax.grid(alpha=0.3)
   214	
   215	    fig.suptitle("Stage 4 Method 4.2: Phantom routing space hidden state geometry (Qwen3-VL-4B B1 cls)",
   216	                  fontsize=12, fontweight="bold")
   217	    fig.tight_layout()
   218	    out.parent.mkdir(parents=True, exist_ok=True)
   219	    fig.savefig(out, bbox_inches="tight")
   220	    print(f"[stage4] figure → {out}")
   221	
   222	
   223	if __name__ == "__main__":
   224	    main()

exec
/bin/bash -lc "nl -ba scripts/analysis/stage4_axis2_layer_profile.py | sed -n '1,240p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Exp 1: Axis-2 (prompt-family) layer profile.
     3	
     4	Method 4.2 cosine gap at L17 places (DOM, P-prompt) and (P-text, P-SoM) into
     5	two text-format clusters with prompt-family making essentially zero geometric
     6	contribution to that single layer. The four-fold drop-in property + forest
     7	plot show P-SoM uniquely earns drop-one hero status, but axis-2 mechanism is
     8	not visible at L17.
     9	
    10	This script asks: across ALL 37 layers, where does prompt-family contribute
    11	to residual-stream geometry?
    12	
    13	Pairs computed:
    14	  - axis-2-only (prompt swap, text fixed):
    15	      DOM <-> P-prompt    (both hierarchical AXTree)
    16	      P-text <-> P-SoM    (both flat [SOM_MARKS])
    17	  - axis-1-only (text swap, prompt fixed) — reference:
    18	      DOM <-> P-text      (both DOM-prompt)
    19	      P-prompt <-> P-SoM  (both SoM-prompt)
    20	  - image-axis reference (scale calibration):
    21	      P-SoM <-> SoM
    22	
    23	Outputs:
    24	  - docs/checkpoints/mechanism/results/axis2_layer_profile.md
    25	  - results/phantom_paper/figures/fig_axis2_prompt_layer_profile.png
    26	"""
    27	from __future__ import annotations
    28	
    29	import argparse
    30	from pathlib import Path
    31	
    32	import matplotlib.pyplot as plt
    33	import numpy as np
    34	
    35	ROOT = Path(__file__).resolve().parents[2]
    36	DEFAULT_NPZ_CLS = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz"
    37	DEFAULT_NPZ_RED = ROOT / "results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz"
    38	DEFAULT_OUT_MD = ROOT / "docs/checkpoints/mechanism/results/axis2_layer_profile.md"
    39	DEFAULT_OUT_FIG = ROOT / "results/phantom_paper/figures/fig_axis2_prompt_layer_profile.png"
    40	
    41	PAIRS = [
    42	    # (mode_a, mode_b, label, group, color, linestyle)
    43	    ("dom",          "phantom_prompt", "DOM ↔ P-prompt  (axis-2 only, hierarchical)", "axis-2", "#d62728", "-"),
    44	    ("phantom_text", "phantom_som",    "P-text ↔ P-SoM  (axis-2 only, flat)",         "axis-2", "#ff7f0e", "-"),
    45	    ("dom",          "phantom_text",   "DOM ↔ P-text    (axis-1 only, DOM-prompt)",   "axis-1", "#1f77b4", "--"),
    46	    ("phantom_prompt","phantom_som",   "P-prompt ↔ P-SoM (axis-1 only, SoM-prompt)",  "axis-1", "#2ca02c", "--"),
    47	    ("phantom_som",  "som",            "P-SoM ↔ SoM     (image-axis reference)",      "image",  "#9467bd", ":"),
    48	]
    49	
    50	
    51	def cosine_gap(a: np.ndarray, b: np.ndarray) -> float:
    52	    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    53	
    54	
    55	def compute_pair_curves(npz_path: Path) -> tuple[dict, int, dict]:
    56	    d = np.load(npz_path, allow_pickle=True)
    57	    H = d["hidden_states"]
    58	    ml = d["mode_labels_str"]
    59	    n_layers = H.shape[1]
    60	    means = {}
    61	    for m in {p[0] for p in PAIRS} | {p[1] for p in PAIRS}:
    62	        mask = ml == m
    63	        if mask.sum() == 0:
    64	            continue
    65	        means[m] = H[mask].mean(axis=0)
    66	
    67	    curves = {}
    68	    for a, b, label, group, color, ls in PAIRS:
    69	        if a not in means or b not in means:
    70	            continue
    71	        curve = np.array([cosine_gap(means[a][L], means[b][L]) for L in range(n_layers)])
    72	        curves[label] = {
    73	            "curve": curve,
    74	            "group": group,
    75	            "color": color,
    76	            "linestyle": ls,
    77	            "mode_a": a,
    78	            "mode_b": b,
    79	            "peak_L": int(np.argmax(curve)),
    80	            "peak_gap": float(curve.max()),
    81	            "L17": float(curve[17]) if n_layers > 17 else None,
    82	            "L4": float(curve[4]) if n_layers > 4 else None,
    83	            "L0": float(curve[0]),
    84	            "L_last": float(curve[-1]),
    85	        }
    86	    return curves, n_layers, means
    87	
    88	
    89	def write_md(curves_cls: dict, curves_red: dict, n_layers: int, out: Path):
    90	    out.parent.mkdir(parents=True, exist_ok=True)
    91	    lines = [
    92	        "# Exp 1 — Axis-2 (prompt-family) layer profile",
    93	        "",
    94	        "**Question**: Method 4.2 at L17 shows prompt-family makes ~0 geometric contribution to residual stream",
    95	        "(P-SoM↔P-text 0.0028, DOM↔P-prompt 0.0013). But forest plot drop-one places P-SoM as unique hero,",
    96	        "implying axis-2 (prompt) contributes behaviorally. **Where in the model does axis-2 act?**",
    97	        "",
    98	        "**Method**: For each prompt-only pair (text format fixed, prompt swap), compute full 37-layer cosine gap.",
    99	        "Overlay axis-1-only (text swap, prompt fixed) + image-axis P-SoM↔SoM reference curves to calibrate scale.",
   100	        "",
   101	        "## Results — classifieds site (stage4_multimode_b1_cls, 288 ex)",
   102	        "",
   103	        f"| Pair | Group | L0 | L4 | L17 | L{n_layers-1} | Peak L | Peak gap |",
   104	        f"|---|---|---:|---:|---:|---:|---:|---:|",
   105	    ]
   106	    for label, info in curves_cls.items():
   107	        lines.append(
   108	            f"| {label} | {info['group']} | {info['L0']:.4f} | {info['L4']:.4f} | {info['L17']:.4f} | "
   109	            f"{info['L_last']:.4f} | **L{info['peak_L']}** | {info['peak_gap']:.4f} |"
   110	        )
   111	
   112	    lines += [
   113	        "",
   114	        "## Results — reddit site (stage4_multimode_b1_reddit, 288 ex)",
   115	        "",
   116	        f"| Pair | Group | L0 | L4 | L17 | L{n_layers-1} | Peak L | Peak gap |",
   117	        f"|---|---|---:|---:|---:|---:|---:|---:|",
   118	    ]
   119	    for label, info in curves_red.items():
   120	        lines.append(
   121	            f"| {label} | {info['group']} | {info['L0']:.4f} | {info['L4']:.4f} | {info['L17']:.4f} | "
   122	            f"{info['L_last']:.4f} | **L{info['peak_L']}** | {info['peak_gap']:.4f} |"
   123	        )
   124	
   125	    lines += [
   126	        "",
   127	        "## Interpretation",
   128	        "",
   129	        "Three hypotheses about axis-2 mechanism layer:",
   130	        "",
   131	        "1. **Truly null geometry** — axis-2 pair curves flat <0.01 at all layers. Prompt-family bypasses residual stream entirely (acts at attention pattern or output head). → Next: Exp 3 logit lens or Exp 4 attention probe.",
   132	        "2. **Late-layer spike** — axis-2 pair curves spike at L25+ but flat at mid-layer. Prompt prior re-emerges at output decoding. → Next: Exp 5 late-layer patching.",
   133	        "3. **Early-layer spike absorbed** — axis-2 pair curves spike at L0-L5 then collapse to ~0. Prompt embedding effect absorbed by mid-layer fusion. → Next: Exp 3 logit lens to verify if it re-emerges in output distribution.",
   134	        "",
   135	        "Compare peak layers above against axis-1 (text-format) pairs (the established mechanism with L17 peak) and image-axis reference (~0.04 magnitude). If axis-2 pair peak < 0.01 at all layers, hypothesis 1 holds.",
   136	    ]
   137	    out.write_text("\n".join(lines) + "\n")
   138	    print(f"summary → {out}")
   139	
   140	
   141	def plot(curves_cls: dict, curves_red: dict, n_layers: int, out: Path):
   142	    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
   143	    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
   144	
   145	    for ax, curves, title in [(axes[0], curves_cls, "classifieds"), (axes[1], curves_red, "reddit")]:
   146	        layers = np.arange(n_layers)
   147	        for label, info in curves.items():
   148	            lw = 2.5 if info["group"] == "axis-2" else 1.5
   149	            alpha = 1.0 if info["group"] == "axis-2" else 0.7
   150	            ax.plot(layers, info["curve"], color=info["color"], linestyle=info["linestyle"],
   151	                    linewidth=lw, alpha=alpha, label=label)
   152	        ax.axhline(0.01, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
   153	        ax.axvline(17, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
   154	        ax.set_xlabel("Layer index (L0 = embedding, L36 = final block)")
   155	        ax.set_title(f"{title}  (axis-2 = solid, axis-1 = dashed, image = dotted)", fontsize=10)
   156	        ax.grid(True, alpha=0.3)
   157	        ax.legend(fontsize=7, loc="upper left")
   158	
   159	    axes[0].set_ylabel("Cosine gap")
   160	    fig.suptitle("Exp 1: Axis-2 (prompt-family) layer profile — where does the prompt act?", fontsize=11)
   161	    fig.tight_layout()
   162	    out.parent.mkdir(parents=True, exist_ok=True)
   163	    fig.savefig(out)
   164	    print(f"figure → {out}")
   165	
   166	
   167	def main():
   168	    p = argparse.ArgumentParser()
   169	    p.add_argument("--cls-npz", type=Path, default=DEFAULT_NPZ_CLS)
   170	    p.add_argument("--red-npz", type=Path, default=DEFAULT_NPZ_RED)
   171	    p.add_argument("--output-md", type=Path, default=DEFAULT_OUT_MD)
   172	    p.add_argument("--output-fig", type=Path, default=DEFAULT_OUT_FIG)
   173	    args = p.parse_args()
   174	
   175	    print(f"Loading cls: {args.cls_npz}")
   176	    curves_cls, n_layers_cls, _ = compute_pair_curves(args.cls_npz)
   177	    print(f"  {len(curves_cls)} pairs, {n_layers_cls} layers")
   178	
   179	    print(f"Loading reddit: {args.red_npz}")
   180	    curves_red, n_layers_red, _ = compute_pair_curves(args.red_npz)
   181	    print(f"  {len(curves_red)} pairs, {n_layers_red} layers")
   182	
   183	    assert n_layers_cls == n_layers_red, f"layer count mismatch cls={n_layers_cls} red={n_layers_red}"
   184	
   185	    write_md(curves_cls, curves_red, n_layers_cls, args.output_md)
   186	    plot(curves_cls, curves_red, n_layers_cls, args.output_fig)
   187	
   188	
   189	if __name__ == "__main__":
   190	    main()

exec
/bin/bash -lc "nl -ba scripts/analysis/stage4_logit_lens_axis2.py | sed -n '1,290p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Exp 3: Logit lens at late layers — does axis-2 prompt-family signal
     3	re-emerge in output distribution even though mid-layer residual stream
     4	shows only weak (~0.011) signal at L23?
     5	
     6	Method: Apply Qwen3-VL-4B's lm_head + final_norm to each per-layer hidden
     7	state mean, get a token distribution per (mode, layer). For each axis-2
     8	pair (P-text vs P-SoM at same task) compute:
     9	  - top-1 token disagreement rate per layer
    10	  - KL divergence (P-text || P-SoM) per layer
    11	  - log-prob gap on canonical SoM-prompt vs DOM-prompt action tokens
    12	    (e.g., "click" vs "search", "_pick_", json keys)
    13	
    14	This is Wu et al. tool-calling "knows but says differently" mirror: if
    15	axis-2 cosine gap is 0.011 at L23 but output KL is large at L30-L36,
    16	prompt prior is amplified by late-layer decoding into different output.
    17	
    18	Inputs:
    19	  results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz
    20	  results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz
    21	
    22	Outputs:
    23	  docs/checkpoints/mechanism/results/axis2_logit_lens.md
    24	  results/phantom_paper/figures/fig_axis2_logit_lens.png
    25	"""
    26	from __future__ import annotations
    27	
    28	import argparse
    29	import json
    30	from pathlib import Path
    31	
    32	import matplotlib.pyplot as plt
    33	import numpy as np
    34	import torch
    35	from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
    36	
    37	ROOT = Path(__file__).resolve().parents[2]
    38	DEFAULT_CLS_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz"
    39	DEFAULT_RED_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz"
    40	DEFAULT_MD = ROOT / "docs/checkpoints/mechanism/results/axis2_logit_lens.md"
    41	DEFAULT_FIG = ROOT / "results/phantom_paper/figures/fig_axis2_logit_lens.png"
    42	MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
    43	
    44	AXIS_2_PAIRS = [
    45	    ("phantom_text", "phantom_som", "P-text vs P-SoM  (axis-2 flat-text)"),
    46	    ("dom",          "phantom_prompt", "DOM vs P-prompt  (axis-2 hierarchical)"),
    47	]
    48	AXIS_1_PAIRS = [
    49	    ("dom",           "phantom_text",   "DOM vs P-text    (axis-1 DOM-prompt)"),
    50	    ("phantom_prompt","phantom_som",    "P-prompt vs P-SoM (axis-1 SoM-prompt)"),
    51	]
    52	
    53	
    54	def load_lm_head_and_norm(device="cuda"):
    55	    """Load Qwen3-VL-4B lm_head + final_norm from HF cache (offline)."""
    56	    import os
    57	    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    58	    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    59	    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    60	    print(f"  loading Qwen3VLForConditionalGeneration (lm_head + norm only)")
    61	    model = Qwen3VLForConditionalGeneration.from_pretrained(
    62	        MODEL_PATH, dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    63	    )
    64	    # Qwen3-VL structure (verified via p79/mechanistic/activation_patching.py):
    65	    #   model.model.language_model.layers  (36 decoder layers, no embedding included)
    66	    #   model.model.language_model.norm    (final RMSNorm, sibling of layers)
    67	    #   model.lm_head                       (top-level projection)
    68	    norm = model.model.language_model.norm
    69	    lm_head = model.lm_head
    70	    print(f"  norm: {type(norm).__name__}, lm_head: {type(lm_head).__name__}")
    71	    return tokenizer, lm_head, norm, model
    72	
    73	
    74	@torch.no_grad()
    75	def logits_at_layer(hidden: torch.Tensor, lm_head, norm) -> torch.Tensor:
    76	    """hidden: (D,) → logits (V,) after final_norm + lm_head."""
    77	    h = hidden.unsqueeze(0).to(lm_head.weight.device).to(lm_head.weight.dtype)
    78	    h = norm(h)
    79	    logits = lm_head(h).squeeze(0)
    80	    return logits
    81	
    82	
    83	def kl_divergence(p_logits, q_logits) -> float:
    84	    """KL(P || Q) with softmax on logits."""
    85	    log_p = torch.log_softmax(p_logits, dim=-1)
    86	    log_q = torch.log_softmax(q_logits, dim=-1)
    87	    p = log_p.exp()
    88	    kl = (p * (log_p - log_q)).sum().item()
    89	    return kl
    90	
    91	
    92	def top1_agree(p_logits, q_logits) -> bool:
    93	    return torch.argmax(p_logits).item() == torch.argmax(q_logits).item()
    94	
    95	
    96	def compute_pair_logit_lens(npz: Path, pair_pairs: list, lm_head, norm, n_layers_use: int):
    97	    d = np.load(npz, allow_pickle=True)
    98	    H = d["hidden_states"]  # (N, L, D)
    99	    ml = d["mode_labels_str"]
   100	    means = {}
   101	    for m in {p[0] for p in pair_pairs} | {p[1] for p in pair_pairs}:
   102	        mask = ml == m
   103	        if mask.sum() == 0:
   104	            continue
   105	        means[m] = H[mask].mean(axis=0)
   106	
   107	    result = {}
   108	    for a, b, label in pair_pairs:
   109	        if a not in means or b not in means:
   110	            continue
   111	        layer_kl = []
   112	        layer_disagree = []
   113	        for L in range(n_layers_use):
   114	            h_a = torch.tensor(means[a][L])
   115	            h_b = torch.tensor(means[b][L])
   116	            l_a = logits_at_layer(h_a, lm_head, norm)
   117	            l_b = logits_at_layer(h_b, lm_head, norm)
   118	            layer_kl.append(kl_divergence(l_a, l_b))
   119	            layer_disagree.append(0.0 if top1_agree(l_a, l_b) else 1.0)
   120	        result[label] = {
   121	            "kl": np.array(layer_kl),
   122	            "disagree": np.array(layer_disagree),
   123	            "mode_a": a, "mode_b": b,
   124	            "peak_kl_L": int(np.argmax(layer_kl)),
   125	            "peak_kl": float(np.max(layer_kl)),
   126	        }
   127	    return result, n_layers_use
   128	
   129	
   130	def write_md(cls_axis2, cls_axis1, red_axis2, red_axis1, n_layers, out: Path):
   131	    out.parent.mkdir(parents=True, exist_ok=True)
   132	    lines = [
   133	        "# Exp 3 — Logit lens at late layers (axis-2 vs axis-1)",
   134	        "",
   135	        "Apply Qwen3-VL-4B's final_norm + lm_head to per-layer per-mode mean hidden states.",
   136	        "For each axis-isolated pair, compute KL(mode_a || mode_b) and top-1 token disagreement",
   137	        "across all 37 layers. This probes whether axis-2 cosine signal (L23 peak 0.011) gets",
   138	        "amplified into output distribution divergence by late-layer decoding.",
   139	        "",
   140	        "## Classifieds site",
   141	        "",
   142	        "### Axis-2 (prompt-family) pairs:",
   143	        "",
   144	        "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
   145	        "|---|---:|---:|---:|---:|---:|",
   146	    ]
   147	    for label, info in cls_axis2.items():
   148	        kl = info["kl"]
   149	        lines.append(
   150	            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
   151	            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
   152	        )
   153	    lines += ["", "### Axis-1 (text-format) pairs:", "",
   154	              "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
   155	              "|---|---:|---:|---:|---:|---:|"]
   156	    for label, info in cls_axis1.items():
   157	        kl = info["kl"]
   158	        lines.append(
   159	            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
   160	            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
   161	        )
   162	
   163	    lines += ["", "## Reddit site", "",
   164	              "### Axis-2 (prompt-family) pairs:", "",
   165	              "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
   166	              "|---|---:|---:|---:|---:|---:|"]
   167	    for label, info in red_axis2.items():
   168	        kl = info["kl"]
   169	        lines.append(
   170	            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
   171	            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
   172	        )
   173	    lines += ["", "### Axis-1 (text-format) pairs:", "",
   174	              "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
   175	              "|---|---:|---:|---:|---:|---:|"]
   176	    for label, info in red_axis1.items():
   177	        kl = info["kl"]
   178	        lines.append(
   179	            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
   180	            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
   181	        )
   182	
   183	    lines += [
   184	        "",
   185	        "## Interpretation",
   186	        "",
   187	        "Three hypotheses tested:",
   188	        "",
   189	        "- **H_A (axis-2 absent from output)**: axis-2 KL flat <0.1 at all layers → prompt-family",
   190	        "  effect bypasses logit lens, only visible via attention heads or runtime decoding.",
   191	        "- **H_B (axis-2 amplified at output)**: axis-2 KL peak at L30+ ≫ cosine 0.011 magnitude →",
   192	        "  late-layer decoding amplifies prompt prior into output divergence (Wu et al. tool calling",
   193	        "  'knows but says differently' mirror).",
   194	        "- **H_C (axis-2 tracks residual stream)**: axis-2 KL peak at L23 same as cosine peak →",
   195	        "  prompt prior signal proportional to mid-layer geometry, no amplification.",
   196	        "",
   197	        "Cross-site replication should hold for any of the three. Compare axis-2 KL magnitudes to",
   198	        "axis-1 KL magnitudes to see whether 3-4x ratio in cosine space persists at output level.",
   199	    ]
   200	    out.write_text("\n".join(lines) + "\n")
   201	    print(f"summary → {out}")
   202	
   203	
   204	def plot(cls_a2, cls_a1, red_a2, red_a1, n_layers, out: Path):
   205	    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
   206	    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
   207	    layers = np.arange(n_layers)
   208	    for ax, a2, a1, site in [(axes[0], cls_a2, cls_a1, "classifieds"),
   209	                              (axes[1], red_a2, red_a1, "reddit")]:
   210	        for label, info in a2.items():
   211	            ax.plot(layers, info["kl"], color="#d62728" if "DOM" in label else "#ff7f0e",
   212	                    linewidth=2.5, label=label)
   213	        for label, info in a1.items():
   214	            ax.plot(layers, info["kl"], color="#1f77b4" if "DOM" in label else "#2ca02c",
   215	                    linestyle="--", linewidth=1.5, alpha=0.7, label=label)
   216	        ax.axvline(17, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
   217	        ax.axvline(23, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
   218	        ax.set_xlabel("Layer index")
   219	        ax.set_title(f"{site}  (axis-2 solid, axis-1 dashed)")
   220	        ax.grid(True, alpha=0.3)
   221	        ax.legend(fontsize=7, loc="upper left")
   222	    axes[0].set_ylabel("KL divergence (logit lens)")
   223	    fig.suptitle("Exp 3: Output distribution divergence per layer via logit lens", fontsize=11)
   224	    fig.tight_layout()
   225	    out.parent.mkdir(parents=True, exist_ok=True)
   226	    fig.savefig(out)
   227	    print(f"figure → {out}")
   228	
   229	
   230	def main():
   231	    p = argparse.ArgumentParser()
   232	    p.add_argument("--cls-npz", type=Path, default=DEFAULT_CLS_NPZ)
   233	    p.add_argument("--red-npz", type=Path, default=DEFAULT_RED_NPZ)
   234	    p.add_argument("--output-md", type=Path, default=DEFAULT_MD)
   235	    p.add_argument("--output-fig", type=Path, default=DEFAULT_FIG)
   236	    args = p.parse_args()
   237	
   238	    print("Loading model...")
   239	    tokenizer, lm_head, norm, _ = load_lm_head_and_norm(device="cuda")
   240	
   241	    print("\n[cls] axis-2 pairs:")
   242	    cls_a2, n_L = compute_pair_logit_lens(args.cls_npz, AXIS_2_PAIRS, lm_head, norm, 37)
   243	    print(f"  done, {len(cls_a2)} pairs")
   244	
   245	    print("[cls] axis-1 pairs:")
   246	    cls_a1, _ = compute_pair_logit_lens(args.cls_npz, AXIS_1_PAIRS, lm_head, norm, 37)
   247	    print(f"  done, {len(cls_a1)} pairs")
   248	
   249	    print("[reddit] axis-2 pairs:")
   250	    red_a2, _ = compute_pair_logit_lens(args.red_npz, AXIS_2_PAIRS, lm_head, norm, 37)
   251	    print("[reddit] axis-1 pairs:")
   252	    red_a1, _ = compute_pair_logit_lens(args.red_npz, AXIS_1_PAIRS, lm_head, norm, 37)
   253	
   254	    write_md(cls_a2, cls_a1, red_a2, red_a1, n_L, args.output_md)
   255	    plot(cls_a2, cls_a1, red_a2, red_a1, n_L, args.output_fig)
   256	
   257	
   258	if __name__ == "__main__":
   259	    main()

exec
/bin/bash -lc "nl -ba scripts/analysis/stage2_layer_significance.py | sed -n '1,210p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Stage 2B/2C layer-resolved significance test.
     3	
     4	Question: Is the apparent mid-layer (L11-L17) disruption in
     5	patching_continuation_results.json statistically significant, or could
     6	it be sampling noise given task heterogeneity?
     7	
     8	Procedure:
     9	1. Load per-task per-layer results from forward (2B) and reverse (2C) runs.
    10	2. For each layer L_n in [0, 5, 11, 17, 23, 29], paired t-test (and Wilcoxon
    11	   signed-rank as non-parametric backup) of:
    12	     H0: overlap_to_target(L_n) >= overlap_to_target(L_35)  (no disruption)
    13	     H1: overlap_to_target(L_n) <  overlap_to_target(L_35)  (one-sided, disruption)
    14	   per-task paired difference (L_n - L_35).
    15	3. Holm-Bonferroni correction across the 6 tested layers (per direction).
    16	4. Cross-direction comparison: Welch's t-test on (L17 - L35) per-task difference
    17	   between forward and reverse, to test if disruption magnitude differs.
    18	
    19	Output: markdown summary written to results/mechanistic/layer_significance_<date>.md
    20	"""
    21	from __future__ import annotations
    22	import argparse
    23	import json
    24	from datetime import datetime
    25	from pathlib import Path
    26	
    27	import numpy as np
    28	from scipy import stats
    29	
    30	
    31	# F12 audit fix 2026-05-09: L0-L35 are **transformer block outputs**, NOT
    32	# embedding output. The patching hooks (`activation_patching.py`
    33	# `register_forward_hook` on each `Qwen3VLTextDecoderLayer`) capture the
    34	# output of block i, so:
    35	#   L0  = output of block 0 (first decoder layer post-norm). Often
    36	#         near-target because only one block has transformed the
    37	#         embedding; NOT the embedding itself.
    38	#   L35 = output of block 35 (final block). Output ≈ target by
    39	#         construction since only final_norm + lm_head remain to produce
    40	#         target tokens.
    41	# If you need true embedding output (n_layers+1 indexing), add a hook to
    42	# `model.model.language_model.embed_tokens` instead.
    43	TEST_LAYERS = [0, 5, 11, 17, 23, 29]
    44	BASELINE_LAYER = 35
    45	
    46	
    47	def load_per_task_results(json_path: Path) -> tuple[dict, list[dict]]:
    48	    data = json.loads(json_path.read_text(encoding="utf-8"))
    49	    return data["config"], data["per_task"]
    50	
    51	
    52	def extract_metric_grid(per_task: list[dict], metric: str) -> np.ndarray:
    53	    """Return shape (n_tasks, n_layers) ndarray for a given metric.
    54	    metric ∈ {token_overlap_to_source, token_overlap_to_target, ld_to_source, ld_to_target}.
    55	    """
    56	    rows = []
    57	    for t in per_task:
    58	        rows.append([pl[metric] for pl in t["per_layer"]])
    59	    return np.asarray(rows, dtype=float)
    60	
    61	
    62	def holm_correct(pvals: list[float]) -> list[tuple[float, bool]]:
    63	    """Holm-Bonferroni step-down. Input p-values, output (adj_p, reject@0.05)."""
    64	    n = len(pvals)
    65	    order = sorted(range(n), key=lambda i: pvals[i])
    66	    adj = [0.0] * n
    67	    reject = [False] * n
    68	    prev_adj = 0.0
    69	    for rank, i in enumerate(order):
    70	        adj[i] = min(1.0, max(prev_adj, pvals[i] * (n - rank)))
    71	        prev_adj = adj[i]
    72	        reject[i] = adj[i] < 0.05
    73	    # Once a rejection is rejected, all weaker p-values cannot reject either
    74	    rejected_so_far = True
    75	    for i in order:
    76	        if not reject[i]:
    77	            rejected_so_far = False
    78	        if not rejected_so_far:
    79	            reject[i] = False
    80	    return [(adj[i], reject[i]) for i in range(n)]
    81	
    82	
    83	def per_direction_layer_test(
    84	    label: str, per_task: list[dict], metric: str = "token_overlap_to_target"
    85	) -> dict:
    86	    """Paired t-test + Wilcoxon for each test-layer vs L35 baseline.
    87	
    88	    Direction convention: H1 = layer disrupts vs baseline. For overlap metrics
    89	    higher means closer to target → disruption is layer < baseline (alternative='less').
    90	    For Levenshtein-distance metrics, lower means closer to target → disruption is
    91	    layer > baseline (alternative='greater'). We auto-detect from metric name.
    92	
    93	    Returns dict with per-layer rows + Holm-adjusted p-values.
    94	    """
    95	    grid = extract_metric_grid(per_task, metric)  # (N, 36)
    96	    n_tasks = grid.shape[0]
    97	    baseline = grid[:, BASELINE_LAYER]
    98	
    99	    # Direction: 'less' for overlap-style, 'greater' for ld-style metrics
   100	    is_distance_metric = metric.startswith("ld_")
   101	    alt_direction = "greater" if is_distance_metric else "less"
   102	
   103	    rng = np.random.default_rng(seed=42)
   104	    n_boot = 10000
   105	
   106	    rows = []
   107	    raw_pvals = []
   108	    for L in TEST_LAYERS:
   109	        layer_vals = grid[:, L]
   110	        diff = layer_vals - baseline
   111	
   112	        # C9 fix: handle constant-column edge case (e.g. cell D L0 has all 1.0
   113	        # → 0 variance → NaN t-stat). Skip test, mark as null.
   114	        if np.std(diff, ddof=1) < 1e-12 if n_tasks > 1 else True:
   115	            t_stat, t_p_one = float("nan"), 1.0
   116	            w_stat, w_p_one = float("nan"), 1.0
   117	            ci_lo, ci_hi = float(diff.mean()), float(diff.mean())
   118	        else:
   119	            # Paired t-test (use scipy's built-in alternative= for clarity)
   120	            t_stat, t_p_one = stats.ttest_rel(layer_vals, baseline, alternative=alt_direction)
   121	
   122	            # Wilcoxon signed-rank (non-parametric backup)
   123	            try:
   124	                w_stat, w_p_one = stats.wilcoxon(layer_vals, baseline,
   125	                                                 zero_method="wilcox", alternative=alt_direction)
   126	            except ValueError:
   127	                w_stat, w_p_one = float("nan"), 1.0
   128	
   129	            # C3 fix: bootstrap percentile 95% CI on mean diff (resample tasks)
   130	            boot_means = np.empty(n_boot)
   131	            for b in range(n_boot):
   132	                idx = rng.integers(0, n_tasks, size=n_tasks)
   133	                boot_means[b] = diff[idx].mean()
   134	            ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
   135	
   136	        rows.append({
   137	            "layer": L,
   138	            "mean_layer": float(layer_vals.mean()),
   139	            "mean_baseline": float(baseline.mean()),
   140	            "mean_diff": float(diff.mean()),
   141	            "std_diff": float(diff.std(ddof=1)) if n_tasks > 1 else 0.0,
   142	            "ci_lo_95": ci_lo,
   143	            "ci_hi_95": ci_hi,
   144	            "t_stat": float(t_stat),
   145	            "t_p_one_sided": float(t_p_one),
   146	            "wilcoxon_p_one_sided": float(w_p_one),
   147	        })
   148	        raw_pvals.append(t_p_one)
   149	
   150	    holm = holm_correct(raw_pvals)
   151	    for row, (adj, rej) in zip(rows, holm):
   152	        row["t_p_holm_adj"] = adj
   153	        row["holm_reject_h0"] = rej
   154	
   155	    return {
   156	        "label": label,
   157	        "metric": metric,
   158	        "n_tasks": n_tasks,
   159	        "baseline_layer": BASELINE_LAYER,
   160	        "rows": rows,
   161	    }
   162	
   163	
   164	def cross_direction_test(
   165	    fwd_per_task: list[dict],
   166	    rev_per_task: list[dict],
   167	    layer: int = 17,
   168	    metric: str = "token_overlap_to_target",
   169	) -> dict:
   170	    """Welch's t-test: are forward and reverse disruption magnitudes equal at layer L?
   171	
   172	    Test H0: mean(fwd L_n - L_35) == mean(rev L_n - L_35).
   173	    """
   174	    fwd_grid = extract_metric_grid(fwd_per_task, metric)
   175	    rev_grid = extract_metric_grid(rev_per_task, metric)
   176	    fwd_diff = fwd_grid[:, layer] - fwd_grid[:, BASELINE_LAYER]
   177	    rev_diff = rev_grid[:, layer] - rev_grid[:, BASELINE_LAYER]
   178	
   179	    t_stat, t_p_two = stats.ttest_ind(fwd_diff, rev_diff, equal_var=False)
   180	
   181	    return {
   182	        "layer": layer,
   183	        "metric": metric,
   184	        "fwd_n": int(fwd_grid.shape[0]),
   185	        "rev_n": int(rev_grid.shape[0]),
   186	        "fwd_mean_diff": float(fwd_diff.mean()),
   187	        "fwd_std_diff": float(fwd_diff.std(ddof=1)),
   188	        "rev_mean_diff": float(rev_diff.mean()),
   189	        "rev_std_diff": float(rev_diff.std(ddof=1)),
   190	        "welch_t": float(t_stat),
   191	        "welch_p_two_sided": float(t_p_two),
   192	    }
   193	
   194	
   195	def fmt_p(p: float) -> str:
   196	    if p < 0.001:
   197	        return "<0.001 ***"
   198	    if p < 0.01:
   199	        return f"{p:.3f} **"
   200	    if p < 0.05:
   201	        return f"{p:.3f} *"
   202	    return f"{p:.3f}"
   203	
   204	
   205	def render_markdown(
   206	    fwd_overlap: dict, fwd_ld: dict,
   207	    rev_overlap: dict, rev_ld: dict,
   208	    cross_overlap_l17: dict, cross_ld_l17: dict,
   209	) -> str:
   210	    today = datetime.now().strftime("%Y-%m-%d")

exec
/bin/bash -lc "nl -ba scripts/analysis/stage2_layer_significance.py | sed -n '211,390p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   211	    out = []
   212	    out.append(f"# Stage 2 Layer-Resolved Significance Test ({today})")
   213	    out.append("")
   214	    out.append("Tests whether mid-layer disruption (L11-L17) in continuation patching")
   215	    out.append("is statistically significant or sampling noise. Procedure: per-task")
   216	    out.append("paired t-test of layer L_n vs L35 baseline, one-sided H1: disruption.")
   217	    out.append("Holm-Bonferroni correction across 6 tested layers per direction.")
   218	    out.append("")
   219	    out.append("Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001 (Holm-adjusted).")
   220	    out.append("")
   221	
   222	    for direction_results in [fwd_overlap, rev_overlap, fwd_ld, rev_ld]:
   223	        out.append(f"## {direction_results['label']} — metric: `{direction_results['metric']}`")
   224	        out.append(f"N tasks: {direction_results['n_tasks']}, baseline layer: L{direction_results['baseline_layer']}")
   225	        out.append("")
   226	        out.append("| Layer | mean(L) | mean(L35) | Δ mean | Δ std | 95% CI (boot) | t-stat | p (raw) | p (Holm) | reject H0 |")
   227	        out.append("|---|---|---|---|---|---|---|---|---|---|")
   228	        for r in direction_results["rows"]:
   229	            ci_str = f"[{r.get('ci_lo_95', 0.0):+.3f}, {r.get('ci_hi_95', 0.0):+.3f}]"
   230	            out.append(
   231	                f"| L{r['layer']:>2} | {r['mean_layer']:.3f} | {r['mean_baseline']:.3f} | "
   232	                f"{r['mean_diff']:+.3f} | {r['std_diff']:.3f} | {ci_str} | {r['t_stat']:+.2f} | "
   233	                f"{fmt_p(r['t_p_one_sided'])} | {fmt_p(r['t_p_holm_adj'])} | "
   234	                f"{'✓ Yes' if r['holm_reject_h0'] else '✗ No'} |"
   235	            )
   236	        out.append("")
   237	
   238	    out.append("## Cross-direction comparison @ L17 (Welch's t-test)")
   239	    out.append("")
   240	    out.append("Are forward and reverse disruption magnitudes statistically distinguishable?")
   241	    out.append("")
   242	    for cross in [cross_overlap_l17, cross_ld_l17]:
   243	        out.append(f"**Metric**: `{cross['metric']}` @ L{cross['layer']}")
   244	        out.append(f"- Forward (N={cross['fwd_n']}): Δ = {cross['fwd_mean_diff']:+.3f} ± {cross['fwd_std_diff']:.3f}")
   245	        out.append(f"- Reverse (N={cross['rev_n']}): Δ = {cross['rev_mean_diff']:+.3f} ± {cross['rev_std_diff']:.3f}")
   246	        out.append(f"- Welch's t: {cross['welch_t']:+.2f}, p (two-sided) = {fmt_p(cross['welch_p_two_sided'])}")
   247	        out.append("")
   248	
   249	    out.append("## Interpretation")
   250	    out.append("")
   251	    out.append("- **Holm-rejected layers** in forward direction = layers where mid-layer disruption survives multiple-comparison correction.")
   252	    out.append("- **Cross-direction p**: small p indicates forward and reverse magnitudes differ; large p means they're indistinguishable (consistent with bidirectional mechanism).")
   253	    out.append("")
   254	    out.append("**Caveats**:")
   255	    out.append("- Forward (24 task) and reverse (15 task) ran on DIFFERENT task subsets")
   256	    out.append("  (curated by directional composite score). Selection-bias artifact not")
   257	    out.append("  separable from genuine bidirectional mechanism without 2x2 control")
   258	    out.append("  (see `qsub_2x2_*_myriad.sh` cells C/D).")
   259	    out.append("- Pattern-based mirage curation (curate_mirage_tasks.py) may correlate")
   260	    out.append("  with patching effect size — strong-tier may be effectively the high-")
   261	    out.append("  effect-size subset of full task population.")
   262	    out.append("")
   263	
   264	    return "\n".join(out)
   265	
   266	
   267	def main():
   268	    p = argparse.ArgumentParser()
   269	    p.add_argument("--fwd-results", default="results/mechanistic/stage2b_curated_b1_cls_myriad/patching_continuation_results.json")
   270	    p.add_argument("--rev-results", default="results/mechanistic/stage2c_reverse_curated_b1_cls_myriad/patching_continuation_results.json")
   271	    p.add_argument("--cellc-results", default="results/mechanistic/stage2b_2x2_fwd_revtasks_myriad/patching_continuation_results.json",
   272	                   help="Cell C: forward direction × reverse-tier 15 tasks (2x2 control)")
   273	    p.add_argument("--celld-results", default="results/mechanistic/stage2c_2x2_rev_strongtasks_myriad/patching_continuation_results.json",
   274	                   help="Cell D: reverse direction × strong-tier 24 tasks (2x2 control)")
   275	    p.add_argument("--output", default=None,
   276	                   help="Output markdown path. Default: results/mechanistic/layer_significance_<date>.md")
   277	    args = p.parse_args()
   278	
   279	    fwd_cfg, fwd_per_task = load_per_task_results(Path(args.fwd_results))
   280	    rev_cfg, rev_per_task = load_per_task_results(Path(args.rev_results))
   281	    cellc_per_task = []
   282	    celld_per_task = []
   283	    if Path(args.cellc_results).exists():
   284	        _, cellc_per_task = load_per_task_results(Path(args.cellc_results))
   285	    if Path(args.celld_results).exists():
   286	        _, celld_per_task = load_per_task_results(Path(args.celld_results))
   287	
   288	    print(f"Cell A (fwd × strong):    N={len(fwd_per_task)} tasks")
   289	    print(f"Cell B (rev × reverse):   N={len(rev_per_task)} tasks")
   290	    print(f"Cell C (fwd × reverse):   N={len(cellc_per_task)} tasks")
   291	    print(f"Cell D (rev × strong):    N={len(celld_per_task)} tasks")
   292	
   293	    cells = [
   294	        ("Cell A: forward × strong-tier (24)", fwd_per_task),
   295	        ("Cell B: reverse × reverse-tier (15)", rev_per_task),
   296	        ("Cell C: forward × reverse-tier (15)", cellc_per_task),
   297	        ("Cell D: reverse × strong-tier (24)", celld_per_task),
   298	    ]
   299	    cell_results_overlap = []
   300	    cell_results_ld = []
   301	    for label, ptasks in cells:
   302	        if not ptasks:
   303	            continue
   304	        cell_results_overlap.append(per_direction_layer_test(
   305	            f"{label} — overlap_to_target", ptasks, metric="token_overlap_to_target",
   306	        ))
   307	        cell_results_ld.append(per_direction_layer_test(
   308	            f"{label} — LD_to_target", ptasks, metric="ld_to_target",
   309	        ))
   310	    # Keep legacy names for renderer
   311	    fwd_overlap = cell_results_overlap[0] if cell_results_overlap else None
   312	    rev_overlap = cell_results_overlap[1] if len(cell_results_overlap) > 1 else None
   313	    fwd_ld = cell_results_ld[0] if cell_results_ld else None
   314	    rev_ld = cell_results_ld[1] if len(cell_results_ld) > 1 else None
   315	    # Note for LD: higher = more disruption (output further from target). The
   316	    # directionality of "less than baseline" inverts. We flip sign of diff
   317	    # internally via the metric name handling — but the test is paired so the
   318	    # interpretation needs care. To keep consistent with overlap interpretation,
   319	    # we manually flip the test direction for LD by negating the metric.
   320	
   321	    cross_overlap_l17 = cross_direction_test(
   322	        fwd_per_task, rev_per_task, layer=17,
   323	        metric="token_overlap_to_target",
   324	    )
   325	    cross_ld_l17 = cross_direction_test(
   326	        fwd_per_task, rev_per_task, layer=17,
   327	        metric="ld_to_target",
   328	    )
   329	
   330	    md = render_markdown(fwd_overlap, fwd_ld, rev_overlap, rev_ld,
   331	                         cross_overlap_l17, cross_ld_l17)
   332	    # Append all 4-cell tables after the legacy 2-cell layout (back-compat)
   333	    extra = ["\n## All cells (2x2 expanded)\n"]
   334	    for r in cell_results_overlap + cell_results_ld:
   335	        extra.append(f"### {r['label']}")
   336	        extra.append(f"N={r['n_tasks']}, baseline L{r['baseline_layer']}")
   337	        extra.append("")
   338	        extra.append("| Layer | mean(L) | mean(L35) | Δ mean | Δ std | 95% CI (boot) | t-stat | p (raw) | p (Holm) | reject H0 |")
   339	        extra.append("|---|---|---|---|---|---|---|---|---|---|")
   340	        for row in r["rows"]:
   341	            ci_str = f"[{row.get('ci_lo_95', 0.0):+.3f}, {row.get('ci_hi_95', 0.0):+.3f}]"
   342	            extra.append(
   343	                f"| L{row['layer']:>2} | {row['mean_layer']:.3f} | {row['mean_baseline']:.3f} | "
   344	                f"{row['mean_diff']:+.3f} | {row['std_diff']:.3f} | {ci_str} | {row['t_stat']:+.2f} | "
   345	                f"{fmt_p(row['t_p_one_sided'])} | {fmt_p(row['t_p_holm_adj'])} | "
   346	                f"{'✓ Yes' if row['holm_reject_h0'] else '✗ No'} |"
   347	            )
   348	        extra.append("")
   349	    md = md + "\n".join(extra)
   350	
   351	    if args.output:
   352	        out_path = Path(args.output)
   353	    else:
   354	        today = datetime.now().strftime("%Y%m%d")
   355	        out_path = Path(f"results/mechanistic/layer_significance_{today}.md")
   356	    out_path.parent.mkdir(parents=True, exist_ok=True)
   357	    out_path.write_text(md, encoding="utf-8")
   358	    print(f"\nWrote: {out_path}")
   359	    print()
   360	    print(md)
   361	
   362	
   363	if __name__ == "__main__":
   364	    main()

exec
/bin/bash -lc "nl -ba scripts/analysis/hero_claim_bootstrap.py | sed -n '1,290p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""W1 hero-claim bootstrap CI (response to /stress W1 attack).
     3	
     4	/stress reviewer 2026-05-12 attacked the paper §1 hero claim:
     5	  "Phantom-SoM matches or modestly exceeds full SoM on reddit
     6	   (13.81% vs 10.48%, N=210); the gap is within 2σ under run-to-run
     7	   variability we observe in same-condition repeats"
     8	
     9	The author's own hedging suggests +3.33pp is statistically marginal.
    10	The reviewer demands per-seed bootstrap 95% CI on the pairwise comparison
    11	and on the drop-one oracle, with strict-positive lower bound or downgrade
    12	the prose.
    13	
    14	This script loads B0 reddit per-task adjusted_success for all 6 completed
    15	modes (DOM, SoM, Vision, P-SoM, P-text, P-prompt), bootstraps 10000 task
    16	resamples (N=210 with replacement), and reports for each comparison:
    17	  - Point estimate
    18	  - Bootstrap 95% percentile CI
    19	  - P(diff > 0) — strict-positive bootstrap probability
    20	  - P(diff > 1pp) — practical-significance bootstrap probability
    21	
    22	Also applies to classifieds for cross-site sanity (expect SoM > P-SoM
    23	on cls = sanity check passes if cls reddit story is calibrated).
    24	
    25	Outputs:
    26	  docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md
    27	"""
    28	from __future__ import annotations
    29	
    30	import argparse
    31	import json
    32	import re
    33	from pathlib import Path
    34	
    35	import numpy as np
    36	
    37	ROOT = Path(__file__).resolve().parents[2]
    38	DEFAULT_OUT_MD = ROOT / "docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md"
    39	
    40	# B0 condition directory mapping per site.
    41	SITES = {
    42	    "reddit": {
    43	        "dom":            "B0_3mode_reddit_20260422/phase1_dom_router_0",
    44	        "som":            "B0_3mode_reddit_20260422/phase1_som_router_0",
    45	        "vision":         "B0_3mode_reddit_20260422/phase1_vision_router_0",
    46	        "phantom_som":    "B0_phantom_som_reddit_20260428/phase1_phantom_som_router_0",
    47	        "phantom_text":   "B0_phantom_text_reddit_20260427/phase1_phantom_dom_router_0",
    48	        "phantom_prompt": "B0_phantom_prompt_reddit_20260429/phase1_phantom_prompt_router_0",
    49	    },
    50	    "classifieds": {
    51	        "dom":            "B0_3mode_classifieds_20260413/phase1_dom_router_0",
    52	        "som":            "B0_3mode_classifieds_20260413/phase1_som_router_0",
    53	        "vision":         "B0_3mode_classifieds_20260413/phase1_vision_router_0",
    54	        "phantom_som":    "B0_phantom_som_classifieds_20260426/phase1_phantom_som_router_0",
    55	        "phantom_text":   "B0_phantom_text_classifieds_20260427/phase1_phantom_dom_router_0",
    56	    },
    57	}
    58	
    59	
    60	def load_adjusted_success(episodes_dir: Path) -> dict[int, bool]:
    61	    """Load per-task adjusted_success bool from episodes/*_summary_v2.json files."""
    62	    out = {}
    63	    if not episodes_dir.exists():
    64	        return out
    65	    for p in sorted(episodes_dir.glob("*_summary_v2.json")):
    66	        # files: <site>_task_<int>_summary_v2.json
    67	        m = re.search(r"task_(\d+)", p.name)
    68	        if not m:
    69	            continue
    70	        tid = int(m.group(1))
    71	        try:
    72	            rec = json.loads(p.read_text())
    73	        except Exception:
    74	            continue
    75	        # adjusted_success preferred; fall back to success
    76	        v = rec.get("adjusted_success", rec.get("success", False))
    77	        out[tid] = bool(v)
    78	    return out
    79	
    80	
    81	def build_success_matrix(site: str) -> tuple[np.ndarray, list[int], list[str]]:
    82	    """Build (N_tasks x N_modes) binary success matrix on the same-task subset."""
    83	    mode_dirs = SITES[site]
    84	    per_mode = {}
    85	    for mode, rel in mode_dirs.items():
    86	        epi_dir = ROOT / "results/visualwebarena/phase1" / rel / "episodes"
    87	        per_mode[mode] = load_adjusted_success(epi_dir)
    88	    # same-task subset: tasks present in ALL modes
    89	    task_sets = [set(d.keys()) for d in per_mode.values()]
    90	    common_tasks = sorted(set.intersection(*task_sets)) if task_sets else []
    91	    modes = list(per_mode.keys())
    92	    n = len(common_tasks)
    93	    M = np.zeros((n, len(modes)), dtype=int)
    94	    for i, t in enumerate(common_tasks):
    95	        for j, m in enumerate(modes):
    96	            M[i, j] = int(per_mode[m].get(t, False))
    97	    return M, common_tasks, modes
    98	
    99	
   100	def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, B: int = 10000, seed: int = 42):
   101	    """Bootstrap 95% percentile CI on (mean(a) - mean(b)) per-task paired."""
   102	    n = len(a)
   103	    rng = np.random.default_rng(seed)
   104	    diffs = np.empty(B)
   105	    for k in range(B):
   106	        idx = rng.integers(0, n, size=n)
   107	        diffs[k] = 100 * (a[idx].mean() - b[idx].mean())
   108	    return {
   109	        "point": float(100 * (a.mean() - b.mean())),
   110	        "ci_lo": float(np.quantile(diffs, 0.025)),
   111	        "ci_hi": float(np.quantile(diffs, 0.975)),
   112	        "p_gt_0": float((diffs > 0).mean()),
   113	        "p_gt_1pp": float((diffs > 1.0).mean()),
   114	        "median": float(np.median(diffs)),
   115	    }
   116	
   117	
   118	def bootstrap_drop_one_ci(M: np.ndarray, drop_mode_idx: int, modes: list[str],
   119	                          B: int = 10000, seed: int = 42):
   120	    """Bootstrap drop-one oracle: oracle SR with all modes vs oracle SR without mode i."""
   121	    n = M.shape[0]
   122	    rng = np.random.default_rng(seed)
   123	    drops = np.empty(B)
   124	    other_idx = [j for j in range(M.shape[1]) if j != drop_mode_idx]
   125	    for k in range(B):
   126	        idx = rng.integers(0, n, size=n)
   127	        Msub = M[idx]
   128	        oracle_all = (Msub.sum(axis=1) > 0).mean()
   129	        oracle_without = (Msub[:, other_idx].sum(axis=1) > 0).mean()
   130	        drops[k] = 100 * (oracle_all - oracle_without)
   131	    return {
   132	        "point": float(100 * ((M.sum(axis=1) > 0).mean() -
   133	                              (M[:, other_idx].sum(axis=1) > 0).mean())),
   134	        "ci_lo": float(np.quantile(drops, 0.025)),
   135	        "ci_hi": float(np.quantile(drops, 0.975)),
   136	        "p_gt_0": float((drops > 0).mean()),
   137	        "p_gt_1pp": float((drops > 1.0).mean()),
   138	        "median": float(np.median(drops)),
   139	    }
   140	
   141	
   142	def main():
   143	    p = argparse.ArgumentParser()
   144	    p.add_argument("--bootstraps", type=int, default=10000)
   145	    p.add_argument("--seed", type=int, default=42)
   146	    p.add_argument("--output-md", type=Path, default=DEFAULT_OUT_MD)
   147	    args = p.parse_args()
   148	
   149	    np.random.seed(args.seed)
   150	    lines = [
   151	        "# Hero-claim bootstrap CI (W1 defuse)",
   152	        "",
   153	        f"Per-seed bootstrap 95% percentile CI on paired adjusted-SR diffs and drop-one oracle. "
   154	        f"B={args.bootstraps}, seed={args.seed}. Tasks resampled with replacement at task level.",
   155	        "",
   156	        "**Defuse target**: /stress W1 attack — paper §1 hero claim 'P-SoM 13.81% > SoM 10.48% reddit' "
   157	        "is statistically marginal under author's own 2σ hedge.",
   158	        "",
   159	    ]
   160	
   161	    for site in ["reddit", "classifieds"]:
   162	        try:
   163	            M, tasks, modes = build_success_matrix(site)
   164	        except Exception as e:
   165	            lines.append(f"## {site}: FAILED to load — {e}")
   166	            continue
   167	        n = M.shape[0]
   168	        lines += [
   169	            f"## {site} (N={n} same-task)",
   170	            "",
   171	            "**Per-mode adjusted SR (%)**:",
   172	            "",
   173	        ]
   174	        for j, m in enumerate(modes):
   175	            sr = 100 * M[:, j].mean()
   176	            lines.append(f"- {m}: {sr:.2f}%")
   177	        lines.append("")
   178	
   179	        # Key pairwise comparisons
   180	        mi = {m: j for j, m in enumerate(modes)}
   181	
   182	        # Define comparisons to run
   183	        comps = []
   184	        if "phantom_som" in mi and "som" in mi:
   185	            comps.append(("P-SoM vs SoM", "phantom_som", "som"))
   186	        if "phantom_som" in mi and "dom" in mi:
   187	            comps.append(("P-SoM vs DOM", "phantom_som", "dom"))
   188	        if "phantom_text" in mi and "dom" in mi:
   189	            comps.append(("P-text vs DOM", "phantom_text", "dom"))
   190	        if "phantom_som" in mi and "phantom_text" in mi:
   191	            comps.append(("P-SoM vs P-text", "phantom_som", "phantom_text"))
   192	
   193	        lines += [
   194	            "**Pairwise SR difference, bootstrap 95% CI:**",
   195	            "",
   196	            "| Comparison | Point (pp) | Median | 95% CI | P(diff > 0) | P(diff > 1pp) |",
   197	            "|---|---:|---:|---:|---:|---:|",
   198	        ]
   199	        for label, a, b in comps:
   200	            r = bootstrap_diff_ci(M[:, mi[a]], M[:, mi[b]], B=args.bootstraps, seed=args.seed)
   201	            sign = "✓ strict-pos" if r["ci_lo"] > 0 else ("✗ crosses 0" if r["ci_hi"] > 0 else "✗ strict-neg")
   202	            lines.append(
   203	                f"| {label} | {r['point']:+.2f} | {r['median']:+.2f} | "
   204	                f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['p_gt_0']:.3f} | {r['p_gt_1pp']:.3f} | "
   205	            )
   206	            lines.append(f"|  | | | {sign} | | |")
   207	        lines.append("")
   208	
   209	        # Drop-one oracle on 4-mode set (DOM, SoM, Vision, P-SoM)
   210	        core_modes = ["dom", "som", "vision", "phantom_som"]
   211	        if all(m in mi for m in core_modes):
   212	            sub_idx = [mi[m] for m in core_modes]
   213	            Msub = M[:, sub_idx]
   214	            lines += [
   215	                f"**Drop-one oracle on {len(core_modes)}-mode set ({', '.join(core_modes)}), bootstrap 95% CI:**",
   216	                "",
   217	                "| Drop mode | Drop-one Δ (pp) | Median | 95% CI | P(Δ > 0) | P(Δ > 1pp) |",
   218	                "|---|---:|---:|---:|---:|---:|",
   219	            ]
   220	            for j, m in enumerate(core_modes):
   221	                r = bootstrap_drop_one_ci(Msub, j, core_modes,
   222	                                           B=args.bootstraps, seed=args.seed + j)
   223	                sign = "✓ strict-pos" if r["ci_lo"] > 0 else "✗ crosses 0"
   224	                lines.append(
   225	                    f"| {m} | {r['point']:+.2f} | {r['median']:+.2f} | "
   226	                    f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['p_gt_0']:.3f} | {r['p_gt_1pp']:.3f} | "
   227	                )
   228	                lines.append(f"|  | | | {sign} | | |")
   229	            lines.append("")
   230	
   231	    # Verdict
   232	    lines += [
   233	        "## Verdict on /stress W1",
   234	        "",
   235	        "Read the **reddit P-SoM vs SoM** row + **reddit drop-one P-SoM** row:",
   236	        "",
   237	        "- If both CIs are strict-positive (ci_lo > 0) AND P(diff > 0) > 0.95 → **W1 attack defused**, "
   238	        "  §1 hero claim is bootstrap-supported. Remove the '2σ hedge' from line 5, lead with the magnitude.",
   239	        "- If CIs cross zero but P(diff > 0) > 0.80 → **W1 partially defused**, the claim is directional",
   240	        "  but not strictly statistically significant. §1 hero must downgrade to 'competitive within 2σ' as",
   241	        "  the author already wrote, but the complementarity (Jaccard / drop-one positive on N=7 tasks) carries",
   242	        "  the structural weight.",
   243	        "- If P(diff > 0) < 0.80 → **W1 sustained**, §1 hero claim must rewrite to 'parity / complementarity",
   244	        "  rather than dominance'. The single-mode comparison is unsupported.",
   245	    ]
   246	
   247	    args.output_md.parent.mkdir(parents=True, exist_ok=True)
   248	    args.output_md.write_text("\n".join(lines) + "\n")
   249	    print(f"summary → {args.output_md}")
   250	
   251	
   252	if __name__ == "__main__":
   253	    main()

exec
/bin/bash -lc "nl -ba scripts/mechanistic/run_stage4_multimode_extract.py | sed -n '1,340p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Stage 4 Method 4.2: extract hidden states for ALL 6 modes on same task set.
     3	
     4	Wraps `p79.mechanistic.extract_hidden_states.HiddenStateExtractor`. For each
     5	(task, step) pair, runs forward pass for all 6 modes (DOM/P-text/P-prompt/
     6	P-SoM/SoM/Vision) and saves per-layer last-token hidden states.
     7	
     8	Output schema matches Stage 1B/1C cache format (npz with hidden_states,
     9	labels, task_ids, step_indices, mode_labels_str), so downstream PCA/cosine
    10	analysis is drop-in.
    11	"""
    12	from __future__ import annotations
    13	
    14	import argparse
    15	import json
    16	import logging
    17	import os
    18	from pathlib import Path
    19	
    20	# B-81h workaround (笔记 §117, commit fda1414): force SDPA math backend so the
    21	# script runs on any GPU architecture. PyTorch's flash + memory-efficient SDPA
    22	# backends only have bf16 cutlass kernels for sm_80+ (A100/H100). On V100
    23	# (sm_70) Myriad nodes the dispatcher raises "cutlassF: no kernel found to
    24	# launch!" instead of falling back. Math backend always works (~2-3x slower
    25	# but correct on any GPU). Opt back in via FORCE_MATH_SDP=0.
    26	if os.environ.get("FORCE_MATH_SDP", "1") != "0":
    27	    try:
    28	        import torch as _torch_for_sdp_setup
    29	        _torch_for_sdp_setup.backends.cuda.enable_flash_sdp(False)
    30	        _torch_for_sdp_setup.backends.cuda.enable_mem_efficient_sdp(False)
    31	        _torch_for_sdp_setup.backends.cuda.enable_math_sdp(True)
    32	    except Exception:
    33	        pass
    34	
    35	import numpy as np
    36	
    37	from p79.mechanistic.extract_hidden_states import HiddenStateExtractor, IMAGE_MAX_SIZE_DEFAULT  # noqa: E402
    38	
    39	logging.basicConfig(level=logging.INFO, format="%(asctime)s [stage4] %(levelname)s: %(message)s",
    40	                    datefmt="%H:%M:%S")
    41	logger = logging.getLogger(__name__)
    42	
    43	ALL_6_MODES = ["dom", "phantom_text", "phantom_prompt", "phantom_som", "som", "vision"]
    44	
    45	
    46	def build_som_marks(obs_text: str) -> str:
    47	    """Extract [SOM_MARKS] block from observation_dom.txt — copy of Stage 2B logic.
    48	
    49	    AXTree dump contains lines like `[N] role 'label'`; we keep those and elide
    50	    the rest.
    51	    """
    52	    import re
    53	    pattern = re.compile(r"^\[\d+\]\s+\w+", re.MULTILINE)
    54	    keep = pattern.findall(obs_text)
    55	    return "\n".join(keep) + "\n[end of som marks]\n"
    56	
    57	
    58	def text_payload_for(mode: str, obs_text: str, som_marks_text: str) -> str:
    59	    """Same mapping as run_stage2b post-bug-fix (2026-05-10)."""
    60	    if mode in ("som", "phantom_som", "phantom_text"):
    61	        return som_marks_text
    62	    if mode in ("phantom_prompt", "dom", "phantom_dom"):
    63	        return obs_text
    64	    if mode == "vision":
    65	        return ""
    66	    return som_marks_text
    67	
    68	
    69	def main():
    70	    parser = argparse.ArgumentParser()
    71	    parser.add_argument("--site", default="classifieds")
    72	    parser.add_argument("--n-tasks", type=int, default=24)
    73	    parser.add_argument("--steps", nargs="+", type=int, default=[2])
    74	    parser.add_argument("--archived-run-dir", required=True,
    75	                        help="archive_subset_b1_<site>/ dir with per-task observation snapshots")
    76	    parser.add_argument("--output", required=True, help="output .npz path")
    77	    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
    78	    parser.add_argument("--modes", nargs="+", default=ALL_6_MODES,
    79	                        help="modes to extract (default: all 6)")
    80	    args = parser.parse_args()
    81	
    82	    archive_dir = Path(args.archived_run_dir)
    83	    if not archive_dir.exists():
    84	        raise SystemExit(f"archive dir missing: {archive_dir}")
    85	
    86	    # Pick first n-tasks task IDs that have artifacts at all requested steps
    87	    task_dirs = sorted(archive_dir.glob(f"{args.site}_task_*"))
    88	    selected = []
    89	    for td in task_dirs:
    90	        if all((td / f"step_{s:03d}" / "observation_dom.txt").exists() and
    91	               (td / f"step_{s:03d}" / "screenshot_annotated.png").exists()
    92	               for s in args.steps):
    93	            tid = int(td.name.rsplit("_", 1)[1])
    94	            selected.append((tid, td))
    95	        if len(selected) >= args.n_tasks:
    96	            break
    97	    logger.info(f"Selected {len(selected)} tasks (target {args.n_tasks})")
    98	    if not selected:
    99	        raise SystemExit("no archived tasks selected; check --site/--steps/--archived-run-dir")
   100	
   101	    # Load intents — use same path as run_stage1_pilot.py (external/visualwebarena/config_files/vwa/test_<site>)
   102	    REPO_ROOT = Path(__file__).resolve().parents[2]
   103	    SITE_TO_CONFIG_DIR = {
   104	        "classifieds": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_classifieds",
   105	        "reddit": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_reddit",
   106	        "shopping": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_shopping",
   107	    }
   108	    intents_by_tid = {}
   109	    cfg_dir = SITE_TO_CONFIG_DIR.get(args.site)
   110	    if cfg_dir and cfg_dir.exists():
   111	        for jf in cfg_dir.glob("*.json"):
   112	            try:
   113	                d = json.load(open(jf))
   114	                # filename is <task_id>.json (stage1 convention); also fallback to d["task_id"]
   115	                try:
   116	                    tid = int(jf.stem)
   117	                except ValueError:
   118	                    tid = int(d.get("task_id", -1))
   119	                intent = d.get("intent", "")
   120	                if intent and tid >= 0:
   121	                    intents_by_tid[tid] = intent
   122	            except Exception as e:
   123	                logger.warning(f"failed to load {jf}: {e}")
   124	                continue
   125	    logger.info(f"Loaded {len(intents_by_tid)} intents from {cfg_dir}")
   126	    if not intents_by_tid:
   127	        manifest_path = archive_dir / "manifest.json"
   128	        if manifest_path.exists():
   129	            try:
   130	                manifest = json.load(open(manifest_path))
   131	                for bucket in ("strong", "reverse"):
   132	                    for item in manifest.get(bucket, []):
   133	                        tid = int(item.get("task_id", -1))
   134	                        intent = item.get("intent", "")
   135	                        if tid >= 0 and intent:
   136	                            intents_by_tid[tid] = intent
   137	                logger.info(f"Loaded {len(intents_by_tid)} intents from {manifest_path}")
   138	            except Exception as e:
   139	                logger.warning(f"failed to load intents from {manifest_path}: {e}")
   140	    if not intents_by_tid:
   141	        raise SystemExit(
   142	            "no intents loaded from external config or archive manifest; "
   143	            "cannot extract hidden states"
   144	        )
   145	
   146	    extractor = HiddenStateExtractor(model_path=args.model_path)
   147	    logger.info("Model loaded")
   148	
   149	    # Iterate
   150	    all_hs, all_modes, all_tids, all_steps, all_labels = [], [], [], [], []
   151	    mode_to_label = {m: i for i, m in enumerate(args.modes)}
   152	
   153	    for tid, td in selected:
   154	        intent = intents_by_tid.get(tid, "")
   155	        if not intent:
   156	            logger.warning(f"task {tid}: missing intent, skip")
   157	            continue
   158	        for step in args.steps:
   159	            step_dir = td / f"step_{step:03d}"
   160	            obs_text = (step_dir / "observation_dom.txt").read_text(encoding="utf-8")
   161	            som_marks = build_som_marks(obs_text)
   162	            screenshot = step_dir / "screenshot_annotated.png"
   163	            for mode in args.modes:
   164	                payload = text_payload_for(mode, obs_text, som_marks)
   165	                # vision and som need image; phantom_*/dom do not
   166	                if mode in ("som", "vision"):
   167	                    img = str(screenshot) if screenshot.exists() else None
   168	                else:
   169	                    img = None
   170	                try:
   171	                    hs = extractor.extract(intent, mode, observation_text=payload, image_path=img)
   172	                    # hs shape (n_layers+1, hidden_dim)
   173	                    all_hs.append(hs.cpu().numpy().astype(np.float32))
   174	                    all_modes.append(mode)
   175	                    all_tids.append(tid)
   176	                    all_steps.append(step)
   177	                    all_labels.append(mode_to_label[mode])
   178	                except Exception as e:
   179	                    logger.error(f"task {tid} step {step} mode {mode} failed: {e}")
   180	
   181	    if not all_hs:
   182	        raise SystemExit("no hidden states extracted; all selected tasks/modes failed")
   183	    H = np.stack(all_hs)
   184	    out = Path(args.output)
   185	    out.parent.mkdir(parents=True, exist_ok=True)
   186	    np.savez_compressed(out,
   187	                        hidden_states=H,
   188	                        labels=np.array(all_labels, dtype=np.int64),
   189	                        task_ids=np.array(all_tids, dtype=np.int64),
   190	                        step_indices=np.array(all_steps, dtype=np.int64),
   191	                        mode_labels_str=np.array(all_modes, dtype="<U16"))
   192	    logger.info(f"Saved {len(all_hs)} examples → {out} ({H.nbytes / 1e6:.1f} MB before compression)")
   193	    logger.info(f"Modes: {dict(zip(*np.unique(all_modes, return_counts=True)))}")
   194	
   195	
   196	if __name__ == "__main__":
   197	    main()

exec
/bin/bash -lc "nl -ba scripts/mechanistic/run_stage4_format_variation_extract.py | sed -n '1,320p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Stage 4 H1 test: text format variation across industry-relevant indexed-list styles.
     3	
     4	Tests refined H1 hypothesis (pretraining co-occurrence shortcut):
     5	  "input contains mark-like indexed region list → activates visual grounding pathway"
     6	
     7	If H1 is correct:
     8	  - All marks-like variants (SoM / Browser-Use @ / AppAgent / Tarsier / numbered / XML)
     9	    should cluster with baseline P-text → image-axis cosine gap peaks at L17+
    10	  - Non-marks variants (plain sentence / hash IDs)
    11	    should cluster with baseline DOM (AXTree) → image-axis cosine gap peaks at L4
    12	
    13	8 industry-relevant variants applied to 24 cls strong-tier tasks × 2 steps:
    14	
    15	  som_standard      [0] role 'label'                    (P79 baseline)
    16	  browser_use_at    @0 label                            (Browser Use SDK style)
    17	  appagent_id       id_0: label                         (AppAgent-v2 style)
    18	  tarsier_typed     [B0:role:label]                     (Tarsier Reworkd)
    19	  plain_numbered    0. label                            (generic numbered)
    20	  xml_tagged        <el_0 role="role">label</el_0>      (OmniParser-style)
    21	  hash_id_control   #aBc7 label                         (random hash, no integer index — control)
    22	  plain_sentence    "label0, label1, label2..."         (no list structure — control)
    23	
    24	Reuses dom (AXTree) + som (image+marks) baseline modes for cross-comparison.
    25	
    26	Output: results/mechanistic/stage4_format_variation_b1_cls/hidden_states.npz
    27	shape (10 modes × 24 tasks × 2 steps = 480, n_layers+1, hidden_dim)
    28	"""
    29	from __future__ import annotations
    30	
    31	import argparse
    32	import json
    33	import logging
    34	import os
    35	import re
    36	import sys
    37	import hashlib
    38	from pathlib import Path
    39	
    40	if os.environ.get("FORCE_MATH_SDP", "1") != "0":
    41	    try:
    42	        import torch as _t
    43	        _t.backends.cuda.enable_flash_sdp(False)
    44	        _t.backends.cuda.enable_mem_efficient_sdp(False)
    45	        _t.backends.cuda.enable_math_sdp(True)
    46	    except Exception:
    47	        pass
    48	
    49	import numpy as np
    50	
    51	sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    52	from p79.mechanistic.extract_hidden_states import HiddenStateExtractor  # noqa: E402
    53	
    54	logging.basicConfig(level=logging.INFO, format="%(asctime)s [stage4fv] %(levelname)s: %(message)s",
    55	                    datefmt="%H:%M:%S")
    56	logger = logging.getLogger(__name__)
    57	
    58	
    59	# Parse [N] role 'label' lines from observation_dom.txt → list of (n, role, label)
    60	MARK_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s+(\S+)\s+'([^']*)'")
    61	
    62	
    63	def extract_marks(obs_text: str) -> list[tuple[int, str, str]]:
    64	    """Extract (idx, role, label) tuples from AXTree-style observation text."""
    65	    out = []
    66	    for line in obs_text.split("\n"):
    67	        m = MARK_LINE_RE.match(line.strip())
    68	        if m:
    69	            out.append((int(m.group(1)), m.group(2), m.group(3)))
    70	    return out
    71	
    72	
    73	def hash_id(n: int) -> str:
    74	    """Deterministic 4-char alphanumeric hash, no integer pattern."""
    75	    h = hashlib.md5(str(n).encode()).hexdigest()
    76	    # Avoid pure digits — mix in letters
    77	    return f"{h[0]}{h[5]}{h[10]}{h[15]}"
    78	
    79	
    80	# Format transformers — each takes obs_text, returns text payload string
    81	def fmt_som_standard(obs_text):
    82	    """Baseline [SOM_MARKS] — same as run_stage2b build_som_marks."""
    83	    return "\n".join(line.strip() for line in obs_text.split("\n")
    84	                      if line.strip().startswith("[") and "]" in line.strip()[:6])
    85	
    86	
    87	def fmt_browser_use_at(obs_text):
    88	    marks = extract_marks(obs_text)
    89	    return "\n".join(f"@{n} {label}" for n, role, label in marks)
    90	
    91	
    92	def fmt_appagent_id(obs_text):
    93	    marks = extract_marks(obs_text)
    94	    return "\n".join(f"id_{n}: {label}" for n, role, label in marks)
    95	
    96	
    97	def fmt_tarsier_typed(obs_text):
    98	    marks = extract_marks(obs_text)
    99	    return "\n".join(f"[B{n}:{role}:{label}]" for n, role, label in marks)
   100	
   101	
   102	def fmt_plain_numbered(obs_text):
   103	    marks = extract_marks(obs_text)
   104	    return "\n".join(f"{n}. {label}" for n, role, label in marks)
   105	
   106	
   107	def fmt_xml_tagged(obs_text):
   108	    marks = extract_marks(obs_text)
   109	    return "\n".join(f'<el_{n} role="{role}">{label}</el_{n}>' for n, role, label in marks)
   110	
   111	
   112	def fmt_hash_id_control(obs_text):
   113	    """Control: replace integer index with non-integer hash. Tests whether integer index is the trigger."""
   114	    marks = extract_marks(obs_text)
   115	    return "\n".join(f"#{hash_id(n)} {label}" for n, role, label in marks)
   116	
   117	
   118	def fmt_plain_sentence(obs_text):
   119	    """Control: drop list structure entirely. Tests whether 'list' pattern is the trigger."""
   120	    marks = extract_marks(obs_text)
   121	    return ", ".join(label for n, role, label in marks)
   122	
   123	
   124	VARIANTS = {
   125	    "som_standard":     fmt_som_standard,
   126	    "browser_use_at":   fmt_browser_use_at,
   127	    "appagent_id":      fmt_appagent_id,
   128	    "tarsier_typed":    fmt_tarsier_typed,
   129	    "plain_numbered":   fmt_plain_numbered,
   130	    "xml_tagged":       fmt_xml_tagged,
   131	    "hash_id_control":  fmt_hash_id_control,
   132	    "plain_sentence":   fmt_plain_sentence,
   133	}
   134	
   135	
   136	def find_archive_dir(p79_root: Path) -> Path:
   137	    """Locate manifest archive — on DGX vs Myriad have different paths."""
   138	    cand = p79_root / "results/mechanistic/archive_subset_b1_cls"
   139	    if cand.exists():
   140	        return cand
   141	    raise FileNotFoundError(f"archive not found at {cand}")
   142	
   143	
   144	def main():
   145	    p = argparse.ArgumentParser()
   146	    p.add_argument("--archived-run-dir", required=True, help="Path to archive_subset_b1_<site>")
   147	    p.add_argument("--site", default="classifieds",
   148	                   help="Task dir prefix in archive: <site>_task_<tid>/. classifieds (default) or reddit.")
   149	    p.add_argument("--output", default=None)
   150	    p.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
   151	    p.add_argument("--tier", default="strong")
   152	    p.add_argument("--n-tasks", type=int, default=24)
   153	    p.add_argument("--steps", default="2,5")
   154	    p.add_argument("--min-free-vram-gb", type=float, default=0.0)
   155	    args = p.parse_args()
   156	
   157	    steps = [int(x) for x in args.steps.split(",")]
   158	    archive_dir = Path(args.archived_run_dir)
   159	    manifest_path = archive_dir / "manifest.json"
   160	    manifest = json.loads(manifest_path.read_text())
   161	    tasks = manifest[args.tier][:args.n_tasks]
   162	    logger.info(f"Loaded {len(tasks)} tasks (tier={args.tier})")
   163	
   164	    intents_by_tid = {int(t["task_id"]): t["intent"] for t in tasks}
   165	
   166	    extractor = HiddenStateExtractor(model_path=args.model_path, min_free_vram_gb=args.min_free_vram_gb)
   167	    logger.info(f"Model loaded: {args.model_path}")
   168	
   169	    # Plus the 2 reused baselines: 'dom' (AXTree) and 'som' (image + marks)
   170	    BASELINES = ["dom", "som"]
   171	    ALL_MODES = list(VARIANTS.keys()) + BASELINES
   172	
   173	    all_hidden = []
   174	    all_meta = []  # tuples of (task_id, step, mode_label)
   175	
   176	    for tid in sorted(intents_by_tid):
   177	        intent = intents_by_tid[tid]
   178	        for step in steps:
   179	            task_dir = archive_dir / f"{args.site}_task_{tid}" / f"step_{step:03d}"
   180	            obs_path = task_dir / "observation_dom.txt"
   181	            if not obs_path.exists():
   182	                logger.warning(f"missing {obs_path}; skip")
   183	                continue
   184	            obs_text = obs_path.read_text(encoding="utf-8")
   185	            screenshot = task_dir / "screenshot_annotated.png"
   186	
   187	            for mode in ALL_MODES:
   188	                if mode in VARIANTS:
   189	                    # Variant: DOM-prompt + variant text + no image
   190	                    variant_text = VARIANTS[mode](obs_text)
   191	                    try:
   192	                        h = extractor.extract(intent=intent, mode="dom",
   193	                                                observation_text=variant_text, image_path=None)
   194	                    except Exception as e:
   195	                        logger.error(f"task {tid} step {step} variant {mode} failed: {e}")
   196	                        continue
   197	                elif mode == "dom":
   198	                    # baseline DOM: AXTree text + DOM-prompt + no image
   199	                    h = extractor.extract(intent=intent, mode="dom",
   200	                                            observation_text=obs_text, image_path=None)
   201	                elif mode == "som":
   202	                    # baseline SoM: marks text + SoM-prompt + WITH image
   203	                    marks_text = fmt_som_standard(obs_text)
   204	                    img_path = screenshot if screenshot.exists() else None
   205	                    h = extractor.extract(intent=intent, mode="som",
   206	                                            observation_text=marks_text, image_path=img_path)
   207	                else:
   208	                    continue
   209	
   210	                all_hidden.append(h.numpy() if hasattr(h, "numpy") else h)
   211	                all_meta.append((tid, step, mode))
   212	            logger.info(f"  task={tid} step={step} done ({len(ALL_MODES)} modes)")
   213	
   214	    if not all_hidden:
   215	        raise SystemExit("no hidden states extracted")
   216	
   217	    H = np.stack(all_hidden)  # (N, n_layers+1, hidden_dim)
   218	    task_ids = np.array([m[0] for m in all_meta])
   219	    step_indices = np.array([m[1] for m in all_meta])
   220	    mode_labels = np.array([m[2] for m in all_meta])
   221	
   222	    out_path = Path(args.output or (Path(__file__).resolve().parents[2]
   223	                                       / "results/mechanistic/stage4_format_variation_b1_cls/hidden_states.npz"))
   224	    out_path.parent.mkdir(parents=True, exist_ok=True)
   225	    labels = np.array([list(ALL_MODES).index(m) for m in mode_labels.tolist()])
   226	    np.savez_compressed(
   227	        out_path,
   228	        hidden_states=H,
   229	        labels=labels,
   230	        task_ids=task_ids,
   231	        step_indices=step_indices,
   232	        mode_labels_str=mode_labels,
   233	    )
   234	    logger.info(f"saved: {out_path}  shape={H.shape}  modes={ALL_MODES}")
   235	
   236	    # Quick pilot_summary marker so auto_pull Phase 0 sentinel passes
   237	    summary_path = out_path.parent / "pilot_summary.md"
   238	    summary_path.write_text(
   239	        f"# Stage 4 Format Variation Extract\n\n"
   240	        f"Shape: {H.shape}\n"
   241	        f"Modes: {ALL_MODES}\n"
   242	        f"Tasks: {len(set(task_ids.tolist()))}\n"
   243	        f"Steps: {sorted(set(step_indices.tolist()))}\n"
   244	    )
   245	    logger.info(f"sentinel: {summary_path}")
   246	
   247	
   248	if __name__ == "__main__":
   249	    main()

exec
/bin/bash -lc "nl -ba scripts/mechanistic/run_stage4_method44_steering.py | sed -n '1,360p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Stage 4 Method 4.4: Counterfactual Activation Steering.
     3	
     4	Validates the (mean_P-SoM − mean_DOM) direction at L17 is CAUSALLY operative:
     5	adding α·v to the DOM forward pass at L17 should pull the generated tokens
     6	toward P-SoM ground truth (not just shift them randomly).
     7	
     8	Ports Tool Calling Linear Steerable Circuit (Anonymous 2026 ACL): Qwen3-4B
     9	80-93% tool switch accuracy via mid-layer mean-difference steering.
    10	
    11	For each (task, step) in cls strong-tier 24:
    12	  - Generate DOM baseline (no steering)
    13	  - Generate P-SoM baseline (no steering)
    14	  - For α ∈ {0.5, 1.0, 2.0, 5.0}: generate from DOM inputs with L17 += α·v
    15	
    16	Metrics per (task, step, α):
    17	  - token_overlap to DOM / to P-SoM (Jaccard)
    18	  - levenshtein_norm to DOM / to P-SoM
    19	  - first_token_match_psom (boolean)
    20	
    21	Output:
    22	  - results/mechanistic/stage4_multimode_b1_cls/method44_steering.json
    23	  - docs/checkpoints/stage4_method44_results.md
    24	"""
    25	from __future__ import annotations
    26	
    27	import argparse
    28	import json
    29	import logging
    30	import os
    31	import sys
    32	from pathlib import Path
    33	
    34	# B-81h workaround for V100; harmless on A100
    35	if os.environ.get("FORCE_MATH_SDP", "1") != "0":
    36	    try:
    37	        import torch as _t
    38	        _t.backends.cuda.enable_flash_sdp(False)
    39	        _t.backends.cuda.enable_mem_efficient_sdp(False)
    40	        _t.backends.cuda.enable_math_sdp(True)
    41	    except Exception:
    42	        pass
    43	
    44	import numpy as np
    45	import torch
    46	
    47	sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    48	from p79.mechanistic.extract_hidden_states import HiddenStateExtractor  # noqa: E402
    49	from p79.mechanistic.activation_patching import ActivationPatcher, _token_seq_overlap, _levenshtein_token  # noqa: E402
    50	
    51	logging.basicConfig(level=logging.INFO, format="%(asctime)s [stage4mm44] %(levelname)s: %(message)s",
    52	                    datefmt="%H:%M:%S")
    53	logger = logging.getLogger(__name__)
    54	
    55	ROOT = Path(__file__).resolve().parents[2]
    56	NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz"
    57	ARCHIVE = ROOT / "results/mechanistic/archive_subset_b1_cls"
    58	MANIFEST = ARCHIVE / "manifest.json"
    59	OUT_JSON = ROOT / "results/mechanistic/stage4_multimode_b1_cls/method44_steering.json"
    60	OUT_MD = ROOT / "docs/checkpoints/stage4_method44_results.md"
    61	
    62	
    63	def build_som_marks(obs_text: str) -> str:
    64	    lines = []
    65	    for line in obs_text.split("\n"):
    66	        s = line.strip()
    67	        if s.startswith("[") and "]" in s[:6]:
    68	            lines.append(s)
    69	    return "\n".join(lines)
    70	
    71	
    72	def build_inputs(extractor: HiddenStateExtractor, intent: str, mode: str, obs_text: str):
    73	    """No-image inputs (Stage 4 modes that don't need image: dom / phantom_som / phantom_text / phantom_prompt)."""
    74	    user_text = extractor._build_user_text(intent, mode, obs_text)
    75	    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    76	    text = extractor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    77	    inputs = extractor.processor(text=[text], padding=True, return_tensors="pt")
    78	    return {k: v.to(extractor.model.device) for k, v in inputs.items()}
    79	
    80	
    81	def compute_direction(npz_path: Path, layer: int) -> np.ndarray:
    82	    d = np.load(npz_path, allow_pickle=True)
    83	    H = d["hidden_states"]
    84	    ml = d["mode_labels_str"]
    85	    v = H[ml == "phantom_som"][:, layer, :].mean(0) - H[ml == "dom"][:, layer, :].mean(0)
    86	    return v
    87	
    88	
    89	def jaccard(a, b) -> float:
    90	    sa, sb = set(a), set(b)
    91	    return len(sa & sb) / max(1, len(sa | sb))
    92	
    93	
    94	def lev_norm(a, b) -> float:
    95	    return _levenshtein_token(a, b) / max(1, max(len(a), len(b)))
    96	
    97	
    98	def main():
    99	    p = argparse.ArgumentParser()
   100	    p.add_argument("--layer", type=int, default=17, help="Steering target layer")
   101	    p.add_argument("--alphas", default="0.5,1.0,2.0,5.0")
   102	    p.add_argument("--max-new-tokens", type=int, default=15)
   103	    p.add_argument("--min-free-vram-gb", type=float, default=0.0)
   104	    p.add_argument("--tier", default="strong")
   105	    p.add_argument("--limit", type=int, default=None, help="Smoke-test: limit to first N tasks")
   106	    args = p.parse_args()
   107	    alphas = [float(a) for a in args.alphas.split(",")]
   108	
   109	    logger.info(f"Loading direction from {NPZ} layer={args.layer}")
   110	    v_np = compute_direction(NPZ, args.layer)
   111	    v_norm = float(np.linalg.norm(v_np))
   112	    logger.info(f"Direction norm = {v_norm:.4f} (will be scaled by alpha)")
   113	
   114	    logger.info(f"Loading manifest {MANIFEST}")
   115	    manifest = json.loads(MANIFEST.read_text())
   116	    tasks = manifest[args.tier]
   117	    if args.limit:
   118	        tasks = tasks[:args.limit]
   119	    logger.info(f"Loaded {len(tasks)} tasks (tier={args.tier})")
   120	
   121	    extractor = HiddenStateExtractor(min_free_vram_gb=args.min_free_vram_gb)
   122	    patcher = ActivationPatcher(extractor.model, extractor.processor)
   123	    logger.info(f"Model loaded; n_layers={patcher.n_layers}")
   124	    v_torch = torch.tensor(v_np)
   125	
   126	    steps = manifest.get("steps", [2, 5])
   127	    per_task_results = []
   128	
   129	    for ti, t in enumerate(tasks):
   130	        tid = int(t["task_id"])
   131	        intent = t["intent"]
   132	        for step in steps:
   133	            obs_path = ARCHIVE / f"classifieds_task_{tid}" / f"step_{step:03d}" / "observation_dom.txt"
   134	            if not obs_path.exists():
   135	                logger.warning(f"missing {obs_path}; skipping")
   136	                continue
   137	            obs_text = obs_path.read_text(encoding="utf-8")
   138	            som_marks_text = build_som_marks(obs_text)
   139	
   140	            dom_inputs = build_inputs(extractor, intent, "dom", obs_text)
   141	            psom_inputs = build_inputs(extractor, intent, "phantom_som", som_marks_text)
   142	
   143	            # Baselines
   144	            dom_gen = patcher.model.generate(
   145	                **dom_inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
   146	                return_dict_in_generate=True, use_cache=True,
   147	            )
   148	            dom_tokens = dom_gen.sequences[0, dom_inputs["input_ids"].shape[1]:].cpu().tolist()
   149	            dom_text = extractor.processor.tokenizer.decode(dom_tokens, skip_special_tokens=True)
   150	
   151	            psom_gen = patcher.model.generate(
   152	                **psom_inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
   153	                return_dict_in_generate=True, use_cache=True,
   154	            )
   155	            psom_tokens = psom_gen.sequences[0, psom_inputs["input_ids"].shape[1]:].cpu().tolist()
   156	            psom_text = extractor.processor.tokenizer.decode(psom_tokens, skip_special_tokens=True)
   157	
   158	            logger.info(f"  task={tid} step={step}: dom={dom_text!r} | psom={psom_text!r}")
   159	
   160	            # Steered generations
   161	            per_alpha = []
   162	            for alpha in alphas:
   163	                st_tokens = patcher.steered_generate(
   164	                    layer_idx=args.layer, direction=v_torch, alpha=alpha,
   165	                    max_new_tokens=args.max_new_tokens, **dom_inputs,
   166	                ).cpu().tolist()
   167	                st_text = extractor.processor.tokenizer.decode(st_tokens, skip_special_tokens=True)
   168	                ovl_dom = jaccard(st_tokens, dom_tokens)
   169	                ovl_psom = jaccard(st_tokens, psom_tokens)
   170	                ld_dom = lev_norm(st_tokens, dom_tokens)
   171	                ld_psom = lev_norm(st_tokens, psom_tokens)
   172	                first_match_psom = (len(st_tokens) > 0 and len(psom_tokens) > 0 and st_tokens[0] == psom_tokens[0])
   173	                per_alpha.append({
   174	                    "alpha": alpha, "steered_text": st_text,
   175	                    "token_overlap_dom": ovl_dom, "token_overlap_psom": ovl_psom,
   176	                    "levenshtein_dom": ld_dom, "levenshtein_psom": ld_psom,
   177	                    "first_token_match_psom": first_match_psom,
   178	                    "shifted_toward_psom": ovl_psom > ovl_dom,
   179	                })
   180	                logger.info(f"    α={alpha:.1f} → {st_text!r} | overlap dom={ovl_dom:.2f} psom={ovl_psom:.2f} | first_token_psom_match={first_match_psom}")
   181	
   182	            per_task_results.append({
   183	                "task_id": tid, "step": step,
   184	                "dom_text": dom_text, "psom_text": psom_text,
   185	                "dom_tokens": dom_tokens, "psom_tokens": psom_tokens,
   186	                "per_alpha": per_alpha,
   187	            })
   188	
   189	            # Incremental save
   190	            OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
   191	            OUT_JSON.write_text(json.dumps({
   192	                "config": {"layer": args.layer, "alphas": alphas, "tier": args.tier,
   193	                            "max_new_tokens": args.max_new_tokens, "direction_norm": v_norm},
   194	                "results": per_task_results,
   195	            }, indent=2))
   196	
   197	    # Aggregate
   198	    agg = {}
   199	    for alpha in alphas:
   200	        rows = [pa for r in per_task_results for pa in r["per_alpha"] if pa["alpha"] == alpha]
   201	        agg[f"alpha_{alpha}"] = {
   202	            "n": len(rows),
   203	            "mean_overlap_dom": float(np.mean([r["token_overlap_dom"] for r in rows])),
   204	            "mean_overlap_psom": float(np.mean([r["token_overlap_psom"] for r in rows])),
   205	            "mean_lev_dom": float(np.mean([r["levenshtein_dom"] for r in rows])),
   206	            "mean_lev_psom": float(np.mean([r["levenshtein_psom"] for r in rows])),
   207	            "first_token_psom_match_rate": float(np.mean([r["first_token_match_psom"] for r in rows])),
   208	            "shifted_toward_psom_rate": float(np.mean([r["shifted_toward_psom"] for r in rows])),
   209	        }
   210	
   211	    final = {"config": {"layer": args.layer, "alphas": alphas, "tier": args.tier,
   212	                          "max_new_tokens": args.max_new_tokens, "direction_norm": v_norm},
   213	              "aggregate": agg, "results": per_task_results}
   214	    OUT_JSON.write_text(json.dumps(final, indent=2))
   215	    logger.info(f"final → {OUT_JSON}")
   216	
   217	    write_md(final, OUT_MD)
   218	
   219	
   220	def write_md(d: dict, out: Path) -> None:
   221	    cfg, agg = d["config"], d["aggregate"]
   222	    lines = [
   223	        "# Stage 4 Method 4.4: Counterfactual Activation Steering",
   224	        "",
   225	        f"**Config**: layer L{cfg['layer']:02d}, steering direction = mean(P-SoM) − mean(DOM) at L{cfg['layer']}, ‖v‖={cfg['direction_norm']:.4f}",
   226	        f"**Tier**: {cfg['tier']} cls × steps {{2, 5}} × n_tasks variable",
   227	        f"**Max new tokens**: {cfg['max_new_tokens']}",
   228	        "",
   229	        "## Aggregate per α (does adding α·v to DOM forward shift toward P-SoM?)",
   230	        "",
   231	        "| α | n | mean overlap_DOM | mean overlap_P-SoM | shifted-toward-P-SoM rate | first-token P-SoM match |",
   232	        "|---|---|---|---|---|---|",
   233	    ]
   234	    for k, v in agg.items():
   235	        alpha = k.split("_")[1]
   236	        lines.append(f"| {alpha} | {v['n']} | {v['mean_overlap_dom']:.3f} | {v['mean_overlap_psom']:.3f} | {v['shifted_toward_psom_rate']:.0%} | {v['first_token_psom_match_rate']:.0%} |")
   237	    lines.append("")
   238	    lines.append("Interpretation: if α=0 baseline overlap_DOM = 1.0 + overlap_P-SoM = some baseline,")
   239	    lines.append("then as α↑ overlap_P-SoM should rise + overlap_DOM should fall, monotonically.")
   240	    lines.append("Tool Calling paper (Anonymous 2026 ACL) reports 80-93% tool-switch rate at α=2-3.")
   241	    lines.append("")
   242	    out.write_text("\n".join(lines) + "\n")
   243	    print(f"[stage4mm44] summary → {out}")
   244	
   245	
   246	if __name__ == "__main__":
   247	    main()

exec
/bin/bash -lc "nl -ba scripts/mechanistic/run_stage4_method44_v2_sweep.py | sed -n '1,380p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Stage 4 Method 4.4 v2: layer × α sweep for mean-diff steering.
     3	
     4	Diag (diag_stage4_method44_layer_check.py) showed v1 null was α-calibration
     5	+ wrong-layer issue. At α=50 + L17, steering DOES shift output toward P-SoM
     6	content ('red Toyota') but also breaks JSON envelope (over-steers).
     7	
     8	This script measures the dose-response surface:
     9	  layers: [11, 17, 23, 29, 33, 34]   ← mid → late, covers Wu et al. L34 default
    10	                                       and Stage 2/4 L17 mid-locus
    11	  α:      [1, 2, 5, 10, 20]           ← Wu et al. typical α=1, our diag found ≥10 needed
    12	
    13	Per (task, step, layer, α):
    14	  - token overlap to DOM baseline / P-SoM baseline (Jaccard)
    15	  - completeness = shifted_toward_psom rate (overlap_psom > overlap_dom)
    16	  - selectivity = JSON valid rate (envelope preserved; starts with '{' or '{ "')
    17	  - reliability = harmonic mean of completeness × selectivity
    18	    (HDMI framework, Khorasani et al. 2026 arXiv:2605.07631)
    19	
    20	Direction at patcher.layers[L] ←→ npz[:, L+1, :]
    21	(extract_hidden_states stores HF outputs.hidden_states with embedding at idx 0).
    22	
    23	Output: results/mechanistic/stage4_multimode_b1_cls/method44_v2_sweep.json
    24	"""
    25	from __future__ import annotations
    26	
    27	import argparse
    28	import json
    29	import logging
    30	import os
    31	import sys
    32	from pathlib import Path
    33	
    34	if os.environ.get("FORCE_MATH_SDP", "1") != "0":
    35	    try:
    36	        import torch as _t
    37	        _t.backends.cuda.enable_flash_sdp(False)
    38	        _t.backends.cuda.enable_mem_efficient_sdp(False)
    39	        _t.backends.cuda.enable_math_sdp(True)
    40	    except Exception:
    41	        pass
    42	
    43	import numpy as np
    44	import torch
    45	
    46	sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    47	from p79.mechanistic.extract_hidden_states import HiddenStateExtractor  # noqa: E402
    48	from p79.mechanistic.activation_patching import ActivationPatcher, _levenshtein_token  # noqa: E402
    49	
    50	logging.basicConfig(level=logging.INFO, format="%(asctime)s [stage4mm44v2] %(levelname)s: %(message)s",
    51	                    datefmt="%H:%M:%S")
    52	logger = logging.getLogger(__name__)
    53	
    54	ROOT = Path(__file__).resolve().parents[2]
    55	NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz"
    56	ARCHIVE = ROOT / "results/mechanistic/archive_subset_b1_cls"
    57	MANIFEST = ARCHIVE / "manifest.json"
    58	OUT_JSON = ROOT / "results/mechanistic/stage4_multimode_b1_cls/method44_v2_sweep.json"
    59	OUT_MD = ROOT / "docs/checkpoints/stage4_method44_v2_results.md"
    60	
    61	DEFAULT_LAYERS = [11, 17, 23, 29, 33, 34]
    62	DEFAULT_ALPHAS = [1.0, 2.0, 5.0, 10.0, 20.0]
    63	
    64	
    65	def build_som_marks(obs_text):
    66	    return "\n".join(s for line in obs_text.split("\n")
    67	                      if (s := line.strip()).startswith("[") and "]" in s[:6])
    68	
    69	
    70	def build_inputs(extractor, intent, mode, obs_text):
    71	    user_text = extractor._build_user_text(intent, mode, obs_text)
    72	    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    73	    text = extractor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    74	    inputs = extractor.processor(text=[text], padding=True, return_tensors="pt")
    75	    return {k: v.to(extractor.model.device) for k, v in inputs.items()}
    76	
    77	
    78	def jaccard(a, b):
    79	    sa, sb = set(a), set(b)
    80	    return len(sa & sb) / max(1, len(sa | sb))
    81	
    82	
    83	def lev_norm(a, b):
    84	    return _levenshtein_token(a, b) / max(1, max(len(a), len(b)))
    85	
    86	
    87	def is_json_valid(text):
    88	    s = text.strip()
    89	    return s.startswith("{") or s.startswith('"')
    90	
    91	
    92	def main():
    93	    p = argparse.ArgumentParser()
    94	    p.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    95	    p.add_argument("--alphas", default=",".join(map(str, DEFAULT_ALPHAS)))
    96	    p.add_argument("--max-new-tokens", type=int, default=15)
    97	    p.add_argument("--limit", type=int, default=2, help="N tasks (smoke=2, full=24)")
    98	    p.add_argument("--tier", default="strong")
    99	    p.add_argument("--min-free-vram-gb", type=float, default=0.0)
   100	    args = p.parse_args()
   101	    layers = [int(x) for x in args.layers.split(",")]
   102	    alphas = [float(x) for x in args.alphas.split(",")]
   103	
   104	    d = np.load(NPZ, allow_pickle=True)
   105	    H = d["hidden_states"]
   106	    ml = d["mode_labels_str"]
   107	
   108	    # Precompute direction per patcher layer: layers[L] hook output ↔ npz[L+1]
   109	    directions = {}
   110	    for L in layers:
   111	        v = H[ml == "phantom_som"][:, L + 1, :].mean(0) - H[ml == "dom"][:, L + 1, :].mean(0)
   112	        directions[L] = torch.tensor(v)
   113	        logger.info(f"layer {L}: npz idx {L+1}, ||v|| = {float(np.linalg.norm(v)):.4f}")
   114	
   115	    manifest = json.loads(MANIFEST.read_text())
   116	    tasks = manifest[args.tier][:args.limit]
   117	    steps = manifest.get("steps", [2, 5])
   118	
   119	    extractor = HiddenStateExtractor(min_free_vram_gb=args.min_free_vram_gb)
   120	    patcher = ActivationPatcher(extractor.model, extractor.processor)
   121	    logger.info(f"model loaded; n_layers={patcher.n_layers}")
   122	    logger.info(f"sweep {len(tasks)} tasks × {len(steps)} steps × {len(layers)} layers × {len(alphas)} α "
   123	                 f"+ 2 baselines = {len(tasks)*len(steps)*(len(layers)*len(alphas)+2)} generations")
   124	
   125	    per_task = []
   126	    for t in tasks:
   127	        tid = int(t["task_id"])
   128	        intent = t["intent"]
   129	        for step in steps:
   130	            obs_path = ARCHIVE / f"classifieds_task_{tid}" / f"step_{step:03d}" / "observation_dom.txt"
   131	            if not obs_path.exists():
   132	                logger.warning(f"missing {obs_path}; skip")
   133	                continue
   134	            obs_text = obs_path.read_text(encoding="utf-8")
   135	            som_marks_text = build_som_marks(obs_text)
   136	            dom_inputs = build_inputs(extractor, intent, "dom", obs_text)
   137	            psom_inputs = build_inputs(extractor, intent, "phantom_som", som_marks_text)
   138	
   139	            dom_gen = patcher.model.generate(**dom_inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
   140	                                              return_dict_in_generate=True, use_cache=True)
   141	            dom_tokens = dom_gen.sequences[0, dom_inputs["input_ids"].shape[1]:].cpu().tolist()
   142	            dom_text = extractor.processor.tokenizer.decode(dom_tokens, skip_special_tokens=True)
   143	
   144	            psom_gen = patcher.model.generate(**psom_inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
   145	                                                return_dict_in_generate=True, use_cache=True)
   146	            psom_tokens = psom_gen.sequences[0, psom_inputs["input_ids"].shape[1]:].cpu().tolist()
   147	            psom_text = extractor.processor.tokenizer.decode(psom_tokens, skip_special_tokens=True)
   148	            logger.info(f"  task={tid} step={step} | dom: {dom_text!r}")
   149	            logger.info(f"  task={tid} step={step} | psom: {psom_text!r}")
   150	
   151	            per_layer = {}
   152	            for L in layers:
   153	                per_alpha = {}
   154	                for alpha in alphas:
   155	                    st_tokens = patcher.steered_generate(
   156	                        layer_idx=L, direction=directions[L], alpha=alpha,
   157	                        max_new_tokens=args.max_new_tokens, **dom_inputs,
   158	                    ).cpu().tolist()
   159	                    st_text = extractor.processor.tokenizer.decode(st_tokens, skip_special_tokens=True)
   160	                    o_dom = jaccard(st_tokens, dom_tokens)
   161	                    o_psom = jaccard(st_tokens, psom_tokens)
   162	                    per_alpha[str(alpha)] = {
   163	                        "steered_text": st_text,
   164	                        "overlap_dom": o_dom, "overlap_psom": o_psom,
   165	                        "lev_dom": lev_norm(st_tokens, dom_tokens),
   166	                        "lev_psom": lev_norm(st_tokens, psom_tokens),
   167	                        "shifted_toward_psom": o_psom > o_dom,
   168	                        "json_valid": is_json_valid(st_text),
   169	                        "first_token_psom_match": (len(st_tokens) > 0 and len(psom_tokens) > 0 and st_tokens[0] == psom_tokens[0]),
   170	                    }
   171	                    logger.info(f"    L{L:02d} α={alpha:>4.1f}: shift={o_psom > o_dom} json={is_json_valid(st_text)} "
   172	                                 f"odom={o_dom:.2f} opsom={o_psom:.2f} → {st_text!r}")
   173	                per_layer[str(L)] = per_alpha
   174	
   175	            per_task.append({
   176	                "task_id": tid, "step": step,
   177	                "dom_text": dom_text, "psom_text": psom_text,
   178	                "per_layer": per_layer,
   179	            })
   180	
   181	            # Incremental save
   182	            OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
   183	            OUT_JSON.write_text(json.dumps({
   184	                "config": {"layers": layers, "alphas": alphas, "tier": args.tier,
   185	                            "max_new_tokens": args.max_new_tokens},
   186	                "results": per_task,
   187	            }, indent=2))
   188	
   189	    # Aggregate per (layer, alpha)
   190	    agg = {}
   191	    for L in layers:
   192	        for alpha in alphas:
   193	            cells = []
   194	            for r in per_task:
   195	                v = r["per_layer"][str(L)][str(alpha)]
   196	                cells.append(v)
   197	            completeness = float(np.mean([c["shifted_toward_psom"] for c in cells]))
   198	            selectivity = float(np.mean([c["json_valid"] for c in cells]))
   199	            # HDMI reliability metric (Khorasani et al. 2026, arXiv:2605.07631):
   200	            # harmonic mean penalizes "shift target but break structure" failure mode
   201	            hmean = 2 * completeness * selectivity / (completeness + selectivity + 1e-9) if (completeness + selectivity) > 0 else 0.0
   202	            agg[f"L{L:02d}_a{alpha}"] = {
   203	                "n": len(cells),
   204	                "mean_overlap_dom": float(np.mean([c["overlap_dom"] for c in cells])),
   205	                "mean_overlap_psom": float(np.mean([c["overlap_psom"] for c in cells])),
   206	                "completeness": completeness,       # shifted_toward_psom rate
   207	                "selectivity": selectivity,           # json_valid rate
   208	                "reliability": hmean,                  # HDMI harmonic mean
   209	                "shifted_rate": completeness,         # alias for backward compat
   210	                "json_valid_rate": selectivity,
   211	                "first_token_psom_match_rate": float(np.mean([c["first_token_psom_match"] for c in cells])),
   212	            }
   213	    final = {
   214	        "config": {"layers": layers, "alphas": alphas, "tier": args.tier,
   215	                    "max_new_tokens": args.max_new_tokens,
   216	                    "direction_norms": {str(L): float(directions[L].norm()) for L in layers}},
   217	        "aggregate": agg, "results": per_task,
   218	    }
   219	    OUT_JSON.write_text(json.dumps(final, indent=2))
   220	    logger.info(f"final → {OUT_JSON}")
   221	
   222	    write_md(final, OUT_MD, layers, alphas)
   223	
   224	
   225	def write_md(d, out, layers, alphas):
   226	    lines = ["# Stage 4 Method 4.4 v2: Layer × α Sweep", ""]
   227	    lines.append(f"**Config**: tier={d['config']['tier']}, n_tasks×steps={len(d['results'])}, max_new_tokens={d['config']['max_new_tokens']}")
   228	    lines.append(f"**Direction norms per layer**: " + ", ".join(f"L{k}={v:.2f}" for k, v in d['config']['direction_norms'].items()))
   229	    lines.append("")
   230	
   231	    lines.append("## HDMI Reliability — harmonic mean (completeness × selectivity)")
   232	    lines.append("")
   233	    lines.append("Following Khorasani et al. 2026 (arXiv:2605.07631): reliability = 2·c·s/(c+s).")
   234	    lines.append("Penalizes \"shift target but break envelope\" failure mode. Higher = better.")
   235	    lines.append("")
   236	    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
   237	    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
   238	    for L in layers:
   239	        row = [f"L{L:02d}"]
   240	        for a in alphas:
   241	            row.append(f"**{d['aggregate'][f'L{L:02d}_a{a}']['reliability']:.2f}**")
   242	        lines.append("| " + " | ".join(row) + " |")
   243	    lines.append("")
   244	
   245	    lines.append("## Completeness (shifted-toward-P-SoM rate: overlap_psom > overlap_dom)")
   246	    lines.append("")
   247	    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
   248	    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
   249	    for L in layers:
   250	        row = [f"L{L:02d}"]
   251	        for a in alphas:
   252	            row.append(f"{d['aggregate'][f'L{L:02d}_a{a}']['completeness']:.0%}")
   253	        lines.append("| " + " | ".join(row) + " |")
   254	    lines.append("")
   255	
   256	    lines.append("## Selectivity (JSON envelope valid rate: steered output still starts with `{`)")
   257	    lines.append("")
   258	    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
   259	    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
   260	    for L in layers:
   261	        row = [f"L{L:02d}"]
   262	        for a in alphas:
   263	            row.append(f"{d['aggregate'][f'L{L:02d}_a{a}']['selectivity']:.0%}")
   264	        lines.append("| " + " | ".join(row) + " |")
   265	    lines.append("")
   266	
   267	    lines.append("## Token overlap to DOM baseline (1.0 = identical, 0 = different)")
   268	    lines.append("")
   269	    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
   270	    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
   271	    for L in layers:
   272	        row = [f"L{L:02d}"]
   273	        for a in alphas:
   274	            row.append(f"{d['aggregate'][f'L{L:02d}_a{a}']['mean_overlap_dom']:.2f}")
   275	        lines.append("| " + " | ".join(row) + " |")
   276	    lines.append("")
   277	
   278	    lines.append("## Token overlap to P-SoM baseline")
   279	    lines.append("")
   280	    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
   281	    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
   282	    for L in layers:
   283	        row = [f"L{L:02d}"]
   284	        for a in alphas:
   285	            row.append(f"{d['aggregate'][f'L{L:02d}_a{a}']['mean_overlap_psom']:.2f}")
   286	        lines.append("| " + " | ".join(row) + " |")
   287	    lines.append("")
   288	
   289	    out.write_text("\n".join(lines) + "\n")
   290	    print(f"summary → {out}")
   291	
   292	
   293	if __name__ == "__main__":
   294	    main()

exec
/bin/bash -lc "nl -ba scripts/analysis/reaggregate_method44_v2_hmean.py | sed -n '1,260p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Re-aggregate Method 4.4 v2 sweep JSON with HDMI H-mean reliability metric.
     3	
     4	Run after `run_stage4_method44_v2_sweep.py` completes. The v2 sweep script
     5	writes a JSON with per-task per-layer per-α raw cells (shifted_toward_psom,
     6	json_valid, overlap_dom, overlap_psom). This script re-aggregates with the
     7	HDMI completeness × selectivity → harmonic mean reliability metric
     8	(Khorasani et al. 2026 arXiv:2605.07631).
     9	
    10	Idempotent — can re-run any time; just reads the JSON and writes the md.
    11	"""
    12	from __future__ import annotations
    13	
    14	import json
    15	import sys
    16	from pathlib import Path
    17	
    18	import numpy as np
    19	
    20	ROOT = Path(__file__).resolve().parents[2]
    21	JSON_PATH = ROOT / "results/mechanistic/stage4_multimode_b1_cls/method44_v2_sweep.json"
    22	OUT_MD = ROOT / "docs/checkpoints/stage4_method44_v2_results.md"
    23	
    24	
    25	def main():
    26	    d = json.loads(JSON_PATH.read_text())
    27	    cfg = d["config"]
    28	    layers = cfg["layers"]
    29	    alphas = cfg["alphas"]
    30	    per_task = d["results"]
    31	
    32	    agg = {}
    33	    for L in layers:
    34	        for alpha in alphas:
    35	            cells = [r["per_layer"][str(L)][str(alpha)] for r in per_task
    36	                      if str(L) in r["per_layer"] and str(alpha) in r["per_layer"][str(L)]]
    37	            if not cells:
    38	                continue
    39	            completeness = float(np.mean([c["shifted_toward_psom"] for c in cells]))
    40	            selectivity = float(np.mean([c["json_valid"] for c in cells]))
    41	            hmean = (2 * completeness * selectivity / (completeness + selectivity + 1e-9)
    42	                      if (completeness + selectivity) > 0 else 0.0)
    43	            agg[f"L{L:02d}_a{alpha}"] = {
    44	                "n": len(cells),
    45	                "mean_overlap_dom": float(np.mean([c["overlap_dom"] for c in cells])),
    46	                "mean_overlap_psom": float(np.mean([c["overlap_psom"] for c in cells])),
    47	                "completeness": completeness,
    48	                "selectivity": selectivity,
    49	                "reliability": hmean,
    50	                "first_token_psom_match_rate": float(np.mean([c["first_token_psom_match"] for c in cells])),
    51	            }
    52	
    53	    # Also save back to JSON so other tools see H-mean
    54	    d["aggregate"] = agg
    55	    JSON_PATH.write_text(json.dumps(d, indent=2))
    56	
    57	    write_md(d, OUT_MD, layers, alphas)
    58	
    59	
    60	def write_md(d, out, layers, alphas):
    61	    cfg = d["config"]
    62	    n_cells = len(d["results"])
    63	    lines = [
    64	        "# Stage 4 Method 4.4 v2: Layer × α Sweep (HDMI reliability framework)",
    65	        "",
    66	        f"**Config**: tier={cfg['tier']}, n_task×step={n_cells}, max_new_tokens={cfg['max_new_tokens']}",
    67	        f"**Direction norms per layer**: " + ", ".join(f"L{k}={v:.2f}" for k, v in cfg.get('direction_norms', {}).items()),
    68	        "",
    69	        "## HDMI Reliability — harmonic mean (completeness × selectivity)",
    70	        "",
    71	        "Following Khorasani et al. 2026 (arXiv:2605.07631): reliability = 2·c·s/(c+s).",
    72	        "Penalizes \"shift target but break envelope\" failure mode. Higher = better.",
    73	        "",
    74	        "| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |",
    75	        "|---|" + "|".join(["---"] * len(alphas)) + "|",
    76	    ]
    77	    for L in layers:
    78	        row = [f"L{L:02d}"]
    79	        for a in alphas:
    80	            v = d["aggregate"].get(f"L{L:02d}_a{a}", {}).get("reliability", 0.0)
    81	            row.append(f"**{v:.2f}**")
    82	        lines.append("| " + " | ".join(row) + " |")
    83	    lines.append("")
    84	
    85	    lines.append("## Completeness (shifted-toward-P-SoM rate)")
    86	    lines.append("")
    87	    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
    88	    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
    89	    for L in layers:
    90	        row = [f"L{L:02d}"]
    91	        for a in alphas:
    92	            v = d["aggregate"].get(f"L{L:02d}_a{a}", {}).get("completeness", 0.0)
    93	            row.append(f"{v:.0%}")
    94	        lines.append("| " + " | ".join(row) + " |")
    95	    lines.append("")
    96	
    97	    lines.append("## Selectivity (JSON envelope valid rate)")
    98	    lines.append("")
    99	    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
   100	    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
   101	    for L in layers:
   102	        row = [f"L{L:02d}"]
   103	        for a in alphas:
   104	            v = d["aggregate"].get(f"L{L:02d}_a{a}", {}).get("selectivity", 0.0)
   105	            row.append(f"{v:.0%}")
   106	        lines.append("| " + " | ".join(row) + " |")
   107	    lines.append("")
   108	
   109	    lines.append("## Token overlap to DOM baseline")
   110	    lines.append("")
   111	    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
   112	    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
   113	    for L in layers:
   114	        row = [f"L{L:02d}"]
   115	        for a in alphas:
   116	            v = d["aggregate"].get(f"L{L:02d}_a{a}", {}).get("mean_overlap_dom", 0.0)
   117	            row.append(f"{v:.2f}")
   118	        lines.append("| " + " | ".join(row) + " |")
   119	    lines.append("")
   120	
   121	    lines.append("## Token overlap to P-SoM baseline")
   122	    lines.append("")
   123	    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
   124	    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
   125	    for L in layers:
   126	        row = [f"L{L:02d}"]
   127	        for a in alphas:
   128	            v = d["aggregate"].get(f"L{L:02d}_a{a}", {}).get("mean_overlap_psom", 0.0)
   129	            row.append(f"{v:.2f}")
   130	        lines.append("| " + " | ".join(row) + " |")
   131	    lines.append("")
   132	
   133	    # Peak cell summary
   134	    cells_with_rel = [(k, v["reliability"]) for k, v in d["aggregate"].items()]
   135	    cells_with_rel.sort(key=lambda x: -x[1])
   136	    lines.append("## Top-5 cells by reliability")
   137	    lines.append("")
   138	    lines.append("| Rank | (Layer, α) | Reliability | Completeness | Selectivity |")
   139	    lines.append("|---|---|---|---|---|")
   140	    for i, (k, _) in enumerate(cells_with_rel[:5], 1):
   141	        v = d["aggregate"][k]
   142	        lines.append(f"| {i} | {k} | {v['reliability']:.2f} | {v['completeness']:.0%} | {v['selectivity']:.0%} |")
   143	    lines.append("")
   144	
   145	    out.write_text("\n".join(lines) + "\n")
   146	    print(f"summary → {out}")
   147	
   148	
   149	if __name__ == "__main__":
   150	    main()

exec
/bin/bash -lc "nl -ba docs/checkpoints/mechanism/plan.md | sed -n '1,260p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	---
     2	name: mechanism plan
     3	description: Full mechanism workspace — theory, lit anchor stack, methods, identification protocol, current findings, open questions, advisor sync, roadmap. Specialized companion to paper_planning §2; not a duplicate.
     4	type: workspace_plan
     5	last_substantive_update: 2026-05-12
     6	---
     7	
     8	# Mechanism Plan — paper §5
     9	
    10	## 1. Theory framework (1-screen summary, paper_planning §2 is canonical)
    11	
    12	### 1.1 Zoom 1-4 hierarchy
    13	
    14	| Zoom | Level | What our paper claims |
    15	|---|---|---|
    16	| **1** | Architectural | Phantom routing space = "skip annotated image" boundary contains 3 arms (P-text / P-prompt / P-SoM) sharing 4-fold drop-in property |
    17	| **2** | Behavioral (axis effects) | Axis 1 (text payload: AXTree vs [SOM_MARKS]) is PRIMARY; Axis 2 (prompt: SoM-prompt vs DOM-prompt) is secondary; Axis 3 (image presence: in vs out) is gating |
    18	| **3** | Named phenomena (lit-anchored) | Mirage Effect (Asadi 2026) / Scaffold Effect (Vu&Balloccu 2026) / Cross-modal flow (Kaduri) / Prompt-format sensitivity (Sclar 2024) |
    19	| **4** | Model-internal | L17 mid-layer is BOTH discrimination locus (probe AUROC 1.0) AND causally active planning site (Stage 2/3 patching + Method 4.4 v2 reliability) |
    20	
    21	### 1.2 Three-axis hierarchy quantified (Method 4.2 PCA cosine gap, Qwen3-VL-4B B1 cls)
    22	
    23	| Axis | Peak cosine gap | Peak layer | Magnitude ratio |
    24	|---|---|---|---|
    25	| Image-axis (vs SoM / Vision) | 0.06 | L4–L17 | **10×** |
    26	| Text-axis ([SOM_MARKS] vs AXTree) | 0.025 | L23 | **4×** |
    27	| Prompt-axis (SoM-prompt vs DOM-prompt alone) | 0.007 | L36 | **1×** |
    28	
    29	→ Mechanism magnitude image >> text > prompt. Validates `project_phantom_space_axes_format_not_information.md` memory: P-SoM closest mode at every layer is **P-text** (text-axis sibling, L17 cosine 0.0028 vs P-SoM↔SoM 0.0412 = 14.7× more distant).
    30	
    31	### 1.3 Image-axis peak-layer dichotomy (Mirage mechanism signature)
    32	
    33	Method 4.2 reveals image-axis cosine-gap peak shifts based on text format of the no-image side. Clean dichotomy, zero overlap across 8 image-axis pairs:
    34	
    35	| No-image side text | Peak layer | Pairs |
    36	|---|---|---|
    37	| AXTree (hierarchical) | **L04** | DOM↔Vision, DOM↔SoM, P-prompt↔Vision, P-prompt↔SoM |
    38	| [SOM_MARKS] / flat | **L17–L36** | P-text↔Vision, P-text↔SoM, P-SoM↔Vision, P-SoM↔SoM |
    39	
    40	### 1.4 H1 test confirms broader: flat-list (not just indexed) triggers shortcut (2026-05-12)
    41	
    42	Format variation extraction (Myriad job 352998, `stage4_format_variation_b1_cls`, 450 hidden states = 45 task-step × 10 modes). For each text format V, compute image-axis cosine gap V↔som per layer; peak layer reveals shortcut activation:
    43	
    44	| Format | Peak layer | Verdict |
    45	|---|---|---|
    46	| **AXTree hierarchical (DOM)** | **L04** | **SOLE format defeating shortcut** |
    47	| `"a, b, c, ..."` plain sentence | L17 | mid-level trigger |
    48	| `[N] role 'label'` (SoM standard) | L36 | strong trigger |
    49	| `@N label` (Browser Use) | L36 | strong trigger |
    50	| `id_N: label` (AppAgent) | L36 | strong trigger |
    51	| `[BN:r:l]` (Tarsier) | L36 | strong trigger |
    52	| `N. label` (numbered) | L36 | strong trigger |
    53	| `<el_N>label</el_N>` (XML) | L36 | strong trigger |
    54	| `#hash label` (control: no integer) | L36 | **still triggers!** |
    55	
    56	**Refined H1 verdict**: trigger is **flat element listing**, not "indexed list pattern". Even integer-free hash IDs and pure-sentence variants engage the shortcut. AXTree hierarchical depth is the **unique format** that defeats shortcut activation.
    57	
    58	Paper §5 implication: SoM-family web agents (Browser Use, AppAgent, Tarsier, OmniParser, etc.) **all** implicitly exploit the same flat-list-element-grounding shortcut from VLM training distribution. P79 phantom routing space makes this systematic and routes accordingly.
    59	
    60	## 2. Literature anchor stack (5 anchors, all 2026-05-08 except Sclar 2024)
    61	
    62	| Anchor | Role | bib key | What it gives our paper §5 |
    63	|---|---|---|---|
    64	| **Wu et al. 2026** (UCL lab, our advisors) | Method backbone | `wu2026toolcalling` (2605.07990) | Mean-difference activation steering at second-to-last layer, 77–100% switch on tool selection (93–100% at 4B+). Our Method 4.2/4.4 port to multimodal Qwen3-VL-4B web agent |
    65	| **Ma & Rui 2026** | Probe-vs-causal vocabulary | `maRui2026planning` (2605.07984) | "Planning-compatible representation" vs "causally active planning site". Qwen3-family pattern: probe works, causal patching weak (1% rhyme newline causal vs Gemma 67%). Our Method 4.4 v2 50% reliability is consistent with this family pattern |
    66	| **HDMI / Khorasani et al. 2026** | Alt method + evaluation metric | `khorasani2026hdmi` (2605.07631) | Probe-free gradient-based steering. Critically: **completeness × selectivity → harmonic mean reliability** — what our Method 4.4 v2 reports (not raw shift rate) |
    67	| **Lin & Liu 2026 Position paper** | Methodology protocol | `linLiu2026disclosure` (2605.08012) | 5-step identification disclosure norm: state claim / name strategy / enumerate assumptions / stress-test / separate validation. Paper §5 adopts as identification subsection structure |
    68	| **Peale et al. 2026** | §6 routing theory | `peale2026flexibleRouting` (2605.07805) | Uncertainty decomposition (reducible + irreducible) with regret bound. Paper §6 theoretical anchor; 4-fold drop-in maps onto predict/route/abstain trichotomy |
    69	
    70	## 3. Methods (Stage 4 + planned)
    71	
    72	### 3.0 Curation pipeline (Stage 0 setup, shared by Stages 1-4)
    73	
    74	`scripts/mechanistic/curate_mirage_tasks.py` builds a strong-tier task manifest from a B1 production run:
    75	- Filter to tasks where Mirage Effect is well-defined: agent succeeds in `som` mode but fails in `dom` mode (real image-feature dependence, not random failure)
    76	- Pick 24 cls + 24 reddit "strong-tier" tasks (clean Mirage); separate 15 cls reverse-tier (DOM > SoM, selection-bias defense)
    77	- Archive raw observations (AXTree, screenshots, SoM-marks) per (task, step) → `archive_subset_b1_<site>/<site>_task_<tid>/step_<step>/` for offline patching/extraction (Stages 2-4 read from this archive, no live env required)
    78	
    79	Outputs:
    80	- `results/mechanistic/curate_mirage_b1_classifieds/manifest.json` — cls strong/reverse tier task list
    81	- `results/mechanistic/curate_mirage_b1_reddit/manifest.json` — reddit strong tier
    82	- `results/mechanistic/archive_subset_b1_cls/` (17 MB, 144 files, 24 tasks × 6 steps)
    83	- `results/mechanistic/archive_subset_b1_reddit/` (35 MB, 356 files, 24 tasks × ~15 steps)
    84	
    85	### 3.1 Method 4.2 — PCA cosine gap (DONE)
    86	
    87	`scripts/analysis/stage4_pca_cosine_gap.py` + `stage4_robustness.py`. Three metrics per (mode_pair, layer):
    88	- A. Cosine gap = 1 − cos(mean_A, mean_B)
    89	- B. AUROC via (mean_A − mean_B) projection
    90	- C. Per-(mode, layer) PCA top-10 variance explained
    91	
    92	**5/5 robustness pass**:
    93	- Test A label perm: 9.8σ above noise (real 1.000 vs perm 0.629)
    94	- Test B per-task: 100% of 24 tasks positive
    95	- Test C per-step (step 2 vs step 5): invariant
    96	- Test D silhouette ≥ 0.5 at L23 (strong clustering)
    97	- Test E bootstrap 95% CI tight (4-15% of mean)
    98	
    99	### 3.2 Method 4.4 — mean-diff activation steering (v2 in flight)
   100	
   101	`scripts/mechanistic/run_stage4_method44_v2_sweep.py`. Layer × α sweep:
   102	- Layers: [11, 17, 23, 29, 33, 34] — covers mid (Stage 2 disruption locus) → late (Wu et al. second-to-last)
   103	- α: [1, 2, 5, 10, 20] — Wu et al. typical α=1, our diag found ≥5 needed for multi-step JSON
   104	- 24 cls strong-tier tasks × 2 steps × 30 cells = 1440 generations (~2h)
   105	
   106	**HDMI reliability metric**: completeness × selectivity → harmonic mean (Khorasani et al. 2026):
   107	- Completeness = % tasks where overlap_psom > overlap_dom
   108	- Selectivity = % tasks where JSON envelope preserved (starts with `{`)
   109	- Reliability = 2 · c · s / (c + s)
   110	
   111	**Current smoke (8/48 cells)**: L17 α=5 = **0.44** sweet spot (29% shift + 100% JSON valid). L33 α=10 = 0.23 (57% shift but JSON breaks).
   112	
   113	### 3.3 Method 4.5 — LA-HDMI / SAE (future work, paper §8)
   114	
   115	Two alternative paths:
   116	- **LA-HDMI**: probe-free gradient steering (Khorasani 2026 method). Per-input optimization replaces fixed mean-diff direction. May overcome Qwen3-family causal patching weakness
   117	- **SAE feature steering** (Zekun-recommended in advisor recording, paper_planning §108): train SAE on Qwen3-VL-4B residual stream (1-2 week cost, no public SAE exists), find mirage/format feature, steer directly. Differentiates from Wu et al. mean-diff path
   118	
   119	Decision pending Method 4.4 v2 full sweep + Zekun sync.
   120	
   121	## 4. Identification protocol (Lin & Liu 2026 disclosure norm)
   122	
   123	Following Lin & Liu Position paper, paper §5 must explicitly state:
   124	
   125	### 4.1 Causal claim
   126	
   127	> Mid-layer L17 hidden state at last-token position is the causally active planning site for phantom routing space mode selection in Qwen3-VL-4B web agents.
   128	
   129	### 4.2 Identification strategy
   130	
   131	Triangulation of 3 evidence types:
   132	1. **Probe-level** (Method 4.2 PCA cosine gap, AUROC 1.000 across 540 tests)
   133	2. **Replacement patching** (Stage 2/3 Cell A-H, L17 disruption peak, 8/8 Holm-sig)
   134	3. **Additive steering** (Method 4.4 v2, mid-layer L17 α=5 H-mean reliability 0.44)
   135	
   136	### 4.3 Identification assumptions
   137	
   138	| # | Assumption | Stress-test |
   139	|---|---|---|
   140	| A1 | L17 last-token hidden state mediates action selection (not earlier obs token positions) | Stage 2/3 swept all layers, L17 is peak |
   141	| A2 | Mean-difference direction approximates causal axis (Wu et al. hypothesis) | Method 4.4 v2 H-mean 0.44 partial — assumption holds weakly; LA-HDMI would test |
   142	| A3 | 24 strong-tier tasks generalize to broader VWA distribution | Stage 4 robustness Test B: 100% per-task positive, but tier-selection bias possible. Reverse-tier 15 tasks pending |
   143	| A4 | Qwen3-VL-4B mechanism transfers to other VLM sizes/architectures | Not tested. Wu et al. shows family generality on tool-only; multimodal+multi-step unknown |
   144	| A5 | Replacement patching faithfully simulates "natural" model read of the representation | Cell E random-injection control rules out non-specific disruption — content-specific causation confirmed |
   145	
   146	### 4.4 Stress-test result
   147	
   148	Cell E random-injection control: replacing source hidden with Gaussian noise (same μ, σ) yields **null L17 disruption effect**. Confirms our patching effect is source-content-specific, not noise-driven. Most directly stresses A5.
   149	
   150	### 4.5 Validation ≠ identification (Lin & Liu §5)
   151	
   152	- Method 4.2 AUROC 1.000 = validation (decodability)
   153	- Stage 2/3 + Method 4.4 v2 = identification attempts (causal use)
   154	- These are reported SEPARATELY in paper §5; reviewer should not conflate
   155	
   156	## 5. Current findings dashboard
   157	
   158	### 5.1 Stage 4 Method 4.2 (Qwen3-VL-4B B1 cls, 288 examples × 37 layers)
   159	
   160	| Pair @L17 | Cosine gap | 95% CI | AUROC |
   161	|---|---|---|---|
   162	| P-SoM ↔ P-text | 0.0028 | [0.0027, 0.0029] | 1.000 |
   163	| DOM ↔ P-prompt | 0.0013 | [0.0012, 0.0014] | 1.000 |
   164	| P-SoM ↔ SoM | 0.0413 | [0.0403, 0.0422] | 1.000 |
   165	| DOM ↔ Vision | 0.0547 | [0.0531, 0.0563] | 1.000 |
   166	
   167	### 5.2 Stage 2/3 patching disruption (14 cells, B1 cls + reddit)
   168	
   169	**Stage 2 — P-SoM ↔ SoM patching (10 cells):**
   170	
   171	| Cell | Site | Direction | L17 Δoverlap | Holm-sig |
   172	|---|---|---|---|---|
   173	| A | cls | SoM→P-SoM forward | -0.32 | ✓ |
   174	| B | cls | P-SoM→SoM reverse | -0.16 | ✓ |
   175	| C | cls | 2x2 reverse-tier fwd | -0.02 | ✗ (null) |
   176	| D | cls | 2x2 strong-tier rev | -0.18 | ✓ |
   177	| E | cls | random injection | -0.03 (uniform) | ✓ (negative control) |
   178	| F | reddit | SoM→P-SoM forward | -0.21 | ✓ |
   179	| G | reddit | P-SoM→SoM reverse | -0.18 | ✓ |
   180	| Cr/Dr | reddit 2x2 | both directions | -0.15 to -0.18 | ✓ |
   181	| Er | reddit | random injection | ~0 (uniform) | ✓ |
   182	
   183	**Stage 3 — 2x2 mechanism additivity test (SoM → {DOM, P-text, P-prompt}, cls + reddit):**
   184	
   185	| Cell | Site | Source→Target | Best-L overlap→src | L17 Δoverlap→tgt | Path |
   186	|---|---|---|---|---|---|
   187	| H-d-cls | cls | SoM → DOM | L10 (0.192) | -0.33 | `stage3_cellhd_cls_fwd_dom_myriad/` |
   188	| H-p-cls | cls | SoM → P-prompt | L27 (0.219) | -0.22 | `stage3_cellhp_cls_fwd_prompt_myriad/` |
   189	| H-t-cls | cls | SoM → P-text | L28 (0.164) | -0.25 | `stage3_cellht_cls_fwd_text_myriad/` |
   190	| H-p-red | reddit | SoM → P-prompt | L20 (0.209) | -0.19 | `stage3_cellhp_red_fwd_prompt_myriad/` |
   191	| H-t-red | reddit | SoM → P-text | L01 (0.194) | -0.24 | `stage3_cellht_red_fwd_text_myriad/` |
   192	| **H-d-red** | reddit | SoM → DOM | L28 (0.204) | **L11 -0.33 / L17 -0.26** | `stage3_cellhd_red_fwd_dom_myriad/` ✅ done 2026-05-12 19:57 |
   193	
   194	**Stage 3 interpretation (6/6 cells complete 2026-05-12)**: All forward SoM→{no-image-arm} patching cells show mid-layer L11-L17 disruption -0.19 to -0.33 Δoverlap→tgt. Magnitude > random injection control (Cell E -0.03) at all 6. **Mechanism additivity confirmed**: image-feature axis is shared substrate across DOM / P-text / P-prompt arms — single SoM→{any-no-image-arm} patching displaces target prediction toward source. Cross-site cls + reddit both replicate (paper §5 universal mid-layer fusion locus); reddit fusion locus slightly earlier (L11 vs cls L17), magnitude identical.
   195	
   196	Stage 3 cross-site DOM-axis additivity table (paired-test Δoverlap-to-target from `patching_continuation_results.json`):
   197	
   198	| Site | SoM→DOM | SoM→P-text | SoM→P-prompt | best-L Δ range |
   199	|---|---|---|---|---|
   200	| cls | H-d-cls L17 -0.309 / L18 **-0.352** best | H-t-cls L17 -0.255 / L12 **-0.270** best | H-p-cls L17 -0.223 / L13 **-0.273** best | [-0.273, -0.352] |
   201	| reddit | H-d-red L11 -0.335 / L17 -0.255 / L14 **-0.338** best | H-t-red L11 -0.244 / L17 -0.236 / L15 **-0.330** best | H-p-red L11 -0.233 / L17 -0.191 / L14 **-0.322** best | [-0.322, -0.338] |
   202	
   203	All 6 cells best layer 落在 **L12-L18 mid-layer 窗口** (tight 7-layer band), Δ range [-0.27, -0.35]. Cross-site / cross-arm 一致, mid-layer fusion locus 不是 single layer index 而是稳定窗口.
   204	
   205	### 5.3 Stage 4 Method 4.4 v2 (FULL 45/48 cells, finalized 2026-05-11 22:00)
   206	
   207	H-mean reliability (HDMI framework) per (layer, α). **L17 α=5 smoke claim REFUTED by full sweep**; actual sweet spot at L33 α=10:
   208	
   209	| Layer \ α | α=1 | α=2 | α=5 | α=10 | α=20 |
   210	|---|---|---|---|---|---|
   211	| L11 | 0.04 | 0.09 | 0.20 | 0.12 | 0.12 |
   212	| L17 | 0.00 | 0.12 | **0.16** (was 0.44 smoke) | 0.12 | 0.09 |
   213	| L23 | 0.00 | 0.09 | 0.09 | 0.16 | 0.00 |
   214	| L29 | 0.00 | 0.00 | 0.00 | 0.04 | 0.04 |
   215	| **L33** | 0.04 | 0.00 | 0.00 | **0.33** ⭐ | 0.00 |
   216	| L34 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
   217	
   218	**Layer-specialization** (probe-causal dissociation):
   219	- Mid-layer (L11-L23): **selectivity 100%** at all α (JSON envelope preserved), but completeness 0-11% (modest shift)
   220	- Late-layer (L33): completeness 38% (highest), but selectivity drops to 29% (over-steers JSON)
   221	- L33 α=10 H-mean 0.33 = max reliability cell
   222	
   223	**Smoke variance lesson** (笔记 §126 + §127): 4-cell smoke H-mean 0.44 on L17 was statistical artifact (1/4 hit = inflated rate). Full 45-cell H-mean 0.16 is true rate. Future mechanism findings require n ≥ 30 cells before "sweet spot" claims.
   224	
   225	### 5.4 Image-axis peak-layer dichotomy (Method 4.2, 8 pairs)
   226	
   227	`docs/checkpoints/mechanism/results/layer_axis_emergence.md`. AXTree-no-image side → L04 peak (4/4); [SOM_MARKS]-no-image side → L17–L36 peak (4/4). Zero overlap. Mirage Effect mechanism signature.
   228	
   229	### 5.5 H1 test: flat-list format variation (Method 4.2 extension, 2026-05-12)
   230	
   231	`docs/checkpoints/mechanism/results/format_variation_h1_test.md`. 8 industry-relevant text formats + 2 controls. AXTree hierarchical (DOM) is **unique format** preserving L04 image-axis peak; all 8 flat-list variants (SoM standard, Browser Use @, AppAgent id_, Tarsier typed, plain numbered, XML tagged, hash-ID control, plain-sentence control) shift peak to L17–L36. Trigger is flat element listing, not specific token pattern.
   232	
   233	## 6. Open questions (paper-grade gaps)
   234	
   235	| Q | Status | Next action |
   236	|---|---|---|
   237	| ✅ Method 4.4 v2 full 48-cell sweep — sweet spot stable? | **Closed 2026-05-11 22:00**: L17 α=5 smoke 0.44 → full 0.16 (smoke variance artifact). **Real sweet spot L33 α=10 H-mean 0.33** | — |
   238	| ✅ H1 test: do all flat-list formats trigger shortcut? | **Closed 2026-05-12 00:00**: YES, including hash-ID + plain-sentence controls. AXTree-DOM is sole defeating format | — |
   239	| Reverse-tier 15 tasks vs strong-tier 24 — does L33 + H1 finding generalize beyond selection bias? | Med-High | qsub Stage 4 multimode + format variation with --tier reverse |
   240	| ✅ Cross-site Method 4.2 — does cls finding replicate on reddit? | **Closed 2026-05-12 16:30**: P-SoM↔DOM L17=0.0098 + P-SoM↔SoM L17=0.0423, AUROC 1.0 → Mirage signature replicated. See §7.3.1 | — |
   241	| ✅ Stage 3 reddit 2x2 closure — H-d-red | **Closed 2026-05-12 19:57** (Myriad 358831). L11 Δ=-0.33 / L17 Δ=-0.26. Cross-site additivity confirmed — see §5.2 Stage 3 table | — |
   242	| LA-HDMI vs mean-diff — does gradient steering beat 0.33 ceiling? | Med | Pending Zekun reply + attribution decision |
   243	| SAE feature steering feasibility — is 1-2 week self-training Qwen3-VL-4B SAE worth it? | Low-Med | Depends on Zekun reply + paper §8 prose direction |
   244	| B0 (proxy API) — paper §5 Qwen-specific or generalizable? | Low | Cannot test on B0; cite Wu et al. cross-family generality as proxy |
   245	| AXTree-defeats-shortcut mechanism — *why* hierarchy beats flat? Cross-modal attention specific to indentation tokens? | High (paper §5 supplement) | Activation patching at L4 with hierarchical-text vs flat-text → see which attention heads pre-disrupt image embedding |
   246	
   247	## 7. Advisor sync state — Zekun (Wu et al. 2026 first author = lab member)
   248	
   249	### 7.1 Timeline confirmed (not scoop)
   250	
   251	- 2026-04-09 笔记 §19: I first grok the paper (then "Anonymous 2026 ACL"), record cosine gap method + L23+ steering 80-93%
   252	- 2026-05-01 笔记 §108.19: upgraded to Zoom 4 anchor stack
   253	- 2026-05-02 commit `6662b91`: anchored into paper_planning §2 + paper.bib placeholder
   254	- 2026-05-09 advisor recording: Zekun explicitly recommended "SAE feature steering — 前所未有 inference time steering, 单独发 paper" — directed me to differentiating path
   255	- 2026-05-11: arxiv landed publicly; identity confirmed as lab paper
   256	
   257	**Net**: Zekun explicitly invited mechanism extension. Method 4.4 multimodal port is on his recommendation; SAE Method 4.5 is his next-step suggestion.
   258	
   259	### 7.2 Message draft (v3, paste-ready 2026-05-12)
   260	

exec
/bin/bash -lc "nl -ba docs/checkpoints/mechanism/plan.md | sed -n '261,520p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   261	Updated after v2 full sweep + H1 test. Key revisions from §125.10 draft:
   262	- ❌ Removed: "L17 α=5 H-mean 0.44 mid-layer sweet spot" (smoke variance artifact, full data refutes)
   263	- ✓ Added: **L33 α=10 H-mean 0.33** = matches your second-to-last-layer choice; multi-step JSON selectivity drop explains 38% vs your 93% gap
   264	- ✓ Added: H1 test finding — flat-list format universally triggers shortcut (8/8 variants), only AXTree hierarchical defeats; implication for industry SoM-family agents
   265	- ✓ Three asks: (a) attribution co-author vs cite + independent; (b) your ablation on mid- vs late-layer (we see selectivity tradeoff); (c) SAE direction priority given mean-diff ceiling
   266	
   267	Final message (Chinese, casual WeChat tone):
   268	
   269	> Zekun 早, 你那篇 Tool Calling 上 arxiv 我看了, 恭喜! 我前几天按你说的开始 mechanism work, 跑出来一些东西想跟你 sync 一下, 顺便问几个方向问题。
   270	>
   271	> # Context
   272	> P79 paper 在做 VisualWebArena 的 phantom routing space — agent 6 种 obs mode (DOM 文本/SoM 标注图/Vision 裸图 + 3 个 phantom 变体). 模型 Qwen3-VL-4B, 你 Qwen 3 4B 同 base LM。
   273	>
   274	> # 1. Method 4.2 PCA cosine gap port 到 6 modes
   275	> 24 cls strong-tier × 2 step × 6 mode = 288 hidden states, 37 layer × 2560 dim。全 540 pair × layer AUROC = 1.000 (perm baseline 0.629, real 9.8σ above). 你方法在 multimodal Qwen 上 readable transfer 干净。
   276	>
   277	> # 2. Method 4.4 mean-diff steering (HDMI metric)
   278	> 45 task-step × 6 layer × 5 α full sweep. 用 HDMI completeness×selectivity → H-mean 评估:
   279	>
   280	>   - **L33 α=10 H-mean 0.33** (sweet spot, c=38% s=29%) ← matches 你 paper second-to-last-layer
   281	>   - Mid-layer (L11-L23) selectivity 100% 但 completeness 0-11% — readable but not effectively steerable
   282	>   - 你 paper Qwen 3 4B 93% switch vs 我 38% — 我猜原因是 multi-step JSON gen 的 selectivity 是真约束 (你 single-token tool decision selectivity 自动 1.0)
   283	>
   284	> # 3. H1 test: flat-list format variation (Myriad)
   285	> 测了 8 个 industry-relevant text format (Browser Use @, AppAgent id_, Tarsier typed, numbered, XML, hash-ID, plain-sentence + SoM baseline) vs AXTree-DOM:
   286	>
   287	>   - 全 8 flat variants peak L17/L36 (= 都触发 shortcut)
   288	>   - **AXTree hierarchical 是唯一保留 L04 peak 的 format**
   289	>   - 包括 hash-ID (no integer) + plain-sentence (no list) 都触发
   290	>   - = SoM-family agents 全 implicit exploit 同一 VLM shortcut, AXTree 是 sole exception
   291	>
   292	> # 三个 ask
   293	> (1) Attribution: paper §5 mechanism 这块 — cite 你 + 我独立 framing 比较合理, 还是 co-author 一篇 multimodal extension 比较好? 都 OK, 想听你意见。
   294	>
   295	> (2) 你 ablation 里有跑过 mid- vs late-layer 对比吗? 我 mid-layer selectivity 100% 但 shift 弱, late-layer shift 强但 envelope 破 — 不知道你 tool calling 上是不是也有这种 tradeoff。
   296	>
   297	> (3) 你之前 advisor 录音里建议 SAE feature steering, 我也写进 future work 了。现在 mean-diff ceiling ~0.33, 是不是 SAE 这条路更有差异化? Qwen3-VL-4B SAE 没公开, 自训成本 1-2 周, 你觉得值得 commit GPU 吗?
   298	>
   299	> 不急, 你忙完回我就行. paper 写得真漂亮.
   300	
   301	### 7.3 H1 generalization in-flight (2026-05-12 night)
   302	
   303	After per-task fragility revealed 11% strict dichotomy (aggregate statistical, not deterministic), launched 5-priority defense matrix to triangulate H1 across **(tier × site × family/size)**:
   304	
   305	| Pri | Test | Where | Status @ 06:25 | Sentinel |
   306	|---|---|---|---|---|
   307	| **P1** | Per-task fragility audit (24 cls strong) | DGX | ✅ done | `results/h1_per_task_fragility.md` |
   308	| **P2** | Cross-family (Phi-3.5-Vision 4.2B) | DGX | ❌ deferred (HF cas-bridge throttling) | `stage4_h1_phi35_cls/pilot_summary.md` |
   309	| **P3** | Within-family bigger (Qwen2-VL-7B, H1' capacity test) | DGX | ❌ deferred (HF cas-bridge throttling) | `stage4_h1_qwen2vl7b_cls/pilot_summary.md` |
   310	| **P4** | cls reverse-tier (selection-bias defense) | Myriad 353763 | ✅ **done 18:50:46** — shape (260, 37, 2560), 10 modes, 46 MB pulled. Same pattern as cls strong-tier (L36 marks-like + L04 dom). Selection-bias defended | `stage4_format_variation_b1_cls_reverse/hidden_states.npz` |
   311	| **P5a** | reddit format variation (cross-site H1) | Myriad **354382** (3rd attempt) | ✅ **done 08:09:38** — shape (430, 37, 2560), 10 modes, 76 MB pulled | `stage4_format_variation_b1_reddit/hidden_states.npz` |
   312	| **P5b** | reddit Method 4.2 multimode (cross-site Mirage) | Myriad 353890 | ✅ **done 07:31:14** — 288 examples, 6 modes, 51 MB pulled | `stage4_multimode_b1_reddit/hidden_states.npz` |
   313	
   314	**P5a bug history** (3 attempts):
   315	1. Myriad 353764 (00:48) — `no hidden states extracted` after 105 task skips. Root cause: hardcoded `classifieds_task_{tid}` prefix in `run_stage4_format_variation_extract.py:177`, archive uses `reddit_task_*`
   316	2. Myriad 353889 (06:26) — same failure, same root cause
   317	3. Myriad **354382** (07:26) — fixed via commit 3d41953 (add `--site reddit` arg, default classifieds for backcompat)
   318	
   319	**P2/P3 deferred** (2026-05-12 00:31 → 06:30, 3 attempts each):
   320	- `snapshot_download` `thread_map` 8-worker concurrent download hits cas-bridge throttling/timeout
   321	- Each attempt: get `HTTP 206 Partial Content` then concurrent.futures `result_iterator` raises (underlying worker exception masked)
   322	- Cleanup 4×2.3G incomplete blobs to reclaim disk
   323	- **Recovery plan**: tomorrow morning, single-thread CLI:
   324	  ```bash
   325	  HF_HUB_DOWNLOAD_TIMEOUT=600 huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --max-workers 1
   326	  HF_HUB_DOWNLOAD_TIMEOUT=600 huggingface-cli download microsoft/Phi-3.5-vision-instruct --max-workers 1
   327	  ```
   328	- Paper §5 generalization claim still defensible via P4 (selection-bias) + P5a/P5b (cross-site). P2/P3 are nice-to-have (family/size triangulation), not paper-critical.
   329	
   330	**Expected verdict matrix** (most paper-grade interesting):
   331	- P3 7B per-task variability < 4B per-task variability → H1' capacity-limit partially confirmed (training-distribution still creates shortcut, but consistency increases with size)
   332	- P2 cross-family dichotomy holds → H1 is cross-family universal training prior
   333	- P4 reverse-tier holds → not tier-selection-bias
   334	- P5a reddit holds → cross-site universal
   335	
   336	### 7.3.0 Exp 1 axis-2 layer profile (2026-05-12 21:00 — three-axis hierarchy)
   337	
   338	`axis2_layer_profile.md` + `fig_axis2_prompt_layer_profile.png`. Re-examine residual stream geometry per axis-isolated pair, full 37-layer cosine curves on `stage4_multimode_b1_{cls,reddit}` (288 ex each).
   339	
   340	Cls site peak layers + magnitudes:
   341	
   342	| Pair | Group | L17 | L23 | L36 | Peak L | Peak gap |
   343	|---|---|---:|---:|---:|---:|---:|
   344	| P-SoM↔SoM (image-axis ref) | axis-3 | 0.0412 | 0.0400 | 0.0411 | **L17** | 0.0412 |
   345	| DOM↔P-text (text fmt) | axis-1 | 0.0120 | 0.0254 | 0.0201 | **L23** | 0.0254 |
   346	| P-prompt↔P-SoM (text fmt) | axis-1 | 0.0113 | 0.0292 | 0.0201 | **L23** | 0.0292 |
   347	| P-text↔P-SoM (prompt fam, flat) | axis-2 | 0.0028 | **0.0114** | 0.0089 | L23 | 0.0114 |
   348	| DOM↔P-prompt (prompt fam, hier) | axis-2 | 0.0013 | 0.0050 | 0.0067 | L36 | 0.0067 |
   349	
   350	Reddit cross-site replicates: P-text↔P-SoM L23 = 0.0098 (vs cls 0.0114), same rank-order, same peak layer.
   351	
   352	**Three regularities**:
   353	1. **Distinct peak layers**: image L17 (fast sharp), text-format L23 (slower late-mid), prompt-family L23 (same timing as text-format on flat-text)
   354	2. **Distinct magnitudes**: image ~0.04, text-format ~0.03, prompt-family ~0.01 — 4:3:1 ratio
   355	3. **Cross-site rank stable**: reddit identical pattern
   356	
   357	**Reframe**: Axis-2 prompt-family is NOT null at residual stream. It's 3-4x weaker than axis-1 + peaks at L23 not L17. Method 4.2 plan §5.1 L17 snapshot 错失它. New paper §5 framing: layered three-axis hierarchy, image-axis dominant at L17 Mirage locus, text-format + prompt-family late-mid build at L23 parallel.
   358	
   359	→ Paper §5.7 重写为 "Layered Three-Axis Mechanism Hierarchy" (commit pending).
   360	
   361	### 7.3.0b Axis-2 per-task fragility check (2026-05-12 21:50 — /stress W2 defuse)
   362	
   363	`axis2_per_task_fragility.md` + `fig_axis2_per_task_fragility.png`. /stress reviewer 第一次 invocation W2 attack: 怀疑 axis-2 cosine 0.0114 mean 由 2-3 outlier 主导, 类比 h1_per_task_fragility 11% strict per-task. Defuse 实验:
   364	
   365	| Pair | Site | Mean | Median | IQR | % > 0.010 |
   366	|---|---|---|---|---|---|
   367	| **Axis-2 flat (P-text↔P-SoM)** | cls | 0.0132 | 0.0131 | [0.012, 0.014] | **100%** |
   368	| **Axis-2 flat (P-text↔P-SoM)** | reddit | 0.0121 | 0.0120 | [0.011, 0.013] | **100%** |
   369	| Axis-1 ref (DOM↔P-text) | cls | 0.0287 | 0.0280 | [0.025, 0.031] | 100% |
   370	| Axis-1 ref (DOM↔P-text) | reddit | 0.0260 | 0.0263 | [0.023, 0.031] | 100% |
   371	| Axis-3 image (P-SoM↔SoM) | cls | 0.0407 | 0.0415 | [0.035, 0.044] | 100% |
   372	
   373	**3 findings**:
   374	1. **Mean ≈ median** both sites → distribution **NOT right-skewed**, **NOT outlier-driven**
   375	2. **IQR 极窄** (0.002-0.003 wide), 全部 24 task 在 0.010-0.018 范围, zero outlier
   376	3. **Cross-site rank stable** + magnitude near-identical (0.0132 cls vs 0.0121 reddit, < 9% diff)
   377	
   378	**/stress W2 attack defused completely**: axis-2 cosine gap 是 uniform per-task signature, 不是 aggregate artifact. 这与 H1 binary dichotomy 11% strict per-task fragile 形成对比 — H1 因为问 layer-comparison 离散问题易 fragile, axis-2 cosine 是 continuous mode-pair distance 即使 magnitude 小也 robust per-task.
   379	
   380	**Paper §5.7 增强**: 加入 per-task fragility 段, 明确每个 task 都贡献 axis-2 signal, 不是 2-3 outlier mean artifact.
   381	
   382	### 7.3.0a Exp 3 logit lens 输出层 amplification (2026-05-12 21:02)
   383	
   384	`axis2_logit_lens.md` + `fig_axis2_logit_lens.png`. 应用 Qwen3-VL-4B `model.model.language_model.norm` + `model.lm_head` to per-layer per-mode mean hidden states, 算 KL across 37 层.
   385	
   386	| Pair | Site | Peak L (KL) | Peak KL | Exp 1 cosine peak | 放大倍数 |
   387	|---|---|---|---|---|---|
   388	| P-text↔P-SoM (axis-2 flat) | cls | **L23** | 0.162 | 0.011 | ~14x |
   389	| DOM↔P-prompt (axis-2 hier) | cls | L25 | 0.044 | 0.007 | ~7x |
   390	| DOM↔P-text (axis-1) | cls | L23 | 0.551 | 0.025 | 22x |
   391	| P-prompt↔P-SoM (axis-1) | cls | L23 | 0.695 | 0.029 | 24x |
   392	| Cross-site reddit | | L23-L25 | 0.13-0.62 | preserved | preserved |
   393	
   394	**3 findings**:
   395	1. Axis-2 prompt-family **IS in output distribution** — KL 0.16 at L23, NOT null. Exp 1 cosine 0.011 is not the end of the story.
   396	2. **lm_head 10-25x amplification of cosine → KL** but axis-agnostic ratio preserved (axis-1/axis-2 ratio ~4.3 cls, ~4.9 reddit, vs cosine ratio ~3 — slight amplification of stronger axis but not breaking 3-4x rank).
   397	3. **KL @ L36 ≈ 0 paradox**: 因 mean hidden state at last layer collapse to common JSON format header. Mode-distinct signal concentrated in **L23-L25 decoding window** (not final embedding). This is the "knows but says differently" structural mirror of Wu et al. tool calling.
   398	
   399	**Paper §5.7 follow-up paragraph** added: 三轴 hierarchy persists at output distribution with same rank-order. Deployment routing (paper-2) should treat L23-L25 logit-lens features as cheapest highest-signal mode-axis discriminator.
   400	
   401	### 7.3.1 Reddit cross-site results (2026-05-12 16:30 — P5a + P5b analyses landed)
   402	
   403	**P5a — Format variation H1 test on reddit** (`format_variation_h1_test_reddit.md`):
   404	
   405	| Variant | Peak L (reddit) | Peak L (cls baseline) |
   406	|---|---|---|
   407	| som_standard / browser_use_at / tarsier_typed / xml_tagged | **L17** | L36 (last) |
   408	| appagent_id / plain_numbered | **L04** | L36 |
   409	| hash_id_control | **L04** ✓ (acts as control) | L36 (control failed) |
   410	| plain_sentence | **L17** | L17 |
   411	| dom (baseline) | **L04** ✓ | L04 ✓ |
   412	
   413	**Reddit nuance — cleaner mid-layer fusion**: Reddit 上 marks-like 4/6 真 peak 在 L17 (mid-layer), cls 上 L36 是 monotonic increasing artifact (peak hit boundary). Reddit hash_id_control L04 acts as proper "no integer" control (cls 上失败). Reddit data supports Q5 mid-layer fusion hypothesis better than cls.
   414	
   415	Caveats: small n (24×2=48/mode) makes 2/6 marks-like falling to L04 (appagent_id, plain_numbered) plausible as sampling noise; plain_sentence triggering L17 on reddit (not cls) suggests reddit narrative comments may pattern-match list semantics.
   416	
   417	**P5b — Mirage signature on reddit** (`stage4_method42_results_reddit.md`):
   418	
   419	| Test | Value at L17 | cls baseline |
   420	|---|---|---|
   421	| P-SoM ↔ DOM | **0.0098** (nearly 0) | similar (text-axis sibling) |
   422	| P-SoM ↔ SoM | **0.0423** | similar (image-axis split) |
   423	| P-SoM ↔ Vision | 0.0457 | similar |
   424	| DOM ↔ Vision peak | L04 = 0.0687 (AUROC=1.0) | L04 similar |
   425	
   426	→ **Cross-site Mirage replication ✓**: P-SoM behaves as text-axis sibling of DOM at L17 (image-feature reduction), not as image-axis sibling of SoM. paper §5 4-fold (d) drop-one mechanism holds on reddit.
   427	
   428	**Paper §5 cross-site evidence stack now complete**:
   429	1. P-SoM mid-layer mechanism (4-fold drop-one) — cls + reddit replicated ✓
   430	2. Indexed-list format → shortcut activation — directional consistency cls ↔ reddit ✓
   431	3. Mirage signature geometric structure — cls + reddit replicated ✓
   432	
   433	**P4 selection-bias defense (2026-05-12 18:50)** — cls reverse-tier H1 (`format_variation_h1_test_cls_reverse.md`):
   434	
   435	| Variant | strong-tier cls | reverse-tier cls | reddit |
   436	|---|---|---|---|
   437	| 6 marks-like | L36 monotonic | **L36 monotonic** ✓ same | L17 (4/6 真 peak) |
   438	| hash_id_control | L36 (failed control) | **L36** ✓ same | L04 ✓ proper control |
   439	| plain_sentence | L17 | **L22** close to L17 | L17 |
   440	| dom baseline | L04 ✓ | **L04** ✓ | L04 ✓ |
   441	
   442	H1 mechanism in cls is **not tier selection artifact** (strong vs reverse both replicate). Reddit data paradoxically cleaner reveal of true L17 mid-layer fusion locus (cls L36 is monotonic-boundary artifact).
   443	
   444	### 7.4 Decisions pending
   445	
   446	| Decision | Owner | Trigger |
   447	|---|---|---|
   448	| Co-author multimodal extension vs cite + independent framing | Zekun | After Zekun reply to message |
   449	| Method 4.5 path: LA-HDMI vs SAE | Zekun + advisor sync | After v2 full sweep + Zekun reply |
   450	| Paper §5 prose round | Codex + me | After v2 full + Zekun decision |
   451	
   452	## 8. Roadmap (next 2-4 weeks)
   453	
   454	| Week | Milestone | Deliverable |
   455	|---|---|---|
   456	| **Week 1** (now → 2026-05-18) | v2 full sweep land + Zekun sync + paper §5 prose v1 | 48-cell H-mean table + Zekun message + paper §5 §1-4 prose draft |
   457	| **Week 2** (2026-05-19 → 25) | Cross-site Method 4.2 (reddit) + reverse-tier Method 4.4 | Replication results + paper §5 §5 prose |
   458	| **Week 3** (2026-05-26 → 06-01) | Method 4.5 launch (LA-HDMI or SAE per Zekun decision) | Pilot results + paper §5 §6-7 prose |
   459	| **Week 4** (2026-06-02 → 08) | Paper §5 codex round + advisor review | Submission-ready paper §5 |
   460	
   461	## 9. Connection to paper §1 + §6
   462	
   463	- **§1 phantom routing space + 4-fold drop-in property** — completely independent of mechanism work, anchors Outcome / Macro / Efficiency dimensions. NOT in this folder; see `paper_planning.md` §1
   464	- **§6 cost-aware routing** — Peale et al. 2026 uncertainty decomposition anchor adds theoretical layer to phantom routing space's empirical AUROC. Method 4.2 cosine gap could serve as "reducible uncertainty" signal in deployment
   465	
   466	These two stay outside mechanism folder. Mechanism workspace is paper §5-specific.

exec
/bin/bash -lc "nl -ba docs/checkpoints/paper_drafts/section5_mechanism.md | sed -n '1,220p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# 5. Mechanism Interpretability
     2	
     3	## 5.1 Overview and Theoretical Framing
     4	
     5	Why does Phantom-SoM sometimes achieve DOM-like cost while retaining part of the SoM signal? The mechanism evidence points to a phantom routing space in the residual stream: when the model receives flat Set-of-Mark text without the annotated image, it does not simply collapse to DOM. Instead, it occupies a mode whose text-axis geometry is close to DOM/P-text and whose image-axis geometry remains separated from full SoM.
     6	
     7	This section is the Zoom-4 layer of the paper's four-level account. Zoom 1 defines the architectural intervention, "skip the annotated image"; Zoom 2 measures the behavioral axes of text payload, prompt family, and image presence; Zoom 3 links the observed behavior to Mirage-style no-image visual reasoning and prompt-format sensitivity; Zoom 4 asks where the resulting mode is represented and whether it is causally used by the model. We index layers L0-L36, where L0 is the embedding-block output and L1-L36 are the 36 transformer decoder block outputs.
     8	
     9	The analysis builds on the linear-readable and steerable circuit framework of Wu et al., which uses mode means, PCA geometry, and mean-difference activation steering to study tool selection, and on work showing middle-layer cross-modal information flow in VLMs \citep{wu2026toolcalling,kaduri2024whatsintheimage}. Our setting is not a replication of those papers. It is a multimodal web-agent application of the same representation-level question: whether a behaviorally useful routing arm is linearly readable, partially steerable, and causally active inside the model.
    10	
    11	Four mechanism claims organize the evidence. First, observation modes are geometrically separable in the residual stream: the PCA cosine-gap analysis reaches AUROC 1.0 on the main mode contrasts. Second, Phantom-SoM is a mid-layer text-axis sibling of DOM/P-text, not an image-axis sibling of full SoM. Third, SoM-to-no-image activation patching displaces target continuations at mid layers L11-L17 with consistent magnitude across classifieds and reddit. Fourth, the shortcut trigger is **better described as the conjunction of integer-indexed marker and markup-sigil leading delimiter** than as an abstract "flat element list" — AXTree hierarchy preserves the early L04 image-axis peak, but so do indexed variants that lack either the integer (e.g., `hash_id_control`) or the sigil (e.g., `appagent_id`, `plain_numbered`); only the conjunction triggers the late shift. This refinement is **exploratory after W6** and awaits held-out falsifiers (`bare_N`, `bracket_no_int`).
    12	
    13	**Evidence status (added after /codex-stress 2026-05-12, before any reader reaches §5.2)**:
    14	
    15	| Evidence layer | Method | Status |
    16	|---|---|---|
    17	| Geometry (cosine PCA per layer) | Method 4.2 (§5.2) | **Strong** — AUROC 1.0 on key contrasts, 5/5 robustness, 100% per-task above 0.010 threshold |
    18	| Causal continuation patching (SoM → no-image arms) | Stage 2/3 (§5.4) | **Causal** — mid-layer L12-L18 transfers across cls + reddit, additive across DOM/P-text/P-prompt targets, Gaussian-random negative controls at ~0 |
    19	| Causal axis-2 prompt-only patching | Exp 5 cellhprompt (this section) | **Causal continuation evidence on 2 sites N=24 each**; bootstrap CI and content-matched (task-shuffled / random-source) controls **PENDING** (cellhprm_*_rand 359719/359720 in flight) |
    20	| Steering (mean-diff activation) | Method 4.4 (§5.3) | **Weak / partial** — best H-mean 0.33 at L33 α=10, layer-α tradeoff prevents single sweet spot, treated as evidence ceiling not validation |
    21	| Output divergence | Exp 3 logit lens (§5.7) | **Geometry-amplification at L23-L25**, NOT a deployed classifier; deployment claim deferred |
    22	| Trigger attribution (which formats trigger shortcut) | W6 tokenization (§5.5) | **Exploratory** — 6 marks-like variants split 2-vs-4 on first-token sigil; held-out falsifier `bare_N` (integer no sigil) and `bracket_no_int` (sigil no integer) pending |
    23	
    24	The cross-site evidence stack is deliberately defensive. Per-task H1 fragility shows the dichotomy is an aggregate mechanism rather than a deterministic per-task law. Reverse-tier classifieds runs defend against strong-tier selection bias. Reddit format variation replicates the shortcut direction with cleaner mid-layer peaks. Reddit Method 4.2 replicates the Mirage signature: Phantom-SoM remains close to DOM on the text axis while separating from SoM on the image axis. Paper 1 uses these results for mechanism interpretation only; routing implementation is deferred to paper 2, consistent with the paper-planning scope split.
    25	
    26	## 5.2 Method 4.2: PCA Cosine Gap
    27	
    28	Method 4.2 extracts hidden states from Qwen3-VL-4B B1 runs and compares observation modes by layer. For each mode pair and layer, we compute the cosine gap between hidden-state means, evaluate AUROC by projecting examples onto the mean-difference direction, and summarize per-mode geometry through PCA top-10 variance. The classifieds baseline contains 288 examples, formed from 24 strong-tier tasks, two archived steps, and six modes, over 37 indexed layers.
    29	
    30	The robustness suite passes all five checks in the plan. Label permutation leaves the real AUROC 9.8 standard deviations above the permuted baseline. Per-task analysis is positive for all 24 tasks. Step 2 and step 5 curves are invariant at the mechanism level. The L23 silhouette score is at least 0.5, showing nontrivial clustering. Bootstrap 95% confidence intervals are tight, with widths of roughly 4-15% of the corresponding means.
    31	
    32	The key classifieds snapshot is the L17 cosine-gap table:
    33	
    34	| Pair at L17 | Cosine gap | 95% CI | AUROC |
    35	|---|---:|---:|---:|
    36	| P-SoM <-> P-text | 0.0028 | [0.0027, 0.0029] | 1.000 |
    37	| DOM <-> P-prompt | 0.0013 | [0.0012, 0.0014] | 1.000 |
    38	| P-SoM <-> SoM | 0.0413 | [0.0403, 0.0422] | 1.000 |
    39	| DOM <-> Vision | 0.0547 | [0.0531, 0.0563] | 1.000 |
    40	
    41	The reddit replication lands the same qualitative geometry. At L17, P-SoM is close to DOM with cosine gap 0.0098 and close to P-text with gap 0.0027, while P-SoM-to-SoM remains much larger at 0.0423 and P-SoM-to-Vision at 0.0457. The DOM-to-Vision image-axis peak is L04 with cosine gap 0.0687 and AUROC 1.0.
    42	
    43	This is the Mirage signature in geometric form. Phantom-SoM is not represented as a weakened image mode. At the mid-layer disruption locus, it is a text-axis sibling of DOM/P-text, while the image-axis distance to full SoM remains large.
    44	
    45	## 5.3 Method 4.4: Mean-Difference Activation Steering
    46	
    47	Method 4.4 tests whether the readable mode direction can be used as a steering direction. For each layer, we form a mean-difference vector between Phantom-SoM-like and DOM-like hidden states, add it to each input at generation time with scaling factor $\alpha$, and evaluate whether the continuation moves toward the target mode while preserving the JSON action envelope. Following HDMI's evaluation vocabulary, reliability is the harmonic mean of completeness and selectivity, not a raw shift rate \citep{khorasani2026hdmi}.
    48	
    49	The v2 sweep covers layers [11, 17, 23, 29, 33, 34] and $\alpha \in [1,2,5,10,20]$, for 45 completed cells in the plan summary. The original L17, $\alpha=5$ smoke result reported H-mean 0.44, but the full sweep lowers that cell to 0.16. The plan records this as a smoke-variance artifact from notes 126/127: a 4-cell smoke was too small to support a sweet-spot claim.
    50	
    51	The strongest full-sweep cell is L33, $\alpha=10$, with H-mean 0.33. Its completeness is 38% and its selectivity is 29%. The layer profile is the important result: mid layers L11-L23 preserve the JSON envelope with 100% selectivity but have low completeness, while late-layer L33 produces the largest shifts but frequently over-steers the continuation out of the expected JSON format.
    52	
    53	This creates a probe-causal dissociation. The mid-layer geometry is cleanly readable and causally implicated by patching, but fixed mean-difference steering is only partially reliable. The 0.33 H-mean is therefore an evidence ceiling for Method 4.4, not a final control result. Section 8 should treat LA-HDMI and SAE feature steering as future work motivated by this ceiling, without claiming that either method has already improved it.
    54	
    55	## 5.4 Stage 2/3: Activation Patching for a Causal Mid-Layer Mechanism
    56	
    57	Activation patching provides the causal test. For each task, the clean/source run and corrupt/target run use the same archived browser step and deterministic 50-token continuation. In the core SoM-to-Phantom-SoM setup, the source prompt is `som`: task instruction, SoM prompt family, flat `[SOM_MARKS]` text, and annotated screenshot. The target prompt is `phantom_som`: the same instruction, same prompt family, and same `[SOM_MARKS]` text, but no image. Source hidden states are cached by layer, injected into the final input-token position of the target on the first forward pass, and subsequent decoding proceeds normally through the model cache.
    58	
    59	Each patched continuation is scored against the unpatched source and target continuations. The main disruption statistic is the drop in `token_overlap_to_target`; Levenshtein distance to target is the paired backup. Layer-wise tests compare each grid layer to the final-layer reference using task-paired differences and Holm-Bonferroni correction across the canonical grid. Random-injection controls replace source hidden states with Gaussian tensors matched to source activation mean and standard deviation.
    60	
    61	The Stage 2 P-SoM<->SoM dashboard now contains ten completed cells, including reddit F/G and the reddit random control:
    62	
    63	| Cell | Site | Direction | Mid-layer target-overlap drop | Holm status |
    64	|---|---|---|---:|---|
    65	| A | cls | SoM->P-SoM forward | -0.32 at L17 | significant |
    66	| B | cls | P-SoM->SoM reverse | -0.16 at L17 | significant |
    67	| C | cls | reverse-tier forward | -0.02 at L17 | null |
    68	| D | cls | strong-tier reverse | -0.18 at L17 | significant |
    69	| E | cls | random injection | -0.03 uniform | negative control |
    70	| F | reddit | SoM->P-SoM forward | -0.21 at L17 | significant |
    71	| G | reddit | P-SoM->SoM reverse | -0.18 at L17 | significant |
    72	| Cr/Dr | reddit | 2x2 controls | -0.15 to -0.18 | significant |
    73	| Er | reddit | random injection | approximately 0 uniform | negative control |
    74	
    75	Stage 3 extends this from P-SoM to the three no-image arms, testing whether the image-feature axis is shared across DOM, P-text, and P-prompt targets. The table below reports per-task-paired Δoverlap-to-target from the patching_continuation_results.json under each cell directory, with the layer at which the disruption peaks.
    76	
    77	| Site | SoM->DOM | SoM->P-text | SoM->P-prompt | best-L Δ range |
    78	|---|---:|---:|---:|---:|
    79	| cls | -0.309 at L17, -0.352 at L18 (best) | -0.255 at L17, -0.270 at L12 (best) | -0.223 at L17, -0.273 at L13 (best) | [-0.273, -0.352] |
    80	| reddit | -0.335 at L11, -0.255 at L17, -0.338 at L14 (best) | -0.244 at L11, -0.236 at L17, -0.330 at L15 (best) | -0.233 at L11, -0.191 at L17, -0.322 at L14 (best) | [-0.322, -0.338] |
    81	
    82	All six Stage 3 cells are now closed. Two observations carry the cross-site claim. First, every cell's best layer falls inside the L12-L18 mid-layer window, and every cell's best Δoverlap-to-target is between -0.27 and -0.35. The mid-layer fusion locus is therefore not a single layer index but a tight 7-layer window that transfers across cls and reddit. Second, the interpretation is additive rather than arm-specific: a SoM source state displaces DOM, P-text, and P-prompt targets toward the source with similar magnitude, implying a shared image-feature substrate across all three no-image arms. The negative controls, Cell E at -0.03 and Cell Er near zero, rule out a generic nonzero-injection explanation.
    83	
    84	## 5.5 Image-Axis Peak-Layer Dichotomy and H1 Format Variation
    85	
    86	The cleanest single-pair signature is the image-axis peak-layer dichotomy. Across eight image-presence contrasts, the no-image side's text format predicts the peak layer with zero overlap. If the no-image side is AXTree text, the image-axis cosine gap peaks at L04 in all four pairs: DOM<->Vision, DOM<->SoM, P-prompt<->Vision, and P-prompt<->SoM. If the no-image side is `[SOM_MARKS]` or another flat marks text, the peak shifts to L17-L36 in all four pairs: P-text<->Vision, P-text<->SoM, P-SoM<->Vision, and P-SoM<->SoM.
    87	
    88	The refined H1 is a pretraining co-occurrence shortcut: when the input contains a marker token sequence that pretraining data associates with HTML / agent-trace visual grounding (specifically the conjunction of integer index and markup-sigil leading delimiter such as `[`, `<`, `@`), the model activates a visual-grounding pathway even if the image is absent. Flat element-list form alone is **not sufficient** — `appagent_id` (`id_N: label`) and `plain_numbered` (`N. label`) are nominally flat indexed lists but lack the markup-sigil and behave like AXTree-DOM (W6 evidence, exploratory). Prompt-format sensitivity makes this plausible at the input level \citep{sclar2024promptformat}; Method 4.2 shows it as a layer-resolved internal signature.
    89	
    90	The format-variation grid contains ten modes: six marks-like variants, two controls, and DOM/SoM baselines. In the classifieds strong-tier baseline, all six marks-like variants peak at L36, the hash-ID control also peaks at L36, the plain-sentence control peaks at L17, and the DOM baseline preserves the L04 peak. Because L36 is the boundary layer, this is best read as a strong late/monotonic signature rather than as a precise late-layer mechanism.
    91	
    92	The classifieds reverse-tier run reproduces the strong-tier shape. The six marks-like variants and hash-ID control again peak at L36, the plain-sentence control moves to L22, and DOM remains at L04. This defends H1 against the selection-bias concern that strong-tier curation alone created the pattern.
    93	
    94	The reddit format run is cleaner for the mid-layer interpretation. Four of six marks-like variants peak at L17, the plain-sentence control peaks at L17, hash-ID control returns to L04, and DOM remains at L04. **W6 attribution** (`docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md`, exploratory) further finds that the two L04 marks-like variants (`appagent_id`, `plain_numbered`) share a feature with the L04 DOM baseline: their first tokens are alphanumeric, while the four L17-peaking marks-like variants all start with markup-sigil tokens (`[`, `<`, `@`). The hash-ID control (`#a3f7`) starts with a sigil but lacks integer-marker structure and also peaks at L04, suggesting the trigger conjunction is integer-marker + markup-sigil rather than either alone. This is a post-hoc feature-attribution on a small (N=6 marks-like) format set; held-out falsifiers (`bare_N` = integer without sigil, `bracket_no_int` = sigil without integer) are not yet run. Cross-site, the safe claim is directional: marker formats that combine integer indexing with markup-sigil leading delimiters tend to delay image-axis separation into mid/late layers, while AXTree hierarchy and indexed-list variants lacking either feature preserve the early L04 image-axis peak. The reddit curve reveals the true L11-L17 fusion locus more clearly than the classifieds L36 boundary artifact.
    95	
    96	## 5.6 Convergent Four-Vertical-Defense Evidence Stack
    97	
    98	The first defense is per-task fragility. On 45 classifieds task-step pairs, only 11% satisfy the strict per-task dichotomy, even though aggregate marks-like peaks are later than AXTree peaks. This prevents over-claiming: H1 is a population-level mechanism signature with task variability, not a deterministic rule for every trajectory.
    99	
   100	The second defense is selection-bias robustness. The classifieds reverse-tier run replicates the strong-tier H1 pattern, including L36 marks-like peaks and L04 DOM baseline. The shortcut signature is therefore not an artifact of selecting tasks where SoM beats DOM.
   101	
   102	The third defense is cross-site H1. Reddit does not reproduce the exact boundary-layer shape, but it reproduces the direction of the indexed-list shortcut with a cleaner L17 mid-layer peak for four of six marks-like formats. The site changes the curve shape, not the basic interpretation.
   103	
   104	The fourth defense is cross-site Mirage geometry. Reddit Method 4.2 reproduces the central relation: P-SoM is close to DOM/P-text at L17 and far from SoM/Vision on the image axis, with AUROC 1.0 on the key contrasts. This supports cross-site generalization of the mechanism claim, not B0/B1 capability scaling.
   105	
   106	Two additional defenses remain deferred rather than folded into the claim: P2 cross-family Phi-3.5-Vision and P3 larger Qwen2-VL-7B. The current evidence is sufficient for the single-model, cross-site Qwen3-VL-4B mechanism section; family and capacity generalization belong in future work or Section 7.
   107	
   108	## 5.7 Layered Three-Axis Mechanism Hierarchy
   109	
   110	A naive reading of Method 4.2's L17 snapshot suggests the four phantom-boundary modes split into two text-format clusters with prompt-family making no geometric contribution. That reading is incomplete: it inspects the wrong layer. Computing full 37-layer cosine-gap profiles for axis-isolated pairs reveals a layered three-axis hierarchy in the residual stream.
   111	
   112	The pairs are constructed to isolate each axis. Axis-1 (text-format swap, prompt fixed) is measured by DOM<->P-text (both DOM prompts) and P-prompt<->P-SoM (both SoM prompts). Axis-2 (prompt-family swap, text fixed) is measured by DOM<->P-prompt (both hierarchical AXTree) and P-text<->P-SoM (both flat indexed list). Axis-3 (image-feature swap, mode otherwise fixed) is measured by the P-SoM<->SoM reference pair. All five curves are computed on `stage4_multimode_b1_cls/hidden_states.npz` (288 examples, 37 layers) and replicated cross-site on the matching reddit run.
   113	
   114	The peak-layer and magnitude table (cls site, reddit columns omitted but qualitatively identical):
   115	
   116	| Axis | Pair | L17 | L23 | L36 | Peak L | Peak gap |
   117	|---|---|---:|---:|---:|---:|---:|
   118	| Axis-3 image | P-SoM <-> SoM | 0.0412 | 0.0400 | 0.0411 | **L17** | 0.0412 |
   119	| Axis-1 text-format | DOM <-> P-text | 0.0120 | 0.0254 | 0.0201 | **L23** | 0.0254 |
   120	| Axis-1 text-format | P-prompt <-> P-SoM | 0.0113 | 0.0292 | 0.0201 | **L23** | 0.0292 |
   121	| Axis-2 prompt-family | P-text <-> P-SoM | 0.0028 | 0.0114 | 0.0089 | **L23** | 0.0114 |
   122	| Axis-2 prompt-family | DOM <-> P-prompt | 0.0013 | 0.0050 | 0.0067 | **L36** | 0.0067 |
   123	
   124	Three regularities organize the table. First, the three axes have distinct peak layers: image-axis at L17 (fast, sharp), text-format at L23 (slower late-mid build), prompt-family at L23 or L36 (same timing as text-format on the flat-text pair, boundary peak on hierarchical). Second, the three axes have distinct magnitudes: image axis approximately 0.04, text-format approximately 0.03, prompt-family approximately 0.01. Prompt-family is roughly 3 to 4 times smaller than text-format and 4 to 8 times smaller than image. Third, the magnitude rank holds cross-site: the reddit P-text<->P-SoM axis-2 peak is 0.0098 at L23 (versus cls 0.0114), the same rank-order and the same peak layer.
   125	
   126	The L17 snapshot exclusion of axis-2 is therefore a layer-selection artifact rather than a structural absence. Prompt-family is geometrically present in the residual stream; it simply emerges at L23 rather than L17, and at one-third the magnitude of text-format. The mid-layer fusion locus identified in Sections 5.2-5.5 is specifically the image-axis fusion locus (Mirage signature). The text-format and prompt-family axes share a separate late-mid build at L23 that runs in parallel.
   127	
   128	Per-task fragility check confirms the axis-2 signal is broad rather than outlier-driven. At L23 on the P-text<->P-SoM pair, the per-task cosine gap distribution is tight on both sites: classifieds median 0.0131 with IQR [0.0124, 0.0142] and reddit median 0.0120 with IQR [0.0113, 0.0127], with mean approximately equal to median in both cases (not right-skewed). 100% of 24 classifieds tasks and 100% of 24 reddit tasks exceed both the 0.005 noise threshold and the 0.010 cohort-mean magnitude. None of the 48 tasks falls outside the 0.010-0.018 band on either site. This contrasts with the per-task fragility of the binary H1 dichotomy in Section 5.5 (5 of 45 task-step pairs strict, 11%); the H1 dichotomy is fragile because it asks a per-task layer-comparison question, while the axis-2 cosine gap measures a continuous mode-pair distance that has low per-task variance even when the aggregate magnitude is small. The axis-2 claim is therefore not a 2-3 outlier mean artifact but a uniform 0.011-0.013 cosine-gap signature across every task in both sites.
   129	
   130	This layered hierarchy partially resolves the Phantom-SoM hero puzzle without requiring a non-mechanistic explanation. Phantom-SoM occupies three distinct axis positions: **large image-axis separation from SoM** at L17 (0.0412 cosine gap, the largest single signal — note this is a *separation* from SoM, not a proximity; the no-image marks-text reshapes how image-axis divergence accumulates rather than recovering an image-like internal state), **small prompt-family separation from P-text** at L23 (0.011 signal, consistent across cls and reddit), and **medium text-format separation from P-prompt** at L23 (0.029 signal). P-text occupies only the text-format separation; P-prompt occupies only the prompt-family separation against P-SoM; DOM occupies none of the three. The drop-one hero status of Phantom-SoM in `fig_meta_forest.png` therefore corresponds to the only mode that simultaneously expresses all three axis positions. The connection from "three-axis positional uniqueness" to "drop-one task-success contribution" is currently associational, not yet causally bridged — the patching evidence shows residual-stream displacement on 50-token continuation, not direct displacement of task-level SR / drop-one oracle. Closing that behavioral causal bridge is open work.
   131	
   132	This reframing is itself paper-grade contribution and not a downgrade of the original mechanism story. It strengthens Section 5 from "mid-layer image-feature axis explains text-format cluster" to "the residual stream carries three quantitatively distinct axes with image-axis dominant at L17 and text-format + prompt-family at L23". Section 8 inherits a sharper instruction for future single-axis steering: LA-HDMI and SAE residual-stream interventions can target the L23 prompt-family direction directly, but the signal-to-noise ratio is approximately 3-4 times worse than the image-axis intervention, so steering experiments on axis-2 require correspondingly more samples or alternative attribution methods.
   133	
   134	The three-axis hierarchy persists when we move from residual-stream geometry to output distribution. A logit lens test (Exp 3) applies Qwen3-VL-4B's final RMSNorm and lm_head to each per-layer per-mode mean hidden state, then computes KL divergence between mode pairs across all 37 layers. On classifieds the axis-2 prompt-family pair (P-text vs P-SoM) reaches peak KL 0.162 at L23, the axis-1 text-format pair (P-prompt vs P-SoM) reaches peak KL 0.695 at L23, and the reddit replication holds the same rank-order with peak KL 0.126 at L24 and 0.617 at L23 respectively. The output-level axis-1 to axis-2 magnitude ratio is therefore approximately 4.3 on classifieds and 4.9 on reddit, preserving the 3-4x residual-stream rank from Exp 1. The lm_head amplifies the residual-stream cosine signal into output divergence by roughly 14x on the axis-2 flat-text pair (cosine 0.011 to KL 0.16) and 24x on the axis-1 SoM-prompt pair (cosine 0.029 to KL 0.69), but it is axis-agnostic in the ratio it preserves.
   135	
   136	Two corollaries follow. First, the KL trajectory drops to approximately zero at L36 even though L23 KL is substantial. The mean hidden state at the final layer collapses to the shared JSON action-header tokens that every mode emits, so mode-distinct output signal is concentrated in the L23-L25 decoding window rather than at the final embedding. Second, this output-amplification observation is **mechanistic, not a deployment-time classifier claim**: the lm_head acts as an axis-agnostic ratio-preserving projection that scales residual-stream geometry into output-space KL — the L23-L25 KL magnitude is a property of the mean hidden state, not a per-task discriminator. Whether the L23-L25 hidden representation can be used as a held-out mode classifier — with per-task AUROC, random-direction baseline, and competitive comparison to surface-token classifiers — is open work. Routing exploitation, deferred to paper 2, will need to make this case explicitly rather than inheriting it from §5.7.
   137	
   138	## 5.8 Discussion and Limits
   139	
   140	The main limit is the Method 4.4 ceiling. The cosine-gap and patching evidence point to L11-L17 as the readable and causally active fusion region, while the best fixed mean-difference steering cell is late, L33 with $\alpha=10$, and has H-mean 0.33 because completeness and selectivity trade off. This supports a mechanism interpretation but not a strong deployment-time steering claim.
   141	
   142	The second limit is layer precision. Classifieds H1 peaks often hit L36, while reddit reveals cleaner L17 peaks. The robust claim is therefore an effect-direction claim: AXTree hierarchy preserves early image-axis separation, and flat element-list formats delay that separation into mid/late computation. We should not claim that every site or task has an identical peak layer.
   143	
   144	Literature positioning should stay modest. Section 5 applies the linear-readable, steerable, and mid/late-layer circuit framework to multimodal web-agent observation modes \citep{wu2026toolcalling,kaduri2024whatsintheimage,khorasani2026hdmi,fayyaz2026steermoe}. It should not claim novelty as the first such circuit or the first use of marked text. The contribution is controlled scientific characterization of the phantom boundary.
   145	
   146	Finally, AXTree hierarchy is the unique defeating format in the aggregate, but the reason hierarchy defeats the shortcut remains open. The plan records one attribution-pending hypothesis: hierarchy or indentation tokens may redirect cross-modal attention before the flat-list shortcut activates. That should be treated as a supplement question, not as a Section 5 finding.
   147	
   148	## NOTE FOR HUMAN
   149	
   150	Bibkeys audit (2026-05-12 21:18): all 5 core mechanism anchors verified present in `paper.bib` — `wu2026toolcalling`, `khorasani2026hdmi`, `kaduri2024whatsintheimage`, `sclar2024promptformat`, `fayyaz2026steermoe`. Plus 5 method/protocol references added: `wang2023interpretability` (IOI patching), `zhang2024patching` (patching survey, NEEDS_VERIFY exact paper), `holm1979sequentially` (multiple-comparison correction), `lipton2018troubling` (ML scholarship critique), `neurips2024checklist` (reproducibility standard). paper.bib total 67 entries / 638 lines.
   151	
   152	Behavioral content to relocate from current `section5_mechanism_reddit.md`: lines 17-75 should move to Section 4 or a new behavioral-routing subsection. Specifically, lines 17-23 are reddit substrate framing; lines 25-35 are Axis 1 text-payload behavior; lines 37-47 are Axis 2 prompt behavior; lines 49-59 are Axis 3 image behavior; lines 61-67 are compound P-SoM versus DOM behavior; lines 69-75 are scope/noise limitations. Lines 1-15 are method material that was retained conceptually but must use the new L0-L36 layer convention. Line 77 should be deleted or replaced because routing implementation is now paper-2, not paper-1 Section 6.
   153	
   154	Stage 3 numbers verified 2026-05-12 from full per-task paired-test computation on `patching_continuation_results.json` (each cell, 24 tasks × 36 layers). H-d-cls best L18 Δ=-0.352, H-d-red best L14 Δ=-0.338, H-t-cls best L12 -0.270, H-t-red best L15 -0.330, H-p-cls best L13 -0.273, H-p-red best L14 -0.322. All 6 cells' best layer lands in L12-L18 mid-layer window, Δ range [-0.27, -0.35]. The L17-only column previously cited in plan §5.2 reads -0.309/-0.255/-0.223 (cls) and -0.255/-0.236/-0.191 (reddit); plan §5.2 has been updated to record best-layer Δ instead of L17-only Δ.
   155	
   156	Pending items (post 2026-05-12 audit): (a) Method 4.4 sweep description should be "45 completed cells out of a 6x5 layer-alpha grid plus 3 placeholder cells that did not finish", not "45/48-cell sweep" (the 48-cell wording in plan §5.3 implies a 48-cell denominator that was never executed). (b) Bibkey `zhang2024patching` is marked NEEDS_VERIFY in `paper.bib` because the intended reference may be Heimersheim & Nanda 2024 [arXiv:2404.15255] rather than Zhang & Nanda 2024 [arXiv:2309.16042]; verify before submission. (c) Bibkey `fayyaz2026steermoe` is marked NEEDS_VERIFY pending deanon of the ICLR 2026 submission.
   157	
   158	## NOTE FOR HUMAN — /codex-stress 2026-05-12 findings + pending follow-ups
   159	
   160	Codex independent audit (`docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md`) surfaced 6 weak claims + 5 honest gaps that Claude /stress had missed. 3 fixed inline tonight in §5:
   161	
   162	1. ✅ §5.1 ¶4 — "flat element-list trigger" → refined to "integer-marker + markup-sigil conjunction" with W6 exploratory caveat
   163	2. ✅ §5.7 hero paragraph — "proximity to SoM on the image axis... as if image were present" → corrected to "large image-axis SEPARATION from SoM... no-image marks-text reshapes how image-axis divergence accumulates" (removed internal contradiction with §5.2 table where P-SoM↔SoM gap 0.0412 is the largest = a separation, not proximity)
   164	3. ✅ §5.7 corollary 2 — "deployment-time mode classifier on output logprobs has strictly more signal" + "Section 6 routing should treat L23-L25 logit-lens features as the cheapest mode-axis discriminator" → softened to "mechanistic observation, not deployment-time classifier claim; held-out classifier with random-direction baseline is open work"
   165	4. ✅ Evidence status table added at end of §5.1 — geometry strong / patching causal-continuation / Exp 5 axis-2 CI pending / steering weak / output divergence not classifier / W6 trigger exploratory
   166	
   167	**Cross-doc follow-ups NOT fixed tonight** (each on next-action list):
   168	
   169	- **§4 P-text adjusted SR inconsistency**: §4 table line 37 says 11.90, prose line 106 says 12.38, hero_claim_bootstrap_ci.md says 12.38. Need to canonicalize one number from episode-level adjusted-success and update every occurrence. (1h)
   170	- **plan.md:125-135 stale "L17 planning site"**: plan still asserts L17 singular planning site, while new evidence shows cosine peak L23 + patching peak L11-L17 + steering best L33; replace with "patch-sensitive continuation window L11-L17 under final-token replacement patching". (1-2h)
   171	- **§6 + §7 drafts missing**: §1:13 promises Section 6 (Generalization) and Section 7 (Limitations and Implications). Current draft files: no `section6*.md`; §7 either deferred to paper-2 (routing) or merge into §8. Resolve before submission.
   172	- **Exp 5 cellhprompt bootstrap CI + content-matched control**: Gaussian random injection control 359719/359720 in flight; codex notes Gaussian alone is weak — also need task-shuffled (source from different task) and per-task bootstrap CIs.
   173	- **Behavioral causal bridge gap**: patching displaces 50-token continuation, not SR / drop-one. No experiment currently bridges mid-layer patching effect to task-success outcome. Open work.
   174	- **Cross-family**: P2 Phi-3.5-Vision + P3 Qwen2-VL-7B (task #40, #41). At minimum needed for §6 generalization narrative.
   175	
   176	**Codex verdict**: 0.10-0.20 NeurIPS/ICML/ACL main accept probability; would reject @ reviewer-3 4/10. **Workshop / borderline mid-tier conference today**. Codex agrees the drop-one oracle CI + cross-site asymmetry + axis-2 cosine signal breadth are strong; §5 mechanism prose was over-claiming relative to evidence.

exec
/bin/bash -lc 'for f in docs/checkpoints/mechanism/results/*.md; do echo $f; rg -n "Method|Layer|L0|L17|L23|L35|L36|cosine|AUROC|PCA|logit|patch|Holm|Bonferroni|bootstrap|seed|mean|hidden|token|last|embedding|block|KL|RMSNorm|norm|lm_head|source|target|random" "$f"; done' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/mechanism/results/axis2_layer_profile.md
3:**Question**: Method 4.2 at L17 shows prompt-family makes ~0 geometric contribution to residual stream
7:**Method**: For each prompt-only pair (text format fixed, prompt swap), compute full 37-layer cosine gap.
12:| Pair | Group | L0 | L4 | L17 | L36 | Peak L | Peak gap |
14:| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0013 | 0.0067 | **L36** | 0.0067 |
15:| P-text ↔ P-SoM  (axis-2 only, flat) | axis-2 | 0.0000 | 0.0006 | 0.0028 | 0.0089 | **L23** | 0.0114 |
16:| DOM ↔ P-text    (axis-1 only, DOM-prompt) | axis-1 | 0.0000 | 0.0134 | 0.0120 | 0.0201 | **L23** | 0.0254 |
17:| P-prompt ↔ P-SoM (axis-1 only, SoM-prompt) | axis-1 | 0.0000 | 0.0127 | 0.0113 | 0.0201 | **L23** | 0.0292 |
18:| P-SoM ↔ SoM     (image-axis reference) | image | 0.0000 | 0.0394 | 0.0412 | 0.0411 | **L17** | 0.0412 |
22:| Pair | Group | L0 | L4 | L17 | L36 | Peak L | Peak gap |
24:| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0012 | 0.0059 | **L36** | 0.0059 |
25:| P-text ↔ P-SoM  (axis-2 only, flat) | axis-2 | 0.0000 | 0.0006 | 0.0027 | 0.0080 | **L23** | 0.0098 |
26:| DOM ↔ P-text    (axis-1 only, DOM-prompt) | axis-1 | 0.0000 | 0.0125 | 0.0092 | 0.0183 | **L23** | 0.0217 |
27:| P-prompt ↔ P-SoM (axis-1 only, SoM-prompt) | axis-1 | 0.0000 | 0.0115 | 0.0086 | 0.0176 | **L23** | 0.0240 |
34:1. **Truly null geometry** — axis-2 pair curves flat <0.01 at all layers. Prompt-family bypasses residual stream entirely (acts at attention pattern or output head). → Next: Exp 3 logit lens or Exp 4 attention probe.
35:2. **Late-layer spike** — axis-2 pair curves spike at L25+ but flat at mid-layer. Prompt prior re-emerges at output decoding. → Next: Exp 5 late-layer patching.
36:3. **Early-layer spike absorbed** — axis-2 pair curves spike at L0-L5 then collapse to ~0. Prompt embedding effect absorbed by mid-layer fusion. → Next: Exp 3 logit lens to verify if it re-emerges in output distribution.
38:Compare peak layers above against axis-1 (text-format) pairs (the established mechanism with L17 peak) and image-axis reference (~0.04 magnitude). If axis-2 pair peak < 0.01 at all layers, hypothesis 1 holds.
docs/checkpoints/mechanism/results/axis2_logit_lens.md
3:Apply Qwen3-VL-4B's final_norm + lm_head to per-layer per-mode mean hidden states.
4:For each axis-isolated pair, compute KL(mode_a || mode_b) and top-1 token disagreement
5:across all 37 layers. This probes whether axis-2 cosine signal (L23 peak 0.011) gets
12:| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
14:| P-text vs P-SoM  (axis-2 flat-text) | **L23** | 0.1621 | 0.0215 | 0.1621 | 0.0003 |
19:| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
21:| DOM vs P-text    (axis-1 DOM-prompt) | **L23** | 0.5508 | 0.1299 | 0.5508 | 0.0001 |
22:| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L23** | 0.6953 | 0.1069 | 0.6953 | 0.0003 |
28:| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
35:| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
37:| DOM vs P-text    (axis-1 DOM-prompt) | **L23** | 0.5273 | 0.0898 | 0.5273 | 0.0000 |
38:| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L23** | 0.6172 | 0.0806 | 0.6172 | 0.0002 |
44:- **H_A (axis-2 absent from output)**: axis-2 KL flat <0.1 at all layers → prompt-family
45:  effect bypasses logit lens, only visible via attention heads or runtime decoding.
46:- **H_B (axis-2 amplified at output)**: axis-2 KL peak at L30+ ≫ cosine 0.011 magnitude →
49:- **H_C (axis-2 tracks residual stream)**: axis-2 KL peak at L23 same as cosine peak →
52:Cross-site replication should hold for any of the three. Compare axis-2 KL magnitudes to
53:axis-1 KL magnitudes to see whether 3-4x ratio in cosine space persists at output level.
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md
3:Per-task cosine gap distribution at L23 (axis-2 peak per §5.7 / Exp 1).
4:Each task averaged across its 2 steps; cosine gap computed between mode pairs.
6:**Defuse target**: /stress W2 attack — axis-2 mean 0.0114 might be dominated by 2-3 outlier tasks.
26:## Top 5 axis-2 tasks (classifieds, P-text ↔ P-SoM @ L23)
36:## Bottom 5 axis-2 tasks (classifieds, P-text ↔ P-SoM @ L23)
46:## Top 5 axis-2 tasks (reddit, P-text ↔ P-SoM @ L23)
56:## Bottom 5 axis-2 tasks (reddit, P-text ↔ P-SoM @ L23)
69:- cls: **100%** of 24 tasks above the L23 axis-2 mean magnitude
78:Compare to mean: cls=0.0132, reddit=0.0121.
79:If median << mean, the distribution is right-skewed → outlier-driven (consistent with /stress W2 attack).
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md
13:| Layers | 37 (L0-L36, Qwen3-VL-4B language decoder) | matching |
15:**Test logic**: Holding both `image` and `text-format` constant (off + flat) and patching source hidden states from `phantom_som` into a `phantom_text` run isolates whether the residual-stream prompt-family signature has *causal* effect on token continuation, not just *geometric* magnitude (which Exp 1 already showed is small at 0.011 cosine gap @ L23).
17:## Result — mid-layer (L11-L17) patching causal effect
19:| Site | Cell (axes) | overlap→tgt L11 | overlap→tgt L17 | LD→tgt L11 | LD→tgt L17 |
26:(Baseline `overlap→tgt = 1.00` at L35 = full target preservation, no patching effect.)
30:- Axis-2 (prompt) **alone** displaces target output by **0.20-0.30 overlap** units, mid-layer L11-L17 peak.
32:- **Prompt-only captures ~77-100% of the combined effect** (cls 0.21/0.25 = 84%; red @ L17 0.30/0.24 = 125%, **prompt-only stronger on red**).
36:Both cls + red show the same mid-layer L11-L17 peak. Reddit shows *stronger* axis-2 effect at L17 than cls (overlap→tgt 0.70 vs 0.79).
42:Compared with Exp 1 cosine geometry, using best-layer values:
46:| Image (SoM ↔ P-SoM) | 0.041 @ L17 | ~0.04-0.05 (inferred from H-text − cellhprompt diff) |
47:| Text-format (DOM ↔ P-text) | 0.029 @ L23 | (Exp H-d-cls/red, not directly compared here) |
48:| **Prompt-family (P-SoM ↔ P-text)** | **0.011 @ L23** | **~0.20-0.30 @ L11-L17** |
50:**4:3:1 cosine geometry ratio does NOT translate to 4:3:1 causal patching ratio.** Prompt-family has the **smallest** geometric magnitude but the **largest** causal patching weight.
54:Critically, the **layer at which cosine peaks ≠ the layer at which patching has maximal effect** for prompt-family:
56:| Layer | cls overlap→tgt | red overlap→tgt | Interpretation |
58:| L0  | 0.86 | 0.92 | early, signal not yet routed |
60:| L17 | 0.79 | **0.70** | **causal peak (red)** |
61:| **L23** | 0.96 | 0.89 | **cosine geometry peak, but patching weak** — representation stabilized |
63:| L35 | 1.00 | 1.00 | output convergence (baseline preserved) |
65:At **L23** (the cosine peak), patching displaces target output by only **0.04-0.11 overlap units** — much smaller than the **0.20-0.30** displacement at L11-L17.
71:- **L23 is the prompt-family "signature layer"**: representation has stabilized to its most discriminable form (highest cosine separation between P-SoM and P-text). It reflects *what prompt was given* — a state variable.
72:- **L11-L17 is the prompt-family "decision routing layer"**: patching here changes upstream signal that downstream layers consume to drive token continuation. It reflects *how the model uses the prompt* — a causal variable.
74:Activation patching is path-dependent: an upstream patch propagates into all downstream computations, while a downstream patch leaves upstream inputs unchanged so subsequent layers can re-encode the same signal. This is consistent with standard mechanistic-interpretability findings (cf. \citep{wang2023interpretability} IOI circuit: feature *encoded* ≠ feature *used*).
78:1. Residual-stream cosine separation is a **necessary but not sufficient** signal of causal mechanism.
79:2. Prompt-family information is **dispatchable** — small geometric perturbation at the decision layer produces large output displacement when patched.
80:3. **Where a feature is most readable (L23) and where it is most consequential (L11-L17) are different layers** — paper-grade mechanism claims must report both, not collapse them.
85:- Axis-1 (text-format): Exp 1 cosine 0.029 + H-d cells causal patching (prior)
86:- Axis-2 (prompt-family): Exp 1 cosine 0.011 + **Exp 5 cellhprompt causal patching (this)**
87:- Axis-image: Exp 1 cosine 0.041 + indirect (H-text − cellhprompt residual ~0.04-0.05)
91:- 2-site cross-replication (cls + red), N=24 each, paired L0-L36 sweep
94:- Behavioral: exploration rate axis-1 dependent (Exp 1 cosine sigma + §4.5 reddit behavioral)
95:- Causal mechanism: prompt-family mid-layer L11-L17 patching produces output displacement comparable to image-axis flip
99:- N=24 per cell — bootstrap CI on per-layer overlap means would tighten interpretation.
100:- "phantom_som" archive vs "som" archive — the codebase uses same hidden-state extraction infrastructure but `--source-mode phantom_som` extracts without image. The pilot_summary template label "(with image — clean)" is a hardcoded artifact, not a runtime check. Manual verification of the archive contents (hidden state norms) would close a soundness gap.
101:- Patching displacement is a token-level metric; doesn't directly translate to SR / drop-one oracle. Behavioral consequence (which paper §1 hero is about) operates on top of this causal signal.
106:- `patching_continuation_results.json`: per-layer per-task continuation strings + metrics (~1.3 MB each)
107:- `patching_continuation_curves.png`: visual layer profile
112:- Watcher missed GONE events due to silent-miss bug (PR same commit) — auto_pull dispatched manually
docs/checkpoints/mechanism/results/format_variation_h1_test.md
6:**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:
7:- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)
8:- Peak L17+: image-axis divergence delayed → variant DOES trigger marks-shortcut
12:| Variant | Format example | H1 class | Peak layer | Peak cosine gap |
14:| dom | `AXTree (baseline DOM)` | AXTree-baseline | **L04** | 0.0462 |
15:| plain_sentence | `'a, b, c, ...' (no list)` | control (no list) | **L17** | 0.0544 |
16:| som_standard | `[N] role 'label' (SoM)` | marks-like | **L36** | 0.0434 |
17:| browser_use_at | `@N label (Browser Use)` | marks-like | **L36** | 0.0502 |
18:| appagent_id | `id_N: label (AppAgent)` | marks-like | **L36** | 0.0506 |
19:| tarsier_typed | `[BN:role:label] (Tarsier)` | marks-like | **L36** | 0.0468 |
20:| plain_numbered | `N. label (numbered)` | marks-like | **L36** | 0.0502 |
21:| xml_tagged | `<el_N role='..'>label</el_N> (XML)` | marks-like | **L36** | 0.0426 |
22:| hash_id_control | `#hash label (no integer)` | control (no integer) | **L36** | 0.0514 |
26:### marks-like  (mean peak L36)
28:- `[N] role 'label' (SoM)`: peak **L36** = 0.0434
29:- `@N label (Browser Use)`: peak **L36** = 0.0502
30:- `id_N: label (AppAgent)`: peak **L36** = 0.0506
31:- `[BN:role:label] (Tarsier)`: peak **L36** = 0.0468
32:- `N. label (numbered)`: peak **L36** = 0.0502
33:- `<el_N role='..'>label</el_N> (XML)`: peak **L36** = 0.0426
35:### control (no integer)  (mean peak L36)
37:- `#hash label (no integer)`: peak **L36** = 0.0514
39:### control (no list)  (mean peak L17)
41:- `'a, b, c, ...' (no list)`: peak **L17** = 0.0544
43:### AXTree-baseline  (mean peak L4)
45:- `AXTree (baseline DOM)`: peak **L04** = 0.0462
49:- **6 marks-like variants**: mean peak layer = 36, range L36-L36
50:- **2 control variants** (no integer / no list): mean peak layer = 26, range L17-L36
51:- **AXTree-DOM baseline**: peak L04
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md
6:**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:
7:- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)
8:- Peak L17+: image-axis divergence delayed → variant DOES trigger marks-shortcut
12:| Variant | Format example | H1 class | Peak layer | Peak cosine gap |
14:| dom | `AXTree (baseline DOM)` | AXTree-baseline | **L04** | 0.0434 |
16:| som_standard | `[N] role 'label' (SoM)` | marks-like | **L36** | 0.0429 |
17:| browser_use_at | `@N label (Browser Use)` | marks-like | **L36** | 0.0520 |
18:| appagent_id | `id_N: label (AppAgent)` | marks-like | **L36** | 0.0526 |
19:| tarsier_typed | `[BN:role:label] (Tarsier)` | marks-like | **L36** | 0.0475 |
20:| plain_numbered | `N. label (numbered)` | marks-like | **L36** | 0.0518 |
21:| xml_tagged | `<el_N role='..'>label</el_N> (XML)` | marks-like | **L36** | 0.0439 |
22:| hash_id_control | `#hash label (no integer)` | control (no integer) | **L36** | 0.0516 |
26:### marks-like  (mean peak L36)
28:- `[N] role 'label' (SoM)`: peak **L36** = 0.0429
29:- `@N label (Browser Use)`: peak **L36** = 0.0520
30:- `id_N: label (AppAgent)`: peak **L36** = 0.0526
31:- `[BN:role:label] (Tarsier)`: peak **L36** = 0.0475
32:- `N. label (numbered)`: peak **L36** = 0.0518
33:- `<el_N role='..'>label</el_N> (XML)`: peak **L36** = 0.0439
35:### control (no integer)  (mean peak L36)
37:- `#hash label (no integer)`: peak **L36** = 0.0516
39:### control (no list)  (mean peak L22)
43:### AXTree-baseline  (mean peak L4)
45:- `AXTree (baseline DOM)`: peak **L04** = 0.0434
49:- **6 marks-like variants**: mean peak layer = 36, range L36-L36
50:- **2 control variants** (no integer / no list): mean peak layer = 29, range L22-L36
51:- **AXTree-DOM baseline**: peak L04
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md
6:**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:
7:- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)
8:- Peak L17+: image-axis divergence delayed → variant DOES trigger marks-shortcut
12:| Variant | Format example | H1 class | Peak layer | Peak cosine gap |
14:| appagent_id | `id_N: label (AppAgent)` | marks-like | **L04** | 0.0488 |
15:| plain_numbered | `N. label (numbered)` | marks-like | **L04** | 0.0505 |
16:| hash_id_control | `#hash label (no integer)` | control (no integer) | **L04** | 0.0508 |
17:| dom | `AXTree (baseline DOM)` | AXTree-baseline | **L04** | 0.0495 |
18:| som_standard | `[N] role 'label' (SoM)` | marks-like | **L17** | 0.0429 |
19:| browser_use_at | `@N label (Browser Use)` | marks-like | **L17** | 0.0515 |
20:| tarsier_typed | `[BN:role:label] (Tarsier)` | marks-like | **L17** | 0.0457 |
21:| xml_tagged | `<el_N role='..'>label</el_N> (XML)` | marks-like | **L17** | 0.0431 |
22:| plain_sentence | `'a, b, c, ...' (no list)` | control (no list) | **L17** | 0.0521 |
26:### marks-like  (mean peak L13)
28:- `[N] role 'label' (SoM)`: peak **L17** = 0.0429
29:- `@N label (Browser Use)`: peak **L17** = 0.0515
30:- `id_N: label (AppAgent)`: peak **L04** = 0.0488
31:- `[BN:role:label] (Tarsier)`: peak **L17** = 0.0457
32:- `N. label (numbered)`: peak **L04** = 0.0505
33:- `<el_N role='..'>label</el_N> (XML)`: peak **L17** = 0.0431
35:### control (no integer)  (mean peak L4)
37:- `#hash label (no integer)`: peak **L04** = 0.0508
39:### control (no list)  (mean peak L17)
41:- `'a, b, c, ...' (no list)`: peak **L17** = 0.0521
43:### AXTree-baseline  (mean peak L4)
45:- `AXTree (baseline DOM)`: peak **L04** = 0.0495
49:- **6 marks-like variants**: mean peak layer = 13, range L04-L17
50:- **2 control variants** (no integer / no list): mean peak layer = 10, range L04-L17
51:- **AXTree-DOM baseline**: peak L04
docs/checkpoints/mechanism/results/h1_per_task_fragility.md
13:AXTree-DOM peak layer: mean = **27.9**, std = 13.1, range L04-L36
14:Marks-like (avg across 7) peak layer: mean = **31.9**, std = 8.0
25:| 214 | 5 | L04 | L36.0 | **+32.0** |
26:| 228 | 2 | L04 | L36.0 | **+32.0** |
27:| 32 | 5 | L04 | L31.4 | **+27.4** |
28:| 228 | 5 | L04 | L29.4 | **+25.4** |
29:| 9 | 2 | L04 | L24.6 | **+20.6** |
35:| 61 | 5 | L17 | L16.4 | -0.6 |
36:| 20 | 2 | L36 | L33.3 | -2.7 |
37:| 122 | 2 | L36 | L33.3 | -2.7 |
38:| 60 | 5 | L17 | L11.6 | -5.4 |
39:| 37 | 2 | L36 | L28.0 | -8.0 |
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md
1:# Hero-claim bootstrap CI (W1 defuse)
3:Per-seed bootstrap 95% percentile CI on paired adjusted-SR diffs and drop-one oracle. B=10000, seed=42. Tasks resampled with replacement at task level.
5:**Defuse target**: /stress W1 attack — paper §1 hero claim 'P-SoM 13.81% > SoM 10.48% reddit' is statistically marginal under author's own 2σ hedge.
18:**Pairwise SR difference, bootstrap 95% CI:**
31:**Drop-one oracle on 4-mode set (dom, som, vision, phantom_som), bootstrap 95% CI:**
54:**Pairwise SR difference, bootstrap 95% CI:**
67:**Drop-one oracle on 4-mode set (dom, som, vision, phantom_som), bootstrap 95% CI:**
84:- If both CIs are strict-positive (ci_lo > 0) AND P(diff > 0) > 0.95 → **W1 attack defused**,   §1 hero claim is bootstrap-supported. Remove the '2σ hedge' from line 5, lead with the magnitude.
docs/checkpoints/mechanism/results/layer_axis_emergence.md
3:Eight mode pairs differ in image presence (one side has image, the other doesn't). Peak cosine-gap layer reveals **when** image-axis mechanism emerges:
5:| no-image side | image side | no-img text | peak layer | peak cosine gap |
7:| DOM | SoM | AXTree | **L04** | 0.0604 |
8:| DOM | Vision | AXTree | **L04** | 0.0653 |
9:| P-prompt | SoM | AXTree | **L04** | 0.0600 |
10:| P-prompt | Vision | AXTree | **L04** | 0.0649 |
11:| P-SoM | SoM | [SOM_MARKS] | **L17** | 0.0412 |
13:| P-text | Vision | [SOM_MARKS] | **L36** | 0.0614 |
14:| P-SoM | Vision | [SOM_MARKS] | **L36** | 0.0613 |
18:### no-image text = `AXTree` (mean peak L4)
20:- DOM ↔ SoM: peak **L04** = 0.0604
21:- DOM ↔ Vision: peak **L04** = 0.0653
22:- P-prompt ↔ SoM: peak **L04** = 0.0600
23:- P-prompt ↔ Vision: peak **L04** = 0.0649
25:### no-image text = `[SOM_MARKS]` (mean peak L27)
28:- P-text ↔ Vision: peak **L36** = 0.0614
29:- P-SoM ↔ SoM: peak **L17** = 0.0412
30:- P-SoM ↔ Vision: peak **L36** = 0.0613
34:When the no-image side carries `AXTree` text (DOM, P-prompt), the image-axis cosine gap peaks at **L04** — early-layer fresh image-presence detection (vision encoder + cross-modal fusion).
36:When the no-image side carries `[SOM_MARKS]` text (P-text, P-SoM), the image-axis cosine gap peak shifts to **L17–L36** — image yes/no divergence is deferred to mid/output layers.
38:**Mechanism story**: `[SOM_MARKS]` text in input primes an indexed-parsing pathway through mid-layer computation. The model processes marks structurally regardless of whether image is provided, producing image-axis divergence only at late integration stages. This is the direct Method 4.2 empirical anchor for the **Mirage Effect** (Asadi et al. 2026, VLM ~70-80% no-image accuracy) and **Cross-modal flow** (Kaduri et al., middle-layer cross-modal flows store image info in query tokens): the marks-primed mid-layer computation runs *as if image were available*, with image grounding contributed only late.
42:> *Method 4.2 reveals a peak-layer shift signature for the Mirage Effect: image-axis cosine-gap peak transitions from L04 (when no-image side carries AXTree text) to L17–L36 (when no-image side carries [SOM_MARKS] text). The peak-layer shift quantifies how text-payload format primes mid-layer computation pathways — [SOM_MARKS] format triggers indexed-parsing through mid-layers regardless of image presence, with image-axis divergence deferred to late integration. This identifies [SOM_MARKS] as the mechanism trigger for the Mirage Effect, anchoring Asadi et al. 2026's behavioral finding (~70-80% no-image VLM accuracy) and Kaduri et al.'s middle-layer cross-modal flow hypothesis with layer-resolved empirical evidence.*
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md
1:# W6 feature attribution — H1 reddit 2/6 marks-like L04 peak
3:**Setup**: Qwen3-VL-4B tokenizer (Qwen/Qwen3-VL-4B-Instruct). Each marks-like format variant tokenized on a canonical single-element example (N=1, role=button, label=Submit). First-token character class + marker-fingerprint token count compared between L04-peak and L17-peak subgroups.
5:## Per-variant tokenization
7:| Variant | Peak | Example | n_tok | First token | First char class | Marker fp |
9:| appagent_id | L04 | `id_1: Submit` | 5 | `id` | alphanumeric | 4 (`id·_·1·:`) |
10:| plain_numbered | L04 | `1. Submit` | 3 | `1` | alphanumeric | 2 (`1·.`) |
11:| som_standard | L17 | `[1] button 'Submit'` | 7 | `[` | markup-sigil | 3 (`[·1·]`) |
12:| browser_use_at | L17 | `@1 Submit` | 3 | `@` | markup-sigil | 2 (`@·1`) |
13:| tarsier_typed | L17 | `[B1:button:Submit]` | 7 | `[B` | markup-sigil | 7 (`[B·1·:·button·:·Submit·]`) |
14:| xml_tagged | L17 | `<el_1 role='button'>Submit</el_1>` | 14 | `<` | markup-sigil | 4 (`<·el·_·1`) |
15:| hash_id_control | L04 | `#a3f7 Submit` | 5 | `#a` | markup-sigil | 4 (`#a·3·f·7`) |
16:| plain_sentence | L17 | `Submit` | 1 | `Submit` | alphanumeric | 1 (`Submit`) |
17:| dom | L04 | `button: Submit (AXTree)` | 7 | `button` | alphanumeric | 2 (`button·:`) |
18:| som | L17 | `[1] button 'Submit' (+ image marks)` | 11 | `[` | markup-sigil | 3 (`[·1·]`) |
24:| L04-peak (2) | 2 | 0 | 0 | 0 | 0 |
25:| L17-peak (4) | 0 | 4 | 0 | 0 | 0 |
29:✅ **Hypothesis supported (clean split)**: L04-peak variants both start with alphanumeric tokens (2/2); L17-peak variants start with markup-sigil tokens (4/4).
33:- L04-peak mean marker-fp tokens: 3.00
34:- L17-peak mean marker-fp tokens: 4.00
35:- Δ (L17 − L04): +1.00
37:## Full token sequence per variant (marks-like 6)
39:- **appagent_id** (L04, `id_1: Submit`): 5 tokens: `id` · `_` · `1` · `:` · `ĠSubmit`
40:- **plain_numbered** (L04, `1. Submit`): 3 tokens: `1` · `.` · `ĠSubmit`
41:- **som_standard** (L17, `[1] button 'Submit'`): 7 tokens: `[` · `1` · `]` · `Ġbutton` · `Ġ'` · `Submit` · `'`
42:- **browser_use_at** (L17, `@1 Submit`): 3 tokens: `@` · `1` · `ĠSubmit`
43:- **tarsier_typed** (L17, `[B1:button:Submit]`): 7 tokens: `[B` · `1` · `:` · `button` · `:` · `Submit` · `]`
44:- **xml_tagged** (L17, `<el_1 role='button'>Submit</el_1>`): 14 tokens: `<` · `el` · `_` · `1` · `Ġrole` · `='` · `button` · `'>` · `Submit` · `</` · `el` · `_` · `1` · `>`
48:Within the 6 marks-like variants, the L17 vs L04 split corresponds to whether the variant's first tokens are **markup-sigil tokens** (`[`, `<`, `@`) — which co-occur with HTML / web-agent traces in pretraining and trigger the visual-grounding shortcut at mid layers — versus **plain alphanumeric tokens** (`id`, `1`) — which are common in prose / dictionary listings and behave like AXTree-DOM, peaking early at L04 where the image-axis divergence is freshly observable but not yet routed through the shortcut path.
51:- `hash_id_control` (`#a3f7 Submit`): markup-sigil first token but L04 peak. The `#` sigil alone is not sufficient — the marker must contain an **integer index** (which `#a3f7` does not). This is consistent with prior H2 "integer is the trigger token" framing.
52:- `plain_sentence` (`Submit`): alphanumeric first token but L17 peak. With no list/marker structure at all, the divergence path differs — possibly because the text observation drops to bare labels with no positional anchors, which the model handles via a different late-layer routing (likely commitment without grounding).
54:Together these say: the L17 mid-layer shortcut requires **(a) integer-indexed marker + (b) markup-sigil-leading delimiter**. Either alone fails to trigger it.
56:**Paper §5 implication**: H1's binary 'marks-like vs not' prediction is too coarse. The mechanism trigger is the **conjunction** of integer marker + markup-sigil first token, not the abstract concept of 'indexed list'. Variants like `id_N:` and `N.` are nominally indexed but lack the sigil; `hash_id_control` has the sigil but lacks an integer. Both fail to peak at L17. This refines H1 to **'integer marker + markup-sigil delimiter → triggers shortcut at L17'**, which is testable on additional variants and on a `bare_N` falsifier (drop the bracket from `[N]` and re-extract).
58:**Falsifier (concrete next experiment)**: variant `bare_N` = `N button 'Submit'` (no brackets), which has integer + no sigil. Hypothesis predicts L04 peak. If it peaks L17, hypothesis fails.

codex
The prose is not internally settled: `section5_mechanism.md` uses the 37-state extractor convention (`L0` embedding, `L1-L36` blocks), while Stage 2/3 patching uses 36 hook outputs (`L0-L35` blocks). I’m checking whether downstream analysis scripts compensate for that offset consistently, especially steering and patching significance.
exec
/bin/bash -lc "nl -ba scripts/analysis/stage4_robustness.py | sed -n '1,360p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Stage 4 Robustness Suite — addresses 5 reviewer caveats from Method 4.2.
     3	
     4	Tests:
     5	  A. Label permutation neg control — n=100 perms, peak AUROC distribution
     6	     If real signal: real peak AUROC >> permuted (= 0.5 ± noise)
     7	  B. Per-task cosine gap variance — does L17 peak hold across 24 tasks?
     8	  C. Per-step (step 2 vs step 5) cosine gap difference
     9	  D. Silhouette score per layer (within vs between cluster ratio)
    10	  E. Bootstrap 95% CI for key peak cosine gaps (resample tasks)
    11	
    12	Outputs:
    13	  - results/mechanistic/stage4_multimode_b1_cls/method42_robustness.json
    14	  - docs/checkpoints/stage4_method42_robustness.md
    15	"""
    16	from __future__ import annotations
    17	
    18	import json
    19	from pathlib import Path
    20	
    21	import numpy as np
    22	from sklearn.metrics import roc_auc_score, silhouette_score
    23	
    24	ROOT = Path(__file__).resolve().parents[2]
    25	NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz"
    26	OUT_JSON = ROOT / "results/mechanistic/stage4_multimode_b1_cls/method42_robustness.json"
    27	OUT_MD = ROOT / "docs/checkpoints/stage4_method42_robustness.md"
    28	
    29	MODES = ["dom", "phantom_text", "phantom_prompt", "phantom_som", "som", "vision"]
    30	DISPLAY = {"dom": "DOM", "phantom_text": "P-text", "phantom_prompt": "P-prompt",
    31	           "phantom_som": "P-SoM", "som": "SoM", "vision": "Vision"}
    32	RNG = np.random.default_rng(seed=20260511)
    33	
    34	
    35	def cosine_gap(a: np.ndarray, b: np.ndarray) -> float:
    36	    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    37	
    38	
    39	def pair_auroc(X1: np.ndarray, X2: np.ndarray, L: int) -> float:
    40	    """AUROC at single layer via mean-difference direction projection."""
    41	    c1, c2 = X1[:, L, :].mean(0), X2[:, L, :].mean(0)
    42	    direction = (c1 - c2) / (np.linalg.norm(c1 - c2) + 1e-9)
    43	    s1 = X1[:, L, :] @ direction
    44	    s2 = X2[:, L, :] @ direction
    45	    y = np.concatenate([np.ones(len(s1)), np.zeros(len(s2))])
    46	    s = np.concatenate([s1, s2])
    47	    try:
    48	        return roc_auc_score(y, s)
    49	    except Exception:
    50	        return 0.5
    51	
    52	
    53	def test_a_label_permutation(states: dict, n_perm: int = 100, layer: int = 17) -> dict:
    54	    """For P-SoM↔SoM at L17, recompute AUROC under random label shuffles."""
    55	    X1, X2 = states["phantom_som"], states["som"]
    56	    real = pair_auroc(X1, X2, layer)
    57	    pooled = np.concatenate([X1, X2])
    58	    n1 = len(X1)
    59	    perm_aurocs = []
    60	    for _ in range(n_perm):
    61	        idx = RNG.permutation(len(pooled))
    62	        Xp1 = pooled[idx[:n1]]
    63	        Xp2 = pooled[idx[n1:]]
    64	        perm_aurocs.append(pair_auroc(Xp1, Xp2, layer))
    65	    perm = np.array(perm_aurocs)
    66	    pval = float((perm >= real).sum() + 1) / (n_perm + 1)
    67	    return {
    68	        "real_auroc": real,
    69	        "perm_mean": float(perm.mean()),
    70	        "perm_std": float(perm.std()),
    71	        "perm_p25": float(np.percentile(perm, 25)),
    72	        "perm_p975": float(np.percentile(perm, 97.5)),
    73	        "p_value": pval,
    74	        "n_perm": n_perm,
    75	        "layer": layer,
    76	    }
    77	
    78	
    79	def test_b_per_task_cosine_gap(states: dict, task_ids: np.ndarray, mode_labels: np.ndarray,
    80	                                 step_indices: np.ndarray, layer: int = 17) -> dict:
    81	    """For P-SoM↔SoM at L17, compute cosine gap separately per task. n_unique_tasks."""
    82	    pairs = [("phantom_som", "som"), ("phantom_som", "phantom_text"),
    83	              ("phantom_som", "phantom_prompt"), ("dom", "phantom_prompt")]
    84	    results = {}
    85	    unique_tasks = sorted(set(task_ids.tolist()))
    86	    for m1, m2 in pairs:
    87	        per_task = []
    88	        for tid in unique_tasks:
    89	            m1_mask = (mode_labels == m1) & (task_ids == tid)
    90	            m2_mask = (mode_labels == m2) & (task_ids == tid)
    91	            X1 = states[m1][np.in1d(states["__indices__"][m1], np.where(m1_mask)[0])]
    92	            X2 = states[m2][np.in1d(states["__indices__"][m2], np.where(m2_mask)[0])]
    93	            # Simpler: use raw hidden_states with masks
    94	            pass
    95	        # Simpler approach using the original arrays — see below
    96	        results[f"{m1}_vs_{m2}"] = {"task_mean_skip": True}
    97	    return results
    98	
    99	
   100	def test_b_per_task_simple(H: np.ndarray, task_ids: np.ndarray, mode_labels: np.ndarray,
   101	                            layer: int = 17) -> dict:
   102	    """Per-task cosine gap (step-averaged) for 4 key pairs."""
   103	    pairs = [("phantom_som", "som"), ("phantom_som", "phantom_text"),
   104	              ("phantom_som", "phantom_prompt"), ("dom", "phantom_prompt"),
   105	              ("phantom_som", "dom"), ("phantom_text", "som")]
   106	    out = {}
   107	    unique_tasks = sorted(set(task_ids.tolist()))
   108	    for m1, m2 in pairs:
   109	        per_task_gaps = []
   110	        for tid in unique_tasks:
   111	            mask1 = (mode_labels == m1) & (task_ids == tid)
   112	            mask2 = (mode_labels == m2) & (task_ids == tid)
   113	            if mask1.sum() == 0 or mask2.sum() == 0:
   114	                continue
   115	            c1 = H[mask1, layer, :].mean(0)
   116	            c2 = H[mask2, layer, :].mean(0)
   117	            per_task_gaps.append(cosine_gap(c1, c2))
   118	        arr = np.array(per_task_gaps)
   119	        out[f"{m1}_vs_{m2}"] = {
   120	            "n_tasks": len(arr),
   121	            "mean": float(arr.mean()),
   122	            "std": float(arr.std()),
   123	            "min": float(arr.min()),
   124	            "max": float(arr.max()),
   125	            "fraction_positive": float((arr > 0).mean()),
   126	        }
   127	    return out
   128	
   129	
   130	def test_c_per_step(H: np.ndarray, mode_labels: np.ndarray, step_indices: np.ndarray,
   131	                     layer: int = 17) -> dict:
   132	    """Cosine gap at L17 separated by step (2 vs 5)."""
   133	    pairs = [("phantom_som", "som"), ("phantom_som", "phantom_text"),
   134	              ("phantom_som", "phantom_prompt")]
   135	    out = {}
   136	    for m1, m2 in pairs:
   137	        per_step = {}
   138	        for step in [2, 5]:
   139	            m1_mask = (mode_labels == m1) & (step_indices == step)
   140	            m2_mask = (mode_labels == m2) & (step_indices == step)
   141	            c1 = H[m1_mask, layer, :].mean(0)
   142	            c2 = H[m2_mask, layer, :].mean(0)
   143	            per_step[f"step{step}"] = {
   144	                "n_per_mode": int(m1_mask.sum()),
   145	                "cosine_gap": cosine_gap(c1, c2),
   146	            }
   147	        out[f"{m1}_vs_{m2}"] = per_step
   148	    return out
   149	
   150	
   151	def test_d_silhouette(H: np.ndarray, mode_labels: np.ndarray,
   152	                       layers: tuple = (4, 11, 17, 23, 30, 36)) -> dict:
   153	    """Silhouette score per layer (higher = cleaner mode separation)."""
   154	    label_idx = {m: i for i, m in enumerate(MODES)}
   155	    y = np.array([label_idx[m] for m in mode_labels.tolist()])
   156	    out = {}
   157	    for L in layers:
   158	        try:
   159	            X = H[:, L, :]
   160	            # Skip if any feature is constant (e.g., L0 embedding zeros)
   161	            if X.std() < 1e-6:
   162	                out[f"L{L:02d}"] = {"silhouette": None, "reason": "constant features"}
   163	                continue
   164	            s = float(silhouette_score(X, y, metric="cosine"))
   165	            out[f"L{L:02d}"] = {"silhouette": s}
   166	        except Exception as e:
   167	            out[f"L{L:02d}"] = {"silhouette": None, "error": str(e)}
   168	    return out
   169	
   170	
   171	def test_e_bootstrap_ci(H: np.ndarray, task_ids: np.ndarray, mode_labels: np.ndarray,
   172	                         layer: int = 17, n_boot: int = 1000) -> dict:
   173	    """Bootstrap 95% CI for cosine gap by resampling tasks."""
   174	    pairs = [("phantom_som", "som"), ("phantom_som", "phantom_text"),
   175	              ("phantom_som", "phantom_prompt"), ("dom", "phantom_prompt"),
   176	              ("dom", "vision")]
   177	    unique_tasks = np.array(sorted(set(task_ids.tolist())))
   178	    out = {}
   179	    for m1, m2 in pairs:
   180	        gaps = []
   181	        for _ in range(n_boot):
   182	            boot_tids = RNG.choice(unique_tasks, size=len(unique_tasks), replace=True)
   183	            X1 = np.concatenate([H[(mode_labels == m1) & (task_ids == t)] for t in boot_tids])
   184	            X2 = np.concatenate([H[(mode_labels == m2) & (task_ids == t)] for t in boot_tids])
   185	            c1 = X1[:, layer, :].mean(0)
   186	            c2 = X2[:, layer, :].mean(0)
   187	            gaps.append(cosine_gap(c1, c2))
   188	        arr = np.array(gaps)
   189	        out[f"{m1}_vs_{m2}"] = {
   190	            "mean": float(arr.mean()),
   191	            "ci_2.5": float(np.percentile(arr, 2.5)),
   192	            "ci_97.5": float(np.percentile(arr, 97.5)),
   193	            "n_boot": n_boot,
   194	        }
   195	    return out
   196	
   197	
   198	def main() -> None:
   199	    d = np.load(NPZ, allow_pickle=True)
   200	    H = d["hidden_states"]
   201	    mode_labels = d["mode_labels_str"]
   202	    task_ids = d["task_ids"]
   203	    step_indices = d["step_indices"]
   204	    print(f"[robustness] loaded {H.shape}, {len(set(task_ids.tolist()))} tasks, steps={sorted(set(step_indices.tolist()))}")
   205	
   206	    states = {m: H[mode_labels == m] for m in MODES}
   207	
   208	    print("[robustness] Test A: label permutation neg control...")
   209	    A = test_a_label_permutation(states, n_perm=200)
   210	    print(f"  P-SoM↔SoM L17 real AUROC = {A['real_auroc']:.3f}, perm mean = {A['perm_mean']:.3f} ± {A['perm_std']:.3f}, p = {A['p_value']:.4f}")
   211	
   212	    print("[robustness] Test B: per-task cosine gap variance...")
   213	    B = test_b_per_task_simple(H, task_ids, mode_labels, layer=17)
   214	    for k, v in B.items():
   215	        print(f"  {k}: mean={v['mean']:.4f} ± {v['std']:.4f}, range [{v['min']:.4f}, {v['max']:.4f}], +sign {v['fraction_positive']:.0%}")
   216	
   217	    print("[robustness] Test C: per-step comparison...")
   218	    C = test_c_per_step(H, mode_labels, step_indices, layer=17)
   219	    for k, v in C.items():
   220	        print(f"  {k}: step2={v['step2']['cosine_gap']:.4f} (n={v['step2']['n_per_mode']}), step5={v['step5']['cosine_gap']:.4f} (n={v['step5']['n_per_mode']})")
   221	
   222	    print("[robustness] Test D: silhouette across layers...")
   223	    D = test_d_silhouette(H, mode_labels)
   224	    for k, v in D.items():
   225	        s = v.get("silhouette")
   226	        print(f"  {k}: silhouette = {s:.4f}" if s is not None else f"  {k}: skipped ({v.get('reason', v.get('error'))})")
   227	
   228	    print("[robustness] Test E: bootstrap 95% CI (n=1000)...")
   229	    E = test_e_bootstrap_ci(H, task_ids, mode_labels, layer=17, n_boot=1000)
   230	    for k, v in E.items():
   231	        print(f"  {k}: {v['mean']:.4f} [{v['ci_2.5']:.4f}, {v['ci_97.5']:.4f}]")
   232	
   233	    out = {"test_A_permutation": A, "test_B_per_task": B,
   234	            "test_C_per_step": C, "test_D_silhouette": D, "test_E_bootstrap": E}
   235	    OUT_JSON.write_text(json.dumps(out, indent=2))
   236	    print(f"[robustness] metrics → {OUT_JSON}")
   237	
   238	    write_md(out, OUT_MD)
   239	
   240	
   241	def write_md(o: dict, out: Path) -> None:
   242	    A, B, C, D, E = o["test_A_permutation"], o["test_B_per_task"], o["test_C_per_step"], o["test_D_silhouette"], o["test_E_bootstrap"]
   243	    lines = ["# Stage 4 Robustness Suite (Method 4.2 caveat coverage)", ""]
   244	
   245	    lines.append("## Test A: Label Permutation Negative Control")
   246	    lines.append("")
   247	    lines.append("P-SoM↔SoM at L17 — does AUROC=1.000 survive random label shuffles?")
   248	    lines.append("")
   249	    lines.append(f"- **Real AUROC** (true labels): **{A['real_auroc']:.4f}**")
   250	    lines.append(f"- **Permuted AUROC** (n={A['n_perm']} random shuffles): mean = {A['perm_mean']:.4f} ± {A['perm_std']:.4f}")
   251	    lines.append(f"- **95% CI of perm**: [{A['perm_p25']:.4f}, {A['perm_p975']:.4f}]")
   252	    lines.append(f"- **p-value**: {A['p_value']:.4f}")
   253	    lines.append("")
   254	    real_z = (A['real_auroc'] - A['perm_mean']) / (A['perm_std'] + 1e-9)
   255	    lines.append(f"→ Real signal is **{real_z:.1f}σ above permutation baseline**. Cosine-gap AUROC is NOT achievable from random label noise.")
   256	    lines.append("")
   257	
   258	    lines.append("## Test B: Per-Task Cosine Gap Variance")
   259	    lines.append("")
   260	    lines.append("Mean (cosine gap) computed separately per (task × step pair) and aggregated over 24 tasks at L17:")
   261	    lines.append("")
   262	    lines.append("| Mode pair | n tasks | Mean gap | Std | Range | % tasks with positive gap |")
   263	    lines.append("|---|---|---|---|---|---|")
   264	    for k, v in B.items():
   265	        m1, m2 = k.split("_vs_")
   266	        lines.append(f"| {DISPLAY[m1]} vs {DISPLAY[m2]} | {v['n_tasks']} | {v['mean']:.4f} | {v['std']:.4f} | [{v['min']:.4f}, {v['max']:.4f}] | {v['fraction_positive']:.0%} |")
   267	    lines.append("")
   268	
   269	    lines.append("## Test C: Per-Step Comparison (step 2 vs step 5)")
   270	    lines.append("")
   271	    lines.append("| Mode pair | Step 2 gap | Step 5 gap |")
   272	    lines.append("|---|---|---|")
   273	    for k, v in C.items():
   274	        m1, m2 = k.split("_vs_")
   275	        lines.append(f"| {DISPLAY[m1]} vs {DISPLAY[m2]} | {v['step2']['cosine_gap']:.4f} | {v['step5']['cosine_gap']:.4f} |")
   276	    lines.append("")
   277	
   278	    lines.append("## Test D: Silhouette Score Across Layers")
   279	    lines.append("")
   280	    lines.append("Silhouette = (between-cluster - within-cluster) / max, range [-1, 1]. Higher = cleaner mode separation.")
   281	    lines.append("")
   282	    lines.append("| Layer | Silhouette |")
   283	    lines.append("|---|---|")
   284	    for k, v in D.items():
   285	        s = v.get("silhouette")
   286	        lines.append(f"| {k} | {s:.4f} |" if s is not None else f"| {k} | skipped |")
   287	    lines.append("")
   288	
   289	    lines.append("## Test E: Bootstrap 95% CI (n=1000, task-level resample)")
   290	    lines.append("")
   291	    lines.append("| Mode pair | Mean | 95% CI |")
   292	    lines.append("|---|---|---|")
   293	    for k, v in E.items():
   294	        m1, m2 = k.split("_vs_")
   295	        lines.append(f"| {DISPLAY[m1]} vs {DISPLAY[m2]} | {v['mean']:.4f} | [{v['ci_2.5']:.4f}, {v['ci_97.5']:.4f}] |")
   296	    lines.append("")
   297	
   298	    out.write_text("\n".join(lines) + "\n")
   299	    print(f"[robustness] summary → {out}")
   300	
   301	
   302	if __name__ == "__main__":
   303	    main()

exec
/bin/bash -lc "nl -ba scripts/analysis/stage4_axis2_per_task_fragility.py | sed -n '1,300p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Axis-2 per-task fragility check (response to /stress W2 attack).
     3	
     4	The /stress reviewer (2026-05-12) attacked the §5.7 three-axis hierarchy
     5	on grounds that the axis-2 cosine gap mean of 0.0114 is reported without
     6	per-task variance check. By analogy to h1_per_task_fragility (which found
     7	H1 dichotomy holds for only 11% of (task, step) pairs strict), axis-2 at
     8	roughly one-third the magnitude is plausibly dominated by 2-3 outlier
     9	tasks.
    10	
    11	This script computes per-task cosine gap distribution at L23 for:
    12	  - Axis-2 pair: P-text vs P-SoM (flat-text, prompt swap) — main test
    13	  - Axis-2 pair: DOM vs P-prompt (hierarchical, prompt swap) — secondary
    14	  - Axis-1 pair: DOM vs P-text (DOM-prompt, text swap) — magnitude reference
    15	  - Axis-3 pair: P-SoM vs SoM (image-axis reference) — calibration scale
    16	
    17	For each task: average mode hidden states across the 2 steps the task
    18	contributes, then compute cosine gap between the two pair modes at L23.
    19	Report median, IQR, fraction-above-threshold (0.005 and 0.010), top/bottom
    20	5 tasks.
    21	
    22	Outputs:
    23	  - docs/checkpoints/mechanism/results/axis2_per_task_fragility.md
    24	  - results/phantom_paper/figures/fig_axis2_per_task_fragility.png
    25	"""
    26	from __future__ import annotations
    27	
    28	import argparse
    29	from pathlib import Path
    30	
    31	import matplotlib.pyplot as plt
    32	import numpy as np
    33	
    34	ROOT = Path(__file__).resolve().parents[2]
    35	DEFAULT_CLS_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz"
    36	DEFAULT_RED_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz"
    37	DEFAULT_MD = ROOT / "docs/checkpoints/mechanism/results/axis2_per_task_fragility.md"
    38	DEFAULT_FIG = ROOT / "results/phantom_paper/figures/fig_axis2_per_task_fragility.png"
    39	
    40	# (mode_a, mode_b, label, axis, color)
    41	PAIRS = [
    42	    ("phantom_text", "phantom_som",  "P-text ↔ P-SoM   (axis-2 flat-text)",   "axis-2", "#d62728"),
    43	    ("dom",          "phantom_prompt","DOM ↔ P-prompt  (axis-2 hierarchical)", "axis-2", "#ff7f0e"),
    44	    ("dom",          "phantom_text",  "DOM ↔ P-text     (axis-1 reference)",   "axis-1", "#1f77b4"),
    45	    ("phantom_som",  "som",           "P-SoM ↔ SoM     (axis-3 image ref)",    "axis-3", "#9467bd"),
    46	]
    47	
    48	L_TARGET = 23  # paper §5.7 axis-2 peak layer
    49	
    50	
    51	def cosine_gap(a: np.ndarray, b: np.ndarray) -> float:
    52	    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    53	
    54	
    55	def compute_per_task_cosine(npz_path: Path, layer: int):
    56	    """For each task ID, compute mode mean hidden state across steps, then
    57	    cosine gap between mode pairs.
    58	
    59	    Returns dict[pair_label] = {task_id: cosine_gap}
    60	    """
    61	    d = np.load(npz_path, allow_pickle=True)
    62	    H = d["hidden_states"]  # (N, L, D)
    63	    ml = d["mode_labels_str"]
    64	    tids = d["task_ids"]
    65	
    66	    unique_tasks = sorted(set(int(t) for t in tids))
    67	    unique_modes = sorted(set(ml.tolist()))
    68	
    69	    # For each (task, mode), average hidden state across steps at target layer
    70	    task_mode_mean = {}  # (task_id, mode) -> hidden (D,)
    71	    for t in unique_tasks:
    72	        for m in unique_modes:
    73	            mask = (tids == t) & (ml == m)
    74	            if mask.sum() == 0:
    75	                continue
    76	            task_mode_mean[(t, m)] = H[mask, layer].mean(axis=0)
    77	
    78	    per_pair = {}
    79	    for a, b, label, axis, _color in PAIRS:
    80	        per_task = {}
    81	        for t in unique_tasks:
    82	            if (t, a) in task_mode_mean and (t, b) in task_mode_mean:
    83	                per_task[t] = cosine_gap(task_mode_mean[(t, a)], task_mode_mean[(t, b)])
    84	        per_pair[label] = {"per_task": per_task, "axis": axis}
    85	
    86	    return per_pair, unique_tasks
    87	
    88	
    89	def summarize(per_pair: dict, layer: int):
    90	    """For each pair, compute median, IQR, fraction-above-threshold."""
    91	    summary = {}
    92	    for label, info in per_pair.items():
    93	        vals = np.array(list(info["per_task"].values()))
    94	        n = len(vals)
    95	        summary[label] = {
    96	            "axis": info["axis"],
    97	            "n": n,
    98	            "mean": float(vals.mean()),
    99	            "median": float(np.median(vals)),
   100	            "std": float(vals.std()),
   101	            "p25": float(np.percentile(vals, 25)),
   102	            "p75": float(np.percentile(vals, 75)),
   103	            "p10": float(np.percentile(vals, 10)),
   104	            "p90": float(np.percentile(vals, 90)),
   105	            "min": float(vals.min()),
   106	            "max": float(vals.max()),
   107	            "frac_gt_005": float((vals > 0.005).mean()),
   108	            "frac_gt_010": float((vals > 0.010).mean()),
   109	            "frac_gt_020": float((vals > 0.020).mean()),
   110	            "per_task_sorted": sorted(info["per_task"].items(), key=lambda kv: kv[1], reverse=True),
   111	        }
   112	    return summary
   113	
   114	
   115	def write_md(cls_sum: dict, red_sum: dict, layer: int, out: Path):
   116	    out.parent.mkdir(parents=True, exist_ok=True)
   117	    lines = [
   118	        "# Axis-2 per-task fragility check",
   119	        "",
   120	        f"Per-task cosine gap distribution at L{layer} (axis-2 peak per §5.7 / Exp 1).",
   121	        f"Each task averaged across its 2 steps; cosine gap computed between mode pairs.",
   122	        "",
   123	        "**Defuse target**: /stress W2 attack — axis-2 mean 0.0114 might be dominated by 2-3 outlier tasks.",
   124	        "",
   125	        "## Classifieds (24 tasks)",
   126	        "",
   127	        "| Pair | Axis | Mean | Median | IQR | min | max | % > 0.005 | % > 0.010 | % > 0.020 |",
   128	        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
   129	    ]
   130	    for label, s in cls_sum.items():
   131	        iqr = f"[{s['p25']:.4f}, {s['p75']:.4f}]"
   132	        lines.append(
   133	            f"| {label} | {s['axis']} | {s['mean']:.4f} | {s['median']:.4f} | {iqr} | "
   134	            f"{s['min']:.4f} | {s['max']:.4f} | "
   135	            f"{s['frac_gt_005']:.0%} | {s['frac_gt_010']:.0%} | {s['frac_gt_020']:.0%} |"
   136	        )
   137	    lines += [
   138	        "",
   139	        "## Reddit (24 tasks)",
   140	        "",
   141	        "| Pair | Axis | Mean | Median | IQR | min | max | % > 0.005 | % > 0.010 | % > 0.020 |",
   142	        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
   143	    ]
   144	    for label, s in red_sum.items():
   145	        iqr = f"[{s['p25']:.4f}, {s['p75']:.4f}]"
   146	        lines.append(
   147	            f"| {label} | {s['axis']} | {s['mean']:.4f} | {s['median']:.4f} | {iqr} | "
   148	            f"{s['min']:.4f} | {s['max']:.4f} | "
   149	            f"{s['frac_gt_005']:.0%} | {s['frac_gt_010']:.0%} | {s['frac_gt_020']:.0%} |"
   150	        )
   151	
   152	    # Top/bottom 5 tasks for the main axis-2 pair
   153	    main_label = next(k for k in cls_sum.keys() if "P-text ↔ P-SoM" in k)
   154	    for site_name, summ in [("classifieds", cls_sum), ("reddit", red_sum)]:
   155	        s = summ[main_label]
   156	        lines += [
   157	            "",
   158	            f"## Top 5 axis-2 tasks ({site_name}, P-text ↔ P-SoM @ L23)",
   159	            "",
   160	            "| Task ID | Cosine gap |",
   161	            "|---|---:|",
   162	        ]
   163	        for tid, val in s["per_task_sorted"][:5]:
   164	            lines.append(f"| {tid} | {val:.4f} |")
   165	        lines += [
   166	            "",
   167	            f"## Bottom 5 axis-2 tasks ({site_name}, P-text ↔ P-SoM @ L23)",
   168	            "",
   169	            "| Task ID | Cosine gap |",
   170	            "|---|---:|",
   171	        ]
   172	        for tid, val in s["per_task_sorted"][-5:]:
   173	            lines.append(f"| {tid} | {val:.4f} |")
   174	
   175	    lines += [
   176	        "",
   177	        "## Verdict",
   178	        "",
   179	        f"Read the `% > 0.010` column for the axis-2 P-text↔P-SoM pair:",
   180	        f"- cls: **{cls_sum[main_label]['frac_gt_010']:.0%}** of 24 tasks above the L23 axis-2 mean magnitude",
   181	        f"- reddit: **{red_sum[main_label]['frac_gt_010']:.0%}** of 24 tasks above",
   182	        "",
   183	        f"Interpretation tree:",
   184	        f"- If both ≥ 50% → axis-2 signal **broad**, /stress W2 attack defused, §5.7 framing OK",
   185	        f"- If both 25-50% → axis-2 signal **modest but present**, §5.7 needs to add 'task-conditional sparse' qualifier",
   186	        f"- If both < 25% → axis-2 signal **aggregate artifact**, §5.7 three-axis claim must downgrade to 'axis-1 + image-axis with axis-2 weak per-task'",
   187	        "",
   188	        f"Median values: cls={cls_sum[main_label]['median']:.4f}, reddit={red_sum[main_label]['median']:.4f}.",
   189	        f"Compare to mean: cls={cls_sum[main_label]['mean']:.4f}, reddit={red_sum[main_label]['mean']:.4f}.",
   190	        f"If median << mean, the distribution is right-skewed → outlier-driven (consistent with /stress W2 attack).",
   191	    ]
   192	    out.write_text("\n".join(lines) + "\n")
   193	    print(f"summary → {out}")
   194	
   195	
   196	def plot(cls_sum: dict, red_sum: dict, layer: int, cls_per_pair: dict, red_per_pair: dict, out: Path):
   197	    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
   198	    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
   199	
   200	    for col_idx, (site, summ, per_pair) in enumerate([
   201	        ("classifieds", cls_sum, cls_per_pair),
   202	        ("reddit", red_sum, red_per_pair),
   203	    ]):
   204	        # Top row: histograms per pair
   205	        ax_h = axes[0, col_idx]
   206	        for label, info in per_pair.items():
   207	            vals = np.array(list(info["per_task"].values()))
   208	            color = {
   209	                "axis-1": "#1f77b4",
   210	                "axis-2": "#d62728" if "flat" in label else "#ff7f0e",
   211	                "axis-3": "#9467bd",
   212	            }[info["axis"]]
   213	            ax_h.hist(vals, bins=20, alpha=0.6, label=label, color=color)
   214	        ax_h.axvline(0.005, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="0.005 threshold")
   215	        ax_h.axvline(0.010, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="0.010 threshold")
   216	        ax_h.set_xlabel(f"Cosine gap at L{layer}")
   217	        ax_h.set_ylabel("Tasks")
   218	        ax_h.set_title(f"{site}: per-task cosine gap distribution")
   219	        ax_h.legend(fontsize=7, loc="upper right")
   220	        ax_h.grid(True, alpha=0.3)
   221	
   222	        # Bottom row: per-pair box+swarm
   223	        ax_b = axes[1, col_idx]
   224	        labels = list(per_pair.keys())
   225	        data = [list(per_pair[l]["per_task"].values()) for l in labels]
   226	        bp = ax_b.boxplot(data, vert=True, patch_artist=True, labels=[l.split("(")[0].strip() for l in labels])
   227	        for patch, label in zip(bp["boxes"], labels):
   228	            axis = per_pair[label]["axis"]
   229	            patch.set_facecolor({
   230	                "axis-1": "#1f77b4",
   231	                "axis-2": "#d62728" if "flat" in label else "#ff7f0e",
   232	                "axis-3": "#9467bd",
   233	            }[axis])
   234	            patch.set_alpha(0.6)
   235	        # overlay individual task points
   236	        for i, vals in enumerate(data):
   237	            jitter = np.random.normal(0, 0.04, size=len(vals))
   238	            ax_b.scatter(np.full(len(vals), i + 1) + jitter, vals, color="k", s=8, alpha=0.5)
   239	        ax_b.axhline(0.010, color="gray", linestyle="--", linewidth=1, alpha=0.5)
   240	        ax_b.set_ylabel(f"Cosine gap at L{layer}")
   241	        ax_b.set_title(f"{site}: per-task box + swarm")
   242	        ax_b.grid(True, alpha=0.3)
   243	        plt.setp(ax_b.get_xticklabels(), rotation=15, ha="right")
   244	
   245	    fig.suptitle("Axis-2 per-task fragility check (/stress W2 defuse target)", fontsize=11)
   246	    fig.tight_layout()
   247	    out.parent.mkdir(parents=True, exist_ok=True)
   248	    fig.savefig(out)
   249	    print(f"figure → {out}")
   250	
   251	
   252	def main():
   253	    p = argparse.ArgumentParser()
   254	    p.add_argument("--cls-npz", type=Path, default=DEFAULT_CLS_NPZ)
   255	    p.add_argument("--red-npz", type=Path, default=DEFAULT_RED_NPZ)
   256	    p.add_argument("--layer", type=int, default=L_TARGET)
   257	    p.add_argument("--output-md", type=Path, default=DEFAULT_MD)
   258	    p.add_argument("--output-fig", type=Path, default=DEFAULT_FIG)
   259	    args = p.parse_args()
   260	
   261	    np.random.seed(0)
   262	    print(f"Loading cls: {args.cls_npz}")
   263	    cls_per_pair, cls_tasks = compute_per_task_cosine(args.cls_npz, args.layer)
   264	    print(f"  {len(cls_per_pair)} pairs, {len(cls_tasks)} tasks")
   265	
   266	    print(f"Loading reddit: {args.red_npz}")
   267	    red_per_pair, red_tasks = compute_per_task_cosine(args.red_npz, args.layer)
   268	    print(f"  {len(red_per_pair)} pairs, {len(red_tasks)} tasks")
   269	
   270	    cls_sum = summarize(cls_per_pair, args.layer)
   271	    red_sum = summarize(red_per_pair, args.layer)
   272	
   273	    write_md(cls_sum, red_sum, args.layer, args.output_md)
   274	    plot(cls_sum, red_sum, args.layer, cls_per_pair, red_per_pair, args.output_fig)
   275	
   276	
   277	if __name__ == "__main__":
   278	    main()

exec
/bin/bash -lc 'rg -n "L0 = embedding|embedding|block outputs|L0-L35|L0-L36|L36|final block|canonical grid|37|36|Layer index|hook_position|first_forward|random_seed|git_rev|sha|manifest" docs/checkpoints/mechanism docs/checkpoints/paper_drafts scripts p79/mechanistic | head -n 200' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
p79/mechanistic/activation_patching.py:31:        model.model.language_model.layers — 36 × Qwen3VLTextDecoderLayer
p79/mechanistic/activation_patching.py:41:    output of the first decoder layer, NOT the embedding output. To get
p79/mechanistic/activation_patching.py:42:    embedding output, hook `model.model.language_model.embed_tokens`
p79/mechanistic/activation_patching.py:44:    to avoid confusion with the embedding.
p79/mechanistic/activation_patching.py:121:        input_len = inputs["input_ids"].shape[1]
p79/mechanistic/activation_patching.py:177:        input_len = inputs["input_ids"].shape[1]
p79/mechanistic/activation_patching.py:211:                if hs.shape != src.shape:
p79/mechanistic/activation_patching.py:213:                        f"shape mismatch for position='all': target {hs.shape} vs source {src.shape}"
p79/mechanistic/activation_patching.py:389:    src_input_len = source_inputs["input_ids"].shape[1]
p79/mechanistic/activation_patching.py:398:    tgt_input_len = target_inputs["input_ids"].shape[1]
docs/checkpoints/paper_drafts/section5_mechanism.md:7:This section is the Zoom-4 layer of the paper's four-level account. Zoom 1 defines the architectural intervention, "skip the annotated image"; Zoom 2 measures the behavioral axes of text payload, prompt family, and image presence; Zoom 3 links the observed behavior to Mirage-style no-image visual reasoning and prompt-format sensitivity; Zoom 4 asks where the resulting mode is represented and whether it is causally used by the model. We index layers L0-L36, where L0 is the embedding-block output and L1-L36 are the 36 transformer decoder block outputs.
docs/checkpoints/paper_drafts/section5_mechanism.md:28:Method 4.2 extracts hidden states from Qwen3-VL-4B B1 runs and compares observation modes by layer. For each mode pair and layer, we compute the cosine gap between hidden-state means, evaluate AUROC by projecting examples onto the mean-difference direction, and summarize per-mode geometry through PCA top-10 variance. The classifieds baseline contains 288 examples, formed from 24 strong-tier tasks, two archived steps, and six modes, over 37 indexed layers.
docs/checkpoints/paper_drafts/section5_mechanism.md:59:Each patched continuation is scored against the unpatched source and target continuations. The main disruption statistic is the drop in `token_overlap_to_target`; Levenshtein distance to target is the paired backup. Layer-wise tests compare each grid layer to the final-layer reference using task-paired differences and Holm-Bonferroni correction across the canonical grid. Random-injection controls replace source hidden states with Gaussian tensors matched to source activation mean and standard deviation.
docs/checkpoints/paper_drafts/section5_mechanism.md:75:Stage 3 extends this from P-SoM to the three no-image arms, testing whether the image-feature axis is shared across DOM, P-text, and P-prompt targets. The table below reports per-task-paired Δoverlap-to-target from the patching_continuation_results.json under each cell directory, with the layer at which the disruption peaks.
docs/checkpoints/paper_drafts/section5_mechanism.md:80:| reddit | -0.335 at L11, -0.255 at L17, -0.338 at L14 (best) | -0.244 at L11, -0.236 at L17, -0.330 at L15 (best) | -0.233 at L11, -0.191 at L17, -0.322 at L14 (best) | [-0.322, -0.338] |
docs/checkpoints/paper_drafts/section5_mechanism.md:82:All six Stage 3 cells are now closed. Two observations carry the cross-site claim. First, every cell's best layer falls inside the L12-L18 mid-layer window, and every cell's best Δoverlap-to-target is between -0.27 and -0.35. The mid-layer fusion locus is therefore not a single layer index but a tight 7-layer window that transfers across cls and reddit. Second, the interpretation is additive rather than arm-specific: a SoM source state displaces DOM, P-text, and P-prompt targets toward the source with similar magnitude, implying a shared image-feature substrate across all three no-image arms. The negative controls, Cell E at -0.03 and Cell Er near zero, rule out a generic nonzero-injection explanation.
docs/checkpoints/paper_drafts/section5_mechanism.md:86:The cleanest single-pair signature is the image-axis peak-layer dichotomy. Across eight image-presence contrasts, the no-image side's text format predicts the peak layer with zero overlap. If the no-image side is AXTree text, the image-axis cosine gap peaks at L04 in all four pairs: DOM<->Vision, DOM<->SoM, P-prompt<->Vision, and P-prompt<->SoM. If the no-image side is `[SOM_MARKS]` or another flat marks text, the peak shifts to L17-L36 in all four pairs: P-text<->Vision, P-text<->SoM, P-SoM<->Vision, and P-SoM<->SoM.
docs/checkpoints/paper_drafts/section5_mechanism.md:90:The format-variation grid contains ten modes: six marks-like variants, two controls, and DOM/SoM baselines. In the classifieds strong-tier baseline, all six marks-like variants peak at L36, the hash-ID control also peaks at L36, the plain-sentence control peaks at L17, and the DOM baseline preserves the L04 peak. Because L36 is the boundary layer, this is best read as a strong late/monotonic signature rather than as a precise late-layer mechanism.
docs/checkpoints/paper_drafts/section5_mechanism.md:92:The classifieds reverse-tier run reproduces the strong-tier shape. The six marks-like variants and hash-ID control again peak at L36, the plain-sentence control moves to L22, and DOM remains at L04. This defends H1 against the selection-bias concern that strong-tier curation alone created the pattern.
docs/checkpoints/paper_drafts/section5_mechanism.md:94:The reddit format run is cleaner for the mid-layer interpretation. Four of six marks-like variants peak at L17, the plain-sentence control peaks at L17, hash-ID control returns to L04, and DOM remains at L04. **W6 attribution** (`docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md`, exploratory) further finds that the two L04 marks-like variants (`appagent_id`, `plain_numbered`) share a feature with the L04 DOM baseline: their first tokens are alphanumeric, while the four L17-peaking marks-like variants all start with markup-sigil tokens (`[`, `<`, `@`). The hash-ID control (`#a3f7`) starts with a sigil but lacks integer-marker structure and also peaks at L04, suggesting the trigger conjunction is integer-marker + markup-sigil rather than either alone. This is a post-hoc feature-attribution on a small (N=6 marks-like) format set; held-out falsifiers (`bare_N` = integer without sigil, `bracket_no_int` = sigil without integer) are not yet run. Cross-site, the safe claim is directional: marker formats that combine integer indexing with markup-sigil leading delimiters tend to delay image-axis separation into mid/late layers, while AXTree hierarchy and indexed-list variants lacking either feature preserve the early L04 image-axis peak. The reddit curve reveals the true L11-L17 fusion locus more clearly than the classifieds L36 boundary artifact.
docs/checkpoints/paper_drafts/section5_mechanism.md:100:The second defense is selection-bias robustness. The classifieds reverse-tier run replicates the strong-tier H1 pattern, including L36 marks-like peaks and L04 DOM baseline. The shortcut signature is therefore not an artifact of selecting tasks where SoM beats DOM.
docs/checkpoints/paper_drafts/section5_mechanism.md:102:The third defense is cross-site H1. Reddit does not reproduce the exact boundary-layer shape, but it reproduces the direction of the indexed-list shortcut with a cleaner L17 mid-layer peak for four of six marks-like formats. The site changes the curve shape, not the basic interpretation.
docs/checkpoints/paper_drafts/section5_mechanism.md:110:A naive reading of Method 4.2's L17 snapshot suggests the four phantom-boundary modes split into two text-format clusters with prompt-family making no geometric contribution. That reading is incomplete: it inspects the wrong layer. Computing full 37-layer cosine-gap profiles for axis-isolated pairs reveals a layered three-axis hierarchy in the residual stream.
docs/checkpoints/paper_drafts/section5_mechanism.md:112:The pairs are constructed to isolate each axis. Axis-1 (text-format swap, prompt fixed) is measured by DOM<->P-text (both DOM prompts) and P-prompt<->P-SoM (both SoM prompts). Axis-2 (prompt-family swap, text fixed) is measured by DOM<->P-prompt (both hierarchical AXTree) and P-text<->P-SoM (both flat indexed list). Axis-3 (image-feature swap, mode otherwise fixed) is measured by the P-SoM<->SoM reference pair. All five curves are computed on `stage4_multimode_b1_cls/hidden_states.npz` (288 examples, 37 layers) and replicated cross-site on the matching reddit run.
docs/checkpoints/paper_drafts/section5_mechanism.md:116:| Axis | Pair | L17 | L23 | L36 | Peak L | Peak gap |
docs/checkpoints/paper_drafts/section5_mechanism.md:122:| Axis-2 prompt-family | DOM <-> P-prompt | 0.0013 | 0.0050 | 0.0067 | **L36** | 0.0067 |
docs/checkpoints/paper_drafts/section5_mechanism.md:124:Three regularities organize the table. First, the three axes have distinct peak layers: image-axis at L17 (fast, sharp), text-format at L23 (slower late-mid build), prompt-family at L23 or L36 (same timing as text-format on the flat-text pair, boundary peak on hierarchical). Second, the three axes have distinct magnitudes: image axis approximately 0.04, text-format approximately 0.03, prompt-family approximately 0.01. Prompt-family is roughly 3 to 4 times smaller than text-format and 4 to 8 times smaller than image. Third, the magnitude rank holds cross-site: the reddit P-text<->P-SoM axis-2 peak is 0.0098 at L23 (versus cls 0.0114), the same rank-order and the same peak layer.
docs/checkpoints/paper_drafts/section5_mechanism.md:126:The L17 snapshot exclusion of axis-2 is therefore a layer-selection artifact rather than a structural absence. Prompt-family is geometrically present in the residual stream; it simply emerges at L23 rather than L17, and at one-third the magnitude of text-format. The mid-layer fusion locus identified in Sections 5.2-5.5 is specifically the image-axis fusion locus (Mirage signature). The text-format and prompt-family axes share a separate late-mid build at L23 that runs in parallel.
docs/checkpoints/paper_drafts/section5_mechanism.md:130:This layered hierarchy partially resolves the Phantom-SoM hero puzzle without requiring a non-mechanistic explanation. Phantom-SoM occupies three distinct axis positions: **large image-axis separation from SoM** at L17 (0.0412 cosine gap, the largest single signal — note this is a *separation* from SoM, not a proximity; the no-image marks-text reshapes how image-axis divergence accumulates rather than recovering an image-like internal state), **small prompt-family separation from P-text** at L23 (0.011 signal, consistent across cls and reddit), and **medium text-format separation from P-prompt** at L23 (0.029 signal). P-text occupies only the text-format separation; P-prompt occupies only the prompt-family separation against P-SoM; DOM occupies none of the three. The drop-one hero status of Phantom-SoM in `fig_meta_forest.png` therefore corresponds to the only mode that simultaneously expresses all three axis positions. The connection from "three-axis positional uniqueness" to "drop-one task-success contribution" is currently associational, not yet causally bridged — the patching evidence shows residual-stream displacement on 50-token continuation, not direct displacement of task-level SR / drop-one oracle. Closing that behavioral causal bridge is open work.
docs/checkpoints/paper_drafts/section5_mechanism.md:132:This reframing is itself paper-grade contribution and not a downgrade of the original mechanism story. It strengthens Section 5 from "mid-layer image-feature axis explains text-format cluster" to "the residual stream carries three quantitatively distinct axes with image-axis dominant at L17 and text-format + prompt-family at L23". Section 8 inherits a sharper instruction for future single-axis steering: LA-HDMI and SAE residual-stream interventions can target the L23 prompt-family direction directly, but the signal-to-noise ratio is approximately 3-4 times worse than the image-axis intervention, so steering experiments on axis-2 require correspondingly more samples or alternative attribution methods.
docs/checkpoints/paper_drafts/section5_mechanism.md:134:The three-axis hierarchy persists when we move from residual-stream geometry to output distribution. A logit lens test (Exp 3) applies Qwen3-VL-4B's final RMSNorm and lm_head to each per-layer per-mode mean hidden state, then computes KL divergence between mode pairs across all 37 layers. On classifieds the axis-2 prompt-family pair (P-text vs P-SoM) reaches peak KL 0.162 at L23, the axis-1 text-format pair (P-prompt vs P-SoM) reaches peak KL 0.695 at L23, and the reddit replication holds the same rank-order with peak KL 0.126 at L24 and 0.617 at L23 respectively. The output-level axis-1 to axis-2 magnitude ratio is therefore approximately 4.3 on classifieds and 4.9 on reddit, preserving the 3-4x residual-stream rank from Exp 1. The lm_head amplifies the residual-stream cosine signal into output divergence by roughly 14x on the axis-2 flat-text pair (cosine 0.011 to KL 0.16) and 24x on the axis-1 SoM-prompt pair (cosine 0.029 to KL 0.69), but it is axis-agnostic in the ratio it preserves.
docs/checkpoints/paper_drafts/section5_mechanism.md:136:Two corollaries follow. First, the KL trajectory drops to approximately zero at L36 even though L23 KL is substantial. The mean hidden state at the final layer collapses to the shared JSON action-header tokens that every mode emits, so mode-distinct output signal is concentrated in the L23-L25 decoding window rather than at the final embedding. Second, this output-amplification observation is **mechanistic, not a deployment-time classifier claim**: the lm_head acts as an axis-agnostic ratio-preserving projection that scales residual-stream geometry into output-space KL — the L23-L25 KL magnitude is a property of the mean hidden state, not a per-task discriminator. Whether the L23-L25 hidden representation can be used as a held-out mode classifier — with per-task AUROC, random-direction baseline, and competitive comparison to surface-token classifiers — is open work. Routing exploitation, deferred to paper 2, will need to make this case explicitly rather than inheriting it from §5.7.
docs/checkpoints/paper_drafts/section5_mechanism.md:142:The second limit is layer precision. Classifieds H1 peaks often hit L36, while reddit reveals cleaner L17 peaks. The robust claim is therefore an effect-direction claim: AXTree hierarchy preserves early image-axis separation, and flat element-list formats delay that separation into mid/late computation. We should not claim that every site or task has an identical peak layer.
docs/checkpoints/paper_drafts/section5_mechanism.md:152:Behavioral content to relocate from current `section5_mechanism_reddit.md`: lines 17-75 should move to Section 4 or a new behavioral-routing subsection. Specifically, lines 17-23 are reddit substrate framing; lines 25-35 are Axis 1 text-payload behavior; lines 37-47 are Axis 2 prompt behavior; lines 49-59 are Axis 3 image behavior; lines 61-67 are compound P-SoM versus DOM behavior; lines 69-75 are scope/noise limitations. Lines 1-15 are method material that was retained conceptually but must use the new L0-L36 layer convention. Line 77 should be deleted or replaced because routing implementation is now paper-2, not paper-1 Section 6.
docs/checkpoints/paper_drafts/section5_mechanism.md:154:Stage 3 numbers verified 2026-05-12 from full per-task paired-test computation on `patching_continuation_results.json` (each cell, 24 tasks × 36 layers). H-d-cls best L18 Δ=-0.352, H-d-red best L14 Δ=-0.338, H-t-cls best L12 -0.270, H-t-red best L15 -0.330, H-p-cls best L13 -0.273, H-p-red best L14 -0.322. All 6 cells' best layer lands in L12-L18 mid-layer window, Δ range [-0.27, -0.35]. The L17-only column previously cited in plan §5.2 reads -0.309/-0.255/-0.223 (cls) and -0.255/-0.236/-0.191 (reddit); plan §5.2 has been updated to record best-layer Δ instead of L17-only Δ.
docs/checkpoints/paper_drafts/section5_mechanism.md:163:2. ✅ §5.7 hero paragraph — "proximity to SoM on the image axis... as if image were present" → corrected to "large image-axis SEPARATION from SoM... no-image marks-text reshapes how image-axis divergence accumulates" (removed internal contradiction with §5.2 table where P-SoM↔SoM gap 0.0412 is the largest = a separation, not proximity)
docs/checkpoints/paper_drafts/section5_mechanism.md:169:- **§4 P-text adjusted SR inconsistency**: §4 table line 37 says 11.90, prose line 106 says 12.38, hero_claim_bootstrap_ci.md says 12.38. Need to canonicalize one number from episode-level adjusted-success and update every occurrence. (1h)
docs/checkpoints/mechanism/plan.md:16:| **1** | Architectural | Phantom routing space = "skip annotated image" boundary contains 3 arms (P-text / P-prompt / P-SoM) sharing 4-fold drop-in property |
docs/checkpoints/mechanism/plan.md:27:| Prompt-axis (SoM-prompt vs DOM-prompt alone) | 0.007 | L36 | **1×** |
docs/checkpoints/mechanism/plan.md:38:| [SOM_MARKS] / flat | **L17–L36** | P-text↔Vision, P-text↔SoM, P-SoM↔Vision, P-SoM↔SoM |
docs/checkpoints/mechanism/plan.md:48:| `[N] role 'label'` (SoM standard) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:49:| `@N label` (Browser Use) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:50:| `id_N: label` (AppAgent) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:51:| `[BN:r:l]` (Tarsier) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:52:| `N. label` (numbered) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:53:| `<el_N>label</el_N>` (XML) | L36 | strong trigger |
docs/checkpoints/mechanism/plan.md:54:| `#hash label` (control: no integer) | L36 | **still triggers!** |
docs/checkpoints/mechanism/plan.md:72:### 3.0 Curation pipeline (Stage 0 setup, shared by Stages 1-4)
docs/checkpoints/mechanism/plan.md:74:`scripts/mechanistic/curate_mirage_tasks.py` builds a strong-tier task manifest from a B1 production run:
docs/checkpoints/mechanism/plan.md:80:- `results/mechanistic/curate_mirage_b1_classifieds/manifest.json` — cls strong/reverse tier task list
docs/checkpoints/mechanism/plan.md:81:- `results/mechanistic/curate_mirage_b1_reddit/manifest.json` — reddit strong tier
docs/checkpoints/mechanism/plan.md:158:### 5.1 Stage 4 Method 4.2 (Qwen3-VL-4B B1 cls, 288 examples × 37 layers)
docs/checkpoints/mechanism/plan.md:194:**Stage 3 interpretation (6/6 cells complete 2026-05-12)**: All forward SoM→{no-image-arm} patching cells show mid-layer L11-L17 disruption -0.19 to -0.33 Δoverlap→tgt. Magnitude > random injection control (Cell E -0.03) at all 6. **Mechanism additivity confirmed**: image-feature axis is shared substrate across DOM / P-text / P-prompt arms — single SoM→{any-no-image-arm} patching displaces target prediction toward source. Cross-site cls + reddit both replicate (paper §5 universal mid-layer fusion locus); reddit fusion locus slightly earlier (L11 vs cls L17), magnitude identical.
docs/checkpoints/mechanism/plan.md:201:| reddit | H-d-red L11 -0.335 / L17 -0.255 / L14 **-0.338** best | H-t-red L11 -0.244 / L17 -0.236 / L15 **-0.330** best | H-p-red L11 -0.233 / L17 -0.191 / L14 **-0.322** best | [-0.322, -0.338] |
docs/checkpoints/mechanism/plan.md:227:`docs/checkpoints/mechanism/results/layer_axis_emergence.md`. AXTree-no-image side → L04 peak (4/4); [SOM_MARKS]-no-image side → L17–L36 peak (4/4). Zero overlap. Mirage Effect mechanism signature.
docs/checkpoints/mechanism/plan.md:231:`docs/checkpoints/mechanism/results/format_variation_h1_test.md`. 8 industry-relevant text formats + 2 controls. AXTree hierarchical (DOM) is **unique format** preserving L04 image-axis peak; all 8 flat-list variants (SoM standard, Browser Use @, AppAgent id_, Tarsier typed, plain numbered, XML tagged, hash-ID control, plain-sentence control) shift peak to L17–L36. Trigger is flat element listing, not specific token pattern.
docs/checkpoints/mechanism/plan.md:245:| AXTree-defeats-shortcut mechanism — *why* hierarchy beats flat? Cross-modal attention specific to indentation tokens? | High (paper §5 supplement) | Activation patching at L4 with hierarchical-text vs flat-text → see which attention heads pre-disrupt image embedding |
docs/checkpoints/mechanism/plan.md:275:> 24 cls strong-tier × 2 step × 6 mode = 288 hidden states, 37 layer × 2560 dim。全 540 pair × layer AUROC = 1.000 (perm baseline 0.629, real 9.8σ above). 你方法在 multimodal Qwen 上 readable transfer 干净。
docs/checkpoints/mechanism/plan.md:287:>   - 全 8 flat variants peak L17/L36 (= 都触发 shortcut)
docs/checkpoints/mechanism/plan.md:310:| **P4** | cls reverse-tier (selection-bias defense) | Myriad 353763 | ✅ **done 18:50:46** — shape (260, 37, 2560), 10 modes, 46 MB pulled. Same pattern as cls strong-tier (L36 marks-like + L04 dom). Selection-bias defended | `stage4_format_variation_b1_cls_reverse/hidden_states.npz` |
docs/checkpoints/mechanism/plan.md:311:| **P5a** | reddit format variation (cross-site H1) | Myriad **354382** (3rd attempt) | ✅ **done 08:09:38** — shape (430, 37, 2560), 10 modes, 76 MB pulled | `stage4_format_variation_b1_reddit/hidden_states.npz` |
docs/checkpoints/mechanism/plan.md:315:1. Myriad 353764 (00:48) — `no hidden states extracted` after 105 task skips. Root cause: hardcoded `classifieds_task_{tid}` prefix in `run_stage4_format_variation_extract.py:177`, archive uses `reddit_task_*`
docs/checkpoints/mechanism/plan.md:338:`axis2_layer_profile.md` + `fig_axis2_prompt_layer_profile.png`. Re-examine residual stream geometry per axis-isolated pair, full 37-layer cosine curves on `stage4_multimode_b1_{cls,reddit}` (288 ex each).
docs/checkpoints/mechanism/plan.md:342:| Pair | Group | L17 | L23 | L36 | Peak L | Peak gap |
docs/checkpoints/mechanism/plan.md:348:| DOM↔P-prompt (prompt fam, hier) | axis-2 | 0.0013 | 0.0050 | 0.0067 | L36 | 0.0067 |
docs/checkpoints/mechanism/plan.md:353:1. **Distinct peak layers**: image L17 (fast sharp), text-format L23 (slower late-mid), prompt-family L23 (same timing as text-format on flat-text)
docs/checkpoints/mechanism/plan.md:384:`axis2_logit_lens.md` + `fig_axis2_logit_lens.png`. 应用 Qwen3-VL-4B `model.model.language_model.norm` + `model.lm_head` to per-layer per-mode mean hidden states, 算 KL across 37 层.
docs/checkpoints/mechanism/plan.md:397:3. **KL @ L36 ≈ 0 paradox**: 因 mean hidden state at last layer collapse to common JSON format header. Mode-distinct signal concentrated in **L23-L25 decoding window** (not final embedding). This is the "knows but says differently" structural mirror of Wu et al. tool calling.
docs/checkpoints/mechanism/plan.md:407:| som_standard / browser_use_at / tarsier_typed / xml_tagged | **L17** | L36 (last) |
docs/checkpoints/mechanism/plan.md:408:| appagent_id / plain_numbered | **L04** | L36 |
docs/checkpoints/mechanism/plan.md:409:| hash_id_control | **L04** ✓ (acts as control) | L36 (control failed) |
docs/checkpoints/mechanism/plan.md:413:**Reddit nuance — cleaner mid-layer fusion**: Reddit 上 marks-like 4/6 真 peak 在 L17 (mid-layer), cls 上 L36 是 monotonic increasing artifact (peak hit boundary). Reddit hash_id_control L04 acts as proper "no integer" control (cls 上失败). Reddit data supports Q5 mid-layer fusion hypothesis better than cls.
docs/checkpoints/mechanism/plan.md:437:| 6 marks-like | L36 monotonic | **L36 monotonic** ✓ same | L17 (4/6 真 peak) |
docs/checkpoints/mechanism/plan.md:438:| hash_id_control | L36 (failed control) | **L36** ✓ same | L04 ✓ proper control |
docs/checkpoints/mechanism/plan.md:442:H1 mechanism in cls is **not tier selection artifact** (strong vs reverse both replicate). Reddit data paradoxically cleaner reveal of true L17 mid-layer fusion locus (cls L36 is monotonic-boundary artifact).
docs/checkpoints/mechanism/results/axis2_logit_lens.md:5:across all 37 layers. This probes whether axis-2 cosine signal (L23 peak 0.011) gets
docs/checkpoints/mechanism/results/axis2_logit_lens.md:30:| P-text vs P-SoM  (axis-2 flat-text) | **L24** | 0.1260 | 0.0371 | 0.1230 | 0.0002 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:16:| som_standard | `[N] role 'label' (SoM)` | marks-like | **L36** | 0.0434 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:17:| browser_use_at | `@N label (Browser Use)` | marks-like | **L36** | 0.0502 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:18:| appagent_id | `id_N: label (AppAgent)` | marks-like | **L36** | 0.0506 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:19:| tarsier_typed | `[BN:role:label] (Tarsier)` | marks-like | **L36** | 0.0468 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:20:| plain_numbered | `N. label (numbered)` | marks-like | **L36** | 0.0502 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:21:| xml_tagged | `<el_N role='..'>label</el_N> (XML)` | marks-like | **L36** | 0.0426 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:22:| hash_id_control | `#hash label (no integer)` | control (no integer) | **L36** | 0.0514 |
docs/checkpoints/mechanism/results/format_variation_h1_test.md:26:### marks-like  (mean peak L36)
docs/checkpoints/mechanism/results/format_variation_h1_test.md:28:- `[N] role 'label' (SoM)`: peak **L36** = 0.0434
docs/checkpoints/mechanism/results/format_variation_h1_test.md:29:- `@N label (Browser Use)`: peak **L36** = 0.0502
docs/checkpoints/mechanism/results/format_variation_h1_test.md:30:- `id_N: label (AppAgent)`: peak **L36** = 0.0506
docs/checkpoints/mechanism/results/format_variation_h1_test.md:31:- `[BN:role:label] (Tarsier)`: peak **L36** = 0.0468
docs/checkpoints/mechanism/results/format_variation_h1_test.md:32:- `N. label (numbered)`: peak **L36** = 0.0502
docs/checkpoints/mechanism/results/format_variation_h1_test.md:33:- `<el_N role='..'>label</el_N> (XML)`: peak **L36** = 0.0426
docs/checkpoints/mechanism/results/format_variation_h1_test.md:35:### control (no integer)  (mean peak L36)
docs/checkpoints/mechanism/results/format_variation_h1_test.md:37:- `#hash label (no integer)`: peak **L36** = 0.0514
docs/checkpoints/mechanism/results/format_variation_h1_test.md:49:- **6 marks-like variants**: mean peak layer = 36, range L36-L36
docs/checkpoints/mechanism/results/format_variation_h1_test.md:50:- **2 control variants** (no integer / no list): mean peak layer = 26, range L17-L36
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:16:| som_standard | `[N] role 'label' (SoM)` | marks-like | **L36** | 0.0429 |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:17:| browser_use_at | `@N label (Browser Use)` | marks-like | **L36** | 0.0520 |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:18:| appagent_id | `id_N: label (AppAgent)` | marks-like | **L36** | 0.0526 |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:19:| tarsier_typed | `[BN:role:label] (Tarsier)` | marks-like | **L36** | 0.0475 |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:20:| plain_numbered | `N. label (numbered)` | marks-like | **L36** | 0.0518 |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:21:| xml_tagged | `<el_N role='..'>label</el_N> (XML)` | marks-like | **L36** | 0.0439 |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:22:| hash_id_control | `#hash label (no integer)` | control (no integer) | **L36** | 0.0516 |
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:26:### marks-like  (mean peak L36)
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:28:- `[N] role 'label' (SoM)`: peak **L36** = 0.0429
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:29:- `@N label (Browser Use)`: peak **L36** = 0.0520
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:30:- `id_N: label (AppAgent)`: peak **L36** = 0.0526
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:31:- `[BN:role:label] (Tarsier)`: peak **L36** = 0.0475
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:32:- `N. label (numbered)`: peak **L36** = 0.0518
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:33:- `<el_N role='..'>label</el_N> (XML)`: peak **L36** = 0.0439
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:35:### control (no integer)  (mean peak L36)
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:37:- `#hash label (no integer)`: peak **L36** = 0.0516
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:49:- **6 marks-like variants**: mean peak layer = 36, range L36-L36
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md:50:- **2 control variants** (no integer / no list): mean peak layer = 29, range L22-L36
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:49:- som: 21.37%
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:60:| P-SoM vs DOM | +0.43 | +0.43 | [-3.42, +4.70] | 0.538 | 0.374 | 
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:62:| P-text vs DOM | +0.43 | +0.43 | [-3.42, +4.27] | 0.546 | 0.376 | 
docs/checkpoints/mechanism/results/axis2_layer_profile.md:7:**Method**: For each prompt-only pair (text format fixed, prompt swap), compute full 37-layer cosine gap.
docs/checkpoints/mechanism/results/axis2_layer_profile.md:12:| Pair | Group | L0 | L4 | L17 | L36 | Peak L | Peak gap |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:14:| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0013 | 0.0067 | **L36** | 0.0067 |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:22:| Pair | Group | L0 | L4 | L17 | L36 | Peak L | Peak gap |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:24:| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0012 | 0.0059 | **L36** | 0.0059 |
docs/checkpoints/mechanism/results/axis2_layer_profile.md:36:3. **Early-layer spike absorbed** — axis-2 pair curves spike at L0-L5 then collapse to ~0. Prompt embedding effect absorbed by mid-layer fusion. → Next: Exp 3 logit lens to verify if it re-emerges in output distribution.
docs/checkpoints/paper_drafts/section8_limitations.md:19:The statistical design is adequate for medium effects but underpowered for small per-cell effects and exact-layer micro-effects. The blast radius is precision, not directionality: Holm-Bonferroni is applied across the six canonical tested layers (L0/5/11/17/23/29), not all 36 cached layers; this matches the disclosed post-hoc grid but should not be read as a full-layer search correction \citep{holm1979sequentially,wang2023interpretability,zhang2024patching}. Bootstrap intervals use task-paired resampling, random-effects meta-analysis is limited to cells with N>=10 to avoid unstable tau-squared estimates, and complete-case deletion handles crashes or missing artifacts without imputation. Exclusions are listwise only, at <=5% per cell under the B6 lock, so multiple imputation would add modeling assumptions without materially changing paired denominators. Power analysis shows N=15 mechanism cells are not powered for small mid-layer effects (roughly Cohen's d below 0.65 at alpha=0.05), while site-level SR cells mainly detect 4--7pp effects. This affects Sections 4--5: null or marginal cells are interpreted as low-power evidence, and pooled estimates are paired with per-cell uncertainty.
docs/checkpoints/paper_drafts/section8_limitations.md:27:Compute and carbon accounting are approximate because runs span DGX Spark, UCL Condense A100, and UCL Myriad V100/A100 profiles. The blast radius is absolute energy reporting: the per-cell table reports GPU-hours, proxy-API USD cost, and kg-CO2 estimates with hardware provenance, but cross-machine power variation limits precision \citep{qiu2025modserve}. Relative comparisons inside a cell remain valid because modes share the same hardware, benchmark site, and evaluator path. This affects the cost part of the four-fold drop-in claim: token/API cost and latency comparisons are primary, while carbon totals are disclosed as bounded estimates rather than a fine-grained lifecycle assessment.
scripts/vwa_env_remote.sh:9:export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:13:| Layers | 37 (L0-L36, Qwen3-VL-4B language decoder) | matching |
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:91:- 2-site cross-replication (cls + red), N=24 each, paired L0-L36 sweep
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:93:**Reframes hero argument**: The paper §1 framing "text-format shapes exploration; prompt tunes commit" is now backed by:
docs/checkpoints/mechanism/results/layer_axis_emergence.md:13:| P-text | Vision | [SOM_MARKS] | **L36** | 0.0614 |
docs/checkpoints/mechanism/results/layer_axis_emergence.md:14:| P-SoM | Vision | [SOM_MARKS] | **L36** | 0.0613 |
docs/checkpoints/mechanism/results/layer_axis_emergence.md:28:- P-text ↔ Vision: peak **L36** = 0.0614
docs/checkpoints/mechanism/results/layer_axis_emergence.md:30:- P-SoM ↔ Vision: peak **L36** = 0.0613
docs/checkpoints/mechanism/results/layer_axis_emergence.md:36:When the no-image side carries `[SOM_MARKS]` text (P-text, P-SoM), the image-axis cosine gap peak shifts to **L17–L36** — image yes/no divergence is deferred to mid/output layers.
docs/checkpoints/mechanism/results/layer_axis_emergence.md:42:> *Method 4.2 reveals a peak-layer shift signature for the Mirage Effect: image-axis cosine-gap peak transitions from L04 (when no-image side carries AXTree text) to L17–L36 (when no-image side carries [SOM_MARKS] text). The peak-layer shift quantifies how text-payload format primes mid-layer computation pathways — [SOM_MARKS] format triggers indexed-parsing through mid-layers regardless of image presence, with image-axis divergence deferred to late integration. This identifies [SOM_MARKS] as the mechanism trigger for the Mirage Effect, anchoring Asadi et al. 2026's behavioral finding (~70-80% no-image VLM accuracy) and Kaduri et al.'s middle-layer cross-modal flow hypothesis with layer-resolved empirical evidence.*
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:24:| P-SoM ↔ SoM     (axis-3 image ref) | axis-3 | 0.0436 | 0.0439 | [0.0409, 0.0453] | 0.0382 | 0.0535 | 100% | 100% | 100% |
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md:60:| 137 | 0.0111 |
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:13:AXTree-DOM peak layer: mean = **27.9**, std = 13.1, range L04-L36
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:25:| 214 | 5 | L04 | L36.0 | **+32.0** |
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:26:| 228 | 2 | L04 | L36.0 | **+32.0** |
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:36:| 20 | 2 | L36 | L33.3 | -2.7 |
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:37:| 122 | 2 | L36 | L33.3 | -2.7 |
docs/checkpoints/mechanism/results/h1_per_task_fragility.md:39:| 37 | 2 | L36 | L28.0 | -8.0 |
docs/checkpoints/mechanism/README.md:11:P79 phantom routing space modes (DOM / SoM / Vision / P-text / P-prompt / P-SoM) are **linearly readable** from Qwen3-VL-4B residual stream at every layer 4–36 (Method 4.2 PCA cosine gap, AUROC 1.000 × 540 layer-pair tests). The image-axis dominates mechanism magnitude (peak gap 0.06 at L4–L17), text-axis is mid-scale (0.025 at L23), prompt-axis alone is weakest (0.007 at L36) — 10:4:1 hierarchy. P-SoM's closest mode at every layer is **P-text** (14.7× more distant from SoM), refuting "P-SoM = SoM minus image" and validating the format-axis framing. Mid-layer L17 acts as the **causally active planning site** (Stage 2/3 Cell A-H replacement patching disrupts output overlap; HDMI reliability framework gives Method 4.4 v2 mid-layer L17 α=5 = 0.44 reliability vs late-layer L33 α=10 = 0.23, because late-layer over-steers JSON envelope). This positions our work between Wu et al. 2026 (text-only Qwen 3 4B, tool selection 93% switch) and Ma & Rui 2026 (Qwen3 family rhyme newline 1% causal) — 50% mid-layer reliability is the multimodal-multi-step regime's signature.
p79/mechanistic/extract_hidden_states.py:4:per agent line 436) so hidden states reflect identical prompt structure as the
p79/mechanistic/extract_hidden_states.py:91:        """Replicate agent's user content text format (qwen3vl_agent.py:436).
p79/mechanistic/extract_hidden_states.py:129:            Tensor of shape (n_layers + 1, hidden_dim). Layer 0 is embedding output;
p79/mechanistic/extract_hidden_states.py:168:        # outputs.hidden_states is tuple of (n_layers + 1) tensors of shape
p79/mechanistic/linear_probe.py:63:    n_samples, n_layers, hidden_dim = hidden_states.shape
p79/mechanistic/linear_probe.py:64:    if labels.shape[0] != n_samples:
p79/mechanistic/linear_probe.py:65:        raise ValueError(f"labels.shape[0] {labels.shape[0]} != n_samples {n_samples}")
p79/mechanistic/linear_probe.py:81:                effective_pca_dim = min(pca_dim, X_train.shape[0] - 1, X_train.shape[1])
p79/mechanistic/linear_probe.py:146:    ax.set_xlabel("Layer index (0 = embedding output, ≥1 = post-transformer-block)")
docs/checkpoints/paper_drafts/section2_background.md:27:For a web agent, prompt-format sensitivity matters at the trajectory level. The model is not producing a single label; it is choosing whether to search, click, scroll, revisit a page, or finish. Section 4 and Section 5 build on this theoretical anchor: the flat marks list tends to shift exploration toward quick element selection, while AXTree hierarchy tends to support sustained navigation and search. Prompt wording also matters, but our two-knob account separates the layers: text representation shapes how the agent explores, while prompt family tunes when it commits.
docs/checkpoints/paper_drafts/section2_background.md:47:Our contributions follow directly. We define **Phantom-SoM** as the deployment-relevant representative of the phantom routing space: marks text plus SoM prompt family, with the image removed. We show empirically that Phantom-SoM is not a degenerate DOM surrogate: it contributes independent oracle value and has substantial task-pool non-overlap with DOM, SoM, and Vision. The P-text and P-prompt controls establish specificity by separating text-payload flattening from prompt-family effects, rather than attributing all gains to a single novel arm. Finally, we provide mechanism evidence for a two-knob account: representation format shapes exploration, while prompt wording tunes commitment confidence. This motivates the experimental design in Section 3 and the controlled evidence in Section 4.
scripts/queues/queue_phantom_som.sh:196:# Unified per-baseline aggregate gallery — all B0/B1 modes share one URL
scripts/queues/qsub_stage3_cellhp_red_fwd_prompt.sh:34:if [ ! -f "$REPO_DIR/results/mechanistic/archive_subset_b1_reddit/manifest.json" ]; then
scripts/queues/qsub_stage3_cellhp_red_fwd_prompt.sh:35:  echo "FATAL: archive_subset_b1_reddit/manifest.json missing"
scripts/queues/qsub_stage3_cellhprompt_red.sh:35:if [ ! -f "$REPO_DIR/results/mechanistic/archive_subset_b1_reddit/manifest.json" ]; then
scripts/queues/qsub_stage3_cellhprompt_red.sh:36:  echo "FATAL: archive_subset_b1_reddit/manifest.json missing"
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:130:**A1 — Decoding strategy**: B0 uses `temperature=0.0` with `top_p=1.0` (B-37 fix
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:154:{DGX, A100, Myriad} layers L0-L35: max |Δh| < [TBD post-rerun, target <1e-2] at L11 (the
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:160:## §4.X.9 Pre-Phase-A vs post-Phase-A asymmetry (B-01 to B-37 family)
scripts/queues/queue_phantom_text.sh:236:# Unified per-baseline aggregate gallery — all B0/B1 modes share one URL
scripts/queues/queue_pilot_t0.sh:2:# queue_pilot_t0.sh — B-37 Phase A pilot launcher (T=0 + RNG seeding sanity gate)
scripts/queues/queue_pilot_t0.sh:117:echo "[$(date +%H:%M:%S)] Launching B-37 pilot: ${RUN_ID}"
docs/checkpoints/paper_drafts/section3_definition.md:35:**Relative to DOM**, Phantom-SoM is essentially free. The `[SOM_MARKS]` block is produced by a regex filter over the VisualWebArena accessibility-tree text that the DOM baseline already consumes. VWA serializes interactive elements with bracketed numeric IDs of the form `[N] role 'label'`; in our implementation `_extract_text_marks` (see `p79/experiment/som.py`) walks `obs_text` line by line, keeps the lines that match `\[\d+\]`, and returns `(id, label)` pairs that are wrapped in a `[SOM_MARKS] ... [/SOM_MARKS]` block. There is no bounding-box lookup and no image work in this path; bounding boxes are only used by full SoM when drawing numeric labels onto the screenshot. Empirically this leaves text length roughly unchanged: holding the system prompt fixed at the DOM family, median total input is 3437 tokens for DOM versus 3661 for P-text on reddit, and 3008 versus 2948 on classifieds — within ±7% on both sites. The two formats see the same accessibility content; what differs is the surface form (flat indexed list versus nested hierarchy with url/tab metadata). We treat this as a representation property and study its behavioral effect mechanistically in Section 5; for cost accounting the implication is that switching DOM → Phantom-SoM at deployment time costs at most a regex pass over the same observation.
docs/checkpoints/paper_drafts/section3_definition.md:76:The 2x2 is not a routing policy by itself. It is a causal scaffold for Section 5: text-payload structure shapes exploration, while prompt wording tunes commitment confidence. Section 6 promotes this scaffold to a token-monotonic cascade — DOM → P-text (axis 1, text-payload swap, no token increase) → Phantom-SoM (axis 2, system-prompt swap, no data-token increase) → full SoM (axis 3, image embedding cost) — so a routing trigger never has to "add then remove" tokens.
docs/checkpoints/paper_drafts/section3_definition.md:80:All SoM-derived conditions share the same text-marks extractor. `_extract_text_marks` reads `obs_text` (the VisualWebArena accessibility-tree serialization the DOM baseline already uses) line by line, keeps each line whose label matches `\[\d+\]`, and produces `(id, label)` pairs up to a configured cap. `_build_som_result` then wraps those pairs in a `[SOM_MARKS] ... [/SOM_MARKS]` block. This text path **does not require bounding boxes**: the IDs come from the accessibility tree, not from a separate vision pipeline. Bounding boxes are only consulted by full SoM, which uses `obs_nodes_info` to draw numeric labels onto the page screenshot. Phantom-SoM and P-text reuse the exact `[SOM_MARKS]` text and drop the page screenshot; Marks are not re-filtered specifically for Phantom, and the source page state is unchanged.
scripts/queues/qsub_stage3_cellhprompt_cls.sh:38:if [ ! -f "$REPO_DIR/results/mechanistic/archive_subset_b1_cls/manifest.json" ]; then
scripts/queues/qsub_stage3_cellhprompt_cls.sh:39:  echo "FATAL: archive_subset_b1_cls/manifest.json missing"
scripts/queues/queue_phantom_prompt.sh:192:# Unified per-baseline aggregate gallery — all B0/B1 modes share one URL.
scripts/queues/queue_16cell_paper_grade.sh:33:#   Within each chain B0 → B1 sequential (same-site B0/B1 share user account login).
scripts/queues/queue_16cell_paper_grade.sh:36:#   cls chain: B0 (~12h) → B1 (~24h) = 36h
scripts/queues/queue_16cell_paper_grade.sh:39:#   Total wallclock with 3 parallel chains = max(36, 30, 48) = ~48h ≈ 2 days
scripts/queues/qsub_stage3_cellhp_cls_fwd_prompt.sh:37:if [ ! -f "$REPO_DIR/results/mechanistic/archive_subset_b1_cls/manifest.json" ]; then
scripts/queues/qsub_stage3_cellhp_cls_fwd_prompt.sh:38:  echo "FATAL: archive_subset_b1_cls/manifest.json missing"
docs/checkpoints/paper_drafts/section4_empirical_findings.md:29:| Classifieds | DOM | 14.10 | $0.043 | 37.5s | N=234 |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:30:| Classifieds | SoM | **21.37** | $0.042 | 74.0s | N=234 |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:40:The classifieds result is the expected sanity check: when tasks benefit from visual page layout and product imagery, the marked screenshot adds useful grounding and full SoM is clearly best (**SoM 21.37 vs Phantom-SoM 14.53; N=234; adjusted**). Phantom-SoM is close to DOM on classifieds (**14.53 vs 14.10**), but this is not a dominance claim; it is inside the noise floor and far below full SoM.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:63:The overlap view supports the same conclusion. In the four-arm oracle, Phantom-SoM contributes a concrete reddit-only set of seven tasks (**7, 15, 36, 94, 157, 162, 167**) and a non-zero classifieds set as well. Two examples illustrate the kind of work this arm is doing. On reddit task 7, Phantom-SoM searched for the cake-recipe post and navigated directly to the OP recipe comment permalink. On reddit task 162, it searched within /f/wallstreetbets, scrolled hot posts, and returned the GIF URL for the retirement-account-versus-brokerage-account prompt. These are not proof of a universal mechanism by themselves, but they make the drop-one value concrete: the arm is adding recoverable successes, not only shifting aggregate percentages.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:71:> **Text format shapes how the agent explores. Prompt wording tunes when the agent commits.**
docs/checkpoints/paper_drafts/section4_empirical_findings.md:73:The first knob is exploration shape. On the same-task reddit ablation subset, replacing AXTree text with `[SOM_MARKS]` text shifts macro behavior away from DOM-like search loops and toward Phantom-SoM-like quick decisions. The verified search-loop rate is **22.7% for DOM** but **10.8% for Phantom-SoM and 10.8% for P-text** (**N=48; behavior metric; same-task subset**). The prompt change alone does not pull P-text back to DOM-like exploration. This supports the representation-driven part of the hypothesis: the flat marks list, not only the SoM prompt, changes the trajectory distribution.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:104:The evidence chain is consistent across dimensions. At the outcome level, adding P-text to the three-mode baseline contributes oracle value even without the SoM prompt or screenshot (Outcome 0c, +P-text +3.21pp single-phantom lift on the current oracle intersection). At the macro level, the whole-run strategy gradient shows the failure basin directly: reddit search-loop rate falls from DOM to Phantom-SoM and then to full SoM (Macro 1c, search-loop 51.90%->35.71%->31.43%). The axis-1-only macro effect is smaller than the compound prompt path, which is expected if flat text mainly changes which page objects are reachable rather than merely changing the action vocabulary. The micro evidence is sharper: DOM versus P-text has low path overlap for a text-only swap (Micro 2a, URL-path Jaccard 0.573), improves target-page reach (Micro 2b, target-hit +3.47pp), and reduces repeated keyword reuse (Micro 2c, max-keyword-repeat -0.633). The click-target view tells the same story: the two modes choose substantially different element sets even before images enter the system (Micro 2a-extra, click-target Jaccard 0.463).
scripts/queues/qsub_2x2_fwd_revtasks_myriad.sh:55:n_reverse=$(python3 -c "import json; print(len(json.load(open('$REPO_DIR/results/mechanistic/archive_subset_b1_cls/manifest.json'))['reverse']))")
scripts/queues/qsub_stage3_cellhd_red_fwd_dom.sh:36:if [ ! -f "$REPO_DIR/results/mechanistic/archive_subset_b1_reddit/manifest.json" ]; then
scripts/queues/qsub_stage3_cellhd_red_fwd_dom.sh:37:  echo "FATAL: archive_subset_b1_reddit/manifest.json missing"
scripts/queues/qsub_stage3_cellht_cls_fwd_text.sh:37:if [ ! -f "$REPO_DIR/results/mechanistic/archive_subset_b1_cls/manifest.json" ]; then
scripts/queues/qsub_stage3_cellht_cls_fwd_text.sh:38:  echo "FATAL: archive_subset_b1_cls/manifest.json missing"
scripts/queues/_deprecated/queue_b0_with_reset.sh:37:    'reddit':      ('MarvelsGrantMan136',        'test1234'),
docs/checkpoints/paper_drafts/section1_intro.md:5:This paper questions that bundling assumption as an experimental object. We characterize the **phantom routing space**: configurations on the "skip annotated image" boundary that retain some SoM-derived textual or prompt structure while removing the image. Its deployment-relevant representative is **Phantom-SoM**: the agent receives the SoM prompt and the `[SOM_MARKS]` textual element list, but no image. The structural controls are **P-text** (the `[SOM_MARKS]` text under the DOM prompt) and **P-prompt** (the SoM prompt over AXTree text). At the start of this project, Phantom-SoM looked like a broken ablation. The natural expectation was that removing the marked screenshot would collapse SoM into either a weak DOM surrogate or a nonsensical configuration: the prompt still suggests visual marks, but the visual substrate is absent. The data reject that expectation. Phantom-SoM solves tasks that DOM, full SoM, and Vision all miss, and on B0 reddit it matches or modestly exceeds full SoM by adjusted SR (**13.81% vs 10.48%, N=210**; the gap is within 2σ under the run-to-run variability we observe in same-condition repeats), while avoiding image-token cost. On classifieds, full SoM remains clearly stronger (**21.37% vs Phantom-SoM 14.53%, N=234**), the expected sanity check when marked screenshots carry real visual information.
docs/checkpoints/paper_drafts/section1_intro.md:11:Our second contribution is a mechanism account for why the ablation works. A 2-by-2 reddit ablation separates text representation from prompt family: DOM prompt versus SoM prompt, crossed with AXTree versus `[SOM_MARKS]` text. On the verified same-task subset (**N=48**), replacing AXTree with `[SOM_MARKS]` shifts exploration away from DOM-like search loops and toward Phantom-SoM-like quick decisions: the search-loop rate is **22.7% for DOM** but **10.8% for both Phantom-SoM and P-text**. The prompt knob appears elsewhere. DOM-prompt arms show the larger raw-to-adjusted false-positive gap (**6.25 pp; 3 N/A false positives**), while Phantom-SoM under the SoM prompt has a smaller gap (**2.08 pp; 1 N/A false positive**). Section 5 adds mid-layer mechanistic evidence: activation-patching effects concentrate in the L11-L17 region, and the random-injection control distinguishes content-specific phantom information from generic perturbation. The resulting two-knob view is simple: **text representation shapes how the agent explores; prompt wording tunes when it commits**. This aligns with prior evidence that language models are highly sensitive to prompt format \citep{sclar2024promptformat,mishra2022reframing}, but extends the claim from static prediction to multi-step web-agent trajectories.
scripts/queues/queue_chain.sh:4:# that share a single GPU instance (B1 4B local) or any paper-grade sequence.
scripts/queues/qsub_stage2_cellcr_reddit_fwd_revtier.sh:34:if [ ! -f "$REPO_DIR/results/mechanistic/archive_subset_b1_reddit/manifest.json" ]; then
scripts/queues/qsub_stage2_cellcr_reddit_fwd_revtier.sh:35:  echo "FATAL: archive_subset_b1_reddit/manifest.json missing"
scripts/queues/qsub_curate_reddit_myriad.sh:12:# manifest.json + per-task data → enables reddit Stage 2 patching cells F+G.
scripts/queues/qsub_stage3_cellhprompt_red_rand.sh:35:if [ ! -f "$REPO_DIR/results/mechanistic/archive_subset_b1_reddit/manifest.json" ]; then
scripts/queues/qsub_stage3_cellhprompt_red_rand.sh:36:  echo "FATAL: archive_subset_b1_reddit/manifest.json missing"
scripts/queues/qsub_stage3_cellhd_cls_fwd_dom.sh:39:if [ ! -f "$REPO_DIR/results/mechanistic/archive_subset_b1_cls/manifest.json" ]; then
scripts/queues/qsub_stage3_cellhd_cls_fwd_dom.sh:40:  echo "FATAL: archive_subset_b1_cls/manifest.json missing"
scripts/queues/qsub_stage3_cellhprompt_cls_rand.sh:45:if [ ! -f "$REPO_DIR/results/mechanistic/archive_subset_b1_cls/manifest.json" ]; then

exec
/bin/bash -lc "nl -ba docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md | sed -n '1,130p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Exp 5 — Axis-2 Prompt-Family Causal Patching (cellhprompt cls + red)
     2	
     3	**Status**: Closed 2026-05-12 — cellhprompt_cls (359511) + cellhprompt_red (359512) landed via manual auto_pull after silent-miss bug in watcher.
     4	
     5	## Design
     6	
     7	| Variable | cellhprompt (this exp) | H-text baseline (prior) |
     8	|---|---|---|
     9	| Source | `phantom_som` (no image, flat `[SOM_MARKS]`, SoM prompt) | `som` (image, flat `[SOM_MARKS]`, SoM prompt) |
    10	| Target | `phantom_text` (no image, flat `[SOM_MARKS]`, DOM prompt) | `phantom_text` (same) |
    11	| Axes flipped src→tgt | **prompt-family only** | image axis + prompt-family |
    12	| N tasks | 24 (cls strong-tier) / 24 (red strong-tier) | matching |
    13	| Layers | 37 (L0-L36, Qwen3-VL-4B language decoder) | matching |
    14	
    15	**Test logic**: Holding both `image` and `text-format` constant (off + flat) and patching source hidden states from `phantom_som` into a `phantom_text` run isolates whether the residual-stream prompt-family signature has *causal* effect on token continuation, not just *geometric* magnitude (which Exp 1 already showed is small at 0.011 cosine gap @ L23).
    16	
    17	## Result — mid-layer (L11-L17) patching causal effect
    18	
    19	| Site | Cell (axes) | overlap→tgt L11 | overlap→tgt L17 | LD→tgt L11 | LD→tgt L17 |
    20	|---|---|---:|---:|---:|---:|
    21	| cls | H-text (image+prompt) | 0.74 | 0.75 | 9.0 | 9.2 |
    22	| cls | cellhprompt (**prompt only**) | 0.80 | 0.79 | 8.5 | 8.5 |
    23	| red | H-text (image+prompt) | 0.76 | 0.76 | 9.0 | 8.6 |
    24	| red | cellhprompt (**prompt only**) | 0.80 | 0.70 | 7.0 | 8.8 |
    25	
    26	(Baseline `overlap→tgt = 1.00` at L35 = full target preservation, no patching effect.)
    27	
    28	### Causal weight decomposition
    29	
    30	- Axis-2 (prompt) **alone** displaces target output by **0.20-0.30 overlap** units, mid-layer L11-L17 peak.
    31	- Combined image+prompt (H-text) displaces by **0.24-0.26** at same layers.
    32	- **Prompt-only captures ~77-100% of the combined effect** (cls 0.21/0.25 = 84%; red @ L17 0.30/0.24 = 125%, **prompt-only stronger on red**).
    33	- Therefore **image axis contributes a small residual** when prompt-family already differs; prompt-family is the dominant causal driver in this 2-axis subspace.
    34	
    35	### Cross-site replication
    36	Both cls + red show the same mid-layer L11-L17 peak. Reddit shows *stronger* axis-2 effect at L17 than cls (overlap→tgt 0.70 vs 0.79).
    37	
    38	## Geometric ⫨ causal disjoint (two disjoints: magnitude AND layer)
    39	
    40	### Disjoint 1 — magnitude
    41	
    42	Compared with Exp 1 cosine geometry, using best-layer values:
    43	
    44	| Axis | Cosine gap (best layer) | Patching displacement (best causal layer) |
    45	|---|---:|---:|
    46	| Image (SoM ↔ P-SoM) | 0.041 @ L17 | ~0.04-0.05 (inferred from H-text − cellhprompt diff) |
    47	| Text-format (DOM ↔ P-text) | 0.029 @ L23 | (Exp H-d-cls/red, not directly compared here) |
    48	| **Prompt-family (P-SoM ↔ P-text)** | **0.011 @ L23** | **~0.20-0.30 @ L11-L17** |
    49	
    50	**4:3:1 cosine geometry ratio does NOT translate to 4:3:1 causal patching ratio.** Prompt-family has the **smallest** geometric magnitude but the **largest** causal patching weight.
    51	
    52	### Disjoint 2 — layer
    53	
    54	Critically, the **layer at which cosine peaks ≠ the layer at which patching has maximal effect** for prompt-family:
    55	
    56	| Layer | cls overlap→tgt | red overlap→tgt | Interpretation |
    57	|---|---:|---:|---|
    58	| L0  | 0.86 | 0.92 | early, signal not yet routed |
    59	| L11 | 0.80 | 0.80 | **causal peak (cls)** — prompt-family begins routing decision |
    60	| L17 | 0.79 | **0.70** | **causal peak (red)** |
    61	| **L23** | 0.96 | 0.89 | **cosine geometry peak, but patching weak** — representation stabilized |
    62	| L29 | 0.92 | 0.95 | downstream re-encoding |
    63	| L35 | 1.00 | 1.00 | output convergence (baseline preserved) |
    64	
    65	At **L23** (the cosine peak), patching displaces target output by only **0.04-0.11 overlap units** — much smaller than the **0.20-0.30** displacement at L11-L17.
    66	
    67	### Interpretation: signature ≠ use
    68	
    69	This is the **second** geometric/causal disjoint, in addition to magnitude:
    70	
    71	- **L23 is the prompt-family "signature layer"**: representation has stabilized to its most discriminable form (highest cosine separation between P-SoM and P-text). It reflects *what prompt was given* — a state variable.
    72	- **L11-L17 is the prompt-family "decision routing layer"**: patching here changes upstream signal that downstream layers consume to drive token continuation. It reflects *how the model uses the prompt* — a causal variable.
    73	
    74	Activation patching is path-dependent: an upstream patch propagates into all downstream computations, while a downstream patch leaves upstream inputs unchanged so subsequent layers can re-encode the same signal. This is consistent with standard mechanistic-interpretability findings (cf. \citep{wang2023interpretability} IOI circuit: feature *encoded* ≠ feature *used*).
    75	
    76	### Three reads of the data
    77	
    78	1. Residual-stream cosine separation is a **necessary but not sufficient** signal of causal mechanism.
    79	2. Prompt-family information is **dispatchable** — small geometric perturbation at the decision layer produces large output displacement when patched.
    80	3. **Where a feature is most readable (L23) and where it is most consequential (L11-L17) are different layers** — paper-grade mechanism claims must report both, not collapse them.
    81	
    82	## Implications for paper §5
    83	
    84	**Strengthens 3-axis mechanism story**:
    85	- Axis-1 (text-format): Exp 1 cosine 0.029 + H-d cells causal patching (prior)
    86	- Axis-2 (prompt-family): Exp 1 cosine 0.011 + **Exp 5 cellhprompt causal patching (this)**
    87	- Axis-image: Exp 1 cosine 0.041 + indirect (H-text − cellhprompt residual ~0.04-0.05)
    88	
    89	**Defuses /stress critique** "you only have axis-1 mechanism":
    90	- Now have causal evidence for axis-2 separate from axis-1
    91	- 2-site cross-replication (cls + red), N=24 each, paired L0-L36 sweep
    92	
    93	**Reframes hero argument**: The paper §1 framing "text-format shapes exploration; prompt tunes commit" is now backed by:
    94	- Behavioral: exploration rate axis-1 dependent (Exp 1 cosine sigma + §4.5 reddit behavioral)
    95	- Causal mechanism: prompt-family mid-layer L11-L17 patching produces output displacement comparable to image-axis flip
    96	
    97	## Caveats
    98	
    99	- N=24 per cell — bootstrap CI on per-layer overlap means would tighten interpretation.
   100	- "phantom_som" archive vs "som" archive — the codebase uses same hidden-state extraction infrastructure but `--source-mode phantom_som` extracts without image. The pilot_summary template label "(with image — clean)" is a hardcoded artifact, not a runtime check. Manual verification of the archive contents (hidden state norms) would close a soundness gap.
   101	- Patching displacement is a token-level metric; doesn't directly translate to SR / drop-one oracle. Behavioral consequence (which paper §1 hero is about) operates on top of this causal signal.
   102	
   103	## Files
   104	
   105	- `pilot_summary.md`: per-site
   106	- `patching_continuation_results.json`: per-layer per-task continuation strings + metrics (~1.3 MB each)
   107	- `patching_continuation_curves.png`: visual layer profile
   108	
   109	## Provenance
   110	
   111	- Myriad jobs: 359511 (cls) + 359512 (red), submitted 2026-05-12, finished 21:42 + 21:54 UTC
   112	- Watcher missed GONE events due to silent-miss bug (PR same commit) — auto_pull dispatched manually
   113	- Bash invocation:
   114	  ```
   115	  bash scripts/maintenance/auto_pull_myriad_cell.sh 359511 cellhprm_cls stage3_cellhprompt_cls_fwd_ptext_myriad
   116	  bash scripts/maintenance/auto_pull_myriad_cell.sh 359512 cellhprm_red stage3_cellhprompt_red_fwd_ptext_myriad
   117	  ```

exec
/bin/bash -lc "nl -ba docs/checkpoints/mechanism/README.md | sed -n '1,80p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	---
     2	name: mechanism workspace
     3	description: Paper §5 mechanism-specific planning workspace — theory, methods, lit anchors, findings, open Qs, advisor sync. Splits the mechanism subset out of paper_planning.md once it grew paper-within-paper.
     4	type: workspace_index
     5	---
     6	
     7	# Mechanism Workspace — paper §5
     8	
     9	## Elevator pitch (1-paragraph mechanism story)
    10	
    11	P79 phantom routing space modes (DOM / SoM / Vision / P-text / P-prompt / P-SoM) are **linearly readable** from Qwen3-VL-4B residual stream at every layer 4–36 (Method 4.2 PCA cosine gap, AUROC 1.000 × 540 layer-pair tests). The image-axis dominates mechanism magnitude (peak gap 0.06 at L4–L17), text-axis is mid-scale (0.025 at L23), prompt-axis alone is weakest (0.007 at L36) — 10:4:1 hierarchy. P-SoM's closest mode at every layer is **P-text** (14.7× more distant from SoM), refuting "P-SoM = SoM minus image" and validating the format-axis framing. Mid-layer L17 acts as the **causally active planning site** (Stage 2/3 Cell A-H replacement patching disrupts output overlap; HDMI reliability framework gives Method 4.4 v2 mid-layer L17 α=5 = 0.44 reliability vs late-layer L33 α=10 = 0.23, because late-layer over-steers JSON envelope). This positions our work between Wu et al. 2026 (text-only Qwen 3 4B, tool selection 93% switch) and Ma & Rui 2026 (Qwen3 family rhyme newline 1% causal) — 50% mid-layer reliability is the multimodal-multi-step regime's signature.
    12	
    13	## Status snapshot (2026-05-11)
    14	
    15	| Stage | Status | Headline |
    16	|---|---|---|
    17	| Stage 1 (linear probe pilot) | ✅ done | L17 first-token logit shift peak |
    18	| Stage 2/3 (replacement patching, 10 cells) | ✅ done | 8/8 Holm-sig L17 disruption, Cell C cls null asymmetry |
    19	| Stage 4 Method 4.2 (PCA cosine gap) | ✅ done | AUROC 1.000 × 540 tests, 5/5 robustness pass |
    20	| Stage 4 Method 4.4 v2 (mean-diff steering) | ⏳ 8/48 cells (bg) | Smoke: L17 α=5 H-mean 0.44 = current sweet spot |
    21	| Method 4.5 (LA-HDMI / SAE) | 📋 future work | paper §8 anchor; Zekun-recommended SAE path |
    22	| Paper §5 prose rewrite | 📋 pending Codex round | Uses 5-anchor lit + identification protocol |
    23	
    24	## File index
    25	
    26	- **[plan.md](plan.md)** — full mechanism plan: theory framework, lit anchor mapping, method design, identification assumptions table, current findings, open Qs, Zekun sync state, roadmap
    27	
    28	## Cross-references
    29	
    30	| Source | What's there |
    31	|---|---|
    32	| `paper_planning.md` §2 | Full Zoom 1-4 framework with axis 1/2/3 mechanism detail (this folder summarizes, not duplicates) |
    33	| `paper_planning.md` §3 | 4-dimension evidence framework (Outcome / Macro / Micro / Efficiency) — mechanism evidence dimensions |
    34	| `paper_drafts/section5_*.md` | Final paper §5 prose (regenerate via codex rounds) |
    35	| `实验笔记.md` §125 | 2026-05-11 chronicle: Stage 4 lands + Method 4.2/4.4 + Wu/Ma&Rui/HDMI/Position/Peale integration |
    36	| `_status/section/section5.md` | Section-level frontmatter (status, last_codex_round) |
    37	| `_status/cells/cell_*.md` | Per-cell frontmatter (status, last_run_id) for 16 phase-1 cells + 10 mechanistic cells |
    38	| `paper.bib` | 5 mechanism anchors: `wu2026toolcalling` (2605.07990) / `maRui2026planning` (2605.07984) / `khorasani2026hdmi` (2605.07631) / `linLiu2026disclosure` (2605.08012) / `peale2026flexibleRouting` (2605.07805) |
    39	| `docs/literature/hdmi_paper_note.md` | HDMI deep dive (Method 4.4 v2 H-mean rescue) |
    40	| `docs/literature/wheres_the_plan_paper_note.md` | Ma & Rui probe-causal dissociation deep dive |
    41	
    42	## When to update this folder vs other docs
    43	
    44	- **plan.md updates**: when theory framing shifts, new identification assumption surfaces, advisor decision lands, or method strategy changes
    45	- **paper_planning.md §2 stays canonical** for Zoom 1-4 — this folder summarizes for working speed
    46	- **paper_drafts/section5_*.md stays canonical** for prose — this folder is the strategic substrate
    47	- **实验笔记.md** stays canonical for chronicle — this folder doesn't repeat history, points to it
    48	
    49	This folder is the **working substrate** between strategy (paper_planning) and prose (paper_drafts), specialized for mechanism. Treat it as a paper-within-paper workspace.

exec
/bin/bash -lc "nl -ba docs/checkpoints/mechanism/results/layer_axis_emergence.md | sed -n '1,70p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Stage 4: image-axis peak-layer split — Mirage Effect signature
     2	
     3	Eight mode pairs differ in image presence (one side has image, the other doesn't). Peak cosine-gap layer reveals **when** image-axis mechanism emerges:
     4	
     5	| no-image side | image side | no-img text | peak layer | peak cosine gap |
     6	|---|---|---|---|---|
     7	| DOM | SoM | AXTree | **L04** | 0.0604 |
     8	| DOM | Vision | AXTree | **L04** | 0.0653 |
     9	| P-prompt | SoM | AXTree | **L04** | 0.0600 |
    10	| P-prompt | Vision | AXTree | **L04** | 0.0649 |
    11	| P-SoM | SoM | [SOM_MARKS] | **L17** | 0.0412 |
    12	| P-text | SoM | [SOM_MARKS] | **L20** | 0.0494 |
    13	| P-text | Vision | [SOM_MARKS] | **L36** | 0.0614 |
    14	| P-SoM | Vision | [SOM_MARKS] | **L36** | 0.0613 |
    15	
    16	## Grouped by no-image side text format
    17	
    18	### no-image text = `AXTree` (mean peak L4)
    19	
    20	- DOM ↔ SoM: peak **L04** = 0.0604
    21	- DOM ↔ Vision: peak **L04** = 0.0653
    22	- P-prompt ↔ SoM: peak **L04** = 0.0600
    23	- P-prompt ↔ Vision: peak **L04** = 0.0649
    24	
    25	### no-image text = `[SOM_MARKS]` (mean peak L27)
    26	
    27	- P-text ↔ SoM: peak **L20** = 0.0494
    28	- P-text ↔ Vision: peak **L36** = 0.0614
    29	- P-SoM ↔ SoM: peak **L17** = 0.0412
    30	- P-SoM ↔ Vision: peak **L36** = 0.0613
    31	
    32	## Mechanism interpretation (paper §5 v3 Mirage anchor)
    33	
    34	When the no-image side carries `AXTree` text (DOM, P-prompt), the image-axis cosine gap peaks at **L04** — early-layer fresh image-presence detection (vision encoder + cross-modal fusion).
    35	
    36	When the no-image side carries `[SOM_MARKS]` text (P-text, P-SoM), the image-axis cosine gap peak shifts to **L17–L36** — image yes/no divergence is deferred to mid/output layers.
    37	
    38	**Mechanism story**: `[SOM_MARKS]` text in input primes an indexed-parsing pathway through mid-layer computation. The model processes marks structurally regardless of whether image is provided, producing image-axis divergence only at late integration stages. This is the direct Method 4.2 empirical anchor for the **Mirage Effect** (Asadi et al. 2026, VLM ~70-80% no-image accuracy) and **Cross-modal flow** (Kaduri et al., middle-layer cross-modal flows store image info in query tokens): the marks-primed mid-layer computation runs *as if image were available*, with image grounding contributed only late.
    39	
    40	**Paper §5 prose** (suggested):
    41	
    42	> *Method 4.2 reveals a peak-layer shift signature for the Mirage Effect: image-axis cosine-gap peak transitions from L04 (when no-image side carries AXTree text) to L17–L36 (when no-image side carries [SOM_MARKS] text). The peak-layer shift quantifies how text-payload format primes mid-layer computation pathways — [SOM_MARKS] format triggers indexed-parsing through mid-layers regardless of image presence, with image-axis divergence deferred to late integration. This identifies [SOM_MARKS] as the mechanism trigger for the Mirage Effect, anchoring Asadi et al. 2026's behavioral finding (~70-80% no-image VLM accuracy) and Kaduri et al.'s middle-layer cross-modal flow hypothesis with layer-resolved empirical evidence.*

exec
/bin/bash -lc "nl -ba docs/checkpoints/mechanism/results/axis2_logit_lens.md | sed -n '1,70p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Exp 3 — Logit lens at late layers (axis-2 vs axis-1)
     2	
     3	Apply Qwen3-VL-4B's final_norm + lm_head to per-layer per-mode mean hidden states.
     4	For each axis-isolated pair, compute KL(mode_a || mode_b) and top-1 token disagreement
     5	across all 37 layers. This probes whether axis-2 cosine signal (L23 peak 0.011) gets
     6	amplified into output distribution divergence by late-layer decoding.
     7	
     8	## Classifieds site
     9	
    10	### Axis-2 (prompt-family) pairs:
    11	
    12	| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
    13	|---|---:|---:|---:|---:|---:|
    14	| P-text vs P-SoM  (axis-2 flat-text) | **L23** | 0.1621 | 0.0215 | 0.1621 | 0.0003 |
    15	| DOM vs P-prompt  (axis-2 hierarchical) | **L25** | 0.0444 | 0.0184 | 0.0234 | 0.0000 |
    16	
    17	### Axis-1 (text-format) pairs:
    18	
    19	| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
    20	|---|---:|---:|---:|---:|---:|
    21	| DOM vs P-text    (axis-1 DOM-prompt) | **L23** | 0.5508 | 0.1299 | 0.5508 | 0.0001 |
    22	| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L23** | 0.6953 | 0.1069 | 0.6953 | 0.0003 |
    23	
    24	## Reddit site
    25	
    26	### Axis-2 (prompt-family) pairs:
    27	
    28	| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
    29	|---|---:|---:|---:|---:|---:|
    30	| P-text vs P-SoM  (axis-2 flat-text) | **L24** | 0.1260 | 0.0371 | 0.1230 | 0.0002 |
    31	| DOM vs P-prompt  (axis-2 hierarchical) | **L25** | 0.0508 | 0.0228 | 0.0325 | 0.0000 |
    32	
    33	### Axis-1 (text-format) pairs:
    34	
    35	| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
    36	|---|---:|---:|---:|---:|---:|
    37	| DOM vs P-text    (axis-1 DOM-prompt) | **L23** | 0.5273 | 0.0898 | 0.5273 | 0.0000 |
    38	| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L23** | 0.6172 | 0.0806 | 0.6172 | 0.0002 |
    39	
    40	## Interpretation
    41	
    42	Three hypotheses tested:
    43	
    44	- **H_A (axis-2 absent from output)**: axis-2 KL flat <0.1 at all layers → prompt-family
    45	  effect bypasses logit lens, only visible via attention heads or runtime decoding.
    46	- **H_B (axis-2 amplified at output)**: axis-2 KL peak at L30+ ≫ cosine 0.011 magnitude →
    47	  late-layer decoding amplifies prompt prior into output divergence (Wu et al. tool calling
    48	  'knows but says differently' mirror).
    49	- **H_C (axis-2 tracks residual stream)**: axis-2 KL peak at L23 same as cosine peak →
    50	  prompt prior signal proportional to mid-layer geometry, no amplification.
    51	
    52	Cross-site replication should hold for any of the three. Compare axis-2 KL magnitudes to
    53	axis-1 KL magnitudes to see whether 3-4x ratio in cosine space persists at output level.

exec
/bin/bash -lc "nl -ba scripts/provenance/snapshot_env.py | sed -n '1,260p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	"""Environment snapshot for paper-grade run provenance.
     2	
     3	Captures: torch / transformers / Python / git / hostname / GPU compute caps /
     4	HuggingFace model revision SHA. Dumped to <out_path> as JSON.
     5	
     6	Designed to be **fail-soft** — if HF API is unreachable or `git` is unavailable,
     7	records the failure mode in the snapshot rather than crashing the launch.
     8	
     9	Usage (standalone):
    10	    python3 scripts/provenance/snapshot_env.py results/<run_dir>/env_snapshot.json
    11	
    12	Usage (programmatic, hooked into run_experiment.py):
    13	    from scripts.provenance.snapshot_env import capture_env_snapshot
    14	    capture_env_snapshot(run_dir / "env_snapshot.json")
    15	
    16	Output schema (paper §3 / Appendix D quotable fields):
    17	    {
    18	      "captured_at": "2026-05-07T...Z",
    19	      "host": "spark-9ea3",
    20	      "platform": "Linux-6.11.0-aarch64",
    21	      "python_version": "3.12.x",
    22	      "torch": {"version": "2.11.0+cu128", "cuda": "12.8", "compute_caps": [[12,1]]},
    23	      "transformers": {"version": "4.46.x"},
    24	      "models": {"Qwen/Qwen3-VL-4B-Instruct": "<HF revision SHA>"},
    25	      "git": {"commit": "<SHA>", "dirty": false, "branch": "master"},
    26	      "errors": []
    27	    }
    28	"""
    29	
    30	from __future__ import annotations
    31	
    32	import argparse
    33	import hashlib
    34	import json
    35	import logging
    36	import platform
    37	import subprocess
    38	import sys
    39	from datetime import datetime, timezone
    40	from pathlib import Path
    41	from typing import Any
    42	
    43	# Files whose content materially affects scoring (paper-grade evaluator SHA).
    44	# Add new files here when introducing new score-affecting modules.
    45	EVALUATOR_SOURCE_FILES = [
    46	    "p79/experiment/analysis.py",      # compute_adjusted_success + FP rules
    47	    "p79/experiment/environment.py",   # VwaEvaluator wrapper
    48	    "p79/experiment/metrics.py",       # aggregate_condition_metrics (SR roll-up)
    49	]
    50	
    51	logger = logging.getLogger("snapshot-env")
    52	
    53	DEFAULT_MODELS = [
    54	    "Qwen/Qwen3-VL-4B-Instruct",
    55	]
    56	
    57	
    58	def _safe(fn, default=None, errors=None, label=""):
    59	    try:
    60	        return fn()
    61	    except Exception as e:
    62	        if errors is not None:
    63	            errors.append(f"{label}: {type(e).__name__}: {e}")
    64	        return default
    65	
    66	
    67	def capture_env_snapshot(
    68	    out_path: Path | str,
    69	    models: list[str] | None = None,
    70	    extra: dict[str, Any] | None = None,
    71	) -> dict[str, Any]:
    72	    out_path = Path(out_path)
    73	    out_path.parent.mkdir(parents=True, exist_ok=True)
    74	    errors: list[str] = []
    75	
    76	    snap: dict[str, Any] = {
    77	        "captured_at": datetime.now(timezone.utc).isoformat(),
    78	        "host": _safe(lambda: subprocess.check_output(["hostname"]).decode().strip(),
    79	                      default="unknown", errors=errors, label="hostname"),
    80	        "platform": platform.platform(),
    81	        "python_version": sys.version.split()[0],
    82	    }
    83	
    84	    # torch
    85	    def _torch_info():
    86	        import torch
    87	        return {
    88	            "version": torch.__version__,
    89	            "cuda": getattr(torch.version, "cuda", None),
    90	            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    91	            "compute_caps": [
    92	                list(torch.cuda.get_device_capability(i))
    93	                for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0)
    94	            ],
    95	            "device_names": [
    96	                torch.cuda.get_device_name(i)
    97	                for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0)
    98	            ],
    99	        }
   100	    snap["torch"] = _safe(_torch_info, default={}, errors=errors, label="torch")
   101	
   102	    # transformers / qwen-vl-utils
   103	    def _lib_versions():
   104	        out = {}
   105	        for lib in ["transformers", "qwen_vl_utils", "huggingface_hub", "numpy", "scikit_learn"]:
   106	            try:
   107	                mod = __import__(lib.replace("-", "_"))
   108	                out[lib] = getattr(mod, "__version__", "unknown")
   109	            except ImportError:
   110	                out[lib] = None
   111	        return out
   112	    snap["libraries"] = _safe(_lib_versions, default={}, errors=errors, label="libraries")
   113	
   114	    # HuggingFace model revisions (paper-grade: pin model SHA at launch)
   115	    models = models or DEFAULT_MODELS
   116	    snap["models"] = {}
   117	    def _hf_revision(model_id):
   118	        from huggingface_hub import HfApi
   119	        info = HfApi().model_info(model_id)
   120	        return info.sha
   121	    for m in models:
   122	        snap["models"][m] = _safe(
   123	            lambda: _hf_revision(m),
   124	            default="unavailable", errors=errors, label=f"hf:{m}"
   125	        )
   126	
   127	    # Evaluator code SHA — paper-grade: scoring logic must be pinnable independent
   128	    # of git commit (which can change for non-scoring reasons like docs).
   129	    def _evaluator_sha():
   130	        repo_root = Path(__file__).resolve().parents[2]
   131	        h = hashlib.sha256()
   132	        per_file = {}
   133	        for rel_path in EVALUATOR_SOURCE_FILES:
   134	            f = repo_root / rel_path
   135	            if not f.exists():
   136	                per_file[rel_path] = "MISSING"
   137	                continue
   138	            content = f.read_bytes()
   139	            file_h = hashlib.sha256(content).hexdigest()
   140	            per_file[rel_path] = file_h
   141	            h.update(content)
   142	        return {
   143	            "combined_sha256": h.hexdigest(),
   144	            "per_file_sha256": per_file,
   145	            "files": EVALUATOR_SOURCE_FILES,
   146	        }
   147	    snap["evaluator_code"] = _safe(_evaluator_sha, default={}, errors=errors, label="evaluator-sha")
   148	
   149	    # Git
   150	    def _git_info():
   151	        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
   152	        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
   153	        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode()
   154	        return {"commit": commit, "branch": branch, "dirty": bool(status.strip()), "status": status if status else None}
   155	    snap["git"] = _safe(_git_info, default={"unavailable": True}, errors=errors, label="git")
   156	
   157	    # GPU dump (NVML if available)
   158	    def _gpu_info():
   159	        out = subprocess.check_output(
   160	            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"],
   161	            stderr=subprocess.DEVNULL, timeout=5,
   162	        ).decode().strip()
   163	        return [line.strip() for line in out.splitlines()]
   164	    snap["nvidia_smi"] = _safe(_gpu_info, default=[], errors=errors, label="nvidia-smi")
   165	
   166	    if extra:
   167	        snap["extra"] = extra
   168	    snap["errors"] = errors
   169	
   170	    out_path.write_text(json.dumps(snap, indent=2))
   171	    logger.info(f"Env snapshot → {out_path} (errors: {len(errors)})")
   172	    return snap
   173	
   174	
   175	def main():
   176	    p = argparse.ArgumentParser()
   177	    p.add_argument("out_path", help="Output JSON path")
   178	    p.add_argument("--model", action="append", default=None,
   179	                   help="Override default model list (repeat for multiple)")
   180	    args = p.parse_args()
   181	
   182	    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
   183	    snap = capture_env_snapshot(args.out_path, models=args.model)
   184	    print(json.dumps(snap, indent=2))
   185	
   186	
   187	if __name__ == "__main__":
   188	    main()

exec
/bin/bash -lc 'rg -n "def capture_env_snapshot|git|sha|hash|manifest|model_revision|random_seed|CUBLAS|deterministic|manual_seed" scripts/provenance scripts/mechanistic scripts/analysis p79/mechanistic | head -n 200' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
p79/mechanistic/activation_patching.py:121:        input_len = inputs["input_ids"].shape[1]
p79/mechanistic/activation_patching.py:177:        input_len = inputs["input_ids"].shape[1]
p79/mechanistic/activation_patching.py:200:            model output dict (.logits at last position used for downstream metric)
p79/mechanistic/activation_patching.py:211:                if hs.shape != src.shape:
p79/mechanistic/activation_patching.py:213:                        f"shape mismatch for position='all': target {hs.shape} vs source {src.shape}"
p79/mechanistic/activation_patching.py:248:        logit_shift_to_source: (patched_logit_src - target_logit_src) /
p79/mechanistic/activation_patching.py:249:                              (source_logit_src - target_logit_src). 1.0 = full
p79/mechanistic/activation_patching.py:264:    source_logits = source_out.logits[0, -1, :].float().cpu()  # (vocab,)
p79/mechanistic/activation_patching.py:265:    source_probs = torch.softmax(source_logits, dim=-1)
p79/mechanistic/activation_patching.py:269:    target_logits = target_out.logits[0, -1, :].float().cpu()
p79/mechanistic/activation_patching.py:270:    target_probs = torch.softmax(target_logits, dim=-1)
p79/mechanistic/activation_patching.py:275:    logit_shift_to_source = []
p79/mechanistic/activation_patching.py:279:    denom_src_logit = source_logits[source_argmax].item() - target_logits[source_argmax].item()
p79/mechanistic/activation_patching.py:288:        patched_logits = patched_out.logits[0, -1, :].float().cpu()
p79/mechanistic/activation_patching.py:289:        patched_probs = torch.softmax(patched_logits, dim=-1)
p79/mechanistic/activation_patching.py:294:        if abs(denom_src_logit) > 1e-6:
p79/mechanistic/activation_patching.py:296:                (patched_logits[source_argmax].item() - target_logits[source_argmax].item())
p79/mechanistic/activation_patching.py:297:                / denom_src_logit
p79/mechanistic/activation_patching.py:301:        logit_shift_to_source.append(shift)
p79/mechanistic/activation_patching.py:311:                f"logit_shift={shift:.3f}, KL→src={kl_ps:.3f}, KL→tgt={kl_pt:.3f}"
p79/mechanistic/activation_patching.py:317:        "logit_shift_to_source": logit_shift_to_source,
p79/mechanistic/activation_patching.py:322:        "source_logit_at_argmax": float(source_logits[source_argmax].item()),
p79/mechanistic/activation_patching.py:323:        "target_logit_at_argmax": float(target_logits[target_argmax].item()),
p79/mechanistic/activation_patching.py:389:    src_input_len = source_inputs["input_ids"].shape[1]
p79/mechanistic/activation_patching.py:398:    tgt_input_len = target_inputs["input_ids"].shape[1]
p79/mechanistic/linear_probe.py:63:    n_samples, n_layers, hidden_dim = hidden_states.shape
p79/mechanistic/linear_probe.py:64:    if labels.shape[0] != n_samples:
p79/mechanistic/linear_probe.py:65:        raise ValueError(f"labels.shape[0] {labels.shape[0]} != n_samples {n_samples}")
p79/mechanistic/linear_probe.py:81:                effective_pca_dim = min(pca_dim, X_train.shape[0] - 1, X_train.shape[1])
p79/mechanistic/extract_hidden_states.py:41:        model_revision: str = "ebb281ec70b05090aa6165b016eac8ec08e71b17",
p79/mechanistic/extract_hidden_states.py:56:        logger.info(f"Loading {model_path} (revision={model_revision[:12]}...) for hidden state extraction (dtype={dtype})")
p79/mechanistic/extract_hidden_states.py:59:            revision=model_revision,
p79/mechanistic/extract_hidden_states.py:66:            model_path, revision=model_revision, trust_remote_code=True
p79/mechanistic/extract_hidden_states.py:69:        self.model_revision = model_revision
p79/mechanistic/extract_hidden_states.py:129:            Tensor of shape (n_layers + 1, hidden_dim). Layer 0 is embedding output;
p79/mechanistic/extract_hidden_states.py:168:        # outputs.hidden_states is tuple of (n_layers + 1) tensors of shape
scripts/analysis/power_analysis.py:36:    where p_avg ≈ baseline_sr (paired tasks share most variance).
scripts/analysis/stage4_logit_lens_axis2.py:2:"""Exp 3: Logit lens at late layers — does axis-2 prompt-family signal
scripts/analysis/stage4_logit_lens_axis2.py:23:  docs/checkpoints/mechanism/results/axis2_logit_lens.md
scripts/analysis/stage4_logit_lens_axis2.py:24:  results/phantom_paper/figures/fig_axis2_logit_lens.png
scripts/analysis/stage4_logit_lens_axis2.py:40:DEFAULT_MD = ROOT / "docs/checkpoints/mechanism/results/axis2_logit_lens.md"
scripts/analysis/stage4_logit_lens_axis2.py:41:DEFAULT_FIG = ROOT / "results/phantom_paper/figures/fig_axis2_logit_lens.png"
scripts/analysis/stage4_logit_lens_axis2.py:75:def logits_at_layer(hidden: torch.Tensor, lm_head, norm) -> torch.Tensor:
scripts/analysis/stage4_logit_lens_axis2.py:76:    """hidden: (D,) → logits (V,) after final_norm + lm_head."""
scripts/analysis/stage4_logit_lens_axis2.py:79:    logits = lm_head(h).squeeze(0)
scripts/analysis/stage4_logit_lens_axis2.py:80:    return logits
scripts/analysis/stage4_logit_lens_axis2.py:83:def kl_divergence(p_logits, q_logits) -> float:
scripts/analysis/stage4_logit_lens_axis2.py:84:    """KL(P || Q) with softmax on logits."""
scripts/analysis/stage4_logit_lens_axis2.py:85:    log_p = torch.log_softmax(p_logits, dim=-1)
scripts/analysis/stage4_logit_lens_axis2.py:86:    log_q = torch.log_softmax(q_logits, dim=-1)
scripts/analysis/stage4_logit_lens_axis2.py:92:def top1_agree(p_logits, q_logits) -> bool:
scripts/analysis/stage4_logit_lens_axis2.py:93:    return torch.argmax(p_logits).item() == torch.argmax(q_logits).item()
scripts/analysis/stage4_logit_lens_axis2.py:96:def compute_pair_logit_lens(npz: Path, pair_pairs: list, lm_head, norm, n_layers_use: int):
scripts/analysis/stage4_logit_lens_axis2.py:116:            l_a = logits_at_layer(h_a, lm_head, norm)
scripts/analysis/stage4_logit_lens_axis2.py:117:            l_b = logits_at_layer(h_b, lm_head, norm)
scripts/analysis/stage4_logit_lens_axis2.py:133:        "# Exp 3 — Logit lens at late layers (axis-2 vs axis-1)",
scripts/analysis/stage4_logit_lens_axis2.py:190:        "  effect bypasses logit lens, only visible via attention heads or runtime decoding.",
scripts/analysis/stage4_logit_lens_axis2.py:206:    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
scripts/analysis/stage4_logit_lens_axis2.py:222:    axes[0].set_ylabel("KL divergence (logit lens)")
scripts/analysis/stage4_logit_lens_axis2.py:223:    fig.suptitle("Exp 3: Output distribution divergence per layer via logit lens", fontsize=11)
scripts/analysis/stage4_logit_lens_axis2.py:242:    cls_a2, n_L = compute_pair_logit_lens(args.cls_npz, AXIS_2_PAIRS, lm_head, norm, 37)
scripts/analysis/stage4_logit_lens_axis2.py:246:    cls_a1, _ = compute_pair_logit_lens(args.cls_npz, AXIS_1_PAIRS, lm_head, norm, 37)
scripts/analysis/stage4_logit_lens_axis2.py:250:    red_a2, _ = compute_pair_logit_lens(args.red_npz, AXIS_2_PAIRS, lm_head, norm, 37)
scripts/analysis/stage4_logit_lens_axis2.py:252:    red_a1, _ = compute_pair_logit_lens(args.red_npz, AXIS_1_PAIRS, lm_head, norm, 37)
scripts/provenance/snapshot_env.py:3:Captures: torch / transformers / Python / git / hostname / GPU compute caps /
scripts/provenance/snapshot_env.py:6:Designed to be **fail-soft** — if HF API is unreachable or `git` is unavailable,
scripts/provenance/snapshot_env.py:25:      "git": {"commit": "<SHA>", "dirty": false, "branch": "master"},
scripts/provenance/snapshot_env.py:33:import hashlib
scripts/provenance/snapshot_env.py:67:def capture_env_snapshot(
scripts/provenance/snapshot_env.py:120:        return info.sha
scripts/provenance/snapshot_env.py:128:    # of git commit (which can change for non-scoring reasons like docs).
scripts/provenance/snapshot_env.py:129:    def _evaluator_sha():
scripts/provenance/snapshot_env.py:131:        h = hashlib.sha256()
scripts/provenance/snapshot_env.py:139:            file_h = hashlib.sha256(content).hexdigest()
scripts/provenance/snapshot_env.py:143:            "combined_sha256": h.hexdigest(),
scripts/provenance/snapshot_env.py:144:            "per_file_sha256": per_file,
scripts/provenance/snapshot_env.py:147:    snap["evaluator_code"] = _safe(_evaluator_sha, default={}, errors=errors, label="evaluator-sha")
scripts/provenance/snapshot_env.py:150:    def _git_info():
scripts/provenance/snapshot_env.py:151:        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
scripts/provenance/snapshot_env.py:152:        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
scripts/provenance/snapshot_env.py:153:        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode()
scripts/provenance/snapshot_env.py:155:    snap["git"] = _safe(_git_info, default={"unavailable": True}, errors=errors, label="git")
scripts/mechanistic/run_stage1_pilot.py:296:        f"Extracted hidden_states shape={hidden_states.shape} "
scripts/mechanistic/run_stage1_pilot.py:357:- N items: {hidden_states.shape[0]}
scripts/mechanistic/run_stage1_pilot.py:370:- AUROC sharp drop at deep layer → model "abstracts away" mode; mirage becomes task-relevant only
scripts/mechanistic/run_stage4_method44_v2_sweep.py:57:MANIFEST = ARCHIVE / "manifest.json"
scripts/mechanistic/run_stage4_method44_v2_sweep.py:115:    manifest = json.loads(MANIFEST.read_text())
scripts/mechanistic/run_stage4_method44_v2_sweep.py:116:    tasks = manifest[args.tier][:args.limit]
scripts/mechanistic/run_stage4_method44_v2_sweep.py:117:    steps = manifest.get("steps", [2, 5])
scripts/mechanistic/run_stage4_method44_v2_sweep.py:141:            dom_tokens = dom_gen.sequences[0, dom_inputs["input_ids"].shape[1]:].cpu().tolist()
scripts/mechanistic/run_stage4_method44_v2_sweep.py:146:            psom_tokens = psom_gen.sequences[0, psom_inputs["input_ids"].shape[1]:].cpu().tolist()
scripts/provenance/snapshot_vwa.sh:5:# HTML SHA-256 hash. Used for paper §3 / Appendix D byte-equivalence claim
scripts/provenance/snapshot_vwa.sh:86:# Per-site HTTP probe + content hash
scripts/provenance/snapshot_vwa.sh:91:    body_hash = run(["bash", "-c", f"curl -sS --max-time 10 '{url}' | sha256sum | awk '{{print \$1}}'"])
scripts/provenance/snapshot_vwa.sh:95:        "body_sha256": body_hash if body_hash else None,
scripts/analysis/stage4_axis2_layer_profile.py:59:    n_layers = H.shape[1]
scripts/analysis/stage4_axis2_layer_profile.py:131:        "1. **Truly null geometry** — axis-2 pair curves flat <0.01 at all layers. Prompt-family bypasses residual stream entirely (acts at attention pattern or output head). → Next: Exp 3 logit lens or Exp 4 attention probe.",
scripts/analysis/stage4_axis2_layer_profile.py:133:        "3. **Early-layer spike absorbed** — axis-2 pair curves spike at L0-L5 then collapse to ~0. Prompt embedding effect absorbed by mid-layer fusion. → Next: Exp 3 logit lens to verify if it re-emerges in output distribution.",
scripts/analysis/stage4_axis2_layer_profile.py:143:    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
scripts/analysis/analyze_cross_representation.py:373:      url_match excluded: navigating to correct page without finish is legitimate.
scripts/analysis/analyze_cross_representation.py:1515:                # No cost, no steps — pick first deterministically.
scripts/analysis/mechanism_per_task.py:739:def fmt(value: Any, digits: int = 3) -> str:
scripts/analysis/mechanism_per_task.py:743:        return f"{value:.{digits}f}"
scripts/analysis/mechanism_per_task.py:747:def fmt_pct(value: Any, digits: int = 1) -> str:
scripts/analysis/mechanism_per_task.py:750:    return f"{100.0 * float(value):.{digits}f}%"
scripts/analysis/mechanism_per_task.py:954:        "Together, E1 and E2 support a decision-path account: mode swaps change which URL transitions are attempted and how early trajectories split on tasks where outcomes disagree. E3 keeps the commitment-confidence claim separate from path choice: confidence evidence is useful, but existing B0 outputs support it mainly through verbalized and behavioral AUROC rather than token calibration. E4 shows whether those path changes are accompanied by broad policy-shape shifts in the action vocabulary, or whether the same action mix hides different click targets.",
scripts/analysis/aggregate_phantom_lift.py:594:    def maybe_round(value, ndigits=4):
scripts/analysis/aggregate_phantom_lift.py:595:        return None if value is None else round(value, ndigits)
scripts/analysis/aggregate_phantom_lift.py:772:            "Check `results/phantom_paper/run_manifest.yaml` and "
scripts/mechanistic/extract_archive_subset.py:2:subset suitable for git commit + A100/Myriad transfer.
scripts/mechanistic/extract_archive_subset.py:12:Total ~25MB (vs full archive 1.8GB), git-committable for A100 launch.
scripts/mechanistic/extract_archive_subset.py:21:Result manifest: `results/mechanistic/archive_subset_b1_cls/manifest.json`
scripts/mechanistic/extract_archive_subset.py:140:    manifest = {
scripts/mechanistic/extract_archive_subset.py:194:                manifest[tier_name].append(entry)
scripts/mechanistic/extract_archive_subset.py:196:                manifest["skipped"].append({"task_id": task_id, "tier": tier_name, "reason": "no artifact files"})
scripts/mechanistic/extract_archive_subset.py:198:    # 4. Save manifest
scripts/mechanistic/extract_archive_subset.py:199:    manifest_file = output_dir / "manifest.json"
scripts/mechanistic/extract_archive_subset.py:200:    with manifest_file.open("w") as f:
scripts/mechanistic/extract_archive_subset.py:201:        json.dump(manifest, f, indent=2)
scripts/mechanistic/extract_archive_subset.py:202:    logger.info(f"Saved manifest: {manifest_file}")
scripts/mechanistic/extract_archive_subset.py:218:        f"- Strong: {len(manifest['strong'])} tasks × {len(args.steps)} steps = up to {len(manifest['strong']) * len(args.steps)} (task, step) artifacts",
scripts/mechanistic/extract_archive_subset.py:219:        f"- Reverse: {len(manifest['reverse'])} tasks × {len(args.steps)} steps",
scripts/mechanistic/extract_archive_subset.py:220:        f"- Skipped (no artifact): {len(manifest['skipped'])}",
scripts/mechanistic/extract_archive_subset.py:228:    for i, e in enumerate(manifest["strong"][:24], 1):
scripts/mechanistic/extract_archive_subset.py:238:    for i, e in enumerate(manifest["reverse"], 1):
scripts/mechanistic/extract_archive_subset.py:247:        f"Extracted: {len(manifest['strong'])} strong + {len(manifest['reverse'])} reverse tasks\n"
scripts/analysis/stage4_w6_l04_tokenization.py:9:Hypothesis to test: L17-peak variants share special-character-leading tokens
scripts/analysis/stage4_w6_l04_tokenization.py:43:    ("hash_id_control", "L04", "#a3f7 Submit"),
scripts/analysis/stage4_h1_per_task_fragility.py:26:               "plain_numbered", "xml_tagged", "hash_id_control"]  # 7 flat-list (incl no-int control)
scripts/analysis/stage4_h1_per_task_fragility.py:42:    n_layers = H.shape[1]
scripts/analysis/stage4_layer_axis_emergence.py:58:    n_layers = H.shape[1]
scripts/analysis/stage4_format_variation_analyze.py:12:  controls (hash_id_control / plain_sentence) → V↔som peak L04 (like dom)
scripts/analysis/stage4_format_variation_analyze.py:35:             "hash_id_control", "plain_sentence",  # 2 controls
scripts/analysis/stage4_format_variation_analyze.py:45:    "hash_id_control": "#hash label (no integer)",
scripts/analysis/stage4_format_variation_analyze.py:59:    "hash_id_control": "control (no integer)",
scripts/analysis/stage4_format_variation_analyze.py:81:    n_layers = H.shape[1]
scripts/analysis/stage4_format_variation_analyze.py:82:    print(f"loaded {H.shape} from {args.input}, n modes = {len(set(ml.tolist()))}")
scripts/analysis/stage4_format_variation_analyze.py:165:    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
scripts/analysis/stage4_format_variation_analyze.py:189:                       "plain_numbered", "xml_tagged", "hash_id_control", "plain_sentence"]
scripts/analysis/stage4_format_variation_analyze.py:204:                    "+ hash_id_control + plain_sentence)", fontsize=10, fontweight="bold")
scripts/analysis/stage4_format_variation_analyze.py:223:        elif v in ("hash_id_control", "plain_sentence"):
scripts/analysis/stage4_format_variation_analyze.py:239:                  "(broader than indexed-list — hash IDs + plain sentence also trigger; only AXTree hierarchical defeats)",
scripts/mechanistic/run_stage2_patching_pilot.py:177:    metrics = ["argmax_match_source", "logit_shift_to_source", "kl_patched_to_source", "kl_patched_to_target"]
scripts/mechanistic/run_stage2_patching_pilot.py:193:        "logit_shift_to_source": "Logit shift toward source argmax\n(1=full shift, 0=none)",
scripts/mechanistic/run_stage2_patching_pilot.py:217:    ls = np.array(agg["logit_shift_to_source_mean"])
scripts/mechanistic/run_stage2_patching_pilot.py:232:- Best layer for **logit shift to source**: L{best_ls_layer} (shift {ls[best_ls_layer]:.3f})
scripts/mechanistic/run_stage2_patching_pilot.py:239:- logit_shift_to_source: smooth metric; 1.0 = patched output fully matches source on argmax token
scripts/analysis/hero_claim_bootstrap.py:121:    n = M.shape[0]
scripts/analysis/hero_claim_bootstrap.py:124:    other_idx = [j for j in range(M.shape[1]) if j != drop_mode_idx]
scripts/analysis/hero_claim_bootstrap.py:167:        n = M.shape[0]
scripts/mechanistic/run_stage4_multimode_extract.py:127:        manifest_path = archive_dir / "manifest.json"
scripts/mechanistic/run_stage4_multimode_extract.py:128:        if manifest_path.exists():
scripts/mechanistic/run_stage4_multimode_extract.py:130:                manifest = json.load(open(manifest_path))
scripts/mechanistic/run_stage4_multimode_extract.py:132:                    for item in manifest.get(bucket, []):
scripts/mechanistic/run_stage4_multimode_extract.py:137:                logger.info(f"Loaded {len(intents_by_tid)} intents from {manifest_path}")
scripts/mechanistic/run_stage4_multimode_extract.py:139:                logger.warning(f"failed to load intents from {manifest_path}: {e}")
scripts/mechanistic/run_stage4_multimode_extract.py:142:            "no intents loaded from external config or archive manifest; "
scripts/mechanistic/run_stage4_multimode_extract.py:172:                    # hs shape (n_layers+1, hidden_dim)
scripts/analysis/stage2_layer_significance.py:53:    """Return shape (n_tasks, n_layers) ndarray for a given metric.
scripts/analysis/stage2_layer_significance.py:96:    n_tasks = grid.shape[0]
scripts/analysis/stage2_layer_significance.py:184:        "fwd_n": int(fwd_grid.shape[0]),
scripts/analysis/stage2_layer_significance.py:185:        "rev_n": int(rev_grid.shape[0]),
scripts/analysis/stage2_layer_significance.py:240:    out.append("Are forward and reverse disruption magnitudes statistically distinguishable?")
scripts/analysis/stage2_layer_significance.py:252:    out.append("- **Cross-direction p**: small p indicates forward and reverse magnitudes differ; large p means they're indistinguishable (consistent with bidirectional mechanism).")
scripts/analysis/stage4_pca_cosine_gap.py:66:    n_layers = H.shape[1]
scripts/analysis/stage4_pca_cosine_gap.py:67:    print(f"[stage4] loaded {H.shape[0]} examples × {n_layers} layers × {H.shape[2]} dim")
scripts/analysis/stage4_pca_cosine_gap.py:94:            if X.shape[0] >= 11:
scripts/analysis/stage4_pca_cosine_gap.py:95:                n_comp = min(10, X.shape[0] - 1)
scripts/analysis/stage4_pca_cosine_gap.py:105:        "n_examples": int(H.shape[0]), "n_layers": int(n_layers), "n_modes": len(MODES),
scripts/mechanistic/run_stage4_h1_phi35.py:17:import hashlib
scripts/mechanistic/run_stage4_h1_phi35.py:55:def hash_id(n):
scripts/mechanistic/run_stage4_h1_phi35.py:56:    h = hashlib.md5(str(n).encode()).hexdigest()
scripts/mechanistic/run_stage4_h1_phi35.py:85:def fmt_hash_id_control(obs_text):
scripts/mechanistic/run_stage4_h1_phi35.py:86:    return "\n".join(f"#{hash_id(n)} {label}" for n, role, label in extract_marks(obs_text))
scripts/mechanistic/run_stage4_h1_phi35.py:100:    "hash_id_control":  fmt_hash_id_control,
scripts/mechanistic/run_stage4_h1_phi35.py:149:    manifest = json.loads((archive_dir / "manifest.json").read_text())
scripts/mechanistic/run_stage4_h1_phi35.py:150:    tasks = manifest[args.tier][:args.n_tasks]
scripts/mechanistic/run_stage4_h1_phi35.py:206:    logger.info(f"saved: {out_path}  shape={H.shape}  modes={ALL_MODES}")
scripts/mechanistic/run_stage4_h1_phi35.py:211:        f"Shape: {H.shape}\n"
scripts/mechanistic/run_stage4_format_variation_extract.py:10:  - Non-marks variants (plain sentence / hash IDs)
scripts/mechanistic/run_stage4_format_variation_extract.py:21:  hash_id_control   #aBc7 label                         (random hash, no integer index — control)
scripts/mechanistic/run_stage4_format_variation_extract.py:27:shape (10 modes × 24 tasks × 2 steps = 480, n_layers+1, hidden_dim)
scripts/mechanistic/run_stage4_format_variation_extract.py:37:import hashlib
scripts/mechanistic/run_stage4_format_variation_extract.py:73:def hash_id(n: int) -> str:
scripts/mechanistic/run_stage4_format_variation_extract.py:74:    """Deterministic 4-char alphanumeric hash, no integer pattern."""
scripts/mechanistic/run_stage4_format_variation_extract.py:75:    h = hashlib.md5(str(n).encode()).hexdigest()
scripts/mechanistic/run_stage4_format_variation_extract.py:76:    # Avoid pure digits — mix in letters
scripts/mechanistic/run_stage4_format_variation_extract.py:112:def fmt_hash_id_control(obs_text):
scripts/mechanistic/run_stage4_format_variation_extract.py:113:    """Control: replace integer index with non-integer hash. Tests whether integer index is the trigger."""
scripts/mechanistic/run_stage4_format_variation_extract.py:115:    return "\n".join(f"#{hash_id(n)} {label}" for n, role, label in marks)
scripts/mechanistic/run_stage4_format_variation_extract.py:131:    "hash_id_control":  fmt_hash_id_control,
scripts/mechanistic/run_stage4_format_variation_extract.py:137:    """Locate manifest archive — on DGX vs Myriad have different paths."""
scripts/mechanistic/run_stage4_format_variation_extract.py:159:    manifest_path = archive_dir / "manifest.json"
scripts/mechanistic/run_stage4_format_variation_extract.py:160:    manifest = json.loads(manifest_path.read_text())
scripts/mechanistic/run_stage4_format_variation_extract.py:161:    tasks = manifest[args.tier][:args.n_tasks]
scripts/mechanistic/run_stage4_format_variation_extract.py:234:    logger.info(f"saved: {out_path}  shape={H.shape}  modes={ALL_MODES}")
scripts/mechanistic/run_stage4_format_variation_extract.py:240:        f"Shape: {H.shape}\n"
scripts/analysis/aggregate_cross_site.py:285:        help="Run directories (default: paper VWA runs from run_manifest.yaml)",
scripts/analysis/validate_run.py:403:        f"All {len(mode_tasks)} conditions share the same {len(all_tasks)} task IDs",
scripts/analysis/analyze_reason_diagnostics.py:385:    # (e.g. "12") would otherwise match arbitrary digits in DOM text — prices,

exec
/bin/bash -lc "find results/mechanistic -maxdepth 3 \\( -name env_snapshot.json -o -name run_manifest.json -o -name manifest.json \\) | sort | head -n 80" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
results/mechanistic/archive_subset_b1_cls/manifest.json
results/mechanistic/archive_subset_b1_reddit/manifest.json
results/mechanistic/stage2b_2x2_fwd_revtasks_myriad/env_snapshot.json
results/mechanistic/stage2b_2x2_fwd_revtasks_myriad/run_manifest.json
results/mechanistic/stage2b_cellcr_reddit_fwd_revtier_myriad/env_snapshot.json
results/mechanistic/stage2b_cellcr_reddit_fwd_revtier_myriad/run_manifest.json
results/mechanistic/stage2b_celle_random_cls_strong_myriad/env_snapshot.json
results/mechanistic/stage2b_celle_random_cls_strong_myriad/run_manifest.json
results/mechanistic/stage2b_celler_reddit_fwd_random_myriad/env_snapshot.json
results/mechanistic/stage2b_celler_reddit_fwd_random_myriad/run_manifest.json
results/mechanistic/stage2b_cellf_fwd_reddit_strong_myriad/env_snapshot.json
results/mechanistic/stage2b_cellf_fwd_reddit_strong_myriad/run_manifest.json
results/mechanistic/stage2b_curated_b1_cls_myriad/env_snapshot.json
results/mechanistic/stage2b_curated_b1_cls_myriad/run_manifest.json
results/mechanistic/stage2c_2x2_rev_strongtasks_myriad/env_snapshot.json
results/mechanistic/stage2c_2x2_rev_strongtasks_myriad/run_manifest.json
results/mechanistic/stage2c_celldr_reddit_rev_strongtier_myriad/env_snapshot.json
results/mechanistic/stage2c_celldr_reddit_rev_strongtier_myriad/run_manifest.json
results/mechanistic/stage2c_cellg_rev_reddit_reverse_myriad/env_snapshot.json
results/mechanistic/stage2c_cellg_rev_reddit_reverse_myriad/run_manifest.json
results/mechanistic/stage2c_reverse_curated_b1_cls_myriad/env_snapshot.json
results/mechanistic/stage2c_reverse_curated_b1_cls_myriad/run_manifest.json
results/mechanistic/stage3_cellhd_cls_fwd_dom_myriad/env_snapshot.json
results/mechanistic/stage3_cellhd_cls_fwd_dom_myriad/run_manifest.json
results/mechanistic/stage3_cellhd_red_fwd_dom_myriad/env_snapshot.json
results/mechanistic/stage3_cellhd_red_fwd_dom_myriad/run_manifest.json
results/mechanistic/stage3_cellhp_cls_fwd_prompt_myriad/env_snapshot.json
results/mechanistic/stage3_cellhp_cls_fwd_prompt_myriad/run_manifest.json
results/mechanistic/stage3_cellhp_red_fwd_prompt_myriad/env_snapshot.json
results/mechanistic/stage3_cellhp_red_fwd_prompt_myriad/run_manifest.json
results/mechanistic/stage3_cellhprompt_cls_fwd_ptext_myriad/env_snapshot.json
results/mechanistic/stage3_cellhprompt_cls_fwd_ptext_myriad/run_manifest.json
results/mechanistic/stage3_cellhprompt_red_fwd_ptext_myriad/env_snapshot.json
results/mechanistic/stage3_cellhprompt_red_fwd_ptext_myriad/run_manifest.json
results/mechanistic/stage3_cellht_cls_fwd_text_myriad/env_snapshot.json
results/mechanistic/stage3_cellht_cls_fwd_text_myriad/run_manifest.json
results/mechanistic/stage3_cellht_red_fwd_text_myriad/env_snapshot.json
results/mechanistic/stage3_cellht_red_fwd_text_myriad/run_manifest.json

exec
/bin/bash -lc "nl -ba scripts/mechanistic/extract_archive_subset.py | sed -n '1,270p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	"""Extract 24 strong + 11 reverse mirage candidate task artifacts to a compact
     2	subset suitable for git commit + A100/Myriad transfer.
     3	
     4	Reads `results/mechanistic/curate_mirage_b1_classifieds/candidates.jsonl`
     5	(produced by `curate_mirage_tasks.py`, 笔记 §113), filters by composite +
     6	token_overlap criteria, and copies per-(task, step) artifacts:
     7	- observation_dom.txt
     8	- screenshot_annotated.png
     9	
    10	Output: `results/mechanistic/archive_subset_b1_cls/<site>_task_<id>/step_<NNN>/`
    11	
    12	Total ~25MB (vs full archive 1.8GB), git-committable for A100 launch.
    13	
    14	Usage:
    15	    python3 scripts/mechanistic/extract_archive_subset.py
    16	    # or with custom thresholds:
    17	    python3 scripts/mechanistic/extract_archive_subset.py \
    18	      --strong-min-composite 1.0 --strong-max-overlap 0.5 \
    19	      --reverse-max-composite -1.5
    20	
    21	Result manifest: `results/mechanistic/archive_subset_b1_cls/manifest.json`
    22	listing all extracted task IDs + step indices + tier (strong/reverse).
    23	"""
    24	
    25	from __future__ import annotations
    26	
    27	import argparse
    28	import json
    29	import logging
    30	import shutil
    31	import sys
    32	from pathlib import Path
    33	
    34	logging.basicConfig(
    35	    level=logging.INFO,
    36	    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    37	    datefmt="%H:%M:%S",
    38	)
    39	logger = logging.getLogger("extract-subset")
    40	
    41	REPO_ROOT = Path(__file__).resolve().parents[2]
    42	
    43	
    44	def find_artifacts_dir(run_dir: Path) -> Path:
    45	    for child in run_dir.iterdir():
    46	        if child.is_dir() and (child / "artifacts").is_dir():
    47	            return child / "artifacts"
    48	    raise FileNotFoundError(f"No condition subdir with artifacts/ in {run_dir}")
    49	
    50	
    51	def main():
    52	    p = argparse.ArgumentParser()
    53	    p.add_argument(
    54	        "--candidates-jsonl",
    55	        default=str(REPO_ROOT / "results/mechanistic/curate_mirage_b1_classifieds/candidates.jsonl"),
    56	    )
    57	    p.add_argument(
    58	        "--archived-run-dir",
    59	        default=str(REPO_ROOT / "results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428"),
    60	    )
    61	    p.add_argument(
    62	        "--output-dir",
    63	        default=str(REPO_ROOT / "results/mechanistic/archive_subset_b1_cls"),
    64	    )
    65	    p.add_argument("--site", default="classifieds")
    66	    p.add_argument(
    67	        "--steps", nargs="+", type=int, default=[2, 5],
    68	        help="Step indices to extract per task (default [2, 5]).",
    69	    )
    70	    p.add_argument(
    71	        "--strong-min-composite", type=float, default=1.0,
    72	        help="Strong tier: composite >= this (paper-grade mirage candidates)",
    73	    )
    74	    p.add_argument(
    75	        "--strong-max-overlap", type=float, default=0.5,
    76	        help="Strong tier: token_overlap < this (real divergence not envelope)",
    77	    )
    78	    p.add_argument(
    79	        "--reverse-max-composite", type=float, default=-1.5,
    80	        help="Reverse tier: composite <= this (paper §5 robustness check)",
    81	    )
    82	    p.add_argument(
    83	        "--artifacts-subdir", default=None,
    84	        help="Override condition subdir name. For multi-mode archived runs "
    85	             "(e.g. B1_3mode_reddit_20260413 has phase1_{dom,som,vision}_router_0), "
    86	             "find_artifacts_dir picks first-iterated which may be wrong condition. "
    87	             "Set explicitly: e.g. --artifacts-subdir phase1_som_router_0.",
    88	    )
    89	    args = p.parse_args()
    90	
    91	    candidates_jsonl = Path(args.candidates_jsonl)
    92	    if not candidates_jsonl.exists():
    93	        logger.error(f"candidates.jsonl not found: {candidates_jsonl}")
    94	        logger.error("  Run scripts/mechanistic/curate_mirage_tasks.py first.")
    95	        sys.exit(1)
    96	
    97	    archived_dir = Path(args.archived_run_dir)
    98	    if not archived_dir.is_dir():
    99	        logger.error(f"archived run dir not found: {archived_dir}")
   100	        sys.exit(1)
   101	    if args.artifacts_subdir:
   102	        artifacts_dir = archived_dir / args.artifacts_subdir / "artifacts"
   103	        if not artifacts_dir.is_dir():
   104	            logger.error(f"--artifacts-subdir resolved to {artifacts_dir} (does not exist)")
   105	            sys.exit(1)
   106	    else:
   107	        artifacts_dir = find_artifacts_dir(archived_dir)
   108	    logger.info(f"Source artifacts: {artifacts_dir}")
   109	
   110	    output_dir = Path(args.output_dir)
   111	    output_dir.mkdir(parents=True, exist_ok=True)
   112	    logger.info(f"Output: {output_dir}")
   113	
   114	    # 1. Load candidates
   115	    candidates = [json.loads(line) for line in candidates_jsonl.open()]
   116	    logger.info(f"Loaded {len(candidates)} candidates from jsonl")
   117	
   118	    # 2. Filter by tier
   119	    strong = [
   120	        c for c in candidates
   121	        if c["composite"] >= args.strong_min_composite
   122	        and c["token_overlap"] < args.strong_max_overlap
   123	    ]
   124	    reverse = [
   125	        c for c in candidates
   126	        if c["composite"] <= args.reverse_max_composite
   127	    ]
   128	
   129	    # Order: strong by composite desc, reverse by composite asc (most negative first)
   130	    strong.sort(key=lambda c: c["composite"], reverse=True)
   131	    reverse.sort(key=lambda c: c["composite"])
   132	
   133	    logger.info(
   134	        f"Filtered: {len(strong)} strong (composite ≥ {args.strong_min_composite}, "
   135	        f"overlap < {args.strong_max_overlap}), {len(reverse)} reverse "
   136	        f"(composite ≤ {args.reverse_max_composite})"
   137	    )
   138	
   139	    # 3. Copy artifacts
   140	    manifest = {
   141	        "config": vars(args),
   142	        "site": args.site,
   143	        "steps": args.steps,
   144	        "strong": [],
   145	        "reverse": [],
   146	        "skipped": [],
   147	    }
   148	
   149	    total_bytes = 0
   150	
   151	    for tier_name, tier in [("strong", strong), ("reverse", reverse)]:
   152	        for c in tier:
   153	            task_id = c["task_id"]
   154	            entry = {
   155	                "task_id": task_id,
   156	                "intent": c["intent"],
   157	                "composite": c["composite"],
   158	                "token_overlap": c["token_overlap"],
   159	                "src_neg": c["src_neg"],
   160	                "src_aff": c["src_aff"],
   161	                "tgt_neg": c["tgt_neg"],
   162	                "tgt_aff": c["tgt_aff"],
   163	                "source_text": c["source_text"],
   164	                "target_text": c["target_text"],
   165	                "steps_extracted": [],
   166	            }
   167	            task_src = artifacts_dir / f"{args.site}_task_{task_id}"
   168	            task_dst = output_dir / f"{args.site}_task_{task_id}"
   169	
   170	            for step_idx in args.steps:
   171	                step_src = task_src / f"step_{step_idx:03d}"
   172	                if not step_src.is_dir():
   173	                    continue
   174	                step_dst = task_dst / f"step_{step_idx:03d}"
   175	                step_dst.mkdir(parents=True, exist_ok=True)
   176	
   177	                copied_step_files = []
   178	                for fname in ["observation_dom.txt", "screenshot_annotated.png"]:
   179	                    fsrc = step_src / fname
   180	                    if fsrc.exists():
   181	                        fdst = step_dst / fname
   182	                        shutil.copy2(fsrc, fdst)
   183	                        bytes_copied = fdst.stat().st_size
   184	                        total_bytes += bytes_copied
   185	                        copied_step_files.append(fname)
   186	
   187	                if copied_step_files:
   188	                    entry["steps_extracted"].append({
   189	                        "step_idx": step_idx,
   190	                        "files": copied_step_files,
   191	                    })
   192	
   193	            if entry["steps_extracted"]:
   194	                manifest[tier_name].append(entry)
   195	            else:
   196	                manifest["skipped"].append({"task_id": task_id, "tier": tier_name, "reason": "no artifact files"})
   197	
   198	    # 4. Save manifest
   199	    manifest_file = output_dir / "manifest.json"
   200	    with manifest_file.open("w") as f:
   201	        json.dump(manifest, f, indent=2)
   202	    logger.info(f"Saved manifest: {manifest_file}")
   203	
   204	    # 5. Human-readable README
   205	    readme_lines = [
   206	        f"# Mirage Candidate Subset — {args.site} (B1 Qwen3-VL-4B)",
   207	        "",
   208	        f"Extracted from `results/mechanistic/curate_mirage_b1_classifieds/candidates.jsonl` "
   209	        f"(笔记 §113, commit `cd50c34`). Used as paper-grade mirage dataset for Stage 2B "
   210	        f"curated scale-up + Stage 2C reverse-direction asymmetry confirm on A100.",
   211	        "",
   212	        "## Filter criteria",
   213	        f"- **Strong tier** (paper-grade mirage candidates): composite ≥ {args.strong_min_composite} "
   214	        f"AND token_overlap < {args.strong_max_overlap}",
   215	        f"- **Reverse tier** (asymmetry robustness): composite ≤ {args.reverse_max_composite}",
   216	        "",
   217	        "## Counts",
   218	        f"- Strong: {len(manifest['strong'])} tasks × {len(args.steps)} steps = up to {len(manifest['strong']) * len(args.steps)} (task, step) artifacts",
   219	        f"- Reverse: {len(manifest['reverse'])} tasks × {len(args.steps)} steps",
   220	        f"- Skipped (no artifact): {len(manifest['skipped'])}",
   221	        f"- **Total disk**: {total_bytes / 1e6:.1f} MB",
   222	        "",
   223	        "## Strong candidates (top 24, sorted by composite desc)",
   224	        "",
   225	        "| Rank | task_id | composite | overlap | intent (50 char) |",
   226	        "|---|---|---|---|---|",
   227	    ]
   228	    for i, e in enumerate(manifest["strong"][:24], 1):
   229	        readme_lines.append(
   230	            f"| {i} | {e['task_id']} | {e['composite']:+.2f} | {e['token_overlap']:.2f} | "
   231	            f"{e['intent'][:50]} |"
   232	        )
   233	    readme_lines.append("")
   234	    readme_lines.append("## Reverse candidates (sorted by composite asc)")
   235	    readme_lines.append("")
   236	    readme_lines.append("| Rank | task_id | composite | overlap | intent (50 char) |")
   237	    readme_lines.append("|---|---|---|---|---|")
   238	    for i, e in enumerate(manifest["reverse"], 1):
   239	        readme_lines.append(
   240	            f"| {i} | {e['task_id']} | {e['composite']:+.2f} | {e['token_overlap']:.2f} | "
   241	            f"{e['intent'][:50]} |"
   242	        )
   243	    (output_dir / "README.md").write_text("\n".join(readme_lines))
   244	
   245	    logger.info(
   246	        f"\n{'='*60}\n"
   247	        f"Extracted: {len(manifest['strong'])} strong + {len(manifest['reverse'])} reverse tasks\n"
   248	        f"Total disk: {total_bytes / 1e6:.1f} MB\n"
   249	        f"Output: {output_dir}\n"
   250	        f"{'='*60}"
   251	    )
   252	
   253	
   254	if __name__ == "__main__":
   255	    main()

exec
/bin/bash -lc "python3 - <<'PY'
import json
from pathlib import Path
for p in [Path('results/mechanistic/archive_subset_b1_cls/manifest.json'), Path('results/mechanistic/stage3_cellhprompt_cls_fwd_ptext_myriad/run_manifest.json'), Path('results/mechanistic/stage3_cellhprompt_cls_fwd_ptext_myriad/env_snapshot.json')]:
    print('---', p)
    if p.exists():
        d=json.load(open(p))
        def keys(obj, indent=0, depth=2):
            if depth<0: return
            if isinstance(obj, dict):
                for k,v in obj.items():
                    print(' '*indent+str(k)+': '+type(v).__name__)
                    keys(v, indent+2, depth-1)
            elif isinstance(obj, list):
                print(' '*indent+f'len={len(obj)}')
                if obj: keys(obj[0], indent+2, depth-1)
        keys(d, depth=2)
    else:
        print('missing')
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
--- results/mechanistic/archive_subset_b1_cls/manifest.json
config: dict
  candidates_jsonl: str
  archived_run_dir: str
  output_dir: str
  site: str
  steps: list
    len=2
  strong_min_composite: float
  strong_max_overlap: float
  reverse_max_composite: float
site: str
steps: list
  len=2
strong: list
  len=24
    task_id: int
    intent: str
    composite: float
    token_overlap: float
    src_neg: int
    src_aff: int
    tgt_neg: int
    tgt_aff: int
    source_text: str
    target_text: str
    steps_extracted: list
reverse: list
  len=15
    task_id: int
    intent: str
    composite: float
    token_overlap: float
    src_neg: int
    src_aff: int
    tgt_neg: int
    tgt_aff: int
    source_text: str
    target_text: str
    steps_extracted: list
skipped: list
  len=0
--- results/mechanistic/stage3_cellhprompt_cls_fwd_ptext_myriad/run_manifest.json
stage: str
direction: str
site: str
patch_config: dict
  source_mode: str
  target_mode: str
  step_idx: int
  max_new_tokens: int
  n_layers_swept: int
  hook_position: str
  first_forward_only: bool
  min_free_vram_gb: float
model: dict
  path: str
  n_layers: int
input_dataset: dict
  archived_run_dir: str
  n_tasks_requested: int
  n_tasks_completed: int
  task_ids: list
    len=24
outcomes_per_task: list
  len=24
    task_id: int
    step_idx: int
    best_layer_overlap_src: int
    best_overlap_src: float
    L11_overlap_src: float
    L17_overlap_src: float
aggregate: dict
  best_layer_overlap_src_mean: int
  best_overlap_src_mean: float
  best_layer_ld_src_mean: int
  L11_overlap_src_mean: float
  L17_overlap_src_mean: float
env_snapshot_ref: str
results_files: dict
  per_task_jsonl: str
  curves_plot: str
  summary_md: str
--- results/mechanistic/stage3_cellhprompt_cls_fwd_ptext_myriad/env_snapshot.json
captured_at: str
host: str
platform: str
python_version: str
torch: dict
  version: str
  cuda: str
  device_count: int
  compute_caps: list
    len=1
  device_names: list
    len=1
libraries: dict
  transformers: str
  qwen_vl_utils: str
  huggingface_hub: str
  numpy: str
  scikit_learn: NoneType
models: dict
  Qwen/Qwen3-VL-4B-Instruct: str
evaluator_code: dict
  combined_sha256: str
  per_file_sha256: dict
    p79/experiment/analysis.py: str
    p79/experiment/environment.py: str
    p79/experiment/metrics.py: str
  files: list
    len=3
git: dict
  commit: str
  branch: str
  dirty: bool
  status: NoneType
nvidia_smi: list
  len=1
extra: dict
  stage: str
  site: str
  reverse: bool
  tier: str
  random_inject: bool
  random_seed: int
  n_tasks_requested: int
  step: int
  max_new_tokens: int
  source_mode: str
  target_mode: str
errors: list
  len=1

exec
/bin/bash -lc "nl -ba scripts/queues/qsub_stage3_cellhprompt_cls.sh | sed -n '1,140p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/bin/bash -l
     2	# Cell H-prompt-cls: cls forward × strong × source=P-SoM × target=P-text × N=24.
     3	# Exp 5: prompt-only patching (text format held fixed at flat [SOM_MARKS]).
     4	# Tests causal axis-2 mechanism — if mid-late layer L20-L25 patching disrupts
     5	# target continuation, the L23 axis-2 cosine peak is causally used by model.
     6	# Companion to existing Stage 3 SoM→{DOM,P-text,P-prompt} cells which patch
     7	# image-feature axis. This cell isolates prompt-family axis alone.
     8	
     9	#$ -l h_rt=24:0:0
    10	#$ -l mem=64G
    11	#$ -l gpu=1
    12	#$ -wd /home/ucab352/Scratch/p79
    13	#$ -N cellhprm_cls
    14	#$ -o /home/ucab352/Scratch/p79/logs/qsub_cellhprm_cls.$JOB_ID.out
    15	#$ -e /home/ucab352/Scratch/p79/logs/qsub_cellhprm_cls.$JOB_ID.err
    16	#$ -j n
    17	
    18	mkdir -p /home/ucab352/Scratch/p79/logs
    19	
    20	set -euo pipefail
    21	REPO_DIR="/home/ucab352/Scratch/p79"
    22	cd "$REPO_DIR"
    23	
    24	echo "[$(date '+%H:%M:%S')] Job $JOB_ID start on $(hostname)"
    25	nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
    26	
    27	module unload gcc-libs python python3 2>/dev/null || true
    28	module load pytorch/2.1.0/gpu
    29	
    30	export PYTHONUSERBASE="$HOME/Scratch/python_user"
    31	export PYTHONPATH="$PYTHONUSERBASE/lib/python3.9/site-packages:${PYTHONPATH:-}"
    32	export HF_HUB_OFFLINE=1
    33	export TRANSFORMERS_OFFLINE=1
    34	export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    35	
    36	echo "[$(date '+%H:%M:%S')] Repo HEAD: $(git rev-parse --short HEAD)"
    37	
    38	if [ ! -f "$REPO_DIR/results/mechanistic/archive_subset_b1_cls/manifest.json" ]; then
    39	  echo "FATAL: archive_subset_b1_cls/manifest.json missing"
    40	  exit 1
    41	fi
    42	
    43	HF_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
    44	HF_SNAPSHOT_DIR="$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/$HF_REVISION"
    45	if [ ! -f "$HF_SNAPSHOT_DIR/config.json" ]; then
    46	  echo "FATAL: HF model snapshot missing at $HF_SNAPSHOT_DIR"
    47	  exit 1
    48	fi
    49	
    50	OUT_DIR="$REPO_DIR/results/mechanistic/stage3_cellhprompt_cls_fwd_ptext_myriad"
    51	mkdir -p "$OUT_DIR"
    52	
    53	echo "[$(date '+%H:%M:%S')] Cell H-prompt-cls: cls fwd × strong × P-SoM → P-text × N=24"
    54	echo "  Exp 5 axis-2 isolation: text format fixed (flat), prompt swap (SoM-prompt → DOM-prompt)"
    55	
    56	python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    57	    --site classifieds \
    58	    --n-tasks 24 \
    59	    --step 2 \
    60	    --max-new-tokens 50 \
    61	    --source-mode phantom_som \
    62	    --target-mode phantom_text \
    63	    --output-dir "$OUT_DIR" \
    64	    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_cls"
    65	
    66	echo "[$(date '+%H:%M:%S')] Cell H-prompt-cls DONE"
    67	ls -la "$OUT_DIR/"
    68	nvidia-smi --query-gpu=memory.used,memory.free --format=csv

exec
/bin/bash -lc "nl -ba scripts/queues/qsub_stage3_cellhprompt_red.sh | sed -n '1,140p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/bin/bash -l
     2	# Cell H-prompt-red: reddit forward × strong × source=P-SoM × target=P-text × N=24.
     3	# Exp 5 cross-site replication of H-prompt-cls. Isolates axis-2 prompt-family
     4	# mechanism on reddit (text format held fixed at flat [SOM_MARKS]).
     5	
     6	#$ -l h_rt=24:0:0
     7	#$ -l mem=64G
     8	#$ -l gpu=1
     9	#$ -wd /home/ucab352/Scratch/p79
    10	#$ -N cellhprm_red
    11	#$ -o /home/ucab352/Scratch/p79/logs/qsub_cellhprm_red.$JOB_ID.out
    12	#$ -e /home/ucab352/Scratch/p79/logs/qsub_cellhprm_red.$JOB_ID.err
    13	#$ -j n
    14	
    15	mkdir -p /home/ucab352/Scratch/p79/logs
    16	
    17	set -euo pipefail
    18	REPO_DIR="/home/ucab352/Scratch/p79"
    19	cd "$REPO_DIR"
    20	
    21	echo "[$(date '+%H:%M:%S')] Job $JOB_ID start on $(hostname)"
    22	nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
    23	
    24	module unload gcc-libs python python3 2>/dev/null || true
    25	module load pytorch/2.1.0/gpu
    26	
    27	export PYTHONUSERBASE="$HOME/Scratch/python_user"
    28	export PYTHONPATH="$PYTHONUSERBASE/lib/python3.9/site-packages:${PYTHONPATH:-}"
    29	export HF_HUB_OFFLINE=1
    30	export TRANSFORMERS_OFFLINE=1
    31	export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    32	
    33	echo "[$(date '+%H:%M:%S')] Repo HEAD: $(git rev-parse --short HEAD)"
    34	
    35	if [ ! -f "$REPO_DIR/results/mechanistic/archive_subset_b1_reddit/manifest.json" ]; then
    36	  echo "FATAL: archive_subset_b1_reddit/manifest.json missing"
    37	  exit 1
    38	fi
    39	
    40	HF_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
    41	HF_SNAPSHOT_DIR="$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/$HF_REVISION"
    42	if [ ! -f "$HF_SNAPSHOT_DIR/config.json" ]; then
    43	  echo "FATAL: HF model snapshot missing at $HF_SNAPSHOT_DIR"
    44	  exit 1
    45	fi
    46	
    47	OUT_DIR="$REPO_DIR/results/mechanistic/stage3_cellhprompt_red_fwd_ptext_myriad"
    48	mkdir -p "$OUT_DIR"
    49	
    50	echo "[$(date '+%H:%M:%S')] Cell H-prompt-red: reddit fwd × strong × P-SoM → P-text × N=24"
    51	echo "  Exp 5 axis-2 isolation: text format fixed (flat), prompt swap (SoM-prompt → DOM-prompt)"
    52	
    53	python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    54	    --site reddit \
    55	    --n-tasks 24 \
    56	    --step 2 \
    57	    --max-new-tokens 50 \
    58	    --source-mode phantom_som \
    59	    --target-mode phantom_text \
    60	    --output-dir "$OUT_DIR" \
    61	    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_reddit"
    62	
    63	echo "[$(date '+%H:%M:%S')] Cell H-prompt-red DONE"
    64	ls -la "$OUT_DIR/"
    65	nvidia-smi --query-gpu=memory.used,memory.free --format=csv

exec
/bin/bash -lc "python3 - <<'PY'
import json
from pathlib import Path
for p in ['results/mechanistic/stage3_cellhprompt_cls_fwd_ptext_myriad/patching_continuation_results.json','results/mechanistic/stage3_cellhprompt_red_fwd_ptext_myriad/patching_continuation_results.json']:
    d=json.load(open(p)); print(p, d['config'])
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
results/mechanistic/stage3_cellhprompt_cls_fwd_ptext_myriad/patching_continuation_results.json {'site': 'classifieds', 'n_tasks': 24, 'step': 2, 'max_new_tokens': 50, 'source_mode': 'phantom_som', 'target_mode': 'phantom_text', 'source_mode_raw': 'phantom_som', 'target_mode_raw': 'phantom_text', 'reverse': False, 'tier': 'strong', 'random_inject': False, 'random_seed': 42, 'archived_run_dir': '/home/ucab352/Scratch/p79/results/mechanistic/archive_subset_b1_cls', 'model_path': 'Qwen/Qwen3-VL-4B-Instruct', 'n_layers': 36}
results/mechanistic/stage3_cellhprompt_red_fwd_ptext_myriad/patching_continuation_results.json {'site': 'reddit', 'n_tasks': 24, 'step': 2, 'max_new_tokens': 50, 'source_mode': 'phantom_som', 'target_mode': 'phantom_text', 'source_mode_raw': 'phantom_som', 'target_mode_raw': 'phantom_text', 'reverse': False, 'tier': 'strong', 'random_inject': False, 'random_seed': 42, 'archived_run_dir': '/home/ucab352/Scratch/p79/results/mechanistic/archive_subset_b1_reddit', 'model_path': 'Qwen/Qwen3-VL-4B-Instruct', 'n_layers': 36}

exec
/bin/bash -lc 'rg -n "run_stage4_method44_v2_sweep|--limit|--layers|--alphas" scripts/queues scripts/mechanistic docs/checkpoints/mechanism/plan.md docs/checkpoints/paper_drafts/section5_mechanism.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
scripts/mechanistic/run_stage4_method44_v2_sweep.py:94:    p.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
scripts/mechanistic/run_stage4_method44_v2_sweep.py:95:    p.add_argument("--alphas", default=",".join(map(str, DEFAULT_ALPHAS)))
scripts/mechanistic/run_stage4_method44_v2_sweep.py:97:    p.add_argument("--limit", type=int, default=2, help="N tasks (smoke=2, full=24)")
scripts/mechanistic/run_stage4_method44_steering.py:101:    p.add_argument("--alphas", default="0.5,1.0,2.0,5.0")
scripts/mechanistic/run_stage4_method44_steering.py:105:    p.add_argument("--limit", type=int, default=None, help="Smoke-test: limit to first N tasks")
docs/checkpoints/mechanism/plan.md:101:`scripts/mechanistic/run_stage4_method44_v2_sweep.py`. Layer × α sweep:

exec
/bin/bash -lc "if [ -f results/mechanistic/stage4_multimode_b1_cls/method44_v2_sweep.json ]; then python3 - <<'PY'
import json
p='results/mechanistic/stage4_multimode_b1_cls/method44_v2_sweep.json'
d=json.load(open(p)); print(d['config']); print('n_results', len(d['results'])); print('first keys', d['results'][0].keys() if d['results'] else None)
PY
else echo missing; fi" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
{'layers': [11, 17, 23, 29, 33, 34], 'alphas': [1.0, 2.0, 5.0, 10.0, 20.0], 'tier': 'strong', 'max_new_tokens': 15, 'direction_norms': {'11': 4.4820756912231445, '17': 6.101632118225098, '23': 17.12262535095215, '29': 38.21261215209961, '33': 59.188812255859375, '34': 68.24252319335938}}
n_results 45
first keys dict_keys(['task_id', 'step', 'dom_text', 'psom_text', 'per_layer'])

