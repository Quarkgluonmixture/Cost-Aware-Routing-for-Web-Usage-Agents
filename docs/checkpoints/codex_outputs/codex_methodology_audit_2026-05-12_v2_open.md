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
session id: 019e1e25-bee7-7780-bedd-18e5e90fc017
--------
user
# Codex methodology audit — paper-grade protocol soundness check (v2, lean)

You are a senior mechanistic-interpretability researcher (level: David Bau, Neel Nanda, Asma Ghandeharioun, Atticus Geiger) peer-reviewing the methodology behind this paper-1 mechanism work. You have personally implemented activation patching, mean-difference steering, logit lens, and PCA-based representation probing in multiple codebases. You are unimpressed by mechinterp papers whose code does not match their prose.

**Your job today**: audit the **methodology**, not the claims. Read the code that implements the mechanism analyses, cross-reference against the prose and plan, and find protocol bugs, hidden assumptions, statistical errors, or implementation-vs-prose mismatches.

You will set your own attack vectors based on what you actually see in the code. Do not let me preselect them for you — the value of this audit is precisely that you find issues I did not think to list.

## 🚫 Independence requirement

Do NOT read prior audit output or shared-AI artifacts:
- `docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md`
- `docs/checkpoints/codex_outputs/codex_methodology_audit_*.md` (prior runs)
- `.claude/skills/*/SKILL.md`
- `.claude/skills/*/prompt_template.md`
- `docs/checkpoints/process/*_skill_replica.md`

You are doing an independent methodology read from a peer-reviewer perspective.

## What this paper is doing (one paragraph, just so you have context)

The paper characterizes a "phantom routing space" in multimodal web agents (Qwen3-VL on VisualWebArena classifieds + reddit). Mechanism section claims: (a) observation modes are linearly readable in residual stream (Method 4.2 PCA cosine gap, AUROC 1.0); (b) Phantom-SoM is a mid-layer text-axis sibling of DOM/P-text, not image-axis sibling of full SoM; (c) SoM→no-image activation patching displaces target continuations at mid layers; (d) a three-axis hierarchy (image / text-format / prompt-family) emerges across layers L0-L36. Evidence sits in Stage 2/3 patching cells, Stage 4 multimode hidden-state extraction, Exp 1 layer profiles, Exp 3 logit lens, Exp 5 axis-2 prompt-only patching.

## Read scope

Code (this is the primary subject — methodology lives here):

- `scripts/mechanistic/run_stage2b_continuation_pilot.py` (Stage 2/3 + Exp 5 patching)
- `scripts/analysis/stage4_pca_cosine_gap.py` (Method 4.2)
- `scripts/analysis/stage4_axis2_layer_profile.py` (Exp 1)
- `scripts/analysis/stage4_logit_lens_axis2.py` (Exp 3)
- `scripts/analysis/stage4_w6_l04_tokenization.py` (W6 tokenization)
- `scripts/analysis/hero_claim_bootstrap.py` (W1 bootstrap)
- `scripts/analysis/stage2_layer_significance.py` (if exists; Holm-Bonferroni)
- `p79/mechanistic/` (any patching primitives, hooks, mode-mean computation)

Documentation (cross-reference against code):

- `docs/checkpoints/mechanism/plan.md` (the protocol description)
- `docs/checkpoints/paper_drafts/section5_mechanism.md` (prose description of methods)
- Recent evidence files in `docs/checkpoints/mechanism/results/` (claimed numbers — spot-check 2-3 against code)

## Output format

Write one markdown report with these sections. **No more, no less.**

### 1. Verdict (one sentence)
Methodology sound enough for paper-1 submission, or not, and the one most-important reason.

### 2. 🟢 What's right (don't break under refactor)
2-4 protocol pieces that survive expert scrutiny. Be specific. Cite file:line.

### 3. 🔴 Methodology bugs (caught before submission)
For each bug:
- **What** — protocol issue at file:line
- **Why it matters** — which claim depends on this; how it biases results
- **Fix** — concrete code or analysis change
- **Severity** — high (invalidates main claim) / medium (weakens) / low (cosmetic)

### 4. 🟡 Methodology risks (unverified assumptions)
Things that look reasonable but depend on assumptions you cannot verify from code alone. Each with a verification suggestion.

### 5. 📝 Reproducibility audit
Are the analyses runnable by a third party? Seeds set across right RNGs? Source archives uniquely identified? Provenance traceable from results file → command → data?

### 6. One single highest-impact fix
The single methodology change with the largest effect on which claims survive. Concrete file/line.

## Voice

- Technical, specific, no hedging
- Cite code file:line for every claim
- 中文为主双语 — explanations in 中文, technical terms / variable names / file paths in English
- If you cannot determine an answer from the code, say so explicitly ("evidence unavailable for X; would need to inspect Y") — do not make up answers

## What this is NOT

- Not a claim-level reviewer attack (prior /codex-stress did that)
- Not a writing audit
- Not a citation checker
- Specifically: **methodology expert reading code, verifying implementation matches prose, statistical procedures are sound**

codex
I’ll do an independent code/prose audit and avoid the excluded prior-output paths. I’ll first map the relevant files and line numbers, then spot-check result provenance against the implementation before writing the report.
exec
/bin/bash -lc 'pwd && rg --files scripts p79 docs/checkpoints/mechanism docs/checkpoints/paper_drafts | sort' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
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
/bin/bash -lc 'git status --short' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
 M docs/analysis/cross_sites/sr_fp_per_mode.json
 M docs/analysis/cross_sites/sr_fp_per_mode.md
 D docs/reference/MYRIAD_SMOKE_REPORT.md
 D docs/reference/PAPER_STRATEGY_OPEN_QUESTIONS.md
 D docs/reference/RUNPOD_ONBOARDING.md
?? docs/checkpoints/codex_outputs/codex_methodology_audit_2026-05-12_v1_directive.md
?? docs/checkpoints/codex_outputs/codex_methodology_audit_2026-05-12_v2_open.md
?? docs/checkpoints/codex_prompts/codex_methodology_audit_2026-05-12.md
?? scripts/analysis/stage2_transfer_correlation.py

codex
The repository has existing untracked prior audit files under the excluded pattern, so I’m leaving them untouched. I’m going straight to the implementation files, docs, and result files the prompt explicitly allows.
exec
/bin/bash -lc 'nl -ba scripts/mechanistic/run_stage2b_continuation_pilot.py' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
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
/bin/bash -lc 'nl -ba p79/mechanistic/activation_patching.py' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
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

exec
/bin/bash -lc 'nl -ba p79/mechanistic/extract_hidden_states.py' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
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
/bin/bash -lc 'nl -ba p79/mechanistic/linear_probe.py' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	"""Per-layer logistic regression probe with k-fold CV.
     2	
     3	Tests whether mirage label is linearly decodable from hidden states at each
     4	transformer layer. AUROC curve over layers reveals where mirage info is encoded.
     5	"""
     6	
     7	from __future__ import annotations
     8	
     9	import logging
    10	from typing import Optional
    11	
    12	import numpy as np
    13	from sklearn.decomposition import PCA
    14	from sklearn.linear_model import LogisticRegression
    15	from sklearn.model_selection import StratifiedKFold
    16	from sklearn.metrics import roc_auc_score
    17	from sklearn.preprocessing import StandardScaler
    18	
    19	logger = logging.getLogger(__name__)
    20	
    21	
    22	def linear_probe_per_layer(
    23	    hidden_states: np.ndarray,
    24	    labels: np.ndarray,
    25	    n_folds: int = 5,
    26	    seed: int = 42,
    27	    C: float = 0.01,
    28	    max_iter: int = 1000,
    29	    pca_dim: Optional[int] = 50,
    30	) -> dict:
    31	    """Train per-layer LR with stratified k-fold CV; return AUROC curve.
    32	
    33	    Defaults `C=0.01` + `pca_dim=50` are linear-probe-literature standards for
    34	    `n_samples << hidden_dim` regime to prevent trivial overfitting saturation.
    35	    For high-N regime (n_samples >> hidden_dim) consider raising C and disabling
    36	    PCA via `pca_dim=None`.
    37	
    38	    Args:
    39	        hidden_states: (n_samples, n_layers, hidden_dim)
    40	        labels: (n_samples,) binary 0/1
    41	        n_folds: CV folds
    42	        seed: RNG seed for fold split
    43	        C: LR L2 regularization (smaller = more regularization)
    44	        max_iter: LR solver iterations
    45	        pca_dim: if set, fit PCA per fold with this many components (avoids test leak);
    46	            None = no PCA. Auto-clamped to min(pca_dim, n_train - 1).
    47	
    48	    Returns:
    49	        {
    50	            "auroc_mean": list[float] length n_layers,
    51	            "auroc_std": list[float] length n_layers,
    52	            "auroc_per_fold": list[list[float]] (n_layers, n_folds),
    53	            "best_layer": int,
    54	            "best_auroc": float,
    55	            "n_samples": int,
    56	            "n_pos": int,
    57	            "n_layers": int,
    58	            "hidden_dim": int,
    59	            "pca_dim": int | None,
    60	            "C": float,
    61	        }
    62	    """
    63	    n_samples, n_layers, hidden_dim = hidden_states.shape
    64	    if labels.shape[0] != n_samples:
    65	        raise ValueError(f"labels.shape[0] {labels.shape[0]} != n_samples {n_samples}")
    66	    n_pos = int(labels.sum())
    67	    if n_pos == 0 or n_pos == n_samples:
    68	        raise ValueError(f"degenerate labels: {n_pos}/{n_samples} positive")
    69	
    70	    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    71	    auroc_per_fold = np.zeros((n_layers, n_folds), dtype=np.float64)
    72	
    73	    for layer_idx in range(n_layers):
    74	        X = hidden_states[:, layer_idx, :]  # (n_samples, hidden_dim)
    75	        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
    76	            scaler = StandardScaler()
    77	            X_train = scaler.fit_transform(X[train_idx])
    78	            X_test = scaler.transform(X[test_idx])
    79	
    80	            if pca_dim is not None:
    81	                effective_pca_dim = min(pca_dim, X_train.shape[0] - 1, X_train.shape[1])
    82	                pca = PCA(n_components=effective_pca_dim, random_state=seed)
    83	                X_train = pca.fit_transform(X_train)
    84	                X_test = pca.transform(X_test)
    85	
    86	            clf = LogisticRegression(
    87	                C=C, max_iter=max_iter, solver="lbfgs", random_state=seed,
    88	            )
    89	            clf.fit(X_train, labels[train_idx])
    90	            y_score = clf.predict_proba(X_test)[:, 1]
    91	            auroc_per_fold[layer_idx, fold_idx] = roc_auc_score(labels[test_idx], y_score)
    92	        if (layer_idx + 1) % 5 == 0:
    93	            logger.info(
    94	                f"Layer {layer_idx + 1}/{n_layers}: "
    95	                f"AUROC {auroc_per_fold[layer_idx].mean():.4f} ± {auroc_per_fold[layer_idx].std():.4f}"
    96	            )
    97	
    98	    auroc_mean = auroc_per_fold.mean(axis=1)
    99	    auroc_std = auroc_per_fold.std(axis=1)
   100	    best_layer = int(auroc_mean.argmax())
   101	
   102	    return {
   103	        "auroc_mean": auroc_mean.tolist(),
   104	        "auroc_std": auroc_std.tolist(),
   105	        "auroc_per_fold": auroc_per_fold.tolist(),
   106	        "best_layer": best_layer,
   107	        "best_auroc": float(auroc_mean[best_layer]),
   108	        "best_auroc_std": float(auroc_std[best_layer]),
   109	        "n_samples": n_samples,
   110	        "n_pos": n_pos,
   111	        "n_layers": n_layers,
   112	        "hidden_dim": hidden_dim,
   113	        "pca_dim": pca_dim,
   114	        "C": C,
   115	    }
   116	
   117	
   118	def plot_auroc_curve(
   119	    probe_results: dict,
   120	    save_path: str,
   121	    title: Optional[str] = None,
   122	) -> None:
   123	    """Plot per-layer AUROC ± std as a curve."""
   124	    import matplotlib
   125	    matplotlib.use("Agg")
   126	    import matplotlib.pyplot as plt
   127	
   128	    auroc_mean = np.array(probe_results["auroc_mean"])
   129	    auroc_std = np.array(probe_results["auroc_std"])
   130	    best_layer = probe_results["best_layer"]
   131	    n_samples = probe_results["n_samples"]
   132	    n_pos = probe_results["n_pos"]
   133	    layers = np.arange(len(auroc_mean))
   134	
   135	    fig, ax = plt.subplots(figsize=(10, 5))
   136	    ax.plot(layers, auroc_mean, marker="o", lw=1.5, label="AUROC mean")
   137	    ax.fill_between(
   138	        layers, auroc_mean - auroc_std, auroc_mean + auroc_std,
   139	        alpha=0.25, label="±1 std (5-fold CV)",
   140	    )
   141	    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance (0.5)")
   142	    ax.axvline(
   143	        best_layer, color="red", ls=":", lw=0.8,
   144	        label=f"best layer {best_layer} ({auroc_mean[best_layer]:.3f})",
   145	    )
   146	    ax.set_xlabel("Layer index (0 = embedding output, ≥1 = post-transformer-block)")
   147	    ax.set_ylabel("AUROC (5-fold CV)")
   148	    ax.set_ylim(0.45, 1.02)
   149	    ax.grid(alpha=0.3)
   150	    ax.legend(loc="lower right", fontsize=9)
   151	    if title is None:
   152	        title = f"Per-layer linear probe AUROC (N={n_samples}, n_pos={n_pos})"
   153	    ax.set_title(title)
   154	    fig.tight_layout()
   155	    fig.savefig(save_path, dpi=150)
   156	    plt.close(fig)
   157	    logger.info(f"Saved AUROC curve → {save_path}")

exec
/bin/bash -lc 'nl -ba scripts/analysis/stage4_pca_cosine_gap.py' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
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
/bin/bash -lc 'nl -ba scripts/analysis/stage4_axis2_layer_profile.py' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
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
/bin/bash -lc 'nl -ba scripts/analysis/stage4_logit_lens_axis2.py' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
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
/bin/bash -lc 'nl -ba scripts/analysis/stage4_w6_l04_tokenization.py' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""W6 feature attribution: why 2/6 marks-like variants peak at L04 on reddit.
     3	
     4	/stress W6 attack: H1 hypothesis predicts marks-like variants trigger shortcut
     5	(peak L17). But appagent_id + plain_numbered peak at L04 like AXTree-DOM. The
     6	H1 verdict was "MIXED, needs deeper analysis". W6 asks: what's the *feature*
     7	that splits the 6 marks-like variants into L04 vs L17 subgroups?
     8	
     9	Hypothesis to test: L17-peak variants share special-character-leading tokens
    10	(`[`, `<`, `@`) that appear in HTML/markup pretraining; L04-peak variants
    11	(`id_N:`, `N.`) are plain prose patterns common in natural text. Tokenization
    12	captures this — L17-peak first tokens should be non-alphanumeric, L04-peak
    13	first tokens should be alphanumeric.
    14	
    15	Outputs:
    16	- docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md
    17	- Per-variant token breakdown, first-token char class, mean tokens/element
    18	"""
    19	from __future__ import annotations
    20	
    21	import argparse
    22	from pathlib import Path
    23	
    24	ROOT = Path(__file__).resolve().parents[2]
    25	DEFAULT_OUT = ROOT / "docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md"
    26	
    27	# 6 marks-like variants from format_variation_h1_test_reddit.md
    28	# Each entry: (variant_name, peak_layer, example_marker_line)
    29	# Marker line = formatted single-element example with N=1, role=button, label=Submit
    30	VARIANTS = [
    31	    # L04-peak (no shortcut, DOM-like)
    32	    ("appagent_id",    "L04", "id_1: Submit"),
    33	    ("plain_numbered", "L04", "1. Submit"),
    34	    # L17-peak (shortcut triggered, marks-like)
    35	    ("som_standard",   "L17", "[1] button 'Submit'"),
    36	    ("browser_use_at", "L17", "@1 Submit"),
    37	    ("tarsier_typed",  "L17", "[B1:button:Submit]"),
    38	    ("xml_tagged",     "L17", "<el_1 role='button'>Submit</el_1>"),
    39	]
    40	
    41	# Also include controls + AXTree for reference
    42	EXTRAS = [
    43	    ("hash_id_control", "L04", "#a3f7 Submit"),
    44	    ("plain_sentence",  "L17", "Submit"),  # no list/no marker — single label in prose
    45	    ("dom",             "L04", "button: Submit (AXTree)"),
    46	    ("som",             "L17", "[1] button 'Submit' (+ image marks)"),
    47	]
    48	
    49	
    50	def char_class(c: str) -> str:
    51	    if c.isalnum():
    52	        return "alphanumeric"
    53	    if c == "[" or c == "<" or c == "@" or c == "#":
    54	        return "markup-sigil"
    55	    if c == "." or c == "," or c == ":":
    56	        return "punctuation"
    57	    if c == "'" or c == '"':
    58	        return "quote"
    59	    if c.isspace():
    60	        return "whitespace"
    61	    return "other"
    62	
    63	
    64	def main():
    65	    p = argparse.ArgumentParser()
    66	    p.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    67	    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    68	    args = p.parse_args()
    69	
    70	    print(f"Loading tokenizer {args.model_id}...", flush=True)
    71	    from transformers import AutoTokenizer
    72	    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    73	
    74	    rows = []
    75	    for name, peak, example in VARIANTS + EXTRAS:
    76	        # Tokenize the marker portion (= example string)
    77	        ids = tok.encode(example, add_special_tokens=False)
    78	        toks = tok.convert_ids_to_tokens(ids)
    79	        first_tok = toks[0] if toks else ""
    80	        # First char of first decoded token (strip Qwen BPE space marker Ġ if present)
    81	        first_char = first_tok.lstrip("Ġ▁ ").lstrip()[:1] if first_tok else ""
    82	        first_class = char_class(first_char) if first_char else "empty"
    83	        n_tokens = len(ids)
    84	        # Marker-content density: how many tokens encode the bare marker (before label)
    85	        # Heuristic: stop at first space-prefixed token after position 0 → marker fingerprint
    86	        marker_toks = []
    87	        for i, t in enumerate(toks):
    88	            if i > 0 and (t.startswith("Ġ") or t.startswith("▁") or t.startswith(" ")):
    89	                break
    90	            marker_toks.append(t)
    91	        marker_n = len(marker_toks)
    92	        rows.append({
    93	            "name": name,
    94	            "peak": peak,
    95	            "example": example,
    96	            "n_tokens": n_tokens,
    97	            "first_tok": first_tok,
    98	            "first_char": first_char,
    99	            "first_class": first_class,
   100	            "marker_n": marker_n,
   101	            "marker_toks": marker_toks,
   102	            "all_toks": toks,
   103	        })
   104	
   105	    # Verdict: count first-class within L04-peak vs L17-peak subsets of marks-like (skip extras for verdict)
   106	    marks_like = [r for r in rows if r["name"] in {v[0] for v in VARIANTS}]
   107	    l04 = [r for r in marks_like if r["peak"] == "L04"]
   108	    l17 = [r for r in marks_like if r["peak"] == "L17"]
   109	
   110	    def first_class_counts(rs):
   111	        out = {}
   112	        for r in rs:
   113	            out[r["first_class"]] = out.get(r["first_class"], 0) + 1
   114	        return out
   115	
   116	    l04_classes = first_class_counts(l04)
   117	    l17_classes = first_class_counts(l17)
   118	
   119	    md = []
   120	    md.append(f"# W6 feature attribution — H1 reddit 2/6 marks-like L04 peak\n")
   121	    md.append(f"**Setup**: Qwen3-VL-4B tokenizer ({args.model_id}). Each marks-like "
   122	              f"format variant tokenized on a canonical single-element example "
   123	              f"(N=1, role=button, label=Submit). First-token character class + "
   124	              f"marker-fingerprint token count compared between L04-peak and L17-peak "
   125	              f"subgroups.\n")
   126	
   127	    md.append("## Per-variant tokenization\n")
   128	    md.append("| Variant | Peak | Example | n_tok | First token | First char class | Marker fp |")
   129	    md.append("|---|---|---|---:|---|---|---:|")
   130	    for r in rows:
   131	        marker_str = "·".join(r["marker_toks"])
   132	        # Escape pipes for markdown
   133	        ex_safe = r["example"].replace("|", "\\|")
   134	        ft_safe = r["first_tok"].replace("|", "\\|")
   135	        mk_safe = marker_str.replace("|", "\\|")
   136	        md.append(
   137	            f"| {r['name']} | {r['peak']} | `{ex_safe}` | {r['n_tokens']} | "
   138	            f"`{ft_safe}` | {r['first_class']} | {r['marker_n']} (`{mk_safe}`) |"
   139	        )
   140	    md.append("")
   141	
   142	    md.append("## Subgroup first-char-class distribution (6 marks-like only)\n")
   143	    md.append("| Subgroup | alphanumeric | markup-sigil | punctuation | quote | other |")
   144	    md.append("|---|---:|---:|---:|---:|---:|")
   145	    for label, cs in [("L04-peak (2)", l04_classes), ("L17-peak (4)", l17_classes)]:
   146	        md.append(
   147	            f"| {label} | {cs.get('alphanumeric', 0)} | {cs.get('markup-sigil', 0)} | "
   148	            f"{cs.get('punctuation', 0)} | {cs.get('quote', 0)} | {cs.get('other', 0)} |"
   149	        )
   150	    md.append("")
   151	
   152	    # Hypothesis verdict
   153	    md.append("## Hypothesis verdict\n")
   154	    l04_alpha = sum(1 for r in l04 if r["first_class"] == "alphanumeric")
   155	    l17_sigil = sum(1 for r in l17 if r["first_class"] == "markup-sigil")
   156	    if l04_alpha == 2 and l17_sigil >= 3:
   157	        verdict = (
   158	            "✅ **Hypothesis supported (clean split)**: L04-peak variants both start "
   159	            f"with alphanumeric tokens ({l04_alpha}/2); L17-peak variants start with "
   160	            f"markup-sigil tokens ({l17_sigil}/4)."
   161	        )
   162	    elif l04_alpha == 2:
   163	        verdict = (
   164	            "🔸 **Partial support**: L04-peak both alphanumeric-first (2/2), but L17-peak "
   165	            f"only {l17_sigil}/4 markup-sigil-first. Sigil count alone doesn't fully explain "
   166	            "L17 peak — second-order feature (marker length, role-keyword presence, embedded "
   167	            "quotes) likely contributes."
   168	        )
   169	    else:
   170	        verdict = "❌ **Hypothesis rejected**: first-char-class doesn't cleanly split."
   171	    md.append(verdict + "\n")
   172	
   173	    # Secondary features
   174	    md.append("## Secondary features\n")
   175	    l04_marker_mean = sum(r["marker_n"] for r in l04) / max(len(l04), 1)
   176	    l17_marker_mean = sum(r["marker_n"] for r in l17) / max(len(l17), 1)
   177	    md.append(f"- L04-peak mean marker-fp tokens: {l04_marker_mean:.2f}")
   178	    md.append(f"- L17-peak mean marker-fp tokens: {l17_marker_mean:.2f}")
   179	    md.append(f"- Δ (L17 − L04): {l17_marker_mean - l04_marker_mean:+.2f}\n")
   180	
   181	    # Concrete tokens
   182	    md.append("## Full token sequence per variant (marks-like 6)\n")
   183	    for r in marks_like:
   184	        md.append(f"- **{r['name']}** ({r['peak']}, `{r['example']}`): "
   185	                  f"{r['n_tokens']} tokens: " + " · ".join(f"`{t}`" for t in r["all_toks"]))
   186	    md.append("")
   187	
   188	    md.append("## Interpretation\n")
   189	    md.append(
   190	        "The L17 vs L04 split in H1 reddit corresponds to whether the variant's first "
   191	        "tokens are **markup-sigil tokens** (which co-occur with HTML / web-agent traces "
   192	        "in pretraining and trigger the visual-grounding shortcut at mid layers) versus "
   193	        "**plain alphanumeric tokens** (which are common in prose / dictionary listings "
   194	        "and behave like AXTree-DOM, peaking early at L04 where the image-axis divergence "
   195	        "is freshly observable but not yet routed through the shortcut path).\n"
   196	    )
   197	    md.append(
   198	        "**Paper §5 implication**: H1's binary 'marks-like vs not' prediction is too coarse. "
   199	        "The mechanism trigger is **first-token markup-sigil presence**, not the abstract "
   200	        "concept of 'indexed list'. Variants like `id_N:` and `N.` are nominally indexed "
   201	        "but read as plain language → no shortcut. This refines H1 to **'markup-sigil-leading "
   202	        "indexed list → triggers shortcut'**, which is testable on additional variants.\n"
   203	    )
   204	    md.append(
   205	        "**Falsifier**: If we re-tokenize `[N]` without the bracket (e.g., variant `bare_N` = "
   206	        "`N button 'Submit'` with no brackets) and it still peaks at L17, the hypothesis fails. "
   207	        "Would need a follow-up extract.\n"
   208	    )
   209	
   210	    args.output.parent.mkdir(parents=True, exist_ok=True)
   211	    args.output.write_text("\n".join(md))
   212	    print(f"→ {args.output}")
   213	
   214	
   215	if __name__ == "__main__":
   216	    main()

exec
/bin/bash -lc 'nl -ba scripts/analysis/hero_claim_bootstrap.py' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
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
/bin/bash -lc 'nl -ba scripts/analysis/stage2_layer_significance.py' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
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
/bin/bash -lc 'nl -ba scripts/mechanistic/run_stage4_multimode_extract.py' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
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
/bin/bash -lc 'nl -ba scripts/analysis/stage4_layer_axis_emergence.py' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Stage 4: image-axis peak-layer split — Mirage Effect mechanism signature.
     3	
     4	Reads existing Method 4.2 metrics.json + recomputes per-layer cosine gap from
     5	hidden_states.npz. Identifies 8 image-axis mode pairs (one side has image,
     6	other doesn't) and groups by which text payload format is on the no-image side:
     7	
     8	  AXTree no-image side (DOM, P-prompt) → image-axis peak L04 (fresh early check)
     9	  [SOM_MARKS] no-image side (P-text, P-SoM) → image-axis peak L17-L36 (delayed)
    10	
    11	This peak-layer SHIFT is the Method 4.2 mechanism-level signature of the
    12	Mirage Effect (Asadi et al. 2026): [SOM_MARKS] text primes a marks-parsing
    13	pathway through mid-layer computation, making the image yes/no decision
    14	deferred until late layers — explains why VLMs achieve ~70-80% accuracy
    15	on visual tasks without actually seeing the image.
    16	
    17	Outputs:
    18	  - docs/checkpoints/mechanism/results/layer_axis_emergence.md
    19	  - results/phantom_paper/figures/fig_stage4_image_axis_layer_split.png
    20	"""
    21	from __future__ import annotations
    22	
    23	import json
    24	from itertools import combinations
    25	from pathlib import Path
    26	
    27	import matplotlib.pyplot as plt
    28	import numpy as np
    29	
    30	ROOT = Path(__file__).resolve().parents[2]
    31	NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz"
    32	OUT_MD = ROOT / "docs/checkpoints/mechanism/results/layer_axis_emergence.md"
    33	OUT_FIG = ROOT / "results/phantom_paper/figures/fig_stage4_image_axis_layer_split.png"
    34	
    35	MODES = ["dom", "phantom_text", "phantom_prompt", "phantom_som", "som", "vision"]
    36	DISPLAY = {"dom": "DOM", "phantom_text": "P-text", "phantom_prompt": "P-prompt",
    37	           "phantom_som": "P-SoM", "som": "SoM", "vision": "Vision"}
    38	
    39	# Per-mode: (text_format, has_image)
    40	META = {
    41	    "dom":            ("AXTree",      False),
    42	    "phantom_prompt": ("AXTree",      False),
    43	    "phantom_text":   ("[SOM_MARKS]", False),
    44	    "phantom_som":    ("[SOM_MARKS]", False),
    45	    "som":            ("[SOM_MARKS]", True),
    46	    "vision":         ("(no text)",   True),
    47	}
    48	
    49	
    50	def cosine_gap(a, b):
    51	    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    52	
    53	
    54	def main():
    55	    d = np.load(NPZ, allow_pickle=True)
    56	    H = d["hidden_states"]
    57	    ml = d["mode_labels_str"]
    58	    n_layers = H.shape[1]
    59	
    60	    means = {m: H[ml == m].mean(axis=0) for m in MODES}
    61	
    62	    # All 8 image-axis pairs (one side has image, other doesn't)
    63	    image_axis_pairs = []
    64	    for m1, m2 in combinations(MODES, 2):
    65	        if META[m1][1] != META[m2][1]:  # different image presence
    66	            no_img = m1 if not META[m1][1] else m2
    67	            has_img = m2 if not META[m1][1] else m1
    68	            image_axis_pairs.append((no_img, has_img))
    69	
    70	    # Compute per-layer cosine gap
    71	    pair_curves = {}
    72	    for no_img, has_img in image_axis_pairs:
    73	        curve = np.array([cosine_gap(means[no_img][L], means[has_img][L]) for L in range(n_layers)])
    74	        peak_L = int(np.argmax(curve))
    75	        peak_gap = float(curve[peak_L])
    76	        pair_curves[(no_img, has_img)] = {
    77	            "curve": curve, "peak_L": peak_L, "peak_gap": peak_gap,
    78	            "no_img_text": META[no_img][0],
    79	            "has_img_text": META[has_img][0],
    80	        }
    81	
    82	    write_md(pair_curves, OUT_MD)
    83	    plot(pair_curves, OUT_FIG, n_layers)
    84	    print("Peak layer per image-axis pair:")
    85	    for k, v in sorted(pair_curves.items(), key=lambda x: x[1]["peak_L"]):
    86	        no_img, has_img = k
    87	        print(f"  {DISPLAY[no_img]:>9} ↔ {DISPLAY[has_img]:<8} | no-img text={v['no_img_text']:<12} | peak L{v['peak_L']:02d} = {v['peak_gap']:.4f}")
    88	
    89	
    90	def write_md(pair_curves, out):
    91	    out.parent.mkdir(parents=True, exist_ok=True)
    92	    lines = [
    93	        "# Stage 4: image-axis peak-layer split — Mirage Effect signature",
    94	        "",
    95	        "Eight mode pairs differ in image presence (one side has image, the other doesn't). Peak cosine-gap layer reveals **when** image-axis mechanism emerges:",
    96	        "",
    97	        "| no-image side | image side | no-img text | peak layer | peak cosine gap |",
    98	        "|---|---|---|---|---|",
    99	    ]
   100	    for (no_img, has_img), v in sorted(pair_curves.items(), key=lambda x: x[1]["peak_L"]):
   101	        lines.append(f"| {DISPLAY[no_img]} | {DISPLAY[has_img]} | {v['no_img_text']} | **L{v['peak_L']:02d}** | {v['peak_gap']:.4f} |")
   102	    lines.append("")
   103	
   104	    lines.append("## Grouped by no-image side text format")
   105	    lines.append("")
   106	    groups = {}
   107	    for k, v in pair_curves.items():
   108	        groups.setdefault(v["no_img_text"], []).append((k, v))
   109	    for text_fmt in ["AXTree", "[SOM_MARKS]"]:
   110	        pairs = groups.get(text_fmt, [])
   111	        if not pairs:
   112	            continue
   113	        mean_L = np.mean([v["peak_L"] for _, v in pairs])
   114	        lines.append(f"### no-image text = `{text_fmt}` (mean peak L{mean_L:.0f})")
   115	        lines.append("")
   116	        for (no_img, has_img), v in pairs:
   117	            lines.append(f"- {DISPLAY[no_img]} ↔ {DISPLAY[has_img]}: peak **L{v['peak_L']:02d}** = {v['peak_gap']:.4f}")
   118	        lines.append("")
   119	
   120	    lines.append("## Mechanism interpretation (paper §5 v3 Mirage anchor)")
   121	    lines.append("")
   122	    lines.append("When the no-image side carries `AXTree` text (DOM, P-prompt), the image-axis cosine gap peaks at **L04** — early-layer fresh image-presence detection (vision encoder + cross-modal fusion).")
   123	    lines.append("")
   124	    lines.append("When the no-image side carries `[SOM_MARKS]` text (P-text, P-SoM), the image-axis cosine gap peak shifts to **L17–L36** — image yes/no divergence is deferred to mid/output layers.")
   125	    lines.append("")
   126	    lines.append("**Mechanism story**: `[SOM_MARKS]` text in input primes an indexed-parsing pathway through mid-layer computation. The model processes marks structurally regardless of whether image is provided, producing image-axis divergence only at late integration stages. This is the direct Method 4.2 empirical anchor for the **Mirage Effect** (Asadi et al. 2026, VLM ~70-80% no-image accuracy) and **Cross-modal flow** (Kaduri et al., middle-layer cross-modal flows store image info in query tokens): the marks-primed mid-layer computation runs *as if image were available*, with image grounding contributed only late.")
   127	    lines.append("")
   128	    lines.append("**Paper §5 prose** (suggested):")
   129	    lines.append("")
   130	    lines.append("> *Method 4.2 reveals a peak-layer shift signature for the Mirage Effect: image-axis cosine-gap peak transitions from L04 (when no-image side carries AXTree text) to L17–L36 (when no-image side carries [SOM_MARKS] text). The peak-layer shift quantifies how text-payload format primes mid-layer computation pathways — [SOM_MARKS] format triggers indexed-parsing through mid-layers regardless of image presence, with image-axis divergence deferred to late integration. This identifies [SOM_MARKS] as the mechanism trigger for the Mirage Effect, anchoring Asadi et al. 2026's behavioral finding (~70-80% no-image VLM accuracy) and Kaduri et al.'s middle-layer cross-modal flow hypothesis with layer-resolved empirical evidence.*")
   131	    out.write_text("\n".join(lines) + "\n")
   132	    print(f"summary → {out}")
   133	
   134	
   135	def plot(pair_curves, out, n_layers):
   136	    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
   137	    fig, ax = plt.subplots(figsize=(11, 6))
   138	
   139	    for (no_img, has_img), v in pair_curves.items():
   140	        txt = v["no_img_text"]
   141	        if txt == "AXTree":
   142	            color, linestyle = "#cc4444", "-"  # red solid for AXTree no-image (peaks L04)
   143	        else:
   144	            color, linestyle = "#4477aa", "--"  # blue dashed for [SOM_MARKS] no-image (peaks L17+)
   145	        label = f"{DISPLAY[no_img]} ↔ {DISPLAY[has_img]}  (no-img text: {txt})"
   146	        ax.plot(range(n_layers), v["curve"], color=color, linestyle=linestyle, linewidth=1.5, label=label, alpha=0.85)
   147	        ax.scatter([v["peak_L"]], [v["peak_gap"]], color=color, s=60, marker="*", zorder=5, edgecolor="black", linewidth=0.5)
   148	
   149	    ax.axvline(4, color="#cc4444", linestyle=":", alpha=0.4, linewidth=1)
   150	    ax.axvline(17, color="#4477aa", linestyle=":", alpha=0.4, linewidth=1)
   151	    ax.text(4, 0.07, " L4 = AXTree-text\n image-axis peak\n (fresh check)", color="#cc4444", fontsize=8.5, va="top")
   152	    ax.text(17, 0.045, " L17 = [SOM_MARKS]-text\n image-axis peak shifts\n (marks-primed delay)", color="#4477aa", fontsize=8.5, va="top")
   153	
   154	    ax.set_xlabel("Layer index (Qwen3-VL-4B B1 cls)")
   155	    ax.set_ylabel("Cosine gap between mode means")
   156	    ax.set_title("Image-axis peak-layer shift — Mirage Effect signature\n(Method 4.2, 24 cls strong-tier tasks × 2 steps)",
   157	                  fontsize=11, fontweight="bold")
   158	    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
   159	    ax.grid(alpha=0.3)
   160	    ax.set_xlim(-0.5, n_layers - 0.5)
   161	    ax.set_ylim(0, max(v["peak_gap"] for v in pair_curves.values()) * 1.15)
   162	
   163	    fig.tight_layout()
   164	    out.parent.mkdir(parents=True, exist_ok=True)
   165	    fig.savefig(out, bbox_inches="tight")
   166	    print(f"figure → {out}")
   167	
   168	
   169	if __name__ == "__main__":
   170	    main()

codex
I’ve got the core implementation now. Two methodological pressure points are already visible: the PCA/AUROC code appears to classify on a direction fit using all examples, and Stage 4 extraction uses a different SoM-mark formatter than Stage 2B. I’m checking the prose and result files next so the report distinguishes code bugs from presentation drift.
exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/plan.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
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
/bin/bash -lc 'nl -ba docs/checkpoints/paper_drafts/section5_mechanism.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
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
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/axis2_layer_profile.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Exp 1 — Axis-2 (prompt-family) layer profile
     2	
     3	**Question**: Method 4.2 at L17 shows prompt-family makes ~0 geometric contribution to residual stream
     4	(P-SoM↔P-text 0.0028, DOM↔P-prompt 0.0013). But forest plot drop-one places P-SoM as unique hero,
     5	implying axis-2 (prompt) contributes behaviorally. **Where in the model does axis-2 act?**
     6	
     7	**Method**: For each prompt-only pair (text format fixed, prompt swap), compute full 37-layer cosine gap.
     8	Overlay axis-1-only (text swap, prompt fixed) + image-axis P-SoM↔SoM reference curves to calibrate scale.
     9	
    10	## Results — classifieds site (stage4_multimode_b1_cls, 288 ex)
    11	
    12	| Pair | Group | L0 | L4 | L17 | L36 | Peak L | Peak gap |
    13	|---|---|---:|---:|---:|---:|---:|---:|
    14	| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0013 | 0.0067 | **L36** | 0.0067 |
    15	| P-text ↔ P-SoM  (axis-2 only, flat) | axis-2 | 0.0000 | 0.0006 | 0.0028 | 0.0089 | **L23** | 0.0114 |
    16	| DOM ↔ P-text    (axis-1 only, DOM-prompt) | axis-1 | 0.0000 | 0.0134 | 0.0120 | 0.0201 | **L23** | 0.0254 |
    17	| P-prompt ↔ P-SoM (axis-1 only, SoM-prompt) | axis-1 | 0.0000 | 0.0127 | 0.0113 | 0.0201 | **L23** | 0.0292 |
    18	| P-SoM ↔ SoM     (image-axis reference) | image | 0.0000 | 0.0394 | 0.0412 | 0.0411 | **L17** | 0.0412 |
    19	
    20	## Results — reddit site (stage4_multimode_b1_reddit, 288 ex)
    21	
    22	| Pair | Group | L0 | L4 | L17 | L36 | Peak L | Peak gap |
    23	|---|---|---:|---:|---:|---:|---:|---:|
    24	| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0012 | 0.0059 | **L36** | 0.0059 |
    25	| P-text ↔ P-SoM  (axis-2 only, flat) | axis-2 | 0.0000 | 0.0006 | 0.0027 | 0.0080 | **L23** | 0.0098 |
    26	| DOM ↔ P-text    (axis-1 only, DOM-prompt) | axis-1 | 0.0000 | 0.0125 | 0.0092 | 0.0183 | **L23** | 0.0217 |
    27	| P-prompt ↔ P-SoM (axis-1 only, SoM-prompt) | axis-1 | 0.0000 | 0.0115 | 0.0086 | 0.0176 | **L23** | 0.0240 |
    28	| P-SoM ↔ SoM     (image-axis reference) | image | 0.0000 | 0.0434 | 0.0423 | 0.0434 | **L4** | 0.0434 |
    29	
    30	## Interpretation
    31	
    32	Three hypotheses about axis-2 mechanism layer:
    33	
    34	1. **Truly null geometry** — axis-2 pair curves flat <0.01 at all layers. Prompt-family bypasses residual stream entirely (acts at attention pattern or output head). → Next: Exp 3 logit lens or Exp 4 attention probe.
    35	2. **Late-layer spike** — axis-2 pair curves spike at L25+ but flat at mid-layer. Prompt prior re-emerges at output decoding. → Next: Exp 5 late-layer patching.
    36	3. **Early-layer spike absorbed** — axis-2 pair curves spike at L0-L5 then collapse to ~0. Prompt embedding effect absorbed by mid-layer fusion. → Next: Exp 3 logit lens to verify if it re-emerges in output distribution.
    37	
    38	Compare peak layers above against axis-1 (text-format) pairs (the established mechanism with L17 peak) and image-axis reference (~0.04 magnitude). If axis-2 pair peak < 0.01 at all layers, hypothesis 1 holds.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/axis2_logit_lens.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
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
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/layer_axis_emergence.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
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
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# W6 feature attribution — H1 reddit 2/6 marks-like L04 peak
     2	
     3	**Setup**: Qwen3-VL-4B tokenizer (Qwen/Qwen3-VL-4B-Instruct). Each marks-like format variant tokenized on a canonical single-element example (N=1, role=button, label=Submit). First-token character class + marker-fingerprint token count compared between L04-peak and L17-peak subgroups.
     4	
     5	## Per-variant tokenization
     6	
     7	| Variant | Peak | Example | n_tok | First token | First char class | Marker fp |
     8	|---|---|---|---:|---|---|---:|
     9	| appagent_id | L04 | `id_1: Submit` | 5 | `id` | alphanumeric | 4 (`id·_·1·:`) |
    10	| plain_numbered | L04 | `1. Submit` | 3 | `1` | alphanumeric | 2 (`1·.`) |
    11	| som_standard | L17 | `[1] button 'Submit'` | 7 | `[` | markup-sigil | 3 (`[·1·]`) |
    12	| browser_use_at | L17 | `@1 Submit` | 3 | `@` | markup-sigil | 2 (`@·1`) |
    13	| tarsier_typed | L17 | `[B1:button:Submit]` | 7 | `[B` | markup-sigil | 7 (`[B·1·:·button·:·Submit·]`) |
    14	| xml_tagged | L17 | `<el_1 role='button'>Submit</el_1>` | 14 | `<` | markup-sigil | 4 (`<·el·_·1`) |
    15	| hash_id_control | L04 | `#a3f7 Submit` | 5 | `#a` | markup-sigil | 4 (`#a·3·f·7`) |
    16	| plain_sentence | L17 | `Submit` | 1 | `Submit` | alphanumeric | 1 (`Submit`) |
    17	| dom | L04 | `button: Submit (AXTree)` | 7 | `button` | alphanumeric | 2 (`button·:`) |
    18	| som | L17 | `[1] button 'Submit' (+ image marks)` | 11 | `[` | markup-sigil | 3 (`[·1·]`) |
    19	
    20	## Subgroup first-char-class distribution (6 marks-like only)
    21	
    22	| Subgroup | alphanumeric | markup-sigil | punctuation | quote | other |
    23	|---|---:|---:|---:|---:|---:|
    24	| L04-peak (2) | 2 | 0 | 0 | 0 | 0 |
    25	| L17-peak (4) | 0 | 4 | 0 | 0 | 0 |
    26	
    27	## Hypothesis verdict
    28	
    29	✅ **Hypothesis supported (clean split)**: L04-peak variants both start with alphanumeric tokens (2/2); L17-peak variants start with markup-sigil tokens (4/4).
    30	
    31	## Secondary features
    32	
    33	- L04-peak mean marker-fp tokens: 3.00
    34	- L17-peak mean marker-fp tokens: 4.00
    35	- Δ (L17 − L04): +1.00
    36	
    37	## Full token sequence per variant (marks-like 6)
    38	
    39	- **appagent_id** (L04, `id_1: Submit`): 5 tokens: `id` · `_` · `1` · `:` · `ĠSubmit`
    40	- **plain_numbered** (L04, `1. Submit`): 3 tokens: `1` · `.` · `ĠSubmit`
    41	- **som_standard** (L17, `[1] button 'Submit'`): 7 tokens: `[` · `1` · `]` · `Ġbutton` · `Ġ'` · `Submit` · `'`
    42	- **browser_use_at** (L17, `@1 Submit`): 3 tokens: `@` · `1` · `ĠSubmit`
    43	- **tarsier_typed** (L17, `[B1:button:Submit]`): 7 tokens: `[B` · `1` · `:` · `button` · `:` · `Submit` · `]`
    44	- **xml_tagged** (L17, `<el_1 role='button'>Submit</el_1>`): 14 tokens: `<` · `el` · `_` · `1` · `Ġrole` · `='` · `button` · `'>` · `Submit` · `</` · `el` · `_` · `1` · `>`
    45	
    46	## Interpretation
    47	
    48	Within the 6 marks-like variants, the L17 vs L04 split corresponds to whether the variant's first tokens are **markup-sigil tokens** (`[`, `<`, `@`) — which co-occur with HTML / web-agent traces in pretraining and trigger the visual-grounding shortcut at mid layers — versus **plain alphanumeric tokens** (`id`, `1`) — which are common in prose / dictionary listings and behave like AXTree-DOM, peaking early at L04 where the image-axis divergence is freshly observable but not yet routed through the shortcut path.
    49	
    50	**Control variants (counterexamples that refine the rule)**:
    51	- `hash_id_control` (`#a3f7 Submit`): markup-sigil first token but L04 peak. The `#` sigil alone is not sufficient — the marker must contain an **integer index** (which `#a3f7` does not). This is consistent with prior H2 "integer is the trigger token" framing.
    52	- `plain_sentence` (`Submit`): alphanumeric first token but L17 peak. With no list/marker structure at all, the divergence path differs — possibly because the text observation drops to bare labels with no positional anchors, which the model handles via a different late-layer routing (likely commitment without grounding).
    53	
    54	Together these say: the L17 mid-layer shortcut requires **(a) integer-indexed marker + (b) markup-sigil-leading delimiter**. Either alone fails to trigger it.
    55	
    56	**Paper §5 implication**: H1's binary 'marks-like vs not' prediction is too coarse. The mechanism trigger is the **conjunction** of integer marker + markup-sigil first token, not the abstract concept of 'indexed list'. Variants like `id_N:` and `N.` are nominally indexed but lack the sigil; `hash_id_control` has the sigil but lacks an integer. Both fail to peak at L17. This refines H1 to **'integer marker + markup-sigil delimiter → triggers shortcut at L17'**, which is testable on additional variants and on a `bare_N` falsifier (drop the bracket from `[N]` and re-extract).
    57	
    58	**Falsifier (concrete next experiment)**: variant `bare_N` = `N button 'Submit'` (no brackets), which has integer + no sigil. Hypothesis predicts L04 peak. If it peaks L17, hypothesis fails.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Hero-claim bootstrap CI (W1 defuse)
     2	
     3	Per-seed bootstrap 95% percentile CI on paired adjusted-SR diffs and drop-one oracle. B=10000, seed=42. Tasks resampled with replacement at task level.
     4	
     5	**Defuse target**: /stress W1 attack — paper §1 hero claim 'P-SoM 13.81% > SoM 10.48% reddit' is statistically marginal under author's own 2σ hedge.
     6	
     7	## reddit (N=210 same-task)
     8	
     9	**Per-mode adjusted SR (%)**:
    10	
    11	- dom: 9.52%
    12	- som: 10.48%
    13	- vision: 6.67%
    14	- phantom_som: 13.81%
    15	- phantom_text: 12.38%
    16	- phantom_prompt: 9.52%
    17	
    18	**Pairwise SR difference, bootstrap 95% CI:**
    19	
    20	| Comparison | Point (pp) | Median | 95% CI | P(diff > 0) | P(diff > 1pp) |
    21	|---|---:|---:|---:|---:|---:|
    22	| P-SoM vs SoM | +3.33 | +3.33 | [-0.95, +7.62] | 0.914 | 0.828 | 
    23	|  | | | ✗ crosses 0 | | |
    24	| P-SoM vs DOM | +4.29 | +4.29 | [+0.00, +8.57] | 0.963 | 0.914 | 
    25	|  | | | ✗ crosses 0 | | |
    26	| P-text vs DOM | +2.86 | +2.86 | [-0.95, +6.67] | 0.918 | 0.810 | 
    27	|  | | | ✗ crosses 0 | | |
    28	| P-SoM vs P-text | +1.43 | +1.43 | [-1.90, +5.24] | 0.739 | 0.548 | 
    29	|  | | | ✗ crosses 0 | | |
    30	
    31	**Drop-one oracle on 4-mode set (dom, som, vision, phantom_som), bootstrap 95% CI:**
    32	
    33	| Drop mode | Drop-one Δ (pp) | Median | 95% CI | P(Δ > 0) | P(Δ > 1pp) |
    34	|---|---:|---:|---:|---:|---:|
    35	| dom | +1.90 | +1.90 | [+0.48, +3.81] | 0.981 | 0.767 | 
    36	|  | | | ✓ strict-pos | | |
    37	| som | +1.90 | +1.90 | [+0.48, +3.81] | 0.980 | 0.762 | 
    38	|  | | | ✓ strict-pos | | |
    39	| vision | +1.43 | +1.43 | [+0.00, +3.33] | 0.949 | 0.574 | 
    40	|  | | | ✗ crosses 0 | | |
    41	| phantom_som | +3.33 | +3.33 | [+0.95, +6.19] | 0.998 | 0.969 | 
    42	|  | | | ✓ strict-pos | | |
    43	
    44	## classifieds (N=234 same-task)
    45	
    46	**Per-mode adjusted SR (%)**:
    47	
    48	- dom: 14.10%
    49	- som: 21.37%
    50	- vision: 13.68%
    51	- phantom_som: 14.53%
    52	- phantom_text: 14.53%
    53	
    54	**Pairwise SR difference, bootstrap 95% CI:**
    55	
    56	| Comparison | Point (pp) | Median | 95% CI | P(diff > 0) | P(diff > 1pp) |
    57	|---|---:|---:|---:|---:|---:|
    58	| P-SoM vs SoM | -6.84 | -6.84 | [-12.39, -1.28] | 0.005 | 0.001 | 
    59	|  | | | ✗ strict-neg | | |
    60	| P-SoM vs DOM | +0.43 | +0.43 | [-3.42, +4.70] | 0.538 | 0.374 | 
    61	|  | | | ✗ crosses 0 | | |
    62	| P-text vs DOM | +0.43 | +0.43 | [-3.42, +4.27] | 0.546 | 0.376 | 
    63	|  | | | ✗ crosses 0 | | |
    64	| P-SoM vs P-text | +0.00 | +0.00 | [-4.27, +4.27] | 0.464 | 0.317 | 
    65	|  | | | ✗ crosses 0 | | |
    66	
    67	**Drop-one oracle on 4-mode set (dom, som, vision, phantom_som), bootstrap 95% CI:**
    68	
    69	| Drop mode | Drop-one Δ (pp) | Median | 95% CI | P(Δ > 0) | P(Δ > 1pp) |
    70	|---|---:|---:|---:|---:|---:|
    71	| dom | +2.14 | +2.14 | [+0.43, +4.27] | 0.993 | 0.877 | 
    72	|  | | | ✓ strict-pos | | |
    73	| som | +8.55 | +8.55 | [+5.13, +12.39] | 1.000 | 1.000 | 
    74	|  | | | ✓ strict-pos | | |
    75	| vision | +3.42 | +3.42 | [+1.28, +5.98] | 1.000 | 0.988 | 
    76	|  | | | ✓ strict-pos | | |
    77	| phantom_som | +2.56 | +2.56 | [+0.85, +4.70] | 0.999 | 0.943 | 
    78	|  | | | ✓ strict-pos | | |
    79	
    80	## Verdict on /stress W1
    81	
    82	Read the **reddit P-SoM vs SoM** row + **reddit drop-one P-SoM** row:
    83	
    84	- If both CIs are strict-positive (ci_lo > 0) AND P(diff > 0) > 0.95 → **W1 attack defused**,   §1 hero claim is bootstrap-supported. Remove the '2σ hedge' from line 5, lead with the magnitude.
    85	- If CIs cross zero but P(diff > 0) > 0.80 → **W1 partially defused**, the claim is directional
    86	  but not strictly statistically significant. §1 hero must downgrade to 'competitive within 2σ' as
    87	  the author already wrote, but the complementarity (Jaccard / drop-one positive on N=7 tasks) carries
    88	  the structural weight.
    89	- If P(diff > 0) < 0.80 → **W1 sustained**, §1 hero claim must rewrite to 'parity / complementarity
    90	  rather than dominance'. The single-mode comparison is unsupported.

exec
/bin/bash -lc "python - <<'PY'
import json
from pathlib import Path
p=Path('results/mechanistic/stage4_multimode_b1_cls/method42_metrics.json')
print(p.exists(), p)
if p.exists():
 data=json.loads(p.read_text())
 print(json.dumps({k:data[k] for k in ['n_examples','n_layers','n_modes','n_per_mode']}, indent=2))
 for key in ['phantom_som_vs_phantom_text','phantom_som_vs_som','dom_vs_vision','dom_vs_phantom_prompt']:
  # actual canonical key maybe sorted mode order
  for kk in data['pairwise_cosine_gap']:
   if set(kk.split('_vs_'))==set(key.split('_vs_')):
    arr=data['pairwise_cosine_gap'][kk]; au=data['pairwise_auroc'][kk]
    print(kk, 'L17', arr[17], au[17], 'peak', data['peak_disruption_layers'][kk])
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 exited 127 in 0ms:
/bin/bash: line 1: python: command not found

exec
/bin/bash -lc "python3 - <<'PY'
import json
from pathlib import Path
p=Path('results/mechanistic/stage4_multimode_b1_cls/method42_metrics.json')
print(p.exists(), p)
if p.exists():
 data=json.loads(p.read_text())
 print(json.dumps({k:data[k] for k in ['n_examples','n_layers','n_modes','n_per_mode']}, indent=2))
 wanted=[('phantom_som','phantom_text'),('phantom_som','som'),('dom','vision'),('dom','phantom_prompt')]
 for a,b in wanted:
  for kk in data['pairwise_cosine_gap']:
   if set(kk.split('_vs_'))=={a,b}:
    arr=data['pairwise_cosine_gap'][kk]; au=data['pairwise_auroc'][kk]
    print(kk, 'L17', arr[17], au[17], 'peak', data['peak_disruption_layers'][kk])
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
True results/mechanistic/stage4_multimode_b1_cls/method42_metrics.json
{
  "n_examples": 288,
  "n_layers": 37,
  "n_modes": 6,
  "n_per_mode": {
    "dom": 48,
    "phantom_text": 48,
    "phantom_prompt": 48,
    "phantom_som": 48,
    "som": 48,
    "vision": 48
  }
}
phantom_text_vs_phantom_som L17 0.0027894973754882812 1.0 peak {'layer': 23, 'gap': 0.011447668075561523, 'auroc_at_peak': 1.0}
phantom_som_vs_som L17 0.04124343395233154 1.0 peak {'layer': 17, 'gap': 0.04124343395233154, 'auroc_at_peak': 1.0}
dom_vs_vision L17 0.054534733295440674 1.0 peak {'layer': 4, 'gap': 0.06526339054107666, 'auroc_at_peak': 1.0}
dom_vs_phantom_prompt L17 0.0012888908386230469 1.0 peak {'layer': 36, 'gap': 0.006670176982879639, 'auroc_at_peak': 0.997829861111111}

exec
/bin/bash -lc "find results/mechanistic -maxdepth 2 -name patching_continuation_results.json | sort | sed -n '1,120p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
results/mechanistic/stage2b_2x2_fwd_revtasks_myriad/patching_continuation_results.json
results/mechanistic/stage2b_cellcr_reddit_fwd_revtier_myriad/patching_continuation_results.json
results/mechanistic/stage2b_celle_random_cls_strong_myriad/patching_continuation_results.json
results/mechanistic/stage2b_celler_reddit_fwd_random_myriad/patching_continuation_results.json
results/mechanistic/stage2b_cellf_fwd_reddit_strong_myriad/patching_continuation_results.json
results/mechanistic/stage2b_continuation_b1_classifieds_pilot/patching_continuation_results.json
results/mechanistic/stage2b_continuation_b1_classifieds_pilot_reverse/patching_continuation_results.json
results/mechanistic/stage2b_curated_b1_cls_myriad/patching_continuation_results.json
results/mechanistic/stage2c_2x2_rev_strongtasks_myriad/patching_continuation_results.json
results/mechanistic/stage2c_celldr_reddit_rev_strongtier_myriad/patching_continuation_results.json
results/mechanistic/stage2c_cellg_rev_reddit_reverse_myriad/patching_continuation_results.json
results/mechanistic/stage2c_reverse_curated_b1_cls_myriad/patching_continuation_results.json
results/mechanistic/stage3_cellhd_cls_fwd_dom_myriad/patching_continuation_results.json
results/mechanistic/stage3_cellhd_red_fwd_dom_myriad/patching_continuation_results.json
results/mechanistic/stage3_cellhp_cls_fwd_prompt_myriad/patching_continuation_results.json
results/mechanistic/stage3_cellhp_red_fwd_prompt_myriad/patching_continuation_results.json
results/mechanistic/stage3_cellhprompt_cls_fwd_ptext_myriad/patching_continuation_results.json
results/mechanistic/stage3_cellhprompt_red_fwd_ptext_myriad/patching_continuation_results.json
results/mechanistic/stage3_cellht_cls_fwd_text_myriad/patching_continuation_results.json
results/mechanistic/stage3_cellht_red_fwd_text_myriad/patching_continuation_results.json

exec
/bin/bash -lc "find results/mechanistic -maxdepth 2 -name env_snapshot.json -o -name run_manifest.json | sort | sed -n '1,120p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
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
/bin/bash -lc "find results/mechanistic -maxdepth 2 -name hidden_states.npz | sort | sed -n '1,120p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
results/mechanistic/stage1B_archived_b1_classifieds_pilot/hidden_states.npz
results/mechanistic/stage1C_image_axis_b1_cls_pilot/hidden_states.npz
results/mechanistic/stage4_format_variation_b1_cls/hidden_states.npz
results/mechanistic/stage4_format_variation_b1_cls_reverse/hidden_states.npz
results/mechanistic/stage4_format_variation_b1_reddit/hidden_states.npz
results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz
results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz

exec
/bin/bash -lc 'rg -n "max_new_tokens|max-new-tokens|50-token|last-token|L0|embedding|AUROC|mean-difference|PCA|Holm|bootstrap|random|seed" docs/checkpoints/paper_drafts/section5_mechanism.md docs/checkpoints/mechanism/plan.md scripts/mechanistic/run_stage2b_continuation_pilot.py scripts/analysis/stage4_pca_cosine_gap.py scripts/analysis/stage2_layer_significance.py p79/mechanistic/activation_patching.py p79/mechanistic/extract_hidden_states.py' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
p79/mechanistic/extract_hidden_states.py:119:        """Forward pass with output_hidden_states=True. Return last-token hidden states.
p79/mechanistic/extract_hidden_states.py:129:            Tensor of shape (n_layers + 1, hidden_dim). Layer 0 is embedding output;
p79/mechanistic/activation_patching.py:13:- Patch source's layer-L last-token hidden state into target run at same layer.
p79/mechanistic/activation_patching.py:39:    F16 audit clarification 2026-05-09: layers L0..L_{n-1} are transformer
p79/mechanistic/activation_patching.py:40:    BLOCK OUTPUTS (post-attention + ffn + residual + post-norm). L0 is the
p79/mechanistic/activation_patching.py:41:    output of the first decoder layer, NOT the embedding output. To get
p79/mechanistic/activation_patching.py:42:    embedding output, hook `model.model.language_model.embed_tokens`
p79/mechanistic/activation_patching.py:44:    to avoid confusion with the embedding.
p79/mechanistic/activation_patching.py:80:        max_new_tokens: int = 30,
p79/mechanistic/activation_patching.py:83:        """Patch last-token hidden state at layer_idx on FIRST forward, then greedy-generate.
p79/mechanistic/activation_patching.py:113:                max_new_tokens=max_new_tokens,
p79/mechanistic/activation_patching.py:130:        max_new_tokens: int = 15,
p79/mechanistic/activation_patching.py:133:        """Add alpha * direction to last-token hidden at layer_idx on first forward, generate.
p79/mechanistic/activation_patching.py:143:            max_new_tokens: generation length
p79/mechanistic/activation_patching.py:169:                max_new_tokens=max_new_tokens,
p79/mechanistic/activation_patching.py:194:            position: 'last' = patch only last-token position (works across
p79/mechanistic/activation_patching.py:357:    max_new_tokens: int = 15,
p79/mechanistic/activation_patching.py:359:    randomize_source_hidden: bool = False,
p79/mechanistic/activation_patching.py:363:    Per-layer patch source's last-token hidden into target run, then greedy-generate
p79/mechanistic/activation_patching.py:364:    `max_new_tokens` tokens. Compare patched output sequence to source/target baselines.
p79/mechanistic/activation_patching.py:386:        **source_inputs, max_new_tokens=max_new_tokens, do_sample=False,
p79/mechanistic/activation_patching.py:395:        **target_inputs, max_new_tokens=max_new_tokens, do_sample=False,
p79/mechanistic/activation_patching.py:412:    # task-specific structure. If L17 disruption persists with random
p79/mechanistic/activation_patching.py:415:    if randomize_source_hidden:
p79/mechanistic/activation_patching.py:416:        import torch as _torch_for_random
p79/mechanistic/activation_patching.py:417:        randomized = []
p79/mechanistic/activation_patching.py:421:            noise = _torch_for_random.randn_like(h) * std + mean
p79/mechanistic/activation_patching.py:422:            randomized.append(noise)
p79/mechanistic/activation_patching.py:423:        source_cache = randomized
p79/mechanistic/activation_patching.py:435:            max_new_tokens=max_new_tokens,
p79/mechanistic/activation_patching.py:467:        "max_new_tokens": max_new_tokens,
scripts/analysis/stage2_layer_significance.py:15:3. Holm-Bonferroni correction across the 6 tested layers (per direction).
scripts/analysis/stage2_layer_significance.py:31:# F12 audit fix 2026-05-09: L0-L35 are **transformer block outputs**, NOT
scripts/analysis/stage2_layer_significance.py:32:# embedding output. The patching hooks (`activation_patching.py`
scripts/analysis/stage2_layer_significance.py:35:#   L0  = output of block 0 (first decoder layer post-norm). Often
scripts/analysis/stage2_layer_significance.py:37:#         embedding; NOT the embedding itself.
scripts/analysis/stage2_layer_significance.py:41:# If you need true embedding output (n_layers+1 indexing), add a hook to
scripts/analysis/stage2_layer_significance.py:63:    """Holm-Bonferroni step-down. Input p-values, output (adj_p, reject@0.05)."""
scripts/analysis/stage2_layer_significance.py:93:    Returns dict with per-layer rows + Holm-adjusted p-values.
scripts/analysis/stage2_layer_significance.py:103:    rng = np.random.default_rng(seed=42)
scripts/analysis/stage2_layer_significance.py:112:        # C9 fix: handle constant-column edge case (e.g. cell D L0 has all 1.0
scripts/analysis/stage2_layer_significance.py:129:            # C3 fix: bootstrap percentile 95% CI on mean diff (resample tasks)
scripts/analysis/stage2_layer_significance.py:217:    out.append("Holm-Bonferroni correction across 6 tested layers per direction.")
scripts/analysis/stage2_layer_significance.py:219:    out.append("Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001 (Holm-adjusted).")
scripts/analysis/stage2_layer_significance.py:226:        out.append("| Layer | mean(L) | mean(L35) | Δ mean | Δ std | 95% CI (boot) | t-stat | p (raw) | p (Holm) | reject H0 |")
scripts/analysis/stage2_layer_significance.py:251:    out.append("- **Holm-rejected layers** in forward direction = layers where mid-layer disruption survives multiple-comparison correction.")
scripts/analysis/stage2_layer_significance.py:338:        extra.append("| Layer | mean(L) | mean(L35) | Δ mean | Δ std | 95% CI (boot) | t-stat | p (raw) | p (Holm) | reject H0 |")
scripts/analysis/stage4_pca_cosine_gap.py:2:"""Stage 4 Method 4.2: PCA cosine gap analysis of phantom routing space.
scripts/analysis/stage4_pca_cosine_gap.py:10:  B. AUROC: project hidden states onto (mean_A - mean_B) direction, predict mode
scripts/analysis/stage4_pca_cosine_gap.py:11:  C. Per-(mode, layer) PCA top-10 variance explained
scripts/analysis/stage4_pca_cosine_gap.py:26:from sklearn.decomposition import PCA
scripts/analysis/stage4_pca_cosine_gap.py:96:                pca_var[mi, L] = PCA(n_components=n_comp).fit(X[:, L, :]).explained_variance_ratio_.sum()
scripts/analysis/stage4_pca_cosine_gap.py:126:        "# Stage 4 Method 4.2: PCA Cosine Gap Analysis",
scripts/analysis/stage4_pca_cosine_gap.py:135:        "| Mode pair | Peak layer | Cosine gap | AUROC at peak |",
scripts/analysis/stage4_pca_cosine_gap.py:144:    L17_section.append("| Mode pair | L17 cosine gap | L17 AUROC |")
scripts/analysis/stage4_pca_cosine_gap.py:191:    ax.set_title("(b) Pairwise AUROC (project onto Δ-mean direction, classify)")
scripts/analysis/stage4_pca_cosine_gap.py:198:    ax.set_ylabel("Top-10 PCA cumulative variance explained")
scripts/mechanistic/run_stage2b_continuation_pilot.py:12:  greedy-generate max_new_tokens, compare full token sequences.
scripts/mechanistic/run_stage2b_continuation_pilot.py:28:      --site classifieds --n-tasks 3 --step 2 --max-new-tokens 15 \
scripts/mechanistic/run_stage2b_continuation_pilot.py:155:    p.add_argument("--max-new-tokens", type=int, default=15, help="Continuation length (15 covers JSON envelope start)")
scripts/mechanistic/run_stage2b_continuation_pilot.py:175:        "--random-inject", action="store_true",
scripts/mechanistic/run_stage2b_continuation_pilot.py:183:        "--random-seed", type=int, default=42,
scripts/mechanistic/run_stage2b_continuation_pilot.py:184:        help="Seed for --random-inject Gaussian noise (paper-grade reproducibility). "
scripts/mechanistic/run_stage2b_continuation_pilot.py:185:             "Same seed + same input = same noise = byte-identical re-runs. Default 42.",
scripts/mechanistic/run_stage2b_continuation_pilot.py:189:    # C8 fix: seed all RNGs when random-inject is on, for paper-grade
scripts/mechanistic/run_stage2b_continuation_pilot.py:191:    # Default seed=42 means re-running with same data + code produces
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
scripts/mechanistic/run_stage2b_continuation_pilot.py:299:        logger.info(f"task {task_id}: running continuation patching grid (max_new_tokens={args.max_new_tokens})...")
scripts/mechanistic/run_stage2b_continuation_pilot.py:302:            max_new_tokens=args.max_new_tokens,
scripts/mechanistic/run_stage2b_continuation_pilot.py:303:            randomize_source_hidden=args.random_inject,
scripts/mechanistic/run_stage2b_continuation_pilot.py:310:        # F18 audit fix 2026-05-09: include reverse / tier / random_inject /
scripts/mechanistic/run_stage2b_continuation_pilot.py:311:        # random_seed in incremental JSON so downstream stage2 stat scripts
scripts/mechanistic/run_stage2b_continuation_pilot.py:329:                    "max_new_tokens": args.max_new_tokens,
scripts/mechanistic/run_stage2b_continuation_pilot.py:336:                    "random_inject": args.random_inject,
scripts/mechanistic/run_stage2b_continuation_pilot.py:337:                    "random_seed": args.random_seed,
scripts/mechanistic/run_stage2b_continuation_pilot.py:371:        "ld_to_source": f"Levenshtein dist → source\n(0=identical, max~{args.max_new_tokens})",
scripts/mechanistic/run_stage2b_continuation_pilot.py:379:        ax.set_xlabel("Layer index (0=embedding, ≥1=post-block)")
scripts/mechanistic/run_stage2b_continuation_pilot.py:391:        f"max_new_tokens={args.max_new_tokens})",
scripts/mechanistic/run_stage2b_continuation_pilot.py:427:- max_new_tokens: {args.max_new_tokens} (greedy continuation, deterministic)
scripts/mechanistic/run_stage2b_continuation_pilot.py:428:- Random injection: {"YES, seed=" + str(args.random_seed) + " (paper-grade reproducible)" if args.random_inject else "NO (real source hidden injected)"}
scripts/mechanistic/run_stage2b_continuation_pilot.py:438:| L0  | {overlap_src[0]:.2f} | {overlap_tgt[0]:.2f} | {ld_src[0]:.1f} | {ld_tgt[0]:.1f} |
scripts/mechanistic/run_stage2b_continuation_pilot.py:458:- Scale up: 5 task × max_new_tokens=20 (~75 min) for tighter mean ± std
scripts/mechanistic/run_stage2b_continuation_pilot.py:473:            "max_new_tokens": args.max_new_tokens,
docs/checkpoints/mechanism/plan.md:19:| **4** | Model-internal | L17 mid-layer is BOTH discrimination locus (probe AUROC 1.0) AND causally active planning site (Stage 2/3 patching + Method 4.4 v2 reliability) |
docs/checkpoints/mechanism/plan.md:21:### 1.2 Three-axis hierarchy quantified (Method 4.2 PCA cosine gap, Qwen3-VL-4B B1 cls)
docs/checkpoints/mechanism/plan.md:37:| AXTree (hierarchical) | **L04** | DOM↔Vision, DOM↔SoM, P-prompt↔Vision, P-prompt↔SoM |
docs/checkpoints/mechanism/plan.md:46:| **AXTree hierarchical (DOM)** | **L04** | **SOLE format defeating shortcut** |
docs/checkpoints/mechanism/plan.md:75:- Filter to tasks where Mirage Effect is well-defined: agent succeeds in `som` mode but fails in `dom` mode (real image-feature dependence, not random failure)
docs/checkpoints/mechanism/plan.md:85:### 3.1 Method 4.2 — PCA cosine gap (DONE)
docs/checkpoints/mechanism/plan.md:89:- B. AUROC via (mean_A − mean_B) projection
docs/checkpoints/mechanism/plan.md:90:- C. Per-(mode, layer) PCA top-10 variance explained
docs/checkpoints/mechanism/plan.md:97:- Test E bootstrap 95% CI tight (4-15% of mean)
docs/checkpoints/mechanism/plan.md:127:> Mid-layer L17 hidden state at last-token position is the causally active planning site for phantom routing space mode selection in Qwen3-VL-4B web agents.
docs/checkpoints/mechanism/plan.md:132:1. **Probe-level** (Method 4.2 PCA cosine gap, AUROC 1.000 across 540 tests)
docs/checkpoints/mechanism/plan.md:133:2. **Replacement patching** (Stage 2/3 Cell A-H, L17 disruption peak, 8/8 Holm-sig)
docs/checkpoints/mechanism/plan.md:140:| A1 | L17 last-token hidden state mediates action selection (not earlier obs token positions) | Stage 2/3 swept all layers, L17 is peak |
docs/checkpoints/mechanism/plan.md:144:| A5 | Replacement patching faithfully simulates "natural" model read of the representation | Cell E random-injection control rules out non-specific disruption — content-specific causation confirmed |
docs/checkpoints/mechanism/plan.md:148:Cell E random-injection control: replacing source hidden with Gaussian noise (same μ, σ) yields **null L17 disruption effect**. Confirms our patching effect is source-content-specific, not noise-driven. Most directly stresses A5.
docs/checkpoints/mechanism/plan.md:152:- Method 4.2 AUROC 1.000 = validation (decodability)
docs/checkpoints/mechanism/plan.md:160:| Pair @L17 | Cosine gap | 95% CI | AUROC |
docs/checkpoints/mechanism/plan.md:171:| Cell | Site | Direction | L17 Δoverlap | Holm-sig |
docs/checkpoints/mechanism/plan.md:177:| E | cls | random injection | -0.03 (uniform) | ✓ (negative control) |
docs/checkpoints/mechanism/plan.md:181:| Er | reddit | random injection | ~0 (uniform) | ✓ |
docs/checkpoints/mechanism/plan.md:191:| H-t-red | reddit | SoM → P-text | L01 (0.194) | -0.24 | `stage3_cellht_red_fwd_text_myriad/` |
docs/checkpoints/mechanism/plan.md:194:**Stage 3 interpretation (6/6 cells complete 2026-05-12)**: All forward SoM→{no-image-arm} patching cells show mid-layer L11-L17 disruption -0.19 to -0.33 Δoverlap→tgt. Magnitude > random injection control (Cell E -0.03) at all 6. **Mechanism additivity confirmed**: image-feature axis is shared substrate across DOM / P-text / P-prompt arms — single SoM→{any-no-image-arm} patching displaces target prediction toward source. Cross-site cls + reddit both replicate (paper §5 universal mid-layer fusion locus); reddit fusion locus slightly earlier (L11 vs cls L17), magnitude identical.
docs/checkpoints/mechanism/plan.md:227:`docs/checkpoints/mechanism/results/layer_axis_emergence.md`. AXTree-no-image side → L04 peak (4/4); [SOM_MARKS]-no-image side → L17–L36 peak (4/4). Zero overlap. Mirage Effect mechanism signature.
docs/checkpoints/mechanism/plan.md:231:`docs/checkpoints/mechanism/results/format_variation_h1_test.md`. 8 industry-relevant text formats + 2 controls. AXTree hierarchical (DOM) is **unique format** preserving L04 image-axis peak; all 8 flat-list variants (SoM standard, Browser Use @, AppAgent id_, Tarsier typed, plain numbered, XML tagged, hash-ID control, plain-sentence control) shift peak to L17–L36. Trigger is flat element listing, not specific token pattern.
docs/checkpoints/mechanism/plan.md:240:| ✅ Cross-site Method 4.2 — does cls finding replicate on reddit? | **Closed 2026-05-12 16:30**: P-SoM↔DOM L17=0.0098 + P-SoM↔SoM L17=0.0423, AUROC 1.0 → Mirage signature replicated. See §7.3.1 | — |
docs/checkpoints/mechanism/plan.md:245:| AXTree-defeats-shortcut mechanism — *why* hierarchy beats flat? Cross-modal attention specific to indentation tokens? | High (paper §5 supplement) | Activation patching at L4 with hierarchical-text vs flat-text → see which attention heads pre-disrupt image embedding |
docs/checkpoints/mechanism/plan.md:274:> # 1. Method 4.2 PCA cosine gap port 到 6 modes
docs/checkpoints/mechanism/plan.md:275:> 24 cls strong-tier × 2 step × 6 mode = 288 hidden states, 37 layer × 2560 dim。全 540 pair × layer AUROC = 1.000 (perm baseline 0.629, real 9.8σ above). 你方法在 multimodal Qwen 上 readable transfer 干净。
docs/checkpoints/mechanism/plan.md:288:>   - **AXTree hierarchical 是唯一保留 L04 peak 的 format**
docs/checkpoints/mechanism/plan.md:310:| **P4** | cls reverse-tier (selection-bias defense) | Myriad 353763 | ✅ **done 18:50:46** — shape (260, 37, 2560), 10 modes, 46 MB pulled. Same pattern as cls strong-tier (L36 marks-like + L04 dom). Selection-bias defended | `stage4_format_variation_b1_cls_reverse/hidden_states.npz` |
docs/checkpoints/mechanism/plan.md:397:3. **KL @ L36 ≈ 0 paradox**: 因 mean hidden state at last layer collapse to common JSON format header. Mode-distinct signal concentrated in **L23-L25 decoding window** (not final embedding). This is the "knows but says differently" structural mirror of Wu et al. tool calling.
docs/checkpoints/mechanism/plan.md:408:| appagent_id / plain_numbered | **L04** | L36 |
docs/checkpoints/mechanism/plan.md:409:| hash_id_control | **L04** ✓ (acts as control) | L36 (control failed) |
docs/checkpoints/mechanism/plan.md:411:| dom (baseline) | **L04** ✓ | L04 ✓ |
docs/checkpoints/mechanism/plan.md:413:**Reddit nuance — cleaner mid-layer fusion**: Reddit 上 marks-like 4/6 真 peak 在 L17 (mid-layer), cls 上 L36 是 monotonic increasing artifact (peak hit boundary). Reddit hash_id_control L04 acts as proper "no integer" control (cls 上失败). Reddit data supports Q5 mid-layer fusion hypothesis better than cls.
docs/checkpoints/mechanism/plan.md:415:Caveats: small n (24×2=48/mode) makes 2/6 marks-like falling to L04 (appagent_id, plain_numbered) plausible as sampling noise; plain_sentence triggering L17 on reddit (not cls) suggests reddit narrative comments may pattern-match list semantics.
docs/checkpoints/mechanism/plan.md:424:| DOM ↔ Vision peak | L04 = 0.0687 (AUROC=1.0) | L04 similar |
docs/checkpoints/mechanism/plan.md:438:| hash_id_control | L36 (failed control) | **L36** ✓ same | L04 ✓ proper control |
docs/checkpoints/mechanism/plan.md:440:| dom baseline | L04 ✓ | **L04** ✓ | L04 ✓ |
docs/checkpoints/mechanism/plan.md:464:- **§6 cost-aware routing** — Peale et al. 2026 uncertainty decomposition anchor adds theoretical layer to phantom routing space's empirical AUROC. Method 4.2 cosine gap could serve as "reducible uncertainty" signal in deployment
docs/checkpoints/paper_drafts/section5_mechanism.md:7:This section is the Zoom-4 layer of the paper's four-level account. Zoom 1 defines the architectural intervention, "skip the annotated image"; Zoom 2 measures the behavioral axes of text payload, prompt family, and image presence; Zoom 3 links the observed behavior to Mirage-style no-image visual reasoning and prompt-format sensitivity; Zoom 4 asks where the resulting mode is represented and whether it is causally used by the model. We index layers L0-L36, where L0 is the embedding-block output and L1-L36 are the 36 transformer decoder block outputs.
docs/checkpoints/paper_drafts/section5_mechanism.md:9:The analysis builds on the linear-readable and steerable circuit framework of Wu et al., which uses mode means, PCA geometry, and mean-difference activation steering to study tool selection, and on work showing middle-layer cross-modal information flow in VLMs \citep{wu2026toolcalling,kaduri2024whatsintheimage}. Our setting is not a replication of those papers. It is a multimodal web-agent application of the same representation-level question: whether a behaviorally useful routing arm is linearly readable, partially steerable, and causally active inside the model.
docs/checkpoints/paper_drafts/section5_mechanism.md:11:Four mechanism claims organize the evidence. First, observation modes are geometrically separable in the residual stream: the PCA cosine-gap analysis reaches AUROC 1.0 on the main mode contrasts. Second, Phantom-SoM is a mid-layer text-axis sibling of DOM/P-text, not an image-axis sibling of full SoM. Third, SoM-to-no-image activation patching displaces target continuations at mid layers L11-L17 with consistent magnitude across classifieds and reddit. Fourth, the shortcut trigger is **better described as the conjunction of integer-indexed marker and markup-sigil leading delimiter** than as an abstract "flat element list" — AXTree hierarchy preserves the early L04 image-axis peak, but so do indexed variants that lack either the integer (e.g., `hash_id_control`) or the sigil (e.g., `appagent_id`, `plain_numbered`); only the conjunction triggers the late shift. This refinement is **exploratory after W6** and awaits held-out falsifiers (`bare_N`, `bracket_no_int`).
docs/checkpoints/paper_drafts/section5_mechanism.md:17:| Geometry (cosine PCA per layer) | Method 4.2 (§5.2) | **Strong** — AUROC 1.0 on key contrasts, 5/5 robustness, 100% per-task above 0.010 threshold |
docs/checkpoints/paper_drafts/section5_mechanism.md:18:| Causal continuation patching (SoM → no-image arms) | Stage 2/3 (§5.4) | **Causal** — mid-layer L12-L18 transfers across cls + reddit, additive across DOM/P-text/P-prompt targets, Gaussian-random negative controls at ~0 |
docs/checkpoints/paper_drafts/section5_mechanism.md:19:| Causal axis-2 prompt-only patching | Exp 5 cellhprompt (this section) | **Causal continuation evidence on 2 sites N=24 each**; bootstrap CI and content-matched (task-shuffled / random-source) controls **PENDING** (cellhprm_*_rand 359719/359720 in flight) |
docs/checkpoints/paper_drafts/section5_mechanism.md:26:## 5.2 Method 4.2: PCA Cosine Gap
docs/checkpoints/paper_drafts/section5_mechanism.md:28:Method 4.2 extracts hidden states from Qwen3-VL-4B B1 runs and compares observation modes by layer. For each mode pair and layer, we compute the cosine gap between hidden-state means, evaluate AUROC by projecting examples onto the mean-difference direction, and summarize per-mode geometry through PCA top-10 variance. The classifieds baseline contains 288 examples, formed from 24 strong-tier tasks, two archived steps, and six modes, over 37 indexed layers.
docs/checkpoints/paper_drafts/section5_mechanism.md:30:The robustness suite passes all five checks in the plan. Label permutation leaves the real AUROC 9.8 standard deviations above the permuted baseline. Per-task analysis is positive for all 24 tasks. Step 2 and step 5 curves are invariant at the mechanism level. The L23 silhouette score is at least 0.5, showing nontrivial clustering. Bootstrap 95% confidence intervals are tight, with widths of roughly 4-15% of the corresponding means.
docs/checkpoints/paper_drafts/section5_mechanism.md:34:| Pair at L17 | Cosine gap | 95% CI | AUROC |
docs/checkpoints/paper_drafts/section5_mechanism.md:41:The reddit replication lands the same qualitative geometry. At L17, P-SoM is close to DOM with cosine gap 0.0098 and close to P-text with gap 0.0027, while P-SoM-to-SoM remains much larger at 0.0423 and P-SoM-to-Vision at 0.0457. The DOM-to-Vision image-axis peak is L04 with cosine gap 0.0687 and AUROC 1.0.
docs/checkpoints/paper_drafts/section5_mechanism.md:47:Method 4.4 tests whether the readable mode direction can be used as a steering direction. For each layer, we form a mean-difference vector between Phantom-SoM-like and DOM-like hidden states, add it to each input at generation time with scaling factor $\alpha$, and evaluate whether the continuation moves toward the target mode while preserving the JSON action envelope. Following HDMI's evaluation vocabulary, reliability is the harmonic mean of completeness and selectivity, not a raw shift rate \citep{khorasani2026hdmi}.
docs/checkpoints/paper_drafts/section5_mechanism.md:53:This creates a probe-causal dissociation. The mid-layer geometry is cleanly readable and causally implicated by patching, but fixed mean-difference steering is only partially reliable. The 0.33 H-mean is therefore an evidence ceiling for Method 4.4, not a final control result. Section 8 should treat LA-HDMI and SAE feature steering as future work motivated by this ceiling, without claiming that either method has already improved it.
docs/checkpoints/paper_drafts/section5_mechanism.md:57:Activation patching provides the causal test. For each task, the clean/source run and corrupt/target run use the same archived browser step and deterministic 50-token continuation. In the core SoM-to-Phantom-SoM setup, the source prompt is `som`: task instruction, SoM prompt family, flat `[SOM_MARKS]` text, and annotated screenshot. The target prompt is `phantom_som`: the same instruction, same prompt family, and same `[SOM_MARKS]` text, but no image. Source hidden states are cached by layer, injected into the final input-token position of the target on the first forward pass, and subsequent decoding proceeds normally through the model cache.
docs/checkpoints/paper_drafts/section5_mechanism.md:59:Each patched continuation is scored against the unpatched source and target continuations. The main disruption statistic is the drop in `token_overlap_to_target`; Levenshtein distance to target is the paired backup. Layer-wise tests compare each grid layer to the final-layer reference using task-paired differences and Holm-Bonferroni correction across the canonical grid. Random-injection controls replace source hidden states with Gaussian tensors matched to source activation mean and standard deviation.
docs/checkpoints/paper_drafts/section5_mechanism.md:61:The Stage 2 P-SoM<->SoM dashboard now contains ten completed cells, including reddit F/G and the reddit random control:
docs/checkpoints/paper_drafts/section5_mechanism.md:63:| Cell | Site | Direction | Mid-layer target-overlap drop | Holm status |
docs/checkpoints/paper_drafts/section5_mechanism.md:69:| E | cls | random injection | -0.03 uniform | negative control |
docs/checkpoints/paper_drafts/section5_mechanism.md:73:| Er | reddit | random injection | approximately 0 uniform | negative control |
docs/checkpoints/paper_drafts/section5_mechanism.md:86:The cleanest single-pair signature is the image-axis peak-layer dichotomy. Across eight image-presence contrasts, the no-image side's text format predicts the peak layer with zero overlap. If the no-image side is AXTree text, the image-axis cosine gap peaks at L04 in all four pairs: DOM<->Vision, DOM<->SoM, P-prompt<->Vision, and P-prompt<->SoM. If the no-image side is `[SOM_MARKS]` or another flat marks text, the peak shifts to L17-L36 in all four pairs: P-text<->Vision, P-text<->SoM, P-SoM<->Vision, and P-SoM<->SoM.
docs/checkpoints/paper_drafts/section5_mechanism.md:90:The format-variation grid contains ten modes: six marks-like variants, two controls, and DOM/SoM baselines. In the classifieds strong-tier baseline, all six marks-like variants peak at L36, the hash-ID control also peaks at L36, the plain-sentence control peaks at L17, and the DOM baseline preserves the L04 peak. Because L36 is the boundary layer, this is best read as a strong late/monotonic signature rather than as a precise late-layer mechanism.
docs/checkpoints/paper_drafts/section5_mechanism.md:92:The classifieds reverse-tier run reproduces the strong-tier shape. The six marks-like variants and hash-ID control again peak at L36, the plain-sentence control moves to L22, and DOM remains at L04. This defends H1 against the selection-bias concern that strong-tier curation alone created the pattern.
docs/checkpoints/paper_drafts/section5_mechanism.md:94:The reddit format run is cleaner for the mid-layer interpretation. Four of six marks-like variants peak at L17, the plain-sentence control peaks at L17, hash-ID control returns to L04, and DOM remains at L04. **W6 attribution** (`docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md`, exploratory) further finds that the two L04 marks-like variants (`appagent_id`, `plain_numbered`) share a feature with the L04 DOM baseline: their first tokens are alphanumeric, while the four L17-peaking marks-like variants all start with markup-sigil tokens (`[`, `<`, `@`). The hash-ID control (`#a3f7`) starts with a sigil but lacks integer-marker structure and also peaks at L04, suggesting the trigger conjunction is integer-marker + markup-sigil rather than either alone. This is a post-hoc feature-attribution on a small (N=6 marks-like) format set; held-out falsifiers (`bare_N` = integer without sigil, `bracket_no_int` = sigil without integer) are not yet run. Cross-site, the safe claim is directional: marker formats that combine integer indexing with markup-sigil leading delimiters tend to delay image-axis separation into mid/late layers, while AXTree hierarchy and indexed-list variants lacking either feature preserve the early L04 image-axis peak. The reddit curve reveals the true L11-L17 fusion locus more clearly than the classifieds L36 boundary artifact.
docs/checkpoints/paper_drafts/section5_mechanism.md:100:The second defense is selection-bias robustness. The classifieds reverse-tier run replicates the strong-tier H1 pattern, including L36 marks-like peaks and L04 DOM baseline. The shortcut signature is therefore not an artifact of selecting tasks where SoM beats DOM.
docs/checkpoints/paper_drafts/section5_mechanism.md:104:The fourth defense is cross-site Mirage geometry. Reddit Method 4.2 reproduces the central relation: P-SoM is close to DOM/P-text at L17 and far from SoM/Vision on the image axis, with AUROC 1.0 on the key contrasts. This supports cross-site generalization of the mechanism claim, not B0/B1 capability scaling.
docs/checkpoints/paper_drafts/section5_mechanism.md:130:This layered hierarchy partially resolves the Phantom-SoM hero puzzle without requiring a non-mechanistic explanation. Phantom-SoM occupies three distinct axis positions: **large image-axis separation from SoM** at L17 (0.0412 cosine gap, the largest single signal — note this is a *separation* from SoM, not a proximity; the no-image marks-text reshapes how image-axis divergence accumulates rather than recovering an image-like internal state), **small prompt-family separation from P-text** at L23 (0.011 signal, consistent across cls and reddit), and **medium text-format separation from P-prompt** at L23 (0.029 signal). P-text occupies only the text-format separation; P-prompt occupies only the prompt-family separation against P-SoM; DOM occupies none of the three. The drop-one hero status of Phantom-SoM in `fig_meta_forest.png` therefore corresponds to the only mode that simultaneously expresses all three axis positions. The connection from "three-axis positional uniqueness" to "drop-one task-success contribution" is currently associational, not yet causally bridged — the patching evidence shows residual-stream displacement on 50-token continuation, not direct displacement of task-level SR / drop-one oracle. Closing that behavioral causal bridge is open work.
docs/checkpoints/paper_drafts/section5_mechanism.md:136:Two corollaries follow. First, the KL trajectory drops to approximately zero at L36 even though L23 KL is substantial. The mean hidden state at the final layer collapses to the shared JSON action-header tokens that every mode emits, so mode-distinct output signal is concentrated in the L23-L25 decoding window rather than at the final embedding. Second, this output-amplification observation is **mechanistic, not a deployment-time classifier claim**: the lm_head acts as an axis-agnostic ratio-preserving projection that scales residual-stream geometry into output-space KL — the L23-L25 KL magnitude is a property of the mean hidden state, not a per-task discriminator. Whether the L23-L25 hidden representation can be used as a held-out mode classifier — with per-task AUROC, random-direction baseline, and competitive comparison to surface-token classifiers — is open work. Routing exploitation, deferred to paper 2, will need to make this case explicitly rather than inheriting it from §5.7.
docs/checkpoints/paper_drafts/section5_mechanism.md:140:The main limit is the Method 4.4 ceiling. The cosine-gap and patching evidence point to L11-L17 as the readable and causally active fusion region, while the best fixed mean-difference steering cell is late, L33 with $\alpha=10$, and has H-mean 0.33 because completeness and selectivity trade off. This supports a mechanism interpretation but not a strong deployment-time steering claim.
docs/checkpoints/paper_drafts/section5_mechanism.md:152:Behavioral content to relocate from current `section5_mechanism_reddit.md`: lines 17-75 should move to Section 4 or a new behavioral-routing subsection. Specifically, lines 17-23 are reddit substrate framing; lines 25-35 are Axis 1 text-payload behavior; lines 37-47 are Axis 2 prompt behavior; lines 49-59 are Axis 3 image behavior; lines 61-67 are compound P-SoM versus DOM behavior; lines 69-75 are scope/noise limitations. Lines 1-15 are method material that was retained conceptually but must use the new L0-L36 layer convention. Line 77 should be deleted or replaced because routing implementation is now paper-2, not paper-1 Section 6.
docs/checkpoints/paper_drafts/section5_mechanism.md:164:3. ✅ §5.7 corollary 2 — "deployment-time mode classifier on output logprobs has strictly more signal" + "Section 6 routing should treat L23-L25 logit-lens features as the cheapest mode-axis discriminator" → softened to "mechanistic observation, not deployment-time classifier claim; held-out classifier with random-direction baseline is open work"
docs/checkpoints/paper_drafts/section5_mechanism.md:169:- **§4 P-text adjusted SR inconsistency**: §4 table line 37 says 11.90, prose line 106 says 12.38, hero_claim_bootstrap_ci.md says 12.38. Need to canonicalize one number from episode-level adjusted-success and update every occurrence. (1h)
docs/checkpoints/paper_drafts/section5_mechanism.md:172:- **Exp 5 cellhprompt bootstrap CI + content-matched control**: Gaussian random injection control 359719/359720 in flight; codex notes Gaussian alone is weak — also need task-shuffled (source from different task) and per-task bootstrap CIs.
docs/checkpoints/paper_drafts/section5_mechanism.md:173:- **Behavioral causal bridge gap**: patching displaces 50-token continuation, not SR / drop-one. No experiment currently bridges mid-layer patching effect to task-success outcome. Open work.

exec
/bin/bash -lc "python3 - <<'PY'
import json, glob
for p in sorted(glob.glob('results/mechanistic/*/patching_continuation_results.json')):
    d=json.load(open(p))
    cfg=d.get('config',{})
    print(p, 'N', len(d.get('per_task',[])), 'max_new', cfg.get('max_new_tokens'), 'src', cfg.get('source_mode'), 'tgt', cfg.get('target_mode'), 'reverse', cfg.get('reverse'), 'random', cfg.get('random_inject'), 'tier', cfg.get('tier'))
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 103ms:
exec
/bin/bash -lc "python3 - <<'PY'
import json
for p in ['results/mechanistic/stage2b_curated_b1_cls_myriad/run_manifest.json','results/mechanistic/stage2b_celle_random_cls_strong_myriad/run_manifest.json','results/mechanistic/stage3_cellhd_cls_fwd_dom_myriad/run_manifest.json']:
    print('\\n',p)
    d=json.load(open(p))
    print(json.dumps({k:d.get(k) for k in ['stage','direction','site','patch_config','input_dataset','aggregate']}, indent=2)[:3000])
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:

 results/mechanistic/stage2b_curated_b1_cls_myriad/run_manifest.json
{
  "stage": "stage2b_continuation_curated",
  "direction": "forward",
  "site": "classifieds",
  "patch_config": {
    "source_mode": "som",
    "target_mode": "phantom_som",
    "step_idx": 2,
    "max_new_tokens": 50,
    "n_layers_swept": 36,
    "hook_position": "last_token",
    "first_forward_only": true,
    "min_free_vram_gb": 0.0
  },
  "input_dataset": {
    "archived_run_dir": "/home/ucab352/Scratch/p79/results/mechanistic/archive_subset_b1_cls",
    "n_tasks_requested": 24,
    "n_tasks_completed": 24,
    "task_ids": [
      1,
      33,
      40,
      82,
      116,
      161,
      214,
      19,
      20,
      122,
      181,
      9,
      32,
      37,
      60,
      61,
      69,
      73,
      99,
      108,
      109,
      215,
      227,
      228
    ]
  },
  "aggregate": {
    "best_layer_overlap_src_mean": 11,
    "best_overlap_src_mean": 0.16666666666666674,
    "best_layer_ld_src_mean": 15,
    "L11_overlap_src_mean": 0.16666666666666674,
    "L17_overlap_src_mean": 0.15750000000000008
  }
}

 results/mechanistic/stage2b_celle_random_cls_strong_myriad/run_manifest.json
{
  "stage": "stage2b_continuation_curated",
  "direction": "forward",
  "site": "classifieds",
  "patch_config": {
    "source_mode": "som",
    "target_mode": "phantom_som",
    "step_idx": 2,
    "max_new_tokens": 50,
    "n_layers_swept": 36,
    "hook_position": "last_token",
    "first_forward_only": true,
    "min_free_vram_gb": 0.0
  },
  "input_dataset": {
    "archived_run_dir": "/home/ucab352/Scratch/p79/results/mechanistic/archive_subset_b1_cls",
    "n_tasks_requested": 24,
    "n_tasks_completed": 24,
    "task_ids": [
      1,
      33,
      40,
      82,
      116,
      161,
      214,
      19,
      20,
      122,
      181,
      9,
      32,
      37,
      60,
      61,
      69,
      73,
      99,
      108,
      109,
      215,
      227,
      228
    ]
  },
  "aggregate": {
    "best_layer_overlap_src_mean": 2,
    "best_overlap_src_mean": 0.08750000000000001,
    "best_layer_ld_src_mean": 2,
    "L11_overlap_src_mean": 0.016805555555555556,
    "L17_overlap_src_mean": 0.015416666666666667
  }
}

 results/mechanistic/stage3_cellhd_cls_fwd_dom_myriad/run_manifest.json
{
  "stage": "stage2b_continuation_curated",
  "direction": "forward",
  "site": "classifieds",
  "patch_config": {
    "source_mode": "som",
    "target_mode": "dom",
    "step_idx": 2,
    "max_new_tokens": 50,
    "n_layers_swept": 36,
    "hook_position": "last_token",
    "first_forward_only": true,
    "min_free_vram_gb": 0.0
  },
  "input_dataset": {
    "archived_run_dir": "/home/ucab352/Scratch/p79/results/mechanistic/archive_subset_b1_cls",
    "n_tasks_requested": 24,
    "n_tasks_completed": 24,
    "task_ids": [
      1,
      33,
      40,
      82,
      116,
      161,
      214,
      19,
      20,
      122,
      181,
      9,
      32,
      37,
      60,
      61,
      69,
      73,
      99,
      108,
      109,
      215,
      227,
      228
    ]
  },
  "aggregate": {
    "best_layer_overlap_src_mean": 10,
    "best_overlap_src_mean": 0.19166666666666668,
    "best_layer_ld_src_mean": 15,
    "L11_overlap_src_mean": 0.19166666666666665,
    "L17_overlap_src_mean": 0.16750000000000007
  }
}

exec
/bin/bash -lc "python3 - <<'PY'
import json
for p in ['results/mechanistic/stage2b_curated_b1_cls_myriad/env_snapshot.json','results/mechanistic/stage2b_celle_random_cls_strong_myriad/env_snapshot.json','results/mechanistic/stage3_cellhd_cls_fwd_dom_myriad/env_snapshot.json']:
    print('\\n',p)
    d=json.load(open(p))
    print(json.dumps(d, indent=2)[:2500])
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:

 results/mechanistic/stage2b_curated_b1_cls_myriad/env_snapshot.json
{
  "captured_at": "2026-05-09T06:31:21.667346+00:00",
  "host": "node-v00a-001.myriad.ucl.ac.uk",
  "platform": "Linux-3.10.0-1160.147.1.el7.x86_64-x86_64-with-glibc2.17",
  "python_version": "3.9.6",
  "torch": {
    "version": "2.1.0+cu121",
    "cuda": "12.1",
    "device_count": 1,
    "compute_caps": [
      [
        8,
        0
      ]
    ],
    "device_names": [
      "NVIDIA A100 80GB PCIe"
    ]
  },
  "libraries": {
    "transformers": "4.57.6",
    "qwen_vl_utils": "unknown",
    "huggingface_hub": "0.36.2",
    "numpy": "1.26.4",
    "scikit_learn": null
  },
  "models": {
    "Qwen/Qwen3-VL-4B-Instruct": "unavailable"
  },
  "evaluator_code": {
    "combined_sha256": "ba7a9276d59269be30bc8eb27b261d89125272f1c4ccaec8c35c11dbf2dc041a",
    "per_file_sha256": {
      "p79/experiment/analysis.py": "9d6559495b61977daf7e67bd3ace4044053cff24fdc92f5107133ee077a7441b",
      "p79/experiment/environment.py": "e9a12798677fd2333685505915cc6adc68786cbbc981e4b25e4ffda4bcbf2ed5",
      "p79/experiment/metrics.py": "b7361fe724ee4d7041b6f3127d69f225c58212fbc7cf31f05aafbfd780db2a78"
    },
    "files": [
      "p79/experiment/analysis.py",
      "p79/experiment/environment.py",
      "p79/experiment/metrics.py"
    ]
  },
  "git": {
    "commit": "9ea9e53b2123494d347a215a496a2acfd165ddee",
    "branch": "master",
    "dirty": false,
    "status": null
  },
  "nvidia_smi": [
    "NVIDIA A100 80GB PCIe, 550.127.05, 81920 MiB, 8.0"
  ],
  "extra": {
    "stage": "stage2b_curated",
    "reverse": false,
    "site": "classifieds"
  },
  "errors": [
    "hf:Qwen/Qwen3-VL-4B-Instruct: OfflineModeIsEnabled: Cannot reach https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct: offline mode is enabled. To disable it, please unset the `HF_HUB_OFFLINE` environment variable."
  ]
}

 results/mechanistic/stage2b_celle_random_cls_strong_myriad/env_snapshot.json
{
  "captured_at": "2026-05-09T08:31:47.039277+00:00",
  "host": "node-e00a-003.myriad.ucl.ac.uk",
  "platform": "Linux-3.10.0-1160.147.1.el7.x86_64-x86_64-with-glibc2.17",
  "python_version": "3.9.6",
  "torch": {
    "version": "2.1.0+cu121",
    "cuda": "12.1",
    "device_count": 1,
    "compute_caps": [
      [
        7,
        0
      ]
    ],
    "device_names": [
      "Tesla V100-PCIE-32GB"
    ]
  },
  "libraries": {
    "transformers": "4.57.6",
    "qwen_vl_utils": "unknown",
    "huggingface_hub": "0.36.2",
    "numpy": "1.26.4",
    "scikit_learn": null
  },
  "models": {
    "Qwen/Qwen3-VL-4B-Instruct": "unavailable"
  },
  "evaluator_code": {
    "combined_sha256": "ba7a9276d59269be30bc8eb27b261d89125272f1c4ccaec8c35c11dbf2dc041a",
    "per_file_sha256": {
      "p79/experiment/analysis.py": "9d6559495b61977daf7e67bd3ace4044053cff24fdc92f5107133ee077a7441b",
      "p79/experiment/environment.py": "e9a12798677fd2333685505915cc6adc68786cbbc981e4b25e4ffda4bcbf2ed5",
      "p79/experiment/metrics.py": "b7361fe724ee4d7041b6f3127d69f225c58212fbc7cf31f05aafbfd780db2a78"
    },
    "files": [
      "p79/experiment/analysis.py",
      "p79/experiment/environment.py",
      "p79/experiment/metrics.py"
    ]
  },
  "git": {
    "commit": "fbbbcd31e1df8d68081ad4037828814dd09a27f9",
    "branch": "master",
    "dirty": false,
    "status": null
  },
  "nvidia_smi": [
    "Tesla V100-PCIE-32GB, 550.127.05, 32768 MiB, 7.0"
  ],
  "extra": {
    "stage": "stage2b_curated",
    "reverse": false,
    "site": "classifieds"
  },
  "errors": [
    "hf:Qwen/Qwen3-VL-4B-Instruct: OfflineModeIsEnabled: Cannot reach https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct: offline mode is enabled. To disable it, please unset the `HF_HUB_OFFLINE` environment variable."
  ]
}

 results/mechanistic/stage3_cellhd_cls_fwd_dom_myriad/env_snapshot.json
{
  "captured_at": "2026-05-10T19:55:12.771393+00:00",
  "host": "node-e00a-013.myriad.ucl.ac.uk",
  "platform": "Linux-3.10.0-1160.147.1.el7.x86_64-x86_64-with-glibc2.17",
  "python_version": "3.9.6",
  "torch": {
    "version": "2.1.0+cu121",
    "cuda": "12.1",
    "device_count": 1,
    "compute_caps": [
      [
        7,
        0
      ]
    ],
    "device_names": [
      "Tesla V100-PCIE-32GB"
    ]
  },
  "libraries": {
    "transformers": "4.57.6",
    "qwen_vl_utils": "unknown",
    "huggingface_hub": "0.36.2",
    "numpy": "1.26.4",
    "scikit_learn": null
  },
  "models": {
    "Qwen/Qwen3-VL-4B-Instruct": "unavailable"
  },
  "evaluator_code": {
    "combined_sha256": "ba7a9276d59269be30bc8eb27b261d89125272f1c4ccaec8c35c11dbf2dc041a",
    "per_file_sha256": {
      "p79/experiment/analysis.py": "9d6559495b61977daf7e67bd3ace4044053cff24fdc92f5107133ee077a7441b",
      "p79/experiment/environment.py": "e9a12798677fd2333685505915cc6adc68786cbbc981e4b25e4ffda4bcbf2ed5",
      "p79/experiment/metrics.py": "b7361fe724ee4d7041b6f3127d69f225c58212fbc7cf31f05aafbfd780db2a78"
    },
    "files": [
      "p79/experiment/analysis.py",
      "p79/experiment/environment.py",
      "p79/experiment/metrics.py"
    ]
  },
  "git": {
    "commit": "46653aac65ee30595e0295adfeff0e2452c13e4f",
    "branch": "master",
    "dirty": true,
    "status": " M scripts/mechanistic/run_stage2b_continuation_pilot.py\n?? scripts/queues/qsub_stage2_cellcr_reddit_fwd_revtier.sh\n?? scripts/queues/qsub_stage2_celldr_reddit_rev_strongtier.sh\n?? scripts/queues/qsub_stage2_celler_reddit_fwd_random.sh\n?? scripts/queues/qsub_stage3_cellhd_cls_fwd_dom.sh\n?? scripts/queues/qsub_stage3_cellhp_cls_fwd_prompt.sh\n?? scripts/queues/qsub_stage3_cellhp_red_fwd_prompt.sh\n?? scripts/queues/qsub_stage3_cellht_cls_fwd_text.sh\n?? scripts/queues/qsub_stage3_cellht_red_fwd_text.sh\n"
  },
  "nvidia_smi": [
    "Tesla V100-PCIE-32GB, 550.127.05, 32768 MiB, 7.0"
  ],
  "extra": {
    "stage": "stage2b_curated",
    "site": "classifieds",
    "reverse": false,
    "tier": "strong",
    "random_inject": false,
    "random_seed": 42,
    "n_tasks_requested": 24,
    "step": 2,
    "max_new_tokens": 50,
    "source_mode": "som",
    "target_mode": "dom"
  },
  "errors": [
    "hf:Qwen/Qwen3-VL-4B-Instruct: OfflineModeIsEnabled: Cannot reach https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct: offline mode is enabled. To disable it, please unset the `HF_HUB_OFFLINE` environment variable."


exec
/bin/bash -lc 'rg -n "stage4_multimode|stage2b_curated|stage3_cellhprompt|stage4_pca_cosine_gap|axis2_layer_profile|logit_lens|layer_significance|run_stage2b_continuation" scripts/queues docs/checkpoints/mechanism/plan.md docs/checkpoints/mechanism/results scripts/README.md scripts/analysis/README.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/mechanism/plan.md:87:`scripts/analysis/stage4_pca_cosine_gap.py` + `stage4_robustness.py`. Three metrics per (mode_pair, layer):
docs/checkpoints/mechanism/plan.md:312:| **P5b** | reddit Method 4.2 multimode (cross-site Mirage) | Myriad 353890 | ✅ **done 07:31:14** — 288 examples, 6 modes, 51 MB pulled | `stage4_multimode_b1_reddit/hidden_states.npz` |
docs/checkpoints/mechanism/plan.md:338:`axis2_layer_profile.md` + `fig_axis2_prompt_layer_profile.png`. Re-examine residual stream geometry per axis-isolated pair, full 37-layer cosine curves on `stage4_multimode_b1_{cls,reddit}` (288 ex each).
docs/checkpoints/mechanism/plan.md:384:`axis2_logit_lens.md` + `fig_axis2_logit_lens.png`. 应用 Qwen3-VL-4B `model.model.language_model.norm` + `model.lm_head` to per-layer per-mode mean hidden states, 算 KL across 37 层.
scripts/queues/qsub_stage3_cellhp_red_fwd_prompt.sh:51:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
scripts/queues/qsub_stage2_cellcr_reddit_fwd_revtier.sh:52:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
scripts/queues/qsub_stage3_cellhprompt_cls_rand.sh:5:# Companion to qsub_stage3_cellhprompt_cls.sh (real-source axis-2 patching).
scripts/queues/qsub_stage3_cellhprompt_cls_rand.sh:57:OUT_DIR="$REPO_DIR/results/mechanistic/stage3_cellhprompt_cls_fwd_ptext_rand_myriad"
scripts/queues/qsub_stage3_cellhprompt_cls_rand.sh:63:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
docs/checkpoints/mechanism/results/axis2_layer_profile.md:10:## Results — classifieds site (stage4_multimode_b1_cls, 288 ex)
docs/checkpoints/mechanism/results/axis2_layer_profile.md:20:## Results — reddit site (stage4_multimode_b1_reddit, 288 ex)
scripts/queues/qsub_stage3_cellht_red_fwd_text.sh:51:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
scripts/queues/qsub_stage2_celldr_reddit_rev_strongtier.sh:52:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
scripts/queues/qsub_stage2b_myriad.sh:23:#   results/mechanistic/stage2b_curated_b1_cls_myriad/
scripts/queues/qsub_stage2b_myriad.sh:114:OUT_DIR="$REPO_DIR/results/mechanistic/stage2b_curated_b1_cls_myriad"
scripts/queues/qsub_stage2b_myriad.sh:119:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:115:  bash scripts/maintenance/auto_pull_myriad_cell.sh 359511 cellhprm_cls stage3_cellhprompt_cls_fwd_ptext_myriad
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md:116:  bash scripts/maintenance/auto_pull_myriad_cell.sh 359512 cellhprm_red stage3_cellhprompt_red_fwd_ptext_myriad
scripts/queues/qsub_stage4_multimode_extract_cls.sh:4:# Output → results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz
scripts/queues/qsub_stage4_multimode_extract_cls.sh:40:OUT_DIR="$REPO_DIR/results/mechanistic/stage4_multimode_b1_cls"
scripts/queues/qsub_stage4_multimode_extract_cls.sh:46:python3 scripts/mechanistic/run_stage4_multimode_extract.py \
scripts/queues/qsub_stage3_cellhd_cls_fwd_dom.sh:56:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
scripts/queues/qsub_stage3_cellhd_red_fwd_dom.sh:53:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
scripts/queues/qsub_stage3_cellhprompt_red_rand.sh:4:# Cross-site replication of qsub_stage3_cellhprompt_cls_rand.sh.
scripts/queues/qsub_stage3_cellhprompt_red_rand.sh:47:OUT_DIR="$REPO_DIR/results/mechanistic/stage3_cellhprompt_red_fwd_ptext_rand_myriad"
scripts/queues/qsub_stage3_cellhprompt_red_rand.sh:53:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
scripts/queues/qsub_stage2c_myriad.sh:98:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
scripts/queues/qsub_stage3_cellhprompt_red.sh:47:OUT_DIR="$REPO_DIR/results/mechanistic/stage3_cellhprompt_red_fwd_ptext_myriad"
scripts/queues/qsub_stage3_cellhprompt_red.sh:53:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
scripts/queues/qsub_celle_random_cls_strong_myriad.sh:55:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
scripts/queues/qsub_stage2_celler_reddit_fwd_random.sh:53:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
scripts/queues/qsub_2x2_fwd_revtasks_myriad.sh:62:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
scripts/queues/qsub_2x2_rev_strongtasks_myriad.sh:59:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
scripts/queues/qsub_stage3_cellhp_cls_fwd_prompt.sh:54:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
scripts/queues/qsub_cellg_rev_reddit_reverse_myriad.sh:42:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
scripts/queues/qsub_cellf_fwd_reddit_strong_myriad.sh:42:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
scripts/queues/qsub_stage3_cellht_cls_fwd_text.sh:54:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
scripts/queues/qsub_stage4mm_reddit.sh:3:# Output → results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz
scripts/queues/qsub_stage4mm_reddit.sh:30:OUT_DIR="$REPO_DIR/results/mechanistic/stage4_multimode_b1_reddit"
scripts/queues/qsub_stage4mm_reddit.sh:34:python3 scripts/mechanistic/run_stage4_multimode_extract.py \
scripts/queues/qsub_stage3_cellhprompt_cls.sh:50:OUT_DIR="$REPO_DIR/results/mechanistic/stage3_cellhprompt_cls_fwd_ptext_myriad"
scripts/queues/qsub_stage3_cellhprompt_cls.sh:56:python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \

exec
/bin/bash -lc 'rg -n "random_seed|seed|manual_seed|np.random|default_rng|deterministic|do_sample|model_revision|revision|HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE" scripts/mechanistic scripts/analysis p79/mechanistic' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
p79/mechanistic/activation_patching.py:114:                do_sample=False,
p79/mechanistic/activation_patching.py:170:                do_sample=False,
p79/mechanistic/activation_patching.py:386:        **source_inputs, max_new_tokens=max_new_tokens, do_sample=False,
p79/mechanistic/activation_patching.py:395:        **target_inputs, max_new_tokens=max_new_tokens, do_sample=False,
scripts/mechanistic/run_stage1_pilot.py:37:- Single seed (42). Cross-seed stability = Stage 1D.
scripts/mechanistic/run_stage1_pilot.py:224:    parser.add_argument("--seed", type=int, default=42)
scripts/mechanistic/run_stage1_pilot.py:314:        f"Running per-layer linear probe ({args.n_folds}-fold CV, seed={args.seed}, "
scripts/mechanistic/run_stage1_pilot.py:319:        n_folds=args.n_folds, seed=args.seed,
scripts/mechanistic/run_stage1_pilot.py:324:    probe_results["seed"] = args.seed
scripts/mechanistic/run_stage1_pilot.py:358:- CV: {args.n_folds}-fold StratifiedKFold, seed {args.seed}
p79/mechanistic/linear_probe.py:26:    seed: int = 42,
p79/mechanistic/linear_probe.py:42:        seed: RNG seed for fold split
p79/mechanistic/linear_probe.py:70:    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
p79/mechanistic/linear_probe.py:82:                pca = PCA(n_components=effective_pca_dim, random_state=seed)
p79/mechanistic/linear_probe.py:87:                C=C, max_iter=max_iter, solver="lbfgs", random_state=seed,
scripts/analysis/analyze_cross_representation.py:1515:                # No cost, no steps — pick first deterministically.
scripts/mechanistic/run_stage4_method44_v2_sweep.py:139:            dom_gen = patcher.model.generate(**dom_inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
scripts/mechanistic/run_stage4_method44_v2_sweep.py:144:            psom_gen = patcher.model.generate(**psom_inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
p79/mechanistic/extract_hidden_states.py:40:        # Paper-grade: pin HF revision SHA — DGX baseline lock 2026-05-07 (笔记 §114)
p79/mechanistic/extract_hidden_states.py:41:        model_revision: str = "ebb281ec70b05090aa6165b016eac8ec08e71b17",
p79/mechanistic/extract_hidden_states.py:56:        logger.info(f"Loading {model_path} (revision={model_revision[:12]}...) for hidden state extraction (dtype={dtype})")
p79/mechanistic/extract_hidden_states.py:59:            revision=model_revision,
p79/mechanistic/extract_hidden_states.py:66:            model_path, revision=model_revision, trust_remote_code=True
p79/mechanistic/extract_hidden_states.py:69:        self.model_revision = model_revision
scripts/mechanistic/diag_stage4_method44_layer_check.py:101:            out = patcher.model.generate(**dom_inputs, max_new_tokens=max_tok, do_sample=False,
scripts/mechanistic/diag_stage4_method44_layer_check.py:127:    rng = np.random.default_rng(20260511)
scripts/analysis/aggregate_phantom_lift.py:124:def bootstrap_lift_ci(in_3: np.ndarray, in_5: np.ndarray, B: int = 1000, seed: int = 42
scripts/analysis/aggregate_phantom_lift.py:128:    rng = np.random.default_rng(seed)
scripts/analysis/aggregate_phantom_lift.py:203:                              B: int = 1000, seed: int = 42, ci: float = 0.95
scripts/analysis/aggregate_phantom_lift.py:220:    rng = np.random.default_rng(seed)
scripts/analysis/aggregate_phantom_lift.py:230:                                  delta_pp: float = 1.0, B: int = 1000, seed: int = 42
scripts/analysis/aggregate_phantom_lift.py:266:    rng = np.random.default_rng(seed)
scripts/analysis/aggregate_phantom_lift.py:282:                                   B: int = 1000, seed: int = 42,
scripts/analysis/aggregate_phantom_lift.py:303:    rng = np.random.default_rng(seed)
scripts/mechanistic/curate_mirage_tasks.py:130:            do_sample=False,
scripts/mechanistic/run_stage4_method44_steering.py:145:                **dom_inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
scripts/mechanistic/run_stage4_method44_steering.py:152:                **psom_inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
scripts/analysis/stage4_logit_lens_axis2.py:57:    os.environ.setdefault("HF_HUB_OFFLINE", "1")
scripts/analysis/stage4_logit_lens_axis2.py:58:    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
scripts/mechanistic/run_stage2b_continuation_pilot.py:183:        "--random-seed", type=int, default=42,
scripts/mechanistic/run_stage2b_continuation_pilot.py:185:             "Same seed + same input = same noise = byte-identical re-runs. Default 42.",
scripts/mechanistic/run_stage2b_continuation_pilot.py:189:    # C8 fix: seed all RNGs when random-inject is on, for paper-grade
scripts/mechanistic/run_stage2b_continuation_pilot.py:191:    # Default seed=42 means re-running with same data + code produces
scripts/mechanistic/run_stage2b_continuation_pilot.py:197:        _rnd.seed(args.random_seed)
scripts/mechanistic/run_stage2b_continuation_pilot.py:198:        _np.random.seed(args.random_seed)
scripts/mechanistic/run_stage2b_continuation_pilot.py:199:        _t.manual_seed(args.random_seed)
scripts/mechanistic/run_stage2b_continuation_pilot.py:201:            _t.cuda.manual_seed_all(args.random_seed)
scripts/mechanistic/run_stage2b_continuation_pilot.py:202:        # Defense-in-depth visibility (防忘): log prominently + this seed
scripts/mechanistic/run_stage2b_continuation_pilot.py:204:        print(f"\n{'=' * 60}\n[RANDOM-INJECT SEED] {args.random_seed} "
scripts/mechanistic/run_stage2b_continuation_pilot.py:207:        # Note: cell E (job 335404) is currently running with NO seed (commit
scripts/mechanistic/run_stage2b_continuation_pilot.py:209:        # grade re-runs will be byte-reproducible with --random-seed 42.
scripts/mechanistic/run_stage2b_continuation_pilot.py:227:                "random_seed": args.random_seed,
scripts/mechanistic/run_stage2b_continuation_pilot.py:311:        # random_seed in incremental JSON so downstream stage2 stat scripts
scripts/mechanistic/run_stage2b_continuation_pilot.py:337:                    "random_seed": args.random_seed,
scripts/mechanistic/run_stage2b_continuation_pilot.py:427:- max_new_tokens: {args.max_new_tokens} (greedy continuation, deterministic)
scripts/mechanistic/run_stage2b_continuation_pilot.py:428:- Random injection: {"YES, seed=" + str(args.random_seed) + " (paper-grade reproducible)" if args.random_inject else "NO (real source hidden injected)"}
scripts/analysis/stage4_axis2_per_task_fragility.py:237:            jitter = np.random.normal(0, 0.04, size=len(vals))
scripts/analysis/stage4_axis2_per_task_fragility.py:261:    np.random.seed(0)
scripts/analysis/analyze_confidence_calibration.py:1098:    n_boot: int = 2000, ci: float = 0.95, seed: int = 42,
scripts/analysis/analyze_confidence_calibration.py:1105:    rng = np.random.default_rng(seed)
scripts/analysis/analyze_confidence_calibration.py:1166:    *, loo: bool = True, n_bootstrap: int = 500, seed: int = 42,
scripts/analysis/analyze_confidence_calibration.py:1200:    rng = np.random.default_rng(seed)
scripts/analysis/hero_claim_bootstrap.py:10:The reviewer demands per-seed bootstrap 95% CI on the pairwise comparison
scripts/analysis/hero_claim_bootstrap.py:100:def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, B: int = 10000, seed: int = 42):
scripts/analysis/hero_claim_bootstrap.py:103:    rng = np.random.default_rng(seed)
scripts/analysis/hero_claim_bootstrap.py:119:                          B: int = 10000, seed: int = 42):
scripts/analysis/hero_claim_bootstrap.py:122:    rng = np.random.default_rng(seed)
scripts/analysis/hero_claim_bootstrap.py:145:    p.add_argument("--seed", type=int, default=42)
scripts/analysis/hero_claim_bootstrap.py:149:    np.random.seed(args.seed)
scripts/analysis/hero_claim_bootstrap.py:153:        f"Per-seed bootstrap 95% percentile CI on paired adjusted-SR diffs and drop-one oracle. "
scripts/analysis/hero_claim_bootstrap.py:154:        f"B={args.bootstraps}, seed={args.seed}. Tasks resampled with replacement at task level.",
scripts/analysis/hero_claim_bootstrap.py:200:            r = bootstrap_diff_ci(M[:, mi[a]], M[:, mi[b]], B=args.bootstraps, seed=args.seed)
scripts/analysis/hero_claim_bootstrap.py:222:                                           B=args.bootstraps, seed=args.seed + j)
scripts/analysis/axis_effect_size.py:238:def bootstrap_ci(diffs: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
scripts/analysis/axis_effect_size.py:239:    rng = random.Random(seed)
scripts/analysis/preregistration_decision_test.py:43:    python3 scripts/analysis/preregistration_decision_test.py --synthetic --seed 0
scripts/analysis/preregistration_decision_test.py:223:def generate_synthetic_cells(seed: int = 0, n_cells: int = 16, scenario: str = "h1_pass") -> list[dict]:
scripts/analysis/preregistration_decision_test.py:232:    rng = random.Random(seed)
scripts/analysis/preregistration_decision_test.py:292:    p.add_argument("--seed", type=int, default=0)
scripts/analysis/preregistration_decision_test.py:308:        cells = generate_synthetic_cells(seed=args.seed, scenario=args.scenario)
scripts/analysis/stage2_layer_significance.py:103:    rng = np.random.default_rng(seed=42)
scripts/analysis/validate_run.py:1503:    random.seed(42)
scripts/analysis/stage4_robustness.py:32:RNG = np.random.default_rng(seed=20260511)
scripts/analysis/figures/fig_mechanism_pilot.py:273:        rng = np.random.default_rng(42)
scripts/analysis/figures/fig_mechanism_pilot.py:304:             "bootstrap n=2000, seed=42  |  green=Holm-sig, red=NULL",
scripts/analysis/figures/fig0c_drop_one_oracle.py:164:    seed: int = 42,
scripts/analysis/figures/fig0c_drop_one_oracle.py:172:    rng = np.random.default_rng(seed)
scripts/analysis/stage2_heterogeneity_figure.py:70:    rng = np.random.default_rng(seed=42)  # reproducible jitter

exec
/bin/bash -lc "nl -ba scripts/queues/qsub_stage4_multimode_extract_cls.sh | sed -n '1,90p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/bin/bash -l
     2	# Stage 4 Method 4.2: extract hidden states for all 6 modes on cls strong-tier
     3	# tasks (24 tasks × step 2 + step 5 × 6 modes = 288 forward passes).
     4	# Output → results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz
     5	#
     6	# Used for PCA cosine gap analysis (paper §5 Zoom 4 mechanism evidence).
     7	
     8	#$ -l h_rt=12:0:0
     9	#$ -l mem=64G
    10	#$ -l gpu=1
    11	#$ -wd /home/ucab352/Scratch/p79
    12	#$ -N stage4mm_cls
    13	#$ -o /home/ucab352/Scratch/p79/logs/qsub_stage4mm_cls.$JOB_ID.out
    14	#$ -e /home/ucab352/Scratch/p79/logs/qsub_stage4mm_cls.$JOB_ID.err
    15	#$ -j n
    16	
    17	mkdir -p /home/ucab352/Scratch/p79/logs
    18	
    19	set -euo pipefail
    20	REPO_DIR="/home/ucab352/Scratch/p79"
    21	cd "$REPO_DIR"
    22	
    23	echo "[$(date '+%H:%M:%S')] Job $JOB_ID start on $(hostname)"
    24	nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    25	
    26	module unload gcc-libs python python3 2>/dev/null || true
    27	module load pytorch/2.1.0/gpu
    28	
    29	export PYTHONUSERBASE="$HOME/Scratch/python_user"
    30	export PYTHONPATH="$PYTHONUSERBASE/lib/python3.9/site-packages:${PYTHONPATH:-}"
    31	export HF_HUB_OFFLINE=1
    32	export TRANSFORMERS_OFFLINE=1
    33	export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    34	
    35	if [ ! -d "$REPO_DIR/results/mechanistic/archive_subset_b1_cls" ]; then
    36	  echo "FATAL: archive_subset_b1_cls missing"
    37	  exit 1
    38	fi
    39	
    40	OUT_DIR="$REPO_DIR/results/mechanistic/stage4_multimode_b1_cls"
    41	mkdir -p "$OUT_DIR"
    42	OUT_NPZ="$OUT_DIR/hidden_states.npz"
    43	
    44	echo "[$(date '+%H:%M:%S')] Stage 4 Method 4.2: 24 tasks × 2 steps × 6 modes = 288 forward passes"
    45	
    46	python3 scripts/mechanistic/run_stage4_multimode_extract.py \
    47	    --site classifieds \
    48	    --n-tasks 24 \
    49	    --steps 2 5 \
    50	    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_cls" \
    51	    --output "$OUT_NPZ" \
    52	    --modes dom phantom_text phantom_prompt phantom_som som vision
    53	
    54	# Add a sentinel for Phase 0 done-check
    55	touch "$OUT_DIR/pilot_summary.md"
    56	echo "Stage 4 multimode extraction complete" > "$OUT_DIR/pilot_summary.md"
    57	echo "Modes: 6 (dom / phantom_text / phantom_prompt / phantom_som / som / vision)" >> "$OUT_DIR/pilot_summary.md"
    58	echo "Tasks: 24 cls strong-tier × 2 steps = 48 (task, step) pairs × 6 modes = 288 examples" >> "$OUT_DIR/pilot_summary.md"
    59	ls -la "$OUT_DIR/" >> "$OUT_DIR/pilot_summary.md"
    60	
    61	echo "[$(date '+%H:%M:%S')] DONE"
    62	ls -la "$OUT_DIR/"

exec
/bin/bash -lc "nl -ba scripts/queues/qsub_stage4mm_reddit.sh | sed -n '1,80p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/bin/bash -l
     2	# P5b: Method 4.2 multimode on reddit strong-tier 24 tasks — cross-site Mirage signature.
     3	# Output → results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz
     4	
     5	#$ -l h_rt=12:0:0
     6	#$ -l mem=64G
     7	#$ -l gpu=1
     8	#$ -wd /home/ucab352/Scratch/p79
     9	#$ -N stage4mm_red
    10	#$ -o /home/ucab352/Scratch/p79/logs/qsub_stage4mm_red.$JOB_ID.out
    11	#$ -e /home/ucab352/Scratch/p79/logs/qsub_stage4mm_red.$JOB_ID.err
    12	#$ -j n
    13	
    14	mkdir -p /home/ucab352/Scratch/p79/logs
    15	set -euo pipefail
    16	REPO_DIR="/home/ucab352/Scratch/p79"
    17	cd "$REPO_DIR"
    18	
    19	echo "[$(date '+%H:%M:%S')] Job $JOB_ID start on $(hostname)"
    20	nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    21	
    22	module unload gcc-libs python python3 2>/dev/null || true
    23	module load pytorch/2.1.0/gpu
    24	
    25	export PYTHONUSERBASE="$HOME/Scratch/python_user"
    26	export PYTHONPATH="$PYTHONUSERBASE/lib/python3.9/site-packages:${PYTHONPATH:-}"
    27	export HF_HUB_OFFLINE=1
    28	export TRANSFORMERS_OFFLINE=1
    29	
    30	OUT_DIR="$REPO_DIR/results/mechanistic/stage4_multimode_b1_reddit"
    31	mkdir -p "$OUT_DIR"
    32	OUT_NPZ="$OUT_DIR/hidden_states.npz"
    33	
    34	python3 scripts/mechanistic/run_stage4_multimode_extract.py \
    35	    --site reddit \
    36	    --n-tasks 24 \
    37	    --steps 2 5 \
    38	    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_reddit" \
    39	    --output "$OUT_NPZ" \
    40	    --modes dom phantom_text phantom_prompt phantom_som som vision
    41	
    42	# Sentinel for Phase 0
    43	echo "Stage 4 multimode reddit extraction complete" > "$OUT_DIR/pilot_summary.md"
    44	echo "Modes: 6 (dom / phantom_text / phantom_prompt / phantom_som / som / vision)" >> "$OUT_DIR/pilot_summary.md"
    45	echo "Tasks: 24 reddit strong-tier × 2 steps × 6 modes = 288 examples" >> "$OUT_DIR/pilot_summary.md"
    46	ls -la "$OUT_DIR/" >> "$OUT_DIR/pilot_summary.md"
    47	
    48	echo "[$(date '+%H:%M:%S')] DONE"
    49	ls -la "$OUT_DIR/"

exec
/bin/bash -lc "nl -ba scripts/queues/qsub_stage2b_myriad.sh | sed -n '100,140p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   100	  echo "  Fix: ssh login node, then:"
   101	  echo "    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE"
   102	  echo "    python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-4B-Instruct', revision='$HF_REVISION')\""
   103	  exit 1
   104	fi
   105	echo "[$(date '+%H:%M:%S')] HF model snapshot OK: $HF_SNAPSHOT_DIR"
   106	
   107	n_strong=$(python3 -c "import json; print(len(json.load(open('$REPO_DIR/results/mechanistic/archive_subset_b1_cls/manifest.json'))['strong']))")
   108	echo "[$(date '+%H:%M:%S')] Dataset: $n_strong strong mirage candidates"
   109	
   110	# ============================================================================
   111	# Run Stage 2B forward direction (paper §5 mechanism scale-up)
   112	# ============================================================================
   113	
   114	OUT_DIR="$REPO_DIR/results/mechanistic/stage2b_curated_b1_cls_myriad"
   115	mkdir -p "$OUT_DIR"
   116	
   117	echo "[$(date '+%H:%M:%S')] Launching Stage 2B forward (24 task × all layers × 50 max_new_tokens)..."
   118	
   119	python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
   120	    --site classifieds \
   121	    --n-tasks 24 \
   122	    --step 2 \
   123	    --max-new-tokens 50 \
   124	    --source-mode som \
   125	    --target-mode phantom_som \
   126	    --output-dir "$OUT_DIR" \
   127	    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_cls"
   128	
   129	# ============================================================================
   130	# Done
   131	# ============================================================================
   132	
   133	echo "[$(date '+%H:%M:%S')] Stage 2B forward DONE"
   134	echo "[$(date '+%H:%M:%S')] Output: $OUT_DIR"
   135	ls -la "$OUT_DIR/"
   136	
   137	# Final GPU stats
   138	echo "[$(date '+%H:%M:%S')] Final GPU memory:"
   139	nvidia-smi --query-gpu=memory.used,memory.free --format=csv

exec
/bin/bash -lc "nl -ba scripts/queues/qsub_celle_random_cls_strong_myriad.sh | sed -n '45,75p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
    45	HF_SNAPSHOT_DIR="$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/$HF_REVISION"
    46	if [ ! -f "$HF_SNAPSHOT_DIR/config.json" ]; then
    47	  echo "FATAL: HF model snapshot missing at $HF_SNAPSHOT_DIR/config.json"
    48	  exit 1
    49	fi
    50	
    51	OUT_DIR="$REPO_DIR/results/mechanistic/stage2b_celle_random_cls_strong_myriad"
    52	mkdir -p "$OUT_DIR"
    53	
    54	echo "[$(date '+%H:%M:%S')] Launching FORWARD × strong-tier 24 task with RANDOM source hidden..."
    55	python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    56	    --site classifieds \
    57	    --n-tasks 24 \
    58	    --step 2 \
    59	    --max-new-tokens 50 \
    60	    --source-mode som \
    61	    --target-mode phantom_som \
    62	    --tier strong \
    63	    --random-inject \
    64	    --output-dir "$OUT_DIR" \
    65	    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_cls"
    66	
    67	echo "[$(date '+%H:%M:%S')] DONE → $OUT_DIR"
    68	ls -la "$OUT_DIR/"
    69	nvidia-smi --query-gpu=memory.used,memory.free --format=csv

exec
/bin/bash -lc "find results/mechanistic/archive_subset_b1_cls -path '*step_002/observation_dom.txt' | head -1 | xargs -r sed -n '1,20p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
Tab 0 (current): 404 Not Found | Tab 1: Classifieds

[2843] RootWebArea '404 Not Found' focused: True url: http://100.95.81.103:7770/24-pcs-lot-wild-animal-cupcake-wrappers-birthday-cake-party-decoration-supplies-lion-monkey-zebra-elephant.html%20%7CAND%7C%20http://100.95.81.103:9980
	[3107] link 'My Account' url: http://100.95.81.103:7770/customer/account/
	[3111] link 'My Wish List' url: http://100.95.81.103:7770/wishlist/
	[3121] link 'Sign In' url: http://100.95.81.103:7770/customer/account/login/
	[3143] StaticText 'Welcome to One Stop Market'
	[3150] link 'Create an Account' url: http://100.95.81.103:7770/customer/account/create/
	[3152] link 'Skip to Content' url: http://100.95.81.103:7770/24-pcs-lot-wild-animal-cupcake-wrappers-birthday-cake-party-decoration-supplies-lion-monkey-zebra-elephant.html%20%7CAND%7C%20http://100.95.81.103:9980#contentarea
	[3165] link 'store logo' url: http://100.95.81.103:7770/
		[3167] image 'one_stop_market_logo' url: http://100.95.81.103:7770/media/logo/websites/1/image_15__1.png
	[3174] link '\ue611 My Cart' url: http://100.95.81.103:7770/checkout/cart/
	[3285] StaticText 'Search'
	[3290] combobox '\ue615 Search' autocomplete: both hasPopup: listbox required: False expanded: False
	[3296] link 'Advanced Search' url: http://100.95.81.103:7770/catalogsearch/advanced/
	[3304] button 'Search' disabled: True
	[3318] tablist '' multiselectable: False orientation: horizontal
		[DROPDOWN OPTIONS] "My Account", "My Wish List", "Sign In", "Compare Products", "Create an Account"
		[3326] tabpanel ''
			[3330] menu '' orientation: vertical

exec
/bin/bash -lc "rg -n \"def _extract_text_marks|_extract_text_marks\" p79/experiment/som.py && nl -ba p79/experiment/som.py | sed -n '1,180p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
24:def _extract_text_marks(obs_text: str, max_marks: int = 200) -> List[Dict[str, Any]]:
205:    text_marks = _extract_text_marks(obs_text)
225:    # into obs_text (the full AXTree), but _extract_text_marks strips them because
     1	from __future__ import annotations
     2	
     3	import logging
     4	import os
     5	import re
     6	from dataclasses import dataclass
     7	from pathlib import Path
     8	from typing import Any, Dict, List, Optional
     9	
    10	from PIL import ImageDraw, ImageFont
    11	
    12	logger = logging.getLogger(__name__)
    13	
    14	
    15	@dataclass
    16	class SomResult:
    17	    som_text: str
    18	    marked_image_path: Optional[str]
    19	    marked_image: Optional[Any]  # PIL Image with bounding boxes drawn, None if unavailable
    20	    degraded_som: bool
    21	    mark_count: int
    22	
    23	
    24	def _extract_text_marks(obs_text: str, max_marks: int = 200) -> List[Dict[str, Any]]:
    25	    marks: List[Dict[str, Any]] = []
    26	    for line in (obs_text or "").splitlines():
    27	        m = re.search(r"\[(\d+)\]", line)
    28	        if not m:
    29	            continue
    30	        eid = int(m.group(1))
    31	        label = re.sub(r"\[(\d+)\]", "", line).strip()
    32	        marks.append({"id": eid, "label": label})
    33	        if len(marks) >= max_marks:
    34	            break
    35	    return marks
    36	
    37	
    38	def _collect_bbox_map(raw: Any, bbox_map: Dict[int, List[float]]) -> None:
    39	    if isinstance(raw, dict):
    40	        maybe_id = None
    41	        for id_key in ("id", "node_id", "nodeId", "element_id"):
    42	            if id_key in raw:
    43	                try:
    44	                    maybe_id = int(raw[id_key])
    45	                    break
    46	                except Exception:
    47	                    logger.debug("Failed to parse element ID from key %s=%r", id_key, raw[id_key])
    48	                    maybe_id = None
    49	
    50	        bbox = None
    51	        for bbox_key in ("bbox", "bounding_box", "bounds", "rect"):
    52	            if bbox_key in raw and isinstance(raw[bbox_key], (list, tuple)) and len(raw[bbox_key]) == 4:
    53	                bbox = [float(x) for x in raw[bbox_key]]
    54	                break
    55	
    56	        if maybe_id is not None and bbox is not None:
    57	            bbox_map[maybe_id] = bbox
    58	
    59	        for v in raw.values():
    60	            _collect_bbox_map(v, bbox_map)
    61	    elif isinstance(raw, list):
    62	        for v in raw:
    63	            _collect_bbox_map(v, bbox_map)
    64	
    65	
    66	_FONT_CACHE: Dict[int, Any] = {}
    67	
    68	_CANDIDATE_FONTS = [
    69	    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    70	    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    71	    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    72	    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    73	    "/System/Library/Fonts/Helvetica.ttc",
    74	    "/Windows/Fonts/arialbd.ttf",
    75	]
    76	
    77	
    78	def _get_font(size: int = 14) -> Any:
    79	    if size in _FONT_CACHE:
    80	        return _FONT_CACHE[size]
    81	    font = None
    82	    for path in _CANDIDATE_FONTS:
    83	        if os.path.exists(path):
    84	            try:
    85	                font = ImageFont.truetype(path, size)
    86	                break
    87	            except Exception:
    88	                continue
    89	    if font is None:
    90	        try:
    91	            font = ImageFont.load_default(size=size)  # Pillow >= 9.2.0
    92	        except TypeError:
    93	            font = ImageFont.load_default()
    94	    _FONT_CACHE[size] = font
    95	    return font
    96	
    97	
    98	def _draw_label(
    99	    draw: ImageDraw.ImageDraw,
   100	    x: float,
   101	    y: float,
   102	    text: str,
   103	    font: Any,
   104	    bg_color: str = "#00BCD4",
   105	    fg_color: str = "white",
   106	    pad: int = 2,
   107	) -> None:
   108	    """Draw a filled pill-shaped label with white text at (x, y)."""
   109	    try:
   110	        bb = font.getbbox(text)
   111	        tw, th = bb[2] - bb[0], bb[3] - bb[1]
   112	    except AttributeError:
   113	        tw, th = 8 * len(text), 12
   114	    label_h = th + 2 * pad
   115	    rx0 = int(x)
   116	    # Place label above the bbox; fall back to inside if near top edge
   117	    ry0 = int(y) - label_h - 1 if int(y) - label_h - 1 >= 0 else int(y)
   118	    rx1, ry1 = rx0 + tw + 2 * pad, ry0 + label_h
   119	    draw.rectangle([rx0, ry0, rx1, ry1], fill=bg_color)
   120	    draw.text((rx0 + pad, ry0 + pad), text, fill=fg_color, font=font)
   121	
   122	
   123	def _normalize_bbox(bbox: List[float], width: int, height: int) -> List[float]:
   124	    x1, y1, x2, y2 = bbox
   125	    # If normalized values in [0, 1], scale to pixels.
   126	    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
   127	        x1, x2 = x1 * width, x2 * width
   128	        y1, y2 = y1 * height, y2 * height
   129	    return [x1, y1, x2, y2]
   130	
   131	
   132	def prepare_observation_for_mode(
   133	    obs: Any,
   134	    mode: str,
   135	    artifact_dir: Path,
   136	    step_idx: int,
   137	) -> SomResult:
   138	    """Prepare observation representation for the given observation mode.
   139	
   140	    mode == "dom":            Full AXTree text, no image (consistent: prompt expects AXTree, gets it).
   141	    mode == "som":            SOM_MARKS compressed index + marked image (consistent SoM).
   142	    mode == "phantom_som":    SOM_MARKS index, NO image (P-SoM: image-mismatched — prompt
   143	                              promises screenshot but agent gets none).
   144	    mode == "phantom_dom"
   145	       or "phantom_text":     SOM_MARKS index, NO image, but DOM-prompt (P-text: text-mismatched —
   146	                              prompt expects AXTree, agent receives [SOM_MARKS]).
   147	                              phantom_dom is the legacy mode value (paper-grade run dirs use it);
   148	                              phantom_text is the current name. Both dispatch identically.
   149	    mode == "phantom_prompt": Full AXTree text, NO image, with SoM-prompt (P-prompt: text+image
   150	                              mismatched — symmetric counterpart of phantom_text across the
   151	                              prompt × text axes; isolates prompt swap effect on AXTree text).
   152	    mode == "vision":         Empty text, raw screenshot as image.
   153	    """
   154	    obs_text = getattr(obs, "text", "") or ""
   155	
   156	    if mode == "vision":
   157	        return SomResult(
   158	            som_text="",
   159	            marked_image_path=None,
   160	            marked_image=getattr(obs, "image", None),
   161	            degraded_som=False,
   162	            mark_count=0,
   163	        )
   164	
   165	    if mode in ("phantom_som", "phantom_dom", "phantom_text"):
   166	        # phantom_som: SoM prompt + [SOM_MARKS] text + no image (image-mismatched)
   167	        # phantom_dom / phantom_text: DOM prompt + [SOM_MARKS] text + no image (text-mismatched)
   168	        # Obs construction identical — only system prompt differs (handled in agent).
   169	        result = _build_som_result(obs, obs_text, artifact_dir, step_idx)
   170	        return SomResult(
   171	            som_text=result.som_text,
   172	            marked_image_path=result.marked_image_path,  # keep artifact for inspection
   173	            marked_image=None,                           # model receives no image
   174	            degraded_som=result.degraded_som,
   175	            mark_count=result.mark_count,
   176	        )
   177	
   178	    if mode == "phantom_prompt":
   179	        # P-prompt: AXTree text (same as DOM mode) + no image, but SoM prompt (set in agent).
   180	        # Symmetric ablation of phantom_text: only the prompt axis is swapped from DOM.

exec
/bin/bash -lc "nl -ba p79/experiment/som.py | sed -n '180,260p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   180	        # Symmetric ablation of phantom_text: only the prompt axis is swapped from DOM.
   181	        return SomResult(som_text=obs_text, marked_image_path=None, marked_image=None,
   182	                         degraded_som=False, mark_count=0)
   183	
   184	    if mode != "som":
   185	        # "dom" mode or any unknown mode — full AXTree, no image
   186	        return SomResult(som_text=obs_text, marked_image_path=None, marked_image=None, degraded_som=False, mark_count=0)
   187	
   188	    # --- "som" mode: SOM_MARKS compressed index + marked image ---
   189	    return _build_som_result(obs, obs_text, artifact_dir, step_idx)
   190	
   191	
   192	def _build_som_result(
   193	    obs: Any,
   194	    obs_text: str,
   195	    artifact_dir: Path,
   196	    step_idx: int,
   197	    include_full_axtree: bool = False,
   198	) -> SomResult:
   199	    """Core SOM logic: build SOM_MARKS index + marked image.
   200	
   201	    Args:
   202	        include_full_axtree: If True, appends the full AXTree after SOM_MARKS
   203	            (legacy behavior). If False (new SOM mode), only SOM_MARKS is returned.
   204	    """
   205	    text_marks = _extract_text_marks(obs_text)
   206	    # Only zero marks is a hard SOM fallback.
   207	    # A single mark can still be a valid interactive page and should not be
   208	    # forced into text-only degradation.
   209	    if len(text_marks) == 0:
   210	        # Do NOT leak the full AXTree into SOM mode — the model uses a SOM-specific
   211	        # system prompt and expects [SOM_MARKS] format. An empty block signals
   212	        # "no interactive elements detected" while keeping prompt/input consistent.
   213	        # Fall back to the raw (unmarked) screenshot so the model can still use vision.
   214	        return SomResult(
   215	            som_text="[SOM_MARKS]\n[/SOM_MARKS]",
   216	            marked_image_path=None,
   217	            marked_image=getattr(obs, "image", None),
   218	            degraded_som=True,
   219	            mark_count=0,
   220	        )
   221	
   222	    # Build options map: eid -> annotation line (e.g. '[OPTIONS] "a","b"' or
   223	    # '[OPTIONS: currently selected="x"] ...' or '[DROPDOWN OPTIONS] ...').
   224	    # _inject_select_options / _inject_css_dropdown_options already wrote these
   225	    # into obs_text (the full AXTree), but _extract_text_marks strips them because
   226	    # they have no [N] element id.  Re-scan obs_text to recover them.
   227	    _options_map: Dict[int, str] = {}
   228	    _obs_lines = (obs_text or "").splitlines()
   229	    for _i, _line in enumerate(_obs_lines):
   230	        _m = re.search(r"\[(\d+)\]", _line)
   231	        if not _m:
   232	            continue
   233	        _eid = int(_m.group(1))
   234	        for _j in range(_i + 1, min(_i + 3, len(_obs_lines))):
   235	            _stripped = _obs_lines[_j].strip()
   236	            if _stripped.startswith("[OPTIONS") or _stripped.startswith("[DROPDOWN OPTIONS"):
   237	                _options_map[_eid] = _stripped
   238	                break
   239	            if re.search(r"\[(\d+)\]", _obs_lines[_j]):
   240	                break  # next element reached
   241	
   242	    mark_lines = []
   243	    for _mark in text_marks:
   244	        _entry = f"[id={_mark['id']}] {_mark['label']}"
   245	        if _mark["id"] in _options_map:
   246	            _entry += f"\n    {_options_map[_mark['id']]}"
   247	        mark_lines.append(_entry)
   248	
   249	    som_header = "\n".join(["[SOM_MARKS]"] + mark_lines + ["[/SOM_MARKS]"])
   250	    som_text = f"{som_header}\n\n{obs_text}" if include_full_axtree else som_header
   251	
   252	    bbox_map: Dict[int, List[float]] = {}
   253	
   254	    # Prefer VWA's obs_nodes_info (populated from CDP via observation_metadata).
   255	    # Each entry: str(element_id) -> {"union_bound": [x, y, width, height], ...}
   256	    # Convert [x, y, w, h] → [x1, y1, x2, y2] for _normalize_bbox.
   257	    obs_nodes_info = getattr(obs, "obs_nodes_info", None)
   258	    if obs_nodes_info:
   259	        for node_id_str, node_info in obs_nodes_info.items():
   260	            try:

exec
/bin/bash -lc "python3 - <<'PY'
from pathlib import Path
import json
for base in ['results/mechanistic/archive_subset_b1_cls','results/mechanistic/archive_subset_b1_reddit']:
 p=Path(base)
 print('\\n',base, 'exists',p.exists())
 if p.exists():
  dirs=sorted([d.name for d in p.glob('*_task_*')])[:30]
  print('first dirs lex:', dirs)
  mf=p/'manifest.json'
  if mf.exists():
   m=json.load(open(mf))
   print('manifest keys', {k: len(v) for k,v in m.items() if isinstance(v,list)})
   for k,v in m.items():
    if isinstance(v,list) and v:
     print(k, [int(x['task_id']) for x in v[:10]])
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 exited 1 in 122ms:
exec
/bin/bash -lc "python3 - <<'PY'
from pathlib import Path
import json, numpy as np
for base in ['results/mechanistic/archive_subset_b1_cls','results/mechanistic/archive_subset_b1_reddit']:
 p=Path(base)
 print('\\n',base)
 mf=p/'manifest.json'
 if mf.exists():
  m=json.load(open(mf))
  print('manifest list lengths', {k: len(v) for k,v in m.items() if isinstance(v,list) and (not v or isinstance(v[0],dict))})
  for k,v in m.items():
   if isinstance(v,list) and v and isinstance(v[0],dict):
    print(k, [int(x['task_id']) for x in v[:10]])
  lex=[]
  for td in sorted(p.glob('*_task_*')):
   tid=int(td.name.rsplit('_',1)[1]); lex.append(tid)
  print('first 24 lex tids', lex[:24])
for npzp in ['results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz','results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz']:
 d=np.load(npzp, allow_pickle=True)
 tids=sorted(set(map(int,d['task_ids'])))
 print('\\n',npzp, len(tids), tids)
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 exited 1 in 0ms:
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
ModuleNotFoundError: No module named 'numpy'

exec
/bin/bash -lc "python3 - <<'PY'
from pathlib import Path
import json
for base in ['results/mechanistic/archive_subset_b1_cls','results/mechanistic/archive_subset_b1_reddit']:
 p=Path(base)
 print('\\n',base)
 mf=p/'manifest.json'
 if mf.exists():
  m=json.load(open(mf))
  print('manifest list lengths', {k: len(v) for k,v in m.items() if isinstance(v,list) and (not v or isinstance(v[0],dict))})
  manifest_sets={}
  for k,v in m.items():
   if isinstance(v,list) and v and isinstance(v[0],dict):
    tids=[int(x['task_id']) for x in v]
    manifest_sets[k]=set(tids)
    print(k, tids[:24])
  lex=[]
  for td in sorted(p.glob('*_task_*')):
   tid=int(td.name.rsplit('_',1)[1]); lex.append(tid)
  first=lex[:24]
  print('first 24 lex tids', first)
  for k,s in manifest_sets.items():
   print('overlap with', k, len(set(first)&s), 'extra', sorted(set(first)-s)[:20])
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:

 results/mechanistic/archive_subset_b1_cls
manifest list lengths {'strong': 24, 'reverse': 15, 'skipped': 0}
strong [1, 33, 40, 82, 116, 161, 214, 19, 20, 122, 181, 9, 32, 37, 60, 61, 69, 73, 99, 108, 109, 215, 227, 228]
reverse [211, 4, 152, 159, 10, 123, 130, 151, 155, 156, 157, 160, 188, 191, 200]
first 24 lex tids [1, 10, 108, 109, 116, 122, 123, 130, 151, 152, 155, 156, 157, 159, 160, 161, 181, 188, 19, 191, 20, 200, 211, 214]
overlap with strong 10 extra [10, 123, 130, 151, 152, 155, 156, 157, 159, 160, 188, 191, 200, 211]
overlap with reverse 14 extra [1, 19, 20, 108, 109, 116, 122, 161, 181, 214]

 results/mechanistic/archive_subset_b1_reddit
manifest list lengths {'strong': 47, 'reverse': 48, 'skipped': 0}
strong [93, 116, 79, 98, 99, 124, 197, 78, 123, 151, 42, 65, 125, 2, 4, 18, 32, 115, 147, 156, 180, 194, 207, 35]
reverse [43, 185, 186, 106, 172, 103, 181, 199, 62, 158, 165, 208, 26, 41, 47, 56, 63, 67, 135, 142, 164, 195, 81, 111]
first 24 lex tids [100, 103, 106, 107, 111, 112, 113, 115, 116, 117, 119, 122, 123, 124, 125, 132, 133, 135, 136, 137, 142, 145, 147, 148]
overlap with strong 10 extra [103, 106, 107, 111, 112, 117, 119, 122, 132, 135, 136, 142, 145, 148]
overlap with reverse 14 extra [100, 113, 115, 116, 123, 124, 125, 133, 137, 147]

exec
/bin/bash -lc "python3 - <<'PY'
import zipfile, ast, struct
from pathlib import Path

def read_npy_from_npz(npz, name):
    with zipfile.ZipFile(npz) as z:
        b=z.read(name+'.npy')
    if not b.startswith(b'\\x93NUMPY'):
        raise ValueError('not npy')
    major=b[6]
    if major == 1:
        hlen=struct.unpack('<H', b[8:10])[0]; off=10
    else:
        hlen=struct.unpack('<I', b[8:12])[0]; off=12
    header=ast.literal_eval(b[off:off+hlen].decode('latin1'))
    data=b[off+hlen:]
    return header, data

def read_int64_array(npz, name):
    h,data=read_npy_from_npz(npz,name)
    assert h['descr']=='<i8', h
    n=1
    for s in h['shape']: n*=s
    return list(struct.unpack('<'+'q'*n, data[:8*n])), h

for npzp in ['results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz','results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz']:
    vals,h=read_int64_array(npzp,'task_ids')
    print('\\n',npzp, h)
    print('unique tids', sorted(set(vals)), 'n_unique', len(set(vals)))
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:

 results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz {'descr': '<i8', 'fortran_order': False, 'shape': (288,)}
unique tids [1, 10, 19, 20, 32, 33, 108, 109, 116, 122, 123, 130, 155, 156, 157, 160, 161, 181, 191, 200, 211, 214, 227, 228] n_unique 24

 results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz {'descr': '<i8', 'fortran_order': False, 'shape': (288,)}
unique tids [100, 106, 107, 111, 112, 113, 115, 116, 117, 119, 122, 123, 132, 133, 135, 136, 137, 142, 145, 147, 148, 154, 158, 162] n_unique 24

exec
/bin/bash -lc "python3 - <<'PY'
import json, zipfile, ast, struct
from pathlib import Path

def read_i8(npz,name):
 b=zipfile.ZipFile(npz).read(name+'.npy'); major=b[6]
 off=10 if major==1 else 12; hlen=struct.unpack('<H' if major==1 else '<I', b[8:off])[0]
 h=ast.literal_eval(b[off:off+hlen].decode('latin1')); data=b[off+hlen:]
 n=1
 for s in h['shape']: n*=s
 return set(struct.unpack('<'+'q'*n, data[:8*n]))
for base,npzp in [('results/mechanistic/archive_subset_b1_cls','results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz'),('results/mechanistic/archive_subset_b1_reddit','results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz')]:
 m=json.load(open(Path(base)/'manifest.json'))
 strong={int(x['task_id']) for x in m['strong']}; rev={int(x['task_id']) for x in m['reverse']}
 tids=read_i8(npzp,'task_ids')
 print('\\n',npzp)
 print('tids', sorted(tids))
 print('strong overlap', len(tids&strong), sorted(tids&strong))
 print('reverse overlap', len(tids&rev), sorted(tids&rev))
 print('neither', sorted(tids-strong-rev))
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:

 results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz
tids [1, 10, 19, 20, 32, 33, 108, 109, 116, 122, 123, 130, 155, 156, 157, 160, 161, 181, 191, 200, 211, 214, 227, 228]
strong overlap 14 [1, 19, 20, 32, 33, 108, 109, 116, 122, 161, 181, 214, 227, 228]
reverse overlap 10 [10, 123, 130, 155, 156, 157, 160, 191, 200, 211]
neither []

 results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz
tids [100, 106, 107, 111, 112, 113, 115, 116, 117, 119, 122, 123, 132, 133, 135, 136, 137, 142, 145, 147, 148, 154, 158, 162]
strong overlap 10 [100, 113, 115, 116, 123, 133, 137, 147, 154, 162]
reverse overlap 14 [106, 107, 111, 112, 117, 119, 122, 132, 135, 136, 142, 145, 148, 158]
neither []

codex
The biggest implementation-vs-prose mismatch is now clear: Stage 4 is not actually selecting the declared strong-tier manifest; it lexicographically samples task directories from archives that contain both strong and reverse tasks. I also found Stage 4’s SoM text reconstruction drops most labels, which means the PCA geometry is not on the same `[SOM_MARKS]` payload used by Stage 2 or production.
exec
/bin/bash -lc "nl -ba scripts/analysis/stage4_robustness.py | sed -n '1,260p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
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

codex
### 1. Verdict (one sentence)
Methodology **not sound enough for paper-1 submission as written**, 最关键原因是 Stage 4 几何证据的输入数据不是 prose 声称的 strong-tier + production `[SOM_MARKS]` protocol，而是 lexicographic mixed-tier task sample 且 SoM text 被 lossy 重构，直接威胁 Method 4.2 / three-axis hierarchy 的主证据链。

### 2. 🟢 What's right (don't break under refactor)
- Stage 2/3 patching primitive 的 causal intervention 定义清楚：cache source hidden states，然后只在 target first forward 的 last input token 位置替换 hidden state，后续 generation 走 KV cache；这和 prose 的“final input-token position / subsequent decoding proceeds normally”匹配。见 `p79/mechanistic/activation_patching.py:83-89`, `p79/mechanistic/activation_patching.py:98-117`, `docs/checkpoints/paper_drafts/section5_mechanism.md:57`.

- Patching continuations 使用 deterministic greedy decoding，避免 sampling noise 混进 layer profile。`do_sample=False` 同时用于 source baseline、target baseline 和 patched generation。见 `p79/mechanistic/activation_patching.py:384-400`, `p79/mechanistic/activation_patching.py:432-439`.

- Stage 2B 当前脚本已把 source/target mode、reverse、tier、random control、max_new_tokens、model path 写入结果 JSON config；这是正确的 provenance 方向。见 `scripts/mechanistic/run_stage2b_continuation_pilot.py:325-343`.

- `HiddenStateExtractor` pin 了 Qwen3-VL revision，并从 `Qwen3VLAgent` 读取 prompt builder，原则上比手抄 prompt 更稳。见 `p79/mechanistic/extract_hidden_states.py:40-66`, `p79/mechanistic/extract_hidden_states.py:71-83`.

### 3. 🔴 Methodology bugs (caught before submission)
**Bug 1 — Stage 4 task set is not the claimed strong-tier set.**  
**What** — Stage 4 extractor ignores `manifest.json` tiers and selects the first task directories after `sorted(archive_dir.glob(...))`; no `--tier` argument exists. See `scripts/mechanistic/run_stage4_multimode_extract.py:86-96`. The qsub files describe “strong-tier 24 tasks” and pass the full archive dir, not a tier-filtered list: `scripts/queues/qsub_stage4_multimode_extract_cls.sh:1-4`, `scripts/queues/qsub_stage4_multimode_extract_cls.sh:46-52`, `scripts/queues/qsub_stage4mm_reddit.sh:1-3`, `scripts/queues/qsub_stage4mm_reddit.sh:34-40`. The plan says the archive contains separate strong and reverse tiers, not only strong-tier tasks: `docs/checkpoints/mechanism/plan.md:74-83`.  
**Why it matters** — Method 4.2, Axis-2 profile, logit lens, and layer-axis emergence all inherit `hidden_states.npz`; the paper describes them as “24 strong-tier tasks” (`docs/checkpoints/paper_drafts/section5_mechanism.md:28`, `docs/checkpoints/mechanism/results/axis2_layer_profile.md:10`, `docs/checkpoints/mechanism/results/axis2_layer_profile.md:20`). If the task set is mixed strong/reverse, then “strong-tier Mirage mechanism” and selection-bias defenses are not what the code ran. Bias direction is uncontrolled.  
**Fix** — Add `--tier {strong,reverse,all}` to `run_stage4_multimode_extract.py`; load task IDs from `manifest.json` in manifest order; fail if selected IDs differ from requested tier. Re-extract all Stage 4 NPZs and regenerate Method 4.2, Exp 1, Exp 3, layer-axis emergence.  
**Severity** — high.

**Bug 2 — Stage 4 `[SOM_MARKS]` text is not production/Stage 2B `[SOM_MARKS]`.**  
**What** — Stage 4 claims its `build_som_marks` is a copy of Stage 2B logic, but it uses `re.compile(r"^\[\d+\]\s+\w+")` and `findall`, which keeps only the bracket id plus role token and drops labels/options. See `scripts/mechanistic/run_stage4_multimode_extract.py:46-55`. Stage 2B uses `_extract_text_marks` and emits `[id=...] {label}` lines: `scripts/mechanistic/run_stage2b_continuation_pilot.py:121-126`. Production SoM construction also preserves labels and option lines: `p79/experiment/som.py:242-250`.  
**Why it matters** — Method 4.2’s central geometry compares modes whose flat text payload is materially different from the agent and patching payload. This can manufacture or shrink text-format / prompt-family axes. Claims depending on “P-SoM as P-text sibling” and three-axis hierarchy are therefore not tied to the actual intervention protocol.  
**Fix** — Replace Stage 4 `build_som_marks` with the production `_build_som_result` path or the exact Stage 2B `_extract_text_marks` formatter including labels/options. Re-extract hidden states. Add a unit test that Stage 4 `som`, `phantom_som`, and `phantom_text` text payloads are byte-identical to Stage 2B/production for the same archived step.  
**Severity** — high.

**Bug 3 — Method 4.2 AUROC is in-sample mean-difference classification, not held-out linear readability.**  
**What** — `stage4_pca_cosine_gap.py` computes per-mode means on all examples, defines `direction = c1 - c2`, then scores those same examples for AUROC. See `scripts/analysis/stage4_pca_cosine_gap.py:69-86`. The robustness permutation repeats the same in-sample fitting/evaluation pattern: `scripts/analysis/stage4_robustness.py:39-48`, `scripts/analysis/stage4_robustness.py:53-64`.  
**Why it matters** — “AUROC 1.0” is not a cross-validated probe result and should not be described as held-out decodability. In high dimension with n=48/mode, fitting the direction on the evaluation examples inflates separability; the permutation baseline being far above chance is itself expected under this protocol. This weakens the linear-readable claim in `docs/checkpoints/paper_drafts/section5_mechanism.md:17` and `docs/checkpoints/paper_drafts/section5_mechanism.md:28-30`.  
**Fix** — Use leave-one-task-out or grouped k-fold by `task_id`; fit mean direction on train tasks only, score held-out task-step pairs. Report held-out AUROC, not in-sample AUROC; keep in-sample cosine gap as descriptive geometry.  
**Severity** — high for AUROC/decodability wording, medium for descriptive cosine-gap geometry.

**Bug 4 — Layer indices are being compared across incompatible conventions.**  
**What** — Patching hooks expose 36 transformer block outputs, where patching `L0` is block 0 output and not embedding: `p79/mechanistic/activation_patching.py:39-44`. Hidden-state extraction saves 37 layers, where index 0 is embedding and indices 1..36 are block outputs: `p79/mechanistic/extract_hidden_states.py:168-173`. The paper uses a unified L0-L36 convention: `docs/checkpoints/paper_drafts/section5_mechanism.md:7`, while the plan still says “L17 hidden state” as a singular planning site: `docs/checkpoints/mechanism/plan.md:125-144`.  
**Why it matters** — Stage 2 patching `L17` corresponds to hidden-extraction block output index 18 under the 37-layer convention, not Method 4.2 `L17`. Cross-method claims about “same L17 locus” are off by one unless explicitly converted.  
**Fix** — Rename patching layers `B0..B35` and extraction layers `E0, B0..B35`, or convert all reports to block index. Regenerate tables with both raw index and block index.  
**Severity** — medium-high.

**Bug 5 — Logit lens may use a different model revision than the hidden states.**  
**What** — Hidden extraction pins revision `ebb281ec...`: `p79/mechanistic/extract_hidden_states.py:40-66`. Logit lens loads tokenizer/model from `MODEL_PATH` without passing the same revision: `scripts/analysis/stage4_logit_lens_axis2.py:54-63`.  
**Why it matters** — Applying `norm + lm_head` from a different cached revision to hidden states from a pinned revision makes KL magnitudes non-reproducible and potentially invalid. This affects Exp 3 claims in `docs/checkpoints/mechanism/results/axis2_logit_lens.md:3-6`.  
**Fix** — Add `MODEL_REVISION` constant or read revision metadata from the NPZ manifest; pass `revision=...` to both `AutoTokenizer` and `Qwen3VLForConditionalGeneration`.  
**Severity** — medium.

**Bug 6 — Layer significance baseline is a treated final-layer patch, not an unpatched null.**  
**What** — The significance test defines H0 by comparing layer `L_n` to `L35`: `scripts/analysis/stage2_layer_significance.py:8-15`, `scripts/analysis/stage2_layer_significance.py:43-44`, `scripts/analysis/stage2_layer_significance.py:95-101`. But `L35` is still a source-hidden replacement intervention, not the unpatched target continuation.  
**Why it matters** — Holm-significant “disruption vs L35” tests relative layer specificity, not whether patching disrupts target versus a genuine no-intervention baseline. It should not be used alone to claim causal displacement.  
**Fix** — Test each layer against the unpatched target metric (`token_overlap_to_target=1`, `ld_to_target=0`) and separately test mid-layer specificity against a pre-registered late-layer reference.  
**Severity** — medium.

### 4. 🟡 Methodology risks (unverified assumptions)
- Last-token-only patching may miss distributed causal sites over observation/image tokens. The intervention is explicitly last-token only: `p79/mechanistic/activation_patching.py:83-89`, `p79/mechanistic/activation_patching.py:102-105`. Verification: run position sweeps over image-token summary positions, observation span tokens, and generation prompt token; report whether L11-L17 survives.

- Gaussian random injection is a weak specificity control. It randomizes full source hidden tensors by per-layer scalar mean/std: `p79/mechanistic/activation_patching.py:415-423`. Verification: add task-shuffled source controls, same-task wrong-step controls, and same-mode different-task controls. Gaussian-null only rules out one OOD disruption class.

- Strong-tier curation is behaviorally selected on SoM/DOM success and may enrich for large internal separations. The plan states this curation explicitly: `docs/checkpoints/mechanism/plan.md:72-83`. Verification: rerun Method 4.2 and patching on random same-site tasks plus reverse-tier tasks with the corrected manifest selection.

- W6 trigger attribution is post-hoc and small-N. The W6 script hardcodes six marks-like variants and two controls: `scripts/analysis/stage4_w6_l04_tokenization.py:27-47`, and the prose correctly calls held-out falsifiers pending: `docs/checkpoints/paper_drafts/section5_mechanism.md:94`. Verification: run `bare_N`, `bracket_no_int`, `hash_int`, and matched label-length variants before claiming a trigger rule.

### 5. 📝 Reproducibility audit
Third-party rerunnability is partial, not paper-grade.

Seeds: deterministic generation is fine through `do_sample=False` in patching (`p79/mechanistic/activation_patching.py:384-400`, `p79/mechanistic/activation_patching.py:432-439`). Random controls now seed Python/NumPy/Torch when `--random-inject` is enabled: `scripts/mechanistic/run_stage2b_continuation_pilot.py:189-206`. Bootstrap scripts use fixed RNGs: `scripts/analysis/hero_claim_bootstrap.py:100-115`, `scripts/analysis/stage2_layer_significance.py:103-134`.

Provenance: Stage 2/3 now emits `env_snapshot.json` and `run_manifest.json`: `scripts/mechanistic/run_stage2b_continuation_pilot.py:216-236`, `scripts/mechanistic/run_stage2b_continuation_pilot.py:463-515`. But several result JSONs were generated before the current config fields, so direct result provenance is incomplete for legacy cells.

Stage 4 provenance is weak. `run_stage4_multimode_extract.py` writes only arrays to NPZ, with no command, git SHA, model revision, archive path, tier, selected task IDs, or formatter hash: `scripts/mechanistic/run_stage4_multimode_extract.py:183-192`. Because all Method 4.2/Exp 1/Exp 3 analyses consume that NPZ, provenance should be embedded in the NPZ or a sidecar manifest.

Model revision traceability is inconsistent: extraction pins revision (`p79/mechanistic/extract_hidden_states.py:40-66`), but logit lens does not (`scripts/analysis/stage4_logit_lens_axis2.py:54-63`). Also `env_snapshot.json` can report HF model as unavailable under offline mode, which blocks independent model identity verification.

### 6. One single highest-impact fix
Fix `scripts/mechanistic/run_stage4_multimode_extract.py` first: replace lexicographic directory selection with manifest-tier selection at `scripts/mechanistic/run_stage4_multimode_extract.py:86-96`, and replace lossy `build_som_marks` at `scripts/mechanistic/run_stage4_multimode_extract.py:46-55` with the production `[SOM_MARKS]` formatter from `p79/experiment/som.py:242-250`; then re-extract Stage 4 NPZs and regenerate Method 4.2, Exp 1, Exp 3, and layer-axis results.
tokens used
153,956
### 1. Verdict (one sentence)
Methodology **not sound enough for paper-1 submission as written**, 最关键原因是 Stage 4 几何证据的输入数据不是 prose 声称的 strong-tier + production `[SOM_MARKS]` protocol，而是 lexicographic mixed-tier task sample 且 SoM text 被 lossy 重构，直接威胁 Method 4.2 / three-axis hierarchy 的主证据链。

### 2. 🟢 What's right (don't break under refactor)
- Stage 2/3 patching primitive 的 causal intervention 定义清楚：cache source hidden states，然后只在 target first forward 的 last input token 位置替换 hidden state，后续 generation 走 KV cache；这和 prose 的“final input-token position / subsequent decoding proceeds normally”匹配。见 `p79/mechanistic/activation_patching.py:83-89`, `p79/mechanistic/activation_patching.py:98-117`, `docs/checkpoints/paper_drafts/section5_mechanism.md:57`.

- Patching continuations 使用 deterministic greedy decoding，避免 sampling noise 混进 layer profile。`do_sample=False` 同时用于 source baseline、target baseline 和 patched generation。见 `p79/mechanistic/activation_patching.py:384-400`, `p79/mechanistic/activation_patching.py:432-439`.

- Stage 2B 当前脚本已把 source/target mode、reverse、tier、random control、max_new_tokens、model path 写入结果 JSON config；这是正确的 provenance 方向。见 `scripts/mechanistic/run_stage2b_continuation_pilot.py:325-343`.

- `HiddenStateExtractor` pin 了 Qwen3-VL revision，并从 `Qwen3VLAgent` 读取 prompt builder，原则上比手抄 prompt 更稳。见 `p79/mechanistic/extract_hidden_states.py:40-66`, `p79/mechanistic/extract_hidden_states.py:71-83`.

### 3. 🔴 Methodology bugs (caught before submission)
**Bug 1 — Stage 4 task set is not the claimed strong-tier set.**  
**What** — Stage 4 extractor ignores `manifest.json` tiers and selects the first task directories after `sorted(archive_dir.glob(...))`; no `--tier` argument exists. See `scripts/mechanistic/run_stage4_multimode_extract.py:86-96`. The qsub files describe “strong-tier 24 tasks” and pass the full archive dir, not a tier-filtered list: `scripts/queues/qsub_stage4_multimode_extract_cls.sh:1-4`, `scripts/queues/qsub_stage4_multimode_extract_cls.sh:46-52`, `scripts/queues/qsub_stage4mm_reddit.sh:1-3`, `scripts/queues/qsub_stage4mm_reddit.sh:34-40`. The plan says the archive contains separate strong and reverse tiers, not only strong-tier tasks: `docs/checkpoints/mechanism/plan.md:74-83`.  
**Why it matters** — Method 4.2, Axis-2 profile, logit lens, and layer-axis emergence all inherit `hidden_states.npz`; the paper describes them as “24 strong-tier tasks” (`docs/checkpoints/paper_drafts/section5_mechanism.md:28`, `docs/checkpoints/mechanism/results/axis2_layer_profile.md:10`, `docs/checkpoints/mechanism/results/axis2_layer_profile.md:20`). If the task set is mixed strong/reverse, then “strong-tier Mirage mechanism” and selection-bias defenses are not what the code ran. Bias direction is uncontrolled.  
**Fix** — Add `--tier {strong,reverse,all}` to `run_stage4_multimode_extract.py`; load task IDs from `manifest.json` in manifest order; fail if selected IDs differ from requested tier. Re-extract all Stage 4 NPZs and regenerate Method 4.2, Exp 1, Exp 3, layer-axis emergence.  
**Severity** — high.

**Bug 2 — Stage 4 `[SOM_MARKS]` text is not production/Stage 2B `[SOM_MARKS]`.**  
**What** — Stage 4 claims its `build_som_marks` is a copy of Stage 2B logic, but it uses `re.compile(r"^\[\d+\]\s+\w+")` and `findall`, which keeps only the bracket id plus role token and drops labels/options. See `scripts/mechanistic/run_stage4_multimode_extract.py:46-55`. Stage 2B uses `_extract_text_marks` and emits `[id=...] {label}` lines: `scripts/mechanistic/run_stage2b_continuation_pilot.py:121-126`. Production SoM construction also preserves labels and option lines: `p79/experiment/som.py:242-250`.  
**Why it matters** — Method 4.2’s central geometry compares modes whose flat text payload is materially different from the agent and patching payload. This can manufacture or shrink text-format / prompt-family axes. Claims depending on “P-SoM as P-text sibling” and three-axis hierarchy are therefore not tied to the actual intervention protocol.  
**Fix** — Replace Stage 4 `build_som_marks` with the production `_build_som_result` path or the exact Stage 2B `_extract_text_marks` formatter including labels/options. Re-extract hidden states. Add a unit test that Stage 4 `som`, `phantom_som`, and `phantom_text` text payloads are byte-identical to Stage 2B/production for the same archived step.  
**Severity** — high.

**Bug 3 — Method 4.2 AUROC is in-sample mean-difference classification, not held-out linear readability.**  
**What** — `stage4_pca_cosine_gap.py` computes per-mode means on all examples, defines `direction = c1 - c2`, then scores those same examples for AUROC. See `scripts/analysis/stage4_pca_cosine_gap.py:69-86`. The robustness permutation repeats the same in-sample fitting/evaluation pattern: `scripts/analysis/stage4_robustness.py:39-48`, `scripts/analysis/stage4_robustness.py:53-64`.  
**Why it matters** — “AUROC 1.0” is not a cross-validated probe result and should not be described as held-out decodability. In high dimension with n=48/mode, fitting the direction on the evaluation examples inflates separability; the permutation baseline being far above chance is itself expected under this protocol. This weakens the linear-readable claim in `docs/checkpoints/paper_drafts/section5_mechanism.md:17` and `docs/checkpoints/paper_drafts/section5_mechanism.md:28-30`.  
**Fix** — Use leave-one-task-out or grouped k-fold by `task_id`; fit mean direction on train tasks only, score held-out task-step pairs. Report held-out AUROC, not in-sample AUROC; keep in-sample cosine gap as descriptive geometry.  
**Severity** — high for AUROC/decodability wording, medium for descriptive cosine-gap geometry.

**Bug 4 — Layer indices are being compared across incompatible conventions.**  
**What** — Patching hooks expose 36 transformer block outputs, where patching `L0` is block 0 output and not embedding: `p79/mechanistic/activation_patching.py:39-44`. Hidden-state extraction saves 37 layers, where index 0 is embedding and indices 1..36 are block outputs: `p79/mechanistic/extract_hidden_states.py:168-173`. The paper uses a unified L0-L36 convention: `docs/checkpoints/paper_drafts/section5_mechanism.md:7`, while the plan still says “L17 hidden state” as a singular planning site: `docs/checkpoints/mechanism/plan.md:125-144`.  
**Why it matters** — Stage 2 patching `L17` corresponds to hidden-extraction block output index 18 under the 37-layer convention, not Method 4.2 `L17`. Cross-method claims about “same L17 locus” are off by one unless explicitly converted.  
**Fix** — Rename patching layers `B0..B35` and extraction layers `E0, B0..B35`, or convert all reports to block index. Regenerate tables with both raw index and block index.  
**Severity** — medium-high.

**Bug 5 — Logit lens may use a different model revision than the hidden states.**  
**What** — Hidden extraction pins revision `ebb281ec...`: `p79/mechanistic/extract_hidden_states.py:40-66`. Logit lens loads tokenizer/model from `MODEL_PATH` without passing the same revision: `scripts/analysis/stage4_logit_lens_axis2.py:54-63`.  
**Why it matters** — Applying `norm + lm_head` from a different cached revision to hidden states from a pinned revision makes KL magnitudes non-reproducible and potentially invalid. This affects Exp 3 claims in `docs/checkpoints/mechanism/results/axis2_logit_lens.md:3-6`.  
**Fix** — Add `MODEL_REVISION` constant or read revision metadata from the NPZ manifest; pass `revision=...` to both `AutoTokenizer` and `Qwen3VLForConditionalGeneration`.  
**Severity** — medium.

**Bug 6 — Layer significance baseline is a treated final-layer patch, not an unpatched null.**  
**What** — The significance test defines H0 by comparing layer `L_n` to `L35`: `scripts/analysis/stage2_layer_significance.py:8-15`, `scripts/analysis/stage2_layer_significance.py:43-44`, `scripts/analysis/stage2_layer_significance.py:95-101`. But `L35` is still a source-hidden replacement intervention, not the unpatched target continuation.  
**Why it matters** — Holm-significant “disruption vs L35” tests relative layer specificity, not whether patching disrupts target versus a genuine no-intervention baseline. It should not be used alone to claim causal displacement.  
**Fix** — Test each layer against the unpatched target metric (`token_overlap_to_target=1`, `ld_to_target=0`) and separately test mid-layer specificity against a pre-registered late-layer reference.  
**Severity** — medium.

### 4. 🟡 Methodology risks (unverified assumptions)
- Last-token-only patching may miss distributed causal sites over observation/image tokens. The intervention is explicitly last-token only: `p79/mechanistic/activation_patching.py:83-89`, `p79/mechanistic/activation_patching.py:102-105`. Verification: run position sweeps over image-token summary positions, observation span tokens, and generation prompt token; report whether L11-L17 survives.

- Gaussian random injection is a weak specificity control. It randomizes full source hidden tensors by per-layer scalar mean/std: `p79/mechanistic/activation_patching.py:415-423`. Verification: add task-shuffled source controls, same-task wrong-step controls, and same-mode different-task controls. Gaussian-null only rules out one OOD disruption class.

- Strong-tier curation is behaviorally selected on SoM/DOM success and may enrich for large internal separations. The plan states this curation explicitly: `docs/checkpoints/mechanism/plan.md:72-83`. Verification: rerun Method 4.2 and patching on random same-site tasks plus reverse-tier tasks with the corrected manifest selection.

- W6 trigger attribution is post-hoc and small-N. The W6 script hardcodes six marks-like variants and two controls: `scripts/analysis/stage4_w6_l04_tokenization.py:27-47`, and the prose correctly calls held-out falsifiers pending: `docs/checkpoints/paper_drafts/section5_mechanism.md:94`. Verification: run `bare_N`, `bracket_no_int`, `hash_int`, and matched label-length variants before claiming a trigger rule.

### 5. 📝 Reproducibility audit
Third-party rerunnability is partial, not paper-grade.

Seeds: deterministic generation is fine through `do_sample=False` in patching (`p79/mechanistic/activation_patching.py:384-400`, `p79/mechanistic/activation_patching.py:432-439`). Random controls now seed Python/NumPy/Torch when `--random-inject` is enabled: `scripts/mechanistic/run_stage2b_continuation_pilot.py:189-206`. Bootstrap scripts use fixed RNGs: `scripts/analysis/hero_claim_bootstrap.py:100-115`, `scripts/analysis/stage2_layer_significance.py:103-134`.

Provenance: Stage 2/3 now emits `env_snapshot.json` and `run_manifest.json`: `scripts/mechanistic/run_stage2b_continuation_pilot.py:216-236`, `scripts/mechanistic/run_stage2b_continuation_pilot.py:463-515`. But several result JSONs were generated before the current config fields, so direct result provenance is incomplete for legacy cells.

Stage 4 provenance is weak. `run_stage4_multimode_extract.py` writes only arrays to NPZ, with no command, git SHA, model revision, archive path, tier, selected task IDs, or formatter hash: `scripts/mechanistic/run_stage4_multimode_extract.py:183-192`. Because all Method 4.2/Exp 1/Exp 3 analyses consume that NPZ, provenance should be embedded in the NPZ or a sidecar manifest.

Model revision traceability is inconsistent: extraction pins revision (`p79/mechanistic/extract_hidden_states.py:40-66`), but logit lens does not (`scripts/analysis/stage4_logit_lens_axis2.py:54-63`). Also `env_snapshot.json` can report HF model as unavailable under offline mode, which blocks independent model identity verification.

### 6. One single highest-impact fix
Fix `scripts/mechanistic/run_stage4_multimode_extract.py` first: replace lexicographic directory selection with manifest-tier selection at `scripts/mechanistic/run_stage4_multimode_extract.py:86-96`, and replace lossy `build_som_marks` at `scripts/mechanistic/run_stage4_multimode_extract.py:46-55` with the production `[SOM_MARKS]` formatter from `p79/experiment/som.py:242-250`; then re-extract Stage 4 NPZs and regenerate Method 4.2, Exp 1, Exp 3, and layer-axis results.
