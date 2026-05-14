"""Stage 2B — Multi-token continuation activation patching for B1 mirage analysis.

Addresses Stage 2A trivial first-token-agree problem (source argmax == target argmax
== JSON `{` opener forced by chat template). Generate 10-15 tokens past `{` so action_type
+ element_id divergence between source/target emerges, then measure how patching at each
layer pulls patched continuation toward source.

Setup (same as Stage 2A):
- Source = SoM (with screenshot_annotated.png, [SOM_MARKS], SoM prompt)
- Target = P-SoM (no image, [SOM_MARKS], SoM prompt) — mirage condition
- Per (task) × per (layer L = 0..35): patch source's L-th hidden state into target,
  greedy-generate max_new_tokens, compare full token sequences.

Metrics:
- token_overlap_to_source: ratio of positions where patched matches source (1=identical)
- token_overlap_to_target: same vs target baseline
- ld_to_source: Levenshtein edit distance to source token sequence
- ld_to_target: Levenshtein edit distance to target token sequence

Output:
    results/mechanistic/stage2b_continuation_b1_cls_pilot/
      patching_continuation_results.json
      patching_continuation_curves.png
      pilot_summary.md

Usage:
    python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
      --site classifieds --n-tasks 3 --step 2 --max-new-tokens 15 \
      --archived-run-dir results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428

ETA: 3 task × (1 source-gen + 1 target-gen + 1 source-cache + 36 patched-gen)
   = 3 × 39 generation passes × ~15 forward each = ~1750 forwards × 1.5s
   ≈ 45 min compute + 2 min model load = ~50 min total.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# B-81h workaround (笔记 §117): force SDPA math backend so the script runs on
# any GPU architecture. PyTorch's flash + memory-efficient SDPA backends only
# have bf16 cutlass kernels for sm_80+ (A100/H100). On V100 (sm_70) or T4
# (sm_75) Myriad nodes, SDPA dispatcher raises "cutlassF: no kernel found to
# launch!" instead of falling back. The math backend always works (slower
# ~2-3x but correct on any GPU). Disabled via FORCE_MATH_SDP=0 to opt back in.
if os.environ.get("FORCE_MATH_SDP", "1") != "0":
    try:
        import torch as _torch_for_sdp_setup
        _torch_for_sdp_setup.backends.cuda.enable_flash_sdp(False)
        _torch_for_sdp_setup.backends.cuda.enable_mem_efficient_sdp(False)
        _torch_for_sdp_setup.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from p79.mechanistic.activation_patching import ActivationPatcher, patching_grid_continuation
from p79.mechanistic.extract_hidden_states import HiddenStateExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage2b-continuation")

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_TO_CONFIG_DIR = {
    "classifieds": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_classifieds",
    "reddit": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_reddit",
    "shopping": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_shopping",
}


def load_intents(site: str, n_tasks: int) -> list[tuple[int, str]]:
    """Load intents from VWA config_files (full repo with submodule init)."""
    config_dir = SITE_TO_CONFIG_DIR[site]
    json_files = sorted(config_dir.glob("*.json"), key=lambda p: int(p.stem))
    intents = []
    for jf in json_files[:n_tasks]:
        d = json.loads(jf.read_text(encoding="utf-8"))
        if d.get("intent"):
            intents.append((int(jf.stem), d["intent"]))
    return intents


def load_intents_from_subset_manifest(manifest_path: Path, tier: str, n_tasks: int) -> list[tuple[int, str]]:
    """Load intents from archive_subset manifest.json (cross-machine paper-grade
    dataset). Used on Myriad / A100 where VWA submodule isn't init'd."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get(tier, [])[:n_tasks]
    return [(int(e["task_id"]), e["intent"]) for e in entries]


def find_artifacts_dir(run_dir: Path) -> Path:
    """Find artifacts directory; supports two layouts:
    (a) nested:  <run>/<condition>/artifacts/<site>_task_X/step_NNN/
    (b) flat:    <subset>/<site>_task_X/step_NNN/  (extract_archive_subset.py output)
    """
    # Layout (a): nested condition/artifacts
    for child in run_dir.iterdir():
        if child.is_dir() and (child / "artifacts").is_dir():
            return child / "artifacts"
    # Layout (b): flat subset (run_dir IS the artifacts dir)
    for child in run_dir.iterdir():
        if child.is_dir() and any(
            child.name.startswith(prefix)
            for prefix in ("classifieds_task_", "reddit_task_", "shopping_task_")
        ):
            return run_dir
    raise FileNotFoundError(f"No artifacts in {run_dir} (tried nested + flat layouts)")


def build_som_marks(obs_text: str, max_marks: int = 200) -> str:
    from p79.experiment.som import _extract_text_marks
    marks = _extract_text_marks(obs_text, max_marks=max_marks)
    if not marks:
        return "[SOM_MARKS]\n[/SOM_MARKS]"
    return "\n".join(["[SOM_MARKS]"] + [f"[id={m['id']}] {m['label']}" for m in marks] + ["[/SOM_MARKS]"])


def build_inputs(extractor: HiddenStateExtractor, intent: str, mode: str, obs_text: str, image_path):
    user_text = extractor._build_user_text(intent, mode, obs_text)
    content = []
    if image_path is not None:
        img = HiddenStateExtractor._load_resize_image(image_path)
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": user_text})
    messages = [{"role": "user", "content": content}]
    text = extractor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if image_path is not None:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = extractor.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
    else:
        inputs = extractor.processor(text=[text], padding=True, return_tensors="pt")
    return {k: v.to(extractor.model.device) for k, v in inputs.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", default="classifieds", choices=list(SITE_TO_CONFIG_DIR))
    p.add_argument("--n-tasks", type=int, default=3, help="Default 3 for fast pilot, scale to 5+ for paper")
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=15, help="Continuation length (15 covers JSON envelope start)")
    p.add_argument("--archived-run-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument(
        "--model-revision",
        default="ebb281ec70b05090aa6165b016eac8ec08e71b17",
        help="HF revision SHA. Must match Stage 4 v2 extraction.",
    )
    p.add_argument("--source-mode", default="som")
    p.add_argument("--target-mode", default="phantom_som")
    p.add_argument("--min-free-vram-gb", type=float, default=0.0)
    p.add_argument(
        "--reverse", action="store_true",
        help="Swap source ↔ target: patch target's hidden state into source run "
             "(asymmetry control test). Output dir gets _reverse suffix.",
    )
    p.add_argument(
        "--tier", default=None, choices=["strong", "reverse"],
        help="Override manifest tier subset (strong=24 forward-mirage / reverse=15 reverse-mirage). "
             "If unset, auto-derives from --reverse flag (default behavior). "
             "Set explicitly to enable 2x2 cross-subset control: e.g. --reverse --tier strong "
             "tests reverse direction on forward-easy tasks (selection-bias control).",
    )
    p.add_argument(
        "--random-inject", action="store_true",
        help="Random-injection control (paper §5 reviewer Q): replace cached source "
             "hidden state with Gaussian noise matched to per-layer mean/std. Tests "
             "whether mid-layer disruption depends on source-content specificity vs "
             "any non-zero injection. Expected null at all layers if mechanism is "
             "source-content-specific.",
    )
    p.add_argument(
        "--random-seed", type=int, default=42,
        help="Seed for --random-inject Gaussian noise (paper-grade reproducibility). "
             "Same seed + same input = same noise = byte-identical re-runs. Default 42.",
    )
    p.add_argument(
        "--task-shuffle", action="store_true",
        help="Task-shuffled content-specificity control (codex methodology audit "
             "2026-05-12 Bug: Gaussian random-injection is a WEAK specificity baseline "
             "because variance-matched noise breaks residual normalization regardless "
             "of axis content). Task-shuffle replaces SOURCE intent + obs + screenshot "
             "with those of a DIFFERENT task while preserving source mode (= same "
             "residual-stream statistics, different content). If patching effect is "
             "content-specific, task-shuffled should produce MUCH SMALLER displacement "
             "than real-source same-task. Incompatible with --random-inject (they are "
             "alternative controls).",
    )
    p.add_argument(
        "--task-shuffle-seed", type=int, default=42,
        help="Seed for --task-shuffle deterministic permutation (reproducibility).",
    )
    args = p.parse_args()
    if args.task_shuffle and args.random_inject:
        raise SystemExit(
            "--task-shuffle and --random-inject are mutually exclusive controls. "
            "Pick one — they measure different specificity dimensions."
        )

    # C8 fix: seed all RNGs when random-inject is on, for paper-grade
    # reproducibility. Affects torch.randn_like in patching_grid_continuation.
    # Default seed=42 means re-running with same data + code produces
    # byte-identical noise + byte-identical patched outputs.
    if args.random_inject:
        import random as _rnd
        import numpy as _np
        import torch as _t
        _rnd.seed(args.random_seed)
        _np.random.seed(args.random_seed)
        _t.manual_seed(args.random_seed)
        if _t.cuda.is_available():
            _t.cuda.manual_seed_all(args.random_seed)
        # Defense-in-depth visibility (防忘): log prominently + this seed
        # value also flows to env_snapshot.json + pilot_summary.md (below).
        print(f"\n{'=' * 60}\n[RANDOM-INJECT SEED] {args.random_seed} "
              f"(reproducibility — see pilot_summary.md + env_snapshot.json)\n"
              f"{'=' * 60}\n", flush=True)
        # Note: cell E (job 335404) is currently running with NO seed (commit
        # before this fix). That run is one valid realization; future paper-
        # grade re-runs will be byte-reproducible with --random-seed 42.

    suffix = "_reverse" if args.reverse else ""
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / f"results/mechanistic/stage2b_continuation_b1_{args.site}_pilot{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    # Paper-grade provenance: dump env snapshot at run start (Gap 1+3, 笔记 §114)
    try:
        from scripts.provenance.snapshot_env import capture_env_snapshot
        capture_env_snapshot(
            out_dir / "env_snapshot.json",
            extra={
                "stage": "stage2b_curated",
                "site": args.site,
                "reverse": args.reverse,
                "tier": args.tier or ("reverse" if args.reverse else "strong"),
                "random_inject": args.random_inject,
                "random_seed": args.random_seed,
                "task_shuffle": args.task_shuffle,
                "task_shuffle_seed": args.task_shuffle_seed,
                "n_tasks_requested": args.n_tasks,
                "step": args.step,
                "max_new_tokens": args.max_new_tokens,
                "source_mode": args.source_mode,
                "target_mode": args.target_mode,
                "model_path": args.model_path,
                "model_revision": args.model_revision,
            },
        )
    except Exception as e:
        logger.warning(f"Env snapshot failed (non-fatal): {e}")

    archived_dir = Path(args.archived_run_dir)

    # Auto-detect: if archived_run_dir contains manifest.json, it's a subset
    # (extract_archive_subset.py output). Use intents from manifest, support
    # flat layout. This enables cross-machine paper-grade workflow (Myriad / A100)
    # without needing the full B1_phantom_som_classifieds_20260428 archive (~1.8GB).
    subset_manifest = archived_dir / "manifest.json"
    if subset_manifest.exists():
        tier = args.tier if args.tier else ("reverse" if args.reverse else "strong")
        intents = load_intents_from_subset_manifest(subset_manifest, tier=tier, n_tasks=args.n_tasks)
        logger.info(
            f"Subset mode: loaded {len(intents)} intents from manifest "
            f"(tier={tier}, reverse_dir={args.reverse}, "
            f"cross_subset={'YES' if args.tier and args.tier != ('reverse' if args.reverse else 'strong') else 'NO'})"
        )
    else:
        intents = load_intents(args.site, args.n_tasks)
        logger.info(f"Full archive mode: loaded {len(intents)} intents from VWA config_files")

    artifacts_dir = find_artifacts_dir(archived_dir)
    logger.info(f"Archived artifacts: {artifacts_dir}")

    extractor = HiddenStateExtractor(
        model_path=args.model_path,
        model_revision=args.model_revision,
        min_free_vram_gb=args.min_free_vram_gb,
    )
    patcher = ActivationPatcher(extractor.model, extractor.processor)
    logger.info(f"Model loaded (revision={args.model_revision[:12]}...); n_layers={patcher.n_layers}")

    # Task-shuffled control (codex methodology audit 2026-05-12 Bug 6/G3 follow-up):
    # build a deterministic permutation so that target task T_i uses source artifacts
    # from a DIFFERENT task T_j (j != i). Preserves residual-stream statistics (same
    # source mode = same variance), breaks content correspondence.
    target_to_source_task: dict[int, int] = {}
    if args.task_shuffle:
        import random as _rnd
        task_ids_ordered = [tid for tid, _ in intents]
        intent_by_tid = dict(intents)
        rng = _rnd.Random(args.task_shuffle_seed)
        shuffled = list(task_ids_ordered)
        # Derangement: no fixed point. Simple swap-based: rotate by 1, then random
        # swaps that preserve the no-fixed-point property.
        if len(shuffled) > 1:
            shuffled = shuffled[1:] + shuffled[:1]  # rotation guarantees no fixed point
            # Optional: a few random swaps that maintain derangement
            for _ in range(len(shuffled)):
                i = rng.randrange(len(shuffled))
                j = rng.randrange(len(shuffled))
                if (task_ids_ordered[i] != shuffled[j] and
                        task_ids_ordered[j] != shuffled[i]):
                    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        target_to_source_task = dict(zip(task_ids_ordered, shuffled))
        for tt, ss in list(target_to_source_task.items())[:5]:
            logger.info(f"task-shuffle: target task {tt} → source task {ss}")
        # Sanity: no fixed point
        n_fixed = sum(1 for k, v in target_to_source_task.items() if k == v)
        if n_fixed > 0:
            logger.warning(f"task-shuffle: {n_fixed} tasks are fixed points (should be 0); shuffle weak")

    per_task_results = []
    for task_id, intent in intents:
        step_dir = artifacts_dir / f"{args.site}_task_{task_id}" / f"step_{args.step:03d}"
        obs_file = step_dir / "observation_dom.txt"
        screenshot_annotated = step_dir / "screenshot_annotated.png"
        if not obs_file.exists() or not screenshot_annotated.exists():
            logger.warning(f"task {task_id}: missing artifacts, skip")
            continue
        obs_text = obs_file.read_text(encoding="utf-8")
        som_marks_text = build_som_marks(obs_text)

        # 2026-05-10 bug fix: text payload depends on observation mode.
        # phantom_prompt = SoM prompt + AXTree (no marks); previously hardcoded som_marks_text
        # for all modes, which made phantom_prompt byte-identical to phantom_som (Cell F vs
        # H-prompt-red 24/24 patched_text identical, confirming bug).
        text_payload_for = lambda mode: (
            som_marks_text if mode in ("som", "phantom_som", "phantom_text")
            else obs_text  if mode in ("phantom_prompt", "dom", "phantom_dom")
            else ""        if mode == "vision"
            else som_marks_text
        )

        # SOURCE side: by default same task as target; if --task-shuffle, use
        # a different task's artifacts (intent + obs + screenshot) for source.
        if args.task_shuffle:
            source_task_id = target_to_source_task[task_id]
            source_step_dir = artifacts_dir / f"{args.site}_task_{source_task_id}" / f"step_{args.step:03d}"
            source_obs_file = source_step_dir / "observation_dom.txt"
            source_screenshot = source_step_dir / "screenshot_annotated.png"
            if not source_obs_file.exists() or not source_screenshot.exists():
                logger.warning(f"task-shuffle: target {task_id} → source {source_task_id} "
                                "missing artifacts, falling back to same-task source")
                source_intent = intent
                source_obs_text = obs_text
                source_som_marks = som_marks_text
                source_screenshot_path = str(screenshot_annotated)
            else:
                source_intent = intent_by_tid.get(source_task_id, intent)
                source_obs_text = source_obs_file.read_text(encoding="utf-8")
                source_som_marks = build_som_marks(source_obs_text)
                source_screenshot_path = str(source_screenshot)
            logger.info(f"task {task_id}: SHUFFLED source = task {source_task_id}")
        else:
            source_intent = intent
            source_obs_text = obs_text
            source_som_marks = som_marks_text
            source_screenshot_path = str(screenshot_annotated)

        source_text_payload_for = lambda mode: (
            source_som_marks if mode in ("som", "phantom_som", "phantom_text")
            else source_obs_text if mode in ("phantom_prompt", "dom", "phantom_dom")
            else ""              if mode == "vision"
            else source_som_marks
        )
        source_text = source_text_payload_for(args.source_mode)
        target_text = text_payload_for(args.target_mode)

        source_inputs_orig = build_inputs(extractor, source_intent, args.source_mode, source_text, source_screenshot_path)
        target_inputs_orig = build_inputs(extractor, intent, args.target_mode, target_text, None)

        # --reverse: swap roles. patch target's hidden into source run = "remove image content"
        if args.reverse:
            source_inputs, target_inputs = target_inputs_orig, source_inputs_orig
            logger.info(f"task {task_id}: REVERSE direction (patching {args.target_mode} → {args.source_mode})")
        else:
            source_inputs, target_inputs = source_inputs_orig, target_inputs_orig
            logger.info(f"task {task_id}: forward direction (patching {args.source_mode} → {args.target_mode})")

        logger.info(f"task {task_id}: running continuation patching grid (max_new_tokens={args.max_new_tokens})...")
        result = patching_grid_continuation(
            patcher, source_inputs, target_inputs,
            max_new_tokens=args.max_new_tokens,
            randomize_source_hidden=args.random_inject,
        )
        result["task_id"] = task_id
        result["step_idx"] = args.step
        result["intent"] = intent
        per_task_results.append(result)

        # F18 audit fix 2026-05-09: include reverse / tier / random_inject /
        # random_seed in incremental JSON so downstream stage2 stat scripts
        # can reconstruct which causal/control cell produced these numbers.
        # (Previously only env_snapshot.json carried these; analysis scripts
        # consume the results JSON directly.)
        # F19 audit fix 2026-05-09: source_mode/target_mode reported here
        # are the role labels AFTER the reverse swap (i.e. what was actually
        # patched into what), matching the per-task INFO log line above.
        if args.reverse:
            logged_source_mode = args.target_mode
            logged_target_mode = args.source_mode
        else:
            logged_source_mode = args.source_mode
            logged_target_mode = args.target_mode

        with (out_dir / "patching_continuation_results.json").open("w") as f:
            json.dump({
                "config": {
                    "site": args.site, "n_tasks": args.n_tasks, "step": args.step,
                    "max_new_tokens": args.max_new_tokens,
                    "source_mode": logged_source_mode,
                    "target_mode": logged_target_mode,
                    "source_mode_raw": args.source_mode,
                    "target_mode_raw": args.target_mode,
                    "reverse": args.reverse,
                    "tier": args.tier or ("reverse" if args.reverse else "strong"),
                    "random_inject": args.random_inject,
                    "random_seed": args.random_seed,
                    "task_shuffle": args.task_shuffle,
                    "task_shuffle_seed": args.task_shuffle_seed,
                    "task_shuffle_map": (
                        {str(k): v for k, v in target_to_source_task.items()}
                        if args.task_shuffle else None
                    ),
                    "archived_run_dir": str(archived_dir),
                    "model_path": args.model_path,
                    "model_revision": args.model_revision,
                    "n_layers": patcher.n_layers,
                    "layer_index_convention": "patcher layer L = decoder block L output; Stage 4 NPZ block L is H[:, L+1, :]",
                },
                "per_task": per_task_results,
            }, f, indent=2)

    if not per_task_results:
        logger.error("No tasks completed; aborting plot")
        return

    # Aggregate per-layer mean ± std across tasks
    n_layers = patcher.n_layers
    metric_names = ["token_overlap_to_source", "token_overlap_to_target", "ld_to_source", "ld_to_target"]
    agg = {}
    for m in metric_names:
        arr = np.array([
            [layer_r[m] for layer_r in t["per_layer"]]
            for t in per_task_results
        ])  # (n_tasks, n_layers)
        agg[f"{m}_mean"] = arr.mean(axis=0).tolist()
        agg[f"{m}_std"] = arr.std(axis=0).tolist()

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    layers_x = np.arange(n_layers)
    titles = {
        "token_overlap_to_source": "Token overlap → source\n(1=patched matches source position-by-position)",
        "token_overlap_to_target": "Token overlap → target\n(higher = patch had no effect)",
        "ld_to_source": f"Levenshtein dist → source\n(0=identical, max~{args.max_new_tokens})",
        "ld_to_target": f"Levenshtein dist → target\n(higher = patch pulled away from target)",
    }
    for ax, m in zip(axes.flat, metric_names):
        mean = np.array(agg[f"{m}_mean"])
        std = np.array(agg[f"{m}_std"])
        ax.plot(layers_x, mean, marker="o", lw=1.5, label=f"mean (N={len(per_task_results)})")
        ax.fill_between(layers_x, mean - std, mean + std, alpha=0.25, label="±1 std")
        ax.set_xlabel("Block index (B0=decoder block 0 output; no embedding hook)")
        ax.set_title(titles[m], fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    if args.reverse:
        direction_label = f"{args.target_mode}→{args.source_mode} (reverse)"
    else:
        direction_label = f"{args.source_mode}→{args.target_mode} (forward)"
    tier_label = args.tier if args.tier else ("reverse" if args.reverse else "strong")
    fig.suptitle(
        f"Stage 2B Continuation Activation Patching — {direction_label} "
        f"({args.site} N={len(per_task_results)} {tier_label}-tier task × step_{args.step:03d}, "
        f"max_new_tokens={args.max_new_tokens})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "patching_continuation_curves.png", dpi=150)
    plt.close(fig)
    logger.info("Saved patching_continuation_curves.png")

    # Summary
    overlap_src = np.array(agg["token_overlap_to_source_mean"])
    overlap_tgt = np.array(agg["token_overlap_to_target_mean"])
    ld_src = np.array(agg["ld_to_source_mean"])
    ld_tgt = np.array(agg["ld_to_target_mean"])

    best_overlap_layer = int(overlap_src.argmax())
    best_ld_layer = int(ld_src.argmin())

    # Sample some patched outputs for qualitative check
    qualitative = []
    for t in per_task_results[:1]:  # first task only for brevity
        qualitative.append(f"\n### Task {t['task_id']} (intent: {t['intent'][:80]})")
        qualitative.append(f"  source: {t['source_text']!r}")
        qualitative.append(f"  target: {t['target_text']!r}")
        for L in [0, 5, 11, 17, 23, 29, 35]:
            r = t["per_layer"][L]
            qualitative.append(f"  B{L:2d} patched: {r['patched_text']!r}  (overlap→src={r['token_overlap_to_source']:.2f}, LD→src={r['ld_to_source']})")
    qual_block = "\n".join(qualitative)

    summary = f"""# Stage 2B Continuation Activation Patching — Summary

## Setup
- Model: {args.model_path}
- Model revision: {args.model_revision}
- Site: {args.site}, N task: {len(per_task_results)} × step_{args.step:03d}
- Source: `{args.source_mode}` (with image — clean) / Target: `{args.target_mode}` (no image — mirage)
- Direction: {"reverse (target→source)" if args.reverse else "forward (source→target)"}
- Tier: {args.tier or ("reverse" if args.reverse else "strong")}
- max_new_tokens: {args.max_new_tokens} (greedy continuation, deterministic)
- Random injection: {"YES, seed=" + str(args.random_seed) + " (paper-grade reproducible)" if args.random_inject else "NO (real source hidden injected)"}
- Archived: {args.archived_run_dir}

## Result (per-block mean across tasks)
- Best block for **token overlap → source**: B{best_overlap_layer} (overlap {overlap_src[best_overlap_layer]:.3f})
- Best block for **min Levenshtein → source**: B{best_ld_layer} (LD {ld_src[best_ld_layer]:.2f})

## Block-resolved curves (source-side metrics):
| Block | overlap→src | overlap→tgt | LD→src | LD→tgt |
|---|---|---|---|---|
| B0  | {overlap_src[0]:.2f} | {overlap_tgt[0]:.2f} | {ld_src[0]:.1f} | {ld_tgt[0]:.1f} |
| B5  | {overlap_src[5]:.2f} | {overlap_tgt[5]:.2f} | {ld_src[5]:.1f} | {ld_tgt[5]:.1f} |
| B11 | {overlap_src[11]:.2f} | {overlap_tgt[11]:.2f} | {ld_src[11]:.1f} | {ld_tgt[11]:.1f} |
| B17 | {overlap_src[17]:.2f} | {overlap_tgt[17]:.2f} | {ld_src[17]:.1f} | {ld_tgt[17]:.1f} |
| B23 | {overlap_src[23]:.2f} | {overlap_tgt[23]:.2f} | {ld_src[23]:.1f} | {ld_tgt[23]:.1f} |
| B29 | {overlap_src[29]:.2f} | {overlap_tgt[29]:.2f} | {ld_src[29]:.1f} | {ld_tgt[29]:.1f} |
| B35 | {overlap_src[35]:.2f} | {overlap_tgt[35]:.2f} | {ld_src[35]:.1f} | {ld_tgt[35]:.1f} |

## Interpretation
- overlap→src curve climbs monotonically with depth → mirage info accumulates layer-by-layer (deep layer wins)
- overlap→src peaks at middle layer then decays → "computed feature" emerges mid then abstracts
- overlap→src flat ~0 → patching has no causal effect (mirage info distributed elsewhere)
- LD→src minimum identifies "most source-like patched output" layer — opposite signal of overlap→src

## Qualitative samples (first task)
{qual_block}

## Next steps
- If mid-block peak emerges (e.g. B17-B25) → consistent with Stage 2A logit_shift block-window finding ✓
- If late-layer monotone climb → mirage signature is residual-stream cumulative, no single causal layer
- Scale up: 5 task × max_new_tokens=20 (~75 min) for tighter mean ± std
- Then Stage 2C: reverse direction (target→source patching) for asymmetry check
"""
    (out_dir / "pilot_summary.md").write_text(summary)

    # Paper-grade run manifest (Gap 3, 笔记 §114) — single-file roll-up of
    # patch config + per-task outcomes for OSF DOI lock + cross-machine compare.
    run_manifest = {
        "stage": "stage2b_continuation_curated" if args.n_tasks > 5 else "stage2b_continuation_pilot",
        "direction": "reverse" if args.reverse else "forward",
        "site": args.site,
        "patch_config": {
            "source_mode": args.source_mode,
            "target_mode": args.target_mode,
            "step_idx": args.step,
            "max_new_tokens": args.max_new_tokens,
            "n_layers_swept": int(patcher.n_layers),
            "hook_position": "last_token",
            "first_forward_only": True,
            "min_free_vram_gb": args.min_free_vram_gb,
        },
        "model": {
            "path": args.model_path,
            "revision": args.model_revision,
            "n_layers": int(patcher.n_layers),
            "layer_index_convention": "patcher layer L = decoder block L output; Stage 4 NPZ block L is H[:, L+1, :]",
        },
        "input_dataset": {
            "archived_run_dir": str(archived_dir),
            "n_tasks_requested": args.n_tasks,
            "n_tasks_completed": len(per_task_results),
            "task_ids": [int(t["task_id"]) for t in per_task_results],
        },
        "outcomes_per_task": [
            {
                "task_id": int(t["task_id"]),
                "step_idx": int(t["step_idx"]),
                "best_layer_overlap_src": int(np.argmax([r["token_overlap_to_source"] for r in t["per_layer"]])),
                "best_overlap_src": float(max(r["token_overlap_to_source"] for r in t["per_layer"])),
                "L11_overlap_src": float(t["per_layer"][11]["token_overlap_to_source"]) if patcher.n_layers > 11 else None,
                "L17_overlap_src": float(t["per_layer"][17]["token_overlap_to_source"]) if patcher.n_layers > 17 else None,
            }
            for t in per_task_results
        ],
        "aggregate": {
            "best_layer_overlap_src_mean": int(best_overlap_layer),
            "best_overlap_src_mean": float(overlap_src[best_overlap_layer]),
            "best_layer_ld_src_mean": int(best_ld_layer),
            "L11_overlap_src_mean": float(overlap_src[11]) if patcher.n_layers > 11 else None,
            "L17_overlap_src_mean": float(overlap_src[17]) if patcher.n_layers > 17 else None,
        },
        "env_snapshot_ref": "env_snapshot.json",
        "results_files": {
            "per_task_jsonl": "patching_continuation_results.json",
            "curves_plot": "patching_continuation_curves.png",
            "summary_md": "pilot_summary.md",
        },
    }
    with (out_dir / "run_manifest.json").open("w") as f:
        json.dump(run_manifest, f, indent=2)
    logger.info(f"run_manifest.json emitted ({len(per_task_results)} tasks)")
    logger.info(f"Stage 2B continuation patching pilot DONE → {out_dir}")


if __name__ == "__main__":
    main()
