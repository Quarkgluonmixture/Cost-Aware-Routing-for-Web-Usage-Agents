"""Stage 2 — Activation patching pilot for B1 (Qwen3-VL-4B) mirage analysis.

Per advisor 5/5: "patch 到哪一层的时候, 它的结果就切换了" — find layer L
where source-into-target patching causes output flip → causal mirage layer.

Setup:
- Source = SoM mode (with screenshot_annotated.png, [SOM_MARKS], SoM prompt)
- Target = P-SoM mode (no image, [SOM_MARKS], SoM prompt) — mirage condition
- Same task + same archived observation; only image presence differs
- Per (task) × per (layer L = 0..35): patch source's layer-L last-token hidden
  state into target run → measure 4 causal-effect metrics

Output:
    results/mechanistic/stage2_patching_b1_cls_pilot/
      patching_results.json     (per-task per-layer 4 metrics)
      patching_curves.png       (mean ± std over tasks for each metric)
      pilot_summary.md

Usage:
    python3 scripts/mechanistic/run_stage2_patching_pilot.py \
      --site classifieds --n-tasks 5 \
      --archived-run-dir results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428 \
      --step 2

ETA: 5 task × 36 layer × ~3s/forward (image source slower) ≈ 12-18 min compute
+ 2 min model load = ~15-20 min total.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from p79.mechanistic.activation_patching import ActivationPatcher, patching_grid
from p79.mechanistic.extract_hidden_states import HiddenStateExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage2-patching")

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_TO_CONFIG_DIR = {
    "classifieds": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_classifieds",
    "reddit": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_reddit",
    "shopping": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_shopping",
}


def load_intents(site: str, n_tasks: int) -> list[tuple[int, str]]:
    config_dir = SITE_TO_CONFIG_DIR[site]
    json_files = sorted(config_dir.glob("*.json"), key=lambda p: int(p.stem))
    intents = []
    for jf in json_files[:n_tasks]:
        d = json.loads(jf.read_text())
        if d.get("intent"):
            intents.append((int(jf.stem), d["intent"]))
    return intents


def find_artifacts_dir(run_dir: Path) -> Path:
    for child in run_dir.iterdir():
        if child.is_dir() and (child / "artifacts").is_dir():
            return child / "artifacts"
    raise FileNotFoundError(f"No condition subdir with artifacts/ in {run_dir}")


def build_som_marks(obs_text: str, max_marks: int = 200) -> str:
    """Canonical [SOM_MARKS] builder — delegates to the single source of truth.

    master bug B-82 fix (2026-05-14): prior local impl dropped the
    `_options_map` dropdown-options recovery. Now delegates to
    `p79.experiment.som.build_som_text_from_obs_text`.
    """
    from p79.experiment.som import build_som_text_from_obs_text
    return build_som_text_from_obs_text(obs_text, max_marks=max_marks)


def build_inputs(extractor: HiddenStateExtractor, intent: str, mode: str, obs_text: str, image_path):
    """Build model kwargs (input_ids, attention_mask, pixel_values, etc.) for a forward pass."""
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
    p.add_argument("--n-tasks", type=int, default=5)
    p.add_argument("--step", type=int, default=2, help="Step index (default 2 = post-homepage navigation)")
    p.add_argument("--archived-run-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--source-mode", default="som")
    p.add_argument("--target-mode", default="phantom_som")
    p.add_argument("--min-free-vram-gb", type=float, default=0.0)
    args = p.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / f"results/mechanistic/stage2_patching_b1_{args.site}_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    intents = load_intents(args.site, args.n_tasks)
    logger.info(f"Loaded {len(intents)} task intents")

    archived_dir = Path(args.archived_run_dir)
    artifacts_dir = find_artifacts_dir(archived_dir)
    logger.info(f"Archived artifacts: {artifacts_dir}")

    extractor = HiddenStateExtractor(model_path=args.model_path, min_free_vram_gb=args.min_free_vram_gb)
    patcher = ActivationPatcher(extractor.model, extractor.processor)
    logger.info(f"Model loaded; n_layers={patcher.n_layers}")

    per_task_results = []
    for task_id, intent in intents:
        step_dir = artifacts_dir / f"{args.site}_task_{task_id}" / f"step_{args.step:03d}"
        obs_file = step_dir / "observation_dom.txt"
        screenshot_annotated = step_dir / "screenshot_annotated.png"
        if not obs_file.exists() or not screenshot_annotated.exists():
            logger.warning(f"task {task_id} step {args.step}: missing artifacts, skip")
            continue
        obs_text = obs_file.read_text()
        som_marks_text = build_som_marks(obs_text)

        # Source = SoM with image; Target = P-SoM no image. Same [SOM_MARKS] text + same SoM prompt.
        source_inputs = build_inputs(extractor, intent, args.source_mode, som_marks_text, str(screenshot_annotated))
        target_inputs = build_inputs(extractor, intent, args.target_mode, som_marks_text, None)

        logger.info(f"task {task_id}: running patching grid over {patcher.n_layers} layers...")
        result = patching_grid(patcher, source_inputs, target_inputs)
        result["task_id"] = task_id
        result["step_idx"] = args.step
        result["intent"] = intent
        per_task_results.append(result)

        # Save incrementally so partial run still recoverable
        with (out_dir / "patching_results.json").open("w") as f:
            json.dump({
                "config": {
                    "site": args.site, "n_tasks": args.n_tasks, "step": args.step,
                    "source_mode": args.source_mode, "target_mode": args.target_mode,
                    "archived_run_dir": str(archived_dir),
                    "model_path": args.model_path,
                    "n_layers": patcher.n_layers,
                },
                "per_task": per_task_results,
            }, f, indent=2)

    if not per_task_results:
        logger.error("No tasks had complete artifacts; aborting plot")
        return

    # Aggregate: mean ± std across tasks per layer per metric
    n_layers = patcher.n_layers
    metrics = ["argmax_match_source", "logit_shift_to_source", "kl_patched_to_source", "kl_patched_to_target"]
    agg = {}
    for m in metrics:
        arr = np.array([r[m] for r in per_task_results])  # (n_tasks, n_layers)
        agg[f"{m}_mean"] = arr.mean(axis=0).tolist()
        agg[f"{m}_std"] = arr.std(axis=0).tolist()

    # Plot 4-panel
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    layers_x = np.arange(n_layers)
    metric_titles = {
        "argmax_match_source": "Argmax match → source (1=full flip)",
        "logit_shift_to_source": "Logit shift toward source argmax\n(1=full shift, 0=none)",
        "kl_patched_to_source": "KL(patched ‖ source) — lower = closer to source",
        "kl_patched_to_target": "KL(patched ‖ target) — higher = further from target",
    }
    for ax, m in zip(axes.flat, metrics):
        mean = np.array(agg[f"{m}_mean"])
        std = np.array(agg[f"{m}_std"])
        ax.plot(layers_x, mean, marker="o", lw=1.5, label=f"mean (N={len(per_task_results)})")
        ax.fill_between(layers_x, mean - std, mean + std, alpha=0.25, label="±1 std")
        ax.set_xlabel("Layer index (0=embedding, ≥1=post-block)")
        ax.set_title(metric_titles[m], fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Stage 2 Activation Patching — {args.source_mode}→{args.target_mode} ({args.site} N={len(per_task_results)} task × step_{args.step:03d})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "patching_curves.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved patching_curves.png")

    # Summary
    am = np.array(agg["argmax_match_source_mean"])
    ls = np.array(agg["logit_shift_to_source_mean"])
    best_am_layer = int(am.argmax())
    best_ls_layer = int(ls.argmax())
    summary = f"""# Stage 2 Activation Patching Pilot — Summary

## Setup
- Model: {args.model_path}
- Site: {args.site}, N task: {len(per_task_results)} × step_{args.step:03d}
- Source mode: `{args.source_mode}` (with image — clean condition)
- Target mode: `{args.target_mode}` (no image — mirage condition)
- Archived run: {args.archived_run_dir}
- Layers tested: 0..{n_layers - 1} (n_layers={n_layers})

## Result (per-layer mean across tasks)
- Best layer for **argmax flip to source**: L{best_am_layer} (rate {am[best_am_layer]:.3f})
- Best layer for **logit shift to source**: L{best_ls_layer} (shift {ls[best_ls_layer]:.3f})

## Interpretation guide
- argmax_match_source ~ 0 throughout → patching has no causal effect (model ignores patch)
- argmax_match_source rises at deep layer (≥ L25) → mirage info concentrated in late layers
- argmax_match_source rises at middle layer (L10-L20) → mirage computed in mid layers
- argmax_match_source rises at early layer (L1-L5) → mirage signature already present from input encoding
- logit_shift_to_source: smooth metric; 1.0 = patched output fully matches source on argmax token

## Next steps
- If clear peak emerges → Stage 3 SAE feature steering at peak layer (deferred)
- If diffuse signal → check forward direction (target→source instead) for asymmetry
- Scale up: 30 task × 3 step → tighter mean ± std curves
"""
    (out_dir / "pilot_summary.md").write_text(summary)
    logger.info(f"Stage 2 patching pilot DONE → {out_dir}")


if __name__ == "__main__":
    main()
