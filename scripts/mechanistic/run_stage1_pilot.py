"""Stage 1 mechanistic pilot — per-layer linear probe on B1 (Qwen3-VL-4B).

Two modes:

(A) Empty observation (Stage 1A): system prompt + intent, no observation. Fast
    infra smoke test, mode-axis is "system prompt structure only".

(B) Archived observation (Stage 1B): load observation_dom.txt from an archived
    paper-grade run; for each (task, step) build counterfactual prompts under
    {DOM, P-SoM} on the same observation. Mode-axis is "system prompt + text
    payload format on same page state". Cleaner contrastive — only 1 axis varies.

Pipeline:
1. Load N task configs (intent fields)
2. (B only) Load observation_dom.txt per (task, step) from archived run
3. For each (task[, step]) × {mode_a, mode_b}: forward pass extract last-token hidden state
4. Run per-layer 5-fold CV linear probe predicting mode label
5. Save: hidden_states.npz / probe_results.json / auroc_curve.png / pilot_summary.md

Usage:
    # Stage 1A (empty obs, fast pilot):
    python3 scripts/mechanistic/run_stage1_pilot.py --site classifieds --n-tasks 30

    # Stage 1B (archived obs, cleaner contrastive):
    python3 scripts/mechanistic/run_stage1_pilot.py \
      --site classifieds \
      --n-tasks 30 \
      --archived-run-dir results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428 \
      --steps 2 5

Stage 1 caveats (paper §X disclosure when promoting to paper-grade):
- (A) Empty observation: validates infra; doesn't reflect actual prompt context
- (B) [SOM_MARKS] from `_extract_text_marks` regex; production also injects
  dropdown OPTIONS via `_options_map` (slight drift, paper §X disclose)
- Mode label = binary (mode_a=0, mode_b=1). Doesn't isolate "mirage signature
  within P-SoM" (per-step mirage attribution = future Stage 1C).
- Single seed (42). Cross-seed stability = Stage 1D.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for p79

from p79.mechanistic import HiddenStateExtractor, linear_probe_per_layer, plot_auroc_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage1-pilot")

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_TO_CONFIG_DIR = {
    "classifieds": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_classifieds",
    "reddit": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_reddit",
    "shopping": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_shopping",
}


def load_task_intents(site: str, n_tasks: int) -> list[tuple[int, str]]:
    """Load (task_id, intent) pairs from VWA config dir."""
    config_dir = SITE_TO_CONFIG_DIR[site]
    if not config_dir.exists():
        raise FileNotFoundError(f"Config dir not found: {config_dir}")
    json_files = sorted(config_dir.glob("*.json"), key=lambda p: int(p.stem))
    intents = []
    for jf in json_files[:n_tasks]:
        with jf.open() as f:
            d = json.load(f)
        intent = d.get("intent")
        if not intent:
            logger.warning(f"Task {jf.stem} has no intent field, skipping")
            continue
        intents.append((int(jf.stem), intent))
    logger.info(f"Loaded {len(intents)} task intents from {site}")
    return intents


def _build_som_marks_text(obs_text: str, max_marks: int = 200) -> str:
    """Build [SOM_MARKS] text from raw AXTree (regex-filter via p79.experiment.som).

    Note: production `_build_som_result` ALSO injects dropdown OPTIONS via
    `_options_map`. We skip that here (slight prompt drift, paper §X disclose).
    """
    from p79.experiment.som import _extract_text_marks
    marks = _extract_text_marks(obs_text, max_marks=max_marks)
    if not marks:
        return "[SOM_MARKS]\n[/SOM_MARKS]"
    mark_lines = [f"[id={m['id']}] {m['label']}" for m in marks]
    return "\n".join(["[SOM_MARKS]"] + mark_lines + ["[/SOM_MARKS]"])


def _find_artifacts_dir(run_dir: Path) -> Path:
    """Find the condition subdir containing artifacts/ in an archived run."""
    for child in run_dir.iterdir():
        if child.is_dir() and (child / "artifacts").is_dir():
            return child / "artifacts"
    raise FileNotFoundError(f"No condition subdir with artifacts/ in {run_dir}")


def load_archived_items(
    run_dir: Path,
    site: str,
    intents: list[tuple[int, str]],
    steps: list[int],
    modes: tuple[str, str],
) -> list[dict]:
    """Load (task, step) × mode items from archived run.

    For each (task_id, step_idx) in archived run:
      - Read observation_dom.txt (raw AXTree)
      - Per mode: select obs format + image path (None / screenshot / annotated)

    Mode → (obs_text, image_file) mapping:
      - dom              → (full AXTree, None)
      - som              → ([SOM_MARKS], screenshot_annotated.png)
      - vision           → ("", screenshot.png)
      - phantom_som      → ([SOM_MARKS], None) — image-mismatched (mirage axis)
      - phantom_text     → ([SOM_MARKS], None) — text-mismatched
      - phantom_prompt   → (full AXTree, None) — prompt-only swap

    Returns list of dicts: {task_id, step_idx, intent, mode, observation_text, image_path}
    """
    artifacts_dir = _find_artifacts_dir(run_dir)
    logger.info(f"Loading archived observations from {artifacts_dir}")

    items = []
    skipped = 0
    skipped_no_img = 0
    for task_id, intent in intents:
        task_dir = artifacts_dir / f"{site}_task_{task_id}"
        if not task_dir.is_dir():
            skipped += 1
            continue
        for step_idx in steps:
            step_dir = task_dir / f"step_{step_idx:03d}"
            obs_file = step_dir / "observation_dom.txt"
            if not obs_file.exists():
                continue
            obs_text = obs_file.read_text()
            som_marks_text = _build_som_marks_text(obs_text)
            screenshot_annotated = step_dir / "screenshot_annotated.png"
            screenshot_raw = step_dir / "screenshot.png"

            for mode in modes:
                # Mode → obs format + image
                if mode == "dom":
                    obs_for_mode, img_for_mode = obs_text, None
                elif mode == "som":
                    obs_for_mode = som_marks_text
                    img_for_mode = screenshot_annotated if screenshot_annotated.exists() else None
                elif mode == "vision":
                    obs_for_mode = ""
                    img_for_mode = screenshot_raw if screenshot_raw.exists() else None
                elif mode in ("phantom_som", "phantom_text", "phantom_dom"):
                    obs_for_mode, img_for_mode = som_marks_text, None
                elif mode == "phantom_prompt":
                    obs_for_mode, img_for_mode = obs_text, None
                else:
                    obs_for_mode, img_for_mode = obs_text, None

                # Skip image-required modes if image missing (e.g. som mode without screenshot_annotated)
                if mode in ("som", "vision") and img_for_mode is None:
                    skipped_no_img += 1
                    continue

                items.append({
                    "task_id": task_id,
                    "step_idx": step_idx,
                    "intent": intent,
                    "mode": mode,
                    "observation_text": obs_for_mode,
                    "image_path": str(img_for_mode) if img_for_mode is not None else None,
                })
    logger.info(
        f"Loaded {len(items)} archived items "
        f"(modes={list(modes)}); skipped {skipped} tasks not in artifacts dir, "
        f"{skipped_no_img} samples with missing image artifact"
    )
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="classifieds", choices=list(SITE_TO_CONFIG_DIR))
    parser.add_argument("--n-tasks", type=int, default=30, help="N tasks (default 30 for fast pilot)")
    parser.add_argument(
        "--modes",
        nargs=2,
        default=["dom", "phantom_som"],
        help="2 modes for binary contrastive (label 0 vs 1)",
    )
    parser.add_argument(
        "--archived-run-dir",
        default=None,
        help="Stage 1B: load observations from this archived run dir "
             "(e.g. results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428). "
             "If not set, Stage 1A empty-observation mode is used.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[2, 5],
        help="Step indices to sample per task (Stage 1B only). Default [2, 5] = "
             "step_002 + step_005 (mid-navigation, more cross-task variance than step_000).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dir (default results/mechanistic/stage1_b1_<site>_<stage>_pilot)",
    )
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-free-vram-gb", type=float, default=12.0)
    parser.add_argument(
        "--pca-dim", type=int, default=50,
        help="PCA dim per fold before LR (default 50). Set 0 to disable.",
    )
    parser.add_argument(
        "--probe-C", type=float, default=0.01,
        help="LR L2 regularization C (smaller = more reg, default 0.01 for low-N regime).",
    )
    args = parser.parse_args()

    is_stage_1b = args.archived_run_dir is not None
    stage_label = "1B_archived" if is_stage_1b else "1A_empty"

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / f"results/mechanistic/stage{stage_label}_b1_{args.site}_pilot"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Stage {stage_label} output dir: {out_dir}")

    # 1. Load task intents
    intents = load_task_intents(args.site, args.n_tasks)
    if len(intents) < 2:
        raise RuntimeError(f"Too few tasks loaded ({len(intents)}); need ≥ 2")

    # 2. Build item list — (intent, mode, observation_text)
    mode_a, mode_b = args.modes
    items_for_extractor = []  # list of (intent, mode, obs_text)
    item_metadata = []  # parallel: list of {task_id, step_idx, mode}

    if is_stage_1b:
        archived_dir = Path(args.archived_run_dir)
        if not archived_dir.is_dir():
            raise FileNotFoundError(f"--archived-run-dir not found: {archived_dir}")
        archived_items = load_archived_items(
            archived_dir, args.site, intents, args.steps, (mode_a, mode_b),
        )
        for it in archived_items:
            items_for_extractor.append((
                it["intent"], it["mode"], it["observation_text"], it["image_path"],
            ))
            item_metadata.append({
                "task_id": it["task_id"],
                "step_idx": it["step_idx"],
                "mode": it["mode"],
                "image_path": it["image_path"],
            })
    else:
        # Stage 1A: empty observation, no image
        for task_id, intent in intents:
            for mode in (mode_a, mode_b):
                items_for_extractor.append((intent, mode, None, None))
                item_metadata.append({"task_id": task_id, "step_idx": -1, "mode": mode, "image_path": None})

    logger.info(
        f"Will extract {len(items_for_extractor)} hidden state vectors "
        f"({len(items_for_extractor) // 2} (task, step) pairs × 2 modes)"
    )

    # 3. Load model + extract
    extractor = HiddenStateExtractor(
        model_path=args.model_path,
        min_free_vram_gb=args.min_free_vram_gb,
    )
    hidden_states_torch, mode_labels_str = extractor.extract_batch(items_for_extractor)
    hidden_states = hidden_states_torch.numpy()  # (n_items, n_layers + 1, hidden_dim)
    labels = np.array([0 if m == mode_a else 1 for m in mode_labels_str], dtype=np.int64)
    logger.info(
        f"Extracted hidden_states shape={hidden_states.shape} "
        f"(n_items, n_layers+1, hidden_dim); n_pos={int(labels.sum())}"
    )

    # 4. Save raw artifacts
    np.savez_compressed(
        out_dir / "hidden_states.npz",
        hidden_states=hidden_states,
        labels=labels,
        task_ids=np.array([m["task_id"] for m in item_metadata]),
        step_indices=np.array([m["step_idx"] for m in item_metadata]),
        mode_labels_str=np.array(mode_labels_str),
    )
    logger.info(f"Saved hidden_states.npz ({hidden_states.nbytes / 1e6:.1f} MB)")

    # 5. Per-layer linear probe
    pca_dim = args.pca_dim if args.pca_dim > 0 else None
    logger.info(
        f"Running per-layer linear probe ({args.n_folds}-fold CV, seed={args.seed}, "
        f"pca_dim={pca_dim}, C={args.probe_C})"
    )
    probe_results = linear_probe_per_layer(
        hidden_states, labels,
        n_folds=args.n_folds, seed=args.seed,
        pca_dim=pca_dim, C=args.probe_C,
    )
    probe_results["site"] = args.site
    probe_results["modes"] = list(args.modes)
    probe_results["seed"] = args.seed
    probe_results["model_path"] = args.model_path
    probe_results["n_tasks"] = len(intents)

    with (out_dir / "probe_results.json").open("w") as f:
        json.dump(probe_results, f, indent=2)
    logger.info(f"Saved probe_results.json: best layer {probe_results['best_layer']}, "
                f"AUROC {probe_results['best_auroc']:.4f}")

    # 6. Plot AUROC curve
    plot_auroc_curve(
        probe_results,
        save_path=str(out_dir / "auroc_curve.png"),
        title=(
            f"Stage 1 Linear Probe — {args.site} {mode_a}↔{mode_b} "
            f"(N={len(intents)} tasks × 2 modes)"
        ),
    )

    # 7. Pilot summary
    obs_desc = (
        f"archived run `{args.archived_run_dir}`, steps {args.steps}, "
        f"mode-conditional observation reconstruction"
        if is_stage_1b
        else "empty (system prompt + intent only — Stage 1A simplification)"
    )
    summary = f"""# Stage {stage_label} Mechanistic Pilot — Linear Probe AUROC

## Setup
- Model: {args.model_path}
- Site: {args.site}, N tasks: {len(intents)}
- Contrastive modes: `{mode_a}` (label 0) vs `{mode_b}` (label 1)
- Observation: {obs_desc}
- N items: {hidden_states.shape[0]}
- CV: {args.n_folds}-fold StratifiedKFold, seed {args.seed}

## Result
- Best layer: **{probe_results['best_layer']}** / {probe_results['n_layers']}
- Best AUROC: **{probe_results['best_auroc']:.4f} ± {probe_results['best_auroc_std']:.4f}**
- Layer-wise AUROC: see `auroc_curve.png` and `probe_results.json::auroc_mean`

## Interpretation guide
- AUROC ~ 0.5 at all layers → mode label not linearly decodable (unexpected; check pipeline)
- AUROC ~ 1.0 at embedding then decay → mode is "input-text-only" feature (separable from raw tokens)
- AUROC stable high across layers → mode preserved as persistent feature
- AUROC peak at middle layer (e.g. L14-L20) → mode "computed" feature; abstraction emerges then decays
- AUROC sharp drop at deep layer → model "abstracts away" mode; mirage becomes task-relevant only

## Next steps after Stage {stage_label}
{'- Stage 1A passes → run Stage 1B (--archived-run-dir flag) for production-grade prompt context' if not is_stage_1b else '- Stage 1B passes → Stage 2 activation patching (causal patch per layer to identify pivot layer)'}
- All-passes path → Stage 3 SAE feature steering (deferred; Qwen3-VL-4B 公开 SAE 不存在)
"""
    with (out_dir / "pilot_summary.md").open("w") as f:
        f.write(summary)
    logger.info(f"Saved pilot_summary.md")
    logger.info(f"\n{'='*60}\nStage {stage_label} pilot DONE — output: {out_dir}\n{'='*60}")


if __name__ == "__main__":
    main()
