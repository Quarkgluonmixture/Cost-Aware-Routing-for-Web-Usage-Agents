"""Stage 1 mechanistic pilot — per-layer linear probe on B1 (Qwen3-VL-4B).

Pipeline:
1. Load N classifieds task configs (intent fields)
2. For each task × {DOM, P-SoM} mode: forward pass extract last-token hidden state
3. Run per-layer 5-fold CV linear probe predicting mode label
4. Save: hidden_states.npz / probe_results.json / auroc_curve.png / pilot_summary.md

Usage:
    python3 scripts/mechanistic/run_stage1_pilot.py \
      --site classifieds \
      --n-tasks 30 \
      --output-dir results/mechanistic/stage1_b1_cls_pilot

For full 234-task pilot:
    python3 scripts/mechanistic/run_stage1_pilot.py --site classifieds --n-tasks 234

Stage 1 caveats (paper §X disclosure when promoting to paper-grade):
- Empty observation (system prompt + intent only). Stage 2+ should swap in
  archived observations to capture full prompt-conditional state.
- Mode label = (DOM=0, P-SoM=1). This validates infra; doesn't isolate
  "mirage signature within P-SoM" (that requires per-step mirage attribution).
- Single seed (42). Cross-seed stability check left to Stage 1B.
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
        "--output-dir",
        default=None,
        help="Output dir (default results/mechanistic/stage1_b1_<site>_pilot)",
    )
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-free-vram-gb", type=float, default=12.0)
    args = parser.parse_args()

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / f"results/mechanistic/stage1_b1_{args.site}_pilot"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    # 1. Load task intents
    intents = load_task_intents(args.site, args.n_tasks)
    if len(intents) < 2:
        raise RuntimeError(f"Too few tasks loaded ({len(intents)}); need ≥ 2")

    # 2. Build (intent, mode, observation_text=None) item list
    mode_a, mode_b = args.modes
    items = []
    for task_id, intent in intents:
        items.append((intent, mode_a, None))
        items.append((intent, mode_b, None))
    logger.info(f"Will extract {len(items)} hidden state vectors ({len(intents)} tasks × 2 modes)")

    # 3. Load model + extract
    extractor = HiddenStateExtractor(
        model_path=args.model_path,
        min_free_vram_gb=args.min_free_vram_gb,
    )
    hidden_states_torch, mode_labels_str = extractor.extract_batch(items)
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
        task_ids=np.array([tid for tid, _ in intents] * 2),
        mode_labels_str=np.array(mode_labels_str),
    )
    logger.info(f"Saved hidden_states.npz ({hidden_states.nbytes / 1e6:.1f} MB)")

    # 5. Per-layer linear probe
    logger.info(f"Running per-layer linear probe ({args.n_folds}-fold CV, seed={args.seed})")
    probe_results = linear_probe_per_layer(
        hidden_states, labels, n_folds=args.n_folds, seed=args.seed,
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
    summary = f"""# Stage 1 Mechanistic Pilot — Linear Probe AUROC

## Setup
- Model: {args.model_path}
- Site: {args.site}, N tasks: {len(intents)}
- Contrastive modes: `{mode_a}` (label 0) vs `{mode_b}` (label 1)
- Observation: empty (system prompt + intent only — Stage 1 simplification)
- CV: {args.n_folds}-fold StratifiedKFold, seed {args.seed}

## Result
- Best layer: **{probe_results['best_layer']}** / {probe_results['n_layers']}
- Best AUROC: **{probe_results['best_auroc']:.4f} ± {probe_results['best_auroc_std']:.4f}**
- Layer-wise AUROC: see `auroc_curve.png` and `probe_results.json::auroc_mean`

## Interpretation guide
- AUROC ~ 0.5 at all layers → mode label not linearly decodable (unexpected)
- AUROC ~ 1.0 at embedding then decay → mode is "input-text-only" feature
- AUROC stable high → mode preserved as feature throughout
- AUROC peak at middle layer → mode "computed" feature, decays after task abstraction

## Next steps
- If AUROC > 0.7 at any layer: validates infra, proceed to Stage 1B (full N=234)
- Then Stage 2: activation patching to identify causal layer
"""
    with (out_dir / "pilot_summary.md").open("w") as f:
        f.write(summary)
    logger.info(f"Saved pilot_summary.md")
    logger.info(f"\n{'='*60}\nStage 1 pilot DONE — output: {out_dir}\n{'='*60}")


if __name__ == "__main__":
    main()
