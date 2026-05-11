#!/usr/bin/env python3
"""Stage 4 Method 4.2: extract hidden states for ALL 6 modes on same task set.

Wraps `p79.mechanistic.extract_hidden_states.HiddenStateExtractor`. For each
(task, step) pair, runs forward pass for all 6 modes (DOM/P-text/P-prompt/
P-SoM/SoM/Vision) and saves per-layer last-token hidden states.

Output schema matches Stage 1B/1C cache format (npz with hidden_states,
labels, task_ids, step_indices, mode_labels_str), so downstream PCA/cosine
analysis is drop-in.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from p79.mechanistic.extract_hidden_states import HiddenStateExtractor, IMAGE_MAX_SIZE_DEFAULT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [stage4] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ALL_6_MODES = ["dom", "phantom_text", "phantom_prompt", "phantom_som", "som", "vision"]


def build_som_marks(obs_text: str) -> str:
    """Extract [SOM_MARKS] block from observation_dom.txt — copy of Stage 2B logic.

    AXTree dump contains lines like `[N] role 'label'`; we keep those and elide
    the rest.
    """
    import re
    pattern = re.compile(r"^\[\d+\]\s+\w+", re.MULTILINE)
    keep = pattern.findall(obs_text)
    return "\n".join(keep) + "\n[end of som marks]\n"


def text_payload_for(mode: str, obs_text: str, som_marks_text: str) -> str:
    """Same mapping as run_stage2b post-bug-fix (2026-05-10)."""
    if mode in ("som", "phantom_som", "phantom_text"):
        return som_marks_text
    if mode in ("phantom_prompt", "dom", "phantom_dom"):
        return obs_text
    if mode == "vision":
        return ""
    return som_marks_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="classifieds")
    parser.add_argument("--n-tasks", type=int, default=24)
    parser.add_argument("--steps", nargs="+", type=int, default=[2])
    parser.add_argument("--archived-run-dir", required=True,
                        help="archive_subset_b1_<site>/ dir with per-task observation snapshots")
    parser.add_argument("--output", required=True, help="output .npz path")
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--modes", nargs="+", default=ALL_6_MODES,
                        help="modes to extract (default: all 6)")
    args = parser.parse_args()

    archive_dir = Path(args.archived_run_dir)
    if not archive_dir.exists():
        raise SystemExit(f"archive dir missing: {archive_dir}")

    # Pick first n-tasks task IDs that have artifacts at all requested steps
    task_dirs = sorted(archive_dir.glob(f"{args.site}_task_*"))
    selected = []
    for td in task_dirs:
        if all((td / f"step_{s:03d}" / "observation_dom.txt").exists() and
               (td / f"step_{s:03d}" / "screenshot_annotated.png").exists()
               for s in args.steps):
            tid = int(td.name.rsplit("_", 1)[1])
            selected.append((tid, td))
        if len(selected) >= args.n_tasks:
            break
    logger.info(f"Selected {len(selected)} tasks (target {args.n_tasks})")
    if not selected:
        raise SystemExit("no archived tasks selected; check --site/--steps/--archived-run-dir")

    # Load intents — use same path as run_stage1_pilot.py (external/visualwebarena/config_files/vwa/test_<site>)
    REPO_ROOT = Path(__file__).resolve().parents[2]
    SITE_TO_CONFIG_DIR = {
        "classifieds": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_classifieds",
        "reddit": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_reddit",
        "shopping": REPO_ROOT / "external/visualwebarena/config_files/vwa/test_shopping",
    }
    intents_by_tid = {}
    cfg_dir = SITE_TO_CONFIG_DIR.get(args.site)
    if cfg_dir and cfg_dir.exists():
        for jf in cfg_dir.glob("*.json"):
            try:
                d = json.load(open(jf))
                # filename is <task_id>.json (stage1 convention); also fallback to d["task_id"]
                try:
                    tid = int(jf.stem)
                except ValueError:
                    tid = int(d.get("task_id", -1))
                intent = d.get("intent", "")
                if intent and tid >= 0:
                    intents_by_tid[tid] = intent
            except Exception as e:
                logger.warning(f"failed to load {jf}: {e}")
                continue
    logger.info(f"Loaded {len(intents_by_tid)} intents from {cfg_dir}")
    if not intents_by_tid:
        manifest_path = archive_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.load(open(manifest_path))
                for bucket in ("strong", "reverse"):
                    for item in manifest.get(bucket, []):
                        tid = int(item.get("task_id", -1))
                        intent = item.get("intent", "")
                        if tid >= 0 and intent:
                            intents_by_tid[tid] = intent
                logger.info(f"Loaded {len(intents_by_tid)} intents from {manifest_path}")
            except Exception as e:
                logger.warning(f"failed to load intents from {manifest_path}: {e}")
    if not intents_by_tid:
        raise SystemExit(
            "no intents loaded from external config or archive manifest; "
            "cannot extract hidden states"
        )

    extractor = HiddenStateExtractor(model_path=args.model_path)
    logger.info("Model loaded")

    # Iterate
    all_hs, all_modes, all_tids, all_steps, all_labels = [], [], [], [], []
    mode_to_label = {m: i for i, m in enumerate(args.modes)}

    for tid, td in selected:
        intent = intents_by_tid.get(tid, "")
        if not intent:
            logger.warning(f"task {tid}: missing intent, skip")
            continue
        for step in args.steps:
            step_dir = td / f"step_{step:03d}"
            obs_text = (step_dir / "observation_dom.txt").read_text(encoding="utf-8")
            som_marks = build_som_marks(obs_text)
            screenshot = step_dir / "screenshot_annotated.png"
            for mode in args.modes:
                payload = text_payload_for(mode, obs_text, som_marks)
                # vision and som need image; phantom_*/dom do not
                if mode in ("som", "vision"):
                    img = str(screenshot) if screenshot.exists() else None
                else:
                    img = None
                try:
                    hs = extractor.extract(intent, mode, observation_text=payload, image_path=img)
                    # hs shape (n_layers+1, hidden_dim)
                    all_hs.append(hs.cpu().numpy().astype(np.float32))
                    all_modes.append(mode)
                    all_tids.append(tid)
                    all_steps.append(step)
                    all_labels.append(mode_to_label[mode])
                except Exception as e:
                    logger.error(f"task {tid} step {step} mode {mode} failed: {e}")

    if not all_hs:
        raise SystemExit("no hidden states extracted; all selected tasks/modes failed")
    H = np.stack(all_hs)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
                        hidden_states=H,
                        labels=np.array(all_labels, dtype=np.int64),
                        task_ids=np.array(all_tids, dtype=np.int64),
                        step_indices=np.array(all_steps, dtype=np.int64),
                        mode_labels_str=np.array(all_modes, dtype="<U16"))
    logger.info(f"Saved {len(all_hs)} examples → {out} ({H.nbytes / 1e6:.1f} MB before compression)")
    logger.info(f"Modes: {dict(zip(*np.unique(all_modes, return_counts=True)))}")


if __name__ == "__main__":
    main()
