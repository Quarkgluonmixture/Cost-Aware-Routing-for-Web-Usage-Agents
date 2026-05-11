#!/usr/bin/env python3
"""Stage 4 H1 test: text format variation across industry-relevant indexed-list styles.

Tests refined H1 hypothesis (pretraining co-occurrence shortcut):
  "input contains mark-like indexed region list → activates visual grounding pathway"

If H1 is correct:
  - All marks-like variants (SoM / Browser-Use @ / AppAgent / Tarsier / numbered / XML)
    should cluster with baseline P-text → image-axis cosine gap peaks at L17+
  - Non-marks variants (plain sentence / hash IDs)
    should cluster with baseline DOM (AXTree) → image-axis cosine gap peaks at L4

8 industry-relevant variants applied to 24 cls strong-tier tasks × 2 steps:

  som_standard      [0] role 'label'                    (P79 baseline)
  browser_use_at    @0 label                            (Browser Use SDK style)
  appagent_id       id_0: label                         (AppAgent-v2 style)
  tarsier_typed     [B0:role:label]                     (Tarsier Reworkd)
  plain_numbered    0. label                            (generic numbered)
  xml_tagged        <el_0 role="role">label</el_0>      (OmniParser-style)
  hash_id_control   #aBc7 label                         (random hash, no integer index — control)
  plain_sentence    "label0, label1, label2..."         (no list structure — control)

Reuses dom (AXTree) + som (image+marks) baseline modes for cross-comparison.

Output: results/mechanistic/stage4_format_variation_b1_cls/hidden_states.npz
shape (10 modes × 24 tasks × 2 steps = 480, n_layers+1, hidden_dim)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import hashlib
from pathlib import Path

if os.environ.get("FORCE_MATH_SDP", "1") != "0":
    try:
        import torch as _t
        _t.backends.cuda.enable_flash_sdp(False)
        _t.backends.cuda.enable_mem_efficient_sdp(False)
        _t.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from p79.mechanistic.extract_hidden_states import HiddenStateExtractor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [stage4fv] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# Parse [N] role 'label' lines from observation_dom.txt → list of (n, role, label)
MARK_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s+(\S+)\s+'([^']*)'")


def extract_marks(obs_text: str) -> list[tuple[int, str, str]]:
    """Extract (idx, role, label) tuples from AXTree-style observation text."""
    out = []
    for line in obs_text.split("\n"):
        m = MARK_LINE_RE.match(line.strip())
        if m:
            out.append((int(m.group(1)), m.group(2), m.group(3)))
    return out


def hash_id(n: int) -> str:
    """Deterministic 4-char alphanumeric hash, no integer pattern."""
    h = hashlib.md5(str(n).encode()).hexdigest()
    # Avoid pure digits — mix in letters
    return f"{h[0]}{h[5]}{h[10]}{h[15]}"


# Format transformers — each takes obs_text, returns text payload string
def fmt_som_standard(obs_text):
    """Baseline [SOM_MARKS] — same as run_stage2b build_som_marks."""
    return "\n".join(line.strip() for line in obs_text.split("\n")
                      if line.strip().startswith("[") and "]" in line.strip()[:6])


def fmt_browser_use_at(obs_text):
    marks = extract_marks(obs_text)
    return "\n".join(f"@{n} {label}" for n, role, label in marks)


def fmt_appagent_id(obs_text):
    marks = extract_marks(obs_text)
    return "\n".join(f"id_{n}: {label}" for n, role, label in marks)


def fmt_tarsier_typed(obs_text):
    marks = extract_marks(obs_text)
    return "\n".join(f"[B{n}:{role}:{label}]" for n, role, label in marks)


def fmt_plain_numbered(obs_text):
    marks = extract_marks(obs_text)
    return "\n".join(f"{n}. {label}" for n, role, label in marks)


def fmt_xml_tagged(obs_text):
    marks = extract_marks(obs_text)
    return "\n".join(f'<el_{n} role="{role}">{label}</el_{n}>' for n, role, label in marks)


def fmt_hash_id_control(obs_text):
    """Control: replace integer index with non-integer hash. Tests whether integer index is the trigger."""
    marks = extract_marks(obs_text)
    return "\n".join(f"#{hash_id(n)} {label}" for n, role, label in marks)


def fmt_plain_sentence(obs_text):
    """Control: drop list structure entirely. Tests whether 'list' pattern is the trigger."""
    marks = extract_marks(obs_text)
    return ", ".join(label for n, role, label in marks)


VARIANTS = {
    "som_standard":     fmt_som_standard,
    "browser_use_at":   fmt_browser_use_at,
    "appagent_id":      fmt_appagent_id,
    "tarsier_typed":    fmt_tarsier_typed,
    "plain_numbered":   fmt_plain_numbered,
    "xml_tagged":       fmt_xml_tagged,
    "hash_id_control":  fmt_hash_id_control,
    "plain_sentence":   fmt_plain_sentence,
}


def find_archive_dir(p79_root: Path) -> Path:
    """Locate manifest archive — on DGX vs Myriad have different paths."""
    cand = p79_root / "results/mechanistic/archive_subset_b1_cls"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"archive not found at {cand}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archived-run-dir", required=True, help="Path to archive_subset_b1_cls")
    p.add_argument("--output", default=None)
    p.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--tier", default="strong")
    p.add_argument("--n-tasks", type=int, default=24)
    p.add_argument("--steps", default="2,5")
    p.add_argument("--min-free-vram-gb", type=float, default=0.0)
    args = p.parse_args()

    steps = [int(x) for x in args.steps.split(",")]
    archive_dir = Path(args.archived_run_dir)
    manifest_path = archive_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tasks = manifest[args.tier][:args.n_tasks]
    logger.info(f"Loaded {len(tasks)} tasks (tier={args.tier})")

    intents_by_tid = {int(t["task_id"]): t["intent"] for t in tasks}

    extractor = HiddenStateExtractor(model_path=args.model_path, min_free_vram_gb=args.min_free_vram_gb)
    logger.info(f"Model loaded: {args.model_path}")

    # Plus the 2 reused baselines: 'dom' (AXTree) and 'som' (image + marks)
    BASELINES = ["dom", "som"]
    ALL_MODES = list(VARIANTS.keys()) + BASELINES

    all_hidden = []
    all_meta = []  # tuples of (task_id, step, mode_label)

    for tid in sorted(intents_by_tid):
        intent = intents_by_tid[tid]
        for step in steps:
            task_dir = archive_dir / f"classifieds_task_{tid}" / f"step_{step:03d}"
            obs_path = task_dir / "observation_dom.txt"
            if not obs_path.exists():
                logger.warning(f"missing {obs_path}; skip")
                continue
            obs_text = obs_path.read_text(encoding="utf-8")
            screenshot = task_dir / "screenshot_annotated.png"

            for mode in ALL_MODES:
                if mode in VARIANTS:
                    # Variant: DOM-prompt + variant text + no image
                    variant_text = VARIANTS[mode](obs_text)
                    try:
                        h = extractor.extract(intent=intent, mode="dom",
                                                observation_text=variant_text, image_path=None)
                    except Exception as e:
                        logger.error(f"task {tid} step {step} variant {mode} failed: {e}")
                        continue
                elif mode == "dom":
                    # baseline DOM: AXTree text + DOM-prompt + no image
                    h = extractor.extract(intent=intent, mode="dom",
                                            observation_text=obs_text, image_path=None)
                elif mode == "som":
                    # baseline SoM: marks text + SoM-prompt + WITH image
                    marks_text = fmt_som_standard(obs_text)
                    img_path = screenshot if screenshot.exists() else None
                    h = extractor.extract(intent=intent, mode="som",
                                            observation_text=marks_text, image_path=img_path)
                else:
                    continue

                all_hidden.append(h.numpy() if hasattr(h, "numpy") else h)
                all_meta.append((tid, step, mode))
            logger.info(f"  task={tid} step={step} done ({len(ALL_MODES)} modes)")

    if not all_hidden:
        raise SystemExit("no hidden states extracted")

    H = np.stack(all_hidden)  # (N, n_layers+1, hidden_dim)
    task_ids = np.array([m[0] for m in all_meta])
    step_indices = np.array([m[1] for m in all_meta])
    mode_labels = np.array([m[2] for m in all_meta])

    out_path = Path(args.output or (Path(__file__).resolve().parents[2]
                                       / "results/mechanistic/stage4_format_variation_b1_cls/hidden_states.npz"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.array([list(ALL_MODES).index(m) for m in mode_labels.tolist()])
    np.savez_compressed(
        out_path,
        hidden_states=H,
        labels=labels,
        task_ids=task_ids,
        step_indices=step_indices,
        mode_labels_str=mode_labels,
    )
    logger.info(f"saved: {out_path}  shape={H.shape}  modes={ALL_MODES}")

    # Quick pilot_summary marker so auto_pull Phase 0 sentinel passes
    summary_path = out_path.parent / "pilot_summary.md"
    summary_path.write_text(
        f"# Stage 4 Format Variation Extract\n\n"
        f"Shape: {H.shape}\n"
        f"Modes: {ALL_MODES}\n"
        f"Tasks: {len(set(task_ids.tolist()))}\n"
        f"Steps: {sorted(set(step_indices.tolist()))}\n"
    )
    logger.info(f"sentinel: {summary_path}")


if __name__ == "__main__":
    main()
