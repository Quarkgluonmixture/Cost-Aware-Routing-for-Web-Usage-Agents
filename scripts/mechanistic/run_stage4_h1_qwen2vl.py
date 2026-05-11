#!/usr/bin/env python3
"""P3: H1 test on Qwen2-VL-7B-Instruct (within-Qwen-family bigger model).

Directly tests H1' (4B capacity-limit alternative hypothesis):
  If H1 finding (flat-list triggers shortcut, AXTree defeats it) reflects
  4B capacity limit rather than training-distribution prior, then 7B (in
  same Qwen family) should show DIFFERENT pattern (e.g. dichotomy weaker
  or disappear).

Same 8 text format variants + dom baseline. Skip som baseline (image-
required) for simplicity — text-only forward enough for H1 dichotomy test.

Uses Qwen2-VL-7B-Instruct (Qwen2VLForConditionalGeneration). Same 24 cls
strong-tier tasks × 2 steps from archive_subset_b1_cls.

Output: results/mechanistic/stage4_h1_qwen2vl7b_cls/hidden_states.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [stage4_h1_qwen2vl] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Format transformers — copy from run_stage4_format_variation_extract.py
MARK_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s+(\S+)\s+'([^']*)'")


def extract_marks(obs_text):
    out = []
    for line in obs_text.split("\n"):
        m = MARK_LINE_RE.match(line.strip())
        if m:
            out.append((int(m.group(1)), m.group(2), m.group(3)))
    return out


def hash_id(n):
    h = hashlib.md5(str(n).encode()).hexdigest()
    return f"{h[0]}{h[5]}{h[10]}{h[15]}"


def fmt_som_standard(obs_text):
    return "\n".join(line.strip() for line in obs_text.split("\n")
                      if line.strip().startswith("[") and "]" in line.strip()[:6])


def fmt_browser_use_at(obs_text):
    return "\n".join(f"@{n} {label}" for n, role, label in extract_marks(obs_text))


def fmt_appagent_id(obs_text):
    return "\n".join(f"id_{n}: {label}" for n, role, label in extract_marks(obs_text))


def fmt_tarsier_typed(obs_text):
    return "\n".join(f"[B{n}:{role}:{label}]" for n, role, label in extract_marks(obs_text))


def fmt_plain_numbered(obs_text):
    return "\n".join(f"{n}. {label}" for n, role, label in extract_marks(obs_text))


def fmt_xml_tagged(obs_text):
    return "\n".join(f'<el_{n} role="{role}">{label}</el_{n}>' for n, role, label in extract_marks(obs_text))


def fmt_hash_id_control(obs_text):
    return "\n".join(f"#{hash_id(n)} {label}" for n, role, label in extract_marks(obs_text))


def fmt_plain_sentence(obs_text):
    return ", ".join(label for n, role, label in extract_marks(obs_text))


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


def extract_qwen2vl_hidden(model, processor, intent, observation_text):
    """Text-only forward pass for Qwen2-VL. Returns (n_layers+1, hidden_dim) tensor."""
    # Qwen2-VL chat template: same pattern as Qwen3-VL
    user_text = f"Task: {intent}\n[observation]\n{observation_text}"
    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    # outputs.hidden_states: tuple of (n_layers+1) tensors (batch, seq, hidden)
    # Take last-token of each layer
    hidden = torch.stack(
        [h[0, -1, :].detach().float().cpu() for h in outputs.hidden_states],
        dim=0,
    )
    return hidden


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archived-run-dir", default="results/mechanistic/archive_subset_b1_cls")
    p.add_argument("--output", default="results/mechanistic/stage4_h1_qwen2vl7b_cls/hidden_states.npz")
    p.add_argument("--model-id", default="Qwen/Qwen2-VL-7B-Instruct")
    p.add_argument("--tier", default="strong")
    p.add_argument("--n-tasks", type=int, default=24)
    p.add_argument("--steps", default="2,5")
    args = p.parse_args()
    steps = [int(x) for x in args.steps.split(",")]

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    archive_dir = Path(args.archived_run_dir)
    manifest = json.loads((archive_dir / "manifest.json").read_text())
    tasks = manifest[args.tier][:args.n_tasks]
    logger.info(f"loaded {len(tasks)} tasks (tier={args.tier})")

    logger.info(f"loading {args.model_id} (this may take a few minutes on first run)")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    logger.info(f"model loaded — n_layers = {len(model.model.layers)}")

    ALL_MODES = list(VARIANTS.keys()) + ["dom"]  # 8 variants + AXTree baseline (no image, no som)

    all_hidden = []
    all_meta = []

    for t in tasks:
        tid = int(t["task_id"])
        intent = t["intent"]
        for step in steps:
            task_dir = archive_dir / f"classifieds_task_{tid}" / f"step_{step:03d}"
            obs_path = task_dir / "observation_dom.txt"
            if not obs_path.exists():
                logger.warning(f"missing {obs_path}; skip")
                continue
            obs_text = obs_path.read_text(encoding="utf-8")

            for mode in ALL_MODES:
                if mode == "dom":
                    variant_text = obs_text  # full AXTree
                else:
                    variant_text = VARIANTS[mode](obs_text)
                try:
                    h = extract_qwen2vl_hidden(model, processor, intent, variant_text)
                except Exception as e:
                    logger.error(f"task {tid} step {step} mode {mode} failed: {e}")
                    continue
                all_hidden.append(h.numpy())
                all_meta.append((tid, step, mode))
            logger.info(f"  task={tid} step={step} done ({len(ALL_MODES)} modes)")

    if not all_hidden:
        raise SystemExit("no hidden states extracted")

    H = np.stack(all_hidden)
    task_ids = np.array([m[0] for m in all_meta])
    step_indices = np.array([m[1] for m in all_meta])
    mode_labels = np.array([m[2] for m in all_meta])
    labels = np.array([list(ALL_MODES).index(m) for m in mode_labels.tolist()])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, hidden_states=H, labels=labels,
                          task_ids=task_ids, step_indices=step_indices, mode_labels_str=mode_labels)
    logger.info(f"saved: {out_path}  shape={H.shape}  modes={ALL_MODES}")

    summary = out_path.parent / "pilot_summary.md"
    summary.write_text(
        f"# Qwen2-VL-7B P3 extraction\n\n"
        f"Shape: {H.shape}\n"
        f"Modes: {ALL_MODES}\n"
        f"Tasks: {len(set(task_ids.tolist()))}\n"
        f"Note: text-only (no som baseline). For H1' capacity-limit test.\n"
    )
    logger.info(f"sentinel: {summary}")


if __name__ == "__main__":
    main()
