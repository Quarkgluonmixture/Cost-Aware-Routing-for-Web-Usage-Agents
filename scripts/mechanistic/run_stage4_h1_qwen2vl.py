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

# v6 /stress 2026-05-14 — Bug 2 + F1 fix propagation. Previously this script
# reimplemented private MARK_LINE_RE (v1 buggy regex) and dropped production
# system prompt. Now uses substrate.
#
# /stress A1.4 F1 backlog doc (2026-05-15): this cross-family script
# INTENTIONALLY imports `_extract_text_marks` directly rather than the
# canonical `build_som_text_from_obs_text`. Reason: this script tests format
# variation across cross-family models (Phi-3.5 / Qwen2-VL) by emitting raw
# marks under 8 different formatters (`[N] label`, `@N label`, `id_N: label`,
# etc.) — those format variants do NOT use the `[SOM_MARKS]...[/SOM_MARKS]`
# wrapper. As a side effect the `_options_map` recovery for dropdown `[OPTIONS]`
# lines (per master bug B-82) is NOT applied here, so cross-family inputs may
# differ from production SoM input on pages with select / dropdown elements.
# This is a known design tradeoff; mechanism §5 is paused per advisor §138
# (frozen archive, no re-activation planned) so the divergence is documented
# rather than corrected. If §5 ever re-activates, cross-family scripts should
# either (a) switch to canonical wrappers and accept format-loss, or (b)
# explicitly inject the [OPTIONS] lines per-formatter.
from p79.experiment.som import _extract_text_marks  # noqa: E402
from p79.agents.qwen3vl_agent import Qwen3VLAgent  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [stage4_h1_qwen2vl] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def hash_id(n):
    h = hashlib.md5(str(n).encode()).hexdigest()
    return f"{h[0]}{h[5]}{h[10]}{h[15]}"


def fmt_som_standard(obs_text):
    return "\n".join(f"[{m['id']}] {m['label']}" for m in _extract_text_marks(obs_text))


def fmt_browser_use_at(obs_text):
    return "\n".join(f"@{m['id']} {m['label']}" for m in _extract_text_marks(obs_text))


def fmt_appagent_id(obs_text):
    return "\n".join(f"id_{m['id']}: {m['label']}" for m in _extract_text_marks(obs_text))


def fmt_tarsier_typed(obs_text):
    # NOTE: production SoM payload has no explicit role; tarsier variant uses id + label only
    return "\n".join(f"[B{m['id']}:{m['label']}]" for m in _extract_text_marks(obs_text))


def fmt_plain_numbered(obs_text):
    return "\n".join(f"{m['id']}. {m['label']}" for m in _extract_text_marks(obs_text))


def fmt_xml_tagged(obs_text):
    return "\n".join(f'<el_{m["id"]}>{m["label"]}</el_{m["id"]}>' for m in _extract_text_marks(obs_text))


def fmt_hash_id_control(obs_text):
    return "\n".join(f"#{hash_id(m['id'])} {m['label']}" for m in _extract_text_marks(obs_text))


def fmt_plain_sentence(obs_text):
    return ", ".join(m["label"] for m in _extract_text_marks(obs_text))


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


def _build_user_text(intent: str, mode: str, observation_text: str, dom_prompt: str, som_prompt: str) -> str:
    """v6 /stress 2026-05-14 — F1 fix. Replicate substrate `_build_user_text`.

    /stress A1.4 B-103 fix (2026-05-15): production agent prepends
    `Accessibility Tree:\\n` for DOM-style modes (qwen3vl_agent.py:441-450);
    cross-family scripts were missing this prefix, producing input not
    byte-identical to production. Mechanism §5 is paused per advisor §138
    but the byte-divergence is corrected here for code consistency.
    """
    som_modes = {"som", "som_standard", "browser_use_at", "appagent_id", "tarsier_typed",
                 "plain_numbered", "xml_tagged", "hash_id_control", "plain_sentence"}
    sys_prompt = som_prompt if mode in som_modes else dom_prompt
    text = f"Task: {intent}\nSystem: {sys_prompt}\n"
    if observation_text:
        # DOM-style modes (= not in som_modes) carry the "Accessibility Tree:" header.
        if mode not in som_modes:
            text += f"Accessibility Tree:\n{observation_text}"
        else:
            text += observation_text
    return text


def extract_qwen2vl_hidden(model, processor, intent, mode, observation_text, dom_prompt, som_prompt):
    """Text-only forward pass for Qwen2-VL. v6 fix: mode-conditional system prompt."""
    user_text = _build_user_text(intent, mode, observation_text, dom_prompt, som_prompt)
    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden = torch.stack(
        [h[0, -1, :].detach().float().cpu() for h in outputs.hidden_states],
        dim=0,
    )
    return hidden


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archived-run-dir", default="results/mechanistic/archive_subset_b1_cls")
    p.add_argument("--output", default="results/mechanistic/stage4_h1_qwen2vl7b_cls/hidden_states_v2_fixed.npz")
    p.add_argument("--model-id", default="Qwen/Qwen2-VL-7B-Instruct")
    # v6 /stress 2026-05-14 — F3 fix (Bug 5 propagation). Pin revision for reproducibility.
    p.add_argument("--model-revision", default=None,
                   help="HF revision SHA to pin (paper-grade reproducibility). "
                        "Default None = use HF Hub current; first run records SHA in provenance.")
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

    # v6 F1 fix — load production system prompts.
    # /stress A1.1 B-92 propagation (2026-05-15): _make_*_prompt are @staticmethod
    # since commit 11d6fd9, so `(None)` argument now raises TypeError. Drop the arg.
    dom_prompt = Qwen3VLAgent._make_dom_prompt()
    som_prompt = Qwen3VLAgent._make_som_prompt()
    logger.info(f"loaded production prompts: DOM={len(dom_prompt)}c, SoM={len(som_prompt)}c")

    logger.info(f"loading {args.model_id} revision={args.model_revision or '(latest)'}")
    model_kwargs = dict(torch_dtype=torch.bfloat16, device_map="cuda")
    if args.model_revision:
        model_kwargs["revision"] = args.model_revision
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    model.eval()
    proc_kwargs = {}
    if args.model_revision:
        proc_kwargs["revision"] = args.model_revision
    processor = AutoProcessor.from_pretrained(args.model_id, **proc_kwargs)
    n_layers = len(model.model.layers)
    actual_revision = getattr(model.config, "_commit_hash", args.model_revision or "(unknown)")
    logger.info(f"model loaded — n_layers = {n_layers}, revision={actual_revision}")

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
                    h = extract_qwen2vl_hidden(model, processor, intent, mode, variant_text,
                                                 dom_prompt, som_prompt)
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

    # v6 /stress 2026-05-14 — provenance sidecar (Bug 5 fix propagation)
    import subprocess
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = "(unknown)"
    provenance = out_path.parent / "provenance.json"
    provenance.write_text(json.dumps({
        "script": "run_stage4_h1_qwen2vl.py",
        "model_id": args.model_id,
        "model_revision_arg": args.model_revision,
        "model_revision_actual": actual_revision,
        "tier": args.tier,
        "n_tasks": args.n_tasks,
        "steps": steps,
        "modes": ALL_MODES,
        "n_layers": n_layers,
        "shape": list(H.shape),
        "task_ids_unique": sorted(set(task_ids.tolist())),
        "git_sha": git_sha,
        "som_extractor": "p79.experiment.som._extract_text_marks (production, Bug 2 fix)",
        "system_prompts": "Qwen3VLAgent._make_{dom,som}_prompt (cross-arch canonical)",
    }, indent=2))
    logger.info(f"provenance: {provenance}")

    summary = out_path.parent / "pilot_summary.md"
    summary.write_text(
        f"# Qwen2-VL-7B P3 extraction\n\n"
        f"Shape: {H.shape}\n"
        f"Modes: {ALL_MODES}\n"
        f"Tasks: {len(set(task_ids.tolist()))}\n"
        f"Model: {args.model_id} @ {actual_revision}\n"
        f"Note: text-only (no som baseline). For H1' capacity-limit test. v6 fix landed 2026-05-14.\n"
    )
    logger.info(f"sentinel: {summary}")


if __name__ == "__main__":
    main()
