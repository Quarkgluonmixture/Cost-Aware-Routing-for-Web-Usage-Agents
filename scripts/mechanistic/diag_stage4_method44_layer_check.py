#!/usr/bin/env python3
"""Diag: layer-index off-by-one + steering hook firing check.

Hypothesis: smoke test null is because we read direction from npz[17]
(= post-block-16 in patcher convention) but apply hook at patcher.layers[17]
(= post-block-17). Off by 1.

Tests on 1 task (cls task_1 step_2):
  1. baseline (no steering)
  2. α=50 steering at patcher.layers[17]   ← what smoke test did
  3. α=50 steering at patcher.layers[16]   ← corrected position
  4. α=50 RANDOM direction at patcher.layers[16]   ← control: should produce gibberish if hook fires

Verdict:
  if (2)=(1)=baseline + (3)=gibberish + (4)=gibberish:
    → off-by-one bug, fix = use layer_idx-1 in patcher
  if (2)=(3)=(1)=baseline + (4)=baseline:
    → hook not firing AT ALL (model arch mismatch?)
  if (2),(3) shift but (4) shifts more:
    → direction works, but smoke alpha was too small (try larger)
"""
from __future__ import annotations

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
from p79.mechanistic.extract_hidden_states import HiddenStateExtractor  # noqa: E402
from p79.mechanistic.activation_patching import ActivationPatcher  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [diag] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz"
ARCHIVE = ROOT / "results/mechanistic/archive_subset_b1_cls"
MANIFEST = ARCHIVE / "manifest.json"

ALPHA = 50.0  # extreme, force any hook-firing to show up


def build_som_marks(obs_text: str, max_marks: int = 200) -> str:
    """Canonical [SOM_MARKS] builder — delegates to the single source of truth.

    master bug B-82 fix (2026-05-14): prior local impl was a crude AXTree
    line-grep. Now delegates to `p79.experiment.som.build_som_text_from_obs_text`.
    """
    from p79.experiment.som import build_som_text_from_obs_text
    return build_som_text_from_obs_text(obs_text, max_marks=max_marks)


def build_inputs(extractor, intent, mode, obs_text):
    user_text = extractor._build_user_text(intent, mode, obs_text)
    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    text = extractor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = extractor.processor(text=[text], padding=True, return_tensors="pt")
    return {k: v.to(extractor.model.device) for k, v in inputs.items()}


def main():
    d = np.load(NPZ, allow_pickle=True)
    H = d["hidden_states"]  # (288, 37, 2560)
    ml = d["mode_labels_str"]

    # Direction at npz idx 17 (= patcher.layers[16] output in our hypothesis)
    v_npz17 = H[ml == "phantom_som"][:, 17, :].mean(0) - H[ml == "dom"][:, 17, :].mean(0)
    # Direction at npz idx 18 (= patcher.layers[17] output)
    v_npz18 = H[ml == "phantom_som"][:, 18, :].mean(0) - H[ml == "dom"][:, 18, :].mean(0)
    logger.info(f"||v_npz17|| = {np.linalg.norm(v_npz17):.4f}, ||v_npz18|| = {np.linalg.norm(v_npz18):.4f}")

    manifest = json.loads(MANIFEST.read_text())
    t = manifest["strong"][0]  # cls task 1
    tid = int(t["task_id"])
    intent = t["intent"]
    obs_path = ARCHIVE / f"classifieds_task_{tid}" / "step_002" / "observation_dom.txt"
    obs_text = obs_path.read_text(encoding="utf-8")
    logger.info(f"task={tid} intent={intent!r}")

    extractor = HiddenStateExtractor(min_free_vram_gb=0)
    patcher = ActivationPatcher(extractor.model, extractor.processor)
    logger.info(f"n_layers={patcher.n_layers}")
    dom_inputs = build_inputs(extractor, intent, "dom", obs_text)

    def gen(layer_idx=None, direction=None, alpha=0.0, max_tok=15):
        if layer_idx is None:
            out = patcher.model.generate(**dom_inputs, max_new_tokens=max_tok, do_sample=False,
                                          return_dict_in_generate=True, use_cache=True)
            toks = out.sequences[0, dom_inputs["input_ids"].shape[1]:].cpu().tolist()
        else:
            v_t = torch.tensor(direction)
            toks = patcher.steered_generate(layer_idx=layer_idx, direction=v_t, alpha=alpha,
                                              max_new_tokens=max_tok, **dom_inputs).cpu().tolist()
        return extractor.processor.tokenizer.decode(toks, skip_special_tokens=True), toks

    logger.info("Test 1: baseline (no steering)")
    txt0, toks0 = gen()
    logger.info(f"  baseline: {txt0!r}")

    logger.info("Test 2: α=50 at patcher.layers[17] = original smoke test position")
    txt2, toks2 = gen(layer_idx=17, direction=v_npz17, alpha=ALPHA)
    logger.info(f"  layers[17] + v_npz17: {txt2!r}")

    logger.info("Test 3: α=50 at patcher.layers[16] = corrected position (matches direction extraction)")
    txt3, toks3 = gen(layer_idx=16, direction=v_npz17, alpha=ALPHA)
    logger.info(f"  layers[16] + v_npz17: {txt3!r}")

    logger.info("Test 4: α=50 at patcher.layers[17] with direction from npz idx 18 = both at layers[17]")
    txt4, toks4 = gen(layer_idx=17, direction=v_npz18, alpha=ALPHA)
    logger.info(f"  layers[17] + v_npz18: {txt4!r}")

    logger.info("Test 5: α=50 RANDOM direction at patcher.layers[16] (gibberish control)")
    rng = np.random.default_rng(20260511)
    v_rand = rng.standard_normal(v_npz17.shape) * np.linalg.norm(v_npz17) / np.sqrt(len(v_npz17))
    txt5, toks5 = gen(layer_idx=16, direction=v_rand, alpha=ALPHA)
    logger.info(f"  layers[16] + RANDOM: {txt5!r}")

    logger.info("\n=== VERDICT ===")
    for label, t in [("baseline", txt0), ("layers[17]+npz17", txt2), ("layers[16]+npz17", txt3),
                       ("layers[17]+npz18", txt4), ("layers[16]+RANDOM", txt5)]:
        matches_baseline = (t == txt0)
        logger.info(f"  {label:30s} | matches baseline: {matches_baseline} | text: {t!r}")


if __name__ == "__main__":
    main()
