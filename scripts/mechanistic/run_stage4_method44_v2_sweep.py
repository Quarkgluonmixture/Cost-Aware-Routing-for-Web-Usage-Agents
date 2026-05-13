#!/usr/bin/env python3
"""Stage 4 Method 4.4 v2: layer × α sweep for mean-diff steering.

Diag (diag_stage4_method44_layer_check.py) showed v1 null was α-calibration
+ wrong-layer issue. At α=50 + L17, steering DOES shift output toward P-SoM
content ('red Toyota') but also breaks JSON envelope (over-steers).

This script measures the dose-response surface:
  layers: [11, 17, 23, 29, 33, 34]   ← mid → late, covers Wu et al. L34 default
                                       and Stage 2/4 L17 mid-locus
  α:      [1, 2, 5, 10, 20]           ← Wu et al. typical α=1, our diag found ≥10 needed

Per (task, step, layer, α):
  - token overlap to DOM baseline / P-SoM baseline (Jaccard)
  - completeness = shifted_toward_psom rate (overlap_psom > overlap_dom)
  - selectivity = JSON valid rate (envelope preserved; starts with '{' or '{ "')
  - reliability = harmonic mean of completeness × selectivity
    (HDMI framework, Khorasani et al. 2026 arXiv:2605.07631)

Direction at patcher.layers[L] ←→ npz[:, L+1, :]
(extract_hidden_states stores HF outputs.hidden_states with embedding at idx 0).

Output: results/mechanistic/stage4_multimode_b1_cls/method44_v2_sweep.json
"""
from __future__ import annotations

import argparse
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
from p79.mechanistic.activation_patching import ActivationPatcher, _levenshtein_token  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [stage4mm44v2] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
# Pipeline audit P0-5 fix (2026-05-13): was hidden_states.npz (v1 buggy SOM_MARKS
# regex dropping 71/72 marks). Other Stage 4 scripts default to v2_fixed; Method
# 4.4 was incoherent cross-pipeline. Steering direction now computed on same NPZ
# that cosine_gap / logit_lens / layer_profile read.
NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states_v2_fixed.npz"
ARCHIVE = ROOT / "results/mechanistic/archive_subset_b1_cls"
MANIFEST = ARCHIVE / "manifest.json"
OUT_JSON = ROOT / "results/mechanistic/stage4_multimode_b1_cls/method44_v2_sweep.json"
OUT_MD = ROOT / "docs/checkpoints/stage4_method44_v2_results.md"

DEFAULT_LAYERS = [11, 17, 23, 29, 33, 34]
DEFAULT_ALPHAS = [1.0, 2.0, 5.0, 10.0, 20.0]


def build_som_marks(obs_text):
    return "\n".join(s for line in obs_text.split("\n")
                      if (s := line.strip()).startswith("[") and "]" in s[:6])


def build_inputs(extractor, intent, mode, obs_text):
    user_text = extractor._build_user_text(intent, mode, obs_text)
    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    text = extractor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = extractor.processor(text=[text], padding=True, return_tensors="pt")
    return {k: v.to(extractor.model.device) for k, v in inputs.items()}


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def lev_norm(a, b):
    return _levenshtein_token(a, b) / max(1, max(len(a), len(b)))


def is_json_valid(text):
    s = text.strip()
    return s.startswith("{") or s.startswith('"')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    p.add_argument("--alphas", default=",".join(map(str, DEFAULT_ALPHAS)))
    p.add_argument("--max-new-tokens", type=int, default=15)
    p.add_argument("--limit", type=int, default=2, help="N tasks (smoke=2, full=24)")
    p.add_argument("--tier", default="strong")
    p.add_argument("--min-free-vram-gb", type=float, default=0.0)
    args = p.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    d = np.load(NPZ, allow_pickle=True)
    H = d["hidden_states"]
    ml = d["mode_labels_str"]

    # Precompute direction per patcher layer: layers[L] hook output ↔ npz[L+1]
    directions = {}
    for L in layers:
        v = H[ml == "phantom_som"][:, L + 1, :].mean(0) - H[ml == "dom"][:, L + 1, :].mean(0)
        directions[L] = torch.tensor(v)
        logger.info(f"layer {L}: npz idx {L+1}, ||v|| = {float(np.linalg.norm(v)):.4f}")

    manifest = json.loads(MANIFEST.read_text())
    tasks = manifest[args.tier][:args.limit]
    steps = manifest.get("steps", [2, 5])

    extractor = HiddenStateExtractor(min_free_vram_gb=args.min_free_vram_gb)
    patcher = ActivationPatcher(extractor.model, extractor.processor)
    logger.info(f"model loaded; n_layers={patcher.n_layers}")
    logger.info(f"sweep {len(tasks)} tasks × {len(steps)} steps × {len(layers)} layers × {len(alphas)} α "
                 f"+ 2 baselines = {len(tasks)*len(steps)*(len(layers)*len(alphas)+2)} generations")

    per_task = []
    for t in tasks:
        tid = int(t["task_id"])
        intent = t["intent"]
        for step in steps:
            obs_path = ARCHIVE / f"classifieds_task_{tid}" / f"step_{step:03d}" / "observation_dom.txt"
            if not obs_path.exists():
                logger.warning(f"missing {obs_path}; skip")
                continue
            obs_text = obs_path.read_text(encoding="utf-8")
            som_marks_text = build_som_marks(obs_text)
            dom_inputs = build_inputs(extractor, intent, "dom", obs_text)
            psom_inputs = build_inputs(extractor, intent, "phantom_som", som_marks_text)

            dom_gen = patcher.model.generate(**dom_inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                              return_dict_in_generate=True, use_cache=True)
            dom_tokens = dom_gen.sequences[0, dom_inputs["input_ids"].shape[1]:].cpu().tolist()
            dom_text = extractor.processor.tokenizer.decode(dom_tokens, skip_special_tokens=True)

            psom_gen = patcher.model.generate(**psom_inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                                return_dict_in_generate=True, use_cache=True)
            psom_tokens = psom_gen.sequences[0, psom_inputs["input_ids"].shape[1]:].cpu().tolist()
            psom_text = extractor.processor.tokenizer.decode(psom_tokens, skip_special_tokens=True)
            logger.info(f"  task={tid} step={step} | dom: {dom_text!r}")
            logger.info(f"  task={tid} step={step} | psom: {psom_text!r}")

            per_layer = {}
            for L in layers:
                per_alpha = {}
                for alpha in alphas:
                    st_tokens = patcher.steered_generate(
                        layer_idx=L, direction=directions[L], alpha=alpha,
                        max_new_tokens=args.max_new_tokens, **dom_inputs,
                    ).cpu().tolist()
                    st_text = extractor.processor.tokenizer.decode(st_tokens, skip_special_tokens=True)
                    o_dom = jaccard(st_tokens, dom_tokens)
                    o_psom = jaccard(st_tokens, psom_tokens)
                    per_alpha[str(alpha)] = {
                        "steered_text": st_text,
                        "overlap_dom": o_dom, "overlap_psom": o_psom,
                        "lev_dom": lev_norm(st_tokens, dom_tokens),
                        "lev_psom": lev_norm(st_tokens, psom_tokens),
                        "shifted_toward_psom": o_psom > o_dom,
                        "json_valid": is_json_valid(st_text),
                        "first_token_psom_match": (len(st_tokens) > 0 and len(psom_tokens) > 0 and st_tokens[0] == psom_tokens[0]),
                    }
                    logger.info(f"    L{L:02d} α={alpha:>4.1f}: shift={o_psom > o_dom} json={is_json_valid(st_text)} "
                                 f"odom={o_dom:.2f} opsom={o_psom:.2f} → {st_text!r}")
                per_layer[str(L)] = per_alpha

            per_task.append({
                "task_id": tid, "step": step,
                "dom_text": dom_text, "psom_text": psom_text,
                "per_layer": per_layer,
            })

            # Incremental save
            OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
            OUT_JSON.write_text(json.dumps({
                "config": {"layers": layers, "alphas": alphas, "tier": args.tier,
                            "max_new_tokens": args.max_new_tokens},
                "results": per_task,
            }, indent=2))

    # Aggregate per (layer, alpha)
    agg = {}
    for L in layers:
        for alpha in alphas:
            cells = []
            for r in per_task:
                v = r["per_layer"][str(L)][str(alpha)]
                cells.append(v)
            completeness = float(np.mean([c["shifted_toward_psom"] for c in cells]))
            selectivity = float(np.mean([c["json_valid"] for c in cells]))
            # HDMI reliability metric (Khorasani et al. 2026, arXiv:2605.07631):
            # harmonic mean penalizes "shift target but break structure" failure mode
            hmean = 2 * completeness * selectivity / (completeness + selectivity + 1e-9) if (completeness + selectivity) > 0 else 0.0
            agg[f"L{L:02d}_a{alpha}"] = {
                "n": len(cells),
                "mean_overlap_dom": float(np.mean([c["overlap_dom"] for c in cells])),
                "mean_overlap_psom": float(np.mean([c["overlap_psom"] for c in cells])),
                "completeness": completeness,       # shifted_toward_psom rate
                "selectivity": selectivity,           # json_valid rate
                "reliability": hmean,                  # HDMI harmonic mean
                "shifted_rate": completeness,         # alias for backward compat
                "json_valid_rate": selectivity,
                "first_token_psom_match_rate": float(np.mean([c["first_token_psom_match"] for c in cells])),
            }
    final = {
        "config": {"layers": layers, "alphas": alphas, "tier": args.tier,
                    "max_new_tokens": args.max_new_tokens,
                    "direction_norms": {str(L): float(directions[L].norm()) for L in layers}},
        "aggregate": agg, "results": per_task,
    }
    OUT_JSON.write_text(json.dumps(final, indent=2))
    logger.info(f"final → {OUT_JSON}")

    write_md(final, OUT_MD, layers, alphas)


def write_md(d, out, layers, alphas):
    lines = ["# Stage 4 Method 4.4 v2: Layer × α Sweep", ""]
    lines.append(f"**Config**: tier={d['config']['tier']}, n_tasks×steps={len(d['results'])}, max_new_tokens={d['config']['max_new_tokens']}")
    lines.append(f"**Direction norms per layer**: " + ", ".join(f"L{k}={v:.2f}" for k, v in d['config']['direction_norms'].items()))
    lines.append("")

    lines.append("## HDMI Reliability — harmonic mean (completeness × selectivity)")
    lines.append("")
    lines.append("Following Khorasani et al. 2026 (arXiv:2605.07631): reliability = 2·c·s/(c+s).")
    lines.append("Penalizes \"shift target but break envelope\" failure mode. Higher = better.")
    lines.append("")
    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
    for L in layers:
        row = [f"L{L:02d}"]
        for a in alphas:
            row.append(f"**{d['aggregate'][f'L{L:02d}_a{a}']['reliability']:.2f}**")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Completeness (shifted-toward-P-SoM rate: overlap_psom > overlap_dom)")
    lines.append("")
    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
    for L in layers:
        row = [f"L{L:02d}"]
        for a in alphas:
            row.append(f"{d['aggregate'][f'L{L:02d}_a{a}']['completeness']:.0%}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Selectivity (JSON envelope valid rate: steered output still starts with `{`)")
    lines.append("")
    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
    for L in layers:
        row = [f"L{L:02d}"]
        for a in alphas:
            row.append(f"{d['aggregate'][f'L{L:02d}_a{a}']['selectivity']:.0%}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Token overlap to DOM baseline (1.0 = identical, 0 = different)")
    lines.append("")
    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
    for L in layers:
        row = [f"L{L:02d}"]
        for a in alphas:
            row.append(f"{d['aggregate'][f'L{L:02d}_a{a}']['mean_overlap_dom']:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Token overlap to P-SoM baseline")
    lines.append("")
    lines.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
    lines.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
    for L in layers:
        row = [f"L{L:02d}"]
        for a in alphas:
            row.append(f"{d['aggregate'][f'L{L:02d}_a{a}']['mean_overlap_psom']:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    out.write_text("\n".join(lines) + "\n")
    print(f"summary → {out}")


if __name__ == "__main__":
    main()
