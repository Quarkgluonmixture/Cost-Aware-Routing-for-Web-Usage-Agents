#!/usr/bin/env python3
"""Stage 4 Method 4.4: Counterfactual Activation Steering.

Validates the (mean_P-SoM − mean_DOM) direction at L17 is CAUSALLY operative:
adding α·v to the DOM forward pass at L17 should pull the generated tokens
toward P-SoM ground truth (not just shift them randomly).

Ports Tool Calling Linear Steerable Circuit (Anonymous 2026 ACL): Qwen3-4B
80-93% tool switch accuracy via mid-layer mean-difference steering.

For each (task, step) in cls strong-tier 24:
  - Generate DOM baseline (no steering)
  - Generate P-SoM baseline (no steering)
  - For α ∈ {0.5, 1.0, 2.0, 5.0}: generate from DOM inputs with L17 += α·v

Metrics per (task, step, α):
  - token_overlap to DOM / to P-SoM (Jaccard)
  - levenshtein_norm to DOM / to P-SoM
  - first_token_match_psom (boolean)

Output:
  - results/mechanistic/stage4_multimode_b1_cls/method44_steering.json
  - docs/checkpoints/stage4_method44_results.md
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# B-81h workaround for V100; harmless on A100
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
from p79.mechanistic.activation_patching import ActivationPatcher, _token_seq_overlap, _levenshtein_token  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [stage4mm44] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz"
ARCHIVE = ROOT / "results/mechanistic/archive_subset_b1_cls"
MANIFEST = ARCHIVE / "manifest.json"
OUT_JSON = ROOT / "results/mechanistic/stage4_multimode_b1_cls/method44_steering.json"
OUT_MD = ROOT / "docs/checkpoints/stage4_method44_results.md"


def build_som_marks(obs_text: str, max_marks=None) -> str:
    """Canonical [SOM_MARKS] builder — delegates to the single source of truth.

    master bug B-82 fix (2026-05-14): prior local impl was a crude AXTree
    line-grep, not production SoM text. Now delegates to the production
    builder. NOTE: this legacy steering script also has an off-by-one layer
    bug (codex Mode B C4) — outputs invalid until both are fixed + re-run.
    """
    from p79.experiment.som import build_som_text_from_obs_text
    return build_som_text_from_obs_text(obs_text, max_marks=max_marks)


def build_inputs(extractor: HiddenStateExtractor, intent: str, mode: str, obs_text: str):
    """No-image inputs (Stage 4 modes that don't need image: dom / phantom_som / phantom_text / phantom_prompt)."""
    user_text = extractor._build_user_text(intent, mode, obs_text)
    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    text = extractor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = extractor.processor(text=[text], padding=True, return_tensors="pt")
    return {k: v.to(extractor.model.device) for k, v in inputs.items()}


def compute_direction(npz_path: Path, layer: int) -> np.ndarray:
    d = np.load(npz_path, allow_pickle=True)
    H = d["hidden_states"]
    ml = d["mode_labels_str"]
    v = H[ml == "phantom_som"][:, layer, :].mean(0) - H[ml == "dom"][:, layer, :].mean(0)
    return v


def jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def lev_norm(a, b) -> float:
    return _levenshtein_token(a, b) / max(1, max(len(a), len(b)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=17, help="Steering target layer")
    p.add_argument("--alphas", default="0.5,1.0,2.0,5.0")
    p.add_argument("--max-new-tokens", type=int, default=15)
    p.add_argument("--min-free-vram-gb", type=float, default=0.0)
    p.add_argument("--tier", default="strong")
    p.add_argument("--limit", type=int, default=None, help="Smoke-test: limit to first N tasks")
    args = p.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]

    logger.info(f"Loading direction from {NPZ} layer={args.layer}")
    v_np = compute_direction(NPZ, args.layer)
    v_norm = float(np.linalg.norm(v_np))
    logger.info(f"Direction norm = {v_norm:.4f} (will be scaled by alpha)")

    logger.info(f"Loading manifest {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text())
    tasks = manifest[args.tier]
    if args.limit:
        tasks = tasks[:args.limit]
    logger.info(f"Loaded {len(tasks)} tasks (tier={args.tier})")

    extractor = HiddenStateExtractor(min_free_vram_gb=args.min_free_vram_gb)
    patcher = ActivationPatcher(extractor.model, extractor.processor)
    logger.info(f"Model loaded; n_layers={patcher.n_layers}")
    v_torch = torch.tensor(v_np)

    steps = manifest.get("steps", [2, 5])
    per_task_results = []

    for ti, t in enumerate(tasks):
        tid = int(t["task_id"])
        intent = t["intent"]
        for step in steps:
            obs_path = ARCHIVE / f"classifieds_task_{tid}" / f"step_{step:03d}" / "observation_dom.txt"
            if not obs_path.exists():
                logger.warning(f"missing {obs_path}; skipping")
                continue
            obs_text = obs_path.read_text(encoding="utf-8")
            som_marks_text = build_som_marks(obs_text)

            dom_inputs = build_inputs(extractor, intent, "dom", obs_text)
            psom_inputs = build_inputs(extractor, intent, "phantom_som", som_marks_text)

            # Baselines
            dom_gen = patcher.model.generate(
                **dom_inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                return_dict_in_generate=True, use_cache=True,
            )
            dom_tokens = dom_gen.sequences[0, dom_inputs["input_ids"].shape[1]:].cpu().tolist()
            dom_text = extractor.processor.tokenizer.decode(dom_tokens, skip_special_tokens=True)

            psom_gen = patcher.model.generate(
                **psom_inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                return_dict_in_generate=True, use_cache=True,
            )
            psom_tokens = psom_gen.sequences[0, psom_inputs["input_ids"].shape[1]:].cpu().tolist()
            psom_text = extractor.processor.tokenizer.decode(psom_tokens, skip_special_tokens=True)

            logger.info(f"  task={tid} step={step}: dom={dom_text!r} | psom={psom_text!r}")

            # Steered generations
            per_alpha = []
            for alpha in alphas:
                st_tokens = patcher.steered_generate(
                    layer_idx=args.layer, direction=v_torch, alpha=alpha,
                    max_new_tokens=args.max_new_tokens, **dom_inputs,
                ).cpu().tolist()
                st_text = extractor.processor.tokenizer.decode(st_tokens, skip_special_tokens=True)
                ovl_dom = jaccard(st_tokens, dom_tokens)
                ovl_psom = jaccard(st_tokens, psom_tokens)
                ld_dom = lev_norm(st_tokens, dom_tokens)
                ld_psom = lev_norm(st_tokens, psom_tokens)
                first_match_psom = (len(st_tokens) > 0 and len(psom_tokens) > 0 and st_tokens[0] == psom_tokens[0])
                per_alpha.append({
                    "alpha": alpha, "steered_text": st_text,
                    "token_overlap_dom": ovl_dom, "token_overlap_psom": ovl_psom,
                    "levenshtein_dom": ld_dom, "levenshtein_psom": ld_psom,
                    "first_token_match_psom": first_match_psom,
                    "shifted_toward_psom": ovl_psom > ovl_dom,
                })
                logger.info(f"    α={alpha:.1f} → {st_text!r} | overlap dom={ovl_dom:.2f} psom={ovl_psom:.2f} | first_token_psom_match={first_match_psom}")

            per_task_results.append({
                "task_id": tid, "step": step,
                "dom_text": dom_text, "psom_text": psom_text,
                "dom_tokens": dom_tokens, "psom_tokens": psom_tokens,
                "per_alpha": per_alpha,
            })

            # Incremental save
            OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
            OUT_JSON.write_text(json.dumps({
                "config": {"layer": args.layer, "alphas": alphas, "tier": args.tier,
                            "max_new_tokens": args.max_new_tokens, "direction_norm": v_norm},
                "results": per_task_results,
            }, indent=2))

    # Aggregate
    agg = {}
    for alpha in alphas:
        rows = [pa for r in per_task_results for pa in r["per_alpha"] if pa["alpha"] == alpha]
        agg[f"alpha_{alpha}"] = {
            "n": len(rows),
            "mean_overlap_dom": float(np.mean([r["token_overlap_dom"] for r in rows])),
            "mean_overlap_psom": float(np.mean([r["token_overlap_psom"] for r in rows])),
            "mean_lev_dom": float(np.mean([r["levenshtein_dom"] for r in rows])),
            "mean_lev_psom": float(np.mean([r["levenshtein_psom"] for r in rows])),
            "first_token_psom_match_rate": float(np.mean([r["first_token_match_psom"] for r in rows])),
            "shifted_toward_psom_rate": float(np.mean([r["shifted_toward_psom"] for r in rows])),
        }

    final = {"config": {"layer": args.layer, "alphas": alphas, "tier": args.tier,
                          "max_new_tokens": args.max_new_tokens, "direction_norm": v_norm},
              "aggregate": agg, "results": per_task_results}
    OUT_JSON.write_text(json.dumps(final, indent=2))
    logger.info(f"final → {OUT_JSON}")

    write_md(final, OUT_MD)


def write_md(d: dict, out: Path) -> None:
    cfg, agg = d["config"], d["aggregate"]
    lines = [
        "# Stage 4 Method 4.4: Counterfactual Activation Steering",
        "",
        f"**Config**: layer L{cfg['layer']:02d}, steering direction = mean(P-SoM) − mean(DOM) at L{cfg['layer']}, ‖v‖={cfg['direction_norm']:.4f}",
        f"**Tier**: {cfg['tier']} cls × steps {{2, 5}} × n_tasks variable",
        f"**Max new tokens**: {cfg['max_new_tokens']}",
        "",
        "## Aggregate per α (does adding α·v to DOM forward shift toward P-SoM?)",
        "",
        "| α | n | mean overlap_DOM | mean overlap_P-SoM | shifted-toward-P-SoM rate | first-token P-SoM match |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in agg.items():
        alpha = k.split("_")[1]
        lines.append(f"| {alpha} | {v['n']} | {v['mean_overlap_dom']:.3f} | {v['mean_overlap_psom']:.3f} | {v['shifted_toward_psom_rate']:.0%} | {v['first_token_psom_match_rate']:.0%} |")
    lines.append("")
    lines.append("Interpretation: if α=0 baseline overlap_DOM = 1.0 + overlap_P-SoM = some baseline,")
    lines.append("then as α↑ overlap_P-SoM should rise + overlap_DOM should fall, monotonically.")
    lines.append("Tool Calling paper (Anonymous 2026 ACL) reports 80-93% tool-switch rate at α=2-3.")
    lines.append("")
    out.write_text("\n".join(lines) + "\n")
    print(f"[stage4mm44] summary → {out}")


if __name__ == "__main__":
    main()
