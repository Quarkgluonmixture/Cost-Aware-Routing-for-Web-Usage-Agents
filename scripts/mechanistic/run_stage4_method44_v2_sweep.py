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
    # Pipeline audit P0-4 fix (2026-05-13): train/eval split for steering direction.
    # Old script fit direction on ALL 24 tasks then evaluated on SAME 24 → in-sample
    # H-mean inflated. Reviewer-3 demands held-out. Default 16-train / 8-eval at
    # split_seed=20260513 (matches _paired_npz_helpers RNG idiom).
    p.add_argument("--n-train-tasks", type=int, default=16,
                   help="N tasks used to compute direction; rest are held-out eval.")
    p.add_argument("--split-seed", type=int, default=20260513,
                   help="Deterministic train/eval split seed.")
    p.add_argument("--also-report-in-sample", action="store_true", default=True,
                   help="Also run sweep on train tasks for in-sample comparison column.")
    p.add_argument("--no-split", action="store_true", default=False,
                   help="Disable split — direction from ALL tasks, sweep on ALL. Legacy mode.")
    args = p.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    d = np.load(NPZ, allow_pickle=True)
    H = d["hidden_states"]
    ml = d["mode_labels_str"]
    npz_tids = d["task_ids"]  # P0-4: needed for train/eval mask
    # Pipeline audit P1-7 fix (2026-05-13): layer-index convention assertion.
    # H[:, 0, :] = embedding; H[:, L+1, :] = decoder block L output.
    # CRITICAL: this script uses patcher.layers[L] ↔ H[:, L+1, :] (off-by-one
    # vs analysis scripts which index H[:, L, :] directly). E.g., "L17 steering"
    # here means patcher.layers[17] hook = H[:, 18, :], which is decoder block 17
    # output. In cosine_gap / logit_lens / layer_profile scripts, "L17" means
    # H[:, 17, :] = decoder block 16 output. See plan.md §1.4.
    assert H.shape[1] == 37, f"expected 37 layers (embed + 36 blocks), got {H.shape[1]}"

    manifest = json.loads(MANIFEST.read_text())
    all_tasks = manifest[args.tier][:args.limit]
    steps = manifest.get("steps", [2, 5])

    # P0-4: deterministic train/eval split.
    # Permute task_ids list (NOT step list) so paired (tid, step) integrity preserved.
    all_tids_in_manifest = np.array([int(t["task_id"]) for t in all_tasks])
    if args.no_split:
        train_tids = eval_tids = all_tids_in_manifest
        train_tasks = eval_tasks = all_tasks
        logger.warning("--no-split: legacy in-sample mode, direction fits on ALL tasks")
    else:
        rng = np.random.default_rng(seed=args.split_seed)
        perm = rng.permutation(all_tids_in_manifest)
        n_train = min(args.n_train_tasks, len(perm))
        train_tids = perm[:n_train]
        eval_tids = perm[n_train:]
        train_tasks = [t for t in all_tasks if int(t["task_id"]) in set(train_tids)]
        eval_tasks = [t for t in all_tasks if int(t["task_id"]) in set(eval_tids)]
        logger.info(f"P0-4 split (seed={args.split_seed}): "
                     f"train={len(train_tids)} tasks {sorted(train_tids.tolist())} | "
                     f"eval={len(eval_tids)} tasks {sorted(eval_tids.tolist())}")

    # Direction computed ONLY from train rows (held-out eval untouched).
    train_mask = np.isin(npz_tids, train_tids)
    n_train_psom_rows = int(((ml == "phantom_som") & train_mask).sum())
    n_train_dom_rows = int(((ml == "dom") & train_mask).sum())
    logger.info(f"direction fitted on {n_train_psom_rows} P-SoM rows + {n_train_dom_rows} DOM rows "
                 f"(train_tids only)")
    if n_train_psom_rows == 0 or n_train_dom_rows == 0:
        raise SystemExit(
            f"P0-4 FATAL: zero train rows for direction. "
            f"P-SoM={n_train_psom_rows}, DOM={n_train_dom_rows}. "
            f"Check NPZ task_ids overlap with manifest."
        )
    directions = {}
    for L in layers:
        psom_train_rows = (ml == "phantom_som") & train_mask
        dom_train_rows = (ml == "dom") & train_mask
        v = H[psom_train_rows][:, L + 1, :].mean(0) - H[dom_train_rows][:, L + 1, :].mean(0)
        directions[L] = torch.tensor(v)
        logger.info(f"layer {L}: npz idx {L+1}, ||v|| = {float(np.linalg.norm(v)):.4f} (train-only)")

    extractor = HiddenStateExtractor(min_free_vram_gb=args.min_free_vram_gb)
    patcher = ActivationPatcher(extractor.model, extractor.processor)
    logger.info(f"model loaded; n_layers={patcher.n_layers}")

    # Build final output structure incrementally; both eval + in_sample share JSON.
    final = {
        "config": {
            "layers": layers, "alphas": alphas, "tier": args.tier,
            "max_new_tokens": args.max_new_tokens,
            "direction_norms": {str(L): float(directions[L].norm()) for L in layers},
            "split_seed": int(args.split_seed),
            "n_train_tasks": int(args.n_train_tasks),
            "no_split": bool(args.no_split),
            "train_task_ids": sorted(train_tids.tolist()),
            "eval_task_ids": sorted(eval_tids.tolist()),
            "also_report_in_sample": bool(args.also_report_in_sample and not args.no_split),
        },
    }

    # P0-4: held-out eval sweep is the paper-grade headline.
    eval_per_task = run_sweep(eval_tasks, steps, extractor, patcher, directions,
                              layers, alphas, args, label="eval", final=final)
    eval_agg = aggregate_per_task(eval_per_task, layers, alphas)
    final["per_task_eval"] = eval_per_task
    final["aggregate_eval"] = eval_agg
    # Backward-compat aliases (figures + downstream readers expect these keys).
    final["results"] = eval_per_task
    final["aggregate"] = eval_agg

    # Optional in-sample column for reviewer comparison.
    if args.also_report_in_sample and not args.no_split:
        logger.info("--also-report-in-sample: running sweep on train tasks (in-sample)")
        in_sample_per_task = run_sweep(train_tasks, steps, extractor, patcher, directions,
                                       layers, alphas, args, label="in_sample", final=final)
        in_sample_agg = aggregate_per_task(in_sample_per_task, layers, alphas)
        final["per_task_in_sample"] = in_sample_per_task
        final["aggregate_in_sample"] = in_sample_agg

    OUT_JSON.write_text(json.dumps(final, indent=2))
    logger.info(f"final → {OUT_JSON}")

    write_md(final, OUT_MD, layers, alphas)


def run_sweep(tasks, steps, extractor, patcher, directions, layers, alphas, args,
              label: str, final: dict):
    """Run dose-response sweep on given tasks. Incrementally writes JSON checkpoint."""
    logger.info(f"[{label}] sweep {len(tasks)} tasks × {len(steps)} steps × {len(layers)} layers "
                 f"× {len(alphas)} α + 2 baselines = "
                 f"{len(tasks)*len(steps)*(len(layers)*len(alphas)+2)} generations")
    per_task = []
    for t in tasks:
        tid = int(t["task_id"])
        intent = t["intent"]
        for step in steps:
            obs_path = ARCHIVE / f"classifieds_task_{tid}" / f"step_{step:03d}" / "observation_dom.txt"
            if not obs_path.exists():
                logger.warning(f"[{label}] missing {obs_path}; skip")
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
            logger.info(f"  [{label}] task={tid} step={step} | dom: {dom_text!r}")
            logger.info(f"  [{label}] task={tid} step={step} | psom: {psom_text!r}")

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
                    logger.info(f"    [{label}] L{L:02d} α={alpha:>4.1f}: shift={o_psom > o_dom} json={is_json_valid(st_text)} "
                                 f"odom={o_dom:.2f} opsom={o_psom:.2f} → {st_text!r}")
                per_layer[str(L)] = per_alpha

            per_task.append({
                "task_id": tid, "step": step, "split_label": label,
                "dom_text": dom_text, "psom_text": psom_text,
                "per_layer": per_layer,
            })

            # Incremental save under the split-label key (preserves other label data already written).
            OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
            snapshot = dict(final)  # shallow copy
            snapshot[f"per_task_{label}"] = per_task
            snapshot["config_partial"] = True
            OUT_JSON.write_text(json.dumps(snapshot, indent=2))
    return per_task


def aggregate_per_task(per_task, layers, alphas):
    """Aggregate per-(L, α) HDMI reliability + components across all (task, step) cells."""
    agg = {}
    for L in layers:
        for alpha in alphas:
            cells = []
            for r in per_task:
                v = r["per_layer"][str(L)][str(alpha)]
                cells.append(v)
            if not cells:
                agg[f"L{L:02d}_a{alpha}"] = {"n": 0, "completeness": 0.0, "selectivity": 0.0,
                                              "reliability": 0.0, "mean_overlap_dom": 0.0,
                                              "mean_overlap_psom": 0.0, "shifted_rate": 0.0,
                                              "json_valid_rate": 0.0,
                                              "first_token_psom_match_rate": 0.0}
                continue
            completeness = float(np.mean([c["shifted_toward_psom"] for c in cells]))
            selectivity = float(np.mean([c["json_valid"] for c in cells]))
            # HDMI reliability metric (Khorasani et al. 2026, arXiv:2605.07631):
            # harmonic mean penalizes "shift target but break structure" failure mode
            hmean = 2 * completeness * selectivity / (completeness + selectivity + 1e-9) if (completeness + selectivity) > 0 else 0.0
            agg[f"L{L:02d}_a{alpha}"] = {
                "n": len(cells),
                "mean_overlap_dom": float(np.mean([c["overlap_dom"] for c in cells])),
                "mean_overlap_psom": float(np.mean([c["overlap_psom"] for c in cells])),
                "completeness": completeness,
                "selectivity": selectivity,
                "reliability": hmean,
                "shifted_rate": completeness,
                "json_valid_rate": selectivity,
                "first_token_psom_match_rate": float(np.mean([c["first_token_psom_match"] for c in cells])),
            }
    return agg


def _best_cell(agg, layers, alphas):
    """Return (L*, α*, H-mean*) — best HDMI reliability across the grid."""
    best = (-1, -1.0, -1.0)
    for L in layers:
        for a in alphas:
            h = agg[f"L{L:02d}_a{a}"]["reliability"]
            if h > best[2]:
                best = (L, a, h)
    return best


def _table(agg, layers, alphas, metric, fmt=".2f"):
    out = []
    out.append("| Layer \\ α | " + " | ".join(f"α={a}" for a in alphas) + " |")
    out.append("|---|" + "|".join(["---"] * len(alphas)) + "|")
    for L in layers:
        row = [f"L{L:02d}"]
        for a in alphas:
            v = agg[f"L{L:02d}_a{a}"][metric]
            if metric in ("completeness", "selectivity"):
                row.append(f"{v:.0%}")
            else:
                row.append(f"{v:{fmt}}")
        out.append("| " + " | ".join(row) + " |")
    return out


def write_md(d, out, layers, alphas):
    cfg = d["config"]
    has_in_sample = "aggregate_in_sample" in d
    eval_agg = d["aggregate_eval"]
    in_agg = d.get("aggregate_in_sample")

    lines = ["# Stage 4 Method 4.4 v2: Layer × α Sweep", ""]
    lines.append(f"**Config**: tier={cfg['tier']}, max_new_tokens={cfg['max_new_tokens']}")
    if cfg.get("no_split"):
        lines.append("**Split**: DISABLED (legacy in-sample mode, direction fit on ALL tasks).")
    else:
        lines.append(f"**Split**: train/eval (seed={cfg['split_seed']}, n_train={cfg['n_train_tasks']})")
        lines.append(f"- Train task_ids (direction fit on these): `{cfg['train_task_ids']}`")
        lines.append(f"- Eval task_ids (held-out, headline numbers from these): `{cfg['eval_task_ids']}`")
    lines.append(f"**Direction norms per layer (train-fit only)**: " +
                 ", ".join(f"L{k}={v:.2f}" for k, v in cfg['direction_norms'].items()))
    lines.append(f"**N eval cells (task × step)**: {len(d.get('per_task_eval', []))}")
    if has_in_sample:
        lines.append(f"**N in-sample cells (task × step)**: {len(d.get('per_task_in_sample', []))}")
    lines.append("")

    # Hero summary — distance between held-out and in-sample peak
    L_e, a_e, h_e = _best_cell(eval_agg, layers, alphas)
    lines.append("## Hero summary — held-out vs in-sample peak HDMI")
    lines.append("")
    if has_in_sample:
        L_i, a_i, h_i = _best_cell(in_agg, layers, alphas)
        gap = h_i - h_e
        same_cell = (L_e == L_i and abs(a_e - a_i) < 1e-9)
        lines.append(f"- **Held-out best**: L{L_e:02d}, α={a_e}, H-mean={h_e:.2f}")
        lines.append(f"- **In-sample best**: L{L_i:02d}, α={a_i}, H-mean={h_i:.2f}")
        lines.append(f"- **Generalization gap (in_sample − held_out)**: {gap:+.2f} "
                     f"({'same cell' if same_cell else 'different cell'})")
        if gap > 0.10:
            lines.append("")
            lines.append("> ⚠️  **Reviewer-3 flag**: gap > 0.10 suggests direction may be over-fit to "
                         "training cohort. Paper §5.3 should report held-out as headline.")
        elif gap < -0.05:
            lines.append("")
            lines.append("> ✓  Held-out exceeds in-sample (negative gap) — direction transfers "
                         "BETTER to unseen tasks than to fit tasks. Unusual but possible at small N.")
        else:
            lines.append("")
            lines.append("> ✓  Gap ≤ 0.10 — direction generalizes within tolerance. Paper §5.3 hero "
                         "claim survives held-out evaluation.")
    else:
        lines.append(f"- **Eval best**: L{L_e:02d}, α={a_e}, H-mean={h_e:.2f}")
        lines.append("- **In-sample column**: not run (--no-split or --also-report-in-sample False)")
    lines.append("")

    sections = [
        ("HDMI Reliability — harmonic mean (completeness × selectivity)",
         "Following Khorasani et al. 2026 (arXiv:2605.07631): reliability = 2·c·s/(c+s). "
         "Penalizes \"shift target but break envelope\" failure mode. Higher = better.",
         "reliability"),
        ("Completeness (shifted-toward-P-SoM rate: overlap_psom > overlap_dom)", "",
         "completeness"),
        ("Selectivity (JSON envelope valid rate: steered output still starts with `{`)", "",
         "selectivity"),
        ("Token overlap to DOM baseline (1.0 = identical, 0 = different)", "",
         "mean_overlap_dom"),
        ("Token overlap to P-SoM baseline", "", "mean_overlap_psom"),
    ]
    for title, blurb, metric in sections:
        lines.append(f"## {title}")
        lines.append("")
        if blurb:
            lines.append(blurb)
            lines.append("")
        if has_in_sample:
            lines.append("### Held-out (paper-grade headline)")
            lines.append("")
            lines.extend(_table(eval_agg, layers, alphas, metric))
            lines.append("")
            lines.append("### In-sample (training cohort — for reviewer comparison only)")
            lines.append("")
            lines.extend(_table(in_agg, layers, alphas, metric))
            lines.append("")
        else:
            lines.extend(_table(eval_agg, layers, alphas, metric))
            lines.append("")

    out.write_text("\n".join(lines) + "\n")
    print(f"summary → {out}")


if __name__ == "__main__":
    main()
