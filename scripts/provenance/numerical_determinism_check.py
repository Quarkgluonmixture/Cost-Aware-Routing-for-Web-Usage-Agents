"""Cross-machine numerical determinism check for paper §3 disclosure.

Captures hidden-state activations on a fixed (task, step, mode) input + saves
to a tensor file. Run on DGX + A100 + Myriad → use `compare` mode to compute
max-abs / mean-abs / cosine similarity diffs. Paper §3 reports e.g.
"Cross-machine numerical agreement: max |Δh| < 1e-3 at L11."

Why this matters: cuDNN / matmul precision differs by GPU architecture
(DGX GB10 sm_121 vs A100 sm_80). For SR-level claim this is invisible, but
for paper §5 layer-wise activation magnitude quotes (e.g., "L11 hidden norm
≈ 23.4 ± 0.6"), we want to confirm cross-machine agreement before quoting.

Usage:
    # Capture on each machine (typically L11 for paper §5 case study):
    python3 scripts/provenance/numerical_determinism_check.py capture \\
        --task-id 0 --step 2 --mode som \\
        --layer-indices 0 5 11 17 23 29 35 \\
        --out results/provenance/det_check_$(hostname).pt

    # Compare two captures (DGX vs A100):
    python3 scripts/provenance/numerical_determinism_check.py compare \\
        --files results/provenance/det_check_spark-9ea3.pt \\
                results/provenance/det_check_a100-condense.pt \\
        --out results/provenance/det_diff.json

Output (compare mode):
    {
      "files": [...],
      "machines": ["spark-9ea3", "a100-condense"],
      "per_layer": [
        {"layer": 0, "max_abs_diff": 1.2e-5, "mean_abs_diff": 3.4e-7, "cosine_sim": 0.99999},
        ...
      ],
      "summary": {"max_layer_diff": 1.5e-3, "all_layers_pass": true},
      "captured_at": "..."
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("det-check")


def cmd_capture(args):
    """Capture hidden states on this machine."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from p79.mechanistic.extract_hidden_states import HiddenStateExtractor
    from scripts.mechanistic.run_stage2b_continuation_pilot import build_inputs, build_som_marks

    archived_dir = Path(args.archived_run_dir)
    artifacts_dir = next(c / "artifacts" for c in archived_dir.iterdir() if c.is_dir() and (c / "artifacts").is_dir())
    step_dir = artifacts_dir / f"{args.site}_task_{args.task_id}" / f"step_{args.step:03d}"
    obs_text = (step_dir / "observation_dom.txt").read_text()
    screenshot = str(step_dir / "screenshot_annotated.png")
    som_marks = build_som_marks(obs_text)

    # Need intent — load from task config
    import json as _json
    task_cfg_path = Path("external_code/visualwebarena/config_files") / args.site / f"{args.task_id}.json"
    if not task_cfg_path.exists():
        # Fallback to results-side cached intent if available
        task_cfg_path = artifacts_dir.parent / "episodes" / f"task_{args.task_id}" / "summary.json"
    if task_cfg_path.exists():
        intent = _json.loads(task_cfg_path.read_text()).get("intent", "<intent unavailable>")
    else:
        intent = "<intent unavailable>"

    extractor = HiddenStateExtractor(model_path=args.model_path, min_free_vram_gb=0.0)
    inputs = build_inputs(extractor, intent, args.mode, som_marks, screenshot if args.mode in ("som",) else None)

    # Forward pass capturing all hidden states (output_hidden_states=True)
    with torch.no_grad():
        outputs = extractor.model(**inputs, output_hidden_states=True, return_dict=True)
    all_hs = outputs.hidden_states  # tuple of (n_layers+1) tensors, each (B, T, D)

    # Save selected layers' last-token hidden state (paper §5 quotable)
    capture = {
        "machine": __import__("subprocess").check_output(["hostname"]).decode().strip(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "task_id": args.task_id,
        "step": args.step,
        "mode": args.mode,
        "intent": intent,
        "n_layers": len(all_hs),
        "layers": {},
        "torch_version": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_compute_cap": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    }
    for L in args.layer_indices:
        if L >= len(all_hs):
            logger.warning(f"Layer {L} out of range (n_layers={len(all_hs)}), skip")
            continue
        h_last = all_hs[L][0, -1, :].detach().cpu().float()  # (D,)
        capture["layers"][str(L)] = {
            "tensor": h_last,
            "norm": float(h_last.norm().item()),
            "mean": float(h_last.mean().item()),
            "std": float(h_last.std().item()),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(capture, out)
    logger.info(f"Captured {len(capture['layers'])} layers → {out}")
    logger.info(f"Machine: {capture['machine']} ({capture['gpu_name']}, sm_{''.join(map(str, capture['gpu_compute_cap']))})")


def cmd_compare(args):
    """Compare two captures."""
    import torch
    if len(args.files) != 2:
        logger.error("Need exactly 2 capture files for pairwise compare")
        sys.exit(1)

    cap_a = torch.load(args.files[0], weights_only=False)
    cap_b = torch.load(args.files[1], weights_only=False)

    if cap_a["task_id"] != cap_b["task_id"] or cap_a["step"] != cap_b["step"] or cap_a["mode"] != cap_b["mode"]:
        logger.error(f"Inputs mismatch: {cap_a['task_id']}/{cap_a['step']}/{cap_a['mode']} vs {cap_b['task_id']}/{cap_b['step']}/{cap_b['mode']}")
        sys.exit(1)

    common_layers = sorted(set(cap_a["layers"].keys()) & set(cap_b["layers"].keys()), key=int)
    logger.info(f"Common layers: {common_layers}")

    per_layer = []
    for L in common_layers:
        ha = cap_a["layers"][L]["tensor"].float()
        hb = cap_b["layers"][L]["tensor"].float()
        diff = (ha - hb).abs()
        cos_sim = float(torch.nn.functional.cosine_similarity(ha.unsqueeze(0), hb.unsqueeze(0)).item())
        per_layer.append({
            "layer": int(L),
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
            "norm_a": float(ha.norm().item()),
            "norm_b": float(hb.norm().item()),
            "cosine_sim": cos_sim,
        })

    max_layer_diff = max(p["max_abs_diff"] for p in per_layer) if per_layer else 0.0
    threshold = args.pass_threshold
    all_pass = max_layer_diff < threshold

    result = {
        "files": [str(f) for f in args.files],
        "machines": [cap_a["machine"], cap_b["machine"]],
        "gpu_a": f"{cap_a['gpu_name']} sm_{''.join(map(str, cap_a['gpu_compute_cap']))}",
        "gpu_b": f"{cap_b['gpu_name']} sm_{''.join(map(str, cap_b['gpu_compute_cap']))}",
        "input": {"task_id": cap_a["task_id"], "step": cap_a["step"], "mode": cap_a["mode"]},
        "per_layer": per_layer,
        "summary": {
            "max_layer_diff": max_layer_diff,
            "max_diff_layer": max(per_layer, key=lambda p: p["max_abs_diff"])["layer"] if per_layer else None,
            "min_cosine_sim": min(p["cosine_sim"] for p in per_layer) if per_layer else None,
            "pass_threshold": threshold,
            "all_layers_pass": all_pass,
        },
        "compared_at": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    logger.info(f"Compare result → {out}")
    logger.info(f"  max |Δh| = {max_layer_diff:.3e} at L{result['summary']['max_diff_layer']} "
                f"(threshold {threshold:.0e}, pass={all_pass})")
    logger.info(f"  min cosine_sim = {result['summary']['min_cosine_sim']:.6f}")
    print(f"\nPaper §3 quotable: 'Cross-machine numerical agreement on Qwen3-VL-4B between "
          f"{cap_a['machine']} ({result['gpu_a']}) and {cap_b['machine']} ({result['gpu_b']}): "
          f"max |Δh| < {max_layer_diff:.0e} across L{common_layers[0]}-L{common_layers[-1]}.'\n")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("capture")
    pc.add_argument("--task-id", type=int, default=0)
    pc.add_argument("--step", type=int, default=2)
    pc.add_argument("--mode", default="som", choices=["som", "phantom_som", "dom"])
    pc.add_argument("--site", default="classifieds")
    pc.add_argument("--layer-indices", type=int, nargs="+", default=[0, 5, 11, 17, 23, 29, 35])
    pc.add_argument("--archived-run-dir",
                    default=str(Path(__file__).resolve().parents[2] / "results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428"))
    pc.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
    pc.add_argument("--out", required=True)

    pcm = sub.add_parser("compare")
    pcm.add_argument("--files", nargs=2, required=True, help="Two capture .pt files")
    pcm.add_argument("--pass-threshold", type=float, default=1e-2,
                     help="max |Δh| threshold for pass/fail (default 1e-2 = paper §3 quotable)")
    pcm.add_argument("--out", required=True)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "capture":
        cmd_capture(args)
    elif args.cmd == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
