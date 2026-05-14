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
import os
from pathlib import Path

# B-81h workaround (笔记 §117, commit fda1414): force SDPA math backend so the
# script runs on any GPU architecture. PyTorch's flash + memory-efficient SDPA
# backends only have bf16 cutlass kernels for sm_80+ (A100/H100). On V100
# (sm_70) Myriad nodes the dispatcher raises "cutlassF: no kernel found to
# launch!" instead of falling back. Math backend always works (~2-3x slower
# but correct on any GPU). Opt back in via FORCE_MATH_SDP=0.
if os.environ.get("FORCE_MATH_SDP", "1") != "0":
    try:
        import torch as _torch_for_sdp_setup
        _torch_for_sdp_setup.backends.cuda.enable_flash_sdp(False)
        _torch_for_sdp_setup.backends.cuda.enable_mem_efficient_sdp(False)
        _torch_for_sdp_setup.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

import numpy as np

from p79.mechanistic.extract_hidden_states import HiddenStateExtractor, IMAGE_MAX_SIZE_DEFAULT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [stage4] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ALL_6_MODES = ["dom", "phantom_text", "phantom_prompt", "phantom_som", "som", "vision"]


def build_som_marks(obs_text: str, max_marks: int = 200) -> str:
    """Canonical [SOM_MARKS] builder — delegates to the single source of truth.

    master bug B-82 fix (2026-05-14): the prior local implementation called
    `_extract_text_marks` directly and DROPPED the `_options_map` recovery
    pass, so dropdown `[OPTIONS]` lines (present in 66-81% of archive obs and
    in the production agent SoM) were silently missing from the v2 NPZ. Now
    delegates to `p79.experiment.som.build_som_text_from_obs_text`, byte-
    identical to the production agent SoM text path.
    """
    from p79.experiment.som import build_som_text_from_obs_text
    return build_som_text_from_obs_text(obs_text, max_marks=max_marks)


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
    parser.add_argument(
        "--model-revision",
        default="ebb281ec70b05090aa6165b016eac8ec08e71b17",
        help="HF revision SHA. Must match Stage 2B / agent extraction (Bug 5 fix).",
    )
    parser.add_argument("--modes", nargs="+", default=ALL_6_MODES,
                        help="modes to extract (default: all 6)")
    parser.add_argument(
        "--tier", choices=["strong", "reverse", "all"], default="strong",
        help="Filter archive by manifest tier (Bug 1 fix). Default strong "
             "matches Stage 2/3 patching tier. Use 'all' to ignore manifest "
             "and reproduce legacy lexicographic-glob behavior (NOT recommended).",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Pipeline audit P0-2 fix (2026-05-13): post-extraction grid check "
             "raises SystemExit if (task × step × mode) cells are missing. "
             "Pass --allow-partial to override and ship ragged NPZ anyway. "
             "Without this flag, silent per-mode failures from earlier in the "
             "loop will abort the script BEFORE writing NPZ.",
    )
    args = parser.parse_args()

    archive_dir = Path(args.archived_run_dir)
    if not archive_dir.exists():
        raise SystemExit(f"archive dir missing: {archive_dir}")

    # Bug 1 fix (2026-05-12, /codex-stress methodology audit v2): previous
    # implementation used `sorted(archive_dir.glob(...))` lexicographic
    # selection, ignoring `manifest.json` tier buckets, so the "24 strong-
    # tier" claim in paper §5 was not what the code ran when archives
    # contained mixed strong + reverse tasks. Now load tier from manifest;
    # fall back to legacy behavior only when --tier=all is explicit.
    manifest_path = archive_dir / "manifest.json"
    tier_task_ids: set[int] | None = None
    if args.tier != "all" and manifest_path.exists():
        try:
            manifest = json.load(open(manifest_path))
            tier_task_ids = {int(item["task_id"]) for item in manifest.get(args.tier, [])
                             if "task_id" in item}
            logger.info(f"Manifest tier '{args.tier}': {len(tier_task_ids)} task IDs")
        except Exception as e:
            logger.warning(f"failed to parse manifest tier '{args.tier}': {e}")
            tier_task_ids = None
    if tier_task_ids is not None and not tier_task_ids:
        raise SystemExit(
            f"Manifest contains no tasks under tier '{args.tier}'. "
            "Use --tier=all to bypass tier filter (legacy behavior, NOT recommended)."
        )

    task_dirs = sorted(archive_dir.glob(f"{args.site}_task_*"))
    selected = []
    skipped_off_tier = 0
    for td in task_dirs:
        tid = int(td.name.rsplit("_", 1)[1])
        if tier_task_ids is not None and tid not in tier_task_ids:
            skipped_off_tier += 1
            continue
        if all((td / f"step_{s:03d}" / "observation_dom.txt").exists() and
               (td / f"step_{s:03d}" / "screenshot_annotated.png").exists()
               for s in args.steps):
            selected.append((tid, td))
        if len(selected) >= args.n_tasks:
            break
    logger.info(f"Selected {len(selected)} tasks (target {args.n_tasks}); "
                f"skipped {skipped_off_tier} off-tier")
    if not selected:
        raise SystemExit("no archived tasks selected; check --site/--steps/--tier/--archived-run-dir")
    if len(selected) < args.n_tasks and not args.allow_partial:
        raise SystemExit(
            f"selected only {len(selected)} tasks, target was {args.n_tasks}. "
            "This would ship a smaller-than-claimed NPZ. Pass --allow-partial to override."
        )

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

    extractor = HiddenStateExtractor(
        model_path=args.model_path,
        model_revision=args.model_revision,
    )
    logger.info(f"Model loaded (revision pinned: {args.model_revision[:12]}...)")

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

    # Pipeline audit P0-2 fix (2026-05-13): post-extraction grid check.
    # Previous behavior: per-mode/per-step failures logged at ERROR but
    # NPZ silently shipped with ragged (task × step × mode) coverage.
    # Downstream cosine/logit-lens/steering analyses became uninterpretable
    # without warning. Now: compute expected grid, diff against actual,
    # raise SystemExit unless --allow-partial was passed.
    expected_grid = []
    for tid, _ in selected:
        for step in args.steps:
            for mode in args.modes:
                expected_grid.append((int(tid), int(step), mode))
    actual_grid = list(zip(
        (int(t) for t in all_tids),
        (int(s) for s in all_steps),
        all_modes,
    ))
    missing = sorted(set(expected_grid) - set(actual_grid))
    extra = sorted(set(actual_grid) - set(expected_grid))
    n_expected = len(expected_grid)
    n_actual = len(actual_grid)
    duplicate = n_actual != len(set(actual_grid))
    if missing or extra or duplicate or n_actual != n_expected:
        msg = (
            f"P0-2 grid check FAIL: expected {n_expected} extractions "
            f"(tasks={len(selected)} × steps={len(args.steps)} × modes={len(args.modes)}), "
            f"got {n_actual}. missing={len(missing)} extra={len(extra)}. "
            f"duplicate={duplicate}. "
            f"First 5 missing: {missing[:5]}. "
        )
        if args.allow_partial:
            logger.warning(msg + "Proceeding (--allow-partial).")
        else:
            raise SystemExit(msg + "Pass --allow-partial to override.")
    else:
        logger.info(f"P0-2 grid check OK: {n_actual}/{n_expected} extractions complete")

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

    # Provenance sidecar (added 2026-05-12 after /codex-stress methodology
    # audit v2: previously only the .npz array was written, with no command,
    # git SHA, model revision, archive path, tier, selected task IDs, or
    # formatter hash. All Method 4.2 / Exp 1 / Exp 3 / Exp 5 analyses
    # consume this NPZ, so provenance traceability is paper-grade required.
    import hashlib
    import subprocess
    import sys
    sidecar = out.with_suffix(".provenance.json")
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_sha = "unknown"
    try:
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
        ).decode().strip())
    except Exception:
        git_dirty = None
    # Hash of the build_som_marks source so future audits can verify the
    # formatter has not silently drifted. Includes the function source +
    # the imported _extract_text_marks source for full byte-identity check.
    formatter_src = build_som_marks.__code__.co_consts
    try:
        from p79.experiment import som as _som_mod
        upstream_src = open(_som_mod.__file__, "rb").read()
        formatter_hash = hashlib.sha256(
            (repr(formatter_src) + upstream_src.decode("utf-8", errors="replace")).encode()
        ).hexdigest()
    except Exception:
        formatter_hash = "unknown"
    provenance = {
        "command": " ".join(sys.argv),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "model_path": args.model_path,
        "model_revision": args.model_revision,
        "archive_dir": str(archive_dir.resolve()),
        "tier": args.tier,
        "tier_task_ids_from_manifest": (
            sorted(tier_task_ids) if tier_task_ids is not None else None
        ),
        "selected_task_ids": [tid for tid, _ in selected],
        "n_tasks_target": args.n_tasks,
        "n_tasks_selected": len(selected),
        "modes": args.modes,
        "steps": args.steps,
        "formatter_hash": formatter_hash,
        "formatter_source_module": "p79.experiment.som._extract_text_marks",
        "npz_path": str(out.resolve()),
        "n_examples_saved": len(all_hs),
        "hidden_state_shape": list(H.shape),
        # Pipeline audit P0-2 fix (2026-05-13): grid check status. Future
        # audits can verify NPZ completeness from provenance alone.
        "grid_check": {
            "n_expected": n_expected,
            "n_actual": n_actual,
            "n_missing": len(missing),
            "n_extra": len(extra),
            "missing_first_20": [list(m) for m in missing[:20]],
            "allow_partial": bool(args.allow_partial),
            "status": "OK" if not (missing or extra) else (
                "PARTIAL (--allow-partial)" if args.allow_partial else "FAIL"
            ),
        },
    }
    sidecar.write_text(json.dumps(provenance, indent=2, default=str))
    logger.info(f"Provenance → {sidecar}")


if __name__ == "__main__":
    main()
