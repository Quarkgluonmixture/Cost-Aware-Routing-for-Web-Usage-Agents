"""Environment snapshot for paper-grade run provenance.

Captures: torch / transformers / Python / git / hostname / GPU compute caps /
HuggingFace model **actually-loaded** SHA + evaluator code SHA. Dumped to
<out_path> as JSON.

A1.16 fixes (2026-05-16) — see master_bug_catalog B-238..B-240:
  - B-238 (P0-4, 3-AI overlap): HF SHA now reads `huggingface_hub.scan_cache_dir()`
    for the **actually loaded** revision, with `HfApi().model_info()` registry
    HEAD recorded separately for divergence detection. Pre-fix: only registry
    HEAD captured → paper §3 "model SHA pinned at launch" mismatched actual
    loaded version when HF rolled main between download and snapshot.
  - B-239 (P1-1): gated models (gemma-3-4b-it) require `HF_TOKEN` env. Pre-fix
    `HfApi()` anonymous → `GatedRepoError` → silent "unavailable". Now hard-fail
    on missing token + paper-baseline gated model. `HF_TOKEN_REQUIRED_MODELS` env
    declares which models trigger the gate (default = `DEFAULT_GATED_MODELS`).
  - B-240 (P1-2, 3-AI overlap): evaluator combined_sha256 now (a) fails loud on
    MISSING files (raises) (b) canonical-sorts the file list (c) uses path-aware
    sentinel-delimited hash form (rel_path \0 byte_len \0 content \0). Pre-fix:
    MISSING silently skipped (`continue`) → reviewer without `git submodule init`
    got a valid-looking hash from remaining files. List reorder also changed
    the hash even when bytes unchanged.
  - B-242 (P1-3, gemini-unique OOB): evaluator scope expanded to include
    `configs/exp_v2_*.yaml` + `p79/utils/*.py` (eval-adjacent) + inline
    `pip freeze --all` snapshot. Pre-fix scope: 4 files only → YAML `max_steps`
    or reward weight change altered SR but `combined_sha256` stayed the same.

Designed to be **fail-loud where it matters** (paper-baseline gated models,
missing evaluator files) and fail-soft otherwise. Errors list still records
non-critical failures.

Usage (standalone):
    HF_TOKEN=hf_xxx python3 scripts/provenance/snapshot_env.py results/<run_dir>/env_snapshot.json

Usage (programmatic, hooked into run_experiment.py):
    from scripts.provenance.snapshot_env import capture_env_snapshot
    capture_env_snapshot(run_dir / "env_snapshot.json")

Output schema (paper §3 / Appendix D quotable fields):
    {
      "captured_at": "...",
      "host": "...", "platform": "...", "python_version": "...",
      "torch": {...}, "libraries": {...},
      "models": {
        "Qwen/Qwen3-VL-4B-Instruct": {
          "loaded_revision": "<HF SHA actually in local cache>",  # PRIMARY
          "registry_head": "<HF SHA on main right now>",
          "loaded_from_cache": true|false,
          "divergence": "match"|"runner_used_stale_cache"|"unknown"
        }
      },
      "evaluator_code": {
        "combined_sha256": "<canonical path-aware hash>",
        "per_file_sha256": {...},
        "files": [...],
        "schema_version": "2026-05-16-canonical-v2"
      },
      "pip_freeze_lock": "<full pip freeze --all output>",
      "git": {...},
      "errors": []
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Files whose content materially affects scoring (paper-grade evaluator SHA).
# A1.16 P1-3 (B-242): scope expanded beyond 4-file evaluator core.
EVALUATOR_SOURCE_FILES = sorted([
    # Core scoring / SR aggregation
    "p79/experiment/analysis.py",      # Pareto / scored_task_count / analyze_run
    "p79/experiment/environment.py",   # VwaEvaluator wrapper
    "p79/experiment/metrics.py",       # aggregate_condition_metrics (SR roll-up)
    # External evaluator (B-91 empty-prediction guard)
    "external/visualwebarena/evaluation_harness/helper_functions.py",
    "external/visualwebarena/evaluation_harness/evaluators.py",  # string_match / url_match / program_html
    # Eval-adjacent utilities (B-242 scope expansion)
    "p79/utils/auth_refresh.py",       # auth gate decides which tasks ran NOT-LOGGED-IN
])

# Configs whose values affect SR baselines (max_steps, reward weights,
# observation_mode toggles). Globbed at capture time.
EVALUATOR_CONFIG_GLOB = "configs/exp_v2_*.yaml"

logger = logging.getLogger("snapshot-env")

# B-119 (2026-05-15): default models cover all 3 baselines (B0 = proxy API,
# no HF SHA; B1 = Qwen3-VL-4B-Instruct local; B2 = Gemma3-VL local).
DEFAULT_MODELS = [
    "Qwen/Qwen3-VL-4B-Instruct",        # B1
    "google/gemma-3-4b-it",             # B2 (added 2026-05-14 cross-family matched-capability control)
]

# B-239 (A1.16 P1-1): gated models trigger HF_TOKEN hard-fail. List is paper-baseline
# critical models; non-paper models still get _safe + warn.
DEFAULT_GATED_MODELS = [
    "google/gemma-3-4b-it",
]


def _safe(fn, default=None, errors=None, label=""):
    try:
        return fn()
    except Exception as e:
        if errors is not None:
            errors.append(f"{label}: {type(e).__name__}: {e}")
        return default


def _loaded_revision_from_cache(model_id: str) -> tuple[str | None, str | None]:
    """B-238 A1.16 P0-4: read locally-cached HF revision.

    Returns (loaded_sha, source) where source ∈ {'cache', 'no_cache_entry', 'unavailable'}.
    Falls back to None if huggingface_hub.scan_cache_dir() can't be imported or repo missing.
    """
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == model_id:
                # repo.revisions is a frozenset of CachedRevisionInfo;
                # pick the most-recently-modified (HF cache typically has one per model)
                revs = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
                if revs:
                    return revs[0].commit_hash, "cache"
                return None, "no_revisions"
        return None, "no_cache_entry"
    except ImportError:
        return None, "huggingface_hub_not_installed"
    except Exception as e:
        return None, f"scan_cache_error:{type(e).__name__}"


def _registry_head_revision(model_id: str, token: str | None) -> str | None:
    """B-238 A1.16 P0-4: read HF registry current main SHA (for divergence detection)."""
    from huggingface_hub import HfApi
    info = HfApi(token=token).model_info(model_id)
    return info.sha


def _capture_model_revisions(
    models: list[str],
    gated_models: list[str],
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    """B-238 + B-239 A1.16: capture both cache and registry SHA per model."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    result: dict[str, dict[str, Any]] = {}

    for model_id in models:
        is_gated = model_id in gated_models
        # B-239 P1-1 hard-fail: gated paper-baseline model + no token → bail before
        # silent "unavailable" entry contaminates paper §3.
        if is_gated and not hf_token:
            msg = (
                f"hf:{model_id}: GATED model + HF_TOKEN unset. "
                "Set HF_TOKEN=hf_xxx (with accepted ToS) or remove from DEFAULT_GATED_MODELS."
            )
            errors.append(msg)
            logger.error(msg)
            result[model_id] = {
                "loaded_revision": None,
                "registry_head": None,
                "loaded_from_cache": False,
                "divergence": "gated_no_token",
                "_critical_error": msg,
            }
            continue

        loaded_sha, loaded_source = _loaded_revision_from_cache(model_id)
        registry_sha = _safe(
            lambda: _registry_head_revision(model_id, hf_token),
            default=None, errors=errors, label=f"hf:{model_id}:registry",
        )

        if loaded_sha and registry_sha:
            divergence = "match" if loaded_sha == registry_sha else "runner_used_stale_cache"
        elif loaded_sha and not registry_sha:
            divergence = "registry_unavailable"
        elif registry_sha and not loaded_sha:
            divergence = "no_local_cache"  # runner hasn't loaded this model yet
        else:
            divergence = "unknown"

        result[model_id] = {
            "loaded_revision": loaded_sha,       # PRIMARY (what runner actually used)
            "registry_head": registry_sha,        # SECONDARY (for drift detection)
            "loaded_from_cache": loaded_sha is not None,
            "loaded_source": loaded_source,
            "divergence": divergence,
        }

    return result


def _evaluator_combined_sha(
    repo_root: Path,
    extra_config_files: list[Path],
    errors: list[str],
) -> dict[str, Any]:
    """B-240 + B-242 A1.16: fail-loud on MISSING + canonical path-aware hash.

    Form: sha256( for each file in sorted(files): rel_path \0 byte_len \0 content \0 )
    Includes EVALUATOR_SOURCE_FILES + extra_config_files (sorted union).
    """
    # Canonical union — sort EVALUATOR_SOURCE_FILES + relative paths of configs
    rel_paths = sorted(set(EVALUATOR_SOURCE_FILES) | {
        str(cfg.relative_to(repo_root)) for cfg in extra_config_files
    })

    h = hashlib.sha256()
    per_file: dict[str, dict[str, Any]] = {}

    for rel_path in rel_paths:
        f = repo_root / rel_path
        if not f.exists():
            # B-240 A1.16: fail-loud. Reviewer without `git submodule init` should
            # get an explicit error, not a silently-computed hash from remaining files.
            msg = f"evaluator_code: required source missing: {rel_path}"
            errors.append(msg)
            raise FileNotFoundError(msg + " (paper-grade: snapshot aborts on incomplete evaluator)")
        content = f.read_bytes()
        # Canonical hash form: path + byte_len + content (path-aware,
        # rename-detect, size-detect, content-detect)
        h.update(rel_path.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(len(content)).encode("ascii"))
        h.update(b"\x00")
        h.update(content)
        h.update(b"\x00")
        per_file[rel_path] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }

    return {
        "combined_sha256": h.hexdigest(),
        "per_file": per_file,
        "files_count": len(rel_paths),
        "schema_version": "2026-05-16-canonical-v2",  # B-240 + B-242 fix marker
    }


def _pip_freeze_lock(errors: list[str]) -> str:
    """B-242 A1.16: inline pip freeze --all for full env reproducibility."""
    return _safe(
        lambda: subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            stderr=subprocess.DEVNULL, timeout=30,
        ).decode().strip(),
        default="<pip-freeze-failed>", errors=errors, label="pip-freeze",
    )


def capture_env_snapshot(
    out_path: Path | str,
    models: list[str] | None = None,
    gated_models: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    repo_root = Path(__file__).resolve().parents[2]

    snap: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "host": _safe(lambda: subprocess.check_output(["hostname"]).decode().strip(),
                      default="unknown", errors=errors, label="hostname"),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "schema_version": "2026-05-16-a1.16",  # B-238..B-242 fix marker
    }

    # torch
    def _torch_info():
        import torch
        return {
            "version": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "compute_caps": [
                list(torch.cuda.get_device_capability(i))
                for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0)
            ],
            "device_names": [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0)
            ],
        }
    snap["torch"] = _safe(_torch_info, default={}, errors=errors, label="torch")

    # transformers / qwen-vl-utils
    def _lib_versions():
        out = {}
        for lib in ["transformers", "qwen_vl_utils", "huggingface_hub", "numpy", "scikit_learn"]:
            try:
                mod = __import__(lib.replace("-", "_"))
                out[lib] = getattr(mod, "__version__", "unknown")
            except ImportError:
                out[lib] = None
        return out
    snap["libraries"] = _safe(_lib_versions, default={}, errors=errors, label="libraries")

    # HuggingFace model revisions (B-238 + B-239 A1.16):
    # Loaded SHA (from local cache) is PRIMARY; registry HEAD captured for drift.
    # Gated models without HF_TOKEN raise an `errors` entry but don't crash unless
    # the model is paper-baseline (DEFAULT_GATED_MODELS — runner will fail later
    # at from_pretrained anyway, so we just surface it loud here).
    models = models or DEFAULT_MODELS
    gated_models = gated_models or DEFAULT_GATED_MODELS
    snap["models"] = _capture_model_revisions(models, gated_models, errors)

    # Evaluator code SHA (B-240 + B-242 A1.16): canonical + scope-expanded.
    config_files = sorted((repo_root).glob(EVALUATOR_CONFIG_GLOB))
    try:
        snap["evaluator_code"] = _evaluator_combined_sha(repo_root, config_files, errors)
    except FileNotFoundError as e:
        # Paper-grade fail-loud: incomplete evaluator → exit non-zero so caller
        # knows the snapshot is invalid. But still write what we have so debugging
        # is easier.
        snap["evaluator_code"] = {"error": str(e), "incomplete": True}
        errors.append(f"evaluator_code: FATAL: {e}")

    # B-242: inline pip freeze for env reproducibility
    snap["pip_freeze_lock"] = _pip_freeze_lock(errors)

    # Git
    def _git_info():
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode()
        return {"commit": commit, "branch": branch, "dirty": bool(status.strip()), "status": status if status else None}
    snap["git"] = _safe(_git_info, default={"unavailable": True}, errors=errors, label="git")

    # GPU dump (NVML if available)
    def _gpu_info():
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        return [line.strip() for line in out.splitlines()]
    snap["nvidia_smi"] = _safe(_gpu_info, default=[], errors=errors, label="nvidia-smi")

    if extra:
        snap["extra"] = extra
    snap["errors"] = errors

    out_path.write_text(json.dumps(snap, indent=2))

    # B-239 P1-1 hard-fail: any `_critical_error` in models → exit nonzero so
    # paper-grade launch aborts before producing partially-valid snapshot.
    critical_model_errors = [
        m for m, info in snap["models"].items()
        if isinstance(info, dict) and info.get("_critical_error")
    ]
    if critical_model_errors:
        logger.error(
            f"FATAL: {len(critical_model_errors)} paper-baseline model(s) failed SHA capture: "
            f"{critical_model_errors}. See `errors` field for details. Snapshot written but invalid."
        )

    logger.info(f"Env snapshot → {out_path} (errors: {len(errors)})")
    return snap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_path", help="Output JSON path")
    p.add_argument("--model", action="append", default=None,
                   help="Override default model list (repeat for multiple)")
    p.add_argument("--gated", action="append", default=None,
                   help="Override default gated-model list (these require HF_TOKEN)")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero on any paper-baseline gated model SHA failure (B-239)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    snap = capture_env_snapshot(args.out_path, models=args.model, gated_models=args.gated)
    print(json.dumps(snap, indent=2))

    # B-239 strict mode: paper-grade Phase 1a launch wraps this with --strict so
    # missing HF_TOKEN aborts before runner spawn.
    if args.strict:
        critical = [m for m, info in snap.get("models", {}).items()
                    if isinstance(info, dict) and info.get("_critical_error")]
        if critical:
            print(f"\n[FATAL --strict] {len(critical)} critical model(s): {critical}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
