"""Environment snapshot for paper-grade run provenance.

Captures: torch / transformers / Python / git / hostname / GPU compute caps /
HuggingFace model revision SHA. Dumped to <out_path> as JSON.

Designed to be **fail-soft** — if HF API is unreachable or `git` is unavailable,
records the failure mode in the snapshot rather than crashing the launch.

Usage (standalone):
    python3 scripts/provenance/snapshot_env.py results/<run_dir>/env_snapshot.json

Usage (programmatic, hooked into run_experiment.py):
    from scripts.provenance.snapshot_env import capture_env_snapshot
    capture_env_snapshot(run_dir / "env_snapshot.json")

Output schema (paper §3 / Appendix D quotable fields):
    {
      "captured_at": "2026-05-07T...Z",
      "host": "spark-9ea3",
      "platform": "Linux-6.11.0-aarch64",
      "python_version": "3.12.x",
      "torch": {"version": "2.11.0+cu128", "cuda": "12.8", "compute_caps": [[12,1]]},
      "transformers": {"version": "4.46.x"},
      "models": {"Qwen/Qwen3-VL-4B-Instruct": "<HF revision SHA>"},
      "git": {"commit": "<SHA>", "dirty": false, "branch": "master"},
      "errors": []
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("snapshot-env")

DEFAULT_MODELS = [
    "Qwen/Qwen3-VL-4B-Instruct",
]


def _safe(fn, default=None, errors=None, label=""):
    try:
        return fn()
    except Exception as e:
        if errors is not None:
            errors.append(f"{label}: {type(e).__name__}: {e}")
        return default


def capture_env_snapshot(
    out_path: Path | str,
    models: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    snap: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "host": _safe(lambda: subprocess.check_output(["hostname"]).decode().strip(),
                      default="unknown", errors=errors, label="hostname"),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
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

    # HuggingFace model revisions (paper-grade: pin model SHA at launch)
    models = models or DEFAULT_MODELS
    snap["models"] = {}
    def _hf_revision(model_id):
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_id)
        return info.sha
    for m in models:
        snap["models"][m] = _safe(
            lambda: _hf_revision(m),
            default="unavailable", errors=errors, label=f"hf:{m}"
        )

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
    logger.info(f"Env snapshot → {out_path} (errors: {len(errors)})")
    return snap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_path", help="Output JSON path")
    p.add_argument("--model", action="append", default=None,
                   help="Override default model list (repeat for multiple)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    snap = capture_env_snapshot(args.out_path, models=args.model)
    print(json.dumps(snap, indent=2))


if __name__ == "__main__":
    main()
