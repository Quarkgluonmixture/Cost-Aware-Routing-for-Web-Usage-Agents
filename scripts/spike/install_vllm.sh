#!/bin/bash
# Install vLLM into an ISOLATED venv (.venv-vllm) so the paper-grade HF env
# (.venv: torch 2.11+cu128 / transformers 5.8.1) is never perturbed. Throwaway —
# `rm -rf .venv-vllm` to undo. Writes /tmp/vllm_install.DONE on success,
# /tmp/vllm_install.FAIL on error (Tier-1 markers for the remote monitor).
set -u
cd /home/ubuntu/workspace/p79 || exit 2
rm -f /tmp/vllm_install.DONE /tmp/vllm_install.FAIL
trap 'echo "[install] FAILED"; touch /tmp/vllm_install.FAIL' ERR
set -e

echo "[install] $(date +%H:%M:%S) creating .venv-vllm (python3.10)"
python3.10 -m venv .venv-vllm
.venv-vllm/bin/pip install -q -U pip wheel setuptools
echo "[install] $(date +%H:%M:%S) pip install vllm (this is the long pole)"
.venv-vllm/bin/pip install vllm
echo "[install] $(date +%H:%M:%S) verifying import"
.venv-vllm/bin/python - <<'PY'
import vllm, torch, transformers
print("vllm", vllm.__version__, "| torch", torch.__version__, "| tf", transformers.__version__)
PY
touch /tmp/vllm_install.DONE
echo "[install] $(date +%H:%M:%S) DONE"
