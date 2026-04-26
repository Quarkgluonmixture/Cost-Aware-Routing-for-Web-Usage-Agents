#!/usr/bin/env bash
# Quick refresh: annotate screenshots + regenerate gallery HTML.
# Usage: bash scripts/maintenance/refresh_gallery.sh [RUN_DIR]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON="${REPO_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "ERROR: 找不到 Python 解释器" >&2; exit 127
fi

# Default: latest B0 or B1 run
RUN_DIR="${1:-$(ls -1dt "${REPO_DIR}"/results/visualwebarena/phase1/B[01]_* 2>/dev/null | head -1)}"

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "Usage: bash scripts/maintenance/refresh_gallery.sh [RUN_DIR]" >&2
  exit 1
fi

echo "Annotating: ${RUN_DIR}"
"${PYTHON}" "${REPO_DIR}/scripts/maintenance/annotate_screenshots.py" --run-dir "${RUN_DIR}"

echo "Generating gallery..."
"${PYTHON}" "${REPO_DIR}/scripts/maintenance/generate_gallery.py" --run-dir "${RUN_DIR}"
