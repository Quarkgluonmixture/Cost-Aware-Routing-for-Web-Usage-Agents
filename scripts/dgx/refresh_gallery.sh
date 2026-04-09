#!/usr/bin/env bash
# Quick refresh: annotate screenshots + regenerate gallery HTML.
# Usage: bash scripts/dgx/refresh_gallery.sh [RUN_DIR]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_DIR}/.venv/bin/python"

# Default: latest B0 or B1 run
RUN_DIR="${1:-$(ls -1dt "${REPO_DIR}"/results/visualwebarena/phase1/B[01]_* 2>/dev/null | head -1)}"

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "Usage: bash scripts/dgx/refresh_gallery.sh [RUN_DIR]" >&2
  exit 1
fi

echo "Annotating: ${RUN_DIR}"
"${PYTHON}" "${REPO_DIR}/scripts/annotate_screenshots.py" --run-dir "${RUN_DIR}"

echo "Generating gallery..."
"${PYTHON}" "${REPO_DIR}/scripts/generate_gallery.py" --run-dir "${RUN_DIR}"
