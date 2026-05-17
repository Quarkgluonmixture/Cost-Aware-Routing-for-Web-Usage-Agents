#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VWA_DIR="${PROJECT_DIR}/external/visualwebarena"
ENV_DIR="${VWA_DIR}/environment_docker"
DATA_DIR="${ENV_DIR}/data"

TARGET_DATASET="${SETUP_VWA_TARGET_DATASET:-all}"
SKIP_DOCKER_IMAGES="${SKIP_DOCKER_IMAGES:-0}"
SKIP_HF_CHECK=0
SKIP_CONDA_ACTIVATE=0
PYTHON_BIN=""

usage() {
  cat <<USAGE
Usage: bash scripts/vwa/setup_vwa.sh [options]

Options:
  --target-dataset <list>   all|shopping|reddit|wikipedia|classifieds or comma list
  --skip-docker-images      Skip docker image downloads (dataset files only)
  --skip-hf-check           Skip Hugging Face credential check
  --skip-conda-activate     Do not attempt conda activation
  --python <path>           Explicit python interpreter
  -h, --help                Show this help

Examples:
  bash scripts/vwa/setup_vwa.sh
  bash scripts/vwa/setup_vwa.sh --target-dataset shopping,reddit
  bash scripts/vwa/setup_vwa.sh --target-dataset wikipedia --skip-docker-images
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-dataset)
      TARGET_DATASET="${2:-all}"
      shift 2
      ;;
    --skip-docker-images)
      SKIP_DOCKER_IMAGES=1
      shift
      ;;
    --skip-hf-check)
      SKIP_HF_CHECK=1
      shift
      ;;
    --skip-conda-activate)
      SKIP_CONDA_ACTIVATE=1
      shift
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "No Python interpreter found (.venv/bin/python, python3, python)." >&2
    exit 127
  fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 127
fi

contains_dataset() {
  local needle="$1"
  if [[ "${TARGET_DATASET}" == "all" ]]; then
    return 0
  fi

  local normalized="${TARGET_DATASET// /}"
  IFS=',' read -r -a selected <<< "${normalized}"
  for item in "${selected[@]}"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

ensure_conda() {
  if (( SKIP_CONDA_ACTIVATE == 1 )); then
    return 0
  fi

  if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Conda environment active: ${CONDA_DEFAULT_ENV}"
    return 0
  fi

  if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "/opt/conda/etc/profile.d/conda.sh"
  fi

  if command -v conda >/dev/null 2>&1; then
    conda activate p79_ai >/dev/null 2>&1 || true
  fi
}

check_hf_auth() {
  if (( SKIP_HF_CHECK == 1 )); then
    echo "Skipping Hugging Face auth check (--skip-hf-check)."
    return 0
  fi

  local token_file="$HOME/.huggingface/token"
  if [[ -f "${token_file}" ]]; then
    echo "Hugging Face token detected at ${token_file}"
    return 0
  fi

  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not found and ~/.huggingface/token missing." >&2
    echo "Install huggingface_hub and login, or provide ~/.huggingface/token." >&2
    exit 1
  fi

  if ! huggingface-cli whoami >/dev/null 2>&1; then
    echo "Hugging Face authentication check failed (not logged in)." >&2
    echo "Run: huggingface-cli login" >&2
    exit 1
  fi

  echo "Hugging Face authentication check passed."
}

clone_vwa_if_missing() {
  if [[ -d "${VWA_DIR}" ]] && [[ -n "$(ls -A "${VWA_DIR}" 2>/dev/null || true)" ]]; then
    echo "VisualWebArena already present: ${VWA_DIR}"
    return 0
  fi

  echo "Cloning VisualWebArena..."
  git clone https://github.com/web-arena-x/visualwebarena.git "${VWA_DIR}"
}

python_hf_download() {
  local repo_id="$1"
  local filename="$2"
  "${PYTHON_BIN}" - <<PY
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='${repo_id}', filename='${filename}', repo_type='dataset', local_dir='.')
PY
}

download_shopping_image() {
  if ! contains_dataset "shopping"; then
    return 0
  fi
  if (( SKIP_DOCKER_IMAGES == 1 )); then
    echo "Skipping shopping image (SKIP_DOCKER_IMAGES=1)."
    return 0
  fi
  if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q '^shopping_final_0712:latest$'; then
    echo "Shopping image already exists."
    return 0
  fi

  # HF dataset webarena/Shopping no longer exists (404 RepositoryNotFound,
  # verified 2026-05-14). Use the CMU mirror from environment_docker/README.md
  # (~68GB; wget -c resumes a partial download).
  echo "Downloading shopping image (CMU mirror, ~68GB)..."
  wget -c --tries=3 -O shopping_final_0712.tar \
    "http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar"
  docker load < shopping_final_0712.tar
  rm -f shopping_final_0712.tar
}

download_forum_image() {
  if ! contains_dataset "reddit" && ! contains_dataset "wikipedia"; then
    return 0
  fi
  if (( SKIP_DOCKER_IMAGES == 1 )); then
    echo "Skipping forum image (SKIP_DOCKER_IMAGES=1)."
    return 0
  fi
  if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q '^postmill-populated-exposed-withimg:latest$'; then
    echo "Forum image already exists."
    return 0
  fi

  # HF dataset webarena/Reddit no longer exists — CMU mirror (~53GB).
  echo "Downloading forum image (CMU mirror, ~53GB)..."
  wget -c --tries=3 -O postmill-populated-exposed-withimg.tar \
    "http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar"
  docker load < postmill-populated-exposed-withimg.tar
  rm -f postmill-populated-exposed-withimg.tar
}

download_wikipedia_data() {
  if ! contains_dataset "wikipedia"; then
    return 0
  fi
  mkdir -p "${DATA_DIR}"
  # ZIM version 2025-08 (not VWA upstream 2022-05): P79 queue scripts
  # hardcode WIKIPEDIA_ZIM_VERSION=2025-08 (笔记 §81 fix for Kiwix-newer-than-
  # VWA-config mismatch). Prod (quark Windows docker) also runs 2025-08, so
  # A100 self-host must match — otherwise all wiki URLs return 404.
  local wiki_file="${DATA_DIR}/wikipedia_en_all_maxi_2025-08.zim"
  if [[ -f "${wiki_file}" ]]; then
    echo "Wikipedia ZIM already exists."
    return 0
  fi

  # CMU metis only mirrors 2022-05; pull 2025-08 from kiwix.org canonical.
  echo "Downloading Wikipedia ZIM 2025-08 (kiwix.org, ~95GB)..."
  wget -c --tries=3 -O "${wiki_file}" \
    "https://download.kiwix.org/zim/wikipedia/wikipedia_en_all_maxi_2025-08.zim"
}

download_classifieds_data() {
  if ! contains_dataset "classifieds"; then
    return 0
  fi
  local classifieds_dir="${ENV_DIR}/classifieds_docker_compose"
  if [[ -d "${classifieds_dir}" ]]; then
    echo "Classifieds data already exists."
    return 0
  fi

  # HF dataset webarena/Classifieds no longer exists — archive.org hosts
  # classifieds_docker_compose.zip (a .zip, not .tar.gz; ~25MB).
  echo "Downloading classifieds dataset (archive.org, ~25MB)..."
  wget -c --tries=3 -O classifieds_docker_compose.zip \
    "https://archive.org/download/classifieds_docker_compose/classifieds_docker_compose.zip"
  unzip -o -q classifieds_docker_compose.zip -d "${ENV_DIR}"
  rm -f classifieds_docker_compose.zip
}

main() {
  echo "=== VisualWebArena Setup ==="
  echo "project_dir=${PROJECT_DIR}"
  echo "target_dataset=${TARGET_DATASET}"
  echo "skip_docker_images=${SKIP_DOCKER_IMAGES}"
  echo "python=${PYTHON_BIN}"

  ensure_conda
  check_hf_auth
  clone_vwa_if_missing

  mkdir -p "${DATA_DIR}"

  download_shopping_image
  download_forum_image
  download_wikipedia_data
  download_classifieds_data

  # /stress A1.18-re (B-626 P2-8-B* codex OOB, 2026-05-17): fail-loud if VWA
  # per-task split configs are missing. The 912 gitignored per-task config
  # files (config_files/vwa/test_{site}/{0..N}.json) are derived artifacts and
  # must be regenerated post-clone via `make vwa-generate-configs`. Pre-fix
  # "Setup complete" echoed success even when the substrate was incomplete,
  # producing silent runtime failures during first task launch.
  SUBMODULE_CFG_DIR="${PROJECT_DIR}/external/visualwebarena/config_files/vwa"
  if [ ! -d "${SUBMODULE_CFG_DIR}/test_classifieds" ] || \
     [ ! -d "${SUBMODULE_CFG_DIR}/test_reddit" ] || \
     [ ! -d "${SUBMODULE_CFG_DIR}/test_shopping" ]; then
    echo ""
    echo "⚠️  WARNING: VWA per-task split configs not materialized yet."
    echo "    Expected at: ${SUBMODULE_CFG_DIR}/test_{classifieds,reddit,shopping}/"
    echo "    Run: make vwa-generate-configs"
    echo "    (sets DATASET=visualwebarena + REDDIT/SHOPPING/CLASSIFIEDS env vars + invokes generate_test_data.py)"
    echo ""
  fi

  echo "Setup complete."
}

main "$@"
