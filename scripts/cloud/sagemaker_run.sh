#!/bin/bash
set -euo pipefail

# SageMaker Run Script
# Usage:
#   bash scripts/cloud/sagemaker_run.sh
#   CONFIG_FILE=legacy/configs/exp_shopping.yaml bash scripts/cloud/sagemaker_run.sh
#   SKIP_SUMMARY=1 bash scripts/cloud/sagemaker_run.sh
#   DOCKER_DOWNLOAD_METHOD=gdown bash scripts/cloud/sagemaker_run.sh

echo "=== SageMaker Run ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

CONFIG_FILE="${CONFIG_FILE:-}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs/sagemaker}"

# Force Docker + cache paths onto SageMaker volume (47GB disk).
SAGEMAKER_ROOT="${SAGEMAKER_ROOT:-/home/ec2-user/SageMaker}"
export HF_HOME="${HF_HOME:-$SAGEMAKER_ROOT/hf_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SAGEMAKER_ROOT/pip_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TMPDIR="${TMPDIR:-$SAGEMAKER_ROOT/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SAGEMAKER_ROOT/.cache}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" "$TMPDIR" "$XDG_CACHE_HOME" "$LOG_DIR"

dataset_from_config() {
    local cfg="$1"
    case "$(basename "$cfg")" in
        *shopping*) echo "shopping" ;;
        *reddit*) echo "reddit" ;;
        *wikipedia*) echo "wikipedia" ;;
        *classifieds*) echo "classifieds" ;;
        *)
            echo "Cannot infer dataset from config: $cfg"
            return 1
            ;;
    esac
}

cleanup_after_dataset() {
    local dataset="$1"
    local env_dir="${REPO_DIR}/external/visualwebarena/environment_docker"
    echo "=== Destroy environment for dataset: $dataset ==="

    local container_ids
    container_ids="$(docker ps -aq || true)"
    if [ -n "$container_ids" ]; then
        docker stop $container_ids || true
    fi
    docker system prune -a --volumes -f || true

    # Remove dataset-specific local data so next dataset re-downloads as needed.
    case "$dataset" in
        wikipedia)
            rm -rf "$env_dir/data/wikipedia_en_all_maxi_2022-05.zim"
            ;;
        classifieds)
            rm -rf "$env_dir/classifieds_docker_compose"
            ;;
        *)
            ;;
    esac

    # Remove transient artifacts.
    rm -rf "${REPO_DIR}/external/visualwebarena/.cache"
    rm -f "${REPO_DIR}/"*.tar "${REPO_DIR}/"*.tar.gz 2>/dev/null || true
}

run_single_config() {
    local cfg="$1"
    local dataset="$2"
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local log_file="$LOG_DIR/${dataset}_${ts}.log"
    local rc=0

    echo ""
    echo "=== Running dataset: $dataset ($cfg) ==="
    TARGET_DATASET="$dataset" bash "${REPO_DIR}/scripts/cloud/sagemaker_setup.sh"

    set +e
    if command -v python3 >/dev/null 2>&1; then
        python3 "${REPO_DIR}/scripts/run_experiment.py" --config "$cfg" 2>&1 | tee "$log_file"
    else
        python "${REPO_DIR}/scripts/run_experiment.py" --config "$cfg" 2>&1 | tee "$log_file"
    fi
    rc=${PIPESTATUS[0]}
    set -e

    cleanup_after_dataset "$dataset"
    return $rc
}

echo "HF_HOME=$HF_HOME"
echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "TMPDIR=$TMPDIR"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"

declare -a configs
if [ -n "$CONFIG_FILE" ]; then
    configs=("$CONFIG_FILE")
else
    configs=(
        "legacy/configs/exp_shopping.yaml"
        "legacy/configs/exp_reddit.yaml"
        "legacy/configs/exp_classifieds.yaml"
    )
fi

for cfg in "${configs[@]}"; do
    dataset="$(dataset_from_config "$cfg")"
    run_single_config "$cfg" "$dataset"
done

if [ "$SKIP_SUMMARY" = "0" ]; then
    echo ""
    echo "Unified v2 analyzer uses single-run input: scripts/analyze_experiment.py --run_dir <RUN_DIR>"
    echo "Skipping legacy multi-run summary step."
else
    echo "Skipping result summary (SKIP_SUMMARY=1)"
fi

echo ""
echo "Experiment completed."
