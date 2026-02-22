#!/bin/bash
set -euo pipefail

# SageMaker Run Script
# Usage:
#   bash scripts/sagemaker_run.sh
#   CONFIG_FILE=configs/exp_shopping.yaml bash scripts/sagemaker_run.sh
#   SKIP_SUMMARY=1 bash scripts/sagemaker_run.sh
#   DOCKER_DOWNLOAD_METHOD=gdown bash scripts/sagemaker_run.sh

echo "=== SageMaker Run ==="

CONFIG_FILE="${CONFIG_FILE:-}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"
LOG_DIR="${LOG_DIR:-logs/sagemaker}"

# Force Docker + cache paths onto SageMaker volume (47GB disk).
SAGEMAKER_ROOT="${SAGEMAKER_ROOT:-/home/ec2-user/SageMaker}"
DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-$SAGEMAKER_ROOT/docker_data}"
export HF_HOME="${HF_HOME:-$SAGEMAKER_ROOT/hf_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SAGEMAKER_ROOT/pip_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TMPDIR="${TMPDIR:-$SAGEMAKER_ROOT/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SAGEMAKER_ROOT/.cache}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" "$TMPDIR" "$XDG_CACHE_HOME" "$LOG_DIR"

setup_docker_data_root() {
    echo "=== Configure Docker data-root ==="
    sudo systemctl stop docker
    sudo mkdir -p "$DOCKER_DATA_ROOT"
    echo "{\"data-root\": \"$DOCKER_DATA_ROOT\"}" | sudo tee /etc/docker/daemon.json >/dev/null
    sudo systemctl start docker
    echo "Docker data-root: $DOCKER_DATA_ROOT"
}

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
    local env_dir="external/visualwebarena/environment_docker"
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
    rm -rf external/visualwebarena/.cache
    rm -f ./*.tar ./*.tar.gz
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
    TARGET_DATASET="$dataset" bash scripts/sagemaker_setup.sh

    set +e
    python scripts/run_vwa_batch.py --config "$cfg" 2>&1 | tee "$log_file"
    rc=${PIPESTATUS[0]}
    set -e

    cleanup_after_dataset "$dataset"
    return $rc
}

setup_docker_data_root
echo "HF_HOME=$HF_HOME"
echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "TMPDIR=$TMPDIR"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"

declare -a configs
if [ -n "$CONFIG_FILE" ]; then
    configs=("$CONFIG_FILE")
else
    configs=(
        "configs/exp_shopping.yaml"
        "configs/exp_reddit.yaml"
        "configs/exp_wikipedia.yaml"
        "configs/exp_classifieds.yaml"
    )
fi

for cfg in "${configs[@]}"; do
    dataset="$(dataset_from_config "$cfg")"
    run_single_config "$cfg" "$dataset"
done

if [ "$SKIP_SUMMARY" = "0" ]; then
    echo ""
    echo "=== Summarizing Results ==="
    python scripts/summarize_results.py --results_dir results --output_dir results_summary
else
    echo "Skipping result summary (SKIP_SUMMARY=1)"
fi

echo ""
echo "Experiment completed."
