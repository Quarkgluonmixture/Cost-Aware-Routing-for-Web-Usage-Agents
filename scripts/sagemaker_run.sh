#!/bin/bash
set -e

# SageMaker Run Script
# Usage:
#   bash scripts/sagemaker_run.sh                    # Run all 4 datasets with default docker pull method
#   DOCKER_DOWNLOAD_METHOD=docker_pull bash scripts/sagemaker_run.sh
#   DOCKER_DOWNLOAD_METHOD=gdown bash scripts/sagemaker_run.sh
#   CONFIG_FILE=configs/exp_shopping.yaml bash scripts/sagemaker_run.sh  # Run single dataset

echo "=== SageMaker Run ==="

# 1. Setup Environment
# DOCKER_DOWNLOAD_METHOD can be set to "docker_pull" (default) or "gdown"
# Example: DOCKER_DOWNLOAD_METHOD=gdown bash scripts/sagemaker_run.sh
bash scripts/sagemaker_setup.sh

# 2. Configuration
# You can override config values using env vars or by modifying the yaml
# Default: run all 4 datasets (shopping, reddit, wikipedia, classifieds)
CONFIG_FILE="${CONFIG_FILE:-}"

# 3. Run VWA
# If CONFIG_FILE is set, run a single dataset
# Otherwise, run all 4 datasets sequentially
if [ -n "$CONFIG_FILE" ]; then
    echo "Running single dataset: $CONFIG_FILE"
    python scripts/run_vwa_batch.py --config $CONFIG_FILE
else
    echo "Running all 4 datasets sequentially..."
    
    # Shopping
    echo "=== Running Shopping Dataset ==="
    python scripts/run_vwa_batch.py --config configs/exp_shopping.yaml
    
    # Reddit
    echo "=== Running Reddit Dataset ==="
    python scripts/run_vwa_batch.py --config configs/exp_reddit.yaml
    
    # Wikipedia
    echo "=== Running Wikipedia Dataset ==="
    python scripts/run_vwa_batch.py --config configs/exp_wikipedia.yaml
    
    # Classifieds
    echo "=== Running Classifieds Dataset ==="
    python scripts/run_vwa_batch.py --config configs/exp_classifieds.yaml
fi

echo "Experiment completed."
