#!/bin/bash
set -e

# SageMaker Run Script
# Usage:
#   bash scripts/sagemaker_run.sh                    # Use default docker pull method
#   DOCKER_DOWNLOAD_METHOD=docker_pull bash scripts/sagemaker_run.sh
#   DOCKER_DOWNLOAD_METHOD=gdown bash scripts/sagemaker_run.sh

echo "=== SageMaker Run ==="

# 1. Setup Environment
# DOCKER_DOWNLOAD_METHOD can be set to "docker_pull" (default) or "gdown"
# Example: DOCKER_DOWNLOAD_METHOD=gdown bash scripts/sagemaker_run.sh
bash scripts/sagemaker_setup.sh

# 2. Configuration
# You can override config values using env vars or by modifying the yaml
# Here we ensure we use the default.yaml which now points to Qwen/Qwen3-VL-4B-Instruct
CONFIG_FILE="${CONFIG_FILE:-configs/default.yaml}"

echo "Using config file: $CONFIG_FILE"

# 3. Run VWA
# Assuming run_vwa_batch.py or similar is the entry point
# Adjust the script and arguments as needed for your specific experiment
echo "Starting VWA experiment..."
python scripts/run_vwa_batch.py --config $CONFIG_FILE

echo "Experiment completed."
