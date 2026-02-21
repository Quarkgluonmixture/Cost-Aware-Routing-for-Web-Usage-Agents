#!/bin/bash
set -e

# 1. Setup Environment
source scripts/sagemaker_setup.sh

# 2. Configuration
# You can override config values using env vars or by modifying the yaml
# Here we ensure we use the default.yaml which now points to Qwen/Qwen3-VL-4B-Instruct
CONFIG_FILE="configs/default.yaml"

# 3. Run VWA
# Assuming run_vwa_batch.py or similar is the entry point
# Adjust the script and arguments as needed for your specific experiment
echo "Starting VWA experiment..."
python scripts/run_vwa_batch.py --config $CONFIG_FILE

echo "Experiment completed."
