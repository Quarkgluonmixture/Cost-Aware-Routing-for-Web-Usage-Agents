#!/bin/bash
set -e

# Setup script for SageMaker / Cloud environment
# This ensures all dependencies and data are ready

echo "=== SageMaker Setup ==="

# 1. Source the robust setup script (handles repo cloning and data download)
# We need to install gdown first to ensure the setup script works for downloads
pip install gdown

# Run the setup script
# We run it with 'bash' explicitly in case execute permissions are weird, 
# but sourcing it would be better if we needed env vars (like conda) to persist.
# However, for a setup script, running it as a subprocess is fine as long as it does the work.
bash scripts/setup_vwa.sh

# 2. Install Python Dependencies
echo "Installing Python dependencies..."
pip install -r external/visualwebarena/requirements.txt

# Install specific requirements for Qwen3-VL and other agents
# Force upgrade specific packages to avoid conflicts
pip install -U transformers accelerate qwen_vl_utils bitsandbytes

# 3. Install Playwright browsers
echo "Installing Playwright browsers..."
playwright install --with-deps chromium

echo "SageMaker setup complete."
