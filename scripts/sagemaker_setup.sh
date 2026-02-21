#!/bin/bash
set -e

# Update system packages if necessary (optional, depends on container)
# apt-get update && apt-get install -y xvfb libgbm-dev

# Install dependencies
echo "Installing dependencies..."
pip install -r external/visualwebarena/requirements.txt

# Install specific requirements for Qwen3-VL
# Note: Newer transformers is likely required for Qwen3
pip install -U transformers accelerate qwen_vl_utils bitsandbytes

# Install Playwright browsers
echo "Installing Playwright browsers..."
playwright install --with-deps chromium

echo "Dependencies installed."
