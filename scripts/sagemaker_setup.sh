#!/bin/bash
set -e

# Setup script for SageMaker / Cloud environment
# This ensures all dependencies and data are ready

echo "=== SageMaker Setup ==="

# 1. Install gdown for alternative download method
echo "Installing gdown..."
pip install gdown

# 2. Download Docker Images
# Use environment variable DOCKER_DOWNLOAD_METHOD to choose download method
# Options: "docker_pull" (default) or "gdown"
DOCKER_DOWNLOAD_METHOD="${DOCKER_DOWNLOAD_METHOD:-docker_pull}"

echo "Using download method: $DOCKER_DOWNLOAD_METHOD"

# Shopping Image
if ! docker images | grep -q "shopping_final_0712"; then
    echo "Shopping image not found locally."
    if [ "$DOCKER_DOWNLOAD_METHOD" = "docker_pull" ]; then
        echo "Downloading Shopping Docker image via docker pull..."
        docker pull webarenaimages/shopping_final_0712
    elif [ "$DOCKER_DOWNLOAD_METHOD" = "gdown" ]; then
        echo "Downloading Shopping Docker image via gdown..."
        gdown --id <GDRIVE_ID_SHOPPING> -O shopping_final_0712.tar
        docker load < shopping_final_0712.tar
        rm shopping_final_0712.tar
    else
        echo "Unknown download method: $DOCKER_DOWNLOAD_METHOD"
        exit 1
    fi
else
    echo "Shopping image exists."
fi

# Forum Image
if ! docker images | grep -q "postmill-populated-exposed-withimg"; then
    echo "Forum image not found locally."
    if [ "$DOCKER_DOWNLOAD_METHOD" = "docker_pull" ]; then
        echo "Downloading Forum Docker image via docker pull..."
        docker pull webarenaimages/postmill-populated-exposed-withimg
    elif [ "$DOCKER_DOWNLOAD_METHOD" = "gdown" ]; then
        echo "Downloading Forum Docker image via gdown..."
        gdown --id <GDRIVE_ID_FORUM> -O postmill-populated-exposed-withimg.tar
        docker load < postmill-populated-exposed-withimg.tar
        rm postmill-populated-exposed-withimg.tar
    else
        echo "Unknown download method: $DOCKER_DOWNLOAD_METHOD"
        exit 1
    fi
else
    echo "Forum image exists."
fi

# 3. Source the robust setup script (handles repo cloning and data download)
# Skip Docker image downloads in setup_vwa.sh since we've already handled them
SKIP_DOCKER_IMAGES=1 bash scripts/setup_vwa.sh

# 4. Install Python Dependencies
echo "Installing Python dependencies..."
pip install -r external/visualwebarena/requirements.txt

# Install specific requirements for Qwen3-VL and other agents
# Force upgrade specific packages to avoid conflicts
# Note: bitsandbytes is only needed for quantized models (4bit/8bit)
# For non-quantized models, we only need transformers, accelerate, and qwen_vl_utils
pip install -U transformers accelerate qwen_vl_utils

# 5. Install Playwright browsers
echo "Installing Playwright browsers..."
playwright install --with-deps chromium

echo "SageMaker setup complete."
