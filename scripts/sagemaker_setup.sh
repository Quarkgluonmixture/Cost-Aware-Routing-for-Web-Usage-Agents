#!/bin/bash
set -euo pipefail

# Setup script for SageMaker / Cloud environment
# This ensures all dependencies and data are ready.

echo "=== SageMaker Setup ==="

DOCKER_DOWNLOAD_METHOD="${DOCKER_DOWNLOAD_METHOD:-docker_pull}"
TARGET_DATASET="${TARGET_DATASET:-all}"  # all|shopping|reddit|wikipedia|classifieds
FORCE_PYTHON_SETUP="${FORCE_PYTHON_SETUP:-0}"
SETUP_MARKER=".sagemaker_python_setup_done"

echo "Using download method: $DOCKER_DOWNLOAD_METHOD"
echo "Target dataset: $TARGET_DATASET"

need_dataset() {
    local name="$1"
    if [ "$TARGET_DATASET" = "all" ] || [ "$TARGET_DATASET" = "$name" ]; then
        return 0
    fi
    return 1
}

pull_or_load_image() {
    local image_name="$1"
    local gdown_id="$2"
    local tar_name="$3"

    if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${image_name}:latest$"; then
        echo "Image already exists: ${image_name}:latest"
        return
    fi

    if [ "$DOCKER_DOWNLOAD_METHOD" = "docker_pull" ]; then
        echo "Pulling image: $image_name"
        docker pull "$image_name"
    elif [ "$DOCKER_DOWNLOAD_METHOD" = "gdown" ]; then
        if [ -z "$gdown_id" ]; then
            echo "Missing gdown id for image: $image_name"
            exit 1
        fi
        if ! command -v gdown >/dev/null 2>&1; then
            echo "Installing gdown..."
            pip install gdown
        fi
        echo "Downloading image tar via gdown: $image_name"
        gdown --id "$gdown_id" -O "$tar_name"
        docker load < "$tar_name"
        rm -f "$tar_name"
    else
        echo "Unknown download method: $DOCKER_DOWNLOAD_METHOD"
        exit 1
    fi
}

# Download only required Docker images for the selected dataset.
if need_dataset shopping; then
    pull_or_load_image "webarenaimages/shopping_final_0712" "${GDRIVE_ID_SHOPPING:-}" "shopping_final_0712.tar"
fi

if need_dataset reddit || need_dataset wikipedia; then
    pull_or_load_image "webarenaimages/postmill-populated-exposed-withimg" "${GDRIVE_ID_FORUM:-}" "postmill-populated-exposed-withimg.tar"
fi

# setup_vwa.sh handles repo clone + dataset files (wiki/classifieds).
SETUP_VWA_TARGET_DATASET="$TARGET_DATASET" SKIP_DOCKER_IMAGES=1 bash scripts/setup_vwa.sh

# Install Python dependencies once unless explicitly forced.
if [ ! -f "$SETUP_MARKER" ] || [ "$FORCE_PYTHON_SETUP" = "1" ]; then
    echo "Installing Python dependencies..."
    pip install -r external/visualwebarena/requirements.txt
    pip install -U transformers accelerate qwen_vl_utils
    echo "Installing Playwright browsers..."
    playwright install --with-deps chromium
    date > "$SETUP_MARKER"
else
    echo "Python dependencies already installed (marker: $SETUP_MARKER)."
fi

echo "SageMaker setup complete."
