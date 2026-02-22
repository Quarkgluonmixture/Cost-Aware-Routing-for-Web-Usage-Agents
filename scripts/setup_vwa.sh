#!/bin/bash
set -e

# Setup script for VisualWebArena dependencies and data
# Usage: source scripts/setup_vwa.sh

echo "=== VisualWebArena Setup ==="

# 1. Conda Activation
# Try to find conda and activate the environment if not already active
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "Conda environment not active."
    # Common locations for conda
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    fi
    
    # Try activating the environment (adjust name if needed, assuming 'p79_ai' based on lock file)
    # If the user hasn't created it yet, we might need instructions for that, but assuming it exists or we just activate base
    conda activate p79_ai 2>/dev/null || echo "Warning: Could not activate 'p79_ai' environment. Proceeding with current environment."
else
    echo "Conda environment '$CONDA_DEFAULT_ENV' is active."
fi

# 2. Clone VisualWebArena if missing
VWA_DIR="external/visualwebarena"
if [ ! -d "$VWA_DIR" ] || [ -z "$(ls -A $VWA_DIR)" ]; then
    echo "Cloning VisualWebArena..."
    git clone https://github.com/web-arena-x/visualwebarena.git "$VWA_DIR"
else
    echo "VisualWebArena directory exists."
fi

# 3. Download Large Files for Docker Environment
ENV_DIR="$VWA_DIR/environment_docker"
DATA_DIR="$ENV_DIR/data"
mkdir -p "$DATA_DIR"

echo "Checking and downloading large files..."

# Shopping Image
if ! docker images | grep -q "shopping_final_0712"; then
    echo "Downloading Shopping Docker image..."
    # Note: Google Drive direct links are tricky with wget/curl. 
    # Using gdown if available, or instructing user.
    # Since we can't easily automate GDrive downloads without gdown, we'll try to use a direct link workaround or fallback to clear instructions.
    # For now, providing the command the user can run manually if automation fails is safest, 
    # but I'll try to install gdown if possible or just print the critical info if missing.
    
    if ! docker images | grep -q "shopping_final_0712"; then
        echo "Downloading Shopping Docker image..."
        # Using Hugging Face as a more reliable mirror than Google Drive
        wget "https://huggingface.co/datasets/webarena/Shopping/resolve/main/shopping_final_0712.tar?download=true" -O shopping_final_0712.tar
        docker load < shopping_final_0712.tar
        rm shopping_final_0712.tar
    else
        echo "Shopping image exists."
    fi
else
    echo "Shopping image exists."
fi

# Forum Image
if ! docker images | grep -q "postmill-populated-exposed-withimg"; then
    echo "Downloading Forum Docker image..."
    wget "https://huggingface.co/datasets/webarena/Reddit/resolve/main/postmill-populated-exposed-withimg.tar?download=true" -O postmill-populated-exposed-withimg.tar
    docker load < postmill-populated-exposed-withimg.tar
    rm postmill-populated-exposed-withimg.tar
else
    echo "Forum image exists."
fi

# Wikipedia ZIM
WIKI_FILE="$DATA_DIR/wikipedia_en_all_maxi_2022-05.zim"
if [ ! -f "$WIKI_FILE" ]; then
    echo "Downloading Wikipedia ZIM file..."
    wget "https://huggingface.co/datasets/webarena/Wikipedia/resolve/main/wikipedia_en_all_maxi_2022-05.zim?download=true" -O "$WIKI_FILE"
else
    echo "Wikipedia ZIM file exists."
fi

# Classifieds
CLASSIFIEDS_DIR="$ENV_DIR/classifieds_docker_compose"
if [ ! -d "$CLASSIFIEDS_DIR" ]; then
    echo "Downloading Classifieds..."
    wget "https://huggingface.co/datasets/webarena/Classifieds/resolve/main/classifieds.tar.gz?download=true" -O classifieds.tar.gz
    tar -xzf classifieds.tar.gz -C "$ENV_DIR"
    rm classifieds.tar.gz
else
    echo "Classifieds directory exists."
fi

echo "Setup complete."
