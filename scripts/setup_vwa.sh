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
    conda activate p79_ai 2>/dev/null || echo "Warning: Could not activate 'p79_ai' environment. Proceeding with current environment."
else
    echo "Conda environment '$CONDA_DEFAULT_ENV' is active."
fi

# 2. Check Hugging Face Login
echo "Checking Hugging Face authentication..."

# Check for token file first (works even if huggingface-cli command is not available)
HF_TOKEN_FILE="$HOME/.huggingface/token"
if [ -f "$HF_TOKEN_FILE" ]; then
    echo "Hugging Face token file found at $HF_TOKEN_FILE"
    echo "Authentication check passed."
else
    # Try huggingface-cli as fallback
    if ! command -v huggingface-cli &> /dev/null; then
        echo "huggingface-cli not found. Installing huggingface_hub..."
        pip install -q "huggingface_hub"
    fi

    if ! command -v huggingface-cli &> /dev/null; then
        echo "================================================================"
        echo "ERROR: Hugging Face authentication required but not configured."
        echo "The WebArena datasets require authentication."
        echo ""
        echo "Please follow these steps:"
        echo "1. Create a Hugging Face account at https://huggingface.co/join"
        echo "2. Generate an Access Token (Read permissions) at https://huggingface.co/settings/tokens"
        echo "3. Create the token file manually:"
        echo "   mkdir -p ~/.huggingface"
        echo "   echo 'YOUR_TOKEN_HERE' > ~/.huggingface/token"
        echo "4. IMPORTANT: Visit https://huggingface.co/datasets/webarena/Shopping and accept the terms/conditions if required."
        echo "================================================================"
        exit 1
    fi

    # Check if user is logged in by running whoami
    if ! huggingface-cli whoami &> /dev/null; then
        echo "================================================================"
        echo "ERROR: You are not logged in to Hugging Face."
        echo "The WebArena datasets require authentication."
        echo ""
        echo "Please follow these steps:"
        echo "1. Create a Hugging Face account at https://huggingface.co/join"
        echo "2. Generate an Access Token (Read permissions) at https://huggingface.co/settings/tokens"
        echo "3. Run the following command in your terminal and paste your token:"
        echo "   huggingface-cli login"
        echo "4. IMPORTANT: Visit https://huggingface.co/datasets/webarena/Shopping and accept the terms/conditions if required."
        echo "================================================================"
        exit 1
    else
        echo "Logged in to Hugging Face."
    fi
fi

# 3. Clone VisualWebArena if missing
VWA_DIR="external/visualwebarena"
if [ ! -d "$VWA_DIR" ]; then
    echo "Cloning VisualWebArena..."
    git clone https://github.com/web-arena-x/visualwebarena.git "$VWA_DIR"
else
    echo "VisualWebArena directory exists."
fi

# 4. Download Large Files for Docker Environment
# We use huggingface-cli download which handles authentication automatically
ENV_DIR="$VWA_DIR/environment_docker"
DATA_DIR="$ENV_DIR/data"
mkdir -p "$DATA_DIR"

echo "Checking and downloading large files..."

# Helper function to download if not exists
download_if_missing() {
    local repo_id=$1
    local filename=$2
    local output_path=$3
    local description=$4
    
    if [ -f "$output_path" ]; then
        echo "$description exists at $output_path."
        return
    fi

    echo "Downloading $description..."
    # Download to current dir then move/load
    huggingface-cli download "$repo_id" "$filename" --repo-type dataset --local-dir . --local-dir-use-symlinks False
    
    if [[ "$filename" == *.tar ]]; then
        # If it's a tar for docker load, we load it then delete
        if [[ "$description" == *"Docker image"* ]]; then
             echo "Loading Docker image from $filename..."
             docker load < "$filename"
             rm "$filename"
        fi
    elif [[ "$filename" == *.tar.gz ]]; then
        # Extract if needed (for classifieds)
        if [[ "$description" == "Classifieds" ]]; then
            tar -xzf "$filename" -C "$ENV_DIR"
            rm "$filename"
        fi
    else
        # Just move to target
        mv "$filename" "$output_path"
    fi
}

# Shopping Image
if ! docker images | grep -q "shopping_final_0712"; then
    echo "Downloading Shopping Docker image..."
    huggingface-cli download webarena/Shopping shopping_final_0712.tar --repo-type dataset --local-dir . --local-dir-use-symlinks False
    docker load < shopping_final_0712.tar
    rm shopping_final_0712.tar
else
    echo "Shopping image exists."
fi

# Forum Image
if ! docker images | grep -q "postmill-populated-exposed-withimg"; then
    echo "Downloading Forum Docker image..."
    huggingface-cli download webarena/Reddit postmill-populated-exposed-withimg.tar --repo-type dataset --local-dir . --local-dir-use-symlinks False
    docker load < postmill-populated-exposed-withimg.tar
    rm postmill-populated-exposed-withimg.tar
else
    echo "Forum image exists."
fi

# Wikipedia ZIM
WIKI_FILE="$DATA_DIR/wikipedia_en_all_maxi_2022-05.zim"
if [ ! -f "$WIKI_FILE" ]; then
    echo "Downloading Wikipedia ZIM file..."
    huggingface-cli download webarena/Wikipedia wikipedia_en_all_maxi_2022-05.zim --repo-type dataset --local-dir . --local-dir-use-symlinks False
    mv wikipedia_en_all_maxi_2022-05.zim "$WIKI_FILE"
else
    echo "Wikipedia ZIM file exists."
fi

# Classifieds
CLASSIFIEDS_DIR="$ENV_DIR/classifieds_docker_compose"
if [ ! -d "$CLASSIFIEDS_DIR" ]; then
    echo "Downloading Classifieds..."
    huggingface-cli download webarena/Classifieds classifieds.tar.gz --repo-type dataset --local-dir . --local-dir-use-symlinks False
    tar -xzf classifieds.tar.gz -C "$ENV_DIR"
    rm classifieds.tar.gz
else
    echo "Classifieds directory exists."
fi

echo "Setup complete."
