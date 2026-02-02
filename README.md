# Project 79: Cost-Aware Routing for Web Usage Agents

Research codebase for **VisualWebArena (VWA)** agents with local **Qwen3-VL-4B** inference.
Focused on memory-constrained environments (WSL2, RTX 4060 8GB) and multi-scale model routing.

## 🌟 Key Features

*   **Qwen3-VL Integration**: Uses `Qwen3VLForConditionalGeneration` for state-of-the-art multimodal understanding.
*   **VisualWebArena Wrapper**: A robust `VWAWrapper` (`p79/envs/vwa_wrapper.py`) that bridges modern VLM outputs (coordinates/JSON) to VWA's ID-based action space.
*   **Safety-First Environment**: Custom installation procedure to prevent VWA dependencies from downgrading critical libraries (Transformers).
*   **Cookbook**: Ready-to-run scripts for agent loops and evaluations.

## 🛠️ Environment Setup

To ensure compatibility between Qwen3-VL (requiring new Transformers) and VisualWebArena (often pinning old versions), follow this exact sequence:

### 1. Conda Environment

```bash
conda create -n p79_ai python=3.10 -y
conda activate p79_ai
```

### 2. Core Dependencies (Pinned for Qwen3-VL)

We install these *first* to establish the baseline for the model.

```bash
pip install -U "transformers>=4.57" accelerate "qwen-vl-utils>=0.0.14" bitsandbytes
```

### 3. VisualWebArena (Safe Install)

We install VWA without dependencies to avoid it overwriting our Transformers version, then manually add its runtime requirements.

```bash
# Clone VWA if you haven't
mkdir -p external
cd external
git clone https://github.com/web-arena-x/visualwebarena.git
cd visualwebarena

# Install WITHOUT dependencies
pip install -e . --no-deps

# Install runtime dependencies manually
pip install -U playwright gymnasium lxml beautifulsoup4 numpy pillow requests tqdm

# Install browser kernels
playwright install
cd ../..
```

### 4. Project Package

```bash
pip install -e .
```

## ⚙️ Configuration

1.  **Model Weights**: Ensure you have `Qwen3-VL-4B-Instruct` downloaded locally.
2.  **Edit Config**: Update `configs/default.yaml` with your local path:

```yaml
model:
  path: "/mnt/d/(Gluons)/hf_models/Qwen3-VL-4B-Instruct"  # Update this!
  quantization: "4bit"
  device: "cuda"
```

## 🚀 Quick Start: Cookbook

Run a single episode with the full agent loop (Observation -> Qwen3-VL -> Action -> VWA).

```bash
python scripts/run_one_vwa_episode.py \
    --task_config external/visualwebarena/config_files/vwa/test_classifieds.raw.json \
    --task_id 0 \
    --model_path "/mnt/d/(Gluons)/hf_models/Qwen3-VL-4B-Instruct" \
    --headless
```

**What this does:**
1.  Loads task #0 from the classifieds config.
2.  Initializes `VWAWrapper` (auto-handles environment variables).
3.  Loads Qwen3-VL-4B (4-bit quantized).
4.  Runs the agent loop, validating JSON actions against the Accessibility Tree.
5.  Logs the trajectory to `episode_0.jsonl`.

## 📂 Project Structure

- **`p79/agents`**: Agent logic using `Qwen3VLForConditionalGeneration`. Includes strict JSON schema prompting.
- **`p79/envs`**: `VWAWrapper` for VisualWebArena. Handles observation parsing and action translation.
- **`p79/policies`**: (Placeholder) Cost-aware routing logic.
- **`p79/utils`**: Configuration and logging utilities.
- **`scripts/`**:
    - `run_one_vwa_episode.py`: Main demo script.
    - `smoke_test_vwa.py`: Simple connectivity check.
- **`external/`**: Contains the VisualWebArena submodule.

## ✅ Verification

Run these commands to ensure your environment is healthy:

```bash
# Check Transformers version (should be >= 4.57)
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check VWA Import
export DATASET=visualwebarena
python -c "from browser_env import ScriptBrowserEnv; print('VWA Import OK')"
```
