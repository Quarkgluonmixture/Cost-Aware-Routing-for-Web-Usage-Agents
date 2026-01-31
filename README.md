# Project 79: Cost-Aware Routing for Web Usage Agents

Research codebase for VisualWebArena (VWA) agents with local Qwen3-VL-4B inference.
Focused on memory-constrained environments (WSL2, RTX 4060 8GB).

## Environment Setup

### 1. Conda Environment

```bash
conda create -n p79_ai python=3.10 -y
conda activate p79_ai
```

### 2. Install Dependencies

```bash
pip install -e .
```

**Note:** For 4-bit quantization, `bitsandbytes` is required. Ensure you have CUDA drivers installed in WSL2.

### 3. VWA Setup (TODO)

To use the real environment, you must install VisualWebArena manually (follow their repo instructions).
This project provides a `VWAWrapper` that defaults to a `dry_run` mode if VWA is missing.

## Usage

### Configuration

Edit `configs/default.yaml` to change model path, quantization settings, or logging parameters.

### Run Single Episode (Dry Run)

```bash
python scripts/run_episode.py --task_id test_0 --instruction "Find a red apple"
```

### Run Batch

```bash
python scripts/run_batch.py --tasks_file data/tasks/sample_tasks.jsonl
```

### Summarize Results

```bash
python scripts/summarize_results.py --log_dir logs
```

### Cookbook: Web Browsing Agent Loop

Run a single episode with Qwen2-VL Agent on VisualWebArena (VWA). This script demonstrates the full loop:
Observation -> Agent -> Action -> Environment -> Observation.

```bash
python scripts/run_one_vwa_episode.py \
    --task_config external/visualwebarena/config_files/vwa/test_classifieds.raw.json \
    --task_id 0 \
    --model_path "Qwen/Qwen2-VL-7B-Instruct" \
    --headless
```

**Features:**
- Loads Qwen2-VL-7B-Instruct (4-bit quantization).
- Wraps VWA environment with `VWAWrapper`.
- **ID-based actions**: Agent outputs element IDs matched against the Accessibility Tree (obs.text).
- Logs trajectory to `episode_{task_id}.jsonl`.

## Project Structure

- `p79/agents`: Qwen3-VL agent logic, strict JSON schema enforcement.
- `p79/envs`: Wrapper for VWA (handles `dry_run` and real VWA integration).
- `p79/policies`: Routing policy placeholders.
- `p79/logging`: Structured JSONL logging.
- `data/tasks`: Task definitions.
- `scripts/`: Execution and analysis scripts.

## Checklist for User

- [x] **Install VisualWebArena**: Clone and install VWA dependencies (including Playwright).
- [ ] **Update Config**: Set `env.dry_run: false` in `configs/default.yaml` once VWA is ready.
- [ ] **Model Weights**: Download Qwen model weights and update `model.path` in `configs/default.yaml`.
- [x] **Implement VWA Hooks**: `p79/envs/vwa_wrapper.py` implements the real `reset` and `step` methods calling the VWA environment.
