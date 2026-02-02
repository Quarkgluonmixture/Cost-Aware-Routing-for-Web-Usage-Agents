# Repository Status Report

**Last Updated:** 2026-02-02
**Project:** Cost-Aware Routing for Web Usage Agents (p79)

## 1. Executive Summary

This repository hosts a local web agent framework designed to interact with the **VisualWebArena (VWA)** benchmark. It leverages the **Qwen3-VL-4B** multimodal model to perceive web pages (screenshots and accessibility trees) and execute actions (click, type, scroll).

The current state represents a functional "vertical slice":
- **Environment**: Fully containerized local VWA sites (Shopping, Reddit, etc.) are running.
- **Agent**: A Qwen3-VL based agent is implemented and integrated.
- **Pipeline**: The end-to-end execution loop (observation -> inference -> action -> environment step) is verified and operational.

## 2. Functional Capabilities

### A. Environment & Simulation
- **Local Hosting**: All VWA target websites (Classifieds, Shopping, Reddit, Wikipedia, Homepage) are hosted locally via Docker, ensuring reproducibility and zero external network dependency for the simulation.
- **Headless Browser**: Uses Playwright (via `webarena` wrapper) to interact with websites headlessly, suitable for batch evaluation.
- **State Management**: Automatic authentication and state restoration using pre-generated cookies (in `.auth/`).

### B. Agent Architecture
- **Model**: Qwen3-VL-4B-Instruct, loaded with 4-bit quantization to fit consumer-grade GPUs (e.g., RTX 4060 8GB).
- **Multimodal Input**: Processes high-resolution screenshots and text-based Accessibility Trees (AXTree) simultaneously.
- **Action Generation**: Outputs structured JSON actions (e.g., `{"action_type": "click", "element_id": 42}`).
- **Robustness**: Includes fallback mechanisms for parsing invalid JSON and validating element IDs against the accessibility tree.

### C. Execution Pipeline
- **Batch Runner**: `scripts/run_vwa_batch.py` handles massive parallel evaluation with sharding, auto-resume, and inline evaluation.
- **Single Episode Runner**: `scripts/run_one_vwa_episode.py` allows running specific tasks by ID from raw configuration files.
- **Dynamic Configuration**: Automatically replaces placeholders (e.g., `__SHOPPING__`) in task configs with active local URLs.
- **Logging**: detailed execution logs (steps, actions, observations, rewards) are saved in JSONL format (e.g., `episode_0.jsonl`).

## 3. Completed Work (Changelog)

### Experiment Infrastructure
- [x] **Batch Pipeline**: Implemented `scripts/run_vwa_batch.py` for stable, reproducible experiments.
    - Supports task sharding (for parallel execution).
    - Auto-resumes from last checkpoint.
    - Logs detailed run metadata (prompt version, seeds, etc.).
    - **Optimization**: Fixed OOM issues by ensuring model is loaded only once per process.
- [x] **Official Evaluation**: Integrated VisualWebArena's native evaluation harness.
    - `scripts/eval_vwa_runs.py` computes true Success Rate (SR) using DOM/URL matching rules from VWA.
    - Patched `p79/envs/vwa_wrapper.py` to record raw actions required by the evaluator.
- [x] **Configuration**: Created `configs/exp_shopping_small.yaml` for a controlled baseline experiment (Shopping tasks 0-19).
    - Updated to force `env.dry_run: false` and increase max steps for real environment runs.

### Recent Debug Runs
- **latest_run_1770063000**: Completed 5 tasks with full environment load and evaluation; SR 0.00%, average steps 30. Task 3 failed with `list index out of range`.
- **debug_1770036200**: Full 5-task run; SR 0.00%, average steps 30. Task 3 failed with `list index out of range`.

### Environment Setup
- [x] **Docker Integration**: Verified and fixed Docker execution permissions in WSL.
- [x] **VWA Deployment**: Deployed and health-checked all 5 VWA websites.
- [x] **Configuration**: Created `scripts/vwa_env.sh` for centralized environment variable management.
- [x] **Data Preparation**: Generated authentication states (`.auth/`) required for Shopping and Reddit tasks.

### Codebase Improvements
- [x] **Dependency Resolution**:
    - Installed system-level dependencies (`gcc`, `g++`) via conda to support `triton`/`bitsandbytes`.
    - Resolved complex Python dependency conflicts by upgrading `torch`, `torchvision`, `transformers`, and `accelerate` to versions compatible with Qwen2-VL.
    - Installed `webarena` and `p79` packages in editable mode.
- [x] **Bug Fixes**:
    - **Agent Logic**: Fixed `TypeError` in agent step function by correctly accessing observation objects.
    - **Image Handling**: Patched `p79/envs/vwa_wrapper.py` to convert numpy arrays from Playwright to PIL images, fixing a crash in `qwen_vl_utils`.
    - **Click Normalization**: Added pixel-to-normalized coordinate conversion for mouse clicks.
    - **Config Loading**: Fixed import errors in `run_one_vwa_episode.py` and removed dependencies on missing `Config` class.
    - **URL Handling**: Implemented dynamic URL placeholder replacement to support raw VWA task configs.
    - **Prompting**: Strengthened system prompt to encourage scroll/search before finishing.
    - **Guardrails**: Added no-progress detection to force scroll when actions repeat.

## 4. Pending / Placeholder Items
- **Router Policy**: The `p79/policies/router.py` exists but contains placeholder logic. The "Cost-Aware Routing" feature (dynamically selecting between models based on complexity/cost) is not yet implemented.

## 5. Visualization & Debugging

To inspect the agent's behavior, you have three options:

### A. Screenshot Playback (Recommended)
Every run automatically saves step-by-step screenshots to the `visualization/` directory.
- **Location**: `visualization/episode_{task_id}/step_{step_idx}.png`
- **Usage**: Open these images in VS Code to visually verify what the agent saw at each step.

### B. Execution Logs
The execution trace is saved to a JSONL file (e.g., `episode_0.jsonl`).
- **Content**: Contains the full Accessibility Tree, Action JSON, and execution logs for each step.
- **Usage**: Parse this file to analyze decision making or token usage.

### C. Live Browser View
If running in a desktop environment (or with X11 forwarding), you can watch the browser in real-time:
```bash
python scripts/run_one_vwa_episode.py ... --no-headless
```

## 6. Usage Quickstart

### Setup
```bash
conda activate p79_ai
export CC=$(which x86_64-conda-linux-gnu-gcc)
cp -r external/visualwebarena/.auth . 
source scripts/vwa_env.sh
```

### Run Batch Experiment (Recommended)
```bash
# Run 20 shopping tasks (defined in configs/exp_shopping_small.yaml)
python scripts/run_vwa_batch.py --config configs/exp_shopping_small.yaml

# Resume an interrupted run
python scripts/run_vwa_batch.py --config configs/exp_shopping_small.yaml --resume results/shopping_small/run_TIMESTAMP

# Test pipeline without loading model (Mock Agent)
# Note: Evaluation will fail or return 0% as mock agent performs random actions
python scripts/run_vwa_batch.py --config configs/exp_shopping_small.yaml --mock_agent
```

### Evaluate Results
The batch runner performs inline evaluation. To re-evaluate or aggregate offline:
```bash
python scripts/eval_vwa_runs.py --result_dir results/shopping_small/<TIMESTAMP_RUNID>
```

### Run Single Task (Debug)
```bash
python scripts/run_one_vwa_episode.py \
  --task_config external/visualwebarena/config_files/vwa/test_shopping.raw.json \
  --task_id 0
```
