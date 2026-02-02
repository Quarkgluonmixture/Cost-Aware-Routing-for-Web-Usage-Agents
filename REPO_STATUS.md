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
- **Single Episode Runner**: `scripts/run_one_vwa_episode.py` allows running specific tasks by ID from raw configuration files.
- **Dynamic Configuration**: Automatically replaces placeholders (e.g., `__SHOPPING__`) in task configs with active local URLs.
- **Logging**: detailed execution logs (steps, actions, observations, rewards) are saved in JSONL format (e.g., `episode_0.jsonl`).

## 3. Completed Work (Changelog)

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
    - **Image Handling**: Patched `p79/envs/vwa_wrapper.py` to convert numpy arrays from Playwright to PIL images, fixing a crash in `qwen_vl_utils`.
    - **Config Loading**: Fixed import errors in `run_one_vwa_episode.py` and removed dependencies on missing `Config` class.
    - **URL Handling**: Implemented dynamic URL placeholder replacement to support raw VWA task configs.

## 4. Pending / Placeholder Items
- **Router Policy**: The `p79/policies/router.py` exists but contains placeholder logic. The "Cost-Aware Routing" feature (dynamically selecting between models based on complexity/cost) is not yet implemented.
- **Batch Evaluation**: `scripts/run_batch.py` exists but hasn't been stress-tested with the new environment setup.
- **Evaluation**: The success detection logic in the single episode runner is a proxy; full evaluation requires VWA's evaluation harness integration.

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

1. **Activate Environment**:
   ```bash
   conda activate p79_ai
   ```

2. **Set Up Environment Variables**:
   ```bash
   # Ensure .auth is present in current directory for playwright (prepare.sh generates this)
   cp -r external/visualwebarena/.auth .
   # Source env vars
   source scripts/vwa_env.sh
   # Set CC for triton if needed (WSL specific)
   export CC=$(which x86_64-conda-linux-gnu-gcc)
   ```

3. **Run a Task**:
   ```bash
   python scripts/run_one_vwa_episode.py \
     --task_config external/visualwebarena/config_files/vwa/test_shopping.raw.json \
     --task_id 0
   ```
