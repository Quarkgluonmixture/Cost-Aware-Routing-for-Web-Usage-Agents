# Cost-Aware Routing for Web Usage Agents

## Executive Summary
This repository implements a local VisualWebArena (VWA) agent pipeline with Qwen3-VL-4B inference and a custom environment wrapper. The system runs end-to-end tasks on locally hosted web environments, capturing screenshots and action traces for reproducible evaluation.

## Current Status
- [x] Environment
- [x] Agent

## Screenshot
The following screenshot is captured from the locally hosted Shopping site during a live run.

![VWA Shopping Screenshot](assets/vwa_run_screenshot.png)

## Quick Start
```bash
conda activate p79_ai
export CC=$(which x86_64-conda-linux-gnu-gcc)
cp -r external/visualwebarena/.auth .
source scripts/vwa_env.sh

python scripts/run_vwa_batch.py --config configs/exp_shopping_small.yaml
```

## Project Structure
- `p79/agents`: Qwen3-VL agent implementation
- `p79/envs`: VWA environment wrapper and observation handling
- `scripts/`: batch runner, evaluator, utilities
- `external/visualwebarena/`: upstream environment and evaluator
