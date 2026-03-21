# DGX Spark Machine Quirks (Local Only)

Last updated: 2026-03-21

> This file is **host-specific** and is **not** the default setup path for this repository.
> General users should follow `README.md` only.

## Scope
Applies to local DGX Spark host (`spark-9ea3`, aarch64 + NVIDIA GB10) where we run local Qwen3-VL-4B baseline jobs.

## Known Quirks
1. `python` command may be missing
- Prefer `./.venv/bin/python` or `python3`.

2. CUDA probe may hang unless NVML-based check is enabled
- Use:
```bash
export PYTORCH_NVML_BASED_CUDA_CHECK=1
```

3. Global MPS env can cause instability
- Use empty MPS variables:
```bash
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
```

## DGX-only scripts
Use scripts under `scripts/dgx/`:

```bash
bash scripts/dgx/run_qwen3vl4b_baseline.sh
bash scripts/dgx/start_qwen3vl4b_baseline_when_gpu_idle.sh
bash scripts/dgx/qwen3vl4b_status.sh
```

These scripts intentionally include DGX-specific env handling and should not be treated as general defaults.
