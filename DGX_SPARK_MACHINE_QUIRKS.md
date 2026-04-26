# DGX Spark Machine Quirks (Local Only)

Last updated: 2026-04-03

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

4. `torch` may be installed as CPU-only (`+cpu`), making local 4B runs effectively unusable
- Symptom:
```bash
./.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print("cuda_available=", torch.cuda.is_available())
print("cuda_built=", torch.backends.cuda.is_built())
PY
```
- If you see `+cpu` and `cuda_available=False`, reinstall CUDA wheels explicitly:
```bash
source .venv/bin/activate
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128
```
- Verify after install:
```bash
python - <<'PY'
import torch
print(torch.__version__)
print("cuda_version=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

5. `nvrtc: invalid value for --gpu-architecture` on GB10 (`sm_121`)
- Symptom in baseline logs:
```text
nvrtc: error: invalid value for --gpu-architecture (-arch)
```
- Cause: runtime GPU capability (`sm_121`) is newer than the SM list in some torch wheels.
- Repository default behavior now auto-enables a targeted `torch.prod` fallback in `Qwen3VLAgent`.
- Optional override (disable fallback):
```bash
export P79_DISABLE_NVRTC_PROD_FALLBACK=1
```

6. Remote VWA site mode: avoid `localhost` redirect loops
- Symptom:
  - DGX-side task DOM shows `502 Bad Gateway` at `http://localhost:7770/`
  - But remote `SHOPPING` endpoint itself is reachable.
- Cause:
  - Shopping site base URL on the host machine is still configured as `localhost`,
    so remote requests are redirected to DGX local `localhost`.
- Fix on site host:
```bash
HOST=100.95.81.103
docker exec vwa-shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://${HOST}:7770"
docker exec vwa-shopping mysql -u magentouser -pMyPassword magentodb -e \
  "UPDATE core_config_data SET value='http://${HOST}:7770/' WHERE path IN ('web/unsecure/base_url','web/secure/base_url');"
docker exec vwa-shopping /var/www/magento2/bin/magento cache:flush
```
- Verify from DGX:
```bash
curl -I "$SHOPPING" | grep -iE '^(HTTP/|Location:)'
```
Expected: no `Location: http://localhost:7770/...`.

7. Remote homepage (`:4399`) may need Windows portproxy maintenance
- If DGX cannot reach homepage but WSL local `127.0.0.1:4399` returns 200:
  - re-create Windows `netsh interface portproxy` mapping
  - ensure inbound firewall allows TCP 4399

8. DGX Spark already includes NVIDIA Container Runtime (usually no extra install needed)
- If Docker reports `runtime not found`, first verify runtime registration instead of reinstalling:
```bash
docker info | grep -iE 'Runtimes|Default Runtime'
```
- Permission issue (`permission denied` on Docker socket):
```bash
id
getent group docker
```
  Ask the machine owner/admin to add your user to the `docker` group if needed.
- CUDA mismatch checks (`nvidia-smi` works on host but fails in container, or vice versa):
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```
  If container CUDA stack and host driver are incompatible, align image/toolkit versions with host constraints.

9. NGC CLI must be ARM64 on DGX Spark
- DGX Spark is `aarch64`, so use the ARM64 NGC CLI package only.
- Quick architecture check:
```bash
uname -m
```
Expected: `aarch64`.

10. UMA memory display in `nvidia-smi` is different from discrete-GPU systems
- On UMA platforms, memory figures in `nvidia-smi` can look unusual compared with standalone dGPU expectations; this is expected behavior.
- For debugging memory-reclaim observation bias, you can force cache drop:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
  This requires `sudo`; if you are not the machine owner, confirm with the owner/admin first.

11. All background jobs must use `setsid + nohup` on DGX Spark
- Rule:
```bash
setsid nohup <command> > <log_path> 2>&1 < /dev/null &
```
- Do not use plain `nohup <command> &` or `<command> &` for long-running jobs.
- Reason: this host has had background process lifecycle quirks; using both `setsid` and `nohup` avoids parent-shell/session side effects and makes jobs more stable after disconnect.
- Recommended template:
```bash
LOG_PATH="logs/<job_name>.log"
setsid nohup bash -lc 'cd /abs/repo/path && exec <command>' \
  > "${LOG_PATH}" 2>&1 < /dev/null &
echo "pid=$!"
```

## Experiment queue scripts
Use scripts under `scripts/queues/` for B0/B1 三模式 baseline runs（contain DGX-specific env handling: NVML CUDA check, MPS pipe disable, sm_121 fallback）：

```bash
bash scripts/queues/queue_b1_with_reset.sh        # B1 三模式 VWA
bash scripts/queues/queue_b1_wa_with_reset.sh     # B1 三模式 WA
bash scripts/queues/queue_b0_with_reset.sh        # B0 三模式 VWA
bash scripts/queues/queue_b0_wa_with_reset.sh     # B0 三模式 WA
```

These scripts include DGX-specific env handling and should not be treated as general defaults.
