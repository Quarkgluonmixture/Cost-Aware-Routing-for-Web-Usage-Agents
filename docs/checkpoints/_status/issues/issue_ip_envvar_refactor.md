---
type: issue
category: backlog
status: backlog
priority: medium
action: "替换 \\${VWA_REMOTE_HOST} env var read"
---

# IP env-var-ize 重构 (9 处 hardcoded `100.95.81.103`)

文件: `auth_refresh.py` / `external/visualwebarena/browser_env/envs.py` / `scripts/maintenance/{reset_vwa_sites,retry_b1_single_task,experiment_watchdog}.{sh,py}`. 触发: Myriad onboard 时 (现已废, 如 RunPod 也用 Tailscale 可能 trigger).
