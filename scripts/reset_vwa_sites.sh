#!/usr/bin/env bash
# reset_vwa_sites.sh — VWA 站点 reset 工具（通用，可 source）
#
# 用法（source 后调用）：
#   source scripts/reset_vwa_sites.sh
#   reset_vwa_sites all          # reset classifieds + reddit + shopping
#   reset_vwa_sites classifieds  # 只 reset classifieds
#   reset_vwa_sites reddit
#   reset_vwa_sites shopping
#
# 环境变量覆盖：
#   VWA_RESET_SSH_KEY   私钥路径（默认 ~/.ssh/vwa_windows）
#   VWA_RESET_SSH_HOST  目标主机（默认 quark@100.95.81.103）
#   VWA_RESET_SCRIPT    Windows PowerShell 脚本路径（默认 C:\vwa\reset_vwa.ps1）
#   VWA_RESET_ENABLE    设为 0 可禁用 reset（dry-run/本地调试用）

VWA_RESET_SSH_KEY="${VWA_RESET_SSH_KEY:-${HOME}/.ssh/vwa_windows}"
VWA_RESET_SSH_HOST="${VWA_RESET_SSH_HOST:-quark@100.95.81.103}"
VWA_RESET_SCRIPT="${VWA_RESET_SCRIPT:-C:\\vwa\\reset_vwa.ps1}"
VWA_RESET_ENABLE="${VWA_RESET_ENABLE:-1}"

# reset_vwa_sites <site> [label]
#   site:  all | classifieds | reddit | shopping
#   label: 日志前缀（可选，默认 reset）
reset_vwa_sites() {
    local site="${1:-all}"
    local label="${2:-reset}"

    if [[ "${VWA_RESET_ENABLE}" != "1" ]]; then
        echo "[${label}][reset_vwa] VWA_RESET_ENABLE=0，跳过 reset（site=${site}）"
        return 0
    fi

    if [[ ! -f "${VWA_RESET_SSH_KEY}" ]]; then
        echo "[${label}][reset_vwa] WARNING: SSH 私钥不存在: ${VWA_RESET_SSH_KEY}，跳过 reset"
        return 1
    fi

    echo "[${label}][reset_vwa] 开始 reset site=${site}..."
    ssh -i "${VWA_RESET_SSH_KEY}" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=30 \
        -o ServerAliveInterval=60 \
        "${VWA_RESET_SSH_HOST}" \
        "powershell -File ${VWA_RESET_SCRIPT} -Site ${site}" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[${label}][reset_vwa] reset 完成"
    else
        echo "[${label}][reset_vwa] WARNING: reset 失败 rc=${rc}（继续执行）"
    fi
    return $rc
}
