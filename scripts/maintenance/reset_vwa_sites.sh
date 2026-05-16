#!/usr/bin/env bash
# reset_vwa_sites.sh — VWA 站点 reset 工具（通用，可 source）
#
# 用法（source 后调用）：
#   source scripts/maintenance/reset_vwa_sites.sh
#   reset_vwa_sites all              # reset classifieds + reddit + shopping + shopping_admin
#   reset_vwa_sites classifieds      # 只 reset classifieds
#   reset_vwa_sites reddit
#   reset_vwa_sites shopping
#   reset_vwa_sites shopping_admin   # WA shopping_admin (Magento admin)
#
# 部署模式（VWA_RESET_MODE，默认 auto）：
#   remote  — DGX→quark→Windows PowerShell（默认 SSH 私钥存在时）
#   local   — A100 自托管 docker，本机操作（SSH 私钥不存在时 fallback）
#   auto    — 按 SSH 私钥存在与否自动选择
#
# 环境变量覆盖：
#   VWA_RESET_MODE      remote | local | auto
#   VWA_RESET_SSH_KEY   私钥路径（默认 ~/.ssh/vwa_windows）— remote 模式用
#   VWA_RESET_SSH_HOST  目标主机（默认 必须设置 e.g., quark@YOUR_HOST_IP）— remote 模式用
#   VWA_RESET_SCRIPT    Windows PowerShell 脚本路径（默认 C:\vwa\reset_vwa.ps1）— remote 模式用
#   VWA_RESET_ENABLE    设为 0 可禁用 reset（dry-run/本地调试用）
#   CLASSIFIEDS_RESET_TOKEN  cls reset endpoint token — local 模式用

VWA_RESET_MODE="${VWA_RESET_MODE:-auto}"
VWA_RESET_SSH_KEY="${VWA_RESET_SSH_KEY:-${HOME}/.ssh/vwa_windows}"
VWA_RESET_SSH_HOST="${VWA_RESET_SSH_HOST:?VWA_RESET_SSH_HOST must be set (e.g., quark@YOUR_HOST_IP); see scripts/vwa_env_remote.sh}"
VWA_RESET_SCRIPT="${VWA_RESET_SCRIPT:-C:\\vwa\\reset_vwa.ps1}"
VWA_RESET_ENABLE="${VWA_RESET_ENABLE:-1}"

# --- A100 self-host local-mode helpers ----------------------------------------

# reset_vwa_local_classifieds — HTTP reset endpoint（OSClass / jykoh image）
# BUG-3 fix (2026-05-16, codex Attack 6 + gemini NEW-OOB-2): OSClass controller
# at /usr/src/myapp/oc-includes/osclass/controller/reset.php expects POST + page=reset
# (NOT GET page=reset_database — that has no controller, OSClass silently swallows
# returning homepage 200 → wrapper false-positive; 0 SQL executed across entire
# Phase 1a run; cls DB state contaminated cross-episode). Codex verified live via
# `docker exec mysql query oc_t_item_comment` — 2 stale comments persisted post-"reset".
# 3-5pp drift (gemini estimate) / 0.2-0.8pp bounded (codex on require_reset subset).
_reset_vwa_local_classifieds() {
    local label="$1"
    local token="${CLASSIFIEDS_RESET_TOKEN:-4b61655535e7ed388f0d40a93600254c}"
    local code
    code=$(curl -sS -o /dev/null --max-time 60 -w "%{http_code}" \
           -X POST -d "token=${token}" \
           "http://localhost:9980/index.php?page=reset" 2>/dev/null || echo "000")
    if [[ "${code}" != "200" ]]; then
        echo "[${label}][reset_vwa][local] classifieds HTTP FAIL (http=${code})" >&2
        return 1
    fi
    # Mutation sentinel — verify reset actually executed (HTTP 200 alone is fake-safe
    # for OSClass; only docker-exec DB query confirms SQL ran). Codex's debug method.
    local count
    count=$(docker exec classifieds_db mysql -uroot -ppassword osclass -sN -e \
            "SELECT COUNT(*) FROM oc_t_item_comment WHERE b_active=1;" 2>/dev/null || echo "?")
    if [[ "${count}" != "0" ]]; then
        echo "[${label}][reset_vwa][local] classifieds reset SQL did not execute (oc_t_item_comment count=${count}, expected 0)" >&2
        return 1
    fi
    echo "[${label}][reset_vwa][local] classifieds OK (http=200, sentinel verified)"
    return 0
}

# reset_vwa_local_reddit — docker rm + run（postmill image seeds itself）
# Why: postmill-populated-exposed-withimg has the seed DB built in; recreate
# = fresh state. ~30-60s warm-up after run.
_reset_vwa_local_reddit() {
    local label="$1"
    docker rm -f vwa-reddit 2>/dev/null || true
    if ! docker run -d --name vwa-reddit -p 9999:80 postmill-populated-exposed-withimg >/dev/null; then
        echo "[${label}][reset_vwa][local] reddit docker run FAILED" >&2
        return 1
    fi
    # warm-up wait — postmill cold-start ~60-120s (DB init slower than expected).
    # Poll for HTTP 200 (not just connection-up; 500 = still initializing).
    local i code
    for i in $(seq 1 60); do
        code=$(curl -sS -o /dev/null --max-time 5 -w "%{http_code}" \
            http://localhost:9999/ 2>/dev/null || echo "000")
        if [[ "${code}" == "200" ]]; then
            echo "[${label}][reset_vwa][local] reddit OK (warm-up=$((i*3))s)"
            return 0
        fi
        sleep 3
    done
    echo "[${label}][reset_vwa][local] reddit warm-up TIMEOUT after 180s (last http=${code})" >&2
    return 1
}

# reset_vwa_local_shopping — placeholder for Phase 1b
# B-299 (A1.17 2026-05-16 cross-AI A+B P0): pre-fix returned `return 0` which the
# reset_and_auth_gate treated as success → Phase 1b shopping fires would silently
# proceed against dirty Magento state (cart/customer/session/search-cache from prior
# condition). Now returns 78 ("not implemented" sentinel rc); gate translates to
# hard-fail unless AUTH_GATE_BYPASS=1 explicitly set. Phase 1a (cls+red only) is
# unaffected because it never calls _reset_vwa_local_shopping.
_reset_vwa_local_shopping() {
    local label="$1"
    echo "[${label}][reset_vwa][local] shopping reset NOT YET IMPLEMENTED — Phase 1b launch blocked" >&2
    echo "[${label}][reset_vwa][local] implement Magento SQL-restore + cache flush + cart truncate before Phase 1b" >&2
    return 78
}

# reset_vwa_sites <site> [label]
#   site:  all | classifieds | reddit | shopping | shopping_admin
#   label: 日志前缀（可选，默认 reset）
reset_vwa_sites() {
    local site="${1:-all}"
    local label="${2:-reset}"

    if [[ "${VWA_RESET_ENABLE}" != "1" ]]; then
        echo "[${label}][reset_vwa] VWA_RESET_ENABLE=0，跳过 reset（site=${site}）"
        return 0
    fi

    # Auto-detect mode (B-300 A1.17 2026-05-16 cross-AI A+B P0 OOB):
    # pre-fix used only `[[ -f ${SSH_KEY} ]]` as proxy for "remote path available";
    # broken when A100 VM has legacy SSH key from dotfiles/rsync + paper-grade target
    # is local docker. Now hostname-first: A100 indicators force local, regardless
    # of SSH key presence. Remote only when explicitly DGX-shaped session.
    local mode="${VWA_RESET_MODE}"
    if [[ "${mode}" == "auto" ]]; then
        if [[ "$(hostname)" == *a100* ]] \
           || [[ "$(hostname)" == *condense* ]] \
           || [[ "${P79_PAPER_GRADE_HOST:-0}" == "1" ]] \
           || [[ -d /home/ubuntu/workspace/p79 ]]; then
            mode="local"
            echo "[${label}][reset_vwa] auto-detect: A100 indicator matched → mode=local (ignoring SSH key presence)" >&2
        elif [[ -f "${VWA_RESET_SSH_KEY}" ]]; then
            mode="remote"
        else
            mode="local"
        fi
    fi
    echo "[${label}][reset_vwa] resolved mode=${mode} for site=${site}" >&2

    # Local mode: dispatch per site via _reset_vwa_local_* helpers
    if [[ "${mode}" == "local" ]]; then
        echo "[${label}][reset_vwa] mode=local，开始 reset site=${site}..."
        local rc=0
        case "${site}" in
            all)
                _reset_vwa_local_classifieds "${label}" || rc=$?
                _reset_vwa_local_reddit "${label}" || rc=$?
                _reset_vwa_local_shopping "${label}" || rc=$?
                ;;
            classifieds)         _reset_vwa_local_classifieds "${label}" || rc=$? ;;
            reddit)              _reset_vwa_local_reddit "${label}" || rc=$? ;;
            shopping|shopping_admin)
                                 _reset_vwa_local_shopping "${label}" || rc=$? ;;
            *)
                echo "[${label}][reset_vwa] unknown site=${site}" >&2
                return 2
                ;;
        esac
        return ${rc}
    fi

    # Remote mode: DGX→quark→Windows PowerShell (original path)
    if [[ ! -f "${VWA_RESET_SSH_KEY}" ]]; then
        echo "[${label}][reset_vwa] WARNING: SSH 私钥不存在: ${VWA_RESET_SSH_KEY}，跳过 reset"
        return 1
    fi

    echo "[${label}][reset_vwa] mode=remote，开始 reset site=${site}..."
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

    # DGX-side defensive check: Magento base_url 复发是历史 bug, 持久化在 quark
    # side 已做三层 (commit on quark: magento_baseurl_fix.sh + start_vwa_docker.sh
    # + reset_shopping.sh), 这里加个 reachability+redirect 验证作为 belt-and-
    # suspenders. 仅 warn, 不 fail (避免 reset rc 干扰上游 chain).
    if [[ "${site}" == "shopping" || "${site}" == "shopping_admin" || "${site}" == "all" ]]; then
        for shop_site in shopping shopping_admin; do
            [[ "${site}" != "all" && "${site}" != "${shop_site}" ]] && continue
            local port="7770"
            [[ "${shop_site}" == "shopping_admin" ]] && port="7780"
            local url="${VWA_HOST_URL:-http://localhost}:${port}/"
            local redirect
            redirect=$(curl -sS -o /dev/null --max-time 10 -w "%{redirect_url}" -I "${url}" 2>/dev/null || echo "")
            if [[ "${redirect}" == *metis* ]]; then
                echo "[${label}][reset_vwa] ⚠️ ${shop_site} 又 redirect 到 metis (${redirect}). 需 quark side magento_baseurl_fix.sh 重跑."
            fi
        done
    fi

    return $rc
}
