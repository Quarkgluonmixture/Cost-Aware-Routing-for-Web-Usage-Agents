#!/usr/bin/env bash
# run_b0_shopping_dom.sh — B0 shopping 仅 DOM 模式（带 auth refresh + watchdog）
#
# 用法:
#   setsid nohup bash scripts/dgx/run_b0_shopping_dom.sh \
#     > logs/b0_shopping_dom.log 2>&1 < /dev/null &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

# ---------- 通用 reset 工具 ----------
source "${REPO_DIR}/scripts/reset_vwa_sites.sh"

# ---------- API key 加载 ----------
AUTH_FILE="${REPO_DIR}/.auth/qwen_api"
if [[ -z "${PROXY_API_KEY:-}" ]]; then
  if [[ -f "${AUTH_FILE}" ]]; then
    raw_key="$(grep -m1 '^rp_' "${AUTH_FILE}" | tr -d '[:space:]')"
    if [[ -n "${raw_key}" ]]; then
      export PROXY_API_KEY="${raw_key}"
      export QWEN_API_KEY="${raw_key}"
      export DASHSCOPE_API_KEY="${raw_key}"
      echo "[b0/shopping/dom] Loaded PROXY_API_KEY from ${AUTH_FILE}" >&2
    else
      echo "[error] ${AUTH_FILE} 存在但为空" >&2; exit 1
    fi
  else
    echo "[error] ${AUTH_FILE} 不存在，且 PROXY_API_KEY 未设置" >&2; exit 1
  fi
fi

# ---------- 配置 ----------
BASE_CONFIG="${REPO_DIR}/configs/exp_v2_B0_3mode_shopping.yaml"
LOG_DIR="${REPO_DIR}/logs"
RESULTS_BASE="${REPO_DIR}/results/visualwebarena/phase1"
RUN_ID="${RUN_ID:-B0_3mode_shopping_$(date +%Y%m%d)}"
RUN_DIR="${RESULTS_BASE}/${RUN_ID}"
CONDITION_ID="phase1_dom_router_0"

MAX_RESUME_ATTEMPTS="${MAX_RESUME_ATTEMPTS:-10}"
WATCHDOG_TIMEOUT_MINS="${WATCHDOG_TIMEOUT_MINS:-35}"
WATCHDOG_CHECK_SECS="${WATCHDOG_CHECK_SECS:-60}"

export NTFY_TOPIC="${NTFY_TOPIC:-p79-exp-dgx-spark}"
NTFY_URL="https://ntfy.sh/${NTFY_TOPIC}"

EXP_WATCHDOG_ENABLE="${EXP_WATCHDOG_ENABLE:-1}"
EXP_WATCHDOG_POLL_SECS="${EXP_WATCHDOG_POLL_SECS:-30}"
EXP_WATCHDOG_IDLE_ALERT_MINS="${EXP_WATCHDOG_IDLE_ALERT_MINS:-30}"
EXP_WATCHDOG_NOTIFY_COMPLETION_ENABLE="${EXP_WATCHDOG_NOTIFY_COMPLETION_ENABLE:-0}"
EXP_WATCHDOG_GLM_CONFIG="${EXP_WATCHDOG_GLM_CONFIG:-${REPO_DIR}/.auth/glm}"
EXP_WATCHDOG_PID=""

AGGREGATE_PREFIX="B0_3mode"

mkdir -p "${LOG_DIR}"

# ---------- 环境变量 ----------
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
export OPENAI_API_KEY="${OPENAI_API_KEY:-DUMMY_P79_NON_LLM_EVAL}"
export P79_DISABLE_STALE_CLEANUP="${P79_DISABLE_STALE_CLEANUP:-1}"
export WIKIPEDIA_ZIM_VERSION="${WIKIPEDIA_ZIM_VERSION:-wikipedia_en_all_maxi_2025-08}"

# ---------- Python ----------
if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "[error] 找不到 Python 解释器" >&2; exit 127
fi

# ---------- VWA 站点环境 ----------
if [[ -n "${VWA_ENV_FILE:-}" ]] && [[ -f "${VWA_ENV_FILE}" ]]; then
  source "${VWA_ENV_FILE}" || true
elif [[ -f "${REPO_DIR}/scripts/vwa_env_remote.sh" ]]; then
  source "${REPO_DIR}/scripts/vwa_env_remote.sh" || true
elif [[ -f "${REPO_DIR}/scripts/vwa_env.sh" ]]; then
  source "${REPO_DIR}/scripts/vwa_env.sh" || true
fi

# ---------- 辅助函数 ----------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

ntfy_send() {
  local title="$1" message="$2" priority="${3:-default}"
  curl -s -H "Title: ${title}" -H "Priority: ${priority}" \
    -d "${message}" "${NTFY_URL}" > /dev/null 2>&1 || true
}

count_episode_summaries() {
  find "${1}" -type f -path "*/episodes/*_summary_v2.json" 2>/dev/null | wc -l | tr -d '[:space:]'
}

is_condition_complete() {
  local run_dir="$1" cid="$2"
  local cond_dir="${run_dir}/${cid}"
  local task_total done
  task_total=$(find "${run_dir}/task_configs" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d '[:space:]')
  done=$(find "${cond_dir}" -type f -path "*/episodes/*_summary_v2.json" 2>/dev/null | wc -l | tr -d '[:space:]')
  if [[ -f "${cond_dir}/condition_summary_v2.json" ]]; then
    if [[ "${task_total}" -gt 0 && "${done}" -lt "${task_total}" ]]; then
      log "${cid}: condition_summary 存在但仅完成 ${done}/${task_total} tasks，删除过期 summary 重跑"
      rm -f "${cond_dir}/condition_summary_v2.json"
      return 1
    fi
    return 0
  fi
  [[ "${task_total}" -gt 0 && "${done}" -ge "${task_total}" ]] && return 0
  return 1
}

# ---------- Auth refresh ----------
refresh_site_auth() {
  local site="$1"
  local auth_file="${REPO_DIR}/.auth/${site}_state.json"
  log "刷新 ${site} auth state..."
  DATASET=visualwebarena "${PYTHON_BIN}" - <<PYEOF
import os, sys, time
sys.path.insert(0, '${REPO_DIR}/external/visualwebarena')
from playwright.sync_api import sync_playwright

site = '${site}'
ACCOUNTS = {
    'classifieds': ('blake.sullivan@gmail.com', 'Password.123'),
    'reddit':      ('MarvelsGrantMan136',        'test1234'),
    'shopping':    ('emma.lopez@gmail.com',       'Password.123'),
}
base_urls = {
    'classifieds': os.environ.get('CLASSIFIEDS', 'http://100.95.81.103:9980'),
    'reddit':      os.environ.get('REDDIT',      'http://100.95.81.103:9999'),
    'shopping':    os.environ.get('SHOPPING',    'http://100.95.81.103:7770'),
}
login_paths = {
    'classifieds': '/index.php?page=login',
    'reddit':      '/login',
    'shopping':    '/customer/account/login/',
}
username, password = ACCOUNTS[site]
base_url = base_urls[site]
login_path = login_paths[site]

cm = sync_playwright()
playwright = cm.__enter__()
browser = playwright.chromium.launch(headless=True, args=['--host-resolver-rules=MAP metis.lti.cs.cmu.edu 100.95.81.103'])
context = browser.new_context()
page = context.new_page()
page.goto(base_url + login_path)
if site == 'shopping':
    page.get_by_label('Email', exact=True).fill(username)
    page.get_by_label('Password', exact=True).fill(password)
    page.get_by_role('button', name='Sign In').click()
time.sleep(2)
context.storage_state(path='${auth_file}')
cm.__exit__(None, None, None)
print('${site} auth refreshed -> ' + page.url)
PYEOF
  local rc=$?
  if [[ $rc -eq 0 ]] && [[ -s "${auth_file}" ]]; then
    log "${site} auth state 已刷新"
    return 0
  else
    log "[error] ${site} auth 刷新失败 rc=${rc}"
    return 1
  fi
}

AUTH_REFRESH_MAX_ATTEMPTS="${AUTH_REFRESH_MAX_ATTEMPTS:-5}"
refresh_site_auth_retry() {
  local site="$1" label="${2:-auth}"
  local attempt=0 delay=10
  while [[ $attempt -lt $AUTH_REFRESH_MAX_ATTEMPTS ]]; do
    attempt=$(( attempt + 1 ))
    if refresh_site_auth "${site}"; then
      return 0
    fi
    if [[ $attempt -ge $AUTH_REFRESH_MAX_ATTEMPTS ]]; then break; fi
    log "[${label}] auth 刷新第 ${attempt}/${AUTH_REFRESH_MAX_ATTEMPTS} 次失败，${delay}s 后重试..."
    ntfy_send "P79 [B0/shopping/dom] auth retry" "第 ${attempt} 次失败，${delay}s 后重试" "default"
    sleep "${delay}"
    delay=$(( delay * 2 ))
  done
  log "[${label}][fatal] auth 刷新 ${AUTH_REFRESH_MAX_ATTEMPTS} 次均失败"
  ntfy_send "P79 [B0/shopping/dom] auth FAILED" "${AUTH_REFRESH_MAX_ATTEMPTS} 次失败" "urgent"
  return 1
}

# ---------- experiment_watchdog ----------
start_exp_watchdog() {
  EXP_WATCHDOG_PID=""
  [[ "${EXP_WATCHDOG_ENABLE}" != "1" ]] && return 0
  local ws="${REPO_DIR}/scripts/experiment_watchdog.py"
  [[ -f "${ws}" ]] || return 0

  local _op
  _op="$(ps -eo pid=,args= | awk -v dir="${RUN_DIR}" '/experiment_watchdog\.py/ && $0 ~ dir && !/awk/ {print $1}')"
  if [[ -n "${_op}" ]]; then
    for p in ${_op}; do kill "${p}" 2>/dev/null || true; done; sleep 1
    for p in ${_op}; do kill -9 "${p}" 2>/dev/null || true; done
  fi

  local wlog="${LOG_DIR}/experiment_watchdog_b0_shopping_dom.log"
  local wstate="${LOG_DIR}/experiment_watchdog_b0_shopping_dom.state.json"
  local wcmd=(
    "${PYTHON_BIN}" -u "${ws}"
    --run-dir "${RUN_DIR}"
    --poll-secs "${EXP_WATCHDOG_POLL_SECS}"
    --idle-alert-mins "${EXP_WATCHDOG_IDLE_ALERT_MINS}"
    --ntfy-topic "${NTFY_TOPIC}"
    --state-file "${wstate}"
    --aggregate-prefix "${AGGREGATE_PREFIX}"
  )
  [[ -f "${EXP_WATCHDOG_GLM_CONFIG}" ]] && wcmd+=(--glm-config "${EXP_WATCHDOG_GLM_CONFIG}" --digest-dir "${RUN_DIR}/analysis/digest")
  [[ "${EXP_WATCHDOG_NOTIFY_COMPLETION_ENABLE}" == "1" ]] && wcmd+=(--notify-completion)

  mkdir -p "${RUN_DIR}"
  nohup "${wcmd[@]}" > "${wlog}" 2>&1 < /dev/null &
  local pid=$!; sleep 1
  kill -0 "${pid}" 2>/dev/null \
    && { EXP_WATCHDOG_PID="${pid}"; log "experiment_watchdog pid=${pid}"; } \
    || log "[warn] experiment_watchdog 启动失败"
}

stop_exp_watchdog() {
  local pid="${EXP_WATCHDOG_PID:-}"; [[ -z "${pid}" ]] && return 0
  kill -0 "${pid}" 2>/dev/null && {
    kill "${pid}" 2>/dev/null || true; sleep 2
    kill -9 "${pid}" 2>/dev/null || true
  }
  wait "${pid}" 2>/dev/null || true
  EXP_WATCHDOG_PID=""
}

# ---------- 生成 dom-only config ----------
DOM_CONFIG="/tmp/b0_shopping_dom_only_$$.yaml"
"${PYTHON_BIN}" - << PYEOF
import re
with open("${BASE_CONFIG}") as f:
    content = f.read()
content = re.sub(
    r'observation_mode:\s*\[.*?\]',
    'observation_mode: ["dom"]',
    content
)
with open("${DOM_CONFIG}", "w") as f:
    f.write(content)
print("dom-only config -> ${DOM_CONFIG}")
PYEOF

# ---------- 前台运行（带 watchdog）----------
ACTIVE_RUNNER_PID=""

run_condition_foreground() {
  local log_path="${LOG_DIR}/b0_shopping_dom_${RUN_ID}.log"
  mkdir -p "${RUN_DIR}"

  log "=== [B0/shopping/dom] 启动 run_id=${RUN_ID} ==="

  nohup "${PYTHON_BIN}" scripts/run_experiment.py \
    --config "${DOM_CONFIG}" \
    --run_id "${RUN_ID}" \
    --log_path "${log_path}" \
    >> "${log_path}" 2>&1 < /dev/null &
  local job_pid=$!
  ACTIVE_RUNNER_PID="${job_pid}"
  log "[B0/shopping/dom] PID=${job_pid}"

  local last_count stale_secs=0 watchdog_secs=$(( WATCHDOG_TIMEOUT_MINS * 60 )) next_log_secs=300
  last_count=$(count_episode_summaries "${RUN_DIR}")

  while kill -0 "${job_pid}" 2>/dev/null; do
    sleep "${WATCHDOG_CHECK_SECS}"
    ! kill -0 "${job_pid}" 2>/dev/null && break
    local cur
    cur=$(count_episode_summaries "${RUN_DIR}")
    if [[ "${cur}" -gt "${last_count}" ]]; then
      local new=$(( cur - last_count ))
      log "[B0/shopping/dom] +${new} episode(s) total=${cur}，计时重置"
      last_count="${cur}"; stale_secs=0; next_log_secs=300
    else
      stale_secs=$(( stale_secs + WATCHDOG_CHECK_SECS ))
      [[ "${stale_secs}" -ge "${next_log_secs}" ]] && {
        log "[B0/shopping/dom] $(( stale_secs / 60 ))min 无新 episode（上限 ${WATCHDOG_TIMEOUT_MINS}min）"
        next_log_secs=$(( next_log_secs + 300 ))
      }
      [[ "${stale_secs}" -ge "${watchdog_secs}" ]] && {
        log "[B0/shopping/dom] WATCHDOG: kill PID ${job_pid}"
        ntfy_send "P79 [B0/shopping/dom] WATCHDOG" "${WATCHDOG_TIMEOUT_MINS}min 无进展，kill 准备 resume" "high"
        kill "${job_pid}" 2>/dev/null || true; sleep 10
        kill -9 "${job_pid}" 2>/dev/null || true
        wait "${job_pid}" 2>/dev/null || true
        return 1
      }
    fi
  done
  wait "${job_pid}" 2>/dev/null || true
  log "=== [B0/shopping/dom] 进程退出 ==="
}

# ---------- Cleanup ----------
cleanup() {
  [[ -n "${ACTIVE_RUNNER_PID:-}" ]] && kill -0 "${ACTIVE_RUNNER_PID}" 2>/dev/null \
    && { kill "${ACTIVE_RUNNER_PID}" 2>/dev/null || true; }
  stop_exp_watchdog
  rm -f "${DOM_CONFIG}" 2>/dev/null || true
}
trap cleanup EXIT

# ---------- 主流程 ----------
log "========================================================"
log "=== B0 Shopping DOM-only 启动 ==="
log "=== RUN_ID=${RUN_ID} ==="
log "=== WATCHDOG_TIMEOUT_MINS=${WATCHDOG_TIMEOUT_MINS} ==="
log "========================================================"

ntfy_send "P79 [B0/shopping/dom] 启动" "run_id=${RUN_ID}" "default"

# Auth refresh
refresh_site_auth_retry "shopping" "shopping/dom" || { log "[fatal] auth 失败，中止"; exit 1; }

# Start watchdog
start_exp_watchdog

# Run with auto-resume
attempt=0
while ! is_condition_complete "${RUN_DIR}" "${CONDITION_ID}"; do
  attempt=$(( attempt + 1 ))
  [[ ${attempt} -gt ${MAX_RESUME_ATTEMPTS} ]] && {
    log "[B0/shopping/dom] ERROR: ${MAX_RESUME_ATTEMPTS} 次 resume 后仍未完成"
    ntfy_send "P79 [B0/shopping/dom] 失败" "已重试 ${MAX_RESUME_ATTEMPTS} 次" "urgent"
    exit 1
  }
  [[ ${attempt} -gt 1 ]] && {
    log "[B0/shopping/dom] resume ${attempt}/${MAX_RESUME_ATTEMPTS}..."
    refresh_site_auth_retry "shopping" "shopping/dom/retry${attempt}" || true
    ntfy_send "P79 [B0/shopping/dom] 重试" "第 ${attempt}/${MAX_RESUME_ATTEMPTS} 次 resume" "default"
  }
  run_condition_foreground || true
  log "[B0/shopping/dom] 等待 15s..."
  sleep 15
done

stop_exp_watchdog

log "========================================================"
log "=== B0 Shopping DOM 完成（${attempt} 次）==="
log "========================================================"
ntfy_send "P79 [B0/shopping/dom] 完成!" "run_id=${RUN_ID}，${attempt} 次" "high"

rm -f "${DOM_CONFIG}"
