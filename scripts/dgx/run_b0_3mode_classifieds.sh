#!/usr/bin/env bash
# run_b0_3mode_classifieds.sh — B0 三模式 classifieds（dom → reset → som → reset → vision）
#
# 每个 condition 独立跑，condition 间自动 reset 站点，消除数据污染。
# 三个 condition 共享同一 RUN_ID，结果汇入同一目录，分析脚本无需改动。
#
# API key 来源: .auth/qwen_api（单行 rp_... 格式）
# 站点 reset:   ~/.ssh/vwa_windows → quark@100.95.81.103 → C:\vwa\reset_vwa.ps1
#
# 用法:
#   bash scripts/dgx/run_b0_3mode_classifieds.sh
#   B0_RUN_ID=B0_3mode_classifieds_20260413 bash scripts/dgx/run_b0_3mode_classifieds.sh
#
# Gallery: http://localhost:8765/B0_3mode/gallery.html
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

# ---------- 通用 reset 工具 ----------
source "${REPO_DIR}/scripts/reset_vwa_sites.sh"

# refresh_classifieds_auth — classifieds 站点 reset 后重新登录，刷新 .auth/classifieds_state.json
refresh_classifieds_auth() {
  local auth_file="${REPO_DIR}/.auth/classifieds_state.json"
  local classifieds_url="${CLASSIFIEDS:-http://100.95.81.103:9980}"
  log "[b0_3mode] 刷新 classifieds auth state (${classifieds_url})..."
  DATASET=visualwebarena "${PYTHON_BIN:-python3}" - <<PYEOF
import os, sys, time
sys.path.insert(0, '${REPO_DIR}/external/visualwebarena')
from playwright.sync_api import sync_playwright
url = '${classifieds_url}'
cm = sync_playwright()
playwright = cm.__enter__()
browser = playwright.chromium.launch(headless=True)
context = browser.new_context()
page = context.new_page()
page.goto(url + '/index.php?page=login')
page.locator('#email').fill('blake.sullivan@gmail.com')
page.locator('#password').fill('Password.123')
page.get_by_role('button', name='Log in').click()
time.sleep(2)
context.storage_state(path='${auth_file}')
cm.__exit__(None, None, None)
print('classifieds auth refreshed -> ' + page.url)
PYEOF
  local rc=$?
  if [[ $rc -eq 0 ]] && [[ -s "${auth_file}" ]]; then
    log "[b0_3mode] classifieds auth state 已刷新"
    return 0
  else
    log "[b0_3mode][error] classifieds auth 刷新失败 rc=${rc}（auth_file=$(wc -c < "${auth_file}" 2>/dev/null || echo missing) bytes）"
    return 1
  fi
}

# ---------- API key 加载 ----------
AUTH_FILE="${REPO_DIR}/.auth/qwen_api"
if [[ -z "${PROXY_API_KEY:-}" ]]; then
  if [[ -f "${AUTH_FILE}" ]]; then
    raw_key="$(grep -m1 '^rp_' "${AUTH_FILE}" | tr -d '[:space:]')"
    if [[ -n "${raw_key}" ]]; then
      export PROXY_API_KEY="${raw_key}"
      export QWEN_API_KEY="${raw_key}"
      export DASHSCOPE_API_KEY="${raw_key}"
      echo "[b0_3mode] Loaded PROXY_API_KEY from ${AUTH_FILE}" >&2
    else
      echo "[b0_3mode][error] ${AUTH_FILE} 存在但为空" >&2; exit 1
    fi
  else
    echo "[b0_3mode][error] ${AUTH_FILE} 不存在，且 PROXY_API_KEY 未设置" >&2; exit 1
  fi
else
  echo "[b0_3mode] PROXY_API_KEY 已由环境变量提供" >&2
fi

# ---------- 配置 ----------
BASE_CONFIG="${REPO_DIR}/configs/exp_v2_B0_3mode_classifieds.yaml"
LOG_DIR="${REPO_DIR}/logs"
RESULTS_BASE="${REPO_DIR}/results/visualwebarena/phase1"
mkdir -p "${LOG_DIR}"

NTFY_TOPIC="${NTFY_TOPIC:-p79-exp-dgx-spark}"
NTFY_URL="https://ntfy.sh/${NTFY_TOPIC}"
NTFY_MINIMAL_MODE="${NTFY_MINIMAL_MODE:-1}"
WATCHDOG_ENABLE="${WATCHDOG_ENABLE:-1}"
WATCHDOG_POLL_SECS="${WATCHDOG_POLL_SECS:-30}"
WATCHDOG_IDLE_ALERT_MINS="${WATCHDOG_IDLE_ALERT_MINS:-20}"
WATCHDOG_NOTIFY_COMPLETION_ENABLE="${WATCHDOG_NOTIFY_COMPLETION_ENABLE:-1}"
WATCHDOG_GLM_CONFIG="${REPO_DIR}/.auth/glm"
WATCHDOG_PID=""

REASON_DIAG_ENABLE="${REASON_DIAG_ENABLE:-1}"
MAX_RESUME_ATTEMPTS="${MAX_RESUME_ATTEMPTS:-10}"
WATCHDOG_TIMEOUT_MINS="${WATCHDOG_TIMEOUT_MINS:-35}"
WATCHDOG_CHECK_SECS="${WATCHDOG_CHECK_SECS:-60}"
AGGREGATE_PREFIX="B0_3mode"

# ---------- Run ID（三个 condition 共享）----------
RUN_ID="${B0_RUN_ID:-B0_3mode_classifieds_20260413}"
OUTPUT_DIR="${RESULTS_BASE}/${RUN_ID}"

echo "[b0_3mode] run_id:  ${RUN_ID}" >&2
echo "[b0_3mode] output:  ${OUTPUT_DIR}" >&2
echo "[b0_3mode] model:   qwen3-vl-235b-a22b (proxy API)" >&2
echo "[b0_3mode] gallery: http://localhost:8765/${AGGREGATE_PREFIX}/gallery.html" >&2
echo "[b0_3mode] reset:   VWA_RESET_ENABLE=${VWA_RESET_ENABLE}" >&2

# ---------- Python 解释器 ----------
if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "[b0_3mode][error] 找不到 Python 解释器" >&2; exit 127
fi

# ---------- VWA 站点环境 ----------
if [[ -n "${VWA_ENV_FILE:-}" ]] && [[ -f "${VWA_ENV_FILE}" ]]; then
  source "${VWA_ENV_FILE}" || true
elif [[ -f "${REPO_DIR}/scripts/vwa_env_remote.sh" ]]; then
  source "${REPO_DIR}/scripts/vwa_env_remote.sh" || true
elif [[ -f "${REPO_DIR}/scripts/vwa_env.sh" ]]; then
  source "${REPO_DIR}/scripts/vwa_env.sh" || true
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-DUMMY_P79_NON_LLM_EVAL}"
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
export P79_DISABLE_STALE_CLEANUP="${P79_DISABLE_STALE_CLEANUP:-1}"

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
  local task_total
  task_total=$(find "${run_dir}/task_configs" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d '[:space:]')
  local done
  done=$(find "${cond_dir}" -type f -path "*/episodes/*_summary_v2.json" 2>/dev/null | wc -l | tr -d '[:space:]')
  # If condition_summary exists but episodes are incomplete, the summary is stale
  # (e.g. manually deleted episodes, bug fixes requiring re-runs). Remove it so
  # run_until_complete will restart the runner for the missing tasks.
  if [[ -f "${cond_dir}/condition_summary_v2.json" ]]; then
    if [[ "${task_total}" -gt 0 && "${done}" -lt "${task_total}" ]]; then
      log "[b0_3mode] ${cid}: condition_summary 存在但仅完成 ${done}/${task_total} tasks，删除过期 summary 重跑"
      rm -f "${cond_dir}/condition_summary_v2.json"
      return 1
    fi
    return 0
  fi
  [[ "${task_total}" -gt 0 && "${done}" -ge "${task_total}" ]] && return 0
  return 1
}

# ---------- Gallery 服务器 ----------
GALLERY_PID=""
if ! ss -tlnp 2>/dev/null | grep -q ':8765 '; then
  log "[b0_3mode] 启动 Gallery 服务器 port=8765"
  nohup "${PYTHON_BIN}" -m http.server 8765 \
    --directory "${RESULTS_BASE}" \
    > "${LOG_DIR}/gallery_server_8765.log" 2>&1 < /dev/null &
  GALLERY_PID=$!
  sleep 1
  kill -0 "${GALLERY_PID}" 2>/dev/null \
    && log "[b0_3mode] Gallery pid=${GALLERY_PID}" \
    || { log "[b0_3mode][warn] Gallery 启动失败"; GALLERY_PID=""; }
else
  log "[b0_3mode] 8765 端口已占用，跳过 Gallery 服务器"
fi

# ---------- Watchdog（全程单实例）----------
start_watchdog() {
  WATCHDOG_PID=""
  [[ "${WATCHDOG_ENABLE}" != "1" ]] && return 0
  local ws="${REPO_DIR}/scripts/experiment_watchdog.py"
  [[ -f "${ws}" ]] || { log "[b0_3mode][warn] watchdog script 不存在"; return 0; }

  # kill 遗留（只 kill 监控同一 OUTPUT_DIR 的 watchdog，避免误杀并行实验）
  local _op
  _op="$(ps -eo pid=,args= | awk -v dir="${OUTPUT_DIR}" '/experiment_watchdog\.py/ && $0 ~ dir && !/awk/ {print $1}')"
  if [[ -n "${_op}" ]]; then
    for p in ${_op}; do kill "${p}" 2>/dev/null || true; done; sleep 1
    for p in ${_op}; do kill -9 "${p}" 2>/dev/null || true; done
  fi

  local wlog="${LOG_DIR}/experiment_watchdog_b0_3mode_${RUN_ID}.log"
  local wstate="${LOG_DIR}/experiment_watchdog_b0_3mode_${RUN_ID}.state.json"
  local wcmd=(
    "${PYTHON_BIN}" -u "${ws}"
    --run-dir "${OUTPUT_DIR}"
    --poll-secs "${WATCHDOG_POLL_SECS}"
    --idle-alert-mins "${WATCHDOG_IDLE_ALERT_MINS}"
    --ntfy-topic "${NTFY_TOPIC}"
    --state-file "${wstate}"
    --aggregate-prefix "${AGGREGATE_PREFIX}"
  )
  [[ -f "${WATCHDOG_GLM_CONFIG}" ]] && wcmd+=(--glm-config "${WATCHDOG_GLM_CONFIG}" --digest-dir "${OUTPUT_DIR}/analysis/digest")
  [[ "${WATCHDOG_NOTIFY_COMPLETION_ENABLE}" == "1" ]] && wcmd+=(--notify-completion)

  mkdir -p "${OUTPUT_DIR}"
  nohup "${wcmd[@]}" > "${wlog}" 2>&1 < /dev/null &
  local pid=$!; sleep 1
  kill -0 "${pid}" 2>/dev/null \
    && { WATCHDOG_PID="${pid}"; log "[b0_3mode] watchdog pid=${pid}"; } \
    || log "[b0_3mode][warn] watchdog 启动失败"
}

stop_watchdog() {
  local pid="${WATCHDOG_PID:-}"; [[ -z "${pid}" ]] && return 0
  kill -0 "${pid}" 2>/dev/null && { kill "${pid}" 2>/dev/null || true; sleep 2; kill -9 "${pid}" 2>/dev/null || true; }
  wait "${pid}" 2>/dev/null || true
  WATCHDOG_PID=""
}

# ---------- Cleanup ----------
ACTIVE_RUNNER_PID=""   # 全局跟踪当前 runner，kill 脚本时一并清理

cleanup() {
  [[ -n "${ACTIVE_RUNNER_PID:-}" ]] && kill -0 "${ACTIVE_RUNNER_PID}" 2>/dev/null \
    && { kill "${ACTIVE_RUNNER_PID}" 2>/dev/null || true; }
  stop_watchdog
  [[ -n "${GALLERY_PID}" ]] && kill -0 "${GALLERY_PID}" 2>/dev/null \
    && { kill "${GALLERY_PID}" 2>/dev/null || true; }
  # 清理 /tmp 的临时 config
  rm -f /tmp/b0_3mode_dom_$$.yaml /tmp/b0_3mode_som_$$.yaml /tmp/b0_3mode_vision_$$.yaml
}
trap cleanup EXIT

# ---------- 生成单模式 temp config ----------
make_single_mode_config() {
  local mode="$1" dest="$2"
  "${PYTHON_BIN}" - << PYEOF
import re, sys
with open("${BASE_CONFIG}") as f:
    content = f.read()
# 替换 observation_mode 列表为单模式
content = re.sub(
    r'observation_mode:\s*\[.*?\]',
    'observation_mode: ["${mode}"]',
    content
)
with open("${dest}", "w") as f:
    f.write(content)
print("ok")
PYEOF
}

# ---------- 单 condition 运行（带内层 watchdog loop）----------
run_condition() {
  local mode="$1" tmp_config="$2" cid="$3"
  log "=== [b0_3mode/${mode}] 启动 condition_id=${cid} ==="

  local log_path="${LOG_DIR}/b0_3mode_${mode}_${RUN_ID}.log"
  mkdir -p "${OUTPUT_DIR}"

  nohup "${PYTHON_BIN}" scripts/run_experiment.py \
    --config "${tmp_config}" \
    --run_id "${RUN_ID}" \
    --log_path "${log_path}" \
    >> "${log_path}" 2>&1 < /dev/null &
  local job_pid=$!
  ACTIVE_RUNNER_PID="${job_pid}"
  log "[b0_3mode/${mode}] PID=${job_pid}"

  local last_count stale_secs=0 watchdog_secs=$(( WATCHDOG_TIMEOUT_MINS * 60 )) next_log_secs=300
  last_count=$(count_episode_summaries "${OUTPUT_DIR}")

  while kill -0 "${job_pid}" 2>/dev/null; do
    sleep "${WATCHDOG_CHECK_SECS}"
    ! kill -0 "${job_pid}" 2>/dev/null && break
    local cur
    cur=$(count_episode_summaries "${OUTPUT_DIR}")
    if [[ "${cur}" -gt "${last_count}" ]]; then
      local new=$(( cur - last_count ))
      log "[b0_3mode/${mode}] +${new} episode(s) total=${cur}，计时重置"
      last_count="${cur}"; stale_secs=0; next_log_secs=300
    else
      stale_secs=$(( stale_secs + WATCHDOG_CHECK_SECS ))
      [[ "${stale_secs}" -ge "${next_log_secs}" ]] && {
        log "[b0_3mode/${mode}] $(( stale_secs / 60 ))min 无新 episode（上限 ${WATCHDOG_TIMEOUT_MINS}min）"
        next_log_secs=$(( next_log_secs + 300 ))
      }
      [[ "${stale_secs}" -ge "${watchdog_secs}" ]] && {
        log "[b0_3mode/${mode}] WATCHDOG: kill PID ${job_pid}"
        ntfy_send "P79 [B0/${mode}] WATCHDOG" "${WATCHDOG_TIMEOUT_MINS}min 无进展，kill 准备 resume" "high"
        kill "${job_pid}" 2>/dev/null || true; sleep 10
        kill -9 "${job_pid}" 2>/dev/null || true
        wait "${job_pid}" 2>/dev/null || true
        return 1
      }
    fi
  done
  wait "${job_pid}" 2>/dev/null || true
  log "=== [b0_3mode/${mode}] 进程退出 ==="
}

run_until_complete() {
  local mode="$1" tmp_config="$2" cid="$3"
  local attempt=0

  if is_condition_complete "${OUTPUT_DIR}" "${cid}"; then
    log "[b0_3mode/${mode}] 已完成，跳过"
    return 0
  fi

  while ! is_condition_complete "${OUTPUT_DIR}" "${cid}"; do
    attempt=$(( attempt + 1 ))
    [[ ${attempt} -gt ${MAX_RESUME_ATTEMPTS} ]] && {
      log "[b0_3mode/${mode}] ERROR: ${MAX_RESUME_ATTEMPTS} 次 resume 后仍未完成"
      ntfy_send "P79 [B0/${mode}] 失败" "已重试 ${MAX_RESUME_ATTEMPTS} 次" "urgent"
      return 1
    }
    [[ ${attempt} -gt 1 ]] && {
      log "[b0_3mode/${mode}] resume ${attempt}/${MAX_RESUME_ATTEMPTS}..."
      ntfy_send "P79 [B0/${mode}] 重试" "第 ${attempt}/${MAX_RESUME_ATTEMPTS} 次 resume" "default"
    }
    run_condition "${mode}" "${tmp_config}" "${cid}" || true
    log "[b0_3mode/${mode}] 等待 15s..."
    sleep 15
  done
  log "[b0_3mode/${mode}] 完成（${attempt} 次）"
}

run_reason_diagnostics() {
  [[ "${REASON_DIAG_ENABLE}" != "1" ]] && return 0
  local diag="${REPO_DIR}/scripts/analysis/analyze_reason_diagnostics.py"
  [[ -f "${diag}" ]] || { log "[b0_3mode] reason diagnostics 脚本不存在，跳过"; return 0; }
  log "[b0_3mode] 运行 reason diagnostics..."
  "${PYTHON_BIN}" "${diag}" \
    --run-dir "${OUTPUT_DIR}" --report --report-language zh --samples-per-bucket 5 \
    >> "${LOG_DIR}/b0_3mode_reason_diag.log" 2>&1 \
    && { log "[b0_3mode] reason diagnostics 完成"
         ntfy_send "P79 [B0_3mode] 归因完成" "run_id=${RUN_ID}" "default"; } \
    || { log "[b0_3mode][warn] reason diagnostics 失败（非阻塞）"
         ntfy_send "P79 [B0_3mode] 归因失败" "查看 logs/b0_3mode_reason_diag.log" "default"; }
}

# ---------- 生成 temp configs ----------
DOM_CONFIG="/tmp/b0_3mode_dom_$$.yaml"
SOM_CONFIG="/tmp/b0_3mode_som_$$.yaml"
VISION_CONFIG="/tmp/b0_3mode_vision_$$.yaml"

make_single_mode_config "dom"    "${DOM_CONFIG}"
make_single_mode_config "som"    "${SOM_CONFIG}"
make_single_mode_config "vision" "${VISION_CONFIG}"

# ---------- 主流程 ----------
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
ntfy_send "P79 [B0_3mode] 启动" "run_id=${RUN_ID}" "default"

start_watchdog

# 1) DOM
log "======== [1/3] DOM ========"
ntfy_send "P79 [B0/dom] 开始" "run_id=${RUN_ID}" "default"
run_until_complete "dom" "${DOM_CONFIG}" "phase1_dom_router_0"
[[ "${NTFY_MINIMAL_MODE}" != "1" ]] && ntfy_send "P79 [B0/dom] 完成" "run_id=${RUN_ID}" "default"

# reset → SOM
log "======== reset classifieds before SOM ========"
reset_vwa_sites "classifieds" "b0_3mode"
sleep 10  # 额外缓冲
refresh_classifieds_auth || {
  log "[b0_3mode][error] SOM 前 auth 刷新失败，等待 30s 后重试..."
  sleep 30
  refresh_classifieds_auth || { log "[b0_3mode][fatal] auth 刷新两次均失败，中止"; exit 1; }
}

# 2) SOM
log "======== [2/3] SOM ========"
ntfy_send "P79 [B0/som] 开始" "run_id=${RUN_ID}" "default"
run_until_complete "som" "${SOM_CONFIG}" "phase1_som_router_0"
[[ "${NTFY_MINIMAL_MODE}" != "1" ]] && ntfy_send "P79 [B0/som] 完成" "run_id=${RUN_ID}" "default"

# reset → VISION
log "======== reset classifieds before Vision ========"
reset_vwa_sites "classifieds" "b0_3mode"
sleep 10
refresh_classifieds_auth || {
  log "[b0_3mode][error] Vision 前 auth 刷新失败，等待 30s 后重试..."
  sleep 30
  refresh_classifieds_auth || { log "[b0_3mode][fatal] auth 刷新两次均失败，中止"; exit 1; }
}

# 3) VISION
log "======== [3/3] Vision ========"
ntfy_send "P79 [B0/vision] 开始" "run_id=${RUN_ID}" "default"
run_until_complete "vision" "${VISION_CONFIG}" "phase1_vision_router_0"
[[ "${NTFY_MINIMAL_MODE}" != "1" ]] && ntfy_send "P79 [B0/vision] 完成" "run_id=${RUN_ID}" "default"

log "======== final reset classifieds after Vision ========"
reset_vwa_sites "classifieds" "b0_3mode_final" || true
sleep 5

stop_watchdog

# ---------- 完成 ----------
run_reason_diagnostics

log "========================================================"
log "=== B0 三模式 classifieds 全部完成！==="
log "=== Gallery: http://localhost:8765/${AGGREGATE_PREFIX}/gallery.html ==="
log "========================================================"
ntfy_send "P79 [B0_3mode] 完成!" "run_id=${RUN_ID}；dom+som+vision 全部跑完" "high"
