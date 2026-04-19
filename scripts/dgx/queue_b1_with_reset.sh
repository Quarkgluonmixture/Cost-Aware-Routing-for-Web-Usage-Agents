#!/usr/bin/env bash
# queue_b1_with_reset.sh — B1 三模式 reddit+shopping（每 condition 间 reset 站点）
#
# 每个 condition 独立跑，condition 间自动 reset 站点，消除跨模式数据污染。
# reddit DOM 已有 ~112 个完成 episode（污染的 10 个已清），会自动 resume。
#
# 用法:
#   nohup bash scripts/dgx/queue_b1_with_reset.sh \
#     > logs/queue_b1_with_reset_main.log 2>&1 &
#
# Gallery: http://localhost:8765/visualwebarena/phase1/B1_3mode/gallery.html
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

# ---------- 通用 reset 工具 ----------
source "${REPO_DIR}/scripts/reset_vwa_sites.sh"

# refresh_site_auth <site> — 站点 reset 后重新登录，刷新 .auth/<site>_state.json
# 支持 classifieds / reddit / shopping
refresh_site_auth() {
  local site="$1"
  local auth_file="${REPO_DIR}/.auth/${site}_state.json"
  log "[b1_3mode] 刷新 ${site} auth state..."
  DATASET=visualwebarena "${PYTHON_BIN:-python3}" - <<PYEOF
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
if site == 'classifieds':
    page.locator('#email').fill(username)
    page.locator('#password').fill(password)
    page.get_by_role('button', name='Log in').click()
elif site == 'reddit':
    page.get_by_label('Username').fill(username)
    page.get_by_label('Password').fill(password)
    page.get_by_role('button', name='Log in').click()
elif site == 'shopping':
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
    log "[b1_3mode] ${site} auth state 已刷新"
    return 0
  else
    log "[b1_3mode][error] ${site} auth 刷新失败 rc=${rc}（auth_file=$(wc -c < "${auth_file}" 2>/dev/null || echo missing) bytes）"
    return 1
  fi
}

# refresh_site_auth_retry <site> <label> — 带指数退避的 auth 刷新（最多 5 次）
AUTH_REFRESH_MAX_ATTEMPTS="${AUTH_REFRESH_MAX_ATTEMPTS:-5}"
refresh_site_auth_retry() {
  local site="$1" label="${2:-auth}"
  local attempt=0 delay=10
  while [[ $attempt -lt $AUTH_REFRESH_MAX_ATTEMPTS ]]; do
    attempt=$(( attempt + 1 ))
    if refresh_site_auth "${site}"; then
      return 0
    fi
    if [[ $attempt -ge $AUTH_REFRESH_MAX_ATTEMPTS ]]; then
      break
    fi
    log "[b1_3mode][${label}] auth 刷新第 ${attempt}/${AUTH_REFRESH_MAX_ATTEMPTS} 次失败，${delay}s 后重试..."
    ntfy_send "P79 [B1/${label}] auth retry" "${site} 第 ${attempt} 次失败，${delay}s 后重试" "default"
    sleep "${delay}"
    delay=$(( delay * 2 ))
  done
  log "[b1_3mode][${label}][fatal] ${site} auth 刷新 ${AUTH_REFRESH_MAX_ATTEMPTS} 次均失败"
  ntfy_send "P79 [B1/${label}] auth FAILED" "${site} ${AUTH_REFRESH_MAX_ATTEMPTS} 次失败" "urgent"
  return 1
}

# ---------- 配置 ----------
BASELINE_CONFIG="${REPO_DIR}/configs/exp_v2_qwen3vl4b_B1_baseline.yaml"
LOG_DIR="${REPO_DIR}/logs"
RESULTS_BASE="${REPO_DIR}/results/visualwebarena/phase1"
mkdir -p "${LOG_DIR}"

# --- run_id 配置 ---
RUN_ID_CLASSIFIEDS="${RUN_ID_CLASSIFIEDS:-B1_3mode_classifieds_20260413}"
RUN_ID_REDDIT="${RUN_ID_REDDIT:-B1_3mode_reddit_20260413}"
RUN_ID_SHOPPING="${RUN_ID_SHOPPING:-B1_3mode_shopping_20260413}"

# 每个 condition 最多自动 resume 次数
MAX_RESUME_ATTEMPTS="${MAX_RESUME_ATTEMPTS:-10}"

# 内层进度 watchdog 配置（检测 runner 卡死）
WATCHDOG_TIMEOUT_MINS="${WATCHDOG_TIMEOUT_MINS:-35}"
WATCHDOG_CHECK_SECS="${WATCHDOG_CHECK_SECS:-60}"

# ntfy 推送配置（export for runner.py retry-pass notification）
export NTFY_TOPIC="${NTFY_TOPIC:-p79-exp-dgx-spark}"
NTFY_URL="https://ntfy.sh/${NTFY_TOPIC}"
NTFY_MINIMAL_MODE="${NTFY_MINIMAL_MODE:-1}"

# experiment_watchdog 配置（外层监控，每站点单实例）
EXP_WATCHDOG_ENABLE="${EXP_WATCHDOG_ENABLE:-1}"
EXP_WATCHDOG_POLL_SECS="${EXP_WATCHDOG_POLL_SECS:-30}"
EXP_WATCHDOG_IDLE_ALERT_MINS="${EXP_WATCHDOG_IDLE_ALERT_MINS:-20}"
EXP_WATCHDOG_NOTIFY_COMPLETION_ENABLE="${EXP_WATCHDOG_NOTIFY_COMPLETION_ENABLE:-1}"
EXP_WATCHDOG_GLM_CONFIG="${EXP_WATCHDOG_GLM_CONFIG:-${REPO_DIR}/.auth/glm}"
EXP_WATCHDOG_PID=""

REASON_DIAG_ENABLE="${REASON_DIAG_ENABLE:-1}"
AGGREGATE_PREFIX="B1_3mode"

# DGX Spark 环境变量（本地 GPU 必须设置）
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
export OPENAI_API_KEY="${OPENAI_API_KEY:-DUMMY_P79_NON_LLM_EVAL}"
export P79_DISABLE_STALE_CLEANUP="${P79_DISABLE_STALE_CLEANUP:-1}"
export WIKIPEDIA_ZIM_VERSION="${WIKIPEDIA_ZIM_VERSION:-wikipedia_en_all_maxi_2025-08}"

# ---------- Python 解释器 ----------
if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "[b1_3mode][error] 找不到 Python 解释器" >&2; exit 127
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
  # If condition_summary exists but episodes are incomplete, the summary is stale
  # (e.g. manually deleted episodes, bug fixes requiring re-runs). Remove it so
  # run_until_complete will restart the runner for the missing tasks.
  if [[ -f "${cond_dir}/condition_summary_v2.json" ]]; then
    if [[ "${task_total}" -gt 0 && "${done}" -lt "${task_total}" ]]; then
      log "[queue_b1] ${cid}: condition_summary 存在但仅完成 ${done}/${task_total} tasks，删除过期 summary 重跑"
      rm -f "${cond_dir}/condition_summary_v2.json"
      return 1
    fi
    return 0
  fi
  [[ "${task_total}" -gt 0 && "${done}" -ge "${task_total}" ]] && return 0
  return 1
}

# ---------- 生成 site+mode 组合的 temp config ----------
make_site_mode_config() {
  local site="$1" mode="$2" dest="$3"
  "${PYTHON_BIN}" - << PYEOF
import re
with open("${BASELINE_CONFIG}") as f:
    content = f.read()
content = re.sub(r'include_sites:\s*\[.*?\]', 'include_sites: ["${site}"]', content)
content = re.sub(r'observation_mode:\s*\[.*?\]', 'observation_mode: ["${mode}"]', content)
with open("${dest}", "w") as f:
    f.write(content)
print("ok")
PYEOF
}

# ---------- Gallery 服务器（启动一次，如果未运行）----------
GALLERY_PID=""
start_gallery_server() {
  if ! ss -tlnp 2>/dev/null | grep -q ':8765 '; then
    log "[b1_3mode] 启动 Gallery 服务器 port=8765"
    nohup "${PYTHON_BIN}" -m http.server 8765 \
      --directory "${REPO_DIR}/results" \
      > "${LOG_DIR}/gallery_server_8765.log" 2>&1 < /dev/null &
    GALLERY_PID=$!
    sleep 1
    kill -0 "${GALLERY_PID}" 2>/dev/null \
      && log "[b1_3mode] Gallery pid=${GALLERY_PID} url=http://localhost:8765/visualwebarena/phase1/${AGGREGATE_PREFIX}/gallery.html" \
      || { log "[b1_3mode][warn] Gallery 启动失败"; GALLERY_PID=""; }
  else
    log "[b1_3mode] 8765 端口已占用，跳过 Gallery 服务器（可能已由 B0 启动）"
  fi
}

# ---------- experiment_watchdog（每站点单实例）----------
start_exp_watchdog() {
  local run_id="$1" label="$2"
  EXP_WATCHDOG_PID=""
  [[ "${EXP_WATCHDOG_ENABLE}" != "1" ]] && return 0
  local ws="${REPO_DIR}/scripts/experiment_watchdog.py"
  [[ -f "${ws}" ]] || { log "[${label}][warn] experiment_watchdog.py 不存在，跳过"; return 0; }

  local run_dir="${RESULTS_BASE}/${run_id}"
  # 只 kill 监控同一 run_dir 的遗留 watchdog（避免误杀并行实验的 watchdog）
  local _op
  _op="$(ps -eo pid=,args= | awk -v dir="${run_dir}" '/experiment_watchdog\.py/ && $0 ~ dir && !/awk/ {print $1}')"
  if [[ -n "${_op}" ]]; then
    for p in ${_op}; do kill "${p}" 2>/dev/null || true; done; sleep 1
    for p in ${_op}; do kill -9 "${p}" 2>/dev/null || true; done
  fi

  local wlog="${LOG_DIR}/experiment_watchdog_b1_${label}_${run_id}.log"
  local wstate="${LOG_DIR}/experiment_watchdog_b1_${label}_${run_id}.state.json"
  local wcmd=(
    "${PYTHON_BIN}" -u "${ws}"
    --run-dir "${run_dir}"
    --poll-secs "${EXP_WATCHDOG_POLL_SECS}"
    --idle-alert-mins "${EXP_WATCHDOG_IDLE_ALERT_MINS}"
    --ntfy-topic "${NTFY_TOPIC}"
    --state-file "${wstate}"
    --aggregate-prefix "${AGGREGATE_PREFIX}"
  )
  [[ -f "${EXP_WATCHDOG_GLM_CONFIG}" ]] && wcmd+=(--glm-config "${EXP_WATCHDOG_GLM_CONFIG}" --digest-dir "${run_dir}/analysis/digest")
  [[ "${EXP_WATCHDOG_NOTIFY_COMPLETION_ENABLE}" == "1" ]] && wcmd+=(--notify-completion)

  mkdir -p "${run_dir}"
  nohup "${wcmd[@]}" > "${wlog}" 2>&1 < /dev/null &
  local pid=$!; sleep 1
  kill -0 "${pid}" 2>/dev/null \
    && { EXP_WATCHDOG_PID="${pid}"; log "[${label}] experiment_watchdog pid=${pid}"; } \
    || log "[${label}][warn] experiment_watchdog 启动失败"
}

stop_exp_watchdog() {
  local label="$1"
  local pid="${EXP_WATCHDOG_PID:-}"; [[ -z "${pid}" ]] && return 0
  kill -0 "${pid}" 2>/dev/null && {
    kill "${pid}" 2>/dev/null || true; sleep 2
    kill -9 "${pid}" 2>/dev/null || true
  }
  wait "${pid}" 2>/dev/null || true
  EXP_WATCHDOG_PID=""
}

# ---------- 单 condition 运行（带内层进度 watchdog）----------
run_condition_foreground() {
  local mode="$1" tmp_config="$2" run_dir="$3" label="$4"
  local run_id
  run_id="$(basename "${run_dir}")"

  log "=== [B1/${label}/${mode}] 启动 run_id=${run_id} ==="

  local log_path="${LOG_DIR}/b1_3mode_${label}_${mode}_${run_id}.log"
  mkdir -p "${run_dir}"

  nohup "${PYTHON_BIN}" scripts/run_experiment.py \
    --config "${tmp_config}" \
    --run_id "${run_id}" \
    --log_path "${log_path}" \
    >> "${log_path}" 2>&1 < /dev/null &
  local job_pid=$!
  ACTIVE_RUNNER_PID="${job_pid}"
  log "[B1/${label}/${mode}] PID=${job_pid}"

  local last_count stale_secs=0 watchdog_secs=$(( WATCHDOG_TIMEOUT_MINS * 60 )) next_log_secs=300
  last_count=$(count_episode_summaries "${run_dir}")

  while kill -0 "${job_pid}" 2>/dev/null; do
    sleep "${WATCHDOG_CHECK_SECS}"
    ! kill -0 "${job_pid}" 2>/dev/null && break
    local cur
    cur=$(count_episode_summaries "${run_dir}")
    if [[ "${cur}" -gt "${last_count}" ]]; then
      local new=$(( cur - last_count ))
      log "[B1/${label}/${mode}] +${new} episode(s) total=${cur}，计时重置"
      last_count="${cur}"; stale_secs=0; next_log_secs=300
    else
      stale_secs=$(( stale_secs + WATCHDOG_CHECK_SECS ))
      [[ "${stale_secs}" -ge "${next_log_secs}" ]] && {
        log "[B1/${label}/${mode}] $(( stale_secs / 60 ))min 无新 episode（上限 ${WATCHDOG_TIMEOUT_MINS}min）"
        next_log_secs=$(( next_log_secs + 300 ))
      }
      [[ "${stale_secs}" -ge "${watchdog_secs}" ]] && {
        log "[B1/${label}/${mode}] WATCHDOG: kill PID ${job_pid}"
        ntfy_send "P79 [B1/${label}/${mode}] WATCHDOG" "${WATCHDOG_TIMEOUT_MINS}min 无进展，kill 准备 resume" "high"
        kill "${job_pid}" 2>/dev/null || true; sleep 10
        kill -9 "${job_pid}" 2>/dev/null || true
        wait "${job_pid}" 2>/dev/null || true
        return 1
      }
    fi
  done
  wait "${job_pid}" 2>/dev/null || true
  log "=== [B1/${label}/${mode}] 进程退出 ==="
}

run_condition_until_complete() {
  local mode="$1" tmp_config="$2" run_dir="$3" cid="$4" label="$5"
  local attempt=0

  if is_condition_complete "${run_dir}" "${cid}"; then
    log "[B1/${label}/${mode}] 已完成，跳过"
    return 0
  fi

  while ! is_condition_complete "${run_dir}" "${cid}"; do
    attempt=$(( attempt + 1 ))
    [[ ${attempt} -gt ${MAX_RESUME_ATTEMPTS} ]] && {
      log "[B1/${label}/${mode}] ERROR: ${MAX_RESUME_ATTEMPTS} 次 resume 后仍未完成"
      ntfy_send "P79 [B1/${label}/${mode}] 失败" "已重试 ${MAX_RESUME_ATTEMPTS} 次" "urgent"
      return 1
    }
    [[ ${attempt} -gt 1 ]] && {
      log "[B1/${label}/${mode}] resume ${attempt}/${MAX_RESUME_ATTEMPTS}..."
      ntfy_send "P79 [B1/${label}/${mode}] 重试" "第 ${attempt}/${MAX_RESUME_ATTEMPTS} 次 resume" "default"
    }
    run_condition_foreground "${mode}" "${tmp_config}" "${run_dir}" "${label}" || true
    log "[B1/${label}/${mode}] 等待 15s..."
    sleep 15
  done
  log "[B1/${label}/${mode}] 完成（${attempt} 次）"
}

run_reason_diagnostics() {
  [[ "${REASON_DIAG_ENABLE}" != "1" ]] && return 0
  local run_dir="$1" label="$2"
  local diag="${REPO_DIR}/scripts/analysis/analyze_reason_diagnostics.py"
  [[ -f "${diag}" ]] || { log "[B1/${label}] reason diagnostics 脚本不存在，跳过"; return 0; }
  log "[B1/${label}] 运行 reason diagnostics..."
  "${PYTHON_BIN}" "${diag}" \
    --run-dir "${run_dir}" --report --report-language zh --samples-per-bucket 5 \
    >> "${LOG_DIR}/b1_3mode_reason_diag_${label}.log" 2>&1 \
    && { log "[B1/${label}] reason diagnostics 完成"
         ntfy_send "P79 [B1/${label}] 归因完成" "run_id=$(basename "${run_dir}")" "default"; } \
    || {
      log "[B1/${label}][warn] reason diagnostics 失败（非阻塞）"
      ntfy_send "P79 [B1/${label}] 归因失败" "查看 logs/b1_3mode_reason_diag_${label}.log" "default"
    }
}

# ---------- 单站三模式（dom → reset → som → reset → vision）----------
run_site_3mode_with_reset() {
  local site="$1" run_id="$2"
  local run_dir="${RESULTS_BASE}/${run_id}"
  local label="${site}"

  log "========================================================"
  log "=== [B1/${label}] 开始三模式（dom → reset → som → reset → vision）==="
  log "=== run_id=${run_id} run_dir=${run_dir} ==="
  log "========================================================"

  # 生成三个临时 config（site + mode 各一份）
  local dom_config="/tmp/b1_3mode_${site}_dom_$$.yaml"
  local som_config="/tmp/b1_3mode_${site}_som_$$.yaml"
  local vision_config="/tmp/b1_3mode_${site}_vision_$$.yaml"

  make_site_mode_config "${site}" "dom"    "${dom_config}"
  make_site_mode_config "${site}" "som"    "${som_config}"
  make_site_mode_config "${site}" "vision" "${vision_config}"

  mkdir -p "${run_dir}"
  ntfy_send "P79 [B1/${label}] 启动" "run_id=${run_id}" "default"

  start_exp_watchdog "${run_id}" "${label}"

  # 1) DOM — auth refresh before first condition (SOM/Vision already have it)
  refresh_site_auth_retry "${site}" "${label}/dom" || { log "[b1][fatal] ${site} DOM 前 auth 失败，中止"; exit 1; }
  log "======== [B1/${label} 1/3] DOM ========"
  ntfy_send "P79 [B1/${label}/dom] 开始" "run_id=${run_id}" "default"
  run_condition_until_complete "dom" "${dom_config}" "${run_dir}" "phase1_dom_router_0" "${label}"
  [[ "${NTFY_MINIMAL_MODE}" != "1" ]] && ntfy_send "P79 [B1/${label}/dom] 完成" "run_id=${run_id}" "default"

  # reset → SOM
  log "======== reset ${site} before SOM ========"
  reset_vwa_sites "${site}" "b1_3mode_${site}" || true
  sleep 10
  refresh_site_auth_retry "${site}" "${label}/som" || { log "[b1_3mode][fatal] ${site} SOM 前 auth 失败，中止"; exit 1; }

  # 2) SOM
  log "======== [B1/${label} 2/3] SOM ========"
  ntfy_send "P79 [B1/${label}/som] 开始" "run_id=${run_id}" "default"
  run_condition_until_complete "som" "${som_config}" "${run_dir}" "phase1_som_router_0" "${label}"
  [[ "${NTFY_MINIMAL_MODE}" != "1" ]] && ntfy_send "P79 [B1/${label}/som] 完成" "run_id=${run_id}" "default"

  # reset → VISION
  log "======== reset ${site} before Vision ========"
  reset_vwa_sites "${site}" "b1_3mode_${site}" || true
  sleep 10
  refresh_site_auth_retry "${site}" "${label}/vision" || { log "[b1_3mode][fatal] ${site} Vision 前 auth 失败，中止"; exit 1; }

  # 3) VISION
  log "======== [B1/${label} 3/3] Vision ========"
  ntfy_send "P79 [B1/${label}/vision] 开始" "run_id=${run_id}" "default"
  run_condition_until_complete "vision" "${vision_config}" "${run_dir}" "phase1_vision_router_0" "${label}"
  [[ "${NTFY_MINIMAL_MODE}" != "1" ]] && ntfy_send "P79 [B1/${label}/vision] 完成" "run_id=${run_id}" "default"

  log "======== final reset ${site} after Vision ========"
  reset_vwa_sites "${site}" "b1_3mode_${site}_final" || true
  sleep 5

  stop_exp_watchdog "${label}"

  run_reason_diagnostics "${run_dir}" "${label}"

  ntfy_send "P79 [B1/${label}] 完成!" "run_id=${run_id}；dom+som+vision 全部跑完" "high"
  log "========================================================"
  log "=== [B1/${label}] 三模式全部完成！==="
  log "========================================================"

  rm -f "${dom_config}" "${som_config}" "${vision_config}"
}

# ---------- Cleanup ----------
ACTIVE_RUNNER_PID=""   # 全局跟踪当前 runner，kill 脚本时一并清理

cleanup() {
  [[ -n "${ACTIVE_RUNNER_PID:-}" ]] && kill -0 "${ACTIVE_RUNNER_PID}" 2>/dev/null \
    && { kill "${ACTIVE_RUNNER_PID}" 2>/dev/null || true; }
  stop_exp_watchdog "cleanup"
  [[ -n "${GALLERY_PID:-}" ]] && kill -0 "${GALLERY_PID}" 2>/dev/null \
    && { kill "${GALLERY_PID}" 2>/dev/null || true; }
  rm -f /tmp/b1_3mode_*_$$.yaml 2>/dev/null || true
}
trap cleanup EXIT

# ---------- 主流程 ----------
log "========================================================"
log "=== B1 三模式队列启动（带 reset）==="
log "=== 顺序: classifieds → reddit → shopping ==="
log "=== RUN_ID_CLASSIFIEDS=${RUN_ID_CLASSIFIEDS} ==="
log "=== RUN_ID_REDDIT=${RUN_ID_REDDIT} ==="
log "=== RUN_ID_SHOPPING=${RUN_ID_SHOPPING} ==="
log "=== MAX_RESUME_ATTEMPTS=${MAX_RESUME_ATTEMPTS} ==="
log "=== WATCHDOG_TIMEOUT_MINS=${WATCHDOG_TIMEOUT_MINS} ==="
log "=== AGGREGATE_PREFIX=${AGGREGATE_PREFIX} ==="
log "========================================================"

start_gallery_server

ntfy_send "P79 [B1_3mode] 队列启动" "顺序: classifieds → reddit → shopping，带模式间 reset" "default"

# 1) Classifieds（stub 数据，is_condition_complete 会自动跳过）
run_site_3mode_with_reset "classifieds" "${RUN_ID_CLASSIFIEDS}"
log "classifieds 完成. Waiting 15s..."
sleep 15

# 2) Reddit
run_site_3mode_with_reset "reddit" "${RUN_ID_REDDIT}"
log "reddit 完成. Waiting 15s..."
sleep 15

# 3) Shopping
run_site_3mode_with_reset "shopping" "${RUN_ID_SHOPPING}"

log "========================================================"
log "=== B1 三模式全部完成（classifieds + reddit + shopping）==="
log "=== Gallery: http://localhost:8765/visualwebarena/phase1/${AGGREGATE_PREFIX}/gallery.html ==="
log "========================================================"
ntfy_send "P79 [B1_3mode] 全部完成!" "classifieds + reddit + shopping 三模式全部跑完" "high"
