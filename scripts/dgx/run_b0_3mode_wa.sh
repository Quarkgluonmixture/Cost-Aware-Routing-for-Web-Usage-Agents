#!/usr/bin/env bash
# run_b0_3mode_wa.sh — B0 三模式 WA per-site（shopping → shopping_admin → reddit）
#
# 每站点 dom→reset→som→reset→vision，三站顺序执行。
# 基于 run_b0_3mode_classifieds.sh 模板。
#
# 用法:
#   bash scripts/dgx/run_b0_3mode_wa.sh
#   B0_WA_SITE=shopping bash scripts/dgx/run_b0_3mode_wa.sh  # 只跑一站
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
      echo "[b0_wa] Loaded PROXY_API_KEY from ${AUTH_FILE}" >&2
    else
      echo "[b0_wa][error] ${AUTH_FILE} 存在但为空" >&2; exit 1
    fi
  else
    echo "[b0_wa][error] ${AUTH_FILE} 不存在，且 PROXY_API_KEY 未设置" >&2; exit 1
  fi
else
  echo "[b0_wa] PROXY_API_KEY 已由环境变量提供" >&2
fi

# ---------- 配置 ----------
LOG_DIR="${REPO_DIR}/logs"
RESULTS_BASE="${REPO_DIR}/results/webarena/phase1"
mkdir -p "${LOG_DIR}"

export NTFY_TOPIC="${NTFY_TOPIC:-p79-exp-dgx-spark}"
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
AGGREGATE_PREFIX="B0_wa_3mode"

# 只跑指定站（可选）
B0_WA_SITE="${B0_WA_SITE:-all}"
# per-site configs
CONFIGS_SHOPPING="${REPO_DIR}/configs/exp_v2_B0_3mode_wa_shopping.yaml"
CONFIGS_SHOPPING_ADMIN="${REPO_DIR}/configs/exp_v2_B0_3mode_wa_shopping_admin.yaml"
CONFIGS_REDDIT="${REPO_DIR}/configs/exp_v2_B0_3mode_wa_reddit.yaml"

# ---------- Python 解释器 ----------
if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "[b0_wa][error] 找不到 Python 解释器" >&2; exit 127
fi

# ---------- VWA 站点环境 ----------
if [[ -n "${VWA_ENV_FILE:-}" ]] && [[ -f "${VWA_ENV_FILE}" ]]; then
  source "${VWA_ENV_FILE}" || true
elif [[ -f "${REPO_DIR}/scripts/vwa_env_remote.sh" ]]; then
  source "${REPO_DIR}/scripts/vwa_env_remote.sh" || true
elif [[ -f "${REPO_DIR}/scripts/vwa_env.sh" ]]; then
  source "${REPO_DIR}/scripts/vwa_env.sh" || true
fi
# Override DATASET for WA (vwa_env scripts may export visualwebarena)
export DATASET=webarena

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
  local task_total done
  task_total=$(find "${run_dir}/task_configs" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d '[:space:]')
  done=$(find "${cond_dir}" -type f -path "*/episodes/*_summary_v2.json" 2>/dev/null | wc -l | tr -d '[:space:]')
  if [[ -f "${cond_dir}/condition_summary_v2.json" ]]; then
    if [[ "${task_total}" -gt 0 && "${done}" -lt "${task_total}" ]]; then
      log "[b0_wa] ${cid}: condition_summary 存在但仅完成 ${done}/${task_total} tasks，删除过期 summary 重跑"
      rm -f "${cond_dir}/condition_summary_v2.json"
      return 1
    fi
    return 0
  fi
  [[ "${task_total}" -gt 0 && "${done}" -ge "${task_total}" ]] && return 0
  return 1
}

# ---------- Watchdog ----------
start_watchdog() {
  local output_dir="$1"
  WATCHDOG_PID=""
  [[ "${WATCHDOG_ENABLE}" != "1" ]] && return 0
  local ws="${REPO_DIR}/scripts/experiment_watchdog.py"
  [[ -f "${ws}" ]] || { log "[b0_wa][warn] watchdog script 不存在"; return 0; }

  local _op
  _op="$(ps -eo pid=,args= | awk -v dir="${output_dir}" '/experiment_watchdog\.py/ && $0 ~ dir && !/awk/ {print $1}')"
  if [[ -n "${_op}" ]]; then
    for p in ${_op}; do kill "${p}" 2>/dev/null || true; done; sleep 1
    for p in ${_op}; do kill -9 "${p}" 2>/dev/null || true; done
  fi

  local run_id
  run_id="$(basename "${output_dir}")"
  local wlog="${LOG_DIR}/experiment_watchdog_b0_wa_${run_id}.log"
  local wstate="${LOG_DIR}/experiment_watchdog_b0_wa_${run_id}.state.json"
  local wcmd=(
    "${PYTHON_BIN}" -u "${ws}"
    --run-dir "${output_dir}"
    --poll-secs "${WATCHDOG_POLL_SECS}"
    --idle-alert-mins "${WATCHDOG_IDLE_ALERT_MINS}"
    --ntfy-topic "${NTFY_TOPIC}"
    --state-file "${wstate}"
    --aggregate-prefix "${AGGREGATE_PREFIX}"
  )
  [[ -f "${WATCHDOG_GLM_CONFIG}" ]] && wcmd+=(--glm-config "${WATCHDOG_GLM_CONFIG}" --digest-dir "${output_dir}/analysis/digest")
  [[ "${WATCHDOG_NOTIFY_COMPLETION_ENABLE}" == "1" ]] && wcmd+=(--notify-completion)

  mkdir -p "${output_dir}"
  nohup "${wcmd[@]}" > "${wlog}" 2>&1 < /dev/null &
  local pid=$!; sleep 1
  kill -0 "${pid}" 2>/dev/null \
    && { WATCHDOG_PID="${pid}"; log "[b0_wa] watchdog pid=${pid}"; } \
    || log "[b0_wa][warn] watchdog 启动失败"
}

stop_watchdog() {
  local pid="${WATCHDOG_PID:-}"; [[ -z "${pid}" ]] && return 0
  kill -0 "${pid}" 2>/dev/null && { kill "${pid}" 2>/dev/null || true; sleep 2; kill -9 "${pid}" 2>/dev/null || true; }
  wait "${pid}" 2>/dev/null || true
  WATCHDOG_PID=""
}

# ---------- Auth refresh ----------
refresh_site_auth() {
  local site="$1"
  local auth_file="${REPO_DIR}/.auth/${site}_state.json"
  log "[b0_wa] 刷新 ${site} auth state..."
  local dataset="webarena"  # WA script — all sites are webarena
  DATASET="${dataset}" "${PYTHON_BIN}" - <<PYEOF
import os, sys, time
sys.path.insert(0, '${REPO_DIR}/external/visualwebarena')
from playwright.sync_api import sync_playwright

site = '${site}'
ACCOUNTS = {
    'classifieds': ('blake.sullivan@gmail.com', 'Password.123'),
    'reddit':      ('MarvelsGrantMan136',        'test1234'),
    'shopping':    ('emma.lopez@gmail.com',       'Password.123'),
    'shopping_admin': ('admin',                   'admin1234'),
}
base_urls = {
    'classifieds': os.environ.get('CLASSIFIEDS', 'http://100.95.81.103:9980'),
    'reddit':      os.environ.get('REDDIT',      'http://100.95.81.103:9999'),
    'shopping':    os.environ.get('SHOPPING',    'http://100.95.81.103:7770'),
    'shopping_admin': os.environ.get('SHOPPING_ADMIN', 'http://100.95.81.103:7780'),
}
login_paths = {
    'classifieds': '/index.php?page=login',
    'reddit':      '/login',
    'shopping':    '/customer/account/login/',
    'shopping_admin': '/admin',
}
username, password = ACCOUNTS[site]
base_url = base_urls[site]
login_path = login_paths[site]

cm = sync_playwright()
playwright = cm.__enter__()
browser = playwright.chromium.launch(headless=True)
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
elif site == 'shopping_admin':
    page.locator('#username').fill(username)
    page.locator('#login').fill(password)
    page.get_by_role('button', name='Sign in').click()
time.sleep(2)
context.storage_state(path='${auth_file}')
cm.__exit__(None, None, None)
print('${site} auth refreshed -> ' + page.url)
PYEOF
  local rc=$?
  if [[ $rc -eq 0 ]] && [[ -s "${auth_file}" ]]; then
    log "[b0_wa] ${site} auth state 已刷新"
    return 0
  else
    log "[b0_wa][error] ${site} auth 刷新失败 rc=${rc}"
    return 1
  fi
}

# ---------- Cleanup ----------
ACTIVE_RUNNER_PID=""
GALLERY_PID=""

cleanup() {
  [[ -n "${ACTIVE_RUNNER_PID:-}" ]] && kill -0 "${ACTIVE_RUNNER_PID}" 2>/dev/null \
    && { kill "${ACTIVE_RUNNER_PID}" 2>/dev/null || true; }
  stop_watchdog
  [[ -n "${GALLERY_PID}" ]] && kill -0 "${GALLERY_PID}" 2>/dev/null \
    && { kill "${GALLERY_PID}" 2>/dev/null || true; }
  rm -f /tmp/b0_wa_3mode_*_$$.yaml 2>/dev/null || true
}
trap cleanup EXIT

# ---------- Gallery ----------
if ! ss -tlnp 2>/dev/null | grep -q ':8765 '; then
  log "[b0_wa] 启动 Gallery 服务器 port=8765"
  nohup "${PYTHON_BIN}" -m http.server 8765 \
    --directory "${REPO_DIR}/results" \
    > "${LOG_DIR}/gallery_server_8765.log" 2>&1 < /dev/null &
  GALLERY_PID=$!
  sleep 1
  kill -0 "${GALLERY_PID}" 2>/dev/null \
    && log "[b0_wa] Gallery pid=${GALLERY_PID}" \
    || { log "[b0_wa][warn] Gallery 启动失败"; GALLERY_PID=""; }
else
  log "[b0_wa] 8765 端口已占用，跳过 Gallery 服务器"
fi

# ---------- 生成单模式 temp config ----------
make_single_mode_config() {
  local base_config="$1" mode="$2" dest="$3"
  "${PYTHON_BIN}" - << PYEOF
import re
with open("${base_config}") as f:
    content = f.read()
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

# ---------- 单 condition 运行 ----------
run_condition() {
  local mode="$1" tmp_config="$2" output_dir="$3"
  local run_id
  run_id="$(basename "${output_dir}")"
  log "=== [b0_wa/${mode}] 启动 run_id=${run_id} ==="

  local log_path="${LOG_DIR}/b0_wa_3mode_${mode}_${run_id}.log"
  mkdir -p "${output_dir}"

  nohup "${PYTHON_BIN}" scripts/run_experiment.py \
    --config "${tmp_config}" \
    --run_id "${run_id}" \
    --log_path "${log_path}" \
    >> "${log_path}" 2>&1 < /dev/null &
  local job_pid=$!
  ACTIVE_RUNNER_PID="${job_pid}"
  log "[b0_wa/${mode}] PID=${job_pid}"

  local last_count stale_secs=0 watchdog_secs=$(( WATCHDOG_TIMEOUT_MINS * 60 )) next_log_secs=300
  last_count=$(count_episode_summaries "${output_dir}")

  while kill -0 "${job_pid}" 2>/dev/null; do
    sleep "${WATCHDOG_CHECK_SECS}"
    ! kill -0 "${job_pid}" 2>/dev/null && break
    local cur
    cur=$(count_episode_summaries "${output_dir}")
    if [[ "${cur}" -gt "${last_count}" ]]; then
      local new=$(( cur - last_count ))
      log "[b0_wa/${mode}] +${new} episode(s) total=${cur}，计时重置"
      last_count="${cur}"; stale_secs=0; next_log_secs=300
    else
      stale_secs=$(( stale_secs + WATCHDOG_CHECK_SECS ))
      [[ "${stale_secs}" -ge "${next_log_secs}" ]] && {
        log "[b0_wa/${mode}] $(( stale_secs / 60 ))min 无新 episode（上限 ${WATCHDOG_TIMEOUT_MINS}min）"
        next_log_secs=$(( next_log_secs + 300 ))
      }
      [[ "${stale_secs}" -ge "${watchdog_secs}" ]] && {
        log "[b0_wa/${mode}] WATCHDOG: kill PID ${job_pid}"
        ntfy_send "P79 [B0_WA/${mode}] WATCHDOG" "${WATCHDOG_TIMEOUT_MINS}min 无进展，kill 准备 resume" "high"
        kill "${job_pid}" 2>/dev/null || true; sleep 10
        kill -9 "${job_pid}" 2>/dev/null || true
        wait "${job_pid}" 2>/dev/null || true
        return 1
      }
    fi
  done
  wait "${job_pid}" 2>/dev/null || true
  log "=== [b0_wa/${mode}] 进程退出 ==="
}

run_until_complete() {
  local mode="$1" tmp_config="$2" output_dir="$3" cid="$4"
  local attempt=0

  if is_condition_complete "${output_dir}" "${cid}"; then
    log "[b0_wa/${mode}] 已完成，跳过"
    return 0
  fi

  while ! is_condition_complete "${output_dir}" "${cid}"; do
    attempt=$(( attempt + 1 ))
    [[ ${attempt} -gt ${MAX_RESUME_ATTEMPTS} ]] && {
      log "[b0_wa/${mode}] ERROR: ${MAX_RESUME_ATTEMPTS} 次 resume 后仍未完成"
      ntfy_send "P79 [B0_WA/${mode}] 失败" "已重试 ${MAX_RESUME_ATTEMPTS} 次" "urgent"
      return 1
    }
    [[ ${attempt} -gt 1 ]] && {
      log "[b0_wa/${mode}] resume ${attempt}/${MAX_RESUME_ATTEMPTS}..."
      ntfy_send "P79 [B0_WA/${mode}] 重试" "第 ${attempt}/${MAX_RESUME_ATTEMPTS} 次 resume" "default"
    }
    run_condition "${mode}" "${tmp_config}" "${output_dir}" || true
    log "[b0_wa/${mode}] 等待 15s..."
    sleep 15
  done
  log "[b0_wa/${mode}] 完成（${attempt} 次）"
}

run_reason_diagnostics() {
  local output_dir="$1"
  [[ "${REASON_DIAG_ENABLE}" != "1" ]] && return 0
  local diag="${REPO_DIR}/scripts/analysis/analyze_reason_diagnostics.py"
  [[ -f "${diag}" ]] || { log "[b0_wa] reason diagnostics 脚本不存在，跳过"; return 0; }
  log "[b0_wa] 运行 reason diagnostics..."
  local run_id
  run_id="$(basename "${output_dir}")"
  "${PYTHON_BIN}" "${diag}" \
    --run-dir "${output_dir}" --report --report-language zh --samples-per-bucket 5 \
    >> "${LOG_DIR}/b0_wa_3mode_reason_diag_${run_id}.log" 2>&1 \
    && log "[b0_wa] reason diagnostics 完成" \
    || log "[b0_wa][warn] reason diagnostics 失败（非阻塞）"
}

# ---------- 单站三模式 ----------
run_site_3mode() {
  local site="$1" base_config="$2" run_id="$3"
  local output_dir="${RESULTS_BASE}/${run_id}"

  log "========================================================"
  log "=== [B0_WA/${site}] dom → som → vision ==="
  log "=== run_id=${run_id} ==="
  log "========================================================"

  local dom_config="/tmp/b0_wa_3mode_${site}_dom_$$.yaml"
  local som_config="/tmp/b0_wa_3mode_${site}_som_$$.yaml"
  local vision_config="/tmp/b0_wa_3mode_${site}_vision_$$.yaml"

  make_single_mode_config "${base_config}" "dom"    "${dom_config}"
  make_single_mode_config "${base_config}" "som"    "${som_config}"
  make_single_mode_config "${base_config}" "vision" "${vision_config}"

  mkdir -p "${output_dir}" "${LOG_DIR}"
  ntfy_send "P79 [B0_WA/${site}] 启动" "run_id=${run_id}" "default"

  start_watchdog "${output_dir}"

  # 1) DOM
  log "======== [1/3] DOM ========"
  run_until_complete "dom" "${dom_config}" "${output_dir}" "phase1_dom_router_0"

  # reset → SOM
  log "======== reset ${site} before SOM ========"
  reset_vwa_sites "${site}" "b0_wa_3mode" || true
  sleep 10
  refresh_site_auth "${site}" || {
    sleep 30; refresh_site_auth "${site}" || { log "[b0_wa][fatal] auth 刷新失败"; exit 1; }
  }

  # 2) SOM
  log "======== [2/3] SOM ========"
  run_until_complete "som" "${som_config}" "${output_dir}" "phase1_som_router_0"

  # reset → VISION
  log "======== reset ${site} before Vision ========"
  reset_vwa_sites "${site}" "b0_wa_3mode" || true
  sleep 10
  refresh_site_auth "${site}" || {
    sleep 30; refresh_site_auth "${site}" || { log "[b0_wa][fatal] auth 刷新失败"; exit 1; }
  }

  # 3) VISION
  log "======== [3/3] Vision ========"
  run_until_complete "vision" "${vision_config}" "${output_dir}" "phase1_vision_router_0"

  log "======== final reset ${site} ========"
  reset_vwa_sites "${site}" "b0_wa_3mode_final" || true
  sleep 5

  stop_watchdog
  run_reason_diagnostics "${output_dir}"

  ntfy_send "P79 [B0_WA/${site}] 完成!" "run_id=${run_id}" "high"

  rm -f "${dom_config}" "${som_config}" "${vision_config}"
}

# ---------- 主流程 ----------
log "========================================================"
log "=== B0 WA 三模式启动 ==="
log "=== B0_WA_SITE=${B0_WA_SITE} ==="
log "========================================================"

SITES_TO_RUN=()
if [[ "${B0_WA_SITE}" == "all" ]]; then
  SITES_TO_RUN=("shopping" "shopping_admin" "reddit")
else
  SITES_TO_RUN=("${B0_WA_SITE}")
fi

for site in "${SITES_TO_RUN[@]}"; do
  case "${site}" in
    shopping)
      run_id="${B0_WA_RUN_ID_SHOPPING:-B0_wa_3mode_shopping_$(date +%Y%m%d)}"
      run_site_3mode "shopping" "${CONFIGS_SHOPPING}" "${run_id}"
      ;;
    shopping_admin)
      run_id="${B0_WA_RUN_ID_SHOPPING_ADMIN:-B0_wa_3mode_shopping_admin_$(date +%Y%m%d)}"
      run_site_3mode "shopping_admin" "${CONFIGS_SHOPPING_ADMIN}" "${run_id}"
      ;;
    reddit)
      run_id="${B0_WA_RUN_ID_REDDIT:-B0_wa_3mode_reddit_$(date +%Y%m%d)}"
      run_site_3mode "reddit" "${CONFIGS_REDDIT}" "${run_id}"
      ;;
    *)
      log "[b0_wa][error] Unknown site: ${site}"; exit 1
      ;;
  esac
  sleep 15
done

log "========================================================"
log "=== B0 WA 三模式全部完成！==="
log "========================================================"
ntfy_send "P79 [B0_WA_3mode] 全部完成!" "sites=${SITES_TO_RUN[*]}" "high"
