#!/usr/bin/env bash
# retry_b1_single_task.sh — 重跑 B1 单个 task 的三模式（dom→reset→som→reset→vision）
#
# 用法:
#   B1_SITE=reddit B1_TASK_ID=143 bash scripts/maintenance/retry_b1_single_task.sh
#   # 或后台:
#   B1_SITE=reddit B1_TASK_ID=143 \
#     setsid nohup bash scripts/maintenance/retry_b1_single_task.sh \
#     > logs/retry_b1_reddit_143.log 2>&1 < /dev/null &
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

# ---------- 参数 ----------
SITE="${B1_SITE:?请设置 B1_SITE (reddit/classifieds/shopping)}"
TASK_ID="${B1_TASK_ID:?请设置 B1_TASK_ID}"

# run_id 映射
declare -A DEFAULT_RUN_IDS=(
  [classifieds]="B1_3mode_classifieds_20260413"
  [reddit]="B1_3mode_reddit_20260413"
  [shopping]="B1_3mode_shopping_20260413"
)
RUN_ID="${B1_RUN_ID:-${DEFAULT_RUN_IDS[$SITE]:-B1_3mode_${SITE}_20260413}}"

# ---------- 通用 ----------
source "${REPO_DIR}/scripts/maintenance/reset_vwa_sites.sh"

BASELINE_CONFIG="${REPO_DIR}/configs/_deprecated/exp_v2_qwen3vl4b_B1_baseline.yaml"
RESULTS_BASE="${REPO_DIR}/results/visualwebarena/phase1"
RUN_DIR="${RESULTS_BASE}/${RUN_ID}"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
export OPENAI_API_KEY="${OPENAI_API_KEY:-DUMMY_P79_NON_LLM_EVAL}"
export P79_DISABLE_STALE_CLEANUP="${P79_DISABLE_STALE_CLEANUP:-1}"
export WIKIPEDIA_ZIM_VERSION="${WIKIPEDIA_ZIM_VERSION:-wikipedia_en_all_maxi_2025-08}"

# Python
if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "[error] 找不到 Python 解释器" >&2; exit 127
fi

# VWA 站点环境
if [[ -n "${VWA_ENV_FILE:-}" ]] && [[ -f "${VWA_ENV_FILE}" ]]; then
  source "${VWA_ENV_FILE}" || true
elif [[ -f "${REPO_DIR}/scripts/vwa_env_remote.sh" ]]; then
  source "${REPO_DIR}/scripts/vwa_env_remote.sh" || true
elif [[ -f "${REPO_DIR}/scripts/vwa_env.sh" ]]; then
  source "${REPO_DIR}/scripts/vwa_env.sh" || true
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---------- refresh_site_auth（精简版）----------
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
    'classifieds': os.environ.get('CLASSIFIEDS', 'http://localhost:9980'),
    'reddit':      os.environ.get('REDDIT',      'http://localhost:9999'),
    'shopping':    os.environ.get('SHOPPING',    'http://localhost:7770'),
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
# /stress A1.18 P0-2 (2026-05-16): chromium launch args env-driven; reproducers
# set VWA_CHROMIUM_LAUNCH_ARGS to override DNS for their own VWA Docker host.
_chromium_args = [a for a in os.environ.get('VWA_CHROMIUM_LAUNCH_ARGS', '').split() if a]
browser = playwright.chromium.launch(headless=True, args=_chromium_args)
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
}

# ---------- 生成单 task 的 temp config ----------
make_single_task_config() {
  local site="$1" mode="$2" task_id="$3" dest="$4"
  "${PYTHON_BIN}" - << PYEOF
import re, yaml

with open("${BASELINE_CONFIG}") as f:
    content = f.read()

content = re.sub(r'include_sites:\s*\[.*?\]', 'include_sites: ["${site}"]', content)
content = re.sub(r'observation_mode:\s*\[.*?\]', 'observation_mode: ["${mode}"]', content)

# task_ids 过滤
if 'task_ids:' not in content:
    content = content.rstrip() + '\n'
    # insert under task: section
    content = re.sub(
        r'(task:\n(?:[ \t]+\S.*\n)*)',
        r'\1  task_ids:\n    ${site}: [${task_id}]\n',
        content
    )
else:
    content = re.sub(r'task_ids:\s*\{.*?\}', 'task_ids:\n    ${site}: [${task_id}]', content)

with open("${dest}", "w") as f:
    f.write(content)
print("ok: ${dest}")
PYEOF
}

# ---------- 跑单个 mode ----------
run_single_mode() {
  local mode="$1"
  local cid="phase1_${mode}_router_0"
  local summary="${RUN_DIR}/${cid}/episodes/${SITE}_task_${TASK_ID}_summary_v2.json"

  if [[ -f "${summary}" ]]; then
    log "[${mode}] task ${TASK_ID} 已有 summary，跳过"
    return 0
  fi

  local tmp_config="/tmp/b1_retry_${SITE}_${mode}_${TASK_ID}_$$.yaml"
  make_single_task_config "${SITE}" "${mode}" "${TASK_ID}" "${tmp_config}"

  local log_path="${LOG_DIR}/b1_retry_${SITE}_${mode}_task${TASK_ID}.log"
  log "[${mode}] 启动 task ${TASK_ID}..."

  "${PYTHON_BIN}" scripts/run_experiment.py \
    --config "${tmp_config}" \
    --run_id "${RUN_ID}" \
    --log_path "${log_path}" \
    >> "${log_path}" 2>&1

  rm -f "${tmp_config}"

  if [[ -f "${summary}" ]]; then
    local result
    result=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${summary}')); print(f'success={d[\"success\"]}, steps={d[\"steps\"]}, error={d.get(\"error\",None)}')")
    log "[${mode}] task ${TASK_ID} 完成: ${result}"
  else
    log "[${mode}] task ${TASK_ID} 未生成 summary!"
  fi
}

# ---------- 主流程 ----------
log "========================================================"
log "=== B1 单 task 重试: ${SITE} task ${TASK_ID} ==="
log "=== run_id=${RUN_ID} ==="
log "========================================================"

# 0) 前置 reset — 清除残留状态
log "======== initial reset ${SITE} before DOM ========"
reset_vwa_sites "${SITE}" "retry_${TASK_ID}_initial" || true
sleep 10

# 1) DOM
refresh_site_auth "${SITE}" || { log "[warn] auth 刷新失败，继续尝试..."; }
log "======== [1/3] DOM ========"
run_single_mode "dom"

# reset → SOM
log "======== reset ${SITE} before SOM ========"
reset_vwa_sites "${SITE}" "retry_${TASK_ID}" || true
sleep 10
refresh_site_auth "${SITE}" || true

# 2) SOM
log "======== [2/3] SOM ========"
run_single_mode "som"

# reset → VISION
log "======== reset ${SITE} before Vision ========"
reset_vwa_sites "${SITE}" "retry_${TASK_ID}" || true
sleep 10
refresh_site_auth "${SITE}" || true

# 3) VISION
log "======== [3/3] Vision ========"
run_single_mode "vision"

# final reset
log "======== final reset ========"
reset_vwa_sites "${SITE}" "retry_${TASK_ID}_final" || true

log "========================================================"
log "=== B1 task ${TASK_ID} 三模式重试完成 ==="
log "========================================================"

# 汇总结果
for mode in dom som vision; do
  cid="phase1_${mode}_router_0"
  summary="${RUN_DIR}/${cid}/episodes/${SITE}_task_${TASK_ID}_summary_v2.json"
  if [[ -f "${summary}" ]]; then
    result=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${summary}')); print(f'success={d[\"success\"]}, score={d[\"score\"]}, steps={d[\"steps\"]}')")
    log "  ${mode}: ${result}"
  else
    log "  ${mode}: MISSING"
  fi
done
