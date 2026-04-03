#!/usr/bin/env bash
# ============================================================
# queue_b1_serial.sh — 基于完成度判定 + 进度 watchdog 自动 resume
#
# 顺序: classifieds → reddit → shopping
#
# 完成判定 (任一满足即视为完成):
#   1. condition_summary_v2.json 存在
#   2. episodes/*_summary_v2.json 数量 >= task_configs/ 文件数 (>0)
#
# Watchdog:
#   每 WATCHDOG_CHECK_SECS 秒检查一次新完成的 episode 数量。
#   若 WATCHDOG_TIMEOUT_MINS 分钟内无进展（网络挂死/GPU 卡住），
#   主动 kill 进程，交由 run_until_complete 循环 resume。
#
# 用法:
#   nohup bash scripts/dgx/queue_b1_serial.sh > logs/queue_b1_serial.log 2>&1 &
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

# --- run_id 配置 ---
RUN_ID_CLASSIFIEDS="${RUN_ID_CLASSIFIEDS:-B1_baseline_run_classifieds_20260328_210239}"
RUN_ID_REDDIT="${RUN_ID_REDDIT:-B1_baseline_run_reddit_20260328_210239}"
RUN_ID_SHOPPING="${RUN_ID_SHOPPING:-B1_baseline_run_shopping_20260328_210239}"

RESULTS_BASE="${REPO_DIR}/results/visualwebarena/phase1"

BASELINE_CONFIG="${REPO_DIR}/configs/exp_v2_qwen3vl4b_B1_baseline.yaml"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python"

# 每个 site 最多自动 resume 次数（防止无限循环）
MAX_RESUME_ATTEMPTS="${MAX_RESUME_ATTEMPTS:-10}"

# Watchdog 配置
# 超过此分钟数无新 episode 完成 → kill 进程
WATCHDOG_TIMEOUT_MINS="${WATCHDOG_TIMEOUT_MINS:-35}"
# Watchdog 检查间隔（秒）
WATCHDOG_CHECK_SECS="${WATCHDOG_CHECK_SECS:-60}"

# DGX Spark 环境变量
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
export OPENAI_API_KEY="${OPENAI_API_KEY:-DUMMY_P79_NON_LLM_EVAL}"
export P79_DISABLE_STALE_CLEANUP="${P79_DISABLE_STALE_CLEANUP:-1}"

# ntfy 推送配置
NTFY_TOPIC="${NTFY_TOPIC:-p79-exp-dgx-spark}"
NTFY_URL="https://ntfy.sh/${NTFY_TOPIC}"
NTFY_EPISODE_INTERVAL="${NTFY_EPISODE_INTERVAL:-20}"
# 是否在每个站点完成后自动生成失败/成功归因报告
REASON_DIAG_ENABLE="${REASON_DIAG_ENABLE:-1}"

# 加载 VWA 站点环境
if [[ -f "${REPO_DIR}/scripts/vwa_env_remote.sh" ]]; then
  source "${REPO_DIR}/scripts/vwa_env_remote.sh" || true
elif [[ -f "${REPO_DIR}/scripts/vwa_env.sh" ]]; then
  source "${REPO_DIR}/scripts/vwa_env.sh" || true
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

ntfy_send() {
  local title="$1"
  local message="$2"
  local priority="${3:-default}"
  curl -s \
    -H "Title: ${title}" \
    -H "Priority: ${priority}" \
    -d "${message}" \
    "${NTFY_URL}" > /dev/null 2>&1 || true
}

run_reason_diagnostics() {
  local run_id="$1"
  local label="$2"

  if [[ "${REASON_DIAG_ENABLE}" != "1" ]]; then
    log "[${label}] reason diagnostics disabled (REASON_DIAG_ENABLE=${REASON_DIAG_ENABLE})."
    return 0
  fi

  local run_dir="${RESULTS_BASE}/${run_id}"
  local diag_script="${REPO_DIR}/scripts/analyze_reason_diagnostics.py"
  if [[ ! -d "${run_dir}" ]]; then
    log "[${label}] reason diagnostics skipped: run dir not found (${run_dir})"
    return 0
  fi
  if [[ ! -f "${diag_script}" ]]; then
    log "[${label}] reason diagnostics skipped: script not found (${diag_script})"
    return 0
  fi

  log "[${label}] Running reason diagnostics for ${run_id}..."
  if "${PYTHON_BIN}" "${diag_script}" \
    --run-dir "${run_dir}" \
    --report \
    --report-language zh \
    --samples-per-bucket 5 \
    >> "${REPO_DIR}/logs/queue_b1_serial_reason_diag.log" 2>&1; then
    log "[${label}] reason diagnostics completed."
    ntfy_send "P79 [${label}] 归因报告已生成" "run_id=${run_id}；输出目录: ${run_dir}/analysis/reason_diagnostics" "default"
  else
    log "[${label}] WARNING: reason diagnostics failed (non-blocking)."
    ntfy_send "P79 [${label}] 归因报告失败" "run_id=${run_id}；请检查 logs/queue_b1_serial_reason_diag.log" "default"
  fi
}

progress_hint() {
  local done="$1"
  local total="$2"
  local pending=$(( total - done ))
  if [[ "${pending}" -lt 0 ]]; then
    pending=0
  fi
  echo "进度 ${done}/${total}（已完成/总数），待跑 ${pending}/${total}"
}

count_episode_summaries() {
  local run_dir="$1"
  find "${run_dir}" -type f -path "*/episodes/*_summary_v2.json" 2>/dev/null | wc -l | tr -d '[:space:]'
}

expected_episode_total() {
  local run_dir="$1"
  local total_cond expected_tasks
  total_cond=$(find "${run_dir}" -mindepth 1 -maxdepth 1 -type d -name "phase*" 2>/dev/null | wc -l | tr -d '[:space:]')
  expected_tasks=$(find "${run_dir}/task_configs" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d '[:space:]')
  if [[ "${total_cond}" -gt 0 && "${expected_tasks}" -gt 0 ]]; then
    echo $(( total_cond * expected_tasks ))
  else
    echo 0
  fi
}

format_site_progress() {
  local done="$1"
  local total="$2"
  if [[ "${total}" -gt 0 ]]; then
    echo "${done}/${total}"
  else
    echo "${done}/?"
  fi
}

# ============================================================
# is_run_complete <run_id> <label>
#   检查所有 conditions 是否全部跑完
#   返回 0=完成, 1=未完成
#
#   判定顺序：
#   1. run_summary_v2.json 存在（runner 在所有 condition 全部完成后写入）
#   2. 所有 condition 目录都有 condition_summary_v2.json
#   3. 所有 condition 的 episodes 总数 >= task_configs × condition 数
# ============================================================
is_run_complete() {
  local run_id="$1"
  local label="$2"
  local run_dir="${RESULTS_BASE}/${run_id}"

  if [[ ! -d "${run_dir}" ]]; then
    log "${label}: run dir not found — not started."
    return 1
  fi

  # 主判定: run_summary_v2.json（所有 conditions 完成后由 runner 写入）
  if [[ -f "${run_dir}/run_summary_v2.json" ]]; then
    log "${label}: run_summary_v2.json exists — complete."
    return 0
  fi

  # 次级判定: 所有 condition 目录都有 condition_summary_v2.json
  local total_cond done_cond
  total_cond=$(find "${run_dir}" -mindepth 1 -maxdepth 1 -type d -name "phase*" 2>/dev/null | wc -l | tr -d '[:space:]')
  done_cond=$(find "${run_dir}" -maxdepth 2 -type f -name "condition_summary_v2.json" 2>/dev/null | wc -l | tr -d '[:space:]')

  if [[ "${total_cond}" -gt 0 && "${done_cond}" -ge "${total_cond}" ]]; then
    log "${label}: all ${done_cond}/${total_cond} condition summaries exist — complete."
    return 0
  fi

  # 三级判定: episode 总数 >= task_configs × condition 数
  local done expected_tasks expected
  done=$(count_episode_summaries "${run_dir}")
  expected_tasks=$(find "${run_dir}/task_configs" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d '[:space:]')
  expected=$(( expected_tasks * ( total_cond > 0 ? total_cond : 1 ) ))

  if [[ "${expected_tasks}" -gt 0 && "${total_cond}" -gt 0 && "${done}" -ge "${expected}" ]]; then
    log "${label}: ${done}/${expected} episodes done across ${total_cond} conditions — treating as complete."
    return 0
  fi

  log "${label}: ${done:-0}/${expected:-?} episodes, ${done_cond:-0}/${total_cond:-?} conditions done — incomplete."
  return 1
}

# ============================================================
# run_site_foreground <site> <run_id> <label>
#   后台启动实验进程，watchdog 监控进度。
#   若 WATCHDOG_TIMEOUT_MINS 内无新 episode，主动 kill 进程。
# ============================================================
run_site_foreground() {
  local site="$1"
  local run_id="$2"
  local label="$3"

  log "=== [${label}] Launching site=${site} run_id=${run_id} ==="
  log "=== [${label}] Watchdog: kill if no new episode in ${WATCHDOG_TIMEOUT_MINS}min ==="

  local tmp_config="/tmp/exp_v2_qwen3vl4b_${label}.yaml"
  cp "${BASELINE_CONFIG}" "${tmp_config}"
  sed -i -E "s/include_sites:[[:space:]]*\[[^]]*\]/include_sites: [\"${site}\"]/" "${tmp_config}"
  sed -i -E "s/name:[[:space:]]*\"qwen3vl4b_B1_baseline_phase1\"/name: \"qwen3vl4b_B1_baseline_phase1_${label}\"/" "${tmp_config}"

  local log_path="${REPO_DIR}/logs/B1_baseline_qwen3vl4b_${label}_${run_id}.log"
  ln -sfn "$(basename "${log_path}")" "${REPO_DIR}/logs/latest_${site}.log"
  log "[${label}] Site log: ${log_path}"

  # 后台启动，直接写 site log（避免 pipeline 使 PID 不明确）
  "${PYTHON_BIN}" scripts/run_experiment.py \
    --config "${tmp_config}" \
    --max_steps 30 \
    --run_id "${run_id}" \
    --log_path "${log_path}" \
    >> "${log_path}" 2>&1 &
  local job_pid=$!
  log "[${label}] PID=${job_pid} started."

  # --- Watchdog 循环 ---
  local run_dir="${RESULTS_BASE}/${run_id}"
  local last_count
  # 初始值取当前已完成数（跨所有 conditions），避免 resume 启动期间误触发
  last_count=$(count_episode_summaries "${run_dir}")
  local last_notify_count="${last_count}"
  local stale_secs=0
  local watchdog_secs=$(( WATCHDOG_TIMEOUT_MINS * 60 ))
  local next_log_secs=300  # 每 5 分钟打一次 stale 日志

  while kill -0 "${job_pid}" 2>/dev/null; do
    sleep "${WATCHDOG_CHECK_SECS}"

    # sleep 期间进程可能已正常退出
    if ! kill -0 "${job_pid}" 2>/dev/null; then
      break
    fi

    local current_count
    current_count=$(count_episode_summaries "${run_dir}")

    if [[ "${current_count}" -gt "${last_count}" ]]; then
      local new=$(( current_count - last_count ))
      log "[${label}] Watchdog: +${new} episode(s), total=${current_count}. Stale timer reset."
      last_count="${current_count}"
      stale_secs=0
      next_log_secs=300
      if (( current_count - last_notify_count >= NTFY_EPISODE_INTERVAL )); then
        local total_count progress_text
        total_count=$(expected_episode_total "${run_dir}")
        progress_text=$(format_site_progress "${current_count}" "${total_count}")
        ntfy_send "P79 [${label}] 进度" "每 ${NTFY_EPISODE_INTERVAL} 任务提醒：当前站点进度 ${progress_text}" "default"
        last_notify_count="${current_count}"
      fi
    else
      stale_secs=$(( stale_secs + WATCHDOG_CHECK_SECS ))

      # 每 5 分钟输出一次 stale 警告
      if [[ "${stale_secs}" -ge "${next_log_secs}" ]]; then
        local stale_mins=$(( stale_secs / 60 ))
        log "[${label}] Watchdog: no new episode for ${stale_mins}min (limit=${WATCHDOG_TIMEOUT_MINS}min)"
        next_log_secs=$(( next_log_secs + 300 ))
      fi

      if [[ "${stale_secs}" -ge "${watchdog_secs}" ]]; then
        log "[${label}] WATCHDOG TRIGGERED: no progress for ${WATCHDOG_TIMEOUT_MINS}min — killing PID ${job_pid}"
        ntfy_send "P79 [${label}] WATCHDOG" "${WATCHDOG_TIMEOUT_MINS}min 无进展，进程已 kill，准备 resume" "high"
        kill "${job_pid}" 2>/dev/null || true
        sleep 10
        # 确保进程已死
        kill -9 "${job_pid}" 2>/dev/null || true
        log "[${label}] WATCHDOG: Process killed."
        wait "${job_pid}" 2>/dev/null || true
        return 1
      fi
    fi
  done

  wait "${job_pid}" 2>/dev/null || true
  local rc=$?
  log "=== [${label}] Process exited rc=${rc} ==="
  return ${rc}
}

# ============================================================
# run_until_complete <site> <run_id> <label>
#   检查完成度，未完成则循环 resume，最多 MAX_RESUME_ATTEMPTS 次
# ============================================================
run_until_complete() {
  local site="$1"
  local run_id="$2"
  local label="$3"
  local attempt=0

  log "--- Checking completion: ${label} (site=${site}, run_id=${run_id}) ---"

  if is_run_complete "${run_id}" "${label}"; then
    log "${label}: already complete, skipping."
    return 0
  fi

  while ! is_run_complete "${run_id}" "${label}"; do
    attempt=$((attempt + 1))
    if [[ ${attempt} -gt ${MAX_RESUME_ATTEMPTS} ]]; then
      log "ERROR: ${label} still incomplete after ${MAX_RESUME_ATTEMPTS} resume attempts. Aborting queue."
      ntfy_send "P79 [${label}] 失败" "已重试 ${MAX_RESUME_ATTEMPTS} 次仍未完成，队列中止" "urgent"
      return 1
    fi
    log "${label}: attempt ${attempt}/${MAX_RESUME_ATTEMPTS}..."
    if [[ ${attempt} -gt 1 ]]; then
      ntfy_send "P79 [${label}] 重试" "第 ${attempt}/${MAX_RESUME_ATTEMPTS} 次 resume" "default"
    fi
    run_site_foreground "${site}" "${run_id}" "${label}" || true
    log "${label}: exited. Waiting 15s for GPU memory release..."
    sleep 15
    log "${label}: re-checking completion..."
  done

  log "${label}: confirmed complete after ${attempt} attempt(s)."
}

# ============================================================
# 主流程: classifieds → reddit → shopping
# ============================================================
TOTAL_SITES=3
done_sites=0

log "========================================================"
log "Queue started. Order: classifieds → reddit → shopping"
log "Completion: condition_summary_v2.json OR episodes>=task_configs"
log "Watchdog: kill after ${WATCHDOG_TIMEOUT_MINS}min no new episode"
log "MAX_RESUME_ATTEMPTS=${MAX_RESUME_ATTEMPTS}"
log "ntfy topic: ${NTFY_TOPIC} (interval: ${NTFY_EPISODE_INTERVAL} episodes)"
log "========================================================"
ntfy_send "P79 队列启动" "$(progress_hint "${done_sites}" "${TOTAL_SITES}")；顺序: classifieds → reddit → shopping" "default"

# 1) classifieds
ntfy_send "P79 [classifieds] 开始" "$(progress_hint "${done_sites}" "${TOTAL_SITES}")；run_id=${RUN_ID_CLASSIFIEDS}" "default"
run_until_complete "classifieds" "${RUN_ID_CLASSIFIEDS}" "classifieds"
run_reason_diagnostics "${RUN_ID_CLASSIFIEDS}" "classifieds"
done_sites=$(( done_sites + 1 ))
ntfy_send "P79 [classifieds] 完成" "$(progress_hint "${done_sites}" "${TOTAL_SITES}")；run_id=${RUN_ID_CLASSIFIEDS}" "default"
log "classifieds complete. Waiting 15s..."
sleep 15

# 2) reddit
ntfy_send "P79 [reddit] 开始" "$(progress_hint "${done_sites}" "${TOTAL_SITES}")；run_id=${RUN_ID_REDDIT}" "default"
run_until_complete "reddit" "${RUN_ID_REDDIT}" "reddit"
run_reason_diagnostics "${RUN_ID_REDDIT}" "reddit"
done_sites=$(( done_sites + 1 ))
ntfy_send "P79 [reddit] 完成" "$(progress_hint "${done_sites}" "${TOTAL_SITES}")；run_id=${RUN_ID_REDDIT}" "default"
log "reddit complete. Waiting 15s..."
sleep 15

# 3) shopping
ntfy_send "P79 [shopping] 开始" "$(progress_hint "${done_sites}" "${TOTAL_SITES}")；run_id=${RUN_ID_SHOPPING}" "default"
run_until_complete "shopping" "${RUN_ID_SHOPPING}" "shopping"
run_reason_diagnostics "${RUN_ID_SHOPPING}" "shopping"
done_sites=$(( done_sites + 1 ))
ntfy_send "P79 [shopping] 完成" "$(progress_hint "${done_sites}" "${TOTAL_SITES}")；run_id=${RUN_ID_SHOPPING}" "default"

log "========================================================"
log "=== All B1 baseline sites completed! ==="
log "========================================================"
ntfy_send "P79 全部完成!" "$(progress_hint "${done_sites}" "${TOTAL_SITES}")；所有 B1 baseline 站点已完成" "high"
