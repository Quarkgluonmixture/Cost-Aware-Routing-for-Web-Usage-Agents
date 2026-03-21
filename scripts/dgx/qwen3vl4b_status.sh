#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_DIR}/.autostart"

WATCH_PID_FILE="${STATE_DIR}/qwen3vl4b_watch.pid"
RUN_PID_FILE="${STATE_DIR}/qwen3vl4b_run.pid"
LAUNCH_MARK_FILE="${STATE_DIR}/qwen3vl4b_launched.ok"
RUN_LOG_FILE="${STATE_DIR}/qwen3vl4b_run.log"
WATCH_LOG_FILE="${STATE_DIR}/qwen3vl4b_watch.log"
RUN_EXIT_FILE="${STATE_DIR}/qwen3vl4b_run.exit_code"
RUN_DONE_OK_FILE="${STATE_DIR}/qwen3vl4b_run.done.ok"
RUN_DONE_FAIL_FILE="${STATE_DIR}/qwen3vl4b_run.done.fail"

IDLE_UTIL_THRESHOLD="${IDLE_UTIL_THRESHOLD:-5}"
IDLE_MEM_THRESHOLD_MIB="${IDLE_MEM_THRESHOLD_MIB:-1024}"

trim() {
  sed 's/^[[:space:]]*//; s/[[:space:]]*$//'
}

fmt_time() {
  local file="$1"
  if [[ -f "$file" ]]; then
    date -d "@$(stat -c %Y "$file")" '+%F %T'
  else
    echo "-"
  fi
}

pid_alive() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

pid_elapsed() {
  local pid="$1"
  ps -p "${pid}" -o etime= 2>/dev/null | trim || true
}

pid_cmd() {
  local pid="$1"
  ps -p "${pid}" -o cmd= 2>/dev/null | trim || true
}

show_once() {
  local now watcher_pid run_pid
  local watcher_state watcher_detail
  local run_state run_detail overall
  local run_exit="-"
  local gpu_state="未知"
  local gpu_name="-"
  local gpu_util="-"
  local gpu_mem="-"
  local gpu_proc_count="0"
  local gpu_proc_lines=""

  now="$(date '+%F %T')"
  watcher_pid="$(cat "${WATCH_PID_FILE}" 2>/dev/null || true)"
  run_pid="$(cat "${RUN_PID_FILE}" 2>/dev/null || true)"

  if [[ -n "${watcher_pid}" ]] && pid_alive "${watcher_pid}"; then
    watcher_state="运行中"
    watcher_detail="PID=${watcher_pid}, 已运行=$(pid_elapsed "${watcher_pid}")"
  elif [[ -n "${watcher_pid}" ]]; then
    watcher_state="未运行（PID 文件已过期）"
    watcher_detail="PID=${watcher_pid}"
  else
    watcher_state="未运行"
    watcher_detail="-"
  fi

  if [[ -f "${RUN_EXIT_FILE}" ]]; then
    run_exit="$(cat "${RUN_EXIT_FILE}" 2>/dev/null || echo "?")"
  fi

  if [[ -n "${run_pid}" ]] && pid_alive "${run_pid}"; then
    run_state="运行中"
    run_detail="PID=${run_pid}, 已运行=$(pid_elapsed "${run_pid}")"
  elif [[ -f "${RUN_DONE_OK_FILE}" ]]; then
    run_state="已结束（成功）"
    run_detail="退出码=${run_exit}"
  elif [[ -f "${RUN_DONE_FAIL_FILE}" ]]; then
    run_state="已结束（失败）"
    run_detail="退出码=${run_exit}"
  elif [[ -f "${LAUNCH_MARK_FILE}" ]]; then
    run_state="已触发但未运行"
    run_detail="可能已退出，查看运行日志"
  else
    run_state="未开始"
    run_detail="-"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_line idx util_raw mem_used_raw mem_total_raw util_num mem_num
    gpu_line="$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -n1 || true)"
    if [[ -n "${gpu_line}" ]]; then
      IFS=',' read -r idx gpu_name util_raw mem_used_raw mem_total_raw <<< "${gpu_line}"
      idx="$(echo "${idx}" | trim)"
      gpu_name="$(echo "${gpu_name}" | trim)"
      gpu_util="$(echo "${util_raw}" | trim)%"
      gpu_mem="$(echo "${mem_used_raw}" | trim)/$(echo "${mem_total_raw}" | trim) MiB"
      util_num="$(echo "${util_raw}" | tr -cd '0-9')"
      mem_num="$(echo "${mem_used_raw}" | tr -cd '0-9')"

      gpu_proc_lines="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null | awk 'NF>0' || true)"
      gpu_proc_count="$(printf '%s\n' "${gpu_proc_lines}" | awk 'NF>0' | wc -l | tr -d '[:space:]')"

      if [[ "${gpu_proc_count}" -eq 0 ]] && [[ -n "${util_num}" ]] && [[ "${util_num}" -le "${IDLE_UTIL_THRESHOLD}" ]]; then
        if [[ -z "${mem_num}" ]] || [[ "${mem_num}" -le "${IDLE_MEM_THRESHOLD_MIB}" ]]; then
          gpu_state="空闲"
        else
          gpu_state="占用中（显存高）"
        fi
      else
        gpu_state="占用中"
      fi
    fi
  else
    gpu_state="不可用（nvidia-smi 不存在）"
  fi

  if [[ "${run_state}" == "运行中" ]]; then
    overall="Baseline 正在运行"
  elif [[ "${run_state}" == "已结束（成功）" ]]; then
    overall="Baseline 已完成（成功）"
  elif [[ "${run_state}" == "已结束（失败）" ]]; then
    overall="Baseline 已结束（失败）"
  elif [[ "${watcher_state}" == "运行中" ]]; then
    overall="等待 GPU 空闲后自动启动"
  elif [[ "${run_state}" == "已触发但未运行" ]]; then
    overall="已触发过，但当前未运行"
  else
    overall="未挂起自动任务"
  fi

  echo "========== Qwen3-VL-4B 自动任务状态 =========="
  echo "时间: ${now}"
  echo "总览: ${overall}"
  echo
  echo "[GPU]"
  echo "状态: ${gpu_state}"
  echo "设备: ${gpu_name}"
  echo "利用率: ${gpu_util}"
  echo "显存: ${gpu_mem}"
  echo "计算进程数: ${gpu_proc_count}"
  if [[ -n "${gpu_proc_lines}" ]]; then
    echo "进程列表:"
    echo "${gpu_proc_lines}" | sed 's/^/  - /'
  fi
  echo
  echo "[Watcher]"
  echo "状态: ${watcher_state}"
  echo "详情: ${watcher_detail}"
  echo "PID 文件: ${WATCH_PID_FILE}"
  echo "日志: ${WATCH_LOG_FILE}"
  echo "最近日志:"
  if [[ -f "${WATCH_LOG_FILE}" ]]; then
    tail -n 3 "${WATCH_LOG_FILE}" | sed 's/^/  /'
  else
    echo "  -"
  fi
  echo
  echo "[Baseline]"
  echo "状态: ${run_state}"
  echo "详情: ${run_detail}"
  echo "已触发时间: $(fmt_time "${LAUNCH_MARK_FILE}")"
  echo "退出码文件: ${RUN_EXIT_FILE}"
  echo "运行日志: ${RUN_LOG_FILE}"
  echo "最近日志:"
  if [[ -f "${RUN_LOG_FILE}" ]]; then
    tail -n 5 "${RUN_LOG_FILE}" | sed 's/^/  /'
  else
    echo "  -"
  fi
}

if [[ "${1:-}" == "--watch" ]]; then
  interval="${2:-3}"
  while true; do
    clear
    show_once
    sleep "${interval}"
  done
else
  show_once
fi
