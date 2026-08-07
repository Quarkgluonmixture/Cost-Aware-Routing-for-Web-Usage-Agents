#!/usr/bin/env bash
# Fire-4 RCA Wave 2c M11 — in-flight VWA health telemetry (log/alert ONLY).
#
# User decision 2026-05-19 (post 3-AI /stress audit): "Add lightweight in-flight
# telemetry: disk, docker/container health, curl latency, maybe every 30
# episodes or fixed time interval. Initially log/alert rather than aggressive
# false-positive abort."
#
# Hypothesis under test: Fire-4 task 75 substrate failure may be amplified by
# 3-hour-old VWA cls docker state drift (gemini A1-7 OOB finding) — disk
# pressure / PHP-FPM saturation / Postgres lock contention. This script
# captures the metrics needed to test that hypothesis post-Fire-5.
#
# Outputs a single JSON object on stdout + appends one JSONL line to
# logs/health/<run_id>_health.jsonl if RUN_ID env is set. NEVER exits non-zero
# from threshold breaches — this is observability, NOT enforcement. Exit codes:
#   0 — success (data captured)
#   1 — script error (e.g., curl binary missing) NOT threshold breach
#
# Usage:
#   ./check_vwa_health.sh                  # stdout JSON, no logging
#   RUN_ID=B0_dom_cls_... ./check_vwa_health.sh  # also append JSONL
#   NTFY_TOPIC=p79-exp-dgx-spark ./check_vwa_health.sh  # alert on threshold

set -uo pipefail  # NOTE: NOT set -e; we want to capture each probe's success
                   # independently so partial failures still emit the JSON.

# Hard thresholds (breach → ntfy alert, NOT abort).
DISK_PCT_ALERT_THRESHOLD="${DISK_PCT_ALERT_THRESHOLD:-95}"
CURL_LATENCY_MS_ALERT_THRESHOLD="${CURL_LATENCY_MS_ALERT_THRESHOLD:-5000}"

# Resolve repo root for log directory placement.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TIMESTAMP="$(date -u +%FT%TZ)"
HOSTNAME="$(hostname 2>/dev/null || echo unknown)"

# Probe 1: disk usage on the partition holding the repo (typically / on A100,
# /home/jiaming on DGX). Returns integer 0-100.
disk_pct=0
if df_out=$(df -P "${REPO_ROOT}" 2>/dev/null | awk 'NR==2 {gsub("%",""); print $5}'); then
    disk_pct="${df_out:-0}"
fi

# Probe 2: curl latency to VWA cls (port 9980), median of 3 trials, 10s timeout.
#
# B-1968 (2026-08-07): timeouts used to be recorded as a LATENCY, not as a failure.
# `curl -m 10` still prints `%{time_total}` ≈ 10.001 when it times out, and the old
# `|| echo ""` never fired (curl wrote the value before exiting non-zero), so a dead
# site produced `curl_cls_ms=10001, curl_cls_status=ok` and an alert reading
# "cls latency=10001ms" — i.e. **"the site is slow"** when the truth was **"the site
# did not answer"**. Those are different incidents with different responses.
#
# Observed 2026-08-07: 3 serial probes on A100 returned 200/0.20s, then 000/12.0s,
# then 000/12.0s, while the container sat at 0.00% CPU and its own log showed 200s
# — the signature of the PHP built-in server (single-threaded) head-of-line blocking,
# not of a loaded host. Under the old code that hour looked like "latency crept up".
#
# Now: `%{http_code}` decides. 000 = no HTTP response (timeout / conn refused) and is
# counted as a failure; only real responses contribute to the median. Status is
# ok / degraded (some failed) / unreachable (all failed) so the alert says which.
curl_cls_ms="null"
curl_cls_status="probe_failed"
curl_cls_fail_n=0
if command -v curl > /dev/null; then
    declare -a cls_times_ms=()
    for _ in 1 2 3; do
        probe=$(curl -s -o /dev/null -w '%{time_total} %{http_code}' -m 10 http://localhost:9980/ 2>/dev/null || true)
        t_sec="${probe%% *}"; http_code="${probe##* }"
        if [[ -z "${probe}" ]] || [[ "${http_code}" == "000" ]]; then
            curl_cls_fail_n=$((curl_cls_fail_n + 1))
            continue
        fi
        t_ms=$(awk -v s="${t_sec}" 'BEGIN { printf "%d", s * 1000 }')
        cls_times_ms+=("${t_ms}")
    done
    if [[ "${#cls_times_ms[@]}" -gt 0 ]]; then
        sorted_times=$(printf '%s\n' "${cls_times_ms[@]}" | sort -n)
        median_idx=$(( (${#cls_times_ms[@]} - 1) / 2 ))
        curl_cls_ms=$(printf '%s\n' "${sorted_times}" | awk -v idx="${median_idx}" 'NR==idx+1 { print }')
        if [[ "${curl_cls_fail_n}" -gt 0 ]]; then
            curl_cls_status="degraded"
        else
            curl_cls_status="ok"
        fi
    else
        curl_cls_status="unreachable"
    fi
fi

# Probe 3: docker container memory for cls/red/shop (best-effort, A100-local).
# Format: {"cls": "850MiB", "red": "...", ...}; null if docker not available.
docker_mem_cls="null"
docker_mem_red="null"
docker_mem_shop="null"
if command -v docker > /dev/null; then
    # `docker stats --no-stream --format` gives one-shot snapshot.
    # MemUsage column is "USED / LIMIT", e.g. "850MiB / 16GiB".
    if mem_cls_raw=$(docker stats --no-stream --format '{{.MemUsage}}' classifieds 2>/dev/null); then
        docker_mem_cls="\"${mem_cls_raw}\""
    fi
    if mem_red_raw=$(docker stats --no-stream --format '{{.MemUsage}}' reddit 2>/dev/null); then
        docker_mem_red="\"${mem_red_raw}\""
    fi
    if mem_shop_raw=$(docker stats --no-stream --format '{{.MemUsage}}' shopping 2>/dev/null); then
        docker_mem_shop="\"${mem_shop_raw}\""
    fi
fi

# Probe 4: load average (1-min) — proxy for CPU contention.
load_1m="null"
if loadavg_raw=$(uptime 2>/dev/null | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' '); then
    load_1m="${loadavg_raw:-null}"
fi

# Emit JSON to stdout.
cat <<EOF
{
  "schema": "vwa_health_v1",
  "ts": "${TIMESTAMP}",
  "hostname": "${HOSTNAME}",
  "run_id": "${RUN_ID:-null}",
  "disk_pct": ${disk_pct},
  "disk_pct_threshold": ${DISK_PCT_ALERT_THRESHOLD},
  "curl_cls_ms": ${curl_cls_ms},
  "curl_cls_status": "${curl_cls_status}",
  "curl_latency_threshold_ms": ${CURL_LATENCY_MS_ALERT_THRESHOLD},
  "docker_mem_cls": ${docker_mem_cls},
  "docker_mem_red": ${docker_mem_red},
  "docker_mem_shop": ${docker_mem_shop},
  "load_1m": ${load_1m}
}
EOF

# Append JSONL to log (if RUN_ID set, route under that run's health log).
if [[ -n "${RUN_ID:-}" ]]; then
    log_dir="${REPO_ROOT}/logs/health"
    mkdir -p "${log_dir}" 2>/dev/null || true
    log_path="${log_dir}/${RUN_ID}_health.jsonl"
    # Compact one-liner for log (vs pretty stdout output).
    {
        printf '{"schema":"vwa_health_v1","ts":"%s","hostname":"%s","run_id":"%s","disk_pct":%s,"curl_cls_ms":%s,"curl_cls_status":"%s","curl_cls_fail_n":%s,"docker_mem_cls":%s,"docker_mem_red":%s,"docker_mem_shop":%s,"load_1m":%s}\n' \
            "${TIMESTAMP}" "${HOSTNAME}" "${RUN_ID}" \
            "${disk_pct}" "${curl_cls_ms}" "${curl_cls_status}" "${curl_cls_fail_n}" \
            "${docker_mem_cls}" "${docker_mem_red}" "${docker_mem_shop}" "${load_1m}"
    } >> "${log_path}"
fi

# ntfy alert on threshold breach (log-only side-channel; never abort fire).
alert_msgs=()
if [[ "${disk_pct}" -ge "${DISK_PCT_ALERT_THRESHOLD}" ]]; then
    alert_msgs+=("disk=${disk_pct}% >= ${DISK_PCT_ALERT_THRESHOLD}% (host=${HOSTNAME})")
fi
# B-1968: 三种情况分开报 —— 「慢」「时通时不通」「完全不通」需要的处置不同,
# 而旧版把后两者都渲染成一句 "latency=10001ms", 读起来像第一种。
if [[ "${curl_cls_status}" == "unreachable" ]]; then
    alert_msgs+=("cls UNREACHABLE (3/3 probes got no HTTP response)")
elif [[ "${curl_cls_status}" == "degraded" ]]; then
    alert_msgs+=("cls DEGRADED ${curl_cls_fail_n}/3 probes no-response (median of the rest=${curl_cls_ms}ms)")
elif [[ "${curl_cls_ms}" != "null" ]] && [[ "${curl_cls_ms}" -ge "${CURL_LATENCY_MS_ALERT_THRESHOLD}" ]]; then
    alert_msgs+=("cls slow: latency=${curl_cls_ms}ms >= ${CURL_LATENCY_MS_ALERT_THRESHOLD}ms (all 3 probes answered)")
fi
if [[ "${#alert_msgs[@]}" -gt 0 ]] && [[ -n "${NTFY_TOPIC:-}" ]] && command -v curl > /dev/null; then
    msg="P79 health alert (run=${RUN_ID:-?}): $(IFS='; '; echo "${alert_msgs[*]}")"
    curl -L -d "${msg}" "ntfy.sh/${NTFY_TOPIC}" 2>/dev/null > /dev/null || true
fi

exit 0
