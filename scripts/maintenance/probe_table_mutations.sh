#!/usr/bin/env bash
# Magento 状态面采集 (B-1954 follow-up): 记录 Magento 全表 UPDATE_TIME 随 fire 演进的轨迹 (单次采样, cron 驱动)。
#
# 目的: 穷举式确定「实验会改动哪些表」, 为「保留容器 + 只回滚受影响表」的
#       替代 reset 方案提供实证依据 (docs/reference/shopping_reset_state_surface.md)。
#
# 只读 (SELECT information_schema), 对 fire 无副作用。
# MariaDB 10.6 的 InnoDB UPDATE_TIME 实时且容器创建后不重启 →
# 时间戳自带分段: reindex 期 vs runner 期。容器缺席时静默跳过 (reset 窗口)。
set -uo pipefail

OUT="${PROBE_OUT:-$HOME/workspace/p79/logs/magento_table_probe.tsv}"
CONTAINER="${PROBE_CONTAINER:-vwa-shopping}"

mkdir -p "$(dirname "$OUT")"
[ -s "$OUT" ] || printf 'probe_ts\tcontainer_started\ttable_name\tupdate_time\ttable_rows\tdata_mb\n' > "$OUT"

# 容器不在 (reset 重建窗口) → 记一行 sentinel 后退出, 保留时间轴连续性
STARTED="$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER" 2>/dev/null)"
if [ -z "$STARTED" ]; then
  printf '%s\t-\t(container-absent)\t-\t-\t-\n' "$(date -Is)" >> "$OUT"
  exit 0
fi

TS="$(date -Is)"
# P2-2-A (/stress 2026-08-04): 容器**存在但 mysqld 未 query-ready** 是 reset 期间的
# 真实中间态 (正是 B-1952 处理的那段, 实测可达数十秒)。原实现在这种情况下
# `2>/dev/null` 吞掉错误、一行不写 → 时间轴留下**无标记的洞**, 事后分析看到的是
# 「这十分钟没有任何表变动」, 与「没采到」不可区分 —— 正是
# memory `feedback_absence_of_evidence_vs_measured_zero` 说的那种错误。
# 现在与 `(container-absent)` 并列写 `(db-unavailable)` sentinel。
_tmp="$(mktemp)"
if docker exec -e MYSQL_PWD=MyPassword "$CONTAINER" \
     mysql -u magentouser magentodb -sN -e "
       SELECT '${TS}', '${STARTED}', table_name, UPDATE_TIME, table_rows,
              ROUND(data_length/1048576,2)
       FROM information_schema.tables
       WHERE table_schema='magentodb' AND UPDATE_TIME IS NOT NULL
       ORDER BY UPDATE_TIME;" > "$_tmp" 2>/dev/null && [ -s "$_tmp" ]; then
  cat "$_tmp" >> "$OUT"
else
  printf '%s\t%s\t(db-unavailable)\t-\t-\t-\n' "$TS" "$STARTED" >> "$OUT"
fi
rm -f "$_tmp"
