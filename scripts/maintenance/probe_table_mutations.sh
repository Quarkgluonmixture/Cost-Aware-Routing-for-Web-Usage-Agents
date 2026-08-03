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
docker exec -e MYSQL_PWD=MyPassword "$CONTAINER" \
  mysql -u magentouser magentodb -sN -e "
    SELECT '${TS}', '${STARTED}', table_name, UPDATE_TIME, table_rows,
           ROUND(data_length/1048576,2)
    FROM information_schema.tables
    WHERE table_schema='magentodb' AND UPDATE_TIME IS NOT NULL
    ORDER BY UPDATE_TIME;" >> "$OUT" 2>/dev/null
