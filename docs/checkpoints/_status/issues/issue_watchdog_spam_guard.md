---
type: issue
category: backlog
status: backlog
priority: medium
action: experiment_watchdog.py:1340 增加 episode count vs expected guard
---

# Watchdog AUTO-ANALYSIS spam guard

partial condition_summary 触发 infinite loop (§104 Day 3 04:00 audit). `condition_completed = condition_summary_v2.json.exists()` 应增加 episode count 检查避免 partial 数据 (e.g. 165/234 ep) 触发 Case 3 re-trigger loop。当前 workaround: 不在 in-flight run 上跑 `make rederive`。
