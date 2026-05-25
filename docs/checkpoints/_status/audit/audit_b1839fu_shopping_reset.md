---
type: audit
ref: B-1839-fu
title: Phase 1b shopping reset 须含 per-condition docker restart
status: deferred
priority: P1
effort: 1h (随 shopping reset impl)
phase: phase1b
blocker: ''
---

# B-1839-fu · Phase 1b shopping reset 须含 per-condition docker restart

per-condition restart 覆盖现状: **reddit ✓** (reset=`docker rm+run` 天然 fresh) /
**cls ✓** (B-1839 加 `docker restart classifieds_db classifieds`) / **shopping ✗ 最后缺口**
(`_reset_vwa_local_shopping` 现 placeholder `return 78`; Phase 1b 实现时 = HTTP/SQL reset +
`docker restart vwa-shopping` + db-ready/http-200 warmup, 否则 shopping 同 cls 退化 +
cross-condition latency confound)。

随 shopping reset impl 一起做。Phase 1b launch 前必须完成。
