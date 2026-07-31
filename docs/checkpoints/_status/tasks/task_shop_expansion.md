---
type: task
status: pending
priority: P3
horizon: backlog
order: 9
blocker: "① reset 实测 (实现已 land commit d78fd3b, 但需等 A100 空 ~08-04 才能测 —— 重建 68GB 镜像容器 + indexer reindex 会污染在跑 chain 的 latency); ② 磁盘: A100 42G 是硬上限 (443G 已用里 419G 是 ACTIVE docker images 删不掉), 12 cond × 435 ep ≈ 18.8G ⇒ 需边跑边 rsync 回 DGX"
eta: "user 2026-07-31 决定重开 (原 P3/backlog/2026-09+ 已 supersede)。新论据 = router 标签供给, 不是站点泛化"
detail: CLAUDE.md Phase 1b (shop × {B0,B1,B2} × 6 modes)
created: 2026-07-16
updated: 2026-07-31
---

# Shopping 扩展 (Phase 1b) — 期刊版长线, 刻意排后

**定位裁决 (user 2026-07-16)**: 三选项 (B3 / WA / shop) 中单位算力信息量最低 —
shop 与 cls 同为视觉密集电商类, 第三站的站点泛化边际 < 新模型族或新 benchmark;
且 6 条 B0 条件重新挂 proxy (outage 风险)。**B3 → WA pilot → shop** 排序。

**买到**: 2 站 → 3 站 (B-1295 "N=2 sites" caveat 软化) + §5.6 site-modulated
utility 第三点 + R3→R1/Option D 主文扩展 framing。

**Scope**: shop (435 scored) × {B0, B1, B2} × 6 modes = 18 conditions。
