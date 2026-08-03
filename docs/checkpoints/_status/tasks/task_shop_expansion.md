---
type: task
status: pending
priority: P3
horizon: backlog
order: 9
blocker: "① reset 实测 ✅ 已完成 2026-08-03 (A100 真容器: schema 全对 + 非平凡清理端到端通过 + B-1942 守卫有效; PROTOCOL_NOTE_07 每-task 清 cart 已启用) ② **磁盘/wallclock 实测后大幅上修** — 旧估 12cond≈18.8G 基于错误的每-ep 体积。A100 实测 **4.38 MB/ep** (som 7.46, DGX 上量到的 132KB 是缺 artifacts 的假象): VWA shop 18cond=**33.5GB/38.8天**, WA shop 18cond=**13.3GB/15.4天**, 两个都跑=**46.8GB/54.3天** (hard rule #3 强制串行) vs **A100 仅剩 41GB** ⇒ 全跑装不下, 必须边跑边 rsync 或缩 scope"
eta: "user 2026-07-31 决定重开 (原 P3/backlog/2026-09+ 已 supersede)。新论据 = router 标签供给 + (2026-08-03 另一 session) frame 的站点轴 +1。⚠️ **scope 未定**: 全 3-baseline × 2 站 = 54 天不可行; 若按「backbone 非独立观测」的措辞纪律, **B0-only × 2 站 = 6+6 cond ≈ 13.6 天 / 15.6 GB** 即可交付站点轴 +1"
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
