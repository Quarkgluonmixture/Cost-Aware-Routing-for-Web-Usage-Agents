---
type: task
status: pending
priority: P3
horizon: backlog
order: 9
blocker: "① reset 实测 ✅ 已完成 2026-08-03 (A100 真容器: schema 全对 + 非平凡清理端到端通过 + B-1942 守卫有效; PROTOCOL_NOTE_07 每-task 清 cart 已启用) ② ~~磁盘装不下~~ **已解除 2026-08-12 — 那条量错了盘**。VWA fire 落 **scratch** 不落 vda1 (台账 §301 的 partial symlink: 只 `results/visualwebarena` → scratch, 顶层留原盘因为 git 不 follow symlink 会让 Gate 3 fail-closed)。实测 `/mnt/scratch` **avail 278G** (VWA 数据当前占 53G), 而 vda1 那 60G 是另一块盘、与 fire 无关。⇒ VWA shop 18cond=33.5GB **装得下**, 两站 46.8GB 也装得下。**真约束只剩 wallclock (38.8 天 / 两站 54.3 天, hard rule #3 强制串行) 与 B0 的 API 钱**。⚠️ 排查磁盘前必须 `df -h results/visualwebarena` 或 `readlink`, 不能 `df -h results` —— 后者在这套布局下**永远**报原盘"
eta: "user 2026-07-31 决定重开 (原 P3/backlog/2026-09+ 已 supersede)。新论据 = router 标签供给 + (2026-08-03 另一 session) frame 的站点轴 +1。⚠️ **scope 未定**: 全 3-baseline × 2 站 = 54 天不可行; 若按「backbone 非独立观测」的措辞纪律, **B0-only × 2 站 = 6+6 cond ≈ 13.6 天 / 15.6 GB** 即可交付站点轴 +1。2026-08-12 复核: VWA-shop 的 B0 已有 dom/som/vision (各 435 ep, 08-04~08-10), B1 有 dom + som/P-SoM 在 chain 中 ⇒ 缺口比卡上假设的小; 且 **补 B0 三个 phantom 已判定无价值** (§5 存活的 cost ceiling 'adds no arm', 且每加一臂欠一笔 rerun 对照)"
detail: CLAUDE.md Phase 1b (shop × {B0,B1,B2} × 6 modes)
created: 2026-07-16
updated: 2026-08-12
---

# Shopping 扩展 (Phase 1b) — 期刊版长线, 刻意排后

**定位裁决 (user 2026-07-16)**: 三选项 (B3 / WA / shop) 中单位算力信息量最低 —
shop 与 cls 同为视觉密集电商类, 第三站的站点泛化边际 < 新模型族或新 benchmark;
且 6 条 B0 条件重新挂 proxy (outage 风险)。**B3 → WA pilot → shop** 排序。

**买到**: 2 站 → 3 站 (B-1295 "N=2 sites" caveat 软化) + §5.6 site-modulated
utility 第三点 + R3→R1/Option D 主文扩展 framing。

**Scope**: shop (435 scored) × {B0, B1, B2} × 6 modes = 18 conditions。
