---
type: task
status: active
priority: P0
horizon: now
order: 0
blocker: "verdict 数字等 k=6 (B2_reddit ~07-26/27); 非数字段落与格式转换不等"
eta: "2026-08-05 (REALM @ EMNLP 2026 direct submission)"
detail: docs/checkpoints/paper_drafts/aaai27/aaai27_main.md
created: 2026-07-22
updated: 2026-07-22
---

# Paper A — phantom routing space 现象篇 (REALM @ EMNLP 2026, **non-archival**)

> **venue 决定 (user 2026-07-27)**: **两篇都投 workshop** (REALM @ EMNLP 2026)。
> Paper A 不再为主会保留 non-archival 形态 → 8 页压缩取舍可果断; 双盲 ACL 格式。


**主张**: 菜单存在且非冗余 —— H3 双轴 PASS + 各臂独解任务 + cost 构造性剖面。

**归档选择 = non-archival** (决策 2026-07-22, 笔记 §383.1): REALM 的 non-archival 轨明确允许
"concurrently submitted work", 故**保住之后投 EMNLP main 的选项**; archival 会进 ACL Anthology
= 把主线内容烧在 workshop 上。

**格式**: ACL 2026 style, 双盲, long **8 页正文 + refs/appendix 不限** (camera-ready +1)。
比 AAAI 的 7 页宽 → 原 `cut_prewrites` 砍词工作**作废**, 只需 AAAI→ACL 转换
(`aaai27/latex/` skeleton + convert.sh 改目标模板)。Supplement + 逐 cell 表可全进 appendix。

**待办**
- [ ] AAAI→ACL 格式转换 (不等数据)
- [ ] 非数字段落: method / related work / framing (不等数据)
- [ ] k=6 落地后 splice H1/H3 全部 verdict 数字
- [ ] k=6 后删 Protocol Note 06 两轨制披露 + "5 of 6" 免责段
- [ ] B-1284 cross-family modifier 解除后开放跨族复制主张
- [ ] /stress + Mode B/C chain

**与 Paper B 的切分**: A 讲"菜单存在且非冗余", B 讲"菜单存在≠会点菜"。贡献与 reviewer pool
均不重叠 → 无 dual-submission 问题。
