---
type: task
status: superseded
priority: P0
horizon: archive
order: 0
blocker: ""
eta: "SUPERSEDED 2026-07-28 — 两篇合并为一篇, 见 task_realm_paper_b_router_negative"
superseded_by: docs/checkpoints/_status/tasks/task_realm_paper_b_router_negative.md
detail: docs/checkpoints/实验笔记.md §398.8
created: 2026-07-22
updated: 2026-07-31
---

# ~~Paper A — phantom routing space 现象篇~~ (SUPERSEDED 2026-07-28)

> ## ⛔ 本卡已作废 — 两篇合并为一篇 (user 拍板 2026-07-28, 笔记 §398.8)
>
> **唯一的 REALM 稿现在是 [[task_realm_paper_b_router_negative]]**
> (标题已改为「REALM 稿 (合并 A+B) — 表征路由的上限真实存在, 但既不稳定也不可达」)。
>
> **合并理由不是「两篇都弱」, 而是 A 弱 B 强**: 原 Paper B 是完整自洽的负结果;
> 本卡的 Paper A 则 **H1 FAIL + H3 双轴低于噪声地板**。
>
> 本卡内容降级为合并稿的 **①② 两步**(ceiling 高 + 有结构基础), 后接 ③ 结构小于噪声地板
> (§398.2, 焊接枢纽) + ④ 学不到 (0/6 Pareto)。**下方内容保留作 ①② 的素材来源**,
> 但其中的「两篇切分」「non-archival 保主会选项」等表述**均已作废**。
>
> ⚠️ 本卡下方那句「venue 决定 (user 2026-07-27): 两篇都投 workshop」比合并决策早一天,
> 是它没被及时更新的原因 —— 2026-07-31 的一次 session 里 Claude 读了本卡的 frontmatter
> 并更新了 blocker/eta 字段, 却没意识到「存在两个独立 task」这件事本身已经作废
> (教训见笔记 §405.13)。

---

<details>
<summary>以下为 SUPERSEDED 前的原文 (保留作 ①② 素材)</summary>

> **venue 决定 (user 2026-07-27)**: 两篇都投 workshop (REALM @ EMNLP 2026)。
> ⚠️ 该表述已被 2026-07-28 的合并决策 supersede。


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
⚠️ **该切分已作废** —— 2026-07-28 合并为一篇后, "菜单存在且非冗余" 成为 ①②,
"菜单存在≠会点菜" 成为 ④, 中间插入 ③ (结构小于噪声地板) 作为焊接枢纽。

</details>
