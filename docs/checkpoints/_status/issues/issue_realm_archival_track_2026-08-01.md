---
type: issue
category: venue
status: open
priority: P0
action: 08-05 投 REALM **非归档轨** (Non-archival long, 8 页) — 恢复 07-22 拍板、被 07-28 合并静默删掉的主会保护
---

# REALM 有非归档轨 —— 07-28 合并把主会提交权静默烧掉了

**发现日期**: 2026-08-01（user 问「workshop 是不是没有投了以后锁你的规则」触发核查）

## 事实（CFP 原文核实 2026-08-01）

REALM = **The 2nd Workshop for Research on Agent Language Models @ EMNLP 2026**

| 项 | 值 |
|---|---|
| 三条轨 | Long (**archival**) · Short (**archival**) · **Non-archival (long or short)** |
| 归档轨约束 | *"Archival submissions must not be under review at any other venue for the duration of the REALM review period."* |
| 归档轨去向 | *"Accepted papers will appear in the ACL Anthology workshop proceedings."* |
| 非归档轨 | *"Non-archival papers may present previously published work, work under review elsewhere, or preliminary results."* · *"will not appear in the proceedings."* |
| 页数 | 两轨同为 **8 页正文** + refs/appendix 不限 |
| 截止 | Direct **2026-08-05**（两轨相同）· **ARR commitment 已从 08-10 推到 08-31** |
| Topic 命中 | 8 个里 3 个正中：**Agent Quality Evaluation** / **Agent Architectures** / **Data and simulation environments** |

## 为什么这是个 issue 而不只是一条信息

**项目自己定过非归档方案，然后在一次以「篇幅」为动机的合并里把它删掉了，且没有重新讨论。**

| 时间 | 记录 | 出处 |
|---|---|---|
| 2026-05-14 | 学长建议：「先投非 archival 占时间戳 + 展示，**不烧 archival 提交权**，为后续 main conference 铺路」 | `paper_planning §19` decision log + 笔记 §137.2 |
| 2026-07-22 | user 拍板：「Paper A = phantom 现象篇 **(non-archival, 保主会选项)** / Paper B = 阴性结果 (archival)」 | `next_steps:590` |
| **2026-07-28** | **两篇合并成单篇 8 页 archival** | `next_steps:192` · `REBUILD_PLAN.md:36` |

合并的动机是**页数与工作量**（两篇 8 页写不完）。归档属性**没有进那次决策的视野**。

## 可迁移的教训

> 合并两个对象时，**可加的属性**（页数、工作量、章节数）会被讨论；
> 而**取最严值的属性**（归档权、双盲、匿名期、license）会被**静默继承**。
> 后者往往才是不可逆的那个。

memory: [[feedback-merge-inherits-strictest-property]]

## 处置

**投非归档轨。** 它同时满足三件事，没有取舍：

1. 时间戳 + 上台讲 + 真审稿反馈 —— 归档轨能给的，非归档轨都给；
2. **零主会提交权消耗** —— 明确允许 *"work under review elsewhere"*；
3. **降低本轮定稿压力** —— 非归档稿不需要是终稿，可以带着已知缺口（SoM 无重跑地板 / n=2 工作负载）去要反馈。

连带后果：**页数之战降级**。8 页仍是上限，但一份非归档稿写得紧不紧，代价完全不同 ⇒
可以放心执行 2026-08-01 的框架转向（`笔记 §407`），而不是把旧框架硬压进 8 页。

## 待 user 确认（08-03 对账）

- [ ] 认非归档轨（我方建议：认）
- [ ] 之后冲主会的目标 venue（cascade 原记为「SL / 可能 MLSys」，名称一直待确认）
- [ ] ARR commitment 08-31 这条路要不要同时保留
