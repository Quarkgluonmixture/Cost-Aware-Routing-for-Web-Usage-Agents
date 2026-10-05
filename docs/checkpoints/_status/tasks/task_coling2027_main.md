---
type: task
status: active
priority: P0
horizon: now
order: 3
blocker: "A100 不可达 (UCL VPN 未连 + condenser 证书 2026-09-28 已过期), 补实验暂时起不了"
eta: "**ARR submission 2026-10-12** (AoE) → commit COLING **2026-12-23** → notif 2027-02-10。已核 2027.coling-iccl.org 2026-10-06"
detail: docs/checkpoints/paper_planning.md
created: 2026-08-08
updated: 2026-10-06
---

# COLING 2027 — 完整版目标（2026-10-06 由 NAACL 2027 改投）

**目标会议 2026-10-06 由 NAACL 2027 改为 COLING 2027**（user 定）。本文件原名 `task_naacl2027_main.md`;
2026-08-08 定 NAACL 的原始记录在 `paper_planning.md` 决策表与笔记 §443, 不改。

**REALM 是当前版本的落点; COLING 2027 是这项工作的完整版目标** —— 两者不冲突, 非归档轨本就适合当反馈场再扩成主会稿。

## 硬事实 (已核 `https://2027.coling-iccl.org/`, 2026-10-06)

| | |
|---|---|
| **ARR submission** | **2026-10-12** (Mon), 23:59 AoE |
| commitment (meta-review 后) | **2026-12-23** |
| 录取通知 | 2027-02-10 |
| 会议 | 2027-05-09 ~ 05-14, **澳门**（线上部分 05-06~07） |

**改投不改 ARR 周期**: COLING 2027 是第一届走 ACL Rolling Review 的 COLING, 收的正是 10 月 ARR 周期 ——
和原 NAACL 计划**同一个 10-12 截稿**。ARR 投稿本身不绑会议, 改的是**拿到 review 之后 commit 到哪**。
⇒ 写作与补实验的截止不变, 变的只是 12-23 的 commit 目标。
⚠️ 页数上限 / 模板官网首页未列, 投前去 CFP 页核, ⛔ 别按 NAACL 的记忆写。

**路径可行性**: ACL 系列 main conference 明确接受 measurement study / negative findings / resource / reproduction, 不要求"发明新模型" —— 本项目的 agent routing / cost-accuracy / empirical measurement 路线**本身就匹配**。

## 真实时间窗口 ⚠️（2026-10-06 重算）

```
10-06 今天 ──6天── 10-12 ARR 截稿 ──~10周── 12-23 COLING commit ──7周── 02-10 notif
```

- **只剩 6 天**: 10-12 前能进稿的只有**已有数据 + 0-compute 分析**（见 `docs/analysis/*_2026-09-22.md` 里列的 0-compute 项）。
- 新 GPU 实验（trajectory rescue Gate 1 等）**赶不上 10-12**, 只能作为 ARR author response / 下一轮的增补。
- A100 当前不可达 (2026-10-06 实测): 跳板机 `ssh.condenser.arc.ucl.ac.uk` 连接超时（UCL VPN 未连）, 且证书 09-28 已过期。
  A100 上本项目数据已于 2026-09-23 全量拉到 `E:100-condenser-backup\`（核对结果见该目录 `_transfer/verify_report.txt`）。

## 要跨过去的是「证据强度」, 不是 polish

当前 REALM 稿 = 一篇很完整的 MSc measurement + routing study。workshop 靠清楚的问题 + 扎实实验 + 有意思的结果就能成立; COLING reviewer 会继续追下去。**七个最可能的攻击面**（学长/user 2026-08-08 列）:

- [ ] 1. routing 的**泛化**到底怎么样
- [ ] 2. 是否**跨 site / benchmark** — 手上有 WA pilot + shop Phase 1b (pre-fix) 可用
- [ ] 3. **baseline 是否足够强**
- [ ] 4. router 是否真正 outperform **简单 heuristic** — ⚠️ 这条项目内已有硬结论: `§387.16.4` 的两道控制 (always-cheapest 固定策略 + label-shuffle 零分布) 显示**路由的两半都失败且败因不同**; COLING 稿必须正面处理, 不能绕
- [ ] 5. **cost-accuracy trade-off 是否稳定**
- [ ] 6. DOM / SoM / Vision 的观察能否形成**更一般化的结论**
- [ ] 7. 近乎完美的 **AUROC** 是 task 易区分, 还是 **leakage / construction artifact**

### 第 7 条: 已拆过的雷, 风险在「只报一半」

台账里这条已经被自己诊断并裁定过了 —— **结论正是 artifact**:
- `§111.2` Stage-1 linear probe 三个 setup 全 `L1+ AUROC=1.0`, 裁定 **trivial**: 根因是 probe 在 last input token position **永远 trivially 编码 input 差异**（text 内容/长度/image tokens 本就不同）⇒ **linear probe 对该 contrastive setup 是 wrong tool**, mirage signature 必须用 patching (causal) 测
- `§127.1` 另一处 AUROC 1.0 已标 **in-sample、非 held-out**
- `§394` **RETRACTED** router 的 "AUROC 0.65-0.72 in 5/6 cells" 叙述: 第 6 格 red·B2 是 **0.483（低于随机）**, 而它偏偏是**唯一显著的那格**。替代表述 = 「全局判别 (AUROC) 与尾部可用性是两个性质, 本数据上二者解耦; base SR 2-27% 的 regime 里 AUROC 高既不必要也不充分」

⇒ 风险不是"被 reviewer 发现", 而是**稿子里只写好看的那一半**。写作时逐条带上 caveat，反而是加分项（negative finding 是 ACL 系列明确接受的类型）。
⚠️ 另注: mechanism 线 (§5) 自 2026-05-14 起 shelved。若 COLING 稿不含 mechanism, 第 7 条的 probe 部分不适用; 若含, **必须带上"linear probe 是 wrong tool"的说明**。

## 输入与顺序

1. **REALM 审稿意见 (09-07 应已到)** — 免费的一轮 top-tier 反馈。决定「按原框架投还是重构」的主要依据（仓库里未见落盘, 用前先找）。
2. **REALM 稿本体** (#192, 正文 8 页) + 四步论证结构
3. **毕设全稿 (09-01)** — 为 rubric 补的文献图谱 / benchmark EDA / 形式化公式, 部分可反哺会议稿 appendix

## 与毕设的关系

**不是同一份东西**。毕设 = problem-first / concept-first / 文献综述与 EDA 齐备的长文; COLING 稿 = 单一论证线（页数以 CFP 为准）。**共享数据与图, 不共享结构。**

## 为什么这不是瞎抬目标

不是"没有论文然后幻想冲主会", 而是已有 8 页主文 + 大量 appendix + 完整实验 pipeline。针对上面七条补实验 + 重新 framing, **COLING 主会是合理的 stretch target**。若成 (MSc 一作、从 dissertation 长出), CV 信号与"硕士有篇 workshop"不是一个量级。
