---
type: task
status: planned
priority: P0
horizon: next
order: 3
blocker: "毕设 09-01 交付前不占写作时间; 但补实验可在写毕设期间并行跑 (GPU 不与写作抢资源)"
eta: "**ARR submission 2026-10-12** (已核 2027.naacl.org 2026-08-08)。会议 2027-06-01~05 旧金山。commitment deadline 官网标 'stay tuned for details' 尚未公布"
detail: docs/checkpoints/paper_planning.md
created: 2026-08-08
updated: 2026-08-08
---

# NAACL 2027 main conference — 完整版目标

学长 2026-08-08 定的下一站。**REALM 是当前版本的落点; NAACL 2027 是这项工作的完整版目标** —— 两者不冲突, 非归档轨本就适合当反馈场再扩成主会稿。

## 硬事实 (已核 `https://2027.naacl.org/`, 2026-08-08)

| | |
|---|---|
| **ARR submission** | **2026-10-12** |
| 会议 | 2027-06-01 ~ 06-05, San Francisco |
| commitment deadline | 官网 "stay tuned for details" — **未公布**, 临近再查 |

**定位**: ACL / EMNLP / NAACL 属同一第一梯队 (声望排序 ACL ≳ EMNLP ≳ NAACL, 非官方且随方向差异大)。NAACL 2024 主会 565/2434 = **23.2%** 录取, 另 12.5% 进 Findings。**NAACL Main 是正经顶会论文**, 不是"次一级小会"。

**路径可行性**: ACL 系列 main conference 明确接受 measurement study / negative findings / resource / reproduction, 不要求"发明新模型" —— 本项目的 agent routing / cost-accuracy / empirical measurement 路线**本身就匹配**。

## 真实时间窗口 ⚠️

```
08-08 今天 ──24天── 09-01 毕设硬截止 ──6天── 09-07 REALM notif ──35天── 10-12 ARR
                                    └──────────── 41 天 ────────────┘
```

"距 deadline 两个月"是从今天算的; **毕设占掉前 24 天**。扣掉后:
- 毕设交付 → ARR = **41 天**
- REALM 审稿意见到手 → ARR = **35 天**（要在这 5 周内消化意见 + 补实验 + 重写）

⇒ **补实验必须与毕设写作并行**。这是可行的: 毕设是人力/写作, 补实验是 GPU, **不抢同一资源**。等 09-01 之后再启动实验，35 天窗口会非常紧。

## 要跨过去的是「证据强度」, 不是 polish

当前 REALM 稿 = 一篇很完整的 MSc measurement + routing study。workshop 靠清楚的问题 + 扎实实验 + 有意思的结果就能成立; NAACL reviewer 会继续追下去。**七个最可能的攻击面**（学长/user 2026-08-08 列）:

- [ ] 1. routing 的**泛化**到底怎么样
- [ ] 2. 是否**跨 site / benchmark** — 手上有 WA pilot + shop Phase 1b (pre-fix) 可用
- [ ] 3. **baseline 是否足够强**
- [ ] 4. router 是否真正 outperform **简单 heuristic** — ⚠️ 这条项目内已有硬结论: `§387.16.4` 的两道控制 (always-cheapest 固定策略 + label-shuffle 零分布) 显示**路由的两半都失败且败因不同**; NAACL 稿必须正面处理, 不能绕
- [ ] 5. **cost-accuracy trade-off 是否稳定**
- [ ] 6. DOM / SoM / Vision 的观察能否形成**更一般化的结论**
- [ ] 7. 近乎完美的 **AUROC** 是 task 易区分, 还是 **leakage / construction artifact**

### 第 7 条: 已拆过的雷, 风险在「只报一半」

台账里这条已经被自己诊断并裁定过了 —— **结论正是 artifact**:
- `§111.2` Stage-1 linear probe 三个 setup 全 `L1+ AUROC=1.0`, 裁定 **trivial**: 根因是 probe 在 last input token position **永远 trivially 编码 input 差异**（text 内容/长度/image tokens 本就不同）⇒ **linear probe 对该 contrastive setup 是 wrong tool**, mirage signature 必须用 patching (causal) 测
- `§127.1` 另一处 AUROC 1.0 已标 **in-sample、非 held-out**
- `§394` **RETRACTED** router 的 "AUROC 0.65-0.72 in 5/6 cells" 叙述: 第 6 格 red·B2 是 **0.483（低于随机）**, 而它偏偏是**唯一显著的那格**。替代表述 = 「全局判别 (AUROC) 与尾部可用性是两个性质, 本数据上二者解耦; base SR 2-27% 的 regime 里 AUROC 高既不必要也不充分」

⇒ 风险不是"被 reviewer 发现", 而是**稿子里只写好看的那一半**。写作时逐条带上 caveat，反而是加分项（negative finding 是 ACL 系列明确接受的类型）。
⚠️ 另注: mechanism 线 (§5) 自 2026-05-14 起 shelved。若 NAACL 稿不含 mechanism, 第 7 条的 probe 部分不适用; 若含, **必须带上"linear probe 是 wrong tool"的说明**。

## 输入与顺序

1. **REALM 审稿意见 (09-07)** — 免费的一轮 top-tier 反馈, 落在毕设交付之后。决定「按原框架投还是重构」的主要依据, **等它到手再定稿件形态**。
2. **REALM 稿本体** (#192, 正文 8 页) + 四步论证结构
3. **毕设全稿 (09-01)** — 为 rubric 补的文献图谱 / benchmark EDA / 形式化公式, 部分可反哺会议稿 appendix

## 与毕设的关系

**不是同一份东西**。毕设 = problem-first / concept-first / 文献综述与 EDA 齐备的长文; NAACL 稿 = 8-9 页单一论证线。**共享数据与图, 不共享结构。**

## 为什么这不是瞎抬目标

不是"没有论文然后幻想冲 NAACL", 而是已有 8 页主文 + 大量 appendix + 完整实验 pipeline。接下来两个月针对上面七条补实验 + 重新 framing, **NAACL 是合理的 stretch target**。若成 (MSc 一作、从 dissertation 长出), CV 信号与"硕士有篇 workshop"不是一个量级。
