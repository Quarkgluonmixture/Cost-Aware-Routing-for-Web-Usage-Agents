---
type: plan
status: active
created: 2026-07-28
purpose: Phase 1 — turn the 2033-record ledger into a conclusion layer, read in full by Claude
supersedes: REBUILD_PLAN.md Phase 1 ("claim inventory of both papers"), which was scoped too narrowly
---

# Phase 1 — 结论提取

## 为什么范围不是 paperA/B

REBUILD_PLAN 原写 "Every claim currently in `paperA/` + `paperB/`"。**范围太窄**（user 2026-07-28）：
上个 session 五次错误里有三次不涉及任何 paper claim（replicate 是否存在、
mechanistic sweep 状态、凭空造出的算力冲突）。只核论文 = 只清理会被 reviewer 看到的那部分。

## 为什么不是"再跑一遍扫描"

我做过三轮自动依赖扫描，全部基于**数字指纹**，结论是这条路不够：

| | 条数 | 自动方法能碰到吗 |
|---|---|---|
| 带可核数字 | 1090 (53.6%) | ✅ 已验 99.6% 可追溯 |
| **纯文字** | **943 (46.4%)** | ❌ **全部碰不到** |

而那 943 条里主要是 **ADJUDICATED（裁定）+ RETRACTED（作废）** —— 恰好是"这事定过了"
和"这话废了"，防重做的核心。数字指纹在台账最核心的用途上失效。

## 产出必须是聚合，不是转写

这是 Phase 1 值不值得做的分水岭。

台账已经是笔记的转写（且验过 99.6% 可追溯）。**再做一次逐条转写只会引入二阶损失**，
而且新那层没有可机械核对的锚点 —— 结论是文字，不像数字能比对。

所以产出是**按主题聚合**：

```
台账形态（碎片）                          结论层形态（聚合）
§135 estimand 锁 FE                       ┌ 主题: pooling estimand
§143.6 gemini 反攻 FE, OSF lock 阻塞  ──> │ 当前状态: FE inverse-variance (Decision 3A)
§172.4 SE floor = 1.0pp 经验校准          │ 演化: DL → 争议 → FE; DL 降为 Appendix
§209 power 48.3% 是理论上界, 实测 81%     │ 已否决: N-aware SE / exclude-degenerate (附理由)
§253 ... 共 20 余条散落                   │ 代码状态: 实现已对, 注释 stale (两处)
                                          └ 未决: 无
```

查"estimand 怎么定的"得到一个完整答案，而不是 20 条碎片 —— **这才是防重做要的形态**，
也是逐条索引给不了的。

## 分批（context 现实）

我的 context ~337k，可用约一半。台账全文 ~300K token，**一次读不完**。且读过的内容
不会因写文件而释放。因此:

- 每批 **30–40K token**，读完**立即落盘**，落盘内容不依赖 context 保留
- 中途被压缩也能续 —— 下一批只需 `ledger.jsonl` + 已落盘的结论文件
- 无数字的先读（最大盲区），带数字的后读（已有 99.6% 数字核验托底）

| 批次 | 内容 | 条数 | 落盘 |
|---|---|---|---|
| A | ADJUDICATED 无数字 | ~501 | `conclusions/adjudicated.md` |
| B | RETRACTED 全部 + CLAIM_UNVERIFIED 全部 | 248 | `conclusions/retracted.md` |
| C | MEASURED 无数字 | ~350 | `conclusions/measured_qualitative.md` |
| D | MEASURED 带数字 | ~555 | `conclusions/measured_numeric.md` |
| E | DATA + 残余 | ~380 | `conclusions/data_inventory.md` |

批内按 § 升序读，保留时间顺序 —— 演化路径（A→B→C）只有按时序读才看得出来。

## 每批的产出格式

```markdown
## 主题名

**当前状态**: 一句话 — 现在算数的是什么
**演化**: §N 定 X → §M 改 Y（因为 Z） → §P 现状
**已否决**: 提过但被拒的方案 + 拒的理由（防止重新提起）
**未决**: 悬而未决的 + 卡在什么上
**证据**: §号 + artifact 路径
**⚠️ 矛盾**: 台账内部不一致的地方，两侧并列，不擅自调和
```

主题不预设，读的过程中涌现。初始框架（从今天已读内容归纳，非凭空）：
估计量与统计门槛 / 观测模式与表征定义 / 路由器 / 数据等级与 run 管理 /
评测器与 FP / 基础设施与流程 / 论文 framing 与 scope / 机制分析（已搁置）

## 质量保证 —— 这一层怎么防错

台账那层用"数字是否出现在源文档"验（99.6%）。结论层没有数字锚点，改用三条：

1. **每个主题必须列 § 号**，声称的演化路径可回台账逐条查
2. **矛盾不调和** —— 台账里两条打架的，结论层并列写明，不选边。选边就是在没有新证据的
   情况下制造确定性，那正是本次重建要修的病
3. **不引入台账里没有的判断** —— 结论层只做聚合、排序、去重。任何新推论必须显式标
   `[我的推论，非台账内容]`，且不能作为结论陈述

## 验收

- [ ] 五批全部落盘，`docs/reference/known/conclusions/`
- [ ] 每个主题可回溯到 § 号
- [ ] 矛盾清单单独成节（供 user 判 —— 有一类只有 user 能核：裁定是不是真这么定的）
- [ ] 一份 `INDEX.md` 汇总所有主题 + 当前状态一句话

## 不在 Phase 1 范围

- **回 artifact 复算数字** —— 那是 Phase 1 之后按结论层筛出的重点做，不是全量
- **bug sub-paper** —— 独立线，以 `master_bug_catalog.md`（318 条目 / 9874 行 / 61 处
  上游 benchmark bug）为主源，台账仅补漏笔记里散落未进 catalog 的观察。不必等 Phase 1
- **改 paper prose** —— 仍冻结至 Phase 4
