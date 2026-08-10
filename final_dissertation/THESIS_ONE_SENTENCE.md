# THESIS_ONE_SENTENCE

> 起手文件 1/5（Guide §26）。只放 problem / RQ / headline answer，各一句。
> 这三句一旦稳定，其余四个文件和全部 prose 都对齐它。
> **改这里 = 改整篇论文**，所以改之前先问：是证据变了，还是只是措辞不顺。

**建立日期**：2026-08-09 · **状态**：v1 草案，待 supervisor 确认

---

## Problem

Web agent 每一步都默认喂进最贵的那种上下文（标注截图 + 完整 DOM），
但**没有人验证过这笔开销是不是每一步都必要**——而它按步计费、按步耗时、按步烧算力。

## Research Question

> **昂贵的多模态上下文在什么时候才真正必要，以及这个"何时"能不能被便宜到不抵消收益地预测出来？**

（对齐 Guide §14.1。两半分别对应现象问题与方法问题。）

## Headline Answer

> **必要性确实是状态依赖的——上限真实存在；但这个"何时"在当前体制下预测不出来，
> 而失败的原因是结构性的，不是模型选得不好。**

三个分句各有确切所指，写作时不得含混：

| 分句 | 确切含义 | 主证据 |
|---|---|---|
| 上限真实存在 | oracle 相对最强单模 **+3.45 ~ +16.07pp** SR，成本低 **13.7–35.3%**（⚠️ **VWA 6 格口径**，上界 +16.07pp = `cls·B0`；含 WA 两格的 **8 格口径上界是 +16.35pp** = `wa_reddit·B0`，见 CLAIM_EVIDENCE_MATRIX C1。两个数都对，指代的 cell 集不同——**引用时必须带 cell 集**）<br>🔴 **且必须紧跟重跑基线**：一次同 arm 重跑就买到 **2.0–7.6pp**（`noise_floor_inventory.md` §3 ①）。这是全文唯一还站着的正面主张，限定不能省 | `phase1_full_prereg_decision.json` · `router_objective_ordering.md:21-24, 146-154` · `noise_floor_inventory.md` |
| 预测不出来 | 真嵌套 CV 下 **0/6 cell** 能 Pareto 胜过平凡的 always-cheapest 固定策略 | `router_triage_learnability.md` |
| 原因是结构性的 | 瓶颈是标签的**产生率**：标签只在任务被解开时诞生，而 base SR 只有 2–27% | `router_label_supply_diagnosis.md` |

---

## ⚠️ 这篇论文的答案里有一个 "no"，这是有意的

Guide §14.1 给出的 RQ 第二半（"can we predict…"），**本项目实测答案是否定的**。

这不是要换问题。Guide §1.6 与 §4.2 明确把这种情况列为优秀论文的特征——
RL climate thesis 直接写 *"the answer … is on average: no"* 并因此加分。
但它确实改变两处写法，必须提前锁死：

1. **Ch6 Evaluation 不能按"router 赢了多少"组织**，要按
   **"它在哪一步断掉、以及为什么那一步是结构性的"** 组织。
2. **Ch7 Discussion 才是科学贡献所在**：本文交付的不是一个 router，
   而是**这个体制的边界**——在什么条件下表征路由不可能学得起来。

一个 negative result 要撑得起 MSc thesis，必须同时满足两条，本项目都满足：

- **上限先被证明存在**（否则"学不到"可以被解释成"根本没东西可学"）
- **失败原因被定位到机制**（否则"学不到"可以被解释成"你方法不行"）

## 反过来说，这篇论文**不主张**什么

写作时任何一句越过下面这条线，都要降级或补证据（Guide T10 / §11 claim ladder）：

- ❌ 不主张 P-SoM 能替代 SoM（价值在**互补**，不在替代）
- ❌ 不主张"丢掉图性能几乎不掉"或"这是更便宜的 SoM"
- ❌ 不主张 routing 在别的模型 / 别的 benchmark 上也一定学不到——
      本文只标定了**在观察到的 base SR 2–27% 体制内**它为什么学不到
- ❌ 不把 token / 美元成本代理偷换成能耗或碳排（Guide T14 / §22.7）
- ❌ **不主张三个 phantom arm 的互补结构"幅度超过重复采样地板"**（2026-08-10 加）——
      方向一致（22/24 arm drop-one 为正）可以说，但 pooled 双轴效应 **1.35 / 2.09pp**
      低于最宽松的已测重跑地板 **2.0pp**，且"过带 ≠ 过噪声"（要 **3.82–4.15pp** 才不太可能
      由单次重跑产生）。原文判词：*"does not survive as a positive claim"*
      （`noise_floor_inventory.md` §3 ②）。正确措辞见 CLAIM_EVIDENCE_MATRIX C2

---

→ 下一个文件：[CLAIM_EVIDENCE_MATRIX.md](CLAIM_EVIDENCE_MATRIX.md)
