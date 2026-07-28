---
type: progress
status: active
created: 2026-07-28
purpose: Phase 1 结论提取的进度与接力点 — 新 session 从这里接手
---

# Phase 1 进度 / 接力

> **新 session 读这一个文件就够。** 不要重读 `实验笔记.md`，不要重建台账。
> 计划在 `docs/checkpoints/PHASE1_PLAN.md`，数据在 `docs/reference/known/ledger.jsonl`。

## 一句话现状

台账 2033 条已建成并核验（99.6% 可追溯）。现在把它**按主题聚合**成结论层。
分五批，合计约 **591K token** —— 单个 session 的 context 装不下，所以分 session 接力。

## 批次状态

| 批 | 内容 | 条数 | token | 谁做 | 状态 | 产出 |
|---|---|---|---|---|---|---|
| **B** | RETRACTED + CLAIM_UNVERIFIED | 248 | 59.6K | Claude 主 session | ✅ **完成** 07-28 | `retracted.md` |
| **C** | MEASURED 无数字 | 30 | 7.6K | Claude 主 session | ✅ **完成** 07-28 | `measured_qualitative.md` |
| **E** | DATA（49 条） | 49 | 14.4K | Claude 主 session | ✅ **完成** 07-28 | `data_inventory.md` |
| **A** | ADJUDICATED（裁定） | 831 | ~250K | Claude 分轮 | ⬜ **唯一剩余批次**，需新 session（2–3 轮） | `adjudicated.md` |
| **D** | MEASURED 带数字 | 875 | ~260K | subagent ×4 | ✅ **D1/D2/D3/D4 全部完成**（D4: 58 主题 / 13 矛盾 / §397.10 五条修正全部落位） | `measured_D1..D4.md` |

## 分批数据在哪

```
/tmp/claude-1012/-home-jiaming-workspace-Cost-Aware-Routing-for-Web-Usage-Agents/
  ed15cb9e-3b51-4b2a-95da-c59606b0a51e/scratchpad/batches/
    A_adjudicated_nonum.jsonl   B_retracted_unverified.jsonl
    C_measured_nonum.jsonl      D1..D4.jsonl   E_rest.jsonl
```

⚠️ scratchpad 是 session 专属的。**新 session 若发现路径不在，用这条重新切分**：

```bash
# 切分逻辑见 PHASE1_PLAN.md；或直接按 type 从 ledger.jsonl 过滤
.venv/bin/python3 -c "
import json
rs=[json.loads(l) for l in open('docs/reference/known/ledger.jsonl') if l.strip()]
print({t: sum(1 for r in rs if r['type']==t) for t in
       ('MEASURED','ADJUDICATED','RETRACTED','CLAIM_UNVERIFIED','DATA')})"
```

## 为什么 A 必须由 Claude 亲读、D 可以交 subagent

**A（831 条裁定）** = 「这事定过了 + 为什么」。防重做的核心就是这个「为什么」——
一旦被概括就失去作用。实证：B-1806（measured-cost tie-break 被否）之所以会被重新提起，
正是因为**理由**没被记住，而不是结论没被记住。所以 A 不外包。

**D（875 条带数字测量）** = 已经过数字核验（99.6% 可追溯），是唯一被机制覆盖过的一批，
外包风险最低。且已强制要求每条结论附**原文片段**，主 session 可抽查。

## 每批的产出格式

见 `PHASE1_PLAN.md`。要点：**聚合不是转写** —— 把散落几十个 § 讲同一件事的记录
归成一个主题，给出「当前值 / 演变 / 已作废 / caveats / 证据 / 原文片段」。

## 三条不可违反

1. **数字原样抄，绝不做算术。** §302 已 RETRACT 一条线性分解为 category error
   （跨 model/modality/serving/perturbation 四个不可比维度）。noise 类数字
   （self_drop 6.7/7.6pp · discordance 14.3pp · κ 0.614 · H3 轴 1.35/2.09pp ·
   跨 GPU ±3-5pp · id-shuffle 20.0%/12.5% · AMENDMENT_07 Δ−3.2pp）一律各自带 scope 并列。
2. **caveats 一字不丢。** 尤其工具自带的
   `instability proxy, NOT H1 drop-one bias correction; 小样本/可能混代码版本 = upper-bound risk trigger`。
3. **矛盾不调和。** 两条打架就并列标 ⚠️，不选边 —— 在没有新证据时制造确定性，正是这次重建要修的病。

## 已知需在结论层标注的坑

- **§397.10 是 CORRECTION 节**，作废了 §397.4 与 §397.9 的部分结论，并追加 (4)(5)。
  读 §397.4 / §397.9 必须连它一起读。
- **§397.4「全 archive 只有一对同模式重跑」是假的** —— manifest 19 组 ≥2-run，
  `results/repro_replicates/` 有**两个** clean replicate。
- **§397.9 的 id-namespace 表不完整** —— compact 1..K 是**三个** mode
  （som / phantom_som / phantom_text）；**Vision 零 element_id**，其幻觉率 0.000 是
  结构性不适用而非「native」。主 session 已实证（模型输出 id：p-som 1/12/68 ·
  p-text 1/13/72 vs p-prompt 139/4074/26235 · dom 2/3606/61833）。
- **`preregistration_decision_test.py` 注释 stale**：第 35 / 46-47 行仍写
  `PRIMARY GATE = DerSimonian-Laird`，而实现已是 FE-only。实现对、注释错。

## 接力时怎么开头

新 session 第一句可以是：

> 读 `docs/reference/known/conclusions/PROGRESS.md`，接着做未完成的批次。

然后按表格挑一个 ⬜ 或 🔄 的批开始，做完**立即落盘并更新本文件的状态列**。

## 主 session 已裁的悬案（新 session 不必重裁）

- **§397.9「符号相反 = 真交互」仍然成立。** 台账给它挂了 `named by RETRACTED §397.10` 的 flag，
  D4 保守并列（其矛盾 #9）。裁定依据：§397.10(1) 修正的是"compact namespace 只有两个 mode"
  这个隐含说法（实为三个，SoM 也在内），而该论证用的两组比较（DOM/P-prompt 都 native、
  P-text/P-SoM 都 1..K）**恰恰被 §397.10 确认**；主 session 另有实测支持（模型输出的 element_id：
  p-som 1/12/68 · p-text 1/13/72 vs p-prompt 139/4074/26235 · dom 2/3606/61833）。

## 合并阶段待做（只有主 session 能做，subagent 结构上做不到）

1. **跨批作废交叉标注**：B 批持有全部 156 条 RETRACTED，D1–D4 各自看不见它们
   （按 type 切批导致同一主题的 MEASURED 与 RETRACTED 分家）。D1 已实证撞上这个：
   我给它的例子里 §103「少 ~50%」它找不到，因为那条是 RETRACTED 在 B 批。
   ⇒ 合并时须按 § 号把 B 批作废回标到 D1–D4 的主题上。
2. **D3 报「§299.4 的 Δ−3.2pp 不在其批次」** —— 须确认该记录归到了哪批，没有整体漏掉。
3. **矛盾清单汇总**：D1 13 条 + D2 10 条 + D3 6 条 + D4 13 条 + B 批若干，去重后成一张表。
   其中"只有 user 能判"的那类（"user 拍板 X" / "advisor 收口 Y"）单独分节。
