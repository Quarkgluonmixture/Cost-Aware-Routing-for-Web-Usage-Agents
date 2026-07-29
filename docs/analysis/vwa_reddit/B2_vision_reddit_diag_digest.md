# /diag digest — B2 × `vision` × reddit

*生成 2026-07-27（Tier-1 全量 + Tier-2 未深挖）*

> **定位声明**：本 digest 是**单 condition** 的失败归因，其中的 per-rule 分布只描述它自己。
>
> ✅ **discover-then-freeze 已完成**（2026-07-27）：reddit 规则批 P41–P46 + B-1890 修复 + P33
> reddit 路径扩展已落码，`RULESET_VERSION` = `8-reddit-p41p46-b1890fix`，**全部 36 个 canonical
> condition（reddit 18 + cls 18）已在该版本下重扫**，版本一致性由
> `scripts/analysis/diag_rescan_all.py` 校验 → **cross-mode / cross-model 定量聚合现已解锁**。
>
> ⚠️ v7→v8 的 cls 行为**不是**字节不变，差异全部经过定性核实：`P35`/`P39` 的旧命中因
> B-1890 死字段修复而移除（抽查确认那些 episode 确实有 6–8 个突变步，旧命中是错的）；
> `P33` 在 cls 上 +1 例（cls task 233 的 intent 实际要求访问 reddit，旧正则漏检）。


## 1. Header

| 字段 | 值 |
|---|---|
| **Run** | `B2_vision_reddit_20260719` |
| **Condition** | `phase1_vision_router_0` |
| **Site / Mode / Model** | reddit / `vision` / B2 = Gemma3-VL `google/gemma-3-4b-it` (local) |
| **Episodes** | 205 |
| **SR** | **2.44%** (5 success / 200 failed) |
| **ruleset_version** | `8-reddit-p41p46-b1890fix` |
| **Tier-1 三子集** | failed+hit 199 · **failed-NO-hit 1** · success+hit 2 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 456 | 84 |
| `P5` | 感知缺失循环 | 281 | 147 |
| `P31` | budget 耗尽未完成 | 170 | 170 |
| `P14` | URL 自环 | 94 | 79 |
| `P1` | 元素中心越界 | 57 | 9 |
| `P33` | 导航至裸图片 URL 幻觉 | 33 | 33 |
| `P25` | 跨站任务跳过其中一站 | 33 | 33 |
| `P12` | 从不翻页 | 15 | 15 |
| `P46` | COMMENT_INTENT_NO_TYPE | 15 | 15 |
| `P10` | 跨步数值记忆失败 | 3 | 3 |
| `P27` | 找不到即放弃 | 1 | 1 |

**success 侧 fire 的规则（presence-only 误报审计对象）**: `P33`×1, `P41`×1

**failed-NO-hit episode（deterministic 盲区）**: [64]

**success episode**: [77, 78, 98, 120, 160]


## 3. Tier-2 深挖

**本轮未做 Tier-2 深挖。**

依 /diag skill 的跨-condition 预算纪律，Tier-2 只投给 (a) SR 异常低 / (b) 新 site-mode / (c) no-hit 比例 >25% 的 condition。本 condition 的 SR 落在该 model 的常规区间、no-hit 比例为 0.5%（<25%），故本轮排在 B2 六条之后。

**待深挖子集已就绪**：failed-NO-hit 1 个（见 §2 列表）+ success-with-hits 2 个（presence-only 误报审计）。

⚠️ 因此本 digest 的三分类**不完整** —— 未深挖不等于「无 scaffold-bug / 无 benchmark-FP」，只代表本轮没有查。请勿据此下「pipeline 干净」结论。

## 4. 🔁 Self-evolving — 提议规则

待 Tier-2 深挖后补。

## 5. Actionable

- ⚠️ **本 cell 的 success 含 task 160（B-1889 benchmark-FP）**。若排除，SR 2.44% → 1.95%。排除与否属 prereg 级改动，**待 user / advisor 决策**，本 digest 不自行调整数字。
- scaffold-bug 情况未知（Tier-2 未做）。

---

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物

---

## Tier-2 补记（2026-07-29）— 此前本 digest 只有 Tier-1

`§401.6` 查出 B2·reddit 六个 mode **全部只做了 Tier-1**，三分类因此不完整。本轮补齐。

**Tier-1 全扫（ruleset `8-reddit-p41p46-b1890fix`）**：205 episode，scaffold 规则命中 **0**。
Tier-2 深挖目标 = 本 cell 的 no-hit failed + success-with-hit。

**Tier-2 结论（六 mode 合计 14 个 no-hit failed）**：**全部 agent-limit**，
high confidence ×14；**scaffold-bug 0 · benchmark-FP 0**。
两个候选显式排除：task 179 `invalid_select_option` = parse guard 正常工作
（`consumes_agent_action_budget=false`，无预算泄漏）；task 64/vision
`policy_blocked_offsite` = 护栏按设计工作。

⇒ 本 cell 的 ~1–4% SR 是**真能力地板**（与 §338 六源收敛一致，现有 Tier-2 逐条背书）。

⚠️ **但 success 侧另有发现，不属本 cell 的能力问题**：reddit 有 **7 个 task 共读
`#sidebar > section > ul`**，而 `require_reset` 在 reddit 是 no-op（`envs.py:172`），
**订阅状态在 run 内 205 个 episode 间累积** ⇒ 这批 task 的成败由**执行顺序**决定。
实证 B2·dom 的 178/188/189 三个判成功却从未访问过所需 forum。
task 58/160 已 protocol-excluded，**170/171/178/188/189/190 仍在 scored universe**，
裁定待 user/advisor。详见 实验笔记 §402.5 与 `known/conclusions/measured_D4.md` 附录 B。

**新 P-rule 提议**（success-FP 全 0）：P43 正则修 · URLMATCH_FINISH_ON_SEARCH
（P19 补 `/search?q=` —— 它此前对整个 reddit 站失明）· CONJUNCTIVE_EVAL_PARTIAL ·
SUBMIT_INTENT_NEVER_REACHED_FORM · SELF_ACCOUNT_ONLY。
**SUCCESSFUL_NOOP_REPEAT 明确不落地**（success-FP 16.8%，反例 B2·som task 130）。
