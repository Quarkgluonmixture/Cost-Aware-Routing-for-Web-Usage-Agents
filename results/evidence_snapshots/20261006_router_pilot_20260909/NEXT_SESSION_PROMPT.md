# 新 session prompt — router 整体重想

> 直接复制下面整段作为新 session 的第一条消息。

---

我要重新想清楚这个项目的 router 该怎么做。2026-09-09 我跑了一批探索性实验，数据都在
`results/router_llm_pilot_20260909/`（20 个脚本 + 41 份结果 + `tables/full_table2.json`）。
**先读那个目录的 `README.md`，尤其是「已知缺陷」10 条 —— 里面第 10 条是我自己当天推翻的一个
错误结论。** 数据可信，我对数据的**解读**不一定可信，请当作待检验的假设而不是既定结论。

## 背景（30 秒）

论文 = REALM @ EMNLP 2026 已接受（非归档），camera-ready 09-14 已处理完，不要动它。
下一个目标 = **NAACL via ACL ARR 2026-10-12**。论文 §5 现有五个 routing construction 全部
失败，核心解释是 supply-value coupling（which-mode 标签只在「某个 mode 解出」的任务上存在，
所以监督按 agent 成功率产出）。审稿人 sVJH 点名说没测 richer LLM-based routing / contextual
bandit / RL / online routing after partial interaction。

数据规模：**18,294 episodes**，11 个 cell（VWA cls×{B0,B1,B2,B5} / red×{B0,B1,B2} /
shop×{B0,B1} + WA red×{B0,B1}），6 个 observation mode。

## 当天做了什么（结果都在盘上，可直接复用）

1. **LLM router 三版**（router 模型 = GPT-5.6 luna/terra，走 AWS proxy）
   - v1 只给 intent，zero-shot；v2 加全 40 条 failure 画像 + few-shot 标签 + base rate；
     v3 再加三轴画像（SR/cost/latency）+ 显式 cost-aware 目标
   - 11 cell 全跑；另有 GroupKFold（按 `intent_template_id` 分组）4 cell
2. **3-class 版**（READ/LOOK/BOTH，arm-matched，代表臂在 fold 内选），11 cell
3. **论文自己那个 0-token 正则**改在 cost 轴上评估（`regex_costaware.py`）
4. 一堆分析：三轴 Pareto、严格支配、per-step splice 窗口、best−2nd gap 相关性、
   三个降本来源分解

## 我当天的解读（**请逐条质疑**，这些是要重想的东西）

- 三版 LLM router 的 SR 增益**全部落在 rerun band 内**（5 个有 replicate 的 cell）
- `corr(best−2nd gap, Δbest) = −0.696`，比 `corr(labelled%, Δbest) = −0.564` 更强
  ⇒ 我当时的说法是「routing 有没有活路取决于有没有两个差不多好的臂，而不是标签多少」。
  **n=10，极不稳，这条最需要重新检验。**
- GroupKFold 下增益归零（+0.89→−0.45 等）⇒ 稳态下那点优势来自模板记忆。
  ⚠️ user 已裁定 template sibling **不是 leakage**（真实部署确实见得到同类历史任务），
  随机 fold 与 GroupKFold 回答两个不同的部署问题，都合法，不要再叫它泄漏。
- 免费正则在 6/10 cell 上打赢付费 LLM router
- 三个降本来源：**无解题→最便宜臂 26.9%** ≫ 同解→便宜臂 3.3% > 图像侧 som→vision 1.0%
  ⇒ 我当时的建议是 router 应该做 abstention（判断值不值得做）而不是 mode selection

## ⚠️ 当天推翻的一个结论（教训本身值得看）

我先说「per-step / online routing 是死路，中位分岔窗口 1 步」，**那是拿错字段**：
`obs_url` 是动作**执行之后**的 URL，而路由决策发生在**看到页面之后、动作之前**，
该用 `state_digest.url_before`。重算（B0·cls 四臂）：

- **step0 的 `url_before` 四 mode 一致 = 224/224 = 100%** ⇒ 第 0 步 splice 严格合法
- 中位分岔 **2 步**；决策窗口 ≥1 步 100%、≥2 步 64%、≥3 步 24–37%

## ⭐ 最可能改写论文核心解释的一条（优先想这个）

论文 `~/overleaf-aaai27/sections/6_gap.tex` 现在写着：

> too few labels, and too few tasks where a choice even exists. **They are one wall.**
> … across our eight cells they track the best single mode's success rate at **ρ = 0.952**
> … it predicts its own reversal: run this measurement on an agent that solves most of
> these tasks and the label supply and the contested set grow together.

论文把两堵墙合并，理由是两者都是成功率的函数。**但 2026-09-09 的数据显示还有第三个、
与成功率不同向的量**：`best−2nd gap`（最好的臂领先次好臂多少）。

| cell | labelled%（供给） | best−2nd gap | Δbest (v3 router) |
|---|---|---|---|
| cls_B5 | **51.3%（最高）** | **12.9pp（最大）** | **−2.23** |
| cls_B0 | 43.3% | 2.2pp | +0.89 |
| red_B0 | 26.8%（低） | **0.0pp** | **+1.46** |

`cls_B5` 与 `cls_B0` 都是高成功率、供给充足的格，但 gap 差 6 倍，router 收益一正一负。
⇒ **供给充足 ≠ 有可交换空间。**

要想清楚的是：
- gap 是不是真的与 supply 正交？（先算 `corr(labelled%, gap)`，n=10 要做检验）
- 如果是，论文 §6 那句「它预言自己的反转」**可能反了**：更强的 agent 同时带来更多标签
  **和** 更大的 gap，两者可能相消甚至净负。B5（全场最强 backbone）正是这个方向的
  第一个数据点，而它是全场 router 表现最差的一格。
- 这条如果站得住，是对论文核心解释的**深化而非推翻**，对 ARR 有分量。

## 我想重想的核心问题

**能不能做一个真正 online learning 的 learned router**，它的知识全部来自这 18,294 个
episode 本身，而不是（a）FrugalGPT 那种机械路由——它根本不知道什么算成功，
（b）LLM router 那种预训练先验（「som 厉害」「说蓝色就用 vision」）。

具体想探的形态是 **one-step-lookahead**：先用便宜 mode 跑第 0 步，拿它的
`action.thought`（中位 132–135 字符，内容是 agent 对这题的理解）+ `confidence`
（`mean_logprob` / `mean_margin`，B0 填 4/6 字段）+ observation，再决定要不要升级。
第 0 步 splice 合法（状态 100% 相同），所以**可以用现有数据离线评估，不需要新 fire**。

请帮我想清楚：

1. 这个设计的**监督**从哪来？输入端信息量确实涨了，但路由目标的标签仍只在
   （便宜的失败 ∧ 贵的成功）上有定义 —— supply-value coupling 这一半没被绕开。
   有没有办法把 step-level 的丰富信号变成 step-level 的**标签**？
2. 它和论文已有的 confidence cascade（§5 第四个 policy，基于 `kadavath2022know` 的信号）
   到底差在哪？会不会只是它的换皮？如果是，那论文的负面结论已经覆盖它了。
3. `best−2nd gap` 那条相关性如果站得住，它对这个新设计意味着什么？
   （gap 大的 cell 是不是无论什么 router 都没救）
4. 如果要投 ARR，这条线是作为**正面结果**还是**又一个负面结果**写？
   两种都可以写，但决定了要跑什么。

## 纪律（这个项目的硬规矩）

- **效应量一律对着 rerun band 读**，不是对着零。band 来源 `docs/analysis/cross_sites/
  noise_floor_inventory.json`；11 个 cell 里只有 5 个有 replicate，其余读不了
- 动手前先查防重做台账：`.venv/bin/python3 scripts/maintenance/known.py <关键词>`
- 术语：condition = (site, model, mode)；cell = (site, model)，别混
- `cls_B5` 是**自路由**（router 与 agent 同为 GPT-5.6），该格结论带混淆
- 别碰 camera-ready 的稿子（`~/overleaf-aaai27`，09-14 已定稿 push）
