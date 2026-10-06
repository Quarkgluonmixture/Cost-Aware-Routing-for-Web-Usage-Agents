---
type: analysis
status: audit
created: 2026-09-22
purpose: 判定「把 task-level cost-aware router 升级成 trajectory-level adaptive-compute controller」这条方向在现有数据上成不成立；先盘数据、再盘已做过的、最后裁决
scope_warning: 本文件不是阶段权威。stage / 进度 / 下一步以 docs/checkpoints/phase1_plan.md + next_steps.md 为准，本文件只给判断与证据指针。
---

# Trajectory-level adaptive compute — grounded research audit

> 触发：2026-09-22 用户提出把静态 DOM/Vision router 升级为 π(H_t) → (model, perception, verify/recover) 的控制器，
> 要求**先审数据再判断**，不要凭感觉 brainstorm。
>
> **一句话结论**：这个方向的**动机是对的、执行路径基本已经走过、而且在"换表征 / 换模型"这个 actuator 上已被本项目自己的数据否掉**。
> 真正还活着的是**同一个想法换一个控制变量**：控 step budget（已有 2/2 prospective 预注册 PASS），
> 外加一条从没碰过、而且 per-step 标签稠密的轴：**grounding path routing**。

---

## 0. 读这份文件之前必须知道的两件事

### 0.1 本地台账是旧的

| | 记录数 | 最后一条 |
|---|---:|---|
| 本地 `docs/reference/known/ledger.jsonl` | 2207 | §448.6 / 2026-08-09 |
| E: 镜像 `E:\dgx-jiaming-backup\...\ledger.jsonl` | 2919 | §527.7 / 2026-09-15 |

**§449 之后的全部内容只在 E: 上**——包括 abstention、early-abort、retry-vs-switch、2026-09-09 的 LLM-router pilot、
one-step lookahead、contextual bandit、budget router。若在本地 `known.py` 查不到某个主题，**不等于没做过**。
查 E: 侧用：

```bash
python "E:/dgx-jiaming-backup/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/scripts/maintenance/known.py" --no-color <terms>
```

最关键的单一文件：`E:\...\docs\analysis\cross_sites\one_step_lookahead_2026-09-09.md`（308 行 / 25 节，下称 **§505 digest**）。
它已经回答了本次提案的大部分问题。**本文件不复述它的表格，只引结论 + 指针。**

### 0.2 提案里的两个前提，一个对一个错

- ✅ **四个模型是对的**，且第四个确实是 GPT。实测自 run_meta：
  `B0 = qwen.qwen3-vl-235b-a22b`（AWS proxy）· `B1 = Qwen/Qwen3-VL-4B-Instruct`（本地）·
  `B2 = google/gemma-3-4b-it`（本地）· `B5 = global.openai.gpt-5.6-terra`（AWS proxy）。
  （`B3 = XiaomiMiMo/MiMo-VL-7B-RL-2508` 只有 config、**0 episode**；`B4 = eu.anthropic.claude-sonnet-5` 只有 2 次单集 smoke。
  另外 VWA 的 LLM judge 是 `gpt-4o-mini`——即**评测器本身也是 GPT**，见 `docs/archive/paper_drafts_pre_rewrite_2026-08-03/section3_definition.md`；
  submodule 本地未 checkout，源码级 UNVERIFIED。）
  ⚠️ **B4/B5 的 config 只在 E: 镜像上**（本地 `configs/` 只有 B0/B1/B2/B3），所以在本地仓库 grep `gpt|openai`
  会得出"项目没用 GPT"的**错误结论**——我自己第一轮就是这么错的。
- ⚠️ **`AGENTS.md` 的 fire state 是旧的**：它写「Phase 1a 截至 2026-05-17 尚未在 A100 点火」，
  而 bundle 里 36 个标 paper-grade 的 condition 来自 2026-05-25 及以后的 run。
  以 `docs/checkpoints/phase1_plan.md` 为准；这条不影响本审计的结论，但影响"paper-grade 这个标签算不算数"。
- ❌ **"agent 没开 reason / plan 模块"这条只对了一半**：two-stage planner/grounding 模块 `m4` 在 runner 里是**实现完整的**
  （`p79/experiment/runner/main.py:3064-3081`，多一次 model call 产出 sub-goal），只是 Phase-1 全部 condition 的
  `module_flags` 都是 False。同理，**mid-episode 模态升级也是实现完整、从未开过**（见 §3.4）。

---

## 1. 实际数据 inventory（实测）

### 1.1 数据在哪

| 位置 | 内容 | 有逐步数据吗 |
|---|---|---|
| 本仓库 `results/` | 只有小体积策展子集 + provenance JSON；`results/*` 基本 gitignored | ❌ |
| `_incoming_evidence_20260920/p79_smalltext_20260920.tgz` | A100 paper-grade 主机 2026-09-20 导出：**20,284 个 episode summary** + 123 condition_meta + 112 condition_summary + 96 run_summary | ❌ 只有 episode 级聚合 |
| **`E:\dgx-jiaming-backup\...\results\`**（DGX 只读镜像） | **101 个 condition 目录、18,816 个 `*_steps_v2.jsonl`**，覆盖 B0/B1/B2/B5 × 6 modes × cls/reddit/shopping + WA | ✅ **全量逐步日志在这里** |
| 任何位置的 `artifacts/` | **0 个目录**（镜像里 `find -type d -name artifacts` = 0） | ❌ **原始 DOM / 截图一份都没有** |

> ⚠️ **最重要的一条数据边界**：逐步 telemetry 在，**逐步 observation 不在**。
> 这直接决定哪些 RQ 能做（见 §4）：凡是需要"当时页面上有什么""点到的元素文本是什么"
> "thought 与 observation 是否一致"的分析，**本机做不了**，要么去 A100 拉、要么重跑。

**逐步数据的分布**（E: 镜像，18,816 个 episode 有 `*_steps_v2.jsonl`）：

| baseline | classifieds | reddit (VWA) | shopping (VWA) | wa_reddit | 合计 |
|---|---:|---:|---:|---:|---:|
| B0 qwen3-vl-235b | 2,689 | 2,607 | 1,302 | 634 | **7,232** |
| B1 Qwen3-VL-4B | 2,020 | 1,640 | 2,604 | 1,040 | **7,304** |
| B2 gemma-3-4b | 1,344 | 1,230 | — | — | **2,574** |
| B5 gpt-5.6-terra | 1,706 | — | — | — | **1,706** |
| B3 / B4 | — | — | — | — | **0** |

`wa_shopping` / `wa_shopping_admin` 在任何地方都没有逐步数据。

> ✅ **好消息，且是本次审计最实用的一条**：manifest 标 `paper-grade` 的 **36 个 (run_dir, condition) 全部在 E: 上有逐步 JSONL**，
> 且抽 3 个 condition 做 md5 比对，**bundle 与 E: 的 episode summary 6/6 字节相同**——同一批 run，不是另一次 fire。
> ⇒ **paper-grade 的逐步分析不需要再去 A100 拉数据。**
> bundle 缺步数据的 20 个 pair 全是 `_archive_*` 半跑 / B4-B5 单集 smoke / `latest_*` 软链（1,909 episode，全非 paper-grade）。

### 1.2 逐步 schema（`p79/experiment/types.py:55` StepRecordV2，实测一行确认）

每步真实落盘的字段（非推测，读自 `B0_dom_classifieds_..._R31194/episodes/classifieds_task_100_steps_v2.jsonl`）：

- `action` / `raw_action`：含 **`thought`（自由文本）+ `confidence`（verbalized 0–1）** + action 参数
- `confidence`：`{mean_logprob, min_logprob, mean_margin, min_margin, verbalized}`
- `latency_ms`（total / obs_prepare / backend_infer / env_step / …）· `tokens` · `cost_usd` · `energy`
- `action_success` · `page_changed` · `agent_visible_changed` · `page_change_reasons`（12 类）· `text_similarity`
- `state_digest`：`{url_before, url_after, title_before/after, dom_complexity, text_length, scroll_y_before/after}`
- `router`：`{enabled, decision, trigger_reason[], overhead_ms{}}`
- `reward`（逐步）· `done` · `error_category` · `parse_valid` · `element_bbox` · `dialog_meta` · `locator_route_meta*`

### 1.3 规模与基线率（实测，20,284 份 episode summary 全量聚合）

摘几行有代表性的（完整 59 行见 `C:\workspace\_p79_audit_scratch\scan_triggers.py` 的输出）：

| cell | n | SR% | mean steps | trigger 全空的 episode% | page_unchanged_rate |
|---|---:|---:|---:|---:|---:|
| B5 dom cls | 452 | **24.1** | 16.2 | 62.6 | 0.038 |
| B0 som cls | 448 | **28.3** | 13.6 | 60.5 | 0.077 |
| B0 dom cls | 224 | 17.4 | 15.6 | 43.8 | 0.133 |
| B0 phantom_text wa_reddit | 104 | **35.6** | 19.8 | 26.0 | 0.347 |
| B1 dom cls | 515 | 6.8 | 21.1 | 35.9 | 0.288 |
| B1 vision reddit | 205 | **2.9** | 23.1 | 19.0 | 0.528 |
| B2 vision cls | 224 | **2.2** | 28.2 | 1.8 | 0.670 |
| B2 phantom_prompt reddit | 205 | **0.5** | 27.7 | 1.5 | 0.602 |

三条读法：

1. **SR 区间 0.4% – 35.6%**。B2 全部 cell 在 0.4–3.9%——这些 cell 上任何 outcome-conditioned 学习都是在 1–8 个正例上做。
2. **步数与失败强相关**：成功的 episode 早早 finish，失败的一路跑到 30 步 cap（§505 digest §21 实测：
   **失败 episode 吃掉 92% 的支出**，60 个 condition 的中位数，范围 76–99%）。
3. **trigger 在 router 关着的时候照样被计算并落盘**（`p79/experiment/router.py:115-163` 在 enable 判断**之前**）。
   这意味着"router 如果开着会在哪儿触发"是一条**全量免费信号**。但见 §2.4：它的精度很差。

---

## 2. 数据质量审计

本项目已有一份逐字段体检：`docs/analysis/_data_quality_audit.md`（2026-08-03，48 condition）。
下面只列**对本方向致命的**几条，外加本次新测的两条。

### 2.1 ⛔ 七个 risk / mutation 计数器从来没被写过（已造成过一次 retraction）

`destructive_action_count` · `delete_remove_count` · `submit_create_count` · `cart_mutation_count` ·
`effective_mutating_action_count` · `cycle_mutating_action_count` · `repeated_same_mutating_action_count`
—— 全部在 `types.py:615-621` 声明、`schema_migrations/v2.py:374-386` 的 `NOT_POPULATED_BY_RUNNER` 里登记、
**runner 里没有任何写入点**。`footprint_risk_score` 更糟：全仓库无赋值，恒 `None`，且因为默认值是 `None` 而非数字，
它**漏在** `tests/test_b1890_reserved_fields.py` 的常量扫描网之外。

本次实测独立确认：20,284 份 episode summary 里，这七个字段**没有一份 > 0**，`footprint_risk_score` 没有一份非 None
（含 4,350 份 shopping episode）。

> **前车之鉴**：2026-07-27 有过一条 headline「256 个成功里 114 个是零 mutation 的 mutation 任务」，
> 就是把这些**默认值**读成了测量值，连同两条下游证据链一起被 retract（`v2.py:356-373`）。
> ⇒ 任何 risk-aware 分析**必须自己从 step JSONL 重算**，不能读这些字段。

### 2.2 `page_changed` 有系统性假阳性

volatile DOM 片段（疑似 CSRF nonce）把 `content_changed` 顶起 → `page_changed=True` 而 url/scroll/复杂度全程不变。
step 级疑似假阳率 **vision 9.02% > som 7.90% > pprompt 6.86% > dom 6.40% > psom 6.32% > ptext 5.48%**；
`page_changed` 恒 True 且 URL 全程不变的 episode = **VWA 148 + WA 13 = 161**，这些 episode 的
`page_unchanged_streak` **永不累积**。按修正口径重算，rule router 的 escalation 触发率被**系统性低估约 9.9%，且跨 mode 不均**
（som 13.0% vs vision 5.4%）。⇒ **任何以 page-no-change 为核心特征的 loop 研究，跨 mode 比较都带偏。**

### 2.3 逐模型的结构性缺失（②类"假 0"，最危险）

- `tokens.{input_image,input_text}` / `latency_ms.{generate,preprocessing}` / `energy.*`：**B0 全 0%**（走 API，架构使然）→ 跨模型比较里 B0 的 0 是假 0。
- `confidence.entropy`：**B0 只有 4/6 子字段**（proxy 只回 top-2 logprob）→ 任何消费 entropy 的 router 分析在 B0 上不可用。
- `tool_call_*` 四件套：只在 B0 填充（30/48 condition 全 None）。
- `element_bbox` / click 的 `locator_route_meta`：**vision 的 click 是纯坐标派发，0/914–0/2270 填充率**——
  vision 的 grounding 指标与其他 mode **分母不同**，不是"vision 没这个问题"。
- `tokens.thinking`：全项目恒 0（B5 的 proxy 不回 reasoning token）⇒ "长 CoT 是 vision 的隐藏成本"在本数据上**不可测**。

### 2.3b 两条**新测出来的**跨模型可比性硬伤（直接影响提案里的 confidence 与 thought 两条线）

**H1 — B5（GPT-5.6-terra）完全没有 logprob confidence。** 它的 `confidence` 字段字面就是 `{"verbalized": 0.98}`，
四个 logprob 键**不存在**（不是 null）。这是**声明的**不是意外：`exp_v2_B5_dom_classifieds.yaml:85`
`logprobs_unavailable: true  # B-1990 — declared, not discovered`（OpenAI 档对 `logprobs` 直接回 400）。
⇒ **任何基于 logprob 的 confidence / router-signal / AUROC 分析都不能包含 B5**，而 B5 恰好是最强、SR 最高的 baseline。

**H2 — B0 的 thought / confidence 有非随机缺失。** 抽样实测填充率：

| 字段 | B0 dom/som | B1 | B2 | B5 |
|---|---:|---:|---:|---:|
| `action.thought` | **85% / 81%** | 100% | 98–100% | 100% |
| `action.confidence`（verbalized） | **76% / 76%** | 100% | 93–97% | 100% |

缺的那些行 **`parse_valid` 全是 true**，横跨 click/type/finish/scroll/back/select_option——
235B 模型就是会省略这两个可选 tool-call 字段。⇒ **"verbalized confidence"的跨模型比较是在 B0 的 76% 步 vs B1/B5 的 100% 步上做的，
而且缺失非随机。** 任何 thought/confidence 分析必须先报这个分母。

### 2.3c 轨迹形状：loop 基本是"小模型现象"

抽样实测（每 condition 30 episode，VWA classifieds）：

| | B0 dom | B0 som | B1 dom | B1 som | B2 dom | B2 som | B5 dom | B5 som |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 步数中位 | 11.5 | 9.5 | **30** | **30** | **30** | **30** | 8.5 | 11.5 |
| 撞 30 步 cap 的比例 | 0.27 | 0.27 | **0.80** | 0.53 | **0.80** | 0.63 | 0.23 | 0.37 |
| **重复同一动作的步率** | 0.002 | 0.005 | **0.056** | **0.135** | **0.120** | **0.125** | **0.000** | **0.000** |
| 页面无变化步率 | 0.176 | 0.139 | 0.318 | 0.400 | **0.606** | **0.609** | 0.127 | 0.087 |
| thought 字符数均值 | 238 | 238 | 271 | 304 | 234 | 233 | **140** | **138** |

⇒ **提案里的 action loop 在强模型上几乎不存在**（B5 实测 0.000，B0 0.2–0.5%），
它是 4B 级模型的行为（B1/B2 5.6–13.5%）。这与 §3.8 的 session-loss 只发生在 B2/B5 的事实合起来说明：
**可路由的"病"和"药"分布在不同的模型上**——需要救援的 cell 恰好是 SR 1–7%、触发器几乎恒亮的那些。
（另注：部分 episode 有 31–34 行 step 记录，超过 `max_steps: 30`——recovery/retry 行会追加在 cap 之后，
任何按 step index 做的对齐都要处理这个。）

### 2.4 本次新测的两条

**(a) `action_success` 与 `page_changed` 近乎同一个特征。** 逐步交叉表（实测）：

| cell | success=F,changed=F | success=T,changed=F | success=T,changed=T |
|---|---:|---:|---:|
| B0 dom cls (3,496 步) | 17.3% | 4.7% | 78.0% |
| B2 vision cls (6,328 步) | 68.9% | 0.4% | 30.8% |

`action_success=False ⇒ page_changed=False` **无例外**。episode 级 `no_op_rate` 与 `page_unchanged_rate`
在**全部 59 个 cell 上数值完全相同**（定义不同，被 finish-step 的排除项抵消）。
⇒ 提案里的 "repeated action / no page change / action failure" **不是三个特征，接近一个**。

**(b) router trigger 在弱 cell 上几乎没有选择性。** B2 vision cls：`action_failed` 触发 4,195 次 / 约 6,300 步 ⇒ **约 2/3 的步在触发**。
SR 1–3% 的 cell 上"哪里卡住了"等于"哪里都卡住了"。trigger 的**精度随 SR 下降而崩塌**，
而低 SR 恰恰是最需要救援的场景。

### 2.5 分析口径的硬规则（GOTCHAS §9）

**分析脚本一律经 `results/phantom_paper/run_manifest.yaml` 白名单取 run，禁止裸 glob**（§442.8）。
⚠️ 本文件里我自己跑的几个探针（`scan_triggers.py` / `precursor_probe.py` / `xtab.py`）
**用的是裸 glob 且单 run**，因此它们是**可行性探针，不是可引用的测量**——表里出现过同一 condition 的两个 run
（例如两份 `B0_som_classifieds`）就是这个原因。

---

## 3. 已经做过的（内部 prior work）——这是本次审计最重的一节

提案里的 12 个方向，**10 个已有直接对应的实验，大多数是负结果**。按提案编号对齐：

### 3.1 RQ1 trajectory-conditioned routing → **已做，决定性负结果**

- **方差分解**（§505 digest §8，双重复 replicate cell）：
  task 主效应 **0.0773**（cls_B0）/ 0.0557（red_B0）；mode 主效应 0.0026 / 0.0004；
  **task×mode 交互（= routing 的靶子）0.0209 / 0.0056**；replicate 噪声 **0.0618 / 0.0459**。
  ⇒ **交互/噪声 = 0.34 / 0.12。靶子比噪声小。** 这是 outcome 矩阵的性质，**与模型类、特征集无关**——
  换更丰富的 trajectory 特征救不了一个不在标签里的信号。要把标签去噪到 interaction > noise/k，
  cls_B0 需要每个 (task, mode) **k ≥ 3 次重复**，red_B0 需要 **k ≥ 9 次**。
- **配对实现检验**（§505 digest §1）：18 对同 condition 重跑、326 个在两次之间翻面的 task，
  对每个 step-0 信号比较"成功那次 vs 失败那次"——
  mean_logprob 0.476 · verbalized 0.523 · thought length 0.507 · latency 0.521，**sign-test p 全部 > 0.30**。
  因为是 task 内配对，难度被完全抵消 ⇒ **routing 需要预测的那个量，在第一个动作之前根本不可见。**
- 学过的 router（0/6 cell Pareto 胜过 always-cheapest）在三轮独立修正后**全部存活为负**（§387.16.4 / §388.4 / §388.7.2 / §392.2），
  在最有利的角落（同 family 池化 × cost-tier 标签）扩展为 **0/26**（§399.1）。
- RouteLLM 式 TF-IDF kNN、FrugalGPT 式 offline cascade、Vardanyan 式 DOM→Vision 失败升级、
  LazyMCoT 式长度触发——**全部被固定策略严格支配**（§374 / §376）。
- **2026-09-09 LLM router pilot**：路由器本身就是 GPT-5.6（luna/terra 两档），v1 intent-only / v2 +40 条失败规则 +few-shot
  / v3 +三轴 profile + cost-aware / 3-class READ-LOOK-BOTH，**11 个 cell 上全部落在 rerun band 之内**。

### 3.2 RQ3 failure precursor / prefix 预测 → **已做两轮，负**

- **L2 prefix-k AUROC**（2026-05-16）：k=3 时 **8/12 cell < 0.65**，full-episode 也只有 0.657–0.788。
- **early-abort learnability**（§459，2026-08-12，正经 CV + shuffle-null + 固定策略对照）：
  AUROC **0.336–0.667**，**多行低于自己的 shuffle-null**，**18/18 个等损失比较输给"第 k 步直接截断"这种傻策略**。
- **本次独立复测**（prefix k=3，4 个 cell，逐特征 AUROC，裸 glob 探针）：

  | 特征 | B0 cls som | B0 cls dom | B5 cls dom |
  |---|---:|---:|---:|
  | mean verbalized conf | 0.502 | 0.489 | 0.583 |
  | min verbalized conf | 0.476 | 0.457 | 0.598 |
  | mean logprob | 0.524 | 0.387 | n/a |
  | n page-unchanged | 0.536 | 0.509 | 0.523 |
  | n action-failed | 0.497 | 0.464 | 0.498 |
  | **n repeated action** | **0.494** | **0.497** | **0.500** |
  | mean text_similarity | 0.400 | 0.377 | 0.524 |
  | n router trigger | 0.486 | 0.479 | 0.504 |

  ⇒ 没有一个便宜特征在 k=3 上有实用判别力；最强的是 `text_similarity`（|AUC−0.5| ≈ 0.12，方向是"成功的轨迹页面变化更大"）。
  **更关键的一条设计更正**：`n_repeated_action` 在 k=3 上**几乎恒为 0**——**前 3 步根本没有 loop**。
  loop 是晚发信号，等它出现时预算已经花掉一大半（§505 digest §14 实测：streak trigger 的中位触发点是 19 步里的**第 7 步**，
  即**episode 成本已花 45%**）。

  ⚠️ **一处必须钉住的数字不一致**：2026-05-16 的 `l2_partial_traj_auroc` 对同名 cell 报 k=3 verbalized AUROC
  **0.615（cls dom）/ 0.642（cls som）**，而我这次测到 **0.489 / 0.502**。两者不能并存，且我**没有**解释它。
  已知的口径差：那份用 n=234/210（**N/A 排除前**的归档 task 集），我用的是当前 run 的 n=224 且剔除 <3 步的 episode；
  它的产出脚本 `scripts/analysis/l2_partial_traj_auroc.py` **已确认被删**（台账 §153.3 标 ABSENT），无法直接复算。
  ⇒ **在钉清楚之前，两个数都不要引**；要用就先重建一份带口径声明的产出脚本。
  （这不影响结论方向：0.49 和 0.62 都远低于可部署阈值，而 §459 用正经 CV + null + 固定策略对照得到的
  "18/18 输给 truncate-at-k" 才是决定性的那条。）

### 3.3 RQ4 confidence calibration → **已做，descriptive 层有结论，可部署读法是负的**

token 级 routing AUROC **0.497**（§7）；verbalized 的高值（0.79–0.85）是 **pre-Phase-A + 不同口径**，
与现制不可合并；evidence 层那些 0.87 级数字是 **episode 聚合 = 偷看未来**，已判定对任何 prefix 决策 **inadmissible**（§459.1 / §460.3）。
**唯一活着的读法是 post-hoc**：用 episode 聚合 confidence 挑最不自信的 25%/50% 重跑，比随机重跑好 **+0.79 / +0.99pp**
（15 对 replicate 中 87%/93% 为正；但**最佳信号是 in-sample 选的 ⇒ 是上界**）。

### 3.4 RQ2 loop→自适应感知 / 触发器 → **机制齐全，但从未开火**

- `RuleBasedRouter`（`p79/experiment/router.py:67-223`）是**完整可执行代码**，不是 stub：
  `dom_size_exceeds_threshold`(12000) · `action_failed` · `page_unchanged_streak`(≥2) · `no_progress_streak`(≥2) ·
  `intent_unfulfilled_streak` · `dom_complexity_high`(500) · `text_length_high`(12000) · checklist 两条（默认关）
  + v6 cascade latch（一触即跳到 `phantom_som` 并锁死）。
- **但它只能换 observation mode，永远换不了模型**：backend 在 episode 循环之外按 condition 绑定一次
  （`runner/main.py:1066`，cache key `(backend_id, seed)`）。
- **`results/` 里不存在任何 `router_on=True` 的 condition**：Phase-1 全是 `phase1_{mode}_router_0`；
  `configs/` 里零个 `router_kind: cascade`；六个 learned router config 走的是**开跑前选一次 mode**的路径
  （`main.py:2986-3001` 显式短路掉 mid-episode 路由）。
- **离线测过它的选择性**（§505 digest §14，cls_B0 dom 失败集）：stuck 信号对"som 会成功"的 AUROC **0.41–0.56**；
  streak trigger 在 **57% 的失败 episode 和 33% 的成功 episode** 上都会触发。
  指示性投影（明确标注"这是对一次 live run 的预测，不是测量"）：**26.3% / $0.0869 vs always-som 27.2% / $0.0724**
  —— 救回 24 个、白切 81 个、**打断 13 个本来会成功的**。
- 好消息：`paper_grade=True` 下**四个 runner 注入控制全部被硬禁**（anti-repeat / no-early-finish / query 净化 /
  baseline retry），`control_intervention` 在全部 48 condition 上是死字段 ⇒ **paper-grade 数据上的 loop 分析没有注入污染**。
- 坏消息：cycle 检测**只写日志不落盘**（`StepRecordV2` / `EpisodeSummaryV2` 里没有任何 `cycle_*` 字段）
  ⇒ loop 研究必须自己从 action 流重算。

### 3.5 RQ5 thought 的增量价值 → **已做，零增量**

- step-0 thought 的 TF-IDF **≤ task intent 的 TF-IDF，29/33 个 (cell, cheap, label) 行**
  （例：cls_B0·dom·self 0.789 vs 0.577）⇒ **step-0 的 thought 只是在复述 intent，而 intent 是免费的。**
- thought 长度在 326 个翻面 task 上 P(succ>fail)=**0.507**。
- 描述层确有跨模型差异（§508.4）：thought 与上一步逐字相同的比例 B2 11.2/14.3% · B1 11.5/11.8% · B0 0/0 · B5 0/0.1%；
  **B0 有 13.8/21.2% 的空 thought**（tool_call 路径丢字段）⇒ 三条输出路径长度不可跨 backbone 比较。
- **未做**：句向量 / 语义重复 / plan-change 检测这类超出 TF-IDF 的表示。这是 RQ5 唯一剩下的增量。

### 3.6 RQ6 transition surprise → **未直接做，但前提被否**

没有"预期效果 vs 实际效果"的实现。但它的最简版本（page-no-change）就是 §3.4 的 streak 信号，
判别力已测（AUROC 0.41–0.56），而且 §2.2 说它本身有 5–9% 的假阳性且跨 mode 不均。

### 3.7 RQ7 progress-aware routing → **被 §505 digest §1/§8 同时否掉**

p_t 的 t=0 版本就是 §3.1 的配对检验（不可见）；Δp_t 需要 task×mode 交互（比噪声小）。

### 3.8 RQ8 risk-aware / irreversible → ⭐ **真正没做过的一条**

- 台账两侧 `counterfactual` / `branching` / `irreversible` **均 0 匹配**。
- 但**事件是真的、有协议、可定位**：全量 `trajectory_events.jsonl`（36 份）实测
  **33 次 `session_lost_contaminated_detected` + 33 次 `session_lost_paper_grade_preserved`**；
  episode 侧 `infra_covariates=['session_lost_preserved']` **39 份**，分布：
  B5 vision cls 15/360 · B2 dom cls 7/224 · B2 phantom_text cls 5/224 · B2 phantom_prompt 2 · B2 phantom_som 2 · B0 phantom_text 2（+ 两份归档）。
  **全部在 classifieds，全部集中在 B2 / B5。**
- 根因有据（§329）：**B2 Gemma 在 R21521 task 4 step 8 点了 `[6561] Logout`，而它的 thought 想去 `[6558] My account`**；
  task 5/6/7 成为受害者。复发率约 **1 波 / 100 task**。处理协议 = B-1868 / `PROTOCOL_NOTE_01`（preserve + covariate，不删不重试）。
- ⚠️ **取证陷阱（我自己先踩了一次）**：click 只存数字 `element_id`，**grep step JSONL 找 "logout" 会假阴**。
  我最初扫 4,168 个 classifieds episode 得到"几乎为 0"，是错的。
  正确口径：`trajectory_events.jsonl` + `infra_covariates`（受害者），或 join observation（肇事者，本机做不到，见 §1.1）。
- **肇事的那一集不打标**（只标受害者）⇒ 要做"哪一步是高风险动作"的监督，标签得自己造。

### 3.9 RQ9 model routing → **做过一次真正的 tier cascade，结论是"并跑有用、检测器没用"**

§505.20（cls, som arm, B0 或 B1 → B5=GPT-5.6）：
B0→B5 都成 41 / 只 B0 成 20 / 只 B5 成 42 / 都不成 121 ⇒ **并集 46.0% vs B5 单独 37.1%**；
pre-flight "B0 会成功" AUROC **0.710**，但**学出来的 triage 只有 −0.4 ~ +2.0pp**，
post-hoc cascade 37.5/41.5/44.2% vs 随机 36.6/40.4/44.1%。
**B1→B5 更差**：AUROC 0.587，triage −1.7 ~ −0.2pp。
⇒ 收益来自**两个都跑**，不是来自任何检测器（检测器对 B0 失败的 AUROC 只有 0.554）。
且 §505.11：**最强的 cell（cls_B5）并没有更可学**——capability 抬高的是 mode 主效应，不是交互，噪声地板没动。

### 3.10 RQ10 cost/latency/token Pareto → **已做，且这里有唯一一条活着的正结果**

- latency 作为独立轴 **已 RETRACT**（§409.2：置换检验 P(≥3)=0.978，观测比随机还差）。
- 四轴（SR/cost/latency/token）：**11 个 cell 里 0 个存在四轴全胜的 mode**；lookahead 在 21 个非退化对里只有 3 个四轴非劣，
  且没有一个落在有 rerun band 的 cell 上。
- ⭐ **真正的杠杆是 step budget，不是表征**（§505 digest §21–24）：
  失败 episode 吃掉 **92%** 的支出；固定 cap 前沿（60 个 condition 中位数）
  cap 25 → −1.79pp / −14% 成本；cap 20 → −2.44 / −28%；**cap 15 → −3.12 / −43%**；cap 10 → −4.91 / −60%。
  pre-flight 两档 budget router（**零 model call**）在 ~41% 节省处：dom +2.5pp vs 随机 +4.5 vs 等成本固定 cap +3.7，
  **6/6 mode 同向**。
- ⭐⭐ **唯一一次预注册的前瞻检验，2/2 PASS**（§510.3 / §515.3）：预测在 2026-09-09 冻结，
  在当时尚未落地的 shop_B1 臂上评估——
  P-text n=216：two_tier 损失 **+1.39pp** vs 等成本固定 cap +1.85 vs 随机 +2.11 ⇒ PASS；
  P-prompt n=432：two_tier **+0.69pp 且成本 −44%** vs cap-15 +2.55 vs 随机 +2.06 ⇒ PASS。
  ⚠️ 台账自己给的限定：单次 run、只有 B1、SR≈5% ⇒ 差距只有 1–2 个 task；**这是方向判据，不是显著性**；shop_B1 没有 rerun band。
- 逐 cell 前沿（§505 digest §24）：**learned two-tier 在 11/11 个 cell、30/40/50% 三个档位上都优于等成本固定 cap**
  （例 cls_B1 在 40% 档：fixed 2.2pp vs learned 0.4pp；red_B1_WA：6.2 vs 4.3）。
  固定 cap 伤得最重的是"成功但慢"的臂（B1 的 som +6.7pp、WA_B0 的 ptext +11.5pp），learned 把它们压到 2.4–8.7。
- 诚实限制（必须一起引）：**GroupKFold 冷启动下 learned 退化成固定 cap**（dom +3.6 vs +3.7）
  ⇒ 那 ~1pp 的 learned 优势是**模板记忆**，不是泛化；
  **planner 式连续 per-task cap 在每个档位都比固定 cap 更差**
  ⇒ **"pre-flight 特征能预测『能不能成』，不能预测『要跑多久』——预算只能分档，不能连续"**。

### 3.11 RQ12 counterfactual / branching → **未做；且只有一个点是 splice-legal**

- **step 0 是唯一严格合法的拼接点**：`state_digest.url_before` 在各 mode 间 **224/224 = 100% 相同**；
  但到 step 0 的 `url_after`，cheap 与 rich 已经在 **27–70%** 的 task 上不同。
  （注意：早先"中位分歧在第 1 步 ⇒ per-step routing 死路"的说法当天被 retract——读错了字段；
  修正后中位分歧在第 2 步、≥2 步窗口 64%。但 §505 自己又把它收窄：**URL 相等 ≠ 状态相等**，所以只有 step-0 peek 严格合法。
  ⚠️ `b5_splice_window.py` 仍是错字段版本，**不要引用它的文件内容**。）
- **没有任何 checkpoint/restore 基础设施**：`checkpoint_id` / `checkpoint_hash_before` / `substrate_restored_from_checkpoint`
  在 `types.py:592-595` 声明后**全仓库再无出现**，`types.py:583` 自己写着"等基础设施落地前全是 None"。
- 环境只有一个入口：`VWAWrapper.reset(config_file)` 全量重置；`navigate_to(url)` 只还原 URL，不还原 cookie/滚动/标签页/表单。
- **确定性不成立（实测）**：同 condition 重跑 per-task 翻面率 **12.1%（B0 cls dom）/ 14.3%（B0 cls vision，κ=0.614）**，
  且这些翻面被归因为 `model_nondeterm`（不是环境污染）。⇒ 重放 action 前缀**不能假定**能重建 H_t。
- **站点重置代价**（读自 `_lib_paper_grade_gates.sh` / `reset_vwa_sites.sh` 的自述测量）：
  classifieds **5–15 秒** · reddit **60–160 秒** · shopping **实测 > 3769 秒**（光 indexer 就 >1800 秒）。
  且 `require_reset` **只对 classifieds 实现**，reddit/shopping 是 no-op（存在跨 episode 状态泄漏的先例）。
  ⇒ **branching 实验实际只能在 classifieds 上做。**

### 3.12 「Jev」

提案里的 "Jev / fast policy" 在本仓库与两份台账里都查无对应。若你指的是某个具体系统，需要你给个准确名字或链接；
若只是"便宜的结构化 action selector"这个概念，那它已经被 §374 的 RouteLLM-kNN / FrugalGPT-offline 基线覆盖，且被固定策略支配。

---

## 4. Feasibility matrix

分级：**A** = 现有数据够（E: 逐步日志 + summary）· **B** = 需要轻量重跑或去 A100 拉数据 ·
**C** = 需要 branching / 新 fire · **D** = 不可行或已被否

| # | 方向 | 现有数据够吗 | 已做? | 判定 |
|---|---|---|---|---|
| 1 | trajectory-conditioned which-mode routing | A | ✅ 已做 | **D — 靶子(0.0209)小于噪声(0.0618)，换特征救不了** |
| 2 | loop 成因分类 → 差异化升级 | A（但 obs 缺失 ⇒ 成因分类只能靠代理） | 部分 | **C** — 分类可做，**验证"升级有用"必须 branching** |
| 3 | failure precursor P(fail in next k) | A | ✅ 已做两轮 | **D — 18/18 输给 truncate-at-k；k=3 无信号** |
| 4 | confidence 校准（描述层） | A（B0 缺 entropy） | ✅ 已做 | **A 可做但低价值**；可部署读法只剩 post-hoc 重跑挑选 |
| 5 | thought 增量价值（TF-IDF 层） | A | ✅ 已做 | **D**；句向量/plan-change 层 = **A，未做** |
| 6 | transition surprise | A | 前提已否 | **B** — 最简版=page-no-change，已知 5–9% 假阳 |
| 7 | progress-aware Δp_t 控制 | A | 间接已否 | **D** |
| 8 | ⭐ risk / irreversible action | A（事件 39+66 例）+ 需重算 mutation | ❌ **没做过** | **A（描述+机制）/ C（干预验证）** |
| 9 | model-capacity routing | A | ✅ 已做（B*→B5） | **D 作检测器 / A 作"并跑价值"的描述** |
| 10 | ⭐ cost–latency–token Pareto + **budget routing** | A | ✅ 已做 + **2/2 前瞻 PASS** | **A — 唯一有前瞻正证据的主线** |
| 11 | cheap-policy 提案 + verify（speculative） | A（verifier 已测） | 部分 | **B** — cross-arm agreement verifier 已测：P(对\|一致) 35% vs P(对\|不一致) 19%，8/8 cell 同向，但 **coverage 10–70%、成本 2×** |
| 12 | branching counterfactual | ❌ | ❌ | **C，且只在 classifieds 可行**（reset 5–15s；reddit 1–3min；shopping ~1h/branch） |
| 13 | ⭐ **grounding-path routing**（identifier contract） | A（~13k 动作级标签） | ❌ **未 featurise 未建模** | **A — 唯一 per-step 标签稠密、且完全没碰过的轴** |

---

## 5. 外部新颖性审计

见本文件 §5-附（由独立文献 sweep 填充）。要点先行：本项目的**内部**否定结果比外部文献更先约束这个方向——
即使某条子想法在文献上是新的，它在**本数据上已经被测过且是负的**，novelty 也救不了它。
新颖性真正还站得住的是 §4 里标 ⭐ 的三条（8 / 10 / 13）。

---

## 6. 优先级最高的三条

### P1 — 「Route the budget, not the representation」（控预算，不控表征）

- **RQ**：给定 pre-flight 特征，能否在不做任何 model call 的前提下，为每个 task 分配 step budget，
  使得在等成本下 SR 损失低于固定 cap 与随机分配？
- **假设**：难度（task 主效应 0.0773）是可学的，而 task×mode 匹配（0.0209）不可学；
  所以控制变量应该是**跑多久**，不是**看什么**。
- **需要的数据**：已有。E: 逐步日志 + `per_task_sr.csv` + noise floor inventory。
- **指标**：等成本下的 SR 损失（pp）· 成本节省% · 与 rerun band 的关系。
- **基线（两道都必须有）**：等成本**固定 cap**（不是随机！）+ 随机分配。
- **Ablation**：two_tier / three_tier / 连续 cap；GroupKFold 冷启动 vs 同模板；train-on-A/apply-to-B。
- **统计**：预注册 + 前瞻评估（已有范式：freeze file + `budget_router_prospective_eval.py`）。
- **主图**：cost-saving × SR-loss 前沿，learned / fixed-cap / random 三条线 + rerun band 阴影。
- **最可能的失败模式**：冷启动下 learned 塌成固定 cap（**已经观测到**）⇒ 论文主张必须是
  "**budget 这个控制变量有效**"，而**不是**"learned budget 优于 fixed cap"——后者 11/11 cell 是负的。
- **值不值得成文**：**值**。这是全程序里唯一有**冻结在前、评估在后**的正结果，且已有候选标题与 reframe 裁决（§505.27）。

### P2 — Grounding-path routing（identifier contract 路由）

- **RQ**：把路由的对象从"看什么表征"换成"用哪条 grounding 路径派发动作"（compact id / native nodeId / 坐标），
  能否在动作级稠密标签上学到一个**能提升 episode SR** 的策略？
- **假设**：文本臂内部的动作成功率差 **17.3pp**（P-text 75.0 / P-SoM 66.5 / SoM 61.8 / DOM 59.9 / P-prompt 57.7）
  已经接近 Vision↔DOM 的 19.1pp ⇒ 瓶颈可能在 dispatch 契约而非模态。
- **需要的数据**：已有，`locator_route_meta.success` ~13k 条动作级标签。
- **必须先回答的两个问题**（否则整条线是空的）：
  1. **动作级成功 ⇏ episode 成功**——两者关系从未测过。先测这个。
  2. **vision 的 click 根本没有 locator_route_meta**（0% 填充）⇒ 跨 mode 比较**分母不同**，
     必须按 action_type 分层，不能把 vision 的 0 读成 0。
- **指标**：动作级 AUROC → episode 级 SR 的传导率；per-(mode, action_type) 分层后的成功率。
- **失败模式**：它是 scaffold 问题不是表征问题 ⇒ **是否算 routing 主张范围，需要你/导师裁定**（台账 §505.6 明写这条待裁）。
- **值不值得成文**：**值得先探**。它是 REALM 审稿人 sVJH 提出的 grounding confound 的正面回应，
  且是"graded trajectory supervision"在现有数据上唯一能落地的形态。

### P3 — 不可逆动作的未定价尾部风险（risk-aware routing 的诚实版本）

- **RQ**：把 agent 造成的不可恢复状态破坏（session loss）计入成本核算后，
  "路由到便宜模型"的成本优势还剩多少？
- **假设**：破坏状态的是**便宜模型**（实测：39 份受害 episode 集中在 B2 / B5，B0 / B1 极少），
  而现行成本口径（token/USD/latency）**完全没有给这类事件定价**。
- **需要的数据**：已有事件（33+33 events / 39 covariate episodes）；
  mutation 计数**必须从 step JSONL 重算**（schema 里那七个字段是死的，§2.1）。
- **指标**：per-(model, site) 的不可逆事件率 · 每次事件的 blast radius（受害 episode 数 × 其成本）
  · 计入后的 cost-per-success。
- **基线**：现行不计价口径。
- **失败模式（要正面写）**：**n 太小**——33 例、单站点、肇事集不打标。
  ⇒ 这条**不能**作为统计主张，只能作为**成本口径的披露 + 一条机制性论证**。
  若要变成统计主张，需要专门 fire（classifieds、B2、多次重复），而这正是 branching 之外的另一种 C 级代价。
- **值不值得成文**：**作为一节值得**（论文的 limitation / cost-accounting 章节），**作为一篇不够**。

---

## 7. 最小分析计划（不烧新 compute，全部在 E: 逐步日志上做）

按"最便宜地杀死或点亮一个假设"排序。每条都先说它能杀死什么。

| # | 动作 | 它能杀死/点亮什么 | 预估 |
|---|---|---|---|
| 0 | **把 E: 的 §449+ 台账与 §505 digest 同步进本地仓库** | 否则每次会话都会重做已做过的事（本次就险些） | 0.5h |
| 1 | 动作级成功 → episode 成功的传导率（P2 前置问题 1） | 传导率若 ≈0，**P2 整条线当场死** | 2–3h |
| 2 | 按 (mode × action_type) 分层重算 grounding 成功率（P2 前置问题 2） | 若分层后 17.3pp 塌掉，P2 死 | 2h |
| 3 | 从 step JSONL 重算 mutation / 不可逆动作代理，逐 (model, site) | 给 P3 一个真实的分母 | 3h |
| 4 | budget router 在 **cls / reddit** 上复刻 shop_B1 的前瞻范式 | 若只有 shopping 成立 ⇒ P1 降级为单站点结论 | 4–6h |
| 5 | thought 的句向量 / plan-change 特征（TF-IDF 之外唯一没做的） | 若仍 ≤ intent，RQ5 彻底封棺 | 3–4h |
| 6 | loop 成因分类（仅描述，不做干预声明） | 给 P2/P3 提供 taxonomy；**不能**用来声称升级有用 | 3h |

⛔ **不做**：任何形式的 which-mode learned router 重试（§3.1 已决定性）；
任何以 k≤3 prefix 为输入的 failure predictor（§3.2 已决定性）；
任何消费 `destructive_action_count` 等七个死字段的分析（§2.1）。

---

## 8. 最终裁决

> **现有数据支持把 cost–accuracy router 推进成 trajectory-level adaptive compute controller 吗？**

**支持，但只有当控制变量不是"看什么 / 用哪个模型"的时候。**

- 作为 **π(H_t) → (perception, model)** 的控制器：**不支持，而且已经被本项目自己的数据否掉了**，
  不是"没试对方法"——task×mode 交互只有噪声的 1/3 到 1/8（cls 0.34 / red 0.12），
  且需要预测的那个量在第一个动作之前不可见（326 个翻面 task 上全部信号 P≈0.5）。
  提案里的 RQ1/3/5/7/9 都落在这条线上。**这些不是可以靠更丰富的 trajectory 特征绕过去的，是标签本身没有信号。**
- 作为 **π(H_t) → (how much compute)** 的控制器：**支持**，而且已经有本程序里**唯一一次冻结在前的前瞻检验（2/2 PASS）**，
  机制清晰（失败 episode 吃掉 92% 支出），跨 6/6 mode 同向。

**最有证据、最有新意、又最现实的一条主线**：

> **把决策变量从"观察什么"换成"投多少"，并用 grounding-path 作为第二条（正交的）决策轴。**
> 主线 = P1 budget routing（有前瞻正证据）· 新机制 = P2 grounding-path（唯一 per-step 稠密标签且没人碰过）·
> 诚实的代价披露 = P3 不可逆风险（便宜模型的未定价尾部）。

**明确否掉**：trajectory-conditioned which-mode routing · prefix failure precursor · thought 的 TF-IDF 层增量 ·
progress-delta 控制 · 作为检测器的 model cascade。这五条在本数据上都已测、都是负的，
而且否定它们的证据是**结构性的**（方差分解、配对检验），不是"我们没调好"。

**需要你拍板的三件事**（见 §9）。

---

## 9. 需要用户/导师裁定

1. **P2 算不算 routing 主张的范围？** grounding path 是 dispatch/scaffold 问题，不是表征问题。
   台账 §505.6 明写这条待裁。若不算，P2 降为附录。
2. **P1 要不要扩站点？** 目前前瞻 PASS 只有 shop_B1（SR≈5%，差距 1–2 个 task，无 rerun band）。
   扩到 cls/red 不烧新 compute，但要占分析时间。
3. **要不要为 P3 单独 fire？** classifieds × B2 的重复 run 是唯一能把 33 例变成统计主张的路径；
   否则 P3 只能是一节披露。

> ⚠️ **§6–§9 已被下面的 v2 修订部分取代**（trajectory routing 的判定被推翻、推荐路线重排）。
> 读到这里请继续往下读 v2。

---

## 10. 本文件的证据出处

- 我自己实测的（裸 glob 探针，**非 canonical 口径**，脚本在 `C:\workspace\_p79_audit_scratch\`）：
  `scan_triggers.py`（59 cell 的 n/SR/steps/trigger）· `precursor_probe.py`（prefix-3 逐特征 AUROC）·
  `xtab.py`（action_success × page_changed）· `scan_session_lost.py`（infra_covariates 全量）·
  `scan_traj_events.py`（trajectory_events 全量）
- 代码事实：`p79/experiment/{types,router,state_change}.py` · `p79/experiment/runner/main.py` ·
  `p79/policies/{learned_router,router_features}.py` · `p79/agents/_shared_vl_utils.py`
- 项目既有文档：`docs/analysis/_data_quality_audit.md` · `docs/checkpoints/router/*.md` ·
  `final_dissertation/{THESIS_ONE_SENTENCE,CLAIM_EVIDENCE_MATRIX}.md` · `docs/reference/GOTCHAS.md`
- **E: 镜像独有**：`docs/analysis/cross_sites/one_step_lookahead_2026-09-09.md`（§505 digest）·
  `results/router_llm_pilot_20260909/` · 台账 §449–§527

---
---

# v2 修订（2026-09-22 当日，对外部评审意见的再裁决）

> 用户拿这份 audit 去做了一轮独立文献复核，提出 7 条反驳。本节逐条裁决，**并把我判错的地方改掉**。
> 结论先说：**7 条里我认 5 条半**。v1 §3.1 的核心推论**被推翻**，§3.2 / §3.5 / §3.6 的判定**范围收窄**，
> §3.12 判**错**。但这轮复核也挖出一条**对反驳方自己不利**的事实：它主张的重做方向，三周前已经有人发表了。

## v2.1 我判错的（改判）

### 改判 1（最重要）：方差分解**不能**否定 dynamic routing

v1 §3.1 用 task×mode 交互（0.0209）小于 replicate 噪声（0.0618）去否掉 **π(H_t) → perception/model**。
**这个推论无效。** 方差分解量的是**固定臂**处理效应的异质性：

    Y(always DOM)  vs  Y(always Vision)

而 dynamic routing 的估计量是

    τ(H_t) = E[Y | do(a_t = Vision), H_t] − E[Y | do(a_t = DOM), H_t]

后者不被前者 bound。一个 20 步任务里只有第 11–12 步需要视觉，这种任务的 always-Vision 可以不优于 always-DOM
（其余 18 步视觉更吵更贵），**交互项可以接近 0，而切换策略仍然严格更优**。
更具体地说：本项目实测 cls·B0 有 **127/224 个 universal-fail 任务**（六个固定 mode 无一能解）——
"固定 mode 都解不了"**不蕴含**"混合策略解不了"。六臂 oracle 43.3% 是**固定臂**的上界，不是动态策略的上界。

⇒ **v1 §3.1 表格里 "trajectory-conditioned which-mode routing = D" 收回**，正确判定是 **未测试**。
被方差分解真正否掉的只有 **"这个 task 天生更适合 X 模态"** 这一类 pre-flight 主张。

（保留的部分：配对实现检验（326 个翻面 task，全部信号 P≈0.5）依然成立，但它只否掉 **step-0** 可预测性。
反驳方说得对——step-0 不可预测恰恰是"要往后看"的论据，不是"往后看也没用"的论据。）

### 改判 2：prefix 失败预测的**标签**可能一直是错的

v1 §3.2 判 RQ3 = D，依据是 L2 prefix-k AUROC + §459 early-abort + 我自己的 k=3 探针。
**这三个用的是同一个标签：把 episode 的最终 success/failure 倒灌给它的每一个前缀。**
一条最终失败的轨迹，前 3 步完全可能是对的。把它们标成 failure prefix 是**标签噪声**，
而这正好解释了为什么我测到的 AUROC 贴着 0.5、以及为什么 §459 有多行低于自己的 shuffle-null。

**外部证据（已用 arXiv API 核实，非二手）**：`arXiv:2609.02057v1`（2026-09-02）
*Monitoring Web Agents Without Internal Signals: Observable Trajectories and Key-Step Supervision*
（Pan, Shen, Lu, Ding, Cheng, Wang）。摘要原文：

> *"Instead of inheriting the final result label, we label the first critical error that remains uncorrected in
> the observed continuation and is associated with final failure as a key-step boundary, **preserving valid
> early prefixes of failed trajectories as on track**."*

它在 WebArena-Lite + Online-Mind2Web、5 个开源/闭源 backbone 上测出：**observable trajectory signal 与
internal-signal baseline 相当**，支持固定 false-cut 预算下的早期干预，并能跨 held-out 站点类别迁移。

⇒ **v1 §3.2 判定收窄**："**terminal-label** 的 prefix 预测已被否（三条独立证据）；
**key-step / local-hazard** 形式**未测试**，且外部证据显示这个重述是对的。"

### 改判 3：thought 的判定写过头了

v1 §3.5 标题写"已做，零增量"，正文却自己写着"未做：句向量/语义重复/plan-change"。
标题改为 **"step-0 的 TF-IDF 层已否；thought 的时序动态未测试"**。
实测掉的只有两样：step-0 thought 的 TF-IDF（29/33 行 ≤ intent）与 thought 长度（P=0.507）。
**没测过的是差分量**：thought_t 相对 thought_{t−1} 的语义重复 / 假设持续 / 计划变更 / 不确定语升级 /
恢复语 / intent–action 不一致。§508.4 只测了"逐字相同"的比例
（B2 11.2–14.3% · B1 11.5–11.8% · B0 0% · B5 0–0.1%），那是描述统计，不是预测检验。
⚠️ 做这条必须先报 §2.3b 的 H2：**B0 有 15–19% 的步没有 thought**、B5 的 thought 短 40%，跨模型比较有偏。

### 改判 4：`page_changed` 弱 ≠ transition surprise 被否

v1 §3.6 用 page-no-change 的判别力去否 transition surprise。两者不是一回事：
`page_changed` 只是 1[o_{t+1} ≠ o_t]，transition surprise 是 d(ô_{t+1}, o_{t+1})。
**本项目自己的 logout 案例就是这个区别的教科书例子**：thought 说"点 My account 打开账户设置"，
实际跳到 login page——**页面确实变了**，二值口径判为"正常 transition"，语义口径是一次极大的 surprise。
⇒ 改判为 **未测试**，但见 v2.6 的可行性限制（本机没有观测，只能做**结构化降级版**）。

### 改判 5：speculative agent planning 不是 query-routing（v1 §3.12 判错）

v1 说"Jev 式 cheap-policy 已被 §374 的 RouteLLM-kNN / FrugalGPT baseline 覆盖"——**概念分类错误**。
query → 选模型，和 cheap policy 先出 action、再由 authoritative actor 验证/提交，是两条不同的文献线。
已核实存在：`arXiv:2609.03236`（2026-09-03）*Speculative Macro Commit for Faster Tool-Using Agents*、
`arXiv:2605.22154` *IdleSpec*、`arXiv:2410.00079` *Interactive Speculative Planning*。**§3.12 作废。**

### 半认：「第 7 步触发 = 已花 45%」是双刃的

v1 只当负面证据用。公平的读法：对 **rescue** 确实晚（只剩 55% 的步可用），
但对 **abort** 正是价值所在——失败轨迹吃掉 92% 的支出，第 7 步砍掉就省掉后面一大段。
⇒ 改成中性陈述，并直接接到 v2.4 的免费实验。

## v2.2 我不认的三条（带证据）

### ① B0 ∪ B5 = 46.0% **不是** rescue 的 headroom

反驳方用「B5 单独 37.1% vs 并集 46.0% ⇒ 8.9pp headroom」论证 rescue 有空间。
**这个并集是两个模型各自从 clean reset 跑完整 episode 得到的。** 中途接手不是这回事：
接手时**站点状态已被 cheap arm 改过**，新模型继承的是一段**由失败动作构成的 8 步 history**。
并集是"两个都跑"的上界，不是"第 7 步换人"的可达值。§505.20 自己写明收益来自**跑两遍**，
而不是来自任何检测器（检测器对 B0 失败的 AUROC 只有 0.554）。

### ② 有三条**不受 terminal-label 污染**的证据在反对 rescue，反驳方没有处理

这三条量的是「**给定 cheap arm 以某种方式失败了，rich arm 会不会成功**」——比方差分解更接近 τ：

- 失败原因桶 → rich-arm 成功的 oof-AUROC **0.341–0.567**（dom→som 0.567，vision→som 0.341）；switch-only 0.345–0.542
- **`fail_max_steps` 状态下 dom → rich 成功 0/14**——"跑满 30 步还没成"那类失败，换强臂一个都没救回来
- near-miss 极少：page 侧占失败的 1–6%，answer 侧每臂 4–5 个 ⇒ **"差一步就对"对这个 agent 几乎不存在**

这些不是决定性的（单 cell、n 小），但它们是现存**最相关**的证据，方向是负的。
任何 rescue 提案必须正面回应，不能只靠"方差分解不适用"绕过去。

### ③ `harm rate` 在随机化下**不可按反驳方的定义识别**

反驳方要测「本来 cheap 能成功、却被 intervention 搞坏了多少」。这是**个体反事实**，
随机化给不出来（每个 episode 只能观测一个 arm）。随机化能识别的是**组间 SR 差**。
要按 arm 分解出"救回来的"和"打断的"，需要的恰恰是 v1 §3.11 说做不了的同状态分支。
⇒ 该指标必须改写成可识别的形式：`P(success | triggered, arm)` 的组间差，
外加一个**预注册的非劣门槛**（intervention 臂的 SR 不得低于 continue 臂超过 δ）。

## v2.3 外部新颖性（已用 arXiv API 逐条核实，含 id 与日期）

控制查询先跑通（`id_list=2307.13854` → 200 + WebArena 条目）才下结论。
⚠️ `http://export.arxiv.org` 会回 **301 空响应**，必须用 `https` + `-L`，否则"查无此文"是假的。

| 论文 | arXiv id | 日期 | 它占掉了哪一格 |
|---|---|---|---|
| Monitoring Web Agents Without Internal Signals | 2609.02057v1 | 2026-09-02 | **web agent 的 prefix 风险监测 + key-step 标签**，2 benchmark × 5 backbone，含早期干预与跨站迁移 |
| SWE-Router | 2607.00053v1 | 2026-06-30 | **partial trajectory → 是否升级到强模型**，附 Bayes-optimality 定理（SWE 域，非 web） |
| ProgRouter | 2608.25992v2 | 2026-08-26 | **在线 progress-guided 的逐步模型编排**，quality–cost 目标 |
| BrowseConf | 2510.23458v2 | 2025-10-27 | **web agent 的 verbalized-confidence 驱动 test-time scaling** |
| WebUncertainty | 2604.17821v2 | 2026-04-20 | web agent 双层不确定性驱动的规划与推理 |
| Signal-Driven Observation (SDO) | 2606.06708v2 | 2026-06-04 | **"观测频率不该绑定动作频率；由 URL 变化 / 新可交互元素 / 动作失败等轻量信号触发重新观测"**——但它是 position paper（原文 "we outline the open problems … call on the community"），**没有实验** |
| InferAct（Preemptive Detection and Correction of Misaligned Actions） | 2407.11843v4 | 2024-07-16 | **不可逆动作执行前检测**（buy-now 类） |
| Speculative Macro Commit / IdleSpec / Interactive Speculative Planning | 2609.03236 / 2605.22154 / 2410.00079 | 2026-09 / 2026-05 / 2024-09 | agent 级 speculative execution 与 commit |

**三条后果**：

1. **"trajectory-level routing" 这个词本身不能当 novelty 了**——上面 8 篇里 5 篇直接占位。
2. **反驳方主张的"重做 key-step monitor"，三周前已被 2609.02057 做完**（2 benchmark × 5 backbone + 跨站迁移）。
   我们在 VWA 上重做一遍**是复现，不是贡献**。
3. **剩下的空格是"买哪一种算力"**：上面每篇的 actuator 都是**单一**的——
   SDO 只决定要不要重新观测、SWE-Router 只决定要不要换强模型、BrowseConf 只决定要不要再采样、
   2609.02057 只决定要不要 cut、InferAct 只决定要不要拦这一步。
   **没有一篇在同一个触发点上比较「多看 / 换脑子 / 验证 / 放弃」并给出成本核算。**
   ⇒ 这才是可主张的那一格，而且它比"trajectory-level routing"窄得多、也诚实得多。

**我们相对 2609.02057 的现成差异化**：它的前提是"internal signal 不可用"。
**我们两样都有**（B0/B1/B2 有 logprob；全体有 verbalized），所以能回答它答不了的问题：
**internal signal 相对 observable signal 到底有没有增量**——且**零新 compute**。
（⚠️ B5 没有 logprob，见 §2.3b H1，这条分析必须排除 B5。）

## v2.4 双方都漏掉的一个**免费且决定性**的实验：reactive vs pre-flight 预算分配

在设计任何 fire 之前，有一个实验能**直接回答"H_t 到底有没有超出 task 的决策价值"**，
而且**不烧新 compute、不需要新标签、不需要 branching**。

**为什么能离线精确评估**：§505 digest §22 已确认 "truncation is evaluated exactly (success steps are known)"
——每个 episode 的成功发生在第几步是已知的，所以任意"在第 t 步砍掉"的策略，其成功与否**完全确定**，
逐步累计成本也已抽出（13,537 个 canonical episode）。

| 策略 | 决策信息 | 现状 |
|---|---|---|
| A 固定 cap | 无 | 已测（cap 15 → −3.1pp / −43%） |
| B pre-flight 学习分档 | **仅 task** | 已测（11/11 cell 优于 A） |
| **C reactive 触发式截断** | **H_t（streak / hazard）** | **未测** |
| **D 混合（pre-flight 定档 + reactive 提前砍）** | task + H_t | **未测** |

**判据（动手前写死）**：C 或 D 在等成本下显著优于 B ⇒ **轨迹信息在我们自己的数据上有决策价值**，
rescue 那条线才值得花钱 fire。C ≈ B ⇒ **在唯一能精确评估的 actuator 上，轨迹信息没有超出 task 的增量**——
这比 v1 写的任何一条都是更强的负结论，而且零 compute 就能拿到。

⚠️ 必须用 §2.2 的**修正口径**定义触发（`page_changed` 假阳性跨 mode 不均：som 13.0% vs vision 5.4%），
否则 C 的触发率本身带 mode 偏。

**这个实验应排在所有其他事情前面**，含 key-step 标注。

## v2.5 trigger-time 随机化 rescue 实验（设计 + 老实的 power 账）

反驳方的核心方法学贡献：**不需要 checkpoint/restore**，在**第一次触发**时随机分臂即可，
随机化保证各臂的 triggered state 在期望上可比。**这条我完全接受**——它把 v1 §3.11 的门槛
从"必须建 branching 基础设施"降到"改 config + 一处小代码"。

**实现代价（已核过代码）**：

- **perception 臂**：runner 已完整支持 episode 内换 mode（`router.py:67-223` + `main.py:3036-3075`），
  只需一个 `router_kind: cascade` 变体 + 在 `decide()` 里加随机分配 ⇒ **config 级改动**
- **model 臂**：**目前不支持**——backend 在 episode 循环外按 condition 绑定一次（`main.py:1066`）。
  cache key 已是 `(backend_id, seed)`，把 `_get_backend` 挪进循环是小改动，但**是代码改动**，
  且按 §2.5 硬规则必须走 queue script，并与 `paper_grade` 互斥（以 `sr_excluded=True` 的独立协议跑）
- **站点**：**只能 classifieds**（`require_reset` 只对 cls 实现；reset 5–15 秒 vs reddit 60–160 秒 vs shopping >3769 秒）
- **模型**：建议 **B1（Qwen3-VL-4B）**——有明显 loop/stall（重复动作率 5.6–13.5%、撞 cap 53–80%），
  又不像 B2 那样贴地板（cls SR 6.8% vs B2 1.3%）

**power —— 这才是真正的约束，不是机制**。
触发后的条件 SR 需现算；用 §505 digest §14 的触发率（失败集 57% / 成功集 33%）在 cls·B0 上推算约 **11%**，B1 更低。
两比例检验、α=0.05、power 0.8：

| 要检出的 SR 提升 | 每臂 n | 2 臂总触发 episode | 4 臂总触发 episode |
|---|---:|---:|---:|
| +10pp（11%→21%） | ~215 | ~430 | ~860 |
| +5pp（11%→16%） | ~724 | ~1,448 | ~2,896 |

触发率约 50% ⇒ 2 臂检出 +10pp 需跑 **~860 个 episode ≈ 4 个 condition-run**（cls 224 题/run）；
4 臂要 ~8 个 run；+5pp 完全不现实。⇒ **设计必须按这个账来**：

1. **先 2 臂**（continue vs 一个 rescue 臂），不要一上来 4 臂或 2×2 factorial
2. **主要终点用连续量**（cost-per-success、steps saved），power 远高于二值 SR；
   **SR 作预注册的非劣终点**（见 v2.2 ③）
3. **收紧触发条件来富集**（streak ≥ 3 或 hazard 分数过阈），代价是样本变少——这是要预先定死的权衡

## v2.6 一个会卡住上述计划的**物理**限制

反驳方的第一阶段（key-step 重标注、intent/action/expected-state-change 一致性、语义 transition surprise）
**都需要看到当时的页面**。而 v1 §1.1 实测：**E: 镜像里 `artifacts/` 目录数为 0**——
逐步的 DOM 文本与截图一份都没备份。本机只有结构化的 `state_digest`（url / title / 行数 / 文本长度 / scroll）
+ `text_similarity` + `page_change_reasons` + `dialog_meta`。

⇒ 本机能做的是**降级版**：URL 变化、标题变化、dialog 是否弹出、文本长度与相似度、
是否出现新的可交互元素（只能用 `interactive_elements_changed` 这个 reason 代理）。
**做不了**：目标元素是否消失/出现、点到的元素文本是什么、thought–observation 一致性。

**⇒ 这是前置问题不是细节**：需要确认 A100 上是否还留着 `artifacts/`。
若在，先把 classifieds 相关 condition 的 artifacts 拉回来（这是 I/O 不是 compute）；
若不在，key-step 标注与语义 surprise 只能降级做，或随 rescue fire 一起重新产出。

## v2.7 修订后的判定表

| 方向 | v1 | **v2** | 依据 |
|---|---|---|---|
| static task → mode/model | D | **D**（维持） | 方差分解 + 配对检验 + 0/6 + 0/26 |
| step-0 model routing | D | **D**（维持） | 326 翻面 task 全部 P≈0.5 |
| **trajectory perception/model routing** | D | **未测试** | 改判 1；但 v2.2 ①② 的负面证据必须正面回应 |
| terminal-label prefix 预测 | D | **D**（维持，范围写清） | L2 + §459 + 本次探针 |
| **key-step / local hazard 监测** | D | **未测试，但已被外部做掉** | 改判 2 + 2609.02057 |
| thought TF-IDF | D | **D**（维持） | 29/33 行 ≤ intent |
| **thought 时序动态** | 基本 D | **未测试** | 改判 3；注意 H2 缺失偏差 |
| page-no-change | 弱 | **弱**（维持） | 5–9% 假阳、跨 mode 不均 |
| **语义 transition surprise** | 基本 D | **未测试（本机只能降级做）** | 改判 4 + v2.6 |
| budget routing | 主线 | **主线，但先做 v2.4 的 reactive 对照** | 2/2 前瞻 PASS + §22 精确截断 |
| grounding-path routing | 强候选 | **降为并列候选** | 反驳方说得对：更像 scaffold 论文 |
| irreversible risk | 强候选 | **进 controller 作为一层，不单独成篇** | n=33 + InferAct 已占位 |
| speculative cheap-policy | 已被覆盖 | **判错，是独立文献线** | 改判 5 |

## v2.8 修订后的推荐路线

**Gate 0（零 compute，必须先做）**：v2.4 的 reactive vs pre-flight 预算对照。
这是唯一一个**在我们自己的数据上、用精确评估、不需要新标签**就能回答"H_t 有没有决策价值"的实验。

**Gate 1（零新 compute，若 Gate 0 过）**：
(a) internal vs observable signal 的增量（能答 2609.02057 答不了的问题，排除 B5）；
(b) thought 时序动态 + 结构化 transition surprise 特征；
(c) key-step 标注**降级版**（无观测），只作 Gate 2 的触发器候选，**不作独立贡献**。

**Gate 2（要 fire，若 Gate 1 有信号）**：v2.5 的 2 臂 trigger-time 随机化，cls × B1，
连续量作主要终点，约 4 个 condition-run。

**定位**（避开 5 篇近邻）：不是 "trajectory-level routing"，不是 "failure monitoring"，
而是**同一触发点上的 actuator 选择 + 成本核算**——"值不值得买、以及买哪一种算力"。

> ⚠️ **本节（v2.8 路线图）与 v2.5 的终点选择已被 v3 取代**：
> Gate 0 降级为 spend gate（不再是 kill gate）、新增 SUTVA 处理、终点改 co-primary、
> 第一轮干预限定为纯 perception。以 **v3.7** 为准。

---
---

# v3 修订（2026-09-22，第三轮外部评审的裁决）

> 第三轮意见针对 v2。**五条方法学意见我全部接受**，其中两条（SUTVA、终点设计）v2 确实漏了，
> 且 SUTVA 这条本项目有实测过的违例实例。文献上它补的两篇都属实，而且比 v2 列的那批更近。
> 我另外补了两篇它和我都漏的。
> **v2 的 Gate 路线图作废，以 v3.7 为准。**

## v3.1 文献核实更新（arXiv API 实测，含控制查询）

⚠️ 本轮又踩到一个新坑：Python `urllib` 默认 UA 会被 arXiv 回 **406 Not Acceptable**，
六条查询全部"查无此文"。换 `curl -sL -A <ua>` 后全部 200。
**教训与 301 那条同类：空结果先验证通道，不要当成阴性结论。**（控制查询 = 已知 id 打得通才下结论。）

### 评审补的两篇：都属实，转述也准确

| 论文 | arXiv id | 日期 | 摘要核对 |
|---|---|---|---|
| **AVR** — Adaptive Vision-Language Model Routing for Computer Use Agents | **2603.12823v1** | **2026-03-13** | ✅ 属实。逐 tool call 路由：多模态 embedding 估 action difficulty + 探一个小 VLM 的 confidence → 选"预测精度满足 reliability threshold 的最便宜模型"；形式化为 cost–accuracy 权衡 + 阈值策略；**高风险 action 直接升到最强模型**（配 Visual Confused Deputy guardrail）。报告**推理成本最多降 78%**、距 all-large baseline 2pp 以内 |
| **COTA** — Don't Solve, Just Compare: Tiny Advisors for Runtime Intervention in LLM Agents | **2608.21027v1** | **2026-08-21** | ✅ 属实。小 comparator 判断"采样出的候选延续是否优于 actor 的提议"，**重复比较决定何时干预**；训练监督来自 **same-prefix counterfactual branches**；返回的是**非绑定建议**，由原 actor 重新规划。WebShop / ALFWorld / τ³-Retail × 3 个 actor，9/9 设置改善 |

**评审对这两篇的定位判断我同意，但各补一条它没强调的**：

- **AVR 的评测不是 end-to-end**：它跑的是 **ScreenSpot-Pro grounding 数据 + OpenClaw agent routing benchmark**，
  成本下降是 **projected**。它测的是 action grounding 的模型选择，不是"整条 web 任务的 cost–success 前沿"。
  ⇒ 这正是我们的楔子，而且是站得住的楔子。但 **"action-level 风险升级"这一格它确实占了**，
  v1/v2 里 P3 那条不可逆风险路由的 novelty 要按这个重新降级。
- **COTA 的监督来自 same-prefix counterfactual branches** —— 它**有分支能力**，因为 WebShop / ALFWorld / τ³
  是可廉价分支的模拟/文本环境。我们在 VWA 上做不到（v1 §3.11：无 checkpoint、重跑翻面率 12–14%、
  shopping 重置 >3769 秒）。**这既是我们的限制，也是差异**：真实浏览器 + 有状态站点 + 不可逆动作，
  是 COTA 那套 branch-based 监督**不能直接搬过来**的场景。

### 我这轮额外找到、双方都漏的两篇

| 论文 | arXiv id | 日期 | 为什么相关 |
|---|---|---|---|
| Screenshots or Tools? Eliciting Tool Use and Managing Multimodal Context in Hybrid GUI-MCP Computer-Use Agents | 2608.03327v2 | 2026-08-04 | ⚠️ **直接占了"感知 actuator"的一部分**：同一 harness 下 screenshot vs 文本工具；发现"一次成功的工具调用常让下一张截图变冗余"，丢掉它并把图像历史减半，**input token 降约 1/3**，重训后压缩版 37.8% vs 未压缩 33.0% 且只花 53% input 成本，并在**预注册的 degraded 子集**上把 rich-lean gap 收到 0 |
| TwinRouterBench: Fast Static and Live Dynamic Evaluation for Realistic Agentic LLM Routing | 2605.18859v2 | 2026-05-14 | agentic LLM routing 的评测基准（仅标题层核实，内容 UNVERIFIED） |
| Why Are GUI Agents Correct but Late? Decode on the Decision-Time Critical Path | 2607.28399v1 | 2026-07-30 | GUI agent 的延迟轴（仅标题层核实，内容 UNVERIFIED） |

**2608.03327 对我们的影响最大**：它已经在做"什么时候不需要那张图"，而且是端到端 + 预注册。
它和 SDO（2606.06708，position paper 无实验）合起来，把"按需感知"这一格压得比 v2 写的还紧。
我们剩下的差异只有一条是硬的：**它是按规则/重训做上下文压缩，不是按 runtime 轨迹状态触发的临时升级**。

### 修订后的新颖性判断

v2 §2.3 说"剩下的空格是买哪一种算力"。**这句话要收紧**，因为 AVR 已经同时做了
"逐 action 选模型"+"高风险直接升最强"，COTA 已经做了"何时干预 + 给什么方向"。
可主张的只剩一句更窄的：

> **在同一个 runtime state 上，把 perception / model / verification / termination 当作可互换的 actuator，
> 在 end-to-end 有状态 web 任务上统一核算 cost–success 权衡，并报告它们各自的可达收益。**

评审自己也写了"这只能说这轮检索没找到"。我同意，并加一条风险：
**这个空格的半衰期很短**——6 个月内进来了 AVR / SDO / SWE-Router / ProgRouter / 2608.03327 / 2609.02057 / COTA 七篇。
投稿前必须再做一次系统 prior-art sweep，而且**路线图要按"gap 可能在我们点火前被关掉"来排优先级**。

## v3.2 (a) Gate 0 的定位 —— 接受降级，但保留它作为**决策**门

评审对：reactive truncation 只能估 **V(H_t; STOP)**。
"第 8 步卡住 → stop 没用、continue 没用、Vision 有用" 这个情形完全可能，
此时 Gate 0 阴性而 rescue 阳性。**v2 写的 "C≈B ⇒ rescue 不值得 fire" 收回。**

改写为：

- **Gate 0 是 logical diagnostic，不是 kill gate。** 正确表述：
  *a cheap screening test for whether the realized trajectory carries actionable **stopping** information.*
- **但它仍是决策门**，理由与逻辑无关：STOP 是**唯一**能在现有数据上**精确**评估的 actuator
  （成功发生在第几步已知 ⇒ 截断的反事实完全确定），其余三个 actuator 都要花钱点火。
  所以 Gate 0 阳性 = 有证据支持花钱；Gate 0 阴性 = **不构成反驳，但先验下调**
  （在标签最密、噪声最小、评估最干净的那个 actuator 上都读不出信号，
  对"更稀疏、更贵、更难评估的 actuator 能读出信号"的先验应当下调）。
- ⇒ 路线图里它从 "kill gate" 改为 **"spend gate"**：它决定**要不要掏钱**，不决定**假设真不真**。

## v3.3 (b) baseline 阶梯 —— 接受，并补两条评审没写的必要条件

评审指出的缺陷成立：v2 的 C 若含 step index 而 B 只有 task，C 赢可能只是
**"跑到第 17 步还没结束"本身就是信息**（survival time），而不是行为遥测有价值。
它要看 **D−C** 而不是 **D−B**，这是对的。采纳它的六档，但我的版本加两条：

| 档 | 决策信息 | 作用 |
|---|---|---|
| A | 无（固定 cap） | 成本-SR 前沿的地板 |
| B | 仅 task（pre-flight 分档） | 已测，11/11 cell 优于 A |
| **C** | task + **step index + 已花成本/token** | **survival-time 对照 —— 关键新增** |
| D | C + 结构化轨迹遥测（page-change / action 重复 / streak / text_similarity / error_category） | |
| E | D + confidence（verbalized；logprob 仅 B0/B1/B2） | |
| F | E + thought 时序动态 | |

**主对比 = D−C**（行为遥测的增量）；次要 = E−D（不确定性的增量）、F−E（thought 的增量）。

**我加的两条必要条件**：

1. ⭐ **决策时点必须跨档一致。** 如果 B 在 t=0 决定、C/D 在触发时决定，两者不可比——差的是信息 **和** 时机两样。
   所有档都要实现成"**可在任意步停止的策略**"，B 的停止规则就是"task 条件化的固定 cap"。
   否则 D−C 会把"允许晚决定"的收益算成"遥测的价值"。
2. **每一档都要配同一套 CV 协议 + 自己的 shuffle-null + 等成本对照**，并对着 rerun band 读。
   §459 的教训是多行低于自己的 null；没有逐档 null，阶梯上的单调性会被过拟合伪造出来。

⚠️ 还有一条口径：**C 的"已花成本"在跨 baseline 时不可直接用**（见 v3.6 的单位问题），
Gate 0 若只在单 baseline 内做则无此问题——建议 Gate 0 **逐 cell 内部做，不跨 baseline 池化**。

## v3.4 (c) SUTVA —— 接受，这是 v2 最大的漏洞；并给出实测代价

评审对。本项目有**实测过的违例实例**：§329 的 R21521，task 4 点了 Logout，
**task 5/6/7 成为受害者**（`infra_covariates=['session_lost_preserved']`）。
在随机化设计里这正是 SUTVA 破裂：episode 4 分到 rescue 而避免了 logout，episodes 5–7 的潜在结果就被改写了。

**现状实测（查了代码，不是推测）**：

- **站点重置**：upstream 的 per-task `require_reset` **只对 classifieds 实现**，
  且只覆盖**被标记的那部分任务 —— cls 共 22 次 per-task 全站重置**
  （`p79/experiment/config.py:130`、`p79/utils/shopping_cart_reset.py:62`、preregistration PROTOCOL_NOTE_07）。
  reddit 0 次；shopping 0 次（P79 自己加了 per-task 清购物车）。
- **重置代价**：classifieds = `curl` 触发 + **~12s SQL sentinel** + ~1s cache/session 清理，
  `_lib_paper_grade_gates.sh:720` 自述 "typical reset 5-15s"。
- **认证刷新**：`should_refresh` 默认 **每 5 个 episode 或 1200 秒**触发一次重新登录
  （`p79/utils/auth_refresh.py`，B-35：PHP `session.gc_maxlifetime`=1440s）。

⇒ **污染横向传播的射程 ≈ 5 个 episode**（下一次 auth gate 之前），与实测的 3-episode 污染波吻合；
而**站点状态**的污染射程更长——两次 per-task reset 之间可以横跨几十个 episode。

**我的建议（按优劣排序）**：

1. ⭐ **首选：RCT 专用协议下对 classifieds 强制每 episode 全站重置 + 每 episode 重新认证。**
   代价实测可算：224 × ~13s ≈ **48 分钟/condition-run** 的额外开销（重置）；
   加上每 episode 重新登录大约再翻一倍，量级 **1–1.5 小时/run**。
   相对一次 cls run 的总时长这是可接受的，而且**classifieds 是唯一实现了 reset 的站点**——
   这一条同时也再次证明 Gate 1/2 只能在 cls 上做。
   ⚠️ 代价不在时间而在**估计量**：这会让 RCT 的 SR 与 Phase-1 各 cell **不可直接比较**
   （Phase-1 是共享状态协议）。因为 RCT 本来就要以 `sr_excluded=True` 的独立协议跑，这个代价可以接受，
   但**必须在预注册里写明"本实验的 SR 不与 Phase-1 数字并排"**。
2. **次选（不强制重置时的最低要求，三条缺一不可）**：
   (i) **block 级随机化**（以两次 reset / auth gate 之间的 episode 段为 block，整块同臂）；
   (ii) **预注册排除受害 episode**（`infra_covariates` 含 session-lost 的），排除规则写在看数据之前；
   (iii) **把 spillover 本身作为结局报告**——逐臂的 session-lost 事件率。
   这一条我们**已经有现成仪器**（`trajectory_events.jsonl` + `infra_covariates`），零额外开发。
3. ⛔ **不可接受**：把 224 个 task 当 i.i.d. Bernoulli 直接做 episode 级随机化 —— 这正是 v2 写的东西。

**顺带一个对我们有利的观察**：方案 2(iii) 让"便宜模型会毁状态"这件事从一条**披露**
升级成**随机实验里的一个结局变量**——如果 rescue 臂的 session-lost 率显著低于 continue 臂，
那 v1 P3 那条"不可逆风险未被定价"就有了因果证据，而不只是 33 例观察。

## v3.5 (d) 终点设计 —— 基本接受，但改成 co-primary

评审的两条批评都对：
`cost-per-success = ΣCost/ΣSuccess` 是**比值统计量**，在 SR≈5% 时分母只有几个 task，极不稳定；
`steps saved` 会奖励"第 7 步全 abort"这种糟糕策略。v2 写"主要终点用连续量"是错的。

但只用 ΔSR 作主终点也不行——v2 §2.5 的 power 账摆在那里：
11% 基线下检出 +10pp 每臂要 ~215 个触发 episode，检出 +5pp 要 ~724。
ΔSR 作唯一主终点 = 这个实验只对大效应有话说。

⇒ **我的版本：co-primary，两个都要过**

- **Co-primary 1（efficacy，非劣）**：`ΔSR_triggered = P(success|rescue,triggered) − P(success|continue,triggered)`，
  **预注册非劣边界 δ**（建议 δ 取 rerun band 的下沿，cls 有实测 band 可用；band 缺失的 cell 不做主终点）。
- **Co-primary 2（cost，优效）**：**每 episode 的成本**（不是比值），单臂分布可直接做秩检验，power 远高于二值 SR。
- **次级**：tokens、latency、剩余步数、session-lost 事件率（= v3.4 的 spillover 结局）。
- **主图**：triggered 子集上的 cost–success Pareto，两臂各带 bootstrap CI + rerun band 阴影。
- **若要 scalar**：`U_i = Success_i − λ·Cost_i`，**λ 必须与 freeze 文件一起预注册**，
  并且**同时报告一条 λ 敏感性曲线**（本项目有先例：budget router 的 freeze + tag 模式，§505.28）。

一条评审没提但必须写死的：**非劣边界和 λ 都要在看到任何 triggered 数据之前 commit + git tag**，
照 `budget-router-prospective-20260909` 的做法。

## v3.6 (e) 第一轮只做 perception —— 接受，而且我有一条比"混杂太多"更硬的理由

评审的理由是"model switch 同时引入 backend / capability / prompt / confidence schema / history / 成本差异"。
方向对，但"混杂"在**随机化实验里不影响效度**——`switch to B5` 是一个合法的复合处理，
随机化照样能识别它的平均因果效应。它影响的是**机制可解释性**（分不清是"看得不够"还是"想得不够"）。

**更硬的理由有两条，都是实现层的**：

1. ⭐ **成本终点会被单位污染。** B1 是 `electricity_usd_derived`，B0/B5 是 `api_usd`，
   两者尺度差约 1000×（`types.py:241-249`、`metrics.py:868`）。
   一个 B1→B5 的 rescue 臂，其 episode 成本是**两种单位相加**——
   v3.5 的 co-primary 2 直接失效，除非另立一套可比的成本口径。
   **纯 perception 臂（B1 内部换 mode）两臂单位相同，成本终点干净。**
2. **mid-episode 换模型目前根本没实现**：backend 在 episode 循环外按 condition 绑定一次
   （`main.py:1066`，cache key 已是 `(backend_id, seed)`）。换 mode 是 config 级，换模型是代码改动。

⇒ **第一轮干预 = B1 + DOM，触发时随机分到 {continue-DOM, 一次性 Vision/SoM}，模型完全不变。**
这一臂还有一个论证上的好处：它**直接检验"loop → 开 vision"这个最朴素的直觉**，
阴性也是一个干净、可发表的否定。

**但我要加一条评审和 v2 都没注意到的实现风险**：
**mode 之间的 element id 空间不同。** SoM 系走 CDP `getFullAXTree` nodeId 并会 **re-key**
（`p79/experiment/som.py:240` 注释即 "re-key the canonical"；§294 记过 nodeId 乱序不连续）。
"看一步 SoM 然后退回 DOM"会让 agent 在下一步 DOM 里**引用上一步 SoM 的 id**，而那套 id 已经失效。
两个可选缓解，必须二选一并写进预注册：
(i) rescue 臂 **至少两步**（看 + 在同一 id 空间内动作），再退回；
(ii) 升级后**保持该 mode 直到下一次页面变化**。
⛔ 不要用"严格一步"的版本——它会把 id 失配造成的失败算成"Vision 没用"。

## v3.7 修订后的 Gate 路线图（取代 v2 §2.8）

| Gate | 内容 | 代价 | 通过判据 |
|---|---|---|---|
| **−1** | 确认 A100 上 `artifacts/`（逐步 DOM + 截图）还在不在，在就拉回 | I/O，无 compute（**正在另一进程执行，等其报告**） | 有 artifacts ⇒ Gate 1 的特征集可用完整语义版；没有 ⇒ 全程降级版（见 v2 §2.6） |
| **0** | reactive vs pre-flight 截断，**六档阶梯 A–F**，逐 cell 内做，主对比 **D−C** | 零 compute | **spend gate**：D−C 在等成本下显著为正 ⇒ 有证据支持掏钱点火。阴性**不否定 rescue**，只下调先验 |
| **1** | **纯 perception rescue RCT**：cls × B1 × DOM，首次触发随机化 {continue, Vision/SoM ≥2 步}，每 episode 重置 + 重认证 | ~4 个 condition-run + 每 run 约 1–1.5h 重置开销 | co-primary：ΔSR 非劣（预注册 δ）+ 每-episode 成本优效 |
| **2** | **model rescue**：同感知，B1 → B5/B0 一次性升级 | Gate 1 的代价 + 代码改动（`_get_backend` 移进循环）+ **需先解决成本单位问题** | 同上 |
| **3** | **multi-actuator 选择**：同一触发点比较 {continue, perceive, reason, verify, stop} | 最大 | 报告四个 actuator 各自的可达收益与 cost–success 前沿 |

**论文定位**（按 v3.1 收紧后）：不是 trajectory routing，不是 failure monitoring，不是 "何时干预"——
而是 **同一 runtime state 上 actuator 之间的可替代性与各自的性价比**，在 end-to-end 有状态 web 任务上核算。

## v3.8 我对评审总裁决的回应

它说 v1 认 60%、v2 认 85%。我这边的对称陈述：
**v3 相对 v2 改动的全部是方法学，研究方向本身没有再变。** 三条修正（Gate 0 降级、SUTVA、终点）
都使实验更难、更贵、更慢，但都使结论更可信。我接受这个交换。

**我仍然坚持的一条**（v2 §2.2 ② 未被本轮回应）：
存在三条**不受 terminal-label 污染**的现存证据在反对 rescue ——
失败桶→rich 成功 AUROC 0.341–0.567、`fail_max_steps` 下 dom→rich **0/14**、near-miss 占失败仅 1–6%。
它们量的正是 τ 的一个可观测投影。Gate 1 的预注册里**必须把它们写成"事前的悲观先验"**，
并预先说明：如果 Gate 1 阴性，结论是"**在这个 agent 能力区间、这个触发定义下** perception rescue 无效"，
而不是把它写成又一个"我们没做出来"。
