# WA reddit — benchmark 级发现（跨 model 跨 mode）

*生成 2026-08-03（B0 六 mode /diag 的 Tier-2/3 产物；全部结论已做 0-token 全量复核）*

> 本文件收 **不属于任何单一 model / mode** 的结构性问题 —— 站点态污染、evaluator 缺陷、
> 框架 bug。12 个 condition（B0×6 + B1×6，各 104 episode）全部纳入复核范围。
>
> 单 condition 的 per-rule 分布与 episode 明细见 `B0_<mode>_wa_reddit_diag_digest.md`；
> B1 的 cell 级发现见 [[_cell_cross_mode_findings]]（本文件**不复述**它已坐实的
> F1 `select_option` / F2 `walk_fail` / F3 task 66 / F4 artifact 落盘四条）。
>
> ⚠️ **这是 WebArena，不是 VisualWebArena。** 任务集不同（WA 104 scored / VWA 205 collected），
> WA reddit **0/104** 任务带 image、**104/104** 单站。
>
> ℹ️ 数据源 `results/diag_scans/v9_wa/`（`RULESET_VERSION = 9-wa-p47p48`，12/12 condition
> 同版本，`config_missing=0`）。

---

## 0. 一句话结论

WA reddit 的 104 个任务里，**50 个是发帖类**（48%）。这 50 个被一个站点级限流机制
系统性压制，而且**压制强度跨 mode 极不均匀**（3/50 到 19/50）。另有 **6 个任务因
evaluator 的单词 token 精确匹配被判假**、**1 个任务结构性不可通过**。

按"重跑能否改变结果"分类：

| 类别 | 条目 | 重跑有意义吗 |
|---|---|---|
| **站点态污染（run 内）** | B1 发帖限流 · B3 sidebar 订阅累积 | ⚠️ 需 **run 内** cooldown / 周期 reset；per-condition reset **已生效**，挡不住 |
| **上游 evaluator/config 缺陷** | B2 tokenize 假阴性 · B4 task 646 大小写 | ❌ 否 —— 确定性判错，跑一万遍一样 |
| **P79 框架 bug** | B5 `_format_history` 丢 thought | ⚠️ 修了会改变 agent 行为 → 属能力改进而非污染修复 |
| **diag 规则 bug** | B6 P40 在 reddit 空转（B1 已记为 A6） | ❌ 否 —— 改规则重扫，0 token |
| **模型行为，非 bug** | B7 字面 `\n` 16 例 | ❌ 否 |

---

## B1. Postmill 发帖限流 —— 93 episode，跨 mode 不均

### 证据

站点原文横幅 `"You cannot post more. Wait a while before trying again."`

**必须区分横幅来源**（第一版粗扫把两者混为一谈，是本轮方法学教训）：

| 口径 | 判据 | 计数 |
|---|---|---|
| **真限流** | observation 侧（非 `action` 字段）出现站点原文 | **93 ep** |
| **幻觉限流** | 模型 thought/answer 自述限流，但页面**从未**出现站点原文 | **16 ep** |

```
                 真限流   幻觉限流(task)
B0 dom             19     4  [607, 612, 613, 641]
B0 som             16     1  [634]
B0 vision           3     0
B0 phantom_text     8     1  [639]
B0 phantom_prompt  18     5  [606, 607, 630, 632, 633]
B0 phantom_som     14     3  [631, 633, 640]
B1 dom              1     0
B1 som             10     0
B1 vision           1     2  [604, 716]
B1 phantom_text     2     0
B1 phantom_prompt   1     0
B1 phantom_som      0     0
```

幻觉限流 **14/16 集中在 B0** —— 更强的模型更会为自己的失败编一个合理外因。
归因时这 16 个是 **agent-limit**（幻觉式放弃），不是 scaffold。

### 时序结构：滑动窗口，不是配额永久耗尽

按执行顺序把 50 个发帖任务切五段：

```
B0 dom            第1段 3/10 · 第2段 2/10 · 第3段 5/10 · 第4段 4/10 · 第5段 5/10
B0 phantom_prompt 第1段 3/10 · 第2段 5/10 · 第3段 5/10 · 第4段 1/10 · 第5段 4/10
首次限流出现在第 6–7 个发帖任务（run 开始约 20 分钟）
```

命中率在 20–50% 之间波动、**不单调上升** → 是"发几个→触发冷却→恢复→再触发"的滑动窗口，
不是一次性耗尽后锁死。**加 per-post cooldown 即可缓解，不必换账号。**

### 危害：不是降低 SR，而是降得不均

| | 发帖类 SR | 被限流 | 非发帖 SR |
|---|---:|---:|---:|
| B0 dom | 8.0% | 19/50 (38%) | 44.4% |
| B0 som | 14.0% | 16/50 (32%) | 29.6% |
| B0 vision | 8.0% | 3/50 (6%) | 29.6% |
| B0 phantom_text | 20.0% | 8/50 (16%) | 50.0% |
| B0 phantom_prompt | 8.0% | 18/50 (36%) | 42.6% |
| B0 phantom_som | 10.0% | 14/50 (28%) | 38.9% |
| B1 dom | 4.0% | 1/50 | 27.8% |
| B1 som | 6.0% | 10/50 | 20.4% |
| B1 vision | 0.0% | 1/50 | 18.5% |
| B1 phantom_text | 8.0% | 2/50 | 24.1% |
| B1 phantom_prompt | 8.0% | 1/50 | 24.1% |
| B1 phantom_som | 2.0% | 0/50 | 20.4% |

**dom 和 vision 的发帖类 SR 都是 8.0%，但成因相反**：dom 走到了提交那步被拦（38%），
vision 根本走不到（6%）。cross-mode 比较会把这个差异误读成表征能力差异。

⚠️ **被限流数本身是"能力"的代理变量** —— 越能干的 condition 越容易走到提交、越容易撞限流。
所以不能简单地"把被限流的判成功"，那会反向奖励弱 condition。

### 根因

reddit `require_reset=false`（项目既有 quirk，见 memory `reference_vwa_design_quirks`）
+ 50 个发帖任务在同一账号 `MarvelsGrantMan136` 上连续排队，任务间无 cooldown。

### 处置建议

- **若重跑**：只需重跑发帖类 50 个（另 54 个干净），前置条件 = 任务间 cooldown + 每 condition 前 reset
- **若不重跑**：§8 disclose，并以**非发帖子集（54 task）**作为干净对照报告 —— 该子集 B0 29.6–50.0% / B1 18.5–27.8%，跨 mode 排序仍成立

---

## B2. Evaluator 单词 token 精确匹配 —— 确诊 6 task / 74 (task×condition)

### 机制（代码级）

`external/visualwebarena/evaluation_harness/evaluators.py:166-176`：

```python
def must_include(ref: str, pred: str) -> float:
    clean_ref  = StringEvaluator.clean_answer(ref)
    clean_pred = StringEvaluator.clean_answer(pred)
    if len(word_tokenize(clean_ref)) == 1:
        tok_pred = word_tokenize(clean_pred)
        return float(clean_ref in tok_pred)      # ← 精确 token 成员
    else:
        return float(clean_ref in clean_pred)    # ← 子串
```

**单 token 参考词走精确 token 成员判定**，agent 写出任何屈折/复合形式都判 0。

> ⚠️ **更正 [[_cell_cross_mode_findings]] §3**：那里把 `must_include` 描述为"纯子串匹配
> （`clean_ref in clean_pred`）"，只覆盖了 else 分支。P40 族参考答案 `"0"` 恰恰是单 token，
> 走的是 if 分支。结论（该族送分）不受影响，但**机制描述需更正** —— 而正是这个误解让 B1
> 那轮不会去查"为什么 `cheating` 不含 `cheat`"，因而漏掉了本节的 6 个 task。

### 确诊结果

判据：agent 提交文本中 W 不作为独立 token 出现，但存在以 W 为前缀的更长词。

| task | must_include | agent 实际写的 | 受害 condition |
|---|---|---|---|
| 620 | `long` + `relation` | "long-distance relationship" | **12/12** |
| 630 | `headphone` | "headphones" | **12/12** |
| 635 | `headphone` | "headphones" | 11/12 |
| 624 | `break` | "break-up" | 11/12 |
| 638 | `product` (+`life`) | "products" | 8/12 |
| 621 | `cheat` | "cheating" | 7/12 |

**8 个 (task, word) 对 · 6 个 task · 74 个 (task×condition) episode。**
四个 Tier-2 sub-agent 在不同 mode 上各自独立撞到同一机制（621/624/620/632）。

全量风险面：104 个 task 里 **27 个**含单 token `must_include` 项（含 task 599 `MachineLearning`
—— 它 7/12 成功，说明单 token 本身不是问题，**只有当 agent 写出变体时**才触发）。

### 影响上界

以 B0 dom 为例，受害 6 task。若全部改判成功，SR 从 26.92% → 约 32.7%（+5.8pp）。
⚠️ **这是上界** —— 确诊的只是"该词以变体形式存在"，不保证同 task 其余 must_include 项也满足。

### 处置

重跑无意义（确定性判错）。应做**预注册排除**或修 evaluator（比照 task 160 / task 58 的既往处理）。

### ⭐ VWA 主线对照 —— 基本不受影响（阴性对照，2026-08-03 全量核）

paper §1 的数据源是 **VWA** cls + reddit，不是 WA。既然 `evaluators.py` 从未被 P79 审计触及
（见下），必须核 VWA 主线是否同样中招。**36 个 canonical condition 全扫**（`results/diag_scans/v9_vwa/`
的 B0/B1/B2 × 6 mode × 2 site，定位失败 0）：

| 检查项 | WA reddit | **VWA cls** | **VWA reddit** |
|---|---|---|---|
| reference_url 路径段大小写不一致 | **2**（646 不可通过 / 625 有 OR 逃生） | **0** | **0** |
| 单 token `must_include` 风险面 | 27 task | 15 task | 40 task |
| —— 其中 `program_html`（比对页面长文本） | 多数 | 少数 | 少数 |
| **tokenize 假阴性确诊** | **6 task / 74 ep** | **0** | **2 task / 8 ep** |

VWA 确诊的 2 例逐个核过，**都不是决定性败因**：

- **reddit task 29**（`must_include: cat`，7 condition）：agent 打的全部文本是 `'aww aww cats aww aww'`
  —— 写了 `cats` 没写 `cat` 确实触发假阴性，但它**根本没写 intent 要求的故事**
- **reddit task 208**（`must_include: monitor`，1 condition）：agent 写 `monitors` 触发假阴性，
  但同一 `must_include` 还要求两个 SKU，agent 写的是 `"SKUs: 13 and 1"` —— **SKU 也不匹配**，
  即使 `monitor` 判过也仍然失败

**结论**：`must_include` 单 token 缺陷在 VWA 主线上**风险面存在但实际影响 ≈ 0**。
机制原因 —— VWA 的单 token `must_include` 绝大多数走 `string_match`（比对 agent 的**短答案**，
"red" 不会写成 "reddish"），而 WA 那 6 个受害 task 全部走 `program_html`（比对 agent
**写进页面的自然语言长文本**，"cheat" 必然写成 "cheating"）。

> ⚠️ **一般教训**：确诊"该词以变体形式出现"**不等于**"该 episode 应判成功" —— 同一 `must_include`
> 里的其他项可能也不满足。本节两例都是如此。§B2 给 WA 报的 +5.8pp 同样是**上界**。

---

## B3. sidebar 订阅累积 —— **不是新发现**，是已裁定 §402.5/§402.7 在 WA 侧的复现

> ⚠️ **本节经两次自我更正**。初稿写成"跨 condition 状态残留"是**错的**，见下方"证伪过程"。

### 结论

WA reddit 的 **task 595–599** 与 VWA reddit 那 7 个 sidebar 任务是**同一族**，eval 结构完全同构：

```
url      = http://localhost:9999/                                  ← 首页
locator  = document.querySelector("#sidebar > section").outerText  ← 同一个 sidebar 选择器
required = must_include: [forum 名]
```

sidebar 渲染的是**账号的订阅列表**，所以只要**本 run 内更早的某个 episode** 订阅过该 forum，
后续这个 task 就自动通过 —— 判定由**执行顺序**决定而非 agent 能力。

实证支持：同模板的 task 597（consoles）拿 **11/12**，task 595（space）只有 **6/12**。
同难度同模板的两个任务成功率差这么多，差异来自"该 forum 有没有被更早 episode 订阅过"。

**这正是 §402.5 [MEASURED] 已实证、§402.7 [ADJUDICATED] 2026-07-29 已裁定的问题**：
> reddit sidebar 泄漏归入独立 benchmark bug paper，主 paper 只做一句披露 + 指针；
> 不扩展 AMENDMENT_08，scored universe 保持 203。

WA 侧无需重新决策，**直接适用同一裁定**。本轮的增量只有一个：该缺陷在 WA reddit 上表现为
**5 个 task（595–599）**，VWA 是 7 个。

### 证伪过程（两次自我更正，记录以免重犯）

**更正 1 — 不是"跨 condition 残留"，reset 是生效的。**
初稿断言后跑的 condition 继承先跑 condition 的状态。用"首次限流位置"反推：

```
B0:  dom 7 → som 8 → vision 7 → ptext 4 → pprompt 6 → psom 4     (第 N 个发帖任务首次撞限流)
B1:  dom 8 → som 3 → vision 22 → ptext 6 → pprompt 30 → psom None
```

若配额跨 condition 继承，最后跑的应撞得最狠甚至第 1 个就撞；而 **B1 最后一个 condition
完全没撞（0 次）**。→ **每个 condition 前确实 reset 了**（`wa_reset_supported` 对
WA reddit 返回 0，路由到 `_reset_vwa_local_reddit` = docker rm+run）。
污染是 **run 内**（同一 condition 的 104 个 episode 之间），reset 挡不住——与 §402.5 一致。

**更正 2 — 粗正则把 thought 当成了动作。**
初稿引用 sub-agent 结论"vision task 597 从未点过 Subscribe"，我用
`re.search("subscribe", json.dumps(action))` 复核，得到 12/12 都"有 Subscribe 动作"，
据此判 sub-agent 错。**按字段拆开后**：`subscribe` 只出现在 **`thought`** 字段（模型在说
"我要去订阅"），30 步的实际动作全是 `click coord` + `back`——**sub-agent 是对的**。

> 这与 §B1 自己总结的"区分模型说的 vs 环境给的要按字段拆"是同一个教训，**在同一轮里犯了两次**。
> 凡是在 `json.dumps(record)` 上跑正则得出的结论，一律不可信。

---

## B4. task 646 结构性不可通过（大小写）

`EvaluatorComb.__call__` 是 `score *= cur_score`（evaluators.py:618-621，AND 语义），
`URLExactEvaluator` 的 `GOLD in PRED` 是 `ref in pred`（evaluators.py:332-334，**大小写敏感子串**，
`clean_url` 不做 casefold）。

全量 config 审计（104 task）发现 2 个 forum 大小写不一致：

| forum | 少数派 | 多数派 | 站点实测渲染 |
|---|---|---|---|
| `diy` | task **646** 用 `/f/diy` | task 636 用 `/f/DIY` | `DIY` **330 次**，小写 **0 次** |
| `machinelearning` | task **625** 用 `/f/machinelearning` | task 599/604/609 用 `/f/MachineLearning` | `MachineLearning` **668 次**，小写 **0 次** |

- **task 646 → 不可通过**：ref 只有 `/f/diy` 一个候选，url_match 恒 0 → 乘积恒 0
- **task 625 → 可通过**：ref 是 `/f/machinelearning |OR| /f/deeplearning`，而 `deeplearning`
  站点本来就是小写（实测 **485 次**）→ 第二分支可匹配

> ⚠️ **"0/12 全灭"不能证明"不可通过"**：task 636 大小写完全正确却也是 0/12（败于发帖难度）。
> 证伪结构性断言必须走机制路径（evaluator 源码 + 站点实测渲染），不能靠战绩推断。

与 B1 已记录的 F3（task 66 硬编码 `www.reddit.com`）合并，**WA reddit 已知结构性不可通过任务 = 2 个：66, 646**。

---

## B5. `_format_history` 丢弃 thought —— 跨全部 baseline 的框架 bug

`p79/agents/proxy_api_agent.py:521-552`（B0 用）与 `p79/agents/_shared_vl_utils.py:351-384`
（B1+B2 共用）在拼装 step 历史时只输出：

```
  Step N: {action_type}{detail} -> {result} [{url}]
```

**从不读取 `action['thought']`**（代码已逐行核对，两个版本一致）。

**后果**：需要跨多个页面收集信息再汇总的任务（`intent_template_id=17`，"Among the top N
post... show me..."），模型在 step 8 明确写下"我记下 The Hobbit"，step 9 起该句永久不可见 ——
finish 时只能凭当前观测重拼答案。vision task 67 / 68 两个独立 episode 完全相同的败因：
中途确认过的实体在 finish answer 里消失。

**这不是 8 步窗口截断**（两例的丢失都发生在窗口内），是历史格式化函数本身丢弃了推理文本。

**处置**：修复（给 history 加 thought 摘要或跨 step scratchpad）会改变 agent 行为 →
所有数据需重跑。属于**能力改进**而非污染修复，是否做由 paper scope 决定。

---

## B6. P40 在 reddit 上结构性空转（B1 已记为 A6，本轮独立复现）

`check_p40` 的"访问过详情页"豁免 marker 是 `("page=item", "product_id=", "/product/")`
（`diag_pattern_match.py:1591`）—— 纯 shopping/classifieds URL 惯例，Postmill 上永不出现，
豁免分支在 reddit **永远不触发**。

本轮 B0 success 侧 9 条 P40 命中里 **8 条是误报**（agent 确实走到 `/user/<name>/comments`
逐条读了真实票数）。唯一 1 条真信号（phantom_prompt task 30）还是"蒙对了但理由不对"——
agent 走错版块（进 `/f/memes` 而非 `/f/space`），在错误用户的空评论区答 "0"，
靠该任务模板 ground truth 高度退化（答案近乎恒为 "0"）兜底。

**建议收窄**：reddit 专属 marker `r"/user/[^/]+(/comments)?$"`，并额外核对访问过的
`/f/<forum>/` slug 是否匹配 `instantiation_dict` 里的 forum（否则"蒙对错误用户"仍漏网）。

---

## B7. 字面 `\n` 16 例 —— 模型行为，不是 scaffold bug

sub-agent 曾主张 task 652 的 `type` text 字节 `[3f,3f,3f,5c,6e]`（字面 backslash+n 而非
真换行 `0x0a`）是 tool-call 参数双重转义损坏，可能广泛污染 exact_match 任务。

**全量扫描推翻**：

```
type action 总数            = 6495
含真换行 0x0a               = 3521 (54.2%)
含字面 backslash+n (0x5c6e) =   16 (0.25%)
```

若是 scaffold 双重转义，应当系统性发生（接近 100% 的换行都坏）。实测 3521 条正确
vs 16 条字面 → **转义路径本身正常**，那 16 条是模型自己输出了字面 `\n`。
其中落在 exact_match task 上的仅 4 条（全是 task 652）。归 **agent-limit**。

---

## B8. 为什么既有 benchmark 修复没拦住 B2/B4 —— 审计范围的盲区

VWA submodule 在 `p79-patches` 分支上确实带着一批 P79 修复跑（用户 2026-08-03 提出的核查点）。
逐 commit 核过覆盖范围：

| P79 eval 层 commit | 触及文件 |
|---|---|
| `f0c835b` B-91 空预测守卫（llm_fuzzy_match / llm_ua_match） | `evaluation_harness/helper_functions.py` |
| `eb5cbd8` A1.18 15-finding 3-AI audit sweep | `evaluation_harness/helper_functions.py` |
| `1c3a615` A1.25 GRL Chunk 4（B-535/538/539/540） | `evaluation_harness/helper_functions.py` |
| `2f9b0b4` A1.18-re Chunk 1 11-fix substrate sweep | `evaluation_harness/helper_functions.py` |
| `3f9ceca` runtime patches | `evaluation_harness/helper_functions.py` |

**P79 一次都没有修改过 `evaluation_harness/evaluators.py`。**
`git log -- evaluation_harness/evaluators.py` 的全部 6 个 commit 都来自上游 VWA：

- `54139d9` Initial commit —— **单 token 精确匹配分支从这里就在，此后从未被改动**
- `7301362` / `229fc7b` / `2021fe6` —— 三次 `must_include` 修复，**修的全是 `|OR|` 逻辑**
- `a8a1648` —— `clean_url` 加了 `localhost → 127.0.0.1` 替换和去尾斜杠，**没有 casefold**
- `ff176e4` —— CDP session，无关

分工是清楚的：`helper_functions.py` 装的是 **LLM judge 层**（`llm_fuzzy_match` / `llm_ua_match` /
`reddit_get_post_url`），`evaluators.py` 才是**核心评分函数**（`StringEvaluator.must_include` /
`URLExactEvaluator`）。P79 历次审计（含 3-AI cross-audit）全部落在前者，**后者是覆盖空洞**。

`a8a1648` 尤其说明问题：上游**已经意识到** URL 归一化会影响 eval 一致性，但只处理了 host
（localhost→127.0.0.1），没处理 path 大小写 —— B4 正好卡在这个半成品上。

**教训**：审计覆盖率要按**文件/函数**核，不能按"我审过 eval 层"这种粒度自我确认。
一个模块里最核心的评分函数，可能恰恰因为"看起来是上游的、稳定的、不该动的"而从未被读过。

---

## 8. 可 actionable 项

| # | 事项 | 类型 | 优先级 | B-number 候选 |
|---|---|---|---|---|
| C1 | reddit 发帖任务间加 cooldown + 每 condition 前 reset（B1） | infra / 实验设置 | **P0** | 待开 |
| C2 | `must_include` 单 token 精确匹配假阴性 → 6 task 预注册排除或修 evaluator（B2） | benchmark-FP | **P0** | 待开 |
| C3 | task 646 从计分集剔除（B4）；与 B1 的 A5（task 66）合并处理 | benchmark-FP | P1 | 与 A5 合并 |
| C4 | 更正 [[_cell_cross_mode_findings]] §3 对 `must_include` 的机制描述（B2） | 文档 | P1 | — |
| C5 | `_format_history` 加 thought 摘要（B5）—— **需先定 paper scope，修了要全量重跑** | scaffold | P1（决策待定） | 待开 |
| C6 | ~~sidebar 订阅泄漏~~ **已裁定** —— §402.7（2026-07-29 user）归独立 benchmark bug paper，主 paper 一句披露；WA 侧 task 595–599 直接适用，**无需重新决策**（B3） | benchmark-FP | ✅ 已裁定 | — |
| C7 | P40 加 reddit 专属 detail marker（B6）—— 与 B1 的 A6 同条 | 规则 | P1 | 与 A6 合并 |
| C8 | 落新规则 P49–P51（见各 digest §Self-evolving），bump RULESET_VERSION，全量重扫 | 规则 | P2 | — |

---

## 9. 方法学教训（写进流程）

1. **"0/N 全灭"不是"不可通过"的证据** —— task 636 大小写正确却也 0/12。证伪结构性断言必须走
   机制路径。本轮推翻两个 sub-agent 的一致主张（task 409 硬编码 ID 不可通过），靠的是
   **同构反例 task 410（4/12 成功）** —— 同构反例比再多失败样本都有力。
2. **区分"模型说的"和"环境给的"要按字段拆** —— 第一版限流扫描在整行 JSONL 上跑正则，
   thought 里的复述与页面横幅无法区分，正好被"这是幻觉"的反主张击中。改成 `action` 字段
   vs 其余字段分离后才拿到 93/16 这个干净答案。**归因分析里，信息来源比信息内容更能定性。**
3. **规则库跨 benchmark 迁移会静默失效** —— P40 在 shopping/classifieds 上判据合理，迁到
   reddit 后 marker 永假 → 退化成"只要答案是数字就 fire"。不报错、不崩溃，只是悄悄产出误报。
4. **sub-agent 的机制断言必须 0-token 全量复核** —— 本轮 3 条被推翻（409 结构不可通过 ×2、
   652 转义损坏、607 幻觉/真限流之争的粗判），2 条被证实（`_format_history`、真限流）。
   延续 v8 轮的教训：**"没有 X"和"这是系统性的 X"都是最不可靠的两类断言。**
