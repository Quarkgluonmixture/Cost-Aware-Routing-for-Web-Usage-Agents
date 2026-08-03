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

### 证据（⚠️ 本节证据链在 2026-08-03 验证轮被**整体推翻并重建**，见 §B1-R）

**① 站点侧机制 —— 源码坐实（硬证据）**

A100 `vwa-reddit` 容器内 Postmill 源码：

```php
// /var/www/html/src/DataObject/SubmissionData.php:18-19
@RateLimit(period="5 minutes", max=15, groups={"create"},                    entityClass=Submission::class)
@RateLimit(period="1 hour",    max=3,  groups={"unwhitelisted_user_create"}, entityClass=Submission::class)

// var/cache/dev/translations/catalogue.*.php
'ratelimit.error' => 'You cannot post more. Wait a while before trying again.'
```

实现在 `src/Validator/RateLimit{,Validator}.php`；`CommentData.php` 带同类注解。
**「未白名单用户 1 小时 3 帖」**精确解释了实测时序：首次触发在第 4-8 个发帖任务
（约 20 分钟内提交 3-4 次）、之后稳定 ~40%、各段 20-50% 波动不单调上升。

**② episode 侧观测 —— 109 个 episode 出现限流自述**

| 口径 | 判据 | 计数 |
|---|---|---|
| 逐字复现站点文案 | thought/answer 含 `cannot post more` / `wait a while before trying again` | **93 ep** |
| 宽松措辞 | 含 `posting limit` / `rate limit` / `posting restriction` 等 | **+16 ep** |

（分 mode 计数见下方表格的"被限流"列，口径 = 逐字复现那 93 个。）

**③ 无法做的事：逐 episode 核验**

`steps_v2.jsonl` 的 step record **不含任何页面文本字段** —— 只有 `obs_url` 和
`state_digest.{text_length, title, dom_complexity, ...}`。页面文本在 `artifacts/`，
而这批 run 的 artifacts 已被清空。`text_length` 波动做间接检验**无区分度**
（自述组中位数 Δ=922 vs 未自述组 1327，且量级 ~1000 chars 远大于一句横幅约 60 chars）。

**综合判定**：站点机制存在 = 硬证据；93 个 episode 逐字复现 `ratelimit.error` 原文 = 强间接
证据（Postmill 专用翻译键，模型凭空一字不差生成的可能性低）；但**单个 episode 是否真撞了限流
无法逐条判定**，存在替代解释（必填字段缺失触发表单校验 —— 盲复检对 task 631/609 给出过该解释）。
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

## B1-R. 证伪与重建 —— 本条结论对了，但原来的证据链是错的

**这是本轮最值得记住的一段。** 初稿给 B1 的核心证据是"93/93 的横幅命中都在 observation 侧，
0 个只在 thought 里"，据此判定限流为真、并推翻了一个 sub-agent 的"幻觉"主张。

**R1 — 那个 93/0 是扫描 bug 造出来的。** 判据写的是
`{k: v for k, v in record.items() if k != "action"}`，**只排除了 `action`，没排除 `raw_action`**
（后者是 action 的原始副本，同样含 `thought`/`answer`）。修正后重扫：

```
真 observation = 0  ·  仅模型自述 = 93        ← 与初稿完全相反
```

**R2 — 但"0 个 observation"同样不能证明页面上没有横幅。** 进一步查 step record 的字段结构后
发现：**它根本不存页面文本**（见上方 §③）。在一个不含页面文本的记录里搜页面文本，两个方向的
结论都无效 —— 初稿的 93 是假阳性，修正后的 0 是假阴性。**方法本身不适用于这个问题。**

**R3 — 换证据源才解决。** 去 A100 容器读 Postmill 源码，拿到 `@RateLimit` 注解和
`ratelimit.error` 翻译键（见 §①）。结论保住了，但**靠的是完全不同的一条证据链**。

**R4 — 触发这次证伪的是一个"被污染"的 sub-agent。** 盲复检 B 读了 `master_bug_catalog`
（我的 prompt 没禁止），本该作废；但恰恰因为它知道既有结论，才发现 task 631/609 与之矛盾并
坚持报告"该措辞只出现在 thought，未见于 observation"。**独立性是为了避免锚定，但知情的
反对意见有独立性换不来的价值** —— 两者都要，不能互相替代。

### 教训（三条，写进流程）

1. **排除模型输出字段时，必须枚举全部副本字段**（`action` / `raw_action` / 以及任何未来新增的
   镜像字段），不能用"排除 action"这种单点否定。更稳的写法是**白名单**：只在明确属于环境侧的
   字段上匹配，而不是"除了 X 之外都算环境"。
2. **先确认数据里有没有你要找的东西，再设计判据。** 本轮在一个不含页面文本的记录上做了两轮
   "页面文本在不在"的判定，两轮都无效。字段结构检查应该是**第一步**，不是出错后的补救。
3. **"结论对"和"证据对"是两件事。** B1 的结论从头到尾成立，但支撑它的证据换了一整条。
   如果没有这次复核，落进 paper 的会是一个正确结论 + 一条经不起追问的证据链。

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

## V. 验证轮（2026-08-03，补 /diag 主轮跳过的验证）

主轮的 Tier-2 只覆盖 92/475 = 19.4% 的 failed，且 **383 个 `failed+hit` 一个都没做因果验证**
（SKILL.md 要求"每主导规则抽 2-3"）。本节是补做的四项。

### V1. 主导规则的因果验证 —— P36 / P31 双双是 risk-marker

抽"仅单一规则命中"的 episode（归属最干净），各 3 个：

| 规则 | failed 覆盖 | 判定 | 真死因（逐例） |
|---|---:|---|---|
| **P36** WALK_FAIL | 236 (51.0%) | **3/3 risk-marker，0/3 死因** | task 66 walk_fail 靠 fallback 自愈，真死因是 finish 阶段答案汇总缺陷；task 603 真死因是 agent 把任务要求的 typo `budge` "纠正"成 `budget`；task 610 真死因是发帖限流 |
| **P31** budget 耗尽 | 231 (49.9%) | **3/3 risk-marker，0/3 死因** | task 717 是 30 步 URL 零变化的方向性错误；task 610/614(som) 是"误把 navbar 锚点当提交按钮"的自我强化循环 |

**⚠️ digest §2 的 per-rule 分布表是症状分布，不是死因分布。** 引用时必须这样标注。

### V2. P5 / P45 —— 任务依赖，不能统一裁定

二者从不单独命中（solo=0），需判是独立死因还是伴随信号。结论**因任务而异**：

- **task 721 (phantom_som)：death-cause。** step14 成功点赞第 1 项后，steps15-29 连续 15 次点同一
  `element_id=47` 失败，吞掉 step14 后 **100% 的剩余预算**；剩 7 项 × 约 2 步 = 14 步，恰在 16 步
  可用预算内 → 无此死锁大概率能完成。
- **task 28 (dom)：risk-marker。** 两个 P5 窗口只吞 23% 预算且**都被打破恢复**（step22/26 成功换页）；
  P31 的真正上游是任务本身要求枚举未知长度的分页列表。

**结构性解释**：只要死锁吞掉"足够多"剩余预算，P31 必然作为下游终态标签共现——但"足够多"因任务而异，
所以"P5/P45 从不单独命中"这个统计事实**不足以推出**它们是伴随信号。

### V3. P48 的二次 gate —— `_finish_answer()` 从不读 `thought`

`P48` 由 **B1** 那轮 Tier-2 提议并落码，却在 B1 上 **0 命中**、B0 上 15 命中。排查出两道 gate：

1. `len(steps) > 4` 硬上限 —— B1 的同族样本跑 18–20 步，全被排除（docstring 已承认该局限）
2. **`_finish_answer()`（`diag_pattern_match.py:373-378`）只读 `action.answer`/`action.text`，
   从不 fallback 读 `action.thought`** —— B1 把"无结果"结论完整写在 `thought` 里而 `answer` 留空

第 2 条是**过窄漏检而非保守取舍**：规则要检测的机制本就是"agent 在 thought 里认定无结果"。
实测 B1 另有 **9 个 episode 已满足 ≤4 步门槛**，仅因 `answer` 为空被 gate 2 单独排除。

⚠️ **影响面超出 P48**：`P22 / P24 / P27 / P29 / P46` 同样依赖 `_finish_answer()`。
B0（proxy 235B）习惯把结论复述进 `answer`，B1（本地 4B）倾向只写 `thought` ——
**这会让所有依赖该函数的规则产生跨模型的系统性偏差**。

### V4. 盲复检 —— 独立一致率 5/8 = 62.5%

前两次复检因 prompt 未禁止读结论文档而**污染作废**（agent 主动"交叉验证"了 digest 与 bug catalog）。
第三次明确禁读 `docs/` 后：

| task | 主轮判定 | 盲复检 | |
|---|---|---|---|
| 67 / 641 / 714 / 622 | agent-limit / scaffold / agent-limit / agent-limit | 同 | ✓ |
| 584 | agent-limit（填错字段） | agent-limit（**点错 navbar Submit**） | ✓ 类别同、机制不同 |
| **600** | benchmark-FP（开放式 intent 被压成单一 golden） | **agent-limit**（未找 consoles 即选 gaming） | ✗ |
| **647** | agent-limit（绕开字面关键词） | **benchmark-FP**（同义词 `aid` vs 必需字面词 `help`） | ✗ |
| **651** | scaffold-bug（INERT_CLICK_LOOP） | **unclear**（low confidence） | ✗ |

**三处分歧的性质**：600 与 647 都是 **agent-limit ↔ benchmark-FP 的边界解读**（同一事实、两种归责），
不是事实错误；651 是**主轮过度自信**——盲复检诚实标 unclear 并给出了主轮遗漏的 scaffold 线索
（→ **B-1926**）。

**结论：Tier-2 的类别判定有约 1/3 的分歧率，且分歧集中在 agent-limit ↔ benchmark-FP 边界。**
digest 的三分类统计应按此打折引用。

### V5. 新规则候选 P49（已做 success-safe 全量检验）

**判据**：`/submit/` 页上 ≥2 次点击 `locator_route_meta.target_tag == 'A'` 的元素
**且** episode 终态 `obs_url` 仍在 `/submit/`（从未跳到帖子永久链接）。

| 档位 | failed 命中 | success 误伤 | 误伤率 |
|---|---:|---:|---:|
| ≥2 次（无终态条件） | 106 | 14 | 11.7% |
| **≥2 次 + 终态卡 `/submit/`** | **71** | **0** | **0.0%** |
| ≥3 次 + 终态卡 `/submit/` | 47 | 0 | 0.0% |

与已有规则的关系：`P49 ∩ P47 = 2`（几乎互斥，P47 要求 finish 前无 click）、`P49 ∩ P31 = 55`。
**价值不在扩大覆盖（相对 P47+P31 净新增仅 14），而在归因质量** —— 把 55 个原本只标
"P31 预算耗尽"（V1 已判为零信息量的 risk-marker）的 episode，升级为具体死因机制。
vision mode 天然 0 命中（坐标点击无 `target_tag`）→ 规则 mode-specific。

**建议**：落码为 P49，bump `RULESET_VERSION` 到 `10-*`，全量重扫 36 VWA + 12 WA condition。
（未落码，待裁定。）

---

## W. 波及面 —— 本轮发现对 VWA 36 份 diag 与 Macro/Micro 证据层的影响

> 本节**跨 benchmark**，不限于 WA reddit。回答"这些问题会不会波及其他 diag 和所有 macro/micro"。

### W1. 对 VWA 36 份 diag 的波及

| 本轮发现 | 波及 VWA | 台账是否已有 |
|---|---|---|
| **P36 / P31 是 risk-marker**（V1） | ✅ 36 份 digest 的 per-rule 表同为**症状分布** | **§317 已实证** — B1 som cls 上 `P31 failed 集 causal verify 0/4`、`success-fire 10/10 presence-only` |
| **`_finish_answer()` 不读 `thought`**（V3） | ✅ `P22/P24/P27/P29/P46` 在全部 36 condition 上都受影响，且是**跨模型**偏差（B0 写 `answer`，B1 只写 `thought`） | ❌ 新 |
| **B-1926 `page_changed` 假阳性** | ✅ **VWA 148 episode**（WA 仅 13），集中在 B2_ptext_red 20 · B2_psom_red 16 · B0_som_red 15 · B2_psom_cls 11 | **`router.py:95-99` 注释已记 VWA B2_phantom_text_reddit task 103**（同现象，触发 reason 是 `scroll_changed`）；§105 有同类先例 |
| **B-1925 `_format_history` 丢 thought** | ✅ 所有 baseline 所有 run | ❌ 新 |
| **B2 tokenize 假阴性** | 已查 → VWA 仅 2 task / 8 ep，且都不是决定性败因（见 §B2 VWA 对照） | §361.4 `token-granularity 族` |

### W2. 对 Micro 维度的波及（四维度框架）

`per_mode_four_dimension_profile.py` 有 **4 个 Micro 指标 + 1 个 Macro 指标**直接建在
`page_changed` 上：`no_change_rate` / `noop_inert_rate` / `scroll_inert_rate` /
`visibility_gap_rate`（Micro）+ `url_revisit_rate`（Macro）。

**step 级疑似假阳性率**（判据：`page_changed=True` 且 url 不变 且 scroll_y 不变 且
`text_similarity > 0.95`），VWA 36 condition：

```
vision 9.02% > som 7.90% > phantom_prompt 6.86% > dom 6.40% > phantom_som 6.32% > phantom_text 5.48%
```

**⚠️ `visibility_gap_rate` 抓不到这类假阳性** —— task 651 的 `agent_visible_changed` 是
**30/30 全 True**，volatile 片段同时顶起了 `page_changed` 和 `agent_visible_changed`，
所以 `page_changed=True AND agent_visible_changed=False` 命中 0。这个指标覆盖的是另一类问题。

**敏感性分析（把疑似假阳性 step 重算为 no-change，看跨 mode 排序是否翻转）**：

| cell | `no_change_rate` 最高 | 最低 | |
|---|---|---|---|
| B0/cls | vision → vision | som → **phantom_text** | ⚠ 最低端翻转 |
| B0/red · B1/cls · B1/red · B2/cls · B2/red | vision → vision | 各自不变 | ✓ |

**结论：主结论（`no_change_rate` 最高 = Vision）在 6/6 cell 稳健**；污染方向与现有结论一致
（vision 假阳性率最高 9.02%，修正只会加强），**最低端有 1/6 cell 翻转** → 涉及"哪个 mode
最低"的表述需要加脚注或改用区间。

### W3. 对 Macro 维度的波及

Macro 的四个条目（1a Tier-1 hook / 1b cascade / 1c strategy gradient / 1d action vocabulary）
分别源自 `axis_effect_size.json` 和 `mechanism_per_task.json`，**不直接消费 `page_changed`**。
唯一相关的是 `url_revisit_rate`（建在 URL 序列上，不受 page-change 判定影响）。
→ **Macro 维度受本轮发现的影响小。**

### W4. 对 router（paper §6）的波及 —— 这条最需要注意

`p79/experiment/router.py:80-83` 的 `unchanged_streak` 直接由 `prev_page_changed` 驱动，
`page_unchanged_streak` 是 escalation 触发器之一。在那 **161 个（VWA 148 + WA 13）
`page_changed` 恒 True 的 episode 上，router 的这条触发器永不累积** ——
`trigger_distribution` 全空，episode 卡满预算却"什么都没登记"。

`router.py` 自己的注释已经记录了这个现象（VWA B2 task 103）。**§105 先例明写同类污染的后果**：
> router signal AUROC / wasted_cost / no_op_rate 都受污染，paper §5/§6 数字需校

→ **paper §6 router 分析里凡涉及 rule-based escalation 触发率的数字，需要按此校验。**

### W5. 尚未评估的

- **377 个 `failed+hit` 的死因**（本轮只验了 6 个）· **B1 那批 80 个 Tier-2 归因**（一个没验）
- **VWA 36 份 digest 的 Tier-2 归因**（本轮完全没碰）
- `Efficiency` 与 `Outcome` 两个维度对 `page_changed` 的依赖（本节只查了 Macro/Micro）

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
