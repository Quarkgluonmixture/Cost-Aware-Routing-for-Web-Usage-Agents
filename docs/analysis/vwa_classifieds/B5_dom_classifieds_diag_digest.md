# B5 dom classifieds — /diag failure attribution digest

> **per-condition** (site × model × mode)。B5 = GPT-5.6 (`gpt-5.6-terra` via AWS proxy, `structured_output: response_format`)。
> **B5 不在预注册 cell 集合里** —— `run_manifest.yaml` 的 `extension:` 节 (笔记 §509)。本 digest 的数字可与 B0/B1/B2 的 v11 数字块并表 (同 ruleset),
> 但**不进**任何 pooled / K-of-N / forest 统计。跨 mode 共性、规则缺陷与提议集中在 [`B5_classifieds_cross_mode_diag_summary.md`](B5_classifieds_cross_mode_diag_summary.md)。

## 1. Header

| 字段 | 值 |
|---|---|
| Run | `B5_dom_classifieds_20260820_202158_076182888_2491046_R29736` (manifest `extension:` 条目) |
| Condition | `phase1_dom_router_0` |
| Site / Model / Mode | classifieds / **B5** / **dom (AXTree 文本, 无截图)** |
| N episodes | 224 |
| SR | **23.66%** (53/224) |
| **ruleset_version** | **`11-intent-text-fallback`** (与 48 个既有 condition 同版本) |
| 三子集 | failed+hit 154 · **failed NO-hit 17** · success+hit 42 |
| Tier-2 | no-hit **17/17 全覆盖** + hit 审计 4 ep; sonnet, 按 task 跨 mode 分批, 喂压缩轨迹 + 静态 catalog 的 item 查询 |
| Diag date | 2026-09-11 (B5 首次 diag) |

## 2. 三分类统计

| 类别 | 数 | 说明 |
|---|---:|---|
| **agent-limit** | 16 (no-hit) + Tier-1 命中的 154 | Tier-1 命中的全部是 agent-limit 类规则 |
| **scaffold-bug** | 1 | t210 → B-1998; 另: 本 condition 有 **89** 步 `multiple_actions` (其中 85 步页面变了), 全部落在 B-1998 的缝里 |
| **benchmark-FP 嫌疑** | 0 | 无 |
| unclear | 0 | — |

⚠️ **P31 漏标**: 本 condition 有 **26** 个失败 episode 跑满预算 (`trajectory_incomplete`) 却没有 P31 —— B-1999 (P31 豁免在 cls 上恒成立)。
读 P31 行时按「下界」读。

## 3. Tier-1 规则分布 (v11, 全 224 ep)

| 规则 | 含义 | step 级 | episode 级 | 其中 failed | 其中 success |
|---|---|---:|---:|---:|---:|
| `P6` | 视觉任务 DOM 必然失败 | 106 | 106 | 73 | 33 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 64 | 64 | 64 | 0 |
| `P16` | 视觉图像内容DOM必败 | 53 | 53 | 41 | 12 |
| `P31` | budget耗尽未完成 | 40 | 40 | 40 | 0 |
| `P17` | click-back振荡 | 35 | 35 | 35 | 0 |
| `P33` | 导航至裸图片URL幻觉 | 32 | 32 | 28 | 4 |
| `P10` | 跨步数值记忆失败 | 18 | 17 | 10 | 7 |
| `P20` | 评测目标页从未访问 | 15 | 15 | 15 | 0 |
| `P7` | sCity=州名 | 13 | 12 | 10 | 2 |
| `P36` | WALK_FAIL_DEGENERATE | 29 | 11 | 11 | 0 |
| `P30` | 到达正确item后离开 | 7 | 7 | 7 | 0 |
| `P15` | gallery行位置DOM不可定位 | 6 | 6 | 5 | 1 |
| `P5` | 感知缺失循环 | 4 | 4 | 4 | 0 |
| `P18` | cheapest漏价格排序 | 3 | 3 | 3 | 0 |
| `P37` | URL_HALLUCINATION | 3 | 3 | 3 | 0 |
| `P14` | URL 自环 | 3 | 3 | 3 | 0 |
| `P28` | benchmark-FP货币tokenize | 2 | 2 | 2 | 0 |
| `P13` | 搜索代替浏览 | 2 | 2 | 2 | 0 |
| `P44` | HALLUCINATED_ELEMENT_REF | 2 | 2 | 2 | 0 |
| `P22` | 图上数字dom不可读 | 2 | 2 | 2 | 0 |
| `P21` | dom模式视觉幻觉 | 2 | 2 | 2 | 0 |
| `P11` | 最新+地点组合 | 1 | 1 | 1 | 0 |
| `P24` | 不确定仍finish | 1 | 1 | 1 | 0 |
| `P19` | url_match过早搜索页finish | 1 | 1 | 1 | 0 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 1 | 1 | 1 | 0 |
| `P46` | COMMENT_INTENT_NO_TYPE | 1 | 1 | 1 | 0 |
| `P25` | 跨站任务跳过其中一站 | 1 | 1 | 1 | 0 |

> ⚠️ 解读约束 (同 v11 数字块): ① 症状分布, 不是死因分布 (P36/P31 是 risk-marker); ② `P2`/`P4` 依赖 `element_bbox`;
> ③ `P10` 在 B5 上主要是误报: 成功 episode 也有 30 个触发 (B1 只有 1 个), 审计 4/4 非死因; 其中千分位拆分缺陷 (B-2000, `$6,400` 被读成 6 和 400)
>    只解释 B5 失败侧 61→50, 成功侧 31→30 基本不变 ⇒ 主因是日期 / 型号数字与价格混比; ④ `P31` 见上方 B-1999。

## 4. Tier-2 深挖

| no-hit 子类 | 数 | 含义 |
|---|---:|---|
| `keyword-literal-proxy` | 6 | 把字面关键词搜索当作语义/视觉判断 (参考 item 标题里没有这个词) |
| `loop` | 5 | 搜索或滚动循环, 跑满 30 步 (P31 因 B-1999 没标出) |
| `no-visual-channel` | 2 | 该 mode 无图, 任务要看页面图像 (结构性不可解) |
| `wrong-item-visual-match` | 1 | 视觉/颜色/外观判断选错 item |
| `scaffold:parse-abort` | 1 | scaffold: multiple_actions 被判无效却已执行, 连续 3 次 abort (B-1998) |
| `constraint-dropped` | 1 | 丢了类目 / 物品类型 / 排序约束 |
| `answered-wrong-value` | 1 | string_match 答错数值 / 计数 / 属性 |

**Tier-2 (17/17 no-hit 全覆盖)**: 主体仍是 cls 上 B0/B1 见过的两类 —— 字面关键词代理 (6) 与循环 (5)。
dom 无页面截图, 遇到「颜色 / 封面 / 屏幕内容」只能退化为搜颜色词或名词 (t123 搜 `yellow` 跨类目乱点; t92 要看电视屏幕在播 NFL)。
唯一 scaffold 是 **t210**: 三步都被判 `multiple_actions` 却都执行了, 撞 `max_consecutive_parse_errors=3` 被杀 (B-1998)。

**success-hit 42 个**主要是 `P6`/`P16` (dom 视觉任务类, 早已知是 presence-only 风险标记), 本轮不再抽审。

**replicate 对照 (R15476, 同配置第二次跑)**: no-hit 17 vs 17, 头部规则 episode 数 `P6` 106/106 · `P43` 64/62 · `P16` 53/53 ·
`P31` 40/37 · `P17` 35/42 · `P33` 32/27 —— 规则层的分布比 SR (23.66 vs 25.00) 还稳, 与 §508.2「行为指标比 SR 稳」同向。

**hit 审计** (success-hit / failed-hit 抽样):

| task | 类型 | 规则 | 真死因? | 依据 |
|---|---|---|---|---|
| 203 | SUCCESS-HIT | `P33` | 否 | dom 无图；step5 打开裸图，step6 back 后未对图片内容做任何具体断言，最终评论'Do you have a USB-C cable?'是基于'页面文字未说明兼容 USB-C'的缺失证据推理，不是编造的视觉内容。 |
| 63 | SUCCESS-HIT | `P10` | 否 | thought 6400.0 与最终答案 '$6,400.00' 实为同一价格；output num 被记成 '6.0' 是数字提取正则在千分位逗号处截断成 '6' 的解析 bug，而非模型记忆错误。 |
| 23 | FAILED-HIT | `P17` | 否 | item 8744 被反复重开(steps 1/3/6/8)但每次都指向同一(看似正确)的车辆本身没有变化，最终失败是里程数被读错('164,000' vs 参考 '64,000')这一读数错误，与是否重复开合该 item 并无直接因果。 |
| 12 | FAILED-HIT | `P33` | 是 | dom 无图；step2 定位到的 item 9068 正是参考答案(reference_url 一致)，但 step6 打开裸图、step7 back 后没有任何文字线索支持颜色，step8 却断言颜色为'White'——纯编造，而参考答案是'red'，这个编造直接就是最终的错误输出。 |

## 5. 代表 episode

- **t188** [`keyword-literal-proxy`] The task requires visually spotting which book COVER PHOTO depicts a baby, but the agent runs a literal text search for the word 'baby' against listing titles/descriptions. The reference item's title ('Christmas in Ameri  
  证据: step_0: type 'baby' -> search hits; step_1: click id=56253 ('Baby-Sitters Club Graphic Novels', title contains 'baby'); step_2: finish (thought: 'the likely baby-related target') -- picked by title ke
- **t40** [`keyword-literal-proxy`] 已有先例定案(docs/analysis/vwa_classifieds/B0_vision_classifieds_R32024_diag_digest.md, task 40)：真正'最新的不锈钢洗碗机'是 LG(id=18607)，但其标题不含'stainless steel'字样；agent 只用 'stainless'/'stainless steel' 关键词搜索，搜不到 LG，只找到标题明写 'Stainless Stee  
  证据: step_3 type 'stainless\n'; step_11 click element_id=10764 url=...id=70627; step_12 finish answer 'GE Profile'——thought 称'the visible listing explicitly described as a stainless steel dishwasher is the
- **t97** [`loop`] Repeatedly re-ran literal 'animal'/'animal shaped' keyword searches (which never match the reference item's title, a deer-decoration listing) and burned 20+ steps oscillating scroll direction without systematically cover  
  证据: step_10-20: repeated 'scroll up'/'scroll down' reversals at the same sPattern=animal URL with thoughts like 'the current keyword results mostly mention animals rather than being shaped like one'; step
- **t90** [`loop`] Agent repeatedly re-issues the literal keyword search "discoloring"/"discolor" against Cars+trucks (0 hits every time), resets via the category breadcrumb, and repeats the identical cycle instead of switching to visual/g  
  证据: step_1: type 'discoloring' -> 0 hits; step_3: click category breadcrumb resets sPattern; step_4: type 'discoloring' again -> identical cycle repeats through step_30 (>13 times); agent never visits any
- **t123** [`no-visual-channel`] dom has no observation image of the current page (AXTree text only), so colour of on-page listings is unknowable from text; agent substitutes a global keyword search for 'yellow' which pulls an unrelated Household item i  
  证据: step_0: type element_id=183 text='yellow' | thought 'The object being used is yellow, so I will search listings for yellow to locate the matching clothing item.' -> abandons 'this page' scope immediat
- **t92** [`no-visual-channel`] The task requires visually confirming a TV's screen 'displays an ongoing NFL game', which is structurally impossible to judge from dom mode's AXTree-only text (zero image channel); the agent also wasted early steps on su  
  证据: step_0: type 'TV' (2 chars) and step_2: type 'NFL' (3 chars), both below the site's apparent 4-character search minimum (subsequent identical-looking searches for 'television'/'football' were needed a
- **t117** [`wrong-item-visual-match`] Used a plain-text 'blue' keyword search plus price-ascending sort as a proxy for verifying visual color similarity to the reference outfit photo, then finished on the first ($10) result without ever opening/comparing the  
  证据: step_1: type element_id 141 'blue'; step_2: sort 'Lower price first'; step_3-4: click id=59840 then finish, with no intervening step that inspects the item image for color.
- **t210** [`scaffold:parse-abort`] All 3 model turns were classified parse_valid=False/parse_failure_reason=multiple_actions by the action parser even though each single action was still applied and matched its thought (action_intent_fulfilled=true, URL p  
  证据: step_2: select_option 'Lower price first' -> parse_valid=false, parse_failure_reason='multiple_actions'; summary: model_call_attempt_count=3, runner_iteration_count=3, agent_action_step_count=0, valid

## 6. 失败桶 (reason_bucket 五桶, `failure_modes_per_cell.md` Extension cells)

N=224, failed=171

| 失败桶 | 数 | 占 failed | 占全部 |
|---|---:|---:|---:|
| early-finish/wrong-commit | 105 | 61.4% | 46.9% |
| max-steps-other | 29 | 17.0% | 12.9% |
| search-loop | 20 | 11.7% | 8.9% |
| visual-hijack/click-loop | 15 | 8.8% | 6.7% |
| error/noise | 2 | 1.2% | 0.9% |

## 7. Self-evolving 与 actionable

- 本 condition **不单独提规则**: 所有提议在 cross-mode summary §5 统一给出, 每条带 0-token 全量复核结果 (两条最常被 sub-agent 提的「字面关键词」「放弃给定页」复核后**都不 success-safe**, 只当风险标记)。
- scaffold → **B-1998** (`multiple_actions` 执行却按 wait 记账); diag 规则 → **B-1999** (P31) · **B-2000** (P10 千分位)。
- 无 task 需要排除; task 41 的布局依赖是跨 baseline 的 (32/32 run 全错), 登记在 summary, 是否剔除待 user 定。
