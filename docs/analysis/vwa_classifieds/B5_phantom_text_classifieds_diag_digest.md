# B5 phantom_text classifieds — /diag failure attribution digest

> **per-condition** (site × model × mode)。B5 = GPT-5.6 (`gpt-5.6-terra` via AWS proxy, `structured_output: response_format`)。
> **B5 不在预注册 cell 集合里** —— `run_manifest.yaml` 的 `extension:` 节 (笔记 §509)。本 digest 的数字可与 B0/B1/B2 的 v11 数字块并表 (同 ruleset),
> 但**不进**任何 pooled / K-of-N / forest 统计。跨 mode 共性、规则缺陷与提议集中在 [`B5_classifieds_cross_mode_diag_summary.md`](B5_classifieds_cross_mode_diag_summary.md)。

## 1. Header

| 字段 | 值 |
|---|---|
| Run | `B5_phantom_text_classifieds_20260828_131337_301148671_3809444_R4968` (manifest `extension:` 条目) |
| Condition | `phase1_phantom_text_router_0` |
| Site / Model / Mode | classifieds / **B5** / **phantom_text / P-text ([SOM_MARKS] 文本 + DOM 风格 prompt, 无图)** |
| N episodes | 224 |
| SR | **24.11%** (54/224) |
| **ruleset_version** | **`11-intent-text-fallback`** (与 48 个既有 condition 同版本) |
| 三子集 | failed+hit 136 · **failed NO-hit 34** · success+hit 12 |
| Tier-2 | no-hit **34/34 全覆盖** + hit 审计 1 ep; sonnet, 按 task 跨 mode 分批, 喂压缩轨迹 + 静态 catalog 的 item 查询 |
| Diag date | 2026-09-11 (B5 首次 diag) |

## 2. 三分类统计

| 类别 | 数 | 说明 |
|---|---:|---|
| **agent-limit** | 33 (no-hit) + Tier-1 命中的 136 | Tier-1 命中的全部是 agent-limit 类规则 |
| **scaffold-bug** | 0 | 无; 另: 本 condition 有 **86** 步 `multiple_actions` (其中 80 步页面变了), 全部落在 B-1998 的缝里 |
| **benchmark-FP 嫌疑** | 1 | task 41 (「第二行」依赖布局, 见 cross-mode summary §3) |
| unclear | 0 | — |

⚠️ **P31 漏标**: 本 condition 有 **30** 个失败 episode 跑满预算 (`trajectory_incomplete`) 却没有 P31 —— B-1999 (P31 豁免在 cls 上恒成立)。
读 P31 行时按「下界」读。

## 3. Tier-1 规则分布 (v11, 全 224 ep)

| 规则 | 含义 | step 级 | episode 级 | 其中 failed | 其中 success |
|---|---|---:|---:|---:|---:|
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 62 | 62 | 62 | 0 |
| `P17` | click-back振荡 | 49 | 49 | 49 | 0 |
| `P31` | budget耗尽未完成 | 49 | 49 | 49 | 0 |
| `P33` | 导航至裸图片URL幻觉 | 35 | 35 | 29 | 6 |
| `P10` | 跨步数值记忆失败 | 24 | 21 | 16 | 5 |
| `P36` | WALK_FAIL_DEGENERATE | 53 | 16 | 16 | 0 |
| `P20` | 评测目标页从未访问 | 13 | 13 | 13 | 0 |
| `P7` | sCity=州名 | 14 | 13 | 12 | 1 |
| `P5` | 感知缺失循环 | 7 | 7 | 7 | 0 |
| `P30` | 到达正确item后离开 | 7 | 7 | 7 | 0 |
| `P14` | URL 自环 | 5 | 5 | 5 | 0 |
| `P18` | cheapest漏价格排序 | 4 | 4 | 4 | 0 |
| `P25` | 跨站任务跳过其中一站 | 4 | 4 | 4 | 0 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 3 | 3 | 3 | 0 |
| `P28` | benchmark-FP货币tokenize | 3 | 3 | 3 | 0 |
| `P37` | URL_HALLUCINATION | 3 | 3 | 3 | 0 |
| `P22` | 图上数字dom不可读 | 2 | 2 | 2 | 0 |
| `P13` | 搜索代替浏览 | 1 | 1 | 1 | 0 |
| `P11` | 最新+地点组合 | 1 | 1 | 1 | 0 |
| `P46` | COMMENT_INTENT_NO_TYPE | 1 | 1 | 1 | 0 |

> ⚠️ 解读约束 (同 v11 数字块): ① 症状分布, 不是死因分布 (P36/P31 是 risk-marker); ② `P2`/`P4` 依赖 `element_bbox`;
> ③ `P10` 在 B5 上主要是误报: 成功 episode 也有 30 个触发 (B1 只有 1 个), 审计 4/4 非死因; 其中千分位拆分缺陷 (B-2000, `$6,400` 被读成 6 和 400)
>    只解释 B5 失败侧 61→50, 成功侧 31→30 基本不变 ⇒ 主因是日期 / 型号数字与价格混比; ④ `P31` 见上方 B-1999。

## 4. Tier-2 深挖

| no-hit 子类 | 数 | 含义 |
|---|---:|---|
| `keyword-literal-proxy` | 9 | 把字面关键词搜索当作语义/视觉判断 (参考 item 标题里没有这个词) |
| `wrong-item-visual-match` | 7 | 视觉/颜色/外观判断选错 item |
| `loop` | 6 | 搜索或滚动循环, 跑满 30 步 (P31 因 B-1999 没标出) |
| `stated-uncertainty-then-finish` | 3 | thought 说要核实, 下一步不核实就 finish |
| `no-visual-channel` | 2 | 该 mode 无图, 任务要看页面图像 (结构性不可解) |
| `constraint-dropped` | 2 | 丢了类目 / 物品类型 / 排序约束 |
| `other` | 2 | 其他 (见下方代表 episode) |
| `benchmark:task41` | 1 | benchmark-FP 嫌疑: task 41「第二行」依赖布局 |
| `misread-intent` | 1 | 读错 intent 的语义 |
| `given-page-scope-abandoned` | 1 | intent 指「这一页」, 第一步就搜索离开给定页 |

**Tier-2 (34/34 no-hit 全覆盖)**: 无图 mode 的典型分布 —— 字面关键词 (9) + 视觉判断 (7) + 循环 (6)。
- t47: 把 intent 的「the site」当成「我自己的帖子」, 进 My-Account 页连续 scroll 29 步, 从没搜索。
- t105 / t169: thought 里已写出「需要核实 / 这些是配件不是手机」, 下一步照样点开同一个配件并 finish —— 「说了要核实却不核实」在本 mode 出现 3 次。
- t11 / t123: 任务要判断颜色, 本 mode 结构上没有任何图像输入, 只能搜颜色词或凭标题猜。

**hit 审计** (success-hit / failed-hit 抽样):

| task | 类型 | 规则 | 真死因? | 依据 |
|---|---|---|---|---|
| 208 | SUCCESS-HIT | `P33` | 是 | phantom_text 完全无图；step2 打开裸图、step3 立刻 back 未获取任何信息，但最终评论仍写'the insect...is a butterfly'——这是在零视觉输入下的纯编造；任务判为 success 只是因为 program_html 判据只核对评论标题'Questions by Bla |

## 5. 代表 episode

- **t188** [`keyword-literal-proxy`] The task requires visually spotting which book COVER PHOTO depicts a baby, but the agent runs a literal text search for the word 'baby' against listing titles/descriptions. The reference item's title ('Christmas in Ameri  
  证据: step_0-2: scrolls/paginates the Books gallery without a match; step_3: type 'baby' search; step_5: click id=9538 ('What to Expect When You're Expecting & Toddler Years', thought: 'most likely to have 
- **t67** [`keyword-literal-proxy`] Same as som: exhausted keyword permutations (basketball / NBA 2K / game / basketball game) under price-ascending sort but never dropped to a keyword-free category browse; the $3 reference item's text doesn't contain any   
  证据: step_10: click element_id 60 -> item 5219 under sPattern='NBA 2K basketball' sorted ascending; step_11 finish on the $5 SNES bundle, identical wrong item to the som-mode run.
- **t117** [`wrong-item-visual-match`] Applied the same 'blue' keyword filter (here after first sorting by price) as the other no-image modes and finished on the top result without any color verification, unsurprising since phantom_text has no image to check   
  证据: step_2: sort 'Lower price first'; step_3: type 'blue'; step_4-5: click id=59840 then finish.
- **t66** [`wrong-item-visual-match`] Same greedy sort-then-click-cheapest pattern as som, landing on the identical wrong item id=69961; justifies with 'description includes ESPN NFL Football' -- a different specific game title than som's 'NFL 2K4' claim for  
  证据: step_2: click element_id=50 -> item&id=69961 | thought '...its description includes ESPN NFL Football, matching the image.' Also note step_0/step_1 both logged PARSE_INVALID(multiple_actions) (known p
- **t90** [`loop`] Agent repeatedly re-issues the literal keyword search "discoloring"/"discolor" against Cars+trucks (0 hits every time), resets via the category breadcrumb, and repeats the identical cycle instead of switching to visual/g  
  证据: step_0: switch to gallery view but does not inspect it; step_1: type 'discoloring' -> 0 hits; step_2: click category breadcrumb; step_3: type 'discoloring' again -> identical click(9)/type(12) cycle r
- **t47** [`loop`] Agent assumed "the site" meant the logged-in user's own listings, navigated to My-Account items and then scrolled up/down 29 times without ever trying the site search or category filter, exhausting the 30-step budget on   
  证据: step_0: "since this is the logged-in user's content, I will inspect My account for the user's listings"; steps_1-29: all scroll actions on url=index.php?page=user&action=items, never recovers (one Ctr
- **t97** [`stated-uncertainty-then-finish`] Skipped keyword search entirely, opened only the single top listing in the unfiltered default Household category ('2 Zebra Pillows', which is animal-*patterned* not animal-*shaped*) and finished after just 4 steps withou  
  证据: step_2: click element_id 38 -> item id=65205 'I need inspect the newest listings...starting with the top candidate's detail page'; step_3: finish immediately after a single item view.
- **t105** [`stated-uncertainty-then-finish`] Agent explicitly stated it needed to 'verify whether it includes a red velvet lined case' on the top $1-sorted item, then finished in the very next step with no intervening inspection action confirming that description t  
  证据: step_12 thought: '...I will inspect the first guitar listing at $1 to verify whether it includes a red velvet lined case'; step_13: finish immediately with no scroll/observe action in between.
- **t123** [`no-visual-channel`] phantom_text has no observation image ([SOM_MARKS] text only); agent cannot see listing colours and defaults to a site-wide 'yellow' keyword search, even briefly opening a wrong item explicitly noted as 'sage green' befo  
  证据: step_3: back | thought 'The reference object is yellow, while this opened listing is sage green.' -- agent is reasoning about colour words in text, never about pixels; final click id=5469 (Household) 
- **t11** [`no-visual-channel`] phantom_text carries zero image data to the model by construction, so 'blue' is unverifiable; the agent browsed the Bikes category page (no keyword search this time) and simply opened the first bike it scrolled past, wit  
  证据: step_2: click id=76502 — thought 'I need inspect the first bike listing's detail page to determine whether it is the first blue bike and find its wheel size' (no color signal available in this mode); 

## 6. 失败桶 (reason_bucket 五桶, `failure_modes_per_cell.md` Extension cells)

N=224, failed=170

| 失败桶 | 数 | 占 failed | 占全部 |
|---|---:|---:|---:|
| early-finish/wrong-commit | 91 | 53.5% | 40.6% |
| visual-hijack/click-loop | 31 | 18.2% | 13.8% |
| search-loop | 27 | 15.9% | 12.1% |
| max-steps-other | 21 | 12.4% | 9.4% |

## 7. Self-evolving 与 actionable

- 本 condition **不单独提规则**: 所有提议在 cross-mode summary §5 统一给出, 每条带 0-token 全量复核结果 (两条最常被 sub-agent 提的「字面关键词」「放弃给定页」复核后**都不 success-safe**, 只当风险标记)。
- scaffold → **B-1998** (`multiple_actions` 执行却按 wait 记账); diag 规则 → **B-1999** (P31) · **B-2000** (P10 千分位)。
- 无 task 需要排除; task 41 的布局依赖是跨 baseline 的 (32/32 run 全错), 登记在 summary, 是否剔除待 user 定。
