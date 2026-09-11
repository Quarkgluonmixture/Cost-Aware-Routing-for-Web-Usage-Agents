# B5 som classifieds — /diag failure attribution digest

> **per-condition** (site × model × mode)。B5 = GPT-5.6 (`gpt-5.6-terra` via AWS proxy, `structured_output: response_format`)。
> **B5 不在预注册 cell 集合里** —— `run_manifest.yaml` 的 `extension:` 节 (笔记 §509)。本 digest 的数字可与 B0/B1/B2 的 v11 数字块并表 (同 ruleset),
> 但**不进**任何 pooled / K-of-N / forest 统计。跨 mode 共性、规则缺陷与提议集中在 [`B5_classifieds_cross_mode_diag_summary.md`](B5_classifieds_cross_mode_diag_summary.md)。

## 1. Header

| 字段 | 值 |
|---|---|
| Run | `B5_som_classifieds_20260826_084506_787426966_3444289_R31483` (manifest `extension:` 条目) |
| Condition | `phase1_som_router_0` |
| Site / Model / Mode | classifieds / **B5** / **som (SoM 标注截图 + [SOM_MARKS] 文本)** |
| N episodes | 224 |
| SR | **37.05%** (83/224) |
| **ruleset_version** | **`11-intent-text-fallback`** (与 48 个既有 condition 同版本) |
| 三子集 | failed+hit 87 · **failed NO-hit 54** · success+hit 14 |
| Tier-2 | no-hit **54/54 全覆盖** + hit 审计 8 ep; sonnet, 按 task 跨 mode 分批, 喂压缩轨迹 + 静态 catalog 的 item 查询 |
| Diag date | 2026-09-11 (B5 首次 diag) |

## 2. 三分类统计

| 类别 | 数 | 说明 |
|---|---:|---|
| **agent-limit** | 53 (no-hit) + Tier-1 命中的 87 | Tier-1 命中的全部是 agent-limit 类规则 |
| **scaffold-bug** | 0 | 无; 另: 本 condition 有 **72** 步 `multiple_actions` (其中 69 步页面变了), 全部落在 B-1998 的缝里 |
| **benchmark-FP 嫌疑** | 1 | task 41 (「第二行」依赖布局, 见 cross-mode summary §3) |
| unclear | 0 | — |

⚠️ **P31 漏标**: 本 condition 有 **32** 个失败 episode 跑满预算 (`trajectory_incomplete`) 却没有 P31 —— B-1999 (P31 豁免在 cls 上恒成立)。
读 P31 行时按「下界」读。

## 3. Tier-1 规则分布 (v11, 全 224 ep)

| 规则 | 含义 | step 级 | episode 级 | 其中 failed | 其中 success |
|---|---|---:|---:|---:|---:|
| `P31` | budget耗尽未完成 | 35 | 35 | 35 | 0 |
| `P10` | 跨步数值记忆失败 | 19 | 18 | 11 | 7 |
| `P17` | click-back振荡 | 17 | 17 | 17 | 0 |
| `P20` | 评测目标页从未访问 | 14 | 14 | 14 | 0 |
| `P7` | sCity=州名 | 13 | 12 | 10 | 2 |
| `P36` | WALK_FAIL_DEGENERATE | 9 | 9 | 9 | 0 |
| `P33` | 导航至裸图片URL幻觉 | 9 | 9 | 4 | 5 |
| `P25` | 跨站任务跳过其中一站 | 7 | 7 | 6 | 1 |
| `P18` | cheapest漏价格排序 | 5 | 5 | 5 | 0 |
| `P30` | 到达正确item后离开 | 5 | 5 | 5 | 0 |
| `P14` | URL 自环 | 8 | 5 | 5 | 0 |
| `P13` | 搜索代替浏览 | 2 | 2 | 2 | 0 |
| `P28` | benchmark-FP货币tokenize | 2 | 2 | 2 | 0 |
| `P22` | 图上数字dom不可读 | 2 | 2 | 2 | 0 |
| `P37` | URL_HALLUCINATION | 2 | 2 | 2 | 0 |
| `P5` | 感知缺失循环 | 5 | 2 | 2 | 0 |
| `P11` | 最新+地点组合 | 1 | 1 | 1 | 0 |
| `P12` | 从不翻页 | 1 | 1 | 1 | 0 |

> ⚠️ 解读约束 (同 v11 数字块): ① 症状分布, 不是死因分布 (P36/P31 是 risk-marker); ② `P2`/`P4` 依赖 `element_bbox`;
> ③ `P10` 在 B5 上主要是误报: 成功 episode 也有 30 个触发 (B1 只有 1 个), 审计 4/4 非死因; 其中千分位拆分缺陷 (B-2000, `$6,400` 被读成 6 和 400)
>    只解释 B5 失败侧 61→50, 成功侧 31→30 基本不变 ⇒ 主因是日期 / 型号数字与价格混比; ④ `P31` 见上方 B-1999。

## 4. Tier-2 深挖

| no-hit 子类 | 数 | 含义 |
|---|---:|---|
| `keyword-literal-proxy` | 19 | 把字面关键词搜索当作语义/视觉判断 (参考 item 标题里没有这个词) |
| `wrong-item-visual-match` | 13 | 视觉/颜色/外观判断选错 item |
| `loop` | 5 | 搜索或滚动循环, 跑满 30 步 (P31 因 B-1999 没标出) |
| `constraint-dropped` | 4 | 丢了类目 / 物品类型 / 排序约束 |
| `other` | 4 | 其他 (见下方代表 episode) |
| `given-page-scope-abandoned` | 3 | intent 指「这一页」, 第一步就搜索离开给定页 |
| `misread-intent` | 2 | 读错 intent 的语义 |
| `answered-wrong-value` | 2 | string_match 答错数值 / 计数 / 属性 |
| `benchmark:task41` | 1 | benchmark-FP 嫌疑: task 41「第二行」依赖布局 |
| `wrong-item-ordering` | 1 | 没用 / 用错排序 (most recent / cheapest) |

**Tier-2 (54/54 no-hit 全覆盖)**: B5 六个 mode 里 SR 最高 (37.05%), 但 no-hit 也最多 —— 失败里 Tier-1 规则能认出的比例最低 (87/141 = 62%),
说明强模型在有图 mode 上的失败更少落进「循环 / 卡住」这些规则擅长的形状, 更多是「看起来合理、选错了」。
- **有截图也走文本捷径** (13 个视觉判断错 + 19 个字面关键词): t117 / t123 与四个无图 mode 走同一条「搜颜色词 → 价格排序 → 点第一个」, 截图没被用来核对颜色。
- **丢约束**: t169 把标题含 Black 的手机壳当「黑色手机」; t19 `sea` 只有 3 个字符被站内搜索拒绝, 于是放弃「画的是海」这个约束, 改搜 `painting` 取最新一条。
- **放弃给定页** (t94 / t121): start_url 已是题目预设的类目 / 排序 / gallery 页, 第一步就改搜关键词, 把预设参数丢了。
- t49: 在第 6 步已经打开了正确 item, 滚一下就判断「没有 RAM 字段」离开 —— 近似 P30, 但任务是 string_match, P30 不覆盖。

**P33 在 som 上是误报** (审计 4/4 `hit_causal=false`): som 每步有截图, 点开裸图片 URL 是放大看图的合理策略 (t148 / t199 直接从大图读出答案)。

**hit 审计** (success-hit / failed-hit 抽样):

| task | 类型 | 规则 | 真死因? | 依据 |
|---|---|---|---|---|
| 118 | SUCCESS-HIT | `P33` | 否 | som 每步有截图；agent 在 step6 自认'opened photo 不显示手机屏幕'并未借此编造，最终答案 3:03 来自此前 item 详情页缩略图截图（step3-4），裸图事件是无害死路，不构成幻觉。 |
| 148 | SUCCESS-HIT | `P33` | 否 | step1 打开裸图后，step3 直接引用截图内容'the rating line...reads 128 ratings'作答，证明 som 截图通道确实提供了可读信息，是放大读图而非幻觉。 |
| 199 | SUCCESS-HIT | `P33` | 否 | step0 打开裸图，step1 直接读取截图水印'KAIYO.COM'作答，是典型的截图放大读取，非幻觉。 |
| 20 | SUCCESS-HIT | `P10` | 否 | step8 thought 中的 [2023,11,16] 是列表日期字段，output num 200.0 是价格字段，两者是不同语义量的正常共存，并非同一数值被跨步记错。 |
| 76 | SUCCESS-HIT | `P10` | 否 | thought 中的 250.0 是表单里即将被替换掉的旧价格（step5 之前），output 85.5 是任务要求写入的新价格，二者本就该不同，是正常的'替换'动作而非记忆失败。 |
| 17 | FAILED-HIT | `P17` | 是 | 同一 item 10865 被反复重开(steps 3/6/9)，agent 在 step4/5/8 已自认'不含红把手'仍原地打转，从未去看筛选结果里的另一候选，最终仍以该已知不符的 item 收尾——thrash 直接导致错误答案。 |
| 32 | FAILED-HIT | `P17` | 是 | item 23486 被重开 22 次、来回 back 3 次，全程锁死在这一个(错误)候选上反复找评论框，真正目标 item 9689 从未被访问(P20 同时命中)，30 步耗尽仍未完成——是导致 trajectory_incomplete 的直接死因。 |
| 128 | FAILED-HIT | `P33` | 否 | som 有截图；step2 打开裸图后 step3-4 用 Ctrl++ 放大截图并读出'99,5,81,13'，是基于真实图像的读数，只是把 80 错读成 81、漏读 92、多出一个 5——是视觉误读而非'无内容可读→编造'。 |

## 5. 代表 episode

- **t210** [`keyword-literal-proxy`] Searched literal keyword 'lamb' inside category Farm+garden and sorted price ascending, then trusted the cheapest hit without verifying it was actually the livestock 'lamb' the intent means; the top hit was 'Lambs Ear pl  
  证据: step_4: finish answer 'Lambs Ear plants Perennials Groundcover — $6.00'; ITEM LOOKUP shows agent_final_item id=32759 title contains 'Lambs Ear' (a plant) vs reference id=81060 'Katahdin ewe Fincastle'
- **t67** [`keyword-literal-proxy`] Agent only ever keyword-searched (basketball / NBA 2K / game combos) sorted by price ascending and never browsed the Video-gaming category directly by price; the true cheapest match (NCAA March Madness 2005 Xbox, $3) doe  
  证据: step_11: click element_id 60 -> item 5219 while search pattern is plain 'basketball' sorted ascending; thought claims this is 'the cheapest relevant video game listing visible' without ever trying non
- **t117** [`wrong-item-visual-match`] Despite having the annotated screenshot available in this mode, followed the identical keyword-search+price-sort shortcut as the no-image modes and finished immediately after one click, never visually comparing the bike'  
  证据: step_1-2: type 'blue' then sort 'Lower price first'; step_3-4: click id=59840 then finish with thought 'the first $10 listing explicitly describes a blue bike' — relies on listing text, not the photo,
- **t123** [`wrong-item-visual-match`] som is the one mode among the 5 failing here that DOES have a page screenshot, yet it still defaulted to a global 'yellow' text search instead of visually comparing colours of items already on the page, landing on the sa  
  证据: step_0: type text='yellow' | thought 'The person is using a yellow squeeze bottle, so I should search listings for yellow.'; step_3/5 click id=25 -> item&id=5469 Dyson vacuum, justified purely by titl
- **t97** [`loop`] Same literal keyword blind spot as other modes (queries 'animal', 'shaped', 'animal shaped' never match the reference deer-decoration title) plus indecisive alternating scroll/press-Home cycling consumed the entire 31-st  
  证据: step_16-30: long alternating scroll-up/scroll-down/goto sequence at url=...sPattern=animal&sCategory=18 with thoughts repeatedly saying results 'merely mention animals rather than being animal-shaped'
- **t90** [`loop`] Agent repeatedly re-issues the literal keyword search "discoloring"/"discolor" against Cars+trucks (0 hits every time), resets via the category breadcrumb, and repeats the identical cycle instead of switching to visual/g  
  证据: step_0: type 'discoloring' -> 0 hits; step_1: click category breadcrumb (element_id 9) resets filter; step_2: type 'discoloring' again (element_id 12) -> identical click(9)/type(12) cycle repeats verb
- **t19** [`constraint-dropped`] 屡次搜索 'sea'(<4字符被拒)后放弃'showing the sea'视觉约束，改用宽泛关键词 'painting' 并直接点击排序最新的第一条结果(BIG WALL PAINTING, id=38497, $90, 2023/11/15)，未逐条核实图像是否真的画有海景；该条目日期(11/15)甚至晚于参考答案(11/14)，说明它大概率满足价格/类目但不满足海景约束——真正三项都满足的最新条目才是参考答案。  
  证据: step_9 type 'painting' url=...sPattern=painting...; step_10 click element_id=42 (最新一条结果); step_11 finish 声称 'its image depicts the sea'，但此前从未用 sea/ocean 关键词成功命中过该条目，是放弃视觉关键词搜索后直接采信'最新+价格类目匹配'的第一条。
- **t169** [`constraint-dropped`] 关键词搜'black'+价格升序后，直接把标题含'Black'的手机壳(Pixel 6 Pro Case)当作'黑色手机'交差，未核实该条目其实是配件(Case)而非手机本体，违反 intent 里'phone'的物品类型限定  
  证据: step_6: finish "Opened the cheapest qualifying listing: Pixel 6 Pro Case Stormy Sky Black — $5.00."
- **t49** [`other`] Agent actually opened the item the task config identifies as correct (id=60306) at step_6, scrolled down once then back up, wrongly concluded the RAM spec was not shown, navigated away, and eventually finished on an unre  
  证据: step_6: click id=60306 "the related listing image visibly shows blue LED lights"; step_9: "its RAM specification is not shown in the visible detail text" -> click element_id=2 navigates away to url=(h
- **t78** [`other`] Agent conflates 'item located in Pittsburgh' with 'item that references the city' and repeatedly mis-tracks which search-result row corresponds to which location: it opens three different off-Pittsburgh listings (Rockvil  
  证据: step_1: click id=57243 believing 'the visible listing marked Pittsburgh is the Garfield Books lot', then step_2 back after realizing 'this opened listing is located in Rockville' -- this misjudge-then

## 6. 失败桶 (reason_bucket 五桶, `failure_modes_per_cell.md` Extension cells)

N=224, failed=141

| 失败桶 | 数 | 占 failed | 占全部 |
|---|---:|---:|---:|
| early-finish/wrong-commit | 74 | 52.5% | 33.0% |
| search-loop | 31 | 22.0% | 13.8% |
| max-steps-other | 23 | 16.3% | 10.3% |
| visual-hijack/click-loop | 13 | 9.2% | 5.8% |

## 7. Self-evolving 与 actionable

- 本 condition **不单独提规则**: 所有提议在 cross-mode summary §5 统一给出, 每条带 0-token 全量复核结果 (两条最常被 sub-agent 提的「字面关键词」「放弃给定页」复核后**都不 success-safe**, 只当风险标记)。
- scaffold → **B-1998** (`multiple_actions` 执行却按 wait 记账); diag 规则 → **B-1999** (P31) · **B-2000** (P10 千分位)。
- 无 task 需要排除; task 41 的布局依赖是跨 baseline 的 (32/32 run 全错), 登记在 summary, 是否剔除待 user 定。
