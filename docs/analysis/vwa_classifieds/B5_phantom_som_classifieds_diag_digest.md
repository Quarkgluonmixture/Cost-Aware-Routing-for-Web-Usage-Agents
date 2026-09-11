# B5 phantom_som classifieds — /diag failure attribution digest

> **per-condition** (site × model × mode)。B5 = GPT-5.6 (`gpt-5.6-terra` via AWS proxy, `structured_output: response_format`)。
> **B5 不在预注册 cell 集合里** —— `run_manifest.yaml` 的 `extension:` 节 (笔记 §509)。本 digest 的数字可与 B0/B1/B2 的 v11 数字块并表 (同 ruleset),
> 但**不进**任何 pooled / K-of-N / forest 统计。跨 mode 共性、规则缺陷与提议集中在 [`B5_classifieds_cross_mode_diag_summary.md`](B5_classifieds_cross_mode_diag_summary.md)。

## 1. Header

| 字段 | 值 |
|---|---|
| Run | `B5_phantom_som_classifieds_20260829_201900_676822651_4026859_R18439` (manifest `extension:` 条目) |
| Condition | `phase1_phantom_som_router_0` |
| Site / Model / Mode | classifieds / **B5** / **phantom_som / P-SoM ([SOM_MARKS] + SoM 风格 prompt, 无图)** |
| N episodes | 224 |
| SR | **22.77%** (51/224) |
| **ruleset_version** | **`11-intent-text-fallback`** (与 48 个既有 condition 同版本) |
| 三子集 | failed+hit 136 · **failed NO-hit 37** · success+hit 7 |
| Tier-2 | no-hit **37/37 全覆盖** + hit 审计 1 ep; sonnet, 按 task 跨 mode 分批, 喂压缩轨迹 + 静态 catalog 的 item 查询 |
| Diag date | 2026-09-11 (B5 首次 diag) |

## 2. 三分类统计

| 类别 | 数 | 说明 |
|---|---:|---|
| **agent-limit** | 36 (no-hit) + Tier-1 命中的 136 | Tier-1 命中的全部是 agent-limit 类规则 |
| **scaffold-bug** | 0 | 无; 另: 本 condition 有 **50** 步 `multiple_actions` (其中 42 步页面变了), 全部落在 B-1998 的缝里 |
| **benchmark-FP 嫌疑** | 1 | task 41 (「第二行」依赖布局, 见 cross-mode summary §3) |
| unclear | 0 | — |

⚠️ **P31 漏标**: 本 condition 有 **29** 个失败 episode 跑满预算 (`trajectory_incomplete`) 却没有 P31 —— B-1999 (P31 豁免在 cls 上恒成立)。
读 P31 行时按「下界」读。

## 3. Tier-1 规则分布 (v11, 全 224 ep)

| 规则 | 含义 | step 级 | episode 级 | 其中 failed | 其中 success |
|---|---|---:|---:|---:|---:|
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 65 | 65 | 65 | 0 |
| `P17` | click-back振荡 | 51 | 51 | 51 | 0 |
| `P31` | budget耗尽未完成 | 50 | 50 | 50 | 0 |
| `P33` | 导航至裸图片URL幻觉 | 43 | 43 | 40 | 3 |
| `P10` | 跨步数值记忆失败 | 19 | 17 | 12 | 5 |
| `P36` | WALK_FAIL_DEGENERATE | 43 | 15 | 15 | 0 |
| `P20` | 评测目标页从未访问 | 14 | 14 | 14 | 0 |
| `P7` | sCity=州名 | 13 | 12 | 11 | 1 |
| `P30` | 到达正确item后离开 | 9 | 9 | 9 | 0 |
| `P25` | 跨站任务跳过其中一站 | 8 | 8 | 8 | 0 |
| `P5` | 感知缺失循环 | 7 | 7 | 7 | 0 |
| `P18` | cheapest漏价格排序 | 5 | 5 | 5 | 0 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 3 | 3 | 3 | 0 |
| `P14` | URL 自环 | 4 | 3 | 3 | 0 |
| `P28` | benchmark-FP货币tokenize | 3 | 3 | 3 | 0 |
| `P13` | 搜索代替浏览 | 3 | 3 | 3 | 0 |
| `P37` | URL_HALLUCINATION | 3 | 3 | 3 | 0 |
| `P22` | 图上数字dom不可读 | 2 | 2 | 2 | 0 |
| `P11` | 最新+地点组合 | 1 | 1 | 1 | 0 |
| `P12` | 从不翻页 | 1 | 1 | 1 | 0 |
| `P46` | COMMENT_INTENT_NO_TYPE | 1 | 1 | 1 | 0 |

> ⚠️ 解读约束 (同 v11 数字块): ① 症状分布, 不是死因分布 (P36/P31 是 risk-marker); ② `P2`/`P4` 依赖 `element_bbox`;
> ③ `P10` 在 B5 上主要是误报: 成功 episode 也有 30 个触发 (B1 只有 1 个), 审计 4/4 非死因; 其中千分位拆分缺陷 (B-2000, `$6,400` 被读成 6 和 400)
>    只解释 B5 失败侧 61→50, 成功侧 31→30 基本不变 ⇒ 主因是日期 / 型号数字与价格混比; ④ `P31` 见上方 B-1999。

## 4. Tier-2 深挖

| no-hit 子类 | 数 | 含义 |
|---|---:|---|
| `keyword-literal-proxy` | 11 | 把字面关键词搜索当作语义/视觉判断 (参考 item 标题里没有这个词) |
| `wrong-item-visual-match` | 6 | 视觉/颜色/外观判断选错 item |
| `loop` | 5 | 搜索或滚动循环, 跑满 30 步 (P31 因 B-1999 没标出) |
| `wrong-item-ordering` | 2 | 没用 / 用错排序 (most recent / cheapest) |
| `no-visual-channel` | 2 | 该 mode 无图, 任务要看页面图像 (结构性不可解) |
| `stated-uncertainty-then-finish` | 2 | thought 说要核实, 下一步不核实就 finish |
| `constraint-dropped` | 2 | 丢了类目 / 物品类型 / 排序约束 |
| `other` | 2 | 其他 (见下方代表 episode) |
| `answered-wrong-value` | 2 | string_match 答错数值 / 计数 / 属性 |
| `benchmark:task41` | 1 | benchmark-FP 嫌疑: task 41「第二行」依赖布局 |
| `misread-intent` | 1 | 读错 intent 的语义 |
| `given-page-scope-abandoned` | 1 | intent 指「这一页」, 第一步就搜索离开给定页 |

**Tier-2 (37/37 no-hit 全覆盖)**: 字面关键词 11 + 视觉判断 6 + 循环 5; 排序错 2 (t35 全程没有 `sOrder=dt_pub_date` 却答「最近两条」)。
- t123: 本 mode 无图, thought 却写「那辆 1970 Ford Ranchero 明确是黄色车漆」—— 看不见的东西被说成看见了 (编造, 不是读错)。
- t86: 在 item 页与 item_edit 页之间来回 + scroll >8 轮找「颜色」, 而 Osclass 里颜色没有文本字段。

**hit 审计** (success-hit / failed-hit 抽样):

| task | 类型 | 规则 | 真死因? | 依据 |
|---|---|---|---|---|
| 62 | SUCCESS-HIT | `P10` | 否 | thought 750.0 与最终答案中的 '$750.00' 一致；output num 2.0 来自标题里的型号数字'Spider-Man 2'，属于不同语义实体，并非价格被记错。 |

## 5. 代表 episode

- **t188** [`keyword-literal-proxy`] The task requires visually spotting which book COVER PHOTO depicts a baby, but the agent runs a literal text search for the word 'baby' against listing titles/descriptions. The reference item's title ('Christmas in Ameri  
  证据: step_0: type 'baby' search; step_1: switch to gallery view; step_2: click id=9538 (same pregnancy/toddler book as phantom_text); step_3: finish -- converges with phantom_text on the same wrong item, r
- **t210** [`keyword-literal-proxy`] Same 'lamb' -> 'Lambs Ear' text-collision failure as som/phantom_prompt: cheapest keyword hit taken as answer without verifying it is livestock.  
  证据: step_4: finish answer 'Cheapest lamb listing: Lambs Ear plants Perennials Groundcover — $6.00.' at url=index.php?page=item&id=32759 — identical wrong item across all 3 non-dom modes for this task.
- **t117** [`wrong-item-visual-match`] Same keyword-search-as-color-proxy shortcut as dom/som/phantom_text (select Bikes, sort by price, then search 'blue', click first result, finish) without any visual color verification.  
  证据: step_2: sort 'Lower price first'; step_3: type 'blue'; step_4-5: click id=59840 then finish.
- **t141** [`wrong-item-visual-match`] Agent matches the task's reference photo only at coarse category level (thought: 'reference image shows a carton of eggs') and issues the generic keyword search 'eggs' instead of verifying the specific pictured item's id  
  证据: step_0-1: select 'Farm + garden', search 'eggs'; step_2: click id=35914; step_3-4: returns to results and re-searches 'eggs' sorted by date to 'verify most recent'; step_5: re-clicks the SAME id=35914
- **t90** [`loop`] Agent repeatedly re-issues the literal keyword search "discoloring"/"discolor" against Cars+trucks (0 hits every time), resets via the category breadcrumb, and repeats the identical cycle instead of switching to visual/g  
  证据: step_0: type 'discoloring' -> 0 hits; step_1: click category breadcrumb (element_id 9); step_2: type 'discoloring' again (element_id 12) -> identical click(9)/type(12) cycle repeats (incl. PARSE_INVAL
- **t86** [`loop`] Same structural failure again: phantom_som has [SOM_MARKS] text but no actual item screenshots, so color remains unreadable; agent cycles between item&id=84144 and item&action=item_edit&id=84144 more than any other mode'  
  证据: steps 10-29: cycles between item&id=84144 and item&action=item_edit&id=84144 with scroll actions >8 times ('I need inspect this Toyota listing's details to determine its color'); final agent URL at st
- **t97** [`wrong-item-ordering`] Same literal-'animal'-keyword blind spot as phantom_prompt, with even less exploration (11 steps, a single scroll cycle) before settling on the same 2023/11/13 keyword-matched item, never verifying whether newer non-keyw  
  证据: step_9-10: clicks id=71975 ('the top current result is the most recent listing explicitly featuring animal figures') and finishes immediately, without ever loading an unfiltered/date-only view of the 
- **t35** [`wrong-item-ordering`] 与 som 模式同一问题——全程无 sOrder=dt_pub_date，仅凭默认顺序点开前两条('红色'、'charcoal 除外')作为'最近两条'，实际选中 id=70185/49465 均非参考答案(33441/42263)，排序假设有误导致选错条目。  
  证据: step_0/1/2/6 obs_url 均为 index.php?page=search&sPattern=loveseat&sCategory=17，无 sOrder 参数；step_7 finish 给出 links [70185, 49465]，与参考 [33441, 42263] 完全不重合。
- **t123** [`no-visual-channel`] phantom_som has no observation image ([SOM_MARKS] text + SoM-style prompt, no picture); agent claims a 1970 Ford Ranchero listing has a 'yellow paint scheme' that is 'visible' -- language inconsistent with a mode that ne  
  证据: step_3: click element_id=10 -> item&id=22391 | thought 'the visible 1970 Ford Ranchero listing explicitly has a yellow paint scheme, so opening it navigates to a matching item.' -- 'visible' is not so
- **t11** [`no-visual-channel`] Same structural limitation as phantom_text: no image channel, so color cannot be determined; agent switched to gallery display (a layout toggle, not a filter) and opened the first item shown, landing on the same wrong it  
  证据: step_0: click gallery toggle 'Switching to gallery view will make it easier to identify the first bike that is blue visually' (impossible without image data in this mode); step_1: click id=76502; step

## 6. 失败桶 (reason_bucket 五桶, `failure_modes_per_cell.md` Extension cells)

N=224, failed=173

| 失败桶 | 数 | 占 failed | 占全部 |
|---|---:|---:|---:|
| early-finish/wrong-commit | 94 | 54.3% | 42.0% |
| visual-hijack/click-loop | 34 | 19.7% | 15.2% |
| search-loop | 24 | 13.9% | 10.7% |
| max-steps-other | 21 | 12.1% | 9.4% |

## 7. Self-evolving 与 actionable

- 本 condition **不单独提规则**: 所有提议在 cross-mode summary §5 统一给出, 每条带 0-token 全量复核结果 (两条最常被 sub-agent 提的「字面关键词」「放弃给定页」复核后**都不 success-safe**, 只当风险标记)。
- scaffold → **B-1998** (`multiple_actions` 执行却按 wait 记账); diag 规则 → **B-1999** (P31) · **B-2000** (P10 千分位)。
- 无 task 需要排除; task 41 的布局依赖是跨 baseline 的 (32/32 run 全错), 登记在 summary, 是否剔除待 user 定。
