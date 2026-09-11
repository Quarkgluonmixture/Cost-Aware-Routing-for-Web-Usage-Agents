# B5 phantom_prompt classifieds — /diag failure attribution digest

> **per-condition** (site × model × mode)。B5 = GPT-5.6 (`gpt-5.6-terra` via AWS proxy, `structured_output: response_format`)。
> **B5 不在预注册 cell 集合里** —— `run_manifest.yaml` 的 `extension:` 节 (笔记 §509)。本 digest 的数字可与 B0/B1/B2 的 v11 数字块并表 (同 ruleset),
> 但**不进**任何 pooled / K-of-N / forest 统计。跨 mode 共性、规则缺陷与提议集中在 [`B5_classifieds_cross_mode_diag_summary.md`](B5_classifieds_cross_mode_diag_summary.md)。

## 1. Header

| 字段 | 值 |
|---|---|
| Run | `B5_phantom_prompt_classifieds_20260829_015238_489893385_3902862_R10294` (manifest `extension:` 条目) |
| Condition | `phase1_phantom_prompt_router_0` |
| Site / Model / Mode | classifieds / **B5** / **phantom_prompt / P-prompt (AXTree + SoM 风格 prompt, 无图)** |
| N episodes | 224 |
| SR | **21.88%** (49/224) |
| **ruleset_version** | **`11-intent-text-fallback`** (与 48 个既有 condition 同版本) |
| 三子集 | failed+hit 138 · **failed NO-hit 37** · success+hit 13 |
| Tier-2 | no-hit **37/37 全覆盖** + hit 审计 3 ep; sonnet, 按 task 跨 mode 分批, 喂压缩轨迹 + 静态 catalog 的 item 查询 |
| Diag date | 2026-09-11 (B5 首次 diag) |

## 2. 三分类统计

| 类别 | 数 | 说明 |
|---|---:|---|
| **agent-limit** | 36 (no-hit) + Tier-1 命中的 138 | Tier-1 命中的全部是 agent-limit 类规则 |
| **scaffold-bug** | 0 | 无; 另: 本 condition 有 **49** 步 `multiple_actions` (其中 46 步页面变了), 全部落在 B-1998 的缝里 |
| **benchmark-FP 嫌疑** | 1 | task 41 (「第二行」依赖布局, 见 cross-mode summary §3) |
| unclear | 0 | — |

⚠️ **P31 漏标**: 本 condition 有 **30** 个失败 episode 跑满预算 (`trajectory_incomplete`) 却没有 P31 —— B-1999 (P31 豁免在 cls 上恒成立)。
读 P31 行时按「下界」读。

## 3. Tier-1 规则分布 (v11, 全 224 ep)

| 规则 | 含义 | step 级 | episode 级 | 其中 failed | 其中 success |
|---|---|---:|---:|---:|---:|
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 67 | 67 | 67 | 0 |
| `P17` | click-back振荡 | 60 | 60 | 60 | 0 |
| `P31` | budget耗尽未完成 | 46 | 46 | 46 | 0 |
| `P33` | 导航至裸图片URL幻觉 | 44 | 44 | 39 | 5 |
| `P10` | 跨步数值记忆失败 | 23 | 19 | 12 | 7 |
| `P20` | 评测目标页从未访问 | 15 | 15 | 15 | 0 |
| `P7` | sCity=州名 | 17 | 15 | 13 | 2 |
| `P36` | WALK_FAIL_DEGENERATE | 34 | 11 | 11 | 0 |
| `P30` | 到达正确item后离开 | 7 | 7 | 7 | 0 |
| `P14` | URL 自环 | 8 | 6 | 6 | 0 |
| `P5` | 感知缺失循环 | 5 | 5 | 5 | 0 |
| `P37` | URL_HALLUCINATION | 3 | 3 | 3 | 0 |
| `P18` | cheapest漏价格排序 | 2 | 2 | 2 | 0 |
| `P22` | 图上数字dom不可读 | 2 | 2 | 2 | 0 |
| `P28` | benchmark-FP货币tokenize | 1 | 1 | 1 | 0 |
| `P11` | 最新+地点组合 | 1 | 1 | 1 | 0 |
| `P44` | HALLUCINATED_ELEMENT_REF | 1 | 1 | 1 | 0 |
| `P13` | 搜索代替浏览 | 1 | 1 | 1 | 0 |
| `P46` | COMMENT_INTENT_NO_TYPE | 1 | 1 | 1 | 0 |
| `P25` | 跨站任务跳过其中一站 | 1 | 1 | 1 | 0 |

> ⚠️ 解读约束 (同 v11 数字块): ① 症状分布, 不是死因分布 (P36/P31 是 risk-marker); ② `P2`/`P4` 依赖 `element_bbox`;
> ③ `P10` 在 B5 上主要是误报: 成功 episode 也有 30 个触发 (B1 只有 1 个), 审计 4/4 非死因; 其中千分位拆分缺陷 (B-2000, `$6,400` 被读成 6 和 400)
>    只解释 B5 失败侧 61→50, 成功侧 31→30 基本不变 ⇒ 主因是日期 / 型号数字与价格混比; ④ `P31` 见上方 B-1999。

## 4. Tier-2 深挖

| no-hit 子类 | 数 | 含义 |
|---|---:|---|
| `keyword-literal-proxy` | 10 | 把字面关键词搜索当作语义/视觉判断 (参考 item 标题里没有这个词) |
| `wrong-item-visual-match` | 6 | 视觉/颜色/外观判断选错 item |
| `loop` | 5 | 搜索或滚动循环, 跑满 30 步 (P31 因 B-1999 没标出) |
| `constraint-dropped` | 4 | 丢了类目 / 物品类型 / 排序约束 |
| `stated-uncertainty-then-finish` | 4 | thought 说要核实, 下一步不核实就 finish |
| `other` | 2 | 其他 (见下方代表 episode) |
| `answered-wrong-value` | 2 | string_match 答错数值 / 计数 / 属性 |
| `wrong-item-ordering` | 1 | 没用 / 用错排序 (most recent / cheapest) |
| `no-visual-channel` | 1 | 该 mode 无图, 任务要看页面图像 (结构性不可解) |
| `benchmark:task41` | 1 | benchmark-FP 嫌疑: task 41「第二行」依赖布局 |
| `given-page-scope-abandoned` | 1 | intent 指「这一页」, 第一步就搜索离开给定页 |

**Tier-2 (37/37 no-hit 全覆盖)**: 与 P-text 同形, 「说了要核实却直接交」最多 (4 个: t168 / t207 …), 丢约束 4 个。
- t117: 全程 URL 里没有 `sCategory=7` (Bikes), 变成全站搜 `blue bike`。
- t46: 从参考图把型号认成「FX Cruiser」(实际 FX140), 之后 30 步都在精炼搜索词 + 翻页横跳, 一次 item 页都没开。
- t35: thought 计划「先搜关键词再筛 Furniture」, 下一步直接 finish, 两步结束。

**hit 审计** (success-hit / failed-hit 抽样):

| task | 类型 | 规则 | 真死因? | 依据 |
|---|---|---|---|---|
| 139 | SUCCESS-HIT | `P33` | 否 | phantom_prompt 无图；step7 打开裸图后 step8 立即 back，未从图片提取任何具体断言，最终答案由文本搜索'Babe Ruth'匹配得出，裸图事件是无效死路但没有编造视觉内容。 |
| 1 | FAILED-HIT | `P17` | 否 | 往返的是两个不同候选 item(7846、864)而非同一 item 空转，属于正常的逐个比价查看；真正失败原因是 phantom_prompt 无图无法验证车辆颜色，只能靠价格猜测选错了 item，与'thrash'本身无直接因果。 |
| 88 | FAILED-HIT | `P33` | 是 | phantom_prompt 无图；step23-25 打开的正是参考答案 item 50736(P30 同时命中'reached reference then left')，但 step26 back 后 thought 断言'does not meet the red-vehicle-with-trees cond |

## 5. 代表 episode

- **t188** [`keyword-literal-proxy`] The task requires visually spotting which book COVER PHOTO depicts a baby, but the agent runs a literal text search for the word 'baby' against listing titles/descriptions. The reference item's title ('Christmas in Ameri  
  证据: step_0: type 'baby' search; step_1: click id=64537 ('Children's Disney + Seuss Books'); step_2: finish (thought: 'the identified book listing whose cover includes a baby') -- no actual cover was inspe
- **t210** [`keyword-literal-proxy`] Same collision as som mode: searched 'lamb', sorted ascending price, opened the cheapest hit 'Lambs Ear plants...' ($6) without checking it was a plant not livestock; converged on the identical wrong item id=32759 as som  
  证据: step_3: finish answer 'Lambs Ear plants Perennials Groundcover — $6.00' at url=index.php?page=item&id=32759, same wrong item as som/phantom_som modes.
- **t66** [`wrong-item-visual-match`] Same sort-by-price-ascending-then-click-top-result pattern, again id=69961, again claiming without verification that the listing 'explicitly includes an NFL football game' -- no description text is available to confirm t  
  证据: step_2: click element_id=2891 -> item&id=69961 | thought 'The lowest-priced video gaming listing that explicitly includes an NFL football game is PlayStation 2 Games for $2, so I will open its detail 
- **t141** [`wrong-item-visual-match`] Agent matches the task's reference photo only at coarse category level (thought: 'reference image shows a carton of eggs') and issues the generic keyword search 'eggs' instead of verifying the specific pictured item's id  
  证据: step_0-1: select 'Farm + garden', search page loads; step_2: type 'eggs' sorted by newest; step_3: click id=35914 (thought: 'matches the reference image and requested Farm+garden item'); step_4: finis
- **t218** [`loop`] Alternated scroll up/down 8 times (steps 5-13) re-scanning the same price-sorted list without ever opening the higher-priced candidates to verify relevance, repeating near-identical thoughts each time ('need inspect ...   
  证据: steps 5-13: scroll down/up/down/up/down/up/down/up/down (8 alternations, no click/type interleaved) each with a near-duplicate thought about needing to 'inspect' the list; step_14: click id=27050; ste
- **t46** [`loop`] 从参考图误判具体型号为 'Yamaha FX Cruiser'(实际参考条目是 'Yamaha FX140')，此后 30 步全部耗费在反复精炼搜索词(jet ski→yamaha→yamaha jet ski→Yamaha FX Cruiser→2007 Yamaha FX Cruiser→引号精确匹配→FX Cruiser→加 sCategory=8→分页 iPage=2↔iPage=1 反复横跳)却从未打开任何一条 item 详情  
  证据: step_3 type 'Yamaha FX Cruiser\n' thought 'The reference is a Yamaha FX-series personal watercraft...'; step_17/22/25/27 反复在 iPage=1/iPage=2 之间用 click/goto/back 横跳；全 30 步无一次 url 含 page=item。
- **t117** [`constraint-dropped`] Never restricted the search to the Bikes category (sCategory param is absent from every URL in this trajectory), instead searching 'blue'/'blue bike' site-wide; it explicitly recognized it should check the listing image   
  证据: step_0-6: no step's URL ever contains sCategory=7; step_5: click element_id 4772 (changed=False) with thought 'I should inspect the listing image to verify its color before concluding'; step_6: finish
- **t55** [`constraint-dropped`] Same conflation as the dom episode on this task (invention/aircraft imagery mistaken for 'depicts the inventors'); converges on the identical wrong item (id=64447), one day older than the true most-recent match.  
  证据: step_2: click id=64447 "the ceramic stein listing ... explicitly depicts the Wright Bros, the inventors of the airplane shown"
- **t168** [`stated-uncertainty-then-finish`] 用'pink motorcycle'关键词+价格升序找到$1条目，自己在上一步已明确说需要核实该条目是否真的是粉色，但该 mode 无图，下一步没做任何验证动作就直接 finish  
  证据: step_5: "The results are sorted by lowest price, but the first listing needs inspection to verify whether its motorcycle is actually pink rather than merely matching the keyword broadly." step_6: fini
- **t207** [`stated-uncertainty-then-finish`] Drops the color-matching constraint and treats 'cheapest headphones by keyword search' as the answer; thought at step_10 says it will inspect the item 'for its colors before considering higher-priced options' but the ver  
  证据: step_10: click item id=43172 ($8) 'inspect the $8 headphones listing for its colors before considering higher-priced options'; step_11: finish immediately, no verification step taken

## 6. 失败桶 (reason_bucket 五桶, `failure_modes_per_cell.md` Extension cells)

N=224, failed=175

| 失败桶 | 数 | 占 failed | 占全部 |
|---|---:|---:|---:|
| early-finish/wrong-commit | 99 | 56.6% | 44.2% |
| visual-hijack/click-loop | 31 | 17.7% | 13.8% |
| max-steps-other | 23 | 13.1% | 10.3% |
| search-loop | 22 | 12.6% | 9.8% |

## 7. Self-evolving 与 actionable

- 本 condition **不单独提规则**: 所有提议在 cross-mode summary §5 统一给出, 每条带 0-token 全量复核结果 (两条最常被 sub-agent 提的「字面关键词」「放弃给定页」复核后**都不 success-safe**, 只当风险标记)。
- scaffold → **B-1998** (`multiple_actions` 执行却按 wait 记账); diag 规则 → **B-1999** (P31) · **B-2000** (P10 千分位)。
- 无 task 需要排除; task 41 的布局依赖是跨 baseline 的 (32/32 run 全错), 登记在 summary, 是否剔除待 user 定。
