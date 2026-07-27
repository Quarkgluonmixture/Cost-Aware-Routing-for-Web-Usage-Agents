# B1 phantom_text (P-text) classifieds — /diag failure attribution digest

| field | value |
|---|---|
| **Run** | `B1_phantom_text_classifieds_20260605_194554_941872185_431169_R933` (manifest-bound authoritative) |
| **Condition** | `phase1_phantom_text_router_0` |
| **Site / Model / Mode** | classifieds / **B1 (Qwen3-VL-4B)** / **phantom_text (P-text)** = DOM-style prompt + `[SOM_MARKS]` 编号元素文本, **视觉盲 (无标注图)** |
| **N episodes** | 224 |
| **SR** | **7.6%** (17/224 success) |
| **ruleset_version** | `5-domsomvispsom-b1860coord` ⚠️ **discover-only — 未落码新规则, 未 bump version**; cross-mode 定量比较禁止直至 freeze |
| **Tier-2 深挖** | 43 episode / 7 sonnet sub-agent (25 no-hit 全覆盖 + 6 success-hit FP 审计 + 6 P4 axis-2 机制 + 6 P17/P19 verify) + **observation_dom.txt forensic** (task 20 P4 ground-truth) |
| **姊妹 condition** | §317 (B1 som) · §318 (B1 dom) · §320 (B1 vision) — B1 cls cross-mode discover, ptext = 第 4 mode |

---


## 0v8. v8 freeze 补记（2026-07-27）— cls 行为**不是**字节不变

`RULESET_VERSION` 升至 **`8-reddit-p41p46-b1890fix`**。该批规则源自 **reddit** discover，但有两处**确实改变了 cls 行为**，
均已逐条定性核实（不是回归）：

1. **B-1890 修复**：`P35`/`P39` 原先 guard 在 `effective_mutating_action_count`，而该字段从未被 runner
   填充、恒为 0 → guard 是 **no-op**，规则比其 docstring 声称的更宽松。v8 改为从 step record 派生突变计数。
   抽查确认被移除的旧命中确实有 6–8 个突变步（即**旧命中是错的**）。
2. **P33 正则扩展**：加入 reddit 的 `/submission_images/` 路径。cls 侧因此 **+1 例**（cls task 233 —— 它的
   `sites` 只写 classifieds，但 intent 实际要求"the characters in the image **on Reddit**"，
   该 episode 真的访问了 `localhost:9999`，旧正则漏检）。

本 condition 的 v8 数字 —— **跨 condition / 跨站聚合请用这一组**：

| 指标 | v8 |
|---|---|
| SR | **7.59%** (17/224) |
| failed + hit | 194 |
| **failed NO-hit** | **13** |
| success + hit | 1 |

v8 新规则 failed 侧: {'P43': 69, 'P44': 143, 'P45': 97, 'P46': 1}；success 侧: 无。
（`P43` 在 cls 上大量命中属预期 —— 它标记"intent 需要视觉 + 该 mode 无页面截图"这一**中性组合**，
并非预测失败；§387.10 实测补上截图的增益 ≈0。）

全部 36 个 canonical condition 现处同一版本 → **cross-mode / cross-site 聚合解锁**。

---
## 1. 三分类统计

| 类别 | 占比 (深挖样本) | 说明 |
|---|---|---|
| **agent-limit** | 压倒 (~40/43 深挖) | phantom-family 视觉盲 × 图像依赖任务 (主轴) + axis-2 格式 |
| **scaffold-bug** | 0 confirmed, **1 测量隐患候选** | `walk_fail` fallback 误报 success=True (P4 + task 136, 见 §3/§7) — 非 fatal bug, 测量层 |
| **benchmark-FP** | 0 | P19 组 0 真 FP (eval-type 误触发, 见 §6) |
| **rule-FP (presence-only)** | 大量 | success-hit 6/6 全误报 + P19 诊断 FP + finish-less artifact |

**一句话**: B1 ptext 失败 = phantom-family **视觉盲** (与 §319 pprompt "P34 视觉盲 phantom-family-wide" 一致) × 图像依赖任务, 叠加 **axis-2 格式特有的 P4 root-ref 瞎猜** (ptext 突出, vision=0)。scaffold 干净 (有 1 个 fallback 误报 success 的测量隐患), benchmark-FP 0。

---

## 2. Tier-1 规则分布 (failed-only, episode-level)

```
P31=136  P5=91  P19=69  P17=52  P4=41  P18=35  P14=31  P20=20  P10=19
P25=13  P13=11  P7=11  P12=8  P30=7  P33=6  P2=6  P24=4  P22=2  P23=2  P27=1
```
no-hit failed = **25** (coverage 87.9%); success-fire (FP源) = P5×2 P10×2 P14×2 P4×1 P17×1 P12×1 P31×1。

⚠️ **ptext 特有高频**: **P4=41 (根节点误操作)** —— vision mode 是 **0** · **P17=52 (click-back 振荡)** —— vision 是 14。这俩是 ptext discover 的核心新现象 (§3)。

---

## 3. 🔑 axis-2 finding — P4 root-ref 瞎猜 = ptext 格式特有失败 (本 diag headline, **ground-truthed**)

**现象**: deterministic 规则 **P4 (action 对 element_id∈{0,1} 或全 viewport bbox)** 在 ptext fire **41 个失败 episode**, 同模型 vision mode **0 个**。

**P4 sub-agent log+code 推断 (后被 ground-truth 修正)**: 读 `build_som_text_from_obs_text` (enumerate start=1) → 断言 "`[SOM_MARKS]` 从 1 编号, `[id=1]`=RootWebArea, agent 找不到搜索框时 default 到最小编号 [1]=root"。6/6 判 format-induced-default-to-root, 4/6 真死因 (root_click_count 高达 9, task 20/152 no_op_rate 80%)。

**observation_dom.txt + step 记录 ground-truth (§290 教训第四次实证, 修正 sub-agent)**:
- task 20 step 5 真实 observation: 搜索结果页 **RootWebArea = `[3377]`** (原始 AXTree node ID, **非 [1]**); 元素是 `[3833]`/`[3838]` 等高 ID; **页面确无搜索框** (textbox 不在 AXTree — search-result page header 不暴露 textbox ✓ sub-agent 这点对)。
- step 5 真实 action: `type element_id=1 text="white Xbox console"` thought="I should use the search bar"; `locator_route_meta = {success: False, fallback_used: True, error: 'walk_fail:no_input_within_walk'}`; `element_bbox = [0,0,10,10]` (退化); 但 step 报 **success=True**。
- ∴ **真机制 (修正版)**: 视觉盲 agent 想用搜索框, 但该页无搜索框 + 无截图看不见这点 → **瞎猜 emit `type [1]`** (low-default element_id, **不对应 observation 里任何真实元素**, 不是 [SOM_MARKS] renumber 的 [1]=root) → locator `walk_fail:no_input_within_walk` → 退化 fallback bbox[0,0,10,10] → type 没进任何输入框却 **误报 success=True**。

**结论 (paper-usable, 精准版)**:
> ptext 的 P4 高频 = **axis-2 (numbered-element-ref) × 视觉盲**的复合失败: 当目标元素 (搜索框) 不在当前页 AXTree、且 agent 无截图看不见这点时, 4B 模型在编号引用动作空间里**瞎猜 low element_id (1)**, locator 无法 resolve (`walk_fail`) → 退化 no-op。vision mode 因 agent 看得见页面布局 → P4=0。
> ⚠️ **NOT** "`[SOM_MARKS]` renumber 使 [1]=RootWebArea" (sub-agent 的 clean 故事, ground-truth 证伪: observation 用原始 AXTree ID, RootWebArea=[3377])。

**分类**: **agent-limit 主** (model emit 无效 ref, 视觉盲驱动) + **scaffold 测量隐患** (`walk_fail` fallback 报 success=True, type 实际落空 → 虚高 action_success; 与 task 136 同源 `walk_fail`→category loop)。**B-number 候选**: locator fallback 在 `walk_fail:no_input_within_walk` 时不应报 success=True (cross-ref `locator_dispatch`); 但属测量层非 fatal, post-fire 审。

---

## 4. Tier-2 — no-hit 子集 (25 task, 全 agent-limit) = phantom-family 视觉盲 × 图像依赖

0 scaffold (除 task 136 walk_fail 信号) / 0 benchmark-FP。主轴 = **视觉盲 × 图像依赖任务** (与 §319 pprompt P34 一致), 亚型:

| 亚型 | task | 死因 |
|---|---|---|
| **图内 OCR (结构性必败)** | 110 (游戏数) · 118 (手机时钟) · 119 (钞票面值) · 222 (卷尺读数) | 信息只在图片像素里, 无图必败 |
| **reference image 依赖** | 51 · 78 (城市照片) · 138 · 141 (exact item recall) | task_config 有 image 字段但 ptext 不传图 |
| **图内文字识别** | 173 · 199 (kaiyo.com 印在图里) | ⚠️ **新亚型**: agent 把 DOM 里的 localhost 页面 URL **误当"图片中的网站"** = DOM-URL vs image-text 混淆 (区别于 §319 pprompt 的 img-src-path 混淆) |
| **缩略图视觉选择** | 14 (第二行) · 80 (含书) · 188 (封面有婴儿) · 129 (图内价格) | gallery 行列空间 / 缩略图内容文本无法从扁平 AXTree 推 |
| **颜色过滤** | 35 · 43 (非黑灰 / 红色) | 颜色只在缩略图, 文本搜 "red" keyword 无效 |
| **条件分支 (依赖图判断)** | 213 (Elvis 有无观众 → email vs comment) | 视觉判断错 → 走错分支 → DOM 无 mutation |
| **字面搜索代替分类导航** | 26 (整 intent 当 sPattern) | 日期 facet 需 category drill-down |
| **walk_fail fallback category loop** | 136 | ⚠️ `walk_fail:no_input_within_walk` → fallback 误触 sCategory=3 → 同 URL 死循环 4+ 步 (同 P4 §3 根源) |

---

## 5. Tier-2 — success-hit FP 审计 (6/6 全 presence-only) + P31 第 4 mode confound

**全部 6 success episode 的 hit 都 hit_causal=false**:
- **task 153 = finish-less arrival artifact** (url_match, agent step 0 到达 item/14761, 此后 29 步 click eid=17 never finish, eval `agent_page` 模式凭当前 URL 自动 pass)。**P31 在 ptext = §317 som / §320 vision 同款 finish-less artifact** → **跨 4 mode confound 闭环** (som finish-less / dom 真卡死 / vision finish-less加剧 / **ptext finish-less**)。
- **task 44 P4 success-FP**: agent 已到达正确 URL, 后续 `type eid=1` (root) 是 post-arrival 困惑非死因 → **P4 也有 success-fire FP**, 需 success-safe 收窄。
- P10 FP (87, 93): url_match 任务 finish.answer=URL, 端口/itemid 数字被当"应记忆数字" (同 §320)。P14 FP (94): post-arrival 困惑。

---

## 6. Tier-2 — P17 / P19 verify

- **P17 (click-back 52, ptext 突出) = 多数是 visual-attribute-verification deadlock, 非纯 axis-2**: 4 verify 中 **2 axis-2 相关** (task 12 图片元素瞎点 / task 22 gallery 空间编号), **2 纯视觉盲死锁** (task 13/17)。**task 17 典型**: agent 到达正确 item 79747 (是 reference 答案) 但**视觉盲无法确认 "red handlebars" 属性 → 反复 back 不敢 finish → 失败**。= 干净的"换 SoM/Vision 给视觉确认能救"的 routing 候选证据 (但与 P5 不同, 这是 attribute-verification 不是 navigation)。
- **P19 (69) = 0 真 benchmark-FP, eval-type 误触发**: 2 verify (task 23/28) eval 实际是 string_match / program_html **非 url_match** → P19 误 fire (同 §320 vision 发现)。**P19 需加 `eval_type==url_match` guard**。

---

## 7. 🔁 Self-evolving — 提议 P-rule (discover-only, 留合并 freeze step)

> ⚠️ **本 run 不落码** (与 §317/§318/§320 同步)。下列与 B1 dom/som/vision 提议**合并去重**后统一 freeze step 落码 + bump version + 全量重扫。

**新规则候选**:
1. **WALK_FAIL_DEGENERATE (axis-2 + 测量, 强信号)**: `locator_route_meta.error contains 'walk_fail'` (+ element_bbox≈[0,0,10,10] 或 element_id∈{0,1}) — 覆盖 P4 root-ref 瞎猜 (task 20/152) + task 136 category loop。**0-token, locator_route_meta 直读**。⚠️ 兼含 B-number 候选 (success=True 误报)。
2. **VISUAL_BLIND_IMAGE_TASK (phantom-family 通用)**: `obs_mode in phantom_* AND (task_config.image != null OR intent 含 'in the image'/'picture'/'cover'/'looks like this')` → agent-limit 预期失败。合并 §319 pprompt P34 family-wide。
3. **DOM_URL_AS_IMAGE_TEXT (新亚型)**: `intent 含 'address/website in the image' AND finish_answer 含 'localhost' AND reference 含真实域名` — task 173/199。
4. **MUTATION_MISSING** (§318 P35 / §320 收敛, 跨 mode 稳健): program_html + `.comments_list`/item_edit + effective_mutating=0 — task 213。

**现有规则收窄 (success-safe)**: **P4** success=True 静默 (task 44) · **P31** finish-less url_match 豁免 (task 153) · **P10** finish.answer=URL 跳数字比对 · **P19** 加 `eval_type==url_match` guard · **P14** arrival productive 首步豁免。

---

## 8. 代表 episode

| 类 | task | 一句话 |
|---|---|---|
| **axis-2 P4 root-ref (headline, ground-truthed)** | **20 / 152** | 视觉盲找不到搜索框 → 瞎猜 `type [1]` → walk_fail 退化 no-op (root_click 9×) |
| **visual-attribute-verify deadlock (routing 候选)** | **17** | 到达正确 item 但视觉盲确认不了 "red handlebars" → 反复 back 不敢 finish → 失败 |
| **finish-less artifact (FP, 第4 mode)** | 153 | step0 到达正确 URL → 29 步空转 → url_match 自动 pass, P31 误报 |
| **视觉盲图内 OCR** | 119 / 118 | 钞票面值 / 手机时钟在图里, 无图必败 |
| **DOM-URL vs image-text 混淆 (新亚型)** | 173 / 199 | 把 localhost 页面 URL 当"图片中的网站", 应为 kaiyo.com |
| **walk_fail category loop** | 136 | `walk_fail` fallback 误触 sCategory → 同 URL 死循环 |

---

## 9. Actionable

- **B-number 候选 (测量层, post-fire)**: locator fallback 在 `walk_fail:no_input_within_walk` 时报 `success=True` (type 落空仍算成功 → 虚高 action_success)。cross-ref `locator_dispatch` + `master_bug_catalog`。非 fatal, 不阻 fire。
- **benchmark-FP / exclude**: 无新增 (P19 组 0 真 FP)。
- **routing 证据**: task 17 visual-attribute-verification deadlock = 与 §320 task 40 (text-in-image rescue) 不同的第二类 routing-rescuable failure (属性确认型) → freeze 后 cross-mode 验"换 SoM/Vision 给视觉确认能否救活"。
- **freeze step TODO**: 合并 B1 dom(§318)+som(§317)+vision(§320)+ptext 四 condition 规则提议 → 去重 (MUTATION_MISSING / WALK_FAIL / VISUAL_BLIND 收敛) → 落码 + bump version + 全量重扫 → 解锁 cross-mode 定量。**cross-mode 定量仍禁** (6-mode 未齐, 剩 psom 跑中 R26199)。
