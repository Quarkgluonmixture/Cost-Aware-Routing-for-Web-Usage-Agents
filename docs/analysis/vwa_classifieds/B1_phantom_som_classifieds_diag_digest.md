# B1 phantom_som (P-SoM) classifieds — /diag failure attribution digest

| field | value |
|---|---|
| **Run** | `B1_phantom_som_classifieds_20260606_165421_042595838_568395_R26199` (manifest-bound authoritative) |
| **Condition** | `phase1_phantom_som_router_0` |
| **Site / Model / Mode** | classifieds / **B1 (Qwen3-VL-4B)** / **phantom_som (P-SoM)** = SoM-style prompt + `[SOM_MARKS]` 编号元素文本, **视觉盲 (无标注图)**; paper **HERO / deployment-representative** mode |
| **N episodes** | 224 |
| **SR** | **6.7%** (15/224 success) |
| **ruleset_version** | `5-domsomvispsom-b1860coord` ⚠️ **discover-only — 未落码新规则, 未 bump version**; cross-mode 定量比较禁止直至 freeze |
| **Tier-2 深挖** | 50 episode / 7 sonnet sub-agent (28 no-hit 全覆盖 + 9 success-hit FP 审计 + 6 P4 axis-2 + 7 P33 verify) + **observation_dom.txt forensic** (task 8 P4 ground-truth) |
| **姊妹 condition** | §317 som · §318 dom · §320 vision · §321 ptext — B1 cls cross-mode discover, psom = 第 5 mode (剩 pprompt R32516 跑中, 完则 6-mode 齐 → freeze) |

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
| SR | **6.70%** (15/224) |
| failed + hit | 193 |
| **failed NO-hit** | **16** |
| success + hit | 0 |

v8 新规则 failed 侧: {'P43': 69, 'P45': 103, 'P44': 9, 'P46': 3}；success 侧: 无。
（`P43` 在 cls 上大量命中属预期 —— 它标记"intent 需要视觉 + 该 mode 无页面截图"这一**中性组合**，
并非预测失败；§387.10 实测补上截图的增益 ≈0。）

全部 36 个 canonical condition 现处同一版本 → **cross-mode / cross-site 聚合解锁**。

---
## 1. 三分类统计

| 类别 | 占比 (深挖样本) | 说明 |
|---|---|---|
| **agent-limit** | 压倒 (~45/50) | phantom-family 视觉盲 × 图像依赖 (主轴) + axis-2 P4 + P33 PNG 幻觉 |
| **scaffold-bug** | 0 confirmed | walk_fail success=True 测量隐患已 **B-1869** 登记 (本 run 加 async-DOM 误归因假设, 见 §3) |
| **benchmark-FP** | 0 | — |
| **rule-FP (presence-only)** | 大量 | P4 success 4 ep / P31 finish-less 2 ep / P5/P14/P12/P17 |

**一句话**: P-SoM (HERO mode) 失败 = **视觉盲** (image-dependent task 结构性必败, 与 ptext §321 / pprompt §319 同) + **3 个 phantom/format 特有成本**: ① axis-2 P4 root-ref 瞎猜 (比 ptext 更频, SoM-prompt click-priming) ② P33 img-href→裸 PNG 幻觉 (B1 比 B0 更差) ③ finish-less artifact (点缩略图 element 空转)。这些是 paper "cheapest-but-blind" HERO mode 的真实代价。

---

## 2. Tier-1 规则分布 (failed-only, episode-level)

```
P31=126  P5=96  P19=67  P4=63  P17=48  P14=38  P18=31  P20=20  P25=13
P7=13  P12=13  P10=11  P30=10  P33=7  P13=7  P27=7  P24=5  P2=3  P23=3  P22=2
```
no-hit failed = **28** (coverage 86.6%); success-fire (FP源, hit-level) = **P4×24 (=4 ep)** · P5×7 P14×4 P12×2 P31×2 P17×1。

⚠️ **psom 特有高频**: **P4=63 (比 ptext 41 更高)** + P33=7 (psom 签名)。P4 success-fire 24 hit (=4 episode 各反复点 ghost [1])。

---

## 3. 🔑 axis-2 P4 phantom-family-wide (RESOLVED ground-truth, 解 §321 开放点)

**现象**: P4 (action 对 element_id∈{0,1}) 在 psom fire **63 failed + 4 success-ep (24 hit)** > ptext (41+1) > vision (0)。

**ground-truth (task 8 observation_dom.txt, 解 §321 两难)**:
- psom observation **用原始 AXTree node ID**: RootWebArea=`[2]`, textbox=`[140]`, **无 `[1]` 或 `[0]`** (与 ptext §321 一致: ptext RootWebArea=[2]/[3377])。
- ∴ agent emit `element_id=1` = **幻觉式 low-default** (映射到不存在元素), **NOT** renumber 后的 root。**§321 留的 "要么 renumber-root 要么 hallucinated" 两难 → 确定是 hallucinated** (两 mode observation 都无 [1])。多 sub-agent (§321+§322) 口称 "[1]=RootWebArea" 全 log-only 误判 = **§290 教训第 5 次**。

**机制 (agent F locator_route_meta 证据)**: 视觉盲 agent 想操作但目标不在可见编号里 → 瞎猜 `[1]` → walk_fail → 退化 no-op。**两变体**:
- **`walk_fail:no_input_within_walk`** (type [1], 想用搜索框, 同 ptext §321): 4/6 verify, 全 run 63 task。
- **`walk_fail:no_actionable_within_walk`** (click [1], **psom 特有**): 2/6 verify, 全 run 22 task。psom 的 **SoM-style prompt 把 "[N]" 暗示成可点击标记** → 诱导 agent 点 [1] (ptext 不给此暗示 → 只 type)。**= 干净的 prompt-axis 证据: SoM-prompt 放大 root-ref 误点**。

**结论 (paper-usable)**:
> axis-2 P4 root-ref (视觉盲 agent 瞎猜 low-default element_id → walk_fail no-op) = **phantom-family-wide (ptext + psom)**, 非 ptext-specific (§321 扩写)。psom **更频** 因 SoM-style prompt 把编号暗示成可点击标记 (click 变体)。vision=0 因看得见页面。**分类 agent-limit** (model emit 无效 ref) + **B-1869 测量隐患** (walk_fail 报 success=True)。

**B-1869 refinement (本 run 加)**: agent F 读 `state_change.py` 假设 task 4/8/9 的 `success=True + page_changed=True` (walk_fail 下) = classifieds 后台**异步 DOM 更新** (分页懒加载/hover) 被 `detect_page_state_change` 误归因给 agent 动作。**plausible code-reasoning 假设** (非 artifact 直证) → 并入 B-1869 post-fire verify (action_success gating 时一并查 async-DOM confound)。

---

## 4. 🔑 P33 psom 签名 — img-href→裸 PNG 幻觉, B1 比 B0 更差 (4B 自纠错 gap)

P33 (导航至裸图片 URL 幻觉, §304 B0 psom 落码) verify **7/7 全真死因**。机制: `[SOM_MARKS]` 把 `<img href=".../{id}.png">` 暴露为可点击编号元素 (elem 13-19 区) → agent 点击 → obs_url 落到裸 PNG 页 → 视觉盲下**幻觉"已看到图"并 finish**。

**post-PNG 行为**: 幻觉读图立即 finish **6/7** (task 94/120/124/187/222 + 121) · back+重陷+幻觉 1 · 30 步循环卡死 1 (task 27)。

**⚠️ B1 (4B) 显著比 B0 (235B, §304) 差**: §304 B0 到裸 PNG 后能识别"无内容"并 back; **B1 6/7 直接在 PNG 页 hallucinate-finish** (如 task 120 编造 "Bike Brand Name", task 187 "Lightning McQueen 匹配") = 4B 对"当前是裸 PNG 页"的元认知弱 → 接受 PNG URL 当有效内容页幻觉作答。= **4B vs 235B 自纠错能力差距的干净实证**。

**结构性诱发 (phantom_som 特有)**: dom 不以可点击链接暴露 img href; 真 SoM agent 看截图不需点进 PNG; **psom 视觉盲 + [SOM_MARKS] 暴露 img-href → 点击近乎必然**。= HERO mode 的 [SOM_MARKS] 格式特有成本。

---

## 5. P31 finish-less artifact — 第 5 mode confound 闭环

success-hit 9/9 全 presence-only。**task 87/153 = finish-less artifact** (url_match, step0 到达正确 item URL → 此后 29 步 **click elem=17 (缩略图 img 元素 bbox=[150,303,550,412]) 全 action_success=False** 想"看图" → never finish → eval agent_page 凭当前 URL 自动 pass)。**P31 psom = §317 som/§320 vision/§321 ptext 同款 → 跨 5 mode confound 闭环** (som/dom/vision/ptext/psom)。psom 的 finish-less 由"点缩略图 element 想看图"驱动 (视觉盲特有)。绝不可裸作路由信号。

---

## 6. Tier-2 — no-hit 子集 (28 task, 全 agent-limit) = 视觉盲 × 图像依赖

与 ptext §321 一致 (phantom-family-wide)。亚型:
- **gallery 行列盲** (14/41/42/129): 线性 `[SOM_MARKS]` 文本无法编码 2D grid 行列位置 (task 41/42 step0 即 finish 立即放弃)。
- **reference-image 依赖选错 item** (47/60/62/96/138/142): task image 注入 prompt 但 listing 缩略图不可见 → 选错相似 item。
- **DOM-URL vs image-text 混淆** (173/199): 把 localhost 页面 URL 当"图片里的网站" (应 kaiyo.com) — **§321 ptext 亚型在 psom 复现**。
- **视觉属性/内容** (0 颜色/16 缩略图/80 含书/148 Amazon截图)。
- **MUTATION_MISSING** (75/76 edit 未提交 · 208/213 comment 未提交): program_html + effective_mutating=0 — **§318 P35 / §320 / §321 收敛, psom 再现**。
- **新变体**: task 161 agent emit **raw DOM id (33621) 当 SOM-mark 编号** → `obs_nodes_info missing union_bound` noop (4B grounding 把高 id 当编号); task 163 反复 goto amazon.com 5× → `policy_blocked_offsite` loop。

---

## 7. 🔁 Self-evolving — 提议 P-rule (discover-only, 留合并 freeze)

> ⚠️ **本 run 不落码** (与 §317/318/320/321 同步)。下列与 B1 dom+som+vision+ptext 提议**合并去重**后统一 freeze step 落码 + bump + 全量重扫。

**收敛到已提议规则** (5-condition 一致):
- **WALK_FAIL_DEGENERATE** (§321): `locator_route_meta.error contains 'walk_fail'` — psom 加变体 `no_actionable_within_walk` (click) ∪ `no_input_within_walk` (type)。**= P4 success-safe 收窄的正解** (P4 success-fire 4 ep 全 walk_fail no-op → 用 walk_fail 信号豁免, 不用裸 element_id∈{0,1})。
- **VISUAL_BLIND_IMAGE_TASK** (§319/321): phantom_* + image-dependent intent。
- **DOM_URL_AS_IMAGE_TEXT** (§321): task 173/199 psom 复现。
- **MUTATION_MISSING** (§318 P35): task 75/76/208/213。

**psom 特有 (P33 收窄)**: 连续 ≥2 步 obs_url 含 `.png` 且未 back → P33 `severity=high` (PNG 卡死子类, 区别 B0 §304 单次 back)。

**success-safe 收窄**: **P4** 用 walk_fail 豁免 (非裸 element_id, success 4 ep 全 walk_fail) · **P31** finish-less url_match 豁免 · **P5/P12/P14** 点缩略图 element 循环豁免 · P19 `eval_type==url_match` guard (同 §320/321)。

---

## 8. 代表 episode

| 类 | task | 一句话 |
|---|---|---|
| **axis-2 P4 (headline, ground-truthed)** | **8 / 4** | 视觉盲瞎猜 type/click [1] → walk_fail no-op; observation 实测 RootWebArea=[2] 无[1] = 幻觉 ref |
| **P33 PNG 幻觉 (4B>235B 差)** | **120 / 187** | 点 img-href→裸 PNG→幻觉编造内容 finish (B0 会 back, B1 不会) |
| **finish-less (P31 第5 mode)** | 87 / 153 | step0 到达正确 URL → 29 步点缩略图 img 想看图 → never finish, url_match 自动 pass |
| **gallery 行列盲** | 41 / 42 | step0 即 finish, 线性文本读不出 gallery 第二行/最后两行 |
| **MUTATION_MISSING** | 75 / 208 | edit/comment 改了 desc 没 submit, DOM 无 mutation |

---

## 9. Actionable

- **B-1869 refinement**: 加 async-DOM-noise 误归因假设 (agent F state_change.py code-reasoning, post-fire verify 时一并查)。
- **benchmark-FP / exclude**: 无新增。
- **HERO mode routing 论点**: psom 的 image-dependent / gallery-layout / P33-PNG 失败 = 应 **route 这些 task AWAY from psom → vision/SoM** (psom 视觉盲 + img-href 暴露是结构性成本)。axis-2 P4 + P33 = HERO mode 的 format 特有代价, paper §4 需 disclose。
- **freeze step TODO**: B1 cls 6-mode 齐 (剩 pprompt R32516) → 合并 **5 condition** (dom/som/vision/ptext/psom) 规则提议 → 去重 (WALK_FAIL ∪ click 变体 / MUTATION_MISSING / VISUAL_BLIND / DOM-URL-as-image / P33 severity 收敛) → 落码 + bump + 全量重扫 → 解锁 cross-mode 定量 + B1 cls=第2 cell → k_cells=2 → drop-one gate。**cross-mode 定量仍禁直至 freeze**。
