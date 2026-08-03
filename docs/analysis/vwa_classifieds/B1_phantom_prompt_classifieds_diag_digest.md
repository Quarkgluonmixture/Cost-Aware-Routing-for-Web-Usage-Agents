# B1 phantom_prompt (P-prompt) classifieds — /diag failure attribution digest

| field | value |
|---|---|
| **Run** | `B1_phantom_prompt_classifieds_20260607_135946_736335864_683961_R32516` (manifest-bound authoritative) |
| **Condition** | `phase1_phantom_prompt_router_0` |
| **Site / Model / Mode** | classifieds / **B1 (Qwen3-VL-4B)** / **phantom_prompt (P-prompt)** = **SoM-style prompt + 完整 AXTree text** (元素用原始高 node ID 标注), **视觉盲 (无标注图)** |
| **N episodes** | 224 |
| **SR** | **6.7%** (15/224 success) |
| **ruleset_version** | `5-domsomvispsom-b1860coord` ⚠️ **discover-only — 未落码新规则, 未 bump version**; cross-mode 定量比较禁止直至 freeze |
| **Tier-2 深挖** | 28 episode / 5 sonnet sub-agent (15 unique no-hit 全覆盖 + 7 shared verify + 6 success-hit FP) + **3 项确定性 forensic** (walk_fail 表征全扫 / task 173+199 observation img-src / task 5 delete benchmark-FP) |
| **姊妹 condition** | §317 som · §318 dom · §320 vision · §321 ptext · §322 psom — B1 cls cross-mode discover, **pprompt = 第 6 / 最后 mode → 6-mode 齐, 触发 freeze step** |

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
| failed + hit | 198 |
| **failed NO-hit** | **11** |
| success + hit | 2 |

v8 新规则 failed 侧: {'P43': 69, 'P45': 88, 'P44': 134, 'P46': 2}；success 侧: 无。
（`P43` 在 cls 上大量命中属预期 —— 它标记"intent 需要视觉 + 该 mode 无页面截图"这一**中性组合**，
并非预测失败；§387.10 实测补上截图的增益 ≈0。）

全部 36 个 canonical condition 现处同一版本 → **cross-mode / cross-site 聚合解锁**。

---
## 1. 三分类统计

| 类别 | 占比 (深挖样本) | 说明 |
|---|---|---|
| **agent-limit** | 压倒 (22/22 failed 深挖) | phantom-family **视觉盲 × 图像依赖** (主轴) + **axis: walk_fail 退化引用** (pprompt 特有 surface) |
| **scaffold-bug** | 0 confirmed | walk_fail success=True 测量隐患 = **B-1869** (已登记, 本 run 21.7% 复现) — 非 fatal, 测量层 |
| **benchmark-FP** | **1 (success 侧)** | **task 5 false-success** (delete 任务 0 mutation 却 pass, 抬高 SR) — B1 phantom sweep 首例 (见 §6) |
| **rule-FP (presence-only)** | 大量 | success-hit 6/6 全 FP + finish-less artifact + P18 真 bug |

**一句话**: B1 pprompt 失败 = phantom-family **视觉盲 × 图像依赖任务** (与 §321 ptext / §322 psom 同根, AXTree-vs-`[SOM_MARKS]` text 表征无关), 叠加 pprompt 特有的 **walk_fail 退化引用** (66.5% episode): SoM-style prompt 诱导 agent 操作编号元素, 但 AXTree 的真实高 node ID 在视觉盲下无法 ground → 引用 stale/错类型节点 → walk_fail no-op。**axis headline: P4 root-ref (裸 element_id∈{0,1}) 在 pprompt = 0, 但 walk_fail 信号抓住同一底层失败 — 验证 §322 "WALK_FAIL 信号 mode-robust / 裸 element_id mode-fragile" 决策。**

---

## 2. Tier-1 规则分布 (failed-only, episode-level)

```
P31=131  P5=100  P19=70  P17=59  P14=45  P18=28  P12=18  P20=16  P7=14
P25=10  P2=9  P10=8  P30=8  P13=7  P33=5  P27=4  P24=2  P22=2  P23=2  P28=1
```
no-hit failed = **34** (coverage 83.7%); success-fire (FP源, episode-level) = P17×4 P5×3 P12×2 P14×2 P33×2 P10×1 P31×1。

⚠️ **pprompt 特有 — P4 完全缺席 (=0)**: sibling psom P4=63 / ptext P4=41 / vision P4=0。这是本 diag 的 headline (§3)。全扫确认 pprompt agent **从不 emit element_id∈{0,1}** (0 episode)。

---

## 3. 🔑 axis headline — P4=0 但 walk_fail 主导, "瞎猜 element ref" 换了 surface (ground-truthed)

**现象 (确定性全扫)**: deterministic 规则 **P4 (action 对 element_id∈{0,1})** 在 pprompt fire **0 次** (sibling psom 63 / ptext 41)。但 `locator_route_meta.error contains 'walk_fail'` 在 pprompt fire **880 actions / 149 episode (66.5% 的 ep / 18.4% 的全部 4793 action)**。

**ground-truth (全 run walk_fail 表征扫 + sub-agent forensic, 修正初始假设)**:
- 初始假设 = "pprompt 瞎猜**高** AXTree node ID → walk_fail" → **forensic 部分证伪** (Batch C: task 166 高 ID 10451/10918 反而全 resolve 成功; task 141 eid=148 是小 ID)。
- **真机制 = 视觉盲 agent 无法 ground 元素引用, 两个系统性 surface**:
  - **`click → no_actionable_within_walk`: 606 (69%)** — 多为 **stale-EID-after-nav**: 翻页后沿用上一页 eid → `obs_nodes_info missing union_bound` / 不可点。全扫 **556/880 (63%) 的 walk_fail eid 是本 ep 此前引用过的** (stale-ish 代理)。
  - **`type → no_input_within_walk`: 274 (31%)** — **元素类型错**: agent 想重新搜索, 把 item/搜索结果页的 heading/link 节点当搜索框 type (该页无 textbox + 无截图看不见这点)。

**axis 分解 (mechanism-level qualitative, NOT 定量 gate — discover-then-freeze #1)**:
> | text 表征 | "找不到目标元素" 时的 surface | 检测规则 |
> |---|---|---|
> | `[SOM_MARKS]` 短编号 (psom/ptext) | default 到低 **[1]** (幻觉式 low-default, observation 实测无 [1] — §322) | P4 抓 (63/41) |
> | 完整 AXTree 高 node ID (pprompt) | 引用 **stale/错类型** 真 node ID | P4 漏 (0), **walk_fail 抓 (149 ep)** |
>
> 两 surface 是**同底层 agent-limit** (视觉盲无法 ground 元素引用) 的不同表现, 由 text 表征决定形态。**pprompt = §322 决策的最强证据**: 裸 element_id 信号 (P4) **mode-fragile** (只对 `[SOM_MARKS]` text 有效); **walk_fail 信号 (locator_route_meta) mode-robust** (跨 text 表征抓退化引用)。§322 "P4 success-safe 收窄正解 = walk_fail 豁免 (非裸 element_id∈{0,1})" 在此被独立验证。

**精化 §322 "phantom-family-wide" 声明**: 底层失败 (视觉盲→无效 element ref→walk_fail) 确实 phantom-family-wide; 但 "**root-ref 到 [1]**" 的具体 surface 是 **`[SOM_MARKS]`-text-specific** (ptext+psom), pprompt (AXTree text) 不出现。pprompt 是隔离 text-表征轴的 control。

**分类**: **agent-limit 主** (model emit 无效/stale ref, 视觉盲驱动) + **B-1869 测量隐患** (walk_fail 报 success=True)。

**B-1869 复现 (本 run)**: walk_fail 中 **21.7% (191/880) 报 action_success=True + page_changed=True** (两者完全重合 191 各)。支持 §322 async-DOM 误归因假设 (fallback bbox click 真碰到东西 或 后台 DOM 更新被 `detect_page_state_change` 误归因给 agent 动作 → type/click 落空仍算 success → 虚高 action_success)。**post-fire verify candidate** (非 fire-blocker)。

---

## 4. Tier-2 — no-hit 子集 (34 task, 全 agent-limit) = phantom-family 视觉盲 × 图像依赖

focused 策略: 15 unique no-hit 全挖 + 19 shared-with-psom 抽 7 verify (psom §322 已确认其余 12 = 视觉盲)。**22 深挖 + 12 by-psom-reference = 34/34 覆盖**, 0 scaffold / 0 failed-side benchmark-FP。主轴与 §321/§322 一致 (phantom-family-wide), 亚型:

| 亚型 | task | 死因 |
|---|---|---|
| **图内 OCR (结构性必败)** | 128 (球衣号) · 222 (卷尺读数) · 49 (蓝 LED 灯) | 信息只在图片像素, 无图 1-2 步即放弃 (`input_image=0`) |
| **reference-image recall** | 47 · 136 · 137 · 140 · 141 (exact item recall) | task image 注入 prompt 但 listing 缩略图不可见 → 选相似错 item |
| **缩略图视觉选择 / 颜色** | 14 (第二行) · 80 (含书) · 192 (汽车书封面) · 83 (无珠宝图) | gallery 行列空间 + 缩略图内容文本无法从扁平 AXTree 推 |
| **gallery 行列盲** | 41 (第二行价格区间, step0 即 finish) | 线性 AXTree 读不出 2D grid 行位置 → 单 item 当区间 |
| **DOM-URL vs image-text 混淆 (视觉盲, 表征无关)** | 173 · 199 | kaiyo.com 是像素 obs 无 → 抓页面/img URL 当"图中网址" (见 §5; ⚠️CORRECTION: img-src 实暴露, B0 §319 成立) |
| **MUTATION_MISSING** | 9 (create post 未提交) · 213 (Elvis 条件分支 comment 未提交) · 75/76 (edit price) | program_html + effective_mutating=0 — §318/§320/§321 收敛, pprompt 再现 |
| **条件分支 (依赖图判断)** | 213 (Elvis 有无观众 → email vs comment) | 视觉判断错 → 走错分支 → DOM 无 mutation |
| **correct-item-visited-but-rejected (routing 候选)** | 111 | 到达正确 item 79079 但视觉盲看不到球衣图 → 误判"非球衣"离开 (见 §8) |
| **字面搜索代替分类导航 / 字段错填** | 2 (搜索收窄排除正解) · 192 (type 'cars' 进 sCity 城市框) | 弱 4B 字段/导航错 |

---

## 5. 🔑 DOM-URL-as-image — task 173/199 视觉盲, 表征无关 (⚠️ 2026-06-08 CORRECTION: 撤回初版 "pprompt 不暴露 img-src / 修正 B0 §319")

task 173/199 (intent: "website address/mentioned **in the image**"): agent 答**页面 URL** (`.../page=item&id=14834`) / base URL, 而非图片像素里印的 `kaiyo.com` → 失败。与 psom (§322) / ptext (§321) **同根 = 视觉盲** (kaiyo.com 是图片**像素**, 任何文本表征都没有)。

> ⚠️ **CORRECTION (2026-06-08, code+obs ground-truth 翻案本 §5 初版)**: 初版称 "pprompt AXTree **不暴露** img-src (grep `oc-content/uploads`=False) → 修正 B0 §319 → §290 第6次"。**实测全错**:
> - **img-src 在所有文本模式都暴露**: 上游 VWA `browser_env/processors.py:703` 对每个 `<img>` 注 `, url: {src}` 进 alt → 流入 AXTree → dom / `[SOM_MARKS]` / pprompt **全有**。实测 pprompt task 173 observation `oc-content/uploads`=**5** (含 `[id=17] image '...' url: .../14834/14834.png`), 与 psom **字节级相同** (5/5)。`som.py _extract_text_marks` 的 label = 整行只剥 `[id]` 前缀 → `url:` 原样保留, `[SOM_MARKS]` 也带。
> - **observation_dom.txt = 模型真输入** (runner `main.py:1611` 写 `obs.text`; agent `qwen3vl_agent.py:160/187` 喂 `obs.text` 作 AXTree) → pprompt 模型确实收到 img-src。
> - ∴ **B0 §319 "AXTree 暴露 img-src" 本来就对**; 初版 "修正" 才是误判 (grep 错了文件/step)。**无 psom-vs-pprompt surface 差异** (两者相同)。**"§290 第6次" 撤回** —— 反而这是**反向 §290 案例**: 初版 grep 自己就是 log-only 误判, code+obs 重 ground-truth 把 B0 §319 救回来了。

**真机制 (表征无关视觉盲)**: VWA 注的 `url:` = `uploads/<item_id>/X.png` **文件名** (编码 item-id, 与 link href 冗余), **不是图片内容**; P79 **观测侧不接 captioning** (`vwa_wrapper` 不传 `captioning_fn` → agent obs 无 `description:`; BLIP-2 仅 `environment.py` 评测器给 page_image_query 打分用)。→ 文本 agent 拿到图片**文件名**但读不到**像素内容** (kaiyo.com) → 抓个在场 URL (page/img-src 都在 obs 里) 当答案 → 必败 (img-src 文件名在此还是 **distractor**)。**纯视觉盲, 与 text 表征无关** (§3 的 surface 分解不适用此 task)。⚠️ **附带披露缺口**: §3.5 应明示"观测侧无 caption / 文本-phantom 真 image-free" (见 next_steps forward §0; 改 paper prose 走 /stress)。

---

## 6. 🔑 task 5 — false-success benchmark-FP (B1 phantom sweep 首例, ground-truthed)

success-hit 审计 6/6 hit 全 presence-only (hit_causal=false), 但 **task 5 本身的 success=True 是 false success**:

- intent "Navigate to my listing of the white car and **delete it**"; eval = program_html 检查 `item&id=84144` 返回 "404"。
- forensic: `effective_mutating_action_count=0` · `delete_remove_count=0` · `submit_create_count=0` · `agent_finished=false` · `trajectory_incomplete=true` → **agent 啥也没删** (27/30 步在 user/items 页 scroll, item 84144 只访问 1 次且是 type 动作)。
- **eval 仍 pass** 因 item 84144 在 reset 态本就返回 404 (precondition, 非 agent 动作)。

**分类 = benchmark-FP (false success)**: 与常见的"答案对但被判错"(SR-压低) 相反, 这是 **agent 没做事却得分 → SR-抬高**。本 task 贡献 ~0.4pp 虚高 SR。**B1 cls phantom sweep 首个 benchmark-FP** (sibling som/dom/vision/ptext/psom 均报 0)。⚠️ **需 reset-state 确认** (post-hoc 无法完全证 item 84144 是否本就不存在 / "white car" 真实 id ≠ 84144); 登记为 measurement-hazard 候选, 不阻 fire。

---

## 7. Tier-2 — success-hit FP 审计 (6/6 全 presence-only) + 第 6 mode confound 闭环

**全部 6 success episode 的 hit 都 hit_causal=false**:
- **task 87 = finish-less arrival artifact** (url_match agent_page, step0 到达正确 item 34463, 此后 30 步点 img PNG + back 想"看图" never finish → 凭当前 URL 自动 pass)。**P31 在 pprompt = §317 som/§318 dom/§320 vision/§321 ptext/§322 psom 同款 → 跨 6 mode confound 闭环 (全 mode)**。绝不可裸作路由信号。
- **task 45/153**: 到达正确 item 后 emit finish (定义性 finish), P31+P5 是视觉盲验证式 oscillation overhead 非死因。
- **task 48**: homepage category select-loop (steps 0-10 同 URL select_option 重试) 触发 P5/P14/P17, 但最终 finish 正确 item 20629。
- **task 56 = P18 真 bug (非 presence-only, 是规则漏洞)**: agent **step 3 已按价格升序排序** (`sOrder=i_price&iOrderType=asc`), P18 (cheapest 漏价格排序) 仍 fire = **false positive**。P18 需加 guard: 任一 obs_url 含 `sOrder=i_price` 则豁免。
- **task 5**: P14 (URL 自环 27/30 步 user/items) presence-only + 见 §6 false-success。

---

## 8. routing 证据 — correct-item-visited-but-rejected (第三类 routing-rescuable)

**task 111** (find team name of most-recently-posted hockey jersey): agent step1 即到达正确 item **79079** (reference), 但 pprompt 视觉盲看不到 listing 图里的球衣 → 误判"Beckett Hockey Magazine, 非球衣" → 此后 26 步搜无关品 → finish "Unable to determine"。

= 与 §320 task 40 (text-in-image rescue) / §321 task 17 (attribute-verification deadlock) **不同的第三类 routing-rescuable failure**: agent **已在正确页**但视觉盲无法**识别图像内容确认任务对象** → 换 SoM/Vision 给视觉确认能救活。freeze 后 cross-mode 验。

---

## 9. 🔁 Self-evolving — 提议 P-rule (discover-only, 留 freeze step 合并)

> ⚠️ **本 run 不落码** (与 §317/318/320/321/322 同步)。**pprompt = B1 cls 第 6/最后 mode → 6-mode 齐 → 下面是 freeze step 的合并清单。**

**收敛到已提议规则 (6-condition 一致, freeze 落码主体)**:
- **WALK_FAIL_DEGENERATE** (§321/§322, **pprompt 是最强证据**): `locator_route_meta.error contains 'walk_fail'` (含 `no_input_within_walk` [type 变体] ∪ `no_actionable_within_walk` [click 变体])。**= P4 success-safe 收窄的正解** (用 walk_fail 信号, 非裸 element_id∈{0,1} — pprompt 实证裸 element_id 漏 100%)。⚠️ 内置 B-1869 注意: walk_fail 21.7% 报 success=True, 落码时 success-fire 用 walk_fail-but-success 标 presence-only。
- **VISUAL_BLIND_IMAGE_TASK** (§319/321/322): `obs_mode in phantom_* AND image-dependent intent` (task_config.image≠null OR intent 含 'in the image'/'picture'/'cover'/'color'/'looks like')。⚠️ presence-only 风险 (文本侥幸答对则 success-fire) → 需 success-safe (finish≠ref / `input_image=0 AND steps≤3 AND finish 含 'cannot verify/image not visible'` 类硬屏障 sub-signal)。
- **DOM_URL_AS_IMAGE_TEXT** (§321/322, pprompt 复现): `intent 含 'in the image' + 该页有 img + finish.answer 含 localhost (任一形态: `page=item` 页面 URL / base URL / `oc-content/uploads` img-src 路径 — **三者 obs 都在**, agent 抓任一当"图中网址") + reference 是真实外域 (kaiyo.com=像素内容)`。⚠️ **CORRECTION (见 §5)**: 初版 "pprompt 不暴露 img-src / 两 surface 分支" 已撤 — img-src **全文本模式暴露**, 无 psom/pprompt 差异。
- **MUTATION_MISSING** (§318 P35): program_html + `effective_mutating_action_count=0` / `submit_create_count=0` (task 9 create) / `delete_remove_count=0` (task 5) — task 9/213/75/76 + ⚠️ 兼抓 task 5 false-success delete (见下)。

**新 / 修正 (pprompt 贡献)**:
1. **P18 false-positive 修复 (确定性 bug, 优先)**: task 56 已排序却 fire。P18 加 guard: 任一 obs_url 含 `sOrder=i_price` (或价格排序参数) 则不 fire。**0-token, 直读 URL。**
2. **DELETE_FALSE_SUCCESS / spurious-pass (measurement)**: delete/program_html 任务 success=True 但 `delete_remove_count=0 AND effective_mutating=0` → 标 false-success benchmark-FP (task 5)。**SR-抬高隐患**, freeze 时一并实现 (区别于 SR-压低 FP)。
3. **CORRECT_ITEM_REJECTED (routing 信号, 需 runtime reference_url)**: 到达 reference_url item 后同 URL ≥2 walk_fail 随后离开 (task 111)。第三类 routing-rescuable, 但需 runtime reference 可查, 较复杂 → freeze 后评估。

**现有规则 success-safe 收窄 (0-token, 非 success-label)**: **P4** walk_fail 豁免 (pprompt 0 故无影响, 但 psom/ptext 必需) · **P31** finish-less url_match/agent_page 豁免 (task 87) · **P18** `sOrder=i_price` guard (task 56) · **P5/P14/P17** 点 img-PNG / homepage select-loop / 视觉盲验证式 oscillation 豁免 · **P10** finish.answer=URL 跳数字比对 · **P19** 加 `eval_type==url_match` guard (§320/321 收敛)。

---

## 10. 代表 episode

| 类 | task | 一句话 |
|---|---|---|
| **axis headline — walk_fail 退化引用 (P4=0 替代)** | **111 / 136** | 视觉盲引用 stale/错类型 AXTree node → walk_fail no-op (P4 抓不到, walk_fail 抓 149 ep) |
| **DOM-URL-as-image (视觉盲, 表征无关; 纠 §5)** | **173 / 199** | kaiyo.com=像素 obs 无; img-src 全模式暴露(B0 §319 成立) → 抓 page/img URL 当答案 |
| **false-success benchmark-FP (首例)** | **5** | delete 任务 0 mutation 却 pass (item 84144 本就 404) → 虚高 SR |
| **finish-less artifact (P31 第6 mode 闭环)** | 87 | step0 到达正确 URL → 30 步点 img PNG 想看图 → never finish, agent_page 自动 pass |
| **routing 候选 — correct-item-rejected** | 111 | 到达正确 item 79079 但视觉盲看不到球衣 → 误判离开 |
| **P18 false positive (规则 bug)** | 56 | 已 `sOrder=i_price` 排序, P18 仍 fire |
| **视觉盲图内 OCR / gallery 行列** | 128 / 41 | 球衣号在图里无图必败 / 线性 AXTree 读不出 gallery 第二行 |

---

## 11. Actionable

- **B-1869 (测量层, post-fire)**: walk_fail 报 success=True (21.7%, 本 run 复现) + async-DOM 误归因假设 (§322 refinement) → action_success gating 时一并查。非 fatal, 不阻 fire。cross-ref `locator_dispatch` + `master_bug_catalog`。
- **benchmark-FP / exclude**: **task 5 false-success delete** 登记 measurement-hazard 候选 (需 reset-state 确认 item 84144 precondition)。failed 侧 0 新增。
- **routing 证据 (paper §4/§6)**: pprompt 三类 routing-rescuable 失败齐: text-in-image rescue (§320 t40) / attribute-verification deadlock (§321 t17) / **correct-item-rejected (本 run t111)**。+ HERO mode 论点: 视觉盲 image-dependent / walk_fail 退化引用 = 应 route AWAY → vision/SoM。
- **🧊 FREEZE STEP TODO (pprompt = 第6/最后 mode, 6-mode 齐)**: 合并 **B1 cls 6 condition** (dom §318 / som §317 / vision §320 / ptext §321 / psom §322 / pprompt 本 run) 规则提议 → 去重 (WALK_FAIL ∪ click变体 / VISUAL_BLIND success-safe / DOM-URL 双 surface 分支 / MUTATION_MISSING / **P18 sOrder guard** / **DELETE_FALSE_SUCCESS** / P19 eval_type guard) → 落码 `diag_pattern_match.py` + **bump `RULESET_VERSION` → `6-*`** + `diag_autorun.sh` **全量重扫** 所有 condition 拉齐版本 → **解锁 cross-mode 定量** + **B1 cls = 第 2 完整 cell → k_cells=2 → §1 hero drop-one gate 出数** (现 INSUFFICIENT_DATA)。
- ⚠️ **cross-mode / cross-model 定量比较仍禁直至 freeze** (discover-then-freeze 硬纪律 #1; 本 digest 的 P4 / walk_fail 跨 mode 对照仅 mechanism-level qualitative, 非定量 gate; 数字待 freeze 全量重扫拉齐 `ruleset_version` 后才可比)。

---

### v11 数字块（`11-intent-text-fallback`，2026-08-03 补）

> 本 digest 正文成稿于更早的 ruleset。v10 落了 **+P49 / P36 carve-out / P14 carve-out**，
> v11 给 **P34/P48 换用 `_finish_intent_text()`**（answer 为空时 fallback 读 `thought`——
> B0 惯于把结论写进 `answer`，B1 留在 `thought`，旧口径因此变成了模型行为检测器）。
> 全部 48 个 canonical condition 已在 v11 下重扫，**cross-mode / cross-model 聚合以本块为准**。

| 字段 | 值 |
|---|---|
| Run | `B1_phantom_prompt_classifieds_20260607_135946_736335864_683961_R32516` |
| Episodes | 224（success 15 · SR 6.70%） |
| 三子集 | failed+hit 196 · failed-NO-hit 13 · success+hit 2 |
| config_missing | 0 |

| 规则 | 含义 | step 级 | episode 级 |
|---|---|---:|---:|
| `P36` | WALK_FAIL_DEGENERATE | 795 | 131 |
| `P5` | 感知缺失循环 | 155 | 100 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 69 | 69 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 88 | 65 |
| `P17` | click-back振荡 | 59 | 59 |
| `P31` | budget耗尽未完成 | 54 | 54 |
| `P14` | URL 自环 | 45 | 44 |
| `P44` | HALLUCINATED_ELEMENT_REF | 134 | 33 |
| `P18` | cheapest漏价格排序 | 28 | 28 |
| `P12` | 从不翻页 | 18 | 18 |
| `P20` | 评测目标页从未访问 | 16 | 16 |
| `P7` | sCity=州名 | 16 | 14 |
| `P19` | url_match过早搜索页finish | 11 | 11 |
| `P25` | 跨站任务跳过其中一站 | 10 | 10 |
| `P2` | 容器节点误点 | 26 | 9 |
| `P30` | 到达正确item后离开 | 8 | 8 |
| `P13` | 搜索代替浏览 | 7 | 7 |
| `P10` | 跨步数值记忆失败 | 9 | 5 |
| `P33` | 导航至裸图片URL幻觉 | 5 | 5 |
| `P27` | 找不到即放弃 | 4 | 4 |
| `P37` | URL_HALLUCINATION | 4 | 4 |
| `P24` | 不确定仍finish | 2 | 2 |
| `P22` | 图上数字dom不可读 | 2 | 2 |
| `P23` | oldest误用价格排序 | 2 | 2 |
| `P38` | DOM_URL_AS_IMAGE | 2 | 2 |
| `P35` | MUTATION_MISSING | 2 | 2 |
| `P46` | COMMENT_INTENT_NO_TYPE | 2 | 2 |
| `P28` | benchmark-FP货币tokenize | 1 | 1 |

> ⚠️ **解读约束**（`docs/analysis/_data_quality_audit.md`）：
> ① 本表是**症状分布，不是死因分布** —— P36/P31 经 10 例跨 benchmark 因果验证均判为 risk-marker；
> ② `P2`/`P4` 依赖 `element_bbox`，在 **vision 上结构性为 0（假 0）**；
> ③ `P36` 在 vision 上只覆盖 `type` 步（click 无 `locator_route_meta`）→ **分母与 dom/som 不同**。
