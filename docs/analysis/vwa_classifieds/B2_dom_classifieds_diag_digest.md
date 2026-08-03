# B2 dom classifieds — /diag failure attribution digest

**Run**: `B2_dom_classifieds_20260609_214713_553762009_985526_R21521` (manifest-bound authoritative)
**Condition**: phase1_dom_router_0 · **Site**: classifieds · **Model**: B2 = Gemma3-4B · **Mode**: dom
**N**: 224 ep · **SR**: 3/224 = **1.3%** · **ruleset_version**: `5-domsomvispsom-b1860coord`
**Diag date**: 2026-06-19 (首次 B2 cls diag, Tier-1 全扫 + Tier-2 sonnet 深挖)

> ⚠️ 单 condition digest，不下 cross-mode 结论。per-rule 分布只描述本 condition。cross-mode 定量待 B1+B2 cls freeze。


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
| SR | **1.34%** (3/224) |
| failed + hit | 221 |
| **failed NO-hit** | **0** |
| success + hit | 2 |

v8 新规则 failed 侧: {'P44': 244, 'P45': 254, 'P43': 69}；success 侧: 无。
（`P43` 在 cls 上大量命中属预期 —— 它标记"intent 需要视觉 + 该 mode 无页面截图"这一**中性组合**，
并非预测失败；§387.10 实测补上截图的增益 ≈0。）

全部 36 个 canonical condition 现处同一版本 → **cross-mode / cross-site 聚合解锁**。

---
## 1. 三分类统计

| 类别 | 占比 | 说明 |
|---|---|---|
| **agent-limit** | ~100% (221/221 failed) | Gemma3-4B cls ~1% 地板 (§335-338 六源收敛=真地板非 bug)；机制 = 看不懂列表缩略图照片→编答案→不发 finish→budget death |
| scaffold-bug | 0 | Tier-2 深挖 no-hit [16,221] 未发现框架 bug |
| benchmark-FP | 0 | no-hit finish answer 语义也错 (非评测误判) |

## 2. Tier-1 规则分布 (failed per-rule, hit 总数)

`P5`(感知缺失循环)=272 · `P31`(budget耗尽未完成)=195 · `P14`(URL自环)=138 · `P6`(视觉任务DOM必败, dom-gate)=105 · `P33`(img-href幻觉)=79 · `P12`(从不翻页)=78 · `P16`(视觉图像内容, dom-gate)=52 · `P19`(url_match过早finish)=45

→ **P5+P31 主导** = 精确印证 §335 Gemma 地板机制 (感知缺失 + 不终止烧 budget)。P6/P16 dom-gate 视觉天花板类 fire 105/52 = dom 视觉盲区。

## 3. Tier-2 深挖 (no-hit 盲区 + success 审计)

**no-hit failed (2, 全 agent-limit)**:
- **task 16**: DOM 无图 → B2 按商品名模糊匹配 "coffee grinder"≈"coffee mug" 导航到错误 item → 读错卖家 email (缩略图识别盲区, §306 THUMBNAIL 类)
- **task 221**: "how many bowls" 视觉计数 → B2 把 2 个图片链接误判为 2 个碗 (ref=6) = DOM 模式视觉计数系统性盲区 (任何 DOM-only 模型都败)

**success 审计 (3 个 success 上 P-rule fire, 全 presence-only 误报 `hit_causal=false`)**:
- task 25 / 106 / 110: action_failed/page_unchanged P-rule fire 是 B2 探索混乱 (select→click 序列乱 / email `mailto:` link click-no-navigate / 26 步乱探) 的副产品，非 failure causation — 这些 ep 最终 success。
- ⚠️ **B2 success 多为「路径错答案碰对」** (106 hallucinate 高价 item 含动物图碰对 · 110 错页面猜 "0" 碰对 ref="0|OR|zero") = B2 1.3% SR 本身含运气成分，非可靠能力。

## 4. 🔁 Self-evolving — 提议 P-rule (post-fire candidate, 本轮不落码)

**candidate (success-FP 抑制)**: email/phone `mailto:`/`tel:` link 被 click 但 page_unchanged = UI 特性非 action_failed。检测: `locator_route_meta.success=true` + `page_changed=false` + target_tag=A + `href^=mailto/tel` → 排除出 action_failed P-rule (减 B2 success-ep 上 P2/P5/P14 presence-only 误报)。**ruleset 冻结待 B1+B2 cls freeze 一起改 (§0 diag_freeze_v6_plan)**。

## 5. Actionable

- 无 scaffold-bug B-number (Tier-2 确认框架无 bug)。
- 无 benchmark-FP task 排除。
- **B2 dom cls = 干净 agent-limit 地板**，paper §3-§4 evidence (Gemma 跨族 matched-capability control 的 dom 视觉盲区表现)。

---

### v11 数字块（`11-intent-text-fallback`，2026-08-03 补）

> 本 digest 正文成稿于更早的 ruleset。v10 落了 **+P49 / P36 carve-out / P14 carve-out**，
> v11 给 **P34/P48 换用 `_finish_intent_text()`**（answer 为空时 fallback 读 `thought`——
> B0 惯于把结论写进 `answer`，B1 留在 `thought`，旧口径因此变成了模型行为检测器）。
> 全部 48 个 canonical condition 已在 v11 下重扫，**cross-mode / cross-model 聚合以本块为准**。

| 字段 | 值 |
|---|---|
| Run | `B2_dom_classifieds_20260609_214713_553762009_985526_R21521` |
| Episodes | 224（success 3 · SR 1.34%） |
| 三子集 | failed+hit 221 · failed-NO-hit 0 · success+hit 2 |
| config_missing | 0 |

| 规则 | 含义 | step 级 | episode 级 |
|---|---|---:|---:|
| `P36` | WALK_FAIL_DEGENERATE | 1919 | 183 |
| `P5` | 感知缺失循环 | 272 | 155 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 254 | 148 |
| `P31` | budget耗尽未完成 | 126 | 126 |
| `P14` | URL 自环 | 138 | 113 |
| `P6` | 视觉任务 DOM 必然失败 | 105 | 105 |
| `P44` | HALLUCINATED_ELEMENT_REF | 244 | 82 |
| `P33` | 导航至裸图片URL幻觉 | 79 | 79 |
| `P12` | 从不翻页 | 78 | 78 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 69 | 69 |
| `P16` | 视觉图像内容DOM必败 | 52 | 52 |
| `P18` | cheapest漏价格排序 | 35 | 35 |
| `P20` | 评测目标页从未访问 | 25 | 25 |
| `P2` | 容器节点误点 | 34 | 16 |
| `P25` | 跨站任务跳过其中一站 | 12 | 12 |
| `P15` | gallery行位置DOM不可定位 | 6 | 6 |
| `P17` | click-back振荡 | 6 | 6 |
| `P30` | 到达正确item后离开 | 4 | 4 |
| `P10` | 跨步数值记忆失败 | 4 | 3 |
| `P19` | url_match过早搜索页finish | 3 | 3 |
| `P24` | 不确定仍finish | 2 | 2 |
| `P22` | 图上数字dom不可读 | 2 | 2 |
| `P11` | 最新+地点组合 | 1 | 1 |
| `P37` | URL_HALLUCINATION | 1 | 1 |
| `P38` | DOM_URL_AS_IMAGE | 1 | 1 |

> ⚠️ **解读约束**（`docs/analysis/_data_quality_audit.md`）：
> ① 本表是**症状分布，不是死因分布** —— P36/P31 经 10 例跨 benchmark 因果验证均判为 risk-marker；
> ② `P2`/`P4` 依赖 `element_bbox`，在 **vision 上结构性为 0（假 0）**；
> ③ `P36` 在 vision 上只覆盖 `type` 步（click 无 `locator_route_meta`）→ **分母与 dom/som 不同**。
