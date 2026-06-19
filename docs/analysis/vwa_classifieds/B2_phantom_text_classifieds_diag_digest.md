# B2 phantom_text classifieds — /diag failure attribution digest

**Run**: `B2_phantom_text_classifieds_20260614_020803_377049301_1495224_R14219` (manifest-bound authoritative)
**Condition**: phase1_phantom_text_router_0 · **Site**: classifieds · **Model**: B2 = Gemma3-4B · **Mode**: phantom_text (DOM prompt + [SOM_MARKS] 文本 + 无标注图)
**N**: 224 ep · **SR**: 1/224 = **0.4%** (6-mode 最低) · **ruleset_version**: `5-domsomvispsom-b1860coord`
**Diag date**: 2026-06-19 (Tier-1 全扫 + Tier-2 sonnet 深挖 5 ep)

> ⚠️ 单 condition digest，不下 cross-mode 结论。cross-mode 定量待 B1+B2 cls freeze。

## 1. 三分类统计

| 类别 | 占比 | 说明 |
|---|---|---|
| **agent-limit** | ~100% (223/223 failed) | Gemma3-4B phantom_text 地板, 6-mode 最低 SR。5 ep Tier-2 全 agent-limit |
| scaffold-bug | 0 (1 telemetry gap) | 无 fatal bug；但 task 5 暴露 `effective_mutating_action_count=0` 漏计 GET-based 删除 (telemetry gap, 非 fatal) |
| benchmark-FP | 0 | — |

## 2. Tier-1 规则分布 (failed per-rule, hit 总数)

`P5`(感知缺失循环)=237 · `P31`(budget耗尽未完成)=209 · `P4`(根节点误操作)=**153** · `P14`(URL自环)=99 · `P33`(img-href幻觉)=66 · `P12`=52 · `P19`=45 · `P18`=26

→ P5+P31 主导。`P4`=153 显著 = element_id=1 根节点幻觉 (§322 low-default; phantom_text 裸 element_id + 无图最易幻觉，与 phantom_som P4=278 同源)。

## 3. Tier-2 深挖

**no-hit failed (4, 全 agent-limit)**:
- **task 12**: 类目推断缺失 (未搜 motorcycle，按 URL item-id 顺序猜"最新")
- **task 16**: 视觉识别图片定位 (无图 → 文字搜错 item，读错 email)
- **task 41**: ⭐ **gallery 行结构盲** ([SOM_MARKS] 文本不保留 2D grid 布局 → "second row of this page" 无法定位 = phantom_text 结构性盲区)
- **task 119**: 图片内容读取 (钞票面额读不到 → 用 listing price "9999" 代替 ref "50")
- → 16/41/119 = **phantom_text 结构性盲区** (no-image + no-grid-layout)

**success 确认 (task 5 = 唯一 success, presence-only 伪成功)**:
- agent step 6 **确实成功删除** item 84144 (reward=1.0, DOM 中消失)，但 `agent_finished=false` + `trajectory_incomplete=true` → agent 随后 24 步继续随机删除其他 listing 直到 budget 耗尽，runner 救活。
- ⚠️ **telemetry gap**: `effective_mutating_action_count=0` 即使删除成功 (GET-redirect 删除未被突变追踪器捕获) — 非 fatal，post-fire 记。

## 4. 🔁 Self-evolving — 提议 P-rule (post-fire candidates)

1. **P-vision-required**: intent 含 'in the picture' / 'shown in image' / 'denomination' + mode∈{phantom_text, dom} → agent-limit/vision-required (覆盖 16/119，扩展现有 vision-required 规则族)
2. **P-gallery-row**: intent 含 'second/nth row of this page' + start_url 含 `sShowAs=gallery` + mode∈{phantom_text, dom} → layout-blind (task 41)
3. **P-presence-only** (task 5, 同 vision/som — agent_finished=false + trajectory_incomplete=true + success=true)

→ ruleset 冻结待 B1+B2 cls freeze (§0 diag_freeze_v6_plan)。

## 5. Actionable

- 无 fatal scaffold-bug；**telemetry gap**: `effective_mutating_action_count` 漏计 GET-based 删除 (post-fire 候选，cross-ref B-1869 测量隐患族)。
- **B2 phantom_text cls = 0.4% 最低地板**；⚠️ **唯一 success 是 presence-only** → 名义 SR 虚高，真实 SR ≈ 0。
- **presence-only 伪成功跨 som+vision+phantom_text 三 mode 系统性** = B2 url_match/mutation success 普遍含「runner 救活」。
