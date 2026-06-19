# B2 dom classifieds — /diag failure attribution digest

**Run**: `B2_dom_classifieds_20260609_214713_553762009_985526_R21521` (manifest-bound authoritative)
**Condition**: phase1_dom_router_0 · **Site**: classifieds · **Model**: B2 = Gemma3-4B · **Mode**: dom
**N**: 224 ep · **SR**: 3/224 = **1.3%** · **ruleset_version**: `5-domsomvispsom-b1860coord`
**Diag date**: 2026-06-19 (首次 B2 cls diag, Tier-1 全扫 + Tier-2 sonnet 深挖)

> ⚠️ 单 condition digest，不下 cross-mode 结论。per-rule 分布只描述本 condition。cross-mode 定量待 B1+B2 cls freeze。

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
