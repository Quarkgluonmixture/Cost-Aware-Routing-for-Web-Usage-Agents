# B0 vision reddit — /diag digest (3-tier)

- **Run**: `B0_vision_reddit_20260628_094255_184327569_3222015_R17559` (manifest-bound authoritative, bound 2026-06-29)
- **Condition**: `phase1_vision_router_0` · site=reddit · model=B0 · mode=vision (纯截图, 无 AXTree 文本)
- **N**: 205 ep · **SR**: 7.8% (16/205)
- **ruleset_version**: `7-p6p16clsgate-b1860coord`
- **Diag date**: 2026-07-02 (Tier-2: 6 sonnet sub-agents; 37/37 no-hit 全覆盖 + 5 verify; success_hit=0 无审计需求)

> ⚠️ 单 condition digest, 不下 cross-mode 定量结论 (discover-then-freeze; reddit 6-mode 齐后 v8 freeze 重扫)。
> 与 som digest (`B0_som_reddit_diag_digest.md`) 同日同 site — 跨 mode 定性对照见 §6 (标 provisional)。

## 1. 三分类统计 (Tier-2 深挖 37 no-hit + 5 verify)

| 类别 | 计数 | 说明 |
|---|---|---|
| agent-limit | 40 (35 no-hit + 5 verify 全 AL) | 细分见 §3 |
| benchmark-FP | 2 | task 104 (config 错 URL, **B-1885** 双证) + task 125 (token-granularity '27.0'≠'27', 与 som 125 同款) |
| scaffold-bug | 0 | |
| unclear | 0 | |

## 2. Tier-1 规则分布 (failed-only)

P5=203 hits (99% ep!) · P31=134 · P14=122 · P25=37 · P12=28 · P36=23 · P27=4
(failed_hit=152 / failed_NOHIT=37 / success_hit=0 — success 侧零误报, vision 规则面干净)

## 3. Tier-2 新发现 — vision 特征失败模式

1. **图中图 OCR 崩坏** ⭐ (144/145/146 交易截图金额, 误差 24×-856×; 147/148/149 食物计数连续三 task 全错;
   102 计数漂移; 5 评论数误读): vision 对帖内嵌图的精确数字/计数**根本不可靠** — vision-mode 最系统性的弱点,
   与 cls §327 "OCR✓/photo✗" 分层互补 (此处连数字都错 = 分辨率/UI 遮挡下的 OCR 天花板)。
2. **submission_image_trap 跨 mode 复现** (34/87/88/100/136/191 + verify task 0): click 缩略图 → 裸图 URL
   dead-end。som 同名模式 (87/176/177/193) → **mode-agnostic 规则成立** (som=SoM 标注诱导, vision=坐标点击
   同落点, 机制殊途同归)。
3. **外站逃逸** (2 deadline.com / 96 wsj.com×3 / verify task 4 guardian.com / 100 wikipedia): vision 看不到
   URL 文本上下文 → 点外链无感知, 被 paywall/CAPTCHA 吞步数。som 只 1 例 (205) → **vision 放大项**。
   规则候选 `obs_url 非 localhost` (与 som R-som-7 合并, mode-agnostic 检测 + vision 加重)。
4. **表单/提交残缺** (69 form-submit-fail / 72,74 评论打进搜索栏 y<100 / 138 type 后没点 Save /
   verify 89 type-no-submit / 171,190 subscribe 零 mutation): vision 坐标级交互的固有噪声。
   规则候选: `comment 任务 ∧ submit_create_count==0 ∧ agent_finished` (与 som R-som-2 同源, mode-agnostic)。
5. **语义/导航误判** (99 全站 /comments feed 当帖子评论区 / 206 搜索页 finish 幻觉 top-comment /
   130,105,106,107,171 论坛错订 / 38 拍摄地 vs 画面内容混淆 / 129 同坐标 12 连点)。
6. **task 138 (改用户名) 注记**: agent type 'Patrick' 后未点 Save → 用户名实际未改 → 本条 fire 无 B-1884
   自毁风险 (B0 vision 能力不足反而"保护"了账号; Fix-4 幂等复原仍是承重墙)。

## 4. Verify 批 — Tier-1 因果折扣 (vision·reddit; 与 som 对照)

| 规则 | 样本判定 | vision 折扣 | som 对照 | 备注 |
|---|---|---|---|---|
| P5 (203) | causal (task 0, dead-end) | **3 折** ⚠️ | 9 折 | 99% fire rate 区分度崩; 仅 dead-end URL (.jpg/.atom/外域) 场景作死因 |
| P31 (134) | presence (task 4 = 外站逃逸终态) | 6-7 折 | 7-8 折 | terminal flag |
| P14 (122) | causal (task 11 真震荡) | 7 折 | 5-6 折 | vision 搜索-0-结果震荡是真环 |
| P25 (37) | presence (task 44 = 购物段上游失败) | 7 折 | 9+ 折 | 跨站 task 上游失败也 fire |
| P36 (23) | causal (task 89 type-no-submit) | 8 折 | 5-6 折 | vision fire 少而准 (som 是 thumbnail 混淆下游) |

**方法论 catch**: 同一规则的因果折扣**随 mode 漂移** (P5 som 9折 vs vision 3折; P36 反向) —
freeze v8 时 per-rule 折扣须 per-mode 标定, 不能全局单值。

## 5. 🔁 Self-evolving — 规则候选 (落码 defer 至 v8 freeze)

R-vis-1 = R-som-1 submission_image_trap **mode-agnostic 化** ⭐ · R-vis-2 offsite-drift (`obs_url 非
localhost`, 与 R-som-7 合并) · R-vis-3 searchbar-type (`comment 任务 ∧ type y<100 ∧ submit_create_count==0`) ·
R-vis-4 finish-on-search-page (`finish ∧ url 含 search?q= ∧ string_match`) · R-vis-5 media-mismatch
(`answer URL 后缀 ≠ intent 要求格式`) · R-vis-6 subscribe-no-mutation (`subscribe intent ∧
effective_mutating_action_count==0`) · P5 加 dead-end-URL 子信号分级 (§4 3折问题的结构修复)

## 6. 跨 mode 定性对照 (som ↔ vision, provisional — 禁定量, 待 v8 freeze)

- **共同 (mode-robust, module 不可 route 救)**: submission_image_trap · 提交残缺 (comment/save/subscribe) ·
  token-granularity FP (125 同 task 同款) · config-FP (103/104 全 mode 必败) · 外站逃逸 (vision 放大)。
- **vision 特有/放大 (representation 可 route 救)**: 图中图 OCR 崩坏 (som 有 [SOM_MARKS] 文本兜底处 vision
  裸奔) · 搜索栏误 type (som 有元素 id) · 全站 /comments 语义迷路。
- 与 §320 router crux 一致方向: vision 的独特失败集中在"需要精确文本/结构信息"处 — routing 信号面。

## 7. Actionable

- [x] task 104 config-FP → **B-1885** 登记 (som/vision 2-AI 双证 + 全站扫描: 205 config 恰 103/104 两个)
- [ ] scored_task_count 是否豁免 103/104 = estimand 决策, defer advisor (B-1885 §处置; 现状 205 分母不动)
- [ ] v8 freeze 时: per-rule 折扣 per-mode 标定 (§4 catch) + R-vis-* 落码
- [ ] B0 red 剩余 psom (在跑) / pprompt land 后各自 /diag → reddit B0 4 digest 齐
