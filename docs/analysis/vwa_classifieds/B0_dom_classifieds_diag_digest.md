# B0 dom classifieds — diag digest pointer (dual-run preserved)

> Run-to-run dual-digest preserved per user directive 2026-05-26 (笔记 §297-298 H1 sensitivity 一脉 — dom canonical 是 paper Risk 6 hero number 来源). 默认名指针文件, **不含数字** — paper / cross-mode aggregator follow links 到具体 RXXXXX digest, 不直接读本文。
>
> **补救记录 (2026-05-26)**: R31194 digest 在 commit `3af4c4a` 被 R21557 overwrite (default-name 覆盖陷阱 — 与 vision 同源, 但 dom 是 paper hero data 更关键). 本次从 git `de0ae65` (overwrite 前最后一个 R31194-state commit) 恢复到 `B0_dom_classifieds_R31194_diag_digest.md`, ruleset 已是 `4-domsomvis-b1860coord` 全量重扫后版本 → 与 R21557 **同 ruleset 完全可比**.


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
| SR | **17.41%** (39/224) |
| failed + hit | 179 |
| **failed NO-hit** | **6** |
| success + hit | 29 |

v8 新规则 failed 侧: {'P43': 64, 'P45': 35, 'P44': 34}；success 侧: 无。
（`P43` 在 cls 上大量命中属预期 —— 它标记"intent 需要视觉 + 该 mode 无页面截图"这一**中性组合**，
并非预测失败；§387.10 实测补上截图的增益 ≈0。）

全部 36 个 canonical condition 现处同一版本 → **cross-mode / cross-site 聚合解锁**。

---
## Canonical pointers

| Run | File | Date | SR | Ruleset | Status | Data location |
|---|---|---|---|---|---|---|
| **R21557 (current canonical, paper-grade Phase 1a fire-3)** | [[B0_dom_classifieds_R21557_diag_digest]] | 2026-05-26 | **17.4%** (39/224) | `4-domsomvis-b1860coord` | **active** | `results/visualwebarena/phase1/B0_dom_classifieds_..._R21557/` |
| **R31194 (archive clean replicate; H1 sensitivity replicate)** | [[B0_dom_classifieds_R31194_diag_digest]] | 2026-05-23 (digest), 2026-05-25 全量重扫 ruleset 4 | **15.18%** (34/224) | `4-domsomvis-b1860coord` | archived | A100 `results/repro_replicates/B0_dom_classifieds_R31194_clean_replicate/` (NOT `_archive_amend07_seqid_R31194_dom` — R21557 digest header 写错, 实际是 §297 clean replicate path) |

## Run-to-run summary (paper Risk 6 / H1 sensitivity 实证基底)

| 指标 | R31194 (archive) | R21557 (current) | Δ | 解读 |
|---|---|---|---|---|
| N | 224 | 224 | 0 | manifest 池一致 |
| **SR** | **15.18%** | **17.4%** | **+2.2pp** | **paper §298 hero number** (McNemar p≈0.44 不显著, §242 partial @88 误测 +8pp → canonical +2.2pp 收窄实证) |
| coverage (failed) | 87.9% (167/190) | 88.1% (163/185) | +0.2pp | **highly reproducible** — deterministic 层稳 |
| no-hit failed | 23 | 22 | -1 | **13 个 R31194 no-hit 在 R21557 仍 no-hit** (84/97/106/124/129/131/207/208/216/217/218 等) = 规则盲区是 condition-level 稳态 |
| scaffold-bug | 0 | 1 (T180 widget) | +1 | **sampling artifact** (R21557 sub-agent batch 命中, R31194 错过) — 非新现象 |
| benchmark-FP | 0 | 1 (T216 cross-run 同替代 item) | +1 | 同上 sampling artifact |
| P6 success-fire FP | 30/34 = 88% | 26/30 = 87% | -1pp | **rule-level FP rate sticky** — run-to-run 不漂, 是 condition-level 特征 |

## 拆解 (差异来源)

- **B0 MoE 非确定性** (§242, §298): 字节相同输入 stochastic argmax (主导)
- **per-condition docker fresh restart** (B-1839): cart/listing/comment 空
- **AMENDMENT_07 影响 ≈ 0**: dom 用 native nodeId, sequential-id 改动只动 SoM-family; **dom R31194 ↔ R21557 = fresh-vs-fresh out-of-sample 检验** (R21557 digest "跨 run 一致性" 章节明示)
- **sub-agent sampling artifact** (T180 / T216): batch-by-batch 罕见模式覆盖度差异, 与 SKILL Tier-2 budget cap (50 ep) 取舍一致

## 关键洞察 (paper-grade)

✅ **deterministic 层 highly reproducible** (coverage 87.9↔88.1, no-hit 盲区 ≥13/23 stable) — paper §3.5 sensitivity 透明性强证据
⚠️ **Δ=+2.2pp McNemar 不显著** — paper Risk 6 现 hero claim; §242 partial @88 误测 +8pp 收窄到 canonical 全 224 +2.2pp = paper hero number 收敛证据
⚠️ **P6 success-fire 87-88% FP rate sticky** — P6 carveout 仍 open (R31194/R21557 digest 均列 actionable, 待 6-mode freeze 后落码)
⚠️ **scaffold-bug + benchmark-FP 各 +1 是 sampling artifact** — 不是 substrate degradation, 是 sub-agent batch 命中盲点不同

## Cross-link

- 同 condition 不同 mode (dual-run preserved): [[B0_vision_classifieds_diag_digest]] (vision dual-run R24792/R32024) / [[B0_som_classifieds_diag_digest]] (som single — som-family 受 AMENDMENT_07 影响, archive R9725 待评估是否做 dual-digest)
- 笔记 chronicle: 实验笔记 §297-§300 (run-to-run noise 拆解闭环) / §298 (dom canonical landed +2.2pp 不显著)
- paper Risk 6: paper_planning Risk 6 + AMENDMENT_06/07 + phase1_plan §D4
- AMENDMENT_07 archive policy: 笔记 §301 (Phase 2 disk migration + manifest rebind follow-up; lesson "AMENDMENT_07 archive 应同 commit update manifest")

---

### v11 数字块（`11-intent-text-fallback`，2026-08-03 补）

> 本 digest 正文成稿于更早的 ruleset。v10 落了 **+P49 / P36 carve-out / P14 carve-out**，
> v11 给 **P34/P48 换用 `_finish_intent_text()`**（answer 为空时 fallback 读 `thought`——
> B0 惯于把结论写进 `answer`，B1 留在 `thought`，旧口径因此变成了模型行为检测器）。
> 全部 48 个 canonical condition 已在 v11 下重扫，**cross-mode / cross-model 聚合以本块为准**。

| 字段 | 值 |
|---|---|
| Run | `B0_dom_classifieds_20260525_194618_553890342_530647_R21557` |
| Episodes | 224（success 39 · SR 17.41%） |
| 三子集 | failed+hit 177 · failed-NO-hit 8 · success+hit 29 |
| config_missing | 0 |

| 规则 | 含义 | step 级 | episode 级 |
|---|---|---:|---:|
| `P36` | WALK_FAIL_DEGENERATE | 280 | 84 |
| `P6` | 视觉任务 DOM 必然失败 | 80 | 80 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 64 | 64 |
| `P16` | 视觉图像内容DOM必败 | 46 | 46 |
| `P5` | 感知缺失循环 | 48 | 39 |
| `P17` | click-back振荡 | 32 | 32 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 35 | 28 |
| `P14` | URL 自环 | 26 | 25 |
| `P2` | 容器节点误点 | 68 | 22 |
| `P31` | budget耗尽未完成 | 20 | 20 |
| `P7` | sCity=州名 | 19 | 18 |
| `P20` | 评测目标页从未访问 | 15 | 15 |
| `P44` | HALLUCINATED_ELEMENT_REF | 34 | 15 |
| `P18` | cheapest漏价格排序 | 13 | 13 |
| `P10` | 跨步数值记忆失败 | 13 | 12 |
| `P33` | 导航至裸图片URL幻觉 | 9 | 9 |
| `P23` | oldest误用价格排序 | 8 | 8 |
| `P27` | 找不到即放弃 | 7 | 7 |
| `P19` | url_match过早搜索页finish | 6 | 6 |
| `P15` | gallery行位置DOM不可定位 | 5 | 5 |
| `P30` | 到达正确item后离开 | 5 | 5 |
| `P24` | 不确定仍finish | 5 | 5 |
| `P21` | dom模式视觉幻觉 | 5 | 5 |
| `P25` | 跨站任务跳过其中一站 | 5 | 5 |
| `P22` | 图中数量dom不可数 | 4 | 4 |
| `P12` | 从不翻页 | 4 | 4 |
| `P13` | 搜索代替浏览 | 3 | 3 |
| `P37` | URL_HALLUCINATION | 3 | 3 |
| `P28` | benchmark-FP货币tokenize | 2 | 2 |
| `P4` | 根节点误操作 | 2 | 2 |
| `P38` | DOM_URL_AS_IMAGE | 2 | 2 |
| `P32` | 文本误入价格filter | 1 | 1 |

> ⚠️ **解读约束**（`docs/analysis/_data_quality_audit.md`）：
> ① 本表是**症状分布，不是死因分布** —— P36/P31 经 10 例跨 benchmark 因果验证均判为 risk-marker；
> ② `P2`/`P4` 依赖 `element_bbox`，在 **vision 上结构性为 0（假 0）**；
> ③ `P36` 在 vision 上只覆盖 `type` 步（click 无 `locator_route_meta`）→ **分母与 dom/som 不同**。
