# B0 som reddit — /diag digest (3-tier)

- **Run**: `B0_som_reddit_20260627_035453_162107997_3024022_R20936` (manifest-bound authoritative, bound 2026-06-28)
- **Condition**: `phase1_som_router_0` · site=reddit · model=B0 (Qwen3-VL-235B proxy) · mode=som
- **N**: 205 ep · **SR**: 14.6% (30/205)
- **ruleset_version**: `7-p6p16clsgate-b1860coord`
- **Diag date**: 2026-07-02 (Tier-2: 8 sonnet sub-agents; 40/46 no-hit 采样 + 5 verify + 1 success 审计)

> ⚠️ 单 condition digest, 不下 cross-mode 定量结论 (discover-then-freeze 协议; reddit 6-mode 齐后 v8 freeze 重扫)。
> ⚠️ Tier-2 batch A3 (task 92/95/100/101/102/103, 87-103 簇中段) 首跑撞 sub-agent session limit, 重试结果见
> 末尾补录区 — 未补录前本 digest 三分类统计基于 34/40 深挖样本。


## 0b. v8 freeze 补记（2026-07-27）

`RULESET_VERSION` 已升至 **`8-reddit-p41p46-b1890fix`**（reddit 规则批 P41–P46 + B-1890 修复 + P33 reddit
路径扩展）。**全部 36 个 canonical condition（reddit 18 + cls 18）已在该版本下重扫**，
版本一致性由 `scripts/analysis/diag_rescan_all.py` 校验 → **cross-mode 聚合解锁**
（正文顶部"待 v8 freeze 重扫"的前置条件现已满足）。

本 condition 的 v8 数字 —— **跨 condition 聚合请用这一组**：

| 指标 | v8 |
|---|---|
| SR | **14.78%** (30/203) |
| failed + hit | 147 |
| **failed NO-hit** | **28** |
| success + hit | 10 |

> ⚠️ **B-1913 (2026-07-27)**: 上表 SR 已对齐 AMENDMENT_08 计分集 (reddit **203**)。此前写作 collected 分母 205 —— 那是 scored rate 配 collected 分母。分子同样按计分集重取 (被排除的 task 58/160 若曾计为成功则一并移出)。**权威来源 `docs/analysis/cross_sites/sr_per_mode.json`**；本文件其余 `/205` 计数是 episode 级覆盖率, collected 分母正确, 未改。


v8 新规则 failed 侧命中: {'P46': 8, 'P45': 48, 'P44': 1}；success 侧: 无。

正文的 Tier-2 定性结论不受版本变更影响 —— 新规则只是把此前 no-hit 的 episode 归了类，
未推翻任何已有归因。**若本 condition 的 success 含 task 160 / task 58，请一并读
master_bug_catalog B-1889 / B-1892**（评测器缺陷，非模型表现）。

---
## 1. 三分类统计 (Tier-2 深挖子集, 34 no-hit + 5 verify + 1 success 审计)

| 类别 | 计数 | 说明 |
|---|---|---|
| agent-limit | 41 (37 no-hit + 4 verify-确认) | 压倒多数; 细分见 §3 |
| benchmark-FP | 4 | task 134 (live 漂移) + task 4 (P40 真阳性) + task 95 (NLTK slash-token) + task 103 (config 错 URL) |
| scaffold-bug | 0 | 1 个候选 (task 205 外部导航) Tier-3 forensic 后 defused, 见 §4 |
| unclear | 0 | |

## 2. Tier-1 规则分布 (failed-only; hit ≠ 死因, 折扣见 §5)

P36=321 hits · P5=110 · P31=100 · P14=53 · P12=38 · P25=37 · P4=16 · P10=3 · P27=3
(failed_hit=129 ep / failed_NOHIT=46 ep / success_hit=1 ep [P40])

## 3. Tier-2 新发现 — no-hit 盲区的 6 个可复现模式

1. **submission_image_trap** ⭐ (87, 176, 177, 193; A6 估计其它 reddit 图任务高频): SoM 把帖子缩略图/参考图
   标成可点元素 → click 落到 `/submission_images/<hash>` 裸图页 → back → 再 click 同一元素, 循环 2-13 步。
   最强新规则候选: `obs_url ~ /submission_images/ 且下一步 action=back, 连续 ≥2 对`。0-token 纯 URL+action 序列。
2. **正确答案 + 错误终结动作** (90, 91: 未用 comment form — program_html 任务, mutation 从未发生;
   视觉答案完全正确, task 90 "4 teeth" = exact_match 参考)。
   规则候选: `eval=program_html(comment locator) ∧ submit_create_count==0`。
   **⚠️ 125 已 Tier-3 harness 亲验重分类** (2026-07-02): `environment.py:581` 明写 **string_match is
   finish-answer-based** — answer 路由正常, sub-agent "评测只读 send_msg" 的 claim 错误 (log-only
   over-claim 第 3 例)。真死因 = **tokenize 粒度**: "$27.0B" 语义对但 token '27.0' ≠ must_include '27'
   → **125 重分类 benchmark-FP (token-granularity 族, 与 95 slash-token 同族)**; vision 125 同款 (跨 mode)。
   133 维持 agent-limit (答案 42 vs 参考 23, 真错)。
3. **multi-target 群发只发第一个** (202, 203; intent_template_id=84): 11/7 个收件人只发 1 个即 finish。
   规则候选: `must_include 用户名 ≥5 ∧ steps≤10 ∧ fail`。
4. **参考图 → 错帖/错用户 mis-grounding** (1, 2, 5, 34, 38, 108, 109, 126, 127, 133, 140, 145, 172, 173):
   最大簇, 多数无稳定 0-token signal; 172/173 例外 — `eval_target_url ≠ eval_source_agent_url` (纯 summary 级)
   可写 **P-wrong-post-url**。108/109 连续两 task 把同一 Kim16 帖匹配给不同参考图 = 视觉匹配按 prior 不按图。
5. **过早放弃** (32: user-profile 0-scroll 即 "no submissions"; 197: 把 `f/food` 当搜索词 2 步放弃; 175):
   规则候选: `finish@/user/*/submissions ∧ 无 scroll ∧ answer 含 no-submissions 措辞`。
6. **back-click 短循环** (37: 同一被拒帖 3 次重访): P17 存在但未 fire → P17 阈值窄漏, freeze 时复核。

## 4. Tier-3 forensic 复核 (不信 log-only 判定)

- **task 205 "goto twitter.com 撞 CAPTCHA" ≠ scaffold-bug**: steps JSONL 实证 goto 被 whitelist 正常拦截
  (`dispatch_path=offsite_blocked`, `error_category=policy_blocked_offsite`, Amendment 01 语义正确);
  agent 落在外部新闻站是**更早 click 帖内外链**所致 (VWA 真实浏览器 upstream 语义, 非 P79 层)。维持 agent-limit。
  附带规则候选: `obs_url 非 localhost` 检测 click-施加的 offsite drift (比 goto 检测更通用)。
- **task 4 (P40 success 审计) = 真 benchmark-FP**: agent 走错帖 (f/Washington vs f/OldSchoolCool), 两帖 'wheat'
  评论数巧合同为 1, string_match 放行。P40 设计目的实证有效 (marking-only 不动 SR)。
- **task 134 = 新 FP 类**: live 站点评论数从参考 29 漂到 34 → string_match 必败。**stale-reference numeric drift**
  — 与 memory `reference-vwa-design-quirks` 的站点状态演化一致; freeze 时考虑 FP 规则
  (`答案为计数型 ∧ agent 到达 reference URL ∧ 数值 > 参考值`)。

## 5. Verify 批 (A8) — Tier-1 主导规则因果折扣 (som·reddit)

| 规则 | 样本判定 | 折扣建议 | 备注 |
|---|---|---|---|
| P36 (321 hits) | presence (task 8) | **5-6 折** | 单次 failed-click 粒度过细; 大量为 SoM thumbnail 混淆的下游 |
| P5 (110) | causal (task 14) | 9 折 | streak 高时归因准 |
| P31 (100) | causal-浅层 (task 3) | 7-8 折 | terminal flag 非独立死因; task 3 上游=两阶段 SoM 元素混淆 |
| P14 (53) | presence (task 11) | 5-6 折 | 可被 2 次 click-fail 误触, 无真自环 |
| P25 (37) | causal (task 44) | 9+ 折 | 高特异; task 44 = plan-action gap (thought 知道还有 reddit 段却 finish) |

## 6. 🔁 Self-evolving — 提议规则 (落码 defer 至 reddit 6-mode 齐 v8 freeze, per discover-then-freeze)

R-som-1 submission_image_trap ⭐ (V2 证实 vision 87/88 同机制 → mode-agnostic) · R-som-2 comment-form-miss
(`submit_create_count==0`) · R-som-3 **token-granularity FP** (取代原 "string_match-no-send_msg" 错误前提;
统一覆盖 95 slash-token + 125 '27.0'≠'27': `must_include ref ∉ tokenize(answer) ∧ ref 为 answer 某 token
的前缀/子串` → FP 标记) · R-som-4 multi-target-single-send · R-som-5 wrong-post-url
(`eval_target_url≠eval_source_agent_url`, summary 级) · R-som-6 user-profile-0-scroll-quit ·
R-som-7 offsite-drift (`obs_url 非 localhost`) · P17 阈值复核 · P36 加 "后续是否恢复有效行动" 辅助信号 (降 presence 噪声)

## 7. Actionable

- [x] **task 125 answer-routing 抽查 DONE** (environment.py:581: string_match = finish-answer-based, 路由正常;
      125 重分类 benchmark-FP token-granularity 族, 见 §3.2)
- [ ] **task 103+104 config 错 URL 已全站扫描确认** (205 config 中恰 2 个, 都指向 task 102 帖子;
      全 mode 全 model 不可赢, 对 mode 间对比无偏 [全员同败] 但压 SR 天花板 ~1pp; → B-number 登记)
- [ ] A3 批 6 task (92-103 簇中段) 补录 (session-limit 重试中)
- [ ] reddit 6-mode 齐后: v8 freeze + 全量重扫 (R-som-* 落码窗口)
- [x] task 205 scaffold 疑点 — defused (本 digest §4)

## 8. 补录区 (A3 retry, 2026-07-02 已补) — 92-103 簇中段 6 task

- 92 AL (图中 dog 答 cat) · 100 AL (submission_image_trap 再证, 与 87 同款) · 101 AL (文字帖幻觉有图) ·
  102 AL (计数漂移 "4 red keys"→答 0, 参考 3)
- **95 = benchmark-FP (新类: NLTK slash-token)**: 答案 "purple/pinkish" 被 word_tokenize 保成单 token →
  must_include "purple" 不在 token 列表 → fail。sub-agent 实测 NLTK 复现。规则候选 **R-som-8
  P-FP-slash-token**: answer 匹配 `<ref>/<other>` 粘连模式 ∧ ref ∈ must_include。
- **103 = benchmark-FP (新类: task-config 错 URL, ⭐ B-number 候选)**: task 103 的 `program_html[1].url`
  指向 **task 102 的帖子** (f/MechanicalKeyboards/56362) 而非自己 start_url 的 f/memes/41674 —
  agent 在正确帖发了正确评论仍必败。**影响所有 mode 所有 model = task 103 系统性不可赢**, 属上游 VWA
  task config copy-paste bug (对照 B-91/B-1878 同类上游议题)。规则候选 **R-som-9 P-FP-eval-url-mismatch**
  (config 加载时比对 start_url vs program_html[*].url 路径, 0-token)。
- 87-103 簇定性: **非 transient/infra 簇** — 全部为独立 agent-limit/FP, 相邻只因任务模板同族
  (MechanicalKeyboards/memes 图任务带)。

## 9. Actionable 增补 (post-A3)

- [ ] **task 103 config-URL bug → master_bug_catalog 登记候选** (benchmark-FP 类, 全 mode 影响;
      建议顺手扫全 210 reddit task config 的 start_url↔eval URL 一致性 — R-som-9 即是该扫描)
- [ ] task 95 slash-token FP → freeze 时与 P28/P29 (既有 benchmark-FP tokenize 类) 合并考虑

---

### v11 数字块（`11-intent-text-fallback`，2026-08-03 补）

> 本 digest 正文成稿于更早的 ruleset。v10 落了 **+P49 / P36 carve-out / P14 carve-out**，
> v11 给 **P34/P48 换用 `_finish_intent_text()`**（answer 为空时 fallback 读 `thought`——
> B0 惯于把结论写进 `answer`，B1 留在 `thought`，旧口径因此变成了模型行为检测器）。
> 全部 48 个 canonical condition 已在 v11 下重扫，**cross-mode / cross-model 聚合以本块为准**。

| 字段 | 值 |
|---|---|
| Run | `B0_som_reddit_20260627_035453_162107997_3024022_R20936` |
| Episodes | 205（success 30 · SR 14.63%） |
| 三子集 | failed+hit 146 · failed-NO-hit 29 · success+hit 10 |
| config_missing | 0 |

| 规则 | 含义 | step 级 | episode 级 |
|---|---|---:|---:|
| `P31` | budget耗尽未完成 | 100 | 100 |
| `P5` | 感知缺失循环 | 110 | 63 |
| `P33` | 导航至裸图片URL幻觉 | 52 | 52 |
| `P14` | URL 自环 | 52 | 39 |
| `P12` | 从不翻页 | 38 | 38 |
| `P25` | 跨站任务跳过其中一站 | 37 | 37 |
| `P36` | WALK_FAIL_DEGENERATE | 282 | 35 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 48 | 26 |
| `P4` | 根节点误操作 | 16 | 9 |
| `P46` | COMMENT_INTENT_NO_TYPE | 8 | 8 |
| `P10` | 跨步数值记忆失败 | 3 | 3 |
| `P27` | 找不到即放弃 | 3 | 3 |
| `P48` | PREMATURE_NEGATIVE_AFTER_SEARCH | 2 | 2 |
| `P49` | SUBMIT_PAGE_ANCHOR_MISCLICK | 1 | 1 |
| `P44` | HALLUCINATED_ELEMENT_REF | 1 | 1 |

> ⚠️ **解读约束**（`docs/analysis/_data_quality_audit.md`）：
> ① 本表是**症状分布，不是死因分布** —— P36/P31 经 10 例跨 benchmark 因果验证均判为 risk-marker；
> ② `P2`/`P4` 依赖 `element_bbox`，在 **vision 上结构性为 0（假 0）**；
> ③ `P36` 在 vision 上只覆盖 `type` 步（click 无 `locator_route_meta`）→ **分母与 dom/som 不同**。
