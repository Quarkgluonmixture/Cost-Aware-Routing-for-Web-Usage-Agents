# Paper Strategy Open Questions — Audit (2026-04-30)

**Purpose**: meta-audit of paper framework — find issues that look "fixed" but actually have framing/methodology risks 学长 should sign off on.
**Trigger**: user 提的 early-stop micro bias 问题. 反思整个 paper-grade pipeline 还有什么类似 "looks-OK-but-actually-confound" 的角落.
**Audience**: advisor meeting, methodology review

---

## Methodology summary 的 9 个 open questions

按 paper-impact × discovery-difficulty 排序. 每个含 (a) 问题描述 (b) 当前状态 (c) 选项 + cost (d) 我的 lean.

---

### Q1 🔴 Early-stop bias on micro metrics

**Issue**: agent cycle-detect 早停 truncate trajectory, 让 micro Section 5 metrics 部分 cross-mode 不可比.

| Micro metric | 是否受早停影响 |
|---|---|
| 2a URL Jaccard | 🔴 严重 (短 trajectory URL 数差 3×) |
| 2b target-page hit rate | 🔴 严重 (短 trajectory inherent 概率低) |
| 2c keyword-repeat | 🔴 严重 (短 trajectory 没机会 repeat) |
| 2f first-divergent step | ✅ uncensored by design |

**Current state**: Cluster 3 fuzzy cycle hash 让早停 trigger 准 (B-11 fix), 但**早停存在本身**仍然 introduce mode-dependent truncation.

**Options**:
- (A) 关掉早停, 全跑 max_steps=30. Cost +$1300 on top of Phase A 14-cell ($200 → $1500). Cleanest 但 expensive.
- (B) 保留早停 + length-normalized micro metrics. 不需重跑数据, 改 analysis script ~50 LOC.
- (C) Demote 2a/2b/2c 到 secondary, Section 5 主用 first-divergence (2f) + macro outcome.
- (D) 给 paper-grade rigor 子集 (e.g. 50 tasks all 5 modes ran to max_steps): unbiased subset for micro analysis.

**Lean**: B + C combined. 不需重跑, paper rigor 透明 disclose.

---

### Q2 🟡 B0 pre/post Phase A sampling 不对称

**Issue**: Phase A 之前 B0 用 `temperature: 0.1` (stochastic); Phase A 之后改 `T=0.0` (greedy). 14-cell re-run 用 post-Phase-A 代码 → 数据是不同 sampling regime.

如果合并 archived (T=0.1) + 14-cell re-run (T=0.0), 数据**不来自同分布**. paper claims 应该:

**Options**:
- (A) 只用 14-cell re-run 数据 (post-Phase-A T=0). 抛弃 archived. Cost: archived ~30 cells 的钱浪费, 但 paper data clean.
- (B) 合并使用, disclose "B0 split between two sampling regimes; SR estimates aggregated". reviewer 可能 challenge.
- (C) Re-run archived B0 with T=0.0 too (additional ~30 cells cost ~$200 on RunPod). 两套数据都 paper-grade clean.

**Lean**: A. Phase A 14-cell re-run 已经覆盖 paper 主线 cells (cls + red 5-mode + shopping critical), 不需要重跑 archived.

---

### Q3 🟢 Environment non-determinism (REFRAMED 2026-04-30 from "single seed replication")

**Original framing (incorrect)**: "single seed=42, no across-seed replication".

**Reframe (post user critique)**: At post-Phase A B0 T=0 + top_p=1.0 + B1 `do_sample=False`, **LLM-level replication 没有 variance** to measure (greedy 严格 deterministic up to CUDA float-point ~1% top-1 flip). 真正的 trajectory variance 来源是 **environment non-determinism**:
- Page load timing (Tailscale + Magento response 抖动)
- Async render timing
- Site state mutation (Magento DB writes accumulate from prior agents/tasks)
- busy:1 wait 时长 (~30-50s 抖动)

**Implication**: 同 task 同 seed 跑两次会在某步 diverge — LLM 收到同 obs 进同 action, 但 next obs 因 env 非确定性 differ → trajectory drift. 这跟 LLM 采样变化无关.

**Current state**: paper hasn't disclosed this. Bootstrap CI 是 task-level binomial (~±5pp), 不 capture environment variance.

**Options**:
- (A) Section 4 1-line disclosure: "Post-Phase A LLM is strictly greedy (B1 do_sample=False, B0 T=0+top_p=1.0); residual trajectory variance comes from VWA environment non-determinism (page load timing, async render, site state mutation), not LLM sampling. Bootstrap CI captures task-level binomial variance only."
- (B) 跑 environment determinism probe (3 ep × 3 replicates each task on B1 to measure env-induced trajectory divergence). Cost ~$5 only B1 (B0 too expensive to replicate).
- (C) Skip — accept env variance as inherent to web agent benchmarks (no prior paper has measured this either).

**Lean**: A (single footnote). B 可选 if want to quantify but not required.

**Status**: ❌ **Retracted as advisor-ask** (per user 2026-04-30: greedy 下 LLM-level 没 variance to measure). Self-decision: 加 Section 4 footnote, no replication needed.

---

### Q4 ❌ RETRACTED — Cross-site SR comparability already handled

**Original concern**: cls/red/shop 任务池不同, cross-site SR 不可比, 应该加 disclosure + normalize.

**Retraction reason** (per user 2026-04-30): paper_planning §3 line 363 已经明确 "site-modulated representation × prompt × image effects, 不是 Phantom #1 universal routing arm". Section 5 主线一直是 site-aware. **加 disclosure + normalize 是 over-engineering** — paper 当前 framing 已经 sufficient.

**Status**: ❌ **Retracted**. Current site-modulated framing is paper-grade adequate.

---

### Q5 ❌ RETRACTED — FP filter asymmetry is feature not bug

**Original concern**: FP filter mode-asymmetric, 应该 footnote disclose.

**Retraction reason** (per user 2026-04-30): FP filter 是行为研究**方法论必需的 measurement instrument** (na_fp 防 agent 说 "task impossible" 被 GPT 误判 success; eval_fp 防 string_match 偶然碰对). 不同 mode 产生不同 failure pattern, filter 应用 asymmetric **正是 filter 在做它该做的事 (instrument-aware adjustment)**, 不是 confound.

类比: 不同 thermometer 在不同温度区间灵敏度不同, "灵敏度 asymmetric" 是 instrument 特性, 不是 bias.

**Status**: ❌ **Retracted**. FP filter is measurement, not contamination. Current "raw_success vs adjusted_success" Section 4 framing 已 sufficient.

---

### Q6 🟢 Diamond completion 只 partial done

**Issue**: P-prompt diamond corner 只 reddit done (~210 ep), cls 待跑, shop 没 plan. paper Section 5 mechanism 论证用的 diamond axis-decomposition 在 cls 缺数据.

**Current state**: paper_planning §2 line 437 已经 frame, B0 P-prompt cls 在 Phase A 14-cell rerun 队列里.

**Options**:
- (A) Phase A 14-cell rerun 包含 P-prompt cls + shop, problem solved.
- (B) Section 5 主线只 reddit diamond, cls/shop disclaimer.

**Lean**: A. 14-cell 全跑包括 P-prompt 是 default 计划, 不需特殊 ask.

---

### Q7 🟡 B0 vs B1 cross-baseline 比较的 sampling regime 差

**Issue**: B0 (post-Phase A) T=0 greedy via API; B1 (post-Phase A) `do_sample=False` greedy via local transformers. 不同 sampling code path.

**Subtle issue**: B0 API 服务器 internal 可能 still 有 sampling (proxy 不保证 deterministic), B1 local 严格 greedy. 所以 "B0 deterministic = B1 deterministic" claim weak.

**Options**:
- (A) Disclose in Section 4: "B0 deterministic relies on proxy provider's temperature=0 implementation; B1 strictly greedy via do_sample=False + torch.manual_seed."
- (B) Subset replication on B0 to verify deterministic at proxy level (3 ep × 3 calls each, see if outputs match).
- (C) Skip — accept asymmetry as inherent to API vs local model setup.

**Lean**: A + B. API determinism 验证 cheap (~$5).

---

### Q8 🟢 Drop-one oracle calculation depends on observed mode set

**Issue**: drop-one oracle = max(remaining modes) per task. 数值 depends on 我们 observe 的 mode 集合. 如果以后加 cross-model (Claude Opus) data, drop-one 数字会变.

**Current state**: paper Section 1 报 1.7-3.3pp drop-one 是 5-mode set (DOM/SoM/Vision/P-text/P-SoM).

**Options**:
- (A) Section 4 disclosure: "Drop-one oracle defined on this paper's 5-mode set; numbers may shift with added modes."
- (B) Lock 5-mode set in paper, future cross-model paper redo drop-one analysis.

**Lean**: A. 1-句 footnote 透明 disclose.

---

### Q9 🟢 Routing AUROC 是 in-sample evaluation

**Issue**: Section 6 Phantom-SoM signal AUROC 0.793 等数字是 same-data evaluation (signal computed on same trajectory it's predicting). 没有 train/test split.

**Current state**: paper_planning §8 Router Design 提及 routing 是 Tier 1+2 (heuristic-based + signal-based, no learned router).

**Options**:
- (A) Section 6 cite "AUROC is in-sample, no learned router train/test split needed since signals are deterministic features."
- (B) 加 test split: 用 50% of episodes 算 signal, 测 50% AUROC.

**Lean**: A. Tier 1+2 routing 不是 learned model, in-sample = out-of-sample assumption holds. Disclose framing in Section 6.

---

## 总结 — 修订后 (2026-04-30 user critique 后)

**给学长 ask 收缩到 2 个 (Q1 + Q2)**:

1. **Q1 Early-stop micro bias** 🔴 — 推荐 B+C (length-normalized + demote 2a/2b/2c 到 secondary). 不需重跑.
2. **Q2 B0 pre/post sampling 不对称** 🟡 — 推荐 A (only 14-cell rerun data, 弃 archived). 需 commitment "丢弃 archived B0 数据".

**Retracted (per 2026-04-30 user critique)**:
- ~~Q3 Single seed replication~~ → reframed as Q3 environment non-determinism, demoted to self-decided 1-line footnote (greedy 下 LLM-level 没 variance 测)
- ~~Q4 Cross-site SR comparability~~ → already handled by site-modulated framing in paper_planning §3
- ~~Q5 FP filter mode-asymmetric~~ → FP filter is measurement instrument, asymmetry is feature not bug

**Self-decided footnotes (不 ask 学长)**:
- Q3 environment non-determinism disclosure (1 line)
- Q6 diamond completion (covered by 14-cell rerun)
- Q7 B0 vs B1 determinism asymmetry (1 line + optional $5 verification)
- Q8 drop-one oracle 5-mode set scope (1 line)
- Q9 routing AUROC in-sample assumption (Section 6 framing)

**Net effect of revisions**: advisor sync ask list 从 4 个 framing decisions 收缩到 **2 个**. 减负 + 学长聚焦更高 stakes 的真 framing decisions.

---

## Companion docs

- `docs/checkpoints/paper_planning.md` §6 Critical Risks (4 risks 已 framed, 这 9 个 questions 是 risk 之外的 framing decisions)
- `docs/reference/VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md` §3.5 acknowledged-not-fixed (NOT_A_BUG 类)
- `docs/checkpoints/master_bug_catalog.md` (technical bug detail)
- `docs/reference/PHANTOM_SOM_ADVISOR_MEETING_BRIEF.md` (会议讲稿)
- `docs/reference/ADVISOR_SYNC_DRAFT_2026-04-30.md` (本 sync 草稿主体)
