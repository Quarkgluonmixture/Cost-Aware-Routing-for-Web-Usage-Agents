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

### Q3 🟡 Single seed=42, no replication study

**Issue**: 整篇 paper 数据 single seed run. SR delta 数字 (e.g., red P-SoM 13.81% > SoM 10.48%) 没有 across-seed variance estimate. Bootstrap CI 是 **task-level binomial** (~±5pp), 不 capture **across-run sampling variance**.

**Current state**: paper Section 4 用 bootstrap CI 数字, 但 reviewer 可能 ask "你 seed=42 跑一次, seed=43 一样吗?".

**Options**:
- (A) 加 N=3 seed replication (seed=42/43/44) on critical cells (B0 cls/red 5-mode = 10 cells). Cost ~$60 on RunPod (3× cost of one cell × 10 cells × $2/cell), small相对总 budget.
- (B) Single-seed paper, disclose "Seed=42 single replicate; bootstrap CI captures task-level variance, not seed variance." 风险: reviewer 不 buy.
- (C) Subset replication: only 2-3 most controversial cells (e.g., red P-SoM vs SoM) get N=3. Cheaper (~$20) compromise.

**Lean**: C. 关键 P-SoM vs SoM 配对加 replication 是 paper-grade rigor 最强 ammunition, additional ~$20 cost negligible.

---

### Q4 🟡 Cross-site SR 不直接可比

**Issue**: cls/red/shop 任务池不同 (234/210/466 tasks), 任务难度不同, agent capability 不同 per site. paper 写 "Phantom-SoM 13.81% on red > 10.48% on red" OK, 但 "Phantom-SoM red 13.81% > SoM cls 21.37%" 不直接 mean Phantom 比 SoM 弱.

**Current state**: paper Section 5 已经 site-modulated framing (paper_planning line 363).

**Options**:
- (A) 加 explicit Section 4 limitation note: "We do not claim cross-site SR dominance; comparison is within-site between modes."
- (B) Show normalized cross-site metric (e.g., relative SR vs DOM baseline per site).
- (C) Skip cross-site comparison entirely, paper 只 within-site claims.

**Lean**: A + B. Add disclosure + show relative-to-DOM normalization figure. 不需 new data.

---

### Q5 🟢 FP filter (na_fp / eval_fp) is post-hoc mode-asymmetric

**Issue**: §95 FP filter 体系 trigger depends on agent finish answer + GPT-judge. 不同 mode 产生不同 finish pattern → filter 应用 asymmetric.

**Current state**: §95 chronicle 已经讨论, FP rates per mode 都 < 5%, 整体 cross-mode bias 小. Section 4 标 "adjusted_success" 跟 raw_success 都 cite.

**Options**:
- (A) Status quo: 现 framing 已经 OK, 不需 action.
- (B) Section 4 加 footnote: "FP filter is mode-asymmetric; reported numbers are adjusted_success."

**Lean**: B. 简单 footnote 加 paper-grade rigor, 0 cost.

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

## 总结 — 4 个真 actionable 给学长 raise

按 paper-impact 排序, 给 sync 4-question:

1. **Q1 Early-stop micro bias** — 推荐 B+C (length-normalized + demote 2a/2b/2c). 不需重跑.
2. **Q2 B0 pre/post sampling 不对称** — 推荐 A (only 14-cell rerun data). 需 commitment "弃用 archived B0".
3. **Q3 Single-seed replication** — 推荐 C (~$20 partial replication). 加 paper rigor.
4. **Q4 Cross-site comparability** — 推荐 A+B (disclosure + relative normalization). 不需重跑.

其他 Q5-Q9 都是 1-句 footnote disclosure, 不需 ask 学长 (我自己 sign off 写进 Section 4).

---

## Companion docs

- `docs/checkpoints/paper_planning.md` §6 Critical Risks (4 risks 已 framed, 这 9 个 questions 是 risk 之外的 framing decisions)
- `docs/reference/VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md` §3.5 acknowledged-not-fixed (NOT_A_BUG 类)
- `docs/checkpoints/master_bug_catalog.md` (technical bug detail)
- `docs/reference/PHANTOM_SOM_ADVISOR_MEETING_BRIEF.md` (会议讲稿)
- `docs/reference/ADVISOR_SYNC_DRAFT_2026-04-30.md` (本 sync 草稿主体)
