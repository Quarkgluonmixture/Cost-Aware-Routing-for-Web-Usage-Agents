# 裁定层 A3（§165–§240，229 条 ADJUDICATED，2026-05-16 → 2026-05-20）

Claude 主 session 逐条通读产出，2026-07-28。**聚合非转写**：逐条索引见 `ledger.jsonl`。

这一片是 **fire 前最后冲刺 + Fire-1~Fire-6 实战期**。四条主线：
统计口径最终定型（bootstrap percentile 取代 Wald-Z、DL/HKSJ 四层退役、drop-one 真 null 换成 permutation）、
router 从 E'' 走到 E'''（user 五连 OOB catch 后成为 canonical sklearn Pipeline-in-CV）、
GLM rescue 完全退役 + B0 proxy 协议层收口、以及 **archive 数据全面弃用**。
最后 5 天是真 fire：asyncio race / cross-site contention / quarantine 阈值，全部靠实战暴露。

> ⚠️ **跨批核对提示**：A 批看不到 RETRACTED（B 批）与带数字 MEASURED（D1–D4）。标 `⚠️ 待跨批核对` 处，
> 合并阶段须按 § 号回标。

---

## 一、统计口径最终定型

### 1.1 SE floor 的两次锚定

| § | 值 | 依据 |
|---|---|---|
| §172.4 (05-17) | **保持 const 1.0pp**，**拒绝 (B) N-aware floor 与 (C) exclude degenerate cells** | **archive median SE ≈ 0.98pp ≈ 1.0pp 说明 const 是经验校准**；N-aware 让各 cell floor 不同会**引入额外 noise**；exclude degenerate **违反 prereg L586 的 fixed-cells-as-design 原则**。并在 prereg 加 **Agresti-Coull-style disclosure 段**锚定 archive median |
| §211.4 (05-18) | **B-1003 改 0.68pp**（Agresti-Coull anchor） | prereg amendment intent 与代码对齐 |

> ⚠️ 两条并列。§172.4 的**三条拒绝理由仍然有效**（防止有人重提 N-aware / exclude-degenerate），
> 但**具体数值已由 B-1003 改为 0.68pp**。引用 SE floor 数值时以 §211.4 为准，引用"为什么是 const"以 §172.4 为准。

### 1.2 primary p-value 从 Wald-Z 换成 bootstrap percentile

- **§211.4 B-1009**：primary p-value 用 **bootstrap percentile**，**Z-test 仅 transparency**。
- **§215 B-1301 落实现**：canonical producer 加 **`_pool_bootstrap_percentile_p()`**
  （**Davidson-MacKinnon 2000 / Hall 1992 标准 paired-bootstrap pool**：
  固定点估计 IV 权重 × per-cell bootstrap θ_i_b 矩阵 → pooled θ_FE_b 分布 →
  在 **H0: θ_FE ≤ theta_null_pp** 上取 percentile p）；
  substrate `aggregate_phase1_prereg_gate.py:149-160` 暴露 **per-cell boot_pp float32 数组**（**pre-fix 被丢弃**）；
  `h1_pass` 切到 `gate_passed_bootstrap`；CSV +3 列；
  MD 分 **Primary（bootstrap percentile）/ Transparency（Wald normal-Z）**两节；
  H3 轴同步（B-1302，**legacy Wald CI 降为 `passed_wald_ci_legacy`**）。
  理由：**prose-promise 必须有可复现的 artifact 支撑，否则 OSF reviewer 复现时对不上**。

### 1.3 DL/HKSJ 四层退役（§215 B-1305）

Makefile:262 从 `_aggregate` 里**摘掉 phantom-meta**；新增 `phantom-meta-appendix` alias 并在调用时显式 echo；
`aggregate_phantom_meta.py` docstring **撤回"FE would contradict the paper hook"的理由**（**与 decision 3A 直接矛盾**）；
`fig_meta_forest.py` docstring 改标为 **paper §8 Appendix-D sensitivity figure** 并警告 **§1 hero 必须用 canonical FE pool**。
理由：*"B-1016 的 MD-warning header **只退役了输出 prose label，没退役 pipeline 调用和内部理由文档** ——
**`make analysis` 每次仍跑 DL/HKSJ**"*。

**§172.6 同类**：**B-437** `aggregate_phantom_lift` 输出里的 "Hero/PRIMARY" 措辞改为
**"APPENDIX legacy exploratory"** 并加顶部警告；**真正的 paper §1 H1 PRIMARY 是 `phase1_prereg_gate`**。
理由：**两个产出物都自称 hero 会让 reviewer 无法确定哪个是 preregistered primary**。

**§172.6 B-427**：`use_adjusted` 默认翻成 **False**（`aggregate_cross_site.py:341` + `compare_b0_b1.py:340`），
`--use-legacy-adjusted` 作 opt-in。理由：**3-AI overlap 最高信心 —— 默认走已退役的 adjusted 路径违反 prereg**。

### 1.4 drop-one 的真 null（§207.3 B-893，3-AI overlap）

**drop-one oracle 的真 null = fixed-marginal permutation reference，不是 H0: drop_one=0**。
落代码：`permutation_drop_one_null()` + CLI `--permute-marginal-null` + `--permutation-B`
（**default 10000, prereg-locked**）+ `stratified_bootstrap_lift_ci()`。
理由：*"**设计层修复不能纯 prose，否则 reviewer「可以但你做了吗」→ 必须有真 CLI flag + 测试**"*。

### 1.5 H1/H3 的其余锁定

- **§165.4 P0-7 走 option A**：改 paper §1 prose 的 **B=10000 为 B=1000** 对齐 prereg；
  prereg 与 OSF 保持 1000；hero p-value 在 rerun 后按 B=1000 重算。理由：**最便宜且不动已 lock 的 prereg**。
- **§165.4 Q11**：H3 axis2 universe 选 **(B) universe_6 六臂 complete-case**。
  user 显式：*"paper §1 framing 是 **6-arm 且 P-SoM 是 hero**，P-text 与 P-prompt 是 **co-equal 的 axis-decomposition arms**，
  不是非对称的 'P-prompt is THE axis'"*。
- **§211.4 B-1002**：`P79_PAPER_GRADE=1` 下 **k<6 → canonical producer emit `gate_status=DEGRADED` + `missing_cells` 字段**，
  paper §1 必须 disclose。理由：**k=6 strict gate；防止 cell 不全时静默出 verdict**。
- **§211.4/§211.6 B-1001**（**唯一真正的 Phase-1a-fire blocker**）：canonical producer 的 **H3 KeyError 用
  `.get('passed', False)` 防御**，让 framing 把该轴当 **cannot-evaluate**。
  pre-fix 会在**首次 fire 时 H3 noise-floor filter 剩 <2 cells 就 crash**。
- **§215 B-1311**：§1 prose 明确 **H1 = 对 6 个 planned cell 的单一 one-sided fixed-effects pooled bootstrap
  percentile 检验，不是 per-cell 检验家族**；§4 的 per-cell drop-one + bootstrap CI 是**伴随 transparency 表**。
  理由（gemini unique OOB）：原 prose *"reported as a separate per-cell p-value column in §4"*
  **暗示 6 个 per-cell 检验，制造 multiple-testing surface 混淆**。
- **§240 B-651**：Holm family key 从 `(test, metric)` 改 **`(test, metric, cell_key)`** 做
  **cell-scoped 多重比较校正**（**stratified design 的正确做法**），改动 corrected p-value 与 `holm_family_m`。
- **§211.2**：**P0-4-C（R5 heterogeneity cap）从 P0 降 P1**，走 Appendix-D sensitivity disclosure，不改 prereg 主体。
  依据：**archive I²=0% 表明攻击在 hero arm 上未激活**（Q2 实证核查）。

### 1.6 H2(a) cost falsification（§172.5）

`cost_margin_pct` 默认 **10.0 → 20.0** 对齐 prereg L131-132 的 **1.20× lock**；
**K-of-N transparency 语义改为 strict ALL-pass**（`consistent = (pass_count == n_cells_total)`），
`transparency_K_h2` 参数 DeprecationWarning 并忽略；新增 `n_cells_falsified` + `semantics` + `prereg_anchor` 字段。

**过程注记**：*"gemini 原 claim（gate 不查 cost）**被 verify 推翻** —— `evaluate_h2_cost` 已存在于
`preregistration_decision_test.py:517-557`；**但 grep 发现两个真 bug**：margin 2× 过严导致 false-falsification 膨胀，
且 prereg L131-132+L368 要求 **'ANY condition violated → falsified' 的严格全过语义**，
**K-of-N 是把 H1/H3 的 transparency 语义误植到 H2(a)**"*。

### 1.7 H10 —— 从统计检验改成 operational gate

- **§218.3 Q4=A**：H10 estimand primitive = **K-of-6 PRIMARY descriptive + APPENDIX FE pool sensitivity**。
  理由：**K-of-6 保住 site-asymmetric viability 叙事**（prereg §C5 pre-locked）；
  **FE pool 在 site 方向相反时会互相抵消**。
- **§221.2 B-1550 two-layer operational deployment gate**：
  - **cell-level** = 每 cell 做 **paired bootstrap B=1000 seed=42 on (Cost, SR)**，
    若 router 在 **≥95% bootstrap replicate** 上相对 5 个 single-mode baseline 是 **Pareto non-dominated** 则该 cell pass
    （**重采样单元是 cell 内 per-task label，不是跨格的 cell**）
  - **grid-level** = 6 个 pre-registered cell 里 **≥5 个 pass** 则 H10 deployable
  - **5/6 是 fixed-cell operational deployment criterion，NOT a binomial significance test**

  **为什么这不违反 K-of-N 退役 doctrine**（一字不丢）：*"6 cells = decision 3A 下的 **complete pre-registered
  decision family（finite population）**，prereg 不把 6 cells 当随机样本 → 跨格 binomial 推断不自洽；
  §2.4 的 K-of-N fake-precision 退役 doctrine 针对的是「**跨 cell 在 α 上做 binomial 推断**」，
  而 **operational gate 不设跨格 α、无 Type-I/II 耦合、5/6=0.833 是精确的工程 margin 不是显著性水平**"*。
- **§221.1（user 即时纠正，原话 *"对，你这个反驳是对的，而且很关键"*）**：
  **archive ≠ prereg substrate —— 不能用 archive 模拟的 FAIL 投影去 demote H10**。
  理由：*"archive 自己声明不是 prereg substrate；**若当前 E'' router 从没在 archive 里按同一 protocol 跑过，
  archive verdict 就不能判 H10 死刑**。正确路径是重写 K-of-6 为 operational gate + **删掉 prereg 里所有 archive-derived rationale**"*。

---

## 二、Router 最终设计 —— E'' → E'''

### 2.1 CV 结构：LOCO 被 supersede

**§216.1 Q4=C-modified（05-18）**：H10 primary CV = **(E'') task-held-out 5-fold within fixed cells
+ per-cell LR head + fold-local TF-IDF/MI top-18**；**LOCO 6-fold 降为 Appendix sensitivity**。

user 理由（两条）：
1. **LOCO 与 per-cell LR-head 架构冲突**（每个 cell 需要自己的 LR）
2. **LOCO 不是真正的分布外 held-out** —— 各 cell 共享 Phase 1a task pool；**Phase 1b shop 才是唯一真 OOD**

> ⚠️ **本条 supersede A2 §154.2（2026-05-16）的 "paper §6 主数字锁定在 Phase 1a LOCO" 决定。**
> 引用 router CV 设计必须用 §216.1 之后的版本。

**§218.3 Q1=C hybrid**（**2-AI codex+gemini 推翻 Claude 自审的 Q1=A**）：
within-cell 5-fold CV deployment + honest prose + per-task-lookup baseline。
理由：**8-12h 投入换 credible held-out narrative shape vs 1-2h prose 投入换 hedged in-sample claim；
paper §6 的 contribution shape 比 effort metric 重要**。

### 2.2 (E''') 最终设计（§218.2/§218.3）—— canonical sklearn Pipeline-in-CV

- **global fold-local pooled MI**（每个 fold k **同时排除所有 cell 的 fold-k holdout**，每 fold 出 5 个统一 selector）
- **fold-local TF-IDF**（**不能 pooled-on-full-data 的 transductive**）
- **vectorizer state 按 fold pickle**
- **cell-constant 特征（site / capability_tier）不进 per-cell LR pool**
- **StandardScaler 放进 Pipeline**
- **τ 只在 train fold 内 inner-CV 调**（`StratifiedKFold(5)`, **τ ∈ [0.3..0.7]**，**平手取最高 τ = 保守路由**），
  **绝不在 outer holdout 上调**

来源：**user 的 5 连 OOB catch**（#1 cell-count vs task-id leak / #2 pooled MI leak / #3 fold-local /
#4 global fold-local / #5 GPT-relay 4 条 sklearn 严谨性）。

**§218.7 由此确立**：**user-in-the-loop 被确认为第 4 条 lineage（ML implementation correctness 层）**。
理由：*"**3-AI cycle 在 statistical-methodology-on-paper 层触底**；user 经 GPT relay 带回的
**4 条 sklearn pipeline 正确性规则是 3 家 AI 全漏的**"*。

### 2.3 §224 —— user 30 秒反转 Claude 的 vectorizer 划分

**Q4=B（user 反转）**：router 用 **shared fold-local vectorizer + per-cell LR heads**（最终 E'' 设计），
**否掉 Claude 推荐的 per-cell vectorizer**。

理由：*"per-cell vectorizer + shared `selected_idx` **语义不自洽** ——
**feature-index 0 在 cell A 表示 color 在 cell B 表示 forum**，而 `selected_idx_fold{k}.json`
假设跨 cell 的 feature-index → semantic 映射稳定"*。
**教训（一字不丢）**：*"涉及 shared/per-cell 划分时，**追一个 feature-index 端到端走完 pipeline 验证划分自洽**"*。

### 2.4 feature pool 与 fail-loud

- **§217 Q2=A 从 router feature pool 删掉 `capability_tier`**。
  user 架构层论证：*"per-cell 训练下 `capability_tier` 在 **cell 内对所有 task 恒定** →
  **LR 学不到任何 task-level decision boundary，最多变成 intercept 的一部分**；
  若 MI selector 选进 top features 还会**浪费 feature slot**"*。
- **§224 user mandate**：**infrastructure 级错误（artifact 缺失 / pickle 损坏 / dim mismatch）必须
  raise RuntimeError**（`LearnedRouterArtifactError` + hard-fail）；
  **只有 task 级 signal-strength fallback（`max_prob ≤ τ`）保持静默并计数**。
  user 原话：***"silent fallback 是 H10 最大风险之一"***。
  配套：B-1642（`ALLOW_PARTIAL_BASELINE` paper-grade hard-block）+ B-1643（validate-each-pickle preflight）。
- **§190 Q1=B 不做 aliasing shim** —— `condition_id` 从旧格式改成
  `phase1_learned_router_{backend_id}_{site}` 是 **breaking change 但无需 archive 兼容层**。
  user：*"Pass-2 router 都没开始跑，没有 archive 需要兼容"*。
- **§190 三条 fail-loud**：`conditions.py` 在 `obs_mode='learned'` 且走 `emit_baseline` 路径时 **raise（B-692）**；
  runner 的 LR dispatch 包 try/except 并加 **`_lr_fallback_count` metric + 首次触发 ntfy + log.error（B-693）**；
  **`RuleBasedRouter` 遇到不在 modes 列表里的 `current_mode` 直接 raise 不再静默回落 `modes[0]`（B-697）**。
  理由：**防止 learned router 污染 baseline cell 的 SR 数据**；LR pickle/numpy/version 不匹配时必须**可审计地降级**；
  **静默回落会让 router 行为不可解释**。
- **§166.3 T1-4=B**：learned router 测试 **defer 到 Pass-2 fire 前 1 周**（deadline-driven）。

---

## 三、paper §1 hero claim 的诚实化（3-AI 反复打同一点）

### 3.1 §207 —— 三条 framing 修正

- **B-892（3-AI 独立打到同一点：Claude F2 + codex F3 + gemini F4）**：
  §1 段落 2 的"**empirical 4-fold drop-in property**"拆成 **construction-vs-empirical 两层** ——
  **2 条 by-construction（架构后果）+ 2 条 emergent（实证验证）**。
  理由：*"把 **2 by-construction + 2 emergent 包装成 1 个 emergent claim 是 paper hero 的最大裸露面**"*。
  > 这条直接对应当前 CLAUDE.md 里的 hook 措辞（"constructed substrate, NOT a discovery"）。
- **B-894**：phantom-space boundary 统一为 **axis-based**（**DOM = origin baseline，
  3 perturbation arms = {P-text, P-prompt, P-SoM}**），消解 **3-arm vs 4-corner 两个不兼容定义**。
  > 注：**Claude+gemini 都抓到；codex 反而把 sharp boundary 当强 claim** —— 三家判断不一致的实例。
- **B-900**：§2 的"**metadata-rich vs compact**"改为**结构-格式框架**
  （*same element semantics; preserves hierarchical nesting vs flattened into sequential indexed list*）。
  理由：跟 memory `project_phantom_space_axes_format_not_information.md:27-29` 的 framing rule 矛盾
  （**gemini 的 cross-section consistency 攻击**）。

### 3.2 baseline 比较的因果收窄（§208.4）

- **B0-vs-B1 = joint deployment-class × capability-scale comparison（不是纯 capability ablation）**
- **B1-vs-B2 = parameter-parity robustness check（不是 full matched-capability anchor）**
  理由：P1-13-C/P1-14-C 指出**原文的 causal 过度宣称**。
- **scroll-action vocabulary disclosure 从"asymmetry, not a confound"升级为"execution-layer threat"**
  + **reddit cross-baseline SR 解释上限 caveat**（P1-15-C gemini attack）。

### 3.3 cost 单位不可混算（§181，3-AI overlap 最高信心）

引入 **`cost_unit_basis` schema 枚举 + `cost_total_mixed_unit_warn` 标志 +
paper §1 的 `[^cost-basis-cross-baseline]` footnote**，把**跨 baseline cost claim 收成 within-baseline-ratio framing**。
理由：**B0 API USD 与 B1/B2 electricity USD-equivalent 差 1000 倍，直接混算会被 area chair 攻击**。
（Claude schema 层 + codex producer bypass + gemini paper prose 攻击草稿 三家同时命中。）

### 3.4 canonical cost/latency estimand 三轴（§219，user 拍板）

user 原话：***"更准确的 canonical 选择应该是：Cost canonical = raw/billed cost. Latency canonical =
retry-adjusted latency. Raw latency = sensitivity / operational robustness."***

机制理由：**cost 在结构上本就 retry-clean**（B0 proxy `usage.cost` 按构造**只在成功时计**；B1/B2 无 retry），
**latency 才需要 retry 调整**（B0 网络 retry 等待 **10-70s × backoff** 是 **scaffold 级开销**，B1/B2 无对应）。

> **§232 B-1669 后续扩展**：canonical latency **同时扣掉 retry 与 busy_wait** ——
> `aggregate_cross_site.py:282` 新增 `avg_busy_wait_total_ms`（来自 step record 既有字段 `busy_wait_total_ms`,
> `types.py:337`）+ `avg_total_latency_canonical_ms` 列；**raw 与 minus_retry 保留为 sensitivity**。
> 触发：**99s busy-wait 污染 A2.7 的 latency canonical estimand**。

### 3.5 外部效度的收缩与边界（§216 / §217 / §223）

- **§216.1 Q3=A**：**cross-site 结论定位为 boundary-mapping exploratory observation，
  不是 R1-R5 framing 下的 pre-registered hypothesis**（降低 external-validity 攻击面）。
- **§216.1/§216.4 Q6=A**：**R3 fallback 放弃 Zoom 2-4 框架，退到 efficient-heuristic-arm framing**。
- **§216.1/§216.4 Q7=A**：**workshop 与 main 分叉** ——
  **workshop_R1 = H1 + H2(a) ONLY**（不含 H3 structural）；
  **main_R1 = 完整 H1 + H2(a) + H3(i) + H3(ii) + H10 over Phase 1a + Phase 1b 9-cell**；
  **权威 source = `paper_planning §16.0` submission matrix**。
  触发：**gemini P0-7-C\* R3 framework collapse + P0-8-C\* dual-submission novelty trap**。
- **§211.2 Q3**：**B2 进 main pool，不设 carve-out fallback**。
  依据（**文献三锚**）：**Mirage Effect（Asadi 2026）跨族验证 GPT-4V/4o + Claude 3 + Gemini 1.5 Pro +
  Qwen-VL/LLaVA/InternVL** / **Scaffold Effect（Vu & Balloccu 2026）** / **Sclar 2024 prompt format sensitivity**
  支持跨族外推；**B2 carve-out = cherry-pick 攻击面**；§8 的 **4 条 lineage 轴（chat × IT × tokenizer × vision encoder）
  bounded-comparison 已处理最坏情况**。
- **§217 Q1 第三选项（user 提出，优于 Claude 给的 A/B）**：
  **保留 prereg §7 finite-population FE estimand（B2 留在 pool）**，
  另加 **B2-outcome-gated 的 cross-family R-tier downgrade 作为正交的 claim-tier 轴**。
  user 原话：***"B2 失败 = 不能 claim cross-family / cross-capability robustness，并且 R1 强 claim 要 downgrade。
  也就是说，B2 应该影响 claim tier，但不一定推翻所有现象"***。
  效果：**既保 prereg §7 'cells = design, not population sample' 的 scope 诚实，又给 R5 reviewer 一条具体的 falsification rule**。
- **§212.1/§212.3 Q1**：**B2 mechanism trust = NO fallback**；
  **prereg §7 scope-statement 本身就是 falsification rule**（5-min inline disclaimer 而非 R3 cap）。
- **§223 B-1631 术语边界**：**cross-family = B2 vs Qwen 在 4B parity 下（paper-1 scope 内）**；
  **cross-architecture = {Qwen3-VL, Gemma3-VL} 信封之外的 VLM 族 = 显式 future work**。
  理由：**防 R3 攻击"你们的 Gemma 不就是 cross-arch 吗"**。
- **§223 B-1630 deployment portability 的 operational test**：
  **对首步 observation 跑 `_extract_text_marks` regex 的通过率** ——
  **≥95% 判 drop-in；30-95% 判需要 adapter；<30% 判需要 full rewrite**。
  新增 §3.2 operational portability boundary 表（6 行 × 6 列）。
  理由：**原来 "VWA-AXTree-equivalent" 的可移植性声明只有 gestural enumeration 没有可操作检验**。
- **§223 B-1626**：§8.8 **WA Jaccard 预测加 mini-prereg** —— **分层 50-task + 固定 seed + bootstrap CI +
  三态判定**（**CI 上界 ≤0.5 → Generalizable；CI 下界 >0.7 → VWA-specific；否则 Inconclusive**）。
  理由：原来的 ≤0.5 / >0.7 二元预测**没写样本量**，且 **[0.5, 0.7] 40% 灰区未定义**。

---

## 四、GLM rescue 完全退役 + B0 proxy 协议层收口

### 4.1 三层禁用（§169.4/§169.6，05-16）

**GLM rescue 在 paper-grade run 全面禁用，三层叠加防御，任一层 fail-fast**：
- **B-395**：`P79_PAPER_GRADE=1` env → `normalize_config` 合并到 top-level flag
- **B-396**：**39 个 live B0 yaml** `use_glm_fallback` true→false
- **B-340**：RuntimeError raise

user 2026-05-16 directive（一字不丢）：***"Phase A 永远要求最 clean 的 paper grade，GLM rescue 禁止，paper grade run 没有"***。
→ 落为 memory `feedback_phase_a_always_clean_paper_grade.md`。

### 4.2 完整 retire（§208.3/§208.4，05-17）

**GLM rescue 完整 retire**（prose + config + docs 层由本 session，**code 层由并行 session**）。
**触发（关键实证）**：*"user 实测 **AWS proxy 实际支持 OpenAI-style `tool_choice`**，
返回 **`tool_calls[0].function.arguments` 的 schema-compliant JSON + logprobs + top_logprobs[2]**；
**capability gap 在 P79 payload 端（Anthropic-style 被 400-reject），不在 proxy 端**"*。

> 这条**推翻了 A1 §70（2026-04-16）"proxy 静默忽略 tools 参数"的结论** —— 真因是 P79 发的是 Anthropic-style payload。

### 4.3 §210 B-991 协议层收口（05-17）

- 删 GLM rescue 代码路径（**~155 LOC**：`_load_glm_config` + `_call_glm_extract` + Path-3 fallback +
  `urllib.request` import + GLM init）
- `_WEB_ACTION_TOOL` 从 Anthropic-style 换 **OpenAI-style** `{type:'function', function:{name, parameters}}`
- **`thought` 加入 required**
- `step()` 里在 Anthropic `content[].tool_use` **之前先解析顶层 `body['tool_calls']`**
- payload 加 **`logprobs=True` + `top_logprobs=2`**
- **`use_glm_fallback: true` 无条件 raise RuntimeError**
- **4 个 GLM meta key 保留为 schema v2 zombie 字段**
- `exp_v2_base.yaml` **`use_tool_calling` false→true**

**proxy 形状定性**：**hybrid**（Anthropic endpoint + X-Api-Key header + OpenAI-style tools schema +
顶层 tool_calls/logprobs 响应）。

### 4.4 ⚠️ `tool_choice` 的未完结局

**§210 Q1=A**：`tool_choice='auto'` 作为 production default 落地（**N=30 pilot emit_rate ≥95% 通过**；
probe 显示 **auto 下 emit=100% 且 schema valid=100%**）。

> ⚠️ **两处冲突信号**：
> (a) 台账 flag：§210 的三条记录都挂 **`named by RETRACTED §214`** ——
> *"Q1=A pilot gate PASS: B0 proxy full-stack probe `schema_valid_rate=1.0`（§210）"* 被作废；
> (b) CLAUDE.md 现记载 **Fire-6 RCA 2026-05-20：旧 `"auto"` 在 proxy 下 emit 0% tool call → 改 `required` emit=100%（B-1794）**。
> ⇒ **`auto` 已被 `required` 取代**，但**取代的 § 在 A4**。⚠️ 待跨批核对 §214（B 批）+ A4 的 Fire-6 RCA。

### 4.5 B0 可审计性（§229 B-1588）

`proxy_api_agent.py` 加 **per-step 三字段 instrumentation** ——
`tool_call_emitted: bool` / `tool_call_parse_path: str` / `tool_call_fallback_reason: Optional[str]`。
理由（codex unique OOB）：*"paper §3 关于 B-991 native tool-call substrate 的声明**此前没有 per-step 实证支撑**
（**B0 tool-call emit rate 不可审计**）"*。
> 这条 instrumentation 正是后来 Fire-6 能 RCA 出 "emit 0%" 的前提。

### 4.6 B0 provenance（§200 Q22=C / §229 B-1589）

- **§200 Q22=C**：**B0 provider snapshot id 不实现**，按 prereg §7 line 548 已有的
  *"Verifiable from traces, replayable subject to API access"* 作 **acknowledged disclosure**（proxy 侧不暴露）。
- **§229 B-1589**：`snapshot_env.py` 的 `_capture_api_proxy_provider_info` 改读
  **canonical `repo_root/configs/exp_v2_base.yaml`** 抽 B0 `base_url`，并记录
  **`endpoint_source` / `endpoint_env_value` / `endpoint_config_value` 三元组**（原来记的是 env 而不是 yaml）。

---

## 五、Evaluator 政策的最终形态

### 5.1 极性 bug（§178，最能说明"为什么修上游"）

**修法 (b-fix)**：**反转判断顺序** —— **先判否定短语**
（fuzzy: `incorrect` / `partially correct` / `not correct`；ua: `different` / `not the same` / `not same`）
**再判肯定**，**默认 fail-closed**；**明确不设 ambiguous middle 分类**。

**决策过程（值得记）**：*"user 先说 '其他 VWA paper 也是这样' **倾向不修**，
但随后**自己提出 B-91 precedent** —— 既然 llm fuzzy match 的 NA FP 已经 patch 上游并 disclose，
严格 string 判定也应同样处理；**'no ambiguous middle' 是 user 显式否决**"*。
落地：`external/visualwebarena` submodule commit **1c3a615**。

### 5.2 §178.5 —— evaluator-patch policy 定档

**修上游 evaluator bug 并 disclose divergence；不为了直接可比而保留会系统性 inflate SR 的上游 bug；
内部有效性靠 within-paper paired comparison 而非 cross-paper SR 绝对值比较。**
背景：post-fix P79 evaluator 已是**第 3 处与上游 divergence**（judge model gpt-4o-mini / B-91 empty-pred guard / polarity）；
**VWA / WebArena-Verified / PAE / Aviator-Web 仍用 upstream substring matcher**。

### 5.3 evaluator 失败 = fail-fast（三次收紧）

| § | 变更 |
|---|---|
| §180 | evaluator 在 `paper_grade=True` 时 **hard-raise（新 `EvaluatorUnavailableError`）**；**reward override 机制整体 rip out** 并 disclose。理由：**evaluator fail-open 会造成跨 baseline infra-fragility confound；reward override 造成 estimand schizophrenia** |
| §197 Q3 | evaluator init 失败走 **option (d) 根因修复** —— preflight 加 `--paper-grade` 标志**在子进程里实例化 `VwaEvaluator(paper_grade=True)`**；B-544 的 process-fatal 保留为第二道防线。理由：**根因修复 1.5h 让失败在 10s 探针暴露，而 symptom-quarantine 要 4h 且会在 36-condition 批次的第 1 个 condition 上崩** |
| §232 B-1662 | evaluator 基础设施失败 = **condition fail-fast abort** —— 在 `_FATAL_ENV_REGEX.search()` **之前**先检测 `EvaluatorUnavailableError` 直接 raise，**不再往 canonical `episodes/` 目录写 `needs_reevaluation=True` 的 summary**。理由：避免 **canonical 目录污染 + condition 尾部崩溃** |

**§175b 175.3/175.4 配套**：异常路径 `success=False` 硬编码的修法选 **(A') quarantine + 强制 re-run**，
**拒绝 gemini 的 "infer success from last step `action_success`"**。
理由：***"agent 的 `action_success` 不等于 task-level evaluator outcome，用它推断是 risky overreach"***。

**§197 Q4**：**B-329 composite eval_types 判为 GRL 家族**，走 **option (c) defer + paper §3.5.1 disclosure**，不改代码。
命中 **4 条 GRL 判据**（evaluator 侧不对称 / 静默 SR 偏差 / 跨 baseline 不对称 / 修它会破坏 evaluator 纯度）；
影响面 **30/910 = 3.3% paper-grade 任务（reddit ~7%）**，**paired 比较不受影响，幅度有界**
→ **披露比改代码更强的 reviewer 防御**。

---

## 六、Archive 全面弃用（本片最彻底的一条 doctrine）

| § | 裁定 | 依据 |
|---|---|---|
| §184 Q3 | **schema-version gate**：`_collect_condition_summaries` **硬拒**任何含 `adjusted_success` / `fp_reason` / `raw_success` / `is_na_reference` / `adjusted_reason_bucket` 字段的 archive（opt-in `P79_ANALYZE_ALLOW_STALE_ARCHIVE=1` 才放行） | user 原话 ***"所有基于之前 archive 数据的都不可信，都需要现在 phase1 rerun"*** |
| §185 Q3=A | **paper §1 hero 数字只引用 canonical-rerun-at-2f9b0b4 的结果**，archive 一律不用，**archive 降为 Appendix D 参考** | user 显式"之前 archive 数据都不用"；消除 gemini Mode C 指出的 **Phase 1a archive 与 current HEAD 的 version rift confound** |
| §194 Q2=B | **强 enforce 全部 7 个 episode sentinel**（`PAPER_GRADE_EPISODE_OPTIONAL_KEYS`），**不留向后兼容 hook** | user 一句"archive 不进 paper scope"**解除了向后兼容约束**，把默认从 option A（保 archive）改到 option B |
| §210 | archive 数据 **moot** —— "completely ignore，archive 不进 paper-grade" → **P0-3（archive re-fire decision）+ P0-5（aggregator era boundary `b0_transport_epoch`）立即关闭，不需 advisor sync** | user mid-stream 决策 |
| §221.1 | **archive ≠ prereg substrate**（见 1.7） | user 即时纠正 |

**§184 配套（Q2）**：**raw success 是 canonical paper §1 hero**；
`clean_success_rate`（排除 `benchmark_noise` infra）**只作 transparency appendix**；
**明确不再套 §95 + §78a 的 post-hoc FP framework**。
依据：grep §78 + §95 + memory 三方对齐 —— §78（04-18）三层 post-hoc → §95（04-24）删 visual_fp →
**§139.8（05-14）整层 retired，改走 upstream fix**（B-91 guard + `exclude_na_tasks` 默认 true）；
**memory 显式说那是历史背景不要再套**。

**§184 Q1/Q4**：bootstrap denominator 走 **caption-disclosure 不动算法**
（加 `estimand` / `scored_set_n_hero_table` / `estimand_note` 字段，标 *"Conditional-on-observed estimate;
Hero Table 用 scored set"*）；**Q4 = 删除 cumulative SR plot**
（该图**按 task_id 排序 = 伪 learning curve**，user 一词 "Delete" 拍板）。

---

## 七、大规模退役与删除（fire 前减面）

| § | 删了什么 | 理由 |
|---|---|---|
| §171.4/§171.5 | **D1 = A 删除 heuristic 家族** —— 物理删 `p79/backends/heuristic.py` + `p79/experiment/modules.py`，factory 加 actionable ValueError，**M1-M3 调用内联成 baseline-retry generator**，4 个 test 文件跟改（commit 5799fda，11 files +111/-288） | **0% paper-grade 使用率** + paper-2 可从 git history 恢复；**回收 ~150 LOC + 8 个 test 的 audit budget**（对不可达路径的防御性工程） |
| §196 Q4 | **digest 管线全退役（Option α）** —— 删 `_run_auto_digest` / `_check_digest_completions` / `_purge_digest_records{,_batch}` / `_count_failed_episodes_by_mode` / `_DIGEST_MODES` / `--glm-config` / `--digest-dir` / `seen_digest_completions` state，6 个 queue 脚本同步去 flag；**保留 `analyze_reason_diagnostics.py` 与 `glm_batch_digest.py` 作 operator 手动工具** | paper §1 hero claim **不依赖 `digest_*.jsonl`**，paper §4 证据链走 `trajectory_events.jsonl` + Option K aggregator **独立管线**；**一次退役同时关掉 5 项**（P0-1 phantom mode 覆盖缺口 + P0-4 count-based verify + P2-2 task_id 碰撞 + B-86 GLM 不对称 advisor blocker + "6-layer claim" 诚实性），**3h 工作 vs 逐个修 ~5h** |
| §173.4/§173.7 Q3=A | **先 grep 生产代码确认 0 callsite 再删** hover/clear/upload dead code（B-439），并在 paper §3.5.2 写 **honest scope（locator-route 只覆盖 click+type）** | 避免删掉实际在用的分支；**honest scope 比 silently 保留 dead code 更 defensible** |
| §174.3/§174.4 | **删除 `degraded_som` schema bool 字段（B-449）**，aggregator 侧改用 **`mark_count == 0`** | archive 实证 **0/6471 触发** + 该 bool **混了三种语义本质 broken** |
| §168.3 | 删除 `wait_for_reddit_then_rederive.sh` + Makefile `watch-reddit` target 退役（B-394） | 功能已被 queue_chain + watchdog 覆盖 |
| §173b | **B-463 `fig0e` category heatmap 延后**（Makefile 注释掉），paper §1 在有 live producer + advisor taxonomy 确认前**不再 cite** | codex 发现 fig0e **只有 archived-only 数据源，无 live producer → 不可复现** |
| §201 Chunk α | `sync_a100_results.sh` 加 **`--delete-after`（不用 `--delete`，为 SSH 掉线安全）**；Makefile analysis post-hook 去掉 `glm-refresh-playbook`；`error_scan` 删已退役的 `fp_adjust_error` 正则；`restart_watchdog.sh` 删已退役的 `--glm-config`/`--digest-dir` 注入 | **A100 clear_tasks 清理必须传播到 DGX 镜像**否则 stale finalized 与 in-flight 混合；**crontab 2026-05-13 已删的 `glm-refresh-playbook` 仍从 Makefile post-hook 走另一条路 fire，使当时的省钱决策结构性落空**；`fp_adjust_error` **追已退役的鬼**（legacy log 字符串被 5min cron 报成 severity-82 红色噪音）；**A1.15 Chunk a 删 flag 时漏了 `restart_watchdog` 这个 rebuilder** |
| §201 | **P0-4（GLM 生成的"建议"进入 PLAYBOOK §1 成为决策 substrate）无限期 SKIP 不修** | user Q2：*"PLAYBOOK 不用管，这个最后会 ignore 掉"* —— **对将退役的 substrate 不投入** |

**§196 Q2 = Option E（一条"部分为真"的处理范例）**：
"**6-layer**"说法**保留**，但改成**逐层 code-site 表**（Layer 1-4 与 6 在 watchdog，Layer 5 在 `runner/main.py:762`，
Layer 6 延迟到下个 task 的 step_000 DOM 检查）+ edge case disclosure（新建 paper §4.X.15 stub）。
理由：*"gemini 攻击 '6-layer 只有 4 层显式'；**实证 Layer 5/6 确实存在但跨组件**；
**部分为真的设计声明应做精确披露而非整体撤回**"*。

---

## 八、canonical substrate 建设（§177）

新建 **3 个 canonical substrate**：
1. **`aggregate_phase1_full_prereg_decision.py`** —— **唯一 emit H1+H2(a)+H3+R1-R5 的 producer**
   （理由：**Makefile 里没有单一产物给出完整 prereg 决策**）
2. **`lib/canonical_cells.py`** —— 收口 **`PHASE_1A_CELLS` 硬编码 + `aggregate_phantom_lift.CELLS` 冻结 + registry live 三源**
   （理由：**triple-source-of-truth 会漂移**）
3. **`validate_run_manifest.py`** —— paper-grade promotion 协议 schema 校验
   （理由：**手改 yaml 漏一行会造成静默 k=5 hero pool**）

**§185 SBOM 配方定档**：从 `git diff | sha256sum` 改成 **tree-hash chain**：
`git rev-list <base>..HEAD --format=tformat:'%H %T' | sha256sum`（**env-independent**），
witness = `5c6c5f625f44ca1b2155b9cad280b5aecb3e6939cf0599540fcef0900028fb0f`；写进 prereg §7 + paper §4.X.11 + memory 三处。
理由：**`git diff` 的输出受本地 git config / 环境影响，不是可移植 witness**。
> **§219 补充**：canonical 配方是 **reverse-chronological**（`git rev-list ... --format=tformat:%H %T`，`snapshot_env` L451），
> **不是 `git log --reverse | sha256sum`（chronological）**。A2.7 Chunk 1 用错配方改了常量，Chunk 3 smoke test 抓到 → 回滚。

**§200 A1.16 cold-start 全推荐**：tree-hash chain SBOM 重算**内置进 `snapshot_env.py`**
（`_vwa_submodule_integrity` 重算 Layer1 HEAD + Layer2 upstream base 可达 + Layer3 tree-hash chain，
**caller 侧 divergence 即 `SystemExit(2)`**）；递归 hash 全部 coco_images（**412 jpgs**）；
evaluator scope 从 **7 扩到 17 文件**；deterministic torch flags 在 paper-grade 严格 / dev warn_only；
`VWA_EVAL_MODEL` env 纳入 capture；`generation_manifest.json` 纳入 scope；
新增统一 helper **`snapshot_has_critical_errors(snap)`** 收口 4 个 fail-loud 信号。
**根因诊断**：*"6 个 P0 里 **3 个是 'snapshot 写了信号但没有 caller 读' 模式**；
**prereg §7 line 568 声称 'divergence aborts the run' 但无代码路径执行该契约**"*。

**§185 配套**：`generate_test_data.py` 改为**幂等 clean-rebuild + byte-deterministic JSON** +
输出 `config_files/generation_manifest.json`（逐文件 sha256）供 **OSF byte-equivalence 审计**
（codex 发现原脚本**非幂等且 JSON 字节不确定，破坏 OSF replay**）；
**async OpenAI 调用改为 fail-loud（raise）而非静默返回空串**
（**静默空返回会把 API 失败与真实的 empty prediction 混同，污染 paper-grade SR**）。

**§191 两条 gate 升级**：
- provenance gate 从 **file-state 检查升级为 git-state 4 层检查**
  （ls 存在 / `git ls-files --error-unmatch` 已跟踪 / `git diff --quiet HEAD` 干净 /
  JSON schema 含 `captured_at`+`host` 且 errors 为空）。
  理由：***"ls 只回答文件在不在，不回答是不是 paper-grade 证据；OSF 审计看的是 git log 与 diff 状态"***。
- submodule lock 检查改 **SHA-first，branch 降为 WARN-only**；加 **ancestor fallback**
  （若 actual != expected 但 `git merge-base --is-ancestor expected HEAD` 成立则 WARN 放行），
  `EXPECTED_SHA_STRICT=1` 可回严格。
  理由：***"reviewer 按 pin `git checkout <SHA>` 必然进 detached HEAD，branch-name 相等检查方向反了 ——
  branch 是可变社交元数据，SHA 才是不可变证据"***；硬编码相等则每次 submodule 前进都要手改 7 处 lock 文件。

---

## 九、Fire 前基础设施硬化

### 9.1 queue / gate 层

- **§183 sentinel 完成率阈值从 90% 改为 100% exact**（并改用 **post-exclusion `scored_task_count`**）。
  理由：**历史实证 23/24 次 EXACT；"90% 阈值"是 lazy default 不是 reality-driven**。
  > 这条**修订 A2 §162.2 的 90% 决定**。
- **§183 WA benchmark 的 reset 从静默 skip 改为 hard-fail**：新建 `reset_wa_sites.sh` 返回
  **rc=78 'not implemented' 哨兵**，4 个 queue script 在 `BENCHMARK=wa + RESET_BEFORE=1` 时 FATAL。
  理由：user Q6 选长期完整实现路径，但 **A100 WA docker 状态无法验证**（实证 `docker ps` 只有 VWA 容器）
  → 完整实现属过早；**静默 skip 是 paper-grade-fatal**。
- **§188 Q1=B**：P0-1 **不只修 typo，审整个 `config_for_cmd` 函数**（typo + 用 `read -r -a` 安全分词替代
  `local parts=( $cmd )` + 显式 `queue_phantom_dom.sh` 向后兼容分支 + default 分支 emit `UNKNOWN_SCRIPT` fail-loud）
  并补 5 个单测。**user 选深度而非速度**。
- **§188 Q3=A**：Gate 3 的 VWA provenance snapshot 从 **WARN 改 FAIL**，错误信息带可执行命令。
  理由：***"WARN 等于 provenance theater"***。
- **§189**：orchestrator 与 queue_chain 从 `set -uo pipefail` 升到 **`set -euo pipefail`**，
  并在 `check_gates` 里对非致命命令**逐条加 `|| true` / `|| rc=$?` 防护**。
  理由：**5/7 leaf queue 已是 -euo 只有这两个不是 = sibling drift**；
  但 **naive 开 errexit 会让 collect-all-errors 计数逻辑在第一个失败处直接退出，所以必须逐命令 masking**。
- **§189 preflight 新增 `check_openai_api_key`**（unset / 含 `DUMMY|PLACEHOLDER` / 长度 <20 三种 fail 路径）。
  理由：**`DUMMY_P79_PRECHECK` 占位符掩盖了 VWA LLM judge 对真实 OpenAI key 的运行时依赖
  （`helper_functions.py:613+707`），会晚到 N/A 任务评估时才炸**。
- **§192**：leaf queue 加 **`acquire_site_lock`（FD 7，区别于 queue_chain 的 FD 9）**，
  父层用 **`P79_CHAIN_LOCK_HELD` env 握手**让子层跳过重复获取；preflight 加 `--sites csv` 过滤；
  ntfy curl 全加 `-L`；Gate 1 TBD 检测改宽 + **HTML 注释 allowlist（`<!-- TBD-ALLOW: reason -->`）**；
  Gate 6 pgrep **加正则锚定只匹配真 python 调用**。
  分别对应：manual-leaf-during-chain race / 全 5 站点检查阻断 Phase 1a / **重定向丢通知** / 窄 grep 漏 TBD / **pgrep 匹配到编辑器进程**。
- **§208.8 B-905**：`P79_CHAIN_LOCK_HELD` env-bypass **不能靠删 shortcut 收口**，
  改为 **leaf 校验三条**（(a) env 存在 (b) `kill -0 chain_pid` (c) `readlink /proc/<pid>/fd/9 == expected lock`），
  任一失败 FATAL。理由：**原始"纯 Option C 删 shortcut"在 Linux flock per-OFD 语义下 infeasible**。

### 9.2 站点 reset（§195，一条漂亮的折衷）

**Q2=D' 折衷**：**OSClass 全表 DROP+restore（gemini Option A）延后到 A1.17b**；
本轮只把 **reset sentinel 从 3 张表扩到 5 张（+alerts +latest_searches）并加 PHP cache 与 session 双清**。

**全 DROP 的 5 条真实风险路径**（不调和，原样保留）：
1. **install wizard trap**（若 `seed.sql` 不是 post-install 快照则灾难性）
2. **app cache 滞后**
3. **PHP session 残留**
4. **跨镜像升级 schema drift**
5. **FK 约束顺序**

判断：**折衷可 defuse ~80% 攻击面而只承担 ~10% 风险**；**双清本身是 paper-grade 卫生基线，与 partial-vs-full 选择无关**。

**§195 其余关键裁定**：reset 超时按 site 区分（**reddit 240s**）；MySQL 密码走 **`MYSQL_PWD` env 而非 argv**（5 处 callsite）；
indexer pipeline 改 awk 避开 pipefail 脆弱；`setup_vwa` 缺 substrate 时 **exit 1 而非 warning+'Setup complete'**；
TZ 统一成 `P79_VWA_TZ`；`ALLOW_PARTIAL` 在 PG=1 下 hard-block；
dirty cell 在 PG=1+RESET_BEFORE=1+pgrep 命中时 FATAL；
**sentinel parser 解析 site 为空时 FATAL 不回落 `expected_n=0`**；资产发布走 `.tmp.$$` 原子发布 + 对 manifest 校验 sha256。
**共同主题：把静默降级改成 fail-loud。**

**§227 B-1576（live-DB 实证后的最终边界）**：cls sentinel 边界改为
**`WHERE b_active=1 AND (pk_i_id > 84154 OR dt_pub_date > '2024-01-01')`**。
理由：**实验 post 的 `pk_i_id` 自增必超过 84154**；`dt_pub_date` 的 OR 分支**防 AUTO_INCREMENT reset 边缘情形**。
三条 OOB 攻击（B-1571 seed baseline / B-1572 blake.sullivan / B-1573 guest posting）**仍全部化解，
因为 `pk_i_id` 自增与 `fk_i_user_id` 取值无关**。

**§227 B-1577**：清空 reddit 的 positive selector（`_positive_selectors['reddit'] = ''`）
→ 走既有 `if _positive_sel:` 分支使 `_positive_ok` 默认 True，**以 URL-change negative check 作为 reddit 唯一 positive marker**；
cls / shopping / shopping_admin 不动。理由：**postmill 无显式 Logout 链接**；修后两站 auth 实测 PASS。

### 9.3 auth / 进程 / 并发

- **§193 auth_refresh 多项加固**：subprocess creds **走 env 不走 argv**；`time.sleep(2)` 换
  `wait_for_load_state` + `wait_for_function`；**env 白名单（关掉 LLM API key 泄漏，闭合 B-214）**；
  登录判定改 **per-site 正向 selector 与负向检查 AND**；`auth_dir` 先 mkdir；
  `AUTH_REFRESH_TIMEOUT` `int()` 的 ValueError 捕获；**8 条 return 路径加 `outcome=<tag>` 日志**
  （*"`outcome=` 标签比 enum 重构成本低且 blast radius 小"*）。
- **§226 B-1575**：`P79_PAPER_GRADE=1` 下 watchdog 的 `refresh_site_auth` 走 `auth_required_gate`
  并让 **`AuthRefreshFailure` 向上传播**（原来是 soft-fail）。
  根因（codex OOB sibling-propagation）：**`experiment_watchdog.py:166-197` 的 soft-fail 与
  `auth_refresh.py:394-397` docstring 声明的 paper-grade hard-fail callsite 自相矛盾** →
  mid-run session-loss 退化成静默 warn，**NOT-LOGGED-IN episode 污染 `condition_summary_v2.json`**。
- **§226 B-1574 修复位置纠正**：必须是 **`scripts/vwa_env_remote.sh`**（queue hot path
  `_lib_paper_grade_gates.sh:36-38` 只 source 它），**不是 Mode A 推荐的 `~/.api_keys.env`**（不在 hot path，修了也无效）。
- **§205.2 Q2**：只上 **SIGTERM handler（Option A）**，**不做 runtime singleton lease（Option B）**。
  user 看过 30min vs 4-6h trade-off 后原话「B 我觉得不用强上」；**codex 认为 lease 才是架构上正确的收口，
  但按 fire timing 权衡 defer**。
- **§205.7 技术裁定（跨领域可复用）**：**POSIX advisory flock 不能防 `os.unlink`**；
  需要「删另一个进程可能正在写的路径」时**正确原语 = deletion-intent rename（rename-then-async-reap）**。
  依据：**5 个 P0/P1 race vector 里 3 个撞这个不对称**。
- **§205.7 第二条**：**fire-and-forget `Popen` 必须配**
  (a) launch-time `.lock` marker + (b) success-path `.done` marker + (c) 下个 cron tick 的 stale-lock 检测。
  理由：***"裸 Popen 零失败面"***。
- **§206.3 P0-7 修法**：共享 **`p79.experiment.cleanup.clear_task_files()` API**（Option β / user 口中的 Option K），
  不用 subprocess wrap stop-gap；Option K event **单一 emit 点 `_emit_option_k_event(...)`**。
  user 明说「都应该用 Option K」；**永久关掉 fix-one-forget-other drift**（前车之鉴 A1.4 B-451 `_shared_vl_utils` 4-consumer drift）。
- **§206.3 P0-5 重构为 single-operator 场景**：只保 **cron-vs-manual race + cron 自重叠**
  （`sync_a100` >15min → 下个 cron tick 并发）两个真 P0；修复 scope 从 **1.5h cross-host marker protocol
  降到 45min single-host flock**。
  user：***"A100 也是我，dgx 上是唯一操作员，唯一 sync 就是我自己"*** → **multi-operator state-divergence 场景不存在**。
- **§205.2 Q4**：**P1-11 GLMM `operator_cross_contamination` 协变量 DROPPED**（user「我没有人为 race-induced contamination」）。
- **§204.3 B-856**：3 个 callsite 上 **atomic write（temp+`os.replace`）+ fail-loud load（JSONDecodeError→RuntimeError）
  + mandatory `fcntl.flock(LOCK_EX)`**。
  理由：**cron crash 中途写 → 半写 state JSON → 静默 `{}` 丢失 contaminated-episode 追踪**；
  **两个 operator 并发 append 同 (cid,tid) → 重复 JSONL 行 → count-based aggregator 双计 →
  paper §3 phantom 叙事量级可能翻倍**。
- **§204.5 政策**：**non-atomic-write sibling-propagation** —— 任一 site 加 atomic-write 时
  **必须 grep `write_text|open..w|open..a` 扫 sibling**。
- **§204.2 B-855**：抽出 `scripts/maintenance/glm/glm_client.py`（~200 LOC stdlib-only）承载 5 个 GLM helper；
  **保留 `_underscored` back-compat alias**。
  模式定性：*"**extract→re-export→更新主 caller→alias 保留 ≥1 release cycle** 是 operator-script 场景的
  零停机重构模式（**atomic refactor 在 ad-hoc 脚本环境是敌意的**）"*。

### 9.4 GLM sidecar / digest 工具的健壮性

- **§202**：phantom → canonical 映射按 **surface-similarity（图像+文本格式）而非信息深度**，
  依据 memory `project_phantom_space_axes_format_not_information`（**`[SOM_MARKS]` 是 flattened AXTree，1.00× chars**）。
  **只有 DIGEST 决策分支用 `canonical_mode`，原 `obs_mode` 保留给下游 phantom-vs-canonical 分析。**
- **§202/§203 site 推断统一为锚定正则策略**：`case['site']` 显式有效 →
  `case['run_dir']` 的 **`_<site>_<8位数字>` 锚定正则**（**longest-first: shopping_admin → classifieds → reddit → shopping**
  防子串碰撞）→ 回落空串。**单一策略多 callsite，未来任何 site 处理代码都要沿用。**
- **§202**：`_call_glm_chat` 的 JSON 抽取从 `rfind('{')` / `rfind('}')` 改成 **`_extract_balanced_json`**
  （从首个 `{` 前向走，**跟踪括号深度与字符串状态含 `\"` 转义**，返回最外层平衡对象）。
  根因：嵌套输出时 **rfind 会取到最后一个内层 `{` 与最外层 `}`，切出畸形片段** → json.loads 失败 →
  **重试烧 token → GLM_FAILED 噪音**。（10 个边界用例含 markdown fence 真实样例全过。）
- **§203**：**quarantine 必须是管线强制而非仅 prose** —— `auto_pull_myriad_cell.sh` 在
  `VALIDATE_VERDICT` 为 FAIL 时**硬早退**，不再继续跑 Phase 4 的 `make analysis FAST=1`。
  根因：*"pre-fix 已有 '❌ FAIL (quarantine)' verdict 串 + cell frontmatter `status: quarantined` +
  Phase 5 ntfy 提到 quarantine，**但 Phase 4 照跑** → 被污染数据经 aggregator 链进 paper §5 图；
  **prose 声明不等于管线执行，凡标记 quarantine 处必须有对应决策点的实际守卫**"*。
- **§203 文档漂移修法统一**：**strikethrough + 'RETIRED `<date>`' + 退役 commit 引用**，
  **而不是删掉退役特性的文档**。理由：**保留"试过-因 Y 退役"的语义给后来维护者**。

### 9.5 其余运行时硬化

- **§196 Q1=A**：retry 分类改代码 —— `_classify_episode` 先用 `_NOISE_ERROR_SUBSTRINGS` **5 类子串匹配**
  再落到 `code_bug`；**noise 类错误的重试次数从 N=2 提到 N=3**，Phase 1a 全新 cell 一律用新策略。
  （codex F6 指出 disclosure 与代码不符；**因为是全新 fire 所以策略变更没有回溯问题**。）
- **§196 Q3=B**：pgrep 自匹配修法走 **`args.runner_pid` + `os.kill(pid, 0)` 优先**，
  缺 `runner_pid` 时**假定无 runner 并允许清理**（per-item 的 B-222 mtime 与 `.in_progress` 守卫提供更细粒度保护）。
- **§196**：`--reset-state` 加 **`--recover-and-quarantine` 三级抢救** + 发出 `state_reset_discarded` 事件
  并置 **`covariate_trail_complete=False`**；SIGUSR1 注册的 `except:pass` 改成**窄异常类型 + fail-loud `SystemExit(2)`**。
  理由：**原 reset-state 是失忆式重置**；**SIGUSR1 默认动作是 Term（`man 7 signal` 验证），注册失败被吞掉会导致静默被杀**。
- **§199**：**B-819 Pillow 版本守卫从 assert 改 raise RuntimeError**（**防 `python -O` 剥离**）+
  **上界超限只 warn-loud 不 hard-fail**（dev 环境 Pillow 12.1.1 已违反 `<12.0` pin，硬失败会立刻阻断 → **务实 warn**）。
- **§199 A1.2 cold-start 契约收紧**：bool-as-int 4 处表面；**`scroll_direction` 枚举收成 `{up,down,left,right}`**；
  multiple fenced block 绕过关闭；wrapper 返回 None metadata 关闭；
  抽 **`p79/backends/_shared_stage_prefix.py`** 消除 3 个 wrapper 里字节相同的 `stage_prefix` 复制
  （**第 2 次 `_shared_*` 抽取**，前一次是 A1.4 的 `_shared_vl_utils`）并加 invariant lock test。
- **§193 Q3=B 部分修**：CUDA workaround **修 dtype kwarg 尊重** + 加 `_p79_nvrtc_prod_fallback_count` 计数；
  **显式 SKIP autograd raise**（随 mechanism 暂搁）。
  定性：***"paper-grade 实用主义：修可验证部分，有理由地跳过 mechanism 相关部分"***。
- **§194**：`_validate_against_summary` 恢复 **tri-state `Optional[bool]` 语义**（summary 缺失时返回 **None 而非硬 False**）
  + 新增 `summary_missing` telemetry，`strict_identity` 时缺 summary 直接 raise；
  并加 **import-time invariant assert** 保证 `_STEP_FIELD_TYPES` 的 key 集合等于 `REQUIRED_STEP_FIELDS_V2`。
  理由：***"硬 False 会让 integrity log 记成'已检查且没问题'"***。
- **§175b 175.3/175.4**：**episode_dir wipe 与否用 `.in_progress` marker 区分三种 restart 场景** ——
  **S1 watchdog-clean**（marker 缺失 → wipe OK）/ **S2 runner-crash 重启**（marker 存在 → rename 成 `.stale_<ts>` 保留取证）/
  **S3 resume gate**（整条跳过）；watchdog orphan-cleanup 同步跳过 `.stale_*`。
  过程注记：*"读 §168 6-layer auto-clean + watchdog L1404-1426 + B-222 marker 链之后才浮现该区分；
  **不读层间交互史的话 fix 会写成 'always preserve' 从而破坏 watchdog-clean retry 路径**"*。
- **§232 B-1660**：preflight 的 `check_nltk_resources()` 改成**真的调 `sent_tokenize`** 并验证 punkt + punkt_tab 路径
  （原来只 `import evaluator_router`；**B-793 的 init probe 不触发 `sent_tokenize`**）。
- **§232 四项运行时硬化**：**B-1665** `queue_chain.sh` 的 `wait_for_runner_done` 加
  **`MAX_CONDITION_HOURS=4` 默认上限** + 超时 kill+ntfy；
  **B-1666** preflight 加 `check_docker_container_warmth()`（`docker inspect .State.StartedAt`，uptime <30min 告警）；
  **B-1667** watchdog 加 **`--step-stale-mins` 默认 10**（扫 `episodes/*_steps_v2.jsonl` mtime 作次级 idle 信号）；
  **B-1668** `proxy_api_agent` 加 **`_proxy_global_lock()`**（`fcntl.flock` on
  `/tmp/p79_proxy_<api_key_hash>.lock` 包住 retry loop，非阻塞获取 + 60s 重试，**超时降级不阻塞**）。

---

## 十、Fire 实战（Fire-1 ~ Fire-6）—— 只有真跑才暴露的

### 10.1 asyncio loop（两版修法）

- **§228 B-1581 v1**：把 `_lazy_init` 与 `reset` 里的 hard-fail `else: raise` 换成
  **无条件重装**（`asyncio.set_event_loop(asyncio.new_event_loop())`），
  保留 stale 检测为 `logger.warning` 供 post-fire 诊断（**paper-grade audit 可 grep `'B-1581: stale asyncio loop'` 数泄漏频率**）。
  理由：***"guard 的 prose 假设（Phase 1a callers never have an active loop）被多任务 fire 实证证伪"***。
- **§232 B-1661（v2）**：把无条件 `set_event_loop(new_event_loop())` 的 **band-aid** 换成
  **「关掉 stale env+loop + `self._env=None` 强制重建」**；`reset()` 重构成**先做 asyncio 检测再调 `_lazy_init()`**，
  使 `env=None` 能传播到重建。理由：***"v1 只是遮蔽 stale binding 不是清理"***。
- **§229 B-1582（belt-and-suspenders）**：`metrics.py:124` + `experiment_watchdog.py:325`
  **在 playwright noise 匹配之前特判 `'detected an active asyncio loop'` 子串** → 返回 False/None
  让 caller 路由到 `fatal_env`/`code_bug`。
  根因：**watchdog 的 playwright noise 分类把 asyncio fatal 误判成噪声 → 3× retry 放大**。

**§228 由此加防御第 6 层**：**live multi-task fire（≥3 tasks）** ——
***"只有这层能抓跨 episode 边界的状态泄漏；单任务 smoke（第 4 层）打到 max_steps 干净退出，检测不到泄漏"***。
/stress Phase 0 checklist 加：**状态相关代码（env wrapper / watchdog / runner 主循环）必须先跑 ≥3-task smoke 才能放行 fire**。
背景：**B-159 在 2026-05-16 落地后有 2 周只做 prose 验证的窗口，首次真正的多任务 fire 就炸**。

### 10.2 cross-site contention（§232 B-1663）

**Phase 1a fire 默认 SEQUENTIAL cls→red**（`launch_chain cls` → 60s 轮询等 `_cls_pid` 最多 24h → `launch_chain red`）；
**`PHASE1A_PARALLEL=1` 才是 dev 并行**；并写进 CLAUDE.md hard rule #3。
理由：**cls+red 共享 A100 docker bridge + Postgres/Redis underlay + B0 AWS proxy quota**；
实证 **13:28:06 fire 出现 red 99s busy-wait + asyncio race 疑似 cross-site contention**。

**§238.2 R2-P0-1-B\* 补漏**：新 lib helper **`assert_no_other_site_chain_running`**，
应用于两个 orchestrator 的 **cls/red 单站 case**（原来 sentinel-wait **只在 `case all` 里生效**，
**operator 可以 `launch cls && launch red` 重现 Fire-3 的并行 contention**）。
**R2-P0-2-B\***：`assert_no_cross_mode_collision` 扩成两遍 —— Pass 1 同 baseline 跨 mode（既有 B-858 FATAL），
**Pass 2 跨 baseline 同站（新增 FATAL）**（**跨 baseline 同站通过碰撞检查后 `reset_and_auth_gate` 会把 B0 的 session 洗掉**）。

### 10.3 其余 Fire 修复

- **§229 B-1583**：`_lib_paper_grade_gates.sh:491` 的 SIGTERM trap **容器名从 `reddit-box`/`classifieds_box`/`shopping_box`
  改成 `vwa-reddit`/`classifieds`/`classifieds_db`/`vwa-shopping`/`vwa-wikipedia`**（按 live `docker ps`）。
  **原容器名根本不存在 → trap 是空操作。**
- **§229 B-1584**：`experiment_watchdog.py:1826` 把 `_auto_refresh_auth` 包 try/except，
  遇 `AuthRefreshFailure` 时 **`os.kill(args.runner_pid, SIGTERM)` + urgent ntfy + re-raise**
  （**同步停机，取代 60s 异步轮询滞后**）。根因：**watchdog 因 auth 失败死掉却不给 runner 任何信号**。
- **§229 B-1585~B-1587（Pass-2 没继承 Pass-1 的硬化）**：
  `queue_router_learned.sh:191-216` 的 **soft auth gate 换成单行 `reset_and_auth_gate` lib 调用**；
  Pass-2 Gate 8 用 **anchored pgrep + `2>/dev/null || true`**；
  Pass-2 Gate 2 从 cell-coverage **收紧为 36-mode-exact**（6 cells × 6 modes，canonical `phase1_<MODE>_router_0` glob）。
- **§229 B-1590/B-1591/B-1595**：`queue_chain.sh:145` 改用 **`read -r -a cmd_args <<< "$cmd"`**（3-AI overlap，B-672 sibling）；
  `preregistration.md:20+24` 标题改 **Substance-Locked 2026-05-18**（**frontmatter locked 与 title draft 自相矛盾**）；
  `vwa_env_remote.sh.example:50` 加 3 行说明 **PROXY_API_KEY（B0 substrate 优先）/ OPENAI_API_KEY（仅 evaluator judge）/
  HF_TOKEN（gated models）**的角色分工。
- **§229 B-1594 defer-with-rationale**：`auth_refresh.py:181` —— 重审确认 **urlparse 防御对 `/login?errors=1` 场景是稳的**，
  **推测的 redirect-to-homepage 攻击缺实证** → **延到观察到 live regression 再修**。
  理由：**避免为推测性攻击改动 auth 关键路径**。
- **§227 B-1578/B-1579**（**smoke 1 / smoke 2 两次实跑暴露，`bash -x` trace 定位**）：
  A100 的 `vwa_env_remote.sh` 末尾 append `[ -f ~/.api_keys.env ] && source`
  （**queue 只 source `vwa_env_remote.sh`，导致 `snapshot_env.py` FATAL `['google/gemma-3-4b-it'] gated_no_token`**）；
  `_lib_paper_grade_gates.sh:337` 的 **ls 管道加 `|| true`**（**glob 无匹配时在 `set -euo pipefail` 下静默 exit 2**）。
- **§236.2**：gallery 分组键从 `(site, condition_id)` 改 **`(site, baseline, condition)`**，
  `condition_label` 装饰成 `[B0] Phase1 DOM mode`，episode dict 加 baseline 字段。
  根因：**多 baseline 聚合 gallery 里同一 `condition_id`（`phase1_dom_router_0`）在 B0/B1/B2 三个 run_dir 都存在
  → group_map 把三家折叠成一行，UI 丢 baseline 归属**。
- **§236.3 B-1760 按 Option A 延后修**（**不在 Fire-3 cls B0 DOM cell 期间热补**），四条理由：
  (1) **schema-v2 gating 字段全在**，缺的只是 audit/annotation 用的 screenshot；DOM 模式主证据 `observation_dom.txt` 每步都有
  (2) **热补要 pkill + 重启 + 7-gate preflight，有 asyncio race 复发风险**（B-1581/B-159 史）
  (3) SoM/Vision/phantom_som cell 的 `image_som` 路径完好会自带 screenshot
  (4) 若日后 root-cause，**单独重跑 cls B0 DOM（~12h）即可再生**
- **§237.3 P0-1-ABC\*（3-AI overlap）**：`runner/main.py:2483-2491` 的 **about:blank recovery
  不再往 `page_change_reasons` 里追加** → **`page_changed` 布尔只反映 AGENT 动作**。
  根因：***"runner 的干预此前被当成 agent progress 记账，静默污染 paper §3 的 step 语义"***。
- **§237.3 其余 6 项**：**P0-6-AC** `paper_grade=True × baseline_retry_on_no_progress=True → RuntimeError`
  （与 `diagnostic_controls` 同一 sibling 模式）；
  **P0-7-A** paper_grade 下 `VWA_RESET_ENABLE != 1` → `reset_vwa_sites.sh` 返回 1 硬失败；
  **P1-9-AC\*** `environment.py:331-345` 的 B-329 program_html escape 从 `except Exception: pass`
  **收窄到 `(FileNotFoundError, json.JSONDecodeError, KeyError)` + warn**，其余异常上抛；
  **P1-10-B** watchdog 用**结构化 `needs_reevaluation` 字段**而不是字符串匹配 `error(evaluator)`；
  **P1-13-B** `--reset-state` 改为把 state_file 重命名 `.discarded.<ts>` 并就地 emit `state_reset_discarded` 事件；
  **P1-16-AC** wallclock 预算**按 baseline 分档**（pattern 含 `_B0_` → **8h**，否则 **4h**）。
- **§176.3**：**`anti_repeat` / diagnostic action controls 在 paper_grade 下 hard-block**
  （runner `__init__` raise，**镜像 B-340 GLM 模式**）并在 §3.5.1 写 disclosure 段附 grep-audit 指令。
  理由：***"诊断类干预不能出现在 paper-grade 数据里"***。
- **§176.3 新增两类 telemetry**：**`dialog_meta`**（每 step 累积 dialog 事件写进 `step_record.dialog_meta`，B-509，
  **让 dialog 自动接受可审计**）+ **`runtime_sleep_ms`**（wrapper.step() 累加 `wait_for_timeout`，B-510，
  paper §4 latency 表改**双列**：total 与 total − runtime_sleep）。
  理由：**把 phantom-mode latency 收益与 settle-tax 组分拆开，防 reviewer 攻击 latency 构成**。
- **§175.3**：`_FUZZY_MATCH_JS` 改为返回**结构化对象**
  `{matched, match_stage, target_type, selected_text_before/after, clicked_text, error}`，
  **Python 侧 `success = matched` 而非 "evaluate 没抛异常"**；
  并新建 `aggregate_select_option_dispatch.py` 出 match_rate / fuzzy_share / stage breakdown / error taxonomy，
  用 **`pre_b481_unknown_matched` bucket 暴露 legacy 语义占比**。
  理由：***"证据层必须能区分 dispatch 成功与 JS 没崩"***。
- **§180 B-512 —— 一次避免 10× 过度 scoping 的实例**：action schema 不对称的 scope
  **从 4-6h wrapper 重构缩到 30min schema 补字段 + paper §4.X.6 prose 强化**。
  过程：*"user 直觉选 'wrapper code-align 保证 baseline 一致性'，
  但 Claude **实读 `vwa_wrapper.py:395-414` 发现 wrapper 自 §67 reform 起在执行层已经对齐**，
  **真正的 gap 在证据层（`step_record['action']` 记的是 raw emit）**"*。
  → **§180 由此确立 institutional rule**：**对 cross-AI flagged finding，在启动重活之前先做一次
  "let me verify the actual code state" 的 sanity check，即使 user 直觉已选定方案**。
- **§239.4 Fire-6 blocker 6 修**（user 指令只做 fire blocker 子集）：
  **B-1762** Gate 8 缺 `quarantine_registry.py` 从静默 SKIPPED 改 **FATAL**（镜像 Gate 4 fail-closed）；
  **B-1763** `VALID_CLASSIFICATIONS` 加 **`unreproducible_in_isolation`** 枚举并把 task 75 重分类
  （*"Wave 4 M7 的 `transient_drift` 是**事件后 30h+ 的 fresh-chromium 隔离复现**，**不能证明 mid-fire substrate 健康**"*）；
  **B-1764** `preflight_check` 加 **Rule 2 cross-fire recurrent detector**（`detect_recurrent_failures min_fires=2`），
  **classification 不再能单方面解封**；
  **B-1765** `assert_no_other_site_chain_running` 加 **pidfile 检查关掉 30-90s chain-prep race 窗口**；
  **B-1766** `metrics._avg` 对 canonical latency 字段加 **`require_present=True`**（mixed-vintage 时 **fail-loud 而非静默 0**）；
  **B-1767** `aggregate_cross_site` 的 `canonical_ms` **显式 None-check**（原 `(x or 0)` 把 None 当 0）。

### 10.4 §232 re-fire doctrine 8 条（A2.11 post-mortem）

1. fire 前必跑 `preflight --paper-grade --sites classifieds,reddit`
2. **默认 sequential cls→red**
3. watchdog 次级 idle **10 min**
4. 每 condition **4h wallclock 上限**
5. **evaluator infra 失败 = abort 非静默 quarantine**
6. **B0 proxy 文件锁串行**
7. 用 `queue status` 看 **0-42/42，不到 42 不宣布完成**
8. **latency canonical 同时扣 retry 与 busy_wait**

### 10.5 §227 —— paper-grade 审计的 5 层防御（后被 §228 加到 6 层）

(1) cross-AI Mode A+B+C (2) Mode A 自 OOB 自审 (3) **live-DB 实证** (4) **live-system 端到端 smoke**
(5) **`bash -x` / source 级 tracing**。
**缺 (3)(4)(5) 时 bc54e25 那条声称 "Phase 0/4 verification PASS" 的 commit 会带着 4 个未识别缺陷开 fire。**
核心判断：***"Cross-AI 是 risk-reducer 不是 oracle"*** ——
codex F2 OOB 抓出了 Claude 的 JOIN 修法错误，**但 codex 自己推荐的 12-ID canonical set 也是错的，只有 live-DB 部署才暴露**。

**§229 三条过程规则**：
(1) **cross-AI 的真 ROI 是 trust-boundary 多样性**（codex ssh+grep A100 / gemini mysql shell
**都跑了 Claude session 被 auto-classifier 挡住的 live 读**）
(2) **根因与症状是多层的，单个 P0 可能需要多层修复**（codex F1 regex-miss 症状 + B-1581 asyncio 根因**都要落地**）
(3) **多任务 production fire 是多任务 guard 的唯一验证器**

---

## 十一、OSF DOI 流程

### 11.1 两-DOI 拆分（§230）

- **DOI 1 = pre-outcome-creation witness**（**在任何 outcome artifact 产生前提交**）
- **DOI 2 = post-data reproducibility bundle**，触发条件 = **Pass-1 + Pass-2 + analysis frozen + paper §1-§8 finalized**
  （**不是只 Pass-1 complete**）
- **措辞用三级 empirically-conditional 命名**：
  **pre-outcome-creation > pre-outcome-inspection > pre-analysis**（**取代含糊的 "pre-data"**）
- **锚点应是最早的 outcome-bearing artifact（per-episode summary 的 mtime）而不是 `condition_summary_v2.json`**

**§230 强制段落**：`DOI_1_README` 必须声明 **archive placeholder 数字**
（如 `meta_phantom_lift.csv` archive pilot 的 **theta_fe=2.336 at k=3**）**≠ Phase 1a outcome evidence**。
理由：*"否则 OSF reader 在 DOI 1 bundle 里看到 paper §1 hero 数会把 archive pilot 信号与 Phase 1a clean-run 混同 ——
**反而毁掉这层本该防 salami-slicing 的机制**"*。

**§230 操作排序**（help.osf.io/article/330）：
(a) **submission 捕获 cryptographic UTC timestamp = witness anchor**（提交瞬间即不可变）
(b) **admin approval 才分配 DOI 字符串**（默认 48h 自动）。
paper §4 的引用形式随阶段变：过渡期用 **OSF GUID + submission timestamp**；最终用 **DOI + GUID + submission timestamp**。

**§230 一条 doctrine drift 修复**：恢复 **§F.1（2026-05-05 advisor sync，笔记 §110.3）的 pre-data DOI 排序不变式** ——
`osf_lock_manifest §1` 曾把 "Phase 1a Pass-1 数据齐"列为 DOI mint gate，**与之直接矛盾**。
根因：*"**silent doctrine drift** —— B-1570（同日 ~09:30 UTC）退役了 advisor-email-as-lock-gate，
但**没有显式 re-anchor 依赖该 framing 的 §F.1 pre-data DOI 排序**；
**框架级 commit 必须枚举所有相关排序不变式并逐条 retire/preserve/re-anchor**"*。

### 11.2 lock gate 的退役（§221.8 B-1570）

**advisor email 从 lock gate 退役**（改为 optional post-fire collateral）；
**post-B-1570 的有效 lock blocker 只剩 4 项**：
(1) Phase 1a Pass-1 数据齐（36 conditions on A100）(2) A100 B2 HF SHA pin
(3) A100 侧 `probe_b0_production_path.py` 自验 (4) A100 侧 `snapshot_env.py` 自验。
理由：*"doctrine shift 实际在 **2026-05-14 advisor sync + §209「Advisor sync triage: 0 项真需要 advisor pre-sync」**
两个 anchor 上已落地，但 prereg + osf_lock_manifest 的 lock-gate 措辞还停在 pre-2026-05-14，
**造成 reviewer 可见的自相矛盾**"*。

**§221.8 由此确立过程规则**：*"user 说「X 不需要了 我记得」这类 recall 时，
**立刻跨 canonical doc 的 lock substrate 核验（不只查 memory / chronicle）** ——
**doctrine shift 常落在最新 decision log 但 substrate 文本陈旧，制造自相矛盾**"*。
（B-1570 是该 session **第二次** "user 实时抓到 prereg doctrine drift"，第一次是 §221.1 archive-rationale retract。）

### 11.3 witness 与 retraction（§231 / §233 / §235）

- **§231 witness 两阶段方案（Q1A）**：Stage 1 = **立刻 retract + 出 corrected interim scan**；
  Stage 2 = **fire-3 PID-alive 时抓 canonical witness**。
  措辞按 **Q2A-modified** 保留强度但更精确：**"pre-canonical-outcome-creation for Fire-3"**
  （**不接受 gemini 更弱的 "Pre-Pass1-rerun-witness"**）。Q4A：retraction commit 立刻 push，fresh-witness commit 单独 push。
- **§231 retraction 纪律**：**更正 commit + annotated tag（`retraction/osf-doi1-witness-59c60c4`）+ push，
  绝不 force-push**；**GitHub 保留有 bug 的 commit（不可变），audit trail 结构上完整**。
- **§233 Tier 1 pre-launch witness 策略 supersede "PID-alive 后 5 分钟窗口"计划** ——
  在 **launch 之前**抓（**zero run_dir state → 全 canonical-pattern counts = 0 → tier 1**）。
  理由：*"queue script 设计上是 **preflight → gates-pass → queue_chain spawn → runner**，
  **中间没有 5-15 秒空窗留给 witness**，所以 **post-launch witness 必然踩 tier 2**。
  reviewer 看到的序列：witness **21:16:28Z + zero counts** → Fire-3 **21:27:28Z + non-zero counts**，
  **anchor 早于 fire，paper-grade valid**"*。
- **§235 post-mint doctrine**：**deposit 目录（含 `MANIFEST_SHA256.txt`）是 OSF 归档的冻结镜像，mint 后不得修改**；
  **live `pre_run/` 文件可以 backfill post-DOI 数据**。
  另：**"pre-canonical-outcome-creation" 声明的锚点是 witness 文件在 21:16:28Z 记录的 counts=0，
  不是 OSF 提交时刻**（**Fire-3 在 21:16 到 23:10 之间确实在跑并产生 outcome artifact**）。
  理由：**避免 reviewer 把 OSF submission timestamp 误解为 "repo 历史上无 outcome"**。
- **§237.5 Q2=A（走 gemini 路径）**：**OSF DOI 1 无需 amendment** ——
  运营层硬化（quarantine rate, about:blank 处理）**不改 DOI 1 的假设或决策规则** → 属最终 paper 里的**标准 empirical disclosure**；
  **DOI 1（`10.17605/OSF.IO/9QCWU`，anchor commit `5edac3b`）锁不动，不做 supersede/amendment/re-witness**；
  运营硬化经 **`preregistration.md` Appendix E "Post-DOI 1 Operational Hardening (additive disclosure)"** 进入，
  把 **`quarantine_rate` + `runner_intervention_rate` 作为 Appendix 级 sensitivity 列**；**paper §1 hero 数与 estimand 框架不动**。
- **§239.5 OSF 三档发布计划**（user 2026-05-20）：**DOI 1（substance lock）+ OSF 1.5（post-Fire-4 RCA hardening 的
  过渡发布，吸收 30 个 post-mint commit + SBOM bump + Wave 3 M10 disclosure）+ DOI 2（post-data lock）**。
  理由：**DOI 1 后累积了大量 hardening，需要一个中间可引用层**。

### 11.4 quarantine 的预注册（§238.2，防 cherry-picking）

- **R2-P0-4-C\***：prereg **Appendix E.1 预注册 quarantine 失效阈值** ——
  **每 cell >5% 标记并报告；每 cell >20% 从 primary FE pooled gate 剔除；跨 baseline 对称适用**。
  理由（gemini）：***"没有预注册阈值时 operator 可事后剔掉高 quarantine 的 cell 来救 claim（cherry-picking 攻击）"***。
- **R2-P1-6-C\***：**反复 quarantine 的 task 的终态定义** = **3 次人工重跑仍失败 → `success=False`, `score=0.0`,
  `needs_reevaluation_terminal=True`，且 INCLUDED in `scored_task_count`**。
  理由：**防止 operator 自由裁量控制分母（静默 drop 与 score=0 混淆）造成的 denominator-shrink 幸存者偏差**。

---

## 十二、workshop sub-paper 定型（§179）

**Track A** = **GRL walk-up click ON_TARGET fix family + evidence layer**（B-440 + B-448 + §3.5.2），
~**3-4K word** methodology workshop
**Track B** = **VWA LLM judge polarity bug**（B-91 + B-535）独立 ~**1.5K word** evaluation-systems note

**明确不进 workshop 的三项**：
- **cross-benchmark generalizability**（缺数据）
- **action policy 与 safety primitives**（与 paper-1 纠缠太深）
- **SBOM machinery**（**是 process 不是 result**）

**§175.2 相关**：A1.25 Chunk 2 的 **P0-1 dropdown enrichment 走 (a) disclosure-only，降级 P1-0**；
**明确不做 enrichment-off ablation 也不 runtime disable**。
user Q1 reframe：*"**dropdown enrichment 是 standard VWA agent practice**（Aviator-Web 等同类 wrapper-layer hack ——
**模型接口无法直接观察隐藏的 `<ul>`/`<select>` options**）；paper §1 cost 比较是 **DOM+SoM 内部**所以 vision 不对称不破坏该 claim；
**cross-benchmark generalizability 留 workshop paper**"*。

**§171.5 D2**：paper §3 **GRL（Generated Runtime Layer）framing 决策移交 user 平行 session，不由 3-AI audit 定**。
理由：user 之前 advisor sync 已 frame "bug fixes → future cross-benchmark workshop sub-paper"，paper-1 brief mention 即可；
且 **user 区分 net-new capability（`[DROPDOWN OPTIONS]` 注入）vs traditional bug fix，认为 GRL list 可能不完整**；
**user 的 manual task-by-task archive 审查视角与 3-AI audit 不同**。

**§179 Phase 1a launch go/no-go = GO**（按 Q2=Soften 标准，**9 项 pre-launch check 中 6 项在 commit 前 PASS，
7-9 项由 launch script 运行时补**）。依据：A1.25 GRL 4 chunk 全部完成 + **29 fixes** +
paper §3.5/§3.5.1/§3.5.3/§4.X.11 disclosure 面完整 + **SBOM re-lock 在 1c3a615**。

---

## 十三、disclosure-only 裁定清单（防重提）

| § | 项 | 不改代码的理由 |
|---|---|---|
| §165.4 | **P1-9 scroll vocabulary 跨 baseline 不对称走 (C) disclose-only** | Q14 spot-check 确认 **B0 用固定 ±0.8 而 B1 是自由 delta**；差异真实但**改代码会破坏已跑数据** ⚠️ **后被 §212.3 推翻**，见 §十五 |
| §167b.2 Q4 | **P0-1 router numeric threshold 走 disclosure-only**（paper §3.5 + §4.X.5 + aggregator 加 audit gate），**不 recalibrate 阈值**（defer paper-2） | **viewport fix 后阈值实证 dead，但重标定会改变已跑数据的可比性**；`--audit-fire-rate` gate 让 **reviewer 能跑同样命令拿同样数字** |
| §167b.2 Q3 | **P0-5 retry 与 primary action 混在同一 step record** 的 schema bump **推迟到 paper-2**；本轮只在 §3 加一句 action-identity vs outcome mixing | **schema bump 是 breaking change，paper-1 retry 默认 OFF 所以 latent** |
| §167b.2 Q8 | **P1-9 `form_value_changed` 保持 RUNNER_INTERNAL**，只在 §4.X 加 disclosure（*"cls/red form-heavy task SR 在 similarity > 0.95 情形下可能轻微低估"*） | **改判定逻辑会动已跑数据** |
| §176.2 | **P1-1 coordinate 归一化/非归一化双格式** 由 **USER OVERRIDE 判为 intentional design 而非 contract violation**，放弃 wrapper-as-authoritative 重构（**省 ~2-3h**），降为 §3.5.1 disclosure-only | user：*"我记得坐标是现在支持归一化和非归一化，都是通过函数判断的"*；**与 Chunk 2 Q1（dropdown enrichment = standard practice）同一 reframing 模式** |
| §197 Q2 | **P95-of-P95s 走 prose-only caveat 不改算法** | **改聚合口径会动已有数字**；披露 estimand 即可 |
| §169.4 | **B-404 phantom_dom prose 判定为 ALREADY SATISFIED，不改** | `section3_definition.md:29` **已 disclose** `phantom_dom` = deprecated legacy alias + A1.7 B-261 fail-loud raise + run_registry 向后兼容 |
| §171.4 | **B-417 iframe boundary descent 延后**（2h+ scope，**Phase 1a 0pp ROI**） | locator + injection + form snapshot **都只作用于 top document**；但 **Phase 1a 任务集不涉及 iframe** |
| §172.7 | **A1.19 四项延后**：P1-2（DL meta SE 存储 + zero-variance flooring）→ advisor batch；P1-10（`axis_effect_size` 硬编码 archive path + B2 空）→ 随 §5 mechanism 暂搁；P1-11（drop-one vs 3→5 lift 术语漂移）→ codex round 统一；P1-12（P-prompt baseline exclusion）→ 等 advisor + Phase 1a 数据 | 分别是 **advisor 依赖 / paper-2 scope / prose round / 数据依赖** |
| §174.3 | **gemini P0-4 "cost-aware router framing" 攻击 DROPPED** | user Q1：***"paper-1 = phantom routing space phenomenon ONLY；rule-based router 代码是 paper-2 prep substrate，不在 paper §1 claim 里"*** |
| §174.3 | **codex P0-1 "phantom no-image substrate vulnerability" DROPPED** | user Q2：***"paper §3.2 的 'image=None' 指当前页 screenshot，不是 task reference images"***；§3.5 line 107 已正确 disclose；**reference images 所有 mode 都该有** |

**§187 Q3=C —— 一条显式风险接受（唯一一条"既不修也不 disclose"）**：
**P0-1 source-grep theater 既不修也不 disclose，显式风险接受（假定 reviewer 不深读）**；**记入 next_steps 供 advisor 复议**。
user 明确：*"不 disclose，假定 reviewer 不深读"*；修法（lint-mark 或 full-grep-retire-sprint）成本高。
> ⚠️ 这是本片唯一一条**已知敞口且不做任何处理**的裁定。若 advisor 复议或 reviewer 追问，这是第一个暴露面。

---

## 十四、过程与工程纪律（本片沉淀最多）

### 14.1 /stress skill v7.7 → v7.9

| § | 版本 | 变更 | 触发 |
|---|---|---|---|
| §165.8 | **v7.7** | Q&A 从 exhaustive 23-item 升到 **triaged**：**top tier 3-8 个真 user-decision + bottom tier 折叠为 auto-default reserve**，每项带 "Why user-only:" triage 说明 | user 问"有哪些是最需要我 confirm 的" —— A1.9 的 **27-item wall-of-text** 需手工再 triage 成 **6 真-confirm + 22 auto-default** |
| §198 | **v7.8** | 新增 **Phase 4 claim-realness spot-check**（抽 **N = max(2, ceil(0.3 × total))** 条 finding 优先 P0+OOB，核 file:line / 符号 / 引文 / 具体数字 / 跨文件模式）；**强制 runtime capture**（`$OUTPUT.dispatch.meta` 记 START+pid+scope，缺 meta 即 Phase 3 MISSING 强制重试）；**显式 BAN silent fall-back**（wallclock <60s 或 <预声明 scope 下界 1/3 → 强制重试）；**单条 finding 幻觉只打 ⚠️ 标不整轮重试，但 7 抽 5+ 幻觉视为 audit 级污染强制重试** | 两个独立缺口：Phase 1/2/3 **全是 shape verification 无法验证引用是否真实**（**gemini 尤其会编数字**）；Phase 3 无强制 runtime 记录导致 async notify 丢 dispatch 时间戳 → **静默把 milestone 重分类成 spot-check**。**做成 hard rule 是因为软建议会输给 forward-progress mode 的合理化** |
| §209 | **v7.9 candidate** | **power / SE / 统计类攻击必须对着 aggregator 的实际输出**（如 `meta_phantom_lift.csv`）**验证，不能只用理论上界** | **3-AI 全失手** —— Mode A 用理论 1-sample SE 上界；Mode B 审了 aggregator 代码结构但没跑；Mode C 审 prose 但没 cross-check 实证 FE SE。**等价于 A1.4a dom_size 失效模式在 meta-statistical 层的再现** |
| §213.6 | v7.9 proposal | Phase 0 self-audit 加 **"retired/deleted 概念的 sibling-propagation grep"** —— 退役概念 X 时必须**全 repo grep X**（scripts/ + p79/ + tests/ + docs/checkpoints/）并在 retro 里列出所有 stranded 站点 | **B-1051 的 7-site 全漏** |
| §222 | 建议 | Phase 1 应加**强制的 `grep -l <primitive>` cross-chunk 一致性检查**（在 bug-list-present 之前） | 深入审的 **4 个 OOB 全是同类** —— 前面 8 个 /stress chunk 各自过了自己的 gate，但 **cross-chunk sibling gap 对任一 chunk 的审计范围都不可见** |
| §226 | Phase 0 加条 | 针对 **database-state sentinel**，提交修法建议前**必须先 `find external/ -name '*restore*.sql'` 并 grep canonical seed/restore SQL** | Mode A solo 无法自验 canonical seed 状态；**codex 因为 `--sandbox danger-full-access` 自然做了这一步** |
| §228 | 第 6 层 | 状态相关代码必须先跑 **≥3-task smoke** 才能放行 fire | 见 10.1 |
| §231 | Phase 0 加条 | **任何 numeric/count claim 必须先在 codebase 里核验 canonical schema 来源再写 claim** | ***"memory-as-defense 在时间压力下实证失效 —— `feedback_spotcheck_length_claims` memory 就是为防这类 bug 存在的，我 session 开头读过，仍然犯"*** |

**两次 spec override（记录为张力，不当规则）**：
- **§201**：v7.8 的 "<60s 强制重试" 在 Mode C 本轮被 override（gemini ~32s 但 **Phase 4 spot-check 2/2 PASS**，
  输出 5942B / 7 findings / 4 OOB 且引用可核）。提议 v7.9 改成 **wallclock 与 Phase 4 结果组合判定**
  （**<60s AND Phase 4 fail = retry；<60s AND Phase 4 pass = accept with note**）。
- **§206.7**：Mode C **73s < 160s**（pre-fire band 下界 1/3）本应强制 retry，但按 **PASS-WITH-CAVEAT 接受**
  （7 findings 3 OOB / 9KB / Phase 4 7/8 = **productive-fast 而非 silent-fast**）。
  提议 spec vN+1 应区分「**快且低质=retry**」与「**快但质量达标=lineage strength**」。

**§205.1**：**Phase 4 判定某 finding 为 hallucination 前必须用更宽的 grep 复验再降级**
（gemini 指的 `_compute_resume_fingerprint` 被误判为幻觉，复 grep 确认**存在于 `runner/main.py:645` —— 是我 grep pattern 错**）。

**§167a.7**：**plan-meta / process-artifact / checklist-体系类 stress 这次走 Mode A only，但不写入 memory 作为规则**。
理由：*"user 判断这次省了 ~15-25min cross-AI dispatch 且 Claude self-review 已 catch 主要结构性 gap；
**但这只是 1 个 calibration data point，需重复 2-3 次才 memorialize 以防 over-generalize**"*。

**§238.3**：**对 fix-attempt diff 默认跑多轮 /stress**
（~15 min 的成本远小于推一个含 **9 个 push-blocker** 的 fix-attempt-failed wave 的代价）。

**§239.3 user reframing —— P0 标签膨胀**：/stress 统一 bug list 应默认给 **severity-tier × blast-domain 二维分类**，
不只 P0/P1/P2 一维。**三档 blast-domain** = **Fire-6 blocker**（operational safety + data integrity）/
**Paper blocker**（prose/number/path + stats doctrine + SBOM）/ **Phase 1b blocker**（shop budget）；
user 指令「**你就做 fire blocker**」。
理由：**6-lineage 50-finding 规模下容易把 paper prose stale / 统计口径争议 / launch safety bug 打成等价 P0**。

### 14.2 §182 五条 workflow 规则（institutionalized）

1. **退役某机制后必须 grep 断言该机制存在的测试并跑全套件**
2. **每加一个 kwarg 立即 grep 全部 caller 做 sibling propagation**
3. **paper-grade rail 必须经 shared helper 传到 leaf entry point 而非只在 master orchestrator export**
4. **审计快照要在任何 rescuing/validating mutation 之前取**（raw / post-validate / post-control 三层）
5. **writer-no-reader 是新的 dead-code 模式** —— 每个新 step_record 字段都要有对应 aggregator 消费者
   才敢说 "reproducible from JSONL"

依据：**B-550 regression 逃逸（Phase 2 只跑 2 个 test 文件）+ B-549 sibling 漏 + B-548 leaf dormant +
B-552 快照顺序 + B-555 writer-no-reader 五个实例**。

### 14.3 §183 九条 bash/协议教训（A1.13，各对应一个实证 bug）

1. **comment 与 code 的 date 格式串要 grep 核对**
2. **`${var!r}` 不是 python f-string 而是非法 `${var@op}`**
3. **`set -e` 中途翻转会改变其后所有行的 errexit**
4. **substring break-first 排序结构性错误**（要 longest-prefix-first）
5. **`case` 里 `|''` 分支会让空串静默通过**
6. **任何新 gate/env/arg 必须传播到 5 个 caller**
7. **benchmark 特化分支的静默 skip 是 paper-grade-fatal，要改 hard-fail**
8. **TOCTOU pgrep 检查需要 flock**
9. **历史数据是 paper-grade 证据（优先于开发者直觉）**

### 14.4 多 session 并发协议

- **§184**：catalog append 之前**必须用 `grep -oE 'B-[0-9]+' | sed 's/B-//' | sort -n | tail -1` 取真实最大值**，
  **不能信 catalog 各 section head 的 "Next available" 戳**（**实测有 4 个不同的戳分布在 4035/4089/4182/4261 行**）。
- **§206.4**：**B-### 预留 buffer 从 ≥30 提到 ≥40 minimum**
  （**2 天内第 9 次 parallel-session collision**；**parallel session 已是 base rate 而非例外**）。
- **§196/§201**：**B-number 领用改为 TaskCreate 时立即 commit catalog stub**，不再在 audit 窗口内预留后补
  （**6-8 次碰撞实证**：预留在 audit 开始时取的 max **会在 audit 期间被平行 session 超越**）。
- **§187**：**平行 session 安全提交规则** —— 用 **`git commit --only <files>`**（或提交前 `git restore --staged` 非本人文件），
  **不能 `git add` 后裸 `git commit`**。
  实证事故：*"renumber commit **误把平行 A1.6b session 的 5 个 staged 文件打包进 commit 4cc9064（11 files / 1192 ins）**，
  违反'只 commit 自己的'；经 user 授权 `git reset --soft HEAD~1` 回退后用 `--only` 重提"*。
- **§213.5**：并发 session 冲突用 **hunk-split**（`git diff | python hunk-split` 抽自己的 hunk → reset file →
  apply mine → commit → `git apply` 恢复并发方的 hunk），**不用 stash**
  （检测到 **5+ 并发 Claude session 同改 `p79/experiment/analysis.py`**）。
- **§217**：**prose 里的 `B-###` marker 是 canonical audit trail**；
  并发 session 用 `git add -A` 把我的 prose 编辑吞进他们的 commit 时，**commit subject 归属错乱可接受**
  （**多 session 并发常态；marker 保证可追溯**）。

### 14.5 cross-AI 工具层

- **§207.7**：**codex dispatch 不要同时用 `-o` 和 `apply_patch` 写同一路径**
  （codex 用 apply_patch 建文件后，`-o` 在 final assistant message 阶段把内容**覆写成指针消息**；
  真 audit 只在 trace log 内嵌 diff 里，需 awk-extract 还原）。
- **§214**：**codex dispatch 需加 `-c shell_environment_policy.inherit=all`**，否则 **codex 不 honor `-o` 输出路径**
  （A2.3b 实证：codex 自行把 audit 写到别的文件名，**实质输出仍产生**：23864B, 9 findings, 4 OOB）。
- **§175.1**：**codex 限额中断时不放弃 Mode B**，改用 **detached `setsid nohup` worker 睡到配额重置时间自动 fire
  + 写 `.done` marker + ntfy**，harness 侧用 **Tier 1 file-marker monitor** 等。
  user 指令：*"不能因为没有额度就放弃 codex，设置自动触发不要损失精度"*；
  实测**等 ~75min 后 2.5min 跑完，保住 6 个 paper-grade findings**。
- **§215**：**后台 dispatch 的 wallclock 必须在 backgrounding 之前把 START 写进 `.meta`**（或用 `time` builtin 重定向），
  **不能靠 post-wait 的 `$((END - START))`**。
  实证：`WALL=$((END - START))` 在 `run_in_background` 的 bash 里报 **1779065173s（= END 时间戳本身）**，
  START 变量在 harness wrapper 上下文里丢了；**Mode B+C 都因此 exit 2 但审计输出其实正常**。

### 14.6 审计签名与教训

- **§212.2**：**mental-model ≠ artifact-grep** —— A2.2 的 3-AI cycle 没抓到 §1 L27 stale claim，
  因为 **Mode B+C 把 §1 L27 当 truth 没去 grep VWA upstream**（memory `feedback_spotcheck_length_claims` 规则的**再次应验**）。
- **§224 审计签名**：A2.10 的 **4 个 OOB 全是 internal contradiction 型**
  （paper N vs scored helper / B-660 test vs B-173 code / vectorizer 命名 gate-vs-loader / Pareto strict-< doctrine drift）。
  **签名**：*"当一轮审计产出 N 个内部矛盾而只有 M 个外部不符时，说明 **doc-process drift 快于数据演化**。
  修法 = **认定哪一侧是 canonical claim，对齐另一侧**"*。
- **§240**：**3 个 aspirational test 反过来补 source**（**git pickaxe 证实 guard 从未进过 `analysis.py` 历史，
  自测试加入起一直红**）：**B-650** heatmap 用 `np.ma.masked_invalid` + `cmap.set_bad('#cccccc')` 让 N/A 渲染成灰
  （**原 `fillna(-1)` 把 N/A 涂成与 0=fail 同色**）；**B-659** synthesized/partial cell 写 stub `session_summary.json`；
  **B-661** `_plot_phase2` 对 `phase2_fixed_best/routed` 匹配 >1 行时 **fail-loud raise**（原 `.iloc[0]` 静默选错行）。
- **§186 —— user "全推荐" 覆盖 prose-side**：A1.6b **全部走 code-side 修**（user 明确 *"Code-side 修，全推荐"*）：
  在 `_compute_statistical_tests` 里**真实现 TOST + SR-Wilcoxon + `wilcoxon_skipped.csv` 输出**；
  Holm family 改 `(test, metric, cell_key)` cell-scope；热图按 site 分文件；
  Phase 1 SR bar 改按 `(mode, baseline)` 分组**不跨 baseline 求均值**；
  Pareto 加 `(best_min, last_max)` tie 分支保留并列点；bootstrap 同时输出 **paired 与 per-condition single-arm**。
  理由：***"prose-side 只是掩盖差异，代码补齐才是 paper-grade"***。
- **§186 三条 rng/绘图纪律**：`_analyze_condition` 对 **partial cell 跳过绘图**并输出带 `partial: True` 的 stub；
  `_plot_phase2` 的 `fixed_best` **若非唯一则 raise ValueError**；
  **每个 condition 用 `SeedSequence([42, hash(cid)])` 独立 rng 而非全局单点 `default_rng(42)`**
  （**共享 rng 让 CI 依赖 glob 顺序**）。
- **§187 三类测试补位**：`tests/test_runner_smoke_realprompt.py`（env-gated `RUN_REALMODEL_SMOKE=1` + `@pytest.mark.gpu`，
  **1 task × 1 step 真 Qwen3VL forward**，断言 schema/action/tokens/latency 4 项 prod 契约）/
  `tests/test_learned_router_runtime.py`（17 tests 覆盖 learned_router 全部 4 个函数）/
  `tests/test_cli_smoke.py`（subprocess 验 CLI argparse 契约）。
  理由：**prod path + paper §6 router claim + CLI 三处 0 test**；
  ***"mock-mock smoke（env 与 backend 都 mock）不构成生产路径证据"***。
- **§166.4**：pytest 配置改 **`addopts = '--strict-markers -ra -rs'`** + 注册 `local_data`/`external` markers；
  **Makefile 去掉 `-x` fail-fast 改 `--tb=short`**（**`-x` 会在第一个 fail 就停，隐藏后续 failure**；
  **strict-markers 防 typo marker 静默失效**）。
- **§166.4**：**6-cell topology canonical 修正为 B0+B1+B2 × cls+red**（**原 4 处 fixture 写的是 B0+B1 × cls+red+shop**）。
  理由：**Phase 1a scope 是 cls+red 三模型，不含 shop；fixture 拓扑与实验设计不符会让 gate 测试测错东西**。
- **§175b 175.3**：纠正 *"paper §1 hero risk / 需 codex round 重写 paper"* 的 framing ——
  **所有 A1.x fix 是 pre-fire correctness 而非 post-publication revision**；
  并把 "Current fire state" 段写进 **`AGENTS.md` + `GEMINI.md`** 供 cross-AI 继承。
  user Q1：*"SR 还没 published，Phase 1a 还没在 A100 fire"* —— **之前的 unified-list framing 建立在错误假设上**。

### 14.7 审计范围治理（§167a）

- **phase1_plan §A1 从 21 项扩到 24 项**（新 cluster "A1-横向"：A1.22 cross-baseline parity / A1.23 concurrency+race /
  A1.24 clear_tasks）；**§A2 从 8 拆到 13**（A2.3→a/b/c power+sample / DL-vs-REML+HK / multi-test+TOST δ；
  A2.4→a/b；A2.6→a/b；新增 A2.9 reporting+ethics）。
  理由：**cross-cutting surface 跨多 file class 不适用 `feedback_split_large_scope`**；
  A2.3/2.4/2.6 每项包 3-5 个 sub-claim **违反 split rule**；**顶刊要求 compute cost + broader impact + data-rights statement**；
  **拆细后 advisor 可逐项 confirm 减少打包 reject 风险**。
- **§167a.5**：§E milestone wallclock 从单值改为 **happy/realistic band**：
  **1.5-2.5w happy / 2.5-4w realistic**（**1.3-1.5× overhead** 含 watchdog 重跑 + GLM fallback + sequential 3-baseline lag）。
  依据：**paper-grade fire 历史中位 = happy × 1.3-1.5×，单点估计无 contingency**。
- **§167a.6**：**audit ledger 14 个已勾选项全部 reset 成 `[ ]`**，总数 **37 = A1 24 + A2 13**。
  user 指令"把对勾都取消掉"；rationale = **cross-cutting surface（A1.22-24）引入后旧勾的 scope 责任已变更**，
  需在新 cross-cutting 责任下重走一遍；**substantive audit work 在 git history 未丢，只 reset forward-progress signal**。
- **§168.2**：**A1.5 ladder label 与实际 scope 不符澄清** —— **A1.5 实际审的是 utils+cli 不是 `runner/`；
  `runner/` 从未被 audit**，新开 **A1.5b slot**。
- **§166.3 T1-2=B**：**P0-1~P0-5 五个测试层 P0 全修完才能 launch**（paper-grade hard gate）。
- **§173.8 Q5=B wait-fix-all**：Phase 1a 36-condition queue **等 A1.25 GRL 全 4 chunk 完成再启**；
  预计再 **45-75 个 bug**，paper §1 critical path **延迟 3-5 天**。
  user 选 **paper-grade 数据纯净（无 post-hoc adjustment）优先于 wallclock**。
- **§167b.2 Q6 fire timing**：**先修完 19 个 P0+P1 再 fire Pass-1 baseline**（reddit + shopping × 3 baseline × 6 mode router_off）；
  **cls baseline 已 fire 不动**；**cls router_on（受 P0-2 影响）标 paper-2 backlog 不重跑**。
- **§168.2 T8 = (a) Pre-fire 闭环**：接受 **1-1.5 天 fire 延迟**，换 `trajectory_events.jsonl` → aggregator covariate trail
  **在 fire 前完整通端到端**。理由：***"paper §4 reviewer-bulletproof covariate trail 不能 fire-after-the-fact 补"***。
- **§168.3 B-385 reframe**：P0-4 condition_finalize race **从 runner-side mutex 改写成 aggregator 侧 post-hoc race detection**
  （`condition_summary ∩ task_auto_cleared` events），**不新增 event_type**
  （把 1.5h runner 工作**折叠进 aggregator covariate emission**）。

### 14.8 §220 —— user 从默认 scope 撤回三项

**不写 paper §10 cross-AI methodology 章节（~600-1000 words）/ 不把 process replica 纳入 git 追踪 /
不做 per-AI hallucination rate 表**。user 直接指令「**不要描述 /stress workflow**」。
**但保留**：ethics 文档里 **~90-120 words 的 LLM Use Disclosure 短声明**
（**NeurIPS 2025 policy 要求的是 acknowledgement 不是 workflow description；paper 也没把 AI-assisted-methodology 当科学贡献**）。

**§220.3 submission-day gate**（非 fire gate）：**B-1501 release license matrix**
（P79 MIT / VWA MIT / WA MIT / Qwen Apache 2.0 / **Gemma3-VL Gemma Terms of Use gated** / OpenAI judge ToS）；
**B-1510 carbon 报 PUE 区间 [1.0 下界, 1.5 上界]（Strubell-compliant）**；
**B-1512 Makefile 新增 `pre-release-check` 5 步 target 且实跑 5/5 PASS**。

**§225 B2 显示层约定**：颜色 **`#17becf`（matplotlib tab cyan）而非 `#2ca02c`（tab green）**，
marker **`'D'`（diamond）**，label canonical **`'Gemma3-VL'`**。
理由：**避免与 `MODE_COLORS['Vision']='#54a24b'` 在 baseline 色与 mode 色相邻渲染时撞色**。

**§209 B-950**：CLAUDE.md 里的 condition 数**统一为 42**（local-only 修改，`.claude/` gitignored）；
**对协作者的 canonical authority = preregistration + phase1_plan + OSF manifest（三者早已是 42）**。
> 这条正是当前 CLAUDE.md "Terminology hard rule" 段落的来源。

**§216.2**：**B-1263 Phase 1b 条件数从 +18/+12 改为 +21；B-1264 全库 36→42 sweep**（codex 抓的 cross-document count drift）。

---

## 十五、⚠️ 本片矛盾与待核清单（合并阶段用）

| # | 事项 | 两侧 / 需核 |
|---|---|---|
| 1 | **scroll vocabulary（真矛盾）** | §165.4（05-16）判 **disclose-only 不改代码**，理由"改代码会破坏已跑数据" vs **§212.3（05-18）Chunk α 实际改了代码**（B1/B2 三个 prompt template 从 `delta:[dx,dy]` 改 `scroll_direction`，删 B0 转换）。⚠️ 中间发生了 **archive 全面弃用**（§184/§185/§210），"破坏已跑数据"的顾虑随之消失 —— **但台账未显式记录这次反转的裁定**，两条并列 |
| 2 | **SE floor 数值** | §172.4 **const 1.0pp**（archive median 0.98pp 校准）vs §211.4 B-1003 **0.68pp**（Agresti-Coull anchor）。**后者为准；前者的三条拒绝理由（不用 N-aware / 不 exclude degenerate）仍有效** |
| 3 | **router CV 结构** | A2 §154.2（05-16）**LOCO 为 paper §6 主数字** vs §216.1（05-18）**(E'') task-held-out 5-fold 为 primary，LOCO 降 Appendix**。**§216.1 明确 supersede** |
| 4 | **`tool_choice`** | §210 Q1=A **`auto`** 落地（pilot emit ≥95%）+ 台账 flag `named by RETRACTED §214` vs CLAUDE.md 记 **Fire-6 RCA 改 `required`（emit 从 0% → 100%，B-1794）**。⚠️ 反转 § 在 A4；§214 RETRACTED 全文在 B 批 |
| 5 | **sentinel 完成率阈值** | A2 §162.2 **90%** vs §183 **100% exact**（历史实证 23/24 次 EXACT）。**§183 为准** |
| 6 | **§187 Q3=C source-grep theater** | **既不修也不 disclose**，显式风险接受（"假定 reviewer 不深读"），记 next_steps 供 advisor 复议。⚠️ **本片唯一已知敞口不做处理者** |
| 7 | **B-1581 两版修法** | v1（§228 无条件重装）是 band-aid，v2（§232 B-1661 强制重建）才是清理。**引用 asyncio 修法必须用 v2** |
| 8 | **DL/HKSJ 的"退役"分层** | B-1016 只退役输出 prose label；**§215 B-1305 才退役 pipeline 调用 + 内部理由文档**。查"DL 还在不在跑"要看 §215 之后 |
| 9 | **archive placeholder 数字** | §230 记 `meta_phantom_lift.csv` archive pilot **theta_fe=2.336 at k=3**；PROGRESS.md 记 **P-SoM 行 +2.34pp drop-one / 81% power at k=3**。⚠️ 两者疑似同源不同口径，**数字原样并列，不做换算** |
| 10 | **§172.5 gemini claim 被 verify 推翻** | gemini 原 claim（gate 不查 cost）**错**，但 grep 顺带发现**两个真 bug**。→ 记录方式：**攻击结论可错，攻击引发的核查仍有价值** |
| 11 | **§207 三家判断不一致** | B-894 phantom boundary：**Claude+gemini 抓到定义不兼容，codex 反而把 sharp boundary 当强 claim**。**多数决不适用于定义问题** |
| 12 | **§133b.4 / §136.2 遗留** | A2 记的 Method 4.4 三重污染 + `fmt_som_standard` v1-ish 重抽，本片**未再提及**（随 mechanism 暂搁冻结）。paper-2 resume 时须回查 |
| 13 | **消失/改名的文件** | `docs/checkpoints/stress_grl_audit_2026-05-17.md`（§171.5/§179）/ `docs/checkpoints/router/proposals_v6.md`、`_archive_proposals/`（§153.4/§149.3）—— 引用前需确认路径 |

---

*本文件覆盖 A 批 4 片中的第 3 片（229/831 条）。A1（§5–§119）/ A2（§121–§164）已落盘；A4（§241–§397）见同目录。*
