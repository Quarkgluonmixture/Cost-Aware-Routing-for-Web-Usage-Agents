# 裁定层 A4（§241–§397 + §header，206 条 ADJUDICATED，2026-05-20 → 2026-07-28）

Claude 主 session 逐条通读产出，2026-07-28。**聚合非转写**：逐条索引见 `ledger.jsonl`。

这一片时间跨度最长（两个月），也是**治理结构成型**的一片：
Protocol Reset 把"隐藏的 benchmark mutation"显式化成 GRL 贡献层；
AMENDMENT / PROTOCOL_NOTE 两级 witness 体系建立并跑了 8+6 次；
venue 三次转向（EMNLP → AAAI-27 → REALM workshop 双投）；
paper 拆成 A（现象）+ B（路由阴性结果）；
最后两周是**连续的 estimand 审计**——本 session 重建台账的直接起因就在 §397.10。

> ⚠️ **§397.10 是 CORRECTION 节**：作废了 §397.4 与 §397.9 的部分结论并追加 (4)(5)。
> **读 §397.4 / §397.9 必须连它一起读。**（append-only 纪律：不改历史记录，在 §397.10 里作废。）

---

## 一、Protocol Reset（§243–§251）—— 本片最大的结构性事件

### 1.1 framing：GRL 作为显式贡献层（§243，05-20）

canonical Phase 1a **重定义为 "Upstream-Core VWA semantics + P79-GRL runtime/reliability layer"**：
- **benchmark 行为语义**（prompt / action / termination / max_steps）**尽量回 upstream**
- **GRL 作为明确贡献层保留**：evaluator isolation / DOM screenshot timeout recovery /
  Gate 8 recurrent-failure registry / fail-closed quarantine / telemetry+provenance+SBOM /
  VWA evaluator bug fix / wrapper reliability primitives / backend serialization adapters
- **GRL 边界 = make execution reliable, NOT change task policy**
- **不再模糊叫 "vanilla VWA"**

理由（user framing 决策）：***"隐藏的 benchmark mutation 必须显式化成 contribution 层"***。

**§250.3 GRL 边界的一次实际应用**：**observation enrichment（给匿名 textbox 补 label）越出 GRL 边界，不做** ——
**改 representation = 改 task policy + 污染 phantom mode 对比 + 重新偏离 upstream**；
**GRL 该做的是 telemetry（诊断修已 measure）+ disclosure**。
（schema-prompt 对齐属 serialization-adapter 层，符合 §244 #1。）

### 1.2 §244 —— 12-point canonical decision（后续实施以此为准）

| # | 决定 |
|---|---|
| **#1** | **serialization closure**：B0 tool-call(required) / B1 text JSON / B2 text JSON 是 **capability-forced backend split**（B2 无 native tool-call ABI，chat template NO-TOOL-LOGIC）；三者**不需字节级输出一致但必须共享** semantic action schema + validator + max_steps semantics + failure accounting + cost accounting；**paper 要 disclose backend-specific serialization** |
| **#2** | **B-991 后至 tool_choice fix 前的 B0 数据全 non-canonical** |
| **#3** | canonical protocol = Upstream-Core VWA semantics + P79-GRL runtime/reliability layer |
| **#4** | **few-shot 不 restore**，disclose 为 **zero-shot controlled protocol**，**不 claim absolute SR vs VWA literature**；primary estimand = **within-protocol paired comparison** across observation modes / routing |
| **#5** | action set 先 targeted /stress 再 restore upstream-compatible schema |
| **#6** | **WAIT 非 canonical valid agent action**（`valid_agent_action=False`, `consumes_agent_action_budget=False`） |
| **#7** | **max_steps 修回 upstream 语义 = 30 valid agent-action opportunities**；two-budget = `max_agent_actions 30` + safety（`max_model_attempts` / `max_consecutive_parse_errors` / `max_total_parse_errors`） |
| **#8** | parse_error 三 baseline 统一处理 + **三列 cost**（total_billed / canonical_action / protocol_wasted） |
| **#9** | C1/C1b/Gate8 **per-error-class resolution 保持，严禁 task-level `resolved=true`** |
| **#10** | **B-651 Holm family revert 回 `(test, metric)`** |
| **#11** | B1/B2 speed investigation 独立 session **只做 runtime optimization，禁改 prompt/action/schema/accounting semantics** |
| **#12** | **Fire-6 继续 HOLD** 直到 protocol reset + action-set stress + accounting fixes + small smoke 全过 |

> ⚠️ **#10 revert 了 A3 §240 B-651 的 cell-scoped Holm**。两条并列：§240 认为 stratified design 该 cell-scope，
> §244 #10 revert 回 `(test, metric)`。**台账未记录 revert 的具体理由**（12-point 是 user final decision）。

### 1.3 action set 全恢复（§245）

**user 选 Option 1 = full restore + goto 白名单**，4 层 patch：
- hover/press/new_tab/close_tab 走 **wrapper escape-hatch**（`_json_to_id_action_str` → upstream `create_id_based_action`，**零新执行逻辑**）
- **goto 显式 `step()` 分支 + `_goto_allowed_hosts()` 白名单**（开着的 tab host ∪ env VWA host），
  off-site → `none_action` + `info['goto_meta'].allowed=False`
- prompt 加紧凑的 "Other navigation" 块（rarely needed, prefer above）+ "goto ONLY task websites"

**改 system prompt 的正当性论证（关键）**：*"decision #2 已把 pre-reset 数据全判 non-canonical +
canonical Fire-6 未跑 → **现在改正是定义新 canonical 协议的正确时点**；
prompt 跨 baseline 的对称性由 `build_mode_prompt_dispatch_table` 自动保证"*。

**§245/§248 B-1782 off-site goto 的最终 disposition**（一个精细的三态区分）：
**`valid=False`（非合法 VWA 动作）+ `consumes=True`（仍花 agent 一轮 turn）+
`error_category='fail_policy_blocked_offsite'` + `action_success=False`** ——
**这是 valid≠consumes 的唯一 divergence**。
理由：**防止 agent 用 off-site goto 逃 budget**；与 parse-error（valid=False 且 consumes=False）区分。

**§245 五个 cross-AI fix**（*"Mode A 只验了 4 层的名字一致，executable contract 有真实裂缝 —— **全是 B+C 抓的**"*）：
P1-1-BC coord-hover 静默 no-op → wrapper 加显式 hover 分支；
**P1-2-B\* goto 白名单从 hostname 改 netloc**（**A100 self-host 全是 localhost:port，只比 hostname 会坍缩成
放行任意 localhost 端口**；相对 URL = 空 netloc + 空 scheme = 站内放行，`javascript:`/`data:` 仍挡）；
P1-3-B\* multiple-action ambiguity guard 的两处 signature tuple 补 key/key_comb/url；
P2-1-BC 加 step-level fake-env 合约测试；P2-2-B restored actions 进 `action_executed` telemetry。

### 1.4 two-budget + 三列 cost（§248）

**B-1784 two-budget**（B0/B1/B2 同规则，runner backend-agnostic stamp）：
**WAIT sink**（`action_utils` 把所有 parse/structural 失败 rescue 成 `{'action_type':'wait'}`）
= **internal recovery event**，`valid_agent_action=False` + `consumes_agent_action_budget=False`；
主循环从 `while step_idx < max_steps` 改成 multi-budget
（`agent_action_count < max_agent_actions=30` AND `step_idx < max_model_attempts` AND
`consecutive_parse_errors < 3` AND `total_parse_errors < 5`）；
`config.py` 里 `max_agent_actions` **继承 `max_steps`（零 yaml churn）**，
`max_model_attempts` **派生 = budget + parse_cap + 10**（scale-safe，**永不在 primary budget 耗尽前 cut**）。
定性：**deliberate restoration，不是 /stress 发现的 bug**。

**B-1785 三列 cost**：`total_billed`（所有 billed LLM call）/ `canonical_action`（只算 `valid_agent_action` 的 step）/
`protocol_wasted`（残差：parse-error + policy-blocked + recovery）。
**加和闭合不变式 `canonical + wasted ≡ billed` by construction 且可测。**
classification 与三列计算抽成 `metrics.py` 纯函数（`classify_step_accounting` / `compute_three_column_cost`）；
`_avg_or_none` 保 `Optional[float]` 语义（**legacy vintage cannot compute → None 一路传到 cross-site canonical-cost None-guard**）。

### 1.5 §249 —— §1 主 cost 估计量的一次主动换防（05-21）

**paper §1 主 cost 估计量从 canonical-only 改成 `total_billed_cost` primary**（**诚实的 "what you pay"**）；
canonical / protocol_wasted **降为 §4 的 efficiency 分解**；
同时**主动暴露** `parse_error_rate` / `model_call_attempt` / parse-cap sensitivity 来 defuse。
**§244 的代码不变**（two-budget 对齐 upstream 30-valid-action 是对的）；single-budget 的问题回 advisor，不阻塞 Fire-6。

**gemini 的两条独立挑战**：
1. **canonical-cost 作 §1 hero = cherry-picker's estimand**（把 B0 的 parse-error 成本挪进 protocol_wasted 美化 B0）
2. **two-budget 给 buggy 模型 free-look**（parse-error 不消耗 budget）

**裁定性质（一字不丢）**：*"user Q1=A/Q2=A/Q3=A —— **不是承认 §244 错，而是换一个更难攻的主估计量**"*。
→ **§250.1 随 Amendment 01 一起锁**：primary = `total_billed_cost`；canonical/wasted → §4；
**H10 Pareto 的 Cost 轴也用 total_billed，within-cell**。

### 1.6 §246 —— 加速方案的全面排除（user 终局决策）

**canonical run 禁用 vLLM 与 torch.compile（已证会改 10-28% 的 step action）。**
§244 #11「允许 vLLM/torch.compile」与「禁改 action semantics」的张力
**按解释 (b) reference-fidelity（数值结果层）裁定** → **排除所有改 action 输出的引擎**，
**唯一 action-identical 的是 transformers eager 本身**。

加速只能走 **output-preserving 且必须同时是 action-preserving AND latency-preserving**：
(i) **忍受 eager 慢**；(ii) **多卡独占并行**（每卡一 site-chain，当前无硬件）；
(iii) **env 侧 legitimate 优化**（产出字节一致 obs，**排除降 `sleep_after_execution` 因其改 obs→改 action**）。
**一卡多 eager 实例并发被否** —— **虽 action bit-identical 但 GPU 争抢污染 latency canonical**。
（学长口头约束「用 transformers 不要 vllm」与此立场一致。）

**§246 方法论**：**parity/divergence 类 claim 必须 N≥30 才能下结论**（action-parity 版的 spot-check 原则）。
依据：*"本 session 两次小样本（**n=1 spike 判 benign / n=3 smoke 3/3**）都给了乐观假象，**都被 n=40 推翻**"*。

### 1.7 §251 —— schema≡validator（B-1794 真修复）

**核心原则（跨 baseline 一致性的机制保证）**：**schema 的 `required` 必须 ≡ validator（不严不松）**，
因为三 baseline 共享 `validate_action_detailed`（B1/B2 经 `parse_action_text`，B0 经 tool_call）。
实现：在 `_WEB_ACTION_TOOL` 里用 **per-action 条件 required（`allOf` if/then）**；
`select_option` 等用 **`anyOf(element_id|coordinate)`** 兼容 vision mode。
过程注记：*"我一度给 `type` 多加 text-required **被 user 的一致性框架抓出并移除**（validator 接受 `type(eid, 无 text)`）"*。
user 洞察原文：***"所有 optional 语义必需字段都有 forcing 漏洞，是不是确保 B0 与 B1/B2 一致的关键?"*** = **是**。

**§252 B-1796 第一个反向违例**：`select_option` 的 **schema 比 validator 严**
（只 require `element_id`，而 validator 也接受 coordinate）→ **VISION-mode 的 B0 发不出 B1/B2 能发的 `select_option`**；
修为 `anyOf` 镜像 validator，并**加 bidirectional 反向不变式测试堵单向测试漏洞**。

**§251.5 paper 级含义（重要）**：**所有 pre-fix 的 B0 archive（Fire-3/4/5）被 B-1794 artifact
系统性压低 search/type 的 SR → B0 在历史上是残废 baseline，跨 baseline 对比不公平；Fire-6 是 B0 的首次公平测量。**
机制：**schema-forcing 导致 type/search 动作恒 `invalid_element_id`，与 B1/B2 不对称**。

**§251.4 维持 backend-specific serialization 不统一**：统一成 JSON **会重演 B-991 的 0%-tool-call 失败**；
统一成 upstream 文本串**要重写全部 agent I/O，pre-Fire-6 风险大收益小**。
framing 从 "upstream-core VWA semantics" 改为
**"upstream-aligned task/action/evaluator/termination semantics + P79 structured serialization adapters + P79-GRL reliability layer"**，
**不主张 byte-level 格式等价**。
**已 locked 的文件（prereg / Amendment 01）只能未来 amendment，不 retro-edit（witness/DOI 锁）。**

### 1.8 §247 —— paper-grade 的四道硬闸

| B-# | 内容 | 理由 |
|---|---|---|
| **B-1776** | **paper_grade XOR diagnostic_replay 硬块**（runner `__init__` 顶部 + queue env reject） | 泄漏的 `P79_DIAGNOSTIC_REPLAY=1` 会**静默把 paper-grade fire 变成 non-canonical + sr_excluded + suppress M1 abort = 整轮静默作废**（2-AI overlap） |
| **B-1777** | **paper_grade 下禁止 watchdog delete/retry**；判定抽成 module-level 纯函数 `_can_auto_retry` | **auto-clean 删掉并重跑 error episode = denominator surgery**，且与 mode/site/backend 相关（**B0 proxy error > B1/B2 → 不对称 SR 膨胀**） |
| **B-1778** | paper_grade 下 **B0 的 invalid tool_call 不再静默 fallback 到 text-parse 去执行不同的 action**；新增并**持久化** `action_source` / `tool_call_valid` / `text_fallback_used`（**B-1588 的 emission trio 原本只在 meta 里没落盘**） | **静默 fallback 会换掉 action provenance 并掩盖 tool-call 失败率，造成 cross-baseline 不对称** |
| **B-1780** | C1b 的 **two-layer latency 端到端接线**：runner stamp step 级 `_ms` + episode 级 `_total_ms` + `total_latency_canonical_ms`；metrics 消费（avg + `screenshot_timeout_recovered_episode_rate` sensitivity 列）；**aggregate_cross_site 的 canonical = minus_retry − busy_wait − recovered（第三个扣除项）** | **B-1773 加了 `screenshot_timeout_recovered_count` 并声称让 paper §4 能扣 ~+30s，但只写 telemetry 零消费（write-only）** |

**§241 C3a evidence-gated per-error-class Gate 8 Rule 2**：recurrent task **只有在 EVERY error_class 各自都有 resolution 时才解封**
（episode 需 `diagnostic_replay` + `sr_excluded` + `needs_reevaluation=False` + per-class evidence +
`resolved_by_commit` 落在 `[floor, HEAD]` 区间 via `git merge-base`，preflight 重读 episode）；
**严禁 task-level `resolved=true`**。
理由：*"user 要求 matching error_class 才暴露出 **task 75 的第二个 error class**；
naive 的 task 级 `resolved=true` 会让 **Fire-6 撞回 Fire-4 的 screenshot timeout**"*。

**§247 B-1779**：Gate 8 resolution label **按 profile 分化 `classified_via`** ——
eval_goto 保 `matched_temporal_context_diagnostic_replay`；
**screenshot 改 `architectural_containment_c1b`**（**load-dependent timeout 物理上无法用 small-replay 复现，是 containment 不是 temporal repro**）。
理由：**旧的扁平常量是 tautology**。

**§247 过程规则**：**每个 fix 必须在同一 commit 里加 `master_bug_catalog` 条目并 cross-ref canonical docs 才算 finalize**。
触发：*"user 连点 `workshop_subpaper_plan.md`（GRL = workshop contribution）+ `master_bug_catalog.md`
揭示 **scripts-first 审计的盲区** —— 漏读叙述层导致 P1-1 实为 B-1773 follow-up（只在 catalog 可见）"*。

### 1.9 §253 —— 退化的是 BrowserContext 不是 page

**B-1803（Fire-6 RCA C1b）**：`environment.py::_open_fresh_eval_page()` 把 eval 隔离从
**「同-context new_page」升级到「全新 browser context（`browser.new_context`）」** ——
干净 Chromium profile + **auth 从 task config 的 `storage_state` 文件读**（**不调 live `storage_state()`
以免在退化 context 上也挂**）+ viewport；C1 init 与 retry 双路径都用，**retry 每次开新 context**。
属 Amendment 01 已 witness 的 GRL evaluator isolation 边界内 —— **只改 reliability 不改 scoring**；87 个 eval 测试通过。

---

## 二、witness 治理体系 —— AMENDMENT vs PROTOCOL_NOTE

### 2.1 两级体系的分界（§303.3，05-27 新立）

**AMENDMENT_##** = 动 **H1/H3/H10 estimand 定义、`scored_task_count`、observation-id contract、
eval-context、或 sample-pool composition** 的改动 → **git tag + OSF deposit + prereg 正文/附录**
**PROTOCOL_NOTE_##** = **implementation-alignment / runtime recovery semantics 对齐**
→ **git tag + catalog + prose reconcile + chronicle，但 NO OSF deposit + NO prereg estimand 文本改动**

**§387.15.1 的机械应用**：文档类型是 **AMENDMENT_08 而不是 PROTOCOL_NOTE_07**，
因为两个 task 排除**同时命中 `scored_task_count` 与 sample-pool composition 两条**。
定性原文：***"文档类型不是风格选择，是这条规则的机械后果"***。

### 2.2 §250.1 —— pre-fire protocol witness doctrine

**prereg/DOI lock 之后的协议变更必须有 content-addressed tamper-evident 的 pre-fire Git witness（commit + tag），
OSF upload 只是 visibility layer。**
落地 = `AMENDMENT_01_PROTOCOL_RESET_20260521.md`（兼 Protocol Reset Memo）+ `git_witness_*.txt` + prereg amendment-log 指针；
commit `e1f86f4` + tag `prereg-amendment-01-protocol-reset-20260521` + push。
理由：**Protocol Reset（§242-249）晚于 DOI-1 lock（2026-05-18），DOI-2 是 post-data 不能充当**；user 定为 **Fire-6 硬前提**。

**§252 Addendum 01a（witness-timeline gap 的修补范例）**：
Amendment 01 的 tag `e1f86f4` 是 B-1794 修复 commit `681b9cf` 的**祖先**（实测 `git merge-base --is-ancestor`），
**原 amendment 不提 schema≡validator → 存在 witness-timeline gap**。
Addendum 把 B-1794 + B-1796~B-1802 **纳入 pre-fire witness chain**（均在 GRL serialization-adapter 边界内）
并 **supersede 过时的 "~5% parse_error" 表述**；tag `prereg-amendment-01a-schema-validator-20260521` + push；user 已上传 OSF（kv9sf）。
原则：***"witness chain 必须覆盖 fire 前的每一次协议层改动"***。

### 2.3 各 amendment / note 的裁定要点

| 文档 | § | 裁定 | tier 理由 |
|---|---|---|---|
| **AMENDMENT_02**（gate ladder） | §274 (05-23) | **路线 C'** —— H1-strict 6-mode gate **一字节不动**；只 (a) 修 power 表的 estimand **LABEL**（零 archive 数字），(b) 拓宽 R5 报告范围（Route C'-S / C'-R / F 三分支）；**不做 R5→R3 framing rescue** | 乙-full（降 6-mode 为 secondary）会**反转一条 code-enforced anti-rescue guard**（`preregistration_decision_test.py:15-16` 明确禁 R5→R3 rescue failed FE-H1）+ 反转 **user 5 天前亲手锁的**表结构；且**部分动机是怕 H1 挂 → 高 hostile-reviewer 风险** |
| **AMENDMENT_03**（FWER） | §360 (07-02) | **RETIRED 不走**（prereg 不动，论文保持两层表述） | 「能用披露解决的不动锁定文档」—— **amendment 流程成本（witness/tag/OSF）只在数值受影响时才值得**，本案是**零数值影响的语义收敛** |
| **AMENDMENT_04**（analysis alignment） | §281 (05-24) | analysis-layer implementation alignment witness（ADD-label demotion / H10 entropy DEFER / cost total_billed / latency scaffold-adjusted prose / post-R5 route / B2 downgrade），**不改 DOI-locked prereg 正文，无新 estimand/gate/δ/R-ladder** | **gate 算术本身扛住 3-AI 攻击**（bootstrap-percentile / 6-mode-strict drop-one / paired median + 1.20× falsification / provenance 链 / <6 cells fail-closed **全 faithful**），drift 全在外围 producer/figure/README/prose 层；**analysis 层不在 Gate-3 fire import path = fire-safe，但 estimand-adjacent 故仍 witness**（见证时间性：**在任何 paper-grade θ_FE/H10 computed 之前**） |
| **AMENDMENT_05**（coordinate contract） | §285/§288 (05-25) | **首个 estimand-affecting amendment**（coord contract + instruction-strictness relaxed + HARKing 披露）→ tag + **OSF kv9sf 上传**；amendment 文件位置 = `docs/prereg_amendments/`（**不是 `pre_run/`**） | AMENDMENT_01-04 都标 NO-estimand-change，**05 改的是模型输出坐标的解释契约 = 真 estimand 变更** |
| **AMENDMENT_06**（reproducibility sensitivity） | §293 (05-25) | **三平行 mitigation**：主 gate 原样执行 / 另设 **witnessed non-gating reproducibility sensitivity 层** / sensitivity 不稳则 prose 主动降级；**self-oracle 数字必须报 symmetric**（双向 self_drop + discordance + κ）**且只作 instability diagnostic NOT bias estimate** | DOI-1 锁约束**不加 gating-family**；GPT 纠正三点 —— **H10 非仅降 power**（router lucky + baseline unlucky 仍可 false Pareto pass）、**drop-one 正偏非必然**、**self-oracle floor 是 instability diagnostic 而非 bias estimate** |
| **AMENDMENT_07**（SoM identifier contract） | §295 (05-25) | **SoM-family（`[SOM_MARKS]` 文本模式：som / phantom_som / phantom_text）改 deterministic sequential 1..K 编号** + seq-keyed dispatch map 内嵌 native_element_id + fail-closed `_resolve_native_id`；**DOM / P-prompt / vision 保 native CDP nodeId** | **production/标准 SoM 全 sequential → nodeId churn 是 P79 实现 artifact 非 deployment-realistic noise**，**AMENDMENT_06 §4「churn 是真实部署一部分」的前提被证伪** |
| **AMENDMENT_08**（scored-set exclusions） | §387.14/§387.15 (07-27) | task 160 + task 58 **按 §139.8 N/A 先例预注册排除出 scored set** + PROTOCOL_NOTE 披露（选项 a；备选 b=保留但披露 / c=保留不披露） | user 拍板 'a'；**距 REALM 08-05 仅 9 天，k=6 重灌即将执行，该选择会被烤进两篇论文的全部数字，事后更改等于重来** |
| **PROTOCOL_NOTE_01**（session-lost） | §303.3/§330 | paper_grade 下 watchdog 不再 delete+retry session-lost episode，改 **PRESERVE + covariate** | 把**已锁 invariant 应用到 sibling code path，不改测什么** |
| **PROTOCOL_NOTE_02**（transient preflight retry） | §350 (06-21) | episode-level transient retry **收窄到 pre-flight（steps==0）且 class∈{auth,network}**，排除 proxy_5xx（B-1880 管）与 **mid-episode** | 3-AI 收敛：**codex 独家 P0 = mid-episode retry 会造成站点 mutation 污染**（我与 gemini 都漏）；**gemini 关键 defuse = pre-flight-only 一刀同解 mutation + redraw + masking + 组合**；gemini 原判「需 amendment」是基于含 mid-episode 的初版，**收窄后该判定坍缩** |
| **PROTOCOL_NOTE_03**（resume-on-abort） | §352.3/§359.6 (06-22) | **reddit 采用 resume-on-abort（FORCE_NEW=0 从断点续）**；**硬约束 = 仅 reddit**，cls（有 require_reset）与 shop（cart 累积）仍须 FORCE_NEW | **reddit task 实证独立**（`test_reddit.raw.json` 210 task，`instantiation_dict` **0 跨任务 post-ID 碰撞**，全是 find-only/create-only）→ **B-304 的「within-mode 续跑混两套 reset 态」前提被证伪**；resume **只改 which run_id + 省机时不改测什么**；**chunking 反而引入 per-chunk reset 粒度 deviation 故被弃** |
| **PROTOCOL_NOTE_04**（reddit identity reset） | §357.3/§357.6 (06-25) | **在 `env.reset` 内幂等 UPDATE username 复原 reddit 身份**（GRL substrate 复原，镜像 cls 的 `require_reset`） | **诚实标 tier**：不同于 NOTE_01/02 的 estimand 不变，**Fix 4 是为 reddit 定义 estimand = clean per-task**；因 **reddit 无 bound 数据 + cls 零影响（gate on site==reddit）+ 补 cls 既有的 per-task clean-state parity** 才仍归 NOTE tier，**advisor 可升 AMENDMENT_08**。user 硬约束：**不改 auth_refresh/measured execution + episode 期间无 out-of-band 探针** → liveness 门控重登与 verify-then-tolerate 双双出局 |
| **PROTOCOL_NOTE_05**（analysis estimand conformance） | §366 (07-14) | **H3 pool 恢复为「over the 6 planned cells」**（`n_unique<2` 只标 `cell_pass` 不删 cell）；**H1/SR 强制 exact canonical task universe**（新 `lib/canonical_task_universe.py` 委托 `load_tasks` 单源）；`analysis_status` / `h1_verdict` **正交 schema**；伪 0 出清 + 三输出原子事务 | codex **Stop-ship 审计 9 P0**：H3 pool 按 `n_unique≥2` **outcome-dependent 删 cell**（实际 3→2，**代码注释误引 prereg L165 —— 该 floor 只管 cell-level pass**）；H1/SR 不强制 canonical **224/205** exact task set；fig0c bootstrap **从 success-union 反推 universe 致 CI 退化**；**majority baseline 全 cell 标签选多数类 = test-label 泄漏**（smoke 0.533 vs 严格 OOF 0.333）。**修复必须抢在 reddit verdict 数据落地前才是 outcome-blind** |
| **PROTOCOL_NOTE_06**（k=5 早 verdict） | §370/§373 (07-15/16) | **两轨制**：k=5 早 verdict 基线 + B2 后台续跑 + **无条件 k=6 升级**（详见 2.4） | 详见 2.4 |

### 2.4 §370/§373 PROTOCOL_NOTE_06 —— 两轨制（deadline 压力下最谨慎的一条）

- **临时集合预先钉死**为 **cls×3 + B0/B1 Reddit 五格**，沿用同一 paired-bootstrap + FE-IVW
- **所有 verdict 只解释为 "on the five landed cells"，不得冒充六-cell finite-design estimand**
- **双向承诺**：提交前 **B2_red land+bind 即无条件重生 k=6 slotsheet 并作废 k=5**；
  **未齐则 k=5 不因好坏改变**
- 缺 B2_red 后 Gemma 只剩 cls 单 cell → **B-1284 自动一档降级（R1→R2 / R2→R3）**，
  **R-tier 上限 R2，禁 cross-site Gemma replication claim**

**唯一书面理由锁为两条**：
(1) **Jul 28 AoE 硬 deadline + B2 red 实测 1.65d/condition 整链 ETA 07-25**；
(2) **B2_cls 与 B1_red router 不可训令 H10 5/6 已结构性 fail-closed 且与 B2_red outcome 无关**。
**任何 interim H1 数字/分支概率/方向明令禁止作理由。**
**时间上已有 interim，故文书诚实写成「非 pre-outcome-blind 但 decision rule outcome-independent」。**

**§373 advisor 07-15 三项**：① **NOTE_06 两轨制口头 APPROVE 当日激活**（去 DRAFT + 四项披露同变更集）；
② **B3 = MiMo-VL 先行之后扩展其他模型，8 月投稿后启动**；③ **新指令 = router baseline (prior work) efficiency 对比**。
§8.2 splice 采取**加段不删段**（原 Statistics 段的 power/HKSJ/α 披露**非 NOTE_06 授权删除范围**，
K5 段独立成段，**k=6 升级时整段删除即回吐**）。

**§383.1（07-22）**：**B2_reddit（~07-26/27）落地做 k=6 现在够得着且是实质升级** ——
"5 of 6" 免责声明整段消失 + **B-1284 cross-family modifier 解除**（Gemma 两站齐 → 跨族复制主张可用）
+ Protocol Note 06 两轨制披露整块可删。

### 2.5 不走 amendment 的（轻量 tag / 无 witness）

| § | 事项 | 为什么不需要 amendment |
|---|---|---|
| §275 (05-23) | **B-1839 per-condition docker restart**（cls reset 加 `docker restart classifieds_db classifieds` + db `SELECT 1` ready + http 200 warmup） | **substrate hygiene 非 estimand change** → 只打轻量 git tag `b1839-substrate-prefire` + paper methods disclose。**改的是衬底不是 `metrics.py`（零改）→ power 只更稳、对所有 mode 对称无偏** |
| §326 (06-09) | **B-1870 cls reset POST（`page=reset`）加对称 retry+backoff**（initial + 3 retries, 5/10/15s，全耗尽才 return 1） | 同上；**reset 幂等**（000 = 请求没到服务器无 partial；500 也 DROP+seed 幂等）故 re-POST 安全。触发：**B-1839 docker restart 后健康探测有 retry（SELECT 1 ×30 + http200 ×30）但紧接的实际 reset POST 零 retry = 不对称缺口** |
| §344 (06-19) | B0 condition wallclock cap 16h→0（unlimited）+ watchdog idle-alert 30→60min | **不动 max_steps/prompt/evaluator/task set/reset boundary → 无 OSF witness**。理由：**fixed wallclock cap 跨 site 不可移植**（per-condition wallclock = per-step latency × per-task step-count，后者随 site/mode/能力剧烈变化）；**step JSONL 10min 无写入才是可靠 stuck 信号**（B-1667 双信号设计） |
| §347 (06-19) | diag ruleset freeze **v5→v6**（+7 规则 P34-P40 + 3 bug-fix guard + 一批 success-safe 收窄） | **diag 在 `scripts/analysis` 非 fire import 路径且非 estimand → 无 OSF witness**；FP 只做 diag 标记**不动 SR** |
| §349/§352.1 (06-19) | **B-1880 capped exponential backoff**（`retry_backoff_max_s`，None=uncapped 向后兼容）；api_strong max_retries 3→8 / backoff 10 / cap 60（容忍窗口 ~3min→10.7min），后再 8→24（~35min） | **retry 成功返回字节一致 response 故无 OSF witness**。理由：**无上限 doubling 会让单次 sleep 膨胀到 1280s 且 proxy t+50s 恢复却白等 320s**；**step-level retry 是 estimand-SAFE（同 rollout 等出来非重抽）所以"等多久"是唯一旋钮，加厚无估计量代价** |
| §328 (06-09) | **B2 全系数据丢弃 + 开 pan-and-scan**（`do_pan_and_scan: true`）重跑；R17895 **不物理删，降级为现成 config-ablation arm（pas-off 对照）** | 走 prereg **Appendix E.6 + git tag** witness，**不做 hypothesis amendment**：改动在 **B2 canonical 数据存在之前** + **移向 vendor-recommended 是 conservative 方向** + 不动 hypothesis/estimand/gate；**经济账 = B2 沉没成本仅 1 condition**；**paper 叙事收益 = 防"你用 Gemma 最差视觉配置证明 cross-family 差距"的攻击** |
| §383.2 (07-22) | **H10 产物重生 = producer 只跑不改，先进 scratch 验证再 promote** | **避免在修产物时改动 gating 逻辑**（tag `h10-canonical-artifact-regen-k5-20260722`；产物 gitignore 故 SHA256 落 doc） |

### 2.6 §281 —— 一次差点撤销自己决策的 near-miss

**cross_site canonical latency 的 triple-subtract（minus_retry − busy_wait − screenshot）保留**（P1-4=A′），
**改的是 stale 的 prereg §4 / amendment-01 prose 而非 code**。
过程：*"初判 code 多减了是 drift，但**查 `master_bug_catalog` 实证**：triple-subtract 是
**B-1669（Q6=A 2026-05-18 治 red 99s busy-wait 污染）+ B-1780（Q3=A 2026-05-20 commit 526db4b, 1162 tests）
两次 witnessed 的演进**；**盲目改回会撤销用户自己的决策 + 重引入 red 污染 + 反而需新 amendment**"*。

---

## 三、venue 三次转向与 paper 拆分

| 时间 | § | 决定 | 依据 |
|---|---|---|---|
| 2026-05-29 | §309 | **advisor 三点**：(1) **run-to-run noise 作 disclosed limitation 接收，不再深挖不切 provider**；(2) **初步 venue = workshop**；(3) **重心转 router** | **workshop 对 single-run noise 容忍度更高，noise 不是 workshop blocker，router 才是核心交付物** |
| 2026-07-02 | §360/§361.7 | **投 AAAI-27 main track**（abstract 07-21 / full 07-28），**消耗 main-conf 提交权**，EMNLP cascade + D11 early-Sep 被取代；**k<6 降级预案 = a+b**（透明披露投稿 + H10 缺席走 §6 descriptive），**排除 (c) 弃 AAAI**；**B-1885 分母 205 不动**（死任务只披露不豁免，**全员同伤故 paired 对比无偏**） | 见下 |
| 2026-07-22 | §383.1 | **AAAI-27 撤出**（07-28 full deadline 放弃），改 **2026-08-05 双投 REALM @ EMNLP 2026 workshop**：**Paper A（phantom routing 现象）= non-archival**，**Paper B（路由阴性结果 + 标签诊断）= archival** | REALM long = **8 页正文 + refs/appendix 不限**，双盲 ACL 2026 style，notification 09-07 / camera-ready 09-14；**A 保住之后投主会的选项**，**B 本就不太可能单独撑主会，拿 Anthology 记录更划算**；**两篇切法避免 dual-submission**（A = 菜单存在且非冗余；B = 天花板高但学不到） |

**§375 作者与提名**：作者名单 **Jiaming Wei（一作）/ Zekun Wu / Adriano Koshiyama / Maria Perez-Ortiz**
（暂定，Jul 21 AoE 冻结前须确认）；**Reciprocal reviewer 提名 = Zekun Wu**
（**AAAI-27 reciprocal reviewer nomination Jul 21 AoE 冻结，资格作者未提名 = desk-reject**）；
prior-work baseline 范围扩展到 lit review（D7 §2.3）里的 web-agent router / model-switch 类工作。

**§331 advisor 的两个书面 deliverable 节点**（06-10）：
(1) **1 页项目总结**（goals+RQs+experiments）且**学生自设全程 deadline 并持续自查 on-track**，
**self-target 06-12 / official hard ≤06-22**；
(2) **自选一章初稿（lit review）**，**self-target 07-13 / official hard ≤07-20**；
canonical deadline 表 = one-pager §4 **双层**（self-target + official hard）+ live 跟踪在 `_status/tasks` frontmatter `eta`。
理由：**两份文档 = advisor 仅有的正式书面反馈 checkpoint**。

**§385.1 scope 辨析**：**thesis（D7-D11，official early-Sep）≠ REALM workshop 双稿（Paper A/B，08-05，8 页）**；
学长毕设写作 rubric 13 条中 **②⑧⑩⑪ 是 thesis-scale，①④⑥⑫⑬ 对 workshop 稿通用**；
**⑤ disc+concl 合并（limitation 内嵌）直接决定 D8 章结构 = 合并版不分裂**。
理由：***"rubric 每条 = 考官逐点扣分位置，比'写好一点'可执行"***；
逐条标满足度（**基本满足 3 / GAP 8 / 部分 2**）把抽象指导变成 backlog。

**§381 post-submission 路线**（07-16，8 月算力按 rebuttal 日历分配）：
① **B3=MiMo-VL-7B 先行 12 conditions**（cls+red × 6，7 月下旬 DGX 起步 / 8 月上旬 A100 fire）
② **WA 50-task pilot 插空**（B1 本地跑）③ **shop 18-cond 刻意排后** ④ **B0 replicate 附录机会性**。
理由：**B3 堵"cross-family 证据 = 单地板模型"攻击 + 检验"强模型→标签充足→router 可训"假说**；
**WA 兑现 prereg §8.8 B-1296 注册可证伪预测，单位算力信誉资产最高**；
**shop 电商类与 cls 泛化边际低 + 6 条 B0 重挂 proxy**。
**§381 同时否决**：**提交前重跑 B0 replicate**（**outcome-dependent sampling**，NOTE_06 §0 禁令）；**8 月后附录形态合法**。

---

## 四、合并稿的论证结构（原「Paper A / Paper B」）

> ⚠️ **标题已过期，内容没有。** user 2026-07-28 拍板**两篇合并为一篇**（§398.8）——
> 不是因为两篇都弱，而是 Paper B 强、Paper A 弱。本节以下各条仍然全部生效：它们正是
> 合并后四步骨架 ①②③④ 的来源。读时把「Paper A」读成 ①②、「Paper B」读成 ④。
> 第 ③ 步（结构小于同模式重跑地板）两篇原先都没有，见 §398.2 + §406。

### 4.1 Paper B 论点闭合（§383.4，07-22）

**瓶颈不是标签的定义方式，是标签的产生率** —— **标签只在任务被解开时诞生，成功率 7-13% 时无法凭重新切分制造事件**；
**三条独立路径均指向同一瓶颈，论证闭合**（假设类 15 格 sweep / 监督侧三种定义 / 池化换可识别性）；
**且这批证据是 k-无关的 → Paper B 现在就能写完不等数据**。

**§387.16.4 第四条独立路径**：**路由的两半都失败但败因不同** ——
**which-mode 败在标签供给（16-97 个）**；**triage 标签够、AUROC 也够仍然失败，
因为 base SR 只有 2-27% 时真正要赢的对手是一条平凡的固定策略，而那点边际经不起检验**。
**两道控制缺一不可**：*"只加 shuffle 会漏掉 cls/B2（学到的确实非随机，只是不如固定策略）；
只加 always-cheapest 会漏掉 cls/B1（确实赢了固定策略，但赢的量打乱标签也能得到）。
**「比随机好」和「比什么都不做好」是两个不同的问题**"*。

**§396.5 user 拍 (a)** —— 只重写散文 + 换用 `oracle_sr_cost` 行，**不重算标签**（commit 9d11178）。
意外收获（原文）：*"做完发现 (a) 不只是除错，它**把 Paper B 的骨架捋顺了** ——
**triage_only 与 route_only 是天花板的两半，而这两半各自需要的标签正好是 §4 和 §5 分别杀掉的那一个**；
论证从『一个天花板 + 两个失败』变成『**两半天花板，各被一个障碍精确关掉**』，兑现了 contribution #1 原文"*。

**§387.16.1 策略枚举的逻辑收口**：**「先 SR 后 cost」与「先 cost 后 SR」在单次选择下是同一条规则**
（「能解的里面最便宜的」≡「按 cost 升序第一个能解的」）；
**真正分岔只有两种**：**cost 完全压过 SR**（永远挑最便宜 = 固定策略），或**选择变序贯**（级联，成本累加）。
故算了 **4 条真正不同的策略而不是 2 个同义词**。

### 4.2 Paper A 的正面结论被主动削弱（§397.8，本片最重要的科学方向变化）

***"本 session 唯一的科学方向变化：Paper A 的正面结论被有意削弱 —— H3 过门这件事现在在
Limitations 与 §1/§6.1 里都被明确标为弱证据（门测 ≠0，而同策略两次跑本来就 ≠0）。
这不是新发现的缺陷，缺口一直在只是从没写进稿子。**要回到更强的表述需要真的补一次同模式重跑**，
而 A100 排队 ~3 天后才有窗口"***。

**§396.2 攻击的来历**：self-oracle noise floor 攻击**确认为真且两个月未 defuse** ——
H3 没有同模式重跑的噪声地板（**Gemini #4 = codex #8**）；
**项目自己的 `extract_50_features.py:636` 早就警告过 N=1 oracle 需要噪声天花板**；
**这是 2026-05-15 Mode C pilot 提过的同一条攻击**。

**§397.4 的处置**（⚠️ **该处置基于「没有可用 null」这个已被 §397.10(3) 推翻的前提**）：
**保留结构性结论**（两轴都过预注册门、**温度 0.0 贪心已消除解码方差**、预注册 ≥2 独解地板 5/6 格达标）
+ **在不计页数的 Limitations 里把缺口说全**，**且明说 axis-1 的 1.35pp 余量是两轴里更不宽裕的那个**。

**§380 三条 framing 纪律**：
- **F3：framing tier stays R5；C'-S = lower-claim report, not rescue** —— ¶4+§7 明写
  （**codex 判 R5 隐匿；降级路线不得被写成救援**）
- **F9：H10 "pending" 不等于 "可能通过"** —— **结构不可达显式化**
- **F10：五格×六臂 forest 表落 supplement S2.6 + 主文指针**（Protocol Note 06 §1.2 per-cell visibility 义务）
- **教训**：***"verdict-day 审计必须覆盖预写语言，不仅是填槽操作 —— 预写文本自身也会携带 estimand 漂移"***
  （F1 的 "per-arm irreplaceable coverage" **出自预写底稿而非填槽**）

**§378**：k=5 interim 指向 **Branch B（H1-fail + H3-pass = Amendment 02 Route C'-S）**，预写已备
（H1 pooled 不过 +1.0pp 线而 H3 双轴过线）。

### 4.3 verdict-day 工具链（§361.1，防转录漂移）

**`verdict_day_slotsheet.py`**（gate JSON → slot sheet 单命令，**read-only formatter 零 estimand 逻辑，
draft 数字唯一合法来源**）+ `VERDICT_DAY_RUNBOOK.md` + **branch_prewrites**
（Branch A=H1-pass / B=H1-fail+H3-pass=Route C'-S 双分支整段预写）
+ **F2 图画 H1 strict 估计量**（**刻意与旧 `fig_forest_drop_one` 的 vs-3-mode ADD-lift 分开**）。
理由：*"§360.1 四个 P0 **全是转录/压缩漂移** → **结构性修复是让数字不过人手**；
F2 与旧图分开是 **Amendment 02 估计量混淆防线延伸到图表层**"*。

**§360.3/§361.2 聚合准入链**（固化进 NUMBERS_TODO §0）：
A100 fire 侧 `validate_fire_manifest --apply` auto-bind → `fire_manifest.json` →
**手动 promote 进 `results/phantom_paper/run_manifest.yaml` 的 cells（`grade: paper-grade`）** →
**aggregators 读 run_manifest 非 fire_manifest**；**promote 保持手动 deliberate 步骤，cron 只做 gap 检测不自动 promote**。
根因：*"既有 sync cron **只拉 episode 数据，binding 登记无同步路径** = **B1/B2 cls done 三周不进聚合**的根因"*。

### 4.4 单位与可比性纪律

- **§377（07-15）**：**USD 永不跨 backbone；跨模型平面用 tokens / retry-adjusted latency**
  （B0 是 API 计价，B1/B2 是本地电费）。
- **§382（07-16）文献效率定位表硬纪律**：**定位非同平面 —— 不进我方 Pareto、无跨论文相对句**；
  **每数字带「文件:行号」，无本地数字标 UNVERIFIED**；**两篇同名 SCOPE 不得混数**。
  理由：**19 个效率类方法的节省与代价全是自报值，与我方口径不可比**。
- **§315（06-04）routing novelty 支点收紧**：从"第三轴空"收紧为
  **"no peer-reviewed systematic per-task input-representation routing on a fixed web agent"**（五源锁死）；
  **routing 谱系实为约 8 轴，P79 占最细 representation 格**。
  依据：**4 份 deep-research + codex zero-preset 独立交叉 + WebFetch arXiv verify**；
  closest = **Read-More-Think-More（2604.01535, static guideline, arXiv-only 不算 peer-reviewed）**
  + **SeeAct（2401.01614, ICML24, 同模型异 grounding 对照，明说 SoM not effective for web）**；
  **V-GEMS（2603.02626）经核实是 memory+grounding agent 非 representation router = DR characterization error**。
- **§324（06-08）**：**batch-invariance 缺失（He et al. 2025 blog）作为 §302.8 Layer 2
  （AWS dynamic batching + multi-tenant FP reduction）的白盒机制名**进 §4 citation；
  **attribution 拆开**（batch-invariance→He blog **无 arXiv 镜像** / floating-point non-associativity→Yuan 2506.09501）。
  理由：**我们黑盒证据（cross-provider replay + within-B0 paired floor 13.3%）独立 reproduce 了已知现象**；
  **Yuan 未用 "batch invariance" 术语故不做 citation-stretch**。

---

## 五、B2 / B3 模型线

### 5.1 B2（Gemma3-VL）—— 地板确认

- **§328（06-09）B2 全系数据丢弃 + 开 pan-and-scan 重跑**（见 2.5）
- **§335（06-15）**：**不调 Gemma-specific prompt**；把 **B2 reframe 为 cross-family grounding 地板 + scaffold-fit 证据**
  （其 parse_error 梯度 + 终止失败**反而是干净的 axis-2 cross-family 证据**）；§8 disclose 终止 gap + scaffold family-fit confound。
  理由：**救不了地板大头 + mid-fire 改 = 污染 + 破坏同-scaffold 公平**。
  > ⚠️ 台账 flag：*"§335 判断中把 **scaffold family-fit confound over-weight 了**"* 被 **RETRACTED §338**。
- **§338（06-16）B2 ~1% = 真地板非 bug，SETTLED（高置信，六源收敛）**：
  §327 视觉分层崩坏 + digest 95% agent-limit finish 23/224 + §330 0-finish 钉死 +
  **GPT 官方 playbook 审计无 bug** + 文献无 4B 锚 + §335 action 级。
  依据：**GPT 官方 playbook 逐条对账 —— bf16/processor/chat-template/double-BOS/greedy 全官方可辩护**；
  **P0 = 非 Gemma-native agent/termination prompt + 1024-cap×P&S，后者经 probe 关闭**。
  **Path 1 守 shared-scaffold control（推荐）vs Path 2 加 Gemma-native 条件分解 floor 留 advisor**
  （**Path 2 与 byte-identical control 冲突且需 prereg amendment**）。

### 5.2 B3（MiMo-VL）—— 第 4 基线

- **§339（06-16）**：**B3 = 第 4 基线 ADD 不 replace**，首 pilot = **MiMo-VL-7B-RL-2508**
  （GLM-4.6V-Flash 作 gated alternate，**需先过 format-conformance gate**）；**pilot 全程 DGX 不碰 A100 paper-grade fire**。
  **cross-AI 选型三个决定性分歧证明跑两个 AI 的价值**：
  **GLM format-lockout**（GPT 抓 Gemini 漏 —— GLM-4.1V 在 MedCUA 固定 schema **432/432 zero-action**）/
  **InternVL 版本+地板**（2.5=InternLM 干净跨族但 **VWA cls zero-shot 0.4% 地板**）/
  **MiMo-VL**（GPT 独有，跨族 + WebVoyager-SoM 34 + 无 lockout 旗标）。
  **无候选有可核 raw zero-shot VWA SR ≥6% → 地板风险对整个小通用 VLM 类是真的，只能 pilot 答。**
- **§340（06-16）跨模型维定性**：**replication BREADTH 非 controlled ablation**
  （**router 只在同模型内比 representation，从不跨模型比**）→
  **advisor 的 matched-capability control 是 over-constraint**，family/thinking/architecture 降为 §8 披露项，
  **硬约束只剩 4 条**（不地板 / 守 JSON 格式 / 装 40GB / 开源）。
  **且 bolt-on 架构（如 MiMo）优于 native early-fusion** ——
  *"因 phantom 骑在 **language-prior/mirage 机制**上而 **mirage 系文献全是 bolt-on 模型**（原生架构是无人区）"*；
  **多样性碰到机制（mirage）时非中性**；原生 Gemma4/Qwen3.5 属 mechanism-probe（§5 shelved 不碰）。
- **§341（06-17）集成方式**：**`MiMoVLAgent(Qwen3VLAgent)` 子类只 override `__init__`**
  （换 `AutoModelForImageTextToText` load + image_token_id 防御），**继承 240 行 `step()`/prompts/confidence**；
  + `local_mimo` backend + factory lazy dispatch（**零碰 B0/B1/B2 fire 路径**）。
  理由：**MiMo-VL = Qwen2.5-VL 架构 → 单一真相源，避免 Gemma 那种全类复制**。
  离线 verify **2/2 parse-valid 真跑 `step()` 全路径**，think 块被 `action_utils` B-800 自动剥。
- **§342（06-17）**：**放弃 DGX 路径跑 B3 floor pilot**，改为 A100 cls fire 完后在 A100 跑；**不追 task_configs 消失的幽灵**。
  依据：**全库 grep（p79 + scripts + VWA submodule）对 `task_config*` 只有 `tasks.py:134` mkdir，
  零 unlink/rmtree/rename，5 个会动 run-dir 的路径全排除 → 删除来自 P79 之外 = DGX 共享环境幽灵非 B3 code bug**；
  A100 同 runner 跑 17 条无此问题。

---

## 六、Router 的技术裁定

- **§254（05-21）MI feature selection 传 `discrete_features` mask**（binary 恒在 design matrix 末尾 15 列），
  lambda 改 `functools.partial`，`N_SELECTED` 加 `--k` CLI；
  **selected set 只当 operational compact feature subset，不作 causal feature-importance / feature-discovery claim**。
  根因：**sklearn `mutual_info_classif` 对 dense X 默认全 continuous** → **15 个 binary 被 k-NN(Kraskov) 熵估计
  当连续处理**，{0,1} 上 distance-0 ties 靠 seeded 噪声打破 → **binary MI 下偏，binary indicator 在 top-k 被系统性压低**。
  **修复时机零成本**（`raw_features_phase1a.json n_pooled_total=0`，Pass-1 未 land）。
- **§256（05-21）**：**F3（τ 定义在 mode-match accuracy 而非 task-SR）+ C3（inner-CV τ 经 outer-fit MI selector 有 mild leak）
  只做 disclosure + deferred，不真修**。
  理由：**二阶问题**（outer 评估用 outer-holdout 干净，**§254 的 "ZERO leak" 结论不变**）；
  真修需**跨 Stage 改 outcome matrix / 改 user-confirmed pooled selector**，且**数据未 land 无法验证**。
- **§256 H10 加 transparency 指标**：非支配 gate 加 **`router_strictly_better`（θ-CI 下界>0）marginal-utility 层**。
  理由（gemini G4-C）：**Pareto 非支配门槛太低 —— 退化 router 塌成 100% phantom_som 仍被 gate 放行 = learning 幻觉**。
- **§312.3/§312.4（06-02）H10 gate 判据（non-dominance）不动，走路线 (c) prose 降级 + disclosed limitation，不做 amendment**。
  理由链（一字不丢）：*"P0-1 撞 2026-05-18 locked H10 estimand（DOI-1 substance-lock）；
  回溯 §153：**non-dominance 是 v6（2026-05-16）为修 codex+gemini 双 AI 指出的「SR-only superiority = false advertising」
  而深思熟虑选的** → **一行 fix 实为 estimand 钟摆**（SR-only 太严 → non-dominance 太松 →
  strictly-better 堵洞但 cost-constrained θ 仍可能误杀 task-conditional routing）；
  **workshop 不要求 prereg，原则 = 让论文 claim ≤ 已锁 prereg，永不用 amend**"*。
- **§312.5 四个 fire-blocker 修复**：P0-4 cls router wait cap **8h→24h**；
  **P0-3 `learned_router.py:379` τ 缺失从静默默认 0.5 改 hard-fail（estimand-PROTECTING）**；
  P1-9 `extract_50_features.py:117` `bool(success)` missing-field 静默当失败 → **skip+计数（estimand-ADJACENT）**；
  P0-2 `queue_router_learned.sh` leaf gate 改调 `_lib_lr_artifact_validate.py` 验 fold-aware bundle +
  **paper-grade 下 hard-block `ALLOW_NO_LR_MODEL`**。
- **§328（06-09）B-1871 —— twin leak（一次抓在零成本窗口的漏）**：
  **per-cell 独立 `StratifiedKFold` 改为 per-site shared pure `KFold`**（**孪生同 fold by construction**）；
  **stratification 移除**；C1 universe 契约保持，round-robin 废除；JSON schema 不变。
  根因：*"pre-fix **同 site 三 model cell 的 task→fold 不对齐**（stratify 在各自 oracle label 上），
  **holdout task 的逐字同 intent 孪生行约 96% 留在 fold-k selection pool**
  → **共享 vectorizer/MI selector 对 holdout intent 不盲**，
  『Leak: ZERO』与 §6『no holdout-leak』**双声明在 task 层面不成立**"*。
  **修复时机 = fold-aware Stage 1→3 尚未在 paper-grade 数据上跑 = 零数据成本窗口。**
- **§388.4 B-1900**：`cascade_cost_first` **从标 oracle-free 改标 oracle（success-detection）+ 增报无检测器时的真实成本**。
  根因：`if succ[m][t]: break` 的 `succ` 是**评测真值**，**部署时没评测器真级联会在每个任务上跑完 6 个 mode**。
  **教训（可复用）**：***"『oracle-free』要问的不是『训练用没用真值』，而是『运行时每个分支决策所需的信息，部署时拿得到吗』"***。
- **§387.16.5 B-1896**：**白名单从 `run_manifest.yaml` 的 paper-grade 条目生成（不手列）**
  → **router 眼里的 canonical run 不可能与 aggregator 漂开**。
  **教训**：*"一个『有就严格、没有就宽松』的机制，**在白名单落地前等于没有机制**，
  而警告只在有人读日志时存在。§367 读到了、记下了、写明必须先做 —— 然后没人做。**预警不是防线**"*。

- **§399.3（07-28）router Pareto 结果必须三档并列报** —— **非支配**（admissibility）/
  **严格支配**（superiority）/ **相对六固定 mode 菜单非支配**（是否在前沿上）；
  **引用任何 pass 率必须说明是哪一档**。
  根因：`EXP_SPEC_pooled_tier_router` **§2 把假设写成「Pareto 支配」而 §3 把判据锁成
  「95% 非支配」**（§150b.4 / B-1550）。两者不是同一问题 —— 买 **+7pp SR** 花 **+10% cost**
  算非支配、不算支配。首版 verdict 只跑锁定判据，**报出与假设相反方向的 headline**
  （"H-pool SUPPORTED"，而实际 0 个 cell 支配）。
  **教训（可复用）**：***"假设措辞与锁定判据可以在两节文档里悄悄分叉，而单跑锁定判据
  不会报错、只会给出反方向的结论 —— 写规格时假设与判据必须并置在同一节里核对"***。
  ⚠️ 这条**只牵涉规格文档内部**：§387.16.4 / §392.2 的「0/6 Pareto 胜过」本来就是**支配**
  语义（`router_triage_learnability.py:575` 原话「a genuine trade-off point, **not a
  dominating one**」），`INDEX.md` 五层表 ② 行的措辞**是对的**，不需改。

### 6.1 §396.7 —— B-1806 案例（本次台账重建的直接教材）

**B-1806（2026-06-09 F2）已裁定不得静默切到实测成本 tie-break** ——
原文：***"Do NOT silently switch to a measured tie-break — that is an oracle-label estimand change"***。
**本轮 user 选方案 (c) 报两版**：`derive_cost_oracle_label()` 做成**并行函数而非 `derive_oracle_label` 上的开关**
（**刻意不给参数，免得任何 caller 能靠传参切 estimand**）。

**核过的两条理由**：
1. **实测每模式成本序跨 cell 不稳**（P-prompt 在 cls·B0 是第二便宜、在 cls·B1 是最贵）
2. **更要紧的一条**：`total_billed_cost_usd` 是**实际发生的 episode 成本、由步数主导**，
   而**步数取决于这趟走没走成** —— **用它定义标签等于把结果信息写进一个动作前路由器要预测的目标，这是循环**

**§396.7 教训（PROGRESS.md 引用的正是这条）**：
***"跨 AI 审计给的『这条明显该修』建议，在动手前要先查它是不是已经被裁定过 ——
codex 是冷读看不到代码注释里的历史裁定，我看得到，所以核这一步是我的责任而不是它的疏漏；
本轮如果直接执行会 (a) 静默推翻一个有论证的决定 (b) 把循环性写进标签
(c) **让审稿人能直接引用我们自己代码里的注释来打这篇 paper**。另：修完要回头查因此变假的旧陈述"***。

---

## 七、最近两周的 estimand 审计（§388–§397）—— 教训密度最高的一段

### 7.1 universe 一致性的三层修复（B-1906）

**§389.3 根因**：`expected_scored_ids()` 返回 `(ids, sha)`，**只写 `[1]` 就能拿到一个看起来合规的血统标签而数据一行没动**
—— *"**这比完全不接 canonical 更坏**，因为**跨产物 SHA 互校必然放行**（它比的正是那个被抄过来的正确摘要）；
下游 `fig_f2_h1_forest.py:88` 只比 `len()` + 靠 `task_set_sha256` 兜底，
而那道兜底**假设上游 SHA 是照它自己的数据算的** → **串联防线整条失效**。
**根因不是『忘了过滤』，是 API 让『记 SHA』和『过滤数据』成了两个可独立完成的动作**"*。

**§389.6 三层修复（非补丁）**：
1. **新 API `restrict_to_scored()`** —— provenance 的 `content_task_ids_sha256`
   **由裁后实际内容算出不从 canonical 抄**，使「标签对内容宽」**在结构上无法表达**
2. `h10_pareto` 标注改诚实
3. **常驻 lint `test_universe_consumption_lint.py`** 把「修横切必须 grep 普查」**变成每次 `make test` 都跑的普查**，
   带 **`UNIVERSE_TRIAGE_PENDING` 只减不增棘轮**（*"陈旧条目会给下次回归打掩护，故修好必须移出"*）

**§389.6 B-1911 PARTIAL —— 有意不放宽**：fold 缓存是 **pre-AMENDMENT_08 的 205 集**，
**故意让它继续失败**并在报错里指名 B-1904 + 给出正确修法。
理由：*"**静默丢掉那 2 个 fold 条目 = 复用一个在不同 universe 上算出分层的切分，是同一缺陷的更安静版本**"*。

**§389.7 对 §388.7.1 横切教训的两点修正**：
(1) **普查不能靠人记得做 → 它现在是测试不是纪律**（默认拒绝 + 棘轮，新脚本不接 canonical 就红）；
(2) **别只问『谁调用了』，要问『调用后拿它干什么』** ——
*"B-1906 的调用点全都在 Q5 census 里 codex 也列全了，**出问题的是返回值只被消费了一半**。
**`[1]` 这两个字符就是防线和摆设的全部差别**"*。

**§397.5(c) 棘轮被自己绕过**：`aggregate_cross_mode_failure_signatures.py` 经 `_discover_episodes` 读 episode，
**源码里零次出现 `_summary_v2.json` → B-1906 lint 的 grep 看不见它，reddit 分母一直是 205**；棘轮探测已扩到 `_discover_episodes`。
**教训**：***"防线要按能力（谁读了 episode）而不是字面（谁写了那个文件名）来布"*** ——
**B-1906 棘轮是昨天立的，今天我写的新脚本就从 grep 看不见的路径进来了**。

### 7.2 一批可复发的方法论教训（本片密度最高）

| § | 教训 | 实证 |
|---|---|---|
| §388.1 | **审自己的新分析时，先问『这个数在物理上可能吗』再问『代码对不对』** | Mode C 做了一步我自己审三轮都没做的算术 —— **报出的 0.04418 低于任何单 mode 的均值（最低 Vision 0.06481）**，而任何『选一个 mode』的策略不可能低于所有 mode 均值的下界。**一个越过物理下界的数字，不用读代码就知道错了** |
| §388.2 | **空 universe 必须 fail-loud（原为 fail-open）** | 首版交集写 `t in set(expected_ids)` 而 per_task 的 key 是 str、`expected_scored_ids` 返回 int → **交集为空 → `n_paired_tasks_total = 0`，而 producer 依然 rc=0 写出产物并报『H2(a) not falsified』—— 空集上不可能证伪，读起来像通过**。且 **B-1013 那条专为防 str-vs-int drift 而建的断言只检查 common set 内部是否混型，不检查它与外部比较集是否同型** |
| §388.4 (B-1897) | **一句『某某所以不用校正』的豁免，在它守护的假设从佐证升为主张的那一刻就变成承重件 —— 升级承重时要重读旧豁免** | H3 族校正理由原写『闭式 CI 所以不用 p-family』，但 **95% CI 不含 0 与 α=0.05 拒绝同构，两条未校正 CI 膨胀 FWER 到 ~0.0975**；**实质无损**（两轴从容过 Holm：7.5e-07 vs 0.025 / 1.19e-05 vs 0.05）**但理由坏** |
| §388.7.2 (B-1902) | **置换检验的单元必须是『打乱后仍自洽的最小语义包』** —— 只打乱 label 而把 outcome 留在原位，造出的是**既非 H0 也非 H1 的第三种世界**，而它偏哪边不可预测。**『我加了置换零分布』≠『我控制了假阳性』** | 本次**两个 cell 方向相反**（一个偏保守一个偏激进） |
| §392.3 | **三条（B-1906 / B-1904 / B-1903）都不是『算错了』，而是把验证责任交给了一个不会执行它的地方；修法同型 = 让检查发生在数据实际流经的那一点，而不是声称别处会查** | B-1906 交给抄来的 SHA 互校；B-1904 交给不读该产物的下游 gating consumer；B-1903 交给『分数是 OOF 的』这个事实，**而 OOF 的 fold 与外折不同** |
| §393.3 | **两种分母的硬规则**：**success RATE → 计分集**（reddit 203 / cls 224，**一律引 `sr_per_mode.json`**）；**episode 级覆盖率/计数 → 采集集（205）**。**防线放在哪取决于是谁在填** —— 脚本里的字符串 → 测试；**人手填的数字块 → SKILL.md 新增『分母硬规则』章** | ***"测试管不到人下一次手填什么，规则必须写在人看的地方"*** |
| §393.3 | **`git checkout <file>` 会连未 commit 的修复一起丢** | 误用后测试立刻从 4 passed 变 4 failed —— **意外验证了新测试真能抓错** |
| §396.3 | **本轮三条最贵的错（oracle 列错位 / 'preregistered' 误述 / '5–25×'）都不是算错了，而是一个数字或一个词从产生它的地方搬到别处后就没人再核对它了 —— 且这次全部发生在 md→md 的搬运上，不是代码里** | triage_only 搬进散文时丢了策略名；AMENDMENT_08 的 POST-HOC 标签搬进 §2 时丢了；digest 里算错的 summary 搬进 §4.2 时被当成事实 |
| §397.5(a) | **只有在 action-step 与 episode-incidence 两个分母下都成立（都 6/6）的陈述才准进正文**；固化成产物里一张『哪些陈述扛得住换分母』的表 | 交互结论保住了，靠的是**比原来任何一个计数都稳的对比**；**若不换分母，两个塌掉的计数会原样进稿** |
| §397.7 | **稳健性检查与估计量检查是两回事，都要做** —— 换分母/换口径查的是『算得对不对』，**查不出『这个量是不是你说的那个量』**；后者只能靠问『**这个名字的定义是什么，我的计算满足那个定义吗**』 | **本轮三条最贵的错全在估计量层，我的自审一条都没抓到** |
| §397.6 | 三条会复发：(1) **跨 AI 报的那条往往不是最深的那条** —— #9 报『分组错』真问题是『前提假』；#11 报『次数少』真问题是『裁定由 B 决定』；#3 报『标签互换』真问题是『证据类型选错』，**照着报的修 = 修一半**；(2) **修完要用一个正交的口径复核自己**；(3) **自己刚立的防线自己第一个绕过去** | 本轮实证 |
| §387.7 | **0 不带 provenance** —— dataclass 的 default 落盘之后与真实测量值在 JSON 里完全无法区分；**字段名读起来像测量结果、默认值又恰好是一个语义合法的观测值时最危险**。防线：**全库恒为默认值的字段必须在 schema catalog 登记为「未实现」并加测试断言** | B-1887 说的是 `dict.get(key, falsy_default)`；**B-1890 是同一陷阱的 schema 层版本** |
| §383.3 | **`dict.get(key, <falsy default>)` 出现在标签/结局派生路径上，必问『这个 default 会不会把「没测量」伪装成「测量到阴性」』** | 两个 loader 严格度不同**而产标签的那个是松的**（对照 `router_offline_replay` 的 fail-closed 契约） |
| §387.7 | **归因结论的可靠性取决于证据到原始轨迹的距离** | 本次三条结论里**唯一一条建立在统计字段上的是错的**（B-1890），**两条建立在轨迹逐步核对上的是对的**（B-1889 + P36 机制） |
| §387.6 | **低 SR cell 上 success 侧审计的权重要显著高于常规** —— B2 每个 condition 只有 1-8 个成功，**一个伪成功就能定性改变该 cell** | task 160 使 **B2_phantom_prompt 0.49% → 0.00%** |
| §387.9 | **单个实例的分布形状不足以支撑『系统性偏向』这类断言** | **FP 污染量并非随能力单调递减，而是中间高两端低** —— 受影响最重的是 **B1（中等能力）不是 B2**（*"白嫖这类 FP 需要『做了一点但没做对』—— **太强（B0 常真把任务做完）和太弱（B2 连目标页都进不去）都不容易中**"*）。**本 session 第三次『先下的定性描述被后续数据修正』** |
| §387.12/§387.13 | **凡 sub-agent 说『没有 X』，都要用 0-token 全量扫描复核一遍** —— 抽 6-8 个样本得出的『从未出现』是最不可靠的一类断言 | **9 个 sub-agent 提议里 3 条在全量复核后被否**（页面内嵌视觉=结构性不可解 / mutation 任务普遍更难 / missing union_bound 从未出现） |
| §387.8 | **单 condition 的机制观察不要直接当跨 condition 的解释用** | 三口径对照把 sub-agent 的机制假说**适用范围界定清楚**（主结果留下，泛化被砍掉） |
| §387.11 | **监控器的假阳性比假阴性更危险** —— 它把真告警淹掉并训练操作者忽略告警；**每加一个 benchmark 都要回扫监控层的路径常量** | **同根因第 5 次复发** |
| §387.4 | **继承机制的测试必须覆盖比现存最深链再深一层** —— 一个只有一种深度在用的递归结构，等于没测过递归 | 仓库里在跑的 config **全是一级链**，『继承能用』被一级链的成功持续证实了两个月，**而二级链的失败没有任何消费者去触发** |
| §386 | 装第三方管线时**『跑通 self-test』只证明它在自己的 fixture 上工作**；真正决定能不能日常用的是**两个宿主实测**（白名单误命中率 + 阻断策略红灯率），**两条都是装完 5 分钟内量出来的** | 不量就会得到一个『装好了但没人用』的管线 |
| §391.5 | **『注释描述的契约』必须有会红的东西守着** | 今天三条同形 bug 的修法都是**加测试/加断言而不是改注释**（B-1898 'SINGLE source mirrored by' 而两边都是函数局部字面量；B-1906 号称血统可核而只抄摘要；B-1916 注释称 'master chain exports FORCE_NEW=1' 而 `${FORCE_NEW:-0}` + launcher 不 export） |
| §385.2 | **DRY_RUN 必须在收尾前 `exit 0` + 真启动先 rm 陈旧 marker** | self-inflicted bug：DRY_RUN 走完循环仍执行循环外 `touch` marker + notify → **dry run 留下的 `.SWEEP_DONE` 在真 sweep 启动 60s 后误触发 done-monitor + 发假 DONE 推送**。**失败方向是『伪造成功』不是『报错』，比静默死循环更阴** |
| §384.5 | **『暂搁』不等于『冻结无债』**；**同一 artifact 换路径 = 静默失效，取数前先数文件别信目录名**；成本估算要拆出主导项 | §5 搁置两个月期间 **fire 换了数据源 + 落盘布局重构了两次** |
| §387.1 | **共享/长 uptime VM 上自动安全更新会制造『用户态已换、内核态没换』的时间炸弹**，且**只在下一次真正用 GPU 时才炸** —— 长期不跑 GPU 的窗口期结束时应主动验一次 `nvidia-smi` | 本次实证（用 `rmmod nvidia_uvm/nvidia_drm/nvidia_modeset/nvidia` + `modprobe` 原地弥合，**不 reboot** 以绕开 KubeVirt 'reboot 前必须 detach p-79 volume' 的坑） |

### 7.3 §397.9 —— 探测器灵敏度不对称（三家 AI 全漏）

> 🚫 **本节记录的论证本身已被判「别用」，2026-08-01 补标。** `next_steps.md §4`
> 不要引用的数字表明写：*「§397.9 我那套『探测器灵敏度』论证 → 别用 —— 这条 id 噪声
> 早有正经测量」*（`b0_paired_idperturb` probes：B1 本地 temp0 组内一致性 1.000、
> id-shuffle 改变决策 **20.0%**；B0 **12.5%**）。下面的**观察**（两套 id keyspace 稀疏
> vs 稠密）是对的且仍有解释价值，但**不得作为 §4.2 的机制论证**——要谈 id 通道就引
> 那组 probe。本节此前与该作废清单不一致，是结论层内部的一处漂移（结论层生成于
> 07-28/29，作废清单是 07-31 追加的，两者之间没有引用关系）。

**幻觉率指标的判定基准按臂分成两套 key 空间（id-namespace）**：
- **DOM / P-prompt** 的 text 是 AXTree → map 键是**原生 CDP nodeId，稀疏**（median 7839-18729, max 691695）
- **P-text / P-SoM** 的 text 是 legend → `build_som_text_from_obs_text` **重键 1..K，稠密**（median K=15-17, max 176）

**机制**：**稀疏空间下抄错一位几乎必然落在有效集外 → 计数；稠密空间下选错元素仍命中有效 id → 不计数。
所以 legend 臂的低幻觉率不代表引用更准，而是『引用错』这件事这个探测器看不见 →
跨 namespace 比 = 比两个灵敏度不同的探测器。**

**§397.9 由此加的检查清单第 5 条**：***"这个指标的判定基准在各臂是同一个吗？"*** ——
*"一个指标定义为『落在集合 S 外』而 **S 本身随臂变化** → **该比率不是同一个量，它是「行为 × S 覆盖度」的乘积**；
**要么固定 S 比，要么别比**"*。
**三家 AI 全漏的原因**：*"codex 有代码访问但查的是 provenance/分母/格式化；Gemini 只有内联散文看不到代码；
我自审查的是分母和估计量名字，**没查探测器在各臂是否等灵敏**"*。

**连带处置**：给 `write_digests.py:170` 那条『P-SoM 干净 2.3-24.8× vs dom』**加 🚨 注释**
（**它是同一个跨 namespace 比较，也正是旧 §4.2 的来源**）。

> ⚠️ **该 id-namespace 归属表被 §397.10(1) 判为不完整**（漏 SoM 与 Vision）。
> PROGRESS.md 已记主 session 实证：**compact 1..K 是三个 mode（som / phantom_som / phantom_text）；
> Vision 零 element_id，其幻觉率 0.000 是结构性不适用而非「native」**。

### 7.4 §397.1 —— H3 两轴命名的"转置"

**axis 是对 P-SoM 的集合差，差里变的永远是另一个 knob**：
**axis-1（|P-text \ P-SoM|）两臂都带 mark legend；axis-2（|P-prompt \ P-SoM|）两臂都带 SoM prompt**。
**项目代码早就这么命名**（`axis1_microbehavior.py:144-155` 把 (P-prompt, P-SoM) 这对叫 `axis_1_text_alt`；
`axis_effect_size.py:521` 标注 `'controls': 'prompt=SoM, image=no'`）。
**H3 的命名（相对 DOM 的位移 = 预注册口径）与机制分析的命名（差里变的因子）恰好转置，
两套各自自洽，§4.2 把两套混用了。**
裁定：*"**codex 说『两轴标签互换了』只半对**；正确的修法不是换个方向重推，
**那个论证的证据类型就选错了** —— 2x2 跑过了，直接在幻觉引用率上做分解就行"*。

**§397.3**：codex #11 的『与选择步骤共污染』这半**不成立** ——
**mode 选择在 bundle 置换下是不变量**（置换的是 per-task outcome bundle 的次序，每模式 SR/cost 是 multiset 均值）；
**真缺口是稿子没说测的是哪个量**（§5.4 测整格阈值下的 saving，§5.3/Table 6 报嵌套策略，**读者会默认前者在检验后者**）。
> §397.7 记 **codex 反向确认了这个判断** —— ***"跨 AI 也能用来证实，不只是证伪"***。

**§397.7 三条改名裁定**：(1) **Bayes ceiling → in-sample modal agreement**，数字不变，补『仅共享 task』70.3%/74.1%；
(2) **撤掉 'interaction' 一词**；(3) **mode-invariance → 'similar after pooling' + 明写格内 spread**。
（codex 复算与我逐位相同；**Gemini 指出极值条件下 `min−max ≤ 任意其它两两差` 是代数必然**。）

**§397.7 两条基于误读但仍有价值的 Gemini findings**（把『跨 mode 离散度』读成『率本身』；
把 4.975e-3 约整为 5.0e-3 当成用错公式）—— **攻击不成立但两处都指向真·歧义散文，已改措辞**。
***"冷读模型的误读是歧义检测器：它误读的地方审稿人也会。"***

**§397.7 页数教训**：*"页数第三次踩同一个坑但学到新东西 —— **最好的删法不是压缩措辞，
而是删掉『报了又立刻自我折扣』的那组数**（89.9/96.7 —— 报出来又马上说这个上升有算术成分所以论证靠别的数），
让正文直接以真正承重的 agreement 数字开头。**删字反而让论证更直**"*。

### 7.5 §397.10 —— CORRECTION 节与台账重建的起点

**(3) 的根因**：**存在专门工具 `compare_cross_run_same_condition.py:227-247`**
打印 `self_drop archive->current = sum(y1[i] and not y2[i])`，**正是 H3 轴的估计量形式** ——
**该工具早就存在，不该从零重推**。

**根因总结（第 6 条会复发教训，本 session 第 4 次同型）**：
***"§396.7 刚写过『动手前先查有没有被裁定过』，今天在测量维度上又犯三次。
**真正缺的不是『想得更深』而是一个 question → prior-finding 的索引** ——
笔记 2 万行 400 节，grep 只在猜对关键词时有用。
→ 下一步（user 拍板）：**全 phase 重建，Phase 0 先建『已测量 / 已裁定 / 已作废 / 现有数据』结构化台账，
产物必须是可查而不是摘要**"***。
> **这正是本次台账 + 结论层工作的直接起因。**

**追加 (4)**：**Phase 0 台账的 COMPUTE 类必须记驱动 pid / pidfile，不是 worker pid**
（错误形状与 (1)(2)(3) 完全同型：**查代理量（worker pid）而非真对象（sweep 驱动）；四次同型**）。

**追加 (5)（错误形状不同）**：*"前四次是『没查已有的东西』，**这次是『没应用我 context 里已经加载的规则』**
（CLAUDE.md 的三层算力表 + host-role 分工）—— **所以修法也不同**：前者要『可查台账』，
后者要**在推理算力调度前先回读 host-role 表**。→ Phase 5 的『动手前 4 问』加**第 5 问：
这台机器的角色分工是什么，我是不是在按它的角色推理？**"*

**append-only 纪律**：**不改上面的历史记录，在 §397.10 里作废 —— 读 §397.4 / §397.9 必须连 §397.10 一起读。**

---

## 八、基础设施与运维裁定

- **§258（05-21）flock OFD 语义**：**advisory lock 绑 OFD 不绑 fd 号** → 用**共享 `spawn_paper_grade_daemon` helper
  在 spawn 前关 chain fd9 / leaf fd7 / watchdog fd8**；并 **deprecate `phase1a_relaunch_missing.sh`（hard-fail）**；
  新增 **RESUME_MISSING 模式 + `fire_manifest.json` + `validate_fire_manifest` ghost gate**。
  根因：*"leaf `setsid nohup` spawn 的 runner/watchdog **继承 fd 9** → chain `exec 9>&-` 只关自己那份，
  **lingering watchdog 仍持 OFD 副本** → 下个 condition `flock -n 9` 失败，**假报 double-fire**"*。
- **§288（05-25）A100 = source of truth**：**任何 `results/` 的 mv/rm/archive 必须在 A100 端做**
  （DGX 只是 `sync_a100_results.sh --delete-after` 的单向 mirror）；**analysis/diag/读数在 DGX OK**。
  理由：**DGX-only archive 会被 A100 canonical 原件复活，或被 `--delete-after` 删掉 DGX 的 `_archive`**；
  另 **sync 的 phase1 用 `--delete-after` 而 toplevel 不删 = 不对称，跨机清理要全覆盖 4 处**。
- **§301（05-26）partial symlink**：`results/` 整目录 symlink 到 scratch **不可行**，
  改**只 `results/visualwebarena` → scratch**，provenance/mechanistic/repro_replicates 留原盘。
  根因：***"git 不 follow symlink 做 tree traversal"*** —— 整 results symlink 让 git 把整个 subtree 当 deleted
  （**即使物理文件在 scratch 且 ls/realpath/read 都 work**）→ **paper-grade git fingerprint clean check（Gate 3）必 fail-closed**。
- **§301**：**archive 旧 run 目录与 update manifest binding 必须原子同做**（AMENDMENT_07 archive 时漏了 manifest rebind）；
  **修法是 follow-up commit（不 tag 不 OSF）**，因为 **AMENDMENT_07 governance 已 complete，这是 implementation follow-up 不是新 amendment**。
- **§314（06-03）/ §344（06-19）wallclock cap 全部改 unlimited**：condition 级（B1/B2 4h→0，后 B0 16h→0）
  与 orchestrator chain-wait cap（24h→0）**双层都改**；**real deadlock 靠 watchdog idle-alert 兜底不靠 wallclock cap**。
  理由：**弱模型 wallclock 不可从 per-step latency 预估**；**同病灶要查两层**（第二层是重启后才在 log 里发现）；
  **同类 bug 跨参数复发**（06-03 改 B1/B2 时漏改 B0）。
- **§365（07-07）4 天无人值守 boundary orchestrator 上线**（顺序单跑 `queue_*` + per-boundary bind +
  per-boundary ntfy + **proxy 探活门控** + fail-safe 停在安全状态 + 每条 65h ceiling）；
  方案 A 全自动 / **B0 abort 则跳过续 B1 链** / B2 纳入队列尾。
  根因：*"boundary 动作（bind → 探 proxy → 启下一条）一直是手动步骤 → **07-01（15h）与 07-04→06（35h）两次停摆同根因**
  = **monitor 只通知，动作靠人，无 session 就停摆**；cron `inprog=none` 也不报『该跑而没跑』（**监控盲区**）。
  **无人值守自动化的边界应画在动作而非通知**"*。
- **§369（07-15）机会成本调度**：中断 B2 dom（停在 40/205）抢 proxy 窗口插 B0 pprompt reddit resume（96/205 起），
  **纯排序变更，两条 condition 各自协议/estimand 不动**。
  理由：B2 red 实测约 **11.7min/ep = 1.65d/条 ×6 ETA 07-25 压死 Jul 28 AoE**；proxy 窗口历史规律越来越短；
  B0 pprompt 差 109 eps（约 27h）即完成 B0_red cell（**verdict 生死格**）；
  **orchestrator v2 的 pprompt 重试计数已耗尽，不手动插则本进程永不再试**。
  **episode 级并行被三重锁死**：方法节承诺串行+重置 / 并发状态污染×Fix-4 / §5.4 latency 表内可比性 ——
  ***"VWA 式并行恰是本文批判并修复的对象"***。
- **§384.3（07-23）**：**布局漂移在读契约层翻译一次（symlink shim），不改 9 个消费点**
  （对比 **B-82 = 同类跨 extractor 漂移修了一整轮**）。
- **§384.6（07-23）**：**`generation_manifest.json` 恢复必须从 fire host rsync 权威副本，禁止本地重造**。
  理由：**重造脚本 clean-rebuild 912 个 task config 会打断正在读 config 的 runner**；
  且 **URL 从当前环境注入 —— A100 权威副本记 `localhost:9980`（self-hosted），
  在 DGX 重造会写进 Tailscale 地址 → manifest 记录一份 fire 从未用过的 config，provenance 反而更假**。
  （该文件被 submodule `.gitignore:145` 忽略，**任何新 clone/新机器都会缺**。）
- **§390.2/§390.4（07-27）B-1914**：`queue_chain.sh` 的契约是**一个 argv = 一个完整『脚本+参数』串**，
  launcher 每个 step 都是裸 token；修法 = **step 改数组 + 引号 + 两道断言**（step 数 ≠ 6 / 任一 step 不含空格 → exit 3）
  + 加 reddit runner 占用检查 + **修 `$!`**（**`setsid` 的 pid fork 后即死**，改按刚铸 log 路径 pgrep 真 pid）。
  根因：*"这脚本此前**只存在于 A100、不在版本管理** —— **没有 diff / review / 测试，是它带着 bug 活下来的直接原因**"*。
- **§391.1（07-27）WA 走 Route b**（6 臂在同一采集契约下从零跑满 104）—— user 拍板 b。
  理由：**不修 FORCE_NEW 会得到 5 个非主角 mode 各『10 个非 paper-grade + 94 个 paper-grade』混在一个 cell 里**，
  且 **phantom_som 因 pilot 期已被切 full base 跑到 104 而整臂 skip → P-SoM 每个 episode 都是非 paper-grade，
  而 P-SoM 是论文主角臂**。
- **§387.15.6 WA 全量化的实现方式**：**不动已注册的 pilot 采样**（prereg §8.8 的 10-per-site 分层抽样**是可复现的注册对象**），
  改为**新加一层 `configs/exp_v2_wa_full_reddit_base.yaml`**（`task_ids.reddit: {__delete__: true}`，B-574 显式 unset 哨兵），
  6 个 per-condition config 改指向它；**WA shopping / shopping_admin 不放开（没有 reset 实现）**。
  理由：**就地改会让那次注册无法验证**；这是 **B-1888 递归 defaults 修复的第一个实战用例**（三级链）。
- **§387.5**：**WA 跑全 6 mode（user 拍板）**；**注册的是 5-mode universe**（Vision 因 GPU 预算排除）
  → **6-mode 是注册估计量的超集，报告时以注册的 5-mode mean-pairwise Jaccard 为主、6-mode 作探索性**
  （**不破坏 prereg §8.8 注册对象的可验证性**）。

### 8.1 monitor 的三条新教训（§390–§391）

- **§390.3/§390.5 B-1915**：**monitor 收尾动作若是『启动另一个任务』，必须校验那个任务活着** ——
  新 monitor done-condition 用 **pilot 真 PID `kill -0`（Tier 3）+ fire 后 `sleep 60` pgrep 校验真在跑
  + 三结局各自 ntfy**（LAUNCHED / FAILED-to-start high-priority / 24h TIMEOUT）。
  *"CLAUDE.md 的 done-monitor 章节此前只管 self-match 一类失败；**这次是另一类 ——
  done-condition 正常触发了但它守的那件事自己失败了而 monitor 不检查，整条接力静默断裂还报 exit 0，比没有 monitor 更坏**"*。
- **§391.2 B-1917 合并规则**：**monitor 里任何 `if ! <check>` 都必须能回答『这是条件不满足，还是我没测到？』** ——
  B-1915 = 条件触发了但守的事失败了；**B-1917 = 条件根本没触发，只是探测失败了**。
  **远端探测必须让被测方回话**（`P79_ALIVE` / `P79_DEAD` / **空串=不可达永不下判断**，连续 6 次 ~30min 才 high-priority ntfy），
  **不能只看退出码**。
  实证：`if ! timeout 90 ssh HOST "kill -0 PID"` **无法区分 ssh 失败与 pid 不存在**；
  DGX 18:58:12 报『pilot chain exited — firing full chain』**而 pilot chain 直到约 2 分钟后手动 kill 才结束，
  只有 `queue_chain` 的 flock 挡住了这次 double-fire**。
- **§391.4 B-1918**：**一个任务一个 monitor** —— 前一 session 为同一件事留了三个 monitor 其中两个会 fire；
  **修正应该停掉旧的不是再 arm 一个**。
  *"monitor 的可靠性缺陷在单 monitor 下最多是『没接上力』；**在多 monitor 并存下会升级成『并发写同一批 paper-grade 数据』**。
  本次没出事只因为 flock 在"*。
- **§391.2 跨机对时间线先 `date` 两端** —— **A100 跑 UTC / DGX 跑 BST**，chain log 的 17:58:14 与 monitor 的 18:58:12
  **是同一时刻**（一开始误读成『另一个 monitor 抢跑』，**诊断被时区拖慢**）。
- **§387.16.6**：`supervise_mechanistic_canonical.sh` —— **PID-based 存活检测**
  （**不用 pgrep 模式匹配，那会匹配到 supervisor 自己的命令行**，CLAUDE.md 2026-05-09 教训）、
  崩溃自动重启、**连续 3 次 30 秒内秒死则停并推送（防 restart-storm）**。
  理由：**共享 GPU 机器上『外部作业到来、内核回收你』是预期失败模式**。

---

## 九、工具与流程

- **§270（05-22）/stress persona 改场景驱动 5-profile menu**（routing-ML / stats / systems / reproducibility /
  **mechinterp-archive-only**），**机械层一字未动**。
  理由：*"persona 钉死 mechinterp 而 §5 已 2026-05-14 shelved，**反倒 Phase 2 learned router 这周 active 开发
  → persona 指着冻结工作**；user 每次带 scope 覆盖了『读哪些文件』的 staleness，
  **但覆盖不了 persona / OOB bar（它们作 standing context 静默校准攻击手感）**"*。
- **§348（06-19）cross-AI 架构的边界**：**不引入 AI 互辩（debate），也不引入 fix 阶段 cross-review**；
  **裁决权收归单一事实核验者（Claude ground-truth）**；
  唯一落地 = **Phase 4b —— 1-AI unique 的 P0/P1 findings 必 100% 走 Phase 4 citation 验证不 sample**。
  理由：***"AI 互辩收敛的是 rhetoric 不是 truth"***；fix 是**已知 bug + 已知 contract 的定点修复**，
  cross-AI 边际收益不抵 token；**2-AI/3-AI overlap 有天然交叉过滤**（两 lineage 同时幻觉同一 file:line 概率极低），
  **1-AI unique 无此过滤且 blast 最大**。
- **§345（06-19）Mode C 从 gemini CLI 迁到 Antigravity CLI `agy`**，**默认档位 Gemini 3.1 Pro (Low) 而非 (High)**；
  wrapper **从中性 scratch cwd 跑且不给 repo `--add-dir`**；**audit 绝不用 `--dangerously-skip-permissions`**。
  依据：Google **2026-06-18 退役** Gemini Code Assist 个人免费层与 AI Pro/Ultra OAuth（`IneligibleTierError`）；
  **High 档在 print 模式 240s 内不收尾（实测两次空输出）而 Low 约 22s 出 submission 级 audit**
  → ***"审稿质量瓶颈是独立视角不是 thinking budget"***；
  **`--add-dir` repo 根会自动加载 `AGENTS.md`/`GEMINI.md`（指向 CLAUDE.md）破坏 cold-read 独立性并触发探索 loop 超时**；
  **yolo flag 实测把 `agy` 推入 agent tool-loop**（带它 120s 超时 vs 纯 `-p` 13s 干净）。
- **§388.5 零预设 cross-AI 的价值**：*"Mode C **独立重现了我自己找到的 2 条**（universe 不一致 / in-sample 阈值）——
  **2-AI 独立重合不可能是被我的结论引导出来的**；另**独占 3 条我完全没看见的**（per-instance 成本预言机 / cascade 标签 /
  SE-floor 不对称）。**自审能抓『与已有记录冲突』，抓不到『我的心智模型本身有洞』；后者只能靠外部 cold read**"*。
  （**本 session 第 7 次『先下的结论被后续数据修正』，前六次靠查笔记/查全矩阵发现，这次靠换一个 lineage 冷读发现**。）
- **§388.7.4 跨 AI 分工实证**：**codex = 全库普查 + 实际重跑复算；Gemini = 单点算术/物理约束直觉**。
  **2-AI 重合（最高可信）**：in-sample 阈值（3-AI）· task 58 outcome-selected 排除 · universe 不一致。
- **§396.3 Mode B 后检 PASS with caveat**：21 findings 深度足，
  **但它引的文件名（`section3_oracle_complementarity.md` 等）在仓库里不存在，是它按语义自行重命名的**；
  **内容引用与数字全部正确（逐条抽查），所以是路径 confabulation 而非读错文件；行号同样不可直接跳**。
  → **下次 Mode B 的 prompt 要显式给出文件清单并要求逐字回引路径**。
- **§272（05-22）/diag 采用 discover-then-freeze 两阶段协议 + `RULESET_VERSION` 字段**：
  (1) **Discover** —— 每个 mode 各自 self-evolve 建失败字典取并集（**产出是规则字典非 mode 结论**）；
  (2) **Freeze + 全量重扫** —— bump `RULESET_VERSION` → `diag_autorun` 重扫所有 condition → **同版本才可 cross-mode 比较**；
  **当前相位明文禁止 cross-mode 定量比较**。
  纪律原文：***"完备性来自看全 6 mode，可比性来自同字典统一打分；纪律是『任一时刻所有 condition 同版本』而非『绝对完备再 freeze』"***；
  /diag 定位是**内部工具+大致定量（不进 paper）**故 paper-grade held-out 门槛可松。
- **§261.2（05-22）digest 命名从 run_id 改为 per-condition**（`<model>_<mode>_<site>_diag_digest.md`），run_id 保留在 header。
  理由：run_id 随机后缀看不出 condition，**36 condition 堆一个目录会全是随机码**；
  **per-condition 而非 per-cell 因为失败模式强绑 mode**（dom 视觉盲区 vs som/vision 看得到图）。
- **§386（07-24）paper-deslop 三层设计**：**rewrite skill（交互式只出 diff，语义风险全在这层）→
  `invariant_check.py` 词法保真闸 → Vale**；**LLM 改写层永不无人值守，确定性层零语义风险才进 CI**；
  管线 **vendored 到 `tools/paper-deslop/`**（**P79 不是 paper-only repo**），所有 Vale 调用必须 `--config=`；
  skill 本体 tracked 在 `tools/paper-deslop/skill/` 再 symlink 进 `.claude/skills/`（**因为 `.claude/` 整个 gitignore**）。
- **§396.1（07-27）**：另起 `docs/checkpoints/paper_drafts/latex/` **而不改 `aaai27/latex/`**；
  `convert.sh <paperA|paperB> [--submission]` 按 paper 参数化；新增三个 gate（零 `\cite` 给可读诊断 /
  未定义引用计数 / `--submission` 加验 refs 起始页 ≤ 9）；`overleaf_sync.sh` 改双篇并加 `pull --ff-only` 前置。
  理由：原 `convert.sh` 行为被测试 fixture **端到端钉住**；**AAAI-27 已于 07-22 撤出，
  让一份 ACL 投稿住在名为 `aaai27` 的目录里会持续误导**；**原 overleaf 脚本盲推会吞掉 Overleaf 网页端未回流的编辑**。
- **§396.6**：`SINGLE_COL_BODY` 做成 **per-paper 显式清单而不是可推导规则**。
  理由：*"pandoc 只在单元格相对全宽够短时才出 `l` 列，而 **`l` 列不折行** ——
  **单栏下长表头会越界压到邻栏且 LaTeX 不报 Overfull**，只能渲染成图人眼判；
  本 session **两次靠眼看抓到会进 PDF 的错**，两次都是 **LaTeX 静默通过**的"*。
  **ACL 合规**：未编号独立 `\section*{Limitations}` 且**不计页数**（`acl_latex.tex:335`）；
  **页数门改为量 `\label{content-end}` 而非 refs 起始页**。
- **§395.5**：**invariant 与内容修正冲突时按 skill 让 invariant 赢** ——
  §5.4 原句逻辑不通，先改成 '8 vs 4' → invariant FAIL；**skill 不许『解释掉』违规**，
  改为**只用原有的 four 把逻辑说对**（事实一致，且不引入新数字）。
- **§395.3 自建 lint 当天抓到自己**：新诊断脚本里写了 `expected_scored_ids(site)[1]`
  （**正是几小时前为 B-1906 立的禁令**），**没加豁免**，改成从上游 features 产物读 content SHA
  （单一来源 + 与 canonical 交叉核对、不一致 fail closed）。
  ***"『防线要放在数据实际流经的那一点』最直接的一次自证。"***
- **§385.2 24-cell canonical sweep 编排**：优先级排序（P1 主效应→P2 内容特异性→P3 选择偏倚→P4 跨 mode→P5 弱控制）、
  **可续跑**（`pilot_summary.md` 存在即 skip）、**到点截断 `DEADLINE=08-01`**。
- **§384.4 两个例外**：**curate 保 50 tokens / tier_b id-patching 保 256 tokens**。
  理由：(a) **tier 阈值（composite≥1.0 / overlap<0.5 / reverse≤-1.5）是在 50 下标定的，
  换长度等于静默重定义 strong tier**；(b) **§300 实证 64 tokens 会全 PARSE_FAIL**（读不到 action_type）。
  > ⚠️ 台账 flag：*"§384.4 的 mechanistic sweep 规格 = @15 tokens 全 28 cell"* 被 **RETRACTED §385.2**。
- **§303（§302.5/§302.8.3，05-27）noise 的 honest claim 边界**：
  **paper-grade honest claim 只能停在 observable provider-dependent noise floor**，
  **sub-mechanism 不可 isolate without 服务端 audit artifact**；
  **三条 escape hatch 明令不可用**：『switch to DashScope 解决』/『MoE 是主因』/『provider implementation bug』。
  依据：Layer 1 在 DashScope 仍约 **75% partial-nondet**；**MoE under-evidenced**（高 margin 仍漂，
  **multi-token cumulative 才是 surface mechanism**）；**AWS 可能 by-design 做 multi-tenant batching 不是 bug**；
  **`system_fingerprint` 两 provider 都 None = audit gap 行业普遍**。
- **§298.4（05-26）**：**任何机制声明必须消去法 + 实证双锚，只有消去 = 推断不算定论**。
  依据：§282 用消去法 + SOM/DOM 对比推断 id-channel 且自己明示 replay deferred；**§298 用 B1 dense 受控实验做实**；
  *"中途我两次从 surface 推断没读机制没实证被拽回"*。
- **§295 一条 provenance 澄清**：**P79 som 的 nodeId 编号是 design-change 非 bug-fix**
  （git blame：`processors.py:532` 与 `947` **都是 upstream-VWA 原生**，**P79 历史从无 sequential→nodeId 改动**，
  `p79/` 全仓零 renumber 代码，**P79 用 `observation_type=accessibility_tree` 从不调 VWA 原生 image_som**）。
  **结论是 design-change，但因『production/标准 SoM 全 sequential』这一新证据仍决定改 sequential。**
- **§295 axis-1 reframe**：**不再是"纯拍扁同 id"，而是 AXTree-native representation →
  SoM-indexed selection representation（text substrate + identifier contract bundled）**；
  **H3(ii) P-prompt(nodeId)↔P-SoM(seq) 非纯 prompt-axis isolation，用 non-collapse 语言**。
- **§387.7 P36 = agent-limit 不是 scaffold-bug（四路独立一致）→ B2 的低 SR 是真实能力测量，可以进 cross-family 分析**。
  机制：`walk_fail:no_actionable_within_walk` **只在 element_id 已在 `obs_nodes_info` 中找到（union_bound 存在）之后才触发，
  从不代表『引用了不存在的元素』**；**真正机制是 `_JS_RESOLVE_CLICK` 按设计不接受纯文本 `<input>`** ——
  **task 181 在 step_13 把 click 换成 type 后立刻 `target_tag='INPUT'` 成功，直接实锤**。
- **§387.10 / §387.13 routing 空间的外边界**：**『任务需要视觉』这个属性不能预测 routing 增益** ——
  与 §387.8 的 comment/reply 天花板同类，两者都界定 routing 空间的外边界
  （**64 个页面内嵌视觉任务 + 52 个 comment/reply 任务，有重叠，在所有六个表征上都接近地板**）。
  **§387.13 写作纪律**：**discussion 必须把三条并排写**（comment/reply 天花板 4× 18/18 cell ·
  『需要视觉』不预测增益补图≈0pp · 主导失败签名全部 mode-无关），
  ***"否则 oracle 数字会被读成『还有这么多能靠 routing 拿到』"*** ——
  **routing 空间的边界比 drop-one oracle 数字看起来的要窄**。
- **§387.15.2**：`analyze_run` 的 `success_rate_scored_set` **原是半个操作**（分子取全部 observed episode，
  分母取 `scored_task_count`），修成**分子分母同集合** + 新增 `n_success_scored_set` / `n_protocol_excluded_observed` 两个透明列。
  *"这是一个**只在『排除』这个动作上才会犯的错** —— 平时改分子不改分母不会出错，因为分母是固定的 N；
  **一旦分母本身是被改动的对象，『扣分子』就成了半个操作**"*。
- **§387.15.1 排除放 analysis 层不放 load 层**，三条理由：
  (1) 已落盘与将来 run 的 `episodes == scored_task_count` 精确校验（B-1834）**语义保持一致**；
  (2) **with/without 两条敏感性臂对任何 run 都算得出来**；
  (3) **runner import 路径一行不动**（`analysis.py` 只被 `cli/analyze_experiment.py` import）。
  加了测试断言 **registry 永远不得被 import 进 `load_tasks`**。
- **§387.14 两个排除的依据强度不同，必须在 NOTE 里如实标注**：
  **task 160 强**（判据纯 task-config 可推导，**完全 outcome-blind**，与 §139.8 的 N/A 判据同构）；
  **task 58 弱**（**因为被观察到成功才进入视野**，判据不能纯从 config 推导，**带 outcome-adjacent 成分**，
  **正是此前否决『B0 replicate 提交前重跑』时援引的同一类风险**）。
  应对 = **NOTE 中明记依据强度差异 + 附带/不带各自的敏感性分析，让审稿人自行判断**。
- **§387.15.5 测试同时钉住数字和否定掉已被证伪的理由文本**（断言 'no N/A taxonomy' 不得复活）。
  *"一个常量表被部分更新时，**没被更新的那几行往往会长出一个解释**；**那句解释比数字危险得多** ——
  数字错了会被实测抓到，**理由错了会让实测结论被驳回**。只改数字不删理由，下一个作者会把它改回去"*。
- **§387.15.5 一条正交补充**：user 规则『下结论前先查笔记』是对『只有一个数据切面就下描述』那条教训的**正交补充** ——
  **一个防『证据不足』，一个防『与已有记录冲突』；两者都不靠更聪明的推理，只靠多做一次检索**。
- **§389.8**：**判定『代码 vs 文档谁 stale』之前必须查 amendment 台账 + git witness tag**；
  ***"『写了 recommended follow-up 却没有承接机制 = 一定会漂』→ 落成会红的测试而不是再写一条建议"***。
  实证：*"本例里 amendment 不但存在还提前写好了正确修法，**只是没人回来执行** ——
  于是**两个月后一次 /stress 把『文档陈旧』读成『实现越界』，差点据此把 estimand 改回去**"*。

---

## 十、⚠️ 本片矛盾与待核清单（合并阶段用）

| # | 事项 | 两侧 / 需核 |
|---|---|---|
| 1 | **§397.10 是 CORRECTION 节** | 作废 §397.4（"没有可用 null"前提，被 (3) 推翻）与 §397.9（id-namespace 归属表不完整，漏 SoM 与 Vision）；追加 (4)(5)。**读这两节必连 §397.10** |
| 2 | **Holm family key** | A3 §240 B-651 改 `(test, metric, cell_key)` cell-scoped vs **§244 #10 revert 回 `(test, metric)`**。**台账未记 revert 理由**（12-point 是 user final decision）。⚠️ 需核当前实现 |
| 3 | **§335 vs §338** | §335「scaffold family-fit confound」被 **RETRACTED §338** 判为 over-weight；§338 结论是 **B2 ~1% 真地板 SETTLED** |
| 4 | **§384.4 vs §385.2** | §384.4 的 mechanistic sweep 规格被 **RETRACTED §385.2**（"@15 tokens 全 28 cell"）。⚠️ 待 B 批核 |
| 5 | **§302.5 被两处 RETRACT** | (a) `named by RETRACTED §397.4`（"全 archive 只有一对同模式重跑"）—— **而 PROGRESS.md 记该说法本身是假的**（manifest 19 组 ≥2-run，`results/repro_replicates/` 有两个 clean replicate）；(b) `named by RETRACTED §397.10`（§302 的线性减法 12.1% ≈ 10.5% + 1-2pp = category error）。**三层嵌套，引用 §302 需同时读 §397.4 + §397.10 + PROGRESS 的实证更正** |
| 6 | **§242 被两处 RETRACT** | `named by RETRACTED §243`（"Fire-6 继续"）+ `named by RETRACTED §298.4`（"element_id 是 red herring，真机制是 MoE"—— 台账注明"session 中途我自己说的"） |
| 7 | **§383.4 / §387.9 / §387.16.x 被 RETRACT** | §383.4 的『约 1/4 训练标签由 MODES 硬编码顺序的 tie-break 决定』被 **RETRACTED §395.2**；§387.9 的 task 58 取证描述被 **RETRACTED §387.15.3**；§387.9 的 B1_som 数字叙述被 **RETRACTED §393.1**（分子改分母没改，见 PROGRESS M3）；§387.16.1/.4/.5/.6 的『两个存活的 cell 一个都不过 Holm 校正』被 **RETRACTED §388.7.2** |
| 8 | **§293 被 RETRACT** | `named by RETRACTED §309`（§308 提议的 DashScope 同-checkpoint probe + Risk 6 官方 API 减 noise forward note 作为 workshop 前置）—— §309 advisor 已定 noise 作 disclosed limitation 不深挖 |
| 9 | **§137.x 系列被 §387.15.5 点名** | A2 的 §137.2/.3/.4/.5/.7 都挂 `named by RETRACTED §387.15.5`（queue_chain.sh episode gate 注释『WA stays at pre-exclusion since WA has no N/A taxonomy』）。**这是 flag 传播的噪声**（同 § 号被 RETRACTED 记录点名 ≠ 该裁定被作废），合并阶段须区分 |
| 10 | **venue 三次转向** | §309 workshop → §360 AAAI-27 main → §383.1 撤出改 REALM workshop 双投。**引用 venue 必带日期**；当前有效 = **§383.1（2026-08-05 REALM，Paper A non-archival + Paper B archival）** |
| 11 | **k=5 vs k=6** | §370/§373 NOTE_06 两轨制 vs §383.1『B2_reddit 落地做 k=6 现在够得着』。**双向承诺已定：B2_red land+bind → 无条件重生 k=6 并作废 k=5** |
| 12 | **AMENDMENT_08 的 task 58** | 依据**弱**（outcome-adjacent），**与 §381 否决『B0 replicate 提交前重跑』援引的是同一类风险**。台账已要求 NOTE 中明记强度差异 + 双臂敏感性 |
| 13 | **PROGRESS 记的"stale 注释"经实测应撤下** | 2026-07-28 实测 `preregistration_decision_test.py`：(a) 第 **34 行明写 `⚠️ REWRITTEN 2026-05-13 (historical):`** —— 35 行的 `PRIMARY GATE = DL` 属于**显式标注为历史的变更日志块**，不是当前 claim；(b) 46-47 行是**成对的**，47 行写 `(Decision 3A specifies FE; advisor lock pending — see banner above.)` —— **忠实记录了 A2 §143.6 那次"实现漂移、不单方面改估计量、等 advisor"的未决状态**；(c) `Makefile:471` 明写 `Replaces retired-DL preregistration_decision_test`，**主链只调 canonical `aggregate_phase1_full_prereg_decision.py`**（mtime 07-27 活跃 vs 旧脚本 05-18 冻结）。⇒ **不是"实现对、注释错"，而是"整个文件已 retired，其注释忠实记录了写下时的状态"。原描述会误导下一个 session 去"修"一个已退役文件里的正确历史记录。** |

---

## 十一、§header 两条（笔记自身的元规则）

- **§header（2026-05-20）**：笔记顶部「行动规划（Next Steps）」节**冻结为 2026-04-29 历史快照，不再维护**；
  **当前 forward action 一律以 `next_steps.md` 为准**。
  理由：该节含 **RunPod 经费 / Myriad #9073 / 22-config 矩阵 / §80 ratio bug** 等已废 TODO，**保留仅作 chronicle**。
- **§header（unspecified）笔记写作规范**：**笔记 = 索引 + 关键结论，不是详细记录**；
  **`[bug]` 3-5 行 / `[finding]` 1 行仅写 digest 指针 / `[literature]` 8-12 行 / `[infra]` 5-10 行 / `[design]` 8-15 行**；
  详细分析放 `docs/analysis/`，文献放 `docs/literature/`。

---

*本文件覆盖 A 批 4 片中的第 4 片（206/831 条）。A1（§5–§119）/ A2（§121–§164）/ A3（§165–§240）已落盘于同目录。*
*A 批四片合计 831 条 = ledger 中 ADJUDICATED 全量。*
