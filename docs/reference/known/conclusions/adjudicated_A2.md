# 裁定层 A2（§121–§164，177 条 ADJUDICATED，2026-05-09 → 2026-05-16）

Claude 主 session 逐条通读产出，2026-07-28。**聚合非转写**：逐条索引见 `ledger.jsonl`。

这一片是 **pre-fire 审计密集期**，七天内发生了四件大事：
统计估计量被替换三次（DL → 单侧优效 → FE → 被反攻）、FP 架构从后处理层整个退役换成上游根因修、
Gemma3-VL 纳入使 scope 从 24/4 变成 36/6 再变 42、mechanism 从"重点"到"整体暂搁"（同日 180° 转向）。
外加 ~200 条 code fix（B-92 ~ B-302）与三 AI /stress 机制成型。

> ⚠️ **跨批核对提示**：A 批看不到 RETRACTED（B 批）与带数字 MEASURED（D1–D4）。标 `⚠️ 待跨批核对` 处，
> 合并阶段须按 § 号回标。

---

## 一、统计估计量 —— 七天内三换一反攻（本片最重要的链）

### 1.1 完整演化链

| 时间 | § | 变更 | 理由 |
|---|---|---|---|
| 05-13 | §132.4 | **K-of-N 从 gate 重分类为 transparency-only consistency check（无阈值）**；PRIMARY gate 改为 **pooled DerSimonian-Laird 随机效应 meta + TOST 等价 δ=1.0pp** | power analysis（pre-data）显示 K-of-N family 在观测到的 1-3pp effect size 下 **power <10%**，只对 **≥7pp** 校准过；N=4 cells 只有 **5 个可能结果状态（0/4~4/4）没有统计杠杆**。**重分类时间戳必须早于数据揭盲**，否则 reviewer 可指控"看到数据后削弱 gate" |
| 05-13 | §132.8 | **TOST 等价检验措辞 → 无歧义的单侧优效检验**（H0: θ ≤ +1.0pp vs H1: θ > +1.0pp）；TOST 保留为 informational secondary 不 gating | TOST 文献里"**拒绝 H0 = 证明等价**"，与 H1 想表达的方向**相反**，原措辞语义方向有歧义 |
| 05-14 | §135.2 | **Decision 3A**：**DL/REML/Hartung-Knapp 全弃 → FE inverse-variance 加权平均** | 4 个 (site,model) cell 是**研究 design 不是 population sample** → estimand = 对这 4 个 planned cell 的 FE 平均，**没有 τ² → 不用 DL/REML/HK**。**k=4 fragility 本质是 τ²-estimation artifact，τ² 不在 estimand 里就没有 fragility**（这直接溶解 /stress v6 的 F1/F2） |
| 05-14 | §135.3 | **H1 简化为单一检验**：单侧 FE 优效（H0: θ_FE ≤ +1.0pp at α=0.05）；原 H1(i) pooled-meta≠0 + magnitude≥1pp **两个子条件被优效检验逻辑蕴含故折叠**；PRIMARY family **m=1**（只有 H1），H2(a) 退出 m-count；TRANSPARENCY 改为报告 n-of-4 计数无阈值 | **estimand-first 原则：一个量是 design-fixed 或近确定性时，不要假装它是 noisy statistical hypothesis** |
| 05-14 | §135.3 | H3(i)/(ii) 同步 DL → FE；R1-R5 framing rule 相应改写；**I²>75% 不再 block FE pooling**（FE 永远 well-defined），而是**把 hook 上限压在 R3** | 保持整套 gate 的 estimand 一致；FE 下 heterogeneity 不影响估计量定义，**只影响可外推性** |
| 05-15 | §143.6 | **⚠️ FE 被 gemini 反攻，升级为 advisor 决策，OSF lock 阻塞** | 见 1.3 |
| 05-16 | §153.5 | `_fe_pool()` 的 **zero-SE floor 从 1e-9 改为 1.0pp**（Agresti-Coull-style） | codex P0-9：SE ≤ 0 时 floor 到 1e-9 会让 **degenerate cell（e.g. P-SoM-only=0）获得 infinite weight 主宰 FE pool**，paper §1 hero claim 基础设施**可被任一 cell hijack** |

### 1.2 §135.2 Decision 3A 的另一半：H2(a) 不做统计检验

**Decision 3A = Decision 2 选 Option 3 + Decision 1 选 Option A**（学生在 2 轮 codex cross-think 后拍板）。
Decision 2 那一半（Option 3）：
**H2(a) 的"P-SoM cost ≈ DOM"是 by-construction 的 token-accounting 属性**
（regex-filtered AXTree 子集 + 无 image token），用 **falsification check** 验证
（**任一 condition 的 median cost ratio > 1.20× 即证伪**），**不是 sampling-theory gate**。
理由：**对近确定性的 token-count 量跑 TOST 是 category error**。
统一根因（两半共享）：**少推断，多 by-design / by-construction**。

**§133.4 配套 scope 收窄**（05-13）：H2(a) cost 是 **PRIMARY gating**；
**H2(b) latency + H2(c) AUROC + H2(d) folded-drop-one 降为 EXPLORATORY transparency report**，
不参与 R1 gating，**也退出 PRIMARY family m-count**。
理由：**(b) 依赖 serving infra，(c) 依赖 signal universe 选择 —— 都不是模型属性，不适合当 gate**。

### 1.3 ⚠️ §143.6 —— FE 被反攻，且发现实现漂移（未决）

**攻击内容**（gemini P0-2，05-15，从 **generalization-claim-coupling** 角度）：
FE 对 6 个 planned cell 的 inverse-variance pool 意味着"**恰好这 6 个 cell 的平均**"，
与 paper §1 "phantom routing space 是**可泛化性质**"的声称**不能同时成立**。

**三选项交 advisor**（**不自行 auto-fix**，因为估计量是 witness-locked 项，改动需 advisor 见证）：
- (a) 保 FE，把 §1 hook 措辞软化为 "characterizes on these 6 cells"
- (b) 回退 **RE + Knapp-Hartung at k=6**（**gemini 推荐**）
- (c) FE + RE 双报 primary + sensitivity

**OSF lock email 阻塞至 advisor 决定。**

**为什么两轮 audit 结论相反**（台账原文）：*"Claude /stress v6 + codex cross-think 当初锁 FE 是聚焦
**k=4 τ² fragility**（Veroniki/IntHout），gemini 冷读 prose 聚焦 **estimand 可解释性** —— 两条互不覆盖"*。

**⚠️ 同时记录的实现漂移**：`aggregate_phantom_meta.py` 与 `preregistration_decision_test.py`
**当前算的是 DL（第三种）**，与 prereg 锁的 FE 不一致。**不单方面改估计量。**

> **待跨批核对**：CLAUDE.md 当前记载 PRIMARY = FE inverse-variance（Decision 3A）+
> DL/HKSJ + TOST 作 Appendix sensitivity，说明 (a) 或 (c) 最终胜出，但**本片未记录裁决**。
> 另 PROGRESS.md 已记 `preregistration_decision_test.py` 注释仍写 "PRIMARY GATE = DerSimonian-Laird"
> 而实现已是 FE-only —— 与本条的"实现漂移"是同一处的**后续状态**。

### 1.4 canonical producer 的诞生（估计量与实现对齐）

**§150a.1/§150a.8（05-16）estimand drift 收口**：
`aggregate_phantom_lift.py` 的 headline（**3→5 lift over 5 arms**）与 prereg §1 line 68-83 锁的
**FE drop-one over 6 cells** **不是同一个估计量** → phantom_lift 语义**降级为 appendix exploratory**，
新建 canonical producer `scripts/analysis/aggregate_phase1_prereg_gate.py`，
接进 Makefile `_aggregate` chain **排在 phantom-lift 之前**。
发现方式：*"codex B2 主动逐行对照 `preregistration.md` ↔ `aggregate_phantom_lift.py` 才发现"*；
定性：*"**prereg 锁的估计量与 headline 跑的估计量不等价，是复现性审计视角最高 ROI 的攻击类**"*。

**§150a.8 B-184 实现规格**（按 preregistration.md line 68-86）：
- per-cell **θ_i** = 6-mode 观察 universe 交集上 `[t ∈ oracle_6] − [t ∈ oracle_5_drop_PSoM]` 的均值 ×100 (pp)
- **SE_i** = paired bootstrap **B=1000**（**prereg 明锁 1000，不是 `analyze_run` 用的 10000**）
- FE pool **w_i = 1/SE_i²**，**θ_FE = Σw_iθ_i/Σw_i**，**SE_FE = sqrt(1/Σw_i)**
- **z = (θ_FE − 1.0pp)/SE_FE**，`p_one_sided < 0.05` 则拒 H0
- graceful degrade **4 态**：PASS / FAIL / PARTIAL_DATA / INSUFFICIENT_DATA

实现纪律：复用 `aggregate_phantom_lift` 的 `_build_cells` + load + `MIN_EP_FOR_CELL` **避免 cell 枚举漂移**；
**自实现 `_norm_cdf`（math.erf）不依赖 scipy**；加 assertion 自检不变量
`oracle_5_no_psom ⊆ oracle_6 → diff ∈ {0,1}`。测试 17 invariants（tests 251 → 268）。

### 1.5 其他统计裁定

- **§132.6 Smoke-gate B7 改为 outcome-independent**（05-13）：门槛 = **10 episode 全部 `logged_in=True`
  且 ≥9/10 产出完整 artifact bundle 且 evaluator 返回可解析裁定**；**success rate 本身不检查**。
  理由：原文"first 10 episodes 必须 ≥1 success"是 **outcome-dependent stopping** ——
  对真实 SR 5-10% 的 cell，**P(前 10 个 0 成功) ≈ 35-60%**，会无效重启 →
  **系统性抬高低 SR cell = reviewer 一击必杀的弹药**。
- **§132.8 `preregistration_decision_test.py` 全重写**（~370 → ~520 行）：
  H1 = per-(site,model) drop-one oracle ceiling lift 做 pooled DL 随机效应 meta + magnitude ≥1.0pp + 单侧优效；
  H3 = 每轴 pooled DL meta（axis-1 `|P-text∖P-SoM|`，axis-2 `|P-prompt∖P-SoM|`）配 paired bootstrap 方差；
  H2(a) = per-cell 中位成本 ±10% 等价，透明度 K_h2=3/4；K-of-N 只进 JSON 透明行不 gating；R1-R5 framing rule mapper。
  修的是 **codex Flaw 3**：*"原脚本测的是 P-SoM ≥ best DOM/SoM/Vision 而不是 drop-one > 0，与 prereg 直接不符"*。
- **§133.4 heterogeneity 分支**（05-13）：`apply_framing_rule()` 加 —— H1 pooled meta **I² > 75%** 时
  跳过 R1-R5 常规路径，改用 **per-cell 方向一致性（≥3/4 方向为正 AND ≥2 个单独 Holm 显著 → R3，否则 R4）**。
  理由：prereg 写了 do-not-pool 规则但 **script 没实现**（T3，Round A + Round C 双确认）。
  > 注：§135.3 后来改为 I²>75% 不 block FE pooling，改为压 hook 上限在 R3。**两条并存，后者覆盖前者的 block 语义。**
- **§134.3 F1+F2 不自行改**（05-14）：DL 在 k=4 时 **τ² 有偏 / Wald CI anti-conservative**
  → REML 或 Paule-Mandel + Hartung-Knapp，**列为 advisor sync 问题**。
  理由：**估计量选择是方法学战略决策，不能单方面改**。（次日由 Decision 3A 的 FE estimand 溶解 F1/F2。）
- **§135.3 prereg scope-strip**：H7/H8 router + §5.X mechanism layer-selection disclosure 加 **DEFERRED banner**
  （逻辑上属 Appendix B/C，不属本 DOI claim）；stale sweep 16-cell→24-condition / H1-H6→H1-H3 gating /
  8→9 lock decisions；新增 **§2.4 4-cell power acknowledgment + §2.5 8 步 H1 PASS/FAIL 决策流**。
  理由：**prereg 必须只锁本次 DOI 要 claim 的内容，否则 reviewer 会拿未做的 H7/H8 打**。

---

## 二、Scope 演化 —— 16 cells → 24/4 → 36/6 → 42 conditions

### 2.1 演化链

| § | 日期 | scope | 触发 |
|---|---|---|---|
| §132.3 | 05-13 | **16-cell phantom-only → Phase 1a = 24 conditions / 4 cells**（2 sites cls+red × 2 models B0+B1 × 6 modes）；**Phase 1b = +12 conditions**（shop × B0+B1 × 6 modes）deferred 到 workshop 投稿之后 | **codex Flaw 1**：queue 只 launch phantom **没 baseline siblings** → drop-one CI 无从比较 |
| §138.2 | 05-14 | **Gemma3-VL 正式纳入 baseline**（不再是 §137 的"IN scope 待定"） | §137 advisor sync 后与学长讨论收口的 5 个 final 决定之一；满足 advisor"顶刊 ≥3 模型"要求 |
| §140.7/§140.9 | 05-14 | **cls/red Phase 1a = 36 conditions / 6 cells**；shop Phase 1b = 18 conditions | 3-baseline 化 orchestration（dry-run 验证 36/6） |
| §159.2 | 05-16 | **B-264 三头案统一为 42 = 36 Pass-1 baseline + 6 Pass-2 router** | yaml header 原写 24 / preregistration §4 原写 36 / phase1_plan §A 写 42。user：*"preregistration.md 没 OSF 公开是 pending 状态，应该是 42"* |

### 2.2 §132.3 的三条 scope 理由（值得单记）

1. **workshop 先占领地盘用窄 R3 hook，main paper 再补 shop replication**
2. **所有现有数据都是 pre-fix 反正要全重跑，信息成本低而 commit 成本高 → commit late collect early**
3. **N=2 sites 本质是 1-bit 观察，"站点类适应性结构"声称需 N≥3 属 main-paper 领域**

### 2.3 术语锁定（§132.5，05-13）

- **condition** = 1 个 (site, model, mode) 操作单元（Phase 1a 当时 N=24）
- **cell** = 1 个 (site, model) 统计分层单元（Phase 1a 当时 N=4）
- **drop-one 在 cell 内用全部 6 modes 算；pooled meta 输入是 4 cells**

触发原因（**codex Flaw 5 的根因**）：*"cell 在 prereg / decision script / advisor sync / OSF manifest 里
有 **4 种不同定义**（16 vs 18 vs 24 混用）"*。

### 2.4 §132.9 codex 6 个 HIGH severity 设计缺陷全修（05-13）

Flaw 1 queue 补 baseline modes / Flaw 2 K-of-N transparency 全传播 / Flaw 3 decision script 改 drop-one+DL+优效 /
Flaw 4 §3.4 P-prompt 回归 / Flaw 5 scope canonical 24/4 / Flaw 6 smoke gate outcome-independent。
queue 改名 `queue_16cell_paper_grade.sh` → `queue_phase1_paper_grade.sh`（git mv 保历史）。
理由：**pre-launch 窗口，这 6 项任何一个 land 到数据里都不可逆**；改名让文件名反映当前 scope 而非 legacy 16-cell。

**§132.7 配套**：paper §3.4 重写 —— **P-prompt 重新纳入为第 4 个 cell**，承诺 6-mode 框架
（此前 §3 写的是 "intentionally excluded"）。理由：**prereg H3 axis-2 明确需要 P-prompt，而 §3 说排除它 → prereg 与 paper 直接冲突**。

### 2.5 §134.2 —— 一次公开撤回（值得记住的诚实校准）

**撤回此前"Phase 1a launch infrastructure workshop-grade ready"的判断 —— 该判断是错的。**
依据：codex 经验性证明当天 fire 会导致 **6 个 cell 复用 pre-fix 数据 + preflight gate 不阻断 + chain 崩溃静默推进**。

**§134.3 由此产生的 13-task fix wave**：C1 加 `FORCE_NEW=1` 到 5 个 queue 脚本并由 queue_phase1 export +
queue_chain 传播；C2 Gate 4 捕获 preflight 退出码并去掉 `--no-strict-ports`；
C3 queue_chain 要求 `condition_summary_v2.json` 存在才推进否则 abort+ntfy；
C4 phantom queue 的 runner stderr 从 `/dev/null` 改写日志；C5 watchdog spawn 失败改 exit 1；
C6 Gate 6 active-run 检测改 fatal（除非 `ALLOW_ACTIVE_RUNS=1`），Gate 5 GPU/CUDA 也改 blocking；
F3 heterogeneity 检查覆盖 H1+H3 双轴；**F7 bootstrap per-call seed = `base_seed + hash((cell_id, stat_name))`**；
F5 两个 DL 实现**合并为单一 canonical**；F6 修 3 处 stale δ=0.5pp 注释与倒置的 TOST 措辞。
洞察原文：*"**codex 的系统层 finding 是 functional fire-blocker，Claude 的统计 finding 是 advisor-sync 弹药；
两个 persona 抓到互不相交的两层**"*。

---

## 三、FP 架构第五次改 —— 从后处理层退役到上游根因（§139.8）

> 承接 A1 §78→§83→§88→§95 四轮。**这一轮把前四轮建立的整个后处理层拆了。**

### 3.1 根因链（本条最重要）

**§139.8（05-14）**：整个 `compute_adjusted_success` **后处理层退役**，换成**上游根因修复**。

根因：`StringEvaluator` 取 `last_action['answer']`，**agent 没真 finish 时**
VWA `run.py:425-427` 与 P79 `runner/main.py:1426-1432` **补 fake stop `answer=''`**
→ 空 pred 进 LLM judge（`llm_fuzzy_match` / `llm_ua_match`）
→ **GPT-4o-mini 把空答案判 correct/same** → FP。
**确定性 string 法对空 pred 正确返回 0，只有两个 LLM judge 路径 FP**；
**`na_fp` 与 `string_match eval_fp` 是同一根因**。

**触发**：user 升维提问 —— *"eval_fp 必要吗，边界过于难确定，能不能 bug fix 而非 post 修"*。
**理由**：修上游比维护后处理边界可扩展。
**文献佐证反而利好 paper**：WebArena-Verified 把 N/A 评测归为 evaluation mechanism issue /
PAE 报 WebArena "~50% 成功"是 evaluator FP / WONDERBREAD 直接过滤 impossible task；**VWA upstream 无修复**。

修复位置：`external/visualwebarena helper_functions.py`（**submodule commit f0c835b, branch p79-patches**）
+ `p79/experiment/analysis.py`。

### 3.2 三项连带退役

**(a) `has_effective_action` 启发式整个移除**（跨 8 文件：`analysis.py` 分支+param+batch caller /
`runner/main.py` 计算与写入 / `types.py` dataclass 字段 / `schema_migrations/v2.py` catalog /
`analyze_cross_representation.py` + `analyze_reason_diagnostics.py` 兄弟重实现 /
`rederive_episode_summary.py` 5-tuple→4-tuple）。
`program_html eval_fp` 分支删除，**污染由 `RESET_BEFORE` 上游协议防**；**B-85 由此 SUPERSEDED**。
理由：`has_effective_action` **只认 type/select_option 不认 click** → click-causal program_html 任务被误降级，
且**启发式无可扩展边界**。

**(b) N/A 任务在 task-load 时排除**（`task.exclude_na_tasks: true` 默认，
`tasks.py::load_tasks + _is_na_task`，`config.py` 加默认值），**移出 primary SR**。
理由：统一 config 规则**无 per-site edge case**；引 WebArena-Verified / WONDERBREAD 先例。

**(c) `adjusted_success` 彻底退役**（4a `scored_task_count` foundation / 4b 非 live `EXPECTED_N` consumer 切换 /
4c 后处理层删除 `compute_adjusted_success` + `compute_adjusted_success_batch` + 相关派生列与 schema 字段 /
4d 下游 reader 清理）；**`success` 成为 canonical outcome 无 override**。
11 文件改动，py_compile ×10 clean，41 pytest pass 含 end-to-end runner+analysis。

**⚠️ 唯一刻意保留的硬编码**：`power_analysis.py` 保留硬编码 **234/210/466** 并加注释。
理由（一字不丢）：*"power_analysis 是 **pre-registered design-time 计算**，committed prereg 的 power 段锁的是
**pre-exclusion design N**，~4% N 缩减的 MDE 位移可忽略，**改了反而 desync prereg**"*。

### 3.3 后续收尾（三条）

- **§141.2（05-15）memory 指针刷新**：新建 memory `reference_fp_architecture_2026-05-14.md`，更新 3 处 MEMORY.md 指针；
  旧 §95/§78a 保留为历史 context 经 Supersedes 链接。
  理由（教训）：*"MEMORY.md 仍指向 §95（2026-04-24）+ §78a 的 post-hoc FP 框架，而 §139.8 已把整层退役 ——
  **memory stale 会直接导致下一个 session 按废弃规则算数**"*。
- **§143.8（05-15）残留隔离判为无需额外修复**（closed as documentation already sufficient）：
  `analysis.py` 已有 §139.8 退役标记，`scripts/maintenance/*` 与 `figures/fig*.py` 都有 inline 注释
  `'§139.8: adjusted_success retired — success is canonical'`，**reviewer/replicator grep 会落在 §139.8 注释而非 stale 调用**。
  理由：**残留已自我披露，再改是无收益扰动**。
- **§158.6（05-16）⚠️ 上条被 user overrule**：FP architecture 走 **hard-delete 而非 selective-retain alias**；
  2026-05-14 §139.8 的 "alias 保留 == raw counterpart 保 output-schema 稳定" 政策**被推翻**。
  理由四条：**字段名 KEY 携带历史语义**（`adjusted_success` 暗示 post-hoc adjusted）**而 VALUE 是 canonical**
  → reviewer 看到两字段恒等会困惑；fig0b 整个 figure 围绕已退役 FP rate（**always 0**）会画全 0 条；
  `_mark_false_positives` docstring 自称 "thin alias-setter" → **cargo-cult perception**；
  **retire 只 sweep code path 不 sweep narrative footprint 会导致 3 处 surface 数字不自洽**。
- **§158.4（05-16）**：删除 `fig0b_fp_rate_per_mode.py` 整个 figure script + Makefile 注释掉该行
  （post-§139.8 FP rate 恒为 0；paper §1/§4 确认无引用）。
- **§155.3（05-16）术语收口**：paper §4 的三层 SR 术语（Raw / Adjusted / Same-task adjusted）
  **收成单一 canonical "VWA-Success (N/A excluded)"**。
  理由：*"'Adjusted SR' 与已退役的 post-hoc FP adjustment **语义碰撞**，reviewer 会误解"*。
- **§150a.2 B-171 配套**：图题 `(adjusted)` 改 `(N/A excluded at task-load)`。

---

## 四、Mechanism 的兴衰 —— §121-§136 高强度推进，§138.3 整体暂搁

> **读这一节前先知道结局**：§138.3（05-14 晚）advisor *"mechanism 部分先不要管了"* —— §5 整个暂搁。
> 以下大部分是**搁置前**的裁定。保留是因为它们记录了"为什么当时那么做"以及**一批仍然有效的方法论教训**。

### 4.1 evidence stack 一度宣告 closed（§123，05-10）

paper §5 **4-corner evidence stack 宣告 closed**：
cross-site replication ✅（cls A+B + reddit F+G）/ **bidirectional symmetry ✅（Welch p=0.535 cls, 0.705 reddit）** /
selection-bias controlled ✅（cls 4-cell 2x2 + reddit 4-cell 2x2）/ content-specific ✅（cls Cell E + reddit Cell E-r）。
> ⚠️ 台账 flag：*"Cell A L11 ρ=-0.851 强 fusion（transfer correlation 证据）+ §123 的 site-asymmetric mechanism framing"*
> 已被 **RETRACTED §124.4**。读 §123 必连 §124.4。

**§121（05-09）hybrid framing**：§117.7 #3 "bidirectional vs asymmetric"二选一被取代 ——
**在 strong-tier 上 bidirectional（Cell A+D 都 Holm-sig），在 reverse-tier 上 direction-locked（Cell C NULL, Cell B ✓）**；
paper §5 最干净的 claim = *"bidirectional mid-layer mechanism on strong-curated tasks"*。
依据：2x2 off-diagonal 揭示 **direction-tier interaction**。

**§122（05-09）content-specificity claim 成立**：Cell E 回答"你的 L17 disruption 会不会只是 patching 破坏模型"
的 reviewer 攻击 —— **不是，random patching 跨所有层均匀破坏且远比 Cell A 的 L17 drop 灾难性**；
paper §5 应报 **real vs random 同 y-axis 对比图**。

### 4.2 三次 framing 重写（v1 → v2 → v3）

- **§128.5（05-13）v2 —— 三层 claim + cosine-causal disjoint hero**：
  (1) linear readability 全 6 mode lototask **AUROC 1.000**
  (2) geometric magnitude 由 image axis 主导（**~0.04-0.07**），text-format 与 prompt-family 是 **sub-permille**
  (3) L11-L17 causal patching **20-30% target displacement**。
  **关键新颖点 = residual-stream cosine 低估 causal influence 数量级**（Wang et al. 2023 IOI "feature encoded ≠ feature used"）。
  理由：**v2 NPZ 推翻 4:3:1 后需要一个不依赖 layer specialization 的 clean 叙事；disjoint 本身比 hierarchy 更 paper-grade-novel**。
- **§130.1（05-13）plan.md 同步重写 v2**：删 v1 三轴 **4:3:1 hierarchy claim**，换成 cosine-causal disjoint hero
  （**cosine 0.5-1% / KL 5-9% / patching 20-30%**）；新 §0 retraction summary 列 v1 invalidated vs v2 preserved。
  理由：Stage 4 v2 NPZ migration 后 v1 几何证据全部作废，**plan.md 必须同步否则成为 stale 污染源**。
- **§124.5（05-10）v3 —— elicitation diversity 而非 fusion locus**：
  phantom space 3 arms 是**同一 Qwen3-VL-4B 的 non-substitutable policy elicitations**；
  H1 Format-axis（AXTree hierarchical vs [SOM_MARKS] flat）触发不同 **training-distribution-induced priors**；
  H2 Prompt-axis（DOM descriptive vs SoM referential）触发不同 **task-conditional decision priors**；
  4 cells = 2x2（training-prior × decision-prior）**elicitation grid**；
  **drop-one routing value 与 Jaccard 来自 policy-level elicitation diversity 而非 representation-level orthogonality**。
  依据（三条 critique 合成）：**mid-layer hidden states 近乎相同（same-prompt cosine 0.998+，AUROC=1.0
  只是 input-encoding trivially separable），divergence 发生在下游计算 policy**。

### 4.3 文献锚点三件（仍有效）

- **§125.6（05-11）**：Wu et al. 2026（*Tool Calling is Linearly Readable and Steerable*）
  **确认为本实验室论文**（Zekun Wu / Ze Wang / Seonglae Cho / Yufei Yang / Adriano Koshiyama /
  Sahan Bulathwela / Maria Perez-Ortiz，UCL + Holistic AI + Imperial），**不构成 scoop**；
  paper.bib key `anon2026toolcalling` → `wu2026toolcalling`。
  timeline audit：2026-04-09 §19 首次 grok → 05-01 §108.19 升 Zoom 4 → 05-02 commit 6662b91 anchored →
  **05-08 arxiv submitted** → 05-09 advisor 录音 explicit 指向 differentiating path → **05-11 arxiv 公开**；
  **已 cite 为 prior art 一个多月，Zekun explicit 邀请 mechanism extension**。
- **§125.7（05-11）**：采用 Ma & Rui 2026（arXiv:2605.07984）的 **probe-vs-causal dissociation** 作 paper §5 组织框架。
  Ma & Rui 实测（**数字来自该论文**）：linear probe 在所有 model family（Qwen3 / Gemma 3 / Llama 3.1, 270M-70B）
  都找到 planning-compatible representation，但 **activation patching 只有 Gemma-3-27B 因果使用该位点**
  （corrupt-rhyme **67% [57,75] at L33**），**Qwen3-32B 1% / Llama-3.1-70B 2%** ——
  *"我们的 Qwen3-VL-4B 落在 Qwen3 family pattern 内（probe 好用，causal patching 弱于 Gemma）"*。
- **§125.8（05-11）paper §5 重构为 3-anchor triangulation**：
  上锚 Wu et al. 2026（text-only, 93% tool switch）→ 我们 Method 4.2/4.4 v2（web agent multi-step,
  50% partial shift, mid-locus readable）→ 下锚 Ma & Rui 2026（Qwen3 newline causal 1%）；
  覆盖 3 类证据（probe geometry / additive steering / replacement patching），
  **定位在两者之间，不被 scoop 而是互补**。

### 4.4 方法学升级（三条仍有效的通用教训）

- **§126.3 主 metric 改 HDMI**：Method 4.4 主 metric 从 raw shift rate 改为
  **HDMI harmonic-mean reliability = 2·c·s/(c+s)**，c=completeness(shift rate)，s=selectivity(JSON valid rate)。
  理由：**raw shift rate 会把 JSON 破掉的 over-steer 报成 winner**（L33 α=10 shift 50% 但 **JSON 仅 25% valid**），
  reviewer 会直接攻；HDMI（Khorasani 2026）是**已发表的标准 metric**。
- **§126.4 采纳 Lin & Liu 2026 的 5-step identification protocol**（causal claim / identification strategy /
  显式列 A1-A5 假设 / stress-test / **validation ≠ identification**）。
  理由：**reviewer 看到 AUROC 1.000 必问 causal 还是 descriptive**；Lin & Liu 给正式 disclosure 模板可直接 preempt。
- **§131.3 train/eval split**（P0-4）：Method 4.4 steering 改为 **16-train / 8-eval**
  （`np.random.default_rng(seed=20260513).permutation`），direction **只在 train rows 上拟合**；
  `--also-report-in-sample` 默认 True 出双列；**generalization gap > 0.10 触发 reviewer-3 flag**。
  理由：*"原本在全部 examples 上 fit direction 再在同样 examples 上评估 → **in-sample 必然虚高，不是 held-out 可迁移性证据**"*。

### 4.5 §132b.3 —— 一条自我降级（最诚实的一条）

**§5.8 的 A5 counter-claim（"0.33 H-mean ceiling 是 in-sample artifact"）判定成立** →
§5.3 降级：**held-out 0.12 作为 paper-grade headline，in-sample 0.29 仅作 reviewer 对照**；
§5 叙事改为 **probe-causal-steering trichotomy**（probe AUROC 1.000 可读 / patching 0.20-0.35 可迁移 /
**单方向 steering 不可迁移**）。

**机制解释（一字不丢）**：*"patching 用 **per-task paired source-target activation（逐任务反事实）所以迁移**，
steering 用 **population mean-difference 方向（mode-mean）所以不迁移** →
**mid-layer 机制真实但是 per-task 编码的，没有单一固定方向能捕捉**；
这让 SAE / LA-HDMI per-input 自适应 steering 成为**有动机的 future work 而不是"修天花板"**"*。

### 4.6 三个污染裁定（数据层）

- **§124.2 CRITICAL bug**（05-10）：`run_stage2b_continuation_pilot.py:273-276` 对**所有 target_mode
  硬编码 `som_marks_text`** 作 text payload → **phantom_prompt target 实为 P-SoM 重复**
  （H-prompt-red 首跑 340265 的 24 task `patched_text` 与 Cell F **byte-identical，md5 验证**）；
  fix = `text_payload_for(mode)` 映射（som/phantom_som/phantom_text → som_marks_text；
  phantom_prompt/dom/phantom_dom → obs_text AXTree；vision → 空）。
  影响范围仅 Stage 3 H-prompt-cls（340263 qdel）+ H-prompt-red（340265 invalid）；
  修复后 sanity check 显示 **9/24 incidental match（chance level 非 byte-identical）**。commit f325ced。
- **§124.4（05-10）**：既有 failure-mode / SR / action-distribution 分析数据判定为 **pre-§116 contaminated**
  （early-stop bug 在），**不能用来验证 H1+H2 framework predictions**；
  涉及 `phantom_dom_vs_som_diagnostic.md §4 failure buckets` / `axis_effect_size_report.md cascade decomposition` /
  `som_vs_phantom_som_diagnostic.md`。
  **Stage 2 patching 原始数据不受影响**（单步 page-snapshot inference 非多步 trajectory 派生）。
- **§136.4/§136.5（05-14）B-82 —— axis-1 confounded**：
  **axis-1（text-format）pair 被判定为 confounded** —— AXTree modes（dom / phantom_prompt）**保留 options**，
  marks modes（som / phantom_som / phantom_text）**丢失** → 混入"人为的一边有 options 文本一边没有"；
  **axis-2 与 image-axis 干净**（options 对称）。
  **所有 v2 NPZ（cls+reddit）+ Method 4.4 sweep + format-variation 输出在 paper-grade 前必须重抽。**
  理由：**共线性使 axis-1 的几何/KL 数字不能归因于 text-format 本身**。
  **根因**：`_extract_text_marks` 只 keep `[数字]` 行故 `[OPTIONS]`（字母）被 drop；
  **生产 builder 有第二遍 `_options_map` 找回而机制版没有** → `text_payload_for` 让 **options-presence 与 text-format 共线**。
  **定性**：B-82 是 **phantom-design 疏忽而非 B-06 回归**（phantom modes 后于 B-06 dropdown fix 设计）。
  **修法**：`p79/experiment/som.py` 新增 canonical **`build_som_text_from_obs_text`**
  （`_extract_text_marks` + `_options_map` 恢复 + `[SOM_MARKS]` wrapper）作 **single source of truth**，
  生产路径 `_build_som_result` refactor 为调用它（**byte-identical lift**），**9 个机制脚本的本地 `build_som_marks` 全部 delegate**。
- **§136.2（05-14）Method 4.4 v2 sweep 判为"半 fix"**：direction 从 v2 NPZ 算（干净）但
  **P-SoM eval baseline 由 `run_stage4_method44_v2_sweep.py:69-71` 的 crude AXTree line-grep `build_som_marks` 生成（脏）**；
  加上 **C1 缺 `[OPTIONS]`** + **C3 α 乘的是未归一化 mean-diff**（**||v|| L11=2.15 → L33=45.35，20× range 跨层剂量不可比**）
  → **Method 4.4 三重污染，pending**。
  理由：*"steering 结论若基于脏 baseline 与不可比剂量，**任何 held-out 数字都不能上 paper**"*。
- **§136.6（05-14）"三轴因果"口径改写**：**2 条干净轴**（axis-2 via cellhprompt + image via §5.4 core）
  + **1 条分解轴**（axis-1 经 H-d / 2×2 分解）；Method 4.2 与 logit lens 的**轴隔离逻辑判为干净，
  但 substrate（SoM text）有 B-82 confound 故 axis-1 findings 需重抽后复核**。
  依据：§136.2 F2 指出 **H-d cells 是 som→dom 同时翻三轴**；F3 指出 **vision mode 是三轴-different，
  干净 image pair 只有 som↔phantom_som**。

### 4.7 §128.4 —— 损坏范围界定（防过度恐慌的好例子）

Bug 1/2/3/5 的损坏范围**界定为仅 Stage 4 NPZ pipeline**（Method 4.2 / Exp 1 / Exp 3 / layer_axis_emergence）；
**Stage 2/3 activation patching + Exp 5 cellhprompt + §1 hero claims（episode summaries）+ §4 empirical tables
均不受影响**。
依据：Stage 2/3 **直读 archive_subset 且用 production `_extract_text_marks`**；
hero/§4 用 episode summaries **不碰 NPZ** → **paper 仍有完整 causal mechanism evidence base，只 §5.7 framing 需重写**。

**Bug 4 + Bug 6 只文档化，数值不动，code rename defer**：
Bug 4（layer indexing off-by-one：patching L0=block0 output vs hidden-state L0=embedding）与
Bug 6（Holm baseline 用 L35 patched）—— **两者经实证数值 unchanged（L35 patched ≈ unpatched）；
改 code 命名有 regression 风险且不影响结论**。

### 4.8 搁置与冻结

- **§138.3（05-14）Mechanism 整个暂搁**：§5（activation patching / layer probe / logit lens / SAE）全部暂搁；
  SAE 决策 / 自训 VLM SAE / 退守 patching **都不再是 critical path**；
  §133/§136 已 land 的 mechanism v2 工作（Stage 4 四方法 / B-82 fix）**冻结存档不进当前 paper scope**。
  依据：advisor discussion 收口 2026-05-14 晚 *"mechanism 部分先不要管了"*
  —— **相对 §137.4 的"SAE 为重点"是 180° 转向**（同日）。
- **§141.10（05-15）"都要修不只 catalog"政策 + framing 纠正**：
  advisor-paused 的工作（mechanism §5）是 **frozen archive 不是 future-scheduled**，**不能假设会 re-activate**；
  正确 framing = **code-level 立即修 + archived data 标记为 legacy**。
  触发：user push back 抓到 B-103 被 catalog 时**隐含了 re-activation 计划**（"SAE future work"措辞）。
  落地：B-103（`_build_user_text` 加 `'Accessibility Tree:\n'` 前缀与 production agent byte-align，修 3 文件）
  + B-104（B-92 propagation gap：**5 个 callsite 仍用 `(self)`/`(None)` 调 @staticmethod，会 TypeError，
  因 mechanism 暂停而 latent**）；加 **2 个 invariant test 含 repo-wide regex CI guard**。
- **§133b.3/§133b.4（05-14）cross-family extractor 系统性修复**：
  `run_stage4_h1_phi35.py` + `run_stage4_h1_qwen2vl.py` —— import Qwen3VLAgent 的 prompt 方法使 `_build_user_text` 按 mode 分支；
  import production `p79.experiment.som._extract_text_marks` 删掉私有 `MARK_LINE_RE`；
  加 `--model-revision` CLI + provenance JSON sidecar；NPZ 改名 `hidden_states_v2_fixed.npz`。
  理由：**Bug 2 / Bug 5 在 cross-family scope 未闭合，P2/P3 cross-family 抽取在修之前 fire 会重复制造污染数据**。
  **剩余两项挂起**：`format_variation` 的 `fmt_som_standard` 仍是 v1-ish（需 **data-altering 重抽**，待决定）；
  `run_stage1_pilot.py` NPZ schema 可能有 gap（老 pipeline，低优先级）。
  理由：**重抽是 compute 成本决策，不属于 code fix 范畴**。
- **§160.3（05-16）`numerical_determinism_check.py` 5 个 bug 全部 defer**
  （D-1 TF32 matmul blindness / D-2 model dtype non-determinism / D-3 external_code path typo /
  D-4 pass_threshold 1e-2 vs docstring 1e-3 / D-5 capture input 未 SHA-pin）。
  理由：advisor 2026-05-14 显式 defer mechanism；paper-1 主体不依赖。
  **⚠️ 但 D-1 + D-2 会 invalidate "max|Δh| < 1e-3 at L11" 的 paper §3 quote，paper-2 mechanism resume 时是 hard gate。**
- **§143.10（05-15，user option α）**：**完全 archive 现有 mechanism artifacts**；paper-2 将用新数据重跑；
  **paper-1 的 OSF DOI 不引用 paper-2 的 forward stub artifacts**；修 8 个 `pre_run/` 文件。
  发现的两个 pattern：(1) **paper-2 mechanism scope 分离不完整** —— 协议/许可文件仍把 mechanism §5 artifacts
  列进 paper-1 release scope；(2) **§139.8 FP 退役 banner 未应用到协议文件** ——
  `reeval_audit_protocol` 与 `evaluator_change_protocol` **仍把 post-hoc 三层 FP ladder 当 live policy**。
- **§127.5（05-12）§4/§5 边界重划**：旧 `section5_mechanism_reddit.md`（reddit behavioral routing analysis）
  移到新 `section4_empirical_findings.md §4.5`，**§5 只留 mechanism interpretability**。
  理由：**behavioral routing analysis 是 empirical finding 不是 mechanism，错位在 §5 会让 reviewer 混淆两类 evidence**。
- **§126.5（05-11）**：mechanism 工作拆出独立 workspace `docs/checkpoints/mechanism/`（README 1-page + plan.md working cockpit），
  **明确 NOT 复制 `paper_planning.md §2`**（后者保 canonical Zoom 1-4 narrative）。
  理由：**三 source 互不复制各司其职**（strategic narrative / working cockpit / final prose / chronicle）**避免 copy 导致 stale**。

---

## 五、advisor 决策（§137–§138，05-14）—— paper-1 的形状定型

| # | § | 裁定 | 依据 |
|---|---|---|---|
| 1 | §137.2 | **venue cascade**：EMNLP 是近期唯一开放窗口但 deadline 很紧 → 并行投 workshop → **先投非 archival venue 占时间戳/展示不烧 archival 提交权** → 之后再冲更高 tier main conference（用户记为 "SL"，推测 MLSys 待确认） | advisor 建议（学长口头转述） |
| 2 | §137.3 | **router un-defer 成为 paper-1 contribution**；**不限定 router 数量**（学长没说一定 1 个）；举例可做 **unique-task routing**（按哪个 phantom arm 独家解哪些 task 来 route，对应 `fig_phantom_structure_venn` 的 4-corner 韦恩图） | advisor 明确。**⚠️ 与当时 CLAUDE.md / preregistration 的 locked framing 直接冲突，需学生 reconcile** |
| 3 | §137.4 | mechanism 部分 **SAE 是重点**，但不一定有 Qwen3-VL 的 pretrained SAE | advisor 判断。**⚠️ 同日晚 §138.3 整体推翻** |
| 4 | §137.5 | **顶刊最少 3 个模型**，可考虑再加一个（例如 Gemma3-VL）；user confirm 模型外扩没问题 → CLAUDE.md 改为"跨模型族外扩 IN scope" | advisor 对顶级会议模型数量的经验要求 |
| 5 | §137.7 | **论文写作交 advisor** —— *"论文写作不用管，如果能跑出来他可以和我一起写"*；学生 focus = **experiment execution producing results**，paper-prose-blocked 工作（§1-§8 codex round / prose 重写 / §1 hook 回退）**全部去优先级归 advisor-side** | advisor 显式优先级信号。重新校准：**学生 critical path 是 SAE 决策 + 第三模型 matrix 而非 paper §1 prose** |
| 6 | §138.4 | **venue cascade final**：EMNLP（用这 11 天冲，ARR 5/25）→ workshop → **NeurIPS**（用户 05-14 确认）；§137 的 "SL"（推测 MLSys）可能被取代或并存 | advisor discussion 收口 |
| 7 | §138.5 | **Phase 1 critical path**：(1) 审查 bug 和 pipeline 先做 (2) cls + red baseline 完全干净 clean run（**现含 Gemma3-VL → condition count 需重算，不再是 24**）(3) **同步做 router 两条基础路线并行** —— (a) rule-based 按 task 属性/任务区分 route，(b) learned 训练 classifier 做 routing；未来扩展按不同 mode 行为模式 route | advisor 收口把 **router 从 contribution-3 / paper-2 deferred 升为 Phase 1 并行核心线** |
| 8 | §138.6 | **bug 部分可单独再发一篇 paper 投 workshop** —— cross-benchmark bug 聚合研究（例如 agisdk）；**这是独立 paper，不是把主 paper 的 workshop 节点替换成 bug 研究**；主 paper venue cascade 不受影响 | 把项目一直在做的 environment / VWA bug fix 工作（dual-track §109 / B-82 等 **37+ bugs**）显性 frame 成独立 workshop-targeted contribution |
| 9 | §138.8 | **Phase 1 paper-grade 跑在 UCL Condenser A100 VM 的独立 dockerized VWA stack（self-host VWA on VM）**，不再用 DGX 经 Tailscale 访问 quark docker；**Gemma3-VL pin = `google/gemma-3-4b-it`** | 解决 A100 不能直连 quark docker 的 VWA-reach blocker；A100 成为自足平台（单租户/无队列/40GB/自带 VWA docker）；**4B 量级对齐 B1 → matched-capability cross-family control 直接回答 reviewer "是否 Qwen-specific"**；**bf16 unquant 装 40GB，对齐 B1 的 `quantization: none` 避免量化 confound** |

**§157.4（05-16）target host 再确认 + 一条教训**：paper-grade target host 明确为 **A100 self-hosted VWA docker**，
不走 DGX→quark Tailscale；CLAUDE.md 运行环境段 + MEMORY.md index + 新 memory `project_paper_grade_target_host.md` **三处 surface**。
触发：user pushback *"paper grade run 应该是在 A100 起 docker"*。
**教训原文**：*"memory `reference_compute_resources.md:24` 早已写（2026-05-14 standing decision）但**没 surface 到高频 loader**
→ 教训是 **memory presence != memory surfaced**"*。

**§143.1（05-15）scope freeze 传播**：把 paper-1 scope freeze（B2 / 36-cond / k=6 / A100 self-host /
FP source-level / router 进 paper-1 / mechanism 进 paper-2）传播到 **6 个 ground-truth 文档**；
25 条 audit-ledger issue 中 **17 closed，8 partial 或 deferred**。
理由：*"**叙事层与实现层必须同步，否则下一轮 audit 会以 code↔prose mismatch 形式再冒出来**"*。

---

## 六、三 baseline 化 —— Gemma3-VL 落地

### 6.1 agent 层（§140.2，05-14）

Gemma3 agent **复用 Qwen agent 的 3 个 observation-mode prompt 方法** + `_wait_for_vram` / `_compute_confidence` /
`_format_history`（**复用 bound method 而非抄字符串**）；Gemma 特有部分隔离
（`Gemma3ForConditionalGeneration` / 单步 `apply_chat_template` 无 `qwen_vl_utils` / **固定 256-token 每图** / B-37 seed 复现）；
`step()` 返回 contract 与 Qwen agent 全等使 **runner model-agnostic**。
理由：**prompt 逐字节一致是 paper-grade 硬要求，复用 bound method 让 drift 结构上不可能**。

**§140.3 一个陷阱**：`local_gemma` backend 的 `revision` 由 `null` 改为 `'main'`（**非 null placeholder**）。
根因：runner 的 **B-83 逻辑会把顶层（Qwen-specific）`model.revision` 转发给任何 `revision is None` 的 backend**
→ **Gemma 会被误灌 Qwen SHA `ebb281ec` 导致加载崩**。

### 6.2 orchestration 层（§140.7/§140.9）

queue 5 个 launch 脚本加 B2 validation；**`queue_chain.sh` collision 从单值 `other_baseline` 改为遍历 B0 B1 B2
跳过自己逐个 check+wait**；`queue_phase1_paper_grade.sh` 三个 chain builder 加 B2 × 6 modes
→ **cls/red Phase 1a = 36 conditions / 6 cells，shop Phase 1b = 18 conditions**（dry-run 验证 36/6）；
`glm_pre_launch_check` 的 hard-rule prompt 从 "B0 XOR B1" 改为 "single baseline of B0/B1/B2"。
理由：**3 baseline 时旧的二值 collision 逻辑永远漏第三个**。

### 6.3 §140.11 三条硬规则（实战踩坑后的最终方案）

1. **装 VWA 包必须 `pip install -e external/visualwebarena/ --no-deps`**，依赖由 p79 `pyproject.toml` 控制
   （**VWA `requirements.txt` 是 2 年前快照，对 modern stack 是 destructive**）
2. **jykoh/classifieds 用 DGX `docker save | ssh | docker load` 流式导入**绕过 containerd extraction，**不本机 pull**；
   且**永远不要 kill 正在解压的 `docker pull`**（留下不可释放的 lease）
3. **P79 cookies 必须在 repo-root `.auth/`**，`auto_login.py` 跑前 cd 到那里或事后 cp 上去

**§140.12 T17 是真正的前置**：A100-local reset wrapper —— **没有 reset 不能做正式 Phase 1a 跑**（条件间 contamination 风险）。
现有 `reset_vwa_sites.sh` 写死 DGX→quark→Windows PowerShell 路径，A100 自托管需平行实现
（**cls 用 curl reset endpoint 实测 0.1s OK，reddit 用 `docker restart`，shop 用 `docker exec magento`**）。
定性：**A100 与 DGX 的 residual 差异里这是唯一 blocking 项**。

---

## 七、政策转变：disclose-only → code-align by default（§142.7）

### 7.1 转变本身

**旧政策**（memory guard rail）：*"B0/B1 设计不对称（解码策略/max_tokens）已知，论文披露即可，代码不改"* —— **退役**。
**新政策**（§142.7，05-15）：**code-align by default，只在上游协议字面无法暴露该字段时才 disclose，
且每轮 advisor 谈判后重新确认**。
落地：**sed 批量把 `max_new_tokens` 384→4096 覆盖 67 个 config（37 B1 + 30 B2）**，
全部 **110 个 `exp_v2_*.yaml` 统一 4096**（B-116）+ 加 **parity invariant test**。
理由：*"学长正在就 logprob 可得性与官方 API channel 谈判 →
**Class 1（部署固有）vs Class 2（历史代码随意）的区分让旧的 disclose-only 立场对 codebase 可控字段站不住**"*。

### 7.2 仍走 disclosure 的（Class 1）

- **§141.2 B-93 / B-94**：B-93（**B0 有效 scroll 词表是二值 ±0.8 而 B1/B2 是连续 delta**）与
  B-94（**B0 无 `input_image_tokens` → "cost ≈ DOM" 的 paper hero 不是 apples-to-apples**）
  判为 **DISCLOSED-only**，写 paper §3.5.1 披露不改代码。
  理由：属**上游协议/部署固有差异而非代码疏忽**。
  > ⚠️ 台账自带注：该 disclose-only 立场在 **§142.7 被部分推翻** —— 对 codebase 可控字段（如 max_new_tokens）改为 code-align by default。
- **§146.5 P1-B3**（decoding override cascade 不对称，只有 B0 honors T/top_p）判为 **disclosure 不修** ——
  B-137 已把三 baseline temperature 统一 **0.0**，**B1/B2 的 `do_sample=False` 是 reproducibility-by-design**；
  写 paper §3.4 + methodology 披露。定性：**不是缺陷是设计选择，但必须显式披露**。
- **§161.2 B-292**：**fsync 14min overhead 只在 paper §3 disclose，不改代码**。
  理由：**B-198 hero（write-time durability）不能动**；改成 disclosure 让 reviewer 无法攻击 latency cost asymmetry。
- **§164（正文 §159.3）P0-7 Meta+A clear-before-type** 判定为 **intentional P79 design**（clear-before-retry 语义），
  wrapper 层 `locator.fill()` 是 canonical 实现；走 paper §3.5 **double-layer disclosure** 而非改代码。
  user 给出的 rationale：*"agent 在编辑时看不到输入框，重复输入会接在上次后面；**backspace 不是全选变蓝的原因，meta-a 才是**"*。
- **§164（正文 §159.4 Phase F3/F7）**：P79 evaluator 与 upstream 的**三处 divergence**
  （judge model gpt-4o-mini / B-91 empty-pred guard / 后来的 polarity fix）走 **disclose 路线** ——
  paper §3.5 声明 **cross-paper SR 不可直接比，只保内部 paired comparison 有效**。
  理由：**upstream evaluator bug 会系统性 inflate SR；保留 buggy upstream 换 comparability 不值**。

### 7.3 parse / GLM 冻结（advisor parking lot）

**§141.5（05-15）**：建立 parse/GLM parking lot（`docs/checkpoints/parse_advisor_pending.md`）作**单一真相源**，
所有 parse 相关代码改动 **hold 到 advisor 回复 B-86**（proxy API 能否原生返回干净结构化 JSON）；
含 **Option A**（退役 GLM 换干净结构化 API）/ **B**（保留 GLM + 下游过滤 + 披露）/ **C**（hybrid）三分支的
code+config+paper 变更清单；**未来 /stress 触及 parse/GLM 的 finding 一律进该文档 §4 而不进 live catalog**。
背景：AWS proxy 不转发 `tool_choice` → GLM-5.1 修补 B0 不稳定 JSON，**污染 B0 SR**。
> 该文件**现已不存在于工作树**。

**§144.4（05-15）5 项进 advisor parking lot**：P0-2 `use_tool_calling` Plan A vs B（等官方 Qwen API channel）/
P0-7 B0 logprob 6 字段 enable（取决于 DashScope OpenAI-format proxy 是否给 logprob）/
P0-9 完整 T=0 reproducibility audit 的预算 / P1-1 **GLM-5.1 scaffold drop policy**（paper-grade 关掉 还是 透明报告 pre-fallback parse rate）/
P1-4 confidence schema mandatory dict contract（与 P0-7 一起解锁）。

**§159.2 B-262**：GLM fallback **不单方面 code-align**，写入 `parse_advisor_pending.md` 等 advisor 决定（Thread 1 / B-86）。

---

## 八、by-decision dissolution —— 本片最有价值的 process pattern

> **核心洞察**（§146.3 原文）：*"**上游决策让下游 bug 自动消失（by-decision dissolution）比逐个 code fix 高效**"*。

### 8.1 §146.3（05-16）user Q&A 让 A1.2 fix scope 大幅收口

1. **GLM fallback 是 advisor-pending 且整个 module 即将退役** → P0-B2 改为 **disable default 而不修 seed**
2. **heuristic 是 Phase 2 router substrate，Phase 1a 不用** → P0-C1 契约 / P1-B5 dom_mode B2 /
   P2-A5+C3 stage_prefix **全部降权到 Phase 2/3 启用前修**
3. Gemma→Qwen 依赖选 **Option A 正式解耦**（DGX 不爆但 **A100 自托管会 ImportError**）
4. planner/grounder `stage_prefix` 三处复制是 **Phase 3 M4 才触发的 dead path**

**§146.3 P0-C2（`model_calls` 硬编码 1）判为 by-decision dissolution 不修代码**：
区分 **network retry**（HTTP 传输层，**只计费一次 tokens**，B-143 已用 `latency_ms.total_minus_retry` 分到 latency 维度）
与 **model-call retry**（GLM fallback，**真 2 次模型推理**）；
**移除 GLM fallback default 后 `model_calls=1` 就是真实值**，模块退役时彻底删。
定性：**user challenge 逼出的区分**。

### 8.2 §147.2（05-16）A1.3 三个 finding by-design dissolved + 一个 demoted

1. **gemini C1 "invisible turn"** —— `_type_needs_enter` 后的 `env.step(NONE)` 是 **obs refresh 不是独立 step**，
   paper §3 的 step 语义是 **per agent action**，**gemini 误读**
2. **Claude F2 + codex B4 "coord normalize"** —— **>1.0 启发式自动判 pixel vs normalized 是 intentional 设计**，
   因为**模型输出的 `coordinate_type` 字段不可信任**，**两个 AI 都违背设计意图**
3. **Claude F3 "inject_options regex"** —— 重审 VWA AXTree `processors.py:531` 行格式
   `[N] role 'name' key: value` **不含 inline `[N]` brackets**，且 line 868 已有 `m.group(1) in injections` 的
   **dict lookup double-guard** → **fire rate ~0**
4. **gemini C2 "select stale coord" 降 P2** —— 笔记 §51/§54/§60/§61/§62 **已 5 轮修过 select_option**，
   **真正的 attack 是 framework-level（所有 element_id action 都用 `_last_obs_nodes_info` snapshot）非 select 特有**

**结论原文**：*"**cross-AI audit 无法替代 user-level design intent review**；by-decision dissolution 比 code fix 更高效"*。

### 8.3 §148.2 / §148.7 更多 dissolve

- **§148.2 D4 / Finding 6 / gemini C6 的 `dom_size_threshold` 跨 mode fragility 攻击 dissolve by empirical fact
  （ratio 1.00-1.02×）**。触发：user 指出 *"obs_text 每个 mode 都是清洗后的 axtree 长度基本一致"*，**实测证实**。
  → 由此确立 **skill v7.5 hard rule**：**任何 length / token / char / byte / payload 数值 attack
  必须 `wc -c` 或 grep JSONL 实测 artifact，不能套 mental model**。
- **§148.2 phantom mode 的 `som_on=False` 判为 by-design 不修**（user confirm: *"phantom 整个基本和 dom 一个 cost"*），
  符合 paper §3 的 "no annotated image" 边界；gemini C2 的 Phase 2 routed metadata "Frankenstein" 问题 **defer 到 paper-2**。
- **§148.7 A1.4a 分类处置**：**dissolved by-design 5 项**（D4 dom_size 跨 mode / D2 phantom som_on /
  Phase 2 Frankenstein 转 paper-2 / C1 escalation monotonicity 属 cost-aware by-design /
  C5 heuristic injection 默认关且有 git review）；**demoted P2 3 项**（task ordering manifest / zero-step race /
  placeholder provenance）；**defer 4 项**（cycle min_reps / schema v2.0 drift / analysis timeout / phase 3 synergy ceiling）；
  **paper-2 scope 1 项**（dom_size 跨 mode 阈值，待 `router_on=True` 时）。
  统计：**user Q&A 作为第 4 层 audit —— 21 个 finding 中 5 个直接 dissolve（24%），2 个 demote，1 个扩大 scope**。
- **§155.2（05-16）三项 defuse**：
  (a) codex OOB-1/OOB-2（locator route 静默 action 替换 / retry stale action record）**降级为 documentation gap 不修代码**
  —— Q1 拆分 defuse：**trajectory ≠ canonical log**；OOB-2 在 Phase 1a 因 **retry flag 默认 OFF 而 latent**
  （attack range 被 `module_flags.m3_failure_trigger_retry`（modules.py:54）AND
  `runtime.baseline_retry_on_no_progress`（config.py:213）**两个 flag 界定**，**paper-1 数据 retry 永不触发**）
  (b) `ChecklistManagerLite` 语义空洞 finding **全 defuse，checklist 留代码备用** —— paper §3/§4 没引用
  `checklist_completion_rate`，只是内部 metric
  (c) **gemini F4（drop-one oracle vs 任务集 noise floor）判定 70% 已被 defuse，不新增 random-arm baseline mandate**
  —— prereg §1 H1 + B-184 已 mandate **paired bootstrap B=1000 seed=42 + FE pool + 一致性超额 1.0pp z-test**
  （preregistration.md:68-91 + aggregate_phase1_prereg_gate.py），**覆盖统计 noise floor**；
  F4 specific random-arm baseline **由 H3 axis-1 P-text drop-in 同方法横向对比 implicit 覆盖**
  > ⚠️ 这条**只 defuse 了 70%**，留 30% 敞口。PROGRESS.md 三条不可违反第 1 条列的 noise 类数字
  > （self_drop 6.7/7.6pp 等）说明**后批仍在处理这个问题**。⚠️ 待跨批核对。
- **§152.5 两条 process pattern 固化**：
  (1) **"0 caller" 是 strong defuse** —— **单 AI 高严重度攻击必须由另外 2 AI 或 spot-check cross-check**，
  否则浪费 fix cycle（codex 的 3 个 strong-defuse **第一次实证产生具体价值，救下 gemini #1/#4 两个假阳性**）
  (2) **telemetry vs aggregation gap** —— **上游 runner 加字段不等于 paper-grade aggregate 可得**，
  未来加 telemetry **必须同时验证 aggregator 路径已接通**
  （A1.4a 花 4 个 commit stamp `trajectory_incomplete`，**A1.4b-ii 才发现 condition 级断链**）
- **§144.1/§144.9（05-15）**：gemini G2 的 **reference_images leak 攻击判为 intentional design 不是 bug**
  （**DOM mode 也含 ref_img → "cost ≈ DOM" 依然公平**），改走 prereg §2.6 disclosure；
  gemini P2-1 的 **P-prompt header 自相矛盾攻击同样判为 intentional manipulation**
  （*"**这个矛盾就是 2x2 实验设计本身，消解它会让 prompt × text-payload 轴塌掉**"*），改走 paper §3.4.1 disclosure。
  与 A1 §139.6 B-88 的判定一致（ref image 是 task-spec 输入）。
- **§157.2 B-236 删除 `QUARK_TZ`**（5 行 delete + 8 行 BUG-6 NOTE 审计注释保留）。
  理由：**client-side export 不影响 docker container TZ**；
  原本"3-AI agree 2026-05-16 fix"是 **cargo cult** —— **A100 self-hosted docker 路径根本不经过 quark Windows host**。

### 8.4 §148.6 —— 一条"先研究再修"的裁定

**about:blank silent attribution（Claude F1 P0）不直接 patch**，改为 **mini-investigation 路径** ——
建 issue 文件 + 写 `scripts/analysis/about_blank_frequency.py` 做**跨 site × mode × baseline 的触发率 /
触发 action 类型分布 / recovery 后 trajectory SR delta 统计**，**4-phase plan**（measure / trigger pattern /
recovery 后果 / handling decision），**当前不改 runner 任何代码**。
user 原话：*"**应该系统研究下**"* —— *"直接 patch 会在没搞清楚现象前锁死处理方式"*。

---

## 九、/stress skill 演化 v1 → v8.1（本片的元工作）

| § | 版本 | 变更 | 触发 |
|---|---|---|---|
| §127.8 (05-12) | **v1→v2** | 从 PRA-10 **checklist runner** 改为 **hostile top-tier reviewer persona**，设 5 个 milestone auto-trigger；**拒绝 "be gentle" 模式** | 用户两次抓到 Claude forward-progress mode 系统性漏 paper-grade gap（axis-2 mechanism / cross-family coverage）；**checklist 太 mechanical** |
| §128.3 (05-12) | **Mode B** | Claude /stress 完成后**自动 chain /codex-stress**；后升级为 **always-chain**（spot-check 也 chain） | 单 AI self-audit 有系统性 blind spot；**不同 model lineage cold-read prior 不同 → 抓不同 angle**；经验值 **5/6 catch rate** |
| §128.4 (05-12) | **lean prompt** | cross-AI audit prompt 一律用 lean 形式（persona + scope + output format），**不列具体 attack vector**；v4 删掉 15 行 attack list | **enumeration 造成 "checked the listed = audit complete" 的 false confidence + list-shaped blind spot**；v1 directive prompt 让 codex 跟着 list 走**漏掉 Bug 2**，v2 lean 自由 explore **抓到 6 个** |
| §130.3 (05-13) | **v5** | persona 从 generic reviewer 改为 **implementer**（自己写过 activation patching / mean-diff steering / logit lens）；**阅读顺序反转为 scripts FIRST prose SECOND**；**≥1/3 attack 必须是 typical reviewer first-read 会 miss 的 OOB**；显式 **NOT-this-skill list**（非 fact-checker / prose editor / checklist runner / citation auditor）；Mode B 加 pre-flight smoke test | v5 前 6 findings 里 **2/6 是 mechanical fact-check** 不是 paper-grade methodology error；**大多数 paper-grade bug 是 code↔prose mismatch，只有读代码才看得见**；实证 **codex exit 0 但 empty output 仍会被当成 cross-AI diff（fake）** |
| §130.4 (05-13) | **-o flag** | codex 调用一律加 `-o/--output-last-message <FILE>`，让 final assistant message **原子写入指定文件，与 stdout 截断解耦**；加 fallback chain | smoke verified 有效；**比缩窄 prompt 更 native 的修法** |
| §133b.2 (05-14) | **v6** | scope calibration 表（spot-check 2-3 files ≤600w / milestone 5-7 ≤1200w / **pre-fire 8-12 ≤2400w** / submission 10-15 uncapped）；**双语 FAIL CHECK**；**sibling-script propagation check 设为 hard rule**（发现 Bug N 后必须 grep 所有用同一 primitive 的兄弟脚本才能宣布 scope 完成）；Mode A→B context handoff；persona rotation；fix-verification mandate；7 天内 retrospective hook | user 凌晨指出 v5 三个 gap（双语 spec 有但未执行 / **pre-fire 用了 spot-check 深度，7 文件只覆盖 ≤30% pipeline** / ≤600 词 cap 不适合 pre-fire），且 **v6 的 sibling grep 30 秒就证实 user 判断** |
| §136.retro (05-14) | **hard gate** | sibling-propagation check 必须设为 **hard gate**（写 finding 前**强制 `grep -rl <primitive>`**） | §136 实证 **Claude 只在 1 个脚本发现 F1，codex 扩到 4 个**（C2）—— v6 已要求 grep 全 sibling 但 Claude 漏做，**说明 soft rule 不够** |
| §143.2 (05-15) | **Bug Table** | 三个 stress skill 的输出 spec 加**必需的 Bug Table**（3 列：Bug / **Blast Radius** / Launch 卡?），按 P0/P1/P2 分组；Blast Radius 必须 **2-4 句人话**（这个 bug 干什么 / 不修就 fire 会具体发生什么 / 哪个下游步骤被污染） | 上一轮 15 个 finding **只有 severity 标签**，user 无法从 "P2 power_analysis.py 16-cell K rules" 判断该现在修还是 defer |
| §144.2 (05-15) | **v7.3** | **严格顺序 hard constraint**：Mode A 完成 → Mode B 完成 → Mode C 完成 → Claude 汇总统一 cross-AI bug list → **呈现给 user** → user 确认 fix scope → **才开始修**；任何 /stress workflow 内的 Edit/Write 前做 3 问 self-check | user 明确抓到 Claude 在 **Mode A+B 完成但 Mode C 未完成时就开始修** —— *"三家 BC gemini codex 都做完了吗? 做完应该先给我呈现 bug list"* |
| §146.2 (05-16) | **v7.4** | Blast Radius 列**强制双语**（中文 prose 描述 cascade + English code reference 如 file:line / 函数名 / 字段名 / 具体数字，**纯中文或纯英文都违反 spec**）；所有 P0+P1 必须带**"推荐修改" section**（具体改法 / Effort / 风险 / 依赖） | user 反馈 bug table 纯英文 cognitive load 高，且缺"你打算怎么改"的 pre-empt |
| §148.2 (05-16) | **v7.5** | 任何 length/token/char/byte/payload 数值 attack **必须实测**（见 8.3） | dom_size ratio 实测 1.00-1.02× |
| §153.6 (05-16) | **re-dispatch** | cross-AI round-1 若 **confirmation-heavy 则立即 re-dispatch**，带显式 `find new attacks not in {list}` 指令 | round 1 codex 75 秒 8 findings 中 **5 个只是 confirm Claude 的 F1/F3/F4**，没独立攻击 design layer；**round 2 4 分钟出 11 findings 3 OOB**（含 P0-7 execution-substrate hole + P0-9 FE hijack） |
| §156.3 (05-16) | **v7.6** | **Q&A 阶段必须列全部不确定项**（无 Q 数量上限 / **不用 AskUserQuestion** / 6-tier grouping / 每项 2-4 options / effort+dep tags / **Phase 1a fire timing 必问** / tracking-commit-push 必问 / post-feedback patch 要 self-contained 3-6 行） | user 显式 set rule：*"最后的 QA 部分不要局限于 4 个问题，所有不确定的都提问，不要用 question 格式要列出来"*；且**裸 bug ID 无上下文 reader-hostile** |
| §152.4 (05-16) | **v8.1** | gemini CLI `-p` 模式 **3 种 failure mode 系统性编目**并建 wrapper（见下） | 见下 |

### 9.1 §152.4 —— gemini CLI 的三种 failure mode（工具层实证）

1. **chatter injection mid-output**（A1.4b-ii 直接 dispatch 时 G1-G3 finding 行被对话 blob 替换，**结构性数据丢失**）
2. **file-write sidestep**（`--yolo` 触发 `write_file`，`--approval-mode plan` 触发 `exit_plan_mode→write`）
3. **JSON envelope double-wrap**

**修法 Path C** = `--approval-mode plan` + `--output-format json` + `jq -r '.response'`
+ **3 条 prompt discipline**（不写 "Write to 路径" / 显式禁 `write_file` / 显式禁 JSON wrap），
落为 `scripts/maintenance/gemini_stress_clean.sh`（149 LOC）+ SKILL v8.1 Prompt Design rule。
过程：*"经 v2→v5 **四轮迭代**才拿到 **4252 bytes 纯 markdown**；把所有 failure mode 实证编目后**一次性建 wrapper，
永久 close 未来所有 /gemini-stress invocation**"*。
> ⚠️ CLAUDE.md 记载 **2026-06-19 gemini CLI 已死**（OAuth tier 退役），Mode C 迁到 Antigravity `agy`。
> 本条的 wrapper 已被 `agy_stress_clean.sh` 取代。

### 9.2 §133.3 —— fixer-bias 规则（跨领域适用）

**确立**：**修 fixer 的工作必须由独立 reviewer 验；不接受"自己抓 flaw → 自己改 → 自己 self-confirm safe"的闭环**。
依据：*"Round 2 commit message 写 'safe to push, advisor-meeting ready'，**Round A 直接证伪** ——
**textbook fixer-bias false positive**"*。

### 9.3 §163.5 —— worktree 隔离（process 实证）

**每个 /stress session 默认起 git worktree 隔离**（symlink 共享 `.venv`/`.auth`/`results`/`logs`），
**B-number 与 § number 用 `git show master:<file>` 取真值**。
实证对比：*"pre-worktree 的 Chunk 1 踩了 **B-# race**（被迫反向 sed renumber）+ **§ race**（§161 撞 → bump §162）
+ **15 个非本人改动混在 working tree**；post-worktree **三项 race 全 0**；
**cognitive overhead 一次性，race friction 持续**"*。

### 9.4 §148.6 —— B-number reservation 不可靠

**B-number reservation 不可靠 —— 落到 `master_bug_catalog` 之后才 canonical**；
并行 session 用 B-162/B-163 后本 session 的 B-162~B-167 经 sed **重映射为 B-164~B-169**
（commit message 历史保留旧编号，in-tree code + tests + catalog 用新编号）。
定性：*"这是 **cross-session 并行工作的预期 pattern 不是 process bug**（第二次出现，前次是 §145 的 B-150/151/152）"*。

### 9.5 §139.retro —— severity 标注纪律

**severity 标注必须区分"实测"vs"推断"，推断出的 P0 要标 `[P0?实测待定]` 并立刻建 gate task**。
依据：**B-84 一度被定 P0 靠 `dom_size_threshold` 推断，实测后只有 0.2% 触发率**；
**B-83 则是真 P0**（codex C8 的 provenance 修复接线断裂 → 名存实亡，**避免 OSF lock 一个假 SHA**）。

**§139.4 配套的诚实校准**：viewport 硬编码（`current_viewport_only=True` 且不进 `condition_meta.json`）
**从 P0 降为 P2 transparency —— 不是 between-mode bug**。
理由：**所有 mode 同源于一份 viewport-filtered AXTree；审计中途把它 bundle 进 P0 是夸大**。

---

## 十、Router v2 → v7 —— 从 archive 锁参数走回 fresh data

### 10.1 演化链

| § | 版本 | 关键变更 |
|---|---|---|
| §149.3 (05-16) | **v2** | 10 项：全 pipeline 用 `success`（不用 adjusted_success）/ **H9/H10 estimand lock = FE pooled meta 镜像 H1, δ=1.0pp** / P1 的 `has_reference_image` 改 runtime `bool(task.reference_images)` / P1 L1/L2 先设 `state.current_mode` 再一律调 stateful `decide()` / P2 从 primary 删掉 F5（仅留 §6 ablation）/ **P2 维度 cap = 用 train-fold mutual info 取 top-18** / P2 加 test-leak-free 约束 / P1 阈值 pre-locked on archive 不做 post-launch tune / **P2 删掉 cross-model 改 cross-site LOSO** / baseline 改 3-tier random（uniform / freq-weighted / top-3） |
| §150b.1/2 (05-16) | **v3** | 修 user 抓到的 **3 个 OOB**（三家 AI 全漏，见 10.2） |
| §150b.3 (05-16) | **prereg §C 三 patch** | C1 §2 PRIMARY family **加 H9 + H10 + 二者 family-wise Holm correction**，estimand = FE inverse-variance pooled lift over 6 cells（镜像 H1 decision 3A），δ=1.0pp 单侧 FE 优效 α=0.05，内嵌 **H10 DEFER 条件**；C2 §4 best-single-mode 行加 **anchor-flicker fallback**（**Kendall τ < 0.7** → 该 cell anchor 换成 **100 次重采样的 majority winner**）；C3 §4 outcome convention 从 adjusted-SR 改为 `success` |
| §150b.4 (05-16) | **δ 决定** | **δ 保持 1.0pp 不上调到 2×SD（=4.4pp）**，并显式披露**低 power（~12-20% per-cell）** |
| §151.3/§151.4 (05-16) | **v4 Option C** | **archive 从 preregistration lock substrate 降级为 correlated-population sanity check**，所有 lock 移到 **Phase 1a fresh data（train folds）** |
| §153.4 (05-16) | **v6 / D1** | **2D (Cost, SR) Pareto 为 primary estimand**，Latency dominance check 为 secondary；**H9/H10 从 SR-superiority 改为 Pareto non-dominance bootstrap**，新增 **H11 cascade non-dominance** |
| §154.2 (05-16) | **v7** | L1 训练改 **repeated stratified 5-fold × 10 repeats（50 train-test pairs per variant）**；**paper §6 主数字锁定在 Phase 1a LOCO**（Leave-One-Cell-Out：train 5 cells / test 1，repeat 6 次） |

### 10.2 §150b.1/2 —— 三个 OOB，三家 AI 全漏

- **P0-8**：P2 删掉 **audit-derived 的 `task.category` 4-way one-hot**，改 **5 个 runtime intent regex 二值**
  （`has_filter` / `sort` / `compare` / `aggregate` / `account_action`）。
  根因：**P79 的 Cat A/B/C/D 是 codex audit 推导出来的不是 VWA-native** ——
  **VWA task 对象只有 `task_id` / `intent` / `image_url` / `start_url`，无语义 category 字段**。
- **P0-9**：Stage 1 候选池加 **model one-hot + `axtree_element_count_step1`**，
  让 mutual-info top-18 能 surface **(model × density × has_ref_image) 三轴交互**。
- **P0-10**：P1 删掉 `has_ref_image` 的 L1 hard rule，改**真正 capability-blind**
  （只留 intent regex `is_search` → DOM + first-step browser-state escalation + L3 stateful escalation）。

**科学收益**：*"P1 与 P2 的科学区分因此变成 **capability-blind handcrafted single-axis baseline
vs learned capability-aware multi-axis classifier**，顺带 defuse 掉 gemini 的 'phantom renaming' 攻击"*。

### 10.3 §151 walk-back 的六条方法论理由（v4 Option C）

三候选中 user 批 **Option C**（A = 在 Phase 1a 上做完全 nested CV，**~5× compute 且超参不稳**；
B = literature-default 锁**完全 ungrounded**）。**六条具体问题**：
1. archive **pre-bug success 漂移**
2. **同 task ID population pattern leak**
3. **B0-only 不能验 B1-B2**
4. **hijack 机制在 B0 invisible 使 P2 设计中心理论 untestable**
5. pre-locked thresholds 的 **Brownlee-Hastie 辩护在 task-ID overlap 下失效**
6. **"raise δ to 2×SD" 统计层就错**

**5 处 retract/reframe 直接落**：(1) §C1 H10 DEFER trigger 源改 Phase 1a fresh-data train-fold entropy
(2) §C2 anchor-flicker trigger 源改 Phase 1a fresh-data train-fold Kendall τ
(3) **§C1 "raise δ to 2×SD" 规则整体 retract**
(4) P1 阈值 framing 从 "pre-locked on archive" 改为 **literature/typical default**
（**VWA/WebArena 文献 8-15K 取 12K；SeeAct/BrowserGym 400-600 取 500**）+ archive sanity-check 确认方向
(5) `archive_diagnostic` 文档顶部加 **SANITY-CHECK ONLY 警告**，verdict 段改 directional confidence。

**§153.1 同源裁定**：`router_archive_diagnostic.py` 的 G1/G2/G4 verdict
**从 preregistration §C lock substrate 降级为 correlated-population sanity-check**。
user OOB catch：*"archive 用与 Phase 1a **相同的 task ID**（cls 0-233 / red 0-209），属 **pattern leak** 而非 independent test"*。

### 10.4 §150b.4 —— δ 不上调的统计理由（防重提）

**SD 是 lift 估计量的 sampling SE，不是 null-distribution 的 noise floor**；
**δ 应该是 minimum-meaningful-effect floor（镜像 H1 δ=1.0pp 的逻辑）—— 两层不同的量不能 conflate**。
> 台账自带注：该 "raise δ to 2×SD" rule **随后在 §151 被整体 retract**。

### 10.5 §153.4 —— paper title 与 estimand 的 false advertising

**D1 Pareto framing 的触发**（codex #1 + gemini #1 **双 AI OOB P0-8**）：
*"paper title **'Cost-Aware Routing'** 对 **SR-only estimand** 属 **false advertising**，
没有 Pareto / cost-utility / Lagrangian"*。

**D2 L2 trigger 收窄**为 **phantom-only verbose（k>=5）+ universal cycle（max_repeat >= 3, url_revisit >= 4）**。
理由：**L2 anchor category error —— full-episode AUROC 与 step-3 partial 不能 transfer**；**k=5 时只有 phantom modes 全 viable**。

**D4 时机裁定**：**立即 amend preregistration + Appendix A，但 OSF DOI commit 推后到 advisor sync 之后**。
理由：**preregistration 尚未公开属 pending 状态，可继续修订；DOI 一旦 commit 就冻结**。

### 10.6 §154.2 —— LOCO 与双层 honest disclosure

**paper §6 主数字锁定在 Phase 1a LOCO**；**archive sim 数字只作 development sanity supplementary table，
双层 honest disclosure**。
理由：**archive 与 Phase 1a 共享 task id namespace = pattern leak**；
**LOCO 在 6 cells × N=200-240 = ~1200-1400 task pool 上才是 paper-grade**。
L1 改 repeated 5-fold 的理由（user Q4）：**cls archive N=234 时 5-fold 每 fold 只有 47 test，
balanced LR + 8 features 下单次 5-fold variance 太大**。

---

## 十一、Code fix 波次（B-92 ~ B-302）—— 按 blast radius 分层

> 本片 land 了约 **200 条 code fix**。逐条见 `master_bug_catalog`。这里按**为什么这一层重要**归类。

### 11.1 orchestrator 层（blast radius 最高）

**§148.3-§148.5（A1.4a Commit G1-G3，6 项）**：
- **B-164** `_get_backend` 的 `dict(backend_cfg)` 改 **`copy.deepcopy`**
  （即使 B-144 的 `(backend_id, seed)` cache key 正确，**nested `generation`/`model_kwargs`/`headers` 仍共享引用
  导致构造副作用跨 seed 污染**）
- **B-165** reward override 加 **`_real_finish` guard**（`parse_valid AND not fallback_finish`）
  使 **keyword-rescue 触发的 fallback finish 不再走 score 0→1 覆写**
- **B-166 `trajectory_incomplete` telemetry**（max-steps timeout 无显式 finish → fake stop + 空 answer
  → string_match SR=0 的**跨 baseline timeout-rate confound**，采 **Path A disclosure**：SR 保持 canonical 另加透明字段）
- **B-167** `invalid_action` 从 5 类细分到 **10 类 + `unknown_failure` 兜底桶**（保留 2-tuple 向后兼容 wrapper）
- **B-168 partial-step crash 恢复**（`_aggregate_partial_steps` 从已写 JSONL 聚合真实 steps/tokens/cost/latency
  **而不是写 `steps=0,total_cost=0` 覆盖**）
- **B-169 resume identity 6 字段校验**（`schema_version, run_id, condition_id, seed, benchmark_site, task_id`）
  **mismatch 则 quarantine 并重跑**

理由原文：*"**orchestrator 层 blast radius 显著高于 backends/envs/agents** ——
silent attribution / fallback override / schema drift **每一个都直接污染 cross-baseline SR 比较**"*。
tests 185 → 191 → 211 → 225。

### 11.2 生产 runtime 路径首次深度审计（§141.2-§141.4，8 项）

**B-92** Gemma prompt parity 改 `@staticmethod`（原为 `Qwen3VLAgent._make_dom_prompt(None)` 的
**bound-method-on-None 模式**）；**B-95** `image_utils` 加 `over_cap` 字段并复用循环尾 b64 不重复编码；
**B-96** factory 去掉 `cfg.get('type','local_qwen')` **静默 fallback 改 ValueError**；
**B-97** `HeuristicDomBackend` 加 `(backend_id, config)` 构造；**B-98** MockBackend scroll delta 0.5 统一为 0.8；
**B-99** `schema_version` identity 统一（v2 vs 2.0）；**B-100** locator dispatch 4 个函数改 **try/finally 保证 handle dispose**；
**B-101** walk-fail error 标准化为 `walk_fail:<category>` 前缀。
定性：*"这些是**生产 runtime 路径（agents/backends/envs）首次做 code-review 深度审计**，
§139 审的是 data pipeline + analysis **不同面**"*。

**§141.9 B-102**（重要）：修 unknown observation mode 的**两层静默 fallback**
（`som.py prepare_observation_for_mode` 的 `if mode != 'som'` + 3 个 baseline 的 `_system_prompts.get(mode, dom_default)`）
→ 加 **`KNOWN_OBSERVATION_MODES` frozenset 在两层都 strict raise**。
理由：**config typo 如 `'phantum_som'` 会静默跑 DOM-like obs + DOM prompt，paper-grade SR 数据可能含 silent wrong-mode 行**。

**§141.11 A1.3+A1.4 backlog sweep**：B-105 `WALK_UP_MAX_DEPTH=6` 提为 named constant 注入 3 个 JS resolver /
B-106 `_JS_RESOLVE_CLICK` 接受无 href 但有 onclick 的 `<a>` / B-107 ARIA accept list 扩 8 类 /
B-108 `_options_map` look-ahead 去掉 2 行窗口只留 next-mark-id 边界 /
B-109 `_collect_bbox_map` 加 visited set + depth cap 50 / B-110 `apply_som` 发 DeprecationWarning。
**F1 cross-family `_options_map` skip 与 F5 `_FONT_CACHE` 判为 intentional design 写入 header 注释。**
理由：**把跨文件隐式契约（`vwa_wrapper` 注入距离 ↔ `som.py` 恢复距离）显式化**。

### 11.3 跨文件 silent contract（codex 抓到 Claude 全 missed 的两条）

**§142.2 B-111~B-115**：
- **B-111** B0 复用 Qwen3 staticmethod 使**三 baseline prompt byte-identical**，
  并**删掉 Qwen3 prompt 里的 shopping 领域先验**（e.g. `'Electronics'`, `'Jewelry & Watches'`）
- **B-112** 加 `image_meta: Optional[Dict]` 到 `StepRecordV2` 并由 runner 提升 agent meta
  （**此前 archived B0 30K steps 完全无法 audit image encode 行为**）
- **B-113** 加 `image_encode_error` 计数器（**B0 encode 失败静默走 text-only 而 B1/B2 raise → 跨 baseline missingness 不对称**）
- **B-114** 删 `parse_action_text` 的 scroll/back 关键词 fallback
  （**codex 用 archived 数据确认这些 step `action_success=True` 属 silent partial automation**）
- **B-115** runner 顶层 `cfg.model.revision` 转发**按 backend type gate**

理由：*"codex 抓到两条 Claude 完全 missed 的 **cross-file silent contract**（B-112 / B-115 = **B-83 类 bug 在 runner 层再现**）"*。

**§144.3/§144.7 §144 Commit A+B（13 项）**：B-131 runner `_QWEN_CLASS_BACKEND_TYPES` 收窄到 `{local_qwen}`
**关掉 B0 假 revision pin** / B-132 multi-seed condition_id schema drift / B-133 B1+B2 image-encode try-except 与 B0 对齐 /
B-134 runner 保存 `validate_action` 布尔并 emit `runner_invalid_action` / **B-135 B0 max_new_tokens 默认 512→4096** /
B-136 revision strict mode 无硬编码 fallback / **B-137 base yaml 三 baseline temperature 统一 0.0** /
B-138 T=0 greedy consistency probe 脚本 / B-139 B2 `image_token_count_method` enum 进 meta /
B-140 `image_meta` 强制 dict + pipeline label / **B-141 parser 改 `raw_decode` 扫描 + fenced json 优先** /
B-142 `validate_action` 坐标与 delta 形状范围检查 / **B-143 `network_retry_count` + `total_minus_retry` 分离**。
（tests 21 → 25 → 57）

### 11.4 执行层（Phase 1a clean run 会实际命中）

**§147.3 A1.3 Commit F（6 项 P1）**：**B-156** locator-route telemetry 进 step_record（`StepRecordV2.locator_route_meta`）/
**B-157** locator-route click 前 capture tab 数，成功后**检测新 tab 并切换 + `bring_to_front`**
（mirror VWA `browser_env/actions.py:1417-1421`）/ **B-158** dialog handler 从 page-level 提到 **context-level**
让 listener 跟随 context lifetime / **B-159** `_lazy_init`/`reset` 检测到 running loop 时 **raise RuntimeError 带可操作提示**
（替换 silent passthrough）/ **B-160** `navigate_to` 的 URL 用 `json.dumps` 转义 /
**B-161 shadow DOM `elementFromPoint` 穿透**（`_pierceElementFromPoint` 递归下钻 depth ≤5 +
walk-up 经 `getRootNode().host` 跨 shadow 边界）。
理由：*"这些是 **Phase 1a clean run 会实际命中的执行层缺陷**（新 tab / dialog / shadow DOM）"*。tests 177 → 185。

**§146.4 §146（10 项）**：B-144 runner `_backends` cache key 改 `(backend_id, int(seed))` tuple /
**B-145 `use_glm_fallback` 默认 true→false** 且 GLM 块加 **MARKED FOR FULL RETIRE + DeprecationWarning** /
**B-146 新建 `p79/agents/_shared_vl_utils.py`** 抽出 6 个跨 baseline 函数让 **Gemma 不再经 Qwen 拉 `qwen_vl_utils`**
（Qwen classmethods 保留为 backward-compat delegate）/ B-147 `api_proxy` max_new_tokens default 512→4096 /
**B-148 `api_key_env` allowlist 拒绝非白名单值防 yaml-injection** / B-149 mock action 对齐 /
B-153 stale 注释 / B-154 `MockBackend.backend_type = mock_<backend_id>` / **B-155 `pillow>=10.0,<12.0` pin + import-time assert**。

### 11.5 analysis 层（canonical producer，"每条都是 reviewer-3 陷阱"）

**§150a.2 A1.4b-i（14 项）**：B-170 merge key 加 `benchmark_site` / B-171 图题改 `(N/A excluded at task-load)` /
**B-172 Wilcoxon N<5 skip 写 CSV row 带 `skipped_reason`** / B-173 Pareto tie 文档化 /
B-174 `_to_mapping` 静默 JSON parse 失败改 warn + 收集到 `parse_failures.csv` /
**B-175 TOST sig 列改名 `equiv_within_1pp` 并加脚注 "NOT evidence of positive lift"** /
B-176 bootstrap **seed=42 + B=10000** inline 注释 / B-177 obs_prepare 成本 /
**B-178 Holm-Bonferroni 按 (test, metric) 子族分组** / B-179 `_synthesize_condition_summary` 改 delegate canonical aggregator
+ partial 用 hatch 标记 / **B-180 `read_jsonl_dedup` 加 optional 6 字段 identity + steps 基数校验** /
B-181 mixed-phase fail-closed / **B-182 `aggregate_phantom_meta` 加 `family_scope` + `gating_status` 列
明示 RE 是 `APPENDIX_RE_SENSITIVITY`** / B-183 P95 直方图 caption 改 per-episode 非 per-step。
理由：*"`analysis.py` 是 **paper §3 evidence-layer 的 canonical producer**，每条都是 reviewer-3 陷阱"*。tests 225 → 251。

**§152.2 A1.4b-ii（14 项）**：B-187 删 `compute_energy_step`（**0 caller 且 YAML key 错**）/
B-188 删 `compute_waste_breakdown`（**0 caller 且违反数学不变量**）+ 删 `compute_wasted_cost` 的死参数 /
**B-189 paper §3.5 prose 加 seed=42 + B=1000 + 3 个 producer 名** / B-190 加 `STEP_RECORD_V2_DEFAULTS` 对称 catalog 12 字段 /
B-191 `schema_migrations.migrate` 改 deepcopy / B-192 把 `fill_defaults` 接进 `_collect_episode_summaries` /
**B-193 `trajectory_incomplete` + `unknown_failure_reasons` + `partial_recovery_step_count` 进 `EpisodeSummaryV2`
与 `aggregate_condition_metrics`** 使 paper §3.5 transparency metric 可产 /
**B-194 异常路径 `wasted_cost_usd` 用从 partial JSONL 恢复的 total_cost 不再强制 0** / B-195 obs_prepare 成本字段注明 /
B-196 JSONL integrity report emit / **B-197 cost=0 时 `cost_efficiency_ratio` 返 None** /
**B-198 `logger_v2` 三个 callsite 加 directory fsync** /
**B-199 `detect_benchmark_noise` 加 `api_rate_limit` 与 `auth_expired_or_session_invalid` 两类** + 分布 emit /
B-200 p95 显式过滤 None 与 NaN。
user Q&A 定调：**死代码直接删，历史归档不 backfill rederive**（`fill_defaults` 用 baseline 默认值 +
paper prose 加一句 disclaimer 即可）。tests 268 → 292。

### 11.6 config / 一致性

- **§159.2 B-261** `phantom_dom` **全面改名 `phantom_text`**（删 legacy alias + `conditions.py` enum 显式 raise）；
  **`phantom_dom` 只允许存在于 archive 结果**。
  触发：user 指令"全部修复 phantom dom 为 phantom text"；codex 发现
  **`phantom_text` + `resume:true` legacy alias × episode reuse 是攻击面**。（19 yaml obs_mode + queue scripts）
- **§159.3 B-271**：B1/B2 的 **40 个 per-yaml 加 `temperature: 0.0` pin**（post max_new_tokens）。
  理由：与 B0 的 paper-grade audit parity —— **解码参数必须显式可审计**。
- **§159.3 B-270**：`base.yaml` 的 `local_4b` / `local_gemma` 加 **`min_free_vram_gb: 12` 默认**，
  **删除 76 处 per-yaml 的 `: 0` override**。理由：**safety default —— 0 等于关掉 VRAM 保护**。
- **§161.2**：`SCHEMA_VERSION` 全部统一为 **semver `'2.0'`**，`_CHAIN = ['2.0']`。
  理由：`'2.0'` vs `'v2'` 双写会触发 **B-275 latent v3 migration bug**。
- **§156.4 B-213**：`log_cleanup.cleanup_all` 默认改 **`dry_run=True`** + 加 `confirmed: bool` kwarg 安全门，CLI 加 `--confirm`。
  理由：**破坏性默认值**。
- **§158.4**：`power_analysis.py:147-180` 的 **K-of-16 正文段整删**（K=11..14 of 16 表 + K_h1/K_h3 解释 +
  K_h1=12/16 family-wise rule claim）。理由：**header L5-11 已声明 retired-status，正文与 header 矛盾**。

---

## 十二、Infra / process 裁定（跨领域可复用）

- **§124.10 done-monitor 的 post-loop bash 只能 stat-only，不嵌 python one-liner 做 compute/parsing**。
  实证：monitor `bh702x73i` **exit 127** —— curl ntfy message 里嵌 python 计算 Δ，
  **nested 引号在 `bash -c` eval string 里 syntax error**；until-loop **21:36 已完成但 auto_pull 22:50 才 fire**
  （5min cron lag + queue→pull lag），最终**手工 22:55 算出结果**。
  → 后写入 CLAUDE.md 长任务 done-monitor 规则。
- **§124.3 auto_pull 加 Phase 0 done-sentinel check**（commit 3b80a3a）：
  **GONE_HOOK 必须 gate 在逻辑完成**（Myriad 侧存在 `pilot_summary.md` 或 `condition_summary_v2.json` 或
  `hidden_states.npz`）**而非仅 qstat 消失**；bypass = `P79_SKIP_SENTINEL=1`。
  实证两次：**21:23 job 344630 crash 后 sentinel 拦下避免污染**（SKIP + low-priority ntfy，local dir 未动）；
  而 **13:31 cellhp_cls 340263 qdel 时 Phase 0 尚未上线，cron 13:45 拉了 12-task partial buggy data 进本地目录**（后手工清理重跑）。
- **§128.2 经 Windows PowerShell 中转的 SSH chain 不可信 returncode，必须用 stdout sentinel（`__QSTAT_OK__`）+ double-probe guard**。
  根因：**内层 ssh（quark→myriad）silent-fail 时 outer ssh 仍 returncode=0 且 stdout 空**
  → `parse_qstat('')` 返回 `{}` 写进 `state.json` → **整个 job 生命周期 invisible**，Exp 5 两个 cell 的 GONE_HOOK 完全没触发。
- **§129.1 watcher GONE_HOOKS dispatch 改 sorted longest-prefix-first + exact-or-boundary match**
  （`full_name[len(prefix)] == '_'`）。
  根因：**dict insertion-order + `name.startswith(prefix)` 让 `cellhprm_red` 这个短 prefix 吞掉 `cellhprm_red_tsh` 事件**
  → **数据被 dispatch 到错误 remote_dir 并覆盖原 359512 本地数据**。
- **§162.5 删除 `glm_pre_launch_check.py`（159 LOC）**，换成 `launch.sh` 里的 **deterministic shell assert**。
  理由：**5 条 hard rule 里 4 条已被 `queue_chain.sh` / `queue_baseline.sh` 上游 deterministic enforce**；
  唯一 glm-unique 的 config↔site benchmark match 用 grep 即可；
  **GLM dependency = LLM variance + API outage + non-deterministic gate，反而新增攻击面**
  （P1-3 BLOCK collapse / P1-10 regex / P1-11 fail-open）。
- **§130.2 移除 cron 里的真实 GLM API 调用**（`glm-refresh-playbook-s2` 30min + `glm-refresh-playbook` 2h），
  保留 **4 个确定性 cron**（cell-autoupdate / error-scan / check-links / myriad-watcher）；PLAYBOOK 刷新改 on-demand。
- **§129.6 TODO 系统定为 hybrid**：**TaskCreate 只用于 in-session burst**；
  `_status/issues/*.md` + Bases 承载 **paper-grade cross-session TODO**；`next_steps.md` 日/周 horizon；
  `paper_planning.md` decision log；实验笔记 过去 chronicle。
  触发：**36 个 Claude TaskCreate panel 条目不可 scan**。
- **§162.4 / §163.3 Option K Trajectory Event Log**：
  P1-5-B（cell interrupt → relaunch+reset 污染）采用 **α' = Tier 1 分析层 stack + Option K Trajectory Event Log（~11h）**，
  同时覆盖 auth-loss/auto-clear class。
  user cross-talk 洞察：*"**auth-loss/auto-clear 与 P1-5-B 是同 bug class 两个 layer 的 mid-trajectory perturbation**
  （JSONL 与 site state 不一致方向相反但同类）"*；Tier 1 分析层 fix **自然 generalize**（加 `had_auth_clear` covariate）。
  **schema**：`condition_dir/trajectory_events.jsonl` append-only JSONL，每行 `{event_type, task_index, wallclock_ts, metadata}`，
  `event_type ∈ {reset_post_interrupt, task_auto_cleared, auth_refresh_no_clear, runner_restart, watchdog_intervention}`。
  理由：**统一 schema 让 reset-perturbation 与 auth-loss 两类事件在同一 covariate 管线里被 paper §4 GLMM 吸收**。
- **§157.3 Q4 选 shared lib 而非 inline copy**：新建 `scripts/queues/_lib_paper_grade_gates.sh`（174 LOC / 5 functions），
  4 个 queue script 全 source。理由：**sibling-propagation 治本** —— Bug 2 + B-224 + 未来任何新 gate 不再有 sibling-drift 风险。
- **§162.2 B-302 sentinel** 改用 `episodes` 字段 + per-site hardcode `expected_n`（**cls=234/red=210/shop=466/wa_***）
  + **90% completion 阈值**。3 case smoke 通过（valid 234→exit 0，ep=0→exit 3，partial 100/234=43%→exit 4）。
  > 台账自带注：**§183 后来把 90% 阈值改成 100% exact**。⚠️ 待跨批核对（A4）。
- **§160.2 `snapshot_vwa.sh` 的 site probe 从 `/` 改为静态资源 `/robots.txt`** + `VWA_PROBE_PATH_*` 覆盖 + `-b /dev/null` 关 cookie。
  理由：**`/` 是 session-stateful，hash 每次不同 → paper §3 Appendix D 的 byte-equivalence claim 站不住**。
- **§124.7 4×4 evidence pipeline 补齐**：Tier 1 修 Makefile 对已删 `fig_capability_b0_b1.py` 的残留引用
  （line 265+404，**否则 `make analysis` 在新数据上 crash**）+ 去重 `_figures`/`figures` target；
  Tier 2 新增 4 个 paper §1 figure（fig0a SR heatmap / fig0b FP rate / fig0b_extra confidence calibration /
  fig3b image token gap，**全带 empty-data placeholder**）；
  Tier 3 新增 failure-mode pipeline（`aggregate_failure_modes.py` 把细粒度 `reason_buckets` 映射到 **5-bucket paper taxonomy**）。
  `_aggregate` 链 **12 步**，`_figures` 链 **26 图**，**4×4 覆盖 18/20 sub-cells（88%）**；
  剩 **1d Action vocabulary + 2a-extra Click-target divergence 为 JSON-only defer 到 supplement**。
- **§129.5 overnight defuse wave**：4 修（`fig_meta_forest.py:56` 与 `fig_forest_drop_one.py:39` 的
  **`TOST_DELTA_PP` 0.5→1.0**；`validate_run.py:48-66` 加 4 个 `phantom_*` condition ID + mode mapping；
  queue Gate 1b 加 `status: locked` grep）+ **2 defer**（`glm_pre_launch_check` BLOCK exit 码需 design；
  manifest cell promotion 需 human judgment）。
  理由：**TOST δ 与 prereg 不一致会直接改变 gate 结论**；**draft prereg 能 launch paper-grade 是 audit-trail 致命伤**。

---

## 十三、fire timing 与 launch gate

- **§157.7 Q7=B —— fix 完不能立刻 fire**：`preregistration status draft→locked` 是 **hard gate**，
  必须等 advisor 确认 **A2.3 + v7 walk-back + Pareto reformulation**。依据：`phase1_plan §B1` 的 pre-reg lock checkbox 仍 unchecked。
- **§159.2 Q11 fire timing = 所有 P0+P1 全修再 fire（禁止 fire-and-fix-parallel）**。user 显式指令；单批次 14 个 P0+P1 一起落地。
- **§160.6 Q15 = A1.16 的 provenance fix 不 block Phase 1a launch**。
  理由：**paper §3 Appendix D 是 reviewer-facing post-launch artifact，snapshot 在 Pass-1 launch time 跑即可**；
  B-276 `--strict` mode 已实现，何时 wire 进 queue 由用户决定。
- **§159.2 B-266 T0 pilot 放弃** —— 删 3 个 pilot yaml + `queue_pilot_t0.sh` 加 deprecation banner 并 **exit 2**。
  user *"pilot 放弃"*；codex 另发现 **pilot 缺 local seed pin 且仍带 GLM**。
- **§157.6 三项 defer**：P1-1 master orchestrator supervisor（**operational annoyance 非 data-corrupting**
  且 supervisor design 需 ≥1h 讨论 —— cls 挂了是否 abort red / 重启 vs abort）/ P1-5 PID capture race（**zombie restart 边缘场景**）/
  P2-1 Gate 1 prereg grep brittleness（**substring 可被 case/typo 绕过但 user 不会故意绕**，paper-grade lock 风险低）。
- **§156.5 B-227**：evaluator-patch SBOM **不进 main paper Appendix D**，redirect 到独立 bug 研究 paper
  （agisdk-style cross-benchmark bug aggregation，workshop-targeted）。
  理由：**main paper Appendix burden 转 paper-2 substrate，paper-1 不肥大，companion paper credit 独立**；
  companion paper 同时容纳 B-91 / B-01/02/33 / B-90 / B-209 / B-229 parse error taxonomy +
  历史 magento auth / scroll direction / in_viewport_ratio 优先级。
- **§164（正文 §159.4 Phase F2）P0-4 networkidle 等待**改为 `ObservationHandler.get_observation` 里的**单一 barrier**
  （放在 text + image processor 之前），移除 `ImageObservationProcessor` 内部 wait。
  理由：**3-AI overlap = 最高 paper-grade 信心**（Claude code-audit + gemini design-layer + codex code-deepening 三独立角度）；
  原实现造成 **text/image 观测时序不对称**。
- **§164（正文 §159.4）B-26 viewport bug 状态从 NO_FIX 改为 FIXED 2026-04-19**（commit 3f9ceca）；
  paper §4.X.5 重写为 **FIXED + 0.6 threshold 有意义 + 所有 DOM/SoM 已在 fix 后重跑**；
  **§1 hero claim 不再依赖 viewport bug 作 confound source**。
  理由：**实际代码状态与 prose 不符，prose 必须跟上**。

---

## 十四、paper prose 相关（多数 defer）

- **§127.9（05-12）§1 prose 改为 lead with drop-one CI [+0.95,+6.19] strict-positive**；
  head-to-head **显式标 "competitive parity within 2σ"，不 claim exceeds**。
  理由：**/stress W1 预测正确 —— single-mode CI 跨 0**；把 hedge **正式化为 statistical disclosure 而不是含糊措辞**。
- **§133.6（05-13）Tier 4 paper prose drift 项一律 defer 到 Phase 1a 数据 land 之后**：
  §4 仍 5-mode 无 P-prompt / §1 hook 用旧 archived B0-only 数据 / §8 写 "three sites" + **错模型名 Qwen3-Omni-235B** /
  §1:13 引用不存在的 Section 6/7 / paper.bib 缺 DerSimonian-Laird + Higgins-Thompson + Zhang title 漏 "Towards"。
  理由：**prose 方向依赖数据结果和 advisor 对 scope 的签字，现在改会白改**。
- **§135.5（05-14）B0 模型名以 config 为 ground truth = `qwen.qwen3-vl-235b-a22b`**；
  cards 里写的 **`Qwen3-Omni-235B-Thinking` 是错的**；全 sweep 修正 6 个文件
  （preregistration §7 三处 / model_card header+name+modality / locked_versions 两处 /
  ethics_license_coi_statements / topvenue_constraints 三处 / section8_limitations）。
  注：**Qwen3-VL 是 vision-language 无 audio，modality 行也要改**；`codex_prompts/` 里剩 2 处是历史 prompt 输入留 as-is。
- **§144.9（05-15）paper §3.4.1 P-prompt 段改写 + 新增 Disclosure 段做统计前置承诺**：
  (i) **prompt-text ID-space disagreement 是 manipulated variable 不是 confound**
  （删掉原来暗示 graceful-degradation 期望的**事后辩护语气**）
  (ii) **action-parse-success rate 升为 first-class metric**
  (iii) **不把 P-prompt SR collapse 解释为 against H3**
  (iv) **structural claim 锚在 `|P-prompt ∖ P-SoM|` 而非 raw P-prompt SR**
  理由：**防 post-hoc framework switching —— 让 claim 对 collapse 与 partial-success 两种 outcome 都 robust**。
- **§135.4（05-14）pre_run/ 去冗余**：`pre_rerun_audit.md`（678 行 stale aggregator，标题还写 "16-Cell"，
  引用已删的 `PAPER_STRATEGY_OPEN_QUESTIONS.md`）**瘦身为 ~55 行 pointer index**；
  `osf_lock_manifest §3` ↔ `preregistration §6(b)` 改为 **reference 不重列**。
  **根因诊断（可复用）**：*"冗余根因是**文档分层错位** —— spec 类与 derived 类混放，
  **派生视图 copy 了 spec 内容 → spec 一改就 stale**；正确做法是 **derived 视图 reference 而非 copy**"*。
- **§150a.4 B-185/B-186 保持 deferred**（各 0.5-1 day）：B-185（`claim_manifest.json`：paper-claim → producer + input SHA 映射）
  与 B-186（`hero_metrics.json`：4-fold drop-in 单 JSON 收集），**等 A1.4b-ii data plane audit 之后再做**
  （**新增 schema 字段可能影响设计**）；**历史归档数据不 regenerate**。
  user Q&A：*"历史归档不需要处理 + make analysis 看看还缺什么"* → **把做不完的 3 个 new artifact 放进 issue tracker
  而不是死扛，省 6-9h**。
- **§131.1（05-13）27-finding pipeline audit** 结论：**5 个 P0 必须在下一次数据抽取之前修**。
  触发：**user pivot —— 停止追 prose，先审 pipeline**。（`pipeline_audit_2026-05-13.md` ~10K LOC，commit 301b28e）

---

## 十五、⚠️ 本片矛盾与待核清单（合并阶段用）

| # | 事项 | 两侧 / 需核 |
|---|---|---|
| 1 | **FE vs RE 估计量（最重要）** | §135.2 Decision 3A 锁 FE（advisor-witness-locked）vs §143.6 gemini 反攻 + advisor 决策 pending + **实现漂移（脚本当时算 DL）**。**本片无裁决**。⚠️ CLAUDE.md 现记 FE 为 primary，但裁决 § 号需在 A3/A4 找 |
| 2 | **§137.4 SAE 是重点 vs §138.3 整体暂搁** | **同日 180° 转向**（05-14 白天 advisor 说 SAE 重点，当晚收口说 mechanism 先不要管）。台账已标 `named by RETRACTED §138.3` |
| 3 | **disclose-only vs code-align** | §141.2（B-93/B-94 disclose-only）vs §142.7（政策退役，改 code-align by default）。**台账自带注：部分推翻**。判据 = Class 1 部署固有 vs Class 2 历史代码随意 |
| 4 | **§143.8 残留隔离"无需修" vs §158.6 hard-delete** | §143.8 判"documentation already sufficient"，§158.6 **user overrule** 改 hard-delete。后者胜 |
| 5 | **§162.2 90% completion 阈值** | 台账自带注：**§183 后来改成 100% exact**。⚠️ 待 A4 核 |
| 6 | **"raise δ to 2×SD" 规则** | §150b.4 决定"不上调"并给统计理由 → §151 **整体 retract 该 rule**。两者方向一致（都不采纳），但 §151 是把 rule 本身删掉 |
| 7 | **§155.2 gemini F4 只 defuse 70%** | 留 30% 敞口（drop-one oracle vs 任务集 noise floor）。PROGRESS.md 三条不可违反第 1 条列的 noise 数字说明**后批仍在处理**。⚠️ 待跨批核对 |
| 8 | **scope 数字三头案** | §132.3 (24/4) → §140.9 (36/6) → §159.2 (42 = 36+6)。**引用任何一个必须带 § 号和日期**，否则会撞 CLAUDE.md 的 "42 conditions / 6 cells" |
| 9 | **§133.4 I²>75% 分支 vs §135.3 I² 不再 block** | §133.4 加 heterogeneity 分支跳过 R1-R5；§135.3 改为"FE 下 I² 不 block pooling，只压 hook 上限在 R3"。**后者覆盖前者的 block 语义**，但 R3 上限逻辑保留 |
| 10 | **B-number 编号历史** | §145 的 B-150/151/152 与 §148.6 的 B-164~B-169 都是**并行 session 冲突后重映射**；**commit message 里是旧号，catalog 里是新号**。查 bug 时以 catalog 为准 |
| 11 | **消失的文件** | `parse_advisor_pending.md`（§141.5/§144.4/§159.2 三处引用）/ `issue_decision_3a_fe_re_review_2026-05-15.md`（§143.6）/ `docs/checkpoints/hermes_prompts/playbook_refresh.md`（§130.2）—— **均已不在工作树**。引用这些 § 的证据链须注明 |
| 12 | **§136.2 Method 4.4 "三重污染 pending"** | 状态是 **pending 未解决**，随 §138.3 mechanism 暂搁一起冻结。**若 paper-2 resume，这是第一个要处理的** |
| 13 | **§160.3 numerical_determinism D-1/D-2** | defer 但**会 invalidate paper §3 的 "max|Δh| < 1e-3 at L11" quote**；**paper-2 resume 时是 hard gate** |
| 14 | **gemini CLI wrapper 已过时** | §152.4 的 `gemini_stress_clean.sh` 基于 `gemini -p`；CLAUDE.md 记 **2026-06-19 gemini CLI 已死**，改用 `agy` + `agy_stress_clean.sh` |

---

*本文件覆盖 A 批 4 片中的第 2 片（177/831 条）。A1（§5–§119）已落盘；A3（§165–§240）/ A4（§241–§397）见同目录。*
