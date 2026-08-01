---
type: conclusions
batch: D2
status: done
created: 2026-07-28
source: scratchpad/batches/D2.jsonl (219 条 MEASURED, §128.6–§207.6)
scope_note: 2026-05-12 ~ 2026-05-17，router 设计 v1→v7 + §A1 pre-fire 审计密集期
---

# 测量结论 D2 (§128.6–§207.6)

> **本文件是聚合层，不是转写层。** 数字一律原样抄自台账，未做任何算术、换算、平均、合并。
> 台账原文里自带的口径冲突保留在「⚠️ 矛盾清单」，不选边。
> 所有 `[聚合者推论]` 标记的句子是本次聚合新加的判断，不是台账内容。

---

## 1. Stage 4 mechanism v2 NPZ 重算 — axis-2 / axis-1 / logit lens

**当前值**（截至 §136，此后 mechanism 整体随 advisor decision 暂搁）:
- axis-2 layer profile: 全层单调升到 L36，量级 **0.005-0.009**（cls + reddit, B1）
- axis-2 per-task fragility: **IQR [0.0024, 0.0040] tight, 0% outlier**（cls + reddit, B1, 24 task）
- Exp 3 logit lens lm_head amplification (axis-2): **8-10x**，cosine 0.009 → KL 0.09 at L25，双 site 一致（of-means 口径）
- per-task/of-means KL ratio: axis-2 **1.10-1.82×**；reddit axis-1 **3.25-3.95×**
- per-task paired layer profile 的 peak: **cls + reddit 双 site 一律 L36**
- 干净 image pair cosine: **som↔phantom_som ≈ 0.04**
- cls task-shuffle 对照: **peak 从 L14 mid 移到 L0 boundary** → 验证 content-specificity（定性位移，未给显著性检验）

**演变**:
- §129.2 v2 NPZ 重算 → 确认 §5.7 retraction；per-task fragility 从 v1 的 IQR [0.012, 0.014] 降到 [0.0024, 0.0040]（小一个数量级）
- §129.2 报 of-means logit lens 8-10x → §130.5 OOB1 指出是 of-means 而非 per-task → §131.2 per-task 重算给出 ratio
- §131.2 推翻 plan.md v2 §1.3 prose 声称的「cls L36 / reddit L23 发散」，判定那是 **of-means artifact**，per-task 下双 site 收敛 L36
- §136.2 F3 修正 image-axis: 原声称 image axis 0.04-0.07，实为「靠 Vision pairs 撑起来的」

**已作废**:
- v1 的 per-task fragility IQR [0.012, 0.014] — 「v1 数字作废」（§129.2）
- §5.5 v1 的 **text-format dichotomy INVALIDATED**，v2 下是 image-side mode 决定 peak（定性结论，原文未给数值）
- plan.md v2 §1.3 的 cls L36 / reddit L23 发散叙述
- image axis 上界 0.07 的纯 image 轴解读 —「该修正意味着 image-axis 的上界 0.07 不是纯 image 轴; 本 chunk 内未重算」

**caveats**（逐条搬）:
- 「v2 NPZ 仍是 options-less builder 产 (§136 B-82) — axis-1 相关结论需再抽, axis-2 不受影响」
- reddit axis-1 3.25-3.95×: 「§136.4 判定 reddit axis-1 是 B-82 ([OPTIONS] 缺失) 最可疑受害者 (reddit 81% observation 含 dropdown), 该 3.95× surprise 需重抽后复核」，台账已标 `superseded_by: §136`
- axis-2 ratio 接近 1 → 「of-means 表述属 terminology 问题可修辞面修正」
- §5.5 结论「定性结论, 原文未给数值」
- task-shuffle「定性 peak 位移, 未给显著性检验」
- vision mode = vision-prompt + 空 text + image，属**三轴-different**，不能当 image pair

**证据**: §129.2 / §130.5 / §131.2 / §136.2 / §136.4；`docs/checkpoints/mechanism/results/axis2_layer_profile.md`、`axis2_per_task_fragility.md`、`axis2_logit_lens.md`、`axis2_logit_lens_per_task_2026-05-13.md`、`docs/checkpoints/paper_drafts/section5_mechanism.md`；Myriad 359768 (task-shuffle cell)

**原文片段**: 「axis-2 全层单调升到 L36, 量级 0.005-0.009」；「量级比 v1 (IQR [0.012,0.014]) 小一个数量级 — v1 数字作废」；「cls + reddit 双 site 一律 peak L36 (不是 plan.md v2 §1.3 prose 说的 cls L36 / reddit L23 发散 — 那是 of-means artifact)」；「干净 image pair 只有 som↔phantom_som ≈ 0.04, 原来声称的 image axis 0.04-0.07 是靠 Vision pairs 撑起来的」

---

## 2. Method 4.4 activation-patching sweep（三重污染，整条线不可用）

**当前值**:
- Held-out (8 tasks): best cell **L33 α=20, H-mean 0.12, completeness 7% (1/15 paired cells), selectivity 100%, non-zero cells 1/30**
- In-sample (16 tasks): best cell **L11 α=20, H-mean 0.29, completeness 17% (5/30), selectivity 100%, non-zero 9/30**
- generalization gap = **+0.16**
- split（seed=20260513 deterministic permutation, classifieds）: train `[1, 19, 33, 40, 60, 69, 82, 99, 108, 109, 122, 161, 181, 214, 215, 228]`；eval `[9, 20, 32, 37, 61, 73, 116, 227]`

**演变**: §131.3 落 split → §132b.2 出 sweep 数字 → §136.2 F1 + §136.6 判定整条线三重污染

**已作废**: 台账未直接宣告作废，但记为「Method 4.4 整条线三重污染」（下方 caveats）

**caveats**:
- 「eval n=8 极小样本」/「eval n=8 极小」
- 「gap +0.16 超 reviewer-3 可迁移性阈值 0.10」
- 「best peak 从 L11 (in-sample) 迁到 L33 (held-out) 是 peak layer 不一致而不只是量级衰减」
- 「§136.2 F1 + §136.6 指出该 sweep 的 P-SoM eval baseline 由 buggy builder 生成 (半 fix) 且 α 未归一化 → Method 4.4 整条线三重污染」

**证据**: §131.3、§132b.2；`results/mechanistic/stage4_multimode_b1_cls/method44_v2_sweep.json`；`scripts/mechanistic/run_stage4_method44_v2_sweep.py`；Myriad 366792 (V100-PCIE 32GB, 1h26min)

**原文片段**: 「Held-out (8 tasks): best cell L33 α=20, H-mean 0.12, completeness 7% (1/15 paired cells), selectivity 100%, non-zero cells 1/30; In-sample (16 tasks): best cell L11 α=20, H-mean 0.29 ... generalization gap = +0.16」

---

## 3. off-the-shelf VLM SAE 可得性

**当前值**: Qwen 官方 2026-05-01 开源 **Qwen-Scope (arXiv:2605.11887)**，只覆盖 text-only Qwen3/3.5 语言模型（Qwen3-1.7B/8B/30B-A3B + Qwen3.5-2B/9B/27B/35B-A3B，共 **14 组 SAE**，训练数据是 text-only residual-stream activations）；**无任何 VL 变体**，连项目用的 4B/235B 尺寸都没有 → off-the-shelf Qwen3-VL SAE 不存在

**演变**: §137.4 调研核实

**已作废**: 台账标注该条被 RETRACTED §138.3 引用 —「mechanism 部分 SAE 是重点 (§137.4 方向)」这一**方向**已 retract，需在「自训 VLM SAE / 退守 patching / text-domain SAE」三选项中决策

**caveats**: 「来源是 web search 未走 arXiv API 核验 (项目 memory 规则要求 curl arxiv API 确认 id 存在)」

**证据**: §137.4（web search 2026-05-14）

**原文片段**: 「只覆盖 text-only Qwen3/3.5 语言模型 ... 无任何 VL 变体, 连项目用的 4B/235B 尺寸都没有」

---

## 4. 统计门槛之一：K-of-N 的 power 与 power_analysis.md 的 scope 失配

**当前值**（design-time，非实测）:
- N=4 cells × per-cell power ≈ **0.30 at 1.5pp effect** → **P(≥3 of 4 显著) ≈ 8%**
- K-of-N family power 在 **1-3pp effect 下 <10%**，**只对 ≥7pp 校准**

**演变**:
- §132.4 给出上述 design-time 计算（4-cell scope）
- §133.2 codex C-M2: `power_analysis.md` 当时仍写 **K_h1=12/16 + N≥10 + shop pooled N=910**，与 Phase 1a 4-cell scope 完全 mismatch
- §143 的 **B-126** 才把 `power_analysis.py` 改为 **k=6**

**已作废**: `power_analysis.md` 的 K_h1=12/16 / N≥10 / shop pooled N=910 三项（与实际 scope 不匹配）

**caveats**:
- 「design-time 计算不是经验 power」/「Phase 1a 设计期 (4 cells), 非实测数据」
- §132.4 的 scope 是 **4 cells**，而 B-126 之后代码是 **k=6** —— 两个 N 不是同一时期的同一 scope

**证据**: §132.4、§133.2、§143；`docs/analysis/cross_sites/power_analysis.md`

**原文片段**: 「N=4 cells × per-cell power ≈ 0.30 at 1.5pp effect → P(≥3 of 4 显著) ≈ 8%; K-of-N family power 在 1-3pp effect 下 <10%, 只对 ≥7pp 校准」；「文档仍写 K_h1=12/16 + N≥10 + shop pooled N=910, 与 Phase 1a 4-cell scope 完全 mismatch」

---

## 5. 统计门槛之二：FE pooling 的 SE floor

**当前值**: **const 1.0pp**（经验校准值），FE pool zero-SE floor 从 1e-9 改为 1pp

**校准依据**（archive 3 cells 的 P-SoM drop-one lift 与 SE）:
- B0 cls **+2.56pp, SE 0.981pp**
- B0 red **+3.33pp, SE 1.096pp**
- B1 cls **+1.71pp, SE 0.766pp**
- **median SE ≈ 0.98pp**；3-cell 样本中 **0 个 degenerate cell**

**演变**:
- floor 原为 **1e-9** → §166.4 改 **1pp**，对应单元测试期望值 **2.0 → 2.8**（1pp floor 下 weights=[1, 4], θ_FE = 14/5）
- §172.4 用 archive 3 cell 数据回填经验校准依据
- §177 gemini 对 SE floor 提 de-weighting 攻击（数学算错，见 §32 主题），但仍逼出 prose 扩写（Agresti-Coull anchor + archive median + sensitivity table）；user 以「psom unique 应该不会是 0 吧」的实证直觉选 **option A retain**

**已作废**: 1e-9 floor（测试原本锁的是 stale 的 1e-9 floor）

**caveats**:
- 「只有 3 cells, 小样本; 用于论证 const 1.0pp floor 是经验校准而非任意值」
- 本批次**不含** §209 的 power 48.3% / 81% / 97% 数字（§207.6 之后），不要把它们接到这里当同一口径

**证据**: §166.4、§172.4、§177；`results/phantom_paper/meta_phantom_lift.csv`；`scripts/analysis/aggregate_phase1_prereg_gate.py`

**原文片段**: 「B0 cls +2.56pp SE 0.981pp; B0 red +3.33pp SE 1.096pp; B1 cls +1.71pp SE 0.766pp; median SE ≈ 0.98pp; 3-cell 样本中 0 个 degenerate cell」；「test_fe_pool_handles_zero_se_via_floor 期望值 2.0 → 2.8 (1pp floor 下 weights=[1, 4], θ_FE = 14/5)」

---

## 6. 统计门槛之三：δ=1.0pp 单侧优效阈值的 archive 敏感性 + RE 估计量对比

**当前值**:
- δ=1.0pp 单侧优效检验的 **LOO p ≈ 0.044-0.046（压线）**；**P-text LOO-fragile**；archive **P-text I²=71%**
- **HKSJ [0.34, 4.33] vs DL-Wald [1.30, 3.37]**（archive 3 cells, P-SoM arm）

**演变**: §135.1/§135.3 出 LOO 敏感性 → §151 该 archive **被降级为 sanity check，不再作为 lock substrate**（台账标 `superseded_by: §151`）；§172.8 补 HKSJ/DL-Wald CI 对比

**已作废**: 用 bug-fix-pre archive 作 preregistration lock substrate 的做法；DL-Wald 被标为 **legacy descriptive**

**caveats**:
- 「archive 是 pre-fix 数据; p 值压在 0.05 边缘意味着 δ=1.0pp 是地板不是余量」
- 「HKSJ 更宽符合预期; DL-Wald 被标为 legacy descriptive; k=6 时 HK-Wald 反保守 (codex P1-3)」

**证据**: §135.1/§135.3、§151、§172.8；`results/phantom_paper/meta_phantom_lift.csv`；`scripts/analysis/aggregate_phantom_meta.py`

**原文片段**: 「P-text LOO-fragile; δ=1.0pp 优效检验的 LOO p ≈ 0.044-0.046 (压线); archive P-text I²=71%」；「HKSJ [0.34, 4.33] vs DL-Wald [1.30, 3.37]」

---

## 7. 统计门槛之四：bootstrap 重采样次数 B 与 Holm family

**当前值**（本批次内三处并存，见矛盾清单 ⚠️C1）:
- paper §3.5 披露的 canonical: **seed=42 + B=1000**，并列出 3 个 producer 名
- analyze_run 内部使用 **B=10000**
- prereg 明锁 **B=1000** 用于 prereg gate（§150a.8 B-184）
- Holm-Bonferroni 的 (test, metric) family 是 **GLOBAL → m=132**，**within-cell 过校正约 10x**

**演变**: §152.2 记录「两处并存不可混用」→ §165.2 gemini 抓 paper §1:7 prose 写 B=10000 与 prereg/code 的 B=1000 不一致 → §186 codex F3 再报 analysis.py「bootstrap 代码 B=10000 vs prose B=1000」

**已作废**: 台账未宣告任一侧作废

**caveats**:
- 「注意与 analyze_run 内部使用的 B=10000 不同 — prereg 明锁 B=1000 用于 prereg gate (§150a.8 B-184), 两处并存不可混用」
- §165.2 被判定为 **submission-gate 级（fabricated precision by 10x resamples）**；「Claude + codex 都 missed」
- §186 的 bootstrap 还有第二重不匹配:「observed-n 与 prose 声称的 paired 双重不匹配」

**证据**: §152.2、§165.2、§186；`docs/checkpoints/paper_drafts/section3_definition.md`、`section1_intro.md`、`p79/experiment/analysis.py`

**原文片段**: 「seed=42 + B=1000, 并列出 3 个 producer 名」；「paper §1:7 prose 写 B=10000, preregistration + code 是 B=1000」；「bootstrap 代码 B=10000 vs prose B=1000 且 observed-n 与 prose 声称的 paired 双重不匹配; Holm-Bonferroni 的 (test, metric) family 是 GLOBAL → m=132, within-cell 过校正约 10x」

---

## 8. Prereg decision producer：与 prereg 文本的 lock breach + 合成 smoke

**当前值**:
- **B-513**: `evaluate_h1` 仍是 **3-test compound (DL + magnitude + superiority)**，而 prereg §1 L68-86 锁的是 **SINGLE FE superiority test**
- **B-514**: `_effective_gate_pass` 有 **heterogeneity-rescue 分支**能用 per-cell consistency 救回失败的 pooled gate，而 prereg L323 明写高 I² 不阻止 pooling 只 cap the hook，L340-342 明写 **p ≥ 0.05 → H1 FAILS → R5**
- 新 canonical producer 合成 smoke: `r1_pass → R1 STRONGEST (FE + I²=29%)`；`r5_fail → R5 (H1 FE FAIL = paper death)`；`heterogeneity_test I²=94% 只把 R1 cap 成 R3`（符合 prereg L323 cap-only 语义）
- 种子分离: `--data-seed 42` 固定下 `--bootstrap-seed 43 vs 44` 给 **θ_FE 4.11 vs 4.14**

**演变**:
- §132.8/§133.5 早期 4 场景合成 smoke: `r1_pass → R1 STRONGEST (H1 pooled drop-one 4.65pp, 优效 p<0.001, H2 4/4, H3 双轴 PASS)`；`r3_pass → R3 MODERATE (drop-one 5.25pp, H3 双轴 FAIL)`；`r5_fail → R5 (drop-one 0pp)`；`heterogeneity_test → R1 (合成数据 I²=0% 没触发 >75% 分支)`
- §177 A1.21 抓两处 lock breach，并给出新 canonical producer 的 3 场景 smoke（heterogeneity 分支这次被触发到 I²=94%）
- §166.4: prereg gate 测试曾整体 fail —— fixture 缺 `schema_version` → B-283 strict-load 的 lenient mode 把每 cell 100 个 episode 全当 corrupt skip → **θ=0 → 5/17 test fail → `make pre-launch-check` exit != 0 → launch 卡死**；加 1 个 `'schema_version': '2.0'` 字段解决 5 个 test

**已作废**: §132.8/§133.5 的 heterogeneity 分支「未被合成数据触发」状态（§177 已触发）

**caveats**:
- 「全部是 synthetic generator 产的数据, 只验证路由逻辑不构成实证; heterogeneity 分支未被合成数据触发 (逻辑正确但未 end-to-end 验证)」（§132.8/§133.5）
- 「合成数据, 非真实 Phase 1a 数据」（§177）
- 「后果是 reviewer 跑广告里的 decision script 会与 B-184 canonical producer 拿到不同 H1 verdict」

**证据**: §132.8/§133.5、§166.4、§177；`scripts/analysis/preregistration_decision_test.py`、`aggregate_phase1_full_prereg_decision.py`、`tests/test_phase1_prereg_gate.py`

**原文片段**: 「B-513: evaluate_h1 仍是 3-test compound (DL + magnitude + superiority), 而 prereg §1 L68-86 锁的是 SINGLE FE superiority test」；「heterogeneity_test I²=94% 只把 R1 cap 成 R3 (符合 prereg L323 cap-only 语义)」

---

## 9. 分母 canonical：scored_task_count 与 N/A 排除

**当前值**:
- **scored_task_count canonical = cls 224 / red 205 / shop 435**（N/A task 已排除）
- 全 6 站点值: **cls 224 / red 205 / shop 435 / wa 173 / 176 / 104**
- N/A 任务普查: **73 个 N/A 任务全是 string_match + fuzzy_match:'N/A'**（cls 10 / red 5 / shop 31 / wa-shop 19 / wa-admin 6 / wa-red 2），**占 1390 的 5.3%**
- VWA task 总数/id 范围: 三站 min=0, max={233, 209, 465}, unique={234, 210, 466}

**演变**:
- 旧硬编码 `EXPECTED_N` = **234/210/466** → §139.8/§158.5 替换为 **224/205/435**
- §158.2 发现 `scored_task_count==0` 的静默 fallback 链: `run_registry.py:152` 的 `is_complete = actual_n >= expected_n` 在 `0>=0` 时为 True → missing cell 被标 complete；`fig1ab_cascade_diamond.py:146-153` 的 `if n >= min(200, expected)` 在 expected=0 时 n=0 也 complete
- §158.5 起 `strict=True` 时缺文件 **raise FileNotFoundError 而非静默 0**
- §173b 发现 figure 层仍硬编码 **N=234/210**（fig0c / fig0d / fig0e / fig0f 共 4 个），修法是 `expected_n_canonical()` helper

**已作废**: 234/210/466 作为分析层分母

**caveats**:
- 「power_analysis.py 刻意保留旧硬编码以与 committed prereg 的 design-time N 同步」——即 234/210/466 在 power_analysis.py 里是**故意保留**的，不是漏改
- 「是一条统一 config 规则, 无 per-site edge case → 支持在 task-load 时统一排除」
- 「codex 追踪的 silent-fallback 链, 是 paper-grade gate 假阳性 root cause」

**证据**: §139.8、§150a.1、§158.2、§158.5、§173b；`p79/experiment/analysis.py`、`p79/experiment/tasks.py`、`scripts/analysis/lib/run_registry.py`、`scripts/analysis/figures/lib/panels.py`

**原文片段**: 「cls = 224 / red = 205 / shop = 435」；「替代硬编码 EXPECTED_N (234/210/466); power_analysis.py 刻意保留旧硬编码以与 committed prereg 的 design-time N 同步」；「4 个 figure 硬编码 N=234/210, canonical scored_task_count 是 224/205」

---

## 10. task_id 跨站碰撞

**当前值**: 三站 task_id 在 **[0,209] 区间完全 overlap**（cls 0-233 / red 0-209 / shop 0-465）

**演变**:
- §150a.1 实测 → 支撑 **B-170**: `merge(on='task_id')` 跨站碰撞必须改成 `on=['benchmark_site','task_id']`
- §151 同一事实升级为「**archive ≠ independent test**」的核心论据（archive 与 Phase 1a 用同一批 task ID → pattern leak）
- §186 codex F4 P1 OOB: analysis.py 跨 site 热图 `pivot index='task_id'` 时 cls 0-209 与 red 0-209 重合，修法是按 site 分文件出图

**已作废**: 任何以裸 `task_id` 做跨站 join / pivot 的分析产物

**caveats**: 「该实测支撑 B-170 ...; 同一事实随后在 §151 成为 archive ≠ independent test 的核心论据」

**证据**: §150a.1、§151、§186；`p79/experiment/analysis.py`

**原文片段**: 「三站 min=0, max={233, 209, 465}, unique={234, 210, 466}; 三站 task_id 在 [0,209] 区间完全 overlap」

---

## 11. Router 设计 v1→v7 的实测演化

**当前值**（本批次末态）:
- **P1 规则路由数值阈值在 archive 上全部 dead**: `dom_size > 12000` 与 `dom_complexity > 500` 两条 fire **0%**；cls DOM 最大 **5512 chars**，red 最大 **6963 chars** → **P1 v3 ≡ always_phantom_som on archive（决策树退化）**
- **真实 run 的 numeric trigger fire rate**（B1 / classifieds / dom router run, 3237 steps）: `dom_size_exceeds_threshold 7 次 = 0.2162%`；`dom_complexity_high 0 次 = 0.0%`；`text_length_high 6 次 = 0.1854%`；`__streak_or_action_failed__ 822 次 = 25.3939%`；`max_numeric_trigger_rate 0.2162% < 0.5% 阈值 → disclosure_consistent = true`
- **L1 learned router (repeated stratified 5-fold ×10)**: cls **17.84%, Δ +2.02pp, CI [16.30, 19.42]** vs always_phantom_som 15.81%；red **-3.95pp, CI lower bound 9.05 < 14.29**
- **L2 partial-trajectory AUROC 可用率**: k=3 时 **8/12 cells AUROC < 0.65**；k=5 viable **6/11 cells**；phantom modes k=5 **全 viable (5/5)**，baseline modes 仅 **1-3/6**
- **archive diagnostic verdict**: B0 cls G-1 label entropy **0.734**（P2 viable, barely）/ G-2 Kendall τ **0.696**（anchor NOT stable, 临界）/ G-4 噪声 SD **2.23pp** / anchor = **som**；B0 reddit G-1 entropy **0.606**（< log(2)=0.693 → **P2 NOT viable**）/ G-2 τ **0.841**（anchor stable）/ G-4 SD **2.17pp** / anchor = **phantom_som**（majority winner 58.4%, phantom_text 38.0% 第二）/ **86% task (181/210) argmax 在 DOM**
- **执行底座缺口**: `p79/experiment/conditions.py:90` **硬编码 router_on=False**
- **LR artifact**: B0/B1 × cls/red 共 **4 个 LR pkl 已 land (2026-05-16 12:28)**；**B2 (Gemma3-VL) LR artifacts MISSING**（2 个 B2 router_learned yaml 已标 BLOCKED）

**演变**:
- v1 (§149): 3-AI cross-stress **22 findings**（Claude 7/3 OOB + codex 8/3 + gemini 7/4），分类 4 三家共识 / 4 两家 / 7 独家
- v1→v2→v3→v4 (§151.8): **4 个 OOB 全部 user-caught，3 个 AI 一个没抓到**（v2→v3 user 抓 P0-8/P0-9/P0-10；v3→v4 user 抓 P0-11 archive ≠ independent test）
- v3 数值阈值 (§153.2) 实测 fire 0% → 决策树退化
- L1 单 5-fold (v6, §153.2) cls **+2.56pp** / red **-4.76pp** → repeated 5-fold ×10 (v7, §154.2) cls **+2.02pp** / red **-3.95pp**（cls magnitude 衰减 0.54pp）
- L1 三 variant (§153.2): Variant A (uniform LR) **collapse to always-predict-dom**；Variant B (balanced) cls +2.56pp / red -4.76pp；Variant C (binary + hand rule) **all underperform**

**已作废**:
- L1 single 5-fold 的 cls +2.56pp / red -4.76pp（被 §154.2 repeated CV 取代，台账标 `superseded_by: §154.2`）
- archive diagnostic 作为 **preregistration lock substrate** 的地位（台账标 `superseded_by: §151`，降为 correlated-population sanity check / directional confidence）
- v1 里三条 codex 抓到的**静态可验证事实错误**: (1) `adjusted_success` 已退役但 preregistration §359 还在用 → label provenance drift; (2) P1 伪代码的 early-return 会 bypass `RuleBasedRouter.decide()` 的 streak state update; (3) `compute_axtree_complexity` 这个函数不存在（只有 `build_page_state` 的 line-count）→ 会 ImportError

**caveats**:
- 「archive simulation = development sanity supplementary table, NOT paper-grade」
- 「单 fold 47 task/fold variance 大; magnitude 衰减 0.54pp 反映 single-fold optimism; CI lower bound 16.30 > baseline 15.81 所以仍 statistically beat」
- red: 「仍 fail Pareto vs phantom_som; red learned router collapse to majority on text-dominated archive」
- archive diagnostic: 「entropy 低意味着 learned classifier 会退化为 predict cheap default」
- fire rate: 「single run single cell; 结论是 numeric threshold 实际 dead, 真正 routing signal 来自 streak trigger」
- L2: 「sanity sim 非 paper-grade; 用来 confirm F1 (full-episode vs step-3 partial AUROC 不能 transfer)」
- P1 v3 阈值: 「archive simulation 非 fresh data」，**model unspecified, n unspecified**
- router_on=False: 「与 v5 §3.4 声称 'Phase 1a measures L2' 矛盾 = execution-substrate hole (codex pre-fire #2 OOB P0-7)」
- gemini 独家 P0 **Phantom Renaming**: P2 (TF-IDF+LR learned router) 实质等于 preregistration Appendix B 里 deferred 到 paper-2 的 **H7 oracle router 改名 H10 塞回 paper-1** → reviewer 会攻 hollow contribution padding；defuse = P2 增加显式 **test-leak-free 约束（features 只能是 step-1-observable，不含任何 post-run signal）**
- §149 计数自相矛盾（见 ⚠️C5）

**路由设计的机制依据**（§150b.1，台账标明是**引用**而非本处测量）:
- hijack trigger = capability (B0 immune / B1 susceptible) × mark density (33→128 marks 时 B1 num_ids 0→446) × site (reddit 74 / cls 111 marks)；`[SOM_MARKS]` text fallback 让 attention bypass screenshot (num_ids 归 0)
- has_ref_image 效应方向反转: **B1 reddit Cat B (ref-image 类) 反转 -3.6pp**
- 「引用而非本处测量」/「引用自本 chunk 范围之外的笔记行; 用于论证 (has_ref_image × model × density) 三轴交互使 hard-rule 不安全」

**证据**: §149.1/§149.2、§150b.1、§150b.4、§151.8、§153.2、§153.3、§154.2、§159.2、§167b.4；`docs/checkpoints/router/_archive_proposals/proposals_v1.md`、`proposals_v4.md`、`docs/checkpoints/router/proposals_v6.md`、`archive_diagnostic_2026-05-16.md`、`scripts/analysis/l1_archive_simulation.py`、`l2_partial_traj_auroc.py`、`train_l1_router.py`、`p79/experiment/conditions.py`

**原文片段**: 「dom_size > 12000 与 dom_complexity > 500 两条阈值 fire 0%; cls DOM 最大 5512 chars, red 最大 6963 chars」；「Single 5-fold (v6): 18.38%, Δ +2.56pp, CI [13.68, 23.50]; Repeated 5-fold x10 (v7): 17.84%, Δ +2.02pp, CI [16.30, 19.42]」；「4 个 OOB 全部 user-caught, 3 AI 一个没抓到」；「G-1 label entropy 0.606 (< log(2)=0.693 → P2 NOT viable)」

---

## 12. Observation 长度 / 截断 / mark cap 的实际触发率

**当前值**:
- **跨 mode obs_text 长度**（同一 step）: **dom=3804 / som=3879 / vision=3805 字符，ratio 1.00-1.02×**
- **截断触发率**（57435 个 observation_dom.txt 全量）: `>12000 字符截断仅 0.207% step 触发`；`>200-mark cap 仅 0.028% 触发`；**viewport filter 才是真正输入边界（median 3306 / p99 7656 字符）**
- `_extract_text_marks` 的 `re.search` **over-inclusion 率 = 1 行 / 155771 行 = 0.001%**（3000 个文件）

**演变**:
- Bug 1 (axis-1 截断/cap 不对称) 曾定 **P0** → §139.4 实测后**降级**
- 一度怀疑 **B0=12000 / B1=8000 的跨模型 truncation 不对称** → 实测 **wrapper 默认值两边都 12000，agent 里的 8000 是 dead default**（证伪）
- §148.2 推翻三个 AI 共同的错误 mental model（AXTree → flat 应该 -50% chars）

**已作废**:
- 「B0=12000 / B1=8000 跨模型 truncation 不对称」这个怀疑（证伪记录，防止重复怀疑）
- 三 AI 的 -50% chars 直觉（Claude D4 / codex Finding 6 / gemini C6 都套错）

**caveats**:
- 「我此前定 P0 靠的是 dom_size_threshold:12000 的推断不是测量; 教训写入 §139.retro (severity 不能靠推断要实测)」
- over-inclusion: 「user 据此决定不改 (动 fire 路径不值 1/155771)」
- §148.2: 「memory 文件 project_phantom_space_axes_format_not_information.md 早就写过 1.00× chars 实证但没人去看 artifacts; 由此新增 memory rule feedback_spotcheck_length_claims.md + skill v7.5 hard rule」

**证据**: §139.4、§139.6、§148.2；`results/visualwebarena/phase1/B1_3mode_classifieds_20260413` step_000 artifacts；`p79/experiment/som.py`；`docs/checkpoints/_status/issues/issue_about_blank_systematic_2026-05-16.md`

**原文片段**: 「57435 个 observation_dom.txt 中: >12000 字符截断仅 0.207% step 触发; >200-mark cap 仅 0.028% 触发; viewport filter 才是真正输入边界 (median 3306 / p99 7656 字符)」；「同一 step 上 dom=3804 / som=3879 / vision=3805 字符, ratio 1.00-1.02×」

---

## 13. Decoding cap（max_new_tokens 384 → 4096）

**当前值**: **max_new_tokens 384 → 4096**（B1 + B2 configs）

**依据与影响**:
- B1 `max_new_tokens=384` cap 的实际静默截断率 = **0.017% (4/23307 steps)**
- 「max_new_tokens 是 CAP 不是强制长度 (decoder 遇 EOS 即停) → **99.98% step 不变**; 只有 **0.02% outlier step** 现在能落地而非静默截断; GPU 显存不变 (transformers.generate 动态 KV cache); outlier 导致的 **per-cell wall time ≈ +10 min**」
- 生产路径 smoke（5 step, B1 dom classifieds）: `output tokens min=63 max=99 mean=79.0`；`cap hit 0/5`；`dom mode 的 image_meta = None`；`parse_valid Counter({True: 5})`；`5/5 记录 schema 合规`；`action_type = type, scroll, click, type, type`

**演变**: §142.3 codex 用 archived 数据把 Claude F3（假设高概率截断）从 **active emergency fix 降级为 paper §3.5.1 disclosure obligation** → §142.7 决定改 cap → §142.8 生产 smoke

**已作废**: 「B1 会高概率静默截断」这一假设（F3 降级）

**caveats**:
- 「+10 min/cell 是估计不是实测; Phase 1a clean rerun 当时尚未启动 → 无 archived-data divergence, 是完美时间窗」
- 「n=5 step 单 task; 5 step 最大输出 99 token 远低于旧的 384 cap → 印证 codex 的 archived 量化 (cap 对典型 thought+JSON 信封是过度设计)」

**证据**: §142.3、§142.7、§142.8；`configs/exp_v2_base.yaml`；`results/visualwebarena/phase1/B1_dom_classifieds_20260515_190653`

**原文片段**: 「B1 max_new_tokens=384 cap 的实际静默截断率 = 0.017% (4/23307 steps)」；「max_new_tokens 是 CAP 不是强制长度 (decoder 遇 EOS 即停) → 99.98% step 不变」

---

## 14. B1 在 DGX 上的解码速率与 Phase 1a wall-time 外推

**当前值**（DGX Spark GB10 共享 GPU, B1 Qwen3-VL-4B, dom classifieds, 5 step 样本）:
- **~2.6s / output token**（184s generate / 71 tokens）
- **per-step total ~190s**（184s generate + 6s env step）
- **30-step 单任务 ETA ~2h**
- **若 DGX 串行跑完 Phase 1a 单 baseline 单 site ≈ 9 周**

**演变**: §142.8 → 结论「Phase 1a 的 B1/B2 应跑 A100，DGX 留给 B0（proxy 是网络受限不是算力受限）」

**caveats**: 「n=5 step 外推到 9 周; DGX 是共享 GPU 有争抢」

**证据**: §142.8

**原文片段**: 「~2.6s / output token (184s generate / 71 tokens); per-step total ~190s ...; 若 DGX 串行跑完 Phase 1a 单 baseline 单 site ≈ 9 周」

---

## 15. B0 GLM rescue / parse 失败率

**当前值**（archived Phase 1 全量，pre-fix 数据）:
- B0 **GLM rescue 1.488%**（另一口径 §142.3/§170.2 报 **1.49% = 453/30437**）
- B0 **raw parse_fail 0.309%**
- B0 **post-rescue parse_fail 0.003%**
- B1 **true parse_fail 0.060%**
- keyword scroll/back fallback 触发 **≈ 11/53744 = 0.02%**

**演变**: §141.5 首测 → §142.3 codex 用不同口径复核给 1.49%，「与之一致」→ §170.2 gemini 用同一数字论证跨 baseline SR 平均无效

**已作废**: 台账未宣告作废；但记「新 paper-grade run 已 hard-block GLM」，故 1.49% 只描述 legacy archive

**caveats**:
- 「archived pre-fix 数据」
- 「keyword fallback 虽只 0.02% 但属 silent partial automation 故仍删」
- 「gemini 提出这让跨 baseline SR 平均无效; 属 legacy data interpretation gap, 新 paper-grade run 已 hard-block GLM」
- **GLM rescue 是 B0 独有，B1/B2 没有**

**证据**: §141.5、§142.3、§170.2；`docs/checkpoints/parse_advisor_pending.md`

**原文片段**: 「B0 GLM rescue 1.488% / B0 raw parse_fail 0.309% / B0 post-rescue parse_fail 0.003% / B1 true parse_fail 0.060%」；「1.49% safety net (B0 独有, B1/B2 没有)」

---

## 16. 跨 baseline 契约不对称（B0 / B1 / B2）

**当前值**（实测清单）:
- `tokens.input_image`: **B0 = 0 / B1 = 1280**（proxy API 不返回 image-token breakdown，HF processor 明确 split）
- `image_meta_recorded` 原只看 B0 payload → **B1/B2 永远 False**
- `glm_fallback_attempted` 原条件写入 → **audit trail ~47% biased**
- `aggregate_cost_electricity.py:104` 的 else 分支**静默把 B2 当 B1 处理**
- figure 层: **12/26 figures (46%) 只画 2-baseline 4-panel，B2 静默缺席**，而 paper §1 prose 声称 evaluate on B0/B1/B2 三 baseline
- learned router LR artifact: B0/B1 × cls/red **4 个已 land**，**B2 MISSING**

**演变**: §153.3（B2 当 B1）→ §169.4（两个审计字段）→ §170.2/§170.3（input_image）→ §173b（figure 面板，3-AI overlap）

**已作废**: —

**caveats**:
- 「paper §4 cost-fairness 因此是 extrapolation; 2-AI (Mode A 实证 + gemini paper 攻击) 同 root; 处置为 disclose (deferred codex round)」
- 「修后 image-mode 集合收窄为 {som, vision} (phantom_som 排除, som.py:322 确认无 image)」
- 「导致 paper §6 B2 cost story 结构性缺失 (codex #11 P1-13)」
- figure 修法「新建 `scripts/analysis/figures/lib/panels.py` 共享 helper, 从 run_registry.BASELINES + scored_task_count 取真值」；`paper_grade_panels()` smoke 返回 **6 个 PanelSpec (3 baselines x 2 sites), canonical N=224/205, 全部 placeholder=True**（pre-Phase-1a-fire 的正确行为）

**证据**: §153.3、§159.2、§169.4、§170.2/§170.3、§173b；`p79/backends/api_proxy.py`、`p79/experiment/types.py`、`scripts/analysis/aggregate_cost_electricity.py`、`scripts/analysis/figures/lib/panels.py`

**原文片段**: 「B0 = 0 / B1 = 1280 (proxy API 不返回 image-token breakdown, HF processor 明确 split)」；「12/26 figures (46%) 只画 2-baseline 4-panel, B2 静默缺席」

---

## 17. env-layer synthetic injection（[OPTIONS] / [DROPDOWN OPTIONS]）覆盖与披露

**当前值**:
- 覆盖率: **[OPTIONS] 42% + [DROPDOWN OPTIONS] 50% = 92% steps 覆盖**；paper §3 **0 disclosure（pre-fix）**
- archive observation 中含 dropdown 标记的比例: **cls 47/71 + reddit 144/177**（reddit 约 81%）
- canonical SoM builder 修复: **旧版 47 行无 options → 新版 48 行含 [DROPDOWN OPTIONS]**（单例 smoke）

**演变**: §136.4 用 dropdown 比例判定 B-82 爆炸半径 → §136.5 修 builder → §171.2 量化 92% 覆盖并发现 0 disclosure → 「后在 §175 落成 paper §3.5.3 disclosure」

**已作废**: pre-fix 的「无 disclosure」状态

**caveats**:
- 覆盖率一条的 scope 是「archive steps, **site/model unspecified**」
- 「该实测用于判定 [OPTIONS] 缺失对 axis-1 的污染面; reddit 约 81% 有 dropdown 使 reddit axis-1 最可疑」
- builder 修复「单例验证; py_compile ×10 文件 clean, git diff --check 无 whitespace」
- 「2-AI (Mode A F-A1 + Mode C F5/OOB-3) overlap」

**证据**: §136.4、§136.5、§171.2；`p79/experiment/som.py`、`p79/envs/vwa_wrapper.py`

**原文片段**: 「[OPTIONS] 42% + [DROPDOWN OPTIONS] 50% = 92% steps 覆盖; paper §3 0 disclosure (pre-fix)」；「cls 47/71 + reddit 144/177 个 observation 含 dropdown 标记」

---

## 18. VWA evaluator / submodule 缺陷与 SBOM lock chain

**当前值**:
- **LLM judge 极性 bug（paper-grade emergency）**: `helper_functions.py:626-628` 的 `if 'correct' in response: return 1.0` 会把 `'incorrect'` / `'partially correct'` / (llm_ua_match 里的) `'not the same'` 全判成 1.0 —— **monkeypatch 验证过**
- **walk-fail 回退路径**（3-AI overlap，最高信心）: `external/visualwebarena/browser_env/actions.py:1309-1315` 的 fallback 正是 locator-route 想淘汰的 **bbox-center 点击模式**
- **Viewport Paradox prose↔commit 矛盾**: paper §4.X.5 写 'We do not fix' 但 **commit 3f9ceca 实际已 fix (2026-04-19)**
- **硬编码 Tailscale IP 清除规模**: **913 个 task config 改写（3882 处替换）** `http://100.95.81.103:{PORT}` → `__SHOPPING__/__CLASSIFIEDS__/__REDDIT__/__WIKIPEDIA__` placeholder；codex 单独测得 `test_shopping.json` 794 hits + 8 个 P79 script；**最终 grep 全仓库 tracked file 0 IP hits**
- **`.auth/` 残留**: `.auth/` storage state 里**仍含 host IP**，而 paper §4.X.12 声称 hardcoded IP 已 closure

**submodule SHA chain 演变**:
- §157.2 三项校验值: `branch p79-patches / SHA f0c835b / grep -cP 'if not pred or not pred\.strip\(\):' = 2`
- §157.2 caveat: 「后续 A1.25 / A1.18-re 多次 bump 到 **c1765ee → 1c3a615 → 2f9b0b4**」
- §178.3 re-lock: `HEAD eb5cbd8 → 1c3a615`；`diff sha 9c562a3... → f1315dc4...`；`commit count 6 → 8`；同步 sed 到 7 个文件（locked_versions / preregistration / osf_lock_manifest / Makefile / preflight_v2.sh / test_vwa_evaluator_b91_guard.py / section4_limitations_disclosure）
- §185 再次 bump 到 **2f9b0b4 + tree-hash chain**（台账 §157.2 标 `superseded_by: §178`；§178.3 标 `superseded_by: §185`）
- 见 ⚠️C6（eb5cbd8 在链条中的位置不明）

**已作废**: SHA f0c835b / eb5cbd8 / c1765ee / 1c3a615 各自作为 current lock 的地位；paper §4.X.5 的 'We do not fix' 表述；paper §4.X.12 的 IP closure 声称

**caveats**:
- 「B-91 的 empty-pred guard 没有关掉这个洞; paper §1 SR 被系统性 inflate; 是整个 8h sprint 里 paper-grade 最严重的一条」；bug「从 upstream 89f5af2 起长期存在」
- Viewport Paradox: 「Claude Mode A 漏 (只核对 file:line 引用正确就判 disclosure OK, 没读 prose 验证 'we do not fix' 是否仍 true); gemini + codex 2-AI catch」
- IP 清除「排除 _deprecated/ 与 gitignored 的 scripts/vwa_env_remote.sh」
- `.auth/` 「Claude + codex 都漏; 处置是 §4.X.12 写 replayer-host re-capture 协议 + phase1_plan §B0 launch prereq」
- walk-fail: 「Claude A2 + codex B5 + gemini G2 三方独立命中 = 最高信心」

**证据**: §157.2、§164、§173.3、§178.1、§178.3、§185；`external/visualwebarena/evaluation_harness/helper_functions.py`、`browser_env/actions.py`、`external/visualwebarena/config_files`、`docs/checkpoints/pre_run/locked_versions.md`、`docs/checkpoints/paper_drafts/section4_limitations_disclosure.md`

**原文片段**: 「helper_functions.py:626-628 的 if 'correct' in response: return 1.0 会把 'incorrect' / 'partially correct' / (llm_ua_match 里的) 'not the same' 全判成 1.0 — monkeypatch 验证过」；「913 个 task config 改写 (3882 处替换)」

---

## 19. Launch infra gates（queue / preflight / sentinel / 完成率阈值）

**当前值 / 逐项实测**:
- **历史完成率分布**（用于定 sentinel 阈值）: **23/24 次历史 paper-grade fire 是 100% EXACT（cls/red 234/210）**；唯一例外 **B0_dom_shopping 99.78%** 是 pre-fix bug 造成 → 决定改成 **100% exact**
- **queue_chain.sh sentinel schema 不匹配（LAUNCH BLOCKER）**: pre-fix sentinel 读 `total_tasks / num_tasks / scored_task_count` 三个 key **全为 None**；`condition_summary_v2.json` 的 canonical field 实际是 **`episodes: int`（cls 234 / red 210）**，codex 用 sample=20 实测
- **`config_for_cmd` P-SoM 配置名 typo**: `config_for_cmd:228-229` 构造 `..._phantom_${site}.yaml`，实际文件是 `..._phantom_som_${site}.yaml` → **Gate 7 会在 launch all 时硬 FAIL 全部 6 个 P-SoM cell**；「该 typo 从初次 commit 起活过 **7 轮 cross-AI audit**」
- **B-230 auth gate 未传播**: **50% (18/36) Phase 1a conditions** 在 pre-fix 状态下走 soft-warn 路径，**前 1-3 个 phantom task 会 NOT-LOGGED-IN**（B-224 只修了 queue_baseline.sh，没传播到 3 个 phantom sibling）
- **resume-glob 复用 pre-fix archived 目录（codex C1）**: `queue_baseline.sh:133` + 4 个 phantom queue → 「若当天 fire，**6 个 cell 会静默复用被污染数据**」，修法 `FORCE_NEW=1` 强制 timestamped run_id
- **preflight Gate 5「GPU + model load smoke」**: pre-fix 只跑 `torch.cuda.is_available()`；修后加 `AutoConfig.from_pretrained(local_files_only=True)` 探 **Qwen3-VL-4B@ebb281e** 与 **Gemma3-VL@093f9f3**（revision 取自 `configs/exp_v2_base.yaml:103+138`），不分配 VRAM
- **preflight curl 超时 3s → 10s**
- **paper-grade locality 谓词失效**: 白名单模式 `*workspace/p79*` 匹配不上实际仓库目录名 `Cost-Aware-Routing-for-Web-Usage-Agents` → 修法是**逻辑反转**（`P79_PAPER_GRADE=1` 时默认开启，显式 `P79_PAPER_GRADE=0` 才退出）
- **lib extraction LOC**: queue_baseline 285→207 / queue_phantom_som 244→177 / queue_phantom_text 283→223 / queue_phantom_prompt 238→169 = **-274 LOC (-26%) + 174 LOC 新 lib = -100 net**
- **Round 2 commit (e9ddbe3) 自评 safe to push 的实际错误率**: Round A 独立验证发现 **6/7 finding 真实存在** —— queue 根本跑不起来（引用 12 个不存在的 configs，codex 实证 `queue_baseline.sh B0 dom classifieds` fails）；TOST→优效传播在 8+ 处未完成；H2 framing rule 与 prereg gross mismatch；heterogeneity 分支未实现；CSV loader 在真实数据上 `int('0.0')` 直接 crash

**演变**: §133.3（e9ddbe3 错误率）→ §134.2（resume-glob）→ §157.2（auth gate 传播）→ §162.2（sentinel schema）→ §183（100% exact 阈值 + A1.13）→ §188/§189/§192（config typo / Gate 5 / curl timeout）→ §195（locality 谓词）

**已作废**: 90% 完成率阈值方案（user 追问「90 够了吗, 要不要百分百, 现实吗」后由历史证据改 100% exact）；白名单式 locality 判定

**caveats**:
- sentinel: 「pre-fix 任何 100% completion summary 都会 fail validation → chain 跑完 cell 1 立刻 abort; Claude + gemini 都 missed」
- config typo: 「codex cold-read 独家发现; 教训是脚本活得越久审计越默认它 baseline 正确」
- curl timeout: 「Tailscale 冷连接 / A100 docker 首个请求可能 5-8s, 3s 会造成合法基础设施被误判 FAIL」
- lib extraction「净收益是 1 个 source-of-truth, 不只是 LOC」
- Gate 5: codex unique OOB「标题说 model load 实际没 load」

**证据**: §133.3、§134.2、§157.2、§157.3、§162.2、§183、§188、§189、§192、§195；`scripts/queues/queue_chain.sh`、`queue_baseline.sh`、`queue_phase1_paper_grade.sh`、`_lib_paper_grade_gates.sh`、`scripts/preflight_v2.sh`

**原文片段**: 「23/24 次历史 paper-grade fire 是 100% EXACT (cls/red 234/210); 唯一例外 B0_dom_shopping 99.78% 是 pre-fix bug 造成」；「该 typo 从初次 commit 起活过 7 轮 cross-AI audit」；「50% (18/36) Phase 1a conditions 在 pre-fix 状态下走 soft-warn 路径」

---

## 20. A100 self-host bring-up 实测

**当前值**:
- **容器清单**（docker ps 实测 2026-05-17）: `vwa-reddit (postmill-populated-exposed-withimg @9999)` + `vwa-wikipedia (kiwix-serve @8888)` + `vwa-shopping (shopping_final_0712 @7770/7780)` + `classifieds (jykoh/classifieds @9980)` + `classifieds_db (mysql:8.1)`；**无任何 wa-* / webarena / wa_shopping_admin 容器**（7780 是同一 Magento 实例的 admin 口不是独立 WA 容器）
- **依赖版本 skew**: A100 transformers **5.8.1** vs DGX **5.3.0**（torch 未被覆盖，loose pin >=2.0.0）
- **Gemma3-VL 4 mode-family smoke**（pin SHA `093f9f388b31de276ce2de164bdc2081324b9767`）: **ALL PASS** — `dom valid=True img_tok=0 / som valid=True img_tok=256 / vision valid=True img_tok=256 / phantom_text valid=True img_tok=0`
- **B0 AWS proxy 可达性**: (a) A100 → `i5xpracyci.execute-api.eu-west-2.amazonaws.com` 返回 **HTTP 401, 1.18s**；(b) 已认证空 body probe 返回 **HTTP 400 `{'error':'model and messages are required'}`** = key 被接受且请求达 Lambda 后端
- **端到端 paper-grade smoke (T16, B2 dom classifieds)**: 3 task × 30 steps 全 executed；`trigger_distribution = action_failed×46 / page_unchanged×37 / no_progress×37`；**success_rate = 0.0**
- **磁盘需求**: `REQ_GB` 从 **250 提到 350** —— 实际需求 = **216GB raw + 70GB extract = 286GB > 250**（此前 §163.2 已从 130 → 250）

**bring-up 踩到的 4 个互不相关 bug**（§140.11）:
1. HF dataset 全 404 — `webarena/{Shopping,Reddit,Wikipedia,Classifieds}` RepositoryNotFound，`setup_vwa.sh` 4 个 download 函数全死，改用 CMU metis 镜像 + archive.org，速度 1-10MB/s
2. Docker 29.1.3 + containerd-snapshotter 解压死锁 — `jykoh/classifieds:latest` 卡在 layer 91eb47e4f196c，PID STAT=Sl WCHAN=futex_wait_queue_me 0% CPU，dockerd 报 snapshot does not exist + 8h gc lease，`ctr pull` 同样卡
3. DGX home 有 quota，`docker save` 77GB 写 5min 后 disk quota exceeded
4. `pip install -r external/visualwebarena/requirements.txt` 强制降级整 venv（transformers 5.8.1→4.34.0, torch 2.11.0+cu128→2.0.1+cu117, playwright 1.59→1.37）使 Gemma3 smoke 作废

**前置条件（2026-05-14 SSH 实测）**: A100 venv `~/venvs/p79` 当时只有 torch（缺 transformers / qwen_vl_utils）；A100 VM 上无 VWA docker stack；shopping base_url config 需适配 local docker 重核

**已作废**: §140.6 的 Gemma3 smoke 结果被 §140.11 bug 4 的 venv 降级**作废**（原文: 「使 Gemma3 smoke 作废」）

**caveats**:
- T16: 「SR=0.0 是预期且不构成能力测量 (3 个随机 cls task, 4B baseline SR 本就很低, n=3); T16 的验证目标只是 pipeline 端到端跑通; 第一轮因 auto_login.py 把 cookies 写到 CWD-relative external/visualwebarena/.auth 而全 fail」
- Gemma: 「image-token 计数精确 (with-image 恒 256 = Gemma 固定值, no-image 0) → codex 标的 medium-confidence 点 (image_token_id vs 固定 256) resolved; gemma-3-4b-it 是 HF gated repo 需 license 接受 + token」
- proxy: 「`.auth/qwen_api` gitignored 不随 repo 转移, 需 scp DGX→A100 (perms 600)」
- bring-up: 「全是生态侧不可预期问题, dry-run 无法提前发现; 恢复需对 cu12 NVIDIA libs 逐个 force-reinstall (pip force-reinstall 不级联 deps)」
- REQ_GB: 「gemini 称为 Physical Impossibility; 是 v7.5 spot-check 规则 (数值必须算/量而非 'should be enough') 的实例」
- 「provenance 需记录该 skew」

**证据**: §138.8、§140.4、§140.6、§140.10、§140.11、§140.12、§163.2、§183、§195；`scripts/vwa/setup_vwa.sh`、`scripts/setup/a100_self_host_vwa.sh`、`p79/agents/gemma3vl_agent.py`

**原文片段**: 「REQ_GB 从 250 提到 350 — 实际需求 = 216GB raw + 70GB extract = 286GB > 250」；「dom valid=True img_tok=0 / som valid=True img_tok=256 / vision valid=True img_tok=256 / phantom_text valid=True img_tok=0」

---

## 21. DGX 磁盘配额事件

**当前值**: **200G hard quota，实际 216G used 触发 EDQUOT**；清理 blip2 15G + Qwen3-4B 4.9G + `_archive/` + 1 broken pilot **释放 22G → 178G**

**caveats**: 「DGX 总盘 506G 空闲但 admin 设 per-user cap; **Edit 能工作而 Write 不能**（Write 需新 inode/dir entry 触发 quota check）」

**证据**: §129.3（DGX Spark, user jiaming, 2026-05-13 03:00 BST）

**原文片段**: 「200G hard quota, 实际 216G used 触发 EDQUOT ... 释放 22G → 178G」

---

## 22. Carbon / energy 折算

**当前值**:
- `co2e = kwh x 0.220 (UK 2024)`，**PUE = 1.0**；`REGION_INTENSITY['uk'] = 257` 是 **decorative override**
- 真生产路径 = `LightweightEnergyTracker.estimate_step` (`runner/main.py:1472`)；实测 5 个 episode summary 的 **co2e/kwh ≈ 0.22 kg/kWh = 220 g/kWh = UK 电网**
- `compute_energy_step` 有 **0 个 production caller**（gemini #1 攻击对象）；`compute_waste_breakdown` 同样 **0 caller**

**演变**: §152.1 spot-check 推翻 gemini P0（#1 与 #4 均 **P0 → P2**，理由 dead helper 名字误导）→ §155.3 记录两个常数不一致

**已作废**: gemini 把 `compute_energy_step` 当生产路径的 P0 定级

**caveats**:
- 「两个常数不一致 (0.220 vs 257) 是 **disclosure 项而非计算项**; 已写入 paper §8.7」
- 「user 一句 carbon 我记得是有计算的吧 触发 spot-check」

**证据**: §152.1、§155.3；`p79/experiment/metrics.py`、`energy_tracker.py`、`docs/checkpoints/paper_drafts/section8_limitations.md`

**原文片段**: 「gemini #1 攻击的 compute_energy_step 有 0 个 production caller; 真生产走 LightweightEnergyTracker.estimate_step (runner/main.py:1472)」；「REGION_INTENSITY['uk'] = 257 是 decorative override」

---

## 23. 静默失败 / 静默放行的实例集（archive 实证）

**当前值**（逐条，各带 scope，**不合并**）:
- `coordinate_type=normalized` 却接收 pixel 坐标: codex jq 统计 **851 rows（507 no_progress + 344 null error_category）**；Mode A spot-check 分别得 **B0:321 / B1:530**
- `type` 动作无 target 时 `parse_valid=true`: archive **23 rows**（B0 cls task 204 为例）
- 裸 `{'action_type':'scroll'}` `parse_valid=true`: **2 rows**（B1 cls task_174 step_1）
- `select_option` 缺 env-level dispatch telemetry: **195/738 archive steps 无 dispatch meta**
- `bool(row.get('success', False))` 对字符串 `'false'` 判定为 **truthy** → paper §1 hero SR inflation vector
- **ghost fields**: `retry_action_*` + `glm_fallback_*` 共 **5 个字段**被 runner 写进 JSONL 但不在 dataclass 也不在 DEFAULTS catalog
- `state_change.py` 的 `active_element_tag`: 生产数据 **30/30 step 全为 None**（B2 reddit 2026-05-16 sample）
- Stage 4 extraction 静默容忍 per-mode 失败: `run_stage4_multimode_extract.py:227-228` **logger.error 但不 raise**，只有 ALL tasks fail 才 raise
- `select_option_meta.success` 语义漏洞: **「JS evaluate 没抛异常 = success=True」**，被 **4 批独立 audit 在 6 小时窗口内命中**（Mode A C2-A1 + Mode B F2 + 平行 session A1.4 codex B-453）
- `scored_task_count==0` 静默 fallback 链（见主题 9）
- **reference image 加载双重失效**: (1) `glm_batch_digest.py:255` 用 `parents[2]` 得到 `scripts/external/visualwebarena` 该路径不存在（正确是 `parents[3]`）；(2) `_extract_site:957-962` 永远返回空串，而 `_load_reference_images_b64` 在 `if not site` 处早退 → **reference images 从未加载**，paper §3「DOM mode 用任务参考图做视觉匹配诊断」的 prose **在当前代码路径上被证伪**

**防线**: §197 `metrics.py` hero numeric strict guard（20 个字段）活体攻击 `{'steps': True}` / `{'total_cost_usd': '1e309'}` / inf / nan / quarantined **共 5 种输入 → 5/5 raise**；合法路径通过

**caveats**:
- coordinate_type: 「两条独立 lineage 得到同一数字同一 root cause; 属环境层自动纠错隐藏 schema violation」
- Stage 4: 「当前 v2 NPZ 运气上 balanced, 但协议允许 silent imbalance」/「当前数据未受害 (luck-balanced), 是协议层风险; §131.2 Commit C (P0-2 fail-closed grid) 修复」
- active_element_tag: 「测试用合成 'focused input' 造成假阳性; 这是『测试用合成数据验证生产路径不会触发』反模式的实证」
- ghost fields: 「paper §3.5.1 引用的 audit field 是 schema 外黑户」
- strict guard「合成攻击样本」
- reference image「codex Mode B P1-3 OOB; 属 prose↔code claim-mismatch 类」

**证据**: §130.5、§155.3、§161.1、§170.2/§170.3、§171.2、§175.1、§197、§202；`p79/backends/action_utils.py`、`p79/envs/vwa_wrapper.py`、`p79/experiment/io_utils.py`、`types.py`、`state_change.py`、`metrics.py`、`scripts/maintenance/glm/glm_batch_digest.py`

**原文片段**: 「codex jq 统计 851 rows (507 no_progress + 344 null error_category); Mode A spot-check 分别得 B0:321 / B1:530」；「两者叠加 → reference images 从未加载, paper §3 『DOM mode 用任务参考图做视觉匹配诊断』的 prose 在当前代码路径上被证伪」

---

## 24. Dead code / dead config / inert flag

**当前值**:
- **`paper_grade` flag 结构性 inert**: pre-fix **全仓库 0 grep hits**；codex 另 grep 出 **39 个 B0 yaml 写着 `use_glm_fallback: true`**
- **HeuristicDomBackend 家族零使用**: **0/53924 step rows** + **0/119 yaml 有 `dom_mode=heuristic_only`** + **0 处 paper §3 mention**；M1/M2/M3/M4 module flags 生产中全 False（codex `module_flag_true_counts: {}`）
- **`candidate_modes`** 字段在 `p79/` 与 `scripts/` 里 **0 grep 匹配 = dead**
- **`base.yaml max_steps 40` 但 119 个 per-condition yaml 全 override 成 30** = dead default（且是 **33% 成本虚高向量**）
- **`phase2.yaml` / `phase3.yaml`** 在 B-263 retire 之后仍引用**已退役 mode `hybrid` / `dom_only`**
- **`p79/logging/__init__.py` 是 0 字节孤儿包**，而 CLAUDE.md L79 声称是 'structured logging helpers'
- **`conditions.py:90` 硬编码 `router_on=False`**（见主题 11）

**caveats**:
- paper_grade flag: 「三个独立 cold-read (Claude code-trace / codex grep / gemini paper §3 fairness prose 攻击) 都 catch 同一 root」
- HeuristicDom: 「3-AI 共同实证; 是 D1 删除决策的依据」
- logging: 「3-AI 都命中; 处置为 **P2 defer**（删空包+改文档 vs 实现 structlog 是独立决策）」

**证据**: §153.3、§169.3/§169.4、§171.3、§190、§193；`p79/experiment/config.py`、`p79/backends/heuristic.py`、`configs/exp_v2_base.yaml`、`p79/logging/__init__.py`

**原文片段**: 「pre-fix 全仓库 0 grep hits — paper_grade flag 结构性 inert; codex 另 grep 出 39 个 B0 yaml 写着 use_glm_fallback: true」；「0/53924 step rows + 0/119 yaml 有 dom_mode=heuristic_only + 0 处 paper §3 mention」

---

## 25. Watchdog stack

**当前值**:
- **破坏性操作 race window 实测**: 原 paper §4.X.13 与 watchdog L1680 注释写 **'2-3s'**；实测破坏性操作集合**典型 ~10-200ms，最坏约 1s**；event log 本身增加 **~1-10ms**
- **reddit auth 检测正则**: 5 个真实 reddit DOM 文件（6.9-7.2KB, `B0_phantom_som_reddit_20260428` sample）—— **OLD regex 0/5 检出 logged-in，NEW per-site regex 5/5 检出**
- **auto-clean race-prone 行 git blame**: `experiment_watchdog.py L1342` blame 日期 **2026-04-09，6+ 个月未改动**
- **Mode A 自审否定的候选 finding**: A3「5000-byte DOM cap 会截断 auth marker」—— 实测 **reddit DOM 的 marker 在第 563 字节，远在 cap 内** → file 之前自我 downgrade
- **Option K covariate 端到端 smoke**: fake events → merge → aggregator 产出 **5 条 covariate row × 5 field 全 PASS**（retry-path task 10 / session-wave task 20/21/22 wave_size=3 / clean episode task 30 全 False）；idempotent re-merge 返回 0；archived run（`episodes: int(3)`）fallback 到 filesystem scan 产出 3 episodes 不 crash

**已作废**: '2-3s' race window 猜测值

**caveats**:
- 「该 '2-3s' 猜测活过 7 轮 audit 因为没人量过; B-743 digest 退役后最慢的组件已被移除」
- reddit 正则: 「Claude unique OOB; 靠实读 artifact 才发现, 纯 mental model 会漏」
- blame: 「用于纠正 user 记忆混淆 (把 B-314 hook 新增误记成 race semantics 重构); Claude 诚实承认没发生重构」
- Option K「合成数据 smoke, 非生产数据验证」

**证据**: §168.2、§168.4、§196；`scripts/maintenance/experiment_watchdog.py`、`scripts/analysis/aggregate_trajectory_covariates.py`、`docs/checkpoints/paper_drafts/section4_limitations_disclosure.md`

**原文片段**: 「原 paper §4.X.13 与 watchdog L1680 注释写 '2-3s'; 实测破坏性操作集合典型 ~10-200ms, 最坏约 1s」；「OLD regex 0/5 检出 logged-in, NEW per-site regex 5/5 检出」

---

## 26. 可复现性 / determinism / provenance

**当前值**:
- **`hash()` 非确定性**: 同 `--seed 42` 下 `pooled_effect 4.22569 vs 4.23549`（PYTHONHASHSEED 影响）；改用 `hashlib.sha256` 后 **byte-identical 4.276338513017495**
- **seed 分离**: `--data-seed 42` 固定下 `--bootstrap-seed 43 vs 44` → **θ_FE 4.11 vs 4.14**
- **logger_v2 flock 压力测试**: **4 workers × 10 events × 6KB payload → 40/40 valid, 0 torn writes**（加 `fcntl flock LOCK_EX` 之后）
- **snapshot_env.py 修复后 smoke**: `schema_version 2026-05-16-a1.16`；B1 Qwen3-VL-4B-Instruct loaded SHA `ebb281ec70b05090...` 与 registry HEAD 一致 → `divergence: match`；B2 因缺 HF_TOKEN → `divergence: gated_no_token + 1 errors entry`；`evaluator_code.files_count = 125`（6 core + 119 configs）；`pip_freeze_lock 100+ packages`

**caveats**:
- hash: 「codex unique OOB; 属 **OSF byte-reproducibility blocker**」
- flock: 「POSIX O_APPEND 原子性只到 **PIPE_BUF=4096**, 长 metadata (wave_task_index / purged_digest_records) 轻易超过」
- snapshot: 「`snapshot_vwa.sh` 没 smoke (需 docker daemon + live VWA stack), 只做静态检查; runtime 验证待 A100」
- A1.16 里 gemini 的 2 个 P0（TF32 matmul / dtype non-determinism）「落在 mechanism script, 随 paper-2 defer」；`numerical_determinism_check.py` 同样随 paper-2 defer

**证据**: §160.5、§172.3/§172.8、§177、§194、§200；`scripts/analysis/preregistration_decision_test.py`、`scripts/provenance/snapshot_env.py`、`p79/experiment/logger_v2.py`

**原文片段**: 「同 --seed 42 下 pooled_effect 4.22569 vs 4.23549 (PYTHONHASHSEED 影响); 改用 hashlib.sha256 后 byte-identical 4.276338513017495」；「4 workers x 10 events x 6KB payload → 40/40 valid, 0 torn writes」

---

## 27. Cost 计算的两处错值

**当前值**:
- **Phase 2 routed cost 漏算 obs_prepare（B-177 单元测试）**: pre-fix net-saving 被高估为 **0.55**，post-fix **0.45**；差值正好等于 **obs_prepare 0.10**（runner cost = model + router_overhead + obs_prepare，而 `:1538` 只取前两项）
- **paper §3.5 的 cost 倍数错误**: prose 写 **10x，实际 50x**

**caveats**:
- obs_prepare: 「**合成单元测试数值, 非真实实验数据**; 修法改为直接用 canonical `avg_total_cost_usd` 相减 + emit 4-way 分解 + 1e-9 sanity invariant」
- 10x/50x: 「Claude F4 + gemini 2-AI overlap」

**证据**: §150a.2、§185；`p79/experiment/analysis.py`、`docs/checkpoints/paper_drafts/section3_definition.md`

**原文片段**: 「pre-fix net-saving 被高估为 0.55, post-fix 0.45; 差值正好等于 obs_prepare 0.10」；「prose 写 10x, 实际 50x」

---

## 28. FP architecture hard-delete 与 has_effective_action

**当前值**:
- **A1.6 FP hard-delete 验证**: `py_compile` 全 clean（analysis.py + 8 analysis script + 4 maintenance script + 2 figure + power_analysis + 1 新 test）；`tests/test_fp_architecture_invariants.py` **9 passed**；3 个 regression 套件 **41 passed**
- **has_effective_action 若保留会误降级的 episode 构成**: 13 个会被降级的 episode 中**仅 4 个是真 click-only，8 个是 narrow filter 意外抓对的 stale comment FP**

**caveats**:
- 「9 个 invariant 用于 guard future-regression」
- 「该研究支撑 B-85 SUPERSEDED 的判定（启发式的『命中』大多是巧合）」
- 13 = 4 + 8 之外还有 1 个未说明类别 —— **原文未解释余下 1 个**（[聚合者推论] 仅指出原文数字未闭合，不推断其归属）

**证据**: §139.8、§158.5；`tests/test_fp_architecture_invariants.py`、`p79/experiment/analysis.py`

**原文片段**: 「13 个会被降级的 episode 中仅 4 个是真 click-only, 8 个是 narrow filter 意外抓对的 stale comment FP」

---

## 29. paper prose ↔ code/数据 的 claim mismatch 清单

**当前值**（本批次抓到的全部条目）:
- paper §1:7 prose **B=10000** vs prereg+code **B=1000**（submission-gate 级）
- paper §1 「**Zero image tokens**」但 prereg §2.6 明确 **reference_image 保留** → 技术不准确
- paper §1 hero **3.33pp / 2.56pp 是 archive 4-mode universe**，而 prereg H1 是 **6-mode FE-pool target** = **estimand schizophrenia**
- paper §3 prose 缺 hero 数字 **94.4%（locator-route walk success）**「完全不在 paper §3 prose 里」
- paper §3.5 cost 倍数 **10x vs 实际 50x**
- paper §4.X.5 「We do not fix」vs **commit 3f9ceca 已 fix**
- paper §4.X.12 声称 hardcoded IP closure vs **`.auth/` 仍含 host IP**
- paper §4.X.13 race window **'2-3s'** vs 实测 **~10-200ms / 最坏 ~1s**
- paper §3 「DOM mode 用任务参考图做视觉匹配诊断」vs **reference images 从未加载**
- paper §1 prose 称三 baseline vs **12/26 figures 无 B2 面板**
- paper §3.5.1 引用的 audit field 是 **schema 外 ghost fields**
- CLAUDE.md L79 称 `p79/logging/` 是 structured logging helpers vs **0 字节孤儿包**

**caveats**: 「处置是加两个 footnote (`[^image-scope]` + `[^hero-estimand-scope]`) 而非改数字」（§173b 的两条）；多条标注「Claude + codex 都漏，gemini catch」

**证据**: §164、§165.2、§173.3、§173b、§185、§193、§196、§202

**原文片段**: 「(G3) §1 写 'Zero image tokens' 但 prereg §2.6 明确 reference_image 保留 → 技术不准确; (G1) hero 3.33pp/2.56pp 是 archive 4-mode universe, 而 prereg H1 是 6-mode FE-pool target = estimand schizophrenia」

---

## 30. Cross-AI /stress 各轮 finding 收敛（大表）

**当前值**: 下表逐轮抄录，**未做任何求和/平均**。格式 `n findings / m OOB`；空白 = 原文未给。

| § | 轮次 / scope | Mode A (Claude) | Mode B (codex) | Mode C (gemini) | 统一 | fix / B-range |
|---|---|---|---|---|---|---|
| §129.4 | Phase 1 pre-run 全栈 | 并行另抓 2 项 (CL1/CL2) | 10min, 19886 行, 4 HIGH | — | — | — |
| §133.2 | 3-round codex (A/B/C) | — | 26 findings, 11 HIGH | — | 3 cross-AI overlap (T1/T2/T3) | — |
| §134.2 | /stress v6 pre-fire launch infra | 7 / 5 OOB | 9 / 5 OOB + 4 clean negative | — | 16 actionable | — |
| §141.12 | A1.1-A1.4 chain 终态 | — | — | — | 34 findings / 17 code-fixed / 2 disclosed / 4 parking | pytest 88 → 129 |
| §142.1 | A1.1 v7 (三 AI 首次全 PASS) | 5 / 3 OOB, ~30min | 5 / 2 OOB, 120s | 3 weak + 2 strong, 2 OOB, 120s | — | — |
| §143.3 | post-Batch-5 pre-fire | — | 10 (4P0/3P1/3P2), 3 OOB, 7m12s, 8240B | 7 (2P0/2P1/1P2+2 cross-cutting), 3 OOB, 4m30s, 7982B | — | — |
| §144.1 | A1.1 v8 | 7 / 4 | 8 / 5, ~4min | 5 / 3, ~23min | 3-AI overlap 1, 2-AI 1, 单 AI 17 (5/6/5) | — |
| §146.1 | A1.2 v8 backends 2nd pass | 7 / 3 | 5 / 3 | 5 / 2 | 17 distinct / 9 OOB, 仅 2 overlap | — |
| §147.1 | A1.3 v8 envs 2nd pass | 7 / 4 | 7 / 4 | 7 / 3 | 21 raw → 19 distinct / 11 OOB, 1 dual overlap | — |
| §148.1 | A1.4a orchestrator core | 8 / 4 | 8 / 5 | 7 / 4 | 23 raw → 21 distinct / 13 OOB, 4 dual-catch | — |
| §149 | router design v1 | 7 / 3 | 8 / 3 | 7 / 4 | 22 findings; 4 三家 / 4 两家 / 7 独家 | — |
| §150a.1 | A1.4b-i analysis.py | 9 / 4 | 9 (2 OOB + 4 borderline + 3 anti-attack) | 5 / 3 | 17 unique; 4 cross-validated; unique 5/8/1 | — |
| §152.1 | A1.4b-ii data plane | 7 / 3 | 8 (2 OOB+4 borderline+3 strong-defuse+3 quant), ~5min | 5 | — | — |
| §155 | A1.4c auxiliary 7-file | 11 / 3 | 9 / 7（与 Claude 零重叠） | 8 / 4 | 28 unique / 14 OOB | — |
| §156 | A1.5 utils+cli | 9 / 3 | 7 / 5（零重叠） | 5 / 2 | 21 unique | B-211~B-229; 9 fixed/5 deferred/5 disclosed/2 redirected |
| §156.6 | A1.1~A1.5 累计 | — | — | — | B-112~B-229 约 90+ entries, 109+ findings, 50+ OOB | — |
| §157.1 | A1.13+A1.14 queue/preflight | 8 / 5 | 8 / 0 OOB labels (lean retry) | 6 / 3 | 22 → 10 unique | 7 fixed B-230~B-236 |
| §160 | A1.16 provenance | 8 / 5 | 9 / 2 (retry) | 7 / 2 (2 P0 unique) | 22 raw → 11 unique | 7 fixed B-273~B-279 + 5 deferred |
| §161 | A1.8 schema substrate | 9 / 5 | 7 / 4 | 4 / 2 | raw 11+7+4 → 18 unique, 全走 fix | B-280~B-297; pytest 316/316 |
| §162.1 | A1.17 Chunk 1 scripts/vwa | — | 4m49s | 2m25s | 22 attacks = 5P0/12P1/5P2; 1 三方 + 6 双方 + 15 单方 | — |
| §163.2 | A1.17 Chunk 2 | — | — | 3 unique | — | B-309~B-311 |
| §164 | A1.18 submodule | 7 / 4 | 8 / 3+ | 6 / 2 | 15 unique | — |
| §165 | A1.9 metrics/energy/env | 10 / 6 | 9 / 5 (RETRY x1) | 7 / 2 (RETRY x2) | — | 22 fixes B-320~B-341 (8P0/9P1/5P2), 5 commits ~740 LOC + 195 LOC tests |
| §166 | A1.12 tests/ | 8 / 4 | 9 / 3 | 4 / 2 | 17 (1 三方 + 4 双方 + 10 单方) | 17 fixes B-342~B-358; tests 333→398 (+65) |
| §167b.1 | A1.10 control-flow | 12 / 4 | 7 / 4 | 9 / 4 | 28 unique; 1 三方 + 2 双方 + 22 单方 | 17 fix (8P0/9P1) + 8 prose + 5 defer; pytest 406/406 |
| §168.1 | A1.15 watchdog stack | 11 / 3 | 9 / 6, 362s, 433KB | 10 / 5, ~5min, 15KB | — | 11 fixes B-384~B-394 |
| §169.2 | A1.1 agents | 9 / 3 | 9 / 4 (retry 2) | 7 / 4 | 16 distinct (12 unique + 4 overlap); 2 个 3-AI overlap | 11 fixes B-395~B-405; Mode A solo 漏 ~55% |
| §170 | A1.2 backends | 5-6 / 3 | 7 / 5, 15135B | 5 / 3, 7069B | 16; 1 三方 + 2 双方 + 13 单方 | 11 fixes B-406~B-416; pytest 414/414; Mode A solo 漏 ~64% |
| §172 | A1.19 分析管线 | 10 / 4 | 8 / 4, 8.5KB, 6min | 7 / 5, 8.5KB, 24s | 20 unique = 4P0/13P1/3P2 | 13 fixes B-426~B-438 |
| §173b | A1.20 figure scripts | 13 / 4 | 9 / 4, 7721B, 3-4min | 7 / 3, 8192B, 1.6s | 25 unique = 11P0/9P1/5P2 | 19 fixes B-459~B-477 |
| §173.2 | A1.25 GRL Chunk 1 | 10 / 6 | 8 / 5, 10964B, ~3min | 7 / 3, 9885B, ~3min | 25 unique = 8P0/11P1/2P2; 1 三方 + 3 双方 + 18 单方 (11 OOB) | 10 fixes B-439~B-448 |
| §174 | A1.4 SoM extraction chain | — | — | — | 3-AI overlap **0**; 2-AI 3 clusters; 1-AI 17; 共 22+ distinct | 8 fix + 2 defer |
| §175b | A1.5b Phase 1 控制面 | 11 / 5 | 8 / 4, 2m56s, 11.5KB | 8 / 5, 16.8s | 2 三方 + 2 双方 + 17 单方 (A8/B5/C4) | 21 fixes B-485~B-505; Mode A solo 只能抓 8 (38%) |
| §177 | A1.21 决策测+注册表 | — | — | — | 29 unique = 11P0/15P1/3P2 | 21 fixes B-513~B-533; pytest 420/420 |
| §179 | A1.25 GRL 全周期 | — | — | — | Mode B/C 的 OOB catch 占 ~18/29 fixes (62%) | — |
| §180 | A1.5b Phase 2 数据面 | 13 / 4 | 10 / 4 | 4 / 2 | 19 unique = 5P0/9P1/5P2; 7 个 2-AI overlap; 单方 5/5/2 | — |
| §181 | A1.22 跨 baseline 输出契约 | 8 | 8 | 7 | 22 unique | 17 fixes B-560~B-576; pytest 421/421; PAPER_GRADE_KEYS 12→17 |
| §182 | A1.5 (post-Phase 2) | 8 / 5 | 7 weak + 8 bug rows, 10.4KB, ~3min | 3 weak + 4 rows, 5.8KB, 24s | 8 unique (1 三方 + 6 A+B + 1 Claude 独家) | 8 fixes B-548~B-555; 428/8 |
| §183 | A1.13 launch+collision | 7 / 4 | 8 / 4, 6.6KB | 10 / 3, 8.5KB | 3-AI 0, 2-AI 6, 单方 5/4/4 | 18 fixes + 1 scaffold = 19 B-numbers B-630~B-648, 5 commits |
| §184 | A1.6a analysis.py L1-900 | 9 raw → 8 filed (3P0/5P1/1P2/6 OOB) | 6 / 4, 6265B, 2min | 6 / 3, 3.3s | 3-AI 1, 2-AI 4, 单方 3 | 8 fixes B-596~B-603; 427 passed / 8 skipped |
| §185 | A1.18-re submodule | 9 / 4 | 12 / 6 | 7 / 4 | 28 raw; 0 三方, 3 双方, 22 单方 | 26 fixes B-604~B-629; 428/8 |
| §186 | A1.6b analysis.py L900-2012 | 9 / 3 | 6 / 5, 6997B, ~2min | 5 / 3, 6501B, ~3.3s | 16 unique, **13/16 = 81% OOB** | 12 fixes B-650~B-661 + 10 invariants; 436/8/1 |
| §187 | A1.12 cold-start tests/ | 8 / 4 | 7 / 2+, 12119B, ~6.5min | 5 / 5, 8580B, 22.7s | 18 unified = 5P0/10P1/3P2; 0 三方, 4 双方, 14 单方 | 8 fixes B-662~B-674 (user 砍掉 P0-1); pytest 428→443→467 |
| §188/189/191/192 | A1.14 orchestrator+preflight cold-start | 12 / 6 | 8 / 4, ~5min, 270KB (~5KB findings) | 7 / 3 (retry x1) | 22 unique = 4P0/11P1/7P2; 3-AI 0; 单方 6/5/5 | 20 fixes B-672~B-683 + B-703~B-710 (4 chunk); shell tests 20→24→28→37 |
| §190 | A1.7 conditions/configs cold-start | 10 候选 | **FAIL**（3 次重试全在 exec /bin/bash 探索循环耗尽 token, 139KB 未到 synthesis） | PASS (retry x2), 7134B, 5 findings | 13 findings, OOB **8/13 = 62%**; 3-AI 0; A+C 3; Claude 独家 8 | 11 fixes B-691~B-701; 500 passed / 9 skipped |
| §193 | A1.11 utils/cli/logging | — | — | — | 17 unique; 3 三方 + 7 双方 + 7 单方 (codex4/Claude3/gemini1); 分布 18%/41%/41% | 14 fixes B-717~B-730; pytest 500→530 |
| §194 | A1.8 cold-start schema | 8 / 3 | 6, 12198B, 5-7min | 5, 8665B, retry x1 | 13 unique, OOB **8/13 ≈ 62%**; 0 三方, 2 双方, 11 单方 | 10 fixes B-731~B-740; 566/9 (+36 invariants) |
| §195 | A1.17 cold-start vwa+RESET | — | 9 / 7, 13min, 18036B | 7 / 4, 1.5min, 8089B | 21 unique; 1 三方 + 3 A+B + 2 A+C + **0 B+C** + 12 单方 | 17 fixes B-744~B-760 (3P0/14P1), 4 P2 defer; 611/9 (+26) |
| §196 | A1.15 cold-start watchdog+Option K | 7 / 3 | 8 / 5 | 9 / 4 (retry x1, 2 FP 剔除) | 24 attack vectors → 16 unique; 0 三方, 3 双方 | 13 fixes B-741~B-743 + B-761~B-766; 49 invariants; 566→585→641; Mode A solo 漏 ~69% |
| §197 | A1.9 cold-start cost/energy/env | — | — | — | 19 unique = 4P0/9P1/5P2/1 GRL prose; OOB **12/18 = 67%**; 0 三方; A+B 4, A+C 1; 单方 1/5/6 | 17 fixes B-782~B-798; 33 invariants; 674/9 |
| §199 | A1.2 cold-start backends | 11 | 10 (v2, 全部带 `python3 -c` 实测 stdout) | 9 报告但仅 3 条留用 | 22 unique (4P0/9P1/9P2) + 1 honest-gap; OOB **13/22 = 59%**; **91% lineage-unique**; 0 三方, 3 双方, 19 单方 | 23 fixes B-799~B-821; 707/9 (+33) |
| §200 | A1.16 cold-start provenance | 10 / 5 | 7 / 3, 5710B, 183s | 7 / 3, 8763B, 293s | 17 unique (6P0/10P1/3P2) + 1 disclosure; OOB **14/18 = 78%**; 1 三方 + 5 双方 + 12 单方 | 18 fixes B-822~B-839; 733/10 |
| §201 | A1.15b GLM sidecars pre-audit | 12 / 3 | 8 / 3, 5.8min | 7 / 4, ~32s | 22 unique; 3 三方 (sync delete / burn chain / ntfy public) + 1 双方 | Chunk α 4 fix B-841~B-844 |
| §205 | A1.23 concurrency+race | 10 / 5 | 9 / 6, 6889B, 386s | 7 / 4 (retry x1; 首次 16s + 1-byte empty = silent-fast FAIL), 7201B, 248s | 15 unique (3 三方 / 5 双方 / 8 单方) | 14 fixes (5P0/7P1/2P2), P1-11 dropped; 852/10 |
| §206 | A1.24 clear_tasks | 10 / 6, 8 artifacts | 9 / 6, 177s, 10KB, Phase 4 5/5 | 7 / 3, 73s, 9KB, Phase 4 7/8 | 19 unique; 3-AI 2 / 2-AI 3 / 1-AI 14 = **91% lineage-uniqueness** | 811/10 (+20) |
| §207 | **A2.1**（首个 §A2 设计层 audit, paper §1 framing） | 6 / 4 | 8 / 3, 153s, 19839B, Phase 4 3/3 | 5 / 2, 190s, 9397B, Phase 4 2/2 | 10 unique / 6 OOB = **60% OOB 命中率** | — |

**caveats**（全表通用）:
- 「finding 计数为自评」/「计数为自评 tally, 无统一 finding 粒度定义」/「OOB 数为各 AI 自报」多次出现
- §207 的 60%: 「比 §A1 code-layer typical 40-50% 高一档（原文标为「推断」设计层 attack surface 更 dense）」
- §146.1: 「这是 second-pass (first-pass 已修 F1-F6), 仍出 10 个新 finding 含 multi-seed cache 这种 first-pass missed 的 paper-grade blocker → **/stress 不是 one-shot**」
- §174: 「A1.4 的 cross-AI 价值在 **layer 互补 NOT confirmation**（与 A1.1 高 overlap surface 不同 shape）」
- §187 / §194 / §195 / §196 / §197 / §199 / §200 均为 **cold-start**（明确不参考先前 fix / 历史 round）
- §206 的 91%「在已经过 A1.2 cold-start 23-fix + A1.15b 16-fix + A1.23 14-fix 的成熟 substrate 上」
- 多轮报告的 pytest 计数非单调（见 ⚠️C7）；多轮 B-range 存在重叠（见 ⚠️C8）

**证据**: 表内 § 全部；`docs/reference/master_bug_catalog.md`、`docs/checkpoints/codex_outputs/`、`docs/checkpoints/gemini_outputs/`

**原文片段**: 「17 distinct findings 9 OOB ... 只 2 个 overlap, 13 个独家」（§146.1）；「3-AI overlap = 0 distinct findings; 2-AI partial overlap 3 clusters; 1-AI unique 17 findings」（§174）；「3-AI overlap 2 / 2-AI overlap 3 / 1-AI unique 14 = 91% lineage-uniqueness」（§206）

---

## 31. Cross-AI lineage 结构：overlap 演化与 Mode A solo 漏率

**当前值**:
- **Mode A (Claude) solo 漏率**（各轮自报）: **~55%**（§169.2）/ **~64%**（§170）/ **只能抓 8 = 38%，即漏 62%**（§175b）/ **~69%**（§196）/ 「单 Claude 会漏 2 个 P0 OOB」（§161）/「单 Claude 会漏 13 findings 含 6 个 P0/P1 OOB」（§167b.1）
- **lineage-uniqueness 趋势**: 早期轮次有 3-AI overlap（§144.1 1 个 / §169.2 2 个 / §193 3 个 / §201 3 个 / §205 3 个 / §206 2 个），中后期多轮 **3-AI overlap = 0**（§174 / §183 / §185 / §187 / §188 / §190 / §194 / §197 / §199）；§199 与 §206 均达 **91% lineage-unique**
- **Mode B 4 轮验证的 codex-unique vs Claude-unique**（§129.7）: 5-12 早 Mode A pilot **codex 5/6, Claude 1/6**；5-12 晚 lean methodology **codex 6 (4 HIGH), Claude 0**；5-12 凌晨 wrap **codex 4, Claude 4**；5-13 凌晨 pre-run **codex 4 HIGH, Claude 2**
- **A1.25 GRL 全周期**: Mode B/C 的 OOB catch 占 **~18/29 fixes (62%)**
- **两个 lineage-specific catch 的互补实证**（§193）: **B-717** credential 通过 argv 传给 subprocess（`ps auxe` 可见）被 **Claude + gemini 命中而 codex 漏**；**B-718** CLI 的 `sys.path` bootstrap 在 `p79.*` import 之后（绝对路径调用会失败）**只有 codex 靠真跑 `python3 /tmp/...` 复现抓到**，Claude 与 gemini 静读都漏 → 「codex 的**实验性复现 > 静态阅读**（针对 env/subprocess 类 bug）」
- **clean negative check 的价值**（§134.2 codex 4 项）: viewport operator-precedence bug 已修（`processors.py:218` 现为 `(ow*oh)/(w*h)`）；SoM 与 P-SoM 的 text path 共用 `_extract_text_marks`；12 个 baseline config 的 merge 正确；JSONL dedup 弱但 runner 会 unlink stale step JSONL → 非 Phase 1a blocker
- **codex 主动 strong-claims 反攻 spotcheck**（§150a.1）: TOST 公式 `zero-lift → 0.0`, `+2pp lift → 0.905`；DL τ²: `Q=2 df=2 → 0`；sibling task_id propagation grep 全 `scripts/analysis/` 已正确

**caveats**:
- 「negative check 同样有价值 — 避免重复审同一处」
- 「这类『反向验证』（确认某处没问题）是 v6 Skill 的明确收益，防止重复审同一处」
- 各轮「Mode A solo 漏 X%」均为该轮自评，**口径不统一，不可跨轮比较**（[聚合者推论] 仅陈述口径未统一这一事实，不给合并值）

**证据**: §129.7、§134.2、§150a.1、§161、§167b.1、§169.2、§170、§174、§175b、§179、§193、§196、§199、§206

**原文片段**: 「B-718 CLI 的 sys.path bootstrap 在 p79.* import 之后 (绝对路径调用会失败) 只有 codex 靠真跑 python3 /tmp/... 复现抓到, Claude 与 gemini 静读都漏」

---

## 32. Cross-AI 输出质量：幻觉率 / 行号偏移 / FAIL 模式

**当前值**（逐条，各带轮次）:
- **gemini 编造率最差一次（§199 A1.2 cold-start）**: **9 条 findings 里 5 条经 grep 证实为编造**（G1 假的 archive 行 / G2 假的 `delta:[0,300]` / G4 假的 while 循环 / G7 假的 regex 行 / G9 把设计选择当 bug），只有 **3 条留用**（G3 B0 sampling defense 不对称 / G6 RGBA composite dead path / G8 allowlist 刚性）
- **gemini hallucination 单例（§160.1）**: G-7 声称 sm_121 架构 'doesn't exist' — **REJECTED**（CLAUDE.md 已验证 sm_121 真实）
- **gemini false positive（§196）**: F1 lex-timestamp 与 F4 auto-digest reorg 都已被 B-389 / B-391 inline 注释关闭，gemini 没看到 → **9 findings 中 2 条作废，其余 4 OOB 有效**
- **gemini 事实错误（§182）**: 声称 `queue_phase1_paper_grade.sh` 缺 `paper_grade=True` — 实际 master orchestrator 确实 export；codex 把 scope 精确到 **leaf queue 的 `init_paper_grade_env` 不 export**
- **gemini 数学错误（§177）**: SE floor de-weighting 攻击里 **1/√N 算成 0.07pp（应为 7pp）**，且 **FE weight 1/SE² 的方向搞反**
- **gemini 行号准确率**: §197 「spot-check 约 **50% 错**」（RAPLReader 被标在 metrics.py 实际在 energy_tracker.py；声称 metrics.py:539-543 实际是 399+471）；§118「3/5 spot check 里偏 **50-70 行**」；§104「偏 **7-57 行**（60% file coverage）」；§215「C-F4 引 L253 实际 L286（差 33 行），attack 本身成立 → 保留为 B-890 并打 ⚠️ 标记」
- **codex hallucination（§174.6）**: B4 声称 `test_step_schema_validation_required_fields` 会因缺 `locator_route_meta_primary/retry` 而 fail —— **实测 PASS**（fixture line 244/245/249/250 已含）；B1 reddit 数字**报 150 实际 84**
- **Claude 数量误判（§181）**: 声称「all 12 aggregators」消费 `cost_usd`，codex empirical grep 显示实际只有 **3 个**（`aggregate_cost_electricity.py` / `aggregate_cross_site.py` / `aggregate_phase1_full_prereg_decision.py`）—— **mental model 偏 4 倍**
- **codex 整轮 FAIL（§190）**: 3 次重试全在 `exec /bin/bash` 探索循环耗尽 token，139KB 输出未到 synthesis
- **gemini silent-fast FAIL（§205）**: 首次 **16s + 1-byte empty**；需 `GEMINI_KEEP_RAW=1` wrapper 才成功
- **gemini CLI wrapper 三种调用的 token/延迟（§155.2）**: Trial A plan-mode **326ms / 0 tokens**（cache hit, 假成功）；Trial B `--yolo` 直接 **2207B 输出**；Trial C `--yolo` wrapper **1523ms / 4940 tokens**（真 call）
- **v7.8 Phase 4 claim-realness 首轮（§200）**: 跨 Mode B + Mode C 共抽 **7 条引用 spot-check，全部 PASS，本轮 0 条 file:line / field_name / function_name 幻觉**
- **Mode A 自审否定自己的候选 finding（§196）**: A3「5000-byte DOM cap 会截断 auth marker」实测 marker 在第 563 字节 → file 之前自我 downgrade

**caveats**:
- §199 处置: 「用户选 option C『接受 v2 部分结果 + 人工过滤』; 结论是 gemini 需 **tool-call provenance 强制**」
- §160.1: 「同轮 gemini 也贡献了 2 个 P0 unique catch; **hallucination 与 high-value 并存**」
- §174.6: 「推测原因是 codex 做脑内推演而非真跑 pytest; 结论是 **launch-blocker 级 claim 必须实跑验证**」
- §197: 「设计/estimand 层的 4 个 OOB 仍然有效; 合并时标注『行号近似』」；「wrapper 报 2248ms / 6888 tokens 但那是 **API 调用延迟不是 wall time**，实际约 3min」
- §200: 「**单轮结果**; 与 §199 gemini 5/9 编造形成对照」
- §155.2: 「Trial A 的 0 token 是 **silent block 症状** — gemini 读不到 paper drafts 只能 hallucinate narrative」
- §104: 「gemini line numbers 偏 7-57 行, 但 substantive findings 全 code-verified」
- §112: 「gemini 只有 4 findings 未达 pre-fire >=8 阈值 (PASS-with-caveat), 但其**唯一 unique 是 launch-blocker 级**」
- §133、§137: 「gemini 24s 可疑快但 specificity 扎实（直接 quote prereg L131-132）」/「1.6s 可疑快但直接 quote intro L38 与 prereg §2.6」

**证据**: §104、§112、§118、§133、§137、§155.2、§160.1、§174.6、§177、§181、§182、§190、§196、§197、§199、§200、§205、§206

**原文片段**: 「9 条 findings 里 5 条经 grep 证实为编造 ... 只有 3 条 (G3 / G6 / G8) 留用」；「Claude 声称 'all 12 aggregators' 消费 cost_usd, codex empirical grep 显示实际只有 3 个 — mental model 偏 4 倍」

---

## 33. Process 元发现（审计方法本身）

**当前值**:
- **user-caught OOB 全胜（§151.8）**: router v1→v4 四轮中，**v2→v3 user 抓 3 个 OOB（P0-8/P0-9/P0-10）三家 AI 全漏**；**v3→v4 user 抓 1 个 OOB（P0-11 archive ≠ independent test）三家 AI + Claude self-stress 全漏** → **4 个 OOB 全部 user-caught**
- **propagation asymmetry（§143.5, 3 AI 一致确认）**: 叙事层（prereg prose + paper drafts + planning docs）已完全对齐 paper-1 scope freeze；**实现层**（decision script + make gates + manifest registry + per_task_sr producer + configs + power_analysis K rules + snapshot Gemma support + queue collision）**在 10+ 处滞后**，直到 B-117~B-129 修完
- **审计面三层结构（§143.10）**: 完整 audit surface = (a) prose/preregistration + (b) implementation/scripts/configs + (c) protocol/license/governance；cross-AI 覆盖 (a)+(b)，folder-pass Claude review 覆盖 (c)；**三者都需要才算 paper-grade lock ready**
- **sibling-script propagation 实测（§133b.1）**: Bug 2 (SOM_MARKS regex) 存在于 **3 个脚本**（v5 的 7 文件 scope 只看见 1 个）；Bug 5 (model revision pin) 存在于 **5 个脚本**（v5 只看见 2 个）；**grep 耗时约 30 秒**
- **B-number 碰撞（§199/§200）**: 到 §200 为止共 **8 次碰撞**（A1.6b / A1.18-re / A1.7 / A1.11 / A1.8 / A1.17 / A1.2 / A1.16）；预留缓冲建议从 **≥10 → ≥20 (§197) → ≥25 (§199) → ≥30 (§200)**；A1.18-re 的 B-number 两轮 rename **总位移 +27**
- **2026-05-12 死磕日 tally（§128.6）**: 11 commits (ca3c509 → 951d56e)；6 个 Stage 4 methodology bug 被抓（4 修 Bug1/2/3/5 + 2 文档化 defer Bug4/6）；4 cells 数据 land（359511/359512 cellhprompt, 359719/359720 rand）；watcher 3 个 silent-miss patch；**8 distinct weak claims（4 共识 + 4 codex-only + 4 Claude-only）**
- **tests/ 的 source-grep theater（§187）**: **128 条 source-grep 断言**分布在 14 个 `test_stress_a1_*.py`；**behavior retrofit ratio 3.9%**；README 写「81 测试」实际 **436**；**15 处 importorskip**

**caveats**:
- §151.8: 「结构性结论: **cross-AI stress 是 floor 不是 ceiling** — user 持完整 project context ... 能 surface 跨文档 OOB; design-mode /stress 必须有 user 介入 review 的 turn, 这是与 audit-mode /stress 的核心 process 区别」
- §143.5: 「该不对称是 3 AI 共识项（最高置信）; 教训 = prose 改动时 code 必须同步移动」
- §143.10: 「是**过程观察不是实验测量**」
- §128.6: 「8 distinct 与 4+4+4=12 的表述在原文并存（共识项被双计）; 计数为自评」
- §199: 「标准补救是**高位到低位的原子 sed rebase**（源集与目标集不相交, 必要时经 B-9XY 临时区间）; 测试函数名 renumber 要以下划线后缀锚定」
- §187: 「P0-1 source-grep theater 最终被 **user 判定不修不 disclose（风险接受）**」

**证据**: §128.6、§133b.1、§143.5、§143.10、§151.8、§187、§199、§200

**原文片段**: 「4 个 OOB 全部 user-caught, 3 AI 一个没抓到」；「叙事层 ... 已完全对齐 paper-1 scope freeze; 实现层 ... 在 10+ 处滞后」；「128 条 source-grep 断言分布在 14 个 test_stress_a1_*.py; behavior retrofit ratio 3.9%; README 写 '81 测试' 实际 436」

---

## 34. GLM sidecar cluster（A1.15b）

> 🚫 **整个 cluster 已不存在，2026-08-01 补标。** GLM rescue 在 §210（B-991）物理删除，
> digest 管线按 §196 Q4 全部退役，6 个 cron sidecar 随之下线；现存等价物 = `/diag` skill。
> 以下计数与审计结果只作历史 provenance，**不描述任何在运行的东西**。

**当前值**:
- **三 chunk 累计**: α (commit d6dd949) **4 fix B-841~B-844 / smoke only / 30min**；β (commit 08d02fd) **3 fix B-845~B-847 / +33 invariant / 90min**；γ **7 fix B-848~B-854 / 70 targeted PASS / 75min**；合计 **14 fixes B-841~B-854 连续，12 个 unique 文件，~3h**
- **close rate**: **22 unique findings surfaced，16 closed / 4 deferred = 73%**；16 fixes B-841~B-856，~4.5h，4 chunks
- **error_scan**: `PATTERNS 12 → 11`；`error_scan.py --hours 24 --skip-system-checks → 0 errors found`
- **GLM digest 层 phantom mode 覆盖缺口**: 4 个 phantom mode（phantom_som/dom/text/prompt）落到 base prompt，无 mode-specific 字段也不加载标注图 → **Phase 1a 6-mode grid 有 50% condition 只拿到泛化叙述**
- **Chunk γ 7 项运维卫生修复**: B-848 error_scan MAX_TAIL_BYTES **200KB → 2MB**（10x headroom, 最坏 ~300MB 瞬时）；B-849 auto_pull Phase 4 quarantine 早退 + 高优先级 ntfy；B-850 error_scan 注释与 `MAX_HITS_PER_FILE//2=2` 不符（提取 `_PER_KIND_CAP` 常量）；B-851 notify_on_fail 删无效的 sed 反引号转义；B-852 digest_enrich site 默认值 classifieds 改锚定正则推断；B-853 glm_cell_autoupdate frontmatter 改 **temp+rename 原子写**；B-854 automation_overview 退役 digest 的文档漂移
- **B-882 digest cleanup gap 实证**: 文件系统上 **0 个 digest 文件 + 0 个 aggregator consumer**（B-743 retire 已完全传播）→ 降为 **disclose-only + 防御性注释，不写代码**

**caveats**:
- 「phase1_plan line 103 本已把 A1.15b defer 到 post-workshop（operational layer, not paper-grade blocker）; 本轮是 pre-audit 看有无 P0 藏在运维层」
- 「0 errors 与退役后预期一致」
- 「close rate 是 post-δ 快照; Chunk ε 未跑」
- 「仍余 13 项 P1+P2 backlog（~5-6h），Chunk δ 候选是 ntfy rotation / incremental scan / glm_client 抽取 / 原子写扩展 / detect_pid argv / PID-reuse race」
- digest 修法: 「`_normalize_obs_mode` 中心化映射（phantom_som→som, phantom_dom/text→dom, phantom_prompt→dom），原始 obs_mode 在 case record 里保留」

**证据**: §201、§202、§203、§204.6、§204.7、§206.5；`scripts/maintenance/glm/error_scan.py`、`glm_batch_digest.py`、`glm_client.py`

**原文片段**: 「22 unique findings surfaced, 16 closed / 4 deferred = 73%; 16 fixes B-841~B-856, ~4.5h, 4 chunks」；「4 个 phantom mode ... 落到 base prompt, 无 mode-specific 字段也不加载标注图 → Phase 1a 6-mode grid 有 50% condition 只拿到泛化叙述」

---

## 35. pytest 套件规模演化（快照链，非单调）

**当前值**（各 § 报的当时快照，**原样并列，不排序不插值**）:

| § | 计数 | 备注 |
|---|---|---|
| §141.12 | 88 → **129** | pre-§141 是 88，+41 invariant test |
| §146.4/§146.6 | 173 → **177** | Commit D +12, Commit E +4 = 16 new；100% pass 0 regression |
| §155.3 | **286 passed / 1 unrelated fail** | fail = `test_phase1_prereg_gate.py:196` 平行 session B-184 数值 expectation drift |
| §161.5 | **316/316 PASS** | 含 35 个新 negative tests |
| §166 | 333 collected / 5 FAILED → **398 passed / 0 failed / 8 skipped** | +65 tests |
| §167b.3 | **406/406 PASS** | 398 baseline − 1 stale-deferred + 15 新 A1.10 invariants |
| §170 | **414/414** | |
| §177 | **420/420** | |
| §181 | **421/421** | |
| §182 | **428 passed / 8 skipped** | 每个 chunk 边界 |
| §184 | **427 passed / 8 skipped**, 1 deselected | deselect = submodule SHA drift, 平行 session 遗留 |
| §185 | **428 passed / 8 skipped** | |
| §186 | **436 passed / 8 skipped / 1 deselected** | |
| §187 | 428 → 443 → **467** | |
| §190 | **500 passed / 9 skipped** | |
| §193 | 500 → **530** | |
| §194 | **566 passed / 9 skipped** | +36 invariant |
| §195 | **611 passed / 9 skipped** | +26 |
| §196 | 566 → 585 → **641** | 49 invariant tests |
| §197 | **674 passed / 9 skipped** | 33 新 invariant 全过 |
| §199 | **707 passed / 9 skipped** | +33 |
| §200 | **733 passed / 10 skipped** | 新测试 26 pass / 1 skip (HF_TOKEN) |
| §202 | 733 → **766 PASS** | Chunk β 33 新（B-845 11 + B-846 14 + B-847 8） |
| §204.6 | **791 passed / 10 skipped / 0 failed** | was 766 post-Chunk-γ；+25 = Chunk δ |
| §206.6 | **811 / 0 / 10** | pre-A1.24 791 + 20 new；`test_stress_a1_24_clear_tasks.py` ~375 LOC / 20 invariants |
| §205.4 | **852 passed / 10 skipped / 0 failed** | pre-A1.23 = 811；`test_stress_a1_23_concurrency_fixes.py` 41/41 |
| §176 | shell 专项 `test_paper_grade_gates_shell` 20 → 24 → 28 → **37** | A1.14 4 chunk |

**caveats**:
- 每条都是「某一 commit 时点快照」，**§ 序与计数不单调**（§186 报 436 而 §187 报 428→467），原因是 **4+ 并行 Claude session** + deselect + 平行 session 遗留 drift（见 ⚠️C7）
- §175b: 「只跑了 2 个 test 文件而非全套件 — §182 指出这正是 **B-550 regression 逃逸的原因**」（台账标 `superseded_by: §182`）
- §210: A1.15b chunk-scope 58 invariant（β 33 + δ 25）PASS / 0 fail + 11/11 inline Python spot-checks PASS，「**只覆盖 glm substrate 模块**」

**证据**: 表内 § 全部

**原文片段**: 「791 passed / 10 skipped / 0 failed (was 766 post-Chunk-γ; +25 new)」；「只跑了 2 个 test 文件而非全套件 — §182 指出这正是 B-550 regression 逃逸的原因」

---

## 36. Condition / OSF lock 计数

**当前值**: OSF lock 的 condition 数 **36 → 42**

**演变**: §177 Chunk 2 修正；「与 B-264 的 **42 = 36 Pass-1 + 6 Pass-2** 一致」

**已作废**: 36

**证据**: §177；`docs/checkpoints/pre_run/osf_lock_manifest.md`

**原文片段**: 「36 → 42」；「与 B-264 的 42 = 36 Pass-1 + 6 Pass-2 一致」

---

## 37. Registry / manifest 校验状态

**当前值**:
- run manifest 校验器 pre-fire 报 **22 errors**（0 个 paper-grade entry + 2026-05-04 BULK ARCHIVE section/grade drift）
- B-184 producer live 状态: `run_manifest.yaml` 里**没有任何 cell 含全 6 mode**（Phase 1a rerun 仍 in flight）→ 输出 **INSUFFICIENT_DATA**，producer exit 0 不 block `make analysis`

**caveats**:
- 「原文标为 **predicted by design**（pre-fire 本就没有 paper-grade entry）」
- 「Phase 1a 数据 land 后会自动评估; 这是**状态**不是结论」
- 「1 个 pre-existing GRL submodule SHA drift 不算 A1.21 责任」

**证据**: §150a.8、§177；`scripts/analysis/validate_run_manifest.py`、`results/phantom_paper/phase1_prereg_gate.csv`

**原文片段**: 「22 errors (0 个 paper-grade entry + 2026-05-04 BULK ARCHIVE section/grade drift)」；「输出 INSUFFICIENT_DATA, producer exit 0 不 block make analysis」

---

## 38. archived-data 分析脚本的两个数学 bug（codex Round C）

**当前值**:
- **C-1** `aggregate_phantom_lift.py:623-632` 混 denominator（`sr_4_psom` 在 `u_psom` 上算而 `sr_3` 在 `universe_5` 上算）→ **point lift 与 CI desynced**
- **C-2** `:583-587` H3 axis-2 测的是 `universe_5` 而非 P-prompt/P-SoM common universe → **axis-2 unique count 系统性向下偏，可能翻转 H3(ii) 成假阴性 → 错误地把 R1 降级 R2**

**caveats**: 「不阻塞 Phase 1a launch（该脚本只用于 archived 分析）但影响 **Appendix D**; §134.2 F4 再次独立发现同一 denominator 问题并 **defer 到 post-Phase-1a**」

**证据**: §133.2、§134.2；`scripts/analysis/aggregate_phantom_lift.py`

**原文片段**: 「C-2 :583-587 H3 axis-2 测的是 universe_5 而非 P-prompt/P-SoM common universe → axis-2 unique count 系统性向下偏, 可能翻转 H3(ii) 成假阴性 → 错误地把 R1 降级 R2」

---

## 39. `_shared_vl_utils` prompt byte-identity

**当前值**: Qwen / Gemma / `_shared_vl_utils` 三方 prompts 验证**完全相同（byte-identical）**

**证据**: §146.4/§146.6；`p79/agents/_shared_vl_utils.py`

**原文片段**: 「Qwen / Gemma / _shared_vl_utils 三方 prompts 验证完全相同 (byte-identical)」

---

---

## ⚠️ 矛盾清单

**⚠️C1 — bootstrap B 到底哪边是 10000**
- 侧 A（§152.2）: paper §3.5 披露 canonical **B=1000**；analyze_run 内部 **B=10000**；prereg 明锁 B=1000 用于 prereg gate。原文判「**两处并存不可混用**」
- 侧 B（§165.2, gemini）: **paper §1:7 prose 写 B=10000**，preregistration + code 是 **B=1000** → 判 submission-gate 级
- 侧 C（§186, codex F3）: **bootstrap 代码 B=10000 vs prose B=1000**（方向与侧 B 相反），且「observed-n 与 prose 声称的 paired 双重不匹配」
- 三侧涉及的文件不完全相同（paper §1 / paper §3.5 / analysis.py），台账**没有一处把三者拉齐**说明哪个 producer 用哪个 B。**并列，不选边。**

**⚠️C2 — DOM 字符规模的两个"上界"**
- 侧 A（§139.4）: 57435 个 `observation_dom.txt` 全量，viewport filter 后 **median 3306 / p99 7656 字符**（B0+B1, cls+red+shop）
- 侧 B（§153.2）: router archive simulation，**cls DOM 最大 5512 chars，red 最大 6963 chars**（model unspecified, n unspecified, 仅 cls+red）
- [聚合者推论] 两侧 population 不同（侧 A 含 shopping、侧 B 不含），因此**未必真冲突**；但台账两处都被当作「DOM 有多大」的证据且从未互相引用。**并列。**

**⚠️C3 — 2026-05-12 weak claim 计数（原文自承并存）**
- 「**8 distinct weak claims**」 vs 「**4 共识 + 4 codex-only + 4 Claude-only**（=12）」，caveat 自注「共识项被双计」。**并列。**

**⚠️C4 — router design v1 stress 的 finding 总数（原文自承并存）**
- 「**22 findings**」（= 7+8+7 raw） vs 分类「**4 三家共识 / 4 两家 / 7 独家**」（=15）。caveat 自注「原文并列两种计数」。**并列。**

**⚠️C5 — VWA submodule SHA 链条中 `eb5cbd8` 的位置**
- 侧 A（§157.2 caveat）: `f0c835b` →「后续 A1.25 / A1.18-re 多次 bump 到 **c1765ee → 1c3a615 → 2f9b0b4**」
- 侧 B（§178.3）: 「HEAD **eb5cbd8 → 1c3a615**；commit count 6 → 8」
- `eb5cbd8` 未出现在侧 A 的链条里，`c1765ee` 未出现在侧 B。**台账没说清哪条链完整。并列。**

**⚠️C6 — B-number 区间重叠（A1.12 cold-start vs A1.14）**
- §187 A1.12 cold-start: 「8 fixes **B-662~B-674**」
- §188/§189/§191/§192 A1.14: 「20 fixes **B-672~B-683** + B-703~B-710」
- **B-672 / B-673 / B-674 被两轮同时占用**。§199/§200 记录到 §200 为止共 8 次 B-number 碰撞（含 A1.12），但**没有说明这一段最终归属谁**。**并列。**

**⚠️C7 — pytest 全套件计数非单调**
- §186 报 **436 passed**，其后 §187 报 **428 → 443 → 467**；§184 报 **427**，其前 §182 报 **428**
- 台账把原因归为「平行 session」「deselect」「submodule SHA drift 遗留」，但**没有给出任一时点的权威计数**。**并列，不排序成单一曲线。**

**⚠️C8 — A1.9 cold-start 的 finding 分母（原文内部不一致）**
- 同一条记录里写「**19 unique** = 4 P0 + 9 P1 + 5 P2 + 1 GRL prose disclosure」与「**OOB 12/18 = 67%**」——分子分母的 total 一处 19 一处 18。**并列。**

**⚠️C9 — figure 数量单位**
- §173b 的 scope 写「**21 figure scripts**」，同批另一条写「**12/26 figures (46%)**」。script 数与 figure 数是否同一集合，台账未说明。**并列。**

**⚠️C10 — B0 GLM rescue 率的两个小数**
- §141.5 「**1.488%**」 vs §142.3/§170.2 「**1.49% (453/30437)**」。§142.3 caveat 自称「与之一致」，但两个数字字面不同且未给 §141.5 的分子分母。**按原样并列，不做四舍五入判断。**

---

## 未归主题的孤条

- **§133.3**「Round 2 commit (e9ddbe3) 自评 safe to push 的实际错误率 = Round A 独立验证发现 6/7 finding 真实存在」—— 已并入主题 19，但其「自评 safe to push 的错误率」这一**元指标**本身无同类项。
- **§155.2** gemini CLI wrapper 三 trial 的 token/延迟实测（326ms/0 tokens、2207B、1523ms/4940 tokens）—— 已并入主题 32，但属工具链测量而非审计产出。
- **§150b.1** visual hijack probe 三轴触发条件、**§150b.1** has_ref_image -3.6pp 反转 —— 台账明标「引用而非本处测量」（源在本 batch 行范围之外的 §100/§101/§M1 与笔记 line 2305），已作为主题 11 的附属证据保留，但**不属于本批次的原始测量**。
- **§164 (正文标 §159.2)** 的编号异常本身: 「§164 是从重复的 §159 renumber 而来, 正文 sub-header 仍写 159.x」—— 编号治理事实，无同类项。
- **§173b (A1.20)** 的 §-编号写法「§173b (A1.20) 173.1-173.2」与 §175b「§175b (A1.5b) 175.2 + 175.7」—— 同一日多 session 并行导致的 § 编号分叉，无同类项。
- **§158.5** 「strict=True 时缺文件 raise FileNotFoundError 而非静默 0」—— 已并入主题 9，但它同时是主题 23（静默失败）的修复项，跨两个主题。
