---
type: conclusions
batch: D4
status: done
created: 2026-07-28
source: scratchpad/batches/D4.jsonl (217 条 MEASURED，§312.2–§397.10，2026-06→07-28)
---

# 测量结论 D4 (§312.2–§397.10)

> **读法**：每个主题的「当前值」是现在算数的数字；「已作废」的数字**禁引**。
> **本批是全台账出错最多的时段** —— §397.10 是 CORRECTION 节，作废 §397.4 与 §397.9 的部分结论并追加 (4)(5)。
> 凡涉及这两节的条目已在下方 **⚠️ 被 §397.10 修正的条目** 单独成节，且在主题内就地标注。
> **noise 类数字一律各自带 scope 并列，禁止做加减法**（§397.10 明令）。

---

# A. 假设门槛结果 (H1 / H2 / H3 / H10)

## A1. H1 pooled drop-one oracle lift — 主门槛，FAIL

**当前值 (k=6, post-AMENDMENT_08 universe 1281 = 3×224 + 3×203)**:
θ_FE = **0.7897 pp**，**p = 0.807**（vs +1.0pp 门槛），**I² = 0.0%**，**gate FAIL**；
per-cell drop-one **0.0–1.3pp**。
[聚合者复核 — 台账 k=6 只给区间，逐格值取自 `results/phantom_paper/phase1_full_prereg_decision.json`：0.8929 / 0.9852 / 1.3393 / 0.0 / 0.4464 / 0.4926；同文件另有 transparency 口径 normal-approx p = 0.7208，台账引的 0.807 是 bootstrap primary。]

**演变**:
- §360.3 (k=3，全 cls): θ_FE = **+0.975pp [−0.047, 1.997]**, p_one_sided(δ=1.0) = **0.52**；per-cell B0 +0.89 / B1 +1.34 / B2 +0.45。`analysis_status=PARTIAL`，明标 do NOT quote as verdict。
- §378 (k=5，缺 B2·red): **+0.795pp [0.27, 1.49]**, p_boot = **0.743** — 不过线；per-cell B0·cls +0.89 / B0·red +0.98 / B1·cls +1.34 / B1·red **0.00** / B2·cls +0.45。`h1_verdict=NOT_EVALUATED`。
- §395.6 (k=6): θ_FE **0.7897**, p **0.807**, **FAIL**。
- §389.8: B-1898 落地后 **H1 数字一分未动**（0.7896872 / p=0.807 / FAIL）。

**已作废**:
- archive 预期的 reddit P-SoM **pooled +2.34pp (I²=0%)** 未兑现 —— 新 cell B0·red 实测只有 **+0.98pp**（archive 是 pre-reset 非 canonical 数据）。
- §360 写稿时点的 archive drop-one **+2.56pp**（cls P-SoM）同理不可引；同期 fresh cls B0 P-SoM 只有 **+0.89pp [0.00, 2.23]**，而 **P-prompt +2.68pp (sig) 反成 cls 最强 phantom 臂**。

**caveats**: k=3 / k=5 两版均 `analysis_status=PARTIAL`，正式 verdict 走 NOTE_06 授权 slotsheet；k=6 verdict 当时"实质押在 pending reddit"。

**证据**: §360.3 / §378 / §389.8 / §395.6；`results/phantom_paper/phase1_full_prereg_decision.csv|json`

**原文片段**: 「θ_FE = +0.975pp [−0.047, 1.997], p_one_sided(δ=1.0) = 0.52 — cls-only 恰低于 +1.0pp 门槛」(§360.3)；「drop-one 0.0-1.3pp, H1 θ_FE 0.7897, p=0.807, FAIL」(§395.6)

---

## A2. H1 的 SE floor 两套规则 (B-1898) — 实现 vs 预注册散文对不上

**当前值**: 实现口径 `ses<0.68`: n_floored **4**, θ_FE **0.7897**, z vs 1.0pp **−0.585**, gate **FAIL**；
预注册散文口径 `ses<=0`: n_floored **1**, θ_FE **0.6533**, z **−1.417**, gate **FAIL**。

**演变**: §388.3 发现两套并存 → §389.8 落地（散文同步 + 常量从函数局部字面量提到模块级），**H1 数字一分未动**，测试 1622 passed。

**已作废**: 无（两口径同判 FAIL）。但 §388.3 当时推荐的处置「(a) 实现对齐 prereg」已被 **RETRACTED §389.8**。

**caveats**（一字不改）: 「超额 flooring 其实偏向 H1 —— 它把 θ 最小且 SE 也最小的两个 cell 降权, 把 θ_FE 从 0.653 抬到 0.790。不改判决, 且按预注册散文重算 H1 败得更彻底。仍必报: 审稿人照 prereg 重算得 0.6533 而非论文的 0.7897, 数字对不上就是 kill 无论方向」；
AMENDMENT_03 称 canonical producer 是 "SINGLE source mirrored by" transparency producer，但两处都是局部字面量 → 该 mirroring 从来无法被检验，只是碰巧相等。

**证据**: §388.3 / §389.8

**原文片段**: 「实现 ses<0.68: n_floored 4, θ_FE 0.7897, z vs 1.0pp −0.585, gate FAIL; 预注册散文 ses<=0: n_floored 1, θ_FE 0.6533, z −1.417, gate FAIL」(§388.3)

---

## A3. H3 双轴 — 两轴 PASS，但被自己标为弱证据

**当前值 (k=6)**:
- **axis-1 = 1.3528 pp**, p = **1.19e-05**, **5/6 cell 通过**
- **axis-2 = 2.0877 pp**, p = **7.52e-07**, **5/6 cell 通过**
- framing 仍 R5→C_prime_structure
- 相关：P-SoM 独解 **6 cls + 3 red**

**演变**:
- §360.2 (预注册估计量实测): P-text **9 tasks / 4.02pp** + P-prompt **16 tasks / 7.14pp**（用于替换被 stress 打掉的 "most of solvable mass" 表述 → 改为 "36/97 outside best arm"）。
- §360.3 (k=3 interim): axis-1 **+3.20pp [1.58, 4.82]**（k=2，B2 被 ≥2-task floor 排除）；axis-2 **+2.26pp [1.14, 3.38]**（k=3）；CI 均排除 0。
- §366 PROTOCOL_NOTE_05 修复（H3 pool 恢复 over the 6 planned cells）后 dry-run: axis-1 **1.08pp [0.47, 1.98]**（CI 仍排除 0），axis-2 数字不变。
- §378 (k=5): axis-1 **+1.26pp [0.68, 1.99]** p=0.0 (4/5 cell pass)；axis-2 **+2.60pp [1.68, 3.63]** p=0.0 (5/5)。
- §388.2 (universe 统一为 1281 后): axis1 **1.341→1.3528**, axis2 **2.108→2.0877**，两轴仍 PASS。

**已作废**: §360.3 的 axis-1 **3.20pp** 与 draft 里所有旧 interim 数字 = stale-by-correction，禁引。

**caveats**:
- §397.8 明确把两轴标为**弱证据**：「门测 ≠0, 而同策略两次跑本来就 ≠0」。
- §397.10(3) 给出 self-oracle noise 参考值 **6.7 / 7.6pp**（B0·cls vision clean pair）**比 H3 轴大 3-5 倍**，但该值是 **B0-MoE 上界，不可直接外推到本地确定性 backbone** —— 两侧 scope 不同，**禁止做加减法**。
- §397.10(2) 的推论（axis-1 两臂同 id-regime / axis-2 跨 AMENDMENT_07 id-regime 边界，可能解释 axis-2 > axis-1）原文明标**尚未验证**。

**证据**: §360.2 / §360.3 / §366 / §378 / §388.2 / §395.6 / §397.8 / §397.10

**原文片段**: 「axis-1 +1.26pp [0.68, 1.99] p=0.0 (4/5 cell pass); axis-2 +2.60pp [1.68, 3.63] p=0.0 (5/5)」(§378)；「H3 axis-1 1.3528pp p=1.19e-05 (5/6 cell 通过), axis-2 2.0877pp p=7.52e-07 (5/6 cell 通过)」(§395.6)

---

## A4. H2(a) per-task paired median cost ratio — falsification-unrefuted

**当前值 (k=5 报数)**: **5/5 within 1.20× band, 0 falsified**；五格独立口径 **1.01 / 1.04 / 1.08 / 1.00 / 0.98**（标 [A] = 已核）。
k=3 实测三值: **1.0085 / 1.0373 / 1.0765**（全在 1.20× 阈内）。

**演变**: §360.3 从「0/1 cells with data」升级到 3-cell 实测 → §378 k=5 → §388.2 universe 从 205 改 203 且**空集 fail-open 修复**。

**已作废**: §360.1 P0-② 指出的旧写法 —— 「用边际均值比 1.036 冒充 per-task paired median ratio」+ 「把 producer 的 0/1 cells with data 空洞真写成 not falsified」。

**caveats**: 用 **falsification-unrefuted** 措辞（§380 Mode C 改），**不是 "verify"**；page-screenshot tokens = 0 by construction；interim 非 verdict。

**证据**: §360.3 / §378 / §382 / §388.2；`docs/checkpoints/deliverables/router_baseline_master_summary_2026-07-16.md`

**原文片段**: 「1.0085 / 1.0373 / 1.0765, 全在 1.20× falsification 阈内 → unfalsified」(§360.3)；「falsification-unrefuted 措辞 (§380 Mode C 改), 不是 'verify'」(§378)

---

## A5. H10 / Pass-2 router 门槛 — 管道两个月是断的，结论恰好同向

**当前值**: **1/5 cells fully trained**；每 cell **Pass-2 runs = 0** → `k_of_n = "0/0"` 是**独立成因**；`operational_gate_passed = False`。
5-of-6 门槛下只有一个 cell 出完整模型，即 **"structurally unattainable" 比稿子写的更强**；**稿子完全没提 Pass-2 从未 fire**。

**演变**:
- §361.3 rehearsal: 训练链 3 段在隔离 `l1_router_rehearsal_20260702` 全通，`h10_entropy_gate.json` 正常 emit（status=ok, global_min **2.10 bits** > 1.0 DEFER 阈值）；**B2_classifieds router 不可训** —— best-mode 标签仅 **16 task** → 5 fold 全 `insufficient_train_data`。
- §383.2: canonical 目录 `results/phantom_paper/l1_router/` 停在 2026-05-18 的 `{"status":"no_data_yet","n_total_tasks":0}` 占位符，Pass-1 落地后**从未刷新**；真实 Stage-1/2/3 全跑进带日期的兄弟目录 → canonical `h10_entropy_gate.json` **从来不存在**，verdict 一直走 fail-closed 报 `h10_status=entropy_unavailable`。
- 重生后 verdict diff: `h10_status` entropy_unavailable→ok，`entropy_defer_reason` 清空，**结论不变**。

**caveats**: 「两个月没人发现因为**管道理由与科学结论恰好指向同一结果**」；§361.3 当时的预判「若 B2_reddit 同地板, H10 的 ≥5/6 判据数学上最多 4/6 → §6 descriptive 分支概率高; B-1753 解耦下 H10 fail 不伤主线」。

**证据**: §361.3 / §383.2；`results/phantom_paper/l1_router/`、`docs/checkpoints/pre_run/h10_artifact_regen_provenance_2026-07-22.md`

**原文片段**: 「results/phantom_paper/l1_router/ 停在 2026-05-18 的 {"status":"no_data_yet","n_total_tasks":0} 占位符, Pass-1 落地后从未刷新」(§383.2)

---

## A6. 计分 universe / 分母 (AMENDMENT_08)

**当前值**: reddit **collection 205（不变）/ scoring 203**；cls **224 无排除**；三假设统一 universe = **1281 = 3×224 + 3×203**。
task universe SHA: cls `b0f3b8b0…` 逐字节不变；reddit `41b1a918…`(205) → `1ce29c8b…`(203)；`tiers=()` 精确复现旧 SHA。

**演变**: §387.15.1 拆分 collection/scoring → §388.2 发现「同一次运行同一个 JSON: H1 n=203 而 H2(a)+H3 两轴 n=205」→ 修后三者同为 1281。

**已作废**: 所有以 205 为计分分母的论文级 SR / lift / oracle / figure。

**caveats**:
- collection 用于 validate_run / run_registry / paper_grade_check / active_processes / clear_tasks / glm cron；scoring 用于论文所有 SR / lift / oracle / figure。**两者不可混用**。
- 根因 `_six_arm_complete_case_universe`（B-948，当初就是为防 universe drift 而建）从磁盘上凑齐六 mode 的任务推 universe 而不从 `expected_scored_ids` 取 → 「AMENDMENT_08 §3 那张表**被自己的产物证伪**」；「审稿人照文中的 203 重算 H3 会拿到不一样的数」。
- 敏感性臂是**比较**不是重算；universe SHA 写进各产物 `outcome_provenance` 并互校 → 陈旧产物会被现有校验自动抓出。

**证据**: §387.15.1 / §388.2；`docs/analysis/cross_sites/amendment08_sensitivity.md`

**原文片段**: 「同一次运行同一个 JSON: H1 n=203 (post-AMENDMENT_08) 而 H2(a) + H3 两轴 n=205; 修后三者同为 1281 = 3x224 + 3x203」(§388.2)

---

# B. Router 负结果链（Paper B 主线）

## B1. locked LR router OOF replay — 被 SoM 双轴支配

**当前值 (B0_classifieds, n=224, 唯一完整 cell)**:
locked LR router **57/224 = 25.45%** vs best-single SoM **61/224 = 27.23%** → **ΔSR = −1.79pp**；
mean billed cost **0.07555164** vs SoM **0.07235667** → **+4.42%**；
→ **点估计被 SoM 在 SR+cost 双轴支配**。oracle **97/224 = 43.30%**（比 best-single 高 **16.07pp**）。

**bootstrap 稳健性 (§371)**: 5000 次 task-paired bootstrap 里 **SoM 支配 router 仅 72.24%**（双轴 strict **67.26%**）→ **不能升级成 ≥95% 稳健表述**。
B0 oracle **43.30% @ $0.06232** 双轴严格支配六个 fixed + locked router；**deployable frontier = Vision + SoM**。

**caveats**:
- 全部标 **OFFLINE / NON-GATE**，不触碰 canonical `l1_router/`，不替代 live Pass-2/H10。
- **B1_cls partial OOF n=180 仅与 subset SoM 持平 (12.22%)，不可冒充整-cell 数**；B1_cls (4/5-fold diagnostic) router 以同 SR 更低 cost **弱支配** subset SoM，但 bootstrap router-dominates 仅 **47.40%**。
- **成本不跨 cell pool**（B0 api_usd vs B1/B2 electricity-derived）。
- scope: fold-held-out mode + per-fold τ 映射回 paper-grade Pass-1 success 与 canonical `total_billed_cost_usd`。

**证据**: §367 / §371；`results/phantom_paper/l1_router_offline_20260715`、`docs/checkpoints/codex_outputs/router_pareto_reframe_2026-07-15.md`

**原文片段**: 「locked LR router 57/224 = 25.45% vs best-single SoM 61/224 = 27.23% → Δ = −1.79pp; mean billed cost 0.07555164 vs SoM 0.07235667 (+4.42%) → 点估计被 SoM 在 SR+cost 双轴支配」(§367)

---

## B2. 5×3 模型/特征 sweep — 负结果对假设类稳健

**当前值**: **15/15 组合均未超过 SoM 61/224 = 27.23%**；最高仍是 locked LR/MI-18 的 **57/224 = 25.45%**，次高 MLP/full-50 **56/224 = 25.00%**；全量 TF-IDF（每折 119-131 维）也未翻盘。
审计覆盖 **15 组合 / 75 folds / 3360 OOF task records**。

**caveats**（一字不改）: 明标 **NOT the preregistered router, NOT H10-eligible**；scope = 同一 fold map (SHA `91ea10b4...cfd`) / `min_class_n_train=10` / per-fold inner-CV τ + P-SoM fallback / **OFFLINE NON-GATE POST-HOC EXPLORATORY**；
「负结果不是单纯 LR 假设类过弱, 更符合**路由信号偏弱** 和/或 **N=97 单次 oracle 标签噪声/支持不足**」。

**证据**: §368；`docs/checkpoints/codex_outputs/router_model_sweep_2026-07-15.md`

**原文片段**: 「15/15 组合均未超过 SoM 61/224 = 27.23%; 最高仍是 locked LR/MI-18 的 57/224 = 25.45%」(§368)

---

## B3. post-hoc τ 曲线 — 唯一"赢"的配置，但禁止进 H10

**当前值**: 六个二元 OOF 成功头 + fold-train mean cost 决策的 post-hoc τ 曲线，**τ=.10 得 29.91% @ $0.07051，严格支配 SoM 与 locked router**；τ=.05 得 **27.68% @ $0.066199**。

**caveats**: 「τ 曲线因**在同一 replay 上扫 21 个 τ** 明确**禁止进入 H10**」；「原 multiclass max-prob 不是每-mode P(success) 故没有冒充」。

**证据**: §371 / §374

**原文片段**: 「τ 曲线因在同一 replay 上扫 21 个 τ 明确禁止进入 H10」(§371)

---

## B4. prior-work router baselines (kNN / cascade / random floor)

**当前值 (B0 / classifieds / n=224)**:
- **kNN 最好 k=5 = 26.79% @ $0.070682, PGR −0.028** → 双轴严格支配 locked LR (25.45% @ $0.075552, **PGR −0.111**)，但**又被** post-hoc τ=.05 (27.68% @ $0.066199) 与 τ=.10 (29.91% @ $0.070513) 严格支配；k=10/20 降到 **22.32% / 21.43%**。
- **observed confidence cascade 无升级价值** —— primary mean-logprob 一旦 q>0 即**同时加成本降 SR**；四信号最高 SR 也只在 min-logprob q=.70 **追平 best-single 27.23%** 却累加到 **$0.269006/task (PGR=0)**。
- **random floor 约 20.05±1.84% / 20.01±1.77%, PGR ≈ −0.45**。

**caveats**（文献诚实边界，一字不改）:
「RouteLLM 原法是 query embedding + 指数相似度加权 Bradley-Terry preference router, 这里是 advisor 指定的 TF-IDF kNN 简化; FrugalGPT 原法含 learned scoring/cascade optimization, 这里是 independent-reset Pass-1 轨迹的 offline 累加模拟; 两者明确标 -style / not faithful reproduction; APGR 不硬迁移 (原积分轴是 binary strong-model call rate, 六臂 menu + 多轨迹累计成本无忠实同构轴) 故只报 pointwise PGR + ΔSR/ΔUSD; **cascade 是 observed post-trajectory 不是 pre-execution gate**」；confidence 覆盖 20646/20646 steps 与 1344/1344 task×mode episodes。

**证据**: §374；`docs/checkpoints/codex_outputs/router_prior_baselines_2026-07-15.md`

**原文片段**: 「observed confidence cascade 无升级价值 — primary mean-logprob 一旦 q>0 即同时加成本降 SR」(§374)

---

## B5. lit-mined escalation baselines（两条可实现的）

**当前值 (B0 / cls / n=224)**:
- Vardanyan-style DOM→Vision failure escalation: 升级 **117/224 tasks → 23.66% SR @ $0.107321/task, PGR −0.222**
- LazyMCoT-style length-trigger Vision→SoM: 升级 **104/224 tasks → 25.89% @ $0.108248, PGR −0.083**
- **两者都被 always-SoM 严格支配，未改变 deployable/hindsight frontier**。
开采规模: **92 条 prior work 逐项映射（110 个本地文件+行号引用，不足处保留 UNVERIFIED）**。

**caveats**: 只做两条，因其余不需臆造方法细节者稀少 —— WebRouter 的 verified bib 与 routing scan 对 routed object 有冲突且本地无 ca-VIB objective/training recipe **故拒绝捏造**；Read-More-Think-More 在单 model/预算 B0 上退化为已有 fixed policy；Adaptive Re-Ranking 的 oracle-vs-learned 形态已被现有 oracle + locked/cost-aware OOF 覆盖；两条 baseline 均**累计完整 independent-reset Pass-1 轨迹成本**，success 取最终 mode。OFFLINE NON-GATE。

**证据**: §376；`docs/checkpoints/codex_outputs/router_litmined_baselines_2026-07-15.md`

**原文片段**: 「两者都被 always-SoM 严格支配, 未改变 deployable/hindsight frontier」(§376)

---

## B6. cls→red 转移复现 (D1 全冻结 / D2 one-shot)

**当前值 (site=reddit, model=B0, n=205)**:
- **D2 效率式 router**: **23/205 = 11.22% @ $0.101037/task** vs best-single DOM **30/205 = 14.63% @ $0.101278** → **−3.41pp, −$0.000241/task**
- D2 vs DOM task-paired bootstrap: **ΔSR CI [−7.80, +0.98]pp**；**Δcost CI [−$0.00639, +$0.00613]**；router / DOM dominance rate = **4.3% / 45.2%**（5000 次，seed 20260716）
- **D1 全冻结转移**: **7.80% @ $0.097278**（200/205 选 Vision）

**caveats**: **τ\*=0.10 在读 red 结果前冻结**；D2 是「略便宜但明显掉 SR 的 non-dominated trade-off，**未复现 cls 的 joint gain**」；全部标 **OFFLINE / NON-GATE / RECIPE-FROZEN-ON-CLS**；未动 aaai27_main / prereg / gate / run_manifest，未 commit。scope 用 seed-42 site-shared 5-fold map, **205 exact universe**（pre-AMENDMENT_08），content SHA `078df2af…`。

**证据**: §379；`results/phantom_paper/l1_router_offline_20260716_red_replication/`

**原文片段**: 「23/205 = 11.22% @ $0.101037/task vs best-single DOM 30/205 = 14.63% @ $0.101278 → −3.41pp, −$0.000241/task」(§379)

---

## B7. which-mode router 的标签供给 —— 半败在标签而非算法

**当前值 (k=6, B-1904 重抽后)**: **6 cell 全 complete**；reddit **203/203/dropped=[58,160]**，SHA `1ce29c8b` MATCH；**pooled 249 → 260**；6 个 fold 全 `matches_canonical_scored=True`；**Stage3 训练结果与修复前逐格一致 → 1/6**。

**演变**:
- §367 (四 cell canonical 离线): 只有 **B0_cls COMPLETE 5/5**；**B1_cls 4/5**（fold 4 min-class filter 后退化）；**B2_cls 0/5**（16 labeled）；**B1_red 虽 6/6 condition 齐但只有 26 union-success labels**（DOM 14，其余 2-4），`min_class_n_train=10` 后每 fold 不足两类 → **0/5**。
- §383.4: 每 cell 可训练标签 **16-26 个**；`N_MIN_CLASS_TRAIN=10` 让 **3/5 cell 零可训练折**。
- §392.1: 重抽后仍 **1/6**。
- §396.2 codex #4（按实测成本重算 which-mode 标签）: **4/6 不可训变 5/6**（阴性结论反而更强）；§396.7 落地复现 —— 翻的是 **classifieds·B1**（存活类 2→1）。

**caveats**:
- 「**condition 完整 ≠ router 标签支持充分**」；原先只盯 B2_cls 的 H10 trainability 风险**扩成 3/4 cell 无完整 policy**。
- **1/6 不是回归，是 Paper B「which-mode 半败在标签供给」的实证**。
- provenance 修正: extractor 原无 `--manifest` 会 glob 抓旧 B1 3-mode run 成第 7 条，**首次 raw 产物作废**；最终 manifest SHA `9bb985be...ecb1`，四 cell 均 `manifest_used=true` / 6 runs。
- 检查顺序有讲究 —— **universe gate 必须排在 mode-completeness gate 之后**（缺 mode 是更基础的缺陷）。
- V2 tier 化（6 路→2 路"要不要截图"）类数减少但**同样救不回来** —— **绝对标签量才是约束**。
- codex #4 撞上 2026-06-09 已裁定的 **B-1806 (F2)**：报它的理由是替代定义下阴性更强，**不是采纳它的理由**。

**证据**: §367 / §383.4 / §392.1 / §396.2 / §396.7；`scripts/analysis/extract_50_features.py`

**原文片段**: 「教训 = condition 完整 ≠ router 标签支持充分」(§367)；「1/6 不是回归, 是 Paper B『which-mode 半败在标签供给』的实证」(§392.1)

---

## B8. oracle 拆两半：triage_only vs route_only

**当前值 (6 cells, k=6，B-1899 修正后)**:
- **triage_only**: SR **±0.00pp**，成本 **−9.5% ~ −30.6%**；需要二元「有没有 mode 能解」标签，**每 cell 全部 203/224**
- **route_only**: SR **+3.45 ~ +16.07pp**，成本 **−0.2% ~ −11.4%**；需要 which-mode 标签，**每 cell 仅 16-97 个**

**演变 / 已作废**:
- triage_only 成本半原报 **−38% ~ −45%** → **B-1899 推翻为 −9.5% ~ −30.6%**（虚高 **42.9% / 41.2%**；hopeless 子集 cls/B0 0.04438 vs 0.07778、red/B0 0.06144 vs 0.10454）。
- §387.16.3 的「成本这半值 40% 且标签充足所以有戏」**不成立，已加删除线**（§388.1）：修正后 **5/6 cell 里 cheapest 固定策略省得更多**（代价 −1.79 ~ −7.39pp SR）；**cls/B2 零 SR 代价地 Pareto 压制 oracle triage**。
- ⚠️ 台账原文把该修正指向「§387.17」但**笔记中不存在 §387.17** —— 实际修正记在 §388.1。

**caveats**: B-1899 根因 —— `router_objective_ordering.py` 三条策略对"无人能解"的任务用 `min(cost[m][t] for m in SIX_MODES)` = **每任务跨 mode 最小成本**，前者要求事先知道该任务在哪个 mode 上最便宜，**而这三条策略被宣传为"只需要一个二元 solvable/not 标签"**。全部 `post_hoc_exploratory=True` / `h10_eligible=False`。

**证据**: §387.16.3 / §388.1；`docs/analysis/cross_sites/router_objective_ordering.md`、`scripts/analysis/router_objective_ordering.py`

**原文片段**: 「triage_only: SR ±0.00pp, 成本 −9.5% ~ −30.6% (修正后), 需要二元『有没有 mode 能解』标签, 每 cell 全部 203/224」(§387.16.3)

---

## B9. cost-first 级联

**当前值 (6 cells, k=6)**: 拿到 oracle SR 的代价 **+341% ~ +463%**；attempts 直方图 cls/B0 `{1:56, 2:25, …, 6:133}` / red/B2 `{1:4, …, 6:189}`；**60-95% 的任务没有任何 mode 能解**。

**caveats**: 「级联在低 SR 区间是**灾难**(6/6 cell)」；§388.4 B-1900 指出 `cascade_cost_first` **原标 oracle-free 是错的** —— 其 break 条件 `succ` 是**评测真值**，部署时没评测器，真级联会在每个任务上跑完 6 个 mode。`post_hoc_exploratory=True` / `h10_eligible=False`。
⚠️ 本条所在的 §387.16 段另有一句已被 **RETRACTED §388.7.2**（「两个存活的 cell 一个都不过 Holm 校正」，见 B10）。

**证据**: §387.16.2 / §388.4；`docs/analysis/cross_sites/router_objective_ordering.md`

**原文片段**: 「拿到 oracle SR, 代价 +341% ~ +463%; … 60-95% 的任务没有任何 mode 能解」(§387.16.2)

---

## B10. learned triage 的可学性：AUROC / Holm p / 置换检验（三次修正）

**当前值 (6 cells, k=6, Holm at m=6, 阈值 0.008333)**:
四列表（AUROC / 送 cheap 比例 / solv% / Holm p）:
- red·B0 **0.666 / 27% / 26.1 / 1.0000**
- red·B1 **0.685 / 40% / 11.8 / 0.0299**
- cls·B2 **0.651 / 90% / 7.1 / 0.4627**
- red·B2 **0.483 / 95% / 7.4 / 0.0050**
→ **red·B2 是唯一 Holm 通过的 cell，也是唯一 AUROC < 0.5 的 cell**（其最强单协变量 0.711）。
B=10000 重跑: red·B2 **p = 0.00050**（10000 次里 4 次达到），裁定不变但**现在由数据决定**；red·B1 **0.01460 vs 阈值 0.01000 仍不过**；唯一显著格不变。

**演变**:
- §387.16.4 首报: **5/6 cell AUROC 0.65-0.72**，4/6 cell 超过最强单特征 +0.05~+0.09；唯一崩掉的是 red/B2 **0.483**。
- label-shuffle 零分布（200 次置换）首版: 实测 SR-无损节省 / 打乱中位数 / p = cls·B0 0.0%/4.2%/1.000 · red·B0 0.0%/0.0%/1.000 · cls·B1 10.6%/10.1%/0.475 · red·B1 14.4%/0.7%/0.025 · cls·B2 20.8%/20.7%/0.455 · red·B2 26.5%/4.1%/0.035 → 当时结论「**两个存活的 cell 一个都不过 Holm**」。
- **B-1902 (§388.7.2) 推翻该 p**: 置换单元错了（只打乱 y 而 succ/cost 留在原位），且 k/B 应为 **(k+1)/(B+1)** → cls/B1 **0.4776→0.5025**，red/B2 **0.0398→0.0050**（旧偏大 **8×**）→ **red/B2 通过 Holm**（与 codex 报的数字逐位吻合）。
- §397.3: **(k+1)/(B+1) 在 B=200 时地板 = 1/201 = 0.004975**，red·B2 报的 0.0050 **正是这个地板 (k=0)**；Holm 最紧阈值 0.05/6 = **0.008333 就在地板旁边**；「若 B=100 地板 0.009901 > 阈值，那格无论数据多强都**不可能**通过」→ **不是"次数少"，是裁定由 B 而非数据决定** → B=10000 重跑。
- §394.1 补充：**「5/6」的说法必须点明第 6 格是 0.483 且它正是唯一显著格**。

**红队自检 (§388.4)**: 真实标签分数离散度是打乱的 **1.09~1.78 倍**（6/6 实测），red/B2 比值最高 1.78× → 零分布低估选择性乐观 → **p 偏小 (anti-conservative)**。

**尾部富集 (§394.1)**: red·B2 把 **192/203 = 95%** 送去便宜模式而 SR 不损（**3.94% = best-single 的 3.94%**）；与 always-cheapest (100%) 只差 5%，而**那 5% 里恰好含 4 个成功**（nested 8 个成功 vs always-cheapest 4 个）。「p=0.0050 测到的是**尾部富集**；AUROC 衡量全局排序，对"只有极端尾部可判别"不敏感 —— **0.483 与尾部有信号并不冲突，两者测的是不同的东西**」。

**caveats**: red/B2 **7.4% 正类率 × 203 = 15 个正例，5 折后每折 3 个，20 维 LR 必过拟合**；scope = task-held-out 5-fold, seed 42, L2 LR on 20 raw features，mirrors 注册的 H10 split。

**证据**: §387.16.4 / §388.4 / §388.7.2 / §394.1 / §397.3；`scripts/analysis/router_triage_learnability.py`、`docs/analysis/cross_sites/router_triage_learnability.md`

**原文片段**: 「red·B2 是唯一 Holm 通过的 cell, 也是唯一 AUROC < 0.5 的 cell (其最强单协变量 0.711)」(§394.1)；「不是『次数少』, 是裁定由 B 而非数据决定」(§397.3)

---

## B11. 嵌套阈值选择 (B-1903) — 从"半嵌套"到真嵌套

**当前值 (真嵌套, 6 cells, k=6)**: learned triage vs baseline ΔSR = **cls·B0 +1.34 / cls·B1 +0.45 / cls·B2 0 / red·B0 −0.99 / red·B1 −0.49 / red·B2 0 pp**。
**承重句: 0/6 cell 的 learned triage 能 Pareto 胜过 always-cheapest**；两个权衡点 —— cls·B0 SR 高 **1.79pp** 但 cost 高 **11%**；red·B2 SR 高 **1.97pp** 但 cost 高 **2.4%** → **真权衡点非压制点**。

**§399 加严（2026-07-28，最有利角落也不翻）**: 承重句被独立管线在**它最有利的配置**下复测 ——
同族池（B0+B1，which-mode 冲突 45-48% vs 跨族 81.8%）× cost-tier 标签（结构上免疫 §395.2 缺陷，
plug-in 天花板 97.5%）。26 个 arm×cell：**严格支配 always-cheapest 0/26**（最有利角落 0/4）；
**相对六固定 mode 菜单非支配 0/26** —— router 从未落在经验 Pareto 前沿上。
⇒ 可写的加强表述：路由打不过固定廉价策略，**即使同族、标签粗到免疫 tie-break 缺陷、天花板 97.5%**。

**§401 cross-AI 双家复审后的两条修正**：(a) 「同族/粗粒度/池化都不是原因」的因果表述**收回** ——
它建立在「两 arm 都过 0.95 非支配线」上，而反例就在同一张表（reddit·B0 which-mode
15.27%@0.10415 两轴都优于 cost-tier 14.29%@0.10803）；且 reddit tier 标签 63/14
（少数类 18%）严重不平衡，粒度贡献与标签变异混淆。现只报共现与点估计。
(b) 「always-cheapest 是成本下界所以支配不可能」的反驳**已被证伪** —— Vision 只是
per-mode mean 最低，**47.3–70.9% 的 task 上不是最便宜**，per-task cost oracle 便宜
**22.2–46.2%**（该 headroom 表可独立进论文）。**核心结论 0/26 支配 · 0/26 在前沿上 不变。**

**演变 / 已作废**:
- §388.4 in-sample 阈值修复（3-AI 重合发现）: 两个 B0 cell 的节省是 **−0.9% / −0.4%**（几乎为零），「0/6 Pareto 胜过 always-cheapest」在诚实阈值下依然成立 —— **但该"嵌套"其实没真嵌套**（B-1903）。
- §388.7.3 半嵌套 ΔSR: **−1.786, −0.985, −0.446, −0.985, −0.893, 0.0 pp** → 「那个全局看『无损』的设定在半嵌套下就已经不无损」。
- §392.2 真嵌套落地（held-out 行改用只由训练行 refit 的 LR 打分，best_mode/cheap_mode 每外折重选，阈值只对着训练行的内层 CV OOF 选）。
- §399.1/§399.2 同族 pooled × cost-tier 复测（**加严不改判**，见上）。

**caveats**: 「cls·B0 反而升 —— 每折重选 mode 带来**真实的自适应收益**（选择只用训练折，非泄漏），**不是"更诚实就一定更差"**」；`router_objective_ordering` 重跑数字未变；测试 1622 passed。B-1903 当时明记「起稿前若要引用任何 triage 运营数字必须先做」。

⚠️ **§399.4 新增 caveat —— 本节数字的 producer 有 2/20 死特征**：
`router_triage_learnability.py:98` 的 `_feature_row` 从 **step-0 记录**读 `intent_token_count`
与 `reasoning_difficulty`，而 step-0 **不含**这两个字段 ⇒ 静默 0-fill。canonical 路径
（`extract_raw_features` / `extract_50_features:333`）是从 **task config** 取难度、从 intent 串算 token 数。
**方向性**：特征更少只会让 learned 臂更弱，而本节是负结果 ⇒ 这些数字**偏保守，结论不翻**；
但**若要引用本节任何 AUROC 绝对值须先重跑**。同型于 §312.5 P1-9（missing-field 静默当失败）。
§399 的 producer 走 canonical 路径，未继承此缺陷。

**证据**: §388.4 / §388.7.3 / §392.2 / §399；`docs/analysis/cross_sites/router_triage_learnability.md`；
`docs/analysis/cross_sites/router_pooled_tier_learnability.md`

**原文片段**: 「0/6 cell 的 learned triage 能 Pareto 胜过 always-cheapest; cls·B0 SR 高 1.79pp 但 cost 高 11%; red·B2 SR 高 1.97pp 但 cost 高 2.4% — 真权衡点非压制点」(§392.2)

---

## B12. router covariate 基线（LR 相对 trivial 协变量无优势）

**当前值**: 可训的 B0_cls / B1_cls 两 cell —— 18-feat TF-IDF+MI LR 的 **macro-OVR AUROC = .522 / .567**；相对 3 个 trivial 协变量（intent 长度 / 词数 / 是否带参考图）的 scalar 基线 **Δ = −0.013 [−.048, +.018]** 与 **+0.007 [−.054, +.073]** = **无统计可分辨优势**；template-disjoint 掉幅 **0 ~ −0.05**；template one-hot oracle 在 disjoint 下按构造塌缩（阴性对照通过）；canonical split **template 泄漏率 72% / 53%**。

**caveats**（一字不改）: 「定性 = **前瞻性红旗非推翻**(§6 无 landed AUROC claim, Table 4 全 TBD); estimand 纪律 = 本攻击面是 **LR mode-prediction AUROC**, **勿套到 §1 per-mode confidence-signal AUROC (0.766 P-SoM, 无 task 文本特征)**」；scope = rehearsal vintage `l1_router_rehearsal_20260702`（**非 canonical**）。

**证据**: §364（li2026aucnotenough C1 协议迁移）；`docs/analysis/cross_sites/router_covariate_baseline_2026-07-05.md`

**原文片段**: 「Δ = −0.013 [−.048,+.018] 与 +0.007 [−.054,+.073] = 无统计可分辨优势」(§364)

---

## B13. MODES 硬编码顺序 → 标签任意性（结论被翻转）

**当前值 (§395.2 / §396.5 / §396.7)**: **true_tie 在 6 cell 全为 0，tie-break 分支从未触发**；真实缺陷是 **12.5% ~ 54.64%（另处记 54.6%）的标签上，顺序表返回了严格更贵的成功 mode**。
被重标任务数 **53 / 23 / 26 / 9 / 2 / 2**，**恰等于 A.2 里"顺序表选了更贵模式"的计数**（两个独立计算对上了）。

**已作废**: §383.4 的「约 1/4 训练标签由 MODES 硬编码顺序而非数据决定: B0_cls 25/97=26% / B1_cls 16/55=29% / B2_cls 4/16=25% / B0_red 18% / B1_red 15%」—— 被 §395.2 推翻。

**caveats**: MODES 顺序按 "image ⇒ expensive" 排（`p79/policies/router_features.py:72-80` 注释白纸黑字），**而该前提被成本产物否证**；「codex #4 和 #6 不是两条 finding，是**同一个错误前提的两个出口**」；§6.4 "cost tier" 相应更名 **screenshot tier**。

**证据**: §383.4 / §395.2 / §396.5 / §396.7；`p79/policies/router_features.py`

**原文片段**: 「标签在 12.5-54.6% 的行上挑到更贵的 mode」(§396.5)

---

## B14. 池化标签供给诊断（矛盾率 / modal agreement / tier 一致性）

**当前值 (§395.1 重建，6 cells k=6)**:
- 同一 X 配矛盾 y 比例: **cls 57.41% / reddit 56.0%**
- in-sample modal agreement（**旧名 "Bayes 上限"，已改名**）: **cls 79.17% / reddit 83.7%**
- screenshot-tier（原 "cost tier"）一致性: **cls 68.52% / reddit 88.0%**
- 池化标签数: **249 → 260**（B-1904 重抽后）
- reddit 特征向量分组: **78 组里 69 组是单例，占 75% 的行**
- 退回 task 分组后的 resubstitution 单例占比（codex #2）: **cls 50/168 = 29.8%、reddit 34/92 = 37.0%**；仅共享 task 的口径为 **70.3% / 74.1%**

**演变 / 已作废**:
- §383.4 首版（k=5，reddit 只有 2 个 cell）: 池化 **249**；cls 矛盾 **57.4%**（54 共享任务）/ red **45.5%**；"Bayes 上限" cls **79.2%** / red **87.7%**；cost tier red **95.5%** / cls **68.5%**。→ **reddit 三个数字全变**（45.5→56.0 / 87.7→83.7 / 95.5→88.0），**纯因 cell 数从 2 补成 3**。cls 三个数字逐位吻合（57.41/79.17/68.52 vs 57.4/79.2/68.5）= 实现与当年算法一致的对照证明。
- §397.2 一度重算 reddit 天花板为 **89.13**（codex 给 ~91.30，不能照抄）→ §397.5(b) **退回 79.2 / 83.7（= 稿子原数）** → §397.7 #2 **改名 in-sample modal agreement**。

**caveats**:
- 改名理由: 「plug-in 估计在高基数特征上**乐观偏大**，**单例按定义 100% 正确**」。
- 「模型们对『选哪个 mode』分歧巨大，对『要不要那张图』却基本一致」→ 由此得「池化 + tier 二分类是唯一同时解决供给与可识别性的组合」。
- 路由特征全是任务的函数（14 intent 正则 + difficulty + has_reference_image，**无模型信息**）；落在 prereg L447 已注册的 **LOCO appendix sensitivity 槽位**内。
- 原三个 scratch 脚本（label_supply_sweep / label_trainability / pooled_label_conflict）**五天后已全丢**；该对照本身就是「**论点不变但数字会变**」的证明。
- 对取整精度完全不敏感（0-12 位小数下分组数与天花板逐位相同）。
- §397.7 codex #3: **格内最大 spread 15.1 / 27.3 / 48.3 / 36.1pp vs 池化 7.4-13.7pp → mode-invariance 是池化假象**。

**证据**: §383.4 / §395.1 / §397.2 / §397.5 / §397.7；`scripts/analysis/router_label_supply_diagnosis.py`

**原文片段**: 「cls 三个数字逐位吻合 (57.41/79.17/68.52 vs 57.4/79.2/68.5); reddit 三个全变 (45.5→56.0 / 87.7→83.7 / 95.5→88.0)」(§395.1)

---

## B15. archive-vintage intelligent-baseline ladder (S2)

**当前值 (archive vintage / classifieds / B0)**: always-DOM **17.4%** < stump **19.6%** < no-text **22.3%** < router **25.0%** < oracle **43.3%**；意外: **total-billed cost 5 mode 近乎打平（spread 11.7%）**。

**caveats**: 「archive 非 canonical; cost 打平 → **H10 cost 轴近退化为纯 SR 比较**」。

**证据**: §312.2

**原文片段**: 「意外: total-billed cost 5 mode 近乎打平 (spread 11.7%)」(§312.2)

---

## B16. Q4 MODES tie-break 的 measured-cost 验证（B-1806 前身）

**当前值**: landed B0+B1 cls **12 conditions × 224 ep** 的 `total_billed_cost_usd`：all-episode 均值 **vision 反而最低**（B0 .0648 / B1 .0432）；success-only 切片**跨 cell 互逆**（B0: vision .033 最便宜 → dom .051 最贵；B1: dom .018 最便宜 → som .050 最贵），**n_succ 仅 14-61**。

**caveats**: 「prior *text-tier 必然更便宜* 在 **episode-realized 语义**下不成立」；measured tie-break **不可用三连**: cell 间互逆无稳定全局序 / episode cost **内生于行为**（成功路径长度）→ **cost←outcome circularity** / n_succ 太小。决策 = **MODES prior 顺序不动（零 estimand 变更）**，F2 TODO 标 RESOLVED。

**证据**: §328；`p79/policies/router_features.py`

**原文片段**: 「measured tie-break 不可用三连: cell 间互逆无稳定全局序 / episode cost 内生于行为 (成功路径长度) → cost←outcome circularity / n_succ 太小」(§328)

---

# C. Pareto 分解 — 表征轴 vs 模型轴

## C1. 18 臂 cross-object oracle headroom

**当前值 (classifieds, 3 backbone × 6 mode = 18 臂, 同 224 任务, hindsight oracle, unit-free SR 轴)**:
- best single arm (B0/SoM) **27.23%**
- **表征-only oracle（B0 内 6 模式）43.30% (+16.07pp)**
- **模型-only oracle（SoM 固定, 3 backbone）31.25% (+4.02pp)**
- **joint 18 臂 46.43%**
- 边际: **模型选择叠在 B0 表征 oracle 上仅 +3.12pp**（B1/B2 只独解 7 task: 19/50/78/120/121/129/219）；**表征叠在最优模型选择上 +15.18pp**
- → **表征轴携带的 oracle headroom 约为模型轴的 4-5×**

**caveats**（一字不改）:
- menu-specific: 「**B1/B2 是 4B 级弱模型, 模型轴弱可能部分因 menu 能力偏斜**（强-强模型 menu 可能不同）, B3=MiMo 8 月可检验」。
- 单位纪律: 「**USD 永不跨 backbone**（api vs electricity）, 跨模型平面用 tokens / retry-adjusted latency（tokens 前沿 = {B0/SoM, B0/Vision, B1/Vision}; latency 前沿 = {B0/SoM} 单点）」。
- oracle 语义注记: 「本 producer 是 **metric-cheapest-success-else-cheapest**（成本轴最优）, **B0 USD oracle $0.0388 ≠ §371 replay 惯例 oracle $0.0623**（priority-order label）, **引用勿混**」。
- 交叉验证: B1 oracle 55 = 可训标签数 55 / B2 16 / 97+7=104。
- 相关自洽检查 (§396.5): **oracle SR = solvable 列，逐格相等**（oracle 只能选成功过的 mode）；旧稿反着用（声称 "the strongest single mode already covers that union's success count"，而同表 solvable 列自己就否证：43.3% vs 27.23%）。

**证据**: §377 / §396.5；`scripts/analysis/cross_object_pareto.py`、`results/phantom_paper/l1_router_offline_20260715/cross_object/`

**原文片段**: 「模型选择叠在 B0 表征 oracle 上仅 +3.12pp … 表征叠在最优模型选择上 +15.18pp → 表征轴携带的 oracle headroom 约为模型轴的 4-5×」(§377)

---

# D. 幻觉指标 / id-namespace（⚠️ 本批最不可信区）

## D1. 幻觉引用率九格表 — **跨 mode 比较已判定无效**

**已作废（禁止跨 namespace 引用量级）**: locator error（`element_id ∉ obs_nodes_info`）九格表 (site=reddit):
- B0: P-SoM **0.04%** / dom **0.39%** / SoM **0.08%**
- B1: P-SoM **0.12%** / dom **2.98%** / SoM **0.45%**
- B2: P-SoM **7.84%** / dom **18.21%** / SoM **8.84%**

**当前状态**: §397.9 证明判定基准 `obs_nodes_info` 的**键空间按 text payload 分两套**（native sparse vs 1..K compact）→ **跨 namespace 比 = 比两个灵敏度不同的探测器**；§397.10(1) 进一步证明 **compact namespace 是三个 mode（som / phantom_som / phantom_text）不是两个**，且 **Vision 根本没有 element id**。
→ 旧 digest 里 **"SoM 0.08/0.45/8.84 vs dom" 那个对比与 P-SoM-vs-dom 同样跨 namespace 无效**。
→ `write_digests.py:170` 那条 "P-SoM 干净 2.3-24.8× vs dom" 已加 🚨 注释（同一个跨 namespace 比较，也正是旧 §4.2 的来源）。

**仍成立的同 namespace 内计数（scope 受限）**: P-SoM 幻觉引用原始计数 **B0_psom 1/2796 = 0.036%；B1_psom 5/4145 = 0.121%**；walk_fail B0 **356** / B1 **1217**；B1 的 14 个 success 里 **11 个也带 walk_fail**。（sub-agent 只看了 B0/B1 两个 model、只看了 psom 一个 mode；全部复算一致。）

**caveats**: 「判据不依赖 mark_count 推断」；「**禁止把该表用于跨 namespace 的量级主张**」。

**证据**: §387.12 → §397.9 → §397.10(1)

**原文片段**: 「跨 namespace 比 = 比两个灵敏度不同的探测器。禁止把该表用于跨 namespace 的量级主张」(§387.12 caveat / §397.9)

---

## D2. id-namespace 归属表（§397.9 版**不完整**，以 §397.10(1) 为准）

**当前值（§397.10(1) 权威，`runner/main.py:2853-2860`）**:
- **compact 1..K（seq-keyed map）= 三个 mode: som / phantom_som / phantom_text**
- **AXTree modes（留 None / 原生 CDP nodeId）= dom / p-prompt / vision**
- **Vision 零 element_id** —— 实测动作是 `{"coordinate":[494,375],"coordinate_type":"qwen_0_1000"}`；**其幻觉率 0.000 是结构性不适用，标成 "native" 是错的**

**已作废**: §397.9 的两行表 —— 「DOM / P-prompt = 原生 · P-text / P-SoM = 1..K」。原文自评：「**在 4 臂 quadrant 内这是对的，但漏了 SoM 与 Vision**，而这两个的归属恰好推翻了我的框架」。

**支撑量级（同表，scope 内可用）**: element_id 中位数/最大值 —— B1_dom **18729 / 691695**；B1_psom **17 / 134**；B2_dom **7839 / 194037**；B2_psom **15 / 176**。
[聚合者转述 — PROGRESS.md 记主 session 另有实证模型输出 id: p-som 1/12/68 · p-text 1/13/72 vs p-prompt 139/4074/26235 · dom 2/3606/61833；该组数字不在 D4 台账内。]

**caveats**: 「dom 用原生 AXTree id（大稀疏整数），P-SoM 用 [SOM_MARKS] 紧凑编号 1..N；这是机制解释（抄错率差一个数量级）的依据 —— 但**该机制解释本身在 §397.9/§397.10 被重新定性为『探测器灵敏度差异』而非『引用更准』**」。

**证据**: §387.12 / §397.9 / §397.10(1)

**原文片段**: 「Vision 根本没有 element id …它的幻觉率 0.000 是结构性不适用, 我把它标成 "native" 是错的」(§397.10(1))

---

## D3. §4.2 交互结论 — 三重作废

**已作废（整块量化内容）**: §397.1 原报「legend 在 SoM prompt 下 **6/6 格降低幻觉**，而该降幅在 **6/6 格都大于**同一替换在 DOM prompt 下的降幅；机制 = prompt 宣告的 id 方案与 text 实际供给的是否一致（P-prompt 宣告 marks 却给稀疏原生 id，最差）」，scope = 6 cells / k=6 / **action-step 分母**。

**三重作废链**:
1. **§397.5(a)**: 换 **episode incidence** 口径后**三个计数塌了两个**。
2. **§397.7 #1**: 撤掉 **"interaction" 命名** —— 排除极值条件在 action-step 下 **5/6 格是代数强制的**（Gemini 指出）。
3. **§397.9 + §397.10**: 判定**整个跨 namespace 量化内容作废**（两条都是跨 namespace: P-SoM vs P-prompt / P-text vs DOM）。

**§397.9 给出的替代（退回同 namespace，符号级）**:
- DOM → P-prompt（都原生 id）: SoM prompt **升高**（action-step 5/6 · episode 5/6）
- P-text → P-SoM（都 1..K）: SoM prompt **降低**（6/6 · 5/6）
- 「**符号相反 = 真交互，且对探测器不对称免疫** —— 每个符号是各自 namespace 内部的事实，**比符号安全，比跨 namespace 的量级才不安全**」。
⚠️ 但该替代表述本身建立在 §397.9 的**不完整** namespace 表上（§397.10(1) 已作废其结论表述）—— 两侧并列，本聚合不调和。

**caveats**: 原文已注明该组数字「含机械成分」。

**证据**: §397.1 / §397.5 / §397.7 / §397.9 / §397.10；`scripts/analysis/aggregate_cross_mode_failure_signatures.py`

**原文片段**: 「legend 在 SoM prompt 下 6/6 格降低幻觉, 而该降幅在 6/6 格都大于同一替换在 DOM prompt 下的降幅」(§397.1，**已作废**)

---

## D4. id 噪声的正经配对测量（b0_paired_idperturb）

**当前值**: 配对设计 + **id-agnostic 判定**（只有点到**不同物理元素**才算 flip，数字变了点同一元素不算）:
- **B1（本地 temp0 dense）**: 组内一致性 A **1.000** / B **1.000**，**id-shuffle 改变决策 20.0%**
- **B0（proxy MoE）**: **0.867 / 0.890**，**12.5%**
→ 「B1 两组内部都完全确定 → **20% 是纯 id 因果**」。

**caveats**: ⚠️ **§397.10 明令：这些数不许再做加减法，只能各自带 scope 引用** —— 禁止与 self_drop 6.7/7.6pp、discordance 14.3pp、κ 0.614、H3 轴 1.35/2.09pp、跨 GPU ±3-5pp、AMENDMENT_07 Δ−3.2pp 做任何算术。
§397.10(2) 的定位: 「这条 id 噪声**早就被正经量过**，我不该从探测器灵敏度重新推」。

**证据**: §397.10(2)；`scripts/analysis/b0_paired_idperturb_replay.py`、`docs/checkpoints/probes/b0_paired_idperturb_20260529_*.json`

**原文片段**: 「B1 两组内部都完全确定 → 20% 是纯 id 因果」(§397.10)

---

## D5. AMENDMENT_07（SoM-family 换 1..K）的 SR 变化

**当前值**: SR **30.4% → 27.2%，Δ −3.2pp**（§299.4 实证值）。

**caveats**: scope 原文只记 "§299.4 实证值"，未进一步限定；⚠️ **§397.10 明令不许与其他 noise 数字做加减法，只能各自带 scope 引用**。
关联事实: **§298.3 早写明 dom / p-prompt 保原生 nodeId → 仍承担此 id 噪声；SoM-family 被 AMENDMENT_07 换 1..K 消除了它**。

**证据**: §397.10(2)（引 §299.4 / §298.3）

**原文片段**: 「SoM-family 被 AMENDMENT_07 换 1..K 消除了它 (§299.4 实证 SR 30.4%→27.2%, Δ−3.2pp)」(§397.10)

---

## D6. 干净 same-mode replicate pair 的 self_drop / discordance / κ

**当前值 (B0 · classifieds · vision, n=224, post-B-1860 clean replicate)**: **self_drop 6.7pp / 7.6pp；discordance 14.3pp；κ = 0.614**。

**caveats（工具自带，必须一起搬，一字不改）**:
`instability proxy, NOT H1 drop-one bias correction; 小样本/可能混代码版本 = upper-bound risk trigger`
—— 「是 **B0-MoE 上界，不可直接外推到本地确定性 backbone**；真正需要的是**本地格同测量而我们没跑**」。
⚠️ **§397.10 明令不许做加减法**；§302 另已 **RETRACT 线性减法（12.1% ≈ 10.5% + 1-2pp）为 category error**。

**衍生对比（仅并列，不相减）**: H3 两轴 **1.35 / 2.09pp** vs self_drop **6.7 / 7.6pp** —— **小 3-5 倍**；两侧 scope 不同（B0-MoE 上界 vs 本地确定性 backbone 上的 H3）。

**下游影响**: **paperA Limitations 里 "No same-mode replicate exists in our data" 是假陈述**（已提交且已同步 Overleaf），已改成如实叙述 + 明说上界性质。

**证据**: §397.10(3)（引 §302.1）；`results/repro_replicates/B0_vision_classifieds_R24792_clean_replicate/`、`compare_cross_run_same_condition.py:227-247`

**原文片段**: 「instability proxy, NOT H1 drop-one bias correction; 小样本/可能混代码版本 = upper-bound risk trigger」(§397.10 工具自带 caveat)

---

## D7. 跨 GPU type 的 greedy flip 率

**当前值**: **±3-5pp**（跨 GPU type，Myriad V100 / A100 混跑，greedy 解码；§300.2 记录）。

**caveats**: ⚠️ §397.10 明令：**只能各自带 scope 引用，禁止与其他 noise 数字做算术**。

**证据**: §397.10（引 §300.2）

**原文片段**: 「§300.2 记跨 GPU type flip 率 ±3-5pp」(§397.10)

---

## D8. archive 里同 (model, mode, site) 重跑的真实规模 —— §397.4 作废

**当前值（⚠️ 两个数并列，本聚合不调和）**:
- **台账 / 笔记 §397.10(3) 原文**: 「manifest 里 **15 组** (model,mode,site) 有 2 个 run（第二个 `grade=archived`, pre-fix → 确实 confounded）」
- **PROGRESS.md（主 session 已实证）+ 本聚合复核**: **19 组 ≥2-run**，且 `results/repro_replicates/` 有**两个** clean replicate（`B0_dom_classifieds_R31194_clean_replicate` + `B0_vision_classifieds_R24792_clean_replicate`）
  [聚合者复核 — 我按 `results/phantom_paper/run_manifest.yaml` 的 cells+in_flight+archived 三段聚合 (baseline, site, mode)，得 **19 组 ≥2 runs**：B0·cls·dom 5 / B0·red·dom 3 / B0·shop·dom 3 / B0·red 另 5 mode 各 2 / B1·cls 6 mode 各 2-3 / B1·red 3 mode 各 2 / B2·cls·dom 2 / B2·red·dom 2。笔记的 15 疑为只计"第二个 run grade=archived"的子集，但两者**未被任何记录调和**，故并列。]

**已作废**: §397.4 的「**全 archive 只有一对同 (model,mode,site) 重跑且被污染**」—— 原文自评「**我搜漏了**，只在 `results/visualwebarena/phase1/` 按目录名前缀搜」。

**caveats**: 归档 run 用**合并命名**（如 `B0_3mode_reddit_20260422` 一个目录含 DOM/SoM/Vision 三个 mode）→ **前缀搜法完全看不见**；「**早就有专门工具** `compare_cross_run_same_condition.py:227-247` 打印 `self_drop archive->current = sum(y1[i] and not y2[i])`，**正是 H3 轴的估计量形式**」。

**证据**: §397.4 / §397.10(3)；`results/phantom_paper/run_manifest.yaml`、`results/repro_replicates/`

**原文片段**: 「§397.4 的『全 archive 只有一对』是错的 — 只在 results/visualwebarena/phase1/ 按目录名前缀搜」(§397.10)

---

## D9. walk_fail / element_id∈{0,1} (P4) 的跨 mode 结构

**当前值**:
- walk_fail 率（element_id 存在但找不到可操作祖先），site=reddit，**P-SoM / dom**: B0 **13.3% / 23.5%**；B1 **29.5% / 18.5%**；B2 **21.9% / 35.2%** → 「**既非 P-SoM 特有**（3 个模型里 2 个是 dom 更差），**也不随能力单调** —— 在 (model, mode) 格间**就是噪声**」。
- B2_psom 幻觉步中落在退化页面的比例: **374 个幻觉步里只有 123 个落在 mark_count<=2 的退化页，其余 138+ 落在 66-122 marks 的正常页面**（用于排除"退化页面伪影"解释）。
- cls 侧 P4 fire 计数（B1）: **psom 63 failed + 4 success-ep（24 hit）> ptext 41+1 > vision 0**；**pprompt 0 次**（全扫确认 agent 从不 emit element_id∈{0,1}）。
- B2 cls: **psom 278（6-mode 最高）+ ptext 153**。
- B1 pprompt walk_fail: **880 actions / 149 ep (66.5%)**；click→no_actionable_within_walk **606 (69%)**，其中 **556/880 = 63% 是 stale-EID-after-nav**；type→no_input_within_walk **274 (31%)**；B-1869 复现（**walk_fail 21.7% 报 success=True + page_changed=True**）。

**机制（forensic 结论）**:
- **element_id=1 是幻觉式 low-default，不是 renumber 后的 root** —— §321 forensic 证伪了 "[1]=RootWebArea renumber" 机制故事（task 20 step5 真实 observation `RootWebArea=[3377]`，元素是 [3833] 等高 ID）；§322 psom task 8 同样是原始 AXTree node ID（RootWebArea=[2], textbox=[140], 无 [1]/[0]）。
- §346 把该结论从 B1 扩到 B2 并**部分翻转**: B2 psom task 199/201 实证 **element_id=1 在 [SOM_MARKS] 真实存在**（root/body, bbox=[0,0,10,10], action_success=True）= **非 scaffold 构造错**；B2 4B 把 id=1 的 href 当 **low-information fallback 信息源**（task 199 thought 明文）。
- axis 分解 (§323): 「**[SOM_MARKS] 短编号（psom/ptext）找不到目标就 default 到低 [1]（P4 抓）** vs **完整 AXTree 高 node ID（pprompt）引用 stale/错类型真 node ID（P4 漏，walk_fail 抓）** → **裸 element_id 信号 mode-fragile，walk_fail 信号 mode-robust**」。
- walk_fail 两变体 (§322): `no_input_within_walk`（type，同 ptext）+ `no_actionable_within_walk`（click，psom 特有，SoM-style prompt 把 [N] 暗示成可点标记）= **干净 prompt-axis 证据解释 psom > ptext**。

**caveats**: §323 初始假设「瞎猜高 node ID」被 forensic **部分证伪**（task166 高 ID 10451/10918 反而全 resolve；task141 eid=148 小 ID）。

**证据**: §321 / §322 / §323 / §346 / §387.12

**原文片段**: 「既非 P-SoM 特有 (3 个模型里 2 个是 dom 更差), 也不随能力单调 — 在 (model, mode) 格间就是噪声」(§387.12)

---

# E. SR 主表 / 分母 / benchmark-FP

## E1. classifieds 三模型 × 六 mode SR

**当前值 (cls, n=224, 无排除)**:
| mode | B0 (235B) | B1 (Qwen-4B) | B2 (Gemma-4B, pan-and-scan ON) |
|---|---|---|---|
| dom | **17.4%** | **6.2% / 6.25% (14/224)** | **1.3%** |
| som | **27.2% / 27.23% (61/224)** | **14.29% (32/224)** | **2.2%** |
| vision | **25.0%** | **12.5% (28/224)** | **2.2%** |
| phantom_text | **15.6%** | **7.6% (17/224)** | **0.4%** |
| phantom_som | — | **6.7% (15/224)** | **0.9%** |
| phantom_prompt | — | **6.7% (15/224)** | **1.8%** |
（B2 各 224 ep；B0 oracle 43.30% 见 C1）

**演变 / 已作废**:
- §316 的 4 格 canonical: B0 dom 17.41% → som 27.23%（**+9.82pp, ×1.56**）；B1 dom 6.57% → som 13.64%（+7.07pp, ×2.08，**n=198/224 provisional**）→ **B1 som 终值 §317 = 14.29% @ 224**。
- §325 B2 dom R17895 raw SR **2.2% (5/223)** / legit **≈0.9-1.3%** → 该 run 2026-06-09 因 pan-and-scan amendment **降级为 pas-off ablation arm（archive）**，digest 加 SUPERSEDED header。

**caveats**:
- §316: 用 per-episode summary 的 `success` 字段（与官方 condition_summary 224/224 验证一致）；**`reward` 字段 ≠ task score（实测只 9.8% 一致，弃用）**；「单 run 未计 run-to-run 方差 → 涨幅可比**定性稳健、定量待 freeze**」。
- §346 B2 名义 SR **严重虚高**: presence-only 伪成功跨 som(87/124) + vision(121/123/193) + ptext(5) + psom(201) **4 mode 系统性**（agent_finished=false + trajectory_incomplete=true，runner 在 budget 耗尽截最终 URL 救活）；benchmark-FP 集中 phantom 系（psom 142 / pprompt 5,110,142）；**pprompt SR 4/224 拆 = 3 FP + 1 侥幸 → 真实有效 SR ≈ 0**；ptext task 5 有 telemetry gap（`effective_mutating_action_count` 漏计 GET-based 删除）。
- §335 matched-capability 前提: 「**B2 比同为 4B 的 B1 低 3-7× → 同尺寸跨族差距 > 同族降尺寸（235B→4B 仅约半）→ family ≫ size**，advisor 2026-05-14 的 "4B 量级对齐 = matched-capability cross-family control" **前提失效**（B2 vs B1 差异同时混 family + capability）；**SR≈1% 时 1.7-3.3pp drop-one 落在 1-2 ep 翻转噪声里 → B2 对 cross-model meta 近零贡献**」。⚠️ §335 的判断被 **RETRACTED §338** 部分推翻（「§335 判断中把 scaffold family-fit confound over-weight 了」）。

**证据**: §316 / §317 / §318 / §320 / §321 / §322 / §323 / §325 / §335 / §346

**原文片段**: 「dom: B0 17.4% / B1 6.2% / B2 1.3%; som: B0 27.2% / B1 14.3% / B2 2.2%; vision: B0 25.0% / B1 12.5% / B2 2.2%; phantom_text: B0 15.6% / B1 7.6% / B2 0.4%」(§335)

---

## E2. reddit SR 与权威汇总值

**当前值**:
- **两 FP 排除后的汇总 SR = 6.40% = 234/3654**（分母 3690 → 3654 = 203×18），**−0.53pp**，总扣除 **22**。
- per-model 扣除（修正后）: **B0 5 个 mode 各 −1**（dom/pprompt/psom/ptext/vision；只有 som 两个 task 都没成功）；**B1 11**（5 个 mode 各 −2 + vision −1）；**B2 6**。
- 单 condition 值: B0 dom **14.6% (30/205)** · B0 som **14.6%** · B0 vision **7.8%** · B0 phantom_prompt **26/205 = 12.68%**；B1 三 mode 对齐后 Vision **15/203** · SoM **30/203** · DOM **29/203**；B2_phantom_prompt **0.49% → 0.00%**；B2_reddit.sr_pprompt **0.4878 → 0.0**。
- Tier-1 15 条 diag（分母 = 采集集 205）: **B0 reddit SR 11.22–13.66%**（fail-NOhit 18–23%）；**B1 2.93–8.29%**（11–15%）；**B2 0.49–3.90%**（0.5–4%）。

**已作废**: **§387.9 的 6.37% 作废（禁引）**；所有 205 分母的论文级 SR。

**caveats**: 「AMENDMENT_08 / preregistration Appendix A / 敏感性产物**一律用 6.40%**」；Tier-1 那 15 条的分母是**采集集 205，非计分集 203**；「B2 的 no-hit 塌到 0.5–4% 说明失败模式**高度同质**，已知 P-rule 几乎全覆盖」；digest 头部 SR 表**不由 `write_digests.py` 生成**（L298 只追加 v8 补记），是 /diag 流程**人手填**的 → **不能机械改分母**（`write_digests.py` 里含 "205" 的 8 处字符串**只有 2 处是错的**，其余 6 处是 episode 级覆盖率与 task id）。

**证据**: §358 / §361.4 / §378 / §387.6 / §387.15.2 / §389.6 / §393 / §393.2；`docs/analysis/cross_sites/amendment08_sensitivity.md`

**原文片段**: 「6.94% → 6.40% (−0.53pp) = 234/3654; 总扣除 22; 分母 3690 → 3654」(§387.15.2)

---

## E3. benchmark-FP 清单（各站已确认）

**当前值**:
- **reddit task 160 (B-1889)**: 18 cell 里 **13 个判它成功**，轨迹硬约束复核 **13/13 全部不可能真完成**（10 个从未进任何 forum 页，另 3 个只进过非 'i' 开头的 forum）；分布 **B0 2/6 · B1 6/6 · B2 5/6**；**B2_phantom_prompt 唯一成功即此例 → 0.49% 实为 0.00%**。机制 = **must_exclude-only eval + require_reset=true → "什么都不做"自动满足**；硬约束依据 = 订阅按钮只在 `/f/<forum>` 页。**tier A（passive FP, outcome-blind 可判）**。
- **reddit task 58 (B-1892)**: 18 cell 里 **9 个判成功**；未取证的多在 **3–21 步内结束**（B1_dom 仅 3 步）。§387.15.3 修正: **9/9 全部未加载 wikipedia host**（原判据误用 `":8888" in url` 子串匹配）。机制 = **任务选题问题（parametric knowledge / knowledge leakage）**，答案 Reki Kawahara / 刀剑神域恰好是常识。
- **两 FP 合并影响**: **15/18 cell 受影响**（B0 5 cell 各 −1 / B1 11 / B2 6，psom −2）。
- **cls B1 phantom_prompt task 5**: **SR 抬高型 benchmark-FP 首例** —— delete 任务（eval 检查 item 84144 返回 404）: `delete_remove_count=0` / `effective_mutating=0` / `agent_finished=false`，但 **success=True 因 item 84144 在 reset 态本就 404**；估约 **0.4pp 虚高**。sibling som/dom/vision/ptext/psom 均报 benchmark-FP=0。
- **reddit B-1885 (task 103+104)**: config `program_html[1].url` 错指 task 102 帖子（上游 copy-paste）→ **两 task 全 mode 全 model 系统性不可赢，压 SR 天花板约 1pp**；全站扫描 205 config **恰 2 个**。
- **reddit task 138（非 FP）**: 独立复核确认 **真成功改名**（ref-img Patrick 可见 → B0 改名 MarvelsGrantMan136→Patrick → `/user/Patrick/account` program_html PASS）。

**caveats**:
- task 58 的范围限定（防夸大）: 「reddit 有 **40 个多站点任务**，在 18 cell 上总共只产生 **11 个成功**，其中 **8 个未取证且全部是 task 58**」→ §387.15.3 修正为 **9 个未取证**；「**不是多站点任务的系统性问题**，是这一个任务的答案恰好是常识」；tier-B 判据「成功但从未加载声明的第二站」对全部 40 个跨站任务一致适用，**命中且仅命中 task 58**（另两个可解跨站任务 49/66 的 2 次成功都真的加载了第二站）。
- B-1885: 「mode 间对比**无偏**（全员同败）；分母豁免与否是 advisor estimand 议题（**prereg 锁 205 不动，先只 §8 披露**）」。
- cls task 5: 「与常见 SR-压低 FP 相反；**需 reset-state 确认，登记 measurement-hazard 候选不阻 fire**」。
- **B-1901 / B-1907**: 被 AMENDMENT_08 排除的 task 58 / 160 **坐在"各臂独解"这个论文级证据槽里** → `site_unique_psom_union.reddit.task_ids` **[11,12,58,179] n=4 → [11,12,179] n=3**；venn 图 P-SoM-only **B0_reddit 6→5**（污染源 task 160）、**B2_reddit 2→1**（污染源 task 58）。

**证据**: §323 / §358 / §361.4 / §387.7 / §387.9 / §387.15.2 / §387.15.3 / §388.7.1 / §389.4

**原文片段**: 「18 cell 里 13 个判它成功, 轨迹硬约束复核 13/13 全部不可能真完成」(§387.7, task 160)

---

## E4. reddit 任务结构：什么是 routing 救不了的

**当前值**:
- **comment/reply-intent（严口径 52/205 任务）SR 2.11% (19/900) vs 其余 153 任务 8.49% (237/2790) = 4.0× 差距，18/18 cell 方向一致**；B2 五个 mode 在该组上是**绝对零**，B1 有两个 mode 是零。
- **口径敏感性（关键反证）**: 严口径 4.0× / **宽口径**（+post/submit/create/edit/upvote/subscribe…，159 任务）**7.23% vs 6.01% = 0.8×** / **eval 层**（含 program_html 的 mutation 任务，85 任务）**7.45% vs 6.57% = 0.9×** → 「『需要改变站点状态的任务普遍更难』这个解释**被排除**；难的不是 mutation 类别而是 **comment/reply 这个特定子类**」。
- **capability-ceiling**: **25% 的 reddit 任务在所有六个表征上都接近零**。
- **视觉信息三分（210 任务）**: A 有参考图（config.image 非空）**84** · B 页面内嵌视觉（无参考图但 intent 提图）**64** · C 其他 **62**。
- **受控 dom vs som Δ（只差一张标注截图）**: A 组 B0 **+1.27pp** / B1 **+2.53pp** / B2 **−6.33pp**；B 组 B0 **+0.00** / B1 **+1.56** / B2 **+0.00**；C 组 B0 **−1.61** / B1 **+0.00** / B2 **+0.00**。

**caveats**:
- 25% ceiling: 「drop-one oracle 的天花板受此限制；这类失败应在 discussion 里与『routing 可挽救的失败』**明确区分**，否则 oracle 数字会被误读为『还有这么多可以靠 routing 拿到』」。
- 受控 Δ 样本量诚实声明（一字不改）: 「每格 **64 task × 1 rollout**，单元格差异只有 **1-2 个 episode**，**单个 Δ 不可过度解读**；稳的是方向一致性（**9 格里 7 格 |Δ|<=1.6pp**）」。
- comment/reply 计数分母为**采集集 205**（pre-AMENDMENT_08）。
- B 组 = **P34 (VISUAL_BLIND_IMAGE_TASK) 的结构性盲区**（其 gate 是 `if not config.get('image'): return []`）。

**证据**: §387.8 / §387.10

**原文片段**: 「宽口径 (+post/submit/create/edit/upvote/subscribe…) 159 任务 7.23% vs 6.01% = 0.8x … → 『需要改变站点状态的任务普遍更难』这个解释被排除」(§387.8)

---

## E5. cross-mode 失败签名矩阵（首个）+ diag ruleset 冻结

**当前值 (reddit, 36 conditions 同 RULESET_VERSION=8, 3 model 平均 episode 命中率 %, 列序 dom/som/vision/P-text/P-SoM/P-prompt)**:
- **通用**: P31 budget 耗尽 **67.0/64.2/72.7/75.1/73.7/68.5** · P36 walk_fail **57.9/47.0/16.6/45.4/57.1/57.2** · P5 感知缺失循环 **45.0/41.0/64.2/36.1/43.1/49.3** · P14 URL 自环 **23.1/25.9/43.3/22.9/28.0/30.4**
- **mode-specific**: P45 同动作死锁 **38.5/30.9/0.0/22.1/31.1/39.8** · P44 幻觉引用 **24.7/10.4/0.0/12.5/10.4/31.7** · P43 页面视觉缺口 **29.1/0.0/0.0/29.1/29.3/29.6** · P4 根节点误操作 **0.3/9.8/0.0/12.0/14.8/0.5**

**关键解读（原文）**: 「占比最高的四条（P31/P36/P5/P14）**全是通用的 —— 换 mode 救不了，需要 module 层**；mode-specific 四条里 **P44/P45 在 vision 上恒为 0 有平凡解释**（vision 用坐标不用 element_id），**P43 是构造定义使然** → **真正非平凡的只有 P4**（dom 0.3% vs P-SoM 14.8%）」。
⚠️ P44 = 幻觉引用 → 其跨 mode 量级读法受 D1/D2 的 namespace 结论约束 [聚合者推论：台账未就 P44 明说，但 P44 与 D1 同判据族]。

**ruleset 演变**:
- freeze v6 全量重扫: **22 condition 0 failed**；success 误伤（P34-P38 在 success ep）= **0**；**P36 walk_fail 主信号 15205**；P39 success 侧 **12**（presence-only delete FP 预期）；**P34 = P40 = 0**（「硬 sub-signal 严格保守，宁缺勿滥防污染，**不是规则失效**」）。
- v8: `RULESET_VERSION = 8-reddit-p41p46-b1890fix`；修 2 条既有规则（B-1890 P35/P39 死字段 guard；P33 正则加 reddit `/submission_images/`）+ **新增 6 条 P41-P46**；**reddit 18 + cls 18 全部落 v8，36 份 digest 补 v8 数字块**；33 条新回归，全套 **1569 passed**。
- **v7→v8 对 cls 非字节不变**（与 v6→v7 的 cls no-op 不同）: ① P35/P39 旧命中被移除（抽查确认那些 episode 确实有 6-8 个突变步，**旧命中是错的**）；② P33 在 cls **+1 例**（cls task 233 的 sites 只写 classifieds 但 intent 要求 "the characters in the image on Reddit"，该 episode 真访问了 localhost:9999，旧正则漏检）→ 顺带暴露 **sites 字段会漏声明跨站需求**，已记入 P42 的 under-fire 局限。
- P36 密度: **B2_dom 1670 step-hits vs B0_pprompt 367 ≈ 4.5 倍**（§387.6 未定性 → §387.7 **定性为 agent-limit**）。

**证据**: §347 / §387.6 / §387.13；`scripts/analysis/diag_pattern_match.py`、`scripts/analysis/diag_rescan_all.py`

**原文片段**: 「占比最高的四条 (P31/P36/P5/P14) 全是通用的 — 换 mode 救不了, 需要 module 层」(§387.13)

---

## E6. cls 侧 cross-mode oracle 与 per-task 重叠

**当前值**:
- **B2 (Gemma) cls 6-mode**: oracle SR **7.1% vs best-single 2.2% = +4.9pp routable（16 task）**；BUDGET 主导失败（**196-211 跨 mode**）；THUMBNAIL 梯度 dom **6** / som **12** / vision **6**。
- **per-task 成功重叠矩阵**: **B2 som∩dom both-win = 0**（同 35 任务零重叠）；B1 som vs dom: **both 8 / som-only 24 / dom-only 6**；B1 som vs psom: **图带来 25 个新任务**；**B2 两 mode 联合可解 = 6（含 2 假）vs B1 六 mode = 55**。
- **B1 四点 ablation**: dom **6.2%** → phantom_som **6.7%**（[SOM_MARKS] 文本格式效应 **+0.5pp ≈ 0**）→ som **14.3%**（标注图边际 **+7.6pp**）；vision **12.5%**（图单独约 **2× dom**）。B2: dom **2.2%** → som **3.1%**（32 ep partial）= **图边际 ≈ 0**。

**caveats**:
- B2 oracle **+4.9pp 含 runner 救活虚高**（som/vision exclusive-solve 大量是 presence-only 伪成功 87/124/121/123/193），**非纯真 routing value**。
- 「**B2 零重叠 = 成功不可重复 = 噪声地板的直接体现**」。
- B1 ablation 的可检验预测: 「**B2 psom ≈ dom ≈ som 全平且 B2 vision ≤ som**（若 B2 vision 显著 > dom 则本判读错）；som 增益载体 = **照片内容+空间布局**（标号 id 文本里都有），恰是 Gemma 唯一读不出的层」；B2 som 是 **32 ep partial**。
- ⚠️ 这三条（§327 / §335 系）均带 flag：**RETRACTED §338 指出「§335 判断中把 scaffold family-fit confound over-weight 了」**。

**证据**: §327 / §347

**原文片段**: 「B2 som∩dom both-win = 0 (同 35 任务零重叠) … B2 零重叠 = 成功不可重复 = 噪声地板的直接体现」(§327 追加)

---

# F. 模型行为 (B1 / B2 / B3)

## F1. B1 (Qwen3-VL-4B) cls 六 mode 的失败结构

**当前值**（SR 见 E1）:
- **som (R31705)**: 只有 **50.9% episode 主动发 finish**；**P31 (trajectory_incomplete) 占失败 51.6%**；P31 success-fire **10/10 presence-only**；failed 集 P31 causal verify **0/4**；失败集 **median steps 26.5 > mean 18.0（双峰指纹）**，早放弃极端 **22.4%** / can't-stop 极端 **51.6%**。
- **dom (R17188)**: 头号失败 **P5 感知缺失循环 173 + P31 budget 耗尽 131**（对比 B0 dom 头号是 P14 URL 自环 109）；**deterministic coverage 96.7%**（> B0 88%），no-hit 仅 7；dom success finish-rate **14/14 = 100%**；dom overall finish-rate **41.1% < som 50.9%**，failed finish-rate **37.1%**。
- **vision (R28622)**: deterministic coverage **87.2%**（no-hit 25/196）；**P5 在 vision 没消失仍 #1**（123/196 failed）；success-hit **11/11 全 presence-only**，其中 **5 个 finish-less arrival artifact**（task 125/130/151/152/187，全 url_match，no_op 0.93-0.97）。

**关键 caveats（一字不改）**:
- 「**P31 不是死因**，是 B1 finish-less 行为的**结构性 artifact**（url_match 读当前 URL + program_html isolated-context 查 DOM 两类评测器都不需 agent finish）；trajectory_incomplete 与 trigger_distribution 是 **B0↔B1 不可裸比的混淆变量**」。
- 「**P31 在 dom 与 som 含义相反**: som = finish-less arrival artifact（到达但不说停），dom = **真卡死**（P5 scroll/hover 循环 / auth 幻觉登录 / 跨站 working-memory 丢失），causal verify 在 dom task 86/207 是真死因 → **trajectory_incomplete 同时被 model 与 mode confound，不可裸比不可单独作路由信号**」。
- 「**98 个 P6-failed 不能读作 98 个视觉盲**（model 在 dom 仍接收 reference image 并 OCR 出可搜索实体走文本路径）」。
- 「**P14 在 B1 success 上 fire 13 次** = 证伪 diag skill 记录的 P14 v3 修正→success-fire 0」（v3 carve-out 用"有 type=productive"豁免，B1 模式是到达后无 type 的反复无效 click）。
- vision Tier-2 把 **P5 拆成三态**: route-win 实证（task 40 AXTree 丢失的已渲染文字型: **dom 30 步失败 → vision 4 步成功**，截图确认标题可读）/ route 救不了（表征无关导航搜索恢复失败）/ route 救不了（表征无关推理输出错）；**截图 forensic 翻案两例**（task 0 蓝 kayak 搜索结果确实没 kayak / task 1 2007 Yaris 截图确实是银灰）→ 「**4B 视觉感知天花板被 log-only 高估，真天花板在 detail-OCR 不在找不到目标**」。

**证据**: §317 / §318 / §320；`docs/analysis/vwa_classifieds/B1_*_classifieds_diag_digest.md`

**原文片段**: 「P31 在 dom 与 som 含义相反: som = finish-less arrival artifact (到达但不说停), dom = 真卡死」(§318)

---

## F2. B2 (Gemma3-VL) 的地板成因

**当前值**:
- **parse_error_rate**: som **5.04%** / ptext **4.11%** / dom **2.01%**（B-1876）；其中 **97.5% 是语义无效引用**（幻觉 element_id/option/action-type）**非真 unparseable (2.5%)**；`repaired_fenced` 兜 **86.5%** 脏输出。
- **终止失败**: B2 finish 率 **15% vs B1 51%**，**85% 游荡至 max_steps**；**57% mode-collapse**（≥70% 单一动作）+ **52.7% 连续复读**（task_55 select_option [11] 复读 8 次）；全局动作分布 click 52% / select 16% / type 14% / goto 8% / **finish 0.6%**。
- **perseveration (reddit)**: 「在收到**至少 8 轮显式 FAILED 反馈**后仍**连续 27-30 步一字不改**地重复同一个失败点击，直到 30 步预算耗尽」（history_window=8；`format_history()` 确实把 FAILED 反馈进了下一轮 prompt）。
- **B2 dom raw 数据 (R17895)**: action **98% parse OK (6083/6206)**；**224 ep 仅 23 次 emit finish**；P5 感知循环 **268 fires**；P31 **89%**；`invalid_select_option ×77` / `multiple_actions ×19` 经判读**全是 genuine model error**。

**判别结论（原文）**: 「**真地板为主**（非接线 bug，scaffold 能引出全部动作）+ **可救切片**（终止失败，但被 program_html 按末态判分夹死，**上限个位数 pp**）+ **confound**（parsing/repair/finish apparatus 在 Qwen 族调出，Gemma 拟合更差）」；跨 mode 梯度 **som>ptext>dom>vision = [N]-引用诱发 ID 幻觉 = on-axis (axis-2)**。

**caveats**: ⚠️ §335 全组带 flag —— **RETRACTED §338: 「§335 判断中把 scaffold family-fit confound over-weight 了」**。

**证据**: §325 / §335 / §387.7；`docs/reference/master_bug_catalog.md`

**原文片段**: 「B2 finish 率 15% vs B1 51%, 85% 游荡至 max_steps; 57% mode-collapse (≥70% 单一动作) + 52.7% 连续复读」(§335)

---

## F3. Gemma 视觉分层崩坏 + pan-and-scan

**当前值**:
- **单图 3 问 closed probe (task22 step0 som 截图, P79 1024-cap 链)**: Gemma (256 img tok) **标号读取正确**（Apply=26 / PublishAd=6 全对）+ **价格 OCR 正确 (7000.00)** + **照片内容错**（Dark Blue ×4 幻觉，真值银/白/红 3 辆）；**Qwen-4B (576 tok) 三问全对**（Q2 纯像素认出 Hyundai Elantra + Porsche 911）。
- **pan-and-scan A/B (A100, 20 图 × 3 臂, transformers 5.8.1)**: **256-tok (P&S off) 下 3 个独特场景中 2 个出现灾难模式**（油画页 "Wooden Frame-Brown" 复读 ×31 至 token 耗尽；汽车页 6 辆报 12 辆数量爆炸）；**P&S=768 全部消除** + 物体名瞎猜→全对；三臂各 **18/18 同图一致**（A100 greedy 确定）；**照片真内容三臂均未达**；Qwen 优势 = **颜色忠于像素 + 零灾难 + 汽车页 6/6**。
- **pas-on (R21521) vs pas-off (R17895) 同 102 task**: 表面 **1 成功 vs 3**；**legit 口径 1 vs 1**（task 25 = 唯一两 run 都复现的真成功；task 5 = 已审计 false-success；task 45 = audit 定 edge/运气）。
- **1024-cap probe（证伪 cap 是病因）**: 3 图 cap-1024 vs uncapped × P&S=True —— LABEL (26,6) 与 PRICE (7000.00) **两臂逐字相同**；PHOTO 两臂质量相当；**cap 只是 1280→1024 = 1.25×**（GPT 假设 1920 宽高估了）；**1024-cap 是 B1/B2 共享**（`gemma3vl_agent.py:200 = qwen3vl_agent.py:212` 逐字同 default+LANCZOS）**非 Gemma-specific**。

**caveats**:
- 「初始假设 *256 token 下 SoM 标号物理不可读* **被 Q1 证伪** —— 崩坏的是**自然照片内容认知层不是 UI 文字层**」；probe 前必须 grep offload/meta 警告（初版 Qwen 臂权重 offload 到 meta device → vision tower 输出垃圾，**差点被误读成 Qwen 视觉差**）；层内机制（为何 OCR 活照片死）是 **standing 推测不入 claim**。
- 配置归因: 「**bug 无 / 配置次优 = 实质（灾难可消）/ 模型差距仍在（768 仍逊 Qwen 576）**」；DGX 5.3 vs A100 5.8 版本差导致 DGX 侧有漂移。
- 1024-cap probe **诚实 caveat**（一字不改）: 「probe 用**开放式问题混淆 OCR 与照片**（Gemma 大量读 listing 标题文本，如 Canon EF 100-400mm / Mazda RX-8 只能来自标题；task_184 报 #25 Canon lens 而 §327 记该缩略图实为鸟样片 = **读标题没看照片**）→ **不隔离照片识别，§327 受控 photo-blind 结论不被推翻，别 over-claim Gemma 能看照片**」。
- pas 对照: 「**pas-on 无 SR 退化信号**，SR 持平约 1% 符合预判（**pas 买的是灾难模式消除+审稿防御非 SR 跳变**）；**per-task flip 不能全记 pas 头上** —— element-ID churn 同时贡献（§294 noise ceiling B0 ptext 约 **9% per-task flip** 是该类对比的噪声地板）」。
- **double-BOS 坑**: Gemma 模板渲染文本已含字面 bos，processor 再 tokenize 默认又加一个 → 必须 `add_special_tokens=False`；on-device 验证 pas=False 下新旧路径 input_ids/pixel_values **逐 token 全等**。「当日**所有**两步式 probe 均带 double-BOS 跑 → off/on 对比同 double-BOS 故**内部有效性保留**，方向性结论双环境复现稳；首次单图 probe Gemma 臂是一步式单 BOS；修复后 agent 与 probe 臂有 1-token 差，**probe 是配置决策依据非 paper 数据不重跑**」。
- ⚠️ §327 全组带 flag：**RETRACTED §338 指出 §335 over-weight scaffold family-fit confound**。

**证据**: §327 / §328 / §330 / §338；`docs/analysis/vwa_classifieds/B2_gemma_visual_probe_2026-06-09.md`、`B2_gemma_official_playbook_audit_2026-06-16.md`

**原文片段**: 「初始假设 256 token 下 SoM 标号物理不可读 被 Q1 证伪 — 崩坏的是自然照片内容认知层不是 UI 文字层」(§327)

---

## F4. B2 两 run 输入差异的成分（element-ID churn 是唯一源）

**当前值**: cls dom obs grep `token=` / `sessid` / `csrf` = **0 命中**；**12 个 task 的 step_000 diff 内容 100% 是 element-id 整体平移**（[89]→[90] / [115]→[116]），diff 行数 **0-74**（task 14 两 run **字节全同**）；直接实证 task 0 step 0 两 run **[8] vs [5] link Logout**。

**caveats**: 「**收回了"server token 参数差"这个因素（实测证伪）**；B2 = **模型侧确定 + 纯输入侧漂移**，与 B0（serving + 输入侧）不同故**拿 §298 B0 replicate 类比 B2 不严格**；另有 cross-task 状态差异（session-lost wave 位置不同）」。

**证据**: §330

**原文片段**: 「12 个 task 的 step_000 diff 内容 100% 是 element-id 整体平移」(§330)

---

## F5. B3 (MiMo-VL-7B-RL) 引入状态

**当前值**:
- **Stage0 smoke**: 加载干净（Qwen2.5-VL deployment class 确认集成轻）；**3/3 parse-valid + 无 native leak = 无 GLM-lockout**；grounding 连贯（task_184 找对 element 17 PA speaker）。
- **floor pilot 首试 (R300, 10 task / 15 step)**: **10/10 失败于 infra/setup** —— task0 真跑 15 步但 evaluator_error (No such file) / tasks1-3 Config file missing / tasks4-9 VWA_CLASSIFIEDS_USER/PASS missing；速度 **~3min/step**（thinking 模型 + 共享 GB10）→ **25 task ≈ 20-37h 不可行**。

**caveats**: 两新考量 —— (a) **RL 版是 thinking 模型每步输出 think 块 = thinking-vs-not confound**（B0/B1/B2 不 think）；(b) 缺 JSON thought/confidence 字段 —— **后经 §341 离线 verify 修正为 MiMo 直吐 confidence**；踩 GB10 sm_121 nvrtc prod bug 后加 fallback。
floor pilot 根因 = **DGX external/visualwebarena/config_files 不全 + creds 未在 env**（post-A100-migration dev 评测栈未重建）；**MiMo 集成本身 sound**；**floor 结果 DEFERRED 不是本 session 交付**。

**证据**: §339 / §341；`scripts/maintenance/probe_mimo_b3_conformance.py`

**原文片段**: 「10/10 失败于 infra/setup … MiMo 集成本身 sound; floor 结果 DEFERRED 不是本 session 交付」(§341)

---

# G. 运行 / 基础设施

## G1. 实测速度与 wallclock（分档假设两次被推翻）

**当前值**:
- **B1 dom cls (R11094)**: 4h 内只跑 **46 ep** → 推全 224 约 **20h ≫ 4h cap**；SR **4/46 = 8.7%**（对比 B0 dom 17.4%）；**0 parse error**。
- **B0 dom reddit (R4992)**: 约 **17min/task**（16h 只跑 56/205，约 **58h projected**）；对比 **classifieds 实测 142s/task**。
- **MiMo (B3)**: ~3min/step。

**caveats**:
- 「**反直觉: 弱模型比强模型慢**（SR 低 → 几乎每 task 打满 30 max_steps；B0 SR 高提前成功退出）；**B-1665 的 wallclock 分档只按 per-step latency 分，漏了 per-task 步数维度**；graceful SIGTERM (B-859) 保住 46 ep partial」。
- 「reddit dom 是 **max_steps-heavy**（低 SR → 多数 task 烧满 30 步）+ B0 AWS-proxy 高延迟（latency alerts 10001ms）；**16h cap 是按 cls 142s/task 校准的 → mis-kill**；= 2026-06-03 B1/B2 cls 4h-cap saga 的 **B0-reddit 版重演**」。

**证据**: §314 / §344

**原文片段**: 「反直觉: 弱模型比强模型慢 (SR 低 → 几乎每 task 打满 30 max_steps; B0 SR 高提前成功退出)」(§314)

---

## G2. AWS proxy outage 与 resume 协议

**当前值**:
- **outage 时长演进**: abort#3 约 **3min** → abort#5 约 **8-10min** → abort#6 约 **34min** → 07-04 那次约 **49h（史上最长，×2.4）**；对应 retry 预算演进 3 retries (约 3min) → max_retries 8 (约 10.7min) → 24 (约 35min)。
- **outage#4 双路径超时指纹**: gateway **503 @ 约 30s** + Lambda URL 直连 **502 @ 精确 120s**（= Lambda timeout 打满）→ 故障钉在 **Lambda 上游模型服务无响应**非 gateway/Lambda 本身；学长修复 = 上游重启 + Lambda timeout **120s→10min**；07-07 复测双入口 200。
- **resume 实战**: 截至 §362 共 **4 次 live**（R819 abort#7/#8 / R11344 / R32139 task80 / R28173 task87）；单次损失从**整 condition**（§352 之前弃 135-192 ep / 约 16-40h）**降到约 1 task**；abort#7 空窗 **73min**（无 auto-relauncher）→ 配 done-monitor 后 abort#8 空窗压到约 **5min**。
- **B-1882 resume 路径 latent dead bug**: `mint_run_id` 的 stale-check 用 grep `schema_version v2` 查 `condition_meta.json`，但该字段只出现在 v2-named summary 文件名 → **condition_meta 从来没有 → 永远判 stale → resume 形同虚设（dead code 数月）**；修为查真 v2 marker。即时手动补 R819 meta 后 **resume 救回 135 ep / 约 16h**。
- **B-1881 auth-class retry budget**: backoff `min(30·2^(n-1),120)`，n=3 → 约 **3.5min** 窗口；abort#8 的 reddit auth blip 约 **4min** → **just 溢出几十秒**；修为 n=6 → 30+60+120+120+120+120 ≈ **9.5min**。

**caveats**:
- 「proxy outage 在**指数级变长** → 再加阈值是**追移动靶**；两次（abort#6 与 §362）都差约 2min 自愈；**加长 retry 会拉 latency 尾部（canonical estimand）故不动参数靠 resume 协议吸收**」。
- 「resume 让 abort **廉价但不免费**（仍需人肉 classify + resume）；真正免人工要等 PROTOCOL_NOTE_03 背书后的 auto-resume；每次 resume 都产出 `resume_rerun_clean` 证据（B-486 force-rerun 原地重跑出合法 episode = **最硬 in-situ reproduce**）供 G8 classify」。
- B-1881: 「**机制对 ≠ 参数够** —— runner log 的 `retry 3/3 on fresh substrate` 是 B-1881 触发**完全正确**的铁证，失败纯是 budget 尺寸；**estimand-safety 不变**（steps==0 / 零 contamination / no-redraw / proxy_5xx+mid-episode 仍 excluded）」。
- B-1882「链一直 FORCE_NEW=1 所以从没暴露；只影响 FORCE_NEW=0 路径故**链零回归**」。
- 「07-06 的 100s busy-wait **仅 task 0 局部**（busy-page guard × agent 点击循环相互放大）→ **5-18 的 99s busy-wait cross-site contention 归因待修正**」。
- **quarantine registry**: §353.4 DGX 35→37 行（严格前缀不变式成立）；§366.2 DGX 落后 A100 5 事件 → 拉平后双侧 **47 行**；classify 记录含 task104/139/143/149/33/155/80/87 等（**全部 transient_drift**）；**G8 preflight gate 会 HALT 未分类 quarantine**。
- **RESUME_MISSING 停摆根因**: skip 判据 = **manifest-bound ∧ episodes==scored**（B-1825/B-1834），而 bind 是 chain 内步骤 → chain 死在 condition 完成后 bind 前 → 该 condition **永不 bound → 被判未完成会重跑**；ptext 完成后 A100 无 runner/orchestrator **停摆约 15h**（cron verdict `inprog=none` **不报警因完成≠故障**）。SOP: 先 `validate_fire_manifest --populate --apply`（**`--apply` 单用不写，必须配 `--populate`**）再 launch red；collision guard `pgrep -f` 会 **self-match 自己的 ssh 命令串**（用 `[.]` 括号技巧）。
- **cls chain abort 历史频次**: 5/18→6/9 至少 **17+ 次** queue_chain ABORT，几乎全 cls；但 `http=000000` 这个 signature **是首次**（全 log 仅 1 次）；根因多样（衬底退化 B-1839 / eval-timeout B-1836 / asyncio B-1581 / wallclock B-1665 / sentinel-fail）；**教训 = 别只看一次 log 就判孤立 transient**。

**证据**: §326 / §349 / §352 / §352.4 / §353 / §353.2 / §353.3 / §353.4 / §359 / §360.4 / §362 / §365 / §366.2；`PROTOCOL_NOTE_02/03`、`docs/checkpoints/quarantine_registry.jsonl`

**原文片段**: 「proxy outage 在指数级变长 → 再加阈值是追移动靶」(§349/§352/§362)

---

## G3. reddit 站点：破坏性任务 (task 138) 与 VWA upstream 缺陷

**当前值**:
- **B-1884 真根因 (task 138 改名)**: R819 summary success=true + `eval_source_agent_url = /user/Patrick/account` + title "Editing user Patrick"；时间线 = **改名@138 → cached session（绑 user id 13915）撑几个 task → auth_refresh 每 5ep fresh login → 138+5≈143 触发失败 → abort#8**。机制关键 = **cookie 绑 user id 不绑 username** → 改名后 cookie 仍活，**只有 fresh 重登（输用户名）撞墙** → 伪装成间歇 auth blip **骗过 B-1881/B-1883 六个 bug-number**。
- **跨三站扫描**: **全 880 task 只有 reddit 138 一个改凭证**（shopping / cls 各 0）→ 致命 abort **结构上仅 reddit**；但 shopping **无 reset + 大量 add-to-cart/wishlist → 静默跨任务污染（Phase 1b 预警）**；cls 有 reset（`envs.py:172` 唯一实现）受保护，**解释 cls 18 cond 干净**。
- **VWA upstream 缺陷盘点（全 file:line 实证）**: (1) `require_reset` 只实现 classifieds（envs.py:172-178），reddit/shopping 是 **TODO no-op**；(2) `wa_parallel_run.sh` 末行 `run_batch 0 100 200 300 380` = 任务切 4 个连续区间 4 pane 并行，**段内 run.py:323 顺序跑且 4 pane 共享同一 server/账号**；(3) task 138 改名在 VWA 下**污染比 P79 单流更脏**（共享账号的另外 3 个 pane 全污染）；(4) VWA 唯一处理 = `check_error_runs.py --delete_errors --tolerance 2` + until 循环重试，**修不了持久改名**；(5) `session.gc_maxlifetime=1440s (24min)`，reddit+cls 在 A100 都是。
- **Fix 4 live verify 抓到的初版 bug**: postmill users 表有 **normalized_username 列，登录认它不认 username** → username-only restore 不够（首跑因模拟只改 username **误报 LOGIN_OK**）；改成两列都复原后重验通过。

**caveats**:
- 结论（一字不改）: 「**不存在可照搬的 VWA 标准处理破坏性任务，也没有神圣的 canonical 0-209 顺序** → P79 的 reddit fix 是 **estimand 选择 + 透明披露问题**，不是合不合 VWA 问题；P79 加 auth_refresh（每 5ep/20min 重登）是因为**串行批跑一个 condition 约 8h ≫ 24min session**，而串行是因为 **estimand 含 cost+latency（并行会污染 latency）**」。
- 「abort 是 **P79 特有**（串行 + 周期 fresh 重登放大），VWA 并行 + cookie 复用 + delete-error-retry 容忍故不撞」。
- 教训: 「**live verify 必须 replicate 真实 mutation 的全部副作用**，否则弱模拟给假阳性」；restore 走 `docker exec vwa-reddit su - postgres -c psql -d postmill`（peer-auth，db_user/db_name 密码路全失败）。
- forensic 教训 (B-1868): 「click action **只存数字 element_id**，"点了什么"必须 join obs，**grep steps jsonl 找 logout 会假阴**」；B2 agent-induced logout 是**复发行为模式（约 1 波/100 task 起）非 infra/配置 bug**；B-1868 preserve 首次 live 生产验证 = R21521 task 5/6/7 `infra_covariates=['session_lost_preserved']` 原子 patch + trajectory_events 双事件（detected×3 + preserved×3）+ **零物理删除**；task 4 = perpetrator（自点 logout = agent 行为失败**不标**）。
- **B-1895（未定根因）**: **77/720 (10.7%)** 的跨站 episode 里出现过整条 `|AND|` 规格被当单个 URL 导航（`localhost:9999/f/…/38990%20%7CAND%7C%20http://localhost:8888/…`）；分布 **B0 21 / B1 56 / B2 0**；对照 **141/720 = 19.6%** 的跨站 episode 确实落到过 8888 host。「根因未定，**不猜** → catalog 标 DIAGNOSED-未定根因；已排除: 不是初始导航（step 0 干净），不是 new_tab；上游 `browser_env/envs.py:214` **正确拆分并预开两个 tab**；影响有限 —— **不动 AMENDMENT_08 两个排除**，但意味着**跨站类 1.53% 的低 SR 目前不能干净归因为"模型不会跨站取证"**，这成了 AMENDMENT_08 §2 不排除整个跨站类的第二个理由」。
- **口径陷阱**: 「子串口径把 **141 episodes / 2265 steps 虚报成 181 / 2984**；判"是否到过某站"必须 **host 提取 `^https?://([^/]+)`**，不能用 `":8888" in url`；§387.15.3 和这条是**同一个错误的两次显形**」。
- **reddit 参考图 (B-1878)**: reddit 205 task 扫出**仅 2 个本地 reference**（其余 16 走 http URL）；修复 = 建 `coco_images → external/visualwebarena/coco_images` symlink + curl 落 task184 的 `B009P9HODS.1.jpg`。「cls image-eval 走 task-attached reference (B-824) 不碰 CWD 顶层 coco_images，**reddit 是首个用 VWA 原生 fuzzy_image_match 相对路径的站点** → A100 self-hosted 迁移**漏建根 symlink 的 latent 缺口首次暴露**；shopping (Phase 1b) 同隐患；**纯环境层修复零代码零 estimand**」。

**证据**: §329 / §343 / §355 / §356 / §357.2 / §357.6 / §387.15.4；`external/visualwebarena/browser_env/envs.py`、`scripts/maintenance/verify_reddit_identity_fix.sh`

**原文片段**: 「cookie 绑 user id 不绑 username → 改名后 cookie 仍活 episode 续跑, 只有 fresh 重登 (输用户名) 撞墙 → 伪装成间歇 auth blip 骗过 B-1881/B-1883 六个 bug-number」(§355)

---

## G4. 机器层事故（A100 / DGX / GPU）

**当前值**:
- **A100 storage outage (4 天 downtime)**: p-79 (512G) + boot (500G) = **1012Gi vs namespace requestsStorage quota 1024Gi = headroom 仅 12Gi**；迁移时 Harvester 临时翻倍复制撞配额 → 调度失败；底层 = **KubeVirt #17417**（GPU-passthrough VM + attached volume → restart crash-loop）。**初判 "Code=67 unsupported PCI passthrough → 节点不支持 GPU 直通" 被 ARC 同 GPU 接他 host 无问题的实验证伪；Code=67 是次生症状非根因**。
- **A100 nvidia 用户态/内核态裂开**: unattended-upgrades 07-24 06:31 把 `nvidia-*-580-server` 从 **580.159.03 升到 580.173.02**；**uptime 54 天期间没重启过** → 「用户态 libnvidia-ml.so 立刻换新但**已加载的内核模块只能在 rmmod 或重启时才换**；裂缝一出现**所有本地模型 (B1/B2/WA pilot) 全部不可跑**」。
- **CUDA OOM (B-1877)**: live **23.56GiB (60%) + 15.39GiB reserved-unalloc ≫ 16MiB request**；仅末尾 1 次无爬升；其他 5 mode 含 224ep phantom_som **OOM=0**；task34 step19 非异常长；修复 = `expandable_segments:True`。「唯 phantom_prompt 受害 = **原始 AXTree（最长且方差最大）× Gemma3 不终止游荡** → 可变尺寸 KV = 碎片经典诱因 → §335 的终止失败**不只压 SR 还是本次 OOM 的机制因**；paper-grade guard **fail-closed on OOM 是设计正确**（33-ep partial 被 sentinel 拦住没污染 manifest）」。
- **DGX 24-cell mechanistic sweep 首次静默死亡**: 07-23 17:40 起，07-24 11:51 中途死掉，停在 cell 2 的 task 78 L29，**无 traceback 无退出码无 OOM 行**；**1/24 完成**（p1_fwd_strong_cls, 845 min）；cell 2 残缺（无 pilot_summary.md → 续跑会重做）。「GPU 现被他人两个作业占（60GB + 885MB），但那个 60GB 作业 **07-25 23:37 才起，晚于死亡时间 → 不是死因只是现在的障碍**；sweep 脚本本身可续跑但**没有重启，跑完一遍循环就退出**」。
- **ntfy 误报 (B-1893)**: `fire6_monitor.sh` 的 "orch up but NO step in 60min" 每 30min 一条，**10h 内 6 条，占该窗口 ntfy 总量 9 条中的 6 条**；根因 RESULTS 硬编码 `results/visualwebarena/phase1` 而 WA 数据落在 `results/webarena/phase1`；**这是同根因第 5 次复发**（B-1825 / B-1827 / B-1840 / 2026-07-03 FIRELOG）；修成 RESULTS_DIRS 数组后 healthcheck 完全静默。

**证据**: §313 / §337 / §387.1 / §387.11 / §387.16.6

**原文片段**: 「p-79 (512G) + boot (500G) = 1012Gi vs namespace requestsStorage quota 1024Gi = headroom 仅 12Gi」(§313)

---

## G5. WebArena (WA) pilot / 全量

**当前值**:
- **n=10 分辨力**: 每 mode **95% CI ≈ ±25-30pp**；实测 dom 20% / som 30% / vision 10% / ptext 20% **彼此完全无法区分**；「也几乎不会有 drop-one 结构 → 放开到全 **104 个 scored task × 6 mode**（~8.2 min/ep → ETA ~3.5 天）」。
- **采样器 vs 加载器**: 首版实际加载 **27/30**（改为复用 runner 的 `_is_na_task` 谓词后 **30/30 对齐**）。N/A 任务（`reference_answers.fuzzy_match == 'N/A'`）是 §139.8 预注册排除项。
- **"ETA 3.5 天的全量 WA" 实际跑了多少**: **0 个 task、0 个 run_id**；log 全文只有 **37 行**，末尾 `[1/27] queue_baseline.sh` + Usage + `[FATAL] queue script rc=2, no run_id minted` + aborting chain；**monitor 却报 launched chain pid=2650917 且 harness 收到 exit code 0 / status=completed，而 2650917 在 A100 上不存在**；**6 个 step 摊成了 27 个（漏引号）**。
- **pilot 实际时序（与 handoff 记录不符）**: monitor 17:09 判定 pilot 已退出并 fire，但 pilot chain 到 18:41 核查时**仍然 alive，卡在第 5 步 psom**；**psom 直接吃了 104 全量**（现 15/104）→ 走 **Route b（6 臂同契约从零重采）**。
- **A100 后台 chain（§397.8 交接时点）**: chain 2658570 存活 **13h28m 仍在 step 1/6 (dom)**，ETA ~3 天。

**caveats**: 「**fail-closed 生效 → 数据零污染**，也没有与仍在跑的 psom 形成同 site 并发」；「因为 6 个 WA config 的 defaults 已被切到 `exp_v2_wa_full_reddit_base.yaml` 而 psom 是唯一在切换之后启动的那步 → handoff §7 写的『pilot 收尾 → 自动起全量』**时序本身也不对**」；教训「**采样器和加载器必须共用同一个"什么算一个任务"的定义，否则分层配额是虚的**」。
**基建约束 (§397.4)**: 「A100 在跑 WA 全量（~3 天），而 **WA reddit 与 VWA reddit 共用同一 postmill 容器 + 同一 `.auth/reddit_state.json`（B-647），不能并跑** → 补一次重跑不是『1-3 天』而是『**排队等 3 天再跑**』」。

**证据**: §387.5 / §387.15.6 / §390.1 / §390.2 / §390.3 / §397.4 / §397.8

**原文片段**: 「跑了 0 个 task、0 个 run_id; log 全文只有 37 行 … monitor 却报 launched chain pid=2650917 且 harness 收到 exit code 0 / status=completed, 而 2650917 在 A100 上不存在」(§390.1/§390.2)

---

## G6. mechanistic patching 成本与校准（§5，已搁置但仍在跑）

**当前值**:
- **成本**: **16.6 s/层 @15 tokens**；@50 tokens **~13 h/cell × 28 cell = 364 h ≈ 15 天**；@15 tokens **~4 h/cell × 28 cell = 112 h ≈ 4.7 天**（DGX GB10, B1, 36 层, 全跑）。→ 该 28-cell/@15 方案已被 **§385.2 推翻**（改 @50 tokens / 24 cell）。
- **15 vs 50 tokens 校准**: 15 tokens **压效应幅度 2.3-11.7×**（且抹小效应 + 次指标峰层错位），**只换 1.5× 提速**；**峰位置被保住**，变的是幅度与次指标峰层。
- **composite 不预测 patching 效应**: task 70 composite 全场最高 **9.20** 但三指标动态范围**全 0.00** → 「top-24 by composite ≠ 效应最强 24 个，**预期 ~1/3 空转**」= §300.3 异质性再证。
- **canonical artifact normalize**: cls **224 task / 4034 链接**；red **205 task / 4552 链接**（相对 symlink，幂等，遇真实文件不覆盖，落 `normalize_manifest.json`，**下游零改动**）。
- **三站 task config raw json sha256**（DGX vs A100 manifest）: **逐字一致 `5d7322cc` / `beb9fc27` / `68ea7eaa`**（submodule commit 两边均 `2c15d66d`）。

**caveats**: 「主导项是**生成长度不是层数**（summary 只打印 7 个采样点，容易误读成 stride）」；config SHA 一致的意义 = 「两边同源，intent 相同，差异只在 URL 字段，而 **mechanistic 只读 intent 不碰 start_url**」。
⚠️ 该 sweep 的完成状态被 **§397.10(4)(5) 两次纠正** —— 见下方修正节。

**证据**: §384.3 / §384.4 / §384.6 / §385.2；`scripts/queues/queue_mechanistic_canonical.sh`

**原文片段**: 「压幅度 2.3-11.7x (且抹小效应 + 次指标峰层错位), 只换 1.5x 提速」(§385.2)

---

# H. 论文工程 / 跨 AI 审计 / 管线卫生

## H1. pytest 全套件规模轨迹（回归基线锚）

**当前值**: §256 **1242 passed/10 skip** → §271 **1273** → §312 **1397** → §328 **1393** → §366 晨间 **1433+2 failed** → §366.3 日终 **1476/0** → §367 **1485** → §371 **1491** → §374 **1498** → §376 **1503** → §377 **1507** → §379 **1514**（11 skipped）→ §380 **1527** → §383.3 **1531** → §387.4 **1536** → §387.13 **1569** → §389.6 **1619** → §389.8/§392.2 **1622** → §397.8 **1626**。

**caveats**: 「§366 的 2 个 failed 中一个是**三周陈年账**（`test_stress_a2_1` 的 B-897 3-5× 断言自 commit 7b0f456 起 stale-fail 无人发现），一个是当日 **docs commit 引入** → **docs commit 也必须跑 pytest**」。

**证据**: §366 / §377 / §379 / §383.3 / §387.4 / §387.13 / §389.6 / §389.8 / §392.2 / §397.8

**原文片段**: 「§366 晨间 1433+2 failed → §366.3 日终 1476 passed/0 failed」(§366/§377)

---

## H2. universe 泄漏族 —— 「不是漏 3 处，是 35 个入口」

**当前值**: **35 个 analysis 脚本读 `*_summary_v2.json` 却不引用 canonical universe，其中 23 个写入 paper 产物目录**；codex 的 Q7 ledger 覆盖约 **60%**。
「问题不是『还有 3 处漏网』而是『**这一族有 ~35 个入口，此前每次只修看得见的那条调用链**』；§388.7.1 说『三次修 universe 三次漏』其实**低估了**: 是三次修了 ~35 个里的 3 个」。

**已修的具体点与实证 delta**:
- **B-1905（论文级 oracle 效应 universe 泄漏）**: 三个 reddit cell 全部 `n_expected=205 / n_common=205 / lift_6_vs_3_n_universe=205 / is_partial=False`（`aggregate_phantom_lift.py:575/590`，**universe 只是已观测臂交集，从不与 canonical scored IDs 取交集**）→ 修复实证: **90 个字段变化，全在 3 个 reddit cell（cls 逐字节不变 = 干净对照）**；`B2_reddit.sr_pprompt 0.4878 → 0.0`；**修后 36 个 SR 与 `sr_per_mode.json` 0 mismatch**（修复前两个 canonical 产物**一个 205 一个 203，互相矛盾**）。同批必须改 `n_expected`，否则三个 reddit cell 会**因为做对了事而被标 is_partial=True**。
- **B-1904（特征缓存陈旧 universe）**: `raw_features_phase1a.json` 两个 reddit cell 均 `n_total=205`；`B0_reddit_fold_assignment.json` `n_tasks=205` 且**无 scored-universe SHA**；该缓存池**完全没有 B2_reddit**（pre-k=6 vintage）。→ §392.1 重抽（见 B7）。
- **B-1887（未跑的 mode 被当成全失败）**: **B2_reddit 被静默拉入并产出 16 个 oracle 标签，实际只有 4 个完整 mode**（phantom_prompt **零 episode** / phantom_som **76/205**）。根因 `collect_per_task_outcomes` 只 glob 磁盘现存 episode **不断言六 mode 齐全** + `derive_oracle_label` 读 `outcomes.get(m, False)`；「同文件 :125-133 的 P1-9 注释**精确警告过同一语义**，守卫加在 **episode 层，mode 层漏了**」。
- **B-1896（pass1_manifest 白名单缺失）**: B0_reddit **7 canonical Pass-1 runs with no manifest**（多一条被取代的旧 DOM run R819 vs canonical R11344）；B1_classifieds **7**（多的是 §367 点名的 `B1_3mode_classifieds_20260413`）；修后 **6/6 cell 6 runs / manifest_used=True**。「两个 cell 的 per-(task,mode) outcome 被 **newest-wins 无规则覆盖 → oracle 标签不确定**；§367 (07-15) 已写明必须先加白名单，**12 天没做**；副产物: 落白名单暴露 **9 个测试的隔离缺陷**（依赖"仓库里恰好没有这个文件"而通过）」。
- **B-1901 / B-1907**: 见 E3。

**证据**: §383.3 / §387.16.5 / §388.7.1 / §389.4 / §389.5 / §389.6；`tests/test_universe_consumption_lint.py`

**原文片段**: 「35 个 analysis 脚本读 *_summary_v2.json 却不引用 canonical universe, 其中 23 个写入 paper 产物目录」(§389.5)

---

## H3. fire-immutability 证明（两次）

**当前值**:
- **B-1888**: 全部 **123 个 config 解析结果 SHA256 新旧代码对跑** —— **VWA 侧 68 个逐字节不变 (changed=0)**，变的 55 个正是修复目标；`exp_v2_wa_base.yaml` 自身也不变（剔除 `experiment.run_id` 这唯一非确定性字段）。「`exp_v2_wa_base.yaml` 不变是因为它是**一级链**旧代码本就对，恰好**反向印证 bug 边界就是"二级及以上"**」；测试 5 条（含全 config 扫描断言 `default_backend` 必有 type），全套 **1536 passed**。
- **AMENDMENT_08**: 见 A6（cls SHA 逐字节不变 / reddit 205→203 / `tiers=()` 精确复现旧 SHA）。

**证据**: §387.4 / §387.15.1

**原文片段**: 「VWA 侧 68 个逐字节不变 (changed=0); 变的 55 个正是修复目标」(§387.4)

---

## H4. 稿件形态：AAAI-27 → REALM 两篇

**当前值**:
- **Paper B**: 7 节 **6647 词**（markdown）→ LaTeX 正文 **7741 词 + 7 表**，附录 733 词不计；页数 12 → 13（§3 重写 +1）→ 14 总（refs 起 p11 / 正文 10 页，移表 −1）→ §396.6 改 Table 1 单栏后**进 8 页** → §397.8 **两篇正文均 ≤ 8 页**。
- **Paper A**: **53741 词**，8 页需要约 6500 词 → **8× 压缩，等于重写**；实测 TBD 只剩 **14 处**、禁引数字大多已清。
- **两篇引用**: **从 0 到 86 处**；`paper.bib` 原有 **115 key**；新增 3 条方法学经典（`agresti1998approximate` = 0.68pp 退化格阈值来源；`higgins2002quantifying` = I²+Cochran-Q 口径；`cawley2010overfitting` = §5.2 嵌套选择偏倚先例）。
- **AAAI 阶段（已作废路线）**: 正文恰好第 7 页末结束，References pp.8-9，共 9 页，0 undefined；词数轨迹 **5181 → 5450 → 5248 → 5181 → 5197**；abstract **248 词**（verdict-neutral）→ k=5 splice 后 **250 词整**，正文 **6571 raw (+276)**，banned 0 / 残留 verdict 槽 0。**§383.1 AAAI 撤出 → 8 页 REALM，cut_prewrites 工作作废**。
- **k=5 splice 落数**: red P-prompt Tables 2/3 = **12.7 / 6.3** → §389.4 **UNIQ 槽 B0_reddit 6→5 / B2_reddit 2→1**（因 FP 排除）。
- **§396.5 移表**: 5 张表移进附录，**27 个数据行 26 行与原文逐字节一致**，唯一差异是有意的表头更名。

**caveats**:
- Paper A 真实问题（一字不改）: 「**它不是一篇论文，是一份把每条 /stress 结论都内联进散文的审计文档**。Paper B 那 7 节（6647 词）可作为目标形态的样例」；handoff 之前记的「格式转换 + 40 个 TBD + 20 处两轨制」**低估了量级**。
- AAAI 阶段: 「当前**压线合规零余量**，splice 增词由 cut_prewrites 池（161 词）抵消；`paper.bib` **63 个内部 note 字段绕过 convert.sh 编译会泄漏内部文字 + 膨胀约 3 页** → 官方编译必须走 convert.sh；分支 splice 的 abstract 曾**爆词**（278/291 > 250，branch 自账低估 26-35 词）」。
- Paper B **仍超 8 页限 2 页**时的判断: 「剩下的是**取舍不是错误**，最便宜的砍法是 §1（1455 词，其中大量数字在 §3-§5 原地又出现一遍）」。
- **投稿系统实测修正**: AAAI-27 主会走 **OpenReview 非 CMT**；abstract registration **Jul 22 11:59AM UTC-0 (= Jul 21 23:59 AoE)** / full **Jul 29 11:59AM UTC-0 (= Jul 28 23:59 AoE)**；**无独立 supp deadline**；**reciprocal reviewer nomination Jul 21 AoE 冻结，资格作者未提名 = desk-reject**；「repo 此前按 CMT 规划（readiness 审计的 250-word CMT limit 等锚全部改）；旧 07-21/07-28 UTC-12 与新表述**实为同一时刻**；user 立规**所有 deadline 双标注 UTC-0+AoE**」。
- **ACL 模板两坑**: ① `acl.sty:195` 自己发 `\bibliographystyle{acl_natbib}`，skeleton 再发一次 bibtex 直接 "Illegal, another \bibstyle"；② `[review]` 自己匿名作者块，且 **`[final]` 是包默认值 —— 不写 review 会出非匿名 PDF**；16 张表补 caption 的真实原因 = **pandoc 对 pipe table 一律出 longtable，longtable 在两栏非法，caption 行是 table\* 转换的触发条件**。
- **§396.8 收尾**: 两篇 REALM 稿从 markdown 草稿变成**双盲 ACL 8 页、--submission PASS、在 Overleaf 上、被三家 AI 审过一轮、引用从 0 到 86**；12 commit，git 与 Overleaf 均已同步。**§397.8**: 三轮审计（台账 5 条 → 自审 3 条 → 跨 AI 14 条）共 **22 条 findings 全部落地**；**1626 tests pass**；Overleaf 已同步（a4550b2 = repo 2115173，SUBMISSION=1 严格门通过）；距 08-05 还有 8 天。

**证据**: §360 / §366.2 / §366.4 / §380 / §383.1 / §389.4 / §395.4 / §395.7 / §396.1 / §396.4 / §396.5 / §396.6 / §396.8 / §397.8

**原文片段**: 「53741 词, 8 页需要约 6500 词 → 8x 压缩, 等于重写 … 它不是一篇论文, 是一份把每条 /stress 结论都内联进散文的审计文档」(§395.7)

---

## H5. 稿件内容错（压缩 / 搬家引入的 estimand 级错误）

**当前值 — 已确认的稿件错**:
- **AAAI 初稿 4 个 P0**（19 findings 总数: A 7 / B codex 8 / C agy 6）: ① "no image tokens / image-free / text-only cost" **违反 prereg §2.6**（phantom **保留 task reference_images**，正确措辞 = "no per-step page screenshot"）② H2(a) **用边际均值比 1.036 冒充 per-task paired median ratio** + 把 producer 的 "0/1 cells with data" 空洞真写成 not falsified ③ §1/§5.3 **Jaccard-vs-random 方向写反**（源: 观测 0.29-0.49 ≥ 约 3× E[J]≈0.06-0.10，写成 "no more than ~3×"）④ §8 约 97% power 引了 **Amendment 02 已标 OVERSTATES 的 4-mode ADD 估计量**。
- **Paper B §3 稿件 oracle 列用错了列（三家 P0 撞的同一条）**: 稿件 vs 产物 —— cls·B0 **27.23/0.06312/−12.8%（= triage_only 逐位相同）vs oracle_sr_cost 43.30/0.05777/−20.2%**；red·B0 14.78/0.09998/−9.5% vs 26.11/0.09534/−13.7%；cls·B1 14.29/0.04858/−19.4% vs 24.55/0.04340/−28.0%；red·B1 7.39/0.05554/−30.6% vs 11.82/0.05347/−33.2%；cls·B2 2.23/0.07145/−21.3% vs 7.14/0.06981/−23.1%；red·B2 3.94/0.06974/−26.4% vs 7.39/0.06961/−26.6% → **6/6 精确匹配 triage_only → 稿件用了错的那一列**。
- **§6.1 三元组不可重生成**: 稿件写 **7,963 / 7,278 / 685 无任何产物**；按 run_manifest 实测 **39 个能落地的 condition 得 7725 / 7058 / 667**。「**核心主张成立**（7725 个 episode 里 evaluator 只吐 0.0 和 1.0），但具体三元组**不可重生成**」。（原 §383.4 记 VWA score **纯二值 {0.0, 1.0}，7963 episodes 全扫: 7278 / 685** → "连续标签"路堵死。）
- **§1 headline 算术错**: **13.81% / 10.48% / 21.37% / 14.53% 全是 pre-exclusion 分母 (210/234) 算的却标 N=205/224**（13.81%×205 非整数）。
- **paperA Limitations "No same-mode replicate exists in our data" 是假陈述**（见 D6）。

**caveats / 教训（一字不改）**:
- 「**400KB→5K 词压缩的系统性风险是 quantifier / 方向 / 估计量在改写中漂移**，且**已知禁区清单在手仍会违反**（①③ 都在禁区列表里）→ 压缩后必须**逐句对 prereg / 源文回核**」。
- 「§396.3 总结过『**数字搬家后没人再核**』，本 session 又新增三例且**全部是清单与现实脱钩**（EXPECTED_TABLES 只数 table\* / 页数门量 refs-start / overleaf_sync 重命名清单漏了新增的 limitations）；**三次修法都不是改那一处而是加一条会自己发现脱钩的断言**」。
- **§333/§334 staleness pass**: 5 个并行 agent 合计约 **90 findings (7 P0)**，修约 60 处，10 文件 **+175/−103 行**；drafts 冻结在约 2026-05-27，**2026-06-09 一波协议事件全未进 drafts**。§334 复审 **17 findings 三家零重复**（仅 1 个 2-AI overlap），修 16 处（P0×1 + P1×8 + P2×7）；我自己 2 个 OOB —— (a) §4 引 ±2.5pp/14% 指向 §4.X.7-8 但 **grep 证实那里纯定性无数字**（两数还来自不同实验 §308 paired replay vs §302 vision 双 run）= **笔记走私在我身上复刻**；(b) claim-tier gate fail 条款写成 "the B2 cells fail" 字面=全 fail，**比 prereg B-1756 锁的 "either fails / any fails" 更松**；codex 王牌 = judge calls 改 20K-90K 但同段 **$10--$80 USD band 没重算（实算 $1.4-6.5）**。

**证据**: §333 / §334 / §360.1 / §383.4 / §396.2 / §396.3 / §396.5 / §396.8

**原文片段**: 「6/6 精确匹配 triage_only → 稿件用了错的那一列」(§396.2)

---

## H6. 跨 AI 审计（Mode A/B/C）规格与产出

**当前值（各轮）**:
- **§332 lit-review v1.1**: A (Claude) **9 findings**（3 强 unique 全是项目史走私类）/ B (codex 5.5min PASS) 王牌 = **`pan2024webcanvas` bib 条目指错 paper**（2410.17236 = Personalized Web Agents，真 WebCanvas = 2406.12373）+ 4 key 缺 eprint/作者错（**uground 一作实为 Gou 非 Zheng**，agentoccam=Anonymous 残留）/ C (agy gemini **13s** PASS w/note) 2 强（"informationally matched" 过强 + entanglement 反转攻击）；**3-AI overlap ×1**（负存在性 claim 无界标 → 加 "as of June 2026"）。
- **§388.5 Mode C**: Phase 2 深度 8 → **重试后 10 findings / 6 OOB PASS**；Phase 3 runtime **61s → 94s**，低于 submission band (15-40min)，未 silent 降级 scope，**按 v7.8 强制重试 1 次**；Phase 4 引文核实 **8/8 真**。
- **§388.7 Mode B (codex)**: **33.9 min (2033s)**，在 submission band 内（Phase 3 PASS，对比 Mode C 的 61s），**30728B，22 个 P-tag，5 个 OOB**；做了**全库调用点普查**。
- **§396.2/3 首次审两稿**: Mode A (Claude, stats/aggregation persona) **10 findings**；Mode B (codex gpt-5.6-sol xhigh, **23 min, 3.1MB trace**) **21 findings**；Mode C (agy Gemini 3.1 Pro Low, ~4 min) **10 findings**。Mode C 后检 PASS；**Mode B PASS with caveat（路径 confabulation）**。
- **§397.7 再审同一批改动**: Mode B **9 findings, 608s**；Mode C **5 findings, 74s**；**codex 独占 7 条全在代码/产物层，Gemini 独占 3 条全在推理层，两家零重叠** → 「scope split 完全奏效」。

**关键横切结论**:
- 「三家强 finding **完全互补**（Claude=项目记忆类 / codex=真 curl 类 / gemini=examiner 概念类）；**我自己写的句子我没抓到，B+C 双抓 = 自审盲区实证**」。
- **§366.3 日终敌对自审（17 commits）三个 P0 全是当天修复自身引入**: F-01 同名 h1_verdict 双 gate（分歧数据一边 PASS 一边 FAIL）/ F-02 filename vs payload task-ID 穿透 exact-set 守卫 / F-03 convert.sh 在 TODO=0 提交必经终态死于 rg×pipefail；复核重放 **11 FIXED / 2 PARTIALLY / 0 REGRESSED**；复核又抓到**修复过猛**（N-01 slotsheet 要求 producer 从未发出的字段 = 合法输出必拒，且 SHA 检查无真 join）。教训: 「**修复轮的测试只覆盖修复目标场景是结构性盲区**，分歧/错配 fixture 必须由**独立审计者**构造；**校验字段存在 ≠ 校验字段可 join**」。
- **§389.1/§389.2 codex 普查核对**: `scored_task_count` 调用点 —— 第一版 AST 只数到 **21**，codex 报 **36**；补上 alias map（`_scored_task_count` / `_paper_scored_task_count` / clear_tasks.py 里的 `_sct`）后 **TOTAL = 36，与 codex 逐位吻合**。「拿 AST 去核 grep 结论时，**别名解析是必需项不是可选项；一个两字符别名就能让"我独立复核过了"变成假的**」。三份普查表核对结果: Q5 基数 36 ✅ / Q5 flagged clear_tasks:455 ✅ 且更严重 → B-1910 / Q7-1 wider-set ×3 ✅ / Q7-2 replay extra=[58,160] ✅ / Q7-2 fig3 ⚠️归类偏差 → B-1909 / **Q7-2 未列 cross_object_pareto ➕漏列（同型 raise，潜伏）** / Q7-3 七个文件 ✅ 且其中 **4 个连 scored count 都没用** / Q7-5 `power_analysis:176` ✅ **MDE 7.768→7.806** 吻合 → B-1912 / Q7-6 缓存陈旧 ✅ / Q7-7 unique union stale ✅ 已随 B-1901 修好 = [11,12,179] / Q7-8 write_digests 205 分母 ✅ 且同型错误复发 → B-1913 / **Q7-tests "23 passed vs 真实数据即 fail" ✅ 逐位吻合（range(205) synthetic）** / Q4 INFERENCE_IDENTICAL ✅ 5/6 字段逐位吻合（se_pp 差 0.21 = 合成构造差异）。
- 「F3/F4 初判『幻觉』是**我的精确匹配被换行打断**，宽松匹配确认逐字存在」。
- ⚠️ §397.7 三条记录带 flag: **RETRACTED §397.9** 推翻了 §397.7 刚写进 §4.2 的两条量化结论（见 D3）。

**证据**: §332 / §334 / §366.3 / §388.5 / §388.7 / §389.1 / §389.2 / §396.2 / §396.3 / §397.7

**原文片段**: 「三家强 finding 完全互补 (Claude=项目记忆类 / codex=真 curl 类 / gemini=examiner 概念类); 我自己写的句子我没抓到, B+C 双抓 = 自审盲区实证」(§332)

---

## H7. 管线卫生：静默数据缺陷与 CI 策略

**当前值**:
- **§366.2 静默数据缺陷**: `axis1_microbehavior` 的 **direct-script import 会静默退化成空 registry** → **此前机制层输出全是空数据算出来的**；另修 compare bitrot（`get_cell` 全 grade 排序会选中 archived vintage → 改 `get_cells` + grade 过滤）与 fig3 partial 容错；`make analysis` 全链**首次零 Error**。「意味着此前引用过的 axis1 micro-behavior 数字（如 mean_jaccard 相关）**需重算才可信**」。
- **B-1908**: fig0c 是「**失败退出 ≠ 没留下产物**」（先 savefig 后校验，exit=2 却留下一张**时间戳崭新、未通过验证的论文图**）。
- **majority-baseline test-label 泄漏（smoke）**: 泄漏版 **0.533** vs 严格 OOF **0.333**（**smoke 数字非 paper 数据**；修法 = majority fold-local）。
- **Vale / deslop**: 现有 paper_drafts 全量 error 级 alert **551 条**（Paper.EmDash 312 / VerbTricolon 103 / ContrastiveFormulas 32；逐文件 section3 125 · section4_limitations 80 · section5 59 · section8 58 · section6 57 · section1 52）→ **CI 改成棘轮式**（deslopped.txt 白名单阻断 + 全量 lint 报告写 job summary 不阻断 + path filter，分支 master；白名单初始为空 = 零摩擦上线）。**paper-deslop 白名单无词边界子串误命中**: 把 `DOM` 加进 terms.txt 会误命中 random/dominance/dominant/dominated/domain **90 次**；`SoM` 误命中 some/sometimes **7 次** → 「二者恰是改写最高频动的词，**锁死其计数 = 闸门在正当改写上狂报，信噪比崩掉后人就开始无视闸门**」；故 terms.txt **只收 32 条多词短语/长唯一 token**；**上游若加 `\b` 可回收这条限制**。
- **Paper B deslop 收尾**: **33 errors → 0，7/7 invariant PASS，em dash 全零，7 节入 ratchet**；但「批量替换 em dash 造出 **3 处逗号拼接 + 1 处断句崩坏**（§8.1 成对破折号被拆成句号，整句语法坏），逐处修好 —— **正是 skill 'never bare-delete' 警告的副作用**」。
- **Paper A 立论换轴的 invariant_check 形状**: **21 violations 全是 added，零 removed/changed**（= 内容修正该有的形状）。
- **D7 lit-review story 化重写的证据守恒**: **62 引用宏 / 74 bib key / 10 page-pointer / 数字 token 全部 diff-empty**（Claude 独立 grep 复核通过）；词数 **4512→5099 (+13%，在 ±15% 带内)**；§2.0-2.8 编号保留。方法论 = 「把"不许动"的证据做成**机器可查的守恒检查**（引用/指针/数字 multiset diff），story 化只允许动**连接组织**」。原稿备份在 **gitignored 的 deliverables/ 下 = 唯一兜底**。
- **D7 P2-15 指针收尾**: **10 个 claim 全部回原文 PDF 定位到页级指针（0 NOT-FOUND）**；2 处 MISMATCH 当日裁决修正（InfiniteWeb 约 5% 是**联合审计口径非 evaluator-only**；OpenApps >2× 是 **SD 非 variance**）。「gitignored deliverable 的修改**没有 git 审计轨迹**，故在 chronicle 留痕；**MISMATCH 拦截协议 = codex 报告只标记不改数字，orchestrator 裁决后改**」。

**证据**: §366 / §366.2 / §372 / §386 / §389.6 / §395.5 / §395.6

**原文片段**: 「axis1_microbehavior 的 direct-script import 会静默退化成空 registry → 此前机制层输出全是空数据算出来的」(§366.2)

---

## H8. 文献核验（45 篇）与 novelty 存活

**当前值**: **45/45 arXiv ID 真实零幻觉**；**45 篇零篇做 observation/representation 轴路由、零篇在 WA/VWA 上 routing → novelty 完整存活**；LIVE 候选 **12 篇**（**全部 related-work/motivation 权重无一承重**），reviewer-defense 6，paper-2-park 16，background 5，drop 8；数字类声称基本准确（2605.29397 的 100× **实测 290× = 反向保守**）。
**Gemma 3 4B 的 VWA/WebArena 文献锚**: **标准 VWA/WA = N/A in literature**（arXiv API 核 3 个 ID 全真）；最近锚 = **12B+TTI WebArena 26.1%**（2506.07976，post-trained 非 raw）/ **12B-base 易子集约 6.2% TSR**（2603.04364 DMAST）。
**三站非 N/A fuzzy_match 任务数**: **0/0/0** → B-535 暴露面全在 `llm_ua_match` sibling。

**caveats**:
- digest 质量问题: 「§5-anchor 漂移 + 无 bib-dedup 仍在犯；**新发现类比通胀模式**（VRP phantom sibling / ExtractConf 三臂同构 / Deterministic Horizon 理论依据 / TIGER 域读错 **4 实锤**）；**THREAT 定性双向失准**（Plan-Then-Execute 高估，Non-Surjective 定性反了）」。
- 「**WA ≠ VWA 不直接可比，honest 标 N/A**；我方 ~1% 是**首个报告**（novel + 无外锚）；数字**正文级待 curl 复核**」。
- B-535: 「SR delta 判明 **repo 内不可算**（episode 归档不存 raw judge response）→ 改 instrumented paired-replay 协议 blocked-on；**WebArena-Verified 元数据拒编造**（OpenReview-only + bot 防护，待手动）」。
- 打中要害的现成 bib 条目: `enomoto2026observation`（观测缩减只动一个轴）· `feng2025visually`（视觉提示基准很脆）· `zhou2026visualignorance` + `asadi2026mirageillusionvisualunderstanding`（VLM 其实在忽略图像 —— 解释 P-prompt 为什么不崩）。
- **bib-author-bug 第 2 例**: Agent-E bib author `D'Hondt et al.` 经 arXiv API 实测是**错误作者**（真实 Abuelsaad/Akkil/Dey/Jagmohan/Vempaty/Kokku）。

**证据**: §333 / §338 / §363 / §366.2 / §396.4；`docs/literature/raw_digest_triage_2026-07-05.md`

**原文片段**: 「45/45 arXiv ID 真实零幻觉; 45 篇零篇做 observation/representation 轴路由、零篇在 WA/VWA 上 routing → novelty 完整存活」(§363)

---

## H9. 其他工具坑

**当前值**:
- **recharts × React 19 白屏**: recharts **3.8.1** + React **19.2.5** prod 下窗口 resize / fullPage 截图触发 **React #284**（invalid ref on path）→ **整页白屏**；**recharts 3.9.2 更糟**（首渲即崩）；修法 = Meeting0716Brief 完全去 recharts **手写 SVG 柱图**（resize-safe 实证），recharts 锁回 **^3.8.1** 保旧视图首渲。诊断教训: 「**navigate 时快照正常 + 截图后才报错 = resize 路径崩溃，别被首渲正常骗**；图标/ReferenceLine/Cell 全是**红鲱鱼**（二分时每次只该动一个变量）」。
- **`write_digests.py` 里含 "205" 的 8 处字符串只有 2 处是错的**（见 E2）。

**证据**: §368.2 / §393

**原文片段**: 「navigate 时快照正常 + 截图后才报错 = resize 路径崩溃, 别被首渲正常骗」(§368.2)

---

# ⚠️ 矛盾清单

> 一律并列，不选边。

1. **archive 同 (model,mode,site) 重跑组数：15 vs 19**
   - 台账 / 笔记 §397.10(3): 「manifest 里 **15 组**(model,mode,site) 有 2 个 run」
   - PROGRESS.md（主 session 实证）+ 本聚合按 `run_manifest.yaml` 三段复核: **19 组 ≥2-run**
   - 两者从未被任何记录调和。可能差在"是否只计第二个 run `grade=archived`"，**但无记录背书**。→ 见 D8。

2. **clean replicate 个数：一个 vs 两个**
   - §397.10(3) 只点名 `B0_vision_classifieds_R24792_clean_replicate`
   - `results/repro_replicates/` 实有**两个**（另有 `B0_dom_classifieds_R31194_clean_replicate`），PROGRESS.md 亦记两个。→ 见 D8。

3. **H1 p 值口径：0.807 vs 0.7208**
   - 台账全程引 **p = 0.807**（bootstrap primary）
   - 同一产物 JSON 另有 `h1_transparency_p_one_sided_normal_approx = 0.7208`
   - 两者是不同口径而非矛盾，但**引用时必须说清是哪一个**（本条为提示，非真矛盾）。→ 见 A1。

4. **H1 SE floor：实现 0.7897 vs 预注册散文 0.6533**
   - 两个 θ_FE 并存，同判 FAIL；「审稿人照 prereg 重算得 0.6533 而非论文的 0.7897，**数字对不上就是 kill 无论方向**」。→ 见 A2。

5. **triage AUROC「5/6 cell 0.65-0.72」vs「唯一显著格 AUROC 0.483」**
   - §387.16.4 报 5/6 cell AUROC 0.65-0.72 为正面证据
   - §394.1 指出**第 6 格 0.483（低于随机）恰是唯一 Holm 通过的格**，且 AUROC 与尾部富集「测的是不同的东西」。→ 见 B10。

6. **red·B2 permutation p：0.0398 vs 0.0050 vs 0.00050**
   - 原版 (B=200, 错置换单元) 0.0398 → B-1902 修正 0.0050（**恰是 (k+1)/(B+1) 地板 1/201**）→ B=10000 得 0.00050。裁定（唯一显著格）三版不变，但**中间那版是"由 B 决定而非数据决定"**。→ 见 B10。

7. **§383.4 「约 1/4 标签由 MODES 顺序决定」vs §395.2 「true_tie 全 0，tie-break 从未触发」**
   - 后者推翻前者的**机制归因**，但换成另一个更大的缺陷（12.5-54.6% 挑到严格更贵的成功 mode）。→ 见 B13。

8. **§387.16.3「triage 成本半值 40%、标签充足、所以有戏」vs §388.1「5/6 cell cheapest 固定策略省得更多」**
   - 原结论已加删除线，但两版数字都留在产物里。→ 见 B8。

9. **§397.9 的同 namespace 替代结论 vs §397.10(1) 作废其结论表述**
   - §397.9 给出「符号相反 = 真交互」的替代论证，但其 namespace 归属表被 §397.10(1) 判定**不完整**（漏 SoM 与 Vision）。§397.10 未明说该替代论证是否随之失效。→ 见 D3。

10. **§387.16.3 修正指针错**: 台账原文把 triage 成本修正指向「§397.17」，**笔记中不存在 §397.17**（实为 §387.17 笔误，且真实修正记在 §388.1）。

11. **§387.9 汇总 SR 6.37% vs §387.15.2 权威 6.40%**
    - 6.37% 明确**作废、禁引**；但仍可能出现在旧产物里。→ 见 E2。

12. **§390 WA chain 的 harness 状态 vs 现实**
    - monitor 报 "launched chain pid=2650917"、harness 收到 exit 0 / status=completed；**实际 0 task / 0 run_id，且该 pid 在 A100 上不存在**。→ 见 G5。

13. **§390.3 handoff 记录的 pilot→全量时序 vs 实际**
    - handoff §7 写「pilot 收尾 → 自动起全量」；实际是 **psom 直接吃全量，其余 5 mode 稍后补**；且 monitor 17:09 判定已退出而 chain 18:41 仍 alive。→ 见 G5。

---

# ⚠️ 被 §397.10 修正的条目（CORRECTION 节，单独成节）

> **§397.10 是 CORRECTION 节。读 §397.4 / §397.9 必须连它一起读。**
> 原文纪律: 「append-only 纪律: 不改上面的历史记录, 在这里作废。」
> 五条纠正全部来自 user，原文自评根因: 「**我从刚读的代码往前推, 没先查已有的测量/裁定**」。

### (1) §397.9 的 id-namespace 表**不完整** → 其结论表述作废
- 原表: 「DOM / P-prompt = 原生 · P-text / P-SoM = 1..K」→ **在 4 臂 quadrant 内对，但漏了 SoM 与 Vision**。
- 正确（`runner/main.py:2853-2860` 权威）: **SoM 也是 renumber 的** —— SoM-family = **som / phantom_som / phantom_text** 拿 seq-keyed 1..K map；AXTree modes = **dom / p-prompt / vision** 留 None。→ **compact namespace 是三个 mode 不是两个**。
- **Vision 根本没有 element id**（实测动作 `{"coordinate":[494,375],"coordinate_type":"qwen_0_1000"}`）→ **其幻觉率 0.000 是结构性不适用，标成 "native" 是错的**。
- 连带作废: 旧 digest 里 **"SoM 0.08/0.45/8.84 vs dom"** 与 P-SoM-vs-dom **同样跨 namespace 无效**；`write_digests.py:170` 的 "P-SoM 干净 2.3-24.8× vs dom" 已加 🚨。
- **本聚合处理**: D1 / D2 / D3 三个主题全部就地标注；D1 的九格表列为"已作废（禁止跨 namespace 引用量级）"；D2 以 §397.10(1) 为准并保留 §397.9 旧表作为"已作废"对照；D3 的 §397.9 替代论证与本条并列标 ⚠️（矛盾清单 #9）。

### (2) id 噪声**早就被正经量过**，不该从探测器灵敏度重新推
- 权威测量: `b0_paired_idperturb_replay.py` + `docs/checkpoints/probes/b0_paired_idperturb_20260529_*.json`，**配对设计 + id-agnostic 判定**。
- 数值: **B1 组内一致性 1.000 / 1.000，id-shuffle 改变决策 20.0%；B0 0.867 / 0.890，12.5%**。
- 关联: **§298.3 早写明 dom / p-prompt 保原生 nodeId → 仍承担此噪声；SoM-family 被 AMENDMENT_07 换 1..K 消除了它**（§299.4 实证 **SR 30.4%→27.2%, Δ−3.2pp**）。
- **推论（原文明标"尚未验证"）**: axis-1 = |P-text \ P-SoM| **两臂同 regime**（都 1..K）；axis-2 = |P-prompt \ P-SoM| **跨 AMENDMENT_07 的 id-regime 边界** → 「这可能就是 axis-2 (2.09pp) > axis-1 (1.35pp) 的原因 —— 更大的那个轴恰好是跨 regime 的那个」。**此推论尚未验证。**
- **本聚合处理**: D4 / D5 独立成主题；该推论只在 A3 与 D5 的 caveats 里出现，**并明写"尚未验证"，不作为结论**。

### (3) §397.4「全 archive 只有一对同模式重跑且被污染」**是错的**
- 原因: 「**我搜漏了** —— 只在 `results/visualwebarena/phase1/` 按目录名前缀搜」；归档 run 用**合并命名**（`B0_3mode_reddit_20260422` 一个目录含三个 mode）→ 前缀搜法看不见。
- 实际（笔记版）: manifest **15 组**有 2 个 run + `results/repro_replicates/B0_vision_classifieds_R24792_clean_replicate`（§302 明称 post-B-1860 **clean** replicate）。
- 实际（PROGRESS.md + 本聚合复核）: **19 组 ≥2-run** + **两个** clean replicate。⚠️ 两版并列，见矛盾清单 #1/#2。
- **早就有专门工具**: `compare_cross_run_same_condition.py:227-247` 打印 `self_drop archive->current = sum(y1[i] and not y2[i])`，**正是 H3 轴的估计量形式**。
- **§302.1 早就有数**: 干净 vision pair（B0·cls, n=224）**self_drop 6.7pp / 7.6pp，discordance 14.3pp，κ=0.614**；而 **H3 轴是 1.35 / 2.09pp，小 3-5 倍**。
- 下游: **paperA Limitations "No same-mode replicate exists in our data" 是假陈述**（已提交且已同步 Overleaf），已改成如实叙述 + 明说是 **B0-MoE 上界、不可直接外推到本地确定性 backbone、真正需要的是本地格同测量而我们没跑**。
- 工具自带 caveat 必须一起搬: `instability proxy, NOT H1 drop-one bias correction; 小样本/可能混代码版本 = upper-bound risk trigger`。§302 已 RETRACT 线性减法（12.1% ≈ 10.5% + 1-2pp）为 **category error**；§300.2 记跨 GPU type flip 率 **±3-5pp**。
- **「所以这些数不许再做加减法，只能各自带 scope 引用。」**
- **本聚合处理**: D6 / D7 / D8 三个独立主题，每条各自带 scope；A3 里 H3-vs-self_drop 的对比**只并列不相减**；D4/D5/D6/D7 每条 caveats 都复述禁算术令。

### (4) DGX mechanistic「已完成」也是错的（user 第四次纠正）
- 原说法「DGX 38617 已正常完成, pid 已退」→ **只对了一个 cell，整体是在跑**。
- 真对象是 **sweep 驱动 pid 38603**（`logs/mechanistic_canonical/.sweep.pid`）+ supervisor（poll 300s, ≤40 restarts）；`queue_mechanistic_canonical.sh` 是 **24-cell sweep, deadline 2026-08-01**；38617 只是 **cell 2/24** 的 worker（它确实 03:34 干净跑完，log 结尾 `pilot DONE`，24 task，manifest emitted）。现 cell **3/24 `p1_rev_reverse_cls`** 在跑（worker pid 1638252, 21.7 GB VRAM）。
- **单 cell ~800-845min，2 个烧了 ~27h → 08-01 会在第 7-8 个 cell 截断，永远到不了 24**。
- 「错误形状与本节 (1)(2)(3) **完全同型: 查代理量（worker pid）而非真对象（sweep 驱动）**」→ Phase 0 台账的 `COMPUTE` 类必须记 **驱动 pid / pidfile**，不是 worker pid。
- **本聚合处理**: G6 尾部标注；台账中 §397.8 的两条收尾记录（含 A100 chain 状态）均带此 flag，已在 G5/G6 就地标注为"交接时点快照，且完成判断被 (4) 纠正"。

### (5) 凭空造了一个算力冲突（user 第五次纠正）
- 原说法 mechanistic sweep「在挡 `task_b3_mimo` 的 DGX 适配窗口」→ **假的，不挡任何东西**。
- ① `task_b3_mimo` frontmatter 自己写着 **fire 在 A100**（"fire 2026-08 上旬 (A100, 12 conditions ≈ 2-2.5 周)"），引了这行却没读进去；② **DGX 本来就是**留给 dev/curation/mechanistic 的共享争抢机（paper-grade fire 2026-05-14 已迁 A100）；③ sweep 占 **21.7/~128 GB 单进程**，在 CLAUDE.md 明写的"一次 1-2 个进程"范围内，**与 B3 适配可共存**。
- 「这次的错误形状与 (1)-(4) **不同**: 前四次是"没查已有的东西"；这次是**没应用我 context 里已经加载的规则**（CLAUDE.md 的三层算力表 + host-role 分工）」。
- **本聚合处理**: 不作为测量条目收录（无数字），仅在此记录；G6 caveats 指向本节。

---

# 未归主题的孤条

以下记录**未能归入上述任何主题**（多为一次性程序性事实，无同主题可聚合），原样保留指针：

1. **§316 P79 som 变体的无效点击率（VWA-faithful 审计）** — 纯 StaticText 误点 som B0 **2.8%** / B1 **3.4%**；"真正浪费"（非互动 role + `page_changed=False` + `target_tag=None`）som B0 **2.9%** / B1 **14.6%**；dom 对照纯 StaticText B0 3.6% / B1 3.9%，整体浪费 B0 **9.5%** / B1 **11.1%**。
   caveats（一字不改）: 「14.6% 用 `target_tag=None` + 无导航 当 VWA 会屏蔽 的**代理口径，非 Interactable flag 精确复刻**（reviewer-proof 需走 VWA `get_page_bboxes` Interactable 列）；locator route 会从 image/StaticText 的 native id **往上爬到最近可点祖先**（B1 点 image 379 次，369 次爬到 target_tag=A 正常导航）故"点缩略图"多数非浪费」。→ `docs/checkpoints/_status/audit/audit_cv1_vwa_faithful_som.md`
   [聚合者推论：与 D9 walk_fail 同族，但判据（无效点击 vs walk_fail）不同，未合并。]

2. **§325 B-260（输入框不清空导致 query 累积）的站点分布 forensic** — 累积**只在 Magento (7770)**: 64 步 / 4 task；**classifieds (9980) 915 个 type 零累积**；Gemma 每步 type 值恒定 "lion pillow" 而 URL q-param 增长 → **是环境没清空不是 agent 不懂 append-vs-replace**。
   caveats: 「这反驳了既有 B-260 entry 的 "wrapper fill() → paper-grade data unaffected" 断言在 Magento 成立 = B-260 的 Magento gap **非新 bug**；对 Phase 1a cls/red 成立，**对 Phase 1b shop 不成立 = shop fire blocker + 须改 paper §3.5 加 shop scope caveat**」。

3. **§383.2 / §383.4 的 producer 级事实** 已分别归入 A5 / B14，但 **§383.4 的 "V2 tier 化（6 路→2 路）类数减少但同样救不回来 —— 绝对标签量才是约束"** 一句在 B7 caveats 中保留，未单独成题。

4. **§366.2 三站非 N/A fuzzy_match = 0/0/0** 已并入 H8，但其 blocked-on 协议（instrumented paired-replay）本身无数字，未单独成题。

---

*本文件由 D4 批 217 条 MEASURED 记录聚合而成。共 **58 个主题**（A 6 · B 16 · C 1 · D 9 · E 6 · F 5 · G 6 · H 9）+ 孤条 4 条，矛盾清单 13 条，§397.10 修正 5 条。数字一律原样抄写，未做任何算术。*

---

# 附录 — §398 起的补充（结论层建成之后新增）

> D1–D4 四批覆盖到 **§397.10**，是 2026-07-28 结论层建成时的截止点。此后的 § 由本节补。
> §399/§400/§401 已分别写入 B11 节、`INDEX §7`、`measured_qualitative §5`；本节只补 **§398**。

## Z1. §398.2 Phase 0b — 同模式重跑的噪声地板

**当前值（两对 clean replicate，均 B0·classifieds，n=224）**：

| pair | SR archive → current | Δ SR | self_drop a→c | self_drop c→a | discordance | κ | flips 分类 |
|---|---|---|---|---|---|---|---|
| **dom** R31194 ↔ R21557 | 15.2% → 17.4% | +2.2pp | **4.9pp** (11/224) | **7.1pp** (16/224) | 12.1pp | 0.559 | 27 model-nondeterm / 0 reset / 0 未分类 |
| **vision** R24792 ↔ R32024 | 24.1% → 25.0% | +0.9pp | **6.7pp** (15/224) | **7.6pp** (17/224) | 14.3pp | 0.614 | 30 / 0 / 0 |

**三条判读**：
1. **地板不是 vision-specific** —— 同一 (model, site) 的两个不同 mode 给出 4.9–7.6pp
   self_drop，两对均 **0 reset 污染**（起始 URL 全一致）。vision 行**逐位复现 §302.1**
   （6.7/7.6/14.3/0.614）；**dom 行是新的**（README:52 此前只有 partial@88 的快照）。
2. **pre-fix 上界这条路是断的** —— manifest 有 **19 组** ≥2-run（不是计划说的 15），
   但多数第二个 run 目录不在磁盘。唯一存在的 `B1_3mode_classifieds_20260413`：
   目录在、`test -d` 过、manifest 写 `expected_n=234`，三个 subdir **各只剩 1 个 episode**。
   ⇒ **B1/B2 本地格 replicate 一个都没有**（这条直接决定 §8 limitations 怎么写）。
3. **fixed-marginal permutation null 从未被执行过** —— 跑出来 **0/24 arm×cell 为正**；
   且 null 方向不对（6/6 cell 观测 Jaccard 在 null p95 之上，中位数 3.75×–23.04×，
   高重叠必然压低独解）。⚠️ 但它只在旧 omnibus draft 与 `paper_drafts_locked/` 里，
   **`paperA/` grep 全空** ⇒ 不是待投论文的活缺陷。

**caveats**: 工具自带警告须原样携带 —— *"instability proxy, NOT H1 drop-one bias
correction；same-mode discordance ≠ P-SoM-vs-5-competitors false-unique；小样本/
可能混代码版本 = upper-bound risk trigger"*。⚠️ **禁止与其他 noise 数字做算术**（§302 类别错误）。

**证据**: §398.2；`docs/analysis/cross_sites/phase0b_noise_floor.md`；
`compare_cross_run_same_condition.py`（未改代码）

## Z2. §398.2 id-regime 实证 — AMENDMENT_07 确实生效

**当前值**（模型实际输出的 element_id，min/median/max）：
B1·**p-som** 1/12/68 · **p-text** 1/13/72（mark_count 33/35）
vs **p-prompt** 139/4074/26235 · **dom** 2/3606/61833

⇒ SoM-family 拿到的是 1..K 紧凑编号，DOM / P-prompt 仍是原生稀疏 nodeId ——
**§397.10(1) 的 id-namespace 归属正确**，AMENDMENT_07 的重键在生产数据上确实生效。

**证据**: §398.2
