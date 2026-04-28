# Paper 1 Strategy & Notes (Phantom-SoM)

> **Paper-level strategy notebook** for the毕设 paper.
> 含 theory framework / findings 列表 / risks / cascade / router design /
> advisor align checklist. 内容会随 paper writing 逐步落到 paper drafts.
>
> **职能分工**:
> - **paper_planning.md** (此文档): paper strategy, theory, findings, risks
> - **next_steps.md**: action ledger (active processes, codex queue, next 3 actions)
> - **paper drafts** (`docs/analysis/paper_drafts/`): final paper prose
> - **实验笔记** (`docs/checkpoints/实验笔记.md`): time-order chronicle (历史 record)
>
> **Last updated**: 2026-04-28

---

## §1 Paper Hook + Tagline

**Core finding**: Phantom-SoM (SoM prompt + `[SOM_MARKS]` text + no image) is a **hidden 4th routing arm** for web agents with **4-fold drop-in property**:

| Drop-in property | Evidence |
|---|---|
| (a) **Cost ≈ DOM** | `[SOM_MARKS]` 是 AXTree regex filter, 不需 bbox/image (验 `som.py::_extract_text_marks` line 24); text token ±7% (3437 vs 3661 reddit / 3008 vs 2948 cls) |
| (b) **Latency ~50% lower** | cls SoM p95 74s vs Phantom-SoM 18.2s = **4× faster** (no image encoding stage) |
| (c) **Signal AUROC ≥ baseline** | 5-mode 全 `overall_usable=True`; red Phantom-DOM verbalized 0.793 是 5-mode 最高 (超 baseline 0.766) |
| (d) **Drop-one oracle 1.7-3.3pp** | red Phantom-SoM 3.33pp drop-one (≥ SoM 1.90pp); cls 2.56pp |

**Paper one-liner (for advisor pitch)**:
> "Phantom-SoM identifies a hidden text-only routing arm in SoM-style web agents that achieves DOM-level cost and ~50% lower latency while contributing 1.7-3.3pp drop-one oracle value. The arm is created by skipping the marked-image draw and image-token inference path — no model retraining, no prompt change, no infrastructure overhead. We characterize its mechanism via 3-axis ablation (representation × prompt × image), explain its site-modulated effect (cls visual-rich win for SoM, red text-dominated win for Phantom), and demonstrate routing infrastructure drop-in (signal AUROC ≥ baseline)."

---

## §2 Theory Framework — 3-axis Hierarchical (validated)

### Axis 1: Representation (PRIMARY, first-order SR effect)

```
AXTree vs [SOM_MARKS] → action surface + trajectory basin
→ Phantom modes 获得 4th routing arm (因为 [SOM_MARKS] obs)
```

**LLM mechanism**:
- Token distribution shift (hierarchical metadata vs flat indexed list)
- In-context learning bias (pretraining data context)
- Long-context attention degradation (Liu 2023 "Lost in the Middle")
- Output format priming

**Evidence**: fig3 strategy gradient (reddit verified §103 N=48, cls live extension), fig5 category × mode

### Axis 2: Prompt (multi-dimensional task-conditional decision prior)

**Falsified hypothesis**: "prompt 只影响 commitment confidence" (codex `5821387` falsifies)

**Replaced theory**: Prompt acts as task-conditional decision prior over:
- (a) search phrasing
- (b) candidate selection (ranking/disambiguation)
- (c) backtracking strategy
- (d) commitment confidence (FP gap evidence, **subeffect not唯一**)

**Evidence**:
- Phantom-DOM ∩ Phantom-SoM Jaccard 0.45-0.54 (task pool 显著 disjoint despite same SR)
- 6 case studies (codex `5821387` phantom_dom_vs_som_diagnostic.md)
- N=48 verified anchor: Phantom-SoM FP gap 2.08pp vs Phantom-DOM 6.25pp

**Lit support**: Persona priming (Salemi 2024), in-context learning (Min 2022), Sclar 2024 prompt format sensitivity, Mishra 2022. Task-pool divergence Jaccard <0.5 是 paper unique empirical finding.

### Axis 3: Image (8-channel multi-dimensional, codex `7106d2e` validated)

**Falsified hypothesis**: "Visual-hijack 是唯一 image effect" (codex `7106d2e` falsifies)

**Replaced theory**: Image is **bidirectional modality fusion** with multiple sub-channels:

**Helping channels (4)**:
- 3a Spatial grounding (cls 5 tasks)
- 3b Visual context disambiguation (cls 13 / red 2)
- 3c Element disambiguation (cls 10)
- 3d State/action recognition (cls 1 / red 6)

**Harming channels (6, refined per user critique 04-28)**:
- **3e False visual confidence (image-over-text)** [MAIN red 60%, 9/15 failures]
  - Mechanism: image-text alignment pretraining bias → premature commit
  - Lit: Li POPE 2023, Yang 2024, Yu 2024 (object hallucination)
- **3f Text-over-vision fallback fail** (反向 modality bias) [cls task 24 verified]
  - Mechanism: language prior dominance → ignore actual image content
  - Lit: ⭐ Tong 2024 "Eyes wide shut" (NeurIPS), Bitton-Guetta 2023
- **3g Visual saliency hijack on image content** [red task 0/167 verified]
  - 实测: 15× cycles same image link, element_id +2846 each cycle (page reload 不脱)
  - 与 SoM density 不直接相关 (mark_count 117 ≠ outlier vs P95 127)
- **3h SoM occlusion** [§100 ground truth, B0/B1 都 affected]
  - 量化: B0/B1 reddit_task_6 mode-SoM 18%/15% vs NoMarks 78%/75% → **-60pp OCR**
- **3i SoM numeric label attention hijack** [§100 verified, density-dependent]
  - 量化: B1 num_ids 0→**446** at 128 marks; mode-WithText 立即降 0
- **3j Visual misdirection** (visual saliency drift)

### Site-modulated framing (LLM-level explanation)

```
Task 需 vision (cls visual-rich):  3f text-over-vision 主导失败 → Phantom-SoM 失败
Task 不需 vision (red text-dom):    3e image-over-text 主导失败 → SoM 失败

Net effect:
  cls: helping 29 vs harming 13 → image win +6.84pp adj SR
  red: helping 8 vs harming 15 → image lose -3.33pp adj SR
```

### Capability layer (B0 vs B1, lazy minimization §101.九)

**Lazy Minimization Hypothesis** (4B small VLM signal selection):
```
优先级: 数字标签 (高对比) > 文本 (结构化 token) > 截图内容文字 (低对比 + 遮挡)

物理解释 (capability-aware routing 的 mechanism):
  对 4B small VLM, 视觉处理成本/收益比更差
  → text-over-vision bias 在 small VLM 更强 (与 Asadi 2026 anchor 一致)

支持证据 (probe 全 align):
  B1 SoM 高密度 → num_ids 446 (数字 = easy)
  B1 WithText num_ids → 0 (文本可用就忽略截图)
  B1 NoMarks ≈ B0 NoMarks (视觉本身可用, 只是不优先)
```

**B0→B1 cross-site shift** (cross_site_pattern_consolidation `ab86019`):
- SoM hijack flip cross-site: cls **+50.0pp**, red **+33.3pp** (vs aggregate +43.7pp)
- Capability-modulated amplification of harming channels

### Cross-axis interaction LLM mechanism

```
Repr × Prompt: 同 obs token, 不同 prompt prefix → attention(obs|prompt) 不同
Repr × Image: 同 prompt, image vs no-image → cross-modal vs pure self-attention
Prompt × Image: SoM prompt + no image (Phantom-SoM) = "phantom prompt" mismatch
Site × axis: cls visual-bound → image helping dominate; red text-dom → harming dominate
```

### Paper contribution position (Section 5 framing)

**不是发现新 LLM mechanism** (每 axis 都有 prior literature). Contribution 是:
1. **Systematic isolation** (2x2 ablation matrix)
2. **Joint quantification** (Jaccard 0.33-0.55 disjoint task pools)
3. **Site-modulated framework** (cls vs red natural ablation)
4. **Drop-in deployment claim** (4-fold drop-in property)
5. **Paper-grade clean re-run protocol** (watchdog auto-clean + 重跑 verify)

→ Section 5 prose 应当 frame 成 "we don't propose new LLM mechanisms; we systematically decompose them in web-agent multi-step setting + provide drop-in deployment evidence".

### Literature gap 5-dimension (§103 anchor)

| Dimension | 是否有人 isolate marks-text? | 关键 papers |
|---|---|---|
| A. SoM lit | **No** — 全 bundle text+image | Yang 2023, Magma, Ferret-UI 2 |
| B. Representation routing | **No** — 现 routing model-level/modality-level | RouteLLM, Avenir-Web |
| C. AXTree vs flat list | **No** — head-to-head 缺失 | FOCUSAGENT, VWA baseline |
| D. Prompt format sensitivity | **Yes (theory anchor)** — 但无 web agent 应用 | Sclar 2023, Mishra 2022 |
| E. Cost-aware web agent | **No** — focus prune 不 reformat | FOCUSAGENT pruning, ModServe |

**Closest prior** = FOCUSAGENT (text 压缩) + Yang 2023 (SoM with image). 本工作 = unprecedented synthesis.

完整 deep research: `docs/literature/The Novelty and Efficacy of Set-of-Mark Text as an Independent Representation Routing Arm in Web Agents.md`

---

## §3 Findings 列表 (10 paper-grade findings, FRESH 04-28)

1. **DOM-only mode visual-required task 系统性失败** (fig5 Cat B)
2. **Image effect 是 cross-capability + site-modulated + 8-channel 现象** (codex `7106d2e`)
   - Helping × 4 + harming × 6 (含 §100 ground truth)
   - Site-modulated: cls helping dominate +6.84pp, red harming dominate -3.33pp
   - Capability-modulated: B0→B1 +50/+33pp shift cross-site
   - Bidirectional modality (image-over-text 主导 red, text-over-vision 主导 cls vision-needed)
3. **DOM reddit search-loop 22.7%** (vs SoM 12% Phantom 10.8% gradient, fig3)
4. **Phantom-DOM cls adj 14.53% ≈ DOM 14.10%** — prompt-as-decision-prior direct evidence
5. **Phantom-SoM cost ≈ DOM cost** — `[SOM_MARKS]` 是 AXTree regex filter (code-level 验证)
6. **Image tokens per step (measured medians)**: red 733 / cls 1064 (`4d63c9f`)
7. **Cost gap B0 vs B1 ~30×** (fig7 Pareto frontier)
8. **Phantom-SoM unique tasks 验证** (fig8): red 7 task (3.33% drop-one), cls fresh 5 (vs stale 1)
9. **⭐ Phantom 模式 routing signal 完整 + ≥ baseline** (新发现 04-28):
   - 5/5 phantom condition `overall_usable=True`, infra 直接复用 baseline
   - red Phantom-DOM verbalized AUROC 0.793 是 5-mode 最高 (超 baseline 0.766)
   - cls behavioral 主导 (action_diversity), red verbalized 主导
   - Token-level 全 non-discriminative (paper 避免 claim)
   - **paper 4-fold drop-in property 第 (c) 条**
10. **Watchdog auto-clean protocol — paper-grade 数据 100% pure** (04-28 audit):
    - cls (OSClass) NOT LOGGED IN events 全 auto-cleaned + 重跑 (verified mtime + DOM)
    - red 0 events (Postmill cookies 持久)
    - shopping FPC false alarm fixed (`bin/magento cache:disable full_page` + PowerShell hook)
    - paper-grade 数据 0% wasted task

### Honest framing (avoid over-claim)

- Phantom-SoM red 13.81% > SoM 10.48% **不是 unconditional dominance** — within 2σ noise floor
- 主 narrative: **site-modulated representation effect**, NOT "Phantom #1 routing arm"
- cls SoM 21.37% 显著领先 Phantom-SoM 14.53% (+6.84pp adj) 反例必须明示

---

## §4 Paper Section Status (2026-04-28, 8 sections final scope)

| Section | evidence 质量 | 状态 | Hard blocker |
|---|---|---|---|
| 1 Intro | ✅ 已写 (786w + 4-fold drop-in framing + conservative framing) | done `62c1380` `ef29add` | — |
| 2 Background + paper.bib | ✅ 已写 (1514w, 16 entries) | done `206cd93` | 待 codex #10 expand to ~38 |
| 3 Definition + Ablation | ✅ 已写 (863w, token re-estimate corrected) | done `13b9608` `4d63c9f` `48db047` | — |
| 4 Empirical Findings | 🟡 80% (figures FRESH ✅ + B0 5-mode FRESH, prose 待 update) | data ready | codex #11 fresh prose (~30K) |
| 5 Mechanism | 🟡 90% evidence (3-axis × 8-channel × bidirectional × §100) | data 完整 | codex #13 prose (~50K, 待 #10 lit) |
| **6 Routing (Tier 1+2)** ⭐ NEW | 🟡 40% (signal AUROC ≥ baseline `9d7e99f`, infra scaffold) | scaffold ready | Tier 1 prototype (~3 天) + Tier 2 first-step trigger (~7-10 天) |
| 7 Generalization | 🟡 40% (B1 capability profile done) | partial | shopping (跑中) + WA + cross-model (Claude) |
| 8 Discussion + Implications (含 sustainability + 4-fold drop-in summary) | ❌ 未写 | end-stage | 全部 data done |

**Section 1-3 总 prose 3163 words** (paper-ready). Section 4 1725w draft 待 fresh data update. Section 5/6/7/8 待写.

### Section 6 Routing — 详细 outline

paper Section 6 不是 Section 7 Generalization 的子部分, 是**独立 contribution**. 内容:

```
6.1 Routing problem formulation
  - 5-mode arms: DOM / SoM / Vision / Phantom-SoM / Phantom-DOM
  - Per-task feature space (instruction + browser meta + step-1 trigger signals)
  - Target: max adjusted SR / cost-aware / Pareto

6.2 Tier 1 — task-level oracle router (offline supervised)
  - Feature: TF-IDF (task instruction) + binary {has_ref_image, has_finish_string_match}
  - Model: Logistic regression / random forest (interpretable + small data)
  - Train/eval split: 80/20 of cls + red task pool
  - Result: routing pool oracle bound vs learned router gap

6.3 Tier 2 — first-step-trigger router (online cascade)
  - First step on cheaper mode (e.g. DOM) → trigger signal at step 1 → escalate to expensive mode
  - Trigger features: response confidence / action_diversity / behavioral signals
  - No test-time leakage (step-1 obs ≠ task feature)
  - Cost vs SR Pareto improvement

6.4 Routing infra drop-in property (4-fold #c)
  - Phantom modes signal AUROC ≥ baseline (5/5 usable, red Phantom-DOM 0.793 highest)
  - 不需要 retrain signal extraction infra
  - paper claim: "router trained on baseline 可 directly extend to Phantom"

6.5 Visualization (4-fig stack, see §10)
  - Fig A: 3-panel multi-metric Pareto
  - Fig B: Cumulative SR vs Budget curve ⭐ (cost-aware 顶刊套路)
  - Fig C: Routing decision Sankey
  - Fig D: Per-task savings histogram
```

### Section 8 Discussion — 详细 outline (含 sustainability + green AI)

```
8.1 4-fold drop-in property summary
  (a) Cost ≈ DOM
  (b) Latency ~50% lower (cls SoM 74s → Phantom 18s)
  (c) Signal AUROC ≥ baseline (router infra drop-in)
  (d) Drop-one oracle 1.7-3.3pp

8.2 Site/capability-modulated mechanism
  cls visual-rich → image helping channels dominate (3a-3d)
  red text-dominated → image harming channels dominate (3e-3j)
  B1 amplify failure dimensions (lazy minimization, density threshold)

8.3 Sustainability + Green AI implications ⭐
  - cls Phantom-SoM latency 4× improvement (production-relevant)
  - Regional carbon sensitivity (fig9): Phantom-SoM advantage region-dependent
    (large for India 632 g/kWh / Poland 773; small for France 85 / Norway 29)
  - Multi-metric Pareto: cost + latency + carbon 三向 drop-in
  - Lit anchor: Strubell ACL 2019, Patterson 2021

8.4 Limitations + future work
  - Single benchmark family (VWA + WA), single backbone model family (Qwen + Claude)
  - Tier 3 online learning router 留 future work
  - Cross-model meta-policy (cross model family routing) 留 future
```

---

## §5 Final Scope + 顶刊概率

### Final scope (paper 完整版)

```
Benchmark: VWA 3 站 (cls 234 + red 210 + shop 466) + WA 3 站 (red 106 + shop 192 + sa 182)
           = 6 sites, ~1390 task per condition
Models:    B0 (Qwen3-VL-235B proxy) + B1 (Qwen3-VL-4B local) + Claude Opus 4.7
           = 3 model families
Modes:     DOM / SoM / Vision / Phantom-SoM / Phantom-DOM = 5 modes
Cells:     6 sites × 3 models × 5 modes = ~90 cells (~125K episode total)
+ Router:  Tier 1+2 (oracle + first-step trigger), 实际 deploy on agent
+ Multi-metric: cost / P95 latency / carbon (B1 measured + B0 estimate)
```

### 顶刊概率（final scope + multi-metric/green AI 加成 后）

| 投稿目标 | 概率 | 投稿优先级 |
|---|---:|---|
| **NeurIPS / ICLR main** | 45-60% | Tier 1 stretch |
| **ICML** | 40-55% | Tier 1 stretch |
| **ACL / EMNLP main** | 50-65% | Tier 1 |
| **MLSys** | **75-85%** ⭐ | **Tier 1 safe** (drop-in framing 完美 fit) |
| WWW / WSDM | 75-85% | Tier 2 |
| NeurIPS D&B | 70-80% | Tier 2 |
| **TMLR (journal)** | **75-85%** | **保底** |

→ Final scope 完成后, paper 顶刊出版几乎 100% (cascade NeurIPS → ACL/EMNLP → MLSys → TMLR)

### Multi-metric + Green AI axis 加成的 paper-level 价值

1. **Differentiator**: 现 web-agent paper (VWA/WebArena/SeeAct/SoM/FocusAgent) 几乎全不报 carbon
2. **Multi-metric Pareto** 在 ML 顶会近年是 expected
3. **三向 drop-in** (cost+latency+carbon) narrative 立体
4. **Green AI** 是顶会新兴 axis (Strubell ACL 2019, Patterson 2021)

**Caveat**: green AI 是 second-order, 不能抢主线 "hidden routing arm + drop-in deployment"

---

## §6 Critical Risks + Mitigation (4 risks, 决定接收 vs reject)

### Risk 1: Execution quality（顶刊成败 #1 因素 ⚠️⚠️⚠️）

90 cells × ~1390 task = ~125K episode. 任何 cell 跑 sloppy (auth bug / cross-contam / 数据污染 / FP 没处理) 都被 reviewer 抓出.

**Mitigation**:
- 维持 paper-grade re-run 协议: reset between conditions, exclusive same-site B0 XOR B1, watchdog auto-rederive
- 每 cell 完成后立刻 `make analyze` + manual audit gallery
- **不在 execution quality 妥协**

**Status (04-28)**: ✅ B0 cls + red 5-mode 100% paper-grade clean (watchdog auto-clean verified, 0% wasted task)

### Risk 2: Story discipline ⚠️⚠️

6×3×5 cells 容易让 paper 变 "data dump". 顶会 reviewer 反感 "everything but the kitchen sink".

**Single narrative**: "Phantom-SoM is hidden routing arm + we explain why + we route on it + here's the cost saving".

其他 finding (capability shift / category profile) 都是 supporting evidence. Section 4-5 each ≤4 pages, supplementary 装其余.

### Risk 3: Router design ⚠️⚠️

Router 只比 best-single-mode 提升 1-2pp 被 reviewer 说 "不值". Oracle features (test-time leak) 直接 reject.

**Router design tiers**:
- **Tier 1 (must-have)**: Oracle router — task feature → best mode lookup, train/test split
- **Tier 2 (great-to-have)**: First-step-trigger router — 看 step 1 obs 决定 mode, no test leak
- **Tier 3 (stretch)**: Online learning router — mid-trajectory escalation

Tier 1 + Tier 2 就够顶会 contribution; Tier 3 stretch goal.

**Realistic timeline**: ~3-4 周 (vs 之前估 2-3 周)

**Minimum viable router** (start ~3 天 prototype):
```
Feature:  task instruction TF-IDF + binary {has_ref_image, has_finish_string_match}
Target:   max adjusted SR
Model:    Logistic regression (interpretable + small-data friendly)
Train:    cls + red 6 mode, 80/20 split
Baseline: random / best-single-mode / rule-based ("if has_ref_image → SoM else → Phantom-SoM")
```

### Risk 4: Negative results 必须诚实报告 ⚠️

某些 cell 可能反 trend (e.g. Claude shopping Phantom-SoM 不 work). **绝不 cherry-pick**, reviewer 看出直接 reject.

**Mitigation**: 诚实报告反而强化 mechanism claim ("effect 是 task-type/capability bound, 不是 universal").

---

## §7 Investment Cascade Plan

```
Round 1 (T+12 周, paper done):
  → MLSys 2027 (deadline 通常 9-10 月) 或 NeurIPS workshop (Maria 推荐)
  → 75-85% expected outcome  ⭐ first paper friendly venue

Round 2 (rejection 或 timing 错过):
  → ACL / EMNLP main (industry track 友好)
  → 50-65% expected outcome

Round 3 (still rejected):
  → NeurIPS / ICLR main (大幅修改 narrative)
  → 45-60% expected outcome

Round 4 (保底):
  → TMLR (journal rolling review)
  → 75-85% expected outcome
```

**Modified strategy** (per first paper considerations):
- 不把 NeurIPS/ICLR 作 round 1 (lottery + first-paper baggage)
- MLSys 是 strategic safer bet (drop-in framing 完美 fit)
- Maria's 推荐 channel 在 sustainability workshop / AI4SD venue 最有效

期望出版 venue 链 ~99% (5 站 5 model deployed-router scope 没法被全拒).

---

## §8 Router Design (Tier 1+2)

### 5 个关键设计决策点 (each requires ablation)

| 维度 | 选项 | 难点 |
|---|---|---|
| **Feature** | task NLP / browser state / step-1 trigger / capability / audit cat | audit cat 是 leak; small data overfit |
| **Target** | max SR / SR-per-cost / Pareto / budget-constrained | multi-obj weight 选 |
| **Granularity** | task-level / step-level / confidence-triggered | step-level 重跑 2x cost |
| **Cascade** | 单 router / B1→B0 escalation / rule+ML hybrid | escalation 实验代价大 |
| **Baseline** | random / best-single-mode / oracle / rule-based | best-single-mode 是 hardest baseline |

### Realistic timeline (paper 真正最值钱的工作量)

```
Tier 1 (task-level oracle): ~5-7 天
  ├─ Feature engineering (task NLP + browser meta): 2-3 天
  ├─ Train/eval split + baseline 对比:               1-2 天
  └─ Ablation (各 feature 组的 contribution):         1-2 天

Tier 2 (first-step trigger / cascade): ~7-10 天
  ├─ 重新跑 step-1 切换实验:                         3-4 天
  ├─ Confidence threshold tuning:                  1-2 天
  └─ Cascade ablation:                             2-3 天

Total: ~3-4 周
```

### Routing infra 现状 (paper 1 直接用)

- 4 baselines + 5 phantom × `confidence_summary.json` (`overall_usable=True`)
- Behavioral signals AUROC 0.682-0.748 (cls behavioral 主导, red verbalized 主导)
- Verbalized signals AUROC 0.701-0.793 (red Phantom-DOM 0.793 是 5-mode 最高)
- Router scaffold: `p79/experiment/router.py::RuleBasedRouter`
- **Phantom modes 直接复用 baseline signal infra** (drop-in routing claim 第 4 fold)

---

## §9 Advisor Align Checklist

### Meeting #1 (~Week 3, cls+red+shopping done)

| 决策 | Options | 推荐 | 影响 |
|---|---|---|---|
| Router scope | (a) Tier 1 only / (b) Tier 1+2 / (c) Tier 1+2+3 | (b) Tier 1+2 | paper main contribution 强度 |
| Cross-model | (a) Skip / (b) Claude Opus 4.7 only / (c) + GPT-4o/Gemini | (b) Claude only | $70 budget vs scope |
| 单 paper vs 双 paper | (a) Integrated (Paper 1 含 router) / (b) Split (Paper 2 router) | **(a) Integrated** (毕设决策) | publication count vs paper depth |
| Authorship 预期 | TBD with advisor + Zekun | — | first paper credit |
| Investment timing | NeurIPS 2026 ~5 月 / MLSys 2027 ~9 月 / ICLR 2027 ~9 月 | MLSys safer | timeline 紧或松 |

### Meeting #2 (~Week 6-7, WA + Claude done)

| 决策 | Options | 推荐 |
|---|---|---|
| Paper venue (Round 1) | NeurIPS / ICLR / ACL / **MLSys** | **MLSys** (drop-in framing 完美 fit) |
| Section 6 Generalization 范围 | VWA + WA + Claude / + Mind2Web | VWA + WA + Claude 够 |
| 投稿 timing | ASAP vs polish 1-2 周 | polish 后 stable submit |

### 关键 strategic 问题 (advisor align 时主动问)

1. Maria 是否能 referee NeurIPS workshop / Climate Change AI workshop?
2. Holistic AI Zekun 推荐 industry track?
3. Paper review timing: 投稿前 advisor read pass 1 周, 改完 submit
4. 是否要做 Mind2Web pilot (advisor 偏好)
5. Claude Opus 预算: $70 上限 OK?

---

## §10 Visualization Plan (cascade router viz)

**单纯 2D cost-SR Pareto 不够 striking**. 推荐 4-figure stack:

| Figure | 作用 | 设计 |
|---|---|---|
| **Fig A: 3-panel multi-metric Pareto** | 主 figure, fig7 升级 | 3 panel: cost-SR + latency-SR + CO2-SR |
| **Fig B: Cumulative SR vs Budget curve** ⭐ | 最 striking, cost-aware 顶刊套路 | x=budget per task, y=cumulative SR; lines: random/best-single/rule/learned/oracle |
| **Fig C: Routing decision Sankey** | Section 6 解释 router 学到什么 | task category → routed mode → outcome |
| **Fig D: Per-task savings histogram** | Appendix supplementary | distribution: cost saved by routing per task |

**Fig B 详细设计** (参考 RouteLLM ICML 2024 / FocusAgent EMNLP 2025):

```
x: cumulative cost budget per task ($)
y: cumulative SR achievable
lines:
  --- random
  ··· best-single-mode (DOM/SoM/Phantom-SoM 各一条)
  --- rule-based router (handcrafted)
  ▬▬▬ learned ML router (ours) ⭐
  ─── oracle router (upper bound)
fill area: ours vs best-single-mode gap; ours vs oracle gap
```

直观论证: 在 $0.04 budget per task → 我们 router 25% SR vs best-single-mode 21%; oracle 边界 ~30%, learned router 缩小 60% gap.

**反对 3D Pareto**: rotate 才看清, paper 印刷不友好, reviewer 抗拒.

CO2 维度单独 fig E (regional sensitivity, 见 §11).

---

## §11 Cost / Latency / Carbon Multi-metric Plan

### 已有数据状况 (per `condition_summary_v2.json`)

| Backend | Cost | P95 Latency | Energy (kWh) | CO2e (kg) |
|---|---|---|---|---|
| B0 (proxy 235B API) | ✅ | ✅ | ❌ NaN (远端 GPU 不可测) | ❌ NaN (token-estimate-able) |
| B1 (local 4B GPU NVML) | ✅ | ✅ | ✅ | ✅ |

### Carbon tracker 现状 (`p79/experiment/energy_tracker.py`)

- ✅ NVML GPU measurement + 45 region intensity table
- ❌ 未 port: 220+ country DB (CodeCarbon), token-based proxy estimator, cloud provider data
- Default region: UK 220 g/kWh

### Tier 化 paper 利用

| Tier | Metric | 在 paper |
|---|---|---|
| **Tier 1 (主体)** | adjusted SR, drop-one oracle, cost/task | Section 1 hook + Section 4 main + fig7 |
| **Tier 2 (主体辅)** | P95 latency, CO2/task | Section 4 cost-aware table + Section 7 sustainability |
| **Tier 3 (附录)** | wasted cost, energy kWh, cost_efficiency_ratio | supplementary |

### Striking findings 已 measured (paper 直接 cite)

1. **B0 cls SoM P95 lat 74s ≈ 2× DOM 38s** (image inference 拖慢) ⭐
2. **B1 cls SoM energy 0.0020 kWh < DOM 0.0052** (step count 主导, counterintuitive)
3. **B1 reddit SoM > DOM energy** (site-dependent)
4. **Phantom-SoM cls cost ≈ DOM** + latency 4× 改进 = triple win

### Regional Carbon Sensitivity (fig9 already done, codex `d3dfc8f`)

`scripts/analysis/figures/fig9_regional_carbon_sensitivity.py`:
- 45 region × B1 3-mode × cls + red
- Norway 29 g/kWh → South Africa 928 g/kWh (32x range)
- Phantom-SoM advantage region-dependent (large for India/Poland, small for France/Norway)

---

## §12 References / Doc Map

### Paper drafts (final prose, `docs/analysis/paper_drafts/`)

| File | Status | Words |
|---|---|---:|
| `section1_intro.md` | ✅ done | 786 |
| `section2_background.md` (+ paper.bib 16 entries) | ✅ done | 1514 |
| `section3_definition.md` | ✅ done | 863 |
| `section4_empirical_findings.md` | 🟡 stale, codex #11 待 update | 1725 |
| `section5_mechanism.md` | ❌ 未写, codex #13 待 (~50K, 等 #10 lit) | — |
| `section6_generalization.md` | ❌ 未写 (待 WA + Claude data) | — |
| `section7_discussion.md` | ❌ 未写 (paper end-stage) | — |

### paper.bib

`results/phantom_paper/paper.bib` (16 entries, 待 codex #10 expand to ~38)

### Codex analyses (`results/phantom_paper/analyses/`)

| File | Words | Commit |
|---|---:|---|
| `disagreement_clusters.md` (B0+B1 9-cat) | — | `ded0ef6` `c4b52c3` |
| `cross_site_pattern_consolidation.md` (cls vs red shift +50/+33pp) | 1596 | `ab86019` |
| `phantom_dom_vs_som_diagnostic.md` (axis 2 prompt diag) | — | `5821387` |
| `som_vs_phantom_som_diagnostic.md` (axis 3 image 8-ch) | — | `7106d2e` |

### Other analyses

- `docs/analysis/B1_capability_profile.md` (B1 6 sections, 2245w, `03ffb2f`)
- `docs/literature/The Novelty and Efficacy of Set-of-Mark Text...md` (deep research)

### Figures (`results/phantom_paper/figures/`, all FRESH 04-28)

```
fig1 4-mode venn (2x2 B0+B1 cls+red)
fig2 drop-one oracle (2x2)
fig3 strategy gradient (2x4 reddit + cls)
fig4 two-knob diagram schematic
fig5 category × mode heatmap (B0 cls+red)
fig6 capability contrast B0-vs-B1 +43.7pp aggregate
fig7 cost-SR Pareto + deployment callouts
fig8 overlap-depth stacked bar (5-mode)
fig9 regional carbon sensitivity (B1 only)
```

### 实验笔记 § index (key findings)

- §95 adjusted_success canonical + Pareto
- §97 audit ~17500 LOC + 13 YAML
- §99 Magento auth bug + Knockout
- **§100 SoM 截图视觉 probe (B0/B1 OCR + attention)** ⭐ ground truth
- **§101.九 Lazy minimization hypothesis** ⭐
- §102 Phantom-SoM 工程实施
- **§103 Phantom-SoM 4-mode routing arm + paper narrative** ⭐
- §104+ Daily chronicle (Day 1-2 progress moved here from next_steps)

### Recent key commits (~04-27 / 04-28)

```
ae0f8e7  next_steps Day 2 update
8dde2cb  watchdog auto-clean paper-grade 100% pure verified
139afb0  router framing fix (paper 1 not paper 2)
9d7e99f  phantom routing signal AUROC ≥ baseline (4-fold drop-in)
8263d26  axis 2/3 literature deep research plan
b4bbe75  axis 3 image 8-channel framework
81613e0  axis 3 sub-mechanism refinement
00124e4  3-axis hierarchical theory framework
7106d2e  som_vs_phantom_som diag
5821387  phantom_dom_vs_som diag
ab86019  cross-site pattern consolidation
03ffb2f  B1 capability profile
ef29add  drop-in deployment punchline
48db047  Phantom-SoM cost ≈ DOM
93e413f  3-layer cost decomposition
4d63c9f  Section 3.2 token re-estimate (1064/733 measured)
```

详 `git log --oneline --since="2026-04-27"` 看完整历史.

---

## §13 Pending TODO (paper-strategic, not action ledger)

- [ ] codex #10 axis 2/3 literature deep research → expand paper.bib 16→~38 (~Wed)
- [ ] codex #11 Section 4 fresh-data prose update (~Wed)
- [ ] codex #13 Section 5 prose 写 (3-axis hierarchical + lit cite, ~Thu)
- [ ] Section 6 Generalization 草稿 (~Week 6-7, after WA + Claude done)
- [ ] Section 7 Discussion 草稿 (paper end-stage,含 sustainability + lat 4× finding)
- [ ] 二次 deep research (Section 6/7 + 全 paper revisit, paper 终稿前 Week 8+)
- [ ] Router Tier 1 prototype (~3 天, baseline + phantom 全 done 后)
- [ ] Router Tier 2 first-step trigger (~7-10 天)
- [ ] Advisor align meeting #1 prep (~Week 3)
- [ ] Advisor align meeting #2 prep (~Week 6-7)
- [ ] paper writing + revisions (~Week 8-12)
