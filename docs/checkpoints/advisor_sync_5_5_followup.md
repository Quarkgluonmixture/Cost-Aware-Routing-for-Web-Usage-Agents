# 5/5 Sync Follow-up — 想跟您 confirm 几件事

> 学长您好,
>
> 5/5 sync 因您马德里那边网卡了, 后半段 forest plot + threshold 没 propagate 过去, summary 也有点 ambiguous, 我整理一份 follow-up:
>
> - **Part 1**: 简要列我们目前所有的 innovation points (帮您 frame paper 拆分)
> - **Part 2**: paper 拆分 + Mechanistic 该独立还是 nested 这块想跟您 confirm
> - **Part 3**: 来不及讨论的 + 需 email 回的 (主要是 pre-reg threshold)
>
> 已 confirmed 的不重列: early-stop A 全 cancel / GPU 走您 5090 + Rancher / **rerun 16 cell** (我 decide 这个, B0 × {cls,red,shop} × 3 phantom + B1 × {cls,red,shop} × 3 phantom).
>
> 我 5/12-6/1 考试, 这两天能 lock 完我就 launch 实验然后专心复习, 麻烦您扫一眼回个 email 即可!

---

## Part 1 — 我们目前手里的 innovation points

### 1. Phantom routing space (核心发现)

把 web agent 的 observation 拆成 **3 axis** (text payload / system prompt / image presence) → 8-corner cube. 现在 paper 测的 6 modes 里, **4 个 corner 是"不带标注图"** (我们叫 phantom space), 这块以前没人 systematic 测过. 发现 phantom 内部不是塌成一个点, 是 **2-axis 结构** (text-flattening 跟 SoM-prompt 各自能解 P-SoM 解不了的 task). Cube 中心 P-SoM (`[SOM_MARKS]` text + SoM-prompt + 无图) 是 hero, 满足 **4-fold drop-in property**:

| | 数据 |
|---|---|
| (a) cost ≈ DOM | regex filter 不付 image token |
| (b) latency ~50% lower | 省 image inference 那步 (cls SoM 74s → P-SoM 18s, 4×) |
| (c) routing signal AUROC ≥ baseline | 5-mode 全 usable |
| (d) drop-one oracle 1.7-3.8pp per arm | 6-mode oracle vs 3-mode lift +7.14pp [3.81, 10.48] (B0 reddit) |

### 2. Routing (router design + signal infra)

**Signal 基础设施已 ready** (per-condition `confidence_summary.json`, commit `9d7e99f`), 4 个 signal family per mode 已 AUROC 实测:

| Signal family | 提取层 | cls 实测 | red 实测 |
|---|---|---|---|
| Token entropy / logit | model output token 概率 | 多数 mode 不 discriminative | 多数 mode 不 discriminative |
| Perplexity-style entropy | sequence-level | 多数 mode 不 discriminative | 多数 mode 不 discriminative |
| **Behavioral** (`action_diversity` 等) | step-level action 序列 | ⭐ **主导, AUROC 0.682-0.748** | secondary |
| **Verbalized self-confidence** | model 直接 output "I'm X% sure" | secondary | ⭐ **主导, AUROC 0.701-0.793** (P-text red 0.793 = 5-mode max, 超 baseline 0.766) |

**4-fold drop-in (c) 的实证依据**: 5 phantom modes 全部 `overall_usable=True` — phantom mode 直接复用 baseline 的 signal extraction infra, **不需要 retrain**. 这是 "router 部署可以 drop-in" 的硬证据.

**Site asymmetry**: cls 偏 behavioral, red 偏 verbalized — 这意味着 router 的 trigger feature 应当 site-class adaptive (Phantom paper § 7 跟 Routing paper § 6 都会用到这个 finding).

**Router 两 Tier**:
- **Tier 1 offline oracle router** — task-instruction TF-IDF + binary feature (has_ref_image / has_finish_string) + LR / random forest, 80/20 split, 数 paper 会看 oracle bound vs learned router gap
- **Tier 2 first-step trigger online cascade** — DOM 跑 step 1 → 抽 verbalized + behavioral trigger signal → 决定 escalate to P-SoM / SoM / Vision, 不会 leak test-time info (step 1 obs ≠ task feature)

**多指标 Pareto**: cost / P95 latency / regional carbon (B1 measured 45 region, B0 token-based estimator) — 3 向 drop-in 全 ready.

### 3. VWA bug fix (~37 entries)

VWA / WebArena benchmark 本身 broken 的功能我跑实验过程发现的 + 之前 paper 提过没修的 (Patched WebArena 那种). 包括 Magento FPC / Postmill PHP gc / Wikipedia ZIM 版本 / select dropdown 7 层 / Phase A 4-cluster (dispatch+page_changed+fuzzy cycle+RNG). 这个您建议做 ACL position paper / survey + community repo 持续更新.

### 4. Mechanistic interpretability (您 5/5 push 的新方向) ⭐

**简要**: 不看 model 输出, 看 **model 内部** (hidden states / activations / sparse features) 在 mirage / non-mirage 任务下的 signature 差异. 三个工具:

- **Activation patching**: 跑两次 model, A 是 mirage prompt B 是 non-mirage prompt, 把 A 的 layer L hidden state 强行 patch 到 B, 看 B 输出哪一层 patch 后会 flip → 找到 mirage 在哪层最 critical.
- **Linear probe**: 训 linear classifier 用 hidden state 预测"这是 mirage 还是 non-mirage", 看哪层 separable degree 高 → 那层 explicitly encode mirage info.
- **SAE feature steering**: Sparse Autoencoder 把 dense hidden state decompose 成几千个 sparse + interpretable features, 找出 mirage 对应那个 feature, 然后 force activate / inhibit → inference-time intervention 不需要 retrain.

您原话: "**前所未有的 inference time steering... 如果是 cross-model 的, 那这个价值就很大了, 这就是 golden feature**". 我已有 contrastive set (P-text/P-SoM/DOM × site × model 的 mirage / non-mirage 任务对), 直接套就行.

### 5. 其他 supporting innovations (paper_planning §22 列的)

- **Cross-X generalization**: cross-site (cls/red/shop), cross-model (B0 235B vs B1 4B), 计划 cross-family (Claude Opus)
- **Capability-modulated reversal**: B0 偏 text 别扭, B1 偏 image 别扭 (post-hoc N=4 provisional, 16-cell rerun 后 commit)
- **Pre-registration R1-R5 framing rule**: data-conditional hook 选择, 防 reviewer 攻击 cherry-pick
- **~100× deployment-class cost gap**: B0 API $0.04/ep vs B1 electricity $0.0004/ep, 不是 capability ratio
- **9-cell intervention taxonomy**: 3 spectrum (bug fix / synthesis / channel addition) × 3 layer (server-side / agent-pipeline / LLM-internal), 12+ verified industry instances 西方+中国
- **B0 5-call probe (reproducibility 诚实 disclosure)**: API token-non-deterministic 但 action-convergent
- **4-dimension Evidence framework**: Outcome / Macro / Micro / Efficiency 正交
- **Industry-vs-research epistemic distinction**: 工业 deploy at artifact level (单 mode 部署省钱), paper 在 research level 做 controlled cross-mode comparison — 不同 epistemic level

---

## Part 2 — paper 拆分 + Mechanistic nested 还是独立 (核心问题)

### Q1: Mechanistic 该 nested in Phantom 还是独立成 paper? ⭐⭐⭐

录音里您一会说 "单独发一篇比较好的 paper", 一会 summary 又是 "三种不同的 paper", 我没 disambiguate 出来. 想清楚地问一下:

**选项 A — Mechanistic nested in Phantom (3 papers)**:
1. **Phantom paper** (含 mechanistic 章节做 mechanism validation) — Phantom 现象 + 4-fold drop-in + cross-X + **mechanistic 证 mirage 在 model 内部确实有 signature**
2. **Routing paper**
3. **VWA bug position paper / survey**

**选项 B — Mechanistic 独立 (4 papers)**:
1. **Phantom paper** (benchmark study)
2. **Routing paper**
3. **Mechanistic paper** (SAE feature / activation patching / linear probe golden feature)
4. **VWA bug position paper / survey**

**我的想法 (但拿不准)**: Mechanistic nested 进 Phantom 比较自然 — phantom 是 phenomenon, mechanistic 是 model-internal explanation, 一篇 paper 里 phenomenon + mechanism 是经典套路. 但您之前说 "**单独发一篇**" 听起来更像 B, 而且如果 cross-model golden feature claim hold, 单 mechanistic 就是 ICLR/NeurIPS quality.

我倾向: **如果 mechanistic 只在 B1 4B 上 hold 就 nested (A); 如果 cross-model golden feature hold 就独立 (B)**. 但这个判断要等 pilot 跑出来才知道, 您觉得这个 path 合理吗?

### Q2: Phantom paper 投稿定位

录音里您说 "**类似 benchmark study 去发, 这个会稳一些**". 我理解 Phantom paper 主投 NeurIPS Datasets&Benchmarks Track / TMLR / MLSys 这类, 不冲 main track primary contribution.

对吗?

### Q3: VWA bug 是 ACL position paper 还是 survey?

您说 "要么 survey 要么 position paper, ACL 里有 position paper, 难度不高". 我倾向 **ACL position paper** 先 (~37 entries 列 + repo + 1-2 个 manifesto 论点), 比 survey lean. 您觉得?

### Q4: Environment 3-layer framework 进哪 paper?

录音里我跟您讲了 server-side / agent-pipeline / LLM-internal 三层, 您 OK 这个 framework. 这个东西横跨多 paper, 我的 mapping:

- Server-side (NLWeb / select dropdown 信息标签) → **Routing paper** (作为 routing dimension extension)
- Agent-pipeline (OmniParser / Tarsier / agent-browser industry context) → **Phantom paper** (related work + industry positioning)
- LLM-internal → **Mechanistic paper** (或 nested 进 Phantom)

OK 吗? 还是您觉得 framework 应该集中放一篇?

### Q5: Workshop submission 节奏

录音里您 push "**看一下有没有什么 workshop**", "到时候会给您发一些相关的". 我理解为: **小规模数据 (e.g. cls + red 两个 site phantom subset) 跑完后投 workshop**, full version 投 main conference. 节奏对吗?

您方便先发 1-2 个 workshop name 吗? 我排考试后 timeline.

### Q6: Pre-registration 见证 mechanism

我 commit 到 git → 您 email 回 "I have reviewed and witness" → 后续上 OSF 拿 DOI, paper §1 footnote cite. 这个流程 OK 吗?

具体 threshold 在 Part 3.

---

## Part 3 — 来不及讨论 / 需 email 回

### 1. Pre-registration 三个 threshold ⭐⭐⭐ 最 urgent

forest plot 当时 Slack 没传过去您没看到. 我会 commit + push 完发您 GitHub 链接, 您扫一眼 (preregistration.md + 3 张 forest figure), email 回 **"I have reviewed and witness these thresholds"** 即可.

**为什么必须 pre-commit**: paper §1 footnote 会 cite "**pre-registered with advisor email witness on \<date\>, git commit \<SHA\>, OSF DOI**". 数据没出来前 commit, 数据出来后改不了. Reviewer 看到这个 footnote 就不会攻击 "你 cherry-pick 阈值 to make hero pass". 这是 paper rigor 关键, 也是为什么需要您 email 见证 — git commit 时间戳一个人能改, email + OSF 双层 audit trail 改不了.

#### (a) K_h1 = 75% — Hero claim 通过率

**测什么**: P-SoM 在多少 % 的 cell 里 Holm-corrected p < 0.05 显著, 才算 hero claim 通过.

**16-cell rerun 下**: 至少 **12/16 (=75%) cell** 必须 P-SoM 显著, hero claim (paper §1 R1 strongest framing) 才 hold. 不到 → 退到 R3 framing ("P-SoM 是 hidden 4th routing arm" 弱版本) 或 R5 (paper death pivot to VWA bug audit).

**为什么 75% 而不是 50% 或 90%**:
- 50% → 8/16 cell pass 就算赢, 太弱, hero claim 几乎 trivially 成立, reviewer 一眼看穿
- 90% → 14/16 cell 必须 hold, 1-2 个 noise cell 就 fail, paper 死太容易
- **75%** → 容忍 4 个 outlier, 但要求 majority strong. 是 NeurIPS / ICLR 这种 venue reviewer 接受的 norm

**Expected outlier 来源 (在 4 个容忍范围内)**:
- B1 shop SR 极低 (~5%), statistical power 不足
- B1 cls capability-modulated reversal (B0 cls hero pattern 在 B1 反转, post-hoc N=4 已观察到)
- Cell-level cross-mode correlated noise (random seed × site × mode)

**选错风险**: 选低 → hero claim 弱; 选高 → 1 noise cell 破 paper.

#### (b) K_h3 = 67% — Structural claim 通过率

**测什么**: phantom space 不是塌成 1 个点, 是 **2-axis 结构** — P-text (text-flattening axis) 跟 P-prompt (SoM-prompt axis) 各自有 P-SoM 解不了的 unique tasks.

**计算方式 per cell**:
- |P-text ∖ P-SoM| = P-text 解出但 P-SoM 没解出的 task 数
- |P-prompt ∖ P-SoM| = P-prompt 解出但 P-SoM 没解出的 task 数
- Bootstrap (resample tasks with replacement, 1000 次) → 95% CI on each count
- **Cell pass = 两个 CI 下界都 > 0** (P-text + P-prompt 各自 unique-task count 都统计上非零, 同时 ≥ 2 tasks 防 1-task 噪声)

**16-cell 下**: 至少 **11/16 (=67%) cell pass**, structural claim 才 hold. Fail → paper hook framing 退到 R3 (P-SoM hero only, 不分 axis).

**为什么 67% 而不是 75%**:
- Structural 是 **supporting** contribution (axis decomposition), 比 hero deployment claim 弱 commit
- P-text + P-prompt 数据 noisier than P-SoM (single-axis 别扭, performance variance 大)
- 67% (=2/3) 是 "majority of cells but not strict" 的 conventional cutoff, 跟 NeurIPS pre-reg 范例一致

**选错风险**: 选高 → 容易退回 R3, paper claim 弱化; 选低 → structural claim 不 trivial.

#### (c) TOST δ = 1.0pp — Equivalence margin (drop-in property a)

**测什么**: "P-SoM cost ≈ DOM" (4-fold drop-in 第 (a) 项). 这是 **equivalence claim** ("cost 没有 substantive difference"), 不是 difference claim (常规 t-test). 所以用 TOST (Two One-Sided Tests for equivalence).

**δ 是什么**: equivalence 边界. effect 在 [−δ, +δ] 内算 equivalent (i.e. cost 差异 ≤ ±1pp 就算 ≈).

**为什么 1.0pp 而不是 0.5pp 或 3pp**:
- N=234 (cls) / 210 (red) / 466 (shop), 单 task ≈ 0.43pp / 0.48pp / 0.21pp
- Bootstrap iteration noise + cell-level correlated error 实测约 0.7-1.0pp (paper §4 disclosure 已用 archived 数据估)
- δ = 0.5pp → 比 noise floor 还小, TOST 永远测不出 equivalence (Type II error 高)
- δ = 3pp → 太松, 任何接近 cost 都被 declare equivalent, equivalence claim 太 cheap
- **δ = 1.0pp ≈ 2 tasks @ N=234**, noise floor 上方安全 margin

**选错风险**: 选小 → 测不出 (drop-in (a) 实际 hold 但 statistically can't claim); 选大 → 容易 reject equivalence, claim 太 weak.

#### Email confirm 模板

GitHub 链接我 push 完发您, 您扫 `preregistration.md` + 3 张 forest figure (`fig_meta_forest.png` / `fig_forest_drop_one.png` / `fig_phantom_structure_venn.png`), email 回:

> "I have reviewed the pre-registration thresholds (K_h1=0.75, K_h3=0.67, TOST δ=1.0pp at 16-cell rerun scope) on \<date\> and witness them as committed before data unblinding."

(您可以加任何 condition, e.g. "subject to my final review of the rerun protocol", 都 OK. 关键是 timestamp + explicit acknowledge 留 audit trail.)

### 2. Train/test split protocol (Routing paper 用)

| Option | 取舍 |
|---|---|
| **5-fold site-stratified CV** ⭐ 我 lean | data efficiency (k=5, seed=42, min test fold ≥ 40 tasks) |
| **LOSO** (训 cls 测 red, 反之) | reviewer-defensible cross-site claim 但 power 弱 |

您倾向哪个?

### 3. Mechanistic paper scope (B1-only?)

B0 是 Qwen3-VL-235B 走 proxy API, **没 model internals access, 没法 activation patch**. Mechanistic 实际上只能在 **B1 (Qwen3-VL-4B local)** 上做.

两个选项:
- (a) 接受 B1-only limit, claim 限制为 "**Qwen3-VL-4B 上 mirage feature 可分离 + 可 steer**", cross-model golden feature 退到 future work
- (b) 加 **cross-architecture validation** (Qwen2-VL local / Llama-3.2-Vision-Instruct local 也跑 mechanistic), 多 1 个 model family 撑 cross-model claim

(a) lean 但 claim power 弱, (b) 工作量翻倍但 paper 厚. 您倾向哪个?

### 4. 各 paper §1 hook direction

我现在 paper_planning §1 hook 是按 single paper 6-contribution 写的, 拆完后每篇要 reframe. 您方便给个 high-level direction 吗 (一两句话即可)?

- **Phantom paper**:
- **Routing paper**:
- **Mechanistic paper** (如果独立):
- **VWA bug position**:

### 5. OSF DOI 上传 timing

拿到您 threshold witness email 后, 我立刻 OSF 上传 preregistration.md + figures + email 截图 拿 DOI, paper §1 footnote cite. 流程 OK 吗?

---

## Timeline

- **5/6-5/8** (这份 doc 您回完): launch 16-cell paper-grade rerun (no early stop) + mechanistic pilot (B1 cls 一个 cell scout activation patching)
- **5/12-6/1**: 专心复习, 实验 GPU 上自动跑
- **6/1 后**: 数据 ready → make analysis → codex prose pass → workshop 投稿

---

谢谢学长! 麻烦您扫一眼有空回个 email, Q1-Q6 + Part 3 yes/no + lean 即可, 不用打长字.

—— [学生姓名]
2026-05-05
