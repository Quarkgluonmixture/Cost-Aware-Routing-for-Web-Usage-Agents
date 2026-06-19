# Lit Triage — p79-lit-digest cron 2026-06-12 (12 papers)

**Source:** `p79-lit-digest` cronjob (job_id 8034281c1cb9), 2026-06-12, **web-search reconstructed**(E: drive 在 cron sandbox 不可达, digest 自承覆盖面偏窄)。
**Verification:** 12 个 arXiv ID 全部经 arXiv API (`id_list` 批量) 核验 — **12/12 真实存在, 标题/作者/方法与 digest 描述吻合**(逐条对了 method claim)。尽管是 web-search 重建(按 [[reference-arxiv-api-for-verification]] 本应幻觉高发), 这批质量高。**但核验暴露 digest 两处 framing 偏差**(见 §4)。
**Read depth:** 5 篇 live-relevant 新论文由独立 sub-agent 精读全文 (#1/#8/#9/#11/#12); 4 篇 (#2/#3/#7/#10) 从 full abstract 判 disposition(均 park 到 paper-2 §5); 3 篇 (#4/#5/#6) 已在 bib。
**Bottom line:** 真正进 **paper-1 LIVE** 的只有 **2 篇新论文 (WebChallenger #8 + Visual Ignorance #1)** + 1 个已入 bib 但未 position 的 gap (AVR)。digest 主推的 Score-3 多锚到 **§5 mechanism — 已 shelved 到 paper-2** (advisor 2026-05-14), 故对 paper-1 价值被系统性高估。

---

## 0. Verification + scope-map 总表

| # | arXiv | 题 | API 核验 | 已在 bib? | 主映射 | LIVE paper-1? |
|---|---|---|:--:|:--:|---|:--:|
| 1 | 2606.06890 | Diagnosing Visual Ignorance in VLMs | ✅ | — | §1/§2 phenomenon (behavioral) + §5 (mech) | ✅ **add** |
| 2 | 2605.25310 | Tool-Call Dependency Linearly Decodable | ✅ | — | paper-2 §5 (tool-calling circuit) | ✗ park |
| 3 | 2602.04935 | ASA / Lazy Agent | ✅ | — | §1 "knows-but-doesn't" (light) + paper-2 §5 | ~light |
| 4 | 2603.12823 | Adaptive VLM Routing for CUA (AVR) | ✅ | ✅ `liu2026adaptive` | §6.8 router 邻居 | ⚠ **已 bib 未 position** |
| 5 | 2603.21687 | MIRAGE | ✅ | ✅ `asadi2026mirage…` | §1/§2 已引 | — already |
| 6 | 2603.28387 | Scaffold Effect | ✅ | ✅ `vu2026scaffold` | §1/§2 已引 | — already |
| 7 | 2604.27818 | MASCing (MoE steering masks) | ✅ | — | paper-2 §5 (MoE, 伴 SteerMoE) | ✗ park |
| 8 | 2606.10423 | WebChallenger / PageMem | ✅ | — | §2/§6 representation-axis 邻居 | ✅ **add (best find)** |
| 9 | 2605.09252 | When2Tool / Probe&Prefill | ✅ | — | §1/§6 precedent (light) + paper-2 §5 | ~light |
| 10 | 2604.01202 | Therefore I Am. I Think | ✅ | — | paper-2 §5 (CoT post-hoc) | ✗ park |
| 11 | 2604.09377 | Task-Aware LLM Routing (TRouter) | ✅ | — | §2 "task-aware/cold-start" one-liner | ~optional |
| 12 | 2606.02907 | Linear Probes Detect Task Format | ✅ | — | **paper-2 §5 reviewer-defense (must-cite)** | footnote only |

---

## 1. LIVE paper-1 adds (真正有用)

### 1.1 WebChallenger (2606.10423) — best find, §2/§6 representation-axis 邻居
**Method.** PageMem = **确定性结构化 DOM 表示** (非 routing): 4 级 WebsiteMem→PageMem→PageSection(DOM 子树 + LLM 一句话 summary)→Element; 上跑 offline 持久记忆 + multi-stage observation pipeline(选 section→抽细节→合成 task-focused summary = selective attention) + compound action。numbered-list index action(自称对小开源模型比 tool-calling 更可靠)。
**Numbers.** backbone GLM-4-32B-0414 (no finetune); **VWA vision model = Qwen3-VL-4B-Instruct(与 P79 B1 直接重合)**。WebArena **56.3%** 开源 SOTA(+7.9 vs Mobile-Agent-v3.5); VWA 48.7%; harness-isolation: GLM-4-32B 在 minimal GenericAgent **19.4% → WebChallenger 58.8%(+39.4pt)** = "architecture not scale"。
**映射.** 与 P79 **同 representation 轴** — 但 PageMem 是 **fixed** 表示(每页一套确定性结构化 DOM), P79 是 **per-task select** DOM/SoM/Vision/phantom。其 "section selection" = page 内 attention, 非 cross-mode routing。
**Caveats.** 非 P79 baseline(GLM-4-32B controller ≠ Qwen/Gemma; full VWA-910 + exploration phase, SR 不可比); PageMem 捆了 memory+workflows, SR 不能归因到表示单独。
**Verdict. CITE — §2/§6 强邻居(contrast pole, 非 baseline)。** "fixed structured-DOM 表示" 作 P79 "route *among* representations per task" 的 adjacent-but-different 对照; **明确点出 Qwen3-VL-4B-on-VWA 的重合**。借用: 其 Table 4 harness-isolation 设计(同模型 minimal-vs-rich harness 隔离 architecture/scale) = P79 "representation contribution net of model" 的干净模板。

### 1.2 Diagnosing Visual Ignorance (2606.06890) — fresher MIRAGE companion, §1/§2
**Method(dual-lens).** (1) mech: counterfactual layer replacement(GRPO+LoRA debias Qwen2.5-VL-3B decoder, 11.6%→65.2%, 把末 N 层换回 baseline) + layer-wise 3-layer-MLP probe; (2) behavioral: progressive Gaussian blur(8 kernel 1×1→61×61), 追 *连续相同* 答案下界化 language-prior 依赖, **12 VQA bench × 3 VLM**。
**Numbers.** (a) **20–40% 样本在完全视觉遮挡下答案不变**(RePOPE/MMMU ~40%), 子集 acc 近平 = "benchmarks reward visual ignorance"; (b) 两段抑制: 中层取不到细粒度视觉信息, **末 ¼–⅓ 层主动回注 text prior**(language-prior prob >0.6), 中间层仍 ground-truth-dominant; (c) prompt 显式要求 grounding 也关不掉 bias。
**关键 — 超出 MIRAGE+Scaffold 吗?** 部分。**直接引 MIRAGE 并自证为 strict superset method**(批 MIRAGE 的 binary image-present/absent "B-Clean" 被 Yes/No 空间随机猜混淆, 自己的 multi-step consecutive-consistency 是更干净下界)。故 **LIVE §1/§2 = 比 MIRAGE 更新更强的同点 behavioral 引用**(带 20–40% 遮挡不变这个干净数字), 但相对 MIRAGE 是 incremental 非新现象。
**Split.** LIVE-behavioral = blur/invariance + "benchmark rewards visual ignorance"(MIRAGE 的 companion/可替); SHELVED-mech = layer-replacement + probe → paper-2 §5(与 shelved patching/probe 计划同 toolkit, 跑在 VQA)。
**Caveats.** VQA 单图域 ≠ web-agent obs 模式; "routing failure" 是其 metaphor(decoder 内 modality competition)非字面 routing; blur 毁像素 ≠ phantom(phantom 留 [SOM_MARKS] 文本、丢图)→ motivation-only 非 method import。
**Verdict. CITE — LIVE §1/§2 NOW, 与 MIRAGE+Scaffold 并列**(sharpen 非 replace); mech 半边 park paper-2 §5。

### 1.3 ⚠ AVR (2603.12823) — 已在 bib `liu2026adaptive`, 但 §6.8 未 position
`grep` 实测: AVR 在 bib 但 **section6_router.md §6.8 efficiency-routing 表里没有它**。AVR = "Adaptive VLM Routing for Computer Use Agents", 按 action difficulty 在 VLM 池间 route(自称首个 formalize CUA 中 VLM routing), 是 §6.8 最该出现的 "which-VLM in CUA" 邻居。**Action: §6.8 表 + 一句差异化补 AVR**(routes which-VLM-by-difficulty = compute/model 轴; P79 route representation 轴)。

---

## 2. Reviewer-defense / light-cite

### 2.1 Linear Probes Detect Task Format (2606.02907) — paper-2 §5 must-cite (adopt-as-caveat)
**ACL 2026 Trustworthy NLP workshop, v2.** Qwen3-14B 40 层 probe 分 deductive/inductive/abductive: **layer 32 → 100% CV acc**; 但 **source-probe 也 100%**(label 与 dataset source 共变 = perfect format confound); **residualize format vector[source one-hot, n_options, response_length] 后掉到 ≈33.5% ≈ chance**; causal steering targeted-vs-random 40.0% vs 31.7%, **p=0.286 d<0.5(无方向特异因果)**。refute: "高 probe acc 分 reasoning type ⇒ mode-specific 表示" 仅在排除 confound 后成立 — 任何 multi-source 设计自带 format confound = "设计的性质, 非模型的性质"。
**Threat.** **(a) vs shelved §5 Method 4.2(mode-probe AUROC=1.000 across 37 层): 直接命中、naive 引用会被 reviewer 武器化** — P79 六 obs 模式恰好就差在 surface format([SOM_MARKS]/AXTree/raw image, 比 LogiQA-vs-ARC 更赤裸), 按其逻辑 1.000 跨全层 = 教科书 artifact("probe 在读哪个 serializer")。**Defuse 已基本在手:** §5 **已把 probe 配 causal patching**(正是本文抱怨缺的 random-control 因果检验)→ 引本文来 *motivate* patching, probe 当 screen, patching 当 load-bearing; 另可跑 residual control 报 drop。**Reframe: "只是 format" 对 routing story 反而 benign/helpful** — 从 hidden state 读出 format 正是 obs-mode router 要的; 只对 P79 *不需要* 主张的 "mode-specific reasoning circuit" 致命。**(b) vs LIVE §6 router(task-TEXT 特征, 不碰 hidden state): 基本零威胁** — 全文 scope 在 hidden-state probing; §6 router 是 explicit-feature 分类器, 本就 *是* format/text 模型(设计意图非 confound)。
**Caveats.** 单模型(仅 Qwen3-14B, §7 自认无 Llama/GPT 复制); residual 回归 self-admit 过保守(可能抹真信号); reasoning-mode 三分 ≠ obs-mode 多模态(by-analogy 非 by-replication); causal n≤15 欠功率(p=0.286 = 5/20)。
**Verdict. paper-2 §5 must-cite(adopt-as-caveat 非 rebut)**: "we treat probe separability as necessary-but-insufficient screen, establish causal claim via patching; additionally report format-residual controls"。paper-1 仅需 ≤1 行 footnote(scope-police: 防 reviewer 把 shelved probe 和 live router 混为一谈)。引时用 **workshop venue + v2**, 有公开 code。

### 2.2 When2Tool / Probe&Prefill (2605.09252) — §1/§6 precedent(light), 非 baseline
When2Tool 18 env; tool-necessity **AUROC 0.89–0.96 跨 6 模型**(Qwen3-1.7/4/14/32B + Llama), **probe 胜模型自陈 reasoning**(Llama Reason-then-Act 把 tool-calling 砸了 79.5%→31.2%, probe 仍 >0.9 = "signal exists even when generation fails"); Probe&Prefill **−48% calls / −1.7% acc**(真实 Search-o1 上 −20~56% API calls 无掉点)。**轴/特征双错位**: "whether to call a tool" = halt/skip 轴(非 representation 轴); 读 hidden state(P79 live router 用 task-text)→ **非 representation-routing 竞品、非可跑 baseline**。**真正有用**: "routing signal 内部/pre-generation 已在但模型不 act" = phantom 叙事的直系("signal present without the image"); Probe&Prefill = literal hidden-state router = **paper-2 §5 probe-based router 的最近已发表模板**。
**Verdict.** LIVE §1/§6 related-work/precedent(light); hidden-state-router 部分 park paper-2 §5。

### 2.3 ASA / Lazy Agent (2602.04935) — §1 "knows-but-doesn't"(light) + paper-2 §5
"Lazy Agent": tool necessity 从 mid-layer activation 近完美可解码(AUROC~1.0) 但模型不行动; ASA ~20KB steering assets inference-time 插入, **Strict Tool-Use F1 0.18→0.50**。**已被 `wu2026toolcalling` 的 related work 间接引**("Wang et al. 2026 improve binary decisions F1 0.18→0.50")。representation-behavior gap = phantom "知道但不做" 的直系表亲。
**Verdict.** §1 motivation 可借 "knows-but-doesn't" 一句(light); steering method park paper-2 §5。

### 2.4 Task-Aware LLM Routing / TRouter (2604.09377) — §2 optional one-liner
compute/which-model 轴(FrugalGPT/RouteLLM/RouterDC sibling); 域 = 纯文本 QA(Alpaca/GSM8K), 非 web agent。latent task-type 变量 z + ELBO; cold-start 合成数据 $24.37 bootstrap 17,880 QA。**与 P79 完全不同轴/域, 非 baseline。** RouteLLM/FrugalGPT 已占该 taxonomy cell → **仅当要 "task-type-conditioned routing" 先例(parallels learned router)或 "cold-start 训练数据" 防御时引一句, 否则 skip**。

---

## 3. Park 到 paper-2 §5(tool-calling linear circuit / MoE 簇)

这 3 篇 + #3/#9 的 mech 半边构成一个 coherent cluster, 与已引 `wu2026toolcalling` 同脉络, paper-2 §5 复活时成 citation 簇:
- **#2 Tool-Call Dependency (2605.25310)** — Qwen3-32B edge probe 从 residual stream 解码 tool-call 依赖图 + activation patching 证跨层传播; 退化成顺序调用时信号消失(编码抽象拓扑)。paper-2 §5 tool-calling circuit。
- **#10 Therefore I Am (2604.01202)** — linear probe 从 pre-generation activation 解码 tool-calling 决策, 早于首个 reasoning token → CoT 可能事后合理化。paper-2 §5(与 #9 同 "decision-before-reasoning" 簇)。
- **#7 MASCing (2604.27818)** — MoE 行为经 activation steering mask 配置(LSTM 建 routing-logit→行为映射 + sparse mask), jailbreak defense 52.5%→83.9%。paper-2 §5 MoE 维(伴已引 `steermoe`); B0=Qwen3-VL-235B-A22B 是 MoE → 架构相关但 paper-1 不 self-probe(proxy 藏 router logit)。

---

## 4. Digest meta-caveats(cron 质量反馈)

1. **§5-anchor framing bias**: digest 把多数 Score-3 锚到 "§5 mechanism anchor", 但 §5 自 2026-05-14 advisor 决定 shelved 到 paper-2(digest 不知 scope flip)→ 对 paper-1 价值系统性高估。**评分应按 live scope 重定位**(本 note §0–§3 即重排)。
2. **缺 bib-dedup**: MIRAGE/Scaffold/AVR(#4/#5/#6) 已在 bib 仍被 surface(digest 不 cross-check bib)。5-day sweep 应加 bib-dedup gate。
3. 轻微: 同一篇 #9 在两次 push 评分不一致(Score-1 vs Score-3); "TRouter" 昵称 = 论文内部 model 名(非 digest 杜撰, OK)。
4. **正面**: 12/12 ID 真实 + 方法准确, 远好于 web-search 重建的预期 baseline。**但验证仍是必须** — 它本可轻易翻车, 且正是验证发现了 §0 的 bib 重复与 §4.1 的 framing 偏差。

---

## 5. Disposition 行动表

**执行状态 (2026-06-15, user Q=全做)**: ✅ WebChallenger + Visual Ignorance 入 `paper.bib` (详细 note, 97→99); ✅ §6.8 +AVR 行 +WebChallenger 行 + 差异化 prose (fixed-representation contrast pole); ✅ §2.4 language-prior thread (MIRAGE+Scaffold+Visual Ignorance, 顺带激活闲置的 Scaffold cite + 闭合 "丢图不塌" motivation 缺口)。**draft prose 未 commit, 待 /stress** (§6.8 draft-status 已标溯源)。其余 (When2Tool/ASA/TaskRouting light-cite + #2/#7/#10/#12 paper-2 §5 簇) 暂留本表未入 bib。chronicle: 笔记 §336。

| 动作 | 对象 | 优先级 | 落点 |
|---|---|:--:|---|
| 加 bib + §2/§6 contrast pole | WebChallenger `hwang2026webchallenger` | **高** | bib + §6.8 表新增一行 + §2.4 |
| 加 bib + §1/§2 并列 MIRAGE | Visual Ignorance `zhou2026visualignorance` | **高** | bib + §1/§2 behavioral anchor |
| §6.8 补 position(已在 bib) | AVR `liu2026adaptive` | **中** | §6.8 表 + 一句差异化 |
| 加 bib + paper-2 §5 防御 note | challenge `sahoo2026linearprobes` | 中(paper-2) | bib + §5 Identification Assumptions(复活时) |
| 加 bib(可选 light) | When2Tool / ASA / TaskRouting | 低 | §1/§6 一句; 或留 park |
| paper-2 §5 cluster 待引 | #2 / #7 / #10 | paper-2 | 复活时与 `wu2026toolcalling` 成簇 |

**均未入 `paper.bib`(除 #4/#5/#6)。** 建议 bib key: `hwang2026webchallenger` / `zhou2026visualignorance` / `sahoo2026linearprobes` / `sun2026when2tool` / `wang2026asa` / `sun2026toolcalldependency` / `esakkiraja2026therefore` / `telintelo2026mascing` / `liu2026taskaware`。
