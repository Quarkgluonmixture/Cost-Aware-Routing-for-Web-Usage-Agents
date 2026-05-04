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
> **Last updated**: 2026-05-04 late evening (§21 ROUND-3 fact-check integration — Round-2 hallucinations corrected: HMT author Tan/Gao/Wu BIT (not Huang); NLAH dropped (was lying citation); WebAIM 2026 actual 59.1 vs 42 (not 57 vs 27); Operator 41.7% misattribution dropped (was MAI-UI success rate); Doubao bans dropped (no Chinese press verification); K3 Mariner / ActionEngine dropped (hallucinated). New Round-3 verified specifics: Magma uses Qwen3-VL backbone (same family as our paper); ScribeAgent fine-tunes Qwen 7B 6B-token corpus to WebArena 51.3%; NLWeb deployed at Tripadvisor + Shopify with `/ask`+`/mcp` endpoint spec; OmniParser-v2 SPS literal format `<box_start>...<box_end>`; AppAgent-v2 view_state_id JSON schema; Mind2Web 2 WebJudge 3-category taxonomy; cost anchors 241K vs 47-140K tokens; MCP 100% tool spoofing vulnerability)
>
> **2026-05-04 evening**: §21 EXPANDED — DR audit findings integrated: industry precedent stack mapped to 9-cell matrix [NLWeb / OmniParser-v2 / Magma / AppAgent-v2 / ScribeAgent / UI-TARS / HMT]; (ii)×L3 internal 4-tier sub-gradient identified [pretrain / RAG / inference-time / pure-visual], paper-1 occupies inference-time niche; §21.5 candidate paper §1 hook prose with substitution-gradient framing; §21.6 WebAIM 2026 + WebSuite + CAPTCHA counter-evidence stack
>
> **2026-05-04 morning**: §21 NEW — Environment-Agent Intervention Taxonomy 3×3 matrix; 笔记 §1-§108 audit, ~40 entries mapped to 9 cells; paper-1 main hook = (ii)×L3 phantom routing space; identified-but-unfixed (iii)×L2 channel-addition gaps as paper §5 ceiling argument material; paper-level methodology asymmetry inventory §47/§95/§96/§101
>
> **Previous**: 2026-05-03 (pre-registration framework reframe: Hero+Structural+Framing-rule replacing 3-arm a-priori commit; preregistration.md draft + EVIDENCE_LAYER_AUDIT.md §2 anchor; T0a-d evidence-layer infra done)
>
> **Previous**: 2026-05-01 (hook reframe to phantom space 3 arms; §2 cube boundary definition; axis 1/2 LLM mechanism refine)

---

## §1 Paper Hook + Tagline

> **2026-05-03 reframe note**: Paper hook framing is now **data-conditional** per pre-registered framing decision rule (R1-R5; see `docs/checkpoints/preregistration.md` §2). The "core finding" below corresponds to **rule R1 (STRONGEST)** — applies if H1+H2+H3(i)+(ii) all hold post-rerun. If H3 fails, hook falls back to "Phantom-SoM is hidden 4th routing arm" (R3, MODERATE). The Hero (P-SoM deployment) + Structural ablation (P-text/P-prompt non-overlap) + Framing-rule structure replaces the older "3-arm a-priori commit" framing — see `docs/reference/EVIDENCE_LAYER_AUDIT.md` §2 for epistemic rationale.

**Core finding (under R1, contingent on H3 empirical validation)**: We discover a **hidden phantom routing space** for web agents — defined by the boundary "**skip annotated image**" — containing a **2-axis empirical structure** (axis 1 = text payload via P-text; axis 2 = SoM-style prompt via P-prompt) with **P-SoM (cube center, axis 1 + axis 2 compound) as the deployment hero**. P-SoM satisfies a **4-fold drop-in property**; P-text and P-prompt serve as **structural ablation arms** validating axis decomposition:

| Drop-in property | Evidence |
|---|---|
| (a) **Cost ≈ DOM** | `[SOM_MARKS]` 是 AXTree regex filter, 不需 bbox/image (验 `som.py::_extract_text_marks` line 24); text token ±7% (3437 vs 3661 reddit / 3008 vs 2948 cls) |
| (b) **Latency ~50% lower** | cls SoM p95 74s vs Phantom-SoM 18.2s = **4× faster** (no image encoding stage) |
| (c) **Signal AUROC ≥ baseline** | 5-mode 全 `overall_usable=True`; red P-text verbalized 0.793 是 5-mode 最高 (超 baseline 0.766) |
| (d) **Drop-one oracle 1.7-3.8pp per phantom arm** | B0 red: P-text +3.81pp / P-SoM +3.33pp / P-prompt +2.86pp (all sig CI excludes 0); cls: P-text +3.42pp / P-SoM +2.56pp; B1 cls P-SoM +1.71pp. **Phantom space 3 arms 都贡献 unique tasks**, 6-mode oracle vs 3-mode lift +7.14pp [3.81, 10.48] (B0 reddit) |

**Paper one-liner (for advisor pitch)**:
> "We discover a hidden **phantom routing space** in SoM-style web agents — defined by the boundary 'skip annotated image' — containing 3 routing arms (P-text / P-prompt / P-SoM) sharing a **4-fold drop-in property**: cost ≈ DOM (no image embedding tax), ~50% lower latency (no image inference stage), signal AUROC ≥ baseline (routing infra drop-in), drop-one oracle 1.7-3.8pp per arm (all sig). Two LLM mechanisms create this space: (i) text-payload flattening (AXTree → `[SOM_MARKS]`) reframes the agent's task ontology from web-browsing to indexed selection (axis 1); (ii) SoM-style visual prompting without image still activates the agent's visual-mark referencing parsing and recovers a substantial fraction of visual structure information textually (axis 2; **Mirage Effect** Asadi et al. 2026 (arXiv:2603.21687) — VLM 无图准确率 ~70-80% of with-image; **Scaffold Effect** Vu & Balloccu 2026 — prompt mentioning modality alone explains 70-80% performance shift independent of image presence). P-SoM (cube center, axis 1 + axis 2 compound) is the space's representative arm; SoM (image-on cube endpoint) and Vision (image-only, outside cube) anchor the comparison. The space is site-modulated (cls visual-rich requires image; red text-dominated thrives in phantom space) and routing-deployable (B0 red 6-mode oracle lift +7.14pp over 3-mode baseline)."

### Cascade design (token-monotonic, paper Section 6)

```
DOM       (~3K AXTree text + DOM prompt + 无图)        ← cheapest text
  ↓ axis 1: AXTree → [SOM_MARKS] flat (text 结构 swap, ~3K both, prompt 不变)
P-text     ([SOM_MARKS] flat + DOM prompt + 无图)
  ↓ axis 2: dom_prompt → som_prompt (system prompt swap, 0 data token)
P-SoM     ([SOM_MARKS] flat + SoM prompt + 无图)
  ↓ axis 3: + image (~1.5K image embedding tokens)
SoM       ([SOM_MARKS] flat + SoM prompt + 有图)        ← highest text+image
```

**Order rationale**: Step 1+2 都 **0 增量 token**，第 3 步才付 image embedding tax — token-monotonic cascade，trigger router 不需要"先加再删"。Vision 是另条独立路径（image-only, no text），适合纯 visual task。

---

## §2 Theory Framework — Mechanism Activation + Phantom Space Boundary (大重写 2026-05-01)

> **Cross-reference (added 2026-05-04)**: §2 是 **mechanism explanation layer** (Zoom 1-4 解释假说); 跟 §21 **intervention taxonomy** (3-spectrum × 3-layer) 是 orthogonal 维度。**§2 关心 phantom routing space 内部 mechanism** (M1/M2 axis activation, etc.); **§21 关心 substitution gradient 上 paper-1 跟 industry precedents 的 substrate 站位** (NLWeb / OmniParser-v2 / Magma / ScribeAgent / AppAgent-v2 / UI-TARS vs phantom routing). 两个 view 互补不替代。
>
> **Conceptual structure 严格 2 层分离** (笔记 §108 evidence/explanation separation):
> - **Explanation layer** (因果假说, §2 主住所): Zoom 1 architectural / Zoom 2 axis behavioral / Zoom 3 named phenomena / Zoom 4 model-internal
> - **Evidence layer** (观测数据, §3 主住所): Outcome / Macro / Micro / Efficiency × cross-task / mode / site / model
> - 两层不混 — paper writing 时 reviewer 最忌 evidence-as-explanation
>
> **Retract list (历史 framing 已 retract, 不要再用)**:
> - ❌ "5-mode 沿 cube 对角路径" + "4 mismatched 排除 (mismatched parsing tax)" 论证 (§108.2)
> - ❌ 8-corner 2x2x2 cube factorial design 作 paper §2 axis (§108.3)
> - ❌ 6-corner asymmetric grid (a/b × c/¬c × 1/2) (§108.3)
> - ❌ (a)(c) prompt decomposition 作 axis (改用 mechanism activation, §108.3)
> - ❌ "Three-layer mechanism argument" (Layer 1/2/3) 命名 — 改用 evidence/explanation 双层 + Zoom 1-4 (§108.6)
> - ❌ "Approach 1 vs Approach 2" dichotomy — Approach 2 = Zoom 1, Approach 1 不是 single thing (§108.6)
> - ❌ "3rd mechanism (coherence)" 候选升 §2 — 应在 §3 Micro dim 内分析 (§108.4)

### Zoom 1 (architectural): Phantom space boundary + 2-axis activation by design

**Phantom routing space** = subset of 3-axis modal cube characterized by **"skip annotated image"** boundary (axis 3 = no image)。Cube 有 8 corners (3 axes: text payload × system prompt × image)；paper 测 **4 phantom corners (cube image-off 半) + 1 image-on cube endpoint + 1 image-only mode (Vision, cube 之外)** = **5 cube modes + 1 image-only mode = 6 paper modes**:

**Phantom routing space** = subset of 2×2×2 modal cube characterized by **"skip annotated image"** boundary。Cube 有 8 corners (3 axes: text payload × system prompt × image)；paper 测 **4 phantom corners (cube image-off 半) + 1 image-on cube endpoint + 1 image-only mode (Vision, cube 之外)** = **5 cube modes + 1 image-only mode = 6 paper modes**:

| # | text | prompt | image | mode | 在 phantom space? |
|---|---|---|---|---|---|
| 1 | AXTree | DOM-prompt | No | **DOM** | ✅ phantom corner (origin baseline) |
| 2 | AXTree | DOM-prompt | Yes | "DOM+image" | ❌ violates boundary (image embedding tax) |
| 3 | AXTree | SoM-prompt | No | **P-prompt** | ✅ phantom corner (axis 2 alone @ AXTree 锚点) |
| 4 | AXTree | SoM-prompt | Yes | (mismatched + image) | ❌ violates boundary |
| 5 | [SOM_MARKS] | DOM-prompt | No | **P-text** | ✅ phantom corner (axis 1 alone @ DOM-prompt 锚点) |
| 6 | [SOM_MARKS] | DOM-prompt | Yes | "P-text+image" | ❌ violates boundary |
| 7 | [SOM_MARKS] | SoM-prompt | No | **P-SoM** | ✅ phantom corner (cube center, axis 1+2 compound) |
| 8 | [SOM_MARKS] | SoM-prompt | Yes | **SoM** | image-on cube endpoint (paper baseline, NOT phantom) |
| — | none | — | Yes | **Vision** | image-only mode (cube 之外, axis 1 = "no text") |

**Boundary 定义性属性 (axis 3 = "no annotated image")**:
- 4 phantom corners 都 share 4-fold drop-in by construction:
  - (a) cost ≈ DOM — no image embedding tax (`[SOM_MARKS]` 是 AXTree regex filter，~3K text both)
  - (b) latency ~50% lower — no image inference stage
  - (c) signal AUROC ≥ baseline — emergent (5/5 phantom `overall_usable=True`，red P-text 0.793 = 5-mode max)
  - (d) drop-one oracle 1.7-3.8pp positive per arm — emergent (B0 red: P-text +3.81 / P-SoM +3.33 / P-prompt +2.86，all sig CI excludes 0)
- (a)(b) 由 boundary derive (definitional)；(c)(d) 是经验验证的 emergent property —— **definition-then-validation 双层结构**

**Why exclude #2/#4/#6 (3 image-on phantom corners)**: 一旦加 annotated image 回去，cost / latency / carbon 都跟 SoM 拉齐 → 失去 4-fold drop-in property → 不再属于 phantom space。这 3 个 corners 在 routing 维度上是 SoM 的 variants (image cost dominate)，不提供 phantom-class deployment value。即 **boundary 是 "no annotated image"，不是 "matched parsing"**。

**Why P-prompt (#3) 不是 mismatched-redundant** (针对旧 framing 的修正):
- 旧 framing 担心 SoM-prompt + AXTree text "mismatched parsing" 必死 → 实证 falsified
- P-prompt 有真 LLM 机制 (axis 2 effect alone)：**visual prompting without image** —— SoM-style prompt 即使无图仍 activate agent 的 visual-mark referencing parsing，agent 在 AXTree 上自动 fallback 到 element_id 引用，仍 recover 部分 visual 结构信息。**Lit anchor stack** (3 互补 mechanism): (a) **Mirage Effect** Asadi et al. 2026 (arXiv:2603.21687, Stanford) — VLM 无图准确率达有图的 ~70-80% (实验笔记 §18); (b) **Scaffold Effect** Vu & Balloccu 2026 — prompt 仅提及 modality 可用就解释 70-80% performance shift independent of image presence (实验笔记 §25 + phantom_som.md §3.5); (c) **Cross-modal flow** Kaduri et al. — middle-layer cross-modal flows store image info in query tokens enabling image-consistent generation without direct image-token attention (phantom_som.md §2.1)
- 实测 anchor: B0 red P-prompt 4-mode drop-one **+2.86pp [0.95, 5.24] sig** 验证 axis 2 alone 在 AXTree 锚点下有 routing value
- 6-mode oracle vs 5-mode +1.90pp [0.48, 3.81] sig 验证 P-prompt 贡献 incremental unique tasks (6 tasks added, 1 unique to P-prompt)

**Phantom space 内部结构 (4 corners 形成 2×2 ablation diamond)**:
- DOM ↔ P-text: axis 1 alone (DOM-prompt 锚点)
- P-prompt ↔ P-SoM: axis 1 alone (SoM-prompt 锚点) — **双锚点验证 axis 1 effect prompt-anchor-invariant**
- DOM ↔ P-prompt: axis 2 alone (AXTree 锚点)
- P-text ↔ P-SoM: axis 2 alone ([SOM_MARKS] 锚点) — **双锚点验证 axis 2 effect text-anchor-invariant**
- DOM ↔ P-SoM: axis 1 + axis 2 compound (cube diagonal, paper-headline P-SoM 在此)
- P-SoM = cube center, axis 1 + axis 2 compound, paper-headline representative arm

**Image-on extension (cube #8) + image-only (Vision)**: SoM 是 phantom space 之外的 image-on baseline，paper 用作 axis 3 primary endpoint (与 phantom space 任一 corner 对比 isolate axis 3 effect)。Vision 是 cube 之外的 image-only mode (axis 1 = "no text")，paper 用作 image-only routing arm baseline。

**Cascade interpretation (§1 cascade 与 phantom space 的关系)**: §1 cascade `DOM → P-text → P-SoM → SoM` 是 phantom space 内一条 token-monotonic path (axis 1 → axis 2) + 出 phantom space 到 SoM 的 axis 3 跃迁。P-prompt 是 phantom space 第 4 corner，闭合 2×2 ablation diamond，验证 axis 1/2 effect 在双锚点下 invariant。

Paper Section 3 footnote 用此 defense reviewer "why 5+1 not 8" + "why is P-prompt not mismatched-redundant"。

### Zoom 1 (architectural completeness): Approach 2 deductive argument

**Paper §2 framework completeness 不依赖 finite data verification, 而是 architectural deductive argument**:

```
PREMISE 1: Phantom space comparison 锁 image=✗ (axis 3 fixed)
PREMISE 2: Agent's input 只剩 (prompt 文本) + (obs 文本) 两个 component
PREMISE 3: 4 phantom corners 仅 vary 这两个 input component:
            corner ∈ {(b,1)=DOM, (b,2)=P-text, (a-with-c,1)=P-prompt, (a-with-c,2)=P-SoM}
PREMISE 4: LLM 是 deterministic forward function on input tokens
            (T=0 + greedy decoding 假设, Phase A 后真; 但 B0 proxy 仅 decision-level
             convergent, 见 §107.1)
CONCLUSION: 任何 differential 机制 必由 input 差异 trigger
            → 必 attributable to (prompt change) 或 (text change) 或两者
            → M1 (prompt-axis activation) 和 M2 (text-axis activation) 是 exhaustive
            → phantom space 内**没有 hidden 3rd axis** (by construction)
```

**关键性质**: 这是 **deductive argument**, 不是 inductive evidence。即使以后跑 100 个 phantom corners, 也不能反驳 — 因为只要 phantom corners 仍只 vary 这两个 input dimension, M1+M2 仍是 exhaustive。

**Caveat**: Architectural argument 给 axis-level 完备性, 不给 axis 内部 sub-mechanism 完备性 (M1 内部是 Mirage Effect / Scaffold Effect / Cross-modal flow 哪个 dominate 仍需 lit + empirical 区分 — 这归属 Zoom 2/3/4)。

### Zoom 2 (behavioral): M1/M2 mechanism activation 2x2 framework

**Insight (§108.4)**: prompt 文本 coupling ≠ mechanism activation coupling。LLM internal state 层有两个 orthogonal axes:

```
Axis M1 (Image-mirage activation):
  触发条件: prompt 期待 image (SoM-style guidance 自带 image expectation)
  LLM 内部状态: vision-grounded attention pattern 启动, visual tokens 缺席,
              language prior 填补缺席视觉 (Mirage / Scaffold / Cross-modal flow)
  
Axis M2 (Flat-list activation):  
  触发条件: obs text 是 flat indexed list ([SOM_MARKS])
  LLM 内部状态: hierarchical tree-traversal exploration policy 切换到 
               sequential list-scanning policy
```

**4 phantom corners = 2x2 activation pattern**:

| Mode | Axis M1 | Axis M2 | LLM internal state |
|---|:---:|:---:|---|
| DOM | ❌ | ❌ | origin baseline |
| P-text | ❌ | ✅ | M2 alone (Format-swap subspace) |
| P-prompt | ✅ | ❌ | M1 alone (Mirage subspace) |
| P-SoM | ✅ | ✅ | M1 ⊕ M2 compound (cube center, paper hero) |

**P-SoM 是 cube center 的 mechanistic 理由**: 唯一同时 activate M1 + M2 + 它们 nonlinear interaction 的 corner。Compound state ≠ 简单叠加 (transformer attention nonlinear combination), P-SoM unique tasks (3 tasks B0 reddit, 既不在 P-text 也不在 P-prompt) 是 emergent capability。这给 P-SoM 是 paper hero 一个 mechanistic 理由 (不只是几何 cube center)。

### Zoom 2.5 (reverse explanation, NEW 2026-05-01 evening): 别扭 framework + Capability-modulated effect

> **⏸️ Provisional**: 现有数据 N=4 cells (B0 cls/red 含 phantom + B1 cls 5-mode + B1 red 3-mode), 全部 Phase A bug fix 之前 (commit `3c15cd7` 之前). 14-cell rerun 后 statistical commit, 现 framework 标 "provisional pending 14-cell rerun + cross-VLM-family validation"。详 笔记 §108.16。

**Insight**: M1/M2 framework 是 **forward causal upstream** ("what input change happens"). 别扭 framework 是 **reverse causal downstream** ("what gap between expectation and reality"). 两者描述同一现象的不同 layer, 但 别扭 在 phantom space 内提供更 mechanism-aligned 解释。

**别扭 2×2 grid (跟 forward 2×2 isomorphic 但 axis 不同)**:

```
                    No image 别扭         Image 别扭
                    ────────────────      ────────────────
No text 别扭        DOM (origin, 0)       P-SoM (axis B only, 1)
                    SoM (image-on)        
                    
Text 别扭           P-text (axis A only,1) P-prompt (compound A+B, 2)
```

| Mode | Text 别扭 | Image 别扭 | # mismatch |
|---|:---:|:---:|:---:|
| DOM | ❌ | ❌ | 0 (origin baseline) |
| P-text | ✅ (DOM-prompt 期 AXTree, 给 [SOM_MARKS]) | ❌ | 1 |
| P-prompt | ✅ (SoM-prompt 期 [SOM_MARKS], 给 AXTree) | ✅ (SoM 期 image, 无) | 2 (compound) |
| P-SoM | ❌ ([SOM_MARKS] match SoM-prompt) | ✅ (SoM 期 image, 无) | 1 |
| SoM (image-on) | ❌ | ❌ | 0 |
| Vision | n/a | n/a | n/a |

**Per-axis mechanism (跟 Zoom 3 lit anchor 直接对接)**:
- **Text-轴 别扭** (axis A): prompt expects format X, obs gives format Y → format-translation fallback → **Sclar 2024 prompt-format sensitivity / Mishra 2022** lit anchor
- **Image-轴 别扭** (axis B): prompt expects image, none provided → language-prior visual completion → **Mirage Effect Asadi 2026 / Scaffold Effect Vu Balloccu 2026 / Cross-modal flow Kaduri** lit anchor

**4 distinguishing predictions** (vs forward M1/M2):

| Prediction | Forward (M1/M2 activation) | Reverse (别扭) |
|---|---|---|
| 1. Drop-one ranking | P-SoM (compound) > P-text ≈ P-prompt | P-text ≈ P-SoM (1 别扭) > P-prompt (2 别扭) |
| 2. Compound aggregate effect | predicted positive (more activation = better) | predicted **negative** (compound disruption hurts) |
| 3. FP rate | not predicted | image-轴 别扭 → cautious commit → low FP |
| 4. Single 别扭 vs DOM aggregate | (forward predicts ≥) | predicted ≥ DOM (positive aggregate) |

**Empirical cross-cell validation (4 cells, Phase A pre-fix data)**:

| Prediction | B0 reddit | B0 cls | B1 cls | B1 red |
|---|:---:|:---:|:---:|:---:|
| 1. P-prompt drop-one lowest | ✅ 2.86 < 3.81/3.33 | (P-prompt pending) | (P-prompt pending) | n/a |
| 2. P-prompt raw SR < DOM | ✅ 10.48 < 11.43 | (pending) | (pending) | n/a |
| 3. Image-轴 别扭 → low FP | ✅ P-SoM 0.48 lowest | 🟡 P-SoM 1.28 < P-text 2.14 | 🟡 P-SoM 2.56 = DOM | n/a |
| 4. Single 别扭 ≥ DOM | ✅ both > DOM | ✅ both > DOM | ❌ **REVERSED** (P-text/P-SoM both < DOM) | n/a |

→ **B0 cells: 4/4 别扭 predictions confirmed**. **B1 cls: prediction 4 reversed**.

**Capability-modulated discovery (paper §7 finding)**:

Drop-one ranking 跨 capability 反转:
- B0 reddit: P-text 3.81 > P-SoM 3.33 (text-axis > image-axis)
- B0 cls: P-text 3.42 > P-SoM 2.56 (text-axis > image-axis)
- **B1 cls: P-text 0.85 < P-SoM 1.71** (image-axis > text-axis, REVERSED)

**Resolution: 别扭 + Lazy Minimization Hypothesis 联合**:
- 大 VLM (B0 235B-A22B): capability 充分 → text-format mismatch fallback effective → P-text drop-one 高
- 小 VLM (B1 4B): 视觉处理性价比差 → text-over-vision bias 强 → 优先 numeric label / 结构化 token. P-text 上 DOM-prompt vs [SOM_MARKS] obs format mismatch 让小 VLM confused (no capability for fallback). P-SoM 上 SoM-prompt + [SOM_MARKS] obs internally consistent + 仅 image 别扭 — 小 VLM thrives (依赖 numeric label, image absence less cognitive load).

→ **Capability-modulated 别扭 effect**: 小 VLM 偏好 image-轴 别扭, 大 VLM 偏好 text-轴 别扭。Paper §7 cross-capability 章节直接 cite。

**对 paper sections 的 implication**:
- §1 hook drop-one "1.7-3.8pp per arm" 加 capability-modulated caveat ("magnitude 4× weaker on small VLM, direction reverses text-vs-image axis preference")
- §2 Theory: forward (M1/M2) + reverse (别扭) **layered framework** — forward describes design + measurement, reverse describes mechanism + interpretation
- §5 mechanism prose 用 别扭 narrative ("compound mismatch overload" / "image-axis fallback via language priors")
- §7 cross-capability 章节 直接受益 (capability-modulated reversal 是 paper §7 真实证 finding, 解锁 §7 prose ~40% → ~70% completion path)

**Caveat (provisional)**:
- N=4 cells, 1 cell (B0 reddit) full 6-mode, 其他 partial
- Phase A bug pre-fix data — cycle false positives / dispatch noise affect aggregate metrics
- 14-cell rerun 后 8+ 完整 phantom cells, statistical commit time
- B1 reddit phantom 数据缺 (14-cell rerun 必跑) — 不能 yet validate B1 cross-site direction
- Cross-VLM-family (Claude / other 235B-tier) 待 advisor sync 后 决定 scope

### Zoom 3 (named phenomena): Lit-anchored mechanism phenomena

**M1 axis (Image-mirage) Zoom 3 lit anchors** (cross-model behavioral evidence):
- **Mirage Effect** (Asadi et al. 2026, arXiv:2603.21687, Stanford): VLM 无图时仍自信描述视觉特征, **无图准确率达有图的 70-80%** (mirage-mode > guess-mode). 实验笔记 §18 line 457
- **Scaffold Effect** (Vu & Balloccu 2026): prompt 仅提及 modality 可用就解释 **70-80% 性能变化** independent of image presence. 临床 VLM 起源, web agent 同样适用. 实验笔记 §25 + phantom_som.md line 281
- **Cross-modal flow** (Kaduri et al.): middle-layer cross-modal flows store image info in query tokens, allowing image-consistent generation without direct image-token attention (phantom_som.md §2.1 line 83-89) — actually 这个偏 Zoom 4 mechanism

**M2 axis (Flat-list) Zoom 3 lit anchors**:
- **Prompt-format sensitivity** (Sclar 2024 / Mishra 2022): LLMs 对 spurious formatting features 极度敏感, minor prompt changes → major performance shifts
- **Tree-traversal vs list-scanning** (你 deep research SoM novelty doc line 84): "AXTree induces tree traversal trajectory (logical deduction over hierarchy) vs flat SoM induces sequential list scanning trajectory (rapid spatial approximation)"
- Paper draft `section2_background.md` line 27 已 adopt: "the flat marks list tends to shift exploration toward quick element selection, AXTree hierarchy supports sustained navigation and search"

### Zoom 3 expansion (5/6 Gemini DR returns 2026-05-01, Q5 pending)

**Q1 Mirage / visual prompting w/o image — additional cross-model anchors** (`docs/literature/5.1/Cost-Aware Routing...md`):
- **Wang et al. 2025 XLRS-Bench** (CVPR 2025, arXiv:2503.23771): text-only Qwen3-8B 51.6% **>** multimodal GPT-4o 45.2% on aggregate visual remote-sensing tasks — text-only **beats** multimodal in some visual-heavy domains
- **Liu et al. 2025 Plan-and-Act** (ICML 2025): 81.36% text-only SR on WebVoyager, 57.58% on WebArena-Lite — text-only SOTA via two-tier planner-executor architecture
- **Lù et al. 2025 AgentRewardBench** (arXiv:2604.04399): "**distraction phenomenon**" — incorporating both text + image observations *degrades* PRM performance vs text-only inputs (image introduces high-dimensional noise on text-defined semantic boundaries)
- **Koh et al. 2024 VWA empirical**: GPT-4 text-only 7.25% vs GPT-4V multimodal 16.37% = **44% relative retention** (more conservative than Asadi 70-80% on real web tasks vs synthetic mirage benchmarks)
- **Cross-VLM-family confirmed**: GPT-4V/4o (XLRS-Bench, ViLP) + Claude 3 (cross-modal flow) + Gemini 1.5 Pro (WebNavigator 72.9% multi-site) + Qwen-VL/LLaVA/InternVL (text-inertia + Mirage) — all exhibit Mirage Effect, validating cross-architecture generalization

**Q2 system prompt format (multi-step agent)** (`docs/literature/5.1/Sensitivity of LLM Agents...md`):
- **Multi-step compounding finding**: format-induced loop rates compound exponentially over multi-step trajectories (single-turn lit understates effect)
- **Surprise**: "**Text-based DOM-element referencing consistently outperforms Set-of-Mark visual bounding boxes** for prompt referencing due to alignment with textual pre-training" — relevant to paper §2 axis 2 framing (need careful prose distinction, see counter-evidence)
- **Structured hallucination**: forced JSON output induces structured hallucinations + cycle increase, particularly on smaller models (Llama-3-8B)
- **CoT trade-off**: Chain-of-Thought helps single-screen reasoning, **hurts** complex compositional sequences (catastrophic trajectory degradation when agents skip intermediary steps)

**Q3 AXTree vs flat list — head-to-head + SoM-text isolation gap** (`docs/literature/5.1/Empirical Analysis of Observation Modalities...md`):
- ✅ **Paper §1 first-work claim VERIFIED**: "**no study isolates SoM-style flat text as a standalone observation without its accompanying marked screenshot. The target paper fills this unprecedented gap**" — Phantom-SoM "first systematic SoM-text isolation" claim is lit-verified novel
- **Kerboua et al. 2025 FOCUSAGENT** (arXiv:2510.16252): AXTree pruning achieves task-success comparable to full AXTree at significantly reduced token count, but does NOT compare to flat lists
- **Tan et al. 2026 HMT** (arXiv:2603.07024): hierarchical 84.2% recall vs flat 65.8% — applied to **memory architecture** not direct observation, suggests structural format trade-off is task-dependent
- Forward citation: Yang 2023 SoM original paper has near-zero work isolating text from image — reaffirms gap

**Q4 Scaffold Effect cross-domain** (`docs/literature/5.1/Modality Collapse...md`):
- ✅ **No fundamental counter-evidence**: "A comprehensive review of the 2023-2026 literature yields **no fundamental counter-evidence** directly contradicting the core premise of the Scaffold Effect"
- Scaffold + Mirage + Text Inertia identified as 3 manifestations of same underlying "modality collapse" phenomenon
- DPO mitigation contestably fails ("collapsing multi-modal accuracy to random baselines rather than establishing genuine visual grounding")
- → Paper §5 axis 2 Mirage subspace anchored robustly; reviewer attack "Vu & Balloccu only clinical" preempted

**Q6 Lazy Minimization Hypothesis cross-VLM scaling** (`docs/literature/5.1/Examining the Lazy Minimization Hypothesis...md`):
- **Scaling Law of Redundancy**: large VLMs robust to image-token pruning (Qwen-32B MMLU 80.81 → 80.01 under 80% retention pruning); small VLMs collapse harder
- **Pre-training mixture**: Qwen2-VL-7B BUTTERFLY ablation — text-only fine-tuning **slightly outperforms** image-text training (50.50% vs 50.00%) for complex conceptual learning — text provides cleaner gradient signals
- **Idefics2 8B**: trained on OBELICS (350M images / 115B text tokens), text:image ratio fundamentally conditions transformer to prioritize text representations
- **VILA**: re-blends text-only instruction data alongside image-text data during fine-tuning to remedy text-task degradation, inadvertently boosting text inertia

**Q5 Bidirectional modality fusion (Gemini DR 6/6 returned 2026-05-01 evening)** (`docs/literature/5.1/Vision-Language Model Modality Interaction A Comprehensive Analysis of Bidirectional Dominance and Failure Modes.md`):

⭐⭐ **Paper §2 Axis 3 lit anchor 厚度终于到了** — 之前 axis 3 只有单点 anchor (Tong 2024 / Li POPE / Bitton-Guetta)，Q5 给 cross-benchmark 综合 + bidirectional framing 学术名词。

- **Bidirectional Failure framing 是 "Novel synthesis"** (Q5 doc 自己 line 22 标注): "framing VLM modality interaction as exhibiting **dual failure modes that act in opposite directions** constitutes a novel theoretical synthesis. The current 2023-2026 literature largely treats these failure modes in isolation." → Paper §5 axis 3 prose 可 cite Q5 综述 + claim "first systematic web-agent multi-step application"
- **Dual-axis same-paper benchmarks** (验证 axis 3 bidirectional framing 不是 paper artifact):
  - **Guan et al. 2024 HallusionBench** (CVPR, arXiv:2310.14566): 单 benchmark 同时测 "visual illusion" (image-over-text) + "language hallucination" (text-over-vision). GPT-4V 31.42% / 其他开源 <16% on dual-axis control pairs
  - **Sun 2023 MMHal-Bench**: 8 distinct error categories, both visually-grounded failures + language-driven hallucinations on same model
- **Per-axis benchmark mapping** (paper §5 prose anchor):
  - **Image-over-text** (M1 visual saliency hijack): Li et al. 2023 POPE (EMNLP, arXiv:2305.10355) — co-occurrence triggers false positives; CHAIR — autoregressive text fills visual gaps
  - **Text-over-vision** (M2 language prior override): Tong et al. 2024 "Eyes Wide Shut" (CVPR); Fu et al. 2024 BLINK (ECCV, arXiv:2404.12390) — perception primitives at 24-30% near random; Bitton-Guetta et al. 2023 WHOOPS! (ICCV); MM-Vet — OCR/spatial trigger text fallback
- **Scaling law on bidirectional bias** (paper §7 cross-capability anchor):
  - **Large LLM (70B+) + compressed vision** → text-over-vision (parametric memory dominates sparse visual tokens)
  - **Small LLM (7B) + uncompressed dense vision** → image-over-text (dense tokens overwhelm small LLM reasoning)
  - **Long-sequence generation (any size)** → "**vision sink**" / cross-attention decay over time → text-over-vision drift
  - 这给 B0 (235B-A22B MoE + image embedding ~1.5K tokens) 跟 B1 (4B + 同 image budget) 一个不对称 mechanism prediction：B1 image-over-text dominance 更强；B0 long-trajectory text-over-vision drift 更强
- **SoM-specific harming channels formalized** (paper §5 axis 3 8-channel 的 cross-paper validation):
  - **SoM occlusion → image-over-text (saliency hijack)** — markers obscure underlying pixels + act as "embedded geometric shape" distractor (与 §100 ground truth -60pp OCR 一致)
  - **Numeric attention hijack at high mark density → text-over-vision (prior override)** — LLM 优化处理 alphanumeric, "latches onto numeric IDs as primary reasoning anchors" 离开 spatial grounding (与 §100 num_ids 0→446 一致)
  - 注意 Q5 给两条 SoM channels 的 dominance direction 是 opposite — 这正是我们 axis 3 8-channel framework "bidirectional fusion" 的 lit anchor (一个 channel image-over-text，另一个同框架内反向 text-over-vision)
- **Mid-layer attention decay (Liu 2025 "Devils in Middle Layers", arXiv:2512.07730)**: object hallucinations 是 multi-faceted — 同时由 mid-layer visual attention decay + decoding 时 language prior dominance 导致。**Paper §5 axis 3 双因机制 anchor** (从单 cause 升级到 dual cause 解释)
- **Counter-evidence**:
  - "Seeing but Not Believing" (2025): linear probing 显示 visual encoder 准确提取 features，failure 是 **late-stage generative disconnect** 而非 cross-modal interaction failure → reviewer attack vector "你说 cross-modal failure 但其实是 decoding-stage failure"，paper §5 prose 应承认 mechanism granularity (我们的 8-channel 是 behavioral level, 不 commit on encoding vs decoding stage)
  - M3amba (2025): 显式 bidirectional state space (BiMamba blocks) 在 specialized domain (pathology) 可 lossless integrate — 说明 bidirectional dominance 不是架构必然, 是 generalist VLM 的 typical pattern (paper §7 prose specialized exception caveat)

→ **Paper §5 axis 3 prose strategy 升级**: 从 "我们 propose 8-channel" → "我们把 cross-paper 散落的 dominance findings 在 web-agent multi-step setting 系统映射成 8-channel + 提供 first systematic 分类". Bidirectional framing 是 Q5 "novel synthesis"，paper 在 web-agent 应用层是首次。

### Zoom 3 counter-evidence catalog (NEW 2026-05-01 from Gemini DR, mandatory for paper §5 honest framing)

**M1 (Image-mirage) counter-anchors** — text-only fallback fails on these:
- **AsgardBench 2026**: text-only collapses on perception-conditioned execution tasks ("Text-Only performance remains low across models")
- **DailyDroid Benchmark 2026**: 75 tasks × 25 Android apps, multimodal > text-only on dynamic mobile GUI (margin "marginal" but consistent)
- **Mind2Web Deep Research 2025**: text-only struggles with structurally ambiguous DOM environments + noisy spatial layouts
- **FileGramOS / Mind2Web Tracking 2025**: text-only insufficient for behavioral state tracking; rendered page images expose certain operational statistics
- → **Paper §1 hook honest framing**: text-only fallback works on **standard web schema with stable DOM**; fails on perception-conditioned, dynamic GUI, behavioral tracking, structurally ambiguous tasks

**M2 (Flat-list) counter-anchors** — flat-list framing is not universally superior:
- **Tan 2026 HMT**: hierarchical 84.2% recall vs flat 65.8% (memory architecture context, not direct observation)
- "**Flat sequences actively remove critical structural signals** required for human-level spatial reasoning" — failed to indicate line wraps, grouping, spatial organization
- **Q2 surprise**: DOM-element-id referencing > SoM [N] referencing in some prompt-instruction settings — paper §2 prose must distinguish phantom-space P-text uses [SOM_MARKS] **obs format** (different from "SoM-style prompt referencing system" — these are independent dimensions in our 2x2 cube)
- **Zheng et al. 2024 EMNLP** "When 'A Helpful Assistant' Is Not Really Helpful": 162 personas, **NO effect on accuracy** vs no-persona baseline — counter-evidence on persona-style prompt effects
- → **Paper §2 prose distinguish**: phantom space P-text is obs-format swap (axis 1), NOT prompt-instruction-style swap. Sclar/Mishra prompt-format sensitivity applies to former.

**M3 (Image axis bidirectional dominance) counter-anchors** — bidirectional framing not universal (Q5 Gemini DR):
- **"Seeing but Not Believing" 2025**: linear probing on visual encoder shows it accurately extracts features; failure is **late-stage generative disconnect inside LLM**, NOT cross-modal interaction failure. Implication: paper §5 axis 3 prose 不应 over-commit "cross-modal failure" mechanism — 我们 8-channel 是 behavioral level taxonomy, 不 claim encoding-stage vs decoding-stage attribution
- **M3amba 2025**: explicit BiMamba blocks 在 pathology (specialized domain) achieve lossless bidirectional integration — bidirectional dominance 不是架构必然, 是 generalist VLM 的 typical pattern. Paper §7 cross-domain caveat: 我们 finding 限于 generalist VLM (B0 Qwen3-VL-235B, B1 Qwen3-VL-4B), domain-specialized fine-tuned VLM 可能不复现
- **MM-Vet 7-capability decomposition**: failure mode is **task-dependent / both directions** — single dominance direction framing 可能 oversimplify. Paper §5 prose 应保持 "site-modulated" framing 跟 Q5 task-dependent 一致 (cls visual-rich → image-axis dominate; red text-dom → text-axis dominate)
- → **Paper §5 axis 3 honest framing**: bidirectional fusion 是 behavioral-level pattern, 不 commit mechanism stage; site/task/capability modulate which direction dominates

**Capability-scaling (Lazy Minimization) counter-anchors** — small VLMs are not universally inferior:
- **ChartGemma 3B** (early-fusion VLM fine-tuned for chart QA): 3B specialized model **beats GPT-4o**, on par with closed-source frontier VLMs — fine-tuning can override Lazy Minimization in narrow visual domains
- **InternVL3 8B**: under specific high-contrast marker conditions, **matches or exceeds Gemini 2.5 Pro** grounding accuracy (VPBench evaluation)
- **See&Trek**: spatial markers cause performance **drops** on Relative Direction and Object Counting tasks for some small VLMs — signal-priority hierarchy is not universally absolute
- → **Paper §7 prose framing**: Lazy Minimization is general trend for **standard generalist VLMs**, with domain-specialized fine-tuned exceptions

> **🆕 Paper §1 first-work claim verified (Q3 Gemini DR 2026-05-01)**: Gemini deep research synthesis explicitly confirms "no study isolates SoM-style flat text as a standalone observation without its accompanying marked screenshot" — Phantom-SoM paper §1 hook "first systematic SoM-text isolation" claim is lit-verified novel. Reviewer attack vector "you are not first" is preempted via Q3 forward-citation-chain analysis (FOCUSAGENT prunes hierarchy but doesn't compare to flat list; HMT compares hierarchical vs flat in memory architecture, not observation; Yang 2023 SoM original always bundles text with marked image).

### Zoom 4 (model-internal): Mechanistic probe lit anchors (paper §8 future work)

**Paper 不 self-probe Zoom 4** (因 B0 proxy API 不暴露 router logits internals + local deploy Qwen3-VL-235B-A22B 需 ~120GB VRAM 超 RunPod $200 budget). 但 lit anchors 给 mechanism plausibility:

- **Cross-modal flow** (Kaduri et al.): layerwise attention probe 显示 middle-layer cross-modal flows enable image-like representations from query tokens — M1 axis activation 的 mechanistic 解释
- **SteerMoE expert routing** (Fayyaz et al. 2026 ICLR, 学长 2026-05-01 发, 详 §108.9): MoE LLM 用 paired prompts 算 expert Risk Difference, alignment 集中在 subset of experts, alternate routing path 可绕过 ("Alignment Faking")。**对 phantom-SoM 的暗示**: vision-grounding 在 VLM 也可能 concentrated in subset of experts, phantom routing 通过 obs/prompt config 绕过这些 experts。Methodology template for paper sequence follow-up paper (Phantom-SoM Zoom 1+2+3 + SteerMoE-style follow-up Zoom 4 self-probe = paper trilogy)
- **Tool Calling Linear Steerable Circuit** (Anonymous 2026 ACL, 笔记 §19 archived 2026-04-09): 在 **Qwen3-4B** (跟 B1 = Qwen3-VL-4B 是 architectural cousin) 验证 — 15 tools → 10 PCA 方向 (90.2% var), **cosine gap 捕获 92% action-selection 错误**, L23+ layer steering 切 tool 准确率 80-93%, "**knows but cannot say**" (hidden state 77-89% correct, output layer 3-61%)。**对 phantom-SoM 的暗示**: (a) action selection 在 **action-type axis 线性可分** + argument generation 非线性，给 §1 cascade `DOM→P-text→P-SoM→SoM` 的 token-monotonic path 一个 mechanistic 理由 (轴 selection 比 argument 廉价); (b) hidden-state cosine gap 是 B1 白盒 routing signal candidate，AUROC 可对比 logprob (~2300 forward pass，无需重跑 environment); (c) "knows but cannot say" 跟我们 §70 infra 观测到 Bedrock proxy 静默吞 `tools`/`tool_choice` 参数返回纯文本的现象在不同 stack layer 互相印证 (架构层 vs API 层 stack-wide brittleness)。**对 B1-side 平衡 anchor**: SteerMoE / Cross-modal flow 都偏 B0 (235B-A22B MoE) 路径，Tool Calling Linear Circuit 是 Zoom 4 anchor stack 中**唯一在 4B 模型直接验证**的，给 paper §8 future work B1 self-probe 一个直接 method template (output_hidden_states=True forward pass + PCA + cosine gap AUROC, 仅 B1 白盒可行)

**Zoom 4 anchor stack 三角覆盖**:
| Anchor | Probe path | Capability tier | Paper 用途 |
|---|---|---|---|
| Cross-modal flow (Kaduri) | layerwise attention probe | model-agnostic | M1 axis (Mirage / Scaffold) 的 mechanism 解释 |
| SteerMoE (Fayyaz 2026) | expert Risk Difference + routing logits steering | **B0-side** (MoE family, Qwen3-30B-A3B cousin) | follow-up paper Zoom 4 self-probe method template |
| Tool Calling Linear Circuit (Anon 2026) | hidden-state PCA + cosine gap AUROC + L23+ steering | **B1-side** (Qwen3-4B cousin) | B1 白盒 routing signal + cascade action-selection 线性 mechanistic 支撑 |

**B0 = Qwen3-VL-235B-A22B 是 MoE family** (与 SteerMoE 实验 Qwen3-30B-A3B 是 architectural cousin), methodology 几乎可直接 transfer。**B1 = Qwen3-VL-4B** 跟 Tool Calling Linear Circuit 实验的 Qwen3-4B 是 architectural cousin (同 base LM, 加 vision encoder), tool selection 线性可分性合理 transfer 到 web action selection (click/type/scroll/finish 等 8 action 在结构上跟 ACL 论文 15 tools 同质)。两条 architectural cousin 路径让 follow-up paper 能 cover B0 + B1 双侧 mechanistic probe — 但本 paper 不 self-probe, 标 future work 列举。

### Zoom 4 paper sequence implication

```
Phantom-SoM 主 paper (本 paper):
  Zoom 1 ✅ (architectural completeness)
  Zoom 2 ✅ (M1/M2 behavioral activation framework)
  Zoom 3 ✅ (lit-anchored named phenomena)
  Zoom 4 标 future work, 不 self-probe

Follow-up paper (sequence sibling) — dual-tier Zoom 4 self-probe:
  B0-side: SteerMoE-style expert RD on B0 phantom corners
    需要 local deploy 或 API extension (RunPod 4×4090 ~$400-600 cost)
    Method template: SteerMoE paired examples → expert RD → routing logits steering
  B1-side: Tool-Calling-style hidden-state PCA on B1 phantom corners
    仅需 output_hidden_states=True forward pass (~2300 calls 已有 trajectory)
    无需重跑 environment, 单卡 RTX 4090 24h 可完成
    Method template: per-step hidden state → PCA action-direction → cosine gap → AUROC
    cross-validate 跟 logprob signal 比较, 检查 phantom routing 是否在 B1 hidden state 层可读
```

这给 paper §8 discussion 一个清晰 future work direction, 不需要 paper 内 over-commit Zoom 4 mechanism。

### Axis 1: Text payload structure (PRIMARY, first-order SR effect)

```
AXTree (hierarchical) vs [SOM_MARKS] (flat indexed) → action surface + trajectory basin
→ Phantom modes 获得 routing arm (因为 [SOM_MARKS] obs structure)
```

注意：这个 axis 是 **text payload 结构**（agent 看到的 obs 文本），不是抽象的"模型表征"。Token 数大致不变（~3K both），但 layout / parsing pattern 不同。

**LLM mechanism**:
- **Task ontology reframing — web-browsing → indexed selection** (核心 axis 1 effect, NEW 2026-05-01): AXTree (hierarchical tree, agent navigate tree structure 像 browser DOM walk) → `[SOM_MARKS]` (flat indexed list, agent picks ID 像 multiple-choice selection)。改变 LLM 任务 ontology 从 "browse the web" 到 "select from list"，trigger 不同的 in-context behavior。这是 P-text 4-mode drop-one +3.42-3.81pp 的根本机制。**Lit anchor**: deep research `docs/literature/The Novelty and Efficacy of Set-of-Mark Text...` line 84 frame 这个 split 为 "AXTree induces **tree traversal trajectory** (logical deduction over hierarchy) vs flat SoM induces **sequential list scanning trajectory** (rapid spatial approximation)"; paper draft `section2_background.md` line 27 已 adopt: "the flat marks list tends to shift exploration toward **quick element selection**, AXTree hierarchy supports **sustained navigation and search**". Sclar 2024 / Mishra 2022 prompt-format sensitivity theory 提供 transformer-level mechanism (different token distribution → distinct latent state → distinct exploration policy).
- Token distribution shift (hierarchical metadata vs flat indexed list)
- In-context learning bias (pretraining data context — selection-style prompts have stronger few-shot examples in pretraining)
- Long-context attention degradation (Liu 2023 "Lost in the Middle" — flat list 缓解 hierarchical mid-tree attention drop)
- Output format priming (selection prompts produce shorter / more committed outputs)

**Evidence**: fig3 strategy gradient (reddit verified §103 N=48, cls live extension), fig5 category × mode

### Axis 2: Prompt (multi-dimensional task-conditional decision prior)

**Falsified hypothesis**: "prompt 只影响 commitment confidence" (codex `5821387` falsifies)

**Replaced theory**: Prompt acts as task-conditional decision prior over:
- (a) search phrasing
- (b) candidate selection (ranking/disambiguation)
- (c) backtracking strategy
- (d) commitment confidence (FP gap evidence, **subeffect not唯一**)

**Visual prompting without image (P-prompt 的核心 LLM 机制, NEW 2026-05-01)**: SoM-style prompt 即使无 image 仍 activate agent 的 "visual-mark referencing" mental model —— prompt 期望 numerical-marker referencing system，agent 在 AXTree 上自动 fallback 到 element_id 引用，仍 recover **substantial fraction (~70-80%) of visual structure information from textual cues**。**Lit anchor stack (3 互补 mechanism)**: (a) **Mirage Effect** Asadi et al. 2026 (arXiv:2603.21687, Stanford) — VLM 无图时仍自信描述视觉特征，**无图准确率达有图的 70-80%** (mirage-mode > guess-mode), 实验笔记 §18 line 457; (b) **Scaffold Effect** Vu & Balloccu 2026 — prompt 仅提及 MRI 可用就解释 **70-80% 性能变化** independent of image presence (clinical VLM 起源, web agent 同样适用), 实验笔记 §25 + phantom_som.md line 281; (c) **Cross-modal flow** Kaduri et al. — middle-layer cross-modal flows enable VLMs to store image info in query tokens, allowing image-consistent generation without direct image-token attention (phantom_som.md §2.1 line 83-89). 即 axis 2 swap **不是** "prompt label 改"，而是 **task ontology 切换 from textual-action-prompt to visual-mark-referencing-prompt**。这与 axis 1 的 "browse → select" 形成对称 (axis 1 改 obs ontology，axis 2 改 action-referencing ontology)。

**Empirically verified**: B0 red P-prompt 4-mode drop-one **+2.86pp [0.95, 5.24] sig** (axis 2 alone 在 AXTree 锚点下有 routing value)；6-mode vs 5-mode marginal lift +1.90pp sig 验证 P-prompt 贡献 incremental unique tasks。

**Evidence**:
- P-text ∩ Phantom-SoM Jaccard 0.45-0.54 (task pool 显著 disjoint despite same SR)
- 6 case studies (codex `5821387` phantom_dom_vs_som_diagnostic.md)
- N=48 verified anchor: Phantom-SoM FP gap 2.08pp vs P-text 6.25pp
- B0 red P-prompt 4-mode drop-one +2.86pp sig (axis 2 alone @ AXTree 锚点 valid)
- B0 red 6-mode oracle +7.14pp [3.81, 10.48] vs 3-mode (3 phantom arms 都贡献 unique tasks)

**Lit support**: Persona priming (Salemi 2024), in-context learning (Min 2022), Sclar 2024 prompt format sensitivity, Mishra 2022. **Visual prompting without image / Mirage Effect** anchored on: Asadi et al. 2026 (arXiv:2603.21687) Mirage Effect (~70-80% of with-image accuracy); Vu & Balloccu 2026 Scaffold Effect (prompt mentioning modality alone explains 70-80% perf shift); Kaduri et al. cross-modal flow mechanistic explanation (image info stored in query tokens). 三个 mechanism stack 完整支撑 axis 2 alone routing value (B0 red P-prompt drop-one +2.86pp sig). Task-pool divergence Jaccard <0.5 是 paper unique empirical finding.

### Axis 3: Image (8-channel multi-dimensional, codex `7106d2e` validated)

**Falsified hypothesis**: "Visual-hijack 是唯一 image effect" (codex `7106d2e` falsifies)

**Replaced theory**: Image is **bidirectional modality fusion** with multiple sub-channels:

**Cross-paper lit anchor stack (Q5 Gemini DR 2026-05-01 evening)**: Bidirectional dominance framing 在 VLM general literature 是 "novel synthesis" (Q5 doc line 22), web-agent multi-step setting 是 paper §5 first systematic application。
- **Dual-axis same-paper benchmarks**: Guan 2024 HallusionBench (CVPR) "visual illusion" + "language hallucination" 同 suite; Sun 2023 MMHal-Bench 8 categories 同 model; 实证 bidirectional dominance 是 measurable architecture property 不是 paper artifact
- **Per-axis benchmark mapping**: image-over-text → Li 2023 POPE (co-occurrence FP) + CHAIR (autoregressive fill); text-over-vision → Tong 2024 Eyes Wide Shut + Fu 2024 BLINK (perception primitives 24-30% near random) + Bitton-Guetta 2023 WHOOPS! + MM-Vet OCR/spatial fallback
- **Mid-layer mechanism dual-cause** (Liu 2025 "Devils in Middle Layers" arXiv:2512.07730): mid-layer visual attention decay + decoding-time language prior dominance — paper §5 axis 3 dual-cause anchor

**Helping channels (4)**:
- 3a Spatial grounding (cls 5 tasks)
- 3b Visual context disambiguation (cls 13 / red 2)
- 3c Element disambiguation (cls 10)
- 3d State/action recognition (cls 1 / red 6)

**Harming channels (6, refined per user critique 04-28)**:
- **3e False visual confidence (image-over-text)** [MAIN red 60%, 9/15 failures]
  - Mechanism: image-text alignment pretraining bias → premature commit
  - Lit: Li 2023 POPE (EMNLP, co-occurrence FP), Yang 2024, Yu 2024 (object hallucination); Guan 2024 HallusionBench dual-axis benchmark "visual illusion" axis (Q5)
- **3f Text-over-vision fallback fail** (反向 modality bias) [cls task 24 verified]
  - Mechanism: language prior dominance → ignore actual image content
  - Lit: ⭐ Tong 2024 "Eyes wide shut" (CVPR), Bitton-Guetta 2023 WHOOPS! (ICCV); Fu 2024 BLINK (ECCV, perception primitives 24-30%); Liu 2025 Devils mid-layer attention decay; Guan 2024 HallusionBench "language hallucination" axis (Q5)
- **3g Visual saliency hijack on image content** [red task 0/167 verified]
  - 实测: 15× cycles same image link, element_id +2846 each cycle (page reload 不脱)
  - 与 SoM density 不直接相关 (mark_count 117 ≠ outlier vs P95 127)
- **3h SoM occlusion** [§100 ground truth, B0/B1 都 affected]
  - 量化: B0/B1 reddit_task_6 mode-SoM 18%/15% vs NoMarks 78%/75% → **-60pp OCR**
  - Lit (Q5): Q5 doc formalizes SoM occlusion → image-over-text (saliency hijack) — markers obscure pixels + "embedded geometric shape" distractor
- **3i SoM numeric label attention hijack** [§100 verified, density-dependent]
  - 量化: B1 num_ids 0→**446** at 128 marks; mode-WithText 立即降 0
  - Lit (Q5): Q5 doc formalizes numeric hijack → text-over-vision (prior override) — LLM optimized for alphanumeric "latches onto numeric IDs as primary reasoning anchors"; opposite dominance direction vs 3h, 验证 axis 3 同 framework 内 bidirectional
- **3j Visual misdirection** (visual saliency drift)

### Site-modulated framing (LLM-level explanation)

```
Task 需 vision (cls visual-rich):  3f text-over-vision 主导失败 → Phantom-SoM 失败
Task 不需 vision (red text-dom):    3e image-over-text 主导失败 → SoM 失败

Net effect:
  cls: helping 29 vs harming 13 → image win +6.84pp adj SR
  red: helping 8 vs harming 15 → image lose -3.33pp adj SR
```

### Site mechanical substrate (full characterization, 2026-04-29)

Each VWA/WA site has distinct **mechanical substrate** that determines which axis dominates. Section 5 narrative organized as `site × axis × LLM-mechanism` 3-way table, not just "axis effect on aggregate". Source for detailed per-site failure analyses: `docs/analysis/vwa_<site>/B*_{DOM,SoM,Vision}_digest.md` (9 files per site, manually + codex curated).

#### reddit (Postmill, N=210)

| Aspect | Detail |
|---|---|
| Information structure | Forum hierarchy (forum → posts → comments) |
| Navigation affordance | Sidebar `f/<forum>` links + search box (text-rich) |
| Image role | Content (post attachments), NOT navigation affordance |
| Mechanically dominant axis | **Axis 1 (text payload structure)** |
| Mechanism | AXTree hierarchical embeds sidebar in deep tree → search-box becomes shortcut → search-loop pathology. [SOM_MARKS] flat list makes sidebar `f/<forum>` directly clickable → DOM digest §2.1 search-loop 29.6% failure → P-text reduces |
| Image axis sub-effects | Macro 1b axis 3 small (5/5 reddit metrics show image effect d_z<0.16) — image is content not navigation, helping/harming roughly balanced |
| Site-specific failures | `fail_max_steps_search_repeat` (DOM 13.8%), `fail_no_progress` (22.4%), `fail_finish_eval_mismatch` (23.8%, read-and-report tasks) |
| Source digest | `docs/analysis/vwa_reddit/B0_DOM_digest.md` etc. |

#### classifieds (OSClass, N=234)

| Aspect | Detail |
|---|---|
| Information structure | Product listings + categories + search results |
| Navigation affordance | Category dropdown (`select_option`) + search box (intrinsic — most cls tasks are search) |
| Image role | **Product identity** (visual disambiguation critical for color/style) |
| Mechanically dominant axis | **Axis 3 (image)** — Macro 1b cls image axis dominates 5/8 metrics (h=+0.57 finish rate, d=−0.42 action repeat) |
| Mechanism | OSClass query routing (`/index.php?page=item&id=N`) means URL-path is uninformative — visual product comparison required for "find blue motorcycle" tasks. Image absence → P-SoM cls collapses toward DOM (Macro 1a 6/8 cells DOM-like). Image axis recovers at SoM. |
| Axis 1 sub-effects | Smaller; axis 1 (DOM↔P-text) on cls path-only Jaccard 0.904 (path-level same), 0.66 with query (semantic page-id divergence). Reveals macro-vs-micro mismatch: aggregate macro DOM-like but per-task page selection differs |
| Site-specific failures | Latent visual attribute (e.g. "red blanket" without ref image, A2 64% per `codex_audit_shopping_A_refined.json`), aggregation (least/most/cheapest, A3 35%), category navigation (case study task 12) |
| Source digest | `docs/analysis/vwa_classifieds/B0_findings.md`, `_DOM_digest.md` etc. |

#### shopping (Magento, N=466)

| Aspect | Detail |
|---|---|
| Information structure | Product pages + cart + checkout + admin (largest, most complex) |
| Navigation affordance | Product browsing + form interactions (custom-options swatch / qty / cart actions) + admin panel |
| Image role | **Product identification + visual variant selection** (color swatches partially DOM-readable, partially visual) |
| Mechanically dominant axis | **Axis 1 + Axis 3 mixed**, plus axis 2 prompt matters for form-action vs retrieval task split |
| Mechanism | Magento custom-options form interactions (swatch radio / select_option / qty) require precise element selection. §105 swatch state-change bug discovered 04-29 affects 2.4% tasks (DOM/SoM, not Vision). Visual-rich product variants need image axis for color/style disambiguation |
| Axis 1 sub-effects | Form action tasks need select_option for dropdowns; product retrieval needs visual ID |
| Site-specific failures | Aggregation (>50%, A3 dominant per `codex_audit_shopping_A_refined.json`), latent visual attribute (A2 41%), form-stall (swatch loop §105), admin-flow tasks |
| Site-specific quirks | Magento FPC cache full-page-cache requires hook + post-restart curl; custom-option radio swatch bug; review form ratings same bug pattern; long product comparison (12 items × 10 fields per Magento aggregation tasks) |
| Source digest | `docs/analysis/vwa_shopping/` (sparse, 跑中) + `codex_audit_shopping_A_refined.json` (A1/A2/A3/A4 sub-classification) + §105 swatch_form_change_audit.md |

#### Mechanism three-way table (Section 5 narrative scaffold)

```
                    reddit          classifieds         shopping
                    ─────────       ────────────        ──────────
Axis 1 (text)       PRIMARY         secondary           secondary (form-action)
                    sidebar→loop    page-id semantic    select_option matters
                                    divergence

Axis 2 (prompt)     macro driver    type/selfcorr only  prompt × text task split
                    of search/type  (cls aggregate masked)

Axis 3 (image)      weak/balanced   PRIMARY (5/8)       PRIMARY (visual variant)
                    image=content   image=affordance    image=ID + variant select

Site failure mode   search-loop     latent visual /     aggregation /
                    eval-mismatch   aggregation         form-stall / swatch
```

Section 5 prose 用此 3-way table 组织: per-site axis-by-axis mechanism, citing 4-dimension evidence + per-site digests + LLM-level mechanism (axis 1 attention shift / axis 2 task-conditional decision prior / axis 3 bidirectional fusion).

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

## §3 Findings — 4-dimension Evidence + Mechanism Framework (重组 2026-04-29, 2026-05-01 update: evidence/explanation separation)

> **Cross-reference (added 2026-05-04)**: §3 是 **paper finding evidence layer** (4 测量 × 4 cross-X = 16 sub-cells); 跟 §21 **intervention taxonomy** (~40 笔记 § environmental work) 是 orthogonal — §3 关心 phantom routing space 的 SR/cost/AUROC 实证, §21 关心 paper 周边 environmental scaffolding work 的 substrate categorization. **环境侧 fix** (§51-62 / §80 / §107 等 9 条) 是 §21 的 (i)+(ii)×L2 cell 内容, **paper §3 evaluation methodology footnote** acknowledge 这些 fix (跟 Avenir Web ignore env issue 形成 rigor differentiator), 但 detail 不在 §3 主分析。
>
> **重组动因 (§105)**：之前 10 条 finding 是 flat list，paper 写作时不好定位"哪个证据支持哪个 claim"。重组为 **4-dimension framework** —— 每个证据进对应 dimension，每个 paper claim 引用 dimension (e.g. "Outcome 0d Jaccard 0.447 supports routing-arm complementarity")。四个 dimensions 是 **正交** 的（不是 hierarchical layers）。**所有原 10 条 finding 都映射到对应 dimension，未删除**（见末尾索引）。

### Evidence vs Explanation Layer Separation (2026-05-01 update)

Paper conceptual structure 严格 2 层分离 (避免 evidence-as-explanation 混淆):

```
EVIDENCE LAYER (观测数据, 2D organize: 测量类型 × 比较 axis)
   测量类型 (4 dim)          ×    比较 axis (4 cross-X)
   ─────────────────              ─────────────────────
   Outcome (SR/oracle/Jaccard)   cross-task   (within-cell aggregation, 统计 foundation)
   Macro   (action 频率)         cross-mode⭐ (phantom space 主战场, paired)
   Micro   (per-step 决策)       cross-site   (跨 web 环境 generalization)
   Efficiency (cost/lat/carbon)  cross-model  (跨 capability generalization)
   
   = 4 × 4 = 16 evidence sub-cells (cross-mode 那列最 saturated, 是 §3 sub-codes 0a-3d 主居)

EXPLANATION LAYER (因果假说, 1D organize: zoom scale)
   Zoom 1 (coarsest, 架构):   Approach 2 = "phantom space 内 M1/M2 axes by design exhaustive"
                              (§2 Theory Framework 主用, deductive)
   Zoom 2 (中粗, behavioral):  "M1 produces Image-mirage activation, M2 produces list-scanning"
                              (§5 Mechanism prose 主用, inductive 用 Macro/Micro evidence)
   Zoom 3 (中细, named):       Mirage Effect (Asadi 2026) / Scaffold Effect (Vu Balloccu 2026) /
                              prompt-format sensitivity (Sclar 2024)
                              (§5 lit anchor cite, cross-model phenomenon names)
   Zoom 4 (最深, internal):    Cross-modal flow (Kaduri) / SteerMoE expert routing (Fayyaz 2026)
                              (§8 future work 标 lit anchor, paper 不 self-probe)
```

**关键区分**: §3 4-dim 是 evidence layer 的**测量类型轴**, cross-X 是 evidence layer 的**比较 axis 轴**。两者**正交 organize 同一份数据**。Explanation layer 跟 evidence layer 严格分开 — explanation 是 hypothesis (Zoom 1-4), evidence 是 data。Paper writing 时 reviewer 最忌 evidence-explanation 混淆 ("Macro 1c search-loop 51.9→35.7%" 是 evidence, "M1 axis activates list-scanning trajectory" 是 explanation Zoom 2 — 两者必须分写然后 explicit link)。

### Cross-X 比较 axis (4 类) 的 paper section mapping

| Cross-X | 固定 keys | 变 key | Paper section 用 | Saturation status (2026-05-01) |
|---|---|---|---|---|
| **cross-task** | mode + site + model | task | §3/§4 evidence aggregation 基础 | ✅ 234 cls / 210 reddit per cell |
| **cross-mode** ⭐ | task + site + model | mode (4 phantom + 2 image-on/only) | §4/§5 phantom space 主分析 | 🟡 B0 cls/red 5-mode done, B1 phantom 跑中, shop missing |
| **cross-site** | task + mode + model | site (cls/red/shop/WA) | §7 generalization | 🔴 B0 5-mode shop missing, WA 全 missing |
| **cross-model** | task + mode + site | model (B0/B1/Claude) | §7 cross-capability | 🔴 cross-model 数据稀薄, B1 phantom 跑中, Claude 0 |

**Paper §7 (generalization) 当前 ~40% 因为 cross-site + cross-model 两 axis 数据稀薄**, 不是 §7 prose 写作问题 — 是 cross-X saturation 不够。14-cell rerun 后 cross-mode + cross-model partially fill, 但 cross-site 仍需 shop + WA 数据 (Tier 2 expansion)。

### 4-dimension framework 概览

```
Outcome     哪些 task 成功 / 哪些 mode 在哪些 task 上 win
Macro       agent 平均怎么 act（action-type 频率）
Micro       agent per-step 决策（点哪个元素 / 走哪些页 / 搜什么词）
Efficiency  cost / latency / carbon (4-fold drop-in property)
```

每个 dimension 内部 sub-evidence (0a / 1c / 2a / 3d 等 sub-codes) 都标注 source artifact + current 数字（实时 live）。Sub-codes 保留作 figure-internal anchors。

---

### Outcome — task 成功 / 路由 arm 证据

| Sub | 内容 | Source artifact | 当前数字（B0, FRESH 04-29） |
|---|---|---|---|
| **0a** Aggregate raw + adjusted SR per mode | summary_v2.json live | live | red P-SoM **adj 13.81%** > all baseline; red P-text **12.38%** > DOM **9.52%**; cls SoM **21.37%** (best); cls P-text/P-SoM **adj 14.53%** ≈ DOM 14.10% |
| **0b** FP rate per mode | summary_v2.json (raw_succ 与 adj_succ 之差) | live | **red P-SoM 0.48%** (lowest, "honest commit"); cls P-SoM 1.28%; **§3-legacy finding 4 prompt-as-decision-prior 的核心证据** |
| **0c** Routing oracle uplift (3-mode → 4/5-mode) + drop-one | `phantom_lift.{md,csv}` | red 3→5: **+5.24pp** [2.38, 8.11] Wilcoxon p=0.0009 McNemar p=0.0005 ✅; cls +4.70pp [2.14, 7.69] p=0.0009 ✅. red drop-one P-text +3.81pp / P-SoM +3.33pp; cls P-text +3.42pp / P-SoM +2.56pp |
| **0d** Task-pool Jaccard (Scenario C sentinel) | `phantom_lift.md` | red P-SoM↔P-text **0.571** (≤0.7 safe ✅); cls 0.447 ✅. **核心 routing-arm 证据**：cls aggregate SR 同 (P-SoM≈DOM) 但 task-pool 0.53 disjoint —— same SR ≠ same routing pool |
| **0e** Per-category SR (4 cat × 5 mode heatmap) | `fig0e_category_mode_heatmap.png` ✅ live (rebuilt 04-29) | DOM-only Cat B (ref-image) systematic fail; cls Cat A counter-intuitive 8.54% < B 21.30% (codex audit refined → A1/A2/A3/A4 04-29) |
| **0f** Overlap depth (5-mode solve-pool depth distribution) | `fig0f_overlap_stacked_bar.png` ✅ live | red P-SoM 30 succ: 3 unique / 8 d2 / 7 d3 / 9 d4 / 2 d5; **§3-legacy finding 8** |
| **0g** Routing AUROC per mode (signal quality) | `auroc_cross_condition_summary.md` | red P-text 0.793 (5-mode max), P-SoM 0.720; cls P-text 0.737, P-SoM 0.728. **5/5 phantom `overall_usable=True`** —— paper 4-fold drop-in property (c) ✅ |

---

### Macro — agent 平均怎么 act

| Sub | 内容 | Source | 当前数字 |
|---|---|---|---|
| **1a** Tier 1 hook (3-mode coarse: DOM/P-SoM/SoM × 8 metric) | `axis_effect_size.py` (FRESH 04-29) + `axis_effect_size_report.md` | P-SoM "fully independent" cells: **red 4/8 vs cls 1/8**. cls P-SoM 主要"瘫向 DOM" (6/8 DOM-like) —— image axis 决定性, **印证 0d 的 task-pool 复杂性** |
| **1b** Tier 2a Mechanism cascade (3 axes × 8 metric) | `axis_effect_size.py` | **6 antagonistic pairs** (red scroll/text↔prompt 反向相消 / cls finish/prompt↔image 反向); cls **image axis 5/8 dominant** (h=+0.57 finish, d=−0.42 repeat); axis 1 在 macro 0/8 dominant (但 outcome 层 primary, 见 0c) |
| **1c** Strategy gradient (search-loop / type / scroll / selfcorr) | `fig1c_strategy_gradient.png` ✅ FRESH 04-29 全数据 | red DOM **search-loop 51.9%** → P-SoM 35.7% → SoM 31.4% (§3-legacy finding 3 升级版：从 §103 N=48 → N=210 全数据，原"5/5 metrics P-text=P-SoM"已 falsify) |

---

### Micro — per-step 决策

| Sub | 内容 | Source | 当前数字 |
|---|---|---|---|
| **2a** URL signature divergence | `axis1_microbehavior.{py,json,md}` (FRESH 04-29 codex + 我补 compound) | **axis 1 alone**: red Jaccard 0.573 / cls 0.904 (path-only). **compound DOM↔P-SoM**: red 0.481 / cls 0.885 path-only (但 cls path+query 0.66 —— OSClass 用 query routing). **决策真改了，aggregate macro 在 cls 上掩盖** |
| **2b** Target-page hit rate | `axis1_microbehavior.json` | red axis 1 +3.47pp; cls axis 1 +2.33pp; compound red −0.69pp / cls +1.74pp |
| **2c** Search-keyword reuse / repeat | `axis1_microbehavior.json` | red P-text vs DOM 重复 −0.633 (axis 1 减少死循环); cls P-text +0.077 (无 site 损失) |
| **2d** First-action divergence | `axis1_microbehavior.json` | red 21% / cls 14% tasks first-action type differ (axis 1) |
| **2e** Cross-site validity ratio | `axis1_microbehavior.json` `cross_site_validity` | **verdict: generalizes** (red 2.28, cls 1.02). cls 边界 —— 单独 axis 1 在 cls 上 micro≈macro，但 compound DOM↔P-SoM 在 cls 上 path+query Jaccard 0.66 强 divergence |

---

### Efficiency — 4-fold drop-in property

| Sub | 内容 | Source | 当前数字 |
|---|---|---|---|
| **3a** Token cost per step (input) | `condition_summary_v2.json` | P-SoM ≈ DOM (~3K both); SoM +image embedding tax. **4-fold drop-in (a) cost ≈ DOM ✅** |
| **3b** Image embedding tokens (per step median) | `run_summary_collect.json` | red 733 / cls 1064 tokens; **P-SoM 省去这部分**, **§3-legacy finding 6** |
| **3c** Latency per step | `condition_summary_v2.json` | P-SoM ~50% of SoM latency (无 image inference). **4-fold drop-in (b) latency ~50% ✅** |
| **3d** B0 (API) vs B1 (local) deployment-class cost gap | `cost_per_mode.{json,md}` (FRESH 04-29) + `fig3d_cost_sr_frontier.png` | **B0 API ~$0.04/ep (Qwen3-VL-235B-A22B token cost)**; **B1 electricity-equivalent ~$0.0004/ep** (DGX Spark `avg_total_energy_kwh × $0.12/kWh` UK industrial rate). **Ratio ~100×** (red 98× / cls 105×) — **deployment-class gap, NOT capability/parameter ratio**. ⚠️ §103 / §3-legacy "30×" claim **superseded** by FRESH data. Paper presents both classes side-by-side, not a single multiplier. |

---

### Cross-dimension Mechanism Chain（每个 axis 在哪些 dimension 上 first-order）

| Axis | Outcome dimension 贡献 | Macro dimension signature | Micro dimension signature | Efficiency dimension cost |
|---|---|---|---|---|
| **Axis 1 (text payload)** | **PRIMARY** (red P-text +3.81pp drop-one over 3-mode; cls +3.42pp) | 0/8 dominant (但**红 scroll/selfcorr 是 antagonist canceller**) | red URL Jaccard 0.57 / cls 0.90 (axis 1 alone) — 在 reddit 改 WHERE 强 | 0 (text swap 不改 token 量) |
| **Axis 2 (prompt)** | secondary (red P-SoM 加在 P-text 上 +3.33pp; cls +2.56pp) | red 3/8 cascade dominant (search/type/scroll); cls 3/8 (type/selfcorr/click) | URL Jaccard 0.55 (axis 2 alone) | 0 (prompt swap 不改 token 量) |
| **Axis 3 (image)** | secondary (cls SoM 21.37% > P-SoM 14.53%, image 决定性 cls 上) | **cls 5/8 dominant** (finish h=+0.57 medium-effect 最强信号); red 3/8 dominant (efficiency cluster) | image 加上 = URL Jaccard 0.46-0.60 minor change | **+700-1100 image tokens** (Efficiency 3a 主要 cost source) |
| **Compound Axis 1+2 (P-SoM vs DOM)** | red SR delta +2.86pp aggregate; **cls SR delta 0.85pp 但 task-pool Jaccard 0.53** = routing-arm 证据 | cls macro 60-70% DOM-like 但 task-pool 0.53 disjoint —— aggregate 误导 | **path+query Jaccard cls 0.66 / red 0.48** —— per-step decision quality 真改了 | 0 |

---

### Evidence chain — paper claims → dimension support

每个 paper claim 直接 cite dimension + 数字：

| Paper claim | Dimension support |
|---|---|
| **C1**: P-SoM is independent routing arm | 0a (red SR best), 0c (drop-one 3.33pp red / 2.56pp cls), 0d (Jaccard ≤ 0.6), 0g (AUROC ≥ baseline), 1a (red 4/8 macro independent), 2a (red URL Jaccard 0.48 micro divergence) |
| **C2**: 4-fold drop-in property (cost / latency / signal / drop-one) | (a) Efficiency 3a, (b) Efficiency 3c, (c) Outcome 0g, (d) Outcome 0c |
| **C3**: 3-axis hierarchical theory | Macro 1b (cascade decomposition), Micro (axis-by-axis micro), Cross-dimension table |
| **C4**: aggregate macro can mislead about routing potential (cls case) | Macro 1a (cls 6/8 DOM-like macro) + Outcome 0d (cls task-pool Jaccard 0.53) + Micro 2a (cls path+query Jaccard 0.66) |
| **C5**: prompt as task-conditional decision prior (not commit-only) | Outcome 0b (FP rate), Outcome 0d (Jaccard 0.45-0.55 same-SR-different-pool), Macro 1b (cascade axis 2 dominant on red strategy metrics) |
| **C6**: image is bidirectional 8-channel modality fusion | Macro 1b (cls image axis 5/8 dominant), Outcome 0e (codex audit category × mode), codex `7106d2e` channel decomposition |

---

### Mechanism chain — 三个机制阶段

```
Stage 1 (Outcome 层):    P-SoM 的 routing arm 价值在 task-pool complementarity (0d), 不在 aggregate SR
                         ↓ 为什么 P-SoM 拿到 unique tasks?
Stage 2 (Micro 层):       因为 axis 1+2 swap 改变了 per-step 决策 (2a 0.48-0.66 URL Jaccard)
                          ↓ 这些决策具体改了什么?
Stage 3 (Mechanism 层):  axis 1 改 text payload 结构 → 改 in-context attention pattern → 改 element selection 决策
                          axis 2 改 prompt 描述 → 改 task-conditional decision prior → 改 commit / search / 导航策略
                          axis 3 加 image → 改 visual disambiguation → 决定 cls 上的 finish rate
```

**关键 insight**: Macro dimension (action-type 频率) 是 downstream signal，单独看会误导 (cls case)。真正的 mechanism chain 是 Micro dimension (decision quality) + Outcome dimension (task-pool complementarity) 闭环。

---

### Honest framing (avoid over-claim)

- Phantom-SoM red **adj 13.81% > SoM adj 10.48%** —— 这次有数据，是 site-specific dominance（reddit 上）
- cls SoM **adj 21.37% 显著领先 P-SoM 14.53% (+6.84pp)** —— 反例必须明示, image 在 cls 是决定性 axis (Macro 1b 5/8 dominant 印证)
- 主 narrative: **site-modulated representation × prompt × image effects**, 不是 "Phantom #1 universal routing arm"
- Macro dimension 单独 weak on cls (1/8 fully independent) —— paper 必须用 Outcome task-pool + Micro 一起讲，不能只 cite macro
- §103 N=48 "5/5 metrics P-text = P-SoM" 已 **superseded** by N=210 (FRESH 04-29 Macro 1c) — 早期 small-sample artifact

---

### Legacy index (原 10 条 finding 映射)

Naming traceability (04-29): completed filesystem run dirs now distinguish paper-facing text from SoM phantom arms:
`B0_phantom_*` completed runs became `B0_phantom_som_*`, and completed `B0_phantom_dom_*` runs became `B0_phantom_text_*`. Internal mode IDs and condition dirs remain unchanged (`phantom_dom` / `phase1_phantom_dom_router_0`, `phantom_som` / `phase1_phantom_som_router_0`) for backward compatibility with recorded JSONL.

| 原 finding | 映射到 dimension |
|---|---|
| 1 DOM Cat B 系统性失败 | **0e** per-category heatmap |
| 2 Image 8-channel cross-capability | **1b axis 3 + 0e cls cat × image** |
| 3 DOM reddit search-loop 22.7% | **1c strategy gradient** (升级为 N=210 51.9%) |
| 4 P-text cls adj 14.53% ≈ DOM 14.10% | **0a + 0d** (same-SR-different-pool) |
| 5 P-SoM cost ≈ DOM cost | **3a** (4-fold drop-in (a)) |
| 6 Image tokens per step (red 733 / cls 1064) | **3b** |
| 7 B0 vs B1 cost gap | **3d** (修正 04-29: ~100× deployment-class gap, NOT 30× — see `cost_per_mode.md`) |
| 8 Phantom unique tasks (fig8) | **0f** overlap depth |
| 9 AUROC ≥ baseline | **0g** (4-fold drop-in (c)) |
| 10 Watchdog 100% pure | (data quality precondition, 不是 finding) |

新增 finding（§105 04-29）：
| 新 finding | Dimension |
|---|---|
| **N1**: P-prompt 模式必要性（symmetric ablation, AXTree+SoM-prompt+无图） | 设计层 (§2 cube), 数据 pending B0 reddit 跑中 |
| **N2**: Tier 1 hook macro: red 4/8 cells fully independent / cls 1/8 (cls 主要 DOM-like) | **Macro 1a** |
| **N3**: 6 antagonistic mechanism pairs（4-level cascade vs 2-endpoint 比较的核心 paper value） | **Macro 1b** |
| **N4**: cls compound DOM↔P-SoM micro path+query Jaccard 0.66 | **Micro 2a** |
| **N5**: P-SoM cls aggregate SR ≈ DOM 但 task-pool 0.53 (12 unique successes) | **Outcome 0d** (reframes "cls Phantom-SoM 失败" 为 "complementary not dominant") |
| **N6**: red P-SoM FP=0.48% lowest（最 honest commit） | **Outcome 0b** |

---

### Evidence vs Explanation: framework 的真实定位（2026-04-29 反思）

4-dimension framework **不是 paper Section 4/5 的 narrative 结构**，是**分析 scaffold + future-data drop-in 索引**。明确两个层次：

#### 4-dimension = Evidence dimensions（paper Section 4）

观测 evidence: "在 mode/axis swap 下我们 observe 到什么 shift"
- Outcome: 哪些 task 成功（SR / oracle / Jaccard / category / overlap / AUROC）
- Macro: agent 平均怎么 act（action-type 频率 cascade）
- Micro: per-step 决策怎么变（URL / target / keyword）
- Efficiency: 资源 footprint（cost / latency / carbon）

四个 **正交 dimensions**（不是 hierarchical layers），从宏观 outcome 到微观 decision。Paper Section 4 是 evidence catalog，每个 sub-finding 引用一个 dimension 的数据 + figure。

#### LLM mechanism = Explanation layer（paper Section 5）

解释 evidence: "为什么 axis swap 产生这个 shift"——必须 **跨 dimension** 同时 **site × axis × LLM-mechanism** 三阶交互：

```
观测 (evidence): reddit axis 1 swap → search-loop 51.9 → 35.7 (Macro 1c) +
                                       URL Jaccard 0.57 (Micro 2a) +
                                       SR uplift 4.76pp drop-one (Outcome 0c)
解释 (LLM mechanism):
  AXTree (hierarchical, sidebar embedded in tree) → [SOM_MARKS] (flat indexed list)
  ⇒ attention pattern shift: sidebar forum link 在 flat list 显著
  ⇒ agent 直接 click forum link 而非 search-loop
  ⇒ trajectory 变短 + 决策准 + SR up
  ⇒ 横跨 Macro+Micro+Outcome dimensions 的 single mechanism
```

不同 site 触发不同 mechanism (site × axis × LLM):
- **reddit text-heavy forum**: axis 1 主要影响 attention pattern (sidebar visibility)；image axis 几乎冗余
- **cls visual-rich product browsing**: axis 3 image 是 affordance（finish-rate h=+0.57 决定性）；axis 1 主要影响 ID-system parsing efficiency

**paper Section 5 narrative 由 mechanism 驱动**, dimension-organized evidence 作 underlying support — 不是按 dimension 组织 narrative。

#### Axis decomposition（diamond 完整后的 final form）

```
total observed effect (DOM → SoM endpoint)
  = main(axis 1, P-text alone via DOM↔P-text)
  + main(axis 2, P-prompt alone via DOM↔P-prompt)              [DIAMOND ENABLES]
  + main(axis 3, image alone via P-SoM↔SoM)
  + interaction(axis 1, axis 2)                                 [DIAMOND ENABLES]
  + ...higher-order interactions usually 0
```

P-prompt 是必需的，因为它是 **axis 2 在 AXTree-text context 下的唯一测量点**。如果 interaction term ≈ 0 → paper 写 "axis additive, independent first-order"；如果 interaction term ≠ 0 → honest disclose "axis 1 effect is modulated by prompt context"。任一 verdict 都比 cascade-only 多一个 quantitative claim。

#### Framework 的 future-data 弹性

所有 cells 自动落到 dimension × site × axis × baseline 索引：

| Future data | drop-in 到 |
|---|---|
| B1 phantom cls/red 4-cell (P-SoM + P-text × cls + red) | Outcome/Macro/Micro/Efficiency × cls/red × B1 cells |
| B1 P-prompt cls/red (Tier 2) | Diamond axis 2 in B1 capability — Section 7 cross-capability |
| B0/B1 shopping 6-mode | Outcome 0e per-category（shopping-rich audit categories）+ all dimensions shopping cells |
| WA B0/B1 6 sites × 5 modes | Cross-benchmark generalization (Section 7 main) |
| Claude Opus 4.7 5-mode | Cross-model boundary check (Section 7) |
| 其他 benchmark | Same scaffold, no rework |

`make analyze-layered` 是 idempotent 的——新数据 commit 后跑一遍 `layered_status.py` 自动 regenerate `layered_evidence_status.md` + 所有 figures。CLI alias 保留 (`analyze-layered`, `layered_status`, `layered_evidence_status.md`) 是 backward compat — paper-facing 命名是 4-dimension。

#### Caveats / honest framing

- **N=234 cls underpower**: cls Macro dimension 弱信号可能是 statistical power not enough（needs ~800 task to detect d=0.2 small effect with α=0.05, β=0.2）。后续 shopping 466 + WA 480 数据可补强
- **命名约定**: paper-facing 用 "Outcome / Macro / Micro / Efficiency" (4 orthogonal dimensions). Sub-codes (0a / 1c / 2a / 3d) 保留作 figure-internal anchors. Code-level CLI 保留 "layered_*" 别名 (Makefile target / `layered_status.py` / `layered_evidence_status.md`) 作 backward compat
- **不是所有 13 figures 进 paper**：Section 4 只 cherry-pick 5 个代表 figure (e.g. fig0c + fig0d + fig0g + fig1c + fig3d)，其他 supplementary
- **paper Section 5 可能简化**：若 codex prose 过分 dimension cataloging，必须 mechanism-first restructure

---

### Mechanism Tier 1/2/3 escalation plan (Section 5 explanation methodology, 2026-04-29)

4-dimension evidence catalogs *what shifts*; Section 5 mechanism explains *why*. Three escalating tiers, only Tier 1 is currently feasible; Tier 2/3 execute on existing 实验笔记 §19 future-work plan once B1 GPU frees up.

#### Tier 1 — Behavioral mechanism (paper-ready now, B0+B1 data, no GPU work)

Per-task per-step decision-quality metrics, mode-invariant, computable from existing step JSONL:

| Metric | What it measures | Dimension | Status |
|---|---|---|---|
| **E1** click-target Jaccard | per-task `(pre_url, post_url)` transition signature, mode-invariant + step-invariant | Micro | 🟢 codex prompt ready (`mechanism_per_task_explanation.md`) |
| **E2** trajectory boundary | for symmetric-diff success tasks, first divergent step | Micro | 🟢 prompt ready |
| **E3** confidence calibration cross-condition | ECE/MCE/Brier/AUROC per (model, site, mode), aggregating existing `analyze_confidence_calibration.py` per-run output | Outcome 0b + Macro | 🟢 prompt ready |
| **E4** action vocabulary distribution | full action_type × subtype frequency per cell (extends axis_effect_size's 4 metrics) | Macro | 🟢 prompt ready |

Tier 1 deliverables: `scripts/analysis/mechanism_per_task.py` + `docs/analysis/cross_sites/mechanism_per_task.{json,md}`. Adds `make analyze-mechanism` target. ~80K codex tokens. **Trigger anytime**.

#### Tier 2 — Mechanistic interpretability (B1-only, executes 实验笔记 §19 future-work)

实验笔记 §19 已 documented "Tool Calling is a Linear, Steerable Circuit" (ACL 2026, Qwen3 4B verified) 适用于 P79: action selection 是线性电路, cosine gap 预测 92% 错误, L23+ steering 可 80-93% 准确率切换 tool。Section 5 paper-strongest mechanism evidence 走这条路:

| Metric | What it measures | Tooling | Status |
|---|---|---|---|
| **M1** B1 attention pattern probe | feed same task obs through B1 in DOM/P-text/P-SoM modes; extract attention to "forum sidebar link" / "search box" / "post title" tokens; measure shift across modes | `output_attentions=True` forward pass, ~2300 forwards | 🟡 blocked B1 GPU contention |
| **M2** B1 hidden state probing | layer L hidden state → probe "task will succeed"; PCA cosine gap (per §19) → AUROC vs logprob | `output_hidden_states=True` forward pass; PCA + LR | 🟡 blocked B1 GPU |
| **M3** Token-level decision attribution | next-action token distribution per mode; quantify "axis 1 改 token-level decision prior" claim | forward inference, no training | 🟡 blocked B1 GPU |

**Trigger condition**: B1 GPU 空 (~B1 phantom 4-cell chain done, ~30-40d ETA). 不需要重跑 environment — `~2300 task × ~12 steps = ~28K forward passes`, 离线 inference 单 GPU 可在 ~1-2 天 batch 跑完。Code 已部分存在: `analyze_confidence_calibration.py` 处理 logprob, 可扩展 `output_hidden_states/attentions` 提取。

**Paper value**: 比 Tier 1 行为分析更 mechanistic, reviewer 期望顶刊看到。直接对应 ACL 2026 Tool Calling lit。

#### Tier 3 — Causal mechanistic intervention (heavy, may be future paper)

| Metric | What it measures | Tooling | Status |
|---|---|---|---|
| **H1** Activation patching | DOM forward pass at (layer L, step S) → patch hidden state into P-text run → does behavior become DOM-like? | causal scrubbing infrastructure | 🔴 blocked B1 GPU + 1-2 weeks impl |
| **H2** Steering vectors | train PCA / linear probe to find "mode direction" in activation space; add steering vector at inference to induce mode-like behavior without obs/prompt swap | per §19 future work "L23 steering 修正 'know-but-cant-say'" | 🔴 blocked B1 GPU + advanced technique |
| **H3** Attention head ablation | systematic zero-out specific heads; find "axis 1 head" / "axis 2 head" responsible for mode-specific behavior | head-by-head intervention scaffold | 🔴 heaviest, possible split paper |

**Trigger condition**: 顶刊投稿 reviewer 要求 mechanistic 强化 OR 时间允许提前做。可能的 split: H1+H2 进 Section 5, H3 留 future work / paper 2.

**Paper value**: causal claim, 比 correlation-based mechanism (Tier 2) 更强. ACL/NeurIPS mechanistic interpretability track 期望.

#### 总体 Section 5 mechanism narrative cascade

```
Section 5 (顶刊版) 期望证据 stack:
  Tier 1 behavioral (E1-E4)  ← Section 5 fast-write, 现在 ready
  Tier 2 mechanistic (M1-M3) ← Section 5 strengthening, ~30-40d
  Tier 3 causal (H1-H3)      ← Section 5 顶刊 differentiator, optional
```

如果 deadline 紧, Tier 1 + Tier 2 already make Section 5 paper-grade. Tier 3 是 nice-to-have / split-paper option.

---

## §4 Paper Section Status (2026-04-29, 8 sections final scope; 2026-05-04 update with §21 cross-references)

| Section | evidence 质量 | 状态 | Hard blocker | §21 cross-reference (2026-05-04) |
|---|---|---|---|---|
| 1 Intro | ✅ 已写 (786w + 4-fold drop-in framing + conservative framing) | done `62c1380` `ef29add` | **Pending advisor sync 5/5**: substitution-gradient framing rewrite (§21.5 candidate prose ~370w with Magma+ScribeAgent same-Qwen-base differentiator) | §21.5 hook prose, §21.2 industry precedent stack |
| 2 Background + paper.bib | ✅ 已写 (1514w, 16 entries) | done `206cd93` | 待 codex #10 expand to ~38; **加 NLWeb/OmniParser-v2/Magma/ScribeAgent/AppAgent-v2/UI-TARS citations** | §21.5 differentiator table per system |
| 3 Definition + Ablation | ✅ 已写 (863w, token re-estimate corrected) | done `13b9608` `4d63c9f` `48db047` | **Pending**: §3 evaluation methodology footnote acknowledge 9+ environmental scaffolding interventions (Appendix D), cite WebAIM 2026 + Mind2Web 2 WebJudge taxonomy | §21.6 counter-evidence stack |
| 4 Empirical Findings | 🟡 80% (figures FRESH ✅ + B0 5-mode FRESH, prose 待 update) | data ready | codex #11 fresh prose (~30K) | §21.6 cost/latency anchors (241K vs 47-140K tokens) for §1/§6 quantitative positioning |
| 5 Mechanism | 🟡 90% evidence (3-axis × 8-channel × bidirectional × §100) | data 完整 | codex #13 prose (~50K, 待 #10 lit); **Pending**: shared ceiling argument 用 (iii)×L2 7 条 § 共享 root-cause finding | §21.3 (iii)×L2 self-perception 7 条 § |
| **6 Routing (Tier 1+2)** ⭐ NEW | 🟡 40% (signal AUROC ≥ baseline `9d7e99f`, infra scaffold) | scaffold ready | Tier 1 prototype (~3 天) + Tier 2 first-step trigger (~7-10 天) | §21.5 CoAct-1 OSWorld 60.76% task-class routing precedent; §21.6 cost anchors |
| 7 Generalization | 🟡 40% (B1 capability profile done) | partial | shopping (跑中) + WA + cross-model (Claude); **Pending advisor sync 5/5**: 加 §7.x env-side pilot section (Sweet Spot NLWeb-style emit hidden select options) | §21.7 pending decision #2 (env-side pilot) |
| 8 Discussion + Implications (含 sustainability + 4-fold drop-in summary) | ❌ 未写 | end-stage | 全部 data done; **加** IDPI security taxonomy + paper 2 future direction (live web / NLWeb-style env affordance) | §21.6 IDPI security findings + §21.7 paper-2 path |

**Section 1-3 总 prose 3163 words** (paper-ready). Section 4 1725w draft 待 fresh data update. Section 5/6/7/8 待写.

> **2026-05-04 §21 alignment audit**: 8 sections 全 cover 现 §21 9-cell taxonomy 内容, 但 explicit cross-reference 没贯通。advisor sync 5/5 lock 后 paper §1 hook 用 §21.5 prose; codex #11/#13 prose 写作时 cross-reference §21 industry precedents + counter-evidence stack。

### Section 6 Routing — 详细 outline

paper Section 6 不是 Section 7 Generalization 的子部分, 是**独立 contribution**. 内容:

```
6.1 Routing problem formulation
  - 5-mode arms: DOM / SoM / Vision / Phantom-SoM / P-text
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
  - Phantom modes signal AUROC ≥ baseline (5/5 usable, red P-text 0.793 highest)
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
Modes:     DOM / SoM / Vision / Phantom-SoM / P-text = 5 modes
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

### Risk 5: B0 vs B1 reproducibility 不对称 (新增 2026-04-30) ⚠️

**实证 finding** (probe_b37_api_determinism.py 5 calls × T=0+top_p=1.0 + seed=42 forwarded):
- B1 (4B local + `do_sample=False` + `torch.manual_seed`): byte-deterministic by construction
- **B0 (235B proxy API): 5/5 calls produced 5 distinct byte-level outputs** at T=0
- BUT: 5/5 calls selected the **same action** (`click element_id=5`) — decision-level convergent
- Output token count varied 38/45/46/49/49 — model genuinely sampling differently across calls

**Reviewer attack vector**: "你 paper 说 seed=42 reproducible, 但 B0 proxy 不可控, 你 SR 数字哪有 reproducibility?"

**Mitigation** (Section 4 disclosure):
1. Frame B0 vs B1 asymmetry honestly: B1 strict deterministic; B0 decision-level convergent only
2. SR-level conclusions robust (action selection stable across replicates)
3. Token-level metrics (string_match exact-match, thought-text similarity) acknowledged residual variance
4. Empirical evidence anchored in `docs/analysis/cross_sites/probe_b37_api_determinism.md` (5 raw outputs + Section 4 disclosure paragraph drafted)

**Cost saved by NOT pursuing this further**: replication study at full 14-cell scale would cost ~$60-200; instead cheap 5-call probe ($0.005) gave us decisive characterization. Paper Section 4 disclosure paragraph is the deliverable.

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
- Verbalized signals AUROC 0.701-0.793 (red P-text 0.793 是 5-mode 最高)
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

`scripts/analysis/figures/fig3_regional_carbon.py`:
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

`docs/analysis/paper_drafts/paper.bib` (16 entries, 待 codex #10 expand to ~38)

### Codex analyses (`docs/analysis/phantom_paper/`)

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

### A. Codex prose tasks (跟踪 in next_steps §4 codex queue)

- [ ] codex #10 axis 2/3 literature deep research → expand paper.bib 16→~38 (~Wed)
- [ ] codex #11 Section 4 fresh-data prose update (~Wed)
- [ ] codex #13 Section 5 prose 写 (3-axis hierarchical + lit cite, ~Thu)
- [ ] codex #16 Section 6 Routing prose (Week 5-6, after Tier 1+2 prototype)
- [ ] codex #17 Section 7 Generalization 草稿 (~Week 6-7, after WA + Claude done)
- [ ] codex #18 Section 8 Discussion 草稿 (paper end-stage, 含 sustainability + lat 4× finding)
- [ ] codex #19 二次 deep research (Section 6/7/8 + 全 paper revisit, paper 终稿前 Week 8+)

### B. Data analysis pipeline (Python scripts, not codex tokens)

- [x] **统计显著性测试** ✅ done 04-28 — `fig0c_drop_one_oracle.py` 加 `bootstrap_drop_one_ci()` (1000 resample × 4 panel)，error bars + `fig0c_drop_one_bootstrap_ci.csv` 12 rows
  - Section 4 reviewer-grade rigor; codex #11 prose 可直接引用 95% CI
  - Pending: paired permutation test for cross-mode SR delta (lower priority)
- [x] **AUROC aggregation table** ✅ done 04-28 — `scripts/analysis/aggregate_routing_auroc.py` (~110 行)
  - Outputs: `results/phantom_paper/auroc_cross_condition.csv` (188 rows × 5 modes × 4 cells) + `_summary.md` (top-1 per cell, Section 6 claim 证据)
  - Section 6 "AUROC ≥ baseline" claim 部分支持: B0 red P-text 0.793 highest; B0 cls P-text 0.737 ≥ SoM 0.709 baseline; B1 cells 待 chain done
- [x] **Phantom routing lift** ✅ done 04-29 — `scripts/analysis/aggregate_phantom_lift.py` (~180 行)
  - Outputs: `results/phantom_paper/phantom_lift.{csv,md}` — 3-mode → 5-mode oracle ceiling lift + bootstrap CI + per-phantom decomposition
  - **Paper Section 1/4 hook 主 evidence**: B0 cls **+4.70pp [2.14, 7.69]** ✅, B0 red **+5.24pp [2.38, 8.11]** ✅ (CI 排除 0)
  - Decomposition: P-text adds 8 tasks / P-SoM adds 6-7; each phantom 有独家 + overlap 部分 → keep both phantoms in paper
  - B1 cells 待 chain done 自动 cover (script 检测 ep count, ≥50 ep 触发)
- [ ] **Multi-metric Pareto pipeline** (cost + latency + carbon)
  - Section 8 sustainability prose 前置；fig9 已有 carbon B1 only, 需 cost/latency 三向 join
  - Output: 3-panel Pareto figure + per-condition multi-metric table
  - Implementation: extend `scripts/analysis/figures/fig3d_cost_sr_frontier.py`
- [ ] **每 task 特征提取** (Section 6 Tier 1 oracle router 前置)
  - Features: TF-IDF (task instruction) + has_ref_image binary + has_finish_string_match binary + site / category metadata
  - Output: `task_features.parquet` per benchmark
  - Implementation: `scripts/analysis/extract_task_features.py`
- [ ] **B0 token-based carbon estimator** (Section 8 Tier 3 sustainability)
  - 当前 §3.6 marked "optional"; 需 minimum implementation (eu-west-2 default region, token × carbon factor)
  - Source: `condition_summary_v2.json` 含 token counts per condition
  - Implementation: ~20 行 helper in `p79/experiment/metrics.py`，paper Section 8 引用
- [x] **Multiple-comparison correction + TOST equivalence test** ✅ done 2026-05-03 (T0a) — `aggregate_phantom_lift.py` Bonferroni / Holm-Bonferroni / BH FDR q-value / bootstrap TOST p-value (δ=1.0pp) cols + comparison family declaration block
  - Output: `phantom_lift.md` augmented PRIMARY family table + new SECONDARY-family per-arm adjusted subsection
  - Paper rigor: Holm-corrected per pre-registered family (PRIMARY P-SoM gating; SECONDARY P-text/P-prompt exploratory)
- [x] **Cross-cell meta-analysis (DerSimonian-Laird random-effect)** ✅ done 2026-05-03 (T0c) — `aggregate_phantom_meta.py` per arm × all cells with I² heterogeneity + Cochran's Q + τ²
  - Output: `meta_phantom_lift.{md,csv}` (PRIMARY/SECONDARY/TERTIARY family labels)
  - SE_i derived from bootstrap CI as `(CI_hi - CI_lo) / (2 × 1.96)` (normal approx valid at N=210-234)
  - Wired into `make analysis` `_aggregate` target
- [x] **H3 structural test (phantom space 2-axis empirical evidence)** ✅ done 2026-05-03 (T0a) — `aggregate_phantom_lift.py` H3 family with bootstrap CI on |arm ∖ P-SoM| unique-count + per-axis Holm correction
  - Output: `phantom_lift.md` H3 Structural section
  - Tests phantom space is multi-region 2D not collapsed point (paper hook structural claim)
- [x] **Forest plots (per-cell + meta)** ✅ done 2026-05-03 (T0b/T0d) — `fig_forest_drop_one.py` raw 95% CI + Holm-sig marker + TOST band; `fig_meta_forest.py` Hero+Ablation visual hierarchy with weight-sized squares + pooled diamond
- [x] **Phantom space Venn (paper §1 centerpiece)** ✅ done 2026-05-03 (T0d-bis) — `fig_phantom_structure_venn.py` per-cell 3-circle Venn (P-text/P-SoM/P-prompt) showing task-set overlap as visual proof of multi-region structure
- [ ] **Pre-registration document** (T0e, blocks 14-cell rerun launch) — `docs/checkpoints/preregistration.md` (status:draft) 待 advisor sync lock 5 commits + flip to status:locked + git SHA + advisor email witness; OSF DOI optional at paper-time
- [ ] **Hypothesis confirmation matrix viz** (T0f, post-rerun) — `fig_hypothesis_matrix.py` scaffold (~1h post-T0e lock); rerun 数据进来后 fill cell colors

### C. Paper end-stage tasks (Week 8+)

- [ ] Pre-submission checklist (paper_planning §17) execute
- [ ] LaTeX 转换（当前 markdown drafts → LaTeX template per venue）
- [ ] Bib 完整性 check (citations present, format correct)
- [ ] Reproducibility appendix（commit summaries + onboarding instructions）
- [ ] Router Tier 1 prototype (~3 天, baseline + phantom 全 done 后)
- [ ] Router Tier 2 first-step trigger (~7-10 天)
- [ ] Advisor align meeting #1 prep (~Week 3)
- [ ] Advisor align meeting #2 prep (~Week 6-7)
- [ ] paper writing + revisions (~Week 8-12)

---

## §14 Reviewer Attack Anticipation + Pre-Rebuttal

顶刊投稿 reviewer 常见攻击 + 我们的 response (paper integrity hardening):

| Attack | Likely Reviewer Concern | Our Response | Evidence |
|---|---|---|---|
| **Sample size too small** | "VWA cls 234 + red 210 = 444 task, single benchmark" | Final scope = 6 sites × 3 models × 5 modes ≈ 1390 task per condition. Cross-site (cls + red + shopping × VWA + WA), cross-model (Qwen 235B + 4B + Claude Opus 4.7) | §5 Final scope; §3.1 B0 5-mode SR table |
| **Single benchmark family** | "VWA only, no Mind2Web/WebVoyager validation" | + WA (Postmill / Magento / shopping_admin) cross-stack validation. Mind2Web out of scope per advisor align (Plan B) | §7 generalization + paper §6 |
| **Single model family (Qwen)** | "Effect Qwen-specific?" | + Claude Opus 4.7 cross-model after advisor align (~$70). B0 (235B) + B1 (4B) shows capability-dependent shift (+50/+33pp cross-site, §101.九 lazy minimization) | §2 capability layer + cross_site_pattern_consolidation.md |
| **Phantom is just a degraded SoM** | "Why not collapse to DOM if no image?" | Theory C (codex 5821387) verifies prompt knob: cls P-text = Phantom-SoM SR 14.53% but Jaccard 0.447 (task pool 显著 disjoint). Same SR ≠ same routing pool | paper §5; codex `5821387` |
| **Effect size small (drop-one 1.7-3.3pp)** | "Statistically marginal" | (i) Pre-registered Hero (P-SoM) requires pooled magnitude ≥ 1.0pp + TOST equivalence at δ=1.0pp rejected. (ii) P-text/P-prompt are framed as **structural ablation evidence** (low-threshold non-overlap proves phantom space is multi-region 2D), NOT as deployment routing arms — so deployment magnitude bar doesn't apply to them. (iii) Holm-Bonferroni multi-comparison correction applied per pre-registered family. | §1 paper hook (data-conditional R1-R5) + `preregistration.md` H1+H3 + `phantom_lift.md` Holm/TOST cols |
| **Post-hoc hypothesis cherry-picking** ⭐ NEW pre-rebuttal | "你 H-list 是数据进来后 fit 的" | Pre-registration locked before 14-cell rerun via Git SHA + advisor email witness + OSF DOI (paper-time public). Multi-comparison family declared explicitly. Exploratory analyses (H4/H5/H6) marked "post-hoc" in paper prose with explicit non-gating disclosure. Framing decision rule R1-R5 maps data outcome to hook framing transparently — reviewer can verify framing-to-data mapping is deterministic, not chosen post-hoc. | `docs/checkpoints/preregistration.md` + `EVIDENCE_LAYER_AUDIT.md` §2 |
| **Latency claim cherry-picked** | "Just one P95 measurement" | §100 SoM probe ground truth (5 imgs × 3 mode × 2 model = 30 cells measured). cls SoM 74s vs Phantom 18s p95 = 4× slower. Across all conditions consistent | §11 + 实验笔记 §100 |
| **Carbon estimation rough** | "B0 carbon NaN, only B1 measured" | Transparent disclose: B1 NVML measured directly, B0 (proxy API) 远端 GPU 不可测 (per Strubell 2019 / Patterson 2021 estimation acknowledged). fig9 regional sensitivity 用 B1 measured + 45 region intensity table | §11 + fig9 footnote |
| **Router contribution toy** | "Tier 1 oracle is overfit" | Tier 1 train/test split, baseline 对比 (random, best-single-mode, rule-based, oracle, learned). Tier 2 first-step trigger no test leakage | §8 + Section 6 outline §4.6 |
| **No production deployment** | "Drop-in claim hypothetical" | 4-fold drop-in property: code-level verified (`som.py::_extract_text_marks` line 24 regex); routing signal AUROC ≥ baseline (5/5 `overall_usable=True`); 实证 cost+latency+CO2 measured | §1 + §3 finding #5 #9 |
| **Watchdog detection unreliable** | "FPC false alarm undermines paper-grade" | Site-specific audit: cls (real auth issue + auto-clean + 重跑 done), red (0 events), shopping (FPC false alarm fixed). Watchdog auto-clean protocol delete contaminated + runner resume → 0% wasted task. paper-grade 100% pure verified | §18 + 实验笔记 §104 |
| **Mechanism not novel** | "Each axis has prior literature" | Contribution = systematic decomposition + web-agent multi-step setting + drop-in deployment claim. NOT new LLM mechanism. Paper §5 framing 已 acknowledge | §2 paper contribution position |
| **Overfit to VWA visual specifics** | "Effect won't generalize to WA" | §103 falsifiable prediction: WA Phantom-SoM 5-mode oracle gain. WA pilot ≤50 task verify Jaccard ≤0.5 universal vs >0.7 VWA-specific | §103 generalization prediction; pending data |

**Pre-rebuttal strategy**:
- Section 4-5 prose 写时 inline cite this table (proactive defense)
- Section 7 Generalization 必须 explicit address WA + Claude (跨 stack + 跨 model)
- Section 8 Discussion 4.4 limitations 提前 acknowledge known weaknesses

---

## §15 Prior Work Comparison Table

paper Section 2 必备 explicit table (review 加分项):

| Aspect | Yang 2023 SoM (NeurIPS) | VWA Koh 2024 (ICLR) | SeeAct Zheng 2024 (ICML) | FocusAgent Kerboua 2025 (EMNLP) | RouteLLM Ong 2024 (ICML) | **Ours (Phantom-SoM)** |
|---|---|---|---|---|---|---|
| **Marks-text isolation** | ❌ bundled with image | ❌ bundled | ❌ bundled | n/a | n/a | ✅ Phantom-SoM ⭐ |
| **Routing arms** | 1 (single SoM) | 1 (per mode) | 1 (single SoM) | 1 (text prune) | model-level routing | **5-mode** (DOM/SoM/Vision/Phantom-SoM/P-text) ⭐ |
| **Cost-aware Pareto** | ❌ | ❌ | ❌ | ✅ token cost | ✅ model cost | ✅ **multi-metric** (cost+latency+carbon) ⭐ |
| **Cross-site validation** | 4 task domains | 3 sites (cls+red+shop) | 1 site | 2 sites | n/a | **6 sites** (VWA+WA) ⭐ |
| **Cross-model** | 4 models (multimodal) | 6 models (api+local) | 4 models | 1-2 | many (text-only LLM) | 3 models (Qwen 235B+4B + Claude Opus) |
| **Mechanism analysis** | ❌ effect-only | ❌ partial | ❌ baseline | partial (text size effect) | ❌ effect-only | ✅ **3-axis × 8-channel × bidirectional** ⭐ |
| **Drop-in deployment** | ❌ | ❌ | ❌ | partial | partial | ✅ **4-fold property** (cost/latency/signal/oracle) ⭐ |
| **Carbon report** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ **Differentiator** ⭐ |
| **Failure mode taxonomy** | none | 3 categories | none | none | none | **9 categories** + 8-channel image (codex diags) |
| **Sample size** | varies | 910 task total | 50 task subset | 3 sites partial | many | **1390 task** per condition (final scope) |

**Closest prior pairing**: FocusAgent (text 压缩, hierarchy 保持) + Yang 2023 SoM (visual marks). 本工作 = unprecedented synthesis + drop-in deployment claim + multi-metric Pareto + green AI differentiator.

详 deep research: `docs/literature/The Novelty and Efficacy of Set-of-Mark Text as an Independent Representation Routing Arm in Web Agents.md` (5-dimension gap confirmation, §103).

---

## §16 Authorship + Advisor Roles + First-Paper Strategy

### 毕设 paper authorship plan (TBD with advisor align meeting #1)

```
First author: jiaming (毕设 student, primary work)
Co-supervisor: Zekun (Holistic AI, industry collaboration)
Advisor: Maria Perez Ortiz (UCL, AI4SD program director)

Tentative authorship order: jiaming, Zekun, Maria
(final order pending advisor align meeting #1)
```

### Advisor / collaborator roles

| Person | Role | Paper contribution |
|---|---|---|
| **jiaming** | Implementation + experiments + writing + first-paper learning | Main author, all sections, codex orchestration, paper-grade execution |
| **Zekun** | Industry collab + drop-in deployment 视角 + MLSys positioning | Section 8 Discussion deployment angle + venue strategy + reference review |
| **Maria** | Theoretical guidance + AI4SD framing + sustainability + conference network | Section 1/2 background + Section 8 sustainability + referrer pipeline (NeurIPS/Climate Change AI workshop) |

### Personal context (毕设 backdrop, 本 paper 是 first paper)

- 西安交大 undergrad → UCL AI4SD master/PhD transition
- First paper, 经历: paper trajectory 从 "magical noise 怀疑" 到 "4-fold drop-in deployment claim"
- 多次 critique-driven theory refinement (4 rounds: prompt-only / visual-hijack-only / image-over-text / SoM density) — paper integrity discipline 体现
- Holistic AI industry collab 是 publication track signal (industry endorsement)

### First-paper psychology + strategic advice

```
Realistic outcome distribution (per §5 顶刊概率):
  Round 1 (MLSys 2027 ~9月): 75-85% accept (推荐 first paper friendly)
  Round 2 (ACL/EMNLP if rejected): 50-65%
  Round 3 (NeurIPS/ICLR main if still rejected): 45-60%
  Round 4 保底 (TMLR rolling): 75-85%
  
出版概率 cascade 累积: ~99% (基本 lock paper 出版)

Key first-paper considerations:
- Don't put NeurIPS/ICLR as round 1 (lottery + first-paper baggage if rejected)
- MLSys 推荐: drop-in framing 完美 fit + first-paper friendly review
- Maria's referrer pipeline (NeurIPS workshop / Climate Change AI / AI4SD venue) 是 strategic leverage
- Holistic AI industry endorsement → industry track 友好
- Rejection 是 norm, 不要把 rejection 等同于 paper 不行
```

### Acknowledgments draft (paper end-stage)

```
预 draft (Section 8 Acknowledgments):
- Compute: DGX Spark (UCL AI4SD) + remote VWA Docker (Tailscale) + Myriad GPU pending
- Data: VWA + WA benchmarks (open source, properly cited)
- API: Qwen3-VL-235B-A22B (proxy via internal infra), Claude Opus 4.7 (advisor budget)
- Discussions: advisor + co-supervisor + UCL AI4SD group
- COI: Holistic AI industry collab acknowledged
```

---

## §17 Pre-Submission Checklist (~Week 10-12 paper 终稿前)

### Content completeness

- [ ] All 8 sections prose done (Section 1-3 ✅ done, Section 4-8 待 codex)
- [ ] paper.bib expanded to ~38 entries (待 codex #10 deep research)
- [ ] All figures FRESH with latest data + paper-grade captions
- [ ] Section 2 含 prior work comparison table (paper_planning §15)
- [ ] Section 4-5 prose 含 reviewer attack pre-rebuttal (paper_planning §14)
- [ ] Section 5 mechanism 含 §100 SoM probe ground truth + 14 case studies + Tong 2024 cite
- [ ] Section 6 Routing 含 Tier 1+2 implementation + 4-fig stack
- [ ] Section 7 Generalization 含 cross-site (shopping + WA) + cross-model (Claude) data
- [ ] Section 8 Discussion 4 sub-sections (drop-in summary / mechanism / sustainability / limitations)
- [ ] Negative results explicit listed (paper integrity)
- [ ] Limitations section honest (no over-claim)

### Format / Style

- [ ] Page count check (NeurIPS/ICLR 9 page main + unlimited supp; MLSys 12-15 page)
- [ ] Reference style (BibTeX validation, all 38+ entries cite-resolved)
- [ ] Figure resolution 300 DPI (paper print)
- [ ] Code anonymized for review (if double-blind)
- [ ] Supplementary materials packed (data CSVs, configs, analysis dirs)
- [ ] Captions self-contained (figure 不依赖 text)

### Reproducibility

- [ ] Code release path: github / zenodo decided
- [ ] Data: VWA tasks + run results + figures input data (per condition_summary_v2.json)
- [ ] Configs: configs/exp_v2_*.yaml all referenced
- [ ] Reproducibility statement: `make figures` / `make analyze` workflow documented
- [ ] Replication recipe in supplementary

### Authorship + Submission

- [ ] Author order finalized (advisor align #1 + #2)
- [ ] Acknowledgments (compute resources + advisor + collaborators)
- [ ] Conflict of interest declaration (Holistic AI industry collab)
- [ ] Venue-specific format (MLSys vs NeurIPS template chosen)
- [ ] cover letter / abstract polished

### Pre-rebuttal preparedness

- [ ] §14 reviewer attack table integrated to prose (proactive)
- [ ] §15 prior work table integrated to Section 2
- [ ] Limitations explicit (Section 8.4)
- [ ] Cost/budget transparency (acknowledgments)

---

## §18 Watchdog Protocol + Paper-Grade Execution Discipline

> 这部分内容 paper 写时可作为 supplementary "paper-grade execution discipline"
> 引用; 也是 reviewer 信任度的 evidence。

### 6-layer Defense in Depth (per `experiment_watchdog.py`)

```
1. Detection: per-task DOM session check (5000 char window, _check_session_health)
   - Site-specific tab guard (cross-site task skip)
   - Logout / Sign In link regex
2. Alert: streak ≥3 → ntfy notification + ALERT log
3. Refresh: real Playwright sign-in subprocess (auth_refresh.py)
   - Per-site account credentials
   - host-resolver-rules MAP metis → IP (legacy)
   - Verify post-login URL ≠ login_path before storage_state write
4. Cleanup: delete contaminated episodes (auto-clean on login restored)
   - Delete summary_v2.json + steps_v2.jsonl
   - rmtree artifacts/{site}_task_{tid}/
   - Purge digest records
   - Remove from seen_keys
5. Resume: runner re-run with fresh logged-in storage_state
6. Verify: post-cleanup mtime + DOM check (paper-grade integrity audit)
   - State file persists across watchdog restart
```

### Magento history (3 复发 + final fix)

```
2026-04-X (initial): cookie domain split (PHPSESSID under IP, form_key under metis)
                     → fix `7150db8` (quark side base_url 改 IP)
2026-04-27: docker reset 后 base_url 退回 metis
                     → fix `f9cbebf` (DGX defensive curl + quark side scripts)
                     → 3-layer 持久化 (magento_baseurl_fix.sh + start_vwa_docker.sh
                        hook + reset_shopping.sh remove hardcode localhost)
2026-04-28: PowerShell reset chain 没集成 base_url fix → docker reset 仍 invalidate
                     → fix: PowerShell `C:\vwa\reset_vwa.ps1` 加 Configure-MagentoBaseUrl
                       (docker exec config:set + cache:flush, shopping 7770 +
                        shopping_admin 7780 都覆盖)
2026-04-28: Magento Full Page Cache (FPC) homepage cache guest page → false alarm
                     → fix: quark side `bin/magento cache:disable full_page` +
                       PowerShell hook 持久化 (reset 后 auto-disable FPC)
```

### Paper-grade clean re-run protocol

```
Before each new condition:
  1. reset_vwa_sites.sh → DGX SSH quark PowerShell
     PowerShell: docker stop + start vwa-{site} container
     PowerShell: Configure-MagentoBaseUrl (config:set + cache:flush + cache:disable full_page)
     PowerShell: site-specific health check (HTTP 200)
  2. DGX defensive curl: verify redirect ≠ metis (commit f9cbebf)
  3. Refresh storage_state (auth_refresh.py if streak ≥3)
  4. Launch runner with --resume flag
  
During run:
  - Watchdog poll 30s, NOT LOGGED IN streak detection
  - Auto-clean on streak ≥3 + login restored
  - Runner resume picks up missing tasks → fresh re-run
  
Post run:
  - rederive episode summaries (re-compute adjusted_success per FP rules)
  - Auto figures regen (`make figures`)
  - Cross-rep / reason_diag / cross_run analysis (per analyze pipeline)
```

### Paper integrity 论证 (Section 4 / supplementary)

- **0% wasted task data** (Day 2 audit verified): all NOT LOGGED IN events auto-cleaned + 重跑 done. Final episode summaries 全 fresh logged-in.
- **Site-specific noise sources**:
  - cls (OSClass): real auth issue, watchdog auto-clean + 重跑 (~2% early tasks affected per condition)
  - red (Postmill): 0 NOT LOGGED IN events
  - shopping (Magento): FPC false alarm fixed, B0 NEW launch with FPC disabled
- **Cross-mode comparison preserved**: 5 modes 受同一 protocol, drop-one oracle / Jaccard / cost-SR Pareto 都不被 ~2% noise bias
- **Paper-grade discipline**: self-healing data pipeline, 6-layer defense in depth → reviewer 信任 paper data integrity

---

## §19 Decision Log (paper-strategic decisions audit trail)

| Date | Decision | Rationale | Status |
|---|---|---|---|
| 2026-04-27 | Final scope: 6 sites × 3 models × 5 modes + deployed router + multi-metric + green AI | NeurIPS/顶刊 viable scope (paper_planning §5) | ✅ in plan |
| 2026-04-27 | P-text scope 缩减 18→5 cells (mechanism only) | P-text 是 ablation 不是 routing arm 候选 | ✅ in plan |
| 2026-04-27 | Future paper 2 转向 Phase 3 modules (router 整合 paper 1) | 毕设决策, paper 1 含完整 contribution | ✅ in plan |
| 2026-04-27 | First paper 投稿 cascade: round 1 → MLSys (不 NeurIPS) | first-paper friendly + drop-in framing 完美 fit | ✅ in plan |
| 2026-04-27 | Paper hook 升级到 "drop-in deployment intervention" | Phantom-SoM cost ≈ DOM (regex filter), 4-fold property | ✅ commits 48db047 + ef29add |
| 2026-04-28 | B1 shopping DOM 466 ep clear+rerun (paper-grade 协议一致性) | pre-Magento-bug 跑期间, cookie domain split risk | ⏳ 等 Myriad GPU |
| 2026-04-28 | Magento FPC disabled (server-wide) | FPC homepage cache guest false alarm + persistent fix | ✅ done |
| 2026-04-28 | Theory C: prompt as task-conditional decision prior (NOT commit-only) | codex `5821387` Jaccard 0.45-0.54 disjoint task pool | ✅ paper_planning §2 |
| 2026-04-28 | Image axis 8-channel taxonomy (NOT visual-hijack only) | codex `7106d2e` 4 helping + 4 harming, false visual confidence MAIN red 60% | ✅ paper_planning §2 |
| 2026-04-28 | Bidirectional modality framing (image-over-text vs text-over-vision) | user Q3 critique + Tong 2024 "Eyes wide shut" anchor | ✅ paper_planning §2 |
| 2026-04-28 | 4-doc structure (next_steps + paper_planning + drafts + 笔记) | original 1102-line next_steps too dense, separation of concerns | ✅ commit 97cc4ac |
| 2026-04-28 | 8 sections paper structure (含 Section 6 Routing 独立) | router 是 paper independent contribution, not Section 7 sub | ✅ commit 4ca9f66 |
| 2026-05-01 | Paper hook reframe: "P-SoM is hidden 4th routing arm" → "**phantom routing space (3 arms)** sharing 4-fold drop-in" | B0 reddit 6-mode oracle +7.14pp [3.81, 10.48] sig + 3 arms drop-one 全 sig (P-text +3.81 / P-SoM +3.33 / P-prompt +2.86) | ⏸️ provisional pending cls 6-mode + B1 phantom 数据 confirm (advisor sync Q3) |
| 2026-05-01 | Phantom space boundary 重新论证: "no annotated image" 而非 "matched parsing" | (a)(b) 4-fold drop-in by definition derive from no-image; P-prompt mismatched-parsing 论证 falsified by +2.86pp drop-one sig | ✅ paper_planning §2 + 笔记 §108.2 |
| 2026-05-01 | M1/M2 mechanism activation 2x2 framework (新 §2 organizing principle) | prompt textual coupling ≠ mechanism activation coupling; LLM internal state 层 2 axes orthogonal (Image-mirage / Flat-list); P-SoM 是 cube center compound state emergent | ✅ paper_planning §2 + 笔记 §108.4 |
| 2026-05-01 | Architectural completeness argument (Approach 2): phantom space 内 M1/M2 by design exhaustive (deductive) | 不依赖 finite data verify; phantom space 锁 image=✗ → 只 vary 2 input dim → mechanism 必映射到这 2 dim | ✅ paper_planning §2 Zoom 1 + 笔记 §108.5 |
| 2026-05-01 | Evidence vs Explanation layer 严格分离 (paper conceptual structure) | Evidence = 2D organize (4 测量 × 4 cross-X); Explanation = 1D zoom scale (Zoom 1-4); 不混 | ✅ paper_planning §3 顶部 + §2 retract list + 笔记 §108.6 |
| 2026-05-01 | 4 zoom scale of explanation layer | Zoom 1 (架构) / Zoom 2 (behavioral M1/M2) / Zoom 3 (named phenomena Mirage/Scaffold/Sclar) / Zoom 4 (model-internal Cross-modal flow/SteerMoE) | ✅ paper_planning §2 + 笔记 §108.6 |
| 2026-05-01 | (a)(c) prompt decomposition 作 paper §2 axis **retract** | (a) 可独立 (SoM-prompt minus image-mention), (c) 不可独立 (变 vision-mode prompt); prompt structure 跟 mechanism activation 是不同抽象层 | ✅ retract list in §2 + 笔记 §108.3 |
| 2026-05-01 | 8-corner 2x2x2 cube + 6-corner asymmetric grid 实验设计 **retract** | M1/M2 mechanism activation framework 已 saturate phantom space description; ablation cells (p(a-pure)) 价值降级到 nice-to-have for Zoom 2 sub-mechanism granularity | ✅ retract list + 笔记 §108.3 |
| 2026-05-01 | SteerMoE (Fayyaz 2026 ICLR) 作 Zoom 4 lit anchor + paper §8 future work | 学长 2026-05-01 发; B0 = Qwen3-VL-235B-A22B 是 SteerMoE 实验模型 architectural cousin; methodology template 但 paper 不 self-probe (proxy API + budget barrier) | ✅ paper_planning §2 Zoom 4 + 笔记 §108.9 |
| 2026-05-01 | Early-stop mechanism design decision: lean Option A (full cancel), pending advisor align | early-stop 是 cross-dimension systemic confound (不止 micro layer); Option A +$1300 全 cancel / B keep / C hybrid +$200 | ⏸️ advisor sync Q1 重写, lean A pending | 
| 2026-05-01 | 别扭 framework refinement (provisional) — reverse-explanation layer + capability-modulated discovery | (a) Cross-cell empirical validation 4 cells: B0 4/4 别扭 predictions confirmed, B1 cls prediction 4 reversed (small VLM single 别扭 negative aggregate); (b) drop-one direction reversal cross-capability (B0 P-text > P-SoM, B1 cls P-SoM > P-text) → 别扭 + Lazy Minimization 联合 framework; (c) compound 别扭 (P-prompt) 实证 negative aggregate (B0 reddit raw 10.48 < DOM 11.43) but positive complementarity (drop-one +2.86pp) — double-edged property | ⏸️ provisional, pending 14-cell rerun statistical commit + B1 reddit phantom 数据 |
| 2026-05-03 | Pre-registration framework reframe: "3-arm a-priori co-equal" → **Hero (P-SoM) + Structural ablation (P-text/P-prompt non-overlap) + Framing decision rule R1-R5** | "3-arm co-equal" was retrofitting emergent discovery as a-priori hypothesis (epistemically dishonest); new framework pre-registers Hero strict + Structural low-threshold non-overlap + data-conditional framing rule. P-text/P-prompt findings emerged from data, not predicted; their proper role is structural ablation evidence (proving phantom space is multi-region 2D structure, not collapsed point). | ✅ `docs/checkpoints/preregistration.md` (status:draft) + `docs/reference/EVIDENCE_LAYER_AUDIT.md` §2 + paper_planning §1 reframe note |
| 2026-05-03 | H3 structural test = bootstrap CI on \|arm ∖ P-SoM\| unique-count > 0, K_h3=0.67, ≥2 task floor | Structural claim only requires non-emptiness of axis non-overlap, not directional dominance. McNemar tests asymmetry which is wrong test for H3. Bootstrap CI > 0 is correct. | ✅ `aggregate_phantom_lift.py` H3 family + `phantom_lift.md` H3 section |
| 2026-05-03 | Pre-registration commits locked: K_h1=0.75 / K_h3=0.67 / TOST δ=1.0pp / Phase A only main + Appendix D archived / Witness=Git+advisor email + OSF DOI | All 5 commits drafted in `preregistration.md` (status:draft); pending advisor sync to flip status:locked + record git SHA + advisor witness. | ⏸️ pending advisor sync |
| 2026-05-03 | Evidence layer + visualization audit infra (T0a-T0d done) | `aggregate_phantom_lift.py` Bonferroni/Holm/BH/TOST + H3 structural test cols; `aggregate_phantom_meta.py` DerSimonian-Laird random-effect; `fig_forest_drop_one.py` per-cell forest with Holm-sig markers; `fig_meta_forest.py` Hero+Ablation visual hierarchy; `fig_phantom_structure_venn.py` paper §1 centerpiece Venn; `make analysis [FAST=1]` end-to-end wired. | ✅ `docs/reference/EVIDENCE_LAYER_AUDIT.md` §3 T0 4/6 done |

---

## §20 Meta — Doc Update Workflow (when X happens, update which docs)

> Moved from `next_steps.md §10` 2026-05-02 (next_steps 严格 live + future, meta-process 归 paper_planning).

### A. 新 condition 数据 (e.g. B1 phantom_som cls done)

```
✅ 实验笔记: append chronicle entry (§104+ daily)
✅ next_steps §3.1 active: mark done (move to status.base via _status/section*.md frontmatter)
✅ Run `make analysis` (one-shot all-in-one, ~5-10 min):
   - per-run pipeline (rederive + reason-diag + cross-rep + confidence) on 8 paper-grade VWA runs
   - B0 vs B1 site comparison (cls + red → b0_vs_b1_<site>/)
   - cross-condition aggregations: aggregate-cross-site + summary-collect + routing-auroc
   - 9 figures (含 fig2 bootstrap CI)
   ↳ Quick: `make figures` (~10s, 仅 fig regen)
   ↳ Debug: 单独 `make analysis-per-run` / `compare-b0-b1-all` / `aggregate-cross-site` / `summary-collect` / `routing-auroc`
   ↳ 输出 path: `results/phantom_paper/{auroc_cross_condition.*, cross_site/, run_summary_collect.json, figures/*.png}`
   ↳ NOT 自动: GLM digest sidecar (watchdog) + 9 narrow ad-hoc diagnostics (selflink_loop / vision_coordinate / search_over_browse / diag_pattern_match)
🟡 next_steps §0 current state: update if Critical path A 进度变化
🟡 paper_planning §3 finding 列表: if new paper-grade finding emerges
🟡 paper_planning §4 paper section status: if evidence quality 变化
❌ paper drafts: 不动 (等 codex prose update batch)
```

### B. 新 figure (e.g. fig10 cumulative SR vs budget)

```
✅ next_steps §7 figures list: add new path
✅ paper_planning §10 visualization plan: update 4-fig stack
✅ paper_planning §12 figures inventory: add row
🟡 paper_planning §3 finding: if figure reveals new finding
✅ Makefile figures target: add fig10_*.py
```

### B'. 新 cross-condition aggregator (e.g. paired permutation table)

```
✅ scripts/analysis/aggregate_*.py: implement
✅ Makefile: 新 PHONY target + chain into analyze-paper
✅ paper_planning §13.B: mark done with output path
✅ next_steps §4 codex queue / pending scripts: mark ✅
```

### C. 新 codex analysis (e.g. trajectory diff diag)

```
✅ paper_planning §12 codex analyses inventory: add row
✅ next_steps §7 references: add path
🟡 paper_planning §3 finding 列表: if new finding from analysis
🟡 paper_planning §2 theory framework: if mechanism discovery (e.g. axis refinement)
🟡 paper_planning §19 decision log: if framing decision made
✅ next_steps §4 codex queue: mark done
```

### D. 新 paper drafts (e.g. Section 5 prose done by codex)

```
✅ next_steps §2 paper section status: status update (drafted) — via _status/section*.md frontmatter
✅ paper_planning §4 paper section status: same
🟡 paper_planning §2/§3 strategy notes: shrink (move to drafts now in prose)
✅ paper_planning §17 pre-submission checklist: tick off content completeness
❌ next_steps §0: 不变 (除非 strategic shift)
```

### E. 新 decision (e.g. advisor align meeting #1 outcome)

```
✅ paper_planning §19 decision log: append timestamped row
✅ paper_planning §9 advisor align checklist: tick off items
✅ ADVISOR_SYNC §2 framing decisions: status open → discussed → decided
✅ 实验笔记: append decision entry with rationale
🟡 paper_planning §5/§6/§7 (final scope / risks / cascade): if scope decision changes
🟡 paper_planning §16 authorship: if authorship order finalized
```

### F. 新 infra fix (e.g. another bug fix or watchdog upgrade)

```
✅ next_steps §5 open issues: add or move to resolved
✅ 实验笔记: append technical chronicle entry
🟡 paper_planning §18 watchdog protocol + execution discipline: update 6-layer description
🟡 paper_planning §14 reviewer attack: if fix addresses an attack scenario
```

### G. 新 finding (e.g. unexpected mechanism observation)

```
✅ paper_planning §3 finding 列表: add new entry
✅ 实验笔记: append finding with date + evidence
🟡 paper_planning §2 theory framework: if framework refinement needed
🟡 paper_planning §19 decision log: if framing decision triggered
🟡 next_steps §0: if changes paper hook
```

### H. 新 reviewer attack scenario

```
✅ paper_planning §14 reviewer attack: add row
✅ paper drafts (when writing prose): proactive defense in Section 4-5
```

### I. 新 paper section prose done

```
✅ next_steps §2 paper section status: status drafted (via _status/section*.md frontmatter)
✅ paper_planning §4 same
✅ paper_planning §13 pending TODO: tick off
🟡 paper_planning §3 finding: shrink (now in prose) or expand (new finding from prose writing)
🟡 paper_planning §17 pre-submission checklist: tick off content items
```

### General principle

- **Daily** → next_steps.md (live state, codex queue, open issues)
- **Weekly** → paper_planning.md (strategy notebook, when finding/decision emerges)
- **Append-only** → 实验笔记.md (chronicle, never overwrite, append §)
- **Stable until prose write** → paper drafts (only update when Section X prose batch written)
- **Per-meeting** → ADVISOR_SYNC.md (sync prep + post-meeting decision register)

### Quick mental check before update

```
What changed? → 找对应类型 (A-I above)
Mark current status? → next_steps + _status/*.md frontmatter
Add new strategic finding? → paper_planning
Record what happened (history)? → 实验笔记 append §
Modify final paper text? → paper drafts (only when prose batch writing)
Advisor decision context? → ADVISOR_SYNC
```

---

## §21 Environment-Agent Intervention Taxonomy (整合 2026-05-04, 笔记 §1-§108 audit)

> **目的**: 整合分散在笔记里的 environment / agent intervention 类型 work, 给 paper 一个**统一的 contribution scope view**。学长 5/3 push "两头出发" 的 framing 在这里 explicit 化 — 我们一直在做 dual-track work, 只是没显性 frame。
>
> **Scope 不分 paper 1 vs paper 2** — 这一节是 **inventory**, 列"做了的 + 想做的"。具体哪条进 paper 1 / paper 2 / future 后续 advisor sync 决定。
>
> **Source of truth**: 笔记 §X 是历史 chronicle, 这一节是 cross-§ taxonomy view; 任何 § 修改/新增, 这一节同步 update.

### §21.1 Framework — 3 × 3 Taxonomy

**3 个 intervention nature** × **3 个 intervention layer** = 9 cells:

#### Spectrum dimension (intervention 性质)

| Spectrum | 含义 | 跟 Avenir Web 对比 |
|---|---|---|
| **(i) Bug fix** | 恢复 web/agent 已 expose 但 broken 的功能 (web 设计意图本来就有这个 functionality, 但实现/集成 broke 了) | Avenir Web ignore: 把 broken 当 valid failure 不算到 metric 里 |
| **(ii) Affordance synthesis** | 用现有 metadata 补出 agent-readable view (web 已 expose 信息, 但 format 不是 agent-friendly, 加 preprocessing 补出来) | Avenir Web partial: 部分 synthesis (SoM marker) 标准做法, 但 select dropdown / popup 等 ad-hoc |
| **(iii) Channel addition** | 添加 agent 之前 **absent** 的 signal channel (web 根本没 expose 这个信号) | Industry blind spot: Avenir Web / Claude Code / Codex 都 没系统 fix; agent 只能用 model capability cope |

#### Layer dimension (intervention 位置)

| Layer | 含义 | Cost / 修改面 |
|---|---|---|
| **L1 Server-side** | webserver / docker config / web-rendering 层 (改 server 配置, 改页面 HTML, 改 web 标准) | 高: 需要 server 控制权; 但 systematic |
| **L2 Agent-pipeline** | agent script preprocessing / postprocessing layer (改 agent 端 perception 处理, 不动 web) | 中: agent-only 改动; 但 fragile (web 一变就要重写) |
| **L3 Agent-compute** | LLM internal compute substitution (用 prompt / model 内部 compute 替代 explicit affordance) | 低增量: 已有 LLM call, 不加新 stage; 但受 model capability 限制 |

#### 9-Cell 一句话总结表

|  | (i) Bug fix | (ii) Affordance synthesis | (iii) Channel addition |
|---|---|---|---|
| **L1 Server-side** | docker / config 修 (~6 § done)<br/>**Industry**: Atlas StoragePartition isolation | agent-readable web standards (0 done, paper 2 future)<br/>**Industry**: **NLWeb** (Microsoft, May 2025, R.V. Guha — Schema.org → agent JSON channel) ⭐ | agent-specific server channel (0 done, paper 2 / M5 EIP brainstorm)<br/>**Industry**: NLWeb fits here too; A2A protocol (Google, Apr 2025) for agent-to-agent |
| **L2 Agent-pipeline** | runner / wrapper 修 (~28 § done) | script-level affordance overlay (~9 § done, **SoM marker 是 paper-canonical instance**)<br/>**Industry**: **OmniParser-v2** (Microsoft, Feb 2025) — pipeline preprocessing canonical, screenshot → tokenized list | agent-side instrumentation (0 done, **≥7 § identified gaps**) |
| **L3 Agent-compute** | n/a (compute 不能 fix bug) | **Paper hook 4-tier sub-gradient** (重要 distinction):<br/>• Pretraining-time: **Magma** (MS Feb 2025, SoM+ToM grounding 进 weights)<br/>• Offline exploration + RAG retrieve: **AppAgent-v2** (Tencent, agent self-generates text doc, deploy-time RAG)<br/>• Inference-time substitution: **Phantom routing space** ⭐ paper-1 main hook (P-text/P-prompt/P-SoM, no pretraining, no RAG, no offline phase)<br/>• Pure visual VLM: UI-TARS, CogAgent, Magma — opposite end (skip text substitution) | n/a (compute 不能 add absent signal) |

### §21.2 Done items (已做 work, mapped to cells)

#### (i) × L1 — Server-side bug fix (~6 entries)

| § | Item | Affected component |
|---|---|---|
| §39 | PHP `session.gc_maxlifetime=1440s` 太短 → fix 到 86400s (24h) | Postmill PHP config |
| §75 | Magento 302 → `metis.lti.cs.cmu.edu` DGX 不可解析, fix `--host-resolver-rules` 全链路注入 | Magento redirect target |
| §81 | Wikipedia ZIM 2025-08 vs hardcoded 2022-05 mismatch, 三层 fix (URL adapter + tab health check + data cleanup) | Kiwix container ZIM version |
| §103 | Magento auth bug, base_url 改 IP fix dropdown 渲染 | Magento base_url config |
| §86 | Osclass DB 宕机绕过 retry 链, fix mid-episode title 检测 + health check | Osclass server outage detection |
| §82 | 全站 auth refresh enable (跨 L1/L2; queue retry refresh + per-episode 5ep refresh) | Per-site auth lifecycle |

#### (i) × L2 — Agent-pipeline bug fix (~28 entries)

| § Cluster | Examples (笔记 §) | Theme |
|---|---|---|
| **Phase A 4-cluster (paper-grade)** | §107 (commit `3c15cd7`): C1 dispatch / C2 page_changed split / C3 fuzzy cycle hash min_reps=5 / C4 RNG seeding+T=0 | 主 paper-grade rerun trigger |
| Runner reliability | §57 (tab_focus signature) / §58 (shell 孤儿 + stale summary) / §76 (atomic write + retry batch notification) / §85 (P0/P1/P2 batch 9 项) / §90 (keyword_finish 根除) / §87 (evaluator dirty page) / §97 (cross_rep 审计 RU/A/B 类) | Runner / pipeline 稳定性 |
| Action surface | §28 (Vision type bug) / §29 (CDP 焦点) / §30 (np.float32 JSON) / §63 (`<think>` parse_error) / §65 (GLM batch digest) / §67 (parse_error 根治 + scroll 语义化) / §70 (GLM fallback 上线) / §68 (state_change 假阴性/假阳性) | Action parsing / execution |
| Cross-mode symmetry | §44 (B0/B1 三模式 prompt 对称) / §45 (开跑前 9 项) / §46 (B0 ref images 静默丢失) / §42 (11 项审计) | Cross-mode 对称性 fix |
| WA integration | §71 (集成) / §73 (集成审计) / §74 (evaluator 修复) / §83 (E-FP 规则) | WebArena 集成 |
| Watchdog | §84 (跨站 NOT-LOGGED-IN 误判) / §82 (queue retry refresh) / §41 (gallery 幽灵) / §14 (session 丢失) / §12 (cycle 误杀) / §5 (busy:1) | Watchdog / observability fix |
| Misc | §80 (viewport ratio precedence) / §43 (B0 N/A 401) / §40 (B0 prompt 503 retry) / §33 (reddit 7 项) / §34 (ref image processor) / §36 (ref image label) / §59 (BLIP-2 lazy) / §72 (proxy_api dual format) | 杂项 |

#### (ii) × L1 — Server-side affordance synthesis: **0 done** (paper 2 future direction)

**Industry precedent (added 2026-05-04 from DR audit, Round-3 verified)**:

| Industry system | What it does | Date | Relevance to our paper |
|---|---|---|---|
| **NLWeb** (Microsoft, R.V. Guha) | Web emit Schema.org-based agent-readable JSON via `/ask` (conversational) + `/mcp` (Model Context Protocol) endpoints; bypass DOM. **Production deployment at Tripadvisor + Shopify** ✅ Round-3 verified | May 19, 2025 announce ✅ | **Closest industry precedent for our env-side pilot Sweet Spot**. NLWeb 是 deployed standard (不是 proposal), **没 paper 评估 NLWeb-style emission 跟 inference-time L3 substitution (phantom routing) 相对收益** — 我们 env-side pilot 直接 mirror NLWeb spec, 是其首个 controlled comparison |
| **NLWeb literal endpoint spec (Round-3 verified)** | `/ask` returns schema.org JSON: `{"@context":"https://schema.org","@type":"Product","name":"...","offers":{...},"mcp_actionable":{"endpoint":"/mcp/cart/add","parameters":["sku","quantity"]}}` | May 2025 deployed | Sweet Spot env-side pilot for VWA: Postmill/Magento add `/ask`+`/mcp` endpoints emit schema.org JSON, agent reads JSON instead of running AXTree extraction |
| **`agent.txt` / `.well-known/agentbridge.json`** | RFC-stage proposals 给 agent-specific routing hint | 2024-2025 idea stage | Conceptual analog, 没 production deployment |
| **AgentFinder / DataFinder / ModelRouter** (NLWeb subsystems) | NLWeb 内嵌 dynamic RAG indexing + model invocation, 不只是 JSON spec | 2025 deployed | Paper §7 / §8: NLWeb extends beyond passive directory listing — active routing infrastructure |

#### (ii) × L2 — Agent-pipeline affordance synthesis (~9 entries)

| § | Affordance synthesized | Implementation |
|---|---|---|
| §51 | Native `<select>` dropdown action affordance + `[OPTIONS]` AXTree 注入 | `vwa_wrapper.py::_inject_select_options()` |
| §54 | `[OPTIONS: currently selected="X"]` action-effect feedback | `_inject_select_options()` with `selectedOpt.text` |
| §60 | CSS 自定义下拉 `[DROPDOWN OPTIONS]` 注入 | `_inject_css_dropdown_options()` |
| §61 | `[OPTIONS]` cross-mode propagation 到 SoM `[SOM_MARKS]` 列表 | `som.py::_build_som_result()` mark_lines 扫描 |
| §62 | CSS dropdown action surface for Vision `select_option` 路径 | `vwa_wrapper.py` Vision JS path 加 fallback |
| §53 | Native `confirm()` 弹窗自动 accept (synthesizes "auto-accept" affordance) | `page.on("dialog", _on_dialog)` registration |
| §36 | `[Reference image N] This image shows the target item...` prompt augmentation | `runner.py` ref image label |
| §93 | 分析管线 timing/intent/cost/visualization 增强 (eval-side affordance) | `validate_run` 22→27 checks + `reason_diagnostics` 16 cols |
| §38 | VWA 站点 reset infrastructure (一键 docker rm + volume + init_db) | `scripts/maintenance/reset_vwa_sites.sh` |

**(ii) × L2 设计参数 (sub-axis, paper §3 disclosure 必备)**:

| § | Design parameter | Rationale (paper §3) |
|---|---|---|
| §94 | SoM `max_marks` 80→200 | reddit 74% task 被截断, B1 mode 反转部分由此 |
| §96 | SoM 全元素标注 vs 仅可交互 (P79 "Universal SoM" vs VWA "Action-Affordance SoM") | 路由研究控制变量 + Phantom-SoM 消融前提 (DOM↔SoM 切换仅格式差异) |
| §80 | viewport ratio threshold 0.6 | ratio ≥ 0.6 → 元素中心必在 viewport 内 (mathematical guarantee) |
| §101 | P79 vs VWA SoM 范式 fundamental 差异 (universal vs action-affordance) | 1 fundamental + 2 minor (颜色 + placement); paper §3 explicit 论证选 P79 范式 |

#### (ii) × L3 — Agent-compute affordance synthesis (paper-1 main hook)

| § | Item | Status |
|---|---|---|
| §102 | Phantom-SoM 工程实施 (mode + agent prompt + condition config) | ✅ done, all 4 phantom corners ready |
| §103 | Phantom-SoM 4-mode routing arm finding (B0 reddit 6-mode 完整 cell) | ✅ paper §1 main hook, status:provisional pending 14-cell rerun |

**(ii) × L3 内部 4-tier sub-gradient — paper §1 hook 关键 distinction (added 2026-05-04 from DR audit)**:

(ii) × L3 这个 cell 内部不是单一 mechanism, 是**4 种不同的 substitution timing**:

| Sub-tier | When substitution happens | Industry instance | Mechanism |
|---|---|---|---|
| **L3-pretrain** | Training-time (整进 weights) | **Magma** (MS Feb 2025) | SoM + ToM annotations 作 pretraining objectives, vision-text grounding 内化进 model weights |
| **L3-RAG-offline** | Offline exploration → deployment-time retrieve | **AppAgent-v2** (Tencent) | Exploration phase agent **自己** explore 写 text document; deployment phase 用 RAG retrieve over self-generated doc |
| **L3-inference (我们)** ⭐ | Inference-time, no offline, no retrieval | **Phantom routing space** (paper-1 hook) | Agent reads `[SOM_MARKS]` directly from observation processing — no pretraining, no RAG, no offline exploration phase |
| **L3-pure-visual** | (opposite end — skip text substitution entirely) | UI-TARS, CogAgent, OperatorCUA | Pure visual VLM, 完全 skip textual abstraction — coordinate grounding 直接进 weights |

**为什么 L3-inference 是真正没 named precedent 的 cell**:
- L3-pretrain (Magma): 需要 retrain model with SoM/ToM data
- L3-RAG (AppAgent-v2): 需要 offline exploration phase 跑 agent 写 doc
- L3-pure-visual (UI-TARS): 不做 text substitution
- L3-inference (我们): **不需 retrain, 不需 RAG, 不需 offline phase, 不需 special model — pure prompt engineering 在 deployment time 即可。也是 lowest-friction substitution option**

**Paper 1 official contribution scope = L3-inference cell (substitution gradient 的 unfilled niche)**: testing how much (ii) × L2 affordance synthesis (industry instance: OmniParser-v2 pipeline preprocessing) can be substituted by L3-inference compute substitution at deployment time without offline / RAG / pretraining cooperation.

**Industry precedent (added 2026-05-04 from DR audit, Round-3 verified specifics)**:

| Industry system | Cell it occupies | Year | Relevance to our paper |
|---|---|---|---|
| **OmniParser-v2** (Microsoft, Yu/Yang/Wan/Bai) | (ii) × L2 — pipeline preprocessing | Feb 2025 (arXiv:2502.16161) ✅ verified | Canonical instance of L2 affordance synthesis. **MoE token-router shared decoder + Two-Stage Structured-Points-of-Thought (SPOT) prompting**. Literal output token format (Round-3 verified): `<box_start> <x_0.12> <y_0.45> <content_submit> <box_type_button> <box_end>`. Paper §1 / §3 cite as "pipeline preprocessing alternative to inference-time L3 substitution"; literal SPS format vs our `[id=N] role 'label'` format direct contrast |
| **Magma** (Microsoft + UMD + UW, Yang et al.) | (ii) × L3 sub-tier "pretraining-time" | Feb 2025 (arXiv:2502.13130) ✅ verified | **Uses Qwen3-VL backbone (same family as our paper B0/B1!)** Round-3 verified. SoM + ToM 进 model weights via pretraining objectives. **Paper §1 differentiator精确 prose**: "Magma integrates SoM + ToM grounding into Qwen3-VL pretraining; we test inference-time substitution on **non-pretrained Qwen3-VL** to characterize how much routing benefit emerges from prompt structure alone vs requires pretraining" |
| **AppAgent-v2** (Tencent, Zhang et al. arXiv:2408.11824 v1, Zheng et al. v2 arXiv:2411.18279 ✅) | (ii) × L3 sub-tier "RAG-offline-explore" | 2024 (v1), 2025 (v2) ✅ verified | Closest published precedent for textual-surrogate routing. Exploration-phase agent generates **structured JSON document with `view_state_id` + `primary_action_node` + `observed_constraints` + `blocking_conditions` + `error_states_observed`**; deployment-phase RAG retrieves chunked by view_state_id with cosine similarity. **Differs from us in needing offline exploration phase + RAG retrieve**. Paper §1 explicit cite + differentiate prose: "AppAgent-v2 achieves textual-surrogate routing via offline exploration + RAG retrieval; we achieve substitution at single-pass inference time without offline phase or persistent retrieval index" |
| **ScribeAgent** (CMU, Shen/Jain/Xiao et al.) | (ii) × L3 sub-tier "fine-tune" | Dec 2024 (arXiv:2411.15004) ✅ verified | **Uses Qwen base (also same as our paper!)**. **6B token production-scale workflow data from 250+ digital domains (Scribe platform real human demonstration traces, NOT synthetic)**. LoRA fine-tuning, 65K context, 7B + 32B variants. **WebArena: 7B variant 45.7% → 51.3% SOTA** (specific Round-3 verified number). Paper §1 differentiator: "ScribeAgent fine-tunes Qwen 7B on 6B token DOM workflow corpus to achieve WebArena 51.3%; we use **non-fine-tuned Qwen3-VL** + inference-time prompt structure to recover routing benefit without dataset assembly" |
| **UI-TARS** (ByteDance Seed, Qin/Ye/Fang) | Outside (ii) × L3 — pure visual VLM | Jan 2025 (arXiv:2501.12326) ✅ verified | Pure visual coordinate grounding; **opposite end** of substitution gradient. **System-2 reasoning mechanism**: long-form CoT trace **before** physical coordinate action (task decomposition + anticipatory error checking + milestone recognition). Trace gathering across **hundreds of VMs** simultaneously with iterative bootstrap. **UI-TARS-1.5 introduces multi-turn RL** for inference-time scaling (longer reasoning = higher SR). Paper §1 cite as "no-substitution baseline (visual-only) at one extreme of substitution gradient" |
| **HMT (Hierarchical Memory Tree)** Tan, Gao, Wu (BIT) | (ii) × L3 + memory architecture | Mar 2026 (arXiv:2603.07024) ✅ verified | Decouples planning from execution via semantic element descriptions; hierarchical 84.2% recall vs flat 65.8% on memory architecture. Note: Round-2 misattributed authors "Huang et al." — actual is Tan/Gao/Wu (BIT). Paper §2 / §5 cite as "structural format trade-off task-dependent finding" |
| **CoAct-1** (OSWorld) | Hybrid programmatic + visual routing | 2025 ✅ verified | OSWorld 60.76% SOTA via task-class routing: writes Python/Bash for file ops, reserves visual perception for tasks where no programmatic backdoor exists. Different routing axis (programmatic vs visual) but same conceptual family — paper §6 routing chapter cite as "task-class-based routing precedent" |

#### (iii) × L1 — Server-side channel addition: **0 done** (paper 2 future)

笔记中 brainstorm 提及但未 implement:
- 行动规划 line 139 — "EIP 站点先验注入（M5，翻页/排序指南，零额外 token）— §31"
- Avenir Web disclosure (用户提及): "find-expensive task → use sort by" 类 task-specific knowledge channel
- 学长 5/3 brainstorm: "网页给 agent 友好接口" — 类 robots.txt-for-agents / page-emitted hint

**Industry precedent (added 2026-05-04 from DR audit)**:

| Industry system | What it adds (channel) | Year | Relevance |
|---|---|---|---|
| **NLWeb** (Microsoft) | Schema.org-based agent JSON channel; bypass DOM | May 2025 | Same cell as our pilot direction; standard exists 没 deployment evaluation |
| **A2A protocol** (Google) | Agent-card metadata channel for agent-to-agent negotiation | Apr 2025 | Different layer (agent ↔ agent) but conceptually same channel-addition principle |
| **MCP server** (Anthropic, ecosystem-wide) | Tool-execution channel; agent reads tool registry directly | Nov 2024 | Standard widely adopted; channel for tool execution rather than page perception |
| **Doubao Mobile `INJECT_EVENTS`** (ByteDance) | OS-level event channel for cross-app actions | Dec 2025 | Mobile-specific; channel-addition at OS layer rather than web |
| **Atlas StoragePartition** (OpenAI) | Ephemeral session isolation channel | Oct 2025 | Architectural workaround, not affordance addition

#### (iii) × L2 — Agent-side instrumentation channel addition: **0 done, ≥7 § identified gaps** ⚠️

**所有 7 条都指向 same root cause**: agent 缺 self-perception channel (page-position / element-clickable / action-effect / scroll-completion).

| § | Surface symptom | Underlying missing channel |
|---|---|---|
| §52 | B0 TYPE 全选变蓝 (scroll 后 input 消失, agent 不知 focus 变) | `activeElement` change channel |
| §64 | Vision type 非 input 全选变蓝 (类似 §52, 已部分 patch) | Same as §52 |
| §55 | Delete 成功信号缺失 (flash msg 时序, 笔记 explicit "暂不修复, 非结构性缺失") | Action-outcome feedback channel |
| §72 | Qwen scroll_up 3-7% (跨模型, Claude 38% 对照 — agent 不知 page-Y position) | `window.scrollY` / page-position channel |
| §72 | B1 scroll 到底之后不停 scroll → 早停 | Page-bottom-reached signal |
| §31 | task 19/22/58 case study (auto-scroll 不可预测, agent 无 reaction signal) | Page-reflow notification channel |
| §32 | Vision 坐标 misclick Art+crafts→Books 系统性偏移, **零自纠正** | Click-success / element-hit feedback channel |
| §96 | B1 32% click 打到非交互元素 (B0 仅 12.6%) | Element-clickability indicator channel |
| §50 | B0 scroll 方向问题: **无可靠修复** (笔记 explicit 写) | scroll direction / page-position channel |

**Paper §5 mechanism 直接 implication**: phantom routing space 4-fold drop-in 是测**在共享 (iii) × L2 channel-absent ceiling 下**的 (ii) × L3 compute substitution capability, 不是 break ceiling。Paper §4/§5 应 explicit 论证这个 shared-ceiling stance。

#### (iii) × L3 — Agent-compute channel addition: **n/a (impossible by definition)**

Compute 不能 add absent signal (信息论 下界); 但 prompt-side knowledge injection (M5 EIP) 是 L3 prompt-text 模拟 L1 channel 的**不完美 surrogate** — unsystematic (per-task / per-site 写 prior), 但 paper-2 framing 可以是 "L3 prompt prior simulates absent L1 channel"。

### §21.3 Identified-but-not-done items (想做但未做)

#### A. (iii) × L2 instrumentation gaps (Paper §4/§5 disclosure 必备 + paper 2 future)

7 条 § identified, 共 4 个 channel 类:

1. **Page-position channel** (§52/§64/§72/§31/§50) — `window.scrollY` / page-height / scroll-completion / direction
2. **Action-effect feedback channel** (§55/§32/§96) — click-hit / type-success / form-changed indicator
3. **Element-clickability indicator** (§96) — affordance-aware mark
4. **Page-reflow notification** (§31) — auto-scroll / async update reaction

修这些需要 agent script 加 instrumentation (agent 端 read web 已 expose 但没用的 metadata)。**Paper 1 不修, 但 §4/§5 explicit disclose ceiling stance**。

#### B. (ii) × L1 — Server-side affordance synthesis (paper 2)

- Agent-readable web 标准 (扩展 schema.org / ARIA roles for agent)
- Server-rendered SoM markers (vs 现 agent-side script overlay)
- 工作量: 中-大 (改 docker / web framework)

#### C. (iii) × L1 — Server-side channel addition (paper 2)

- Page-emitted task hints (`<meta name="agent-task-hint" value="use sort-by">`)
- Agent-specific status channel (`<meta name="agent-list-total" value="235">` 解 §52 historical-orders fail)
- M5 EIP "翻页/排序指南" — knowledge injection 跟 routing 同 family
- 工作量: 中 (web 加 meta tags + agent 读)

#### D. Agent module M-modules (笔记 行动规划)

| Module | 内容 | Layer |
|---|---|---|
| M1 select fallback | 自动 select_option fallback | (ii) × L2 |
| M2 input fallback | type 失败自动 fallback | (ii) × L2 |
| M3 retry | step 失败自动 retry | (i) × L2 |
| M4 two-stage | planner + grounder 拆分 | (ii) × L3 |
| **M5 EIP 站点先验** | 翻页/排序指南 | (iii) × L1 if server / L3 if prompt-side |

#### E. Cross-stack generalization (paper 2 / future paper)

- 实测 phantom routing space 在 live websites 是否 hold (online mind2web extension)
- 实测在不同 web stack (静态网站 / SPA / form-heavy / e-commerce) 上的 generalize
- 学长说的 "可泛化" 验证

### §21.4 Paper-level methodology asymmetries (跟 9-cell taxonomy 平行的 disclosure list)

不属于 intervention category, 是 paper §3 / §4 必须 disclose 的 methodology design 决策:

| § | Asymmetry | Paper disclosure 位置 |
|---|---|---|
| §47 A1 | B0 temperature=0.1 vs B1 do_sample=False (greedy) | §3 methodology |
| §47 A3 | B0 max_new_tokens=4096 vs B1=384 (B1 偶发 truncation → wait action) | §3 methodology |
| §95 | Visual_fp 删除 + eval_fp 简化为 2-rule (string_match / program_html) | §3 evaluation methodology + Appendix sensitivity analysis |
| §96 | P79 SoM 全元素标注 (rationale: 路由研究控制变量) | §3 ablation 设计 |
| §101 | P79 "Universal SoM" vs VWA "Action-Affordance SoM" 范式选择 | §3 ablation 设计 |
| §80 | viewport ratio threshold 0.6 (mathematical guarantee 论证) | §3 evaluation methodology |
| §89 | Visual / non-visual subset 三套指标 (raw + adjusted + non-visual) | §4 results presentation |

### §21.5 Substitution Gradient — Paper §1 Hook Positioning (added 2026-05-04 from DR Section E)

DR Section E independently characterized substitution gradient (without using "phantom routing" 术语 — circular bias 排除):

```
Server-side                Pipeline-side                 LLM-internal                LLM-internal               LLM-internal
(α/β1)                     (β2)                          (β3 pretrain)              (β-RAG)                     (β-inference)
─────                      ─────                         ─────                       ─────                       ─────
NLWeb (Schema.org JSON)    OmniParser-v2 (parse →         Magma (SoM+ToM            AppAgent-v2 (offline       Phantom routing space ⭐
   substitution at         tokenized list, downstream     grounding 进 weights,      explore 写 doc → deploy-      (paper-1 main hook)
   server emission         LLM consumes only text)        cross-modal 内化)          time RAG retrieve)            inference-time only,
                                                                                                                  no pretrain / no RAG /
                                                                                                                  no offline phase

[opposite end of gradient — no substitution]:
UI-TARS / CogAgent / Operator CUA / Magma — pure visual coordinate grounding
```

**Paper §1 hook 候选 framing** (基于 substitution gradient niche, **Round-3 verified specifics**):

> "Web agents face a substitution gradient between visual affordance perception and textual surrogate processing. Industry has populated nearly every cell of this gradient: server-side native emission (NLWeb, Microsoft, May 2025; deployed at Tripadvisor and Shopify via `/ask` + `/mcp` endpoints emitting schema.org JSON), pipeline-side preprocessing (OmniParser-v2, Microsoft Feb 2025; outputs structured-points sequences via Two-Stage SPOT prompting), pretraining-time LLM-internal grounding (Magma, Microsoft + UMD + UW Feb 2025; integrates Set-of-Mark + Trace-of-Mark annotations into Qwen3-VL pretraining), fine-tuning-time substitution (ScribeAgent, CMU Dec 2024; achieves WebArena 51.3% by fine-tuning Qwen 7B on 6 billion tokens of human demonstration workflows), RAG-time deploy substitution (AppAgent-v2, Tencent; offline exploration generates view-state-indexed JSON document, deployment retrieves via cosine similarity), and pure visual VLM grounding (UI-TARS, ByteDance Jan 2025; pure-pixel coordinate grounding with System-2 CoT reasoning across hundreds of training VMs). One cell remains unfilled: **inference-time LLM-internal substitution requiring no pretraining, no fine-tuning, no retrieval, no offline exploration, no environment cooperation**. We characterize this **phantom routing space**: a class of inference-time agent-compute substitutions of pipeline-side visual affordance synthesis (canonical: SoM marker overlay), recovering most routing benefit at near-zero incremental cost on **non-fine-tuned Qwen3-VL** (the same backbone Magma pretrains and ScribeAgent fine-tunes). P-SoM (cube center, axis 1+2 compound) is the deployment hero satisfying a 4-fold drop-in property: cost ≈ DOM, latency ~50% lower, signal AUROC ≥ baseline, drop-one ≥ 1pp pre-registered. The phantom space exhibits 2-axis empirical structure (axis 1 text payload via P-text; axis 2 SoM-style prompt via P-prompt) — both dimensions contribute non-overlapping unique tasks not solvable by P-SoM alone, evidencing phantom space is a multi-region 2-D structure rather than a collapsed point."

(这版 hook ~370 词, 含 substitution gradient contextualization with Round-3 verified specifics + Qwen-base differentiator + phantom space 主 claim + 2-axis structural)

**关键 differentiator vs each industry precedent (paper §1 必须 cite & contrast, Round-3 specifics)**:

| Industry system | 我们如何 differ |
|---|---|
| OmniParser-v2 | They do (ii)×L2 pipeline preprocessing with literal SPS format `<box_start>...<box_end>`; we do (ii)×L3 LLM compute substitution with `[id=N] role 'label'` format. Different layer + different token economics. |
| Magma | They integrate SoM + ToM **into Qwen3-VL pretraining weights**; we use **non-pretrained Qwen3-VL** at inference time to characterize what's recoverable from prompt structure alone (without retraining). **Same model family** = clean experimental isolation of pretraining contribution. |
| AppAgent-v2 | They do offline exploration phase generating JSON with `view_state_id`/`primary_action_node`/`observed_constraints`, then RAG retrieval at deploy; we do single-pass inference-time substitution. **No offline phase, no persistent index**. |
| ScribeAgent | They fine-tune Qwen 7B on 6B token DOM workflow corpus to WebArena 51.3%; we use **non-fine-tuned Qwen3-VL** (B0 235B / B1 4B) at inference time. **Same Qwen base** = clean experimental isolation of fine-tuning contribution. |
| UI-TARS / CogAgent | Pure visual VLM (no substitution); we sit at the substitution end of the same gradient on standard VLMs. |
| NLWeb (deployed at Tripadvisor + Shopify) | They do (ii)×L1 / (iii)×L1 server-side emission via `/ask` + `/mcp` endpoints with schema.org JSON; we test the LLM-internal compensation when server-side affordance is absent. **Our env-side pilot would mirror NLWeb spec on VWA — first controlled comparison of NLWeb-style emission vs L3 inference substitution.** |
| HMT (Tan/Gao/Wu BIT 2026) | Memory architecture (recall trade-off hierarchical 84.2% vs flat 65.8%); orthogonal axis to phantom routing |
| CoAct-1 (OSWorld 60.76%) | Task-class routing (programmatic Python/Bash vs visual perception); orthogonal axis but same conceptual family — paper §6 routing chapter cite |

### §21.6 Industry counter-evidence stack (added 2026-05-04 from DR Section D)

Empirical findings from DR that **directly support** paper §3 evaluation methodology + §107 audit narrative:

| Finding | Source | Paper integration |
|---|---|---|
| **WebAIM Million 2026 — ARIA pages 59.1 errors avg vs non-ARIA 42 errors** ("no ARIA is better than bad ARIA"; 27% YoY ARIA attribute growth = increasing degradation) ✅ VERIFIED Round-3 | WebAIM annual report 2026 | Paper §3 evaluation methodology: "Web environments are designed for human consumption with systemically broken accessibility metadata. Our 9+ environmental scaffolding interventions (Appendix D) reflect category-level fixes consistent with WebAIM 2026's finding that ARIA-implementing pages have ~40% more errors than non-ARIA pages, with rates worsening year-over-year." (Round-2 had wrong "57 vs 27" numbers; Round-3 verified actual is 59.1 vs 42) |
| **WebSuite trajectory analysis — failed tasks consume 2× steps vs successful** | WebSuite (Harvard, Li & Waldo 2024) | Paper §5 mechanism: connect to §72 scroll under-use / §52 type 全选 / §96 non-interactive click — failed agent task wastes steps on environmental traps, supports phantom routing 4-fold drop-in (lower latency = fewer wasted steps) |
| **CAPTCHA Reasoning Depth + 16% e-commerce checkout deployment** | DR cited research | Paper §1 / §8: real-world deployment friction; phantom routing space doesn't address this (out-of-scope), but acknowledges as ceiling |
| **Online-Mind2Web SOTA Avenir-Web 53.7% with Gemini 3 Pro backbone (Nov 18 2025 public preview)** ✅ VERIFIED Round-3 | Avenir-Web research initiative 2026 / Online-Mind2Web (Xue et al. 2025) | Paper §1 motivation: "Industry SOTA fails ~46% on live web, indicating environment-side hostility is unsolved at any agent-side scale" — **Round-2 "Operator 41.7% failure" claim was misattribution** (actually MAI-UI 41.7% **success** rate on MobileWorld), Round-3 verified Avenir-Web only |
| ~~Doubao Mobile e-commerce bans~~ | (REMOVED — Round-2 hallucination, no Chinese press verification) | **Dropped from paper** — Round-3 fact-check found no verified report of Taobao/Meituan/WeChat Pay banning Doubao-1.5-UI-TARS |
| **Browser-Use 31% → 26% drop when search disabled** ✅ Round-3 verified | OpenReview forum=6jZi4HSs6o | Paper §1 motivation: agents lean on search shortcuts rather than authentic web navigation; phantom routing tests pure-navigation capability |
| **MCP 100% tool spoofing vulnerability** (namespace collision attack) | Anbiaee et al. 2026 "Security Threat Modeling for Emerging AI-Agent Protocols" arXiv:2602.11327 ✅ verified | Paper §8 discussion: agent-readable channels (NLWeb, MCP) introduce IDPI attack surface — tradeoff to acknowledge when proposing env-side cooperation |
| **22 distinct IDPI payload techniques in the wild** (zero-sizing / CSS suppression / invisible chars / payload splitting / multilingual injection) | Unit 42 / Palo Alto Networks 2026 | Paper §8: when adopting environment-side channels, security tradeoff acknowledgment |
| **Persona-based jailbreaking 50-70% reduction in refusal rates** | Zou et al. 2025 arXiv:2507.22171 | Paper §8: agent attack surface even via prompt engineering alone |
| **Sonnet 4.6 explicit IDPI resistance positioning** (Feb 17, 2026 release, computer_20251124 tool) ✅ verified | Anthropic announce news/claude-sonnet-4-6 | Paper §8: industry recognizing IDPI as core agent threat; our work 不 directly address but acknowledge |
| **Mind2Web 2 WebJudge 3-category failure taxonomy**: Agent Failure / Environment Failure / Task Ambiguity. **WebGym infrastructure** for parallel rollout simulations. Online-Mind2Web 300 tasks × 136 live sites; Mind2Web 2 130 tasks × 44 sites. ✅ Round-3 verified | Online-Mind2Web Xue et al. 2025 | **Paper §4 evaluation methodology direct cite**: "Our 9+ environmental scaffolding interventions (Appendix D) prevent conflation of WebJudge **Agent Failure** vs **Environment Failure** — without these patches, evaluation systematically misattributes Environment Failures to agent capability." |
| **Cost / latency quantitative anchors** ✅ Round-3 verified | RAG/MCP/NLWeb Evaluation 2025 + OmniParser-v2 paper | Raw HTML/DOM: **241,000 tokens, 291s/task, F1=0.67**. Server-structural (NLWeb/MCP/RAG + GPT-5): **47K-140K tokens, 50-62s/task, F1=0.87**. OmniParser-v2: sub-second SPS preprocessing. **Paper §1 / §6 quantitative positioning**: phantom routing space sits between (compute substitution at L3 inference, no fine-tune, no offline phase) — measurements paper main contribution row |
| **CoAct-1 OSWorld 60.76% SOTA** via task-class routing (programmatic Python/Bash for file ops + visual perception only when no programmatic backdoor) ✅ verified | OSWorld + CoAct-1 publication 2025 | Paper §6 routing chapter: task-class routing precedent; orthogonal to phantom routing axis but conceptually same family |
| **3D game environments require pipeline SoM injection** (Cradle BAAI 2024 — Red Dead Redemption 2; GenSim Bayesian environment generation) ✅ Round-3 verified | Cradle paper / GenSim paper | Paper §8 future work: substitution gradient extends to game environments; opens paper 2 / 3 path |
| **OS-level metadata broken** (OSWorld + AppWorld + AndroidWorld benchmark consensus): A11y trees rendered blank or wildly inaccurate by custom rendering engines, nested iframes, unlabelled components | OS-Genesis (ACL 2025) / OS-Atlas (ICLR 2025) ✅ verified | Paper §3 / §8: cross-platform pattern — environmental hostility is universal not web-specific |

### §21.7 Pending decisions (后续 discuss / advisor sync)

#### Original (5/3)

1. **Paper 1 §1 hook contextualization** — 是否加 3-spectrum framing (i)/(ii)/(iii)? 现 §1 是 P-SoM hero + structural ablation, 加 contextualization 段 ~150 词
2. **Paper 1 §107 audit 章节 vs 单独 paper** — (i) + (ii) × L2 ~37 entries 是 paper 1 §3/§4 章节 vs Appendix D vs 单独 audit paper
3. **Paper §5 mechanism shared ceiling argument** — (iii) × L2 7 个 manifestation 是否 explicit 进 §5 (作 phantom routing space ceiling 论证)
4. **(ii) × L2 design parameter disclosure** — §94/§96/§80/§101 是否 paper §3 ablation 章节 explicit 论证
5. **paper 1 vs paper 2 scope split** — 现 inventory 不分; 何时 advisor sync decide 切片
6. **Avenir Web disclosure framing** — paper §1 / §3 是否 explicit position rigor differentiator (我们 fix env issues vs Avenir Web ignore)
7. **学长 dual-track framing** — 是否 paper §1 explicit "agent + environment dual-track" framing, 还是隐式 contextualization

#### Added 2026-05-04 (post DR audit)

8. **Adopt §21.5 candidate paper §1 hook prose?** — 用 substitution gradient framing (industry precedent stack contextualization) 替代 / 升级现 §1 hook (~250 词, 含 4-tier sub-gradient contrast + niche positioning)
9. **Fact-check 高 risk industry citations** — paper writing 前必须 verify: HMT (arXiv:2603.07024), NLAH AGENTS.md (arXiv:2602.11327), WebAIM 2026 specific numbers, NLWeb May 2025 announce date, OS-Atlas ICLR 2025 / OS-Genesis ACL 2025 venues. 不 verify 直接引 = paper 审稿 immediate reject risk
10. **Env-side pilot 实施 — Sweet Spot 设计** — 用户提议 "server emit hidden select options" 是 NLWeb-style 实例。Sweet spot 选 (a) inline `<script type="application/agent-marks">` JSON-LD / (b) HTTP header / (c) sidecar endpoint /agent/v1/page-state? 工作量 + paper claim power 跟 14-cell rerun critical path 优先级冲突
11. **AppAgent-v2 differentiation 写哪儿** — paper §1 (closest 工业 precedent) / §2 related work / §5 mechanism (RAG-time vs inference-time substitution 的 mechanism contrast)
12. **OmniParser-v2 跟 phantom routing 对比 prose** — paper §3 / §5 explicit (ii)×L2 vs (ii)×L3 layer 区分; OmniParser 是 industry-side L2 instance, phantom routing 是 paper-side L3 instance
13. **WebAIM 2026 cite 进哪儿** — paper §1 motivation / §3 evaluation methodology / Appendix D environmental fix audit 章节

---
