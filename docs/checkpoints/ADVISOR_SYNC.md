# ADVISOR_SYNC — 跟学长 sync 前的自己梳理

> **一次性 prep notes, 给自己看, 不是给学长发的 message**。列要 cover 的 topic + 自己 lean 哪边 + 关键 talking points。Detail 在 `paper_planning.md` / `实验笔记.md` 里, 这文档只是 sync 时的 reference checklist。
>
> **Sync 目标**: 30-45 min 会, cover §1-§6 各 topic, 拿到 §1.4 / §2.4 / §3.5 / §4.4 / §6.3 advisor input。

---

## 1. Phantom space + 当前 framework

### 1.1 Phantom routing space (3 arms 共享 4-fold drop-in)

- **Boundary 定义**: `"no annotated image"` (axis 3 = ✗) — 不是 "matched parsing"
- **3 arms**: P-text (axis 1 alone @ DOM-prompt 锚点) / P-prompt (axis 2 alone @ AXTree 锚点) / P-SoM (axis 1+2 compound, **paper hero**)
- **4-fold drop-in property**:
  - (a) cost ≈ DOM (no image embedding tax, `[SOM_MARKS]` 是 AXTree regex filter, ~3K text both)
  - (b) latency ~50% lower (no image inference, cls 4×)
  - (c) signal AUROC ≥ baseline (5/5 phantom `overall_usable=True`, red P-text 0.793 = 5-mode max)
  - (d) drop-one oracle 1.7-3.8pp positive per arm (B0 red CI 全 exclude 0)
- **数据 trigger** (B0 reddit 唯一 6-mode 完整 cell): P-text +3.81 / P-SoM +3.33 / P-prompt +2.86 — **3 个 arms 都贡献 unique tasks**, 旧 hook "Phantom-SoM is hidden 4th routing arm" 字面不准

### 1.2 Evidence vs Explanation 双层分离 (笔记 §108 reframe)

- **旧 framing retracted**: "Three-layer mechanism argument" (Layer 1/2/3) — confusing
- **新 framing 严格 2 层**:
  - **Evidence layer** (§3 主住所): 4 测量类型 (Outcome / Macro / Micro / Efficiency) × 4 cross-X 比较 axis (task / mode / site / model) — **2D organize 数据**
  - **Explanation layer** (§2 主住所): Zoom 1-4 hierarchical — **1D zoom scale**
- Reviewer 最忌 evidence-as-explanation: e.g. "Macro 1c search-loop 51.9→35.7%" 是 evidence, "M1 axis activates list-scanning trajectory" 是 explanation Zoom 2 — 必须分写然后 explicit link

### 1.3 Explanation 4-zoom hierarchy

| Zoom | 名称 | Self/Lit | 主要 anchor |
|---|---|---|---|
| 1 | Architectural | self (deductive) | Phantom space premise → M1+M2 by design exhaustive (不依赖 finite data verify) |
| 2 | Behavioral M1/M2 | self (2×2 activation) | M1 (Image-mirage) × M2 (Flat-list) → 4 phantom corners; P-SoM = compound emergent |
| 3 | Named phenomena | lit-anchored | Mirage Effect (Asadi 2026) / Scaffold Effect (Vu Balloccu 2026) / Sclar prompt-format / Q5 bidirectional 6/6 complete |
| 4 | Model-internal | lit-only (future work) | Cross-modal flow (Kaduri) / SteerMoE (Fayyaz B0-side) / **Tool Calling Linear Circuit (B1-side)** |

**Zoom 1 deductive argument** (核心防御): phantom comparison 锁 image=✗ → input 仅剩 (prompt 文本 + obs 文本) → 任何 differential 必由这两 input 差异 trigger → **M1 + M2 by construction exhaustive**。即使跑 100 个 phantom corners 也不能反驳。

### 1.4 想 advisor confirm

1. **Framework reframe** (3-arm 取代 4th-arm) OK 吗?
2. **Phantom space 命名** + `"no annotated image"` boundary 论证够 robust 吗? 旧 "why 5 not 8 (mismatched parsing)" 改用新 boundary 更 sound
3. **Evidence-Explanation 分层 + Zoom 1-4** 是否 reviewer-friendly? 有没 cognitive overload risk?
4. **Zoom 4 future work** 标 "follow-up paper" 是否 OK (vs paper 内 self-probe)?

---

## 2. VWA bug 发现 + 重跑决定

### 2.1 Phase A audit 37 bugs 4-cluster fix

- 5-tier audit + 文献支撑 + Gemini DR 综述对照
- **最严重**: dispatch bug 让 **94.4%** 的失败 click 落在错误 DOM target 上
- **4-cluster patch** (commit `3c15cd7`):
  1. locator-route dispatch
  2. page_changed split
  3. fuzzy cycle hash min_reps (3 → 5)
  4. RNG seeding + T=0
- Pilot 验证 **PASS**

### 2.2 Archived data 影响 (cross-mode 比较仍 valid)

- archived B0/B1 data 全部 Phase A pre-fix
- **但 cross-mode 比较仍 valid 论证**:
  - Symmetric contamination (bug 不偏 specific mode)
  - Vision counter-evidence (Vision 也 affected → 排除 mode-specific artifact)
- Phantom finding 仍 valid, 但 absolute SR 数字会 shift

### 2.3 14-cell rerun 决定

- **Scope**: 8+ phantom cells × 重要 site × cross-capability
- **Cost on RunPod 4090**: ~$70-115 actual + buffer
- **Wallclock**: ~1 周 dedicated (vs DGX shared ~3 周)
- ⚠️ **跟 §1 phantom space reframe 联动**: cell list 是否调整 (现 14-cell 含 P-prompt — 是, 因为 4-corner ablation diamond 完整化需要)

### 2.4 想 advisor input

1. **VWA bug 这一发现要不要单独成文?** (作 paper 1 副 contribution / 或独立 short paper)
2. Paper §4 disclosure 怎么 framing 才 not weaken main finding?
3. archived data 是否完全弃用 vs stratified analysis disclosure?

---

## 3. Myriad blocker → RunPod 申请

### 3.1 Myriad 物理 blocked

- UCL 防火墙 drop Tailscale CGNAT 段
- Myriad 不能 reach 我家里 quark Windows 的 VWA docker
- 物理级 blocked, 没法绕过
- 详 `docs/reference/MYRIAD_SMOKE_REPORT.md`

### 3.2 DGX shared vs dedicated 实测对比

- B1 P-text cls 现跑 30h 完成 225/234 ep (96%)
- **Average**: ~8 min/ep
- **Peak 时段**: 4 ep/h ≈ 15 min/ep
- 对比 B0 proxy (无 GPU 争抢): ~3.5 min/ep
- → DGX shared 比 dedicated 慢 **2-3× average, peak 5-10×**

### 3.3 RunPod $200 budget breakdown

- 4090 dedicated: **$0.6/h**
- 14-cell × 234 ep:
  - DGX shared: ~437h (~18 天 24/7 wallclock)
  - 4090 dedicated 估: **~87-145 GPU hours**
- Cost calc:
  - 4090 hours × $0.6/h = **~$52-87 actual**
  - + 30% buffer (crash/retry/idle): **~$70-115 reasonable**
  - + ad-hoc probe head-room:
    - Q3 B0 multi-call extended verify: ~$10
    - Tier 5 evaluator probe: ~$20
    - P-prompt diamond shop: ~$30
    - Section 5 ad-hoc query: ~$20
  - **= $200 ask**

### 3.4 Wallclock impact (deal-breaker)

- DGX 路径: paper data ready ~3 周
- RunPod 路径: ~1 周
- → paper writing + 学长 review window 从此 unblocked

### 3.5 想 advisor input

1. **RunPod 经费走什么流程?**
2. $200 是否合理 (我 lean 留 buffer 不是 minimal)?

---

## 4. Early-stop design decision

### 4.1 不是 measurement option, 是 design decision

- 之前 framing: "early-stop bias on micro metrics, A/B/C measurement options" — **逃避了 design decision**
- **真问题**: agent system 是否包含 early-stop?
- 影响**全 4 dimension** (不只 micro layer):
  - **Outcome**: SR 上 task 没机会"自然结束" (cycle detection 强制 stop)
  - **Macro**: action freq 是 censored 数据
  - **Micro**: trajectory shape 早停后 truncated
  - **Efficiency**: cost diff 部分来自早停 frequency

### 4.2 Phase A Cluster 3 是 partial mitigation 不是 full cancel

- Cluster 3 fuzzy cycle hash min_reps=5 (从 3 改 5) — 减少 false-positive cycle detection
- 仍保留 cycle detection 早停机制 — partial relax 不是 full cancel

### 4.3 Options

| Option | 内容 | Cost | Paper rigor |
|---|---|---|---|
| **A (我 lean)** | 14-cell rerun 全 cancel early-stop | +$1300 (extra wallclock + token cost) | 全 dim clean, 无需 disclosure |
| B | 全保留 | $0 | accept cross-dim systemic confound + paper §4 disclosure 段 |
| C | hybrid — main 14-cell with early-stop + 1-2 mechanism cells without | +$200 | rigor partial, ablation argument |

### 4.4 我 lean A — 想 advisor input

- Paper rigor vs cost 是真 trade-off
- A: clean across all 4 dim, paper 不需要 disclose systemic confound, reviewer attack vector 关闭
- B/C: 需要 paper §4 disclosure 段落 + 可能 reviewer attack vector
- 这是 design decision 不是我能 unilateral 决定的, 想 advisor 立场

---

## 5. 别扭 framework (简要)

- **Reverse-causal layer**: M1/M2 是 forward causal (input change → activation), 别扭是 reverse causal (mismatch count → SR/FP rank prediction)
- **2×2 别扭 grid**: P-text 1 别扭 / P-prompt 2 别扭 (compound) / P-SoM 1 别扭 / DOM 0 别扭
- **Capability-modulated reversal** (§7 finding): 大 VLM (B0) 偏 text-axis 别扭 / 小 VLM (B1) 偏 image-axis 别扭 — B0/B1 cross-capability **direction reverse**
- **现状**: N=4 cells provisional pending 14-cell rerun
- **Paper §2 Zoom 2.5 候选**: ⚠️ cognitive overload risk — **我 lean §7 cross-capability narrative use, 不进 §2 main framework**。Forward (M1/M2) + reverse (别扭) 双 framework 同时给 reviewer 太累, §7 用作 capability-reversal explanation 即可

---

## 6. paper_planning 里其他有用的 (talking points 汇编)

> 各条 1-2 行 quick reference, detail 在 `paper_planning.md` 对应 § 找

### 6.1 Q5 Gemini DR 6/6 complete (axis 3 lit anchor 厚度补齐)
- Q1-Q4 + Q6 已 integrate 进 paper.bib (commit `4541e0e`)
- Q5 (Vision-Language Model Modality Interaction) 今天 integrate (51 → 56 entries)
- Axis 3 (image bidirectional fusion) lit anchor 现跟 axis 1/2 对齐: HallusionBench / BLINK / POPE / WHOOPS / MM-Vet / Devils

### 6.2 Tool Calling Linear Circuit (ACL 2026, Qwen3-4B) → Zoom 4 B1-side anchor
- 笔记 §19 (4-09) 一直在 backlog, 今天升级到 paper §2 Zoom 4
- 给 §1 cascade `DOM→P-text→P-SoM→SoM` 一个 mechanistic 理由 (action selection 线性可分 + argument generation 非线性 → 粗-细 token-monotonic 顺序合理)
- 跟 §70 infra "Bedrock 静默吞 `tools` 参数" 现象 stack-wide brittleness 互证
- paper.bib 56 → 57 entries

### 6.3 SteerMoE scope (Fayyaz 2026 ICLR, 学长 5/1 发)
- 我 lean **option (i)**: 不 self-probe, paper §8 future work 列举 SteerMoE methodology, 让 paper sequence 自然成 trilogy
- **B1-side dual-tier 解锁**: Tool Calling hidden-state PCA 仅需 ~2300 forward pass + 单卡 RTX 4090 24h, 几乎零成本
- → Follow-up paper 现在是 **dual-tier** (B0-side SteerMoE + B1-side Tool Calling), B1 部分几乎免费
- **想 advisor confirm option (i)** 或考虑 (ii) small-scale architectural proxy / (iii) full-scale B0 self-probe 整合进主 paper

### 6.4 B0 5-call determinism probe ($0.005 cost)
- Setup: 5 calls × T=0 + top_p=1.0 + seed=42 forwarded × 同 prompt × proxy API
- Result: **5/5 byte-level distinct outputs 但 5/5 same action** (`click element_id=5`)
- Conclusion: B0 token-level non-deterministic, **decision-level convergent**
- Paper §4 disclosure: drafted in `docs/analysis/cross_sites/probe_b37_api_determinism.md`

### 6.5 Paper section completion (paper_planning §4)

| Section | Status | Hard blocker |
|---|---|---|
| 1 Intro | ✅ done (786w) | — |
| 2 Background + paper.bib | ✅ done (1514w, **57 entries**) | codex #10 expand |
| 3 Definition + Ablation | ✅ done (863w) | — |
| 4 Empirical Findings | 🟡 80% (figures FRESH, prose 待 update) | codex #11 fresh prose (~30K) |
| 5 Mechanism | 🟡 90% evidence (3-axis × 8-channel × bidirectional × §100) | codex #13 prose (~50K, 待 #10) |
| **6 Routing** ⭐ | 🟡 40% (signal AUROC ≥ baseline ✅, infra scaffold) | Tier 1 prototype (~3 天) + Tier 2 trigger (~7-10 天) |
| 7 Generalization | 🟡 40% (B1 capability profile done) | shopping (跑中) + WA + Claude |
| 8 Discussion + Implications | ❌ end-stage | data done |

- **Section 1-3 总 prose 3163 words paper-ready**
- Section 4-8 待写但 evidence 几乎全 ready

### 6.6 顶刊概率 (paper_planning §5, final scope + multi-metric/Green AI 加成)

| 投稿目标 | 概率 | 优先级 |
|---|---:|---|
| NeurIPS / ICLR main | 45-60% | Tier 1 stretch |
| ICML | 40-55% | Tier 1 stretch |
| ACL / EMNLP main | 50-65% | Tier 1 |
| **MLSys** ⭐ | **75-85%** | **Tier 1 safe** (drop-in framing 完美 fit) |
| WWW / WSDM | 75-85% | Tier 2 |
| NeurIPS D&B | 70-80% | Tier 2 |
| **TMLR (保底)** | **75-85%** | 保底 |

→ Final scope 完成后, paper 顶刊出版**几乎 100%** (cascade NeurIPS → ACL/EMNLP → MLSys → TMLR)

### 6.7 Critical risks (paper_planning §6)
1. **Execution quality** ⚠️⚠️⚠️ (顶刊成败 #1) — bug fix + paper-grade rerun + reproducibility infra
2. **Story discipline** ⚠️⚠️ — 3-arm framing 取代 4th-arm framing 已 mitigate
3. **Router design** ⚠️⚠️ — Tier 1+2 设计 + 5 关键 ablation 决定点
4. **Negative results 必须诚实报告** ⚠️
5. **B0 vs B1 reproducibility 不对称** (新增 4-30) — B0 5-call probe + Section 4 disclosure 已 mitigate

### 6.8 Section 6 Routing + Section 7 Generalization 进度
- Section 6: routing infra scaffold 已 ready, signal AUROC ≥ baseline `9d7e99f`. Tier 1 (offline supervised oracle, 3 天) + Tier 2 (online first-step trigger, 7-10 天) 是 paper 真正最值钱工作量
- Section 7: B1 capability profile done, shopping 跑中 + WA 全未开始 + cross-model (Claude) 待跑

---

## Reference files (sync 时 quick lookup)

- `docs/checkpoints/paper_planning.md` — paper strategy notebook (19 sections)
- `docs/checkpoints/实验笔记.md` — chronicle (§1-§108.19)
- `docs/checkpoints/paper_drafts/` — section1-8 prose + paper.bib (57 entries)
- `docs/checkpoints/next_steps.md` — daily action ledger
- `docs/reference/VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md` — 37 bugs + 4-cluster patch detail
- `docs/reference/MYRIAD_SMOKE_REPORT.md` — Myriad CGNAT block 实证
- `docs/reference/PAPER_STRATEGY_OPEN_QUESTIONS.md` — audit 9 issues (7 自决 disclose, 2 ask advisor)
- `docs/analysis/cross_sites/probe_b37_api_determinism.md` — B0 5-call probe + Section 4 disclosure draft
