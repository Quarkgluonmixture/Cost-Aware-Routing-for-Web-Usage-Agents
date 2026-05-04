# ADVISOR_SYNC — 5/5 sync 自用 reference

> 给我自己开会时看的 reference, 不是发给学长的材料. 顺序按我开会要讲的 flow 来.
>
> **Sync schedule**: 5/5 (周一) 早上, 30-45 min. 学长返回西班牙出差后第一次 sync.

---

## 0. 这次 sync 要 lock 的 7 件事 (TL;DR)

按重要性排:

1. **5 件 pre-registration 阈值** — K_h1 / K_h3 / TOST δ / archived 数据处理 / witness 路径. 拿到 sign-off 后我立刻翻 OSF + git commit
2. **14-cell rerun 的 cell list 最终定** — 现在 manifest 全 archive 了, 等学长一起决定哪 14 个 cell 要重跑
3. **Early-stop 关不关** — Option A (全关) vs B (保留) vs C (hybrid). 我 lean A
4. **RunPod $200 经费怎么走** — 物理路径 + 走流程
5. **Dual-track framing 升级要不要做** — 把 paper §1 hook 换成 research-characterization angle
6. **Env-side pilot 进 paper 1 还是 paper 2** — NLWeb-style server emit JSON 的 ~3-5d small pilot
7. **VWA bug 这块要不要单独成文** — paper 1 副 contribution / 独立 short paper / 不发表

时间分配 (~35 min decision + 5 min buffer):
- §1 phantom space 概念 + 当前结论 + figures: 12 min
- §2 pre-registration 5 件: 5 min
- §3 bug + 重跑: 5 min
- §4 dual-track: 7 min
- §5 RunPod: 3 min
- §6 early-stop: 2 min

---

## 1. Phantom space + paper 主结构 + 当前结论 (12 min)

> 开 `paper_section2_framework.canvas` 一边讲一边看.

### 1.1 Phantom space 是啥 (开 framework canvas)

**核心概念**: phantom routing space 是 modal cube 里"**不带标注图**"那一半. 这块区域我们以前没人 systematic 测过.

**4 个 phantom corner** (按 2 个 axis 展开):

| | DOM-prompt | SoM-prompt |
|---|---|---|
| **AXTree text** | DOM (origin) | P-prompt (axis 2 alone @ AXTree) |
| **`[SOM_MARKS]` text** | P-text (axis 1 alone @ DOM-prompt) | **P-SoM** (axis 1+2 compound, paper hero) |

加上 cube 之外的 **SoM** (image 加回来) + **Vision** (image-only 没文字) = paper 测的 6 modes.

**Hero claim** (P-SoM): 4-fold drop-in property
- (a) cost ≈ DOM (`[SOM_MARKS]` 是 AXTree regex filter, 不付 image token)
- (b) latency ~50% lower (省 image inference 这一步)
- (c) routing signal AUROC ≥ baseline (现有 router 直接能用)
- (d) drop-one oracle ≥ 1pp 显著 lift

**Structural claim** (P-text + P-prompt): phantom space 不是塌成一个点的, 是有 2-axis 结构. 证据: P-text 和 P-prompt 各自能解 P-SoM 解不了的任务. 数据上 cls/red 都 hold (bootstrap CI 排除 0).

### 1.2 别扭 framework (在同一个 canvas 顺手讲)

> 这块顺着 cube 讲, 不单独切节.

**别扭 = 输入里"应该有图但没图"或"text 跟 prompt 不对应"的 mismatch 程度**. 4 corner 别扭程度:
- DOM = 0 别扭 (text 和 prompt 都 native)
- P-text = 1 别扭 (text format 反常)
- P-prompt = 1 别扭 (prompt 反常)
- P-SoM = 1 别扭 (compound — text 和 prompt 都跟 image-on baseline 错位但互相 match)

**反向预测**: 别扭多的 mode 应该 SR 低 / FP 多. B0 4/4 cell 跟假设方向一致, B1 cls 反过来 — 大模型偏 text 别扭, 小模型偏 image 别扭. **Capability-modulated reversal**, paper §7 写 cross-capability discussion 用.

⚠️ 标 post-hoc, 不当 paper hook 主张. 现状 N=4 cells provisional, 14-cell rerun 后再 commit.

### 1.3 Paper 主结构 + 当前结论 (重跑前)

**8 sections final scope**:

| § | 内容 | Status |
|---|---|---|
| 1 | Intro + hook | ✅ 786w done |
| 2 | Background + paper.bib (57 entries) | ✅ 1514w done |
| 3 | Definition + Ablation | ✅ 863w done |
| 4 | Empirical Findings | 🟡 1725w (figures fresh, prose 待 codex update) |
| 5 | Mechanism (3-axis × 8-channel × bidirectional) | 🟡 90% evidence, 待 codex prose |
| 6 | Routing (Tier 1+2 prototype + 5 决策点) | 🟡 40% (signal AUROC ✅, infra scaffold) |
| 7 | Generalization (cross-site / cross-model) | 🟡 40% |
| 8 | Discussion + Implications + sustainability | ❌ end-stage |

**Section 1-3 共 3163 words paper-ready**. 4-8 待写但 evidence 几乎全 ready.

### 1.4 给学长看的 figures (按这顺序打开)

讲到对应 claim 时打开:

| Figure | 讲什么 |
|---|---|
| `fig_phantom_structure_venn.png` | 4-corner unique tasks 韦恩图 — paper §1 centerpiece, P-text/P-prompt/P-SoM 各自有独家解的任务 |
| `fig1ab_cascade_diamond.png` | phantom space 内部 2×2 ablation diamond + cascade DOM→P-text→P-SoM→SoM |
| `fig_meta_forest.png` | Hero + Ablation forest plot, P-SoM hero 顶部黑框 |
| `fig_forest_drop_one.png` | per-cell drop-one with Holm-sig markers |
| `fig0c_phantom_lift_bars.png` | 3-mode vs 4/5/6-mode oracle lift bars |
| `fig3a_token_cost_intra_baseline.png` | cost ≈ DOM 证据 (drop-in property a) |
| `fig3c_latency_per_step.png` | latency 50% lower 证据 (drop-in property b) |
| `fig0g_routing_auroc_heatmap.png` | 5-mode 全 overall_usable=True (drop-in property c) |
| `fig_capability_b0_b1.png` | B0 vs B1 capability profile (cross-capability evidence) |

### 1.5 重要 disclosure: bug 不影响 paper 主 claim

学长可能会问: Phase A 4-cluster bug 修了, 现在 figure 是 pre-fix 数据, 那 paper claim 还成立吗?

我的回答:
- **绝对 SR 数字会 shift** (post-fix 重跑后变, archived 数据弃用)
- **但 cross-mode pattern 仍 valid**, 因为:
  1. **Symmetric contamination** — bug 不偏 specific mode, DOM/SoM/Vision/P-* 都受影响
  2. **Vision counter-evidence** — Vision (image-only, 不走 phantom space) 也 affected → 排除"是 phantom-specific artifact"
- 14-cell rerun 后 absolute 数字会更新, **但 Hero + Structural claim 的 ordinal 关系不应变**

---

## 2. Pre-registration — 5 件具体阈值要 sign off (5 min)

最核心的 ask. 拿到学长一行 email 见证后, 我立刻 OSF + git commit lock.

| # | 决定的事 | 我 lean | 为什么这样选 | 选错了风险 |
|---|---|---|---|---|
| **(1)** | P-SoM 在多少 % 的 cell 里要显著 (Holm-corrected) 才算 hero claim 过 | **K_h1=75%** | 容忍 1-2 个 capability outlier (e.g. B1 shopping power 不足), 但不允许多数失败 | 选低 → paper hero 弱 / 选高 → 1 个 noise cell 把整 paper 破掉 |
| **(2)** | P-text + P-prompt 在多少 % cell 里要有 unique 贡献 (bootstrap CI 排除 0) 才算 structural claim 过 | **K_h3=67%** | 比 (1) 低 — structural 比 deployment 弱 commit, 不该卡那么严 | 选高 → 容易 fail 退回 fallback / 选低 → claim 不 trivial |
| **(3)** | Effect 多大才算"实质非零" (TOST equivalence margin δ) | **δ=1.0pp** (≈ 2 tasks) | 大约是 bootstrap noise floor; 比这小就是采样噪声 | 选小 → TOST 测不出东西 / 选大 → 太容易 reject equivalence, claim 显弱 |
| **(4)** | archived (Phase A 之前) 数据怎么处理 | **paper 主分析只用 post-fix**, archived 进 Appendix D 作 robustness check | 主体 bug-clean + 附录展示 due diligence | 太严 → 浪费 archived 数据 / 太松 → bug 污染主分析 |
| **(5)** | pre-registration 怎么留证据 | **git commit + 学长一行 email 见证 + 投稿前上 OSF 拿 DOI** (paper §1 footnote cite) | 多层 rigor signal | 没 witness → reviewer 怀疑事后改 / 学长不愿见证我自己 lock 也 OK 但稍弱 |

**Mapping table** (数据进来后 paper hook 怎么写, 5 个 case 预先 lock):

| 数据结果 | Paper 怎么写 |
|---|---|
| (A) Hero + (B) Structural 全过 | "phantom routing space 有 2-axis 结构 + P-SoM hero" — **最强 hook** (R1) |
| (A) 过 + (B) 只过 1 个 axis | single-axis 结构 + hero — 次强 (R2) |
| (A) 过 + (B) 全 fail | 退回旧 framing "P-SoM is hidden 4th routing arm" — 中等 (R3) |
| (A) 部分 fail | partial drop-in, 弱 claim (R4) |
| (A) fail | paper 死, pivot 到 VWA bug paper 或者放弃 (R5) |

> 这个 mapping 写在 `preregistration.md` 里, reviewer 能查"数据 → framing"是确定函数, 不是数据进来后挑 fit hook. 如果学长担心这是"披着 pre-registration 外衣的 garden-of-forking-paths" (事后挑 framing), 我的回应: mapping 是数据前写死的, reviewer 一查就知道.

---

## 3. Bug 发现 + 重跑决定 (5 min)

### 3.1 Phase A 4-cluster fix 简介

5-tier audit 完了, 文献支撑 + Gemini DR 综述对照. **最严重的 bug**: dispatch bug 让 **94.4%** 的失败 click 落在错误 DOM target 上.

**4-cluster patch** (commit `3c15cd7`, 4/30 15:35):
1. locator-route dispatch
2. page_changed split
3. fuzzy cycle hash min_reps (3 → 5)
4. RNG seeding + T=0

Pilot 验证 PASS.

### 3.2 现在 manifest 全 archive 了 (我决定的)

5/4 sync 前发现一个 paper-grade 风险: 我之前跑了一些 post-fix run (B1 cls P-prompt 5/1, B1 reddit phantom_som chain 5/4), 但 sibling modes 都还是 pre-bug. 把它们放一起算 oracle / drop-one **会让 cross-mode 比较里 fix-effect 跟 mode-effect 混不开**.

我做了:
- Stop 全部 active runs (B1 reddit chain + B0 cls P-prompt)
- `run_manifest.yaml` 全部 27 entries flip 成 `grade: archived`
- Figure 留在 14:06 last paper-grade-pre-bug-only 状态 (不重 generate)

**学长签了 14-cell rerun cell list 后**, 我 flip 对应 entries 回 paper-grade, aggregator 自动 pick up.

### 3.3 14-cell rerun 想跟学长一起决定的事

**Cost on RunPod 4090**: ~$70-115 实际 + buffer = $200 ask
**Wallclock**: ~1 周 dedicated (vs DGX shared ~3 周)

**Tentative scope** (待学长 confirm):
- B0 × {classifieds, reddit} × {phantom_text, phantom_som, phantom_prompt} = 6
- B1 × {classifieds, reddit} × {phantom_text, phantom_som, phantom_prompt} = 6
- B0 shopping × {phantom_text, phantom_som} = 2 (P-prompt 缓?)
- = 14

可能调整: 13 (砍 B0 shop P-prompt) / 16 (加 B1 shopping). 学长 input.

**还要问**:
1. **VWA bug 这一发现要不要单独成文?** 作 paper 1 副 contribution / 独立 short paper / 不发表
2. Paper §4 disclosure 怎么 framing 才不 weaken main finding?

---

## 4. Dual-track framing — 从 bug fix 自然引出 (7 min)

> **开 `dual_track_taxonomy.canvas` 一边讲一边看** (3×3 grid + bottom annotation).
>
> 这块自然从 bug fix 引出. 学长 5/3 push "两头出发, environment 也做". 我后来想清楚: **我们一直在做 dual-track**, 只是没显性 frame.

### 4.1 我们一直在做 environment fix work

笔记 §1-§108 里有 ~40 条 § 是 environment intervention:
- §51-62 select dropdown 7 层 fix
- §53 confirm 弹窗 auto-accept
- §80 viewport ratio 计算 bug
- §81 Wikipedia ZIM 版本修复
- §103 Magento auth bug
- §107 Phase A 4-cluster ← 这就是

学长 push 的真正意思: 把这件事**显性 frame** 成 dual-track, 不是加新方向.

### 4.2 9-cell taxonomy (3 性质 × 3 layer)

```
                (i) Bug fix    (ii) Affordance synthesis    (iii) Channel addition
L1 Server-side    ~6 ✅           NLWeb (deployed)             A2A protocol
L2 Pipeline       ~28 ✅          OmniParser-v2 / Tarsier等    (gap, ~7 § identified)
L3 LLM-internal   n/a            ⭐ Phantom routing            n/a
                                 (paper-1 niche)
```

**(ii)×L3 内部 4 个 sub-tier** — 这是 paper-1 真 niche:
- Pretraining-time: Magma (用 **Qwen3-VL backbone**, 跟我们同 base!)
- Fine-tuning: ScribeAgent (CMU, **同样用 Qwen 7B**, fine-tune 6B token, WebArena 51.3%)
- RAG-offline: AppAgent-v2 (Tencent, 离线 explore 写 JSON, 部署 retrieve)
- **Inference-time only: 我们 paper** ← 没工业 system 占这格, 不需要 retrain / fine-tune / retrieve / offline

### 4.3 Industry stack (西方 + 中国)

| 工业 system | 跟我们 paper 关系 |
|---|---|
| **NLWeb** (Microsoft 2025-05) | Tripadvisor + Shopify 已部署. server emit schema.org JSON via `/ask` + `/mcp`. 我们想做的 env-side pilot 直接 mirror NLWeb spec |
| **OmniParser-v2** (Microsoft 2025-02) | (ii)×L2 工业 canonical, screenshot → tokenized list |
| **Magma** (Microsoft+UMD 2025-02) | 用 Qwen3-VL backbone 把 SoM+ToM 训进 weights — 我们的 backbone 同 base 不同方法 |
| **ScribeAgent** (CMU 2024-12) | 同样用 Qwen 7B base, fine-tune 6B token DOM corpus, WebArena 45.7→51.3% |
| **agent-browser** (Vercel Labs, 81+ releases) | CDP-direct Rust daemon, 整合 Claude Code / Cursor / Codex / Gemini CLI |
| **Tarsier** (Reworkd) | typed SoM brackets + 内部 benchmark 声称 "**unimodal text beats GPT-4V + Tarsier-Screenshot by 10-20%**" — 最接近的 research-line precedent |
| **Playwright MCP** / **Stagehand** / **Browser Use** / **Skyvern** / **Anchor Browser** / **OpenClaw** (361K stars) | (ii)×L2 工业 SDK 集群 |
| **PageAgent** (Alibaba, 17.5k stars, verified) | 中国 P-text artifact equivalent |
| **UI-TARS** (`arXiv:2501.12326`, ByteDance), **AutoWebGLM** (`arXiv:2404.03648`, Tsinghua), **OS-Atlas** (`arXiv:2410.23218`), **Mobile-Agent v2/v3** (X-PLUG) | 中国 academic L3-pretrain/fine-tune 集群 |
| **Qwen3-VL TR** (`arXiv:2511.21631`) | **直接是我们 paper backbone** (B0=235B-A22B / B1=4B) |

### 4.4 Hook prose 升级方向 (research-characterization angle)

之前 hook 有点 over-claim "first to use phantom configurations". 这站不住, 因为 agent-browser / Tarsier 实际部署过 text-only routing.

**正确 framing** (epistemic-level distinction):
- 工业 deploy at **artifact level** (单 mode 部署省钱)
- Paper 提供 **research characterization** (controlled cross-mode comparison only research can do — industry 单 mode 部署本质上做不了)
- → "First systematic peer-reviewed characterization of routing behavior across phantom routing space configurations on Qwen3-VL via controlled cross-mode comparison"

**Reviewer 防御**:
- "industry already does X" → "他们 deploy, 我们 characterize, 不同性质的问题"
- "为什么不 ablate 站点指纹库 / 短 grammar / FPC fix" → "那些是 SE engineering 模块 (deployment optimization), paper 测的是 cognitive routing (per-axis representation→LLM behavior isolation), 不同范围"
- 同时 acknowledge: paper limit 在 observation-axis, action-grammar (`click @7`) 是 future work axis

完整 ~530w prose 在 `paper_planning §21.5`.

### 4.5 想跟学长 lock 的 3 件 (核心)

1. **Paper §1 hook 用 research-characterization angle 升级吗?** (倾向 yes — `paper_planning §21.5` 有完整 prose)
2. **Env-side pilot 进 paper 1 §7.x 还是 paper 2?** (倾向 paper 1 §7.x, 1 site × 2 mode (phantom_text vs phantom_server), ~3-5 d, NLWeb-style)
3. **Pilot fail 怎么办?** (倾向 negative finding 加分 — 如果 server emit 不复现 routing benefit, 反而证 phantom routing 的 substrate-independence claim 站不住, paper §1 hook 反而要 strict 化)

---

## 5. Myriad blocker → RunPod 申请 (3 min)

### 5.1 Myriad 物理 blocked

UCL 防火墙 drop Tailscale CGNAT 段. Myriad 不能 reach 我家 quark Windows 的 VWA docker. 物理级 blocked, 没法绕. 详 `docs/reference/MYRIAD_SMOKE_REPORT.md`.

### 5.2 DGX shared vs dedicated 实测

- B1 P-text cls 跑 30h 完成 225/234 ep (96%)
- Average ~8 min/ep (peak 时段 15 min/ep)
- 对比 B0 proxy (无 GPU 争抢): ~3.5 min/ep
- → DGX shared 比 dedicated 慢 **2-3× average, peak 5-10×**

### 5.3 RunPod $200 budget

- 4090 dedicated: $0.6/h
- 14-cell × ~234 ep:
  - DGX shared: ~437h (~18 天 24/7)
  - 4090 dedicated 估: ~87-145 GPU hours
- Cost calc:
  - 4090 hours × $0.6/h = ~$52-87 actual
  - + 30% buffer (crash / retry / idle): ~$70-115 reasonable
  - + ad-hoc probe head-room (~$80): Q3 multi-call probe + Tier 5 evaluator probe + P-prompt diamond shop + Section 5 ad-hoc query
  - **= $200 ask**

### 5.4 Wallclock impact (deal-breaker for paper timeline)

- DGX 路径: paper 数据 ready ~3 周
- RunPod 路径: ~1 周
- → paper writing + 学长 review window 从此 unblocked

### 5.5 想 advisor input

1. **RunPod 经费走什么流程?** (UCL 报销 / 学长课题组卡 / 我先垫)
2. $200 是否合理 (我 lean 留 buffer 不是 minimal)?

---

## 6. Early-stop design decision (2 min)

### 6.1 不是 measurement option, 是 design decision

之前 framing "early-stop bias on micro metrics, A/B/C measurement options" — **逃避了 design decision**. 真问题: agent system 是否包含 early-stop?

影响 4 dimension:
- **Outcome**: SR 上 task 没机会"自然结束" (cycle detection 强制 stop)
- **Macro**: action freq 是 censored 数据
- **Micro**: trajectory shape 早停后 truncated
- **Efficiency**: cost diff 部分来自早停 frequency

### 6.2 Phase A Cluster 3 是 partial mitigation

Cluster 3 fuzzy cycle hash min_reps=5 (从 3 改 5) — 减少 false-positive cycle detection. 但**仍保留 cycle detection 早停机制** — partial relax 不是 full cancel.

### 6.3 Options

| Option | 内容 | Cost | Paper rigor |
|---|---|---|---|
| **A (我 lean)** | 14-cell rerun 全 cancel early-stop | +$1300 (extra wallclock + token cost) | 全 dim clean, 无需 disclosure |
| B | 全保留 | $0 | accept cross-dim systemic confound + paper §4 disclosure 段 |
| C | hybrid — main 14-cell with early-stop + 1-2 mechanism cells without | +$200 | rigor partial, ablation argument |

### 6.4 我 lean A — 想 advisor input

Paper rigor vs cost 是真 trade-off. A 全 dim clean, paper 不需要 disclose systemic confound, reviewer attack vector 关闭. B/C 需要 paper §4 disclosure. 这是 design decision 不是我能 unilateral 决定, 想 advisor 立场.

---

## Reference (开会查阅)

**Pre-registration**: `docs/checkpoints/preregistration.md` (status:draft, lock 5 件 commit 后 flip locked)

**Visualizations 开会现场打开**:
- `docs/checkpoints/paper_section2_framework.canvas` — phantom space + 别扭 framework 主 canvas (§1)
- `docs/checkpoints/dual_track_taxonomy.canvas` — Dual-track 9-cell 2D matrix (§4)
- `docs/checkpoints/phantom_space.canvas` — Zoom 1 architectural view (备用)
- `results/phantom_paper/figures/*.png` — 14:06 paper-grade-pre-bug-only state (见 §1.4)

**Detail 文档 (按需深挖)**:
- `docs/checkpoints/paper_planning.md` — §21 dual-track + §1 hook + §19 decision log
- `docs/checkpoints/实验笔记.md` — §1-§109 chronicle (§109.16-19 是 5/4 evening 4-round epistemic upgrade detail)
- `docs/checkpoints/paper_drafts/` — section1-8 prose + paper.bib (57 entries)
- `docs/reference/VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md` — 37 bugs + 4-cluster patch detail
- `docs/reference/MYRIAD_SMOKE_REPORT.md` — Myriad CGNAT block 实证
- `docs/reference/EVIDENCE_LAYER_AUDIT.md` — pre-registration template + meta-rationale
- `docs/analysis/cross_sites/probe_b37_api_determinism.md` — B0 5-call determinism probe (paper §4 disclosure)

**Manifest backup** (5/4 全 archive 决定): `results/phantom_paper/run_manifest.yaml.bak_20260504`
