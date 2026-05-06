---
type: sync-outcome-record
sync_date: 2026-05-05
sync_format: Microsoft Teams (advisor 西班牙马德里, 学生 DGX Spark UK)
sync_duration: ~30 min (网卡断在 threshold detail)
transcript: docs/reference/transcript.md
followup: docs/checkpoints/advisor_sync_5_5_followup.md
status: rolling (Part A confirmed, Part B post-sync, Part C pending advisor email)
last_updated: 2026-05-05
---

# 5/5 Advisor Sync Outcomes — 已确认决议记录

> Sync outcome registry, audit trail purpose. Paper §1 footnote 投稿前 cite 这份 doc 作 decision-provenance reference.
>
> 三块结构:
> - **§A**: Sync 当时 advisor verbal-confirmed (有 transcript quote 支撑)
> - **§B**: Student post-sync decided (advisor email witness pending)
> - **§C**: Pending advisor email reply (指向 followup doc, 不展开)
>
> Email 回复进来后 §C → §A migrate, 同时 update 本文档 + preregistration.md decision log.

---

## §A — Sync 当时已 confirmed (advisor verbal, transcript-grounded)

### A.1 Early-stop = Option A 全 cancel ✅

**决议**: 16-cell rerun 跑全 30 step (max step), **不 early-stop**. archived data 也 obsoleted, 不 partial 用.

**Transcript quote**:
> 学生: "我打算在新的跑的时候就不把它做 early stop 了, 因为之前 early stop 只是为了省钱..."
> advisor: "影响到你对这分析,不想 early stop."
> 学生: "对."

**Rationale (advisor accepted)**: early-stop 是 cross-dim systemic confound (Outcome / Macro / Micro / Efficiency 四 dim 都受影响), 不止 micro. 全 cancel 让 paper §4 不需要 disclosure 段, reviewer attack vector 关闭.

**成本**: +$1300 (extra wallclock + token cost, paper_planning §6 estimate).

---

### A.2 Manifest 全 archive + 必须 16-cell 重跑 ✅

**决议**: archived 27 entries (5/4 commit `8a9f595` 已落地) 仅作 Appendix D robustness check, 主分析只用 post-Phase-A 16-cell rerun 数据.

**Transcript quote**:
> 学生: "之前的所有的内容都受到这个 bug 影响,所以就是之前的只能当做 archive,只能当作归档了。所有的都需要重跑."
> advisor: "对,这是一个问题我之前说的."

**Background**: Phase A 4-cluster fix (commit `3c15cd7`, 4/30 15:35) — dispatch / page_changed split / fuzzy cycle hash min_reps / RNG seeding 4 个 P0 bug 修了. archived 数据是 pre-fix, 跟 post-fix 不能混合分析否则 fix-effect 跟 mode-effect 混不开.

---

### A.3 Paper 拆开发 (split direction confirmed, exact count pending Q1) ✅🟡

**Confirmed direction**: paper split, **不合并成单 paper**.

**Transcript quote**:
> advisor: "我觉得分开比较好。我的感觉是分开比较好。... 包括 Introduction 和 Abstract 会变得超级冗杂超级长... 没有人类的优雅, 人类的优雅是把东西给拆成几篇... 我是建议拆."

**Pending Q1 (followup doc)**: 3 papers vs 4 papers — 具体取决于 Mechanistic nested in Phantom 还是独立 (advisor summary "三种不同的 paper" 但列了 4 项内容, ambiguous).

---

### A.4 VWA bug → ACL position paper / survey + community repo ✅

**决议**: VWA / WebArena benchmark bug catalog (~37 entries from 笔记 §21.2 + Phase A 4-cluster) **不当 normal substantive paper** — 做 ACL position paper 或 survey + community repo 持续更新.

**Transcript quote**:
> advisor: "那个 bug fix 我建议不要当作单独的一篇 paper 去发, 我建议是要么是把它变成一个 survey, 要么是把它变成一个 position paper... ACL 里面有 position paper. 就是难度不高, 但是你可以把这些问题提出来. 或者说你变成个 survey 也可以... 最后的 contribution 也就是把这些错误提出来之后, 给出一个 website 或者 repo, 去持续更新你发现的问题."
> 学生: "对我也这么想, 确实."

**Pending Q3 (followup doc)**: position paper vs survey 二选一 (我 lean position paper for lean execution).

---

### A.5 Routing paper = benchmark study (独立成文) ✅

**决议**: Routing paper 单独成 paper, 投稿定位 **benchmark / D&B track**, 不冲 main conference primary contribution.

**Transcript quote**:
> advisor: "至于 routing 这个方面的话呢, 我是建议把这个当做一个 paper, 把这个当做一个 benchmark, 类似 benchmark study 去发, 这个会稳一些. 这样你就有一个实验 based 的、benchmark based 的 paper."

**Pending Q2 (followup doc)**: Phantom paper 是不是也是 benchmark study 定位 (vs main conference primary).

---

### A.6 Mechanistic interpretability 是 publication-worthy 新方向 ✅

**决议**: 在已有 contrastive set (P-text / P-SoM / DOM × site × model) 上做 SAE feature / activation patching / linear probe — advisor explicit 鼓励, 单 mechanistic 结果就可能够单独 paper 发 (尤其如果 cross-model golden feature hold).

**Transcript quote**:
> advisor: "或者是你用 SAE feature... 直接去 steer 这个 feature. 这会挺有意思的, 这会是前所未有的这种 inference time 的这种 steering."
> advisor: "你只需要 contrastive set, 你现在这已经有 contrastive set 了对不对? 然后你只需要去做 activation patching... 还有就是 train linear probe..."
> advisor: "说不定单是这个结果你就可以拿出来单独发一篇比较好的 paper."

**Specifically advisor 暗示**: 如果找到 cross-model golden feature, "这个价值就很大了... 这就是 golden feature."

**Pending Q1 (followup doc)**: Mechanistic 是 nested in Phantom paper 还是独立成 paper, 这是 paper-split exact count 决定因素.

**Pending Q3 in Part 3 (followup doc)**: Mechanistic scope — B1-only (Qwen3-VL-4B local) 还是加 cross-architecture validation (Llama-3/4 local).

---

### A.7 Workshop submission 节奏 ✅

**决议**: 小规模数据 (e.g. cls + red 两 site phantom subset) 跑完后投 workshop 加 CV; full version 之后投 main conference.

**Transcript quote**:
> advisor: "看一下有没有什么, 就是如果你想搞投的话也看有没有什么 workshop 之类的去投递. 因为到时候你在 CV 里面是可以加上比如说你这个 paper 中了哪个 workshop 或者是带头哪个 workshop, 是可以给你 CV 加分的."
> advisor: "OK 可以. 到时候会给你发一些相关的 workshop 或者会议."
> 学生: "OK OK, 我把这个数据跑完之后我就发一个, 就把那个小规模数据跑完我就发一个."

**Pending Q5 in Part 1 (followup doc)**: advisor 发 workshop names list (advisor 承诺 send, 但 sync 当时没发).

---

### A.8 Compute paths (优先级 ranking) ✅

**决议** (按优先级):

| Priority | Path | Status |
|---|---|---|
| **1** | **Advisor 5090** (advisor 自有, 搬到 AI Center 后长挂, 学生夜间用) | advisor 承诺 "回去把那个 5090 给搬到 AI Center, 然后就一直挂着" |
| **2** | **Rancher / Condenser H100** (UCL 官方, advisor 账号借) | advisor 已申请到 H100 + 几张其他卡, 学生 borrow advisor 账号试 (advisor 自己还没用过) |
| **3** | **RunPod 4090 self-fund** ($0.6/h, $200 budget) | backup if 1+2 不够 |
| **4** | **Myriad** (advisor 之前穿透过, 7-8h limit per block) | 已基本放弃 (Tailscale CGNAT block + 时长限制不适合 single block 实验) |

**Transcript quote**:
> advisor: "我有一张 5090. 我准备这次回去把我那个 5090 给搬到 AI Center, 然后就一直挂着... 那等我不打游戏的时候你来跑半夜跑."
> advisor: "有听过 Condenser 吗? 这个东西? 就是我前段时间申请了, 他给我搞了一个 H100... 它是 UCL 官方的, 但是这个的话, 需要走一个不同的申请流程... 它不是那种 cluster 的, 那种 container, 你可以直接在里面进行长期的那个使用."
> advisor: "如果我能用的话, 你应该可以用我的账号先去试试那个."

**RunPod $200 budget**: advisor 没明确 reject (sync 当时跳过), 学生 keep as backup.

---

### A.9 Pre-registration witness mechanism framework ✅ (具体 threshold pending)

**Confirmed mechanism**: git commit (timestamped, immutable post-push) + advisor email reply (independent timestamp + identity audit) + OSF DOI (paper submission stage external witness) — 三层 audit trail.

**Transcript quote**:
> 学生: "您给我郵件不是有时间戳嘛, 然后把那个相当于保证一份证据嘛. 因为 commit 它时间是可以改的, 我怕到时候会会."
> advisor: "OK, 明白."
> 学生: "我搜了下还有什么一个网站吧到时候 [指 OSF], 但这个不急, 就是特别官方那种的, 它可以直接提交上去然后直接 locked 住那种的. 但这个不急."
> advisor: "明白, 明白明白."

**Pending Part 3 #1 (followup doc)**: 三个具体 threshold (K_h1=0.75 / K_h3=0.67 / TOST δ=1.0pp) advisor email confirm — advisor 当时网卡断没看到 forest plot.

---

### A.10 Environment 3-layer framework accepted (paper mapping pending Q4) ✅

**决议**: 学生提出的 3-layer framework — Server-side / Agent-pipeline / LLM-internal — advisor accept 概念.

**Transcript quote**:
> 学生 (讲 select dropdown 信息标签 idea): "如果 server 侧能够给它这样一个信息... 通过这个按钮去做一些比如说查找最贵的, 查找最便宜的..."
> advisor: "就是增加一些信息嘛, 在它的那个 DOM tree 里."
> 学生: "对对对, 就是增加一些信息. 这个其实我觉得就是 server 侧能做的一个东西."

**Pending Q4 (followup doc)**: 3-layer framework 进哪 paper 的 mapping (server-side → Routing / agent-pipeline → Phantom / LLM-internal → Mechanistic 是 student lean, 学长 lock).

---

## §B — Student post-sync decided (advisor email witness pending)

### B.1 N_cells = 16 ✅ (student decision)

**决议**: paper-grade rerun scope **16 cells** (= B0 × {cls, red} × 3 phantom + B1 × {cls, red} × 3 phantom + B0 shop × 2 phantom + B1 shop × 2 phantom).

**与 sync prep 的 delta**: ADVISOR_SYNC §3.3 当时 lean 14, sync 中没具体讨论 N_cells (网卡断), 学生 post-sync decide 16 让 cross-capability shop coverage 完整 (B0 shop + B1 shop 都有 phantom_text/phantom_som).

**Threshold count auto-adjust**: K_h1 = 0.75 → ≥ 12/16 cells; K_h3 = 0.67 → ≥ 11/16 cells.

**Locked in**:
- `preregistration.md` line 203 (✅ flipped 5/5 to "student-decided")
- `preregistration.md` decision log (5/5 entry added)
- `advisor_sync_5_5_followup.md` 开头 confirmed 区
- `paper_planning.md` / `next_steps.md` / `ADVISOR_SYNC.md` cross-doc sync

---

## §C — Pending advisor email reply (followup doc Q1-Q11)

详见 [`advisor_sync_5_5_followup.md`](advisor_sync_5_5_followup.md). 11 个 Q 简述:

| # | Question | Status |
|---|---|---|
| Part 1 Q1 | Mechanistic nested in Phantom 还是独立 (3 vs 4 papers exact count) | ⭐⭐⭐ critical, 决定 paper split 最终结构 |
| Part 1 Q2 | Phantom paper 投稿定位 = benchmark study? | clarification |
| Part 1 Q3 | VWA bug = ACL position paper 还是 survey | binary lean |
| Part 1 Q4 | Environment 3-layer framework paper mapping | mapping confirm |
| Part 1 Q5 | Workshop submission 节奏 + advisor send names | confirmation + ask |
| Part 1 Q6 | Pre-reg mechanism final OK | meta confirm |
| Part 3 #1 (a) | K_h1 = 0.75 lock | ⭐⭐⭐ paper-grade gating |
| Part 3 #1 (b) | K_h3 = 0.67 lock | ⭐⭐⭐ paper-grade gating |
| Part 3 #1 (c) | TOST δ = 1.0pp lock | ⭐⭐⭐ paper-grade gating |
| Part 3 #2 | Train/test split (5-fold lean vs LOSO) | router 用 |
| Part 3 #3 | Mechanistic scope (B1-only vs +cross-arch) | mechanistic paper claim power |

**Email 已发**: 2026-05-05 (followup doc share GitHub link + Part 3 figures)

**Expected reply window**: advisor 5/5 sync 后回西班牙出差, email 回复估 2-5 天 (5/7-5/10).

**Email 进来后 action**:
1. 本文档 §C → §A migrate, 移对应 row 到 §A 加 advisor quote / paraphrase
2. `preregistration.md` decision log 加 "5/X advisor email witness 确认 K_h1/K_h3/δ" entry, status:draft → status:locked, fill registered_at + registered_git_sha + witnessed_by
3. 落 `.witness/preregistration_witness.eml` (gitignored)
4. OSF 上传 (advisor email reply 后立刻)

---

## §F — OSF DOI 上传 workflow (post-email-witness 决议)

> 这一节 freeze 拿到 advisor email 后 OSF 上传的所有 decisions, 防止届时临时拍脑袋.

### F.1 上传 timing 决议: **早上传** (post-email, NOT paper-submission-stage)

**两种方案对比**:

| Option | Timing | Audit-trail 强度 | 我 lean |
|---|---|---|---|
| (a) **Early** — advisor email 回后 2-3 天内 | 16-cell rerun 启动前/中 | DOI 时间戳锁在 data unblinding **之前**, reviewer 一查就 explicit verify "pre-data" | ⭐ |
| (b) **Late** — paper submission stage | 16-cell rerun done + paper 写完后 | DOI 时间戳可能晚于 data unblinding, "pre-data" claim 弱化, reviewer 需要 cross-reference git SHA 才能 verify | weaker |

**决议: Option (a) early upload**. preregistration.md §6 (b) 写的 "OSF DOI optional at paper-time" 是 fallback wording; 实际 ASAP upload, 让 DOI 时间戳跟 data lock 时间戳关系明确 (DOI < data unblinding).

### F.2 OSF Project naming + 结构

**Project name**: `Phantom-SoM 16-cell Pre-Registration Witness`

**File manifest to upload** (locked snapshot, post-email):

| File | 来源 | 用途 |
|---|---|---|
| `preregistration_locked.pdf` | git locked commit + pandoc render | 主 pre-reg 文档 |
| `preregistration_locked.md` | 同上 source | markdown 版本 (machine-readable) |
| `evidence_layer_audit_§2.pdf` | `docs/reference/EVIDENCE_LAYER_AUDIT.md` §2 节 | 配套 epistemic rationale |
| `advisor_email_witness.pdf` | advisor email screenshot + email header (timestamp + sender + DKIM) | identity audit |
| `fig_meta_forest.png` | `results/phantom_paper/figures/` | Hero + Ablation forest plot |
| `fig_forest_drop_one.png` | 同上 | Per-cell drop-one CI |
| `fig_phantom_structure_venn.png` | 同上 | 4-corner unique tasks |
| `decision_log_excerpt.md` | preregistration.md Appendix A | 决策 audit trail |

**README.md** (OSF 项目顶层):
- Sync date / git SHA at lock / advisor email date / data unblinding boundary date
- 列 8 lock decisions + threshold values
- "DO NOT EDIT this OSF project after upload — locked snapshot for paper §1 footnote citation"

### F.3 DOI 格式 (paper §1 footnote 用)

**预期 footnote prose**:
> "Hypotheses pre-registered prior to 16-cell rerun (OSF DOI 10.17605/OSF.IO/XXXXX, Git SHA `<commit-at-lock>`, witnessed by `<advisor name>` on `<email-reply-date>`)."

**Footnote 占用**: paper §1 第一段末尾 OR §3 (definition + ablation) 顶部 — 取决于 §1 hook reframe 后的结构.

### F.4 Cost / Timeline

- **Cost**: $0 (OSF 免费)
- **Upload time**: ~30 min (含 PDF render + OSF account setup if first time + file upload + project description)
- **DOI generation**: typically < 24h post-upload submission (OSF batch process)
- **总计**: advisor email reply 后 1-3 天内拿到 DOI

### F.5 Pre-upload 必须 cross-doc lock

OSF 上传**之前**必须把以下 doc state freeze (确保 OSF snapshot 跟 git tree state 一致):

1. ✅ `preregistration.md` frontmatter flip — `status: draft` → `status: locked`, fill `registered_at` (advisor email date) + `registered_git_sha` (lock commit SHA) + `witnessed_by` (advisor name)
2. ✅ `preregistration.md` decision log 加 entry — "5/X advisor email reply 确认 K_h1=0.75 / K_h3=0.67 / TOST δ=1.0pp + 8 lock decisions"
3. ✅ 本文档 (`advisor_sync_5_5_outcomes.md`) §C → §A migrate
4. ✅ `advisor_sync_5_5_followup.md` 标 "✅ closed via email reply, see outcomes.md §A"
5. ✅ `paper_planning.md §19` decision log 加对应 entries (8 lock decisions)
6. ✅ 实验笔记 §110 chronicle 写 advisor email reply outcome
7. ✅ git commit + push (commit message 含 "pre-reg locked, advisor witness")
8. ✅ `.witness/preregistration_witness.eml` 落地 (gitignored, advisor email 原始 raw 存一份)

### F.6 Post-DOI maintenance

- DOI 拿到后 → paper drafts/section1_intro.md 加 footnote (codex prose pass 时)
- DOI 写进 `paper_planning.md §22` Multi-Register Novelty Inventory (Register II K row, status flip 🟡 → ✅)
- DOI 写进 `next_steps.md §0` (从 "advisor sync pending" 移除)
- 投稿前再 freeze 一次 OSF (option to add data unblinding-after analyses 作 supplement 但 NOT 改 main pre-reg snapshot — 那个 pre-reg version 永远 locked)

### F.7 Pending decision (sync delta to flag)

⚠️ 当时录音里 advisor 对 OSF 反应是 "OK, 明白, 明白明白" — 没 deep-dive. **没有 explicit lock OSF upload mechanism**. 如果 advisor 后续 push back "OSF 不必, git + email 见证够了", 退回 git+email 双层 (lose external timestamp witness 但其他还都有). Email follow-up 时如果学长有 input, 在本节 F.1 timing 决议处 update.

---

## §D — References

- **Transcript**: `docs/reference/transcript.md` (5/5 sync 完整逐字稿, 网卡断点已标注)
- **Followup doc** (发学长邮件主体): `docs/checkpoints/advisor_sync_5_5_followup.md`
- **Pre-registration source**: `docs/checkpoints/preregistration.md` (16-cell synced)
- **Sync prep snapshot** (现 partly historical): `docs/checkpoints/ADVISOR_SYNC.md` (顶部加了 post-5/5 status note)
- **Forward action**: `docs/checkpoints/next_steps.md` §0
- **Strategy notebook**: `docs/checkpoints/paper_planning.md` §19 decision log + §22 multi-register inventory
- **Chronicle pending**: 实验笔记 §110 (待 append, 5/5 sync chronicle)

---

## §E — Maintenance

- 每次 advisor email 回 → 移对应 §C row 到 §A, 加 quote + 落 paper_planning §19 + preregistration decision log
- 全部 §C cleared 后, 本文档 frontmatter `status: rolling` → `status: locked`, 作 paper §1 footnote audit-trail target
- 不写新 entry — 这是 5/5 single-sync outcome 专属 record. 后续 sync 再开 `advisor_sync_<date>_outcomes.md`
