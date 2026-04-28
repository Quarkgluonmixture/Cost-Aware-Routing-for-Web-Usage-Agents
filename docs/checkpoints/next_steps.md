# Next Steps — Action Ledger

> **Daily action plan**. 当前 active state + 接下来 actions only.
>
> **职能分工**:
> - **next_steps.md** (此文档): action ledger (~300 行)
> - **paper_planning.md**: paper strategy notebook (theory / findings / risks / cascade / router / advisor align)
> - **paper drafts** (`docs/analysis/paper_drafts/`): final paper prose
> - **实验笔记** (`docs/checkpoints/实验笔记.md`): time-order chronicle (历史 record)
>
> **Last updated**: 2026-04-28 14:35

---

## §0 TL;DR

**Paper hook**: "Phantom-SoM is hidden 4th routing arm with **4-fold drop-in property** (cost ≈ DOM / latency ~50% / signal AUROC ≥ baseline / drop-one 1.7-3.3pp)"

**Critical path A B0 部分**: ✅ DONE 2026-04-28 (cls + red 5-mode FRESH paper-grade clean)

**Active 跑中**:
- B1 phantom_som cls (PID 371892, GPU contention 慢, ~7-10d ETA)
- B0 dom shopping pilot (PID 893601 + watchdog 988735, FPC disabled, ~9h ETA — 实测 1.36min/task)

**Next 3 actions**:
1. ~Wed 凌晨 01:00: B0 dom shopping pilot done → verify SR/cost/auth → 启 SoM/Vision/Phantom shopping (~$74)
2. ~Wed 中午: 发 codex #10 (axis 2/3 lit deep research, ~600K) + #11 (Section 4 fresh prose, ~30K)
3. ~Thu: 发 codex #13 (Section 5 prose 写, 3-axis hierarchical + lit cite, ~50K)

**Paper progress**: Section 1/2/3 ✅ done (3163 words), Section 4 figures ✅ FRESH + prose 待 codex update, Section 5 evidence 90% (3-axis + 8-channel + bidirectional + §100 ground truth), Section 6/7 待 cross-site/cross-model data.

---

## §1 Active Processes

| PID | Process | Status | ETA |
|---|---|---|---|
| 371892 | B1_phantom_classifieds runner | 跑中 ~16/234, GPU contention | ~7-10 days ⚠️ |
| 893601 | B0_dom_shopping runner (NEW 14:33) | 跑中 task ~75/466, 1.36 min/task | ~9h (~Wed 01:00) |
| 988735 | B0_dom_shopping watchdog (RESTARTED 16:16, was 893429 crashed @ 14:35 ZeroDivisionError fix line 1509) | active, SR 9/75 = 12.0% | continuous |
| 32263, 4124316, 4124482, 370225, 371979 | Watchdog × 5 | per-condition monitor | continuous |
| 3964734 | queue_b1_after_b0 sequencer | sleeping (will trigger after B1 cls done) | continuous |

**Health checks**:
- Magento HTTP 200 (FPC disabled, PowerShell hook 持久化) ✅
- Watchdog auto-clean protocol verified (paper-grade 100% pure) ✅
- DGX defensive curl post-reset metis check ✅

---

## §2 Paper Section Status (8 sections final scope)

| Section | Status | Path | Blocker |
|---|---|---|---|
| 1 Intro | ✅ done (786w + 4-fold drop-in framing) | `docs/analysis/paper_drafts/section1_intro.md` | — |
| 2 Background + paper.bib (16 entries) | ✅ done (1514w) | `section2_background.md` | codex #10 expand to ~38 |
| 3 Definition + Ablation | ✅ done (863w) | `section3_definition.md` | — |
| 4 Empirical Findings | 🟡 80% (figures FRESH ✅, prose stale 1725w) | `section4_empirical_findings.md` | codex #11 fresh prose (~30K) |
| 5 Mechanism (3-axis × 8-channel) | 🟡 90% evidence ready | (待写) | codex #13 prose (~50K, 待 #10) |
| **6 Routing (Tier 1+2)** ⭐ | 🟡 40% (signal AUROC ≥ baseline ✅, scaffold ready) | (待写) | Tier 1 prototype (~3 天) + Tier 2 (~7-10 天) — Week 4-5 |
| 7 Generalization (cross-site/model) | 🟡 40% (B1 profile done) | (待写) | shopping (跑中) + WA + cross-model |
| 8 Discussion (含 sustainability / green AI) | ❌ 未写 | (end-stage) | 全部 data done |

**Strategic content (未 prose 化)** → see `paper_planning.md`:
- §2 Theory framework (3-axis × 8-channel × bidirectional)
- §3 Findings 列表 (10 paper-grade findings, 含 phantom signal AUROC + 4-fold drop-in)
- §4 Section 6 Routing detailed outline + Section 8 sustainability outline ⭐
- §5/6/7 Final scope + 顶刊概率 + risks + cascade
- §8 Router design (Tier 1+2 with 5 决策维度)
- §9 Advisor align checklist
- §10 Visualization plan (4-fig stack)
- §11 Cost / latency / carbon multi-metric plan

---

## §3 Active Experiments + Pending Cells

### 3.1 Critical path A B0 部分 ✅ DONE (2026-04-28)

| Cell | Done time | raw / adj SR | N |
|---|---|---|---:|
| B0 cls DOM | (baseline) | 14.96% / 14.10% | 234 |
| B0 cls SoM | (baseline) | 23.08% / 21.37% | 234 |
| B0 cls Vision | (baseline) | 15.81% / 13.68% | 234 |
| B0 cls Phantom-DOM | FRESH 04-27 | 16.67% / 14.53% | 234 |
| **B0 cls Phantom-SoM** | FRESH 04-28 01:11 | **15.81% / 14.53%** ⭐ | 234 |
| B0 red DOM | (baseline) | 11.43% / 9.52% | 210 |
| B0 red SoM | (baseline) | 11.90% / 10.48% | 210 |
| B0 red Vision | (baseline) | 8.57% / 6.67% | 210 |
| B0 red Phantom-DOM | FRESH 04-28 02:12 | 13.81% / 11.90% | 210 |
| **B0 red Phantom-SoM** | FRESH 04-28 10:28 | **14.29% / 13.81%** ⭐⭐ | 210 |

⭐⭐ red Phantom-SoM 13.81% > SoM 10.48% (within 2σ noise floor, conservative framing per paper Section 1)

### 3.2 Active 跑中

| Cell | Status | ETA |
|---|---|---|
| B1 phantom_som cls | 跑中 ~16/234 | ~7-10 days (GPU contention from seonglae) |
| B0 dom shopping pilot | 跑中 task 0+ | ~16h (~Wed 06:00) |

### 3.3 自动 follow-up (chain triggered)

| Cell | Trigger |
|---|---|
| B1 phantom_dom cls | chain after B1 phantom_som cls done |
| B1 phantom_som + dom red | queue_b1_after_b0: B1 cls done |

### 3.4 等资源 (manual decision)

| Cell | Blocker | Cost |
|---|---|---|
| B0 shopping SoM/Vision/Phantom (4 cells) | wait B0 dom shopping pilot done + verify ~Wed | ~$74 + ~50h API |
| WA × 3 sites × B0+B1 × som+dom (12 cells) | advisor align + B1 chain done | ~$6 + 24h GPU |
| Cross-model Claude Opus 4.7 cls+red 4-mode | advisor align + agent 适配 | ~$70 |
| B1 shopping 5-mode | Myriad GPU + DGX-side B1 phantom done | ~24h GPU 独占 |

### 3.5 Router experiments (Section 6 paper, ~Week 4-5)

| Cell | Blocker | Implementation |
|---|---|---|
| **Tier 1 oracle router prototype** (TF-IDF + LR, 3 天) | baseline + phantom 全 done | `p79/experiment/router.py::RuleBasedRouter` 扩展 |
| **Tier 2 first-step trigger router** (~7-10 天) | Tier 1 done + step-1 trigger feature engineering | 新增 cascade runner config |
| **Routing signal infra** (Phantom modes 已 verified `9d7e99f`) | ✅ ready | `confidence_summary.json` per-condition (5/5 `overall_usable=True`) |
| **Router eval baseline** (random / best-single / rule-based / oracle / learned) | Tier 1+2 done | benchmark suite |

详见 `paper_planning.md` §8 Router design (5 决策维度: feature / target / granularity / cascade / baseline).

### 3.6 Sustainability / Green AI (Section 8 Discussion)

| Item | Status | Source |
|---|---|---|
| fig9 regional carbon sensitivity (B1 only, 45 region) | ✅ done `d3dfc8f` `0cb26c5` | `scripts/analysis/figures/fig9_regional_carbon_sensitivity.py` |
| B1 measured energy (cls + red × DOM/SoM/Vision) | ✅ ready | `condition_summary_v2.json` |
| B0 carbon (proxy API) token-based estimator | ❌ optional Tier 3 | future deep work |
| Section 8 sustainability prose (含 latency 4× + carbon region-dependent) | ❌ 待 codex #17 (paper end-stage) | 待全部 data done |
| paper.bib green AI lit (Strubell 2019, Patterson 2021) | ✅ already in `paper.bib` | — |

---

## §4 Codex Task Queue (pending only)

完成的 task 移除. 历史 done 列表 see `git log --oneline --since="2026-04-26"` or `paper_planning.md` §13.

| # | Task | Tokens | Blocker | Status |
|---|---|---|---|---|
| **10** | **Axis 2/3 literature deep research + paper.bib expansion** (16→~38, 含 bidirectional modality + Tong 2024 Eyes Wide Shut) ⭐⭐⭐ | ~400-600K | — | 🟢 prompt ready, 发 ~Wed |
| **11** | **Section 4 fresh-data prose update** (3-axis framework + cost/latency 4× finding) | ~30K | 等 #10 一起发 | 🟡 ~Wed |
| **13** | **Section 5 prose 写** (3-axis hierarchical + lit inline cite) | ~50K | 待 #10 + #11 done | 🟡 ~Thu |
| 14 | Codex audit shopping VWA (466) | ~500K | 待 shopping 5-mode 数据 | ⏳ ~Week 2-3 |
| 15 | Codex audit WA tasks (480) | ~500K | 待 WA 数据 | ⏳ Week 4-5 |
| **16** | **Section 6 Routing prose** (Tier 1+2 implementation + eval + 4-fig stack) | ~50K | Tier 1+2 prototype done + figures | ⏳ Week 5-6 |
| 17 | Section 7 Generalization 草稿 (cross-site/model) | ~50K | shopping + WA + Claude done | ⏳ Week 6-7 |
| 18 | Section 8 Discussion 草稿 (含 sustainability + 4-fold drop-in summary) | ~30K | 全部 data done | ⏳ Week 8+ |
| 19 | 二次 deep research (Section 6/7/8 + 全 paper revisit, 终稿前) | ~300K | Week 8+ paper 终稿前 | ⏳ Week 8+ |

**Decisions made (don't redo)**:
- 不重做现有 5 codex docs (lazy integrate via Section 5 prose, saves ~1M tokens)
- Phantom-SoM step trace digest 不必要 (#8/#9 已 14 case studies cover)
- §100 SoM probe + Q1/Q2 audit done, 不需重做

---

## §5 Open Issues (active)

| Issue | Status | Action needed |
|---|---|---|
| **B1 phantom_som cls GPU contention** (seonglae 95% utilization, 5× latency) | 🟡 持续 | 联系 seonglae 协调 GPU sharing or 接受 slow progression |
| **B1 shopping DOM 466 ep pre-Magento-bug** (clear+重跑 决策) | 🟡 等 Myriad GPU | rename to `_pre_magento_fix` (保留 reference), 重跑 with FPC disabled |
| **IP env-var-ize 重构** (9 处 `.py/.sh` hardcoded `100.95.81.103`) | 🟡 backlog | 替换为 `${VWA_REMOTE_HOST}` env var read，让 Myriad / future host 不必 sed。文件: `p79/utils/auth_refresh.py` / `external/visualwebarena/browser_env/envs.py` / `scripts/maintenance/{reset_vwa_sites,retry_b1_single_task,experiment_watchdog}.sh\|.py` / `scripts/queues/queue_b{0,1}_{,wa_}with_reset.sh`。**触发条件**: Myriad onboard 时如果不能 Tailscale reach quark IP，先做这个 |
| **Tier A summary commit decision** (是否 commit `condition_summary_v2.json` + `run_meta.json` 入 git LFS / 直 git for paper-grade archive) | 🟡 待评估 | size: 10 conditions × ~50KB = ~500KB total，git 直管也行；好处: paper repro 时 reviewer 不需 hub access；坏处: 实验未冻结前每次 rederive 改动多 |

**Resolved (recent)**:
- ✅ Magento base_url 三次复发 → PowerShell hook + DGX defensive curl 持久化 (`f9cbebf` + quark side)
- ✅ Magento FPC homepage cached guest → `cache:disable full_page` (quark side, persistent via PowerShell hook)
- ✅ Watchdog auto-clean protocol → paper-grade 100% pure verified (no contamination)
- ✅ VWA submodule reproducibility → fork to `Quarkgluonmixture/visualwebarena` p79-patches branch (`e9f7562` / `5ca2c0f`)
- ✅ Cross-host results sync infra → `make rsync-{to,from,artifacts-from}-hub` (Tier B/C separation)

详 `paper_planning.md` §6 (risks + mitigation) for paper-grade execution discipline.

---

## §6 Decisions Pending (advisor align)

详 `paper_planning.md` §9 (full advisor align checklist).

### Meeting #1 quick checklist (~Week 3)

- [ ] Router scope (Tier 1+2 推荐)
- [ ] Cross-model: Claude Opus 4.7 only (~$70)
- [ ] 单 paper integrated (毕设决策)
- [ ] Authorship 预期
- [ ] Investment timing (MLSys 2027 ~9 月)

### Meeting #2 quick checklist (~Week 6-7, after WA + Claude)

- [ ] Paper venue: MLSys (推荐) / NeurIPS workshop / ACL
- [ ] Section 6 范围 (skip Mind2Web 推荐)
- [ ] 投稿 timing (polish 1-2 周后 submit)

---

## §7 References + Quick Links

### Strategy + planning

- **Paper strategy notebook**: `docs/checkpoints/paper_planning.md` ⭐
- **Time chronicle**: `docs/checkpoints/实验笔记.md` (§100 SoM probe, §101.九 Lazy minimization, §103 paper narrative, §104+ daily chronicle)

### Paper drafts (final prose)

```
docs/analysis/paper_drafts/section1_intro.md          ✅ 786w
docs/analysis/paper_drafts/section2_background.md     ✅ 1514w + paper.bib (16 entries)
docs/analysis/paper_drafts/section3_definition.md     ✅ 863w
docs/analysis/paper_drafts/section4_empirical_findings.md  🟡 1725w stale
docs/analysis/paper_drafts/section5_mechanism.md      ❌ 待 codex #13
docs/analysis/paper_drafts/section6_generalization.md ❌ 待 WA+Claude
docs/analysis/paper_drafts/section7_discussion.md     ❌ paper end-stage
```

### Codex analysis docs

```
docs/analysis/phantom_paper/disagreement_clusters.md           (B0+B1 9-cat)
docs/analysis/phantom_paper/cross_site_pattern_consolidation.md (cls vs red shift +50/+33pp)
docs/analysis/phantom_paper/phantom_dom_vs_som_diagnostic.md   (axis 2 prompt)
docs/analysis/phantom_paper/som_vs_phantom_som_diagnostic.md   (axis 3 image 8-channel)
docs/analysis/B1_capability_profile.md                            (B1 cross-model prep)
docs/literature/The Novelty and Efficacy of Set-of-Mark Text...md (deep research, §103 lit gap)
```

### Figures (`results/phantom_paper/figures/`, all FRESH 04-28)

```
fig1 4-mode venn (2x2 B0+B1 cls+red)
fig2 drop-one oracle (2x2)
fig3 strategy gradient (2x4 reddit + cls)
fig4 two-knob diagram schematic
fig5 category × mode heatmap (B0 cls+red)
fig6 capability contrast B0-vs-B1
fig7 cost-SR Pareto + deployment callouts
fig8 overlap-depth stacked bar (5-mode)
fig9 regional carbon sensitivity (B1 only)
```

### Recent commits

`git log --oneline --since="2026-04-26"`

### Key infra paths

```
configs/exp_v2_*.yaml                 (per-site experiment configs)
scripts/queues/queue_phantom*.sh       (chain orchestration)
scripts/maintenance/reset_vwa_sites.sh (DGX→quark PowerShell reset + defensive curl)
scripts/maintenance/experiment_watchdog.py (auto-clean + post-condition pipeline)
p79/utils/auth_refresh.py              (Playwright sign-in subprocess)
p79/experiment/router.py               (RuleBasedRouter scaffold)
```

---

## §8 What I'd say in a 5-min advisor report (pre-canned)

> "Critical path A B0 部分已完成 — cls + red 5-mode (DOM/SoM/Vision/Phantom-DOM/Phantom-SoM) 全部 fresh paper-grade clean (watchdog auto-clean 协议保证 100% pure data, no contamination).
>
> Paper hook 升级到 4-fold drop-in property: Phantom-SoM cost ≈ DOM (regex filter), latency ~50% lower, signal AUROC ≥ baseline (router infra 复用), drop-one oracle 1.7-3.3pp.
>
> Theory framework 落地到 3-axis hierarchical (representation × prompt × image), 含 8-channel image taxonomy + bidirectional modality. 2 个 codex mechanism diags (axis 2 + axis 3) verified, 14 case studies + §100 ground truth (occlusion -60pp OCR + numeric hijack 0→446 num_ids).
>
> Section 1/2/3 paper drafts 已写 (3163 words). Section 4 figures FRESH (9 PNGs), prose 待 codex update. Section 5 evidence 90% ready, prose 待 codex 写.
>
> 现在 B0 dom shopping single-mode pilot 跑中 (~16h, FPC disabled), B1 phantom_som cls 慢 (GPU contention).
>
> Next 1-2 周: shopping baseline + phantom 5-mode (~$74), Section 4-5 prose update via codex (~3 prompts).
>
> 决策待 align: router scope (Tier 1+2 推荐), Claude Opus budget (~$70), 投稿 timing (MLSys round 1 推荐, first-paper friendly venue)."

---

## §9 学长 Onboarding Reading Order

学长第一次读 paper / 进入 paper context 的推荐 reading order:

### Quick path (~30 min)

1. **(5 min)** `next_steps.md` §0 TL;DR + §8 5-min advisor pre-canned report
   → 当前 status overview + paper hook
2. **(5 min)** `paper_planning.md` §1 Hook + Tagline + §3 Findings 列表
   → 4-fold drop-in property + 10 paper-grade findings
3. **(10 min)** `docs/analysis/paper_drafts/section1_intro.md`
   → Paper intro prose (786w, 含 4-fold drop-in framing)
4. **(5 min)** Browse figures: fig1 venn + fig7 cost-SR + fig8 unique
   → `results/phantom_paper/figures/fig{1,7,8}*.png`
5. **(5 min)** `paper_planning.md` §6 Risks + §9 Advisor align checklist
   → 顶刊 risks 主轴 + advisor align meeting #1 checklist

### Deep dive (optional, ~60 min)

- `paper_planning.md` §2 theory framework (3-axis × 8-channel × bidirectional × LLM rationale)
- `paper_planning.md` §14 reviewer attack 预案 + §15 prior work comparison table
- `paper_drafts/section3_definition.md` (Phantom-SoM 定义 + 2x2 ablation + token re-estimate)
- `paper_drafts/section4_empirical_findings.md` (1725w stale, 数据结构理解, 待 codex update)
- 实验笔记 §103 paper narrative + §100 SoM probe ground truth + §104 chronicle

### Codex analysis docs (paper Section 5 evidence base)

- `docs/analysis/phantom_paper/phantom_dom_vs_som_diagnostic.md` (axis 2 prompt)
- `docs/analysis/phantom_paper/som_vs_phantom_som_diagnostic.md` (axis 3 image 8-ch)
- `docs/analysis/phantom_paper/cross_site_pattern_consolidation.md` (cls vs red shift)
- `docs/analysis/B1_capability_profile.md` (Section 7 cross-model prep)

### Decision audit trail

- `paper_planning.md` §19 Decision log (paper-strategic decisions + timestamp)

---

## §10 Doc Update Workflow (新数据 → docs)

> 当获得新 data / new finding / new decision 时, 该 update 哪些 docs.

### A. 新 condition 数据 (e.g. B1 phantom_som cls done)

```
✅ 实验笔记: append chronicle entry (§104+ daily, Day-by-day finding)
✅ next_steps §1 Active processes: mark done (remove or move to §3.1 done table)
✅ next_steps §3.1 done table: add row (raw/adj SR + N + done time)
✅ Run `make figures` (auto-regen fresh data, 9 figures)
🟡 next_steps §0 TL;DR: update if Critical path A 进度变化
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
✅ next_steps §2 paper section status: status update (drafted)
✅ paper_planning §4 paper section status: same
🟡 paper_planning §2/§3 strategy notes: shrink (move to drafts now in prose)
✅ paper_planning §17 pre-submission checklist: tick off content completeness items
❌ next_steps §0 TL;DR: 不变 (除非 strategic shift)
```

### E. 新 decision (e.g. advisor align meeting #1 outcome)

```
✅ paper_planning §19 decision log: append timestamped row
✅ paper_planning §9 advisor align checklist: tick off items
✅ next_steps §6 decisions pending: remove resolved items
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
🟡 next_steps §0 TL;DR: if changes paper hook
```

### H. 新 reviewer attack scenario discovered

```
✅ paper_planning §14 reviewer attack: add row
✅ paper drafts (when writing prose): proactive defense in Section 4-5
```

### I. 新 paper section prose done

```
✅ next_steps §2 paper section status: status drafted
✅ paper_planning §4 same
✅ paper_planning §13 pending TODO: tick off
🟡 paper_planning §3 finding: shrink (now in prose) or expand (new finding from prose writing)
🟡 paper_planning §17 pre-submission checklist: tick off content items
```

### General principle

- **Daily update**: next_steps.md (active state, codex queue, open issues)
- **Weekly update**: paper_planning.md (strategy notebook, when finding/decision emerges)
- **Append-only**: 实验笔记.md (chronicle, never overwrite, append §)
- **Stable until prose write**: paper drafts (only update when Section X prose batch written)

### Quick mental check before update

```
What changed? → 找对应类型 (A-I)
Mark current status? → next_steps
Add new strategic finding? → paper_planning
Record what happened (history)? → 实验笔记 append §
Modify final paper text? → paper drafts (only when prose batch writing)
```

