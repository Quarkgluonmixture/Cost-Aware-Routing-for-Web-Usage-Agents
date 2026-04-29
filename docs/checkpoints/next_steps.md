# Next Steps — Action Ledger

> **Daily action plan**. 当前 active state + 接下来 actions only.
>
> **职能分工**:
> - **next_steps.md** (此文档): action ledger (~430 行)
> - **paper_planning.md**: paper strategy notebook (theory / findings / risks / cascade / router / advisor align)
> - **paper drafts** (`docs/analysis/paper_drafts/`): final paper prose
> - **实验笔记** (`docs/checkpoints/实验笔记.md`): time-order chronicle (历史 record)
>
> 📖 **新数据/figure/finding/codex 回复 → 该更新哪些文档？** see §10 Doc Update Workflow
> 🔧 **新数据后一键 snapshot**: `make analyze-paper` (~5-10min, **everything**):
>   1. per-run on 8 paper-grade VWA runs: rederive + reason-diag + cross-rep + confidence
>   2. B0 vs B1 site comparison (cls + red, `b0_vs_b1_<site>/`)
>   3. cross-condition aggregations: aggregate-cross-site + summary-collect + routing-auroc
>   4. 9 figures (含 fig2 bootstrap CI)
>
> **NOT** included (intentional): GLM digest sidecar + gallery + annotate (watchdog handles); codex narrative analyses (manual); narrow ad-hoc diagnostics (selflink_loop / vision_coordinate / search_over_browse / diag_pattern_match — invoke per-need)
>
> **Last updated**: 2026-04-29（§105 swatch radio 漏检 fix）

---

## §0 TL;DR

**Paper hook**: "Phantom-SoM is hidden 4th routing arm with **4-fold drop-in property** (cost ≈ DOM / latency ~50% / signal AUROC ≥ baseline / drop-one 1.7-3.3pp)"

**Critical path A B0 部分**: ✅ DONE 2026-04-28 (cls + red 5-mode FRESH paper-grade clean)

**🆕 4-Layer Evidence Framework 落地 (2026-04-29 §106)**: paper_planning §3 重组 + 13 figures rename layer-prefix + Makefile `analyze-layered` 一键 pipeline + `layered_evidence_status.md` live status report. 详见 实验笔记 §106.

**🆕 Cost methodology fix (~100× deployment-class gap)**: §103 legacy "30×" claim **已 superseded**。real ratio 用 electricity-equivalent ($0.12/kWh) 算: B0 API ~$0.04/ep vs B1 local ~$0.0004/ep → **reddit 98× / cls 105×**。`cost_per_mode.md` standalone aggregator + fig3d log-scale Pareto with annotation.

**🆕 P-prompt mode 加入设计 (Diamond ablation)**: P-text 已显著 → P-prompt (AXTree+SoM-prompt+无图) 是 symmetric counterpart, paper Section 5 必需用于 disentangle prompt × text 交互。Code + yaml + queue 已就位; B0 P-prompt reddit 跑中（PID 2075552, ~3-4h ETA）。

**🆕 B0 5-mode reddit + cls 全 done (10/10 cells); B1 cls phantom_som 04-29 17:43 done (4/5 cls cells, P-text pending)**

**🔴 04-29 18:15 same-site collision incident**: queue_chain.sh 自动 advance B1 phantom cls→reddit 时未 check B0 P-prompt reddit 已经在跑 (paper-grade hard rule violation)。30min 后发现 + 立即处理：kill chain (PID 1145483) + B1 phantom reddit runner+watchdog (2244780/2244882) + clear 7 contaminated reddit_task_0-6 episodes via `clear_tasks.py`. B0 P-prompt reddit 完好保留 (PID 2075552, 51/210 progress). Fix: queue_chain.sh 加 `same-site collision check` (commit pending), 自动 wait 对方 baseline 完事再 launch.

**🆕 site_mechanism_dictionary.md done (codex 04-29 17:43)**: 30KB markdown + 30KB JSON, 3 sites × 3 axes × 6 fields，Section 5 prose codex (#13) lookup target. Cross-site invariant: P-SoM Jaccard ≤0.7 sentinel both red/cls + positive single-phantom oracle lift.

**Active 跑中**:
- **B0 P-prompt reddit (PID 2075552, 51/210)** — symmetric ablation 完整保留, ~3-4h ETA → 跑完 fig1ab P-prompt cell auto-fill
- B0 dom shopping clean re-run (PID 1106560, ~9h ETA)
- ~~B1 phantom_som cls (PID 1821957)~~ ✅ done 17:43 (234/234)
- ~~B1 phantom_som reddit (PID 2244780)~~ 🔴 killed 18:15 (collision); contaminated tasks 0-6 cleared
- ~~queue_chain B1 phantom (PID 1145483)~~ 🔴 killed 18:15

**Next 3 actions**:
1. **B0 P-prompt cls** — 等 B1 phantom cls 完事再启（同 site B0 XOR B1 hard rule）
2. **Codex Section 4 prose update** (~30K codex job): 用 4-Layer framework + ~100× cost + N=210 全数据替换 §103 N=48 + 30× narrative
3. **Codex Section 5 mechanism prose** (~50K): 等 P-prompt reddit 跑完 + diamond ablation 数据 ready 后做

**Paper progress**: Section 1/2/3 ✅ done (3163 words), Section 4 figures ✅ FRESH (12 layer-prefixed PNGs) + bootstrap CI ✅ + prose **stale 1725w** 待 codex #11 update, Section 5 evidence **95%** (4-Layer framework + 3-axis cascade diamond + 6 antagonistic pairs + cross-site micro), Section 6 AUROC ≥ baseline ✅, Section 7/8 待 cross-site/cross-model data + multi-metric Pareto.

---

## §1 Active Processes

| PID | Process | Status | ETA |
|---|---|---|---|
| **2075552** | **B0_phantom_prompt_reddit runner** (NEW 04-29 14:35, symmetric ablation for diamond) | 跑中 task ~16/210 → 跑完 fig1ab cell auto-fill | ~6h |
| 2075658 | B0_phantom_prompt_reddit watchdog | active | continuous |
| ~~371892~~ → 1821957 | B1_phantom_classifieds runner (restarted 04-29 09:02 — old PID 371892 stalled 4h due to GPU contention deadlock with seonglae 5 train jobs) | 跑中 task 184/234 → resume 185+, SR 9.7% (17/175); 04-29 watchdog auto-cleaned 5 NOT-LOGGED-IN tasks (0/1/2/3/174) + verified resume | ~10-15d ⚠️ (GPU contention) |
| 1106560 | B0_dom_shopping runner (clean re-run, RESET_BEFORE=1, queue_baseline.sh) | 跑中 task=0+, fresh run_id `B0_dom_shopping_20260428` | ~9h |
| 1106686 | B0_dom_shopping watchdog | active | continuous |
| 32263, 4124316, 4124482, 371979 | Watchdog × 4 (B0 phantom history + B1 cls active) | per-condition monitor | continuous |
| 1145483 | queue_chain B1 phantom 4-cell sequencer | wait B1 cls done → chain 3 next | continuous |

**Health checks**:
- Magento HTTP 200 (FPC disabled, PowerShell hook 持久化) ✅
- Watchdog auto-clean protocol verified (paper-grade 100% pure) ✅
- DGX defensive curl post-reset metis check ✅
- ✅ B0 dom shopping clean re-run launched (was stopped 17:05 due to no-reset audit; old dirty dir 移 `_archive/`)

---

## §2 Paper Section Status (8 sections final scope)

### Final scope (paper 完整版, 详见 `paper_planning.md` §5)

```
Benchmark: VWA 3 站 (cls 234 + red 210 + shop 466) + WA 3 站 (red 106 + shop 192 + sa 182)
           = 6 sites, ~1390 task per condition
Models:    B0 (Qwen3-VL-235B proxy) + B1 (Qwen3-VL-4B local) + Claude Opus 4.7
           = 3 model families
Modes:     DOM / SoM / Vision / P-text / P-SoM (5 base) + P-prompt (diamond control, NEW 04-29)
           = 6 modes (5 paper-headline + 1 mechanism control)
Cells:     6 sites × 3 models × 6 modes = ~108 cells (~135K episodes)
           Tier 1 critical: 5 base modes everywhere
           Tier 2 diamond:  P-prompt only B0 cls+red (Section 5 mechanism), B1 cls+red bonus (Section 7)
+ Router:  Tier 1+2 (oracle TF-IDF+LR / first-step trigger), 实际 deploy on agent,
           measure cost / SR / latency vs best-single-mode
+ Multi-metric: cost / P95 latency / carbon (B1 measured + B0 estimate)
+ Cost class:  B0 = API token $ (~$0.04/ep); B1 = electricity equivalent (~$0.0004/ep);
               ~100× deployment-class gap (~Section 8 sustainability)
```

⚠️ Cells 数实际会少一些 — P-text 在某些 site/model 计划砍掉（control 价值 vs cost：P-SoM 是 hidden 4th routing arm 必跑，P-text 只在 axis 2 prompt-effect 论证关键 site/model 跑）。**P-prompt 只在 cls+red 跑（mechanism Section 5 用），shopping/WA P-prompt 不必**。具体 final cell selection 见 `paper_planning.md` §3 重组后的 layered framework + §5 final scope。

### Section status table

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

### 3.1 Critical path A B0 部分 (5-mode 已 DONE; **diamond P-prompt 跑中 04-29**)

**Naming convention (paper-facing, §106)**: 5 modes = DOM / SoM / Vision / **P-text** (= phantom_dom) / **P-SoM** (= phantom_som). **+ P-prompt** (= phantom_prompt, NEW 04-29) for diamond completion.

| Cell | Done time | raw / adj SR | N |
|---|---|---|---:|
| B0 cls DOM | (baseline) | 14.96% / 14.10% | 234 |
| B0 cls SoM | (baseline) | 23.08% / 21.37% | 234 |
| B0 cls Vision | (baseline) | 15.81% / 13.68% | 234 |
| B0 cls **P-text** | FRESH 04-27 | 16.67% / 14.53% | 234 |
| B0 cls **P-SoM** | FRESH 04-28 01:11 | **15.81% / 14.53%** ⭐ | 234 |
| **B0 cls P-prompt** | ⏳ pending (B1 phantom cls done 后启) | — | 234 |
| B0 red DOM | (baseline) | 11.43% / 9.52% | 210 |
| B0 red SoM | (baseline) | 11.90% / 10.48% | 210 |
| B0 red Vision | (baseline) | 8.57% / 6.67% | 210 |
| B0 red **P-text** | FRESH 04-28 02:12 | 13.81% / 11.90% | 210 |
| B0 red **P-SoM** | FRESH 04-28 10:28 | **14.29% / 13.81%** ⭐⭐ | 210 |
| **B0 red P-prompt** | 🟢 跑中 (PID 2075552, 16/210, ~6h ETA) | — | 210 |

⭐⭐ red P-SoM 13.81% > SoM 10.48% (within 2σ noise floor, conservative framing per paper Section 1)

**Diamond ablation status**: B0 reddit diamond ~6h after P-prompt done; B0 cls diamond blocked by B1 phantom cls (same-site B0 XOR B1 hard rule).

### 3.2 Active 跑中

| Cell | Status | ETA |
|---|---|---|
| **B0 P-prompt reddit (PID 2075552, NEW 04-29)** | 跑中 ~16/210 | ~6h |
| B1 phantom_som cls (PID 1821957) | 跑中 task 184+/234, watchdog auto-cleaned 5 NOT-LOGGED-IN tasks | ~10-15 days (GPU contention) |
| B0 dom shopping clean re-run (PID 1106560, with reset) | 跑中 ~9/466 | ~9h (~Wed 01:00) |

### 3.3 Pending chains — automated via `queue_chain.sh` (sequential)

B1 4B 单 GPU instance + B0 phantom shopping 同 site exclusive → 必须 sequential. 用 `queue_chain.sh` 自动 chain 一组 cells (idempotent 检测 already-running, paper-grade RESET_BEFORE=1 默认):

**Tier 1 (paper-critical, B0 diamond + B1 5-mode)**:
```bash
# B0 P-prompt cls (after B1 phantom_som cls done — same-site B0 XOR B1)
nohup bash scripts/queues/queue_chain.sh \
  "queue_phantom_prompt.sh B0 classifieds" \
  > logs/queue_chain_b0_pprompt_cls.log 2>&1 &

# B1 phantom 4-cell chain (cls already running PID 1821957, chain wait + 3 sequential)
# original 4-cell still active (P-SoM cls/red + P-text cls/red)
nohup bash scripts/queues/queue_chain.sh \
  "queue_phantom_som.sh B1 classifieds" \
  "queue_phantom_som.sh B1 reddit" \
  "queue_phantom_dom.sh B1 classifieds" \
  "queue_phantom_dom.sh B1 reddit" \
  > logs/queue_chain_b1_phantom.log 2>&1 &

# B0 phantom shopping pair (after B0 dom shopping done)
nohup bash scripts/queues/queue_chain.sh \
  "queue_phantom_som.sh B0 shopping" \
  "queue_phantom_dom.sh B0 shopping" \
  > logs/queue_chain_b0_phantom_shop.log 2>&1 &
```

**Tier 2 (diamond completeness, B1 P-prompt — Section 7 cross-capability bonus)**:
```bash
# B1 phantom_prompt 2-cell chain (yamls created 04-29, run after Tier 1 B1 chain done)
nohup bash scripts/queues/queue_chain.sh \
  "queue_phantom_prompt.sh B1 classifieds" \
  "queue_phantom_prompt.sh B1 reddit" \
  > logs/queue_chain_b1_pprompt.log 2>&1 &
```

| Chain | ETA | Status |
|---|---|---|
| **B0 P-prompt cls (Tier 1, NEW)** | ~6h | ⏳ wait B1 phantom_som cls done (~10d) |
| B1 phantom 4-cell (cls som → red som → cls dom → red dom) | ~30-40d (GPU contention 7-10d/cell) | ✅ launched 19:11 PID 1145483 (cls active 1821957) |
| B0 phantom shopping pair (som → dom) | ~24h | wait B0 dom shopping done (~Wed 01:00) |
| **B1 P-prompt 2-cell (Tier 2, NEW)** | ~14-20d | ⏳ wait B1 4-cell Tier 1 done — paper Section 7 cross-capability diamond, not Section 5 critical path |

**Paper-impact triage**:
- **Section 5 (mechanism)**: needs **B0 reddit diamond** (P-text + P-prompt + P-SoM + endpoints) → ~6h后 ready
- **Section 5 cls reinforcement**: needs **B0 cls diamond** → blocked by B1 cls (~10d ETA)
- **Section 7 (generalization)**: B1 5-mode 现 4-cell chain done 后即可 (Tier 2 P-prompt 是 nice-to-have, paper 写 "B1 cross-capability supports B0 mechanism" 即可)

### 3.4 Missing cells — 等资源 (manual decision)

**Updated matrix 04-29 (含 P-prompt diamond)**: 5 modes + 1 diamond control (P-prompt) = **6 cells per (baseline × site)**. Total target: 6 sites × 3 models × 6 modes = 108 cells (~135K episodes; B1 P-prompt + Claude all-modes 是 Tier 2)。

数据 picture (post 04-29):
- VWA cls: B0 5/6 done (P-prompt missing) + B1 1/6 done + 1 跑中 (P-SoM) + 4 missing
- VWA red: B0 5/6 done + 1 跑中 (P-prompt) + B1 1/6 done + 5 missing
- VWA shop: 1/12 跑中 (B0 DOM) + 11 missing
- WA: 0/30 (B0+B1 × 3 sites × 5 modes; +6 if including P-prompt = 36)

**Critical (paper Section 5)**:
| Cell | Blocker | Cost |
|---|---|---|
| **B0 P-prompt cls** (diamond completion) | B1 phantom_som cls done | ~$3 + ~6h API |
| B0 shopping SoM/Vision/Phantom × 3 (3 cells) | B0 dom shopping done + verify | ~$45 + ~24h API |
| B0 shopping P-prompt | B0 phantom shopping pair done (Tier 2) | ~$15 + ~6h API |

**Generalization (paper Section 7)**:
| Cell | Blocker | Cost |
|---|---|---|
| WA × 3 sites × B0+B1 × 5 modes (30 cells; +6 P-prompt opt = 36) | advisor align + B1 VWA chain done | ~$60-72 + 60-72h GPU |
| Cross-model Claude Opus 4.7 cls+red 5-mode (10 cells; +2 P-prompt opt) | advisor align + agent 适配 | ~$100-120 |
| B1 shopping 5-mode + P-prompt (6 cells) | Myriad GPU + DGX-side B1 phantom 全 done | ~24h GPU 独占 each |
| **B1 P-prompt cls + red (Tier 2 diamond cross-cap, NEW yamls 04-29)** | B1 P-SoM + P-text 4-cell chain done | ~14-20d GPU (queued in chain) |
| ~~B1 shopping DOM 466 ep (pre-Magento-bug)~~ | ✅ archived 04-28，待 Myriad clean re-run | — |

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
| fig9 regional carbon sensitivity (B1 only, 45 region) | ✅ done `d3dfc8f` `0cb26c5` | `scripts/analysis/figures/fig3_regional_carbon.py` |
| B1 measured energy (cls + red × DOM/SoM/Vision) | ✅ ready | `condition_summary_v2.json` |
| B0 carbon (proxy API) token-based estimator | ❌ optional Tier 3 | future deep work |
| Section 8 sustainability prose (含 latency 4× + carbon region-dependent) | ❌ 待 codex #17 (paper end-stage) | 待全部 data done |
| paper.bib green AI lit (Strubell 2019, Patterson 2021) | ✅ already in `paper.bib` | — |

---

## §4 Codex Task Queue (pending only)

完成的 task 移除. 历史 done 列表 see `git log --oneline --since="2026-04-26"` or `paper_planning.md` §13.

**🆕 Recently completed (04-29)**:
- ✅ codex axis_effect_size_ablation (3-axis cascade, 47K tok)
- ✅ codex axis_effect_size_ablation_v2 (cascade decomposition + consistency check, 72K tok)
- ✅ codex axis1_microbehavior cross-site (verdict generalizes red 2.28 / cls 1.02, 80K tok)
- ✅ codex shopping A-refine (A1/A2/A3/A4 sub-classification, 50K tok)
- ✅ codex layered refactor (Layer{0,1,2,3} make targets + README + status report, 202K tok)
- ✅ codex fix figures (fig4 cascade diamond + fig7 P-SoM + fig1 5-mode + fig12 + sr_fp aggregator, 145K tok)
- ✅ codex rename figures (12 figs → layer-prefixed + fig3d cost source fix log-scale 100×, 371K tok)

| # | Task | Tokens | Blocker | Status |
|---|---|---|---|---|
| **10** | **Axis 2/3 literature deep research + paper.bib expansion** (16→~38, 含 bidirectional modality + Tong 2024 Eyes Wide Shut) ⭐⭐⭐ | ~400-600K | — | 🟢 prompt ready, 发 ~Wed |
| **11** | **Section 4 fresh-data prose update** (4-Layer framework + ~100× cost framing + N=210 全数据) ⭐ priority next | ~30K | — (现可发) | 🟢 ready, 等用户 trigger |
| **13** | **Section 5 prose 写** (organize as **site × axis × LLM-mechanism 3-way table**, per `paper_planning §2.x site mechanical substrate`); primary input: 9 site digests `docs/analysis/vwa_*/B*_{DOM,SoM,Vision}_digest.md` + `mechanism_per_task.{json,md}` (E1-E4) + `axis_effect_size_report.md` + `disagreement_clusters.md` (04-27 stale, refresh via #14c when phantom data ready) + `axis1_microbehavior_report.md` + cls task-pool 0.53 paradox + 6 antagonistic pairs | ~50-80K | 待 P-prompt reddit done (~6h) + #11 一起发 | 🟡 ~Thu |
| 14 | Codex audit shopping VWA (466) — full 5-mode | ~500K | 待 shopping 5-mode 数据 | ⏳ ~Week 2-3 |
| 14b | Codex audit reddit cat refine (类 shopping A-refine 04-29) | ~50K | red 5-mode FRESH already | 🟢 ready |
| **14c** | **Codex redo `disagreement_clusters.md` 含 5+1 phantom modes** (mechanism narrative for paper Section 5; current md is 04-27 baseline-only snapshot, drives `fig_capability_b0_b1`) | ~80K | B1 phantom 4-cell + B0 P-prompt diamond done (~30-40d) | ⏳ Week 4-5 |
| 15 | Codex audit WA tasks (480) | ~500K | 待 WA 数据 | ⏳ Week 4-5 |
| **16** | **Section 6 Routing prose** (Tier 1+2 implementation + eval + 4-fig stack) | ~50K | Tier 1+2 prototype done + figures | ⏳ Week 5-6 |
| 17 | Section 7 Generalization 草稿 (cross-site/model) | ~50K | shopping + WA + Claude done | ⏳ Week 6-7 |
| 18 | Section 8 Discussion 草稿 (含 sustainability + 4-fold drop-in summary + ~100× cost) | ~30K | 全部 data done | ⏳ Week 8+ |
| 19 | 二次 deep research (Section 6/7/8 + 全 paper revisit, 终稿前) | ~300K | Week 8+ paper 终稿前 | ⏳ Week 8+ |

**Decisions made (don't redo)**:
- 不重做现有 5 codex docs (lazy integrate via Section 5 prose, saves ~1M tokens)
- Phantom-SoM step trace digest 不必要 (#8/#9 已 14 case studies cover)
- §100 SoM probe + Q1/Q2 audit done, 不需重做

### Data analysis backlog (Python scripts, 非 codex)

paper_strategic 数据 pipeline (详 `paper_planning.md` §13.B)。一键 `make analyze-paper` 触发现有 done 项 + figures 重生。

| Task | 用途 | Status |
|---|---|---|
| ✅ A. Bootstrap CI for drop-one oracle | Section 4 显著性 | done `847eaeb` — `fig0c_drop_one_oracle.py` + `fig0c_drop_one_bootstrap_ci.csv` (12 rows × 95% CI) |
| ✅ B. AUROC aggregation per-condition routing signal | Section 6 "AUROC ≥ baseline" supporting table | done `847eaeb` — `aggregate_routing_auroc.py` → `auroc_cross_condition.{csv,md,_summary.md}` (188 rows) |
| ✅ **B'. Phantom routing lift** (3→5-mode oracle ceiling) | **Section 1/4 paper hook 主 evidence**: B0 cls +4.70pp [2.14, 7.69] / B0 red +5.24pp [2.38, 8.11] CI 排除 0 ✅ | done 04-29 — `aggregate_phantom_lift.py` → `phantom_lift.{csv,md}`; chained into `make analyze-paper`; B1 cells 自动 cover when chain done. **Cohen's h + Wilcoxon + McNemar + Scenario C Jaccard sentinel 全 added** |
| ✅ **B''. fig10 phantom_lift_bars** | Section 1/4 hook visualization (5-mode oracle ceiling bar chart with bootstrap CI + lift Δ annotation) | done 04-29 — `figures/fig0c_phantom_lift_bars.py`; in `make figures` chain |
| ✅ **B'''. fig11 routing_auroc_heatmap** | **Section 6 main figure (之前 0 figure)** — cross-condition × signal AUROC heatmap (★ = ≥ 0.7 routing-usable) | done 04-29 — `figures/fig0g_routing_auroc_heatmap.py`; in `make figures` chain |
| ⏳ C. Multi-metric Pareto (cost + lat + carbon) | Section 8 sustainability 前置 | 待 (~2h, fig10 new + carbon estimator integration) |
| ⏳ D. TF-IDF + binary feature extraction | Section 6 Tier 1 router 前置 | 待 (~1h, extend `r1_task_features` in `analyze_cross_representation.py`) |
| ⏳ E. B0 token-based carbon estimator | Section 8 Tier 3 sustainability | 待 (~20 行 helper in `p79/experiment/metrics.py`) |

**`make analyze-paper`** 一键 chain：`aggregate-cross-site` (cross-site SR/cost/lat/energy CSV+plots) + `summary-collect` (run_summary_collect.json) + `routing-auroc` (cross-condition AUROC table) + `figures` (9 PNGs 含 fig2 bootstrap CI). 跑前 paper-grade snapshot 一键生成 — codex #11/#13 prose 写之前 invoke。

---

## §5 Open Issues (active)

| Issue | Status | Action needed |
|---|---|---|
| **B1 phantom_som cls GPU contention** (seonglae 95% utilization, 5× latency) | 🟡 持续 | 联系 seonglae 协调 GPU sharing or 接受 slow progression |
| ~~**B1 shopping DOM 466 ep pre-Magento-bug**~~ | ✅ resolved 04-28 19:00 | archived `_archive/B1_3mode_shopping_20260413_pre_magento_bug` (含 dom 465/466) + som 5ep abandoned condition 删除. 待 Myriad GPU clean re-run via `queue_baseline.sh B1 dom shopping` |
| **IP env-var-ize 重构** (9 处 `.py/.sh` hardcoded `100.95.81.103`) | 🟡 backlog | 替换为 `${VWA_REMOTE_HOST}` env var read，让 Myriad / future host 不必 sed。文件: `p79/utils/auth_refresh.py` / `external/visualwebarena/browser_env/envs.py` / `scripts/maintenance/{reset_vwa_sites,retry_b1_single_task,experiment_watchdog}.sh\|.py`。**触发条件**: Myriad onboard 时如果不能 Tailscale reach quark IP |
| **WA reset mechanism**（queue scripts 当前仅 vwa reset, wa skip） | 🟡 backlog | webarena docker reset 路径未实现。需写 `reset_wa_sites.sh` (类似 `reset_vwa_sites.sh`) 然后 queue scripts 加 `BENCHMARK=wa` 分支调用 |
| **Watchdog AUTO-ANALYSIS spam guard** (partial condition_summary 触发 infinite loop, §104 Day 3 04:00 audit) | 🟡 backlog | `experiment_watchdog.py:1340` `condition_completed = condition_summary_v2.json.exists()` 应增加 episode count vs expected_episodes guard，避免 partial 数据 (e.g. 165/234 ep) 触发 Case 3 re-trigger loop. 当前 workaround: 不要在 in-flight run 上跑 `make rederive RUN=...` |
| **GPU contention deadlock detection** (B1 cls stalled 4h 04-29 05:01-09:00, seonglae 5 train jobs + StreamWriter 8.5GB hogged GPU 95%) | 🟡 backlog | runner state R but ep_pol kernel wait → no progress detection in watchdog. Need watchdog stale-runner heartbeat: 检测 episode mtime 超过 N 分钟无更新 → 自动 SIGTERM runner + queue script idempotent re-spawn. 当前 workaround: manual stop+restart `bash scripts/queues/queue_phantom_som.sh B1 classifieds` |
| **Tier A summary commit decision** (是否 commit `condition_summary_v2.json` + `run_meta.json` 入 git LFS / 直 git for paper-grade archive) | 🟡 待评估 | size: 10 conditions × ~50KB = ~500KB total，git 直管也行；好处: paper repro 时 reviewer 不需 hub access；坏处: 实验未冻结前每次 rederive 改动多 |

**Resolved (recent)**:
- ✅ **§107 (04-29 18:15) queue_chain.sh same-site collision fix** → B1 phantom auto-advance reddit 撞 B0 P-prompt reddit (30min cross-contam, cleaned). Patch added pre-launch collision check waits for opposite baseline to finish before proceeding.
- ✅ **§106 (04-29) 4-Layer Evidence Framework + cost methodology fix + figure rename** → paper_planning §3 重组，13 figures rename layer-prefix，cost ~100× via electricity equivalent (legacy 30× superseded), `make analyze-layered` pipeline, `layered_evidence_status.md` live status. 详见 实验笔记 §106.
- ✅ **§105 (04-29) Magento custom-option radio swatch 漏检** → `state_change.py:_key` 加 value discriminator；同 bug 影响 review form ratings；B0 dom shopping 11/465 ep 受影响（全 fail，9/11 cycle 早停）；DOM/SoM 共享受影响（Vision 不受）；详见 `docs/analysis/cross_sites/swatch_form_change_audit.md`
- ✅ Magento base_url 三次复发 → PowerShell hook + DGX defensive curl 持久化 (`f9cbebf` + quark side)
- ✅ Magento FPC homepage cached guest → `cache:disable full_page` (quark side, persistent via PowerShell hook)
- ✅ Watchdog auto-clean protocol → paper-grade 100% pure verified (no contamination)
- ✅ VWA submodule reproducibility → fork to `Quarkgluonmixture/visualwebarena` p79-patches branch (`e9f7562` / `5ca2c0f`)
- ✅ Cross-host results sync infra → `make rsync-{to,from,artifacts-from}-hub` (Tier B/C separation)

**Pending paper-grade re-run (积累中，全部修完后一次性跑)**:
- ⏳ B0/B1 × DOM/SoM × shopping ✗ Vision — 用户在 dom shopping 上挨个 debug 失败 task，bugs 累计修完后统一 re-run
  - 已 confirmed 影响：§105 swatch radio (DOM+SoM, Vision 不受)
  - 触发：用户 stop debug → 一次性 launch 5 cells（B0 dom/som/vision + B1 dom/som）

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

> "Critical path A B0 部分已完成 — cls + red 5-mode (DOM/SoM/Vision/P-text/Phantom-SoM) 全部 fresh paper-grade clean (watchdog auto-clean 协议保证 100% pure data, no contamination).
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
✅ Run `make analyze-paper` (one-shot **everything**, ~5-10min):
   - per-run pipeline (rederive + reason-diag + cross-rep + confidence) on
     all 8 paper-grade VWA runs (override `RUN_DIRS_PAPER_VWA` if needed)
   - B0 vs B1 site comparison (cls + red → `b0_vs_b1_<site>/`)
   - cross-condition aggregations: aggregate-cross-site + summary-collect +
     routing-auroc
   - 9 figures (含 fig2 bootstrap CI)
   ↳ Quick path: `make figures` (~10s, 仅 fig regen, 不含 per-run + agg)
   ↳ Debug path: 单独 `make analyze-paper-per-run` / `compare-b0-b1-all` /
     `aggregate-cross-site` / `summary-collect` / `routing-auroc`
   ↳ 输出 path: `results/phantom_paper/{auroc_cross_condition.*, cross_site/,
     run_summary_collect.json, figures/*.png + fig0c_drop_one_bootstrap_ci.csv}`
     + `results/visualwebarena/phase1/b0_vs_b1_<site>/`
   ↳ NOT 自动: GLM digest sidecar (watchdog) + 9 narrow ad-hoc diagnostics
     (selflink_loop / vision_coordinate / search_over_browse / diag_pattern_match)
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
✅ Makefile figures target: add fig10_*.py（同时 analyze-paper 自动 chain）
```

### B'. 新 cross-condition aggregator (e.g. paired permutation table)

```
✅ scripts/analysis/aggregate_*.py: implement
✅ Makefile: 新 PHONY target + chain into analyze-paper
✅ paper_planning §13.B: mark done with output path
✅ next_steps §4 Data analysis backlog: mark ✅
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
