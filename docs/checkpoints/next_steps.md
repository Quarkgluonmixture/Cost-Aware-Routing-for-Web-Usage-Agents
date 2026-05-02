---
type: action-ledger
status: rolling
updated: 2026-05-02
---

# Next Steps — Action Ledger

> **Live + Future only**.
> - 过去归档 → [[实验笔记]] (§1-§108 chronicle)
> - 脉络 strategy → [[paper_planning]] (19 sections)
> - Advisor sync → [[ADVISOR_SYNC]] (rolling self-prep notes)
> - 落地 prose → `paper_drafts/section*.md`
>
> 🔧 **新数据 → `make analysis`** (~5-10min, all-in-one): per-run + cross-condition + figures + status

---

## §0 Current state

**Paper hook (provisional, pending cls 6-mode 数据 confirm)**: "We discover a **hidden phantom routing space** for web agents — boundary 'skip annotated image' — containing **3 routing arms** (P-text / P-prompt / P-SoM) sharing a **4-fold drop-in property**. P-SoM (cube center, axis 1+2 compound) is the space's representative arm."

> [!warning] Active blockers
> | 项 | 状态 | 原因 |
> |---|---|---|
> | **14-cell paper-grade re-run** | ⏸️ blocked | DGX GPU 争抢严重 (B1 234 ep 跑 20+ h), 需 RunPod 经费 approval |
> | **Cls pilot wave-3** | ⏸️ blocked | B1 P-text cls 仍跑 (~198/234 ep, ~1-2 d ETA) — same-site B0 XOR B1 hard rule |
> | **学长 sync** | ⏸️ pending | 时间未定, [[ADVISOR_SYNC]] 已 ready |

> [!todo] Next 3 actions (priority order)
> 1. **学长 sync** ⭐ — 时间定后 30-45 min meeting, [[ADVISOR_SYNC]] §1-§5 cover 主 asks (framework reframe sanity check / VWA bug 是否单独成文 / RunPod 经费 / Early-stop A/B/C / SteerMoE scope)
> 2. **RunPod onboarding** (post-approval) — `docs/reference/RUNPOD_ONBOARDING.md` 7-step playbook ready. Pin paper-grade commit hash, scp `.auth/`, smoke test, launch 14-cell parallel (7 cells B0 on DGX + 7 cells B1 on RunPod 4090). ETA ~3-5 d wallclock
> 3. **新数据按 4-dimension framework 整合** — 14-cell 跑完后 `make analysis`. 重 evaluate 4-fold drop-in. Section 4 disclosure paragraphs + codex #11/#13 trigger

---

## §1 Active processes (live)

🔄 **Real-time**: `make active` — scan ps + episode mtime, 不要在此硬编码

```bash
make active                  # human-readable
make active --json           # machine-readable
```

最近 incident 历史 → [[实验笔记]] §107+ chronicle (Phase A bug fix wave / cross-session pilot data destruction / quark wsl shutdown / Myriad rejection / 等)

---

## §2 Paper section status

![[status.base#Paper Sections]]

Per-section detail → `_status/section*.md` notes (frontmatter status/progress/blocker/words). Final scope + 顶刊概率 → [[paper_planning#§5]].

---

## §3 Active experiments + pending cells

> Cells.base view 待 Demo E 完成 (cells 现仍 inline). 暂时表格如下:

### 3.1 Active 跑中

| Cell | Status | ETA |
|---|---|---|
| **B0 P-prompt reddit (PID 2075552)** | ~16/210 | ~6h |
| B1 phantom_som cls (PID 1821957) | task 184+/234, watchdog auto-cleaned 5 NOT-LOGGED-IN | ~10-15 d (GPU contention) |
| B0 dom shopping clean re-run (PID 1106560) | ~9/466 | ~9h |

### 3.2 Pending chains — `queue_chain.sh` (sequential)

B1 4B 单 GPU + B0 phantom shopping 同 site exclusive → sequential. Idempotent + paper-grade `RESET_BEFORE=1` 默认.

**Tier 1 (paper-critical)**:
```bash
nohup bash scripts/queues/queue_chain.sh \
  "queue_phantom_prompt.sh B0 classifieds" \
  > logs/queue_chain_b0_pprompt_cls.log 2>&1 &

nohup bash scripts/queues/queue_chain.sh \
  "queue_phantom_som.sh B1 classifieds" \
  "queue_phantom_som.sh B1 reddit" \
  "queue_phantom_text.sh B1 classifieds" \
  "queue_phantom_text.sh B1 reddit" \
  > logs/queue_chain_b1_phantom.log 2>&1 &

nohup bash scripts/queues/queue_chain.sh \
  "queue_phantom_som.sh B0 shopping" \
  "queue_phantom_text.sh B0 shopping" \
  > logs/queue_chain_b0_phantom_shop.log 2>&1 &
```

**Tier 2 (diamond completeness, B1 P-prompt)**:
```bash
nohup bash scripts/queues/queue_chain.sh \
  "queue_phantom_prompt.sh B1 classifieds" \
  "queue_phantom_prompt.sh B1 reddit" \
  > logs/queue_chain_b1_pprompt.log 2>&1 &
```

| Chain | ETA | Status |
|---|---|---|
| B0 P-prompt cls (Tier 1) | ~6h | ⏳ wait B1 phantom_som cls (~10d) |
| B1 phantom 4-cell (Tier 1) | ~30-40 d (GPU contention) | ✅ launched 04-29 19:11 PID 1145483 |
| B0 phantom shopping pair (Tier 1) | ~24h | wait B0 dom shopping done |
| B1 P-prompt 2-cell (Tier 2) | ~14-20 d | ⏳ wait B1 4-cell Tier 1 done |

### 3.3 Missing cells (post 14-cell rerun target)

**Critical (paper Section 5)**:
| Cell | Blocker | Cost |
|---|---|---|
| B0 P-prompt cls (diamond) | B1 phantom_som cls done | ~$3 + ~6h API |
| B0 shopping SoM/Vision/Phantom × 3 | B0 dom shopping done + verify | ~$45 + ~24h |
| B0 shopping P-prompt | B0 phantom shopping pair done | ~$15 + ~6h |

**Generalization (paper Section 7)**:
| Cell | Blocker | Cost |
|---|---|---|
| WA × 3 sites × B0+B1 × 5 modes (30 cells) | advisor align + B1 VWA chain done | ~$60-72 + 60-72h GPU |
| Cross-model Claude Opus 4.7 cls+red 5-mode (10 cells) | advisor align + agent 适配 | ~$100-120 |
| B1 shopping 5-mode + P-prompt (6 cells) | RunPod 4090 + DGX-side B1 phantom done | ~24h GPU each |
| B1 P-prompt cls + red (Tier 2 cross-cap) | B1 P-SoM + P-text 4-cell chain done | ~14-20 d (queued) |

### 3.4 Router experiments (Section 6, ~Week 4-5)

| Cell | Blocker | Implementation |
|---|---|---|
| **Tier 1 oracle router** (TF-IDF + LR, ~3 d) | baseline + phantom 全 done | `p79/experiment/router.py::RuleBasedRouter` 扩展 |
| **Tier 2 first-step trigger** (~7-10 d) | Tier 1 done + step-1 trigger features | 新增 cascade runner config |
| Routing signal infra | ✅ ready (`9d7e99f`) | `confidence_summary.json` per-condition |

详 [[paper_planning#§8]] (5 决策维度: feature / target / granularity / cascade / baseline).

### 3.5 Sustainability / Green AI (Section 8)

| Item | Status |
|---|---|
| fig9 regional carbon sensitivity (B1, 45 region) | ✅ done |
| B1 measured energy (cls + red × DOM/SoM/Vision) | ✅ ready |
| B0 token-based carbon estimator | ❌ optional Tier 3 future |
| Section 8 prose | ❌ 待 codex #17 (paper end-stage) |

---

## §4 Codex task queue (pending only)

> Codex.base view 待 Demo E 完成. 暂时表格:

| # | Task | Tokens | Blocker | Status |
|---|---|---|---|---|
| **10** | paper.bib formal expansion + §2 Zoom 3 cross-reference. Q5 (bidirectional) 已 6/6 integrate, Tool Calling Linear Circuit 已 Zoom 4 | ~50-100K | — | 🟡 codex 跑中 (paper.bib 部分) |
| **11** | Section 4 fresh-data prose update (4-dim framework + ~100× cost + N=210 全数据) ⭐ | ~30K | — (现可发) | 🟢 ready, 等用户 trigger |
| **13** | Section 5 prose 写 (site × axis × LLM-mechanism 3-way table) | ~50-80K | 待 P-prompt reddit done (~6h) + #11 一起发 | 🟡 ~Thu |
| 14 | Codex audit shopping VWA (466) — full 5-mode | ~500K | 待 shopping 5-mode 数据 | ⏳ Week 2-3 |
| 14b | Codex audit reddit cat refine (类 shopping A-refine) | ~50K | red 5-mode FRESH already | 🟢 ready |
| **14c** | Codex redo `disagreement_clusters.md` 含 5+1 phantom modes | ~80K | B1 phantom 4-cell + B0 P-prompt diamond done (~30-40d) | ⏳ Week 4-5 |
| 15 | Codex audit WA tasks (480) | ~500K | 待 WA 数据 | ⏳ Week 4-5 |
| **16** | Section 6 Routing prose (Tier 1+2 + 4-fig stack) | ~50K | Tier 1+2 prototype done + figures | ⏳ Week 5-6 |
| 17 | Section 7 Generalization 草稿 | ~50K | shopping + WA + Claude done | ⏳ Week 6-7 |
| 18 | Section 8 Discussion 草稿 | ~30K | 全部 data done | ⏳ Week 8+ |
| 19 | 二次 deep research (Section 6/7/8 revisit) | ~300K | Week 8+ paper 终稿前 | ⏳ Week 8+ |

**Pending Python scripts (非 codex)**:
- ⏳ Multi-metric Pareto (cost + lat + carbon) — Section 8 前置 (~2h)
- ⏳ TF-IDF + binary feature extraction — Section 6 Tier 1 router 前置 (~1h)
- ⏳ B0 token-based carbon estimator — Section 8 Tier 3 (~20 行 helper)

---

## §5 Open issues (active)

> Issues.base view 待 Demo E 完成. 暂时表格:

| Issue | Status | Action needed |
|---|---|---|
| **B1 phantom_som cls GPU contention** (seonglae 95% util, 5× latency) | 🟡 持续 | 联系 seonglae or 接受 slow progression |
| **IP env-var-ize 重构** (9 处 hardcoded `100.95.81.103`) | 🟡 backlog | 替换 `${VWA_REMOTE_HOST}` env var (`auth_refresh.py` / `envs.py` / queue scripts). 触发: Myriad onboard 时 (现已废) |
| **WA reset mechanism** (queue scripts 仅 vwa reset, wa skip) | 🟡 backlog | 写 `reset_wa_sites.sh` + queue scripts 加 `BENCHMARK=wa` 分支 |
| **Watchdog AUTO-ANALYSIS spam guard** (partial summary 触发 infinite loop) | 🟡 backlog | `experiment_watchdog.py:1340` 增加 episode count vs expected guard |
| **GPU contention deadlock detection** (B1 cls stalled 4h, kernel wait 不报 progress) | 🟡 backlog | watchdog stale-runner heartbeat: episode mtime > N min → SIGTERM + queue script idempotent re-spawn |
| **Tier A summary commit decision** (`condition_summary_v2.json` + `run_meta.json` 入 git LFS / 直 git?) | 🟡 待评估 | size: 10 cond × ~50KB = ~500KB |
| **B1 shopping DOM 466 ep** (pre-Magento-bug archived) | ⏸️ 待 RunPod | `queue_baseline.sh B1 dom shopping` clean re-run |
| **Pending paper-grade rerun: B0/B1 × DOM/SoM × shopping** | ⏸️ 待 RunPod | bugs 累积修完后一次性 re-run (含 §105 swatch radio fix) |

---

## §6 Advisor align

详 [[ADVISOR_SYNC]] (rolling self-prep notes + 5 framing decisions register)。

---

## §7 References + quick links

### Paper drafts (final prose)
```
docs/checkpoints/paper_drafts/
  section1_intro.md          ✅ 786w
  section2_background.md     ✅ 1514w + paper.bib (57 entries)
  section3_definition.md     ✅ 863w
  section4_findings.md       🟡 1725w stale (待 codex #11)
  section5_mechanism.md      ❌ 待 codex #13
  section6_routing.md        ❌ 待 Tier 1+2 prototype
  section7_generalization.md ❌ 待 WA + Claude
  section8_discussion.md     ❌ paper end-stage
```
Live progress → `![[status.base#Paper Sections]]`

### Codex analysis docs
```
docs/analysis/phantom_paper/disagreement_clusters.md           (B0+B1 9-cat)
docs/analysis/phantom_paper/cross_site_pattern_consolidation.md
docs/analysis/phantom_paper/phantom_dom_vs_som_diagnostic.md   (axis 2)
docs/analysis/phantom_paper/som_vs_phantom_som_diagnostic.md   (axis 3 8-channel)
docs/analysis/B1_capability_profile.md                          (Section 7 prep)
```

### Figures
`results/phantom_paper/figures/` (FRESH 04-28 per `make figures`):
- fig1 4-mode venn / fig2 drop-one oracle / fig3 strategy gradient / fig4 two-knob
- fig5 category × mode heatmap / fig6 capability B0-vs-B1 / fig7 cost-SR Pareto
- fig8 overlap-depth / fig9 regional carbon / fig10 phantom_lift_bars / fig11 routing_auroc_heatmap

### Recent commits
```bash
git log --oneline --since="2026-04-26"
```

### Key infra paths
```
configs/exp_v2_*.yaml                  per-site experiment configs
scripts/queues/queue_phantom*.sh        chain orchestration
scripts/maintenance/reset_vwa_sites.sh  DGX→quark PowerShell reset
scripts/maintenance/experiment_watchdog.py  auto-clean + post-condition pipeline
p79/utils/auth_refresh.py               Playwright sign-in subprocess
p79/experiment/router.py                RuleBasedRouter scaffold
```

---

> 📖 **Doc update workflow** (when X happens, update which docs) → moved to [[paper_planning#§20]]
