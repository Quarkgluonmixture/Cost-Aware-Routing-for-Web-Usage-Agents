---
type: action-ledger
status: rolling
updated: 2026-05-02
---

# Next Steps — Forward Action Ledger

> **Future-only**. Live state 不在这里:
> - Today / 瓶颈 / cron health → [[PLAYBOOK#§1]] + [[PLAYBOOK#§2]] (🤖 GLM @daily 重写)
> - Real-time active runs / GPU → `make active` CLI
> - Cell snapshot (active 跑中 / pending / done) → `cells.base` (Bases view)
> - Paper section progress → `status.base` (Bases view)
> - 过去 chronicle → [[实验笔记]] §1-§108
> - Strategy / theory → [[paper_planning]]
> - Advisor sync prep → [[ADVISOR_SYNC]]
>
> 🔧 **新数据 → `make analysis`** (~5-10min). Cron 每 10 min 自动同步 cell frontmatter (含 re-run detection). 自己只在 brain decision 层做事 — chronicle / planning / hook.

---

## §0 Direction

**Paper hook**: → [[paper_planning#§1]] (canonical, 3 arms / 4-fold drop-in)

> [!todo] Next 3 actions (priority order)
> 1. **学长 sync** ⭐ — 30-45 min, [[ADVISOR_SYNC]] §1-§5 cover 主 asks (framework reframe / VWA bug 单独成文 / RunPod 经费 / Early-stop A/B/C / SteerMoE scope)
> 2. **RunPod onboarding** (post-approval) — `docs/reference/RUNPOD_ONBOARDING.md` 7-step playbook. ETA ~3-5 d wallclock for 14-cell parallel
> 3. **新数据按 4-dimension framework 整合** — 14-cell 跑完后 `make analysis` → re-evaluate 4-fold drop-in → §4 disclosure paragraphs + codex #11/#13

---

## §1 Pending experiment chains — `queue_chain.sh` (sequential)

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

**Pre-launch sanity 必跑**: `make glm-pre-launch-check QUEUE=... BASELINE=... SITE=... RESET=1`

---

## §2 Horizon cells (longer-term, not yet promoted to cells.base)

待 advisor align + RunPod approval 后再 promote 到 `_status/cells/`:

- **B0 shopping SoM/Vision/P-text/P-SoM** (4 cells) — wait B0 dom shopping done. ~$60 + ~24h API
- **B0 shopping P-prompt** (1 cell) — wait B0 phantom shopping pair. ~$15 + ~6h
- **WA × 3 sites × B0+B1 × 6 modes** (36 cells; B1 P-prompt Tier 2 → 33 if dropped) — wait advisor align + B1 VWA chain. ~$60-72 + 60-72h GPU
- **Cross-model Claude Opus 4.7 cls+red 6 modes** (12 cells) — wait advisor align + agent 适配. ~$100-120
- **B1 shopping 6 modes** (6 cells) — wait RunPod 4090 + DGX-side B1 phantom done. ~24h GPU each

---

## §3 Router experiments (Section 6, ~Week 4-5)

| Cell | Blocker | Implementation |
|---|---|---|
| **Tier 1 oracle router** (TF-IDF + LR, ~3 d) | baseline + phantom 全 done | `p79/experiment/router.py::RuleBasedRouter` 扩展 |
| **Tier 2 first-step trigger** (~7-10 d) | Tier 1 done + step-1 trigger features | 新增 cascade runner config |
| Routing signal infra | ✅ ready (`9d7e99f`) | `confidence_summary.json` per-condition |

详 [[paper_planning#§8]] (5 决策维度: feature / target / granularity / cascade / baseline).

---

## §4 Sustainability / Green AI (Section 8)

| Item | Status |
|---|---|
| fig9 regional carbon sensitivity (B1, 45 region) | ✅ done |
| B1 measured energy (cls + red × DOM/SoM/Vision) | ✅ ready |
| B0 token-based carbon estimator | ❌ optional Tier 3 future |
| Section 8 prose | ❌ 待 codex #17 (paper end-stage) |

---

## §5 Codex task queue

![[codex.base#Ready to send (now)]]

![[codex.base#Running / In flight]]

![[codex.base#Blocked / Queued]]

**Pending Python scripts (非 codex)**:
- ⏳ Multi-metric Pareto (cost + lat + carbon) — Section 8 前置 (~2h)
- ⏳ TF-IDF + binary feature extraction — Section 6 Tier 1 router 前置 (~1h)
- ⏳ B0 token-based carbon estimator — Section 8 Tier 3 (~20 行 helper)

---

## §6 Open issues

![[issues.base#Active blockers]]

![[issues.base#Backlog]]

---

## §7 Advisor align

详 [[ADVISOR_SYNC]] (rolling self-prep notes + 5 framing decisions register)。

---

## §8 References + quick links

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

### Key infra paths
```
configs/exp_v2_*.yaml                  per-site experiment configs
scripts/queues/queue_phantom*.sh        chain orchestration
scripts/maintenance/reset_vwa_sites.sh  DGX→quark PowerShell reset
scripts/maintenance/experiment_watchdog.py  auto-clean + post-condition pipeline
scripts/maintenance/glm/glm_*.py        GLM Sidecar (cell sync / playbook refresh / pre-launch check)
p79/utils/auth_refresh.py               Playwright sign-in subprocess
p79/experiment/router.py                RuleBasedRouter scaffold
```

---

> 📖 **Doc update workflow** (when X happens, update which docs) → [[paper_planning#§20]]
