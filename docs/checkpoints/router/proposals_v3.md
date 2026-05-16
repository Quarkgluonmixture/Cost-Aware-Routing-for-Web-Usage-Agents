---
type: design-proposal
status: v3-post-user-3-catches
created: 2026-05-16
purpose: v3 router design folding 3 user-caught OOB findings (P0-8 task.category leak / P0-9 visual hijack triaxis / has_ref_image P1 redesign) — all 3 missed by Claude + codex + gemini
hypothesis-tags: H9 (rule-based router), H10 (learned classifier)
preregistration-anchor: docs/checkpoints/pre_run/preregistration.md §354 + §359 (3 patches applied 2026-05-16 — see Appendix A 2026-05-16 entry + `docs/checkpoints/router/preregistration_C_patches_applied.md`)
supersedes: docs/checkpoints/router/proposals_v2.md
stress-trace:
  mode-a: docs/checkpoints/router/stress_mode_a_2026-05-16.md
  mode-b: docs/checkpoints/codex_outputs/router_design_FINAL_2026-05-16_084921.md
  mode-c: docs/checkpoints/gemini_outputs/router_design_2026-05-16_084921.md
  user-catches: 2026-05-16 user conversation (3 OOB) — P0-8 task.category provenance / P0-9 M1 hijack triaxis / has_ref_image task-attached
---

# Router Design Proposals v3 — folds 3 user-caught OOB findings

> **What changed v2 → v3**: 3 additional findings 全部 user-caught (Claude + codex + gemini 都漏), all P0 / 1 OOB-OOB (was hidden behind sub-doc references that needed cross-reading paper_planning §2 + 实验笔记 §M1 + VWA task schema).

---

## §A v3 NEW findings (user-caught)

### 🔴 P0-8 (user-caught) — `task.category` 4-way 不是 VWA-native, 是 P79 codex audit

**Claim** (v2 spec): `router_proposals_v2.md` Stage 1 candidate pool for P2 lists "6 categorical (site one-hot + manual task category one-hot, **manual category lookup is runtime-known from VWA `task.category`**)".

**Reality**: P79 Cat A/B/C/D 是 codex audit derived (`docs/analysis/cross_sites/codex_audit_classifieds.json` + `codex_audit_reddit.json`). VWA-native task object **has** `task_id` / `intent` / `intent_template` / `eval` / `image_url` / `start_url` / `sites`, **does NOT have** semantic category field. Evidence:
- `docs/analysis/phantom_paper/cross_site_pattern_consolidation.md:25` "B0-64(B) means B0 task 64 with **Codex audit category B**"
- `p79/experiment/tasks.py` grep: no `task.category` field, only `error_category` (post-task failure annotation) and `benchmark_noise_category`

**攻击**: Same-class P0 leak as v1 `has_reference_image` (post-hoc audit label used as router input). 3-AI 都漏，因为我 v2 写 `runtime-known from VWA task.category` 把 audit label 包装成 native field, 没人 grep VWA submodule schema 验证。

**Fix** (P0-8 defuse): Remove Cat A/B/C/D 4-way categorical from P2 Stage 1. Replace with **runtime-derivable intent regex binaries** computed from `task.intent` string only:

| New binary feature | Regex extraction from `task.intent.lower()` |
|---|---|
| `has_filter_keyword` | `re.search(r'\b(filter|narrow|only|with|exclude|less than|more than|under \$\|over \$)\b', intent.lower())` |
| `has_sort_keyword` | `re.search(r'\b(sort|order by|cheapest|most expensive|newest|oldest|highest|lowest)\b', intent.lower())` |
| `has_compare_keyword` | `re.search(r'\b(compare|similar|same as|like this|equivalent)\b', intent.lower())` |
| `has_aggregate_keyword` | `re.search(r'\b(how many|count|total|average|sum)\b', intent.lower())` |
| `has_account_action` | `re.search(r'\b(my account|profile|subscribe|comment|post|publish|submit)\b', intent.lower())` |

These run AT TASK LOAD time on string `task.intent` — **no post-hoc audit, no gold-answer derivation, no human-in-the-loop**. Reviewer can independently re-derive from VWA task json. **Effort**: 30 min code + 1h cross-check coverage vs codex audit Cat A/B/C/D label (target ≥ 70% recall on each, lower than has_ref_image's ≥ 95% target because Cat taxonomy is fuzzier).

### 🔴 P0-9 (user-caught, OOB) — Visual hijack triaxis not in router features

**Claim**: v2 P1 + P2 决策完全没用 P79 已收集的 M1 hijack mechanism evidence。

**Mechanism evidence** (笔记 §100/§101 + §M1, line 2117-2122):

| Trigger axis | B0 (235B) | B1 (4B) | Evidence |
|---|---|---|---|
| Capability | num_ids = 0 (immune) | num_ids 0→446 随 density | line 2003/2117/2122 |
| Mark density | irrelevant | 33: 0 → 41: 12 → 111: 88 → **128: 446** | probe table |
| Site / page density | n/a | reddit ~74 marks / cls ~111 / cls dense ~128 | line 2133 |

**Phantom-SoM 救援 mechanism** (line 2048): "B1 mode-WithText (= Phantom-SoM equivalent) 几乎完全忽略截图 (num_ids 大幅降)" — 给 `[SOM_MARKS]` 文本 fallback → attention 不再被数字 hijack。

**关键预测**: B1/B2 × high-density 页面下, Phantom-SoM > Full SoM (laser-Pareto, paper §1 hero direct mechanism support)。这是 router 决策必须 encode 的 capability × density 交互。

**Fix** (P0-9 defuse): 两 proposal 都加 capability × density 处理:

**P1**: 决策逻辑加 L1.5 capability-aware hijack guard:
```python
# L1.5 hijack avoidance — INSERT BETWEEN existing L1 and L2
mark_density_proxy = obs_1.axtree_element_count   # step-1 AXTree element count ≈ SoM mark count upper bound
if model in ("B1", "B2") and mark_density_proxy > θ_hijack:
    # capability + density 双 trigger → M1 hijack regime
    step_state.preferred_mode = "phantom_som"   # [SOM_MARKS] text bypass screenshot
    step_state.current_mode   = "phantom_som"
    # 不 return; fall through to L3 stateful escalation as usual
```
- `θ_hijack = 90` pre-locked. Justification: 笔记 line 2117 显示 41-marks 阈值未触发, 111-marks 触发 → 90 between保守选。

**P2**: Stage 1 candidate pool 加 2 个 feature:
- `model_categorical` (one-hot 3-way: B0/B1/B2) — capability axis
- `axtree_element_count_step1` (numeric scaled) — density axis
LR 在 train-fold mutual info 选 top-18 时会 surface (model × element_count) 交互 term (sklearn `mutual_info_classif` 检测 non-linear interaction via discretization)。

### 🔴 P0-10 (user-caught, OOB refinement) — `has_ref_image` 在 v2 P1 是 wrong-reason-right-answer

**User insight**: reference image task-attached, passed to ALL modes (`docs/analysis/phantom_paper/B0_dom_shopping_diagnostic.md:87` "task-provided reference images are separate from observation mode and are passed to all modes; DOM removes only the current browser screenshot")。所以 v2 P1 L1 rule `if has_ref_image → SoM` 背后 reasoning "任务需要看图" 不成立 (DOM 也看得到 task image)。

**正确 reasoning** (user insight): has_ref_image task 需要把 **task 图** 跟 **page 上其它 listing 图** 对比。Page 视觉信息只在 SoM/Vision 的 screenshot 里, DOM AXTree 不带 page 截图。所以 has_ref_image **不是 "需要看 task 图"**, 而是 **"需要看 page 图来 match task 图"**。

**但** has_ref_image × capability × density **三轴交互**让 hard-rule 不安全:
- B0 + has_ref_image → SoM 视觉收益 net positive (no hijack)
- B1 + has_ref_image + low density (e.g., cls < 90 marks) → SoM 视觉收益 likely > hijack-loss
- B1 + has_ref_image + high density (cls dense > 90 marks, reddit Cat B) → 笔记 line 2305 实测 -3.6pp, hijack-loss > 视觉收益
- 这个 3-way interaction 不能 pre-data lock — 需要 N=180 训练才能 surface

**Fix** (option b chosen): **P1 L1 完全去掉 `has_ref_image` rule** (capability-blind 哲学一致, zero-training 真 zero-training), 把 (has_ref_image × model × density) 三轴交互**完全推给 P2** 学。
- P1 v3 L1 只保留 `is_search_intent → DOM` (单轴, 不依赖 capability)
- P2 Stage 1 候选 pool 现有 `has_reference_image` binary + 新加 `model` + `axtree_element_count` → LR mutual info 自动 surface 交互
- Paper §6 prose: P1 = "capability-blind handcrafted baseline, single-axis rules"; P2 = "learned classifier captures capability × density × task-attribute interactions"

**Contribution clarification benefit**: P1 vs P2 现在有 **真正的 scientific distinction**:
- P1 holds capability-blind baseline value (cheap, interpretable, locks pre-data)
- P2 contribution = "learned 优于 handcrafted 的 margin 来自 (model × density × ref-image) 三轴交互, 这是 phantom space mechanism 的 router-layer 验证"

---

## §B v3 final spec

### Shared substrate (unchanged from v2)

- Mode universe 6 modes
- Outcome column `success` (no adjusted_success)
- 5-fold site-stratified CV seed=42
- Best-single-mode anchor (train-fold) + 3-tier random baseline (uniform / freq-weighted / top-3)
- Loss = pure SR-max; cost reported as emergent property per deployment class

### P1 v3 — Rule-Based Router (truly capability-blind)

#### Decision logic (single-axis rules only, NO capability check)

```python
def decide_p1_v3(task, obs_1, step_state):
    # ============ Layer 1 — single-axis intent rule (capability-blind) ============
    is_search = bool(re.search(
        r'^(find|search|locate|how many)', task.intent.lower()))
    if is_search:
        step_state.preferred_mode = "dom"          # search 类任务用 cheap DOM
        step_state.current_mode   = "dom"
    # ============ Layer 2 — first-step browser-state escalation ============
    else:
        if obs_1.dom_size > 12000 OR step_state.dom_complexity_history[-1] > 500:
            step_state.preferred_mode = "som"
            step_state.current_mode   = "som"
        else:
            step_state.preferred_mode = "phantom_som"
            step_state.current_mode   = "phantom_som"
    # ============ Layer 3 — ALWAYS call stateful escalation ============
    return RuleBasedRouter.decide(
        preferred_mode = step_state.preferred_mode,
        obs_text       = obs_1.obs_text,
        state          = step_state,
        ...
    )
```

**Differences from v2** (= v3 simplifications):
- **REMOVED** `has_ref_image` rule (P0-10 fix) — task-attached image visible everywhere, capability × density 交互让 hard-rule 不安全
- **REMOVED** capability-aware L1.5 hijack guard — P1 keeps capability-blind, hijack handled by P2

#### P1 v3 feature spec (3 features, all single-axis)

| Layer | Feature | Source |
|---|---|---|
| L1 | `is_search_intent` | regex on `task.intent.lower()` |
| L2 | `dom_size` | `len(obs_text)` at step 1 |
| L2 | `dom_complexity` | `text.count('\n')+1` from `state_change.py` |
| L3 | `unchanged_streak`, `success_streak` | existing `RouterState` |

#### Thresholds pre-locked on archive

- `θ_dom = 12000`, `θ_cmplx = 500` (same as v2, pre-locked from archive)

#### P1 v3 评测

- vs Tier-0a/b/c random ×3
- vs Best-single-mode (preregistration §359)
- vs Oracle ceiling
- vs P2 head-to-head

工程预估 **~1.5 days** (比 v2 短 0.5 day, 因为 L1 rule 减一条 + L1.5 移除)。

### P2 v3 — Learned Classifier (capability × density × ref-image 三轴交互)

#### Test-leak-free constraint (unchanged from v2)

Inference path uses ONLY step-1-observable features. No post-run signals. No audit-derived labels.

#### Stage 1 candidate pool — v3 (changes 标 ★)

| Group | Count | 内容 | v3 change |
|---|---|---|---|
| TF-IDF | 30 | top-30 terms on `task.intent`, English stop-words removed, min_df=3 | unchanged |
| Categorical | **3** | site one-hot (2) + ★ **model one-hot (3-way: B0/B1/B2)** | ★ **P0-9 fix: 加 model axis 让 LR 学 capability × density**, 删 Cat A/B/C/D audit-derived |
| Browser state | **5** | `dom_size`, `dom_complexity`, ★ `axtree_element_count_step1` (mark density proxy), `image_count`, `form_count` | ★ **P0-9 fix: 加 axtree_element_count_step1** |
| Task binary | **15** | `has_reference_image` (runtime extract) + `is_search` + ★ 新 5 runtime regex (`has_filter` / `has_sort` / `has_compare` / `has_aggregate` / `has_account_action`) + 现有 8 (compose / navigation / form_fill / visual_attribute / 等) | ★ **P0-8 fix: 删 audit Cat 4-way, 改 runtime regex 5-binary** |

**总 candidate**: 30 + 3 + 5 + 15 = **53 features**

#### Stage 2 train-fold-only mutual info selection

- Per outer train fold: `SelectKBest(mutual_info_classif, k=18)` fit on train fold only
- Final dim: 18 features × 6 modes = 108 LR weights, Hastie 10-samples-per-feature OK at N≈180

#### Label

`label_t = argmax_m success(t, m)` on train-fold outcomes; tie break = train-fold-frequency-weighted random.

#### Cross-site (replaces dropped cross-model)

- cls-trained → red test; red-trained → cls test
- Cross-model = exploratory disclose only (G2 + Lazy Minimization caveat)

#### Label-distribution diagnostic gate (codex B6)

Pre-Phase-1a 在 archive 上跑 entropy check. If H < log(2) per cell → P2 abandoned (single-route paper)。

#### Sample efficiency curve

Train on 25% / 50% / 75% / 100% of train fold → diagnostic for N=180 adequacy。

#### Ablation rows in paper §6 table

- F-TFIDF only (30)
- F-TFIDF + F-categorical (33: + site + **model**)
- F-TFIDF + F-categorical + F-browser (38: + dom_size + dom_complexity + **axtree_element_count** + image_count + form_count)
- F-TFIDF + F-categorical + F-browser + F-binary-task (53: + 15 binaries) — full P2 v3
- **+F-signal** (v1 F5 verbalized + behavioral AUROC) — paper §6 disclosure ablation only, NOT primary

工程预估 **~6-8 days** (比 v2 +1 day 因为新 5 个 regex binary + axtree_element_count 提取 + 跟 codex Cat audit cross-check coverage)。

### Comparative matrix (v3)

| 维度 | P1 v3 Rule-Based | P2 v3 Learned Classifier |
|---|---|---|
| Capability awareness | ❌ blind (by design) | ✅ via `model` one-hot |
| Hijack mechanism encoding | ❌ none | ✅ via `model × axtree_element_count` mutual-info interaction |
| Ref-image task handling | ❌ no rule (推 P2) | ✅ via `has_reference_image × model × density` LR coefficient |
| Training | 0 | per-cell LR fit on train fold |
| Parameters | 1 threshold + 1 rule | 18 features × 6 modes = 108 LR weights |
| Scientific distinction | Capability-blind baseline | Capability × density × ref-image 三轴交互 learner |
| Engineering effort | ~1.5 days | ~6-8 days |

---

## §C preregistration updates (3 patches, drafted in `preregistration_C_patches.md`)

(Unchanged from v2 §C — applied to preregistration.md only after advisor sync confirms δ_h9 calibration on archive.)

- C1 H9/H10 estimand lock = FE pooled inverse-variance, mirror H1 estimand, δ=1.0pp pending MC re-calibration
- C2 anchor-flicker fallback if Kendall τ < 0.7
- C3 adjusted-SR retirement reflection

---

## §D Remaining gaps (v2 §D unchanged + 1 new)

1-5: G1-G5 from v2 (δ calibration / cross-model downscope / shop / step-level / latency)
**6. (G6 new, P1 capability-blind framing)** — P1 v3 不再 capability-aware. Paper §6 prose 必须明确说 "P1 是 deliberately capability-blind baseline, not best handcrafted router; learned router (P2) captures the capability × density interactions". 否则 reviewer 问 "你的 rule-based router 为啥不用 model info" 没准备答案。

---

## §E tonight 1-3h leverage (extended to verify 3 user catches)

`scripts/analysis/router_archive_diagnostic.py` 现做 **6 gates** (was 4):

| Gate | 输入 | 输出 | 决策 |
|---|---|---|---|
| G-1 P2 viability | per-task per-mode success on B0 archive | label histogram + entropy H per cell | H < log(2) → kill P2 |
| G-2 Anchor stability | per-task per-mode success | Kendall τ × 100 resamples | τ < 0.7 → §C2 fallback |
| G-3 Threshold validation | dom_size + dom_complexity | bucket SR gap on archive | gap < 5pp → re-lock |
| G-4 Noise SD calibration | bootstrap router-vs-anchor lift | router-anchor noise SD | SD > 0.5pp → δ → 2×SD |
| **G-5 ★ NEW** Cat regex coverage | runtime regex on archive intents | recall vs codex audit Cat A/B/C/D | < 70% recall → regex inadequate, paper §6 prose 改 framing |
| **G-6 ★ NEW** Hijack threshold validation | mark density proxy + capability | (B1 + density > 90) cell SR ranking | confirm Phantom-SoM > SoM in this regime |

跑完一晚 → preregistration §C1+§C2 lockable + 3 个新 fix 全验证 → OSF DOI commit + Phase 1a launch unblock。

---

## §F Distance to top-tier (v2 0.85/0.45/0.15 → v3 ?)

v3 vs v2 改动:
- P0-8 fix → 移除 audit Cat leak → R1-R2 reviewer 更难 attack (✅ R2 -0.05 reject prob)
- P0-9 fix → 路由 explicit 用 P79 mechanism evidence → paper §6 与 paper §5 mechanism layer link 更紧 (✅ R1 +0.05 accept prob)
- P0-10 fix → P1 vs P2 真正 scientific distinction (capability-blind vs capability-aware) → 减弱 "P2 = renamed H7 oracle" attack 的 surface (paper-2 H7 是 oracle, P1+P2 都不是)
- G6 P1 capability-blind disclosure → reviewer 知道是 deliberate choice, 不是 design oversight

**v3 估算**:
- Workshop (R3): 0.85 → **0.90**
- Mid-tier (R2): 0.45 → **0.55**
- Top-tier (R1): 0.15 → **0.20**

---

## §G v3 待落地行动

1. ✅ Write `preregistration_C_patches.md` — 3 patches drafted, await advisor sync
2. ✅ Write `scripts/analysis/router_archive_diagnostic.py` — 6 gates
3. After diagnostic land → commit preregistration §C updates → OSF DOI lock → §B Phase 1a launch
4. P1 implementation (~1.5 days) — extend `p79/experiment/router.py::RuleBasedRouter`
5. P2 implementation (~6-8 days) — new `p79/policies/learned_router.py` module
