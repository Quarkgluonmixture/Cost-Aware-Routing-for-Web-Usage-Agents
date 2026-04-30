# Phantom-SoM 代码导览 (Code Tour for Advisor)

**Audience**: PhD advisor reading repo for the first time
**Purpose**: 5 分钟读完, 能 navigate 到 Phantom-SoM + Diamond ablation 的具体实现
**Repo**: https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents
**Pinned commit** (用于稳定 link): `578805b`

---

## 1. Repo 结构 30 秒概览

```
p79/
├── envs/vwa_wrapper.py        ← 接 VWA framework + P79-side action dispatch interception
├── experiment/
│   ├── som.py                 ← ★ Phantom mode observation 构造逻辑
│   ├── runner/main.py         ← 实验 orchestrator (condition × seed × task × step)
│   └── conditions.py          ← Phase 1 condition 生成 (5-mode + diamond)
├── agents/
│   ├── proxy_api_agent.py     ← B0 (235B Qwen3-VL via API) — ★ system prompt selection
│   └── qwen3vl_agent.py       ← B1 (4B local) — 同样的 prompt logic
configs/
└── exp_v2_B0_phantom_*.yaml   ← 6 个 Phantom 实验配置 (3 sites × 2 modes)
```

---

## 2. 核心实现 — Phantom mode observation 构造

**文件**: [`p79/experiment/som.py`](https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents/blob/578805b/p79/experiment/som.py#L132-L189)

### 关键函数 `prepare_observation_for_mode()` (line 132-189)

这是 **paper 5+1 mode 设计的 single source of truth**. 每个 mode 怎么造 observation 都在这一个函数里。

```python
def prepare_observation_for_mode(obs, mode, artifact_dir, step_idx) -> SomResult:
    """
    mode == "dom":            Full AXTree text, no image
    mode == "som":            SOM_MARKS + marked image (consistent SoM)
    mode == "phantom_som":    SOM_MARKS index, NO image (P-SoM: image-mismatched)
    mode == "phantom_text":   SOM_MARKS index, NO image, with DOM-prompt (text-mismatched)
    mode == "phantom_prompt": Full AXTree text, NO image, with SoM-prompt (P-prompt: 我加的 diamond corner)
    mode == "vision":         Empty text, raw screenshot
    """
```

**关键观察**:
- P-SoM 跟 SoM 唯一差别: `marked_image=None` (line 173). 同样的 text + 同样的 prompt, 只是把图删了 — 这就是 4-fold drop-in property 的根 (no image encoding overhead).
- P-text 跟 P-SoM **observation byte-identical** (line 165-176 同 branch). 唯一差别在 system prompt (下一个文件).
- P-prompt 跟 DOM observation byte-identical (都是 AXTree). 唯一差别也在 system prompt.

→ 这个设计让 paper 可以 **paired comparison cleanly attribute 单一 axis 效应**.

---

## 3. System Prompt Selection — axis 2 (prompt) swap

**文件**: [`p79/agents/proxy_api_agent.py`](https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents/blob/578805b/p79/agents/proxy_api_agent.py#L440-L451) (line 440-451)

```python
system_prompts = {
    "dom":            dom_prompt,         # "你的输入是 AXTree 文本"
    "som":            som_prompt,         # "你看图找 mark 数字"
    "vision":         vision_prompt,      # "你只能看图"
    "phantom_som":    som_prompt,         # ← P-SoM 用 SoM-style prompt (但实际没图)
    "phantom_text":   dom_prompt,         # ← P-text 用 DOM-style prompt (text 是 SOM_MARKS)
    "phantom_prompt": som_prompt,         # ← P-prompt 用 SoM-style prompt (text 是 AXTree)
}
```

**关键**: 这 6 个 mode 共用同一个 LLM (Qwen3-VL-235B), 同一个 inference pipeline, 只是 system prompt 不同. 所以 cross-mode 比较没有 model confound.

---

## 4. Diamond Completion — P-prompt 的设计动机

**Diamond 形状的 motivation 在文档里**: [`docs/checkpoints/paper_planning.md`](https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents/blob/578805b/docs/checkpoints/paper_planning.md#L437) line 437

**核心论证**:
- 没有 P-prompt → ablation 是 L-shape: DOM → P-text → P-SoM 三角
- 加 P-prompt → 4-corner factorial 闭合, 可以 separately quantify prompt × text interaction

**实现**: 就是 `som.py:178-182` (P-prompt branch) + `proxy_api_agent.py:451` (P-prompt 用 SoM prompt). 极简 — 4 行代码完成 diamond corner.

```python
# som.py line 178-182
if mode == "phantom_prompt":
    # P-prompt: AXTree text (same as DOM mode) + no image, but SoM prompt (set in agent).
    # Symmetric ablation of phantom_text: only the prompt axis is swapped from DOM.
    return SomResult(som_text=obs_text, marked_image_path=None, marked_image=None,
                     degraded_som=False, mark_count=0)
```

---

## 5. 实验配置 — 6 个 mode 怎么跑

每个 mode + site 一份 yaml: [`configs/`](https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents/tree/578805b/configs)

**关键 configs**:
- `exp_v2_B0_phantom_classifieds.yaml` → P-SoM × cls
- `exp_v2_B0_phantom_text_classifieds.yaml` → P-text × cls
- `exp_v2_B0_phantom_prompt_classifieds.yaml` → P-prompt × cls (Diamond corner)
- 每个 mode 也有 `_reddit.yaml` / `_shopping.yaml` (3 sites × 6 modes ≈ 18 configs)

每个 yaml 都 reference base + override `observation_mode` 字段:
```yaml
variables:
  primary:
    observation_mode: ["phantom_som"]   # or phantom_text, phantom_prompt, dom, som, vision
```

---

## 6. 跑实验的 entry point

[`scripts/queues/queue_phantom_som.sh`](https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents/blob/578805b/scripts/queues/queue_phantom_som.sh) — 启动 P-SoM 实验, 自动加载环境 + watchdog
[`scripts/run_experiment.py`](https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents/blob/578805b/scripts/run_experiment.py) — 真 entry, 读 yaml 跑 ExperimentRunner

---

## 7. 分析脚本 — 怎么从 raw data 算 oracle lift / Jaccard

[`scripts/analysis/figures/`](https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents/tree/578805b/scripts/analysis/figures) — 14 figures 的生成脚本, 一个 figure 对一个 .py:

- `fig0c_drop_one_oracle.py` — drop-one oracle lift (4-fold property d)
- `fig0d_taskpool_jaccard.py` — task pool Jaccard heatmap (complementarity)
- `fig0g_routing_auroc_heatmap.py` — routing signal AUROC (4-fold property c)
- `fig1ab_cascade_diamond.py` — diamond schematic (理论框架图)
- `fig3a_token_cost_intra_baseline.py` — token cost (4-fold property a)

每个脚本 `python3 scripts/analysis/figures/<file>.py` 直接跑, 输出到 `results/phantom_paper/figures/`.

---

## 8. Paper Section 5 lookup table

[`docs/analysis/cross_sites/site_mechanism_dictionary.md`](https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents/blob/578805b/docs/analysis/cross_sites/site_mechanism_dictionary.md) — 30KB structured Section 5 lookup, 3 sites × 3 axes × 6 fields. Section 5 prose 由这个 dictionary 派生.

---

## 9. 验证 — Phantom finding 不是 framework bug 假象

**最近 (2026-04-30) 添加的 4 个 audit + 4 cluster fix**:
- [`docs/checkpoints/master_bug_catalog.md`](https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents/blob/578805b/docs/checkpoints/master_bug_catalog.md) — 37 entries, 9-pass audit log
- [`docs/reference/VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md`](https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents/blob/578805b/docs/reference/VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md) — synopsis + Phantom existence 论证 (§5 6 signals)
- [`p79/envs/locator_dispatch.py`](https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents/blob/578805b/p79/envs/locator_dispatch.py) — Cluster 1 patch (B-33 family fix)
- [`scripts/maintenance/probe_tier10_dispatch_target.py`](https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents/blob/578805b/scripts/maintenance/probe_tier10_dispatch_target.py) — Tier 10 verification probe

**Pilot wave-2 验证** (T=0 sanity, 60 ep): [`docs/analysis/cross_sites/pilot_t0_decision_final.md`](https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents/blob/578805b/docs/analysis/cross_sites/pilot_t0_decision_final.md) — Δ=0pp matched-subset SR

---

## 10. 一句话 summary 给学长 (TL;DR)

> "Phantom-SoM 实现就是 `p79/experiment/som.py:165-182` 4 个 branch (5+1 mode 怎么造 observation) + `p79/agents/proxy_api_agent.py:440-451` 12 行 dict (system prompt selection). 加 P-prompt 让 ablation 闭合成 2×2 factorial diamond. 整个 paper 数据由 `scripts/run_experiment.py` orchestrate, 14 figures 由 `scripts/analysis/figures/` 14 个独立脚本生成. 详细 strategy 见 `docs/checkpoints/paper_planning.md`."

---

## Appendix — 推荐学长 review 顺序

1. **理解设计** (5 min): `docs/checkpoints/paper_planning.md` §1 (paper hook, line 17-29) + §2 (theory framework, line 47-91)
2. **看代码实现** (5 min): `p79/experiment/som.py:132-189` + `p79/agents/proxy_api_agent.py:440-451`
3. **看 figures** (10 min): `results/phantom_paper/figures/` 14 张, 或访问 dashboard `docs/checkpoints/周报/weekly-dashboard/dist/index.html` (双击打开 — 已 build)
4. **bug 副发现** (5 min): `docs/reference/VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md` synopsis

总时长: ~25 分钟可以完整 review.
