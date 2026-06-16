# B3 (第 4 基线) 跨族选型 — cross-AI 综合 + 核验 (2026-06-16)

> 目标 = **ADD**（非 replace）一个 capable cross-family general VLM 作 B3，给 phantom-routing/drop-one
> 一个**不地板**的跨族信号点（B2 Gemma3-4B 留作 cross-family **地板**证据）。两个 AI 各自独立搜
> (zero-preset，prompt 不给结论) → 本文交叉 + arXiv API 核验。GPT 报告 `gpt.md` / Gemini 报告
> `跨家族轻量级开源视觉语言模型...选型报告.md`。配笔记 §339。

## 选型硬标准 (回顾)
非-Qwen LLM **backbone** (按解码器家族判，非模型名) · general-purpose VLM (非 GUI-specialist) ·
够强不地板 (理想 ≥~6% B1 档) · bf16 单 A100-40GB 未量化 (≤~13B) · 开源 HF transformers · 有 VWA/WA 锚 (补 Gemma 无锚短板)。

## cross-AI 3 个决定性分歧 (= 跑两个 AI 的价值)

| 分歧 | GPT | Gemini | 核验后判定 |
|---|---|---|---|
| **GLM format-lockout** | ⚠️ GLM-4.1V 在 MedCUA 固定 schema **432/432 zero-action** (输出自己 proprietary format 不发我们 JSON) | 漏，列 **#1** | **GPT 对**。MedCUA-Bench (`2606.03203`) arXiv 核**真** + 正是 screenshot-only 固定-schema CUA → **GLM 在我们设置可能换个方式重新地板** (format-lockout 非能力)。Gemini #1 降级 |
| **InternVL 版本+地板** | InternVL3.5-8B = **Qwen3 backbone → 拒** | InternVL2.5-8B = **InternLM → #2** | **均对(不同版本)**: 2.5=InternLM 干净跨族, 3/3.5 换 Qwen。**但 Gemini 自引数据**: InternVL2.5-8B VWA **classifieds zero-shot=0.4%** = **在我们站地板** → 跨族干净但没解决地板, Gemini #2 自相矛盾 |
| **MiMo-VL** | **#1** (MiMo-7B 跨族, WebVoyager-SoM=34 信号) | 没搜到 | **GPT 对**。MiMo-VL tech report (`2506.03569`) 核**真** (Xiaomi 自研 MiMo-7B 非 Qwen); 候选里唯一有 SoM-web 非零信号 + 无 format-lockout 旗标 |

## 两份报告**收敛**的清醒事实 ⭐
**没有任何候选有可核验的 raw zero-shot VWA SR ≥6%。** 锚全是 WebVoyager (不同 benchmark, GPT-4V 判分) / WebArena-Lite / grounding。**唯一存在的 VWA-classifieds zero-shot 数 (InternVL2.5=0.4%) 是地板。** → 地板风险对整个"小通用 VLM"类是真的；文献回答不了"谁在 classifieds 不地板 + 跟我们固定 JSON schema" → **只有 pilot 能答**。

## 战略发现 (给学长的 scope 点)
若 pilot 显示 MiMo/GLM 也地板 → 故事 = "**Qwen3-VL 是小模型里少见的 agent-能干异类, 多数小通用 VLM zero-shot VWA 都地板**" (强化 §327 "VWA 对无 agent-SFT 通用模型即地狱"; WebArena GPT-4 ~14% / B0-235B 17.4%)。= finding 非 failure, 也是 cross-family 泛化可测性的 scope 限制。

## 合并候选表 (cross-AI + 核验)

| 候选 | backbone (跨族?) | 尺度 | 信号 (来源) | 风险 | 判定 |
|---|---|---|---|---|---|
| **MiMo-VL-7B-RL-2508** | MiMo-7B ✅ | 8B | WebVoyager-SoM **34.0** (GLM 论文统一 harness) | L-M | **首 pilot** (最稳) |
| GLM-4.6V-Flash / GLM-4V-9B | GLM-4-9B ✅ | 9B | WebVoyager 71.8 (post) / WebArena-Lite **6.1%** zero-shot | M (能力低/**format-lockout 高**) | **gated alternate** (先过 conformance) |
| InternVL2.5-8B | internlm2.5 ✅ | 8B | VWA cls **0.4%** ❌地板 | H(地板) | 拒(地板) |
| Pixtral-12B | Mistral Nemo ✅ | 12B | 无 WA 数 (Agent-RewardBench 感知 76.5%) | M | 备 (12B 紧 40GB) |
| Molmo2-O-7B | OLMo3 ✅ | 8B | 无 web 数 | M | 备 (架构最干净但无信号) |
| Phi-4-multimodal-5.6B | Phi-4-mini ✅ | 5.6B | MedCUA **1.4%** ❌Gemma-like 地板 | H | 拒(地板) |

**拒(Qwen backbone)**: InternVL3.5-8B / Ovis2.5-9B / STEP3-VL / Molmo2-8B / MiniCPM-V-2.6 / Penguin-VL (名跨族实 Qwen 解码器)。**拒(GUI-specialist)**: UI-TARS / OS-Atlas / Aguvis / Holo (非平行通用 VLM)。**拒(显存)**: Mistral-Small-24B / Llama-4-Scout / Ovis-80B-A3B (MoE 全量权重 >40GB)。

## Pilot 计划 (DGX, **不碰 A100 paper-grade**)
- **Stage 0 smoke ✅ PASS (2026-06-16)**: MiMo-VL 在 DGX 加载干净 (`Qwen2_5_VLForConditionalGeneration` class, meta=0; GB10 sm_121 nvrtc prod bug → `apply_nvrtc_prod_fallback_if_needed` 同 agent 即修) + **3/3 parse-valid + 无 native leak = 无 GLM-lockout** + grounding 连贯 (task_184 找对 element 17 PA speaker)。**集成轻确认** (Qwen2.5-VL 处理栈同构 → 复用 `qwen3vl_agent.py` 路径, vs Gemma 全新 class)。**两新考量**: (a) MiMo-VL-**RL** = thinking 模型, 每步 `<think>` 块 = thinking-vs-not confound (B0/B1/B2 不 think) → paper-design 决策; (b) 缺 JSON `thought`/`confidence` (confidence 可 logprob derive)。脚本 `scripts/maintenance/probe_mimo_b3_conformance.py`。
- **Stage 1 format-conformance** (= GLM-lockout 直接测): 喂真实 agent prompt + classifieds 观测, 验 parse-valid 率 + **能否发我们的 `finish`** + 不泄漏 native tokens。
- **Stage 2 floor pilot** (过 1 后): 20-30 classifieds task via **DGX→quark Tailscale VWA** (dev path, 非 A100 localhost) → 真 SR。判: 不地板 (≫1%) → 立 B3 production bring-up (同 §140); 地板 → 战略发现 + 学长 scope 决策。
- **隔离**: A100 R10175 paper-grade fire 全程不受影响 (不同 GPU + 不同 VWA 实例)。

## 部署注意
MiMo-VL-7B-RL 经 **MORL** → §8 诚实标 "general post-trained checkpoint, no VWA-specific SFT/TTI" (同 Qwen3-VL/Gemma3 类, 非 "no RL")。`Qwen2_5_VLForConditionalGeneration` 是部署接口非 backbone。若严格禁 RL-checkpoint → 退 `MiMo-VL-7B-SFT-2508` (但无可核 WebVoyager 数, 防地板证据弱)。

## 源 (arXiv API 核 3 ID 全真; SR 数字正文级待复核)
`2606.03203` MedCUA-Bench (GLM zero-action 源) · `2506.03569` MiMo-VL Tech Report · `2507.01006` GLM-4.5V/4.1V (WebVoyager-SoM 34/69 + format) · GPT/Gemini 全报告存 `docs/literature/{gpt.md, 跨家族...选型报告.md}`。
