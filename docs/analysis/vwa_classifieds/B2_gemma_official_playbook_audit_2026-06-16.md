# B2 (Gemma 3 4B) 官方 playbook 使用审计 — 2026-06-16

> 学长提示"查官方 playbook"→ 用 GPT (web browsing) 对 `google/gemma-3-4b-it` 做官方文档逐条对账，
> 零预设 prompt (不喂结论，让 GPT 独立查官方源并引证)。本文 = 审计结论 + 我方代码核验 refinement +
> 六源收敛。配 §327 probe doc (`B2_gemma_visual_probe_2026-06-09.md`) + 笔记 §338。

## TL;DR

**部署里没有一个低级实现 bug 能单独解释 ~1% SR。** bf16 / 未量化 / `Gemma3ForConditionalGeneration` /
官方 processor + chat template / 单图 / 单 BOS / `pan_and_scan` 已开 —— 官方源逐条确认皆合理。
根因 = **模型能力 + scaffold/post-training 失配 (尤其 termination policy)，非 serialization bug。**

## 优先级表 (GPT 审计 × 我方代码核验)

| 优先级 | 差异 | 解释力 | 我方核验 refinement |
|---|---|---|---|
| **P0** | 非 Gemma-native 的 agent/tool/termination prompt | 极强：直接对应低 finish / 循环 / budget death | ✅ 成立。Gemma 3 无 native computer-use / action-policy / tool-call control tokens；98% parse 只证明会写 JSON，**不证明学会了何时 finish**。与 digest "95% agent-limit, finish 23/224" + §330 "SR 被 0-finish 钉死" 一致 |
| **P0→降级** | processor 前先压到 1024px × P&S | GPT 判强 | ⚠️ **refine**: cap 在 `gemma3vl_agent.py:200` = `qwen3vl_agent.py:212` **逐字相同 = B1/B2 共享**，非 Gemma-specific (Qwen 同样 capped 仍 6-14%)。且 VWA viewport = **1280×720**，cap 只 1280→1024 = **1.25×**，非 GPT 假设的 1920→1024 (1.9×)。§327 已测 "P&S on capped 图 → 红车仍认不出"。→ **✅ uncapped probe (2026-06-16) 实测证伪**: 3 图 cap-1024 vs uncapped × P&S=True，LABEL/PRICE 两臂逐字同 + PHOTO 两臂相当无系统优劣 → **cap 非病因，P0#2 关闭**。caveat: 开放问题混淆 OCR+照片 (Gemma 读标题文本)，不隔离 photo-blind，§327 受控结论存活 |
| **P1** | greedy ≠ checkpoint sampling 默认 | 中：可能强化 mode collapse | checkpoint `generation_config` = `do_sample=true, top_k=64, top_p=0.95` (无 temperature)。**但官方 model card 自己用 `do_sample=False`** → greedy **可辩护，非官方错误**。= disclosure + 可选 ablation，非 fix。顺手清理: `temperature=0` 在 greedy 下 no-op，删 |
| **P1** | `transformers≈5.8` < 当前教程 `>=5.10.1` | 中低 | 发布时只需 `>=4.50`，5.8 非 unsupported。可选独立环境 ablation，勿覆盖 paper env |
| **P2** | SoM / dense UI / coordinate grounding 无官方训练保证 | 很可能能力上限 | Gemma 3 无官方 WA/VWA/SoM 结果；OCR-strong/photo-weak 是**我方 empirical finding** (§327)，非官方已述 limitation |
| 低风险 | eager attn / bf16 / 未量化 / 单图 / processor pipeline | 无证据 = SR 崩溃来源 | ✅ 官方 card 本身用 bf16 |

## 关键官方事实 (GPT 引证，待本地 `curl` 复核)

- **decoding**: checkpoint `generation_config.json` = `{do_sample:true, top_k:64, top_p:0.95}`，无 temperature / repetition_penalty / max_new_tokens。官方无统一推荐输出长度 (card 示例 100/200)。
- **chat template**: Gemma 3 IT 原生只有 `user`/`model` 两角色，**无独立 system turn**；HF template 把 `system` 折进首个 user turn (输入便利接口，非真 system hierarchy)。`tool`/`developer` 角色不支持。
- **image**: 共享 ~400M SigLIP，任意图 → 896² → 4×4 avg-pool → **固定 256 tok/图**。P&S 默认: `do_pan_and_scan=False, min_crop_size=256, max_num_crops=4, min_ratio_to_activate=1.2`；768 tok = 原图 256 + 2 crops×256。**先 crop 后 resize** (故上游 cap 会先丢分辨率再被 P&S 放大)。mean/std=[0.5,0.5,0.5]。官方**无**输入源像素上限。
- **agentic**: Gemma 3 **无 native function-calling 协议** (只能 prompt-engineered)。FunctionGemma = 独立 270M text-only 专训模型。官方强调稳定确定性 API 行为通常需 task-specific fine-tune。
- **vision 官方**: doc/OCR 是明确强项 (DocVQA 75.8 / TextVQA 57.8 IT)；4B 细粒度弱 (pre-train BLINK 38.0 / CountBench 26.1)。官方**未**明述 "OCR 强照片弱"，也无 dense-UI/SoM/WebArena 结果。

## 六源收敛 → "~1% = 真地板非 bug" settled

| 源 | 方法 | 指向 |
|---|---|---|
| §327 probe | 双模型同图同问控制变量 | 视觉分层崩坏 (OCR✓/photo✗) + 0-finish + agent-spec gap |
| B2 dom diag digest | 224ep 错因归因 | 95% agent-limit, finish 23/224, **非 scaffold bug** (98% parse) |
| §330 | pas-on/off 对照 | SR 被 0-finish 钉死 |
| **GPT 官方审计** (本文) | 官方文档对账 | 无单一 bug；主因=能力+termination 失配；用法可辩护 |
| 文献 (arXiv API 核) | 外部锚 | 无 4B VWA/WA 基线；12B+TTI WA 26.1% (post-trained)；12B-base 易子集~6.2% |
| §335 | action 级 | finish 15% vs 51%、element 固着、复读 |

⚠️ **§335 自我修正**: §335 把 "scaffold family-fit confound" 列为三大层之一，**over-weight 了** —— §327 + digest 早已实测排除 (98% parse，scaffold 工作正常)，gap 是真能力差非格式失配。残留开放点 = never-finish 是不懂 finish 格式 (可修) vs 不判断完成 (不可修)，= Path 2 要答的。

## 文献锚 (arXiv API 核 ID 全真；数字正文级待核)

- **Gemma 3 4B 标准 VWA/WA = N/A in literature** → 我方 ~1% = 首个报告 (novel + 无外部锚)。
- 最近锚: 12B + TTI = WebArena **26.1%** (`2506.07976`, post-trained/test-time-interaction，**非 raw**)；12B base ~**6.2% TSR** on curated 易/中子集 + 对抗 (`2603.04364` DMAST，**已在我方 lit [5]**，应高估 vs full VWA)。
- 含义: 连 12B (3× 大) 要 post-train 才 26% / base 易子集才个位数 → 4B full-VWA ~1-2% **方向吻合**，非 bug。WA≠VWA，无一直接可比 → honest 标 N/A + directional。

## 决策叉口 (advisor-level，未定)

- **Path 1 (守 current shared-scaffold control)**: B2 = "固定 scaffold 下跨族 transfer 塌地板"。lean / 可辩护 / 不动 prereg。§8 disclose scaffold-transfer (有意控制) + termination 失配 (机制)。**推荐为主。**
- **Path 2 (加 Gemma-native 条件)**: few-shot termination/loop-recovery，分解 floor scaffold-vs-capability。科学更 richer，喂 router-vs-module 论点 (digest "需 module")。代价: 新条件 → prereg amendment + 算力 + 撞 D4/时间线。**GPT 也只建议 ADD 不 replace** → 当前数据不作废。

## reviewer-defense 价值

本审计本身 = 官方源确认用法正确 + § 327 控制 probe + 文献锚 → **三锁** "~1% 是真地板非误用"。直接进 §8 limitations / B2 baseline justification。

## GPT 建议的最小实验计划 (A0-A5，一次一变量)

A0 current baseline 冻 · A1 只删 1024-cap (image path) · A2 只 Gemma-native prompt (system 入首 user turn + 去 Qwen markers + few-shot finish/loop/uncertainty) · A3 只切 checkpoint sampling (do_sample=True/top_k=64/top_p=0.95, ≥3 seeds) · A4 只升 transformers 5.10.1 · A5 SoM 双图 (raw+SoM)。
→ **A1 作 probe 已跑** (cap 共享，canonical 改它破坏对照，故只 probe)；A2 = Path 2 决策；A3/A4/A5 = 可选 sensitivity/disclosure。

## 官方源 (GPT 引，待 `curl` 复核关键值如 generation_config)

model card huggingface.co/google/gemma-3-4b-it · tech report arXiv:2503.19786 · ai.google.dev/gemma (prompt-structure / huggingface_inference / vision/image / functiongemma) · transformers Gemma3 processing/image_processing 源码 · cookbook Function_Calling_with_HF.ipynb
