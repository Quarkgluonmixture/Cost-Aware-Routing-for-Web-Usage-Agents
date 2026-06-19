# B3 (MiMo-VL-7B-RL-2508) 官方用法审计 — reviewer-defense lock

**Date**: 2026-06-17 (GPT web-audit) + 2026-06-18 (Claude code-verification) · **对标**: §338 / `B2_gemma_official_playbook_audit_2026-06-16.md`（B2 Gemma 同款审计）
**目的**: B3 若 promote 为 paper-grade baseline，确认我们的跑法符合 Xiaomi 官方推荐；逐条判定偏离是 (i) 对齐 / (ii) 可辩护偏离需 §8 disclose / (iii) 真正需改的 misconfiguration。
**Provenance**: GPT browsing-audit (零预设 prompt，见 `docs/checkpoints/codex_prompts/` 或 笔记 §342) 核实官方源 [HF card](https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-RL-2508) + [GitHub](https://github.com/XiaomiMiMo/MiMo-VL) + tech report [arXiv 2506.03569](https://arxiv.org/abs/2506.03569)；Claude 对 3 个"条件性需改"项做 in-repo 代码验证。

---

## 最终判词（先行）

> **无已确认的单一 misconfiguration。** greedy decoding / 替换默认身份 system prompt / 统一 SoM-plus-JSON scaffold 均为跨模型 fixed-scaffold 比较的**可辩护偏离**，应 §8 disclose。greedy 还得官方先例支持（tech report 图像理解评测就用 greedy search）。GPT 提的 3 个"条件性需改"项经 Claude 代码验证：**2 个对齐、1 个 telemetry gap**（见 §3）。

---

## 1. 官方用法核实（[官方明确] / [推断]）

- **解码**: [官方明确] 2508 README + HF card 部署段写 **`temperature=0.3, top_p=0.95`**（README 拼成 `topp`）。`top_k` / `repetition_penalty` / `do_sample` **未指定**。[官方明确] tech report 的**图像理解 benchmark 用 `greedy search`**（max vision pixels `4096×28×28`，max gen `32768`）；纯文本评测用 `temp=0.6, top_p=0.95`。→ [推断] `temp=0.3` 是通用部署推荐、greedy 是官方视觉 benchmark 评测策略，二者不矛盾；官方**无 web-agent SR 上 temp=0 vs 0.3 的消融** → 不能诚实声称 greedy 掉 X pp。
- **system prompt**: [官方明确] `chat_template.json` 在无显式 system message 时自动插入 `"You are MiMo, an AI assistant developed by Xiaomi."`（纯身份句，**不含 GUI/坐标/动作协议**）。模板**显式支持调用方提供 system message**（第一条是 system 则用调用方的）。官方**未**要求保留默认 prompt、未声明替换会破坏 grounding。
- **thinking**: [官方明确] RL-2508 默认 thinking ON（100% 控制成功）；`/no_think` 关思考（**99.84%**），**必须放 user message 最末尾**、其后无文本/图/视频。RL=推荐多数用户使用，SFT=面向继续 SFT/RL 起点；[官方未明确] RL/SFT 的 thinking 默认/控制率差异。
- **图像**: [官方明确] native-resolution ViT（动态分辨率非固定方形）；官方示例 `AutoProcessor.from_pretrained(model_path)` **从 MiMo checkpoint 加载**；`preprocessor_config.json`: `Qwen2_5_VLProcessor` / `Qwen2VLImageProcessor` / `min_pixels=3136 (=4×28×28)` / `max_pixels=12845056 (=16384×28×28)`。[官方明确] **单图/单视频输入要求 image 在 text 之前**（image-before-text 正例，text-before-image 反例）。
- **精度**: [官方明确] `torch_dtype=bfloat16`，arch `Qwen2_5_VLForConditionalGeneration`，HF 标 BF16 权重。
- **web-agent/JSON**: [官方明确] 有 GUI understanding/grounding（`examples/grounding.ipynb`），但**无官方端到端 web-agent runner、无 JSON mode / constrained decoding / 固定 action schema**；tech report RL 训练"No format rewards" → [推断] 严格 JSON 合规**非官方原生保证的接口**（我们同时测 GUI理解+自定义schema+SoM→element-ID映射+agent决策，失败不能一律归因视觉 grounding）。SoM + `[SOM_MARKS]` 文本列表无官方专门规范 → 属我们统一 observation scaffold（跨模型公平有方法学价值，§8 注明）。

---

## 2. 偏离判定表（官方 vs 我们 vs 判定）

| 项 | 官方 | 我们 | 判定 | §8 disclose |
|---|---|---|---|---|
| checkpoint | RL-2508 推荐 | RL-2508 (hash pinned) | 对齐 | 写 hash |
| 精度 | BF16 | BF16 不量化 | 对齐 | 方法记 |
| temperature | 部署 0.3 / 视觉评测 greedy | **0.0 greedy** | **可辩护偏离**(官方先例支持) | **是** |
| top_p | 0.95 | greedy 时无作用 | 可辩护偏离 | 与 greedy 一起写 |
| do_sample | 视觉评测 greedy | False | 可辩护偏离/官方先例 | 是 |
| max_new_tokens | 视觉评测 32768 | **4096** | **条件性**(见 §3.3) | **是+截断率** |
| thinking | 默认 ON | 默认 ON | 对齐 | 方法记 |
| 剥 `<think>` | 无官方 agent recipe | 解析末尾 JSON | 自定义 harness 合理 | 是 |
| 默认 system prompt | 自动插身份句 | **替换为 agent prompt** | **可辩护偏离**(身份句非 GUI prompt) | **是** |
| 自定义 JSON action | 无官方 schema | action_type+element_id | **可辩护偏离**(构念限制) | **是** |
| SoM screenshot + `[SOM_MARKS]` | 无官方规范 | 用 | 可辩护 scaffold | 是 |
| processor 类 | `Qwen2_5_VLProcessor` | Qwen2.5-VL processor | **取决于加载来源**(见 §3.1) | 写清 |
| 单图消息顺序 | image-before-text | (见 §3.2) | (见 §3.2) | — |

---

## 3. GPT "3 个条件性需改" 项 — Claude in-repo 代码验证

### 3.1 processor 从哪个 repo 加载 → ✅ 对齐
`p79/agents/mimo_vl_agent.py:98-102`: `AutoProcessor.from_pretrained(self.model_path, revision=self.model_revision)`，`self.model_path` = `XiaomiMiMo/MiMo-VL-7B-RL-2508`（L45）+ pinned revision。= GPT 的"情况1：从 MiMo checkpoint 加载 = 完全对齐"。min/max_pixels / resize / normalization 全来自 MiMo 自己的 `preprocessor_config.json`。**无需改。**

### 3.2 单图消息 image/text 顺序 → ✅ 对齐（单图）
`p79/agents/qwen3vl_agent.py:248-250`: **无 reference image 时**（纯截图=单图）`content.insert(0, {"type":"image","image":image})` → **image 插到最前 = image-before-text**，合官方单图要求。**有 reference image 时**（多图任务）走 `content.append(text "[Current screenshot]")` + `append(image)`（L246-247）= 多图 regime，官方单图约束不适用。**无需改**（单图路径已对齐；多图 regime 无官方约束，但若 MiMo 对多图顺序敏感可作 §8 次要 note）。

### 3.3 max_new_tokens=4096 是否截断 → ⚠️ telemetry gap（现无法验证）
pilot task0 的 steps JSONL **被 DGX env 幽灵清了**（§342，数据没了）→ 无法回算截断率。且 agent 只记 `output_tokens`（`local_qwen.py:103`）**无 explicit finish_reason/truncation flag**。**风险真实**: MiMo 每步先吐扩展 CoT 再吐 JSON，官方 vision 评测用 **32768**，我们只给 **4096**（注释"对齐 Stage-0 的 512"——但 Stage-0 是简单 probe 非复杂 classifieds 页+长 AXTree/SoM+长推理）。**不对称风险**: cap 太低→JSON 被截→无有效动作=**人为失败**(坏); cap 太高→长推理时慢一点(轻)。→ **A100 run 前 action**(见 §4)。

---

## 4. Action items

**① A100 B3 run 前（真 actionable，不动当前 cls fire 路径）**:
- **抬 `exp_v2_B3_*.yaml` 的 `max_new_tokens` 4096→8192~16384**（对齐官方量级；greedy 不长则提前 EOS，cost 仅长推理时）。
- **加 truncation telemetry**: 最简 = 后处理比 `output_tokens` 接近 cap 的步比例 = 截断率；或加 explicit finish_reason。A100 跑完即可验"cap 没切 JSON"。

**② §8 disclosure 清单**（都可辩护）: greedy(0.0) vs 官方 0.3 · 替换默认身份 system prompt · SoM+element-ID JSON 动作空间 · max_new_tokens cap · Qwen2.5-VL processor(从 MiMo checkpoint 加载)。

**③ 温度敏感性補强**（post-pilot，小规模）: 少量 task × {greedy, temp=0.3 ×3 seeds}，比 success / valid-JSON rate / repeat-action / premature-finish / per-step action disagreement。挡 reviewer"没按推荐参数"，比全重跑划算。诚实措辞: "decoding effect 方向与量级 unknown for this setting，官方无 web-agent temp 消融"。

---

## 5. 与 B2 audit 的 reviewer-defense 平行

B2 (§338): 官方用法审计 = 六源收敛的第 4 源 + 三锁之一。B3 本审计同构 = B3 promote 后的 reviewer-defense lock（"我们按官方文档跑 MiMo；偏离 X/Y/Z 可辩护并 §8 disclose；processor+image-order 经代码验证对齐"）。**前置条件**: B3 floor pilot（A100）confirm MiMo 不地板后才 promote + 才需此 lock 生效。
