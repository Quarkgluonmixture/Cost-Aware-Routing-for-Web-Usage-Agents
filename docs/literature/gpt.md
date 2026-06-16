# 结论

**B3 首选：`XiaomiMiMo/MiMo-VL-7B-RL-2508`。**

它目前最接近你的目标组合：

- **非 Qwen LLM backbone**：语言模型明确是自研 `MiMo-7B`；
- **general-purpose VLM**，不是 GUI specialist；
- **8B total parameters**，A100-40GB bf16 明显可行；
- 有第三方统一 harness 下的 **WebVoyager-SoM 34.0**，不是完全没有 agent signal；
- 它虽然为了部署兼容 `Qwen2_5_VLForConditionalGeneration`，但这只是 implementation interface，**不代表 Qwen backbone**。([Hugging Face](https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-RL))

**第二选择是 `GLM-4.6V-Flash`。**它的名义 web-agent 成绩更强，但对你这种固定外部 JSON action schema 存在明显的格式适配风险。

不过先说明最重要的搜索结论：

> **我没有找到任何一个同时满足全部条件、又有干净 raw zero-shot VisualWebArena/WebArena SR 的 ≤13B 非 Qwen general VLM。**

现有最接近的证据主要来自 `WebVoyager`、`OSWorld-G`、`ScreenSpot-V2`。这些不能直接当 VWA SR：VWA 是 910 个 self-hosted、execution-checked visually grounded tasks；WebVoyager 是 15 个真实网站上的开放式任务，主要用 GPT-4V 判断结果。([arXiv](https://arxiv.org/html/2401.13649v2))

------

## 1. Ranked candidate table

这里的排序轴是：

> **在你的固定 shared prompt + shared JSON action space 下，得到非 floor、可分析信号的概率**

而不是普通 VLM benchmark 的综合排名。

| Rank  | Model                          | LLM backbone（cross-family?）                                | Size                           | bf16@40GB?                                                   | General / GUI                                                | 最好公开 web-agent 或相关信号                                | Floor risk                               | Open weights / HF                                            |
| ----- | ------------------------------ | ------------------------------------------------------------ | ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------- | ------------------------------------------------------------ |
| **1** | **MiMo-VL-7B-RL-2508**         | **MiMo-7B，Y**。Qwen2.5-VL 只是兼容的部署 class，不是语言骨干 | **8B total**                   | **Yes**，bf16 权重约 16GB                                    | General-purpose；包含 broad GUI grounding 能力               | **VWA/WA: N/A**。`WebVoyager-SoM = 34.0`，由 GLM 论文在统一 evaluation framework 中复测；官方另报 `OSWorld-G = 56.1`。released checkpoint 零样本推理，但 checkpoint 本身经过 broad MORL ([arXiv](https://arxiv.org/html/2507.01006v1)) | **L–M**                                  | 官方 HF，BF16；Transformers-compatible ([Hugging Face](https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-RL-2508)) |
| **2** | **GLM-4.6V-Flash**             | **GLM-4-9B-0414，Y**                                         | **9B**                         | **Yes**，约 18GB raw weights                                 | General-purpose；native multimodal function calling，带 GUI/web agent 能力 | **VWA/WA: N/A**。官方 model card 报 `WebVoyager = 71.8`、`OSWorld = 21.1`；没有足够细节证明它使用与你相同的 shared JSON scaffold ([arXiv](https://arxiv.org/html/2507.01006v6)) | **M**：能力风险低，但 format-lock 风险高 | 官方 HF、MIT、Transformers；建议 `transformers>=5.2.0` ([GitHub](https://github.com/zai-org/GLM-V)) |
| **3** | **Molmo2-O-7B**                | **OLMo3-7B-Instruct，Y**                                     | **8B total**                   | **Yes**                                                      | General-purpose image/video/multi-image/grounding            | **VWA/WA: N/A**；未找到可信的端到端 WebVoyager/VWA/WA 数字。相关证据主要是通用 pointing、counting 和 visual grounding，不证明 termination/planning ([Hugging Face](https://huggingface.co/allenai/Molmo2-O-7B)) | **M**                                    | 官方 HF、Apache 2.0；`transformers==4.57.1`、`trust_remote_code=True` ([Hugging Face](https://huggingface.co/allenai/Molmo2-O-7B)) |
| **4** | **Phi-4-Reasoning-Vision-15B** | **Phi-4-Reasoning，Y**                                       | **15B**                        | **Tight / conditional**：约 30GB raw bf16 weights，30-step history 和 3,600 visual tokens 可能把 40GB 挤满 | General-purpose，明确覆盖 GUI grounding / computer-use       | **VWA/WA: N/A**；`ScreenSpot-V2 = 88.2 accuracy`，仅是单步 GUI grounding，不是端到端 SR ([Hugging Face](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B)) | **M**                                    | 官方 HF、MIT；`torch>=2.7.1`、`transformers>=4.57.1`，官方建议 bf16 ([Hugging Face](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B)) |
| **5** | **ZAYA1-VL-8B**                | **ZAYA1-8B MoE，Y**                                          | **10B total；0.7B active LLM** | **Yes**                                                      | General-purpose；Qwen2.5-VL 仅作为 vision encoder            | **VWA/WA: N/A**；相关 grounding：`Point-Bench = 58.0`、`RefCOCO = 84.3`，不是 web-agent SR ([Hugging Face](https://huggingface.co/Zyphra/ZAYA1-VL-8B)) | **H**：无多步 agent/finish 信号          | 官方 HF、Apache 2.0；需要 Zyphra 自己的 Transformers branch ([Hugging Face](https://huggingface.co/Zyphra/ZAYA1-VL-8B)) |
| **6** | **Zamba2-VL-7B**               | **Zamba2-7B，Y**，hybrid SSM–Transformer                     | **8B total**                   | **Yes**                                                      | General-purpose；Qwen2.5-VL 仅作 vision encoder              | **VWA/WA: N/A**；只有 VQA/OCR/counting 等结果，没有 GUI agent SR ([Hugging Face](https://huggingface.co/Zyphra/Zamba2-VL-7B)) | **H**                                    | 官方 HF；需要 custom Transformers fork、`mamba-ssm`、`causal-conv1d`、FlashAttention ([Hugging Face](https://huggingface.co/Zyphra/Zamba2-VL-7B)) |

------

# 2. Top 3 recommendations

## 1. MiMo-VL-7B-RL-2508：最合适的 B3

它是目前最平衡的选择。语言骨干明确是 `MiMo-7B`，总规模只有 8B；官方模型卡虽然说它与 `Qwen2_5_VLForConditionalGeneration` 完全兼容，但那只是为了复用部署代码，不改变 decoder family。([Hugging Face](https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-RL))

更关键的是，`WebVoyager-SoM = 34.0` 不是 Xiaomi 自己随便列出的数字，而是 GLM-4.1V 论文把 MiMo 放进其统一 evaluation framework 得出的复测结果。这至少证明它可以在**外部制定的 SoM web-agent harness** 中产生明显非零信号；这比只有 `ScreenSpot` 或通用 VQA 的候选更接近你的使用条件。([arXiv](https://arxiv.org/html/2507.01006v1))

主要 caveat 是训练制度：

- `MiMo-VL-7B-RL` 做过 **Mixed On-policy Reinforcement Learning**；
- 这是 broad multi-domain RL，覆盖 visual reasoning 和 GUI grounding；
- 它不是在 VWA 上额外 agent-SFT，也没有 evaluation-time TTI；
- 但若你的“raw”定义严格到**不允许 checkpoint 自身包含任何 RL post-training**，它就不满足。此时可以换 `MiMo-VL-7B-SFT-2508`，但后者没有可核验的 WebVoyager 数字，防 floor 证据会弱很多。([arXiv](https://arxiv.org/abs/2506.03569))

**综合判断：B3 就选 `MiMo-VL-7B-RL-2508`。**

------

## 2. GLM-4.6V-Flash：上限可能最高，但必须先做 format gate

它的硬件和架构条件都很漂亮：

- `GLM-4-9B-0414` decoder；
- 9B；
- general-purpose；
- native multimodal function calling；
- 官方报 `WebVoyager = 71.8`。([arXiv](https://arxiv.org/html/2507.01006v6))

但你的 scaffold 恰好踩中 GLM 家族最危险的点。2026 年的 MedCUA-Bench 把 `GLM-4.1V-9B` 接入固定 BrowserGym action schema 后，它在 **432/432 tasks 中没有产生任何可执行 browser action**；论文将其归类为 `zero-action (format lockout)`，即持续输出自己的 proprietary action format。([arXiv](https://arxiv.org/html/2606.03203v1))

这不证明 `GLM-4.6V-Flash` 也会失败——它新增了 native function calling，可能已经改善——但足以让它排在 MiMo 后面。官方仓库还明确提醒 GLM 不同版本使用不同 conversation templates，并为 vLLM 指定 `--tool-call-parser glm45` 与 `--reasoning-parser glm45`，说明模型对 native protocol 的依赖确实较强。([GitHub](https://github.com/zai-org/GLM-V))

**正确用法**：先拿你完全不改的 shared JSON prompt 跑一个 20–30 episode conformance pilot，检查：

- 是否泄漏 `<|begin_of_box|>`、native tool-call 或 GLM action tokens；
- parse-valid 是否仍然高；
- 是否能够发出 `finish`；
- 是否出现 zero-action 或重复 native-format 输出。

只要通过这个 gate，GLM-4.6V-Flash 可能成为比 MiMo 更强的 B3。

------

## 3. Molmo2-O-7B：最干净的 architecture-control 选择

这是候选里最“研究设计干净”的一个：

- `OLMo3-7B-Instruct` decoder；
- 8B total；
- `SigLIP 2` vision backbone；
- general-purpose；
- 模型、数据和训练生态都强调开放性；
- 没有因为名字包装而偷偷落回 Qwen decoder。([Hugging Face](https://huggingface.co/allenai/Molmo2-O-7B))

问题也很直接：**没有端到端 web-agent 成绩**。Molmo 的 pointing/grounding 很强，但 Gemma 已经证明，perception 或 OCR 尚可不代表能够做 multi-step action selection 和 `finish`。因此它更适合作为：

- 禁止 broad RL checkpoint 时的 B3；
- 或者 MiMo/GLM 之外的第二个 cross-family robustness ablation；

而不是你“必须保证不 floor”时的第一选择。

部署上要独立环境：官方模型卡指定 `transformers==4.57.1` 且需要 `trust_remote_code=True`，不建议和你现有 Gemma/Transformers 5.x 环境硬塞在一起。([Hugging Face](https://huggingface.co/allenai/Molmo2-O-7B))

------

# 3. Explicit reject list

## 因 Qwen LLM backbone 排除

- **InternVL3.5-8B**：8B 版本使用 Qwen3-series language model；名字是 InternVL，不代表 decoder 是 InternLM。([Hugging Face](https://huggingface.co/OpenGVLab/InternVL3_5-8B?utm_source=chatgpt.com))
- **Ovis2.5-9B**：官方 NOTICE 明确列出 `Qwen3-8B` 为基础语言模型。([Hugging Face](https://huggingface.co/AIDC-AI/Ovis2.5-9B/blob/main/NOTICE?utm_source=chatgpt.com))
- **STEP3-VL-10B**：明确使用 `Qwen3-8B decoder`。([Hugging Face](https://huggingface.co/stepfun-ai/Step3-VL-10B-Base?utm_source=chatgpt.com))
- **Molmo2-8B**：基于 `Qwen3-8B`；只有 **Molmo2-O-7B** 才是 OLMo3 backbone。([Hugging Face](https://huggingface.co/allenai/Molmo2-8B))
- **MolmoWeb-8B-Native**：使用 Qwen3-8B，而且本身就是 web-specialized checkpoint。([Hugging Face](https://huggingface.co/allenai/MolmoWeb-8B-Native))
- **Penguin-VL-8B**：官方模型卡明确写 `Qwen3 language backbone`。([Hugging Face](https://huggingface.co/tencent/Penguin-VL-8B))

## 因 GUI/computer-use specialist 排除

- `UI-TARS`
- `OS-Atlas`
- `Aguvis`
- `Holo`
- `OpenCUA`
- `MolmoPoint-GUI`

例如 `MolmoPoint-GUI-8B` 官方直接定义为“specialized for GUI pointing”，且只输出单个 point，不是与你其他 baseline 平行的 general instruction VLM。([Hugging Face](https://huggingface.co/allenai/MolmoPoint-GUI-8B))

## 因已有明显 floor 信号排除

- **Phi-4-multimodal-instruct 5.6B**：虽然是 Phi backbone、尺寸合适，但在固定 screenshot-only computer-use harness 上只有 **1.4% strict success**，平均仅 3.5 steps，表现和你的 Gemma floor 太相似。([arXiv](https://arxiv.org/html/2606.03203v1))
- **Idefics3-8B-Llama3**：MedCUA strict success **0.0%**；会发动作，但没有任何任务达到 checker 接受状态。([arXiv](https://arxiv.org/html/2606.03203v1))
- **Kimi-VL-A3B-Thinking**：非 Qwen，但在 MedCUA 仅 **1.9%**，而在 GLM 论文的 `WebVoyager-SoM` 中也只有 **1.8%**，floor 风险过高。([arXiv](https://arxiv.org/html/2606.03203v1))
- **GLM-4.1V-9B-Thinking**：本身 `WebVoyager-SoM = 69.0` 很强，但在固定 BrowserGym schema 中 432/432 zero-action；对你的研究协议风险太高，因此只考虑其后继 `GLM-4.6V-Flash`。([arXiv](https://arxiv.org/html/2507.01006v1))

## 因显存或规模排除

- `Mistral Small 3.1 24B`
- `Llama-4-Scout`
- `GLM-4.5V`
- `GLM-4.6V 106B`
- `Ovis2.6-80B-A3B`

MoE 的 active parameters 较小不等于 bf16 权重只占 active-parameter 大小；你的约束是单卡加载完整 unquantized checkpoint，因此仍然不合格。

------

# 4. Honest uncertainties

### 没有真正满足全部条件的 direct VWA/WA anchor

MiMo、GLM、Molmo、Phi、ZAYA、Zamba 都没有我能核验的标准 raw `VisualWebArena` 或 `WebArena` 数字。因此不能根据 34.0 或 71.8 推断它们在你的 VWA Classifieds 上会得到多少 SR。

尤其是：

- `WebVoyager` 使用真实网站和 GPT-4V-based evaluator；
- VWA 使用固定 self-hosted websites 和 execution-based checker；
- prompt、SoM、accessibility tree、action vocabulary、最大步数都会改变结果。([arXiv](https://arxiv.org/html/2401.13649v2))

### MiMo 的训练边界

`MiMo-VL-7B-RL-2508` 是 evaluation-time zero-shot，但不是“未经 RL 的原始 instruction checkpoint”。它做过 broad MORL。我的判断是这仍然属于与你的 `Qwen3-VL-Instruct`、`Gemma3-it` 相同的“released general-purpose post-trained model”类别，而不是 `model + WebArena-specific TTI`。但论文中必须把这一点写明，不能简称为 “no RL”。

### GLM 的 71.8 协议不够透明

官方 model card 给出了 `WebVoyager = 71.8`，但没有在同一处完整披露：

- exact prompt；
- action schema；
- thinking/non-thinking mode；
- retry policy；
- evaluator version；
- native function-call parser 是否参与。

因此它是强能力信号，不是与你实验可直接横比的 baseline。

### GUI grounding 不保证 termination

`OSWorld-G`、`ScreenSpot-V2`、`Point-Bench` 测的是定位或单步 grounded perception。它们不能证明模型会：

- 识别任务已经完成；
- 发出 `finish`；
- 避免 unchanged-page loop；
- 在 30-step horizon 中维持状态。

这正是你的 Gemma 症状最需要警惕的部分。

------

# 最终选择

## 推荐的正式 B3

```text
B3 = XiaomiMiMo/MiMo-VL-7B-RL-2508
Backbone = MiMo-7B
Size = 8B total
Precision = bf16, unquantized
Prompt/action space = 与 B0/B1/B2 完全相同
Training-label disclosure = general-purpose checkpoint with broad MORL;
                            no VWA-specific SFT, TTI, or online adaptation
```

它不是完美候选，但在 **cross-family、体量、开放部署、general-purpose、外部 web-agent signal、shared-harness 兼容迹象** 这几个条件的交集里，目前最合理。

**GLM-4.6V-Flash 应作为候补 B3′。**先过固定 JSON schema 的小规模 conformance gate；一旦确认不会 native-format lockout，它可能提供比 MiMo 更强的非 floor signal。