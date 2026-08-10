# search_results 核验层

**核验日期**：2026-08-10 · **核验者**：Claude（脚本 + arXiv API + crossref API）

> 五个 GPT 搜索结果文件**保持原样不改**，核验结论集中在本文件。
> 规则来自 `GPT_SEARCH_PROMPTS.md` 自己定的那条：**P2/P3/P4 回来的引文一律不直接采信**，
> 用 arXiv API 核（WebSearch 当月索引滞后，"查无此文"可能是假阴性）。

---

## 1. 总账

| | 数量 | 结果 |
|---|---:|---|
| arXiv ID（`arXiv:` 前缀锚定） | 33 | **30 clean** / 3 需判读 → 实为 **1 处真错** |
| DOI（crossref） | 27 | **18 一次通过** / 9 需判读 → 实为 **0 处编造**，2 处需换引用方式 |

**结论：这批搜索结果的引文质量高。** 唯一的实质错误是一个标题，且不影响该文献的可用性。

### 逐文件

| 文件 | 引文问题 | 说明 |
|---|---|---|
| `web_agent_representation_routing_related_work.md` (P2) | **0** | 27 个 arXiv ID 标题全部相符 |
| `negative_result_figure_design.md` (P4) | **0** | — |
| `methodology_negative_results_full.md` (P3) | **1 真错 + 1 换引法** | 见 §2 |
| `sustainability_measurement_methodology.md` (P5) | **0** | 7 处初判 MISMATCH **全是核验脚本的误报**（该文件表格第二列是长结论文本而非标题，被脚本当成标题去比对） |
| `UCL_AISD_COMP0191_Dissertation_Rules_2025-26.md` (P1) | n/a | 无学术引文；结论本身是"公开资料未见"，见 §3 |

---

## 2. 需要人跟进的 3 条

### ⚠️ 2.1 `arXiv:2502.11027` 标题错 —— **唯一的真错**

| | |
|---|---|
| P3 文件写 | *Diversified Sampling Improves Scaling LLM inference* |
| **arXiv 实际** | **On the Effect of Sampling Diversity in Scaling LLM Inference** |

同一行还把 `DOI:10.3233/HIS-2007-4204`（Bian & Wang, 2007, Hybrid Intelligent Systems）
与这篇 2025 年的 LLM 论文并列 —— 两篇文献缝在一行，**年份跨 18 年**。
两个 ID 各自都真实存在，但**标题↔ID 的绑定错了**。引用时用上表右列。

### ⚠️ 2.2 Holm 1979 只有 JSTOR ID，crossref 查不到

`DOI:10.2307/4615733` 在 crossref 返回 404。**不是编造** —— P3 文件自己标注了
"DOI / JSTOR stable identifier"，JSTOR 收录的老文章常不在 crossref。
Holm, *A Simple Sequentially Rejective Multiple Test Procedure*,
Scandinavian Journal of Statistics 6(2), 1979 是真实经典文献。
**引用时给期刊卷期页，不要给这个 DOI**（examiner 点进去会 404）。

### ✅ 2.3 `arXiv:1906.04908`（SPoC）不是错，是我的脚本误报

初判 MISMATCH 是因为脚本抓了表格的描述列当标题。P3 文件写得是对的：
**pass@k 的原始来源是 Kulal et al., *SPoC: Search-based Pseudocode to Code*
（`arXiv:1906.04908`）**，Chen et al. Codex（`arXiv:2107.03374`）沿用并标准化了无偏估计式。
⇒ 写 F10b 的方法学句时**两篇都引**，只引 Codex 会漏掉出处。

---

## 3. P1 的结论是一个有价值的负结果

9 项里 **8 项"公开资料未见"**，证据等级全 **C（官方但间接）**。但它确立了两件事：

1. ✅ **`COMP0191`** = *MSc Artificial Intelligence for Sustainable Development Project*
   （60 credit，100% dissertation）；**`COMP0190`** 只是 *Project Preparation*。
   ⚠️ 标 **D** —— 公开 catalogue 已切到 2026/27，2025/26 版需从 **Portico Module
   Information Document（Document Year 2025）** 调。
2. ✅ **元规则**（UCL Academic Manual 2025/26）：**若** assessment 设 word count，
   **则必须**在 module/assessment instructions 里说明 figures / tables / appendices /
   references 是否计入。

⇒ **这些数字只可能在 Moodle 或系里，搜公开网页永远搜不到。** 不要再花时间搜。
P1 §13 给了内部来源优先级（COMP0191 Moodle → Portico MID → AISD handbook → CS 系指南），
P1 §14 给了 9 条可直接发出的英文问句。

---

## 4. 核验方法（可复用）

```bash
# 脚本：prefix-anchored arXiv 提取 + arXiv API 批量 + crossref DOI + 标题相似度
.venv/bin/python3 <scratchpad>/verify_citations2.py
```

两条经验，下次直接用：

1. **正则必须锚 `arXiv:` 前缀**。裸 `\d{4}\.\d{4,5}` 会从 DOI 里抓出假 ID
   —— `10.1109/TPAMI.2022.3220744` → "2022.32207"、`10.2202/1544-6115.1585` → "6115.1585"。
   v1 就是这样"发现"了三个不存在的 arXiv ID，全是自己制造的。
2. **必须核标题、不能只核存在**。危险的失效模式不是编造 ID（那一查就死），
   而是**真 ID 配错标题**（§2.1 就是），因为它在 bib 里长得完全正常。
3. 表格结构因文件而异 ⇒ 标题列启发式会误报。**MISMATCH 一律回看原文行**再定性，
   本次 12 处初判 MISMATCH 里 **只有 1 处是真的**。
