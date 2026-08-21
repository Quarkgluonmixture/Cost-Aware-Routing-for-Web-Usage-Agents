# VLM4RWD @ NeurIPS 2026 — 非归档投稿（REALM 稿的模板移植）

**目标 venue**: [Grounded and Faithful Vision-Language Models for Real-World Deployment](https://vlm4rwd.github.io/), NeurIPS 2026 Workshop, Sydney

| 项 | 值 |
|---|---|
| 投稿截止 | **2026-08-30** |
| 通知 | 2026-09-29 |
| Camera-ready | 2026-10 月 |
| 归档性 | **非归档** —— 作者保留全部版权, 可再投会议/期刊 |
| 页数 | 正文 8 页, **references 与 appendix 不计** |
| 评审 | 双盲, 必须完全匿名 |
| 格式 | NeurIPS 2026 (`neurips_2026.sty` 官方 kit, `dblblindworkshop` 档) |
| 提交口 | OpenReview `NeurIPS.cc/2026/Workshop/VLM4RWD` |

CFP 对本次投稿唯一相关的一句原文: *"Relevant published work may be submitted for
presentation but will not be eligible for workshop awards."* —— 即已在别处投递/发表的相关工作
可以投, 只是不参评 workshop 奖项。REALM (#192, 非归档) 与本投稿同为非归档, 不构成 dual-submission 冲突。

## 内容来源

Overleaf 项目 `6a59017b04233a73ed5ec570`, commit `a456bff`（= REALM @ EMNLP 2026 的提交稿）。

⚠️ 该稿的真源在 Overleaf, **不在本仓库**: `sections/` 是在 Overleaf 端手写的散文,
`tables/` 带着那边做的人工修复（7 处 caption + 25 张 table* 转换）。要更新内容,
`git -C ~/overleaf-aaai27 pull` 后重跑本目录的复制步骤, 不要从 `paper_drafts/realm/*.md`
走 `convert.sh` —— 那条路会覆盖掉 Overleaf 端的手写内容（`overleaf_sync.sh:25-31` 有同样警告）。

## ⚠️ 本次移植最重要的一条教训：`wrapfigure` 会静默吃掉 caption

ACL 双栏 11pt A4 → NeurIPS 单栏 10pt US Letter, 同样叫「8 页」, 单栏能装的字数少三到四成。
第一版为了压进 8 页, 把三张小图改成了 `wrapfigure` 文字环绕, 并记录为「纯排版, 不动一个字」。

**那句话是错的。** `wrapfigure` 的垂直空间按预留行数计算, caption 超出部分**直接裁掉且不报
warning**。`sections/4_upperbound.tex` 的 Figure 3 caption 四句, PDF 只渲染到第二句中间
"…and the dark segment", 后面三段——包括本稿核心主张之一
*"Most of the apparent headroom is not attributable to representation diversity."*
——在整个 38 页 PDF 里**一个字都搜不到**。

**它逃过了全部既有检查**: `latexmk` exit 0 · 0 error · 0 undefined reference · 页数达标。
这类 bug 结构上不可能被「编译干净 + 页数对」发现。捕获它的是 codex 的独立审阅
(`docs/checkpoints/codex_outputs/vlm4rwd_template_port_FINAL_2026-08-21_105414.md` P0-1)。

**现在的验证手段**: 逐个 caption 抽取「最长纯文字片段」在 `pdftotext` 输出里比对（需先归一化
en-dash / 右单引号 / ligature, 否则 40+ 条假阳性）。当前 61 个已 input 的 caption 全部通过。

**结论**: 本目录**不再使用 `wrapfigure`**。压页只用「缩图 + 移图入附录」这类不触碰文字的手段。

## 模板移植做了什么

**机械层**（可脚本复现）:
- `table*`/`figure*` → `table`/`figure`（55 + 2 处）—— 单栏下没有全宽浮动体
- `\columnwidth` → `\linewidth`（4 处）
- `acl_natbib` → `plainnat`（`neurips_2026.sty` 自带 natbib）。⚠️ `plainnat` **不支持 `eprint`
  字段**, 四条只有 eprint 的记录（OSWorld / ST-WebAgentBench / budget-matched / routing survey）
  会丢 arXiv 链接 —— 已为它们补 `url` 字段
- `\usepackage[review]{acl}` → `\usepackage[dblblindworkshop]{neurips_2026}` + `\workshoptitle{}`

**页数层**（正文压进 8 页; 每步都量过页数, 全部不触碰文字）:

| 动作 | 收益 |
|---|---|
| 三张小图从 `\linewidth` 缩回 0.62–0.66（它们本是照 ACL 单栏宽 3.1in 画的, 单栏下 `\linewidth`=5.5in 把它们放大了 1.8 倍, 图内字号跟着虚胖） | 10 → 9 页 |
| Limitations 移入附录（Appendix B） | — |
| §7 Threats to validity 移入附录（Appendix A） | 9 → 8 页 |
| `fig_ceilings` 移入附录（Appendix F） | 抵掉加回限定语与对位段落的字数 |

正文因此保留 2 张图（`fig_overview`, `fig_sr_by_class`）; `fig_ceilings` 与
`fig_partition_forest` 在附录。

## 文字改动（全部, 逐条）

**主题对位**（user 2026-08-21 决定加, 经三家审阅后重写）: abstract +12 词点明四个 mode 无截图、
其中一对只差图像; intro 在 Contributions 之前加一段 `\paragraph{Reading this as a statement
about visual grounding.}`。**不含任何新数字。**

> 初版这段有三处硬错, 三家审阅全部命中, 已重写:
> 1. 写「**three** of the six modes are screenshot-free」—— 稿子 `2_setup.tex` 与
>    `final_dissertation/TERMS.md:143` 都是**四个**（DOM · P-prompt · P-text · P-SoM）;
>    且四个里只有两个（P-text / P-SoM）payload 是 `[SOM_MARKS]`, 另两个用完整 AXTree
> 2. 称整套设计是「grounding ablation」—— 实际只有 SoM vs P-SoM 一对在固定 text payload /
>    prompt family / **element-id regime** 的前提下只切图像; 其余对比都同时改动 identifier
>    namespace（`2_setup.tex` 自己写明 payload 决定 id regime）, 按 CLAUDE.md hard rule 属禁止比较
> 3. 拿 AUROC 0.483 当「模型不知道自己看到了什么」的证据 —— 它测的是 triage calibration,
>    且 `appendix_pb.tex:180` 里**本稿自己**论证该值「below chance … is consistent with a real
>    saving」, 原写法与本稿附录直接对立

**正文自足性**（三家共同命中: 移走限定而留下被限定的主张不是中性编辑）。三处限定语已写回正文:
- `1_intro.tex` rerun band 补「measured in two cells and assumed to transfer to the other six」
- `5_lowerbound.tex` 的 +22.54pp 分区补「large and significant only on the capable classifieds
  backbones, and **null on reddit**」
- `5_lowerbound.tex` 的 `red·B2` 例外, 把「Appendix A flips it」换成具体理由「zeroing six
  successes credited from accumulated site state removes that exception」

**原稿既有 bug 顺手修的**（这几条 REALM 在审版本里同样存在, **那边未改**）:
- `tables/tab02.tex` 引用 `tab:t04`（class-ablation 表）论证 non-separability, 而正主
  `tab:t05` 明写「This is not a separability test」, 并指出阈值是 87.5% 不是 tab02 写的 83%
  ——**tab05 的 caption 就是在纠正 tab02, 但 tab02 一直没改**。已改为指向 `tab:t05` 并降级为
  deployment convention
- `4_upperbound.tex` + `appendix.tex` 的 "paper-B precursors" 源项目残留（本稿从未定义 paper B）
- `appendix.tex` 的 "Multiplicity status" 段落落在 `\section` 之前, 被 LaTeX 归进
  Appendix B (Limitations) 名下 —— 已独立为 Appendix C

**交叉引用**: 3 处 `\S\ref{sec:threats}` → `Appendix~\ref{...}`; 1 处
`Figure~\ref{fig:partition}` 与 1 处 `Figure~\ref{fig:ceilings}` → `Appendix Figure~...`。

## 页脚为什么写「Submitted to … NeurIPS 2026」

这是官方 kit 的行为, 不是配置错误。`neurips_2026.sty:398-403`: `\@trackname`（含
`\workshoptitle`）**只在 `[final]` 下打印**, review 模式硬编码主会字样。录用后加 `final`
即正确显示 workshop 名。`main.tex` 的注释已按此更正（初版注释说它会打印 workshoptitle, 是错的）。

## 当前状态

- 正文 **8 页**（`\label{content-end}` 落在 p8）, 全文 39 页
- 编译 **0 error / 0 undefined reference / 0 undefined citation**
- overfull hbox 3 处, 最大 1.04pt（肉眼不可见）
- **61 个 caption 全部完整**（逐个比对 PDF 文本, 见上文验证手段）
- 49 张表全部自带缩放, 单栏下无溢出
- 匿名: 全文无作者名/机构/致谢/仓库链接（`pdftotext | grep -icE` 命中 0）

## 未做

1. **NeurIPS paper checklist**。官方 kit 带 `checklist.tex`, 主会强制、**本 workshop CFP 未要求**,
   故未加。官方原文已备在 `checklist.tex.unused`, 要加就改名并在附录末尾 `\input`。
2. **REALM 在审版本的同源 bug 未修**（tab02 引错表 / paper-B 残留 / Multiplicity 归属）。
   那是另一个提交, 动它需要单独决定。
3. **提交动作本身**。OpenReview 投递需本人操作。

## 构建与自查

```bash
cd deliverables/vlm4rwd && latexmk -pdf main.tex

# 正文必须 ≤ 8 页
grep -oE 'newlabel\{content-end\}\{\{[^}]*\}\{[0-9]+\}' main.aux

# caption 完整性 —— 编译干净不代表没丢字, 见上文
pdftotext main.pdf - | grep -F "Most of the apparent headroom"   # 必须命中
```

## 审阅存档

三家独立审阅（`/stress` Mode A+B+C, milestone scope）。⚠️ 按项目惯例
`docs/checkpoints/{codex,gemini}_outputs/` 整目录 gitignored（库中 0 文件）,
下面两份**只在本机**, clone 后不会有：

- Claude 自审 + Phase 4b 全量核验 + 统一裁决: 本 session
- codex（转换保真度 / Claude 未读章节）: `codex_outputs/vlm4rwd_template_port_FINAL_2026-08-21_105414.md`
  —— 9 findings, **独占捕获 caption 丢失**(P0-1); Phase 4b 核验 9/9 属实
- gemini 3.1 Pro（topic fit / 逐句攻击对位段落）: `gemini_outputs/vlm4rwd_fit_2026-08-21_105414.md`
  —— 6 findings, 独占捕获 id-regime 混淆; 1 条误判已降级

**可追溯的存档在这两处**: `docs/checkpoints/实验笔记.md` §473 与
`docs/reference/known/ledger.jsonl`（11 条, 查 `known.py --section 473`）。
