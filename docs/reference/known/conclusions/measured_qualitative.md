# 定性测量（C 批：30 条无数字 MEASURED，§36–§397.9）

Claude 主 session 逐条通读，2026-07-28。这批全是**实现层事实** —— 没有 SR/成本那类数字，
但每条都是"代码实际怎么样"的实测。聚合非转写。

---

## 1. 散文声称的，代码里没有

这是本批最集中的一类，也是最贵的一类：**paper/plan 写了，实现不存在**。

| 声称 | 实测 | § |
|---|---|---|
| 有 TOST + SR-Wilcoxon + `wilcoxon_skipped.csv` | `_compute_statistical_tests` **只有** McNemar + cost/latency Wilcoxon。三样都没有 | §186 |
| canonical producer 做 Holm/Bonferroni/FDR 校正（prereg 有 **3 处**承诺 transparency Holm-sig count） | `aggregate_phase1_full_prereg_decision.py` 里 `holm\|bonferroni\|fdr` 的 grep 命中 = **0** | §213.1 |
| H2(a) 测 **per-task** cost ratio（paper §1 line 9） | 代码测的是 **median-of-modes**（marginal medians） | §177 |
| prereg §7 的 tree-hash chain witness 有测试保障 | 测试里**完全没有断言**，只验证了可变的 HEAD SHA。原文评价：*该 prereg 契约正是 Claude 自己写的，仍然漏了 test 层 —— 最突出的 implementer blind spot* | §187 |
| `_lib_paper_grade_gates.sh` 注释称 stale-resume 检查会比对 HF revision | 代码根本没检查（B 批 §229，2 周 vapor） | §229 |

**原文片段**（§186）：*"prose 声称有 TOST + SR-Wilcoxon + wilcoxon_skipped.csv；
代码只有 McNemar + cost/latency Wilcoxon"* —— codex unique P0 OOB。

## 2. 类型系统漏洞（同一个 class 在三处数据流复发）

**§194**：`isinstance(True, int) == True`（Python bool 是 int 子类）⇒

- `score=True` 能通过 numeric 校验（gemini OOB）
- `step_idx` 的 `'0'` 字符串不触发 restart 边界
- `needs_reevaluation` 的字符串 `'false'` 被 `bool()` 判 truthy ⇒ **合法 episode 被静默 quarantine**

三处都是 **B-283 string-coercion class 在不同数据流位置的兄弟**。

## 3. 契约漂移：路径变了，消费者没跟

**§384.3**：canonical som run 里 `screenshot_annotated.png` 命中 **0**。
旧契约 `<task>/step_NNN/screenshot_annotated.png` → canonical `<task>/som/step_NNN_som.png`
（`som.py:606`，B-1828 deferred-save 重构副作用）。**该字符串散在 9 个 extractor 里**。

> caveats 原文：*"若不先查就跑，表现是 'missing artifacts, skip' 逐 task 静默跳过"*
> —— 不报错、不中断，只是结果凭空少一批。

## 4. 视觉呈现即数据失真

**§186**（Claude F1 + codex F5 + gemini F2 **三方命中，最高信心**）：
热图 caption 写 `gray=N/A`，但 `imshow(cmap='RdYlGn', vmin=0)` **未调 `set_under`**
⇒ N/A 被渲染成 **dark red**（读者会读成"最差"而非"没数据"）。
`grep set_under` 得 0 hits 实证。原文归类：**visual fraud 类**。

## 5. Router 的负结果（定性侧）

| 测的什么 | 结果 | scope | § |
|---|---|---|---|
| learned triage Pareto 胜过 always-cheapest 的 cell 数 | **0 / 6** | 6 cells, k=6 | §387.16.4 |
| Prereg-style multiclass LR 可训折数 | **0/5 fail closed**（每 fold 只剩 DOM） | B0·reddit, 55 个 oracle-labeled task, min_class_n_train=10 | §379 |
| best_mode 跨折稳定性 | red·B0 五个外折选 **DOM/DOM/SoM/SoM/DOM** | 5 外折 | §392.2 |

§387.16.4 的 caveat 值得整段留着：

> *"cls/B2 尤其说明问题：always-cheapest(Vision) 本就与 best_sr(SoM) SR 打平且便宜 22.1%，
> 而 learned 把 212/224 送去最便宜 = **重新发现『永远用 Vision』**，−20.8% 还不如那个固定策略。"*

§392.2 的意义比"阈值 in-sample"更根本：

> *"用全量结局挑一个 best mode 的管线，不只是对阈值过于乐观，
> **它报告的那个 mode 选择连自己的重采样都复现不出来**。"*

## 6. 幻觉率：同 namespace 内的符号效应

**§397.9**（本批唯一直接进 paper 的结论）：

| 对比 | 两臂 id regime | SoM prompt 的效应 | 两个分母 |
|---|---|---|---|
| DOM → P-prompt | 都原生 nodeId | **升高** | action-step 5/6 · episode 5/6 |
| P-text → P-SoM | 都 1..K | **降低** | 6/6 · 5/6 |

**符号相反 = 真交互**，且对探测器不对称免疫 —— 每个符号是各自 namespace 内部的事实。
原文：*"比符号安全，比跨 namespace 的量级才不安全"*；且比 §397.7 那个"降幅更大"强，
因为后者被 Gemini 指出是排序代数强制的，而**符号翻转不可能被任何单调的探测器偏差造出来**。

⚠️ **台账给这条挂了 `named by RETRACTED §397.10` 的 flag，主 session 裁定该论证仍然成立**：
§397.10(1) 修正的是"compact namespace 只有两个 mode"这个隐含说法（实为三个，SoM 也在内），
而这里用的两组比较（DOM/P-prompt 都 native、P-text/P-SoM 都 1..K）**恰恰被 §397.10 确认**。
主 session 另有实测支持：模型输出的 element_id —— p-som 1/12/68 · p-text 1/13/72
vs p-prompt 139/4074/26235 · dom 2/3606/61833。

**并列记录**（§397.7，同批）：Gemini 指出"交互"多半是代数必然 ——
P-SoM=最小 ∧ P-prompt=最大 ⇒ `min−max ≤ 任意其它两两差`。极值条件在 action-step 下
**5/6 格是强制的**，episode 分母下只 2/6 强制而结论仍 6/6 ⇒ **那 4 格才是非平凡支持**。
原文自评：*"我上一版之所以选中这条来立论，正因为它『换分母也 6/6』—— **稳健 ≠ 有信息**"*

## 7. 上游 provenance（写 paper 必须说清的）

**§251.4**：upstream VWA 用**文本串** `click [id]` / `type [id] [content]`，由 regex 解析
（`prompt_constructor.py::_extract_action`）；`grep 'tool_choice|tools='` = **0**
⇒ 既非 JSON 也非 tool-call。

> 因此 **P79 的 structured serialization（B0 tool-call / B1B2 JSON）是 P79 自创而非继承**
> —— 原文注明这一点必须在 paper 里说清。

## 8. 排版：页数的真凶不是字数

**§396.6**（超 §396.1 早期结论）：`pdftotext` 逐页数表发现表 1-4 **全被推迟堆在第 5 页左栏**，
而引用它们的正文在 2-4 页。根因 = Table 1 声明为 `table*`（跨栏浮动只能落页顶，
排队时 LaTeX 保持浮动体顺序 ⇒ 后面所有单栏表一起被卡住）。
**Table 1 改单栏后：9 页 → 8 页，一个字没再砍。**

对照 §396.1 的早期实测：字号档（`\small`/`\footnotesize`/`\scriptsize`）**对页数完全无影响**。

> 可复用规则（原文）：*"页数超了先 `pdftotext -f N -l N | grep -c '^Table'` 逐页数表
> 看有没有堆叠，再决定砍不砍字 —— **砍字是不可逆的内容损失，浮动体是免费的**"*

## 9. 跨 AI 的独有价值（一条实证）

**§142.3**：gemini Mode C 确认 Claude 的 F4 attack（B0 seed no-op）**已被 paper layer defused**
—— prereg §7 早写了 "server-side determinism is best-effort"。

> caveats 原文：*"这是 cross-AI 的独有价值 —— **Claude single-AI 不读 paper drafts
> 拿不到 paper-layer state**"*

## 10. 零散但值得记的

- **§128.4** format-variation 的 `extract_marks` regex **保留 label**（与 multimode Bug 2 不同）
  ⇒ 8 variant 内部对比 valid。⚠️ 但 `fmt_som_standard` baseline ≠ Stage 2B production envelope，
  且 §133b 表把该脚本 Bug 2 标 ✗ —— **两处记录不一致，未调和**
- **§184** VWA 任务配置的 N/A 判定模式**穷举实测**：cls+red 只出现三种
  （`reference_answers: null` / `fuzzy_match: 'N/A'` / `must_include: [...]`）⇒
  据此把 gemini F4（推测有其他 N/A pattern）降级为 P2
- **§387.7** `[SOM_MARKS]` 与可操作元素集一致（P-SoM scaffold 干净），**但样本仅 8 例**，
  原文建议 B0/B1 交叉核对后再写进正文
- **§391.1** 6 个 WA pilot 目录全 clean。⚠️ 原文提醒：*"『没发生』是事后观察不是契约"*
  —— `P79_PAPER_GRADE=0` 少的两道防线在已落数据上碰巧没触发，附录写不出"paper-grade 采集"
- **§157.5** `mint_run_id` 同秒 3 次调用产生 3 个 distinct ID（`${PID}_R${RANDOM}`）。
  ⚠️ 标 `superseded_by §183`
- **§134.4** 两个 DL 实现合并后一致到 **1e-9**
- **§130.4** codex exec 4 次 fire：1 次完整、**3 次 partial/empty**（audit prompt mid-stream
  cutoff）。归因于 stdin EOF + block-buffered stdout + isatty=false 三因素，**该归因未做对照实验**
- **§131.4** /stress v5（scripts-first）OOB 比例 **4/6**（v5 前典型 2/6）。⚠️ 自评统计，n=1 次调用
- **§167a.2** phase1_plan 自审 8 findings，**2/8 OOB**。⚠️ Mode A only（user 打断了 cross-AI dispatch）
