# Verdict-day 中/高风险砍词预写（2026-07-14）

用途：只作为 verdict-day 人工 splice 候选；不改 `aaai27_main.md`。原文均取自 2026-07-14 Round-2 后的现行 master，定位以节标题和唯一开头短语为准。计数严格使用空白分词的 `wc -w`；因此这里报告的是可复验的机械收益，不是 sentence-splitter 旧估值。

复验本文各标记块的统一 helper：

```bash
PREWRITE=docs/checkpoints/paper_drafts/aaai27/cut_prewrites_verdict_day.md
block_words() {
  awk -v s="<!-- $1_START -->" -v e="<!-- $1_END -->" \
    '$0==s {on=1; next} $0==e {on=0} on' "$PREWRITE" |
    sed 's/^> //' | wc -w
}
```

## C4 — 最近邻与 novelty defense

**现行定位 anchor：** `aaai27_main.md` §2，段首 `**Routing in LLM systems.**` 内，从 `LazyMCoT is an overlapping nearest neighbour` 到 `Adaptive Re-Ranking motivates`（当前 L75）。为保持连续替换块，现行夹在其中的 Vardanyan 最近邻也纳入。

**现行原文（精确）：**

<!-- C4_ORIGINAL_START -->
> LazyMCoT is an overlapping nearest neighbour, but studies natural-image VQA, difficulty-triggered within-visual re-observation per sample, rather than our web, cross-modal/serialization menu and task-level complementarity [@wang2026lazymcot]. A technical report also deploys heuristic accessibility-tree-to-vision escalation [@vardanyan2025browser]. We have not found fixed-model, learned, per-task selection across such a cross-modal menu coupled to drop-one characterization; Agent-E remains within DOM-text variants [@dhondt2024agente]. BoundaryRouter motivates cold-start evaluation [@wang2026boundaryrouter], while Adaptive Re-Ranking motivates the preregistered oracle-vs-realized comparison in §6 [@genc2026adaptivereranking].
<!-- C4_ORIGINAL_END -->

**压缩版全文：**

<!-- C4_COMPRESSED_START -->
> Nearest neighbors remain narrower: LazyMCoT triggers within-visual re-observation for natural-image VQA [@wang2026lazymcot], Vardanyan heuristically escalates accessibility trees to vision [@vardanyan2025browser], and Agent-E stays within DOM-text variants [@dhondt2024agente]. BoundaryRouter motivates cold-start evaluation [@wang2026boundaryrouter], while Adaptive Re-Ranking motivates the pre-registered oracle-vs-realized comparison [@genc2026adaptivereranking]; none couples fixed-model, learned per-task cross-modal/serialization routing with drop-one complementarity.
<!-- C4_COMPRESSED_END -->

**实测：**

```bash
o=$(block_words C4_ORIGINAL); c=$(block_words C4_COMPRESSED)
printf 'original=%s compressed=%s saving=%s\n' "$o" "$c" "$((o-c))"
```

```text
original=73 compressed=51 saving=22
```

**保留的 load-bearing 元素：**

- 五个最近邻全部点名且保留 citation：LazyMCoT、Vardanyan、Agent-E、BoundaryRouter、Adaptive Re-Ranking。
- LazyMCoT 的 natural-image VQA / within-visual re-observation 边界，以及 Agent-E 的 DOM-text-only 边界。
- 本文窄 novelty：fixed-model、learned per-task、cross-modal/serialization routing，并与 drop-one complementarity 联结。
- BoundaryRouter 的 cold-start 用途和 Adaptive Re-Ranking 对 pre-registered oracle-vs-realized comparison 的作用；没有把未落地的 §6 结果写成事实。

**丢失与风险：**删去 LazyMCoT 的 `difficulty-triggered`、`per sample` 两个粒度限定；把 “we have not found” 的文献检索限定压成 `none couples`，语气略硬。候选本体风险为**高**；采用此压缩版后的残余风险为**中**，verdict day 应确认 supplement 的 related-work table 仍保留完整逐篇边界。

## C6 — efficient-observation 最近邻

**现行定位 anchor：** `aaai27_main.md` §2，段首 `**Efficient observations and language priors.**` 的前两句（当前 L79），止于 `[@meng2026sema]`。

**现行原文（精确）：**

<!-- C6_ORIGINAL_START -->
> **Efficient observations and language priors.** FocusAgent and observation-reduction work compress accessibility-tree text [@kerboua2025focusagent; @enomoto2026observation], while ReVision adaptively filters the retained visual stream [@abaskohi2026revision]. Sema's hybrid ablation shows structured text is necessary to close the gap from compact visual tokens to raw-screen performance; because Sema retains a visual channel, it does not test screenshot-omission parity [@meng2026sema].
<!-- C6_ORIGINAL_END -->

**压缩版全文：**

<!-- C6_COMPRESSED_START -->
> **Efficient observations and language priors.** FocusAgent and observation-reduction compress accessibility-tree text [@kerboua2025focusagent; @enomoto2026observation], ReVision filters the retained visual stream [@abaskohi2026revision], and Sema finds structured text necessary to close the compact-visual-to-raw gap but retains vision, so it does not test screenshot omission [@meng2026sema].
<!-- C6_COMPRESSED_END -->

**实测：**

```bash
o=$(block_words C6_ORIGINAL); c=$(block_words C6_COMPRESSED)
printf 'original=%s compressed=%s saving=%s\n' "$o" "$c" "$((o-c))"
```

```text
original=55 compressed=42 saving=13
```

**保留的 load-bearing 元素：**

- FocusAgent / observation-reduction = 压 accessibility-tree text；ReVision = 过滤仍被保留的视觉流。
- Sema 的正面证据只到 “structured text is necessary to close the gap”，并明确 Sema 仍保留 vision。
- 明确 Sema 不测试 screenshot omission，避免滑回 “cheaper SoM” 或“丢图几乎不损性能”的 framing。
- 四组 citation 全保留；这些最近邻没有被删除。

**丢失与风险：**删去 `hybrid ablation` 方法标签，并把 `compact visual tokens to raw-screen performance` 缩成 `compact-visual-to-raw gap`；ReVision 的 `adaptively` 被省略。候选本体风险为**中高**；压缩后残余风险为**中**，主要是方法粒度降低，但核心证据边界仍在。

## C12 — independence-baseline 防误读

**现行定位 anchor：** `aaai27_main.md` §5.3，紧接 `same-task Jaccard ... 0.29–0.49` 后、以 `This is *above* the independence baseline` 开头的第二句（当前 L175）。

**现行原文（精确）：**

<!-- C12_ORIGINAL_START -->
> This is *above* the independence baseline (E[J] ≈ 0.06–0.10 at the observed SRs) — modes agree more than chance, exactly as shared task difficulty predicts — so the complementarity claim rests not on low overlap but on the unique-pass residue that survives it, and those unique-pass sets are distributed across task categories (search, comparison, navigation) rather than concentrating in one family.
<!-- C12_ORIGINAL_END -->

**压缩版全文：**

<!-- C12_COMPRESSED_START -->
> Because Jaccard 0.29–0.49 exceeds the independence baseline (E[J] ≈ 0.06–0.10), modes agree more than chance, as shared task difficulty predicts; complementarity rests on unique passes spanning search, comparison, and navigation—not low overlap.
<!-- C12_COMPRESSED_END -->

**实测：**

```bash
o=$(block_words C12_ORIGINAL); c=$(block_words C12_COMPRESSED)
printf 'original=%s compressed=%s saving=%s\n' "$o" "$c" "$((o-c))"
```

```text
original=61 compressed=32 saving=29
```

**保留的 load-bearing 元素：**

- 观测 Jaccard `0.29–0.49` 与 independence baseline `E[J] ≈ 0.06–0.10` 都保留。
- 明说 overlap **高于**独立基线、modes agree more than chance；不能把 low overlap 读成统计独立。
- shared task difficulty 的解释仍在。
- complementarity 的承重证据仍是跨 search/comparison/navigation 的 unique passes，而非 overlap 数字本身。

**丢失与风险：**删去 `at the observed SRs` 和 “rather than concentrating in one family” 的展开；后者由 `spanning ...` 正向表达替代。候选本体风险为**高**；压缩后残余风险为**低中**，因为 independence 防线与 complementarity 的证据对象均被显式保住。

## C14 — §5.5 behavioural observation + contribution #2 联动

**现行定位 anchor：** `aaai27_main.md` §5.5 全段，从 `On a matched-task Reddit subset` 到 `no mechanism claim is made here`（当前 L183）。

**现行原文（精确）：**

<!-- C14_ORIGINAL_START -->
> On a matched-task Reddit subset (N=48 [V]), the two knobs separate behaviourally: flattening the text payload cuts the search-loop rate roughly in half (DOM 22.7% → P-text/P-SoM 10.8%), while the SoM-prompt arms show a smaller false-positive finish gap than DOM-prompt arms (2.1pp vs. 6.3pp, measured under a since-retired success-adjustment layer — direction-only evidence) — *text representation appears to shape exploration; prompt wording appears to modulate commitment timing*. We advance this strictly as behavioural characterization on an archive substrate; it generates the hypotheses a mechanism study would test, and no mechanism claim is made here.
<!-- C14_ORIGINAL_END -->

**压缩版全文：**

<!-- C14_COMPRESSED_START -->
> On an archive matched-task Reddit subset (N=48 [V]), flattened text roughly halved search loops (22.7% → 10.8%), while SoM-prompt arms had a smaller false-positive finish gap (2.1pp vs. 6.3pp) under a retired success adjustment. These direction-only patterns motivate exploration/commitment hypotheses, not mechanism claims.
<!-- C14_COMPRESSED_END -->

**正文实测：**

```bash
o=$(block_words C14_ORIGINAL); c=$(block_words C14_COMPRESSED)
printf 'section_original=%s section_compressed=%s section_saving=%s\n' "$o" "$c" "$((o-c))"
```

```text
section_original=94 section_compressed=43 section_saving=51
```

### Contribution #2 联动改法（原句 → 改句）

**原句（现行精确文本，`aaai27_main.md` §1 Contributions，当前 L66）：**

<!-- C14_CONTRIB_ORIGINAL_START -->
> 2. **A behavioural hypothesis-generation observation** (§5.5): in a matched-task 2×2 ablation, text representation appears to shape *exploration* (flattened marks cut search-loop rate) while prompt wording appears to modulate *commitment timing* (SoM-prompt arms show smaller false-positive finish gaps). We advance this as behavioural characterization only; mechanism analysis is deferred to follow-up work.
<!-- C14_CONTRIB_ORIGINAL_END -->

**改句：**

<!-- C14_CONTRIB_COMPRESSED_START -->
> 2. **A hypothesis-generating behavioural observation** (§5.5): archive evidence links text format to exploration and prompt family to commitment; no mechanism claim is made.
<!-- C14_CONTRIB_COMPRESSED_END -->

**联动实测与 C14 合计：**

```bash
o=$(block_words C14_CONTRIB_ORIGINAL); c=$(block_words C14_CONTRIB_COMPRESSED)
printf 'contrib_original=%s contrib_compressed=%s contrib_saving=%s\n' "$o" "$c" "$((o-c))"
printf 'C14_coupled_saving=%s\n' "$((51 + o - c))"
```

```text
contrib_original=51 contrib_compressed=23 contrib_saving=28
C14_coupled_saving=79
```

**保留的 load-bearing 元素：**

- archive / matched-task / Reddit / `N=48 [V]` 的证据 vintage 与范围。
- 两个方向及全部关键数值：search loop `22.7% → 10.8%`；false-positive finish gap `2.1pp vs. 6.3pp`。
- retired success adjustment 与 direction-only 限定。
- exploration / commitment 只作为 hypotheses；正文和贡献清单都明确不作 mechanism claim。
- Contribution #2 仍存在，无需重排后续 router contribution 的编号。

**丢失与风险：**正文删去 DOM、P-text/P-SoM、DOM-prompt 的逐臂标签，把 2×2 的具体映射压成 `flattened text` / `SoM-prompt arms`；贡献清单不再展开两项指标，也不再写 “mechanism analysis is deferred”。候选本体风险为**高**；压缩后残余风险为**中高**：科学限定仍全，但贡献 #2 会显得更像次级 observation。若 verdict-day 需要进一步降贡献数量，应另行决定删除该 item 并把 router 重编号，不能只删 §5.5 而保留现行 contribution #2。

## C19 — joint Type-I 历史披露短指针

**现行定位 anchor：** `aaai27_main.md` §8 `**Statistics.**` 段末，以 `The pre-registration disclosed a joint Type-I exposure` 开头（当前 L222）。

**现行原文（精确）：**

<!-- C19_ORIGINAL_START -->
> The pre-registration disclosed a joint Type-I exposure of 0.0975 over {H1, H10} prior to H10's reclassification as an operational criterion; we retain that disclosure for transparency while noting that no joint FWER is defined under the current H10 semantics.
<!-- C19_ORIGINAL_END -->

**压缩版全文：**

<!-- C19_COMPRESSED_START -->
> Pre-registration records the historical 0.0975 joint Type-I exposure for {H1, H10}; under H10's current operational semantics, no joint FWER is defined.
<!-- C19_COMPRESSED_END -->

**实测：**

```bash
o=$(block_words C19_ORIGINAL); c=$(block_words C19_COMPRESSED)
printf 'original=%s compressed=%s saving=%s\n' "$o" "$c" "$((o-c))"
```

```text
original=39 compressed=21 saving=18
```

**保留的 load-bearing 元素：**

- 数字 `0.0975` 原样保留。
- `pre-registration` 是显式短指针，不把历史披露悄悄移出主稿。
- `{H1, H10}` 的原联合对象、H10 后来改为 operational 的语义，以及当前没有 joint FWER 的结论都保留。

**丢失与风险：**删去 `for transparency` 的自我说明和 `under the current H10 semantics` 的展开，用 `historical` 与 `after H10 became operational` 承担时间线。候选本体风险为**高**；压缩后残余风险为**低中**，因为数字、prereg 指针和重分类后果均仍在一句内。

## 汇总词数账

| 候选 | 现行词数 | 压缩后 | 实测收益 | 备注 |
|---|---:|---:|---:|---|
| C4 | 73 | 51 | 22 | 连续块含 Vardanyan |
| C6 | 55 | 42 | 13 | 标题计入前后两侧 |
| C12 | 61 | 32 | 29 | 只替换 independence 解释句 |
| C14 §5.5 | 94 | 43 | 51 | 正文 |
| C14 contribution #2 联动 | 51 | 23 | 28 | 必须与上一行同选 |
| C14 合计 | 145 | 66 | 79 | 汇总时只计此行，不重复计上两行 |
| C19 | 39 | 21 | 18 | 保留 0.0975 + prereg 指针 |
| **五候选全选** | **373** | **212** | **161** | C14 按正文+联动合计 |

以用户指定的现行基数 **5320** 词和 readiness 实测 splice 增量计：

| 场景 | 算式 | verdict-day 净词数 | 相对 5320 |
|---|---|---:|---:|
| 不选本页压缩 + Branch A | 5320 + 49 | 5369 | +49 |
| 五候选全选 + Branch A | 5320 − 161 + 49 | **5208** | **−112** |
| 不选本页压缩 + Branch B | 5320 + 149 | 5469 | +149 |
| 五候选全选 + Branch B | 5320 − 161 + 149 | **5308** | **−12** |

这里严格按任务指定的 `5320 + branch 增量` 口径记账；没有另行套用 checklist HTML comment 中 “interim placeholder 被整体替换” 的内部调整。五项全选提供 161 词 gross saving；它们是中/高风险候选的保守压缩，不等于完整 800–1000 词裁剪计划。

## Banned-phrase 与回归验证

仅抽取本文件六个压缩块（C14 正文与 contribution 分开）运行 master checklist item 9 的现行 pattern：

```bash
awk '
  /<!-- C(4|6|12|14|14_CONTRIB|19)_COMPRESSED_START -->/ {on=1; next}
  /<!-- C(4|6|12|14|14_CONTRIB|19)_COMPRESSED_END -->/ {on=0}
  on
' "$PREWRITE" |
  grep -nE "image[-]free|image-off|no image tokens|text-only cost|both Qwen cells|most of the.*mass" |
  wc -l
```

```text
0
```

全套回归（本任务只新增 Markdown，不修改 tracked 断言对象）：

```bash
.venv/bin/python3 -m pytest -q
```

```text
1476 passed, 11 skipped, 107 warnings in 52.00s
```
