# Codex prompt v2: 3-axis cascade effect size ablation (DOM → P-DOM → P-SoM → SoM)

## 用途

`scripts/analysis/axis_effect_size.py` v1 只算了 2 个 contrast（prompt axis = P-DOM↔P-SoM, image axis = SoM↔P-SoM）。漏了 **axis 1 (text payload structure: AXTree → [SOM_MARKS])**，对应 cascade 第一步 `DOM → P-DOM`。

paper_planning §2 cascade 图明确写了 4-level / 3-transition 结构（`paper_planning.md:31-43`）：

```
DOM       (AXTree + DOM prompt + 无图)
  ↓ axis 1: AXTree → [SOM_MARKS]   (text 结构 swap, prompt + image 不变)
P-DOM     ([SOM_MARKS] + DOM prompt + 无图)
  ↓ axis 2: DOM prompt → SoM prompt (prompt swap, text + image 不变)
P-SoM     ([SOM_MARKS] + SoM prompt + 无图)
  ↓ axis 3: + image                (image swap, text + prompt 不变)
SoM       ([SOM_MARKS] + SoM prompt + 有图)
```

**任务**：把 ablation 扩成 **3 个 axis-by-axis paired contrast**，让每个 axis 在每 metric 上的 signature 都暴露——尤其检验 reddit 上 `scroll_frac` 和 `selfcorr_count` 是否真有 **axis 1 vs axis 2 antagonistic cancellation**（手算粗看 +5.6 vs −4.9 / −0.17 vs +0.11 那种）。

## 三个 paired contrast

| Axis | Contrast | Controls |
|---|---|---|
| **1** (text payload structure) | `P-DOM minus DOM` | prompt=DOM, image=no |
| **2** (prompt) | `P-SoM minus P-DOM` | text=[SOM_MARKS], image=no |
| **3** (image) | `SoM minus P-SoM` | text=[SOM_MARKS], prompt=SoM |

注意 sign convention：每个 contrast 都是 "**later cascade 节点 minus earlier 节点**"，所以 `axis_1 + axis_2 + axis_3 ≈ SoM - DOM`（一致性可验证）。

## 改 `scripts/analysis/axis_effect_size.py`

修改最少必要：
1. 头部 docstring 更新为 3-axis cascade
2. `axes` dict 加 `"text"` 条目（contrast = "P-DOM minus DOM"）
3. site loop 里加第三个 `paired_contrast(modes_data["P-DOM"], modes_data["DOM"], ...)` 调用
4. `dominant()` 函数泛化成接受 dict[str, contrast_result] 而不是只 prompt/image 两个参数；返回 dominant axis name（"text" / "prompt" / "image" / "neither (all small)"）。Tie-break: 选 |effect| 最大的；如果都 < 0.1 (d_z) 或 < 0.1 (h)，返回 "neither"。
5. 输出 JSON 加 `axes.text` block + `results[site][metric]` 加 `text` key (跟 prompt/image 同 schema)
6. **新增 consistency check**: 对每个 site × metric，verify `mean_diff_text + mean_diff_prompt + mean_diff_image ≈ mean(SoM) - mean(DOM)`（容差 0.1pp 或 0.005 fraction）。在 JSON 里写 `consistency_check: pass/fail` per metric per site。

Markdown report 重写：
- 表格扩成 3 列 effect (text / prompt / image) per metric per site
- 加 "Cancellation patterns" 小节：对每个 metric，如果两个 axis 的 effect sign 相反且都 |d| > 0.1，flag 为 "antagonistic"，说明 endpoint (DOM↔SoM) 比较会 mask 内部 mechanism
- "Paper Section 5 implication" 改写为：
  1. 每个 axis 的 dominant metrics
  2. Antagonistic pairs (axis 1 vs axis 2 on scroll/selfcorr if confirmed)
  3. Cascade decomposition 的 paper 价值（"why 4 levels not 2 endpoints"）

## 输出位置

- `scripts/analysis/axis_effect_size.py` (in-place edit, NOT 新文件)
- `docs/analysis/cross_sites/axis_effect_size.json` (overwrite)
- `docs/analysis/cross_sites/axis_effect_size_report.md` (overwrite)

## 验证

跑完 self-check：
- N(reddit) 三个 contrast 都 = 210
- N(cls) 三个 contrast 都 = 234
- consistency check 全 pass per site × metric
- 至少一个 metric 上 axis 1 effect 不可忽略（|d_z| > 0.1 或 |h| > 0.1）—— 否则就和 v1 没区别
- 用 reddit 数据先手算 verify: search% axis 2 应该 ~ -13.8pp, type% axis 2 ~ -6pp, scroll% axis 1 ~ +5.6pp, selfcorr axis 1 ~ -0.17

## token 预算

~30K (一次 read 现有脚本 + 一次 read 现有 JSON + 改脚本 + 跑一次 + 写 markdown)

## 触发命令

```bash
~/.npm-global/bin/codex exec --skip-git-repo-check \
  -C /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents \
  -s danger-full-access \
  --output-last-message logs/codex_axis_v2.last.md \
  - < docs/checkpoints/codex_prompts/axis_effect_size_ablation_v2.md \
  > logs/codex_axis_v2.run.log 2>&1
```
