---
type: reference
status: active
created: 2026-07-28
purpose: anti-redo ledger — "has this already been measured / adjudicated / retracted?"
---

# KNOWN — 防重做台账

`docs/checkpoints/实验笔记.md` 是 2 万行 / 396 节的 append-only 时序记录。**grep 只在你已经猜对关键词时有用**，而"这件事是不是已经做过了"恰恰是猜不到关键词的那类问题。这一层是给那类问题用的。

建于 2026-07-28（REBUILD_PLAN Phase 0），起因是一个 session 在一次会话里重做了已完成的工作**五次**，每次都要用户纠正。

## 怎么查

```bash
.venv/bin/python3 scripts/maintenance/known.py oracle              # 含 "oracle" 的全部
.venv/bin/python3 scripts/maintenance/known.py -t MEASURED self_drop
.venv/bin/python3 scripts/maintenance/known.py --section 302       # §302 及 §302.x
.venv/bin/python3 scripts/maintenance/known.py --flagged           # 被后续作废记录点名的
.venv/bin/python3 scripts/maintenance/known.py --absent            # artifact 确认已丢的
.venv/bin/python3 scripts/maintenance/known.py --stats
```

输出**永远**带 `caveats` 和作废标记。丢 caveat 是这个台账最危险的失效模式 —— 一个没有 scope 的数字正好会招来它本该阻止的那种误用。

## 下结论前问四句

来自五次实际错误，不是通用建议：

1. 这个量**已经被测过**了吗？（查台账，别翻 chronicle）
2. 这个决定**已经被裁定**过了吗？（裁定常常写在代码注释里）
3. 被比较的各臂之间，这个指标的**判定基准是同一个**吗？
4. 这是 in-sample 还是 out-of-sample？

第 5 条不在台账能力范围内，只能靠人：**推理"什么挡着什么"之前，先回读主机角色表**（CLAUDE.md 三层算力）。2026-07-28 那次凭空造出的算力冲突，规则当时就在 context 里。

## 五种记录

| type | 回答的问题 | 数量 |
|---|---|---|
| `MEASURED` | 这个量测过吗？值多少、什么 scope、什么 caveat | 961 |
| `ADJUDICATED` | 这个决定拍过吗？为什么这么定 | 863 |
| `RETRACTED` | 这个说法死了吗？为什么、被什么取代 | 177 |
| `CLAIM_UNVERIFIED` | 这是推论还是测量？（原文自称待验的一律进这里） | 92 |
| `DATA` | 什么数据存在、在哪、什么等级、能支撑什么 | 49 |

共 **2152** 条，覆盖 §1–§412。数据：`docs/reference/known/ledger.jsonl`。

> chunk 9（2026-08-01，+25 条）补的是一处**承重缺口**：§398 整节此前 **0 条**，而合并稿第 ③ 步（结构小于同模式重跑地板）的出处正是 §398.2 —— user 标完 508 条结论账本时，四步骨架里那一步一条都没被裁到。同批补入 §406。§403–405 属 session log，未收录。

## 怎么核验台账本身

```bash
.venv/bin/python3 docs/reference/known/verify_ledger.py              # 汇总
.venv/bin/python3 docs/reference/known/verify_ledger.py --show-fail  # 逐条看存疑的
.venv/bin/python3 docs/reference/known/verify_ledger.py --section 302
```

把每条记录里的数字拿去源文档比对，三级判定：**在被引 §** / **在笔记别处**（引用不够精确，非编造；笔记头部有分类索引表，且 `[finding]` 类按写作规范就是一行指针）/ **在 analysis 文档**（指针型 § 的目标）/ **哪都找不到**。

2026-07-28 首跑：1097 条带可核数字，**1093 条（99.6%）可追溯**，4 条查不到的全是 DATA 类记录里 artifact 文件名的时间戳（笔记正文本不会写，文件实测存在）。**未发现凭空捏造的数字。**

**核验器查不出的三件事**（比它能查的更重要）：
1. **数字被安到错的语义上** —— 数字真实存在，但对应的 quantity/scope 抄错了
2. **笔记本身记错了** —— 台账忠实复制一个错误。台账与笔记同源，互查永远发现不了
3. **漏记** —— 只能验已记录的，验不出该记而没记的

要把某条用于论文，回到 artifact 复算，别停在台账。

## ⚠️ 台账不能做的事

台账本身也可能变成污染源。已知边界：

1. **记录来自 chronicle 文本，不是从数据重算。** "台账说测过 X" = "笔记里记着测过 X"。`artifact_exists ✓` 只说明文件在磁盘上，**不代表那个数字被复核过**。要用于论文，回到 artifact 复算。
2. **`⚑ named by RETRACTED §N` 是待查线索，不是判定。** 它只表示某条作废记录提到了这个 §，谁对谁错要人来判。曾用 token 重叠自动匹配，中文无空格导致约 50% 误报，已改为只认作废记录里**明写的 § 号**（192 → 119 条）。
3. **`superseded_by` 只在同一分块内可靠。** 台账由 7 个 agent 按 § 区间并行抽取，每个只看自己那块，所以"后面的 § 作废了前面的"这类关系**只有跨块 flag 那一条通道**，且是线索级。
4. **19 条 artifact 确认已丢**（`--absent` 可列）。那些 MEASURED 目前只有笔记文本一个来源，不可复核。
5. **21 条 artifact 是"路径搬了家"不是"文件没了"**（如 `docs/analysis/phantom_paper/*` → `results/phantom_paper/*`）。合并时按 basename 全库回查修正了。若不修，台账会说"这个测量的 artifact 不存在"，下游读成"不可复现"→ **正好去重做**。
6. **同一个量在不同 § 有多套不可比数字，一律并列保留、不合并。** 抽取时明确禁止取平均或折算。见 §397.10 的"不许做加减法"清单。

## 重建 / 追加

```bash
.venv/bin/python3 docs/reference/known/rebuild_ledger.py
```

按 § 区间分块并行抽取 → 合并 → artifact basename 回查修复 → 跨块作废链接。新增 § 时按同样 schema 追加到 `ledger.jsonl` 即可，无需重跑全量。

相关：`docs/checkpoints/REBUILD_PLAN.md`（这个台账为什么存在）、`docs/analysis/cross_sites/phase0b_noise_floor.md`（Phase 0b 与之并行的噪声地板测量）。
