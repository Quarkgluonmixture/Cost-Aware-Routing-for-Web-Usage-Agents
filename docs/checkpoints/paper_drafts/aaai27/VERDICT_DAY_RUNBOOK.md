# Verdict-day runbook (队列④, 2026-07-01)

> **触发时点**: 最后一个 pending cell bind 完成（Pass-1 全 36 → H1/H3 verdict day；Pass-2 router 6
> cond 完成 → H10 verdict day，可分两次走）。目标 = 从 "数据 land" 到 "draft slot 全填 + 自检过" 一条
> 命令链走完，**全程零手抄**（本 session 4 个 P0 全是转录/压缩漂移——工具链就是防它）。
> k<6 提前投稿场景（advisor 预案(a)）同样走本 runbook，但 `analysis_status=PARTIAL` 时只允许 verdict-中性披露；不得选择或 splice 分支。

## 0. 前置: 数据 sync（每个 cell land 后即做, 不等 verdict day）

NUMBERS_TODO §0 配方，逐条:
```bash
# ① 拉 fire manifest (A100 → DGX)
rsync -av condense-a100:/home/ubuntu/workspace/p79/docs/checkpoints/pre_run/fire_manifest.json \
  docs/checkpoints/pre_run/fire_manifest.json
# ② 手动 promote 新 bound 条目进 registry (grade: paper-grade, 照 05-28 先例格式)
$EDITOR results/phantom_paper/run_manifest.yaml
# ③ 聚合
make analysis FAST=1
```

## 1. Verdict day 主链（顺序执行）

```bash
# Step 1 — 确认聚合是最新的 (上面 §0 已做则跳过)
make analysis FAST=1

# Step 2 — 生成 canonical router covariate artifact（新 CLI 参数均显式）
.venv/bin/python3 scripts/analysis/router_covariate_baseline.py \
  --raw-features results/phantom_paper/l1_router/raw_features_phase1a.npz \
  --out-json results/phantom_paper/l1_router/covariate_baseline.json

# Step 2b — 生成 slot sheet (唯一允许的数字来源；各 artifact 路径均显式)
.venv/bin/python3 scripts/analysis/verdict_day_slotsheet.py \
  --decision results/phantom_paper/phase1_full_prereg_decision.json \
  --h10 results/phantom_paper/h10_pareto_verdict.json \
  --sr docs/analysis/cross_sites/sr_per_mode.json \
  --fig0c results/phantom_paper/fig0c_drop_one_bootstrap_ci.csv \
  --router results/phantom_paper/l1_router/covariate_baseline.json \
  --out /tmp/claude-1012/slotsheet_$(date +%Y%m%d).md
```

若 **Pass-1 已全 land**（decision 为 `analysis_status=COMPLETE` 且 `h1_verdict` 已是
`PASS`/`FAIL`），但 Pass-2/H10 尚未 land，Step 2 跳过并把 Step 2b 改用
`--h10-pending`：
```bash
.venv/bin/python3 scripts/analysis/verdict_day_slotsheet.py \
  --h10-pending \
  --decision results/phantom_paper/phase1_full_prereg_decision.json \
  --sr docs/analysis/cross_sites/sr_per_mode.json \
  --fig0c results/phantom_paper/fig0c_drop_one_bootstrap_ci.csv \
  --out /tmp/claude-1012/slotsheet_h1_h3_$(date +%Y%m%d).md
```
该形态只开放 H1/H3 槽与 Tables 2/3；H10/router、Table 4 和 abstract H10 槽均显式
fail-closed。它不能用于 `PARTIAL`，也不能与 `--rehearsal` 同用。

当前 `analysis_status=PARTIAL` 时只用下列 **rehearsal 安全形态**；全部输出进 scratch，禁止拿来
splice：
```bash
.venv/bin/python3 scripts/analysis/router_covariate_baseline.py \
  --raw-features results/phantom_paper/l1_router_rehearsal_20260702/raw_features_phase1a.npz \
  --out-json /tmp/p79_rehearsal_20260714/router_covariate_baseline.json \
  --allow-rehearsal
.venv/bin/python3 scripts/analysis/verdict_day_slotsheet.py \
  --rehearsal \
  --decision results/phantom_paper/phase1_full_prereg_decision.json \
  --h10 results/phantom_paper/h10_pareto_verdict.json \
  --sr docs/analysis/cross_sites/sr_per_mode.json \
  --fig0c results/phantom_paper/fig0c_drop_one_bootstrap_ci.csv \
  --router /tmp/p79_rehearsal_20260714/router_covariate_baseline.json \
  --out /tmp/p79_rehearsal_20260714/slotsheet_20260714.md
```

- **Step 3 — 读 sheet §A/§B**: `analysis_status` 必须 = `COMPLETE`，且 `h1_verdict ∈ {PASS,FAIL}`。
  确认 branch 建议后**人工对照** prereg §2.5 + Amendment 02 ladder（sheet 只建议不拍板；
  B2 claim-tier gate / I² cap 两个 modifier 在 sheet §A 里看）。

- **Step 4 — splice 分支**: 打开 `branch_prewrites_s1_abstract.md`，按选定分支替换
  `aaai27_main.md` 三处（abstract 内联句 / §1 ¶3 / §1 ¶4 末两句），«槽» 全部从 **sheet §C** 复制
  （禁止从笔记/next_steps/NUMBERS_TODO 抄——那些是 interim 冻数）。

- **Step 5 — Tables**: sheet §D/§E/§F 直接替换 Table 2/3/4 数据行；[P]→[A] 标签同步 lift；
  §E 带 ⚠️ 的行禁止进 draft。

- **Step 6 — §6 router 双分支**: aaai27_main §6 按 H10 verdict 删一支（sheet §A H10 行);
  `h10_status=entropy_unavailable` = fail-closed，§6 走 descriptive 分支 + Table 4 注保留。

- **Step 7 — (R-CONDITIONAL) 清理**: 全文搜 `(R-CONDITIONAL)` 标记，逐处按 realized R-tier 重写后删标记。

## 2. 自检链（sheet §G，全过才算完）

```bash
cd docs/checkpoints/paper_drafts/aaai27
# ① banned grep — 必须 0 hits
grep -nE "image-free|image-off|no image tokens|text-only cost|both Qwen cells|most of the.*mass" aaai27_main.md | grep -v 'grep -nE'
# ② 残留槽位/标记 — 必须 0 hits
grep -nE "<(H1|H3|H10)-VERDICT>|R-CONDITIONAL|«|⟨TBD⟩|<TBD" aaai27_main.md
# ③ 词数 (strip HTML comment 后 wc -w; strip 命令别贴进 MD comment — item 7 教训)
```

- ④ 词数超 → 按 checklist item 7 候选顺序砍（§2 para4 anchors → §5.5 → §8 statistics para）。
- ⑤ **/stress + codex + gemini chain**（CLAUDE.md auto-trigger #2, 不可跳）→ 修完**重跑①②**。
- ⑥ commit（规范 message）；push 问 user。

## 3. Abstract registration 特别通道（OpenReview, 非 CMT — 2026-07-14 实测修正）

**系统 = OpenReview** (AAAI-27 Main Technical Track 表单)。deadline 双标注:
**Jul 22 2026 11:59 AM UTC-0 = Jul 21 23:59 AoE** (伦敦 Jul 22 12:59 BST)。
只需 Title + TL;DR + Abstract (+Topics/COI 等表单字段): verdict-中性版 abstract + TL;DR
已备好 → **`deliverables/openreview_abstract_tldr_2026-07-14.md` 逐字复制**
(248 词, 槽位 0)。abstract 在 full deadline 前可改。**不要**在 abstract 里写 interim 数字。
⚠️ **Reciprocal reviewer nomination 同在 Jul 21 AoE 冻结** — 有资格作者
(≥2 一作 or ≥5 合著 archival) 未提名 = desk-reject 风险; 作者名单 + 提名在
07-16 学长周会拍板, 冻结前必须落到表单。

## 4. H10 单独 verdict (若 Pass-2 晚于 Pass-1)

Pass-1 全 land、Pass-2 未 land 的第一次 verdict 先用 §1 的 `--h10-pending`，只 splice
H1/H3 与 Tables 2/3，并采用 sheet 给出的固定 abstract H10 pending 短语；§6 与 Table 4 保持
pending，禁止填任何 router 数值。

Pass-2 land 后: `python scripts/analysis/aggregate_h10_pareto.py` → 重跑 slotsheet → 只动 §6 + Table 4
+ abstract 的 `<H10-VERDICT>` 短语。前置: `h10_entropy_gate.json` 必须存在（队列⑤预演产物;
缺失 = fail-closed 不可 claim deployability）。

## 5. Figures 彩排（scratch only）+ 失败模式速查

以下只验证 partial-data 水印与命令可执行性，不产出 draft 可用 figure：
```bash
.venv/bin/python3 scripts/analysis/figures/fig_f2_h1_forest.py \
  --decision results/phantom_paper/phase1_full_prereg_decision.json \
  --out /tmp/p79_rehearsal_20260714/fig_f2_h1_forest \
  --interim
.venv/bin/python3 scripts/analysis/figures/fig0c_drop_one_oracle.py \
  --out /tmp/p79_rehearsal_20260714/fig0c_drop_one_oracle.png \
  --csv-out /tmp/p79_rehearsal_20260714/fig0c_drop_one_bootstrap_ci.csv \
  --allow-partial
.venv/bin/python3 scripts/analysis/figures/fig_f1_diamond_schematic.py \
  --out /tmp/p79_rehearsal_20260714/fig_f1_diamond_schematic
```

| 症状 | 处置 |
|---|---|
| sheet §A analysis_status=PARTIAL 但以为全 land | 查 skipped_cells 行 + fire_manifest bind 数; 回 §0 sync |
| H3 某轴意外 FAIL | branch_prewrites 两支都不适用 → Amendment 02 ladder (C'-R/F) 现写, 先停 splice |
| sheet 数字 vs 旧 interim 引用打架 | 以 sheet 为准; interim 引用全文搜出来清掉 (笔记 §360.3 数字禁入 draft) |
| I² cap_at_R3=True | framing 最高 R3, §1 hook 降级 (prereg I²-cap 条款) |
| figures 崩 (fig0c max-empty / axis1 KeyError) | 已于 2026-07-14 修复 (e35d0b4/559cb47)；若再现 = regression，停止并审查 |

## 6. k<6 提前投稿特例（advisor 预案(a) 获批时）

- `analysis_status=PARTIAL` 是预期态；rehearsal sheet 顶部标 `INVALID_FOR_DRAFT`，§B 只输出 `NO_BRANCH`，并抑制可复制槽值/表格。
- draft 只允许 verdict-中性措辞，并同时落 §4 k<6 透明披露句 + §8 statistics para 对应修改（fixed-cells 设计, k 不齐明写）。
- pooled k<6 数值仅作 interim 诊断；**不得按 legacy `gate_status` 或 bootstrap boolean 选择分支**。最终分支唯一依据是 `analysis_status=COMPLETE` 后的 `h1_verdict`。
- 例外出口（不焊死预案(a)）：若 advisor 确认要在 k<6 提前选支投稿，这构成对 prereg "over the 6 planned cells" estimand 的临时偏离 —— 须先落新的 PROTOCOL_NOTE（明确 k<6 pooled gate 的临时定义 + 披露义务）并打 witness tag；在此之前 slotsheet 刻意不提供任何 k<6 选支机制。runbook 层不得自行拍板。
