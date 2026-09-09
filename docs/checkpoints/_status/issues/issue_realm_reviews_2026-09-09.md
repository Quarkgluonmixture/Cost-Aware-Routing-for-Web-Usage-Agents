---
type: issue
category: venue
status: open
priority: P0
action: camera-ready 09-14 AoE — 三处事实错误 + abstract 人话重写；结构性意见转 ARR 10-12
---

# REALM #192 三份 official review — 逐条对位与处置

**来源**: OpenReview forum `EAplLx6gCD`，意见 2026-08-22 提交、**2026-09-08 15:05 公开**。
**结果**: Accept。3.5 / 2.5 / 3.5；soundness **4 / 2.5 / 3.5**；excitement 3.5 / 2 / 4。
**没有任何一位质疑数字本身**，攻击集中在框架、覆盖度与可读性。

> ⚠️ 提交稿源文件 = `~/overleaf-aaai27/main_restructured.tex` + `sections/`。
> **不是** `main_realm.tex` + `realm_*.tex`（8-05 重构后的废稿，见笔记 §504.1 / 台账 RETRACTED）。

## 1. 三条交集意见（三人全提）

| # | 意见 | 现状 | 落点 |
|---|---|---|---|
| ① | abstract/intro 难读，主张要重建 | **成立** | camera-ready |
| ② | rerun band 覆盖不足 | **一半已被数据推翻** | camera-ready（改述） |
| ③ | 泛化性（cell 少 / 两 reddit 同底层应用 / 「第一性原理可预料」） | **成立** | ARR |

### ① 可读性
- 4s7L：「The abstract and introduction are **very hard to follow**」。全文最清楚的一句在
  **page 7** = `sections/5_lowerbound.tex:17`「the arm the router would route to is
  already the right arm to route everything to」——**要提到顶部**。
- sVJH：「The title and abstract could also **narrow** the learnability claim」。
- 4s7L best-paper 栏：「would become a stronger candidate **if the abstract were
  rewritten**, since the underlying measurement contribution is award relevant even
  though the current framing obscures it」。
- 独立同源：supervisor Maria 2026-09-09 邮件提同一条（「imagine writing so a
  non-native undergrad could follow」）。

### ② rerun 覆盖 —— 稿子落后于数据
`3_noise.tex:5` 现写「Three arms of `cls·B0` … and five modes of `wa_red·B1` on a
ten-task draw. **Nothing else: no VWA-reddit cell and no B2 cell carries a replicate**」。

实测（`docs/analysis/cross_sites/noise_floor_inventory.md`，2026-09-09 现拉）：

| cell | 已注册 replicate 臂 |
|---|---|
| `B0·cls` | **6 / 6** |
| `B0·red` | **6 / 6** |
| `B1·cls` | 3 |
| `B1·red` | 2 |
| `B5·cls` | 1 |
| `B1·wa-red` | 10-task pilot × 5 mode |

**18 对 / 5 格。**「no VWA-reddit cell」失效；仍成立的只剩「no B2 cell」。

### ③ 泛化性 → ARR
反驳弹药已在 §503.2（SoM ⊉ DOM∪Vision，6 cell 合计 81 漏解，2³ 包络下界
cls_B0 **18** / red_B0 **10**，`cls_B2` 交集 **0**）。6vKx 另问「shopping 为什么
没跑」——`B1_*_shopping_20260906` 正在跑，~09-13 落，camera-ready 可一句话答。

## 2. camera-ready 必修（事实错误，非润色）

1. **`3_noise.tex:5`** rerun 覆盖改述（见上）。
2. **`3_noise.tex:3`** flip 数字：现写「49 of 224 … in at least one of the **three**
   replicated arms … 48–52% vs 2.9%」——句子自称三臂、引的是 dom+vision **两臂**产物。
   三臂重算 = **67/224 (29.9%)** · contested **67.0%** vs **5.9%** · enrichment **11.4×**
   （§464.2 已 RETRACT，当时即标注「camera-ready 是窗口」）。
3. **36 vs 48 声明 scope**（4s7L 点名 reconcile）：36 = VWA 六格×6 mode；
   48 = 含 WA 的 6×8。`2_setup.tex:19` 的 **7,686 = 224×18 + 203×18** 是谜底。
   **修法是标 scope，不是改数字。**

## 3. camera-ready 低成本、审稿人明确要求

4. abstract 人话重写（① + Maria），把 `5_lowerbound.tex:17` 那句提到前三句。
5. **加 Related Work 章节**（4s7L：「The paper can add a section on related works」）。
   现状：文献散在正文，`3_complementarity.tex:3` 有 osworld / stwebagentbench 对位。
6. shopping 未跑的原因一句话（6vKx 明确发问）。

## 4. 结构性 → ARR（不进 camera-ready）

7. **删掉最稀疏 cell 的那个 positive routing result**（4s7L：该 cell 被论文自己
   三重否定——太稀疏 / triage 信号低于随机 / leakage check 翻转其判词；
   「Dropping it entirely would make the paper's claim **cleaner, not weaker**」）。
8. headline claims 限定到能支撑的 cell（4s7L：limitations 自己说两个最弱 cell
   「should not carry a comparison」，正文却在 8 格上陈述）。
9. **§3 互补 与 §5 rerun-noise 应合并**（4s7L：「a finding and then its partial
   retraction two sections later」）。注：真稿已是 `3_complementarity` + `3_noise`
   相邻，但仍被读成分离 ⇒ 相邻不够。
10. **noise floor 是否 mode-dependent** —— 4s7L 提，写稿时答不了；**现在可测**
    （`B0·cls` 六臂 10.27–14.29%，`B0·red` 六臂 4.93–11.33%）。见 §504.4。
11. per-mode SR 报点估计而非 mean±sd over reruns（6vKx）。
12. sVJH 要求分离 perception 与 action grounding（§503.1 的 identifier-contract
    分层是部分反驳：文本臂内部差 17.3pp ≈ Vision↔DOM 的 19.1pp ⇒
    **grounding 是混淆项但不沿表征轴分布**）。
13. sVJH 要求测更丰富的 router（LLM-based / contextual bandit / RL / online /
    graded trajectory supervision）。
14. 4s7L 建议：router 性能 vs 可用 label 数的曲线 ⇒ **把负面结果变成预测**
    （直接支持「更强的 agent 会推翻它」这条）。
15. 4s7L 建议：检验 cost saving 是否随「无解任务占比」增长。

## 5. 表单现状（09-09 Edit 入口已开）

- Title：`Routing Is Least Learnable Where It Is Most Valuable: Bounds on
  Representation Routing for Web Agents`
- Authors：Jiaming Wei / Zekun Wu / Adriano Koshiyama / Maria Perez-Ortiz
- **Archival：`Non-archival`** ⚠️ 不可改（改则不可逆地占掉 NAACL ARR 投稿权）
- Cross Submission To：`Plan to submit to ACL ARR 2026 August`
  ⚠️ 目标已改为 **October ARR**（Zekun 2026-09-09 确认），此字段措辞待定
- Serve As Reviewer：Jiaming Wei
- Abstract 余额：**3,395 字符**

## 6. ⛔ camera-ready 期间禁止重跑 `export_ablation_tables.py`（P0-2 未修）

`aggregate_noise_floor_inventory.py:895` 的 `replicated_side="text"` 是**六臂数据上的三臂
gate**（`replicated_arms` 现含 `som` 与 `vision` 两个 image-bearing 臂）。
`export_ablation_tables.py:435-450` 真实消费该字段：

```python
if key == "red_B0" and _covered and _arm_side and _arm_side != _covered:
    _side_ok = False        # verdict 变成 "floor is text-side only; added arm is som"
if key == "red_B0" and _side_ok:
    txt = f"{txt} ({_covered} side)"     # 表里附加 "(text side)"
```

⇒ **一旦为 camera-ready 重跑该脚本，appendix 的表会冒出 `(text side)` 标注，与正文
2026-09-09 新写的 "all six of red·B0" 直接矛盾。**

**处置（2026-09-09）**：不改代码。理由 = analysis 层改动按
`feedback_analysis_layer_fire_immutability_and_witness` 需要 witness tag，而 camera-ready
窗口只剩 5 天，且**不重跑就不受影响**（当前 PDF 里无 `text side` 字样，已 grep 确认）。
**推到 ARR 窗口修**，届时连同产物一起重跑。
