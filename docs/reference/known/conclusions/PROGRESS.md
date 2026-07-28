---
type: progress
status: active
created: 2026-07-28
purpose: Phase 1 结论提取的进度与接力点 — 新 session 从这里接手
---

# Phase 1 进度 / 接力

> **新 session 读这一个文件就够。** 不要重读 `实验笔记.md`，不要重建台账。
> 计划在 `docs/checkpoints/PHASE1_PLAN.md`，数据在 `docs/reference/known/ledger.jsonl`。

## 一句话现状

台账 2033 条已建成并核验（99.6% 可追溯），**五批结论层已全部落盘（2026-07-28）**。
剩下的只有**合并阶段**（见文末）—— 跨批作废回标、矛盾清单汇总、`INDEX.md`。

## 批次状态

| 批 | 内容 | 条数 | token | 谁做 | 状态 | 产出 |
|---|---|---|---|---|---|---|
| **B** | RETRACTED + CLAIM_UNVERIFIED | 248 | 59.6K | Claude 主 session | ✅ **完成** 07-28 | `retracted.md` |
| **C** | MEASURED 无数字 | 30 | 7.6K | Claude 主 session | ✅ **完成** 07-28 | `measured_qualitative.md` |
| **E** | DATA（49 条） | 49 | 14.4K | Claude 主 session | ✅ **完成** 07-28 | `data_inventory.md` |
| **D** | MEASURED 带数字 | 875 | ~260K | subagent ×4 | ✅ **D1/D2/D3/D4 全部完成**（D4: 58 主题 / 13 矛盾 / §397.10 五条修正全部落位） | `measured_D1..D4.md` |
| **A** | ADJUDICATED（裁定） | 831 | ~183K | Claude 主 session 分 4 轮 | ✅ **完成** 07-28（A1–A4 全部落盘，未外包） | `adjudicated_A1..A4.md` |

### A 批分片（按 § 升序保时序，字符量等分，不切开同一 §）

| 片 | § 范围 | 条数 | 时期 | 状态 |
|---|---|---|---|---|
| A1 | §5–§119 | 219 | 04-04 → 05-09 工程建设 + framing 成形 | ✅ 16 节 · 11 条待核 |
| A2 | §121–§164 | 177 | 05-09 → 05-16 pre-fire 审计密集 | ✅ 15 节 · 14 条待核 |
| A3 | §165–§240 | 229 | 05-16 → 05-20 fire 前冲刺 + Fire-1~6 | ✅ 15 节 · 13 条待核 |
| A4 | §241–§397 | 206 | 05-20 → 07-28 Protocol Reset + 治理 + 投稿 | ✅ 11 节 · 13 条待核 |

**覆盖性自检已过**（07-28，实测）：
- `A+B+C+D1–4+E = 2033` = ledger 全量，无遗漏无重复
- A 批 `219+177+229+206 = 831` **= ledger 中 ADJUDICATED 的全部**，无裁定漏到别批
- ⚠️ 文档漂移修正：`PHASE1_PLAN.md` 写 "A = ADJUDICATED 无数字 ~501 条" **是错的**，以 831 为准。
  另 §56 等少数条目含数字却落在 A 批 —— "无数字"切分有渗漏，**不影响产出（数字原样抄）**。

**分片重建命令**（scratchpad 失效时）：见本文件末尾附录。

## 分批数据在哪

```
/tmp/claude-1012/-home-jiaming-workspace-Cost-Aware-Routing-for-Web-Usage-Agents/
  ed15cb9e-3b51-4b2a-95da-c59606b0a51e/scratchpad/batches/
    A_adjudicated_nonum.jsonl   B_retracted_unverified.jsonl
    C_measured_nonum.jsonl      D1..D4.jsonl   E_rest.jsonl
```

⚠️ scratchpad 是 session 专属的。**新 session 若发现路径不在，用这条重新切分**：

```bash
# 切分逻辑见 PHASE1_PLAN.md；或直接按 type 从 ledger.jsonl 过滤
.venv/bin/python3 -c "
import json
rs=[json.loads(l) for l in open('docs/reference/known/ledger.jsonl') if l.strip()]
print({t: sum(1 for r in rs if r['type']==t) for t in
       ('MEASURED','ADJUDICATED','RETRACTED','CLAIM_UNVERIFIED','DATA')})"
```

## 为什么 A 必须由 Claude 亲读、D 可以交 subagent

**A（831 条裁定）** = 「这事定过了 + 为什么」。防重做的核心就是这个「为什么」——
一旦被概括就失去作用。实证：B-1806（measured-cost tie-break 被否）之所以会被重新提起，
正是因为**理由**没被记住，而不是结论没被记住。所以 A 不外包。

**D（875 条带数字测量）** = 已经过数字核验（99.6% 可追溯），是唯一被机制覆盖过的一批，
外包风险最低。且已强制要求每条结论附**原文片段**，主 session 可抽查。

## 每批的产出格式

见 `PHASE1_PLAN.md`。要点：**聚合不是转写** —— 把散落几十个 § 讲同一件事的记录
归成一个主题，给出「当前值 / 演变 / 已作废 / caveats / 证据 / 原文片段」。

## 三条不可违反

1. **数字原样抄，绝不做算术。** §302 已 RETRACT 一条线性分解为 category error
   （跨 model/modality/serving/perturbation 四个不可比维度）。noise 类数字
   （self_drop 6.7/7.6pp · discordance 14.3pp · κ 0.614 · H3 轴 1.35/2.09pp ·
   跨 GPU ±3-5pp · id-shuffle 20.0%/12.5% · AMENDMENT_07 Δ−3.2pp）一律各自带 scope 并列。
2. **caveats 一字不丢。** 尤其工具自带的
   `instability proxy, NOT H1 drop-one bias correction; 小样本/可能混代码版本 = upper-bound risk trigger`。
3. **矛盾不调和。** 两条打架就并列标 ⚠️，不选边 —— 在没有新证据时制造确定性，正是这次重建要修的病。

## 已知需在结论层标注的坑

- **§397.10 是 CORRECTION 节**，作废了 §397.4 与 §397.9 的部分结论，并追加 (4)(5)。
  读 §397.4 / §397.9 必须连它一起读。
- **§397.4「全 archive 只有一对同模式重跑」是假的** —— manifest 19 组 ≥2-run，
  `results/repro_replicates/` 有**两个** clean replicate。
- **§397.9 的 id-namespace 表不完整** —— compact 1..K 是**三个** mode
  （som / phantom_som / phantom_text）；**Vision 零 element_id**，其幻觉率 0.000 是
  结构性不适用而非「native」。主 session 已实证（模型输出 id：p-som 1/12/68 ·
  p-text 1/13/72 vs p-prompt 139/4074/26235 · dom 2/3606/61833）。
- ~~**`preregistration_decision_test.py` 注释 stale**：第 35 / 46-47 行仍写
  `PRIMARY GATE = DerSimonian-Laird`，而实现已是 FE-only。实现对、注释错。~~
  **⛔ 本条经实测撤下（2026-07-28，A 批 session）** —— 原描述会误导下一个 session 去"修"
  一个已退役文件里的**正确历史记录**。实测三条：
  (a) 第 **34 行明写 `⚠️ REWRITTEN 2026-05-13 (historical):`** —— 35 行属**显式标注为历史**的变更日志块，非当前 claim；
  (b) 46-47 行**成对**，47 行写 `(Decision 3A specifies FE; advisor lock pending — see banner above.)`，
  **忠实记录了 §143.6 那次"实现漂移、不单方面改估计量、等 advisor"的未决状态**；
  (c) **`Makefile:471` 明写 `Replaces retired-DL preregistration_decision_test`** ——
  主链只调 canonical `aggregate_phase1_full_prereg_decision.py`（mtime 07-27 活跃 vs 旧脚本 05-18 冻结）。
  ⇒ 结论：**整个文件已 retired，注释忠实记录了写下时的状态。无需修。** 详见 `adjudicated_A4.md` 矛盾表 #13。

## 接力时怎么开头

**五批已全部落盘，没有"未完成的批次"了。** 下一步是**合并阶段**（见文末），新 session 可以说：

> 读 `docs/reference/known/conclusions/PROGRESS.md`，做合并阶段。

若要读某个主题的结论：`adjudicated_A1..A4.md`（裁定 + 为什么）/ `measured_D1..D4.md`（带数字测量）/
`retracted.md`（作废 + 错误模式）/ `measured_qualitative.md`（实现层定性）/ `data_inventory.md`（数据资产）。

## 主 session 已裁的悬案（新 session 不必重裁）

- **§397.9「符号相反 = 真交互」仍然成立。** 台账给它挂了 `named by RETRACTED §397.10` 的 flag，
  D4 保守并列（其矛盾 #9）。裁定依据：§397.10(1) 修正的是"compact namespace 只有两个 mode"
  这个隐含说法（实为三个，SoM 也在内），而该论证用的两组比较（DOM/P-prompt 都 native、
  P-text/P-SoM 都 1..K）**恰恰被 §397.10 确认**；主 session 另有实测支持（模型输出的 element_id：
  p-som 1/12/68 · p-text 1/13/72 vs p-prompt 139/4074/26235 · dom 2/3606/61833）。

## 合并阶段待做（只有主 session 能做，subagent 结构上做不到）

1. **跨批作废交叉标注**：B 批持有全部 156 条 RETRACTED，D1–D4 各自看不见它们
   （按 type 切批导致同一主题的 MEASURED 与 RETRACTED 分家）。D1 已实证撞上这个：
   我给它的例子里 §103「少 ~50%」它找不到，因为那条是 RETRACTED 在 B 批。
   ⇒ 合并时须按 § 号把 B 批作废回标到 D1–D4 的主题上。
2. **D3 报「§299.4 的 Δ−3.2pp 不在其批次」** —— 须确认该记录归到了哪批，没有整体漏掉。
3. **矛盾清单汇总**：D1 13 + D2 10 + D3 6 + D4 13 + **A1 11 + A2 14 + A3 13 + A4 13** + B 批若干，
   去重后成一张表。其中"只有 user 能判"的那类（"user 拍板 X" / "advisor 收口 Y"）单独分节。
   **A 批四片的待核表都在各文件最后一节**（标题含"矛盾与待核清单"），已按同一格式写好可直接合并。
4. **A 批新增的两类合并任务**（07-28）：
   - **演化链跨片拼接** —— 同一主题被 A1–A4 切开的，须按 § 序拼成单条时间线。已识别的主要链：
     FP 体系（§78→§83→§88→§95→**§139.8 上游根因**→§158.6 hard-delete）·
     估计量（K-of-N→DL+TOST→单侧优效→**FE Decision 3A**→被 gemini 反攻→bootstrap percentile→DL/HKSJ 四层退役）·
     scope（16 cells→24/4→36/6→**42 = 36+6**）· venue（EMNLP→workshop→AAAI-27→**REALM 双投**）·
     router CV（archive-locked→Option C walk-back→LOCO→**(E'') task-held-out 5-fold**→E''' shared vectorizer）
   - **flag 噪声区分** —— 台账的 `named by RETRACTED §X` flag 表示"该 § 被某条 RETRACTED 记录点名"，
     **不等于该裁定被作废**。实例：A2 的 §137.2/.3/.5/.7 都挂 `named by RETRACTED §387.15.5`，
     而 §387.15.5 作废的是 `queue_chain.sh` 的一句注释，与 advisor venue 裁定无关。
     ⇒ 合并时须逐条判"点名"vs"作废"，不可一律当作废。
5. **一条"已知坑"经实测已撤下**（见上节 ⛔）—— 合并时不要把它再抄进 INDEX.md。

---

## 附录：A 批分片重建命令（scratchpad 失效时）

四片是从 `ledger.jsonl` 按 `type=ADJUDICATED` 过滤、按 § 升序、按渲染字符量等分切出来的
（不切开同一 §）。渲染成紧凑文本而非 JSONL，省掉约 60K token 的 key 结构噪声：

```bash
.venv/bin/python3 - <<'PY'
import json, re, os
SP = "/tmp/p79_A_slices"; os.makedirs(SP, exist_ok=True)
rs = [json.loads(l) for l in open("docs/reference/known/ledger.jsonl") if l.strip()]
rs = [r for r in rs if r["type"] == "ADJUDICATED"]          # 831 条

def key(r):
    m = re.search(r'(\d+)(?:\.(\d+))?', r.get('source_section', '') or '')
    return (int(m.group(1)) if m else 9999, int(m.group(2) or 0) if m else 0)
rs.sort(key=key)

def render(r):
    out = [f"§{(r.get('source_section') or '?').lstrip('§')} | {r.get('date','?')} | {r.get('decision','')}"]
    if r.get('reasoning'):      out.append(f"  ∵ {r['reasoning']}")
    if r.get('recorded_where'): out.append(f"  @ {r['recorded_where']}")
    for f in r.get('_cross_chunk_flags', []): out.append(f"  ⚑ {f}")
    return "\n".join(out)

tot = sum(len(render(r)) for r in rs); tgt = tot / 4
parts = [[]]; cum = 0
for i, r in enumerate(rs):
    if len(parts) < 4 and cum >= tgt and (i == 0 or key(r)[0] != key(rs[i-1])[0]):
        parts.append([]); cum = 0
    parts[-1].append(r); cum += len(render(r))
for i, p in enumerate(parts, 1):
    open(f"{SP}/A{i}.txt", "w").write("\n\n".join(render(r) for r in p))
    print(f"A{i}: {len(p)} 条 | §{key(p[0])[0]}–§{key(p[-1])[0]}")
PY
```

预期输出（与本次一致即说明切分可复现）：
`A1: 219 条 §5–§119` · `A2: 177 条 §121–§164` · `A3: 229 条 §165–§240` · `A4: 206 条 §241–§9999`

> 每片约 49K token，单次 `Read` 有 25K 上限，需分 2–3 页读。
