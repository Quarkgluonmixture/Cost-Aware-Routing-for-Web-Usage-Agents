---
type: plan
status: active
created: 2026-07-28
purpose: zero-preset rebuild — Phase 0/0b/1 DONE. 状态与接手点见本文顶部「当前状态」
audience: the next session (start here, not at next_steps §0)
---

# Zero-preset rebuild — start here

> **User decision 2026-07-28**: do ALL phases, no scaling down. Previous session's
> handoff (`next_steps.md §0`) is still valid for *state*, but this doc supersedes it for
> *what to do next*.

## 🔴 2026-07-29 — 当前卡点：等 user 标完 508 条结论账本

**账本** `docs/checkpoints/deliverables/advisor_ledger.html`（VSCode → Live Server）。
user 逐条裁定 ✓认可 / ✗有问题 / ?不确定 → 导出 → Claude 与 user 过「有问题/不确定」
→ **基于这些**生成对账议程 → 和学长定 → **才写骨架**。顺序不要跳。

**待办 2 / 4 / 6 已结清**（§399–§404，8 commit 未 push）。剩 **1**（骨架，卡在账本）·
**3**（HKSJ appendix 行）· **5**（逐条核对 conclusion —— 就是 user 正在做的这件事）。

---

## ⏱ 当前状态（2026-07-28 更新，新 session 从这里读）

| Phase | 状态 | 产出 |
|---|---|---|
| −1 清污 | ✅ | limitations 假陈述已改 |
| **0 台账** | ✅ | **2082 条**（2026-08-01 补 chunk 9：§398 整节此前 **0 条**，而 ③ 的出处正是 §398.2）`known/ledger.jsonl`；入口 `KNOWN.md`；查询 `known.py`；核验 99.5% |
| **0b 噪声地板** | ✅ | `phase0b_noise_floor.md` + **`noise_floor_inventory.md`（2026-08-01）**。两轴 1.35/2.09pp 低于 B0 地板 4.9–7.6pp，**也低于新测的 B1 地板 2.0–4.0pp** ⇒ 地板不是 B0/MoE 特有。§7.1 头号待办（B1 replicate）**已关闭**：它一直免费存在于 WA pilot × full 重叠里 |
| **1 结论层** | ✅ | **253 节 / 9377 行** `known/conclusions/` —— **入口 `INDEX.md`**。2026-08-01 补标 9 处（3 处内部不一致 + 6 处 stale） |
| 2 创新与反论 | 🔓 **门已开** | blocker「需先过台账过滤」已解除 —— user 标完 508 条 + 2026-08-01 三桶处理完毕。**不依赖 08-03 对账，可随时发** |
| **3 论文结构** | 🔶 **已拍板：两篇合并成一篇** | 见下 |
| 4 重写 | ⬜ | deadline **08-05**（REALM @ EMNLP，archival，双盲 ACL，**8 页**） |
| 5 memory/CLAUDE.md | ⬜ | |

### 新 session 第一步

> 读 `docs/reference/known/conclusions/INDEX.md`

它含：五批产出导航 · 按问题找 · **M1–M11 错误模式** · 5 条已知限制 ·
已裁悬案 · 演化链 · **「结论层之外还有什么」七节**（canvas / diag / 在跑的实验 /
未来实验 / router 真状态 / 未落地裁定 / per-mode 缺口）。

### 工具

```bash
known.py <kw>              # 查台账单条（强制打印 caveats）
trace_claim.py "6.7pp"     # 四层追溯：结论层→台账→笔记→artifact（带单位查，裸数字会误命中）
verify_ledger.py           # 台账 vs 笔记的数字可追溯性
find_unlanded.py           # 找"定了但没落地"的裁定
```

### 论文结构决策（user 拍板 2026-07-28）

**两篇合并成一篇**。不是因为两篇都弱 —— **Paper B 强、Paper A 弱**。焊接点是 Phase 0b
新增的第 ③ 步：

> ① ceiling 高（+3.4~16.1pp，省 13.7-35.3%）→ ② 有结构基础（H3 双轴独立）→
> ③ **但结构小于同模式重跑地板** → ④ 且学不到（0/6 Pareto）
> ⇒ **表征路由的上限真实存在，但既不稳定也不可达**

第 ③ 步两篇现在都没有，它同时关掉 A 的正面结果、补上 B「为什么不是估计器问题」的另一半。

### 待办（user 2026-07-28 列，按依赖排序）

1. **合并论文骨架 + 取舍清单** ← 先做，它决定下面各项补到什么精度
2. ~~**补 §108 四维 × per-mode 画像**~~ ← ✅ **已执行 2026-07-28，见 §400.2**
   - 产物 `docs/analysis/cross_sites/per_mode_four_dimension_profile.{md,json}`
   - 四维 × 6 mode × 6 cell 全跑。18 指标里 **7 个 6/6 一致极值全是 Vision**，
     经 cross-AI 复审后三分类（§401.2 修正了初版的二分）：**经验发现 0 个** ·
     ◆ **架构下游 3 个**（`scroll_frac` 1.25–6.77× / `action-fail` 1.06–1.60× /
     `no-op` 1.07–1.58× —— 同一条机械链：坐标寻址→点不准→页面不变→被迫滚动）·
     ⚙️ **构造必然 4 个**（`locator fallback` / `tokens` / `cost` / `cost_rel_dom`）。
     **◆ 与 ⚙️ 都不得当行为发现引用。**
   - ⚠️ canvas 数字 stale（仍写 `drop-one 1.7-3.8pp`，k=6 后 H1 已 FAIL）—— 产物零引用 canvas
3. **补 §135.2 HKSJ appendix 行** —— §215 明确承诺过，两稿 0 次，数据现成（§172.8）
4. ~~**router：同族 pooled × cost-tier 可学性**~~ ← ✅ **已执行 2026-07-28，见 §399**
   - 产物 `docs/analysis/cross_sites/router_pooled_tier_learnability.{md,json}`；
     producer `scripts/analysis/router_pooled_tier_learnability.py`
   - **结论：H-pool 不支持 —— 走「不支配 ⇒ 结论加限定后更强」这一支。**
     26 个 arm×cell：严格支配 always-cheapest **0/26**（最有利角落 0/4）；
     相对六固定 mode 菜单非支配 **0/26**（router 从未落在经验 Pareto 前沿上）。
   - 锁定判据（95% 非支配 vs always-cheapest）过了 5/26，**但全在 reddit·B0，且
     同族/跨族、tier/which-mode、pooled/per-cell 三组对照全都过** ⇒ 与本实验要测的
     三个因素无关，真正原因是该格 always-cheapest（Vision 7.39% SR）特别弱。
   - ⚠️ **规格 §2 与 §3 口径分叉**：§2 写「支配」，§3 锁「非支配」。只跑锁定判据会
     报出与假设相反方向的 headline。产物现三档并列报，引用 pass 率必须说明是哪一档。
   - Paper B 可写的加强表述：路由打不过固定廉价策略，**即使同族、标签粗到免疫
     tie-break 缺陷、天花板 97.5%**。→ Phase 3 的合并骨架**不需要因此改动**。
5. **逐条核对 conclusion** —— 待骨架定后，按「承载论文推理的前提」筛（不是按「进不进论文」）
6. ~~**查 `B0·red·P-SoM` 的 `read_jsonl_dedup: summary identity mismatch`**~~
   ← ✅ **已定根因 2026-07-28，见 §400.1**
   - quarantine→resume rerun 写了新 summary 但**没换 steps JSONL**；只影响 task **87 / 149**
   - 全库审计（新 `scripts/analysis/audit_steps_summary_identity.py`）：
     36 组合 / 7686 scored episode 仅此 2 个 = **0.03%**；其余 4 个带 stale/quarantine 的
     condition 全部通过 ⇒ **个案非流程缺陷**
   - 处理：Outcome/Efficiency 读 summary 不受影响（是 clean rerun 且 condition 已 bound）；
     Macro/Micro 排除这 2 个并披露

## Why this exists

The 2026-07-27/28 session redid work that had already been done, and got it wrong, **five
times in one sitting**. Each time the user had to catch it. Four of the five share one shape:
**reasoning forward from just-read code without first asking whether the thing had already
been measured, already been adjudicated, or already exists**.

| I claimed | Reality | Already existed |
|---|---|---|
| "no same-mode replicate exists" | 15 manifest pairs + a dedicated clean replicate dir | `compare_cross_run_same_condition.py` prints the exact statistic; §302.1 has the number (6.7/7.6pp) |
| derived a "detector sensitivity" story for the id channel | the id channel was measured properly, with an id-agnostic metric | `b0_paired_idperturb_replay.py` + `docs/checkpoints/probes/*idperturb*.json` (B1 20%, B0 12.5%) |
| labelled Vision as a native-id arm | Vision is coordinate-based, zero element ids | any step record: `coordinate_type: qwen_0_1000` |
| "the DGX mechanistic job finished" | one *worker* finished; the **24-cell sweep driver** is still running | `logs/mechanistic_canonical/.sweep.pid` + `sweep_supervised_*.log` |

The fifth has a **different shape and therefore a different fix**: I claimed the mechanistic
sweep was "blocking B3's DGX window". It blocks nothing — B3 *fires on the A100* (its own
frontmatter says so, and I had quoted that line), and the DGX is the shared-contention host
*reserved* for mechanistic work precisely because contention does not matter there. That was
not a missing lookup; it was **failing to apply a rule already loaded in context** (CLAUDE.md's
three-tier compute table and host-role split).

So there are two fixes, not one:
- for the first four — **an index from question → prior finding** (Phase 0). The chronicle is
  ~20k lines / ~400 sections, append-only; grep only works when you already guess the keyword.
- for the fifth — **re-read the host-role table before reasoning about scheduling at all.**
  `§396.7` had already recorded the general lesson one round earlier and it still recurred, so
  a resolution is not enough; it has to be a step in a checklist (Phase 5).

## 🚨 Read this before trusting anything

1. **Treat every conclusion the previous session wrote as UNVERIFIED.** Especially
   §397.4–§397.9. `§397.10` is the correction entry — **read it with §397.4 and §397.9 or
   you will inherit retracted framings as fact.**
2. **Do not do arithmetic across the noise numbers.** §302 retracted the linear
   decomposition (`12.1% ≈ 10.5% + 1-2pp`) as a category error across 4 incomparable
   dimensions. §300.2 records ±3-5pp cross-GPU drift on the flip rate. Quote each with its
   scope; never subtract.
3. **The papers are frozen for prose edits.** Do not keep patching sentences — Phase 4
   rewrites them. The only edits allowed before Phase 3 are removals of statements that are
   factually false.

## Phases

### Phase −1 — correct the polluted record ✅ DONE 2026-07-28
- `paperA/limitations.md`: the sentence "No same-mode replicate exists in our data" was
  **false** and had been committed and synced to Overleaf. Replaced with the honest version
  (one clean pair exists; 6.7/7.6pp; upper bound; B0-MoE-specific; local cells unmeasured).
- 笔记 `§397.10` appended as the retraction entry for §397.4 + §397.9.
- `next_steps.md §0` forbidden-numbers table corrected.

### Phase 0 — the anti-redo ledger ⬅️ START HERE
Deliverable is a **queryable** artifact, not a summary. Suggested location
`docs/reference/KNOWN.md` (+ a machine-readable sibling), with MEMORY.md holding only a
pointer — it is large and will grow, so it does not belong in memory files.

Five record types:

```
MEASURED    quantity | value | scope (model/site/mode/n) | caveats | source (§ + artifact path) | superseded_by
ADJUDICATED decision | reasoning | date | where recorded (code comment / prereg / §) | do-not-relitigate
RETRACTED   former claim | why dead | § | what replaced it
DATA        what exists | path | grade (clean / pre-fix / archived) | what it can support
COMPUTE     what is running or planned | host | status | ETA | what it unblocks | what it BLOCKS
```

`COMPUTE` exists because the previous session mis-stated run status twice in opposite
directions: it reported a **running** 24-cell sweep as finished (it had checked a worker pid
instead of the sweep driver), and it did not know a wanted experiment was queued behind an
unrelated one. Same defect as the `DATA` misses: **state that lives only in a process table.**

Two field rules follow from those two errors:
- record the **driver pid or pidfile**, never a worker pid — a child exiting says nothing about
  the job;
- record **what each entry blocks**, not just its ETA — the binding constraint on this project
  is mutual exclusion (shared containers, shared auth state), not duration. And check the
  host-role table before asserting a block: DGX contention is tolerable by design for
  non-paper-grade work, so "same host" does not imply "blocks".

Seed entries already known (verify each, do not trust this table):
- MEASURED: id-perturbation flip B1 **20.0%** / B0 **12.5%**, within-group consistency B1
  **1.000** / B0 0.867 — `docs/checkpoints/probes/b0_paired_idperturb_20260529_*.json`
- MEASURED: same-condition self_drop **6.7pp / 7.6pp**, discordance 14.3pp, κ 0.614 —
  §302.1, B0·vision·classifieds, n=224
- MEASURED: AMENDMENT_07 sequential-id switch SR 30.4% → 27.2%, Δ **−3.2pp** — §299.4
- ADJUDICATED: measured-cost tie-break for oracle labels **rejected** —
  `router_features.py:78-101`, B-1806, 2026-06-09
- ADJUDICATED: K-of-N is transparency-only, not a gate — preregistration §2.5
- RETRACTED: "MoE residual 1-2pp" linear decomposition — §302
- RETRACTED: drop-one hero 1.7-3.3pp — superseded by 6-mode k=6 (0.0-1.3pp, H1 FAIL)
- DATA: `results/repro_replicates/B0_vision_classifieds_R24792_clean_replicate/` — clean
- DATA: 15 (model,mode,site) pairs in `run_manifest.yaml` whose second run is
  `grade=archived` = pre-fix ⇒ confounded. Archived runs use **merged naming**
  (`B0_3mode_reddit_20260422` holds DOM+SoM+Vision) — a directory-prefix search misses them.

COMPUTE — verified 2026-07-28 against `ps` / `_status/tasks/*.md` frontmatter, not from memory:

| what | host | status | ETA | unblocks / BLOCKS |
|---|---|---|---|---|
| WA reddit full, 6 modes × 104 | A100, chain pid 2658570 | **running**, step 1/6 (dom, `B1_dom_wa_reddit_..._R13217`) | ~3 days from 07-27 18:00Z | benchmark-generalisation annex. **BLOCKS every VWA reddit run** — shared postmill container + shared `.auth/reddit_state.json` (B-647). This is what a fresh reddit replicate queues behind |
| **mechanistic canonical sweep, 24 cells** | DGX **sweep driver pid 38603** (`.sweep.pid`) + supervisor (poll 300s, ≤40 restarts) | ⚠️ **RUNNING**, 2/24 done, cell 3/24 `p1_rev_reverse_cls` in flight as worker pid 1638252, **21.7 GB VRAM** | cells take ~800-845 min each; 2 cells burned ~27 h ⇒ the **08-01 sweep deadline truncates it around cell 7-8**, it will never reach 24 | §5 mechanism, **shelved** by advisor 2026-05-14 ⇒ archive only, feeds no current paper. **Blocks nothing.** DGX is the shared-contention host reserved for dev / curation / mechanistic precisely because contention does not matter there; paper-grade fire moved to the A100 on 2026-05-14. It uses 21.7 of ~128 GB and one process, inside CLAUDE.md's "1-2 processes" envelope. `task_b3_mimo` **fires on the A100** (its own frontmatter says so) and only its adaptation touches the DGX, which coexists fine |
| ~~cell 2 `p1_fwd_strong_red`, worker pid 38617~~ | DGX | that ONE CELL finished cleanly 07-28 03:34 (24 tasks, `run_manifest.json` emitted, log ends `pilot DONE`) | — | ⚠️ **2026-07-28: the previous session read this single worker's exit as the whole sweep completing.** Checking a worker pid instead of `.sweep.pid` is the same defect as the DATA misses above — verify the driver, not a child |
| `task_pass1_baseline` | A100 | active, 36/36 landed | — | the k=6 substrate both papers use |
| **`task_pass2_router` — live router pass** | — | **SUPERSEDED, NEVER FIRED** | — | ⚠️ paperA §4.4 discloses this. `k_of_n = 0/0`. Replaced by the offline suite; live router is paper-2 |
| `task_wa_pilot` | A100, B1 local (proxy-immune) | pending | 2026-08, slots into the B3 adaptation gap, ~days | WA generalisation |
| `task_b3_mimo` (B3 = MiMo-VL) | DGX adapt → A100 fire | pending | adapt late 07; fire early 08, **12 conditions ≈ 2-2.5 weeks** | cross-family third model |
| `task_b0_replicate_annex` | A100 + proxy quota | pending | 2026-08 **opportunistic** (grab a proxy window, precedent §369) | ⚠️ a pre-submission B0 rerun was **rejected as outcome-dependent sampling** — do not re-propose it |
| `task_shop_expansion` (Phase 1b: shop × 3 models × 6 modes) | A100 | pending | **2026-09+**, journal/major-revision horizon | R3 → R1 framing decision |
| WA shopping / shopping_admin | — | **deliberately not opened** | — | no reset implementation ⇒ cannot be paper-grade clean |
| `task_grl_subpaper` | — | queued, NOT critical path | 2026-06 slot missed | — |
| `task_doi2_bundle` | — | **blocked** | — | — |
| possible fresh **locally-served** same-mode replicate | A100 | **not scheduled** — Phase 0b decides whether it is needed | queues behind WA reddit (~3 days) | the H3 noise floor for B1/B2 cells |

**Execution note**: chunk the chronicle by § ranges and fan out subagents writing into one
schema. Do not read it linearly — that spends most of a context window and yields a summary
rather than a lookup.

### Phase 0b — self-oracle floor sweep (run in PARALLEL with Phase 0)
Offline, reads cached artifacts, does not touch the sites, does not queue on the A100.
`compare_cross_run_same_condition.py` over every available pair, reporting **separately**:
- clean vs pre-fix pairs (never pool them)
- API-served B0 vs locally-served B1/B2 (B0 carries a ~13% serving-inconsistency floor;
  B1 is deterministic on repeated identical input, so its model-side floor may be ~0)

**This gates whether H3 is a positive result at all**, so it must land before Phase 3.
Open question it answers: is the 6.7/7.6pp floor a B0-vision artifact, or general? If
general, H3's 1.35/2.09pp axes sit inside the noise.

Unverified hypothesis worth testing here: axis-1 (`|P-text \ P-SoM|`) is **within** one
id-regime (both 1..K) while axis-2 (`|P-prompt \ P-SoM|`) **crosses** the AMENDMENT_07
boundary. If the crossing is what makes axis-2 the larger axis, the asymmetry §3.2 calls
"informative" has a mundane cause.

### Phase 1 — claim inventory of both papers, zero preset
Every claim currently in `paperA/` + `paperB/`: claim | number | source artifact | status
(verified / unverified / **contradicted by ledger**). Cross-join against Phase 0. This is
where the previous session's errors surface mechanically instead of by user catch.

### Phase 2 — innovations and counter-arguments
Only after the ledger exists, otherwise both are speculation. **Filter candidate
counter-arguments through the ledger before dispatching any cross-AI round** — in the
2026-07-28 round, 2 of codex's and 2 of Gemini's findings were either re-litigating settled
decisions or misreadings, and the ledger would have caught them for free.

### Phase 3 — decide paper structure
Given what survived. The live fork: Paper B's claim ("the ceiling is real but unlearnable")
depends on neither H3 nor the hallucination metric, so it is the more robust of the two;
Paper A's positive result is exactly the most uncertain thing.

### Phase 4 — rewrite

### Phase 5 — memory + CLAUDE.md
Add a **before-you-claim checklist** derived from the actual failure modes, not generic advice:
1. Has this quantity already been measured? (search the ledger, not the chronicle)
2. Has this decision already been adjudicated? (code comments carry adjudications)
3. Is the metric's basis constant across the arms being compared?
4. Is this in-sample or out-of-sample?

## Scheduling reality
Everything on the critical path is **effort-bound and offline**, including the floor sweep.
Only one thing is wall-clock-bound: *if* the sweep shows a fresh locally-served replicate is
needed, that needs the A100, which is occupied ~3 days by the WA full run — and WA reddit
cannot run beside VWA reddit (shared postmill container, `.auth/reddit_state.json`).

## State at handoff (verify, do not trust)
- git `origin/fix/b1878-reddit-reference-image`, working tree clean, 0 unpushed
- Overleaf synced at repo `9cf49b8`; ⚠️ **the Limitations fix from Phase −1 is NOT yet
  synced** — re-run `bash scripts/maintenance/overleaf_sync.sh` when convenient
- Gates as of Phase −1: `make deslop-ratchet` PASS · both papers content ≤ p8 ·
  `pytest -q` 1626 passed
- Background: A100 chain pid 2658570 alive, WA full, step 1/6 (dom), ETA ~3 days.
  DGX mechanistic 38617 **finished** 07-28 03:34 (§5 mechanism, shelved, archive only)
