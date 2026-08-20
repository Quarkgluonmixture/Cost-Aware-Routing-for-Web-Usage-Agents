# Launch intent — 8-cell noise-floor replicate chain (declared 2026-08-17, BEFORE fire)

**Why this file exists.** codex Mode B Finding 2 (`/stress` 2026-08-17): every cell in this
chain is a same-condition replicate, and replicates are only recognised as such by being
listed in `CLEAN_PAIRS` (`scripts/analysis/aggregate_noise_floor_inventory.py`). Registering
them *after* the runs land makes registration a post-hoc choice — a cell whose floor came
out inconveniently could quietly stay unregistered and be re-labelled contamination. That is
a selective-reporting hazard, and it is invisible in the final artifact.

So the intent is declared here first, and committed before launch. When the runs land, the
registration must match this list exactly: **every cell below gets registered, whatever its
number says.** A cell that lands and is NOT registered requires a written reason in this
file, in the same commit that omits it.

This is also the fix for the concrete failure it describes: `R28065` ran a full 224 episodes
on 2026-08-16 with no such declaration, and `validate_fire_manifest.py` correctly called it a
COMPLETE ghost, raising the fail-closed halt marker at 08:04Z.

## Canonical (arm A) runs these replicate against

Each cell is arm B of a same-condition pair; arm A is the already-bound authoritative run in
`docs/checkpoints/pre_run/fire_manifest.json`.

| # | Cell (condition) | Cost | Purpose |
|---|---|---|---|
| 1 | B0 × classifieds × phantom_text | paid | first clean floor for a **phantom** arm |
| 2 | B0 × classifieds × phantom_prompt | paid | ditto |
| 3 | B0 × classifieds × phantom_som | paid | ditto |
| 4 | B1 × classifieds × vision | free | first B1 floor with any power |
| 5 | B1 × classifieds × dom | free | descriptive |
| 6 | B1 × classifieds × phantom_som | free | descriptive |
| 7 | B1 × classifieds × phantom_text | free | descriptive |
| 8 | B1 × classifieds × phantom_prompt | free | descriptive |

Already landed and registered under this same policy: **B1 × classifieds × som (R28065)**,
registered in `CLEAN_PAIRS` as `B1.cls.som` on 2026-08-17.

## Power, declared up front so the results cannot be oversold later

Floor measurability scales as `d ≈ n × SR × 0.59` (§468 / B-1972). Under that rule:

- **Cells 1-3 (B0 phantom arms): d ≈ 21-26** — the only cells in this chain powered to
  report an interval. These are the reason the chain is worth paying for.
- **Cells 4-8 (B1): d ≈ 8-10**, below the d<10 reporting bar this project set for itself.
  They are inventory and descriptive context, **not** interval evidence.

Declaring this before the data exists is the point: it removes the option of deciding, after
seeing the numbers, which cells were "the real measurement".

## What still is not solved by this file

A per-cell run ID cannot be reserved in advance here — run IDs are minted at launch with a
timestamp and nonce. This file pins the *intent* (which conditions, what they replicate, how
they may be read); it does not pin the identifiers. codex's stronger proposal — reserve IDs
before spawning and record a launch nonce — remains open, and is the durable fix.

---

## AMENDMENT 2026-08-19 — cells 6-8 cancelled, with reason

**Cancelled**: cell 6 (B1 × cls × phantom_som), cell 7 (B1 × cls × phantom_text),
cell 8 (B1 × cls × phantom_prompt). Cells 1-5 stand: 1-3 landed and are registered
in `CLEAN_PAIRS`, 4 landed and is registered, 5 is mid-flight (55/224 at 11:47Z) and
is being allowed to finish.

This file's own rule is "every cell below gets registered, whatever its number says",
and that rule is about **not hiding a number after seeing it**. Cancelling cells 6-8
is a different act — the numbers were never seen, and the reason is stated here before
they could be. Written so the omission is auditable rather than silent.

**Reason 1 — the measurement is already made, and it is a constant.** Cells 4 and 8's
sibling arms have both landed, and both read **0.00%**:

| pair | n | SR A→B | discordance | step-count differences |
|---|---|---|---|---|
| `B1.cls.som` | 224 | 14.29% → 14.29% | **0.00%** | 4 |
| `B1.cls.vision` | 224 | 12.50% → 12.50% | **0.00%** | **0** |

`B1.cls.vision` is the cell this file itself called "first B1 floor with any power"
(d≈16.6, the highest of the five) — so a zero here is not a power artefact. Its 224
episodes reproduce to the step. That matches the mechanism already established in
§298.2 (B1 is dense and runs locally at temperature 0; 133/133 determinism under
controlled replay), which predicts the remaining B1 cells return 0.00% as well.
Cells 6-8 carry d≈8.9/10.1/8.9 — below this project's own d≥10 bar for quoting an
interval — so they were inventory rather than measurement even before this.

**Reason 2 — a B1 floor cannot bound a B0 effect anyway.** The eight registered pairs
now split cleanly: B0's six arms sit at 10.27-14.29%, B1's two at 0.00%. That is the
useful result — it locates the ~12% in B0's serving stack rather than in the benchmark,
the agent, or VWA. But it also means the B1 floor constrains nothing about the effect
sizes measured on B0. Adding four more zeros does not change what any claim may say.

**Reason 3 — what the wall-clock buys instead.** Cells 6-8 run to ~08-22. The
constraint that originally shaped this chain ("08-21 must be free to respond to the
REALM verdict") turned out to rest on a wrong date: the verdict is **09-07**, and the
thesis moved to **09-05** (笔记 §470.9 — the date has now flipped twice; 08-21 is the
stale value). So the window is 19 days, not 2, and the argument for cancelling is no
longer "keep the machine free" but "this is the least valuable thing the machine could
be doing". Higher-value and direction-independent candidates: the reddit phantom
floors (B0 × red × 3 phantom — the cls-side floors are what let §470.3 put a noise
envelope on the unique-solve counts, and reddit currently has none), and validating
the newly wired B5/GPT-5.6 path end-to-end on a real site.

**What is NOT claimed**: that the remaining B1 cells would be 0.00%. They are unrun.
The prediction is stated so it can be checked cheaply later — B1 costs no API budget,
so any of cells 6-8 can be run at any time if a reviewer asks for the fourth, fifth
and sixth zero.

**Mechanics**: the chain orchestrator (`queue_chain.sh`, PID 1962767) is killed by PID.
The cell-5 runner (2294686) and its watchdog (2294723) have PPID=1 and their own process
groups — they are already daemonised and are unaffected. The orchestrator's per-cell
work after a runner exits is *validation* (episode count, condition_id match), not
production, so nothing is lost by its absence; that check is performed by hand when
cell 5 lands, together with the sync and registration steps that were already manual.

---

## CORRECTION 2026-08-20 — 格 5 落地, AMENDMENT 的 Reason 1 前提不成立

格 5 (B1 × cls × dom, `R14980`) 于 2026-08-20 03:32Z 跑满 224 episode，已按本文件规则
注册进 `CLEAN_PAIRS`（commit `37e473d`）。数字是：

| pair | n | SR A→B | discordance | 判据 |
|---|---|---|---|---|
| `B1.cls.som` | 224 | 14.29% → 14.29% | 0.00% | AMENDMENT 引用 |
| `B1.cls.vision` | 224 | 12.50% → 12.50% | 0.00% | AMENDMENT 引用 |
| **`B1.cls.dom`** | 224 | **6.25% → 6.70%** | **3.12% (7/224)** | **本次** |

**AMENDMENT 的 Reason 1 说**「the measurement is already made, and it is a constant」，并
据 §298.2 预测「the remaining B1 cells return 0.00% as well」。格 5 是这条预测的第一次
检验——写 AMENDMENT 时它正在跑（55/224）——**预测不成立**。

**因此需要改述的**：
- ❌「B1 的地板是 0.00%」/「B1 是个常数」——不能再这么写。
- ✅ 仍成立：B0 六臂 10.27–14.29% **≫** B1 三臂 0.00 / 0.00 / 3.12%。数量级差异是稳的，
  Reason 2（B1 地板约束不了 B0 上量到的效应量）**不受影响**，反而更强——B1 自己有地板。
- ⚠️ 新出现的问题，本数据答不了：3.12% 是 **dom 特有**，还是 B1 在 som/vision 上的两个
  0.00% 才是特例。两个 0.00% 是逐 episode 完全复现（vision 连 step count 都零差异），
  而 dom 翻了 7 个 task——这不是"稍微有点噪声"，是**同一模型同一温度下两种截然不同的
  行为**。格 6-8（B1 × cls × 三个 phantom 臂）现在测的正是这个，而 AMENDMENT 砍掉它们
  的理由恰恰是"已经是常数了"。

**AMENDMENT 的 Reason 2 与 Reason 3 不依赖 Reason 1**，所以取消本身未被推翻；被推翻的是
它给出的三条理由里的第一条。格 6-8 是否重排另行决定并记在本文件下方。

**⚠️ 这个 pair 不是同代码复现**：arm A 跑于 2026-06-03，arm B 跑于 2026-08-19，中间隔了
2.5 个月的代码漂移。som / vision 两对同样跨这段漂移却给出 0.00%，所以漂移不能直接解释
dom 的 7 个翻转——但它也没被排除，本文件不主张已排除。
