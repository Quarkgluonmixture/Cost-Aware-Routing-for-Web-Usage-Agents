# p79-lit-digest prompt 增量 patch (2026-07-05)

> **待并入目标**: WeChat 侧 Hermes agent 的 `p79-lit-digest` cron skill prompt
> (job_id `8034281c1cb9`; skill 目录 = Hermes 宿主机的 `.hermes/skills/research/p79-lit-digest/`，
> 即 quark Windows 侧 / Hermes cron sandbox，**不在 DGX 本机** — DGX 只有 2026-06-12 sandbox
> 留下的空壳目录和 `docs/literature/raw/` 的 digest 输出副本)。
> **依据**: `docs/literature/raw_digest_triage_2026-07-05.md` §4 全量核验 (45 papers) 确认的
> 3 个生成端系统性问题：§5-anchor 漂移 / 无 bib-dedup / 类比通胀。06-15 triage
> (`cron_digest_2026-06-12_triage.md` §4) 已诊断前两条，未修。
> **User 操作**: 在 Hermes 端编辑该 skill 的 prompt，把下面 3 段按标注位置粘入；
> 原有结构 (L0-L3 白名单 / narrative mapping / Score 分层 / 双轨输出) 不动。

---

## 增量段 1 — LIVE scope 纠偏（插入位置：评分规则段之前，作为 scope 前提）

```
【LIVE scope 前提 — 评分前必读】
P79 当前 paper-1 LIVE scope = ① phenomenon（phantom routing space：3 个 phantom 臂的
complementarity / drop-one oracle / 独立路由空间）+ ② router（rule-based + learned，
task-text 特征）。**§5 mechanism 已于 2026-05-14 搁置到 paper-2**，不再是 LIVE claim。

因此：mechanism 类论文（activation patching / layer probe / linear probe / SAE /
steering vector / logit lens / circuit 分析）**默认 Score ≤ 2，并在 RELATION 行标注
"paper-2-park"**。唯一例外：论文含 behavioral 层面的 paper-1 接口（如 image-ablation /
遮挡不变性 / benchmark-rewards-blindness 这类可直接进 §1/§2 motivation 的行为学结果），
此时只按 behavioral 半边评分，mechanism 半边仍标 paper-2-park。
「命中 LIVE claim / §」字段不得以 §5 作为 Score 3 依据。
```

## 增量段 2 — bib-dedup（插入位置：筛选流程开头，fetch/web-search 之后、评分之前）

```
【bib 去重 — 评分前置步骤】
评分前先取 P79 已收录文献的 arXiv ID 列表：
  grep -o 'eprint = {[0-9.]*}' docs/checkpoints/paper_drafts/paper.bib
（cron sandbox 拿不到 repo / E: 盘时，用 latest_fetch.json 内随附的 bib-ID 快照；
若快照也缺，在 digest 头部声明「本期未做 bib-dedup」。）

候选论文的 arXiv ID 命中该列表者：标注 "already-in-bib"，**不占 Score 3 名额**，
默认不再展开；仅当有新的 positioning 增量（新版本 / 新实验 / 此前未 position 的
section 落点）时保留一行说明增量本身。
```

## 增量段 3 — 类比纪律（插入位置：RELATION / narrative mapping 规则段之后）

```
【类比纪律 — RELATION 定性门槛】
RELATION 要写「结构同构 / 实证 sibling / 直接佐证 (SUPPORT)」，必须先通过轴对齐论证：
该论文操纵或测量的变量轴，是否与 P79 的 format × prompt-style（representation
routing）轴是同一根轴？须在条目里写出一句对齐论证（"其 X 轴 = P79 的 Y 轴，因为…"）。
写不出同轴论证的，一律降格为「主题呼应 (thematic echo)」或 RELATED-ONLY，
不得用"同构/sibling/佐证"字样。

THREAT 定性同理：必须写出攻击链条的具体传导路径 —— hostile reviewer 拿该论文的
哪个结果、打 P79 的哪条具体 claim、中间经过什么推理步骤。写不出传导路径的，
降格为 caution / awareness（Score 1），不得标 THREAT。
```

---

## 备注

- 3 段均为增量，不改动现有 L0-L3 关键词白名单、双轨输出（vault digest + 微信 push）、
  Score 3 ≤3 篇/期等既有规则。
- 增量段 2 的兜底路径依赖 fetch 脚本：建议顺手让 fetch 脚本把 bib-ID 快照写进
  `latest_fetch.json`（fetch 侧改动，不属本 patch 范围）。
- 并入后可用下一期 digest 验收：① mechanism 论文不再以 §5 为由拿 Score 3；
  ② 已在 bib 的论文带 "already-in-bib" 标注；③ SUPPORT/THREAT 条目带轴对齐 / 传导路径论证。
