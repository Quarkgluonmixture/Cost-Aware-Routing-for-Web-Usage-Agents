# Master Bug Catalog — Index (auto-generated)

> ⚠️ **Auto-generated** by `scripts/maintenance/index_bug_catalog.py --write`. 勿手改 — regenerate after adding entries.
> 🔎 Locate a number: `python3 scripts/maintenance/index_bug_catalog.py --find 1810`.
> 🩺 Health check: `python3 scripts/maintenance/index_bug_catalog.py --lint` (before commit).

**Generated**: 2026-05-24  ·  **Catalog**: master_bug_catalog.md (7898 lines, 128 sections)  ·  **B-numbers**: 793 canonical entries, 1227 referenced, range B-1~B-1857

## Health (lint)

- monotonicity inversions (canonical entries): **143** total (143 grandfathered pre-2026-05-21, **0 new**)
- duplicate bold defs: **7** (summary-table echo + detail-list; not bugs)
- number gaps: **630** — B-100–B-101, B-316–B-319, B-667–B-671, B-685–B-689, B-711, B-713–B-715, B-768–B-780, B-912–B-915, B-919, B-922–B-940, B-958–B-989, B-996–B-999, B-1022–B-1050, B-1065–B-1100, B-1102–B-1110, B-1112–B-1200, B-1209, B-1213–B-1218, B-1220–B-1229, B-1238–B-1259, B-1277–B-1279, B-1297–B-1300, B-1313–B-1319, B-1321–B-1379, B-1381–B-1399, B-1416–B-1419, B-1421–B-1423, B-1425–B-1426, B-1429–B-1430, B-1433–B-1499, B-1513–B-1549, B-1562–B-1569, B-1580, B-1606–B-1610, B-1612–B-1619, B-1632–B-1639, B-1647–B-1649, B-1656–B-1668, B-1680–B-1701, B-1703–B-1761, B-1795

## Section map (file order — find a number by its /stress cluster)

| Line | Section | B-numbers (canonical entries) |
|---|---|---|
| L44 | Master Bug List | B-1–B-37 |
| L883 | Phase 0 — Pre-Phase-A historical fixes (笔记 §5-§97) | B-39–B-80 |
| L1296 | §116 Pre-rerun audit findings (2026-05-08) | B-38, B-81 |
| L1380 | §136 Mechanistic /stress audit findings (2026-05-14) | B-82 |
| L1396 | §139 Pre-fire pipeline audit findings (2026-05-14) | B-83–B-85, B-87–B-99, B-102–B-116 |
| L1565 | §143 Post-Batch-1-5 propagation audit (cross-AI Mode B+C, 2026-05-1… | B-117–B-130 |
| L1614 | Updated Status Counts (post-§116 audit + Phase 0 backfill) | B-131–B-143 |
| L1686 | §145 Pre_run/ folder residual audit (2026-05-15 late evening, user-… | B-144–B-200 |
| L1898 | A1.4c /stress audit (2026-05-16) — B-202 to B-210 (10 entries) | B-202–B-210 |
| L1962 | A1.5 /stress audit (2026-05-16) — B-211 to B-229 (19 entries; 9 fix… | B-211–B-229 |
| L2083 | A1.13 + A1.14 /stress audit (2026-05-16) — B-230 to B-236 (7 entrie… | B-230–B-236 |
| L2134 | §158 /stress A1.6 `p79/experiment/analysis.py` — FP architecture ha… | B-237–B-253 |
| L2254 | /stress A1.7 fix-batch — `conditions.py` + `configs/*.yaml` (2026-0… | B-261–B-272 |
| L2340 | A1.16 /stress audit (2026-05-16) — B-273 to B-279 (7 entries; 7 fix… | B-273–B-279 |
| L2404 | /stress A1.8 fix-batch — schema + JSONL + dedup substrate (2026-05-16) | B-280–B-297 |
| L2519 | /stress A1.17 Chunk 1 — VWA setup + RESET_BEFORE protocol launch-bl… | B-298–B-306 |
| L2590 | §159 /stress A1.18 VWA submodule `p79-patches` — 3-AI audit + full … | B-254–B-260 |
| L2694 | /stress A1.17 Chunk 2 — paper-grade quality + Option K Trajectory E… | B-307–B-314, B-320–B-358 |
| L2909 | /stress A1.10 fix-batch — `p79/experiment/{router, modules, state_c… | B-359–B-383 |
| L3054 | A1.15 /stress audit (2026-05-16) — B-384 to B-394 (11 entries; Pre-… | B-384–B-394 |
| L3138 | A1.1 batch (B-395~B-405, 3-AI cross-audit, 2026-05-16) | B-395–B-405 |
| L3210 | /stress A1.2 batch — `p79/backends/` cross-baseline contract (2026-… | B-406–B-416 |
| L3285 | /stress A1.3 v9 batch — `p79/envs/` env-layer scaffold + D1 heurist… | B-417–B-425 |
| L3347 | A1.19 `scripts/analysis/aggregate_*.py` — pre-fire 分析管线 3-AI cycle … | B-426–B-438 |
| L3427 | A1.25 GRL (Generated Runtime Layer) audit — Chunk 1 batch (2026-05-… | B-439–B-448 |
| L3488 | /stress A1.4 SoM extraction chain unified bug batch (2026-05-17) | B-449–B-458 |
| L3555 | A1.20 `scripts/analysis/figures/*.py` — figure-script pre-fire 3-AI… | B-459–B-477 |
| L3659 | A1.25 GRL (Generated Runtime Layer) audit — Chunk 2 batch (observat… | B-479–B-484 |
| L3703 | A1.5b Phase 1 — `p79/experiment/runner/` control-plane audit (2026-… | B-485–B-505 |
| L3820 | A1.25 GRL (Generated Runtime Layer) audit — Chunk 3 batch (action p… | B-506–B-511 |
| L3871 | /stress A1.21 — `scripts/analysis/preregistration_decision_test.py`… | B-513–B-533 |
| L3989 | A1.25 GRL (Generated Runtime Layer) audit — Chunk 4 batch (VWA subm… | B-535–B-541 |
| L4045 | A1.5b Phase 2 | B-512, B-534, B-542–B-547 |
| L4099 | A1.22 | B-560–B-576 |
| L4192 | /stress A1.5 — post-A1.5b Phase 2 follow-on audit (2026-05-17 ~14:0… | B-548–B-555 |
| L4271 | A1.14 /stress audit Chunk (d) (2026-05-17) — B-703 to B-710 (8 entr… | B-703–B-710 |
| L4325 | A1.14 /stress audit Chunk (c) (2026-05-17) — B-681 to B-683 (3 entr… | B-681–B-683 |
| L4349 | A1.14 /stress audit Chunk (b) (2026-05-17) — B-677 to B-680 (4 entr… | B-677–B-680 |
| L4379 | A1.14 /stress audit Chunk (a) (2026-05-17) — B-672 to B-676 (5 entr… | B-672–B-676 |
| L4415 | A1.13 /stress audit (2026-05-17) — B-630 to B-648 (19 entries; 18 f… | B-630–B-648 |
| L4558 | /stress A1.6a — 2026-05-17 | B-596–B-603 |
| L4638 | /stress A1.18-re — 25 unique findings → 26 fixes (2026-05-17) | B-604–B-629 |
| L4790 | /stress A1.6b — `p79/experiment/analysis.py` Pareto + decision-test… | B-650–B-661 |
| L4883 | /stress A1.12 cold-start — `tests/` directory as paper-grade reprod… | B-662–B-666 |
| L5500 | A1.15b — GLM sidecar cluster Chunk α (2026-05-17) | B-841–B-844 |
| L5561 | A1.15b — GLM sidecar cluster Chunk β (2026-05-17) | B-845–B-847 |
| L5645 | A1.15b — GLM sidecar cluster Chunk γ (2026-05-17) | B-848–B-854 |
| L5713 | A1.15b — GLM sidecar cluster Chunk δ (2026-05-17) | B-855–B-856 |
| L5778 | A1.23 /stress concurrency + race contract — 3-AI cross-AI cycle 14-… | B-858–B-871 |
| L6659 | A2.7 — Confound register / known cross-baseline asymmetries — /stre… | B-1400–B-1414 |
| L7197 | /stress deep repo audit — 6-lineage cross-track Fire-6 blocker subs… | B-1762–B-1767, B-1787–B-1793 |
| L7286 | Fire-6 RCA Stage C — eval-isolation verified + screenshot-recovery … | B-1768–B-1775 |
| L7309 | /stress GRL boundary audit — "reliability not policy" sweep (2026-0… | B-1776–B-1783 |
| L7342 | Protocol Reset accounting (§244 #6/#7/#8, task #70) — 2026-05-20 | B-1784–B-1785 |
| L7359 | Protocol Reset accounting — /stress cross-AI audit (3-AI A+B+C) → B… | B-1786 |
| L7373 | B-1794 — B0 forced-tool-call schema ≡ validator (real fix; descript… | B-1794 |
| L7383 | B-1796 … B-1802 — Pre-Fire-6 /stress (3-AI) schema≡validator comple… | B-1796–B-1802 |
| L7403 | B-1803 — Fire-6 RCA C1b: evaluator isolation → FRESH browser contex… | B-1803 |
| L7415 | B-1804 — L1 router MI feature-selection hygiene: discrete_features … | B-1804 |
| L7433 | B-1805 … B-1809 — router pipeline pre-fire /stress 簇 α: oracle / la… | B-1805–B-1809 |
| L7455 | B-1810, B-1811, B-1818 — router pipeline pre-fire /stress 簇 β: run … | B-1810–B-1811, B-1818 |
| L7475 | B-1812, B-1813 — router pipeline pre-fire /stress 簇 γ: train→serve … | B-1812–B-1813 |
| L7494 | B-1814, B-1815, B-1816 — router pipeline pre-fire /stress 簇 δ: τ ob… | B-1814–B-1816 |
| L7512 | B-1817, B-1819, B-1820 — router pipeline pre-fire /stress standalon… | B-1817, B-1819–B-1821 |
| L7532 | B-1822 — Fire-6 condition-boundary flock fd-inheritance self-collis… | B-907, B-1822 |
| L7556 | B-1823, B-1824, B-1825 — Fire-6 relaunch hardening (/stress 3-AI, 2… | B-1823–B-1827 |
| L7576 | B-1828 — phantom latency estimand: 画框 instrumentation 计入 obs_prepar… | B-1828 |
| L7606 | B-1829 — diag_pattern_match `--failed-only` denominator bug (2026-0… | B-1829 |
| L7618 | B-1830 — vision raw-screenshot save in latency window (_save_artifa… | B-1830 |
| L7632 | B-1831 — env.reset Page.goto transient-timeout retry (2026-05-22, F… | B-1831 |
| L7644 | B-1832 — deferred image save `.tmp` 后缀 → PIL KeyError → som/vision … | B-1832 |
| L7662 | B-1833 — cls docker transient navigation stall 复发 → B-1831 retry bu… | B-1833 |
| L7678 | B-1834 — manifest/resume `episodes >= scored` 非 `==` → over-complet… | B-1834 |
| L7688 | B-1835 — `_run_episode` 局部 `import os` 遮蔽全局 → deferred-save `os.ope… | B-1835 |
| L7702 | B-1836 — eval retry `is_nav_error` 关键词缺 `"timeout"`(无空格)→ Playwrigh… | B-1836 |
| L7722 | B-1837 — eval 5-retry vs agent-step 0-retry asymmetry → differentia… | B-1837 |
| L7736 | B-1838 — sync_a100 rsync rc=24 (files vanished) treated as fatal → … | B-1838 |
| L7748 | B-1839 — per-condition docker restart for classifieds fresh substra… | B-1839 |
| L7772 | B-1840 — fire6_monitor false-positive: orchestrator-name + FIRELOG … | B-1840 |
| L7788 | B-1841 — Gate 3 fresh fire 启动漏 reset fire_manifest.json (Fire-6 R97… | B-1841 |
| L7808 | B-1842 — Parse-error sink: cost 进 wasted 桶但 canonical latency 不扣 (§… | B-1842 |
| L7818 | B-1843 — `parse_error_rate` 实为 injected-wait-sink rate (含模型主动合法 wai… | B-1843 |
| L7826 | B-1844 — Canonical-action validity 只看 parse-validity 不看 env outcome… | B-1844 |
| L7834 | B-1845 — WAIT-rescue soft-retry 非严格 state-neutral (gemini OOB) (202… | B-1845 |
| L7842 | B-1846 — 无 `termination_reason` 字段 (cap-induced 终止只能间接推断) (2026-05-23) | B-1846 |
| L7850 | B-1847 — `paper_grade_check.py` 无 B0 `tool_call_emit_rate` conditio… | B-1847 |
| L7858 | B-1848 — Playwright driver-wedge hang 绕过 operation timeout + runner… | B-1848 |
| L7874 | B-1849~B-1857 — Analysis-layer paper-grade alignment batch (3-AI /s… | B-1849–B-1857 |
