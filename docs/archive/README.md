# docs/archive/ — 归档说明

本目录存放**已 superseded / 已完成 / 历史快照**文档，从 live 文档树移出以避免误读。归档项**不反映当前项目状态**，引用前先看日期。

## 归档边界

| 子目录 / 文件 | 来源 | 归档原因 |
|---|---|---|
| `analysis_pre_2026-05-15/` | `docs/analysis/` 旧布局（commit `d1a63e9` mv） | **FP 体系重构 (2026-05-14) + 6-cell scope 之前**的旧分析（含无前缀 `classifieds/`/`reddit/`、大写 `Analysis/`、WebArena `wa_*`、旧 `paper_drafts/`、`phantom_paper/`）。⚠️ **这些 SR/FP 数字 ≠ clean-run 数字**，勿与 Phase 1a clean-run 混用。 |
| `process_audits_2026-05-15/` | `docs/checkpoints/process/`（2026-05-20 归档） | 2026-05-15 一次性 pre-fire audit 产物（`ground_truth_audit` + 3× `cross_system_docker_audit`），findings 已闭，原错位在 config-replica 目录。 |
| `handoff_fix_31_red_tests.md` | `docs/checkpoints/`（2026-05-20 归档） | cross-session handoff，处方任务已由 commit `d33ae1a` 完成。 |

## 当前 live 文档去向

- 当前 scope / framing → `.claude/CLAUDE.md` + `docs/checkpoints/paper_planning.md`
- 当前分析输出 → `docs/analysis/cross_sites/`
- 历史 chronicle → `docs/checkpoints/实验笔记.md`

_归档边界 README 补于 2026-05-20（doc audit）。_
