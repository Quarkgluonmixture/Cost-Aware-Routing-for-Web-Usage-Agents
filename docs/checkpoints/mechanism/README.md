---
name: mechanism workspace
description: Paper §5 mechanism-specific planning workspace — theory, methods, lit anchors, findings, open Qs, advisor sync. Splits the mechanism subset out of paper_planning.md once it grew paper-within-paper.
type: workspace_index
---

# Mechanism Workspace — paper §5

> ⏸️ **FROZEN 2026-05-14** — mechanism (§5) 整个暂搁 (advisor discussion 2026-05-14)，移至 **paper-2 scope**，不进当前 paper-1。本目录（README / plan / results）为**冻结存档**：下方「Status snapshot (2026-05-11)」「Stage 4 ⏳ 8/48 cells (bg)」「§5 prose pending Codex」等**均不再 active**。当前 forward state 见 [next_steps.md](../next_steps.md) §0a。仅作未来 paper-2 / 解冻参考。

## Elevator pitch (1-paragraph mechanism story)

P79 phantom routing space modes (DOM / SoM / Vision / P-text / P-prompt / P-SoM) are **linearly readable** from Qwen3-VL-4B residual stream at every layer 4–36 (Method 4.2 PCA cosine gap, AUROC 1.000 × 540 layer-pair tests). The image-axis dominates mechanism magnitude (peak gap 0.06 at L4–L17), text-axis is mid-scale (0.025 at L23), prompt-axis alone is weakest (0.007 at L36) — 10:4:1 hierarchy. P-SoM's closest mode at every layer is **P-text** (14.7× more distant from SoM), refuting "P-SoM = SoM minus image" and validating the format-axis framing. Mid-layer L17 acts as the **causally active planning site** (Stage 2/3 Cell A-H replacement patching disrupts output overlap; HDMI reliability framework gives Method 4.4 v2 mid-layer L17 α=5 = 0.44 reliability vs late-layer L33 α=10 = 0.23, because late-layer over-steers JSON envelope). This positions our work between Wu et al. 2026 (text-only Qwen 3 4B, tool selection 93% switch) and Ma & Rui 2026 (Qwen3 family rhyme newline 1% causal) — 50% mid-layer reliability is the multimodal-multi-step regime's signature.

## Status snapshot (2026-05-11)

| Stage | Status | Headline |
|---|---|---|
| Stage 1 (linear probe pilot) | ✅ done | L17 first-token logit shift peak |
| Stage 2/3 (replacement patching, 10 cells) | ✅ done | 8/8 Holm-sig L17 disruption, Cell C cls null asymmetry |
| Stage 4 Method 4.2 (PCA cosine gap) | ✅ done | AUROC 1.000 × 540 tests, 5/5 robustness pass |
| Stage 4 Method 4.4 v2 (mean-diff steering) | ⏳ 8/48 cells (bg) | Smoke: L17 α=5 H-mean 0.44 = current sweet spot |
| Method 4.5 (LA-HDMI / SAE) | 📋 future work | paper §8 anchor; Zekun-recommended SAE path |
| Paper §5 prose rewrite | 📋 pending Codex round | Uses 5-anchor lit + identification protocol |

## File index

- **[plan.md](plan.md)** — full mechanism plan: theory framework, lit anchor mapping, method design, identification assumptions table, current findings, open Qs, Zekun sync state, roadmap

## Cross-references

| Source | What's there |
|---|---|
| `paper_planning.md` §2 | Full Zoom 1-4 framework with axis 1/2/3 mechanism detail (this folder summarizes, not duplicates) |
| `paper_planning.md` §3 | 4-dimension evidence framework (Outcome / Macro / Micro / Efficiency) — mechanism evidence dimensions |
| `paper_drafts/section5_*.md` | Final paper §5 prose (regenerate via codex rounds) |
| `实验笔记.md` §125 | 2026-05-11 chronicle: Stage 4 lands + Method 4.2/4.4 + Wu/Ma&Rui/HDMI/Position/Peale integration |
| `_status/section/section5.md` | Section-level frontmatter (status, last_codex_round) |
| `_status/cells/cell_*.md` | Per-cell frontmatter (status, last_run_id) for 16 phase-1 cells + 10 mechanistic cells |
| `paper.bib` | 5 mechanism anchors: `wu2026toolcalling` (2605.07990) / `maRui2026planning` (2605.07984) / `khorasani2026hdmi` (2605.07631) / `linLiu2026disclosure` (2605.08012) / `peale2026flexibleRouting` (2605.07805) |
| `docs/literature/hdmi_paper_note.md` | HDMI deep dive (Method 4.4 v2 H-mean rescue) |
| `docs/literature/wheres_the_plan_paper_note.md` | Ma & Rui probe-causal dissociation deep dive |

## When to update this folder vs other docs

- **plan.md updates**: when theory framing shifts, new identification assumption surfaces, advisor decision lands, or method strategy changes
- **paper_planning.md §2 stays canonical** for Zoom 1-4 — this folder summarizes for working speed
- **paper_drafts/section5_*.md stays canonical** for prose — this folder is the strategic substrate
- **实验笔记.md** stays canonical for chronicle — this folder doesn't repeat history, points to it

This folder is the **working substrate** between strategy (paper_planning) and prose (paper_drafts), specialized for mechanism. Treat it as a paper-within-paper workspace.
