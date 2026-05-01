# Integrate 5 Gemini DR Zoom 3 lit anchor reports → paper.bib expansion

**Date**: 2026-05-01
**Scope**: Append BibTeX entries from 5 Gemini DR reports into formal `paper.bib` (with dedup + provenance comments).
**Out of scope**: paper_planning §2 updates / paper drafts / paper §5 prose / Q5 (still pending Gemini quota).
**Style**: Mechanical extraction + dedup. Don't rewrite, don't refactor.

---

## Goal

5 Gemini DR reports in `docs/literature/5.1/` each contain a "SECTION 2 — BibTeX entries" block with ~5-10 entries. Aggregate these into `docs/checkpoints/paper_drafts/paper.bib` formal expansion, deduplicating against existing 16 entries.

---

## Input — 5 Gemini DR reports

```
docs/literature/5.1/Cost-Aware Routing for Vision-Language Web Agents An Empirical Analysis of Text-Only Accuracy Retention.md          # Q1 Mirage (~10 entries)
docs/literature/5.1/Sensitivity of Large Language Model Agents to System Prompt Instruction Formatting in Multi-Step Environments.md    # Q2 prompt-format multi-step (~5-10 entries)
docs/literature/5.1/Empirical Analysis of Observation Modalities in Autonomous Web Agents Hierarchical Trees vs. Flat Indexed Lists.md  # Q3 AXTree-vs-flat (~5-10 entries)
docs/literature/5.1/Modality Collapse and the Illusion of Visual Grounding An Exhaustive Analysis of the Scaffold Effect in Vision-Language.md  # Q4 Scaffold (~5-10 entries)
docs/literature/5.1/Examining the Lazy Minimization Hypothesis Scaling Laws, Text-over-Vision Bias, and Routing Dynamics in Vision-Language.md  # Q6 Lazy Minimization (~5-10 entries)
```

Total expected: ~30-50 BibTeX entries across 5 reports.

---

## Deliverable

`docs/checkpoints/paper_drafts/paper.bib` updated:

1. Read existing 16 entries first. Build inventory (key → author + year + title).
2. For each of 5 reports, extract "SECTION 2 — BibTeX entries" block (`@article{...}`, `@inproceedings{...}`, etc.).
3. Dedup: skip new entries whose key OR (author + year + title-substring) matches existing.
4. Append surviving new entries to end of `paper.bib`, grouped by Q-source with provenance comments:

```
% ─────────────────────────────────────────────────────────
% Gemini DR 2026-05-01 — Zoom 3 lit anchor expansion (5/6)
% Q5 (bidirectional modality fusion) pending Gemini quota
% ─────────────────────────────────────────────────────────

% ---- Q1 Mirage / visual prompting w/o image ----
@article{asadi2026mirageillusionvisualunderstanding, ... }
@inproceedings{kaduri2024whatsintheimage, ... }
... (more)

% ---- Q2 system prompt format multi-step ----
... 

% ---- Q3 AXTree vs flat list ----
... 

% ---- Q4 Scaffold Effect cross-domain ----
... 

% ---- Q6 Lazy Minimization scaling ----
... 
```

5. Validate parseability (no syntax errors, balanced braces, valid entry types).

---

## Acceptance criteria

1. **paper.bib parseable** — entries balanced + valid types (@article/@inproceedings/@misc).
2. **Entry count** grows from ~16 to ~40-50 (depending on dedup).
3. **No duplicate keys** in final file (must be unique).
4. **Provenance comments present** — each Q-source has `% ---- Q<N> <topic> ----` header.
5. **Existing 16 entries unchanged** — only append, don't modify.
6. **End-of-file report**: print to stdout count of (new entries appended / duplicates skipped / total entries) when done.

---

## Out of scope

- Don't update paper_planning.md (separate manual task, user is doing).
- Don't update paper drafts.
- Don't update 实验笔记.md (manual chronicle).
- Don't run analysis pipeline.
- Don't rewrite or reformat existing entries.
- Don't add Q5 placeholder.

---

## Reference docs

- `docs/checkpoints/paper_drafts/paper.bib` (target file, existing 16 entries)
- `docs/literature/5.1/*.md` (5 Gemini DR reports, "SECTION 2" contains BibTeX blocks)

---

## Implementation order

1. Read paper.bib, build (key, author+year+title-substring) inventory.
2. For each of 5 reports, grep "SECTION 2 — BibTeX entries" block boundaries, extract @-prefixed entries.
3. Parse entries, build dedup decision per entry.
4. Append to paper.bib with provenance comments, organized by Q-source.
5. Run `grep -c "^@" paper.bib` to count, validate.
6. Print summary report.

Total estimated: ~5-10K tokens output.
