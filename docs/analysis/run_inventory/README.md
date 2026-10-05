---
type: analysis
status: living
created: 2026-10-06
purpose: every run we have, on every host, in one place — the coverage map for completing the evidence layer
producer: scripts/maintenance/inventory_all_runs.py → scripts/maintenance/run_inventory_report.py
---

# Run inventory — all hosts, all runs (2026-10-06)

Regenerate (read-only, ~10 s + ~1 min):

```bash
python scripts/maintenance/inventory_all_runs.py --out docs/analysis/run_inventory
python scripts/maintenance/run_inventory_report.py --inv docs/analysis/run_inventory
```

| file | what |
|---|---|
| `conditions.jsonl` | one row per condition **per copy** (source, path, episode count, artifact dirs, episode fingerprint) |
| `run_inventory.json` | one row per unique (run, condition): completeness, SR over the canonical scored set, registry status, doc citations |
| `run_matrix.md` | site × baseline × mode table of every **full** run with its SR and registry tag |

## 1. Where the data was, and the merged store

Five copies existed. They are now merged into **`E:\p79-runs\`** by
`scripts/maintenance/merge_run_stores.py` — hard links on E: (no extra space; the
original copies are untouched), plain copies only for the ~18 MB that existed solely on C:.
Provenance per file: `E:\p79-runs\_merge\provenance.jsonl`; summary `_merge\summary.json`.

| source | path | what it is |
|---|---|---|
| `local` | `C:\Workspace\Cost-Aware-Routing-for-Web-Usage-Agents\results` | episode text only (no artifacts); synced from A100 **up to 2026-09-24 07:24** — newer than the A100 pull |
| `a100_ws` | `E:\a100-condenser-backup\home-ubuntu\workspace\p79\results` | A100 repo results (WA, replicates, diag), pulled 09-22 |
| `a100_scratch` | `E:\a100-condenser-backup\mnt-scratch\p79_results_active_visualwebarena` | A100 VWA runs **with artifacts** (screenshots / DOM), pulled 09-22~23 → merged as `visualwebarena/` |
| `a100_archives` | `E:\a100-condenser-backup\mnt-scratch\p79_archives` | aborted / partial runs kept for audit → `_a100_archives/` |
| `dgx` | `E:\dgx-jiaming-backup\workspace\Cost-Aware-Routing-for-Web-Usage-Agents\results` | DGX mirror, final snapshot 2026-09-18 (incl. mechanistic) |

Conflict rule (same relative path, different content): **local > A100 > DGX**; the loser is
kept under `E:\p79-runs\_conflicts\<source>\…` and listed in `_merge\conflicts.jsonl`
(702, almost all regenerated analysis outputs). The 5 episode-level conflicts are all
explained: 3 are DGX holding the pre-replacement steps files of R28173 tasks 87/149 and
R3561 task 346 (the §400.1 identity case — A100 and local carry the fixed ones); 1 is
`B1_som_shopping_20260923` task 4, in flight at pull time; 1 is a run-level summary of
`B1_3mode_classifieds_20260413`.

⚠️ DGX mechanistic `_obs_mirror` paths exceed Windows MAX_PATH; both scripts use `\\?\`
paths. Without that, 17,722 directories were silently unlistable.

## 2. What we have

**131** unique conditions; **87 full runs** (see `run_matrix.md`). Registry status:

| status | full | partial | empty |
|---|---|---|---|
| `run_manifest.yaml` paper-grade (cls + red × B0/B1/B2 × 6, + B5 cls × 5) | 41 | — | — |
| `CLEAN_PAIRS` replicate | 18 | — | — |
| unregistered | 28 | 21 | 4 |
| archive / manifest-archived | 1 | 14 | 4 |

"Unregistered" is not by itself a defect: shopping and WA are discovered by glob in the
analysis scripts, not via the manifest. What is a gap is listed in §3.

SR check: `B0_dom_classifieds … R21557` = 39/224 = 17.41%, identical to its manifest note.

## 3. Gaps in the evidence layer (as of 2026-10-06)

1. ~~**The 09-15 local replicate chain is unread.**~~ **Done 2026-10-06**: six pairs registered as
   `B1.wared.*`, recomputed by the producers, `serving_mode_floor` now renders the retraction (笔记 §529.4).
   Original note: WA·B1·reddit six arms all landed
   (09-15 → 09-22), forming six complete same-condition pairs with the July runs — none
   is in `CLEAN_PAIRS`, none is cited anywhere in `docs/`. Reading per the pre-declared
   intent file (`pre_run/local_replicate_chain_launch_intent_20260915.md`, Reading 1) is in
   实验笔记 §529: the three powered arms land at 9.62 / 8.65 / 5.77% — **all above the
   4.93% line that the intent file declared kills C1 (serving-path floor) as stated.**
2. **B1·shop·som replicate (L7) is partial**: 93/435 episodes in the latest local sync
   (09-24). The chain was scheduled to run to ~09-25 and then B2 ×3 (L8–L10) until a hard
   halt 10-05; anything after 09-24 exists only on the A100, which is unreachable.
3. **B0 shopping SoM and Vision** (R12449, R23934; 08-06/07) are complete but cited
   nowhere in `docs/`. B0 shopping has no phantom arms and no replicate (paid, proxy budget).
4. **B5 vision** R24364 is complete but is the known broken coordinate-contract run
   (B-1997) — deliberately out of the manifest; R16160 is a partial earlier attempt.
5. ~~**Conclusion layer lags the ledger by 871 entries.**~~ **Done 2026-10-06** (886 entries from §398, 实验笔记 §530). Original note: `docs/reference/known/conclusions/`
   aggregates §1–§397.10; the ledger now runs to §527 (2,919 entries). The 871 entries of
   §398–§527 (333 MEASURED · 290 ADJUDICATED · 151 RETRACTED · 67 DATA · 30 CLAIM_UNVERIFIED)
   are not in any topic file.
