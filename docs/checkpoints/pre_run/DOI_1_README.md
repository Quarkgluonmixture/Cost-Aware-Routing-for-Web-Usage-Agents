---
type: osf-registration-readme
status: pre-submission-draft
captured_at: 2026-05-18T13:57:22Z
artifact_existence_check: artifact_existence_check_doi1_20260518T135722Z.txt
git_head: 1d73e5c34cf266ad5a7ed9ec10a79cc34e3dcbca
preregistration_locked_sha: ef609a3863adc9b3698789b96a1ee9f709e1c832
preregistration_locked_tag: preregistration-locked
osf_doi: <to-be-assigned post-OSF-public-registration-approval>
osf_guid: <to-be-assigned at submission>
osf_submitted_at_utc: <to-be-recorded at submission>
---

# DOI 1 — Phase 1a Pre-canonical-outcome-creation Witness for Fire-3 (phantom-SoM pre-registration)

## 🚫 Retraction notice — original Fire-2-era witness VOIDED 2026-05-18 ~14:45 UTC

A previous Fire-2-era witness file (`artifact_existence_check_doi1_20260518T135722Z.txt`,
SHA-256 `e0e591f5b19c0248d4a7274cf8b19e54dbcc01706c859bfcbc2530e84de047d6`) was retracted
because its outcome-artifact capture pattern was incorrect (`episodes/*_summary.json` /
`*_steps.jsonl` instead of canonical `episodes/<site>_task_<N>_summary_v2.json` /
`<site>_task_<N>_steps_v2.jsonl` per `p79/experiment/logger_v2.py:111` + `analysis.py:209`).

Fire-2 is treated as an **aborted substrate test** (NLTK punkt missing → B-486 evaluator
failure → SIGTERM at UTC 14:04; watchdog auto-retry cleanup at UTC 14:06). Fire-2 is NOT
used as the canonical Phase 1a outcome source.

The **canonical DOI 1 witness** in this bundle was captured **at Fire-3 launch** (post-
NLTK-substrate-fix, after Fire-3 PID-alive but **before creation or inspection of any
Fire-3 outcome-bearing artifact**). The interim corrected scan at
`artifact_existence_check_doi1_interim_20260518T144258Z.txt` (SHA-256
`7563f0d55b651b604746ef0498fba3439ad7d7e130af97f0adda55e2bc7f1bf8`) documents the
post-fire-2-cleanup, pre-fire-3 substrate state and serves as audit-trail bridge between
retraction and canonical capture.

Full retraction audit trail: `master_bug_catalog.md ## /stress witness pattern bug
retraction` (B-1656~B-1665) + `实验笔记.md §231` + git tag `retraction/osf-doi1-witness-59c60c4`.

---

## Purpose

This OSF public registration is a **pre-canonical-outcome-creation witness** for the
**canonical Fire-3 Phase 1a run** of the phantom-SoM phenomenon paper (paper-1, EMNLP /
workshop target). It deposits the pre-registration document + analysis plan + locked code
state at a public-ledger timestamp that **precedes the creation or inspection of any
Fire-3 outcome-bearing artifact** on the experimental host.

Wording precision: "pre-canonical-outcome-creation" rather than "pre-outcome-creation"
acknowledges that Fire-2 (substrate test) produced + cleaned up partial outputs
between 13:28-14:06 UTC. The DOI 1 timestamp anchors to **Fire-3 = canonical paper-grade
run**, NOT to "no outcomes have ever existed in repo history" (which would be false).

## Empirical pre-canonical-outcome-creation status (canonical capture pending Fire-3)

**Canonical witness placeholder** — to be captured at Fire-3 PID-alive moment with
canonical schema patterns (per `p79/experiment/logger_v2.py:111` + `analysis.py:209`):

| Outcome tier | Canonical pattern | Count (canonical witness, TBD) |
|---|---|---|
| Per-episode outcome (earliest tier) | `episodes/<site>_task_<N>_summary_v2.json` | pending fire-3 launch |
| Condition-level outcome | `condition_summary_v2.json` | pending fire-3 launch |
| Step-level intermediate | `episodes/<site>_task_<N>_steps_v2.jsonl` | pending fire-3 launch |
| Aggregate outputs since fire-3 start | `results/phantom_paper/*.csv` | pending fire-3 launch |

The canonical witness will be captured at `artifact_existence_check_doi1_canonical_<fire3-UTC-TS>.txt`
within 5 min of fire-3 PID-alive + before any episode_summary_v2.json creation.

## Interim corrected scan (post-fire-2-cleanup, pre-fire-3)

UTC 2026-05-18T14:42:58Z scan with canonical schema patterns — documented in
`artifact_existence_check_doi1_interim_20260518T144258Z.txt` (SHA-256
`7563f0d55b651b604746ef0498fba3439ad7d7e130af97f0adda55e2bc7f1bf8`):

| Outcome tier | Canonical pattern | Count (interim) |
|---|---|---|
| Per-episode outcome | `episodes/<site>_task_<N>_summary_v2.json` | **0** (fire-2 cleaned at 14:06 UTC) |
| Condition-level outcome | `condition_summary_v2.json` | **0** |
| Step-level intermediate | `episodes/<site>_task_<N>_steps_v2.jsonl` | **0** |
| Run-dir liveness (any files) | known-positive probe | **0** files in fire-2 run dirs |

The interim scan documents the substrate state between fire-2 cleanup and fire-3 launch.
It is NOT the DOI 1 anchor — its purpose is to bridge the retraction of the buggy
fire-2-era witness and the capture of the canonical fire-3 witness, providing audit-trail
continuity for reviewers.

Pre-existing `fig0c_drop_one_bootstrap_ci.csv` (mtime 2026-05-18T13:25:14Z UTC,
99 bytes, **headers-only with 0 data rows**, **3 minutes BEFORE Fire-2 start at
13:28:06Z**) is verified as **archive analysis pipeline scaffolding**, NOT
Phase 1a outcome.

## Scope disclaimer (paper-grade hygiene — MANDATORY READING for OSF replicators)

This OSF registration documents the **pre-outcome analysis plan**. It
intentionally **excludes** ALL Phase 1a Pass-1 + Pass-2 outcome artifacts.

Numbers appearing in bundled `paper_drafts/section*.md` are EITHER:

(a) **Pre-fire archive placeholders** — e.g., `theta_fe=2.336` at k=3 cells
    from `meta_phantom_lift.csv` archive pilot (~2026-04), explicitly labeled
    "archive ground truth, NOT Phase 1a outcome" in `preregistration.md §2.4`
    empirical SE source row + `osf_lock_manifest.md §2.1` empirical SE source
    row. These are pilot signal levels used for statistical power analysis,
    NOT clean-run evidence.

(b) **Design notes / theoretical anchors** — derived from prior literature
    or pre-fire pilot.

They are **NOT** Phase 1a Pass-1 clean-run evidence. The clean-run outcome
data, final analysis, and finalized paper prose will be deposited separately
in **DOI 2** (mint trigger = Pass-1 + Pass-2 + analysis frozen + paper §1-§8
finalized; see `osf_lock_manifest.md §3b`).

## OSF operational ordering

Per OSF help docs (https://help.osf.io/article/330-welcome-to-registrations
+ https://help.osf.io/article/626-simplifying-the-preregistration-process):

1. **OSF Registration submission** = pre-outcome-creation timestamp anchor.
   The OSF page records the submission datetime as the cryptographic witness
   moment. This is the immutable evidence that the pre-registration was
   committed before outcome data existed.

2. **OSF Admin approval** = DOI string assignment. Default auto-approval at
   ~48h post-submission; can be manually approved sooner by the project
   admin (the user). Registration content is immutable once public.

3. **DOI string** = `10.17605/OSF.IO/XXXXX` once approved. Cite alongside
   the submission timestamp.

### Citation forms

**Before DOI assignment (interim, 0-48h post-submission)**:

```
OSF registration GUID osf.io/xxxxx, submitted 2026-05-18T<HH:MM:SS>Z UTC,
pre-outcome-creation witness (empirical 3-tier artifact-zero check at
submission time per artifact_existence_check_doi1_20260518T135722Z.txt);
DOI <pending OSF admin approval, default auto-approve 48h>.
```

**After DOI assignment (final, post-approval)**:

```
OSF DOI 10.17605/OSF.IO/XXXXX, submitted 2026-05-18T<HH:MM:SS>Z UTC,
pre-outcome-creation witness; registered Git tag `preregistration-locked`
at SHA ef609a3.
```

## Cross-link to DOI 2

**DOI 2 — Phase 1a reproducibility bundle**: `<to-be-assigned post Pass-1 +
Pass-2 + analysis-frozen + paper-finalized>`.

DOI 2's README will explicitly `cited_by` this DOI 1 (immutable forward
reference). This DOI 1 README does **NOT** mention a specific DOI 2 string
because DOI 1 is locked pre-DOI-2 — bidirectional references would be
anachronistic and would defeat OSF's immutable-registration semantics.

## Bundle contents (frozen at Git SHA `ef609a3`)

See `osf_lock_manifest.md §2.1` for the full SHA-locked artifact table. Key
files in this DOI 1 deposit:

| File | Role |
|---|---|
| `preregistration.md` | 14 commit decisions + H1/H2/H3/H10 gating, status: substance-locked 2026-05-18 per §A2 14/16 audit cascade |
| `osf_lock_manifest.md` | This directory's lock manifest, with §3a DOI 1 + §3b DOI 2 workflow split |
| `locked_versions.md` | B0 proxy endpoint + B1 HF SHA `ebb281ec70b05090aa6165b016eac8ec08e71b17` + B2 HF SHA `093f9f388b31de276ce2de164bdc2081324b9767` + VWA submodule HEAD `ac33d2fcd9cec2fcbeddd56d0fa3da58b4c7e927` + tree-hash chain `752caebdc6bd84761b2f308331f21241a9b4a28de65b46ff0007ef27d8c72778` |
| `model_card.md` | Cross-baseline architecture + decoding + capability scope |
| `dataset_card.md` | VWA classifieds + reddit task counts + N/A exclusion protocol |
| `ethics_license_coi_statements.md` | Holistic AI industry COI + license attribution |
| `evaluator_change_protocol.md` | T0-T3 evaluator-change tier classification |
| `compute_cost_carbon_table.md` | A100 GPU-hour estimates per condition |
| `neurips_checklist.md` | Submission integrity checklist |
| `negative_results_registry.md` | Pre-outcome state of negative-results commitment |
| `release_redaction_checklist.md` | Public-release scope per §7 reproducibility |
| `topvenue_constraints.md` | Submission venue analysis (EMNLP / workshop) |
| `pre_rerun_audit_walkthrough_2026-05-18.md` | §A2 cascade closure operational walkthrough |
| `reeval_audit_protocol.md` | FP architecture canonical state post-§139.8 |
| `env_snapshot.json` | A100 pre-fire snapshot per A2.7 B-1408 atomic write |
| `paper_drafts/section{1..8}_*.md` @ `ef609a3` | **Frozen pre-outcome state** — archive placeholder numbers per scope disclaimer above |
| `paper_drafts/paper.bib` @ `ef609a3` | Bibliography frozen pre-outcome |
| `artifact_existence_check_doi1_20260518T135722Z.txt` | **Empirical pre-outcome-creation witness** (3-tier artifact zero check + SHA256 self-verification) |
| `DOI_1_README.md` | This file (scope disclaimer + ordering doctrine + cross-link slot) |

## Doctrine restoration provenance

This two-DOI split + pre-outcome-creation lock + post-launch-pre-inspection
naming **restored** the original 2026-05-05 advisor sync §F.1 outcome
decision:

> "学生 lean DOI 时间戳 < data unblinding 时间戳让 audit trail ordering 明确"

— i.e., pre-data DOI upload — that drifted in `osf_lock_manifest.md §1`
between B-1570 doctrine shift (2026-05-18 ~09:30 UTC, which retired advisor
email as lock gate) and current correction (2026-05-18 ~14:00 UTC).

**Drift detection + correction lineage**:
- Original decision: 实验笔记 §110.3 (2026-05-05 advisor sync outcomes §F.1)
- Drift evidence: `osf_lock_manifest.md §1` pre-correction listed
  "Phase 1a Pass-1 baseline data complete" as DOI mint gate (contradicts
  §F.1 pre-data ordering)
- Correction: `master_bug_catalog.md` ## OSF DOI doctrine restoration
  (B-1650~B-1655) + 实验笔记 §230
- OSF help-doc citations: help.osf.io/article/330 + help.osf.io/article/626
  + this README's "OSF operational ordering" section

## Reviewer / replicator quick-reference

If you are a paper-1 reviewer or independent replicator reading this DOI 1:

1. **The pre-registration timestamp is the OSF submission datetime** —
   verifiable on the OSF page metadata, NOT the DOI assignment datetime
   (which can lag 0-48h).
2. **Numbers in paper drafts are NOT Phase 1a evidence** — see scope
   disclaimer above. Phase 1a clean-run evidence is in DOI 2.
3. **The empirical pre-outcome-creation claim is auditable** — see
   `artifact_existence_check_doi1_20260518T135722Z.txt` for the 3-tier
   zero check + SHA256 + Git HEAD + preregistration-locked tag SHA at
   submission time.
4. **The git substrate is verifiable** — checkout
   `git tag preregistration-locked` (= SHA `ef609a3`) to reproduce the
   pre-outcome code state exactly. All bundled files are tree-hash-chained
   per `preregistration.md §7` VWA submodule SBOM lock recipe.
