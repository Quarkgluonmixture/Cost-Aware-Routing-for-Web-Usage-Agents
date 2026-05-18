---
type: osf-registration-readme
status: pre-submission-witnessed  # tier 1 pre-canonical-outcome-creation witness captured 2026-05-18T21:16:28Z; ready for OSF submission pending user manual sign-off rows
captured_at: 2026-05-18T21:16:28Z  # pre-launch witness capture UTC (zero A100 run_dir state)
captured_tier: pre-outcome-creation  # tier 1 strictest per B-1675 capture_doi1_witness.sh — all canonical-pattern counts = 0 + any_files_in_run_dirs = 0 (no Fire-3 run_dir even exists at capture moment; substrate genuinely empty)
witness_capture_strategy: pre-launch  # captured BEFORE Fire-3 launch invocation rather than between gates-pass and runner-spawn, to side-step the ~5-15 sec runner-startup window that writes first *_steps_v2.jsonl intermediate
artifact_existence_check: artifact_existence_check_doi1_canonical_20260518T211628Z.txt  # in this same pre_run/ dir alongside this README
artifact_existence_check_sha256: 6056b905e25b0880e613eeb43a091f3281e107073736608755331185ffcadbe3  # full-file SHA-256 (matches MANIFEST_SHA256.txt; verifiable via standard `sha256sum`). Witness file's internal line 44 self-doc cites 011fa4c07d76798a57070d9aea8b1653ca50dfe432b9c4dee689b96ee9cc6691 = content-only SHA-256 (file bytes excluding self-doc epilogue line 44 itself; chicken-and-egg — including the SHA inside the file changes the SHA). See §"Witness file hash convention" below.
git_head: 590ad12f9cea69bcb06fb0e12d1e09837ac257da  # master HEAD at pre-launch witness capture; descendant of canonical_substance_lock_sha
canonical_substance_lock_sha: 72b93c939526fb0b9a733013852bd90136d601aa  # substance-lock post Q3=A doctrine fix + B-### renumber
canonical_substance_lock_tag: preregistration-locked-q3a  # at 72b93c9 above
legacy_substance_lock_sha: ef609a3863adc9b3698789b96a1ee9f709e1c832  # audit-trail only, NOT for OSF citation — superseded by Q3=A wave 2026-05-18 evening
legacy_substance_lock_tag: preregistration-locked  # at ef609a3 above, retained on origin for git-history continuity only; witness file's Provenance section shows this legacy tag SHA (99f72f4e) as a script artifact, NOT as the canonical anchor — the canonical anchor is `preregistration-locked-q3a` at 72b93c9 documented in this README body
fire3_launch_planned_ts_utc: 2026-05-18T~21:17-21:18Z  # Fire-3 launch will fire IMMEDIATELY after this README + witness commit lands, ~1-2 min after witness capture
fire3_run_dir_pattern: results/visualwebarena/phase1/B?_*_20260518_21*  # cls/red chain run_dirs will be created post-launch
osf_doi: 10.17605/OSF.IO/9QCWU  # minted 2026-05-18T23:10:06Z UTC at OSF registration submission; archive auto-completed instantly (no 48h wait per OSF Standard Preregistration default). License: CC-By Attribution 4.0 International (OSF-default)
osf_guid: 9qcwu  # OSF Registration GUID (URL: https://osf.io/9qcwu); distinct from parent project GUID kv9sf (URL: https://osf.io/kv9sf) which holds the deposit files
osf_submitted_at_utc: 2026-05-18T23:10:06Z  # precise UTC from OSF API (https://api.osf.io/v2/registrations/9qcwu/ date_registered field); cryptographic external anchor paired with Git tag preregistration-doi1-witnessed-20260518T211628Z @ 5edac3b. Note: chronologically OSF submission UTC is 1h54min after canonical witness capture (21:16:28Z), during which Fire-3 launched and produced outcome data — DOI 1's "pre-canonical-outcome" claim is anchored at the witness file's recorded counts (= 0 at 21:16:28Z), NOT at OSF submission moment. The OSF DOI cryptographically attests that the witness file (with full-file SHA 6056b905...) existed in this state when OSF submission archived it.
osf_witness_tag: preregistration-doi1-witnessed-20260518T211628Z  # tagged at master HEAD on the witness-commit (one commit on top of canonical_substance_lock_sha lineage)
fire2_void_witness: retracted/artifact_existence_check_doi1_20260518T135722Z_VOID_RETRACTION_ONLY.txt  # forensic-only, NOT part of canonical witness chain
fire2_interim_corrected_scan: artifact_existence_check_doi1_interim_20260518T144258Z.txt  # post-fire-2-cleanup audit-trail bridge, NOT DOI 1 anchor (now superseded by canonical_20260518T211628Z)
fire3_attempt3_partial_archived: results/visualwebarena/phase1/_archive_pre_fire3/B0_attempt3_killed_partial_20260518_2105/  # killed Fire-3 attempt #3 partial run_dir; archived for audit trail, NOT part of DOI 1 outcome data
---

# 🟢 WITNESSED — DOI 1 README ready for OSF submission pending user manual sign-off

> **STATUS**: **Tier 1 pre-canonical-outcome-creation witness captured 2026-05-18T21:16:28Z** at zero A100 run_dir state — all canonical-pattern counts are 0 + any_files_in_run_dirs = 0 (no Fire-3 run_dir even exists at capture moment; substrate genuinely empty). This is the strictest possible DOI 1 empirical anchor.
>
> **Witness capture strategy = pre-launch** (per B-1751-fu / Fire-3 attempt #4 strategy). The 4 prior attempts (smoke 6 + attempts #1-#3) all had operational issues:
>   - Attempt #1 (20:53:06Z): preflight FAIL on NLTK `find("tokenizers/punkt_tab")` path-resolution bug (NLTK 3.9+) — Fire-3 didn't launch
>   - Attempt #2 (20:59:34Z): preflight FAIL on Gate 1 un-allowlisted TBD-placeholder marker in prereg frontmatter — Fire-3 didn't launch
>   - Attempt #3 (21:03:35Z): preflight ALL PASS + Fire-3 launched + B0 dom classifieds task 0 ran ~2 min + canonical witness captured at 21:06:50Z BUT landed at tier 2 (pre-outcome-inspection — `summary=0, condition=0, steps=1`) because witness capture happened ~2 min after runner started writing first step jsonl. Attempt killed + partial run_dir archived for audit trail (no contamination of Fire-3 outcome data because run_dir is in `_archive_pre_fire3/`).
>   - Attempt #4 (THIS witness + imminent launch): pre-launch witness capture at zero A100 state → tier 1 strict. Fire-3 launch follows immediately after this README + witness commit + tag.
>
> **READY for OSF submission** — pending only: (a) user manual sign-off rows in `release_redaction_checklist.md` (2 explicit rows blocking submission, NOT auto-fillable); (b) user click on OSF UI submit button. Submission timestamp = the cryptographic external anchor.

## Canonical DOI 1 anchor (single-source-of-truth — cite this block for all OSF-facing claims)

```
Current canonical substance-lock anchor:
  Git SHA       : 72b93c939526fb0b9a733013852bd90136d601aa
  Git tag       : preregistration-locked-q3a   (annotated, at 72b93c9)
  Lock time UTC : 2026-05-18T11:30:00Z (substance) + 2026-05-18T21:16:28Z (empirical witness, pre-launch capture at zero A100 run_dir state — strategy revision attempt #4 per B-1751-fu, NOT post-PID-alive)
  Cite as       : "OSF preregistration registered at Git tag preregistration-locked-q3a (SHA 72b93c9),
                   substance-locked 2026-05-18T11:30Z per Q3=A doctrine fix wave B-1750~B-1759"

Legacy anchors retained for audit trail ONLY (NOT for OSF citation):
  Git SHA       : ef609a3863adc9b3698789b96a1ee9f709e1c832    (pre-doctrine-fix; superseded)
  Git tag       : preregistration-locked                        (legacy, at ef609a3)
  Frontmatter   : 88521b9e254955ab156dc8115da826f171ef5990    (earlier prereg `prereg_registered_git_sha` value)
  These remain on origin for git-history continuity but should NOT be referenced as the DOI 1 anchor.
```

# DOI 1 — Phase 1a Pre-canonical-outcome Witness for Fire-3 (phantom-SoM pre-registration)

## 🚫 Retraction notice — original Fire-2-era witness VOIDED 2026-05-18 ~14:45 UTC

A previous Fire-2-era witness file (`retracted/artifact_existence_check_doi1_20260518T135722Z_VOID_RETRACTION_ONLY.txt`,
SHA-256 `e0e591f5b19c0248d4a7274cf8b19e54dbcc01706c859bfcbc2530e84de047d6`) is **VOIDED**
and **retained only as a retraction artifact**. It is **not** used as the DOI 1 timestamp
anchor or empirical zero-outcome witness.

Reason for retraction: outcome-artifact capture pattern was incorrect
(`episodes/*_summary.json` / `*_steps.jsonl` instead of canonical
`episodes/<site>_task_<N>_summary_v2.json` / `<site>_task_<N>_steps_v2.jsonl` per
`p79/experiment/logger_v2.py:111+114` + `analysis.py:209`).

Fire-2 was an **aborted substrate test** (NLTK punkt missing → B-486 evaluator failure →
SIGTERM at UTC 14:04; watchdog auto-retry cleanup at UTC 14:06). It may have produced
transient run-directory files; the corrected canonical-pattern scan after cleanup found
zero canonical outcome-bearing artifacts. Fire-2 outputs are excluded from DOI 1 and from
canonical Phase 1a analysis.

The **canonical DOI 1 witness** was **captured at 2026-05-18T21:16:28Z** via the
regression-tested capture script `scripts/maintenance/capture_doi1_witness.sh` (B-1675;
canonical patterns hardcoded with schema citation; `tests/test_doi1_witness_pattern.py`
9 cases PASS). Per the attempt-#4 strategy revision (B-1751-fu), capture happened
**pre-launch** at zero A100 run_dir state — **before Fire-3 launch invocation** rather
than between gates-pass and runner-spawn, side-stepping the ~5-15 sec runner-startup
window that writes the first `*_steps_v2.jsonl` intermediate (attempt #3 evidence:
canonical witness at 21:06:50Z landed at tier 2 with `steps=1` already present
because witness was taken ~2 min after runner started). Tier 1 strict (canonical-pattern
counts all 0 + `any_files_in_run_dirs = 0`) is empirically demonstrated; see canonical
witness section below for full counts. The interim corrected scan at
`artifact_existence_check_doi1_interim_20260518T144258Z.txt` (SHA-256
`7563f0d55b651b604746ef0498fba3439ad7d7e130af97f0adda55e2bc7f1bf8`) documents the
post-fire-2-cleanup, pre-fire-3 substrate state and served as audit-trail bridge
between retraction and canonical capture — it is **not** the DOI 1 anchor.

Full retraction audit trail: `master_bug_catalog.md ## /stress witness pattern bug
retraction` (B-1670~B-1679) + `实验笔记.md §231` + git tag `retraction/osf-doi1-witness-59c60c4`.

---

## Purpose

This OSF public registration is a **pre-canonical-outcome witness** for the **canonical
Fire-3 Phase 1a run** of the phantom-SoM phenomenon paper (paper-1, EMNLP / workshop
target). It deposits the pre-registration document + analysis plan + locked code state
at a **public timestamp** that **precedes creation or inspection of any Fire-3 outcome-
bearing artifact** on the experimental host.

**Wording precision**: the title says "pre-canonical-outcome" (not the stronger
"pre-canonical-outcome-creation") until the canonical witness is captured + verified.
Once captured with all canonical-pattern counts zero, the claim strengthens to
**pre-canonical-outcome-creation** and the title updates accordingly. If any canonical
pattern returns nonzero at capture time, the claim tier downgrades automatically per the
capture script's tier-detection logic (pre-canonical-outcome-creation → pre-outcome-
inspection → pre-analysis).

Fire-2 history is acknowledged: it was an aborted substrate test, NOT a canonical run.
DOI 1 timestamp anchors to **Fire-3 = canonical paper-grade run**, NOT to "no outcomes
have ever existed in repo history" (which would be false).

**This wording claim is conditional on the canonical witness counts being zero at
capture time**; verified empirically by `capture_doi1_witness.sh` known-positive probe
(B-1675 P1-4) which detects schema-mismatch (target=0 AND any-files>0 = abort) and
auto-downgrades the tier label if outcomes already exist.

## Canonical Fire-3 witness (CAPTURED 2026-05-18T21:16:28Z — tier 1 strict)

```
file:              artifact_existence_check_doi1_canonical_20260518T211628Z.txt
SHA-256 (full-file): 6056b905e25b0880e613eeb43a091f3281e107073736608755331185ffcadbe3
SHA-256 (content-only, excluding self-doc line 44): 011fa4c07d76798a57070d9aea8b1653ca50dfe432b9c4dee689b96ee9cc6691
captured_utc:      2026-05-18T21:16:28Z
host:              a100-jiaming-test
uptime-since:      2026-05-16 00:44:06
witness strategy:  pre-launch (Fire-3 not yet invoked; A100 substrate at zero run_dir state)
canonical patterns (per p79/experiment/logger_v2.py:111+114 + analysis.py:209):
  episode_summary_v2 count:   0  ✓ tier 1 gate satisfied
  condition_summary_v2 count: 0  ✓
  steps_v2 count:             0  ✓
  run-dir matching pattern (results/visualwebarena/phase1/B?_*_2026*): no matches — substrate genuinely empty
known-positive probe (B-1675 P1-4 schema-mismatch detection):
  any_json_in_run_dirs:        0
  any_files_in_run_dirs:       0  ✓ tier 1 strictest (no scaffold-only false-positive)
provenance at capture:
  Git HEAD:                    590ad12f9cea69bcb06fb0e12d1e09837ac257da
  preregistration-locked tag SHA (legacy, script artifact): 99f72f4e8cc10b90cd4408fde07ce69482ea474b
  VWA submodule HEAD:          ac33d2fcd9cec2fcbeddd56d0fa3da58b4c7e927
STATUS:            pre-outcome-creation (canonical patterns return 0, any-files = 0 = strictest tier)
```

**Status**: ✅ All fields filled per canonical capture 2026-05-18T21:16:28Z.
The DRAFT marker has dropped. Witness file is byte-immutable; MANIFEST_SHA256.txt
hash is the canonical verifiable hash. See §"Witness file hash convention" below for
disambiguation of the two SHAs.

## Witness file hash convention

The witness file (`artifact_existence_check_doi1_canonical_20260518T211628Z.txt`) has
**two legitimate SHA-256 values** that both appear in this README and the audit trail:

| Hash | What it covers | Purpose | Verifiable how |
|---|---|---|---|
| `6056b905e25b0880e613eeb43a091f3281e107073736608755331185ffcadbe3` | **Full file** (all 44 lines including self-doc epilogue) | **MANIFEST + OSF deposit verification** — standard, reviewer-checkable | `sha256sum artifact_existence_check_doi1_canonical_20260518T211628Z.txt` |
| `011fa4c07d76798a57070d9aea8b1653ca50dfe432b9c4dee689b96ee9cc6691` | **Content-only** (lines 1-43, excluding self-doc epilogue line 44 itself) | **Self-documenting capture-script attestation** — recorded inside the file at line 44 | `head -43 <file> \| sha256sum` |

**Why both exist**: line 44 of the witness file is a self-referential SHA-256 attestation
(per `capture_doi1_witness.sh` B-1675 design). Including a file's own SHA inside the file
is mathematically a chicken-and-egg problem (the SHA changes when you include it), so the
capture script writes the **content-only hash** (hash of lines 1-43, before the self-doc
line is added) and labels it inside the witness. The line 44 prose `(full file including
this line's SHA reference)` is **slightly misleading** — the value `011fa4c0...` is
actually the **pre-line-44** content hash, not the full-file hash. This was a
capture-script wording bug, not a content tampering — the hash value itself is
mathematically correct as the content-only hash.

**Canonical for OSF verification**: use `6056b905...` (full-file hash). It matches
MANIFEST_SHA256.txt and any reviewer can verify with standard `sha256sum`. The
content-only hash `011fa4c0...` is retained for audit-trail continuity (it's documented
inside the witness file and was originally written into this README before the
disambiguation was understood).

The witness file itself is **byte-immutable** — its content (including the historically
inaccurate line 44 prose) is preserved verbatim to maintain capture-time integrity. Any
post-capture modification of the witness file would invalidate the witness chain.

## Interim corrected scan (post-fire-2-cleanup, pre-fire-3 — audit-trail bridge only)

UTC 2026-05-18T14:42:58Z scan with canonical schema patterns — documented in
`artifact_existence_check_doi1_interim_20260518T144258Z.txt` (SHA-256
`7563f0d55b651b604746ef0498fba3439ad7d7e130af97f0adda55e2bc7f1bf8`):

| Outcome tier | Canonical pattern | Count (interim) |
|---|---|---|
| Per-episode outcome | `episodes/<site>_task_<N>_summary_v2.json` | **0** (Fire-2 cleaned at 14:06 UTC) |
| Condition-level outcome | `condition_summary_v2.json` | **0** |
| Step-level intermediate | `episodes/<site>_task_<N>_steps_v2.jsonl` | **0** |
| Run-dir liveness check | No Fire-2 run directories or canonical output files remaining after cleanup | **0** files |

The interim scan documents substrate state between Fire-2 cleanup and Fire-3 launch. It
is **NOT** the DOI 1 anchor — its purpose is to bridge the retraction of the buggy
Fire-2-era witness and the capture of the canonical Fire-3 witness, providing audit-trail
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

They are **NOT** Phase 1a Pass-1 clean-run evidence.

**These placeholder numbers are not used to choose or revise the Phase 1a primary
decision rules in this registration.** The H1 / H2(a) / H3 / H10 gating thresholds
(δ=1.0pp, FE-pool inverse-variance estimand, α=0.05 one-sided, m=1 single test, K-of-N
transparency-only) were locked **before** Phase 1a fire and are immutable post-OSF
submission; archive pilot signals informed the power analysis (§2.4) but did not
calibrate the decision rules themselves.

The clean-run outcome data, final analysis, and finalized paper prose will be deposited
separately in **DOI 2** (mint trigger = Pass-1 + Pass-2 + analysis frozen + paper §1-§8
finalized; see `osf_lock_manifest.md §3b`).

## OSF operational ordering

Per OSF help docs (https://help.osf.io/article/330-welcome-to-registrations
+ https://help.osf.io/article/626-simplifying-the-preregistration-process):

1. **OSF Registration submission = public timestamp witness**. The OSF
   registration metadata records the submission / registration timestamp
   as the **public timestamp witness** for the pre-registration. Per OSF
   docs, if administrator does not act, registration auto-approves at ~48h
   and the "registered" date is tied to the submission date; once public,
   registration content is immutable. This timestamp — NOT the later
   DOI-string assignment moment — is what anchors the preregistration
   timing claim.

2. **Bundled SHA-256 manifest provides content-level integrity**. The
   `MANIFEST_SHA256.txt` in the deposit bundle (auto-generated by the
   capture script `--bundle-regen` flag) lists SHA-256 hashes for every
   bundled file, enabling any third party to verify content-identical
   replication of the deposit at the registration moment.

3. **OSF admin approval → DOI string assignment**. After approval (manual
   or 48h auto), OSF assigns the DOI string `10.17605/OSF.IO/XXXXX`. The
   preregistration timing claim **does not depend on DOI-string availability**
   — it depends on the submission/registered timestamp recorded in OSF
   metadata at submission.

### Citation forms (canonical Fire-3 witness must exist before submission)

**Before DOI assignment (interim, 0-48h post-submission, but post-canonical-witness-capture)**:

```
OSF registration GUID osf.io/xxxxx, submitted <Fire-3-launch UTC>,
pre-canonical-outcome witness for Fire-3 (empirical canonical-pattern
zero-count probe at submission time per artifact_existence_check_doi1_canonical_
<fire3-UTC-TS>.txt, SHA-256 <fire3-witness-SHA>); DOI <pending OSF admin
approval, default auto-approve 48h>.
```

**After DOI assignment (final, post-approval)**:

```
OSF DOI 10.17605/OSF.IO/XXXXX, submitted <Fire-3-launch UTC>,
pre-canonical-outcome witness for Fire-3 (registration approved at OSF;
DOI-string assignment timestamp may differ from submission/registered
timestamp — preregistration timing claim relies on the latter, recorded
in OSF metadata); registered Git tag `preregistration-locked` at SHA
ef609a3 + post-retraction commits 59c60c4 + dd5335b + 5c8968a.
```

**Voided Fire-2-era witness** (`retracted/artifact_existence_check_doi1_20260518T135722Z_VOID_RETRACTION_ONLY.txt`,
SHA-256 `e0e591f5...`) is **not cited as empirical witness** in any DOI 1 citation
form — it appears in the bundle only as a retraction audit-trail artifact.

## Cross-link to DOI 2

**DOI 2 — Phase 1a reproducibility bundle**: `<to-be-assigned post Pass-1 + Pass-2 + analysis-frozen + paper-finalized>` # TBD-ALLOW: DOI 2 mints separately ~2-3 weeks post-fire per two-DOI doctrine; will explicitly `cited_by` this DOI 1.

DOI 2's README will cite and cross-link this DOI 1 (immutable forward reference; exact
OSF metadata field-name depends on platform support, but the README prose explicitly
references the DOI 1 string and OSF GUID). This DOI 1 README does **NOT** mention a
specific DOI 2 string because DOI 1 is locked pre-DOI-2 — bidirectional references would
be anachronistic and would defeat OSF's immutable-registration semantics.

## Bundle contents

Core preregistration substance frozen at Git SHA `ef609a3` (`preregistration-locked`
tag, substance-lock 2026-05-18T11:30Z). DOI_1_README + witness files + retraction-wave
doc updates were added in post-retraction commits `59c60c4` (doctrine restoration
B-1650~B-1655) + `e2c1782` + `dd5335b` (witness pattern bug retraction wave
B-1670~B-1679) + `5c8968a` (Stage 2 capture script + regression test B-1675).
See `osf_lock_manifest.md §2.1` for the full SHA-locked artifact table.

| File | Role | Git SHA |
|---|---|---|
| `preregistration.md` | 14 commit decisions + H1/H2/H3/H10 gating, status: substance-locked 2026-05-18 per §A2 14/16 audit cascade | post-retraction (current HEAD) |
| `osf_lock_manifest.md` | Lock manifest, §3a DOI 1 + §3b DOI 2 workflow split | post-retraction |
| `locked_versions.md` | B0 proxy + B1 HF `ebb281e...` + B2 HF `093f9f3...` + VWA submodule `ac33d2f...` + tree-hash chain `752caeb...` | `ef609a3` |
| `model_card.md` | Cross-baseline architecture + decoding + capability scope | `ef609a3` |
| `dataset_card.md` | VWA classifieds + reddit task counts + N/A exclusion protocol | `ef609a3` |
| `ethics_license_coi_statements.md` | Holistic AI industry COI + license attribution | `ef609a3` |
| `evaluator_change_protocol.md` | T0-T3 evaluator-change tier classification | `ef609a3` |
| `compute_cost_carbon_table.md` | A100 GPU-hour estimates per condition | `ef609a3` |
| `neurips_checklist.md` | Submission integrity checklist | `ef609a3` |
| `negative_results_registry.md` | Pre-outcome state of negative-results commitment | `ef609a3` |
| `release_redaction_checklist.md` | Public-release scope per §7 reproducibility | `ef609a3` |
| `topvenue_constraints.md` | Submission venue analysis (EMNLP / workshop) | `ef609a3` |
| `pre_rerun_audit_walkthrough_2026-05-18.md` | §A2 cascade closure operational walkthrough | `ef609a3` |
| `reeval_audit_protocol.md` | FP architecture canonical state post-§139.8 | `ef609a3` |
| `env_snapshot.json` | A100 substrate snapshot at Fire-3 epoch | **N/A for DOI 1** — pre-launch witness strategy (attempt #4) captures zero-state empirical witness instead of post-launch substrate snapshot; full A100 environment snapshot deferred to DOI 2 reproducibility bundle (post-Pass-1 + Pass-2 data landing) |
| `paper_drafts/section{1..8}_*.md` @ `ef609a3` | **Frozen pre-outcome state** — archive placeholder numbers per scope disclaimer above | `ef609a3` |
| `paper_drafts/paper.bib` @ `ef609a3` | Bibliography frozen pre-outcome | `ef609a3` |
| **`artifact_existence_check_doi1_canonical_20260518T211628Z.txt`** | **🟢 CANONICAL DOI 1 EMPIRICAL WITNESS** (captured 2026-05-18T21:16:28Z via `capture_doi1_witness.sh` pre-launch strategy; full-file SHA-256 `6056b905e25b0880e613eeb43a091f3281e107073736608755331185ffcadbe3` verifiable via MANIFEST_SHA256.txt) | git HEAD `590ad12` at capture |
| `artifact_existence_check_doi1_interim_20260518T144258Z.txt` | Audit-trail bridge — post-Fire-2-cleanup pre-Fire-3 scan (NOT the DOI 1 anchor) | `e2c1782` |
| `retracted/artifact_existence_check_doi1_20260518T135722Z_VOID_RETRACTION_ONLY.txt` | **🚫 VOIDED** — retained for retraction audit trail only; NOT used as DOI 1 anchor or empirical witness | `e2c1782` (VOID header in retraction commit) |
| `DOI_1_README.md` | This file (scope disclaimer + ordering doctrine + cross-link slot) | post-retraction |

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

1. **The pre-registration timestamp is the OSF submission/registered datetime** —
   verifiable on the OSF registration metadata page. The DOI-string assignment
   timestamp may lag 0-48h post-submission via OSF admin approval, but **the
   preregistration timing claim relies on the submission/registered timestamp,
   NOT DOI-string availability**. Bundled `MANIFEST_SHA256.txt` provides
   content-level integrity.
2. **Numbers in paper drafts are NOT Phase 1a evidence** — see scope
   disclaimer above. Phase 1a clean-run evidence is in DOI 2. The H1/H2(a)/H3/H10
   decision rules were locked **before** Phase 1a fire; archive placeholder
   numbers informed power analysis but did not calibrate the gates.
3. **The empirical pre-canonical-outcome claim is auditable via the canonical
   Fire-3 witness file** — see `artifact_existence_check_doi1_canonical_<fire3-UTC-TS>.txt`
   (file path filled in at Fire-3 launch) for the 3-tier zero check + SHA-256
   self-doc + Git HEAD + preregistration-locked tag SHA + canonical schema
   citation. **Do NOT cite** the voided Fire-2-era witness
   `retracted/artifact_existence_check_doi1_20260518T135722Z_VOID_RETRACTION_ONLY.txt` — that file is retained
   only for retraction audit-trail, marked with VOID header per B-1670 pattern
   bug retraction (see `master_bug_catalog.md ## /stress witness pattern bug
   retraction` for full context).
4. **The git substrate is verifiable** — checkout `git tag preregistration-locked`
   (= SHA `ef609a3`) to reproduce the substance-locked code state exactly. Post-
   retraction commits 59c60c4 (doctrine restoration) + e2c1782 + dd5335b
   (witness retraction wave) + 5c8968a (Stage 2 capture script) are linear
   children on `master`; `git tag retraction/osf-doi1-witness-59c60c4` is an
   annotated retraction tag pointing at the original buggy doctrine commit. All
   bundled files are tree-hash-chained per `preregistration.md §7` VWA submodule
   SBOM lock recipe.
5. **No force-push doctrine** — GitHub history preserves the Fire-2-era buggy
   commit `59c60c4` and its referenced witness file as immutable audit-trail
   evidence. The annotated retraction tag `retraction/osf-doi1-witness-59c60c4`
   marks this commit as superseded; the canonical Fire-3 witness in this DOI 1
   bundle is the definitive empirical anchor.
