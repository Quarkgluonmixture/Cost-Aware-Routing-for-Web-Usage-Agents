# OSF Deposit Package Manifest — Phase 1a Pass-1 launch event 2026-05-18

> **Purpose**: Pre-fire collection of artifacts ready for OSF DOI mint.
>
> **Updated 2026-05-18 ~14:45 UTC per /stress B-1670~B-1679 witness pattern bug
> retraction wave** (restores 2026-05-05 §F.1 + propagates B-1650~B-1655 two-DOI
> doctrine restoration): OSF DOI is now **two-DOI split** per `osf_lock_manifest.md §3a/§3b`:
>
> - **DOI 1 (pre-canonical-outcome-creation witness for Fire-3)**: minted at Fire-3
>   launch moment (5 min before any episode-summary creation), captures cryptographic
>   pre-outcome anchor for the canonical Phase 1a run. **NOT post-Pass-1-data-complete**
>   (pre-correction doctrine drift superseded per B-1650). Bundle = pre-run/*.md docs
>   + paper_drafts/section{1..8} @ ef609a3-or-fire3-HEAD + env_snapshot.json (A100,
>   fire-3 epoch) + DOI_1_README.md + artifact_existence_check_doi1_canonical_<fire3-UTC>.txt.
>
> - **DOI 2 (Phase 1a reproducibility bundle)**: minted post Pass-1 + Pass-2 +
>   analysis-scripts-frozen + paper §1-§8 finalized (~2-3 weeks post-fire-3). Bundle =
>   42 condition_summary_v2.json + episodes/*_summary_v2.json + episodes/*_steps_v2.jsonl
>   + results/phantom_paper/*.csv + figures + frozen analysis scripts + finalized paper
>   + DOI_2_README.md (mandatory `cited_by: DOI 1`).
>
> **Workflow timing**:
> - **Pre-fire-3 (this manifest)**: artifacts pre-collected + verified
> - **At Fire-3 launch + PID-alive**: 5-min canonical witness capture + git tag
>   `preregistration-doi1-witnessed-<DOI>` post-DOI-assignment; commit + push
> - **Post-Pass-1+Pass-2+analysis-frozen+paper-final**: DOI 2 deposit uploaded
>   to OSF + DOI 2 minted + backfills `osf_lock_manifest.md §2` `<TBD>` cells
>
> **Status**: 🟢 Pre-fire bundle ready 2026-05-18 (subject to fire-time
> Git SHA + paper_drafts snapshot copy at fire event).

---

## Required artifacts for OSF deposit (per `osf_lock_manifest.md §3` Step 7)

### 1. Preregistration document + companion lock docs

| Artifact | Path | Status | Notes |
|---|---|---|---|
| `preregistration.md` (locked) | `docs/checkpoints/pre_run/preregistration.md` | ✅ substance-locked 2026-05-18T11:30Z (status: locked + prereg_registered_git_sha 88521b9e historical, current canonical anchor commit c741c3a/72b93c9 + tag preregistration-locked-q3a; no further "flip" needed at Fire-3 — only canonical empirical witness capture per B-1700~B-1709 Q3=A doctrine alignment) | Contains §1-§7 + Appendix A decision log + §6 §(a) 14 commit decisions |
| `osf_lock_manifest.md` | `docs/checkpoints/pre_run/osf_lock_manifest.md` | ✅ §1-§2 + §3 8-step + §4 post-lock change discipline | Backfill `<TBD>` cells post-mint with Git SHA + DOI |
| `locked_versions.md` | `docs/checkpoints/pre_run/locked_versions.md` | ✅ VWA SHA **`ac33d2f...`** (11th-commit fire-event lock 2026-05-18 — `f883a11` untrack test_shopping.json + `ac33d2f` complement .gitignore typo correction + generation_manifest.json; same logical fix split across 2 commits due to staging oversight; zero substantive content change; supersedes pre-fix `2f9b0b4` at A2.9 prep stage) + B1 HF `ebb281e...` + **B2 HF `093f9f3...` locked 2026-05-18 B-1603** + Playwright 1.58.0 + Chromium revision 1208 | All pins current |
| `model_card.md` | `docs/checkpoints/pre_run/model_card.md` | ✅ B0/B1/B2 cards aligned + B2 HF SHA filled per B-1603 | Mitchell et al. 2019 format |
| `dataset_card.md` | `docs/checkpoints/pre_run/dataset_card.md` | ✅ 3-baseline scope + cls/red/shop task pool hashes | Per audit A12 |
| `evaluator_change_protocol.md` | `docs/checkpoints/pre_run/evaluator_change_protocol.md` | ✅ Protocol A 4-tier T0/T1/T2/T3 classification | — |
| `reeval_audit_protocol.md` | `docs/checkpoints/pre_run/reeval_audit_protocol.md` | ✅ Protocol B `rederive_metadata` audit trail; B-133 §139.8 banner | walkthrough 2026-05-18 `reeval_audit_protocol_walkthrough_2026-05-18.md` |
| `pre_rerun_audit.md` | `docs/checkpoints/pre_run/pre_rerun_audit.md` | ✅ index-only (slimmed 2026-05-14) | walkthrough 2026-05-18 `pre_rerun_audit_walkthrough_2026-05-18.md` |
| `negative_results_registry.md` | `docs/checkpoints/pre_run/negative_results_registry.md` | ✅ 12+ retracted framings + C1 confirmed framing (post-B-1502 42-cond sweep) | — |
| `ethics_license_coi_statements.md` | `docs/checkpoints/pre_run/ethics_license_coi_statements.md` | ✅ MIT cite + release license matrix + Gemma Terms of Use disclosure (B-1501 A2.9) + LLM Use Disclosure NeurIPS 2025 Q16 (B-1507 A2.9) | — |
| `neurips_checklist.md` | `docs/checkpoints/pre_run/neurips_checklist.md` | ✅ NEW Q1-Q16 NeurIPS 2025 paper-time format (B-1506 A2.9 P0-1-AC*) | — |
| `compute_cost_carbon_table.md` | `docs/checkpoints/pre_run/compute_cost_carbon_table.md` | ✅ NEW 15-column skeleton (B-1508 A2.9) | Numerical fill POST-Phase-1a-fire |
| `release_redaction_checklist.md` | `docs/checkpoints/pre_run/release_redaction_checklist.md` | ✅ `make pre-release-check` wired (B-1512 A2.9) + sign-off row 2026-05-18 PASS | — |
| `topvenue_constraints.md` | `docs/checkpoints/pre_run/topvenue_constraints.md` | ✅ 78-constraint scoreboard | Supplanted at paper-submission by `neurips_checklist.md` Q1-Q16 |
| **Walkthrough artifacts (NEW pre-fire 2026-05-18)** | `docs/checkpoints/pre_run/pre_rerun_audit_walkthrough_2026-05-18.md` + `reeval_audit_protocol_walkthrough_2026-05-18.md` | ✅ this prep session | Walkthrough verification artifacts; OSF audit-trail completeness |

### 2. Substrate / paper-grade code lock (per Step 5)

| Artifact | Path | Capture event |
|---|---|---|
| `paper_drafts_locked/` | `docs/checkpoints/paper_drafts_locked/` | `cp -r paper_drafts paper_drafts_locked` AT git tag event (pre-fire snapshot of paper §1-§8 prose) |
| Git tag `preregistration-locked` | `git tag -a preregistration-locked -m "OSF DOI mint <date> Phase 1a Pass-1 launch fire-event Git SHA <SHA>"` | At fire event; commit Git SHA captured |
| `git push origin preregistration-locked` | — | Immediately after tag |
| Submodule pin verification | VWA `external/visualwebarena` HEAD `ac33d2fcd9cec2fcbeddd56d0fa3da58b4c7e927` + tree-hash chain `752caebdc6bd84761b2f308331f21241a9b4a28de65b46ff0007ef27d8c72778` | Captured in `env_snapshot_a100_lock.json` (Step 3) |

### 3. Provenance snapshots (per Step 3 + Step 4)

| Snapshot | Host | Path | Status |
|---|---|---|---|
| `env_snapshot_a100_lock.json` ⭐ paper-1 canonical | A100 (`a100-jiaming-test`) | `results/provenance/env_a100_lock.json` (commit at fire event) | A100 SSH probe verified 2026-05-18 `a100_pre_launch_2026-05-18_071612.json`; will re-snapshot at fire event with `--strict` once HF_TOKEN set for Gemma SHA-verify |
| `env_snapshot_dgx_archive.json` | DGX Spark `spark-9ea3` | `results/provenance/env_dgx_archive.json` | Archive reference only post-2026-05-15 host migration; can be deferred per `preregistration.md §7` Infrastructure migration note |
| `env_snapshot_myriad_audit.json` | Myriad cross-arch | `results/provenance/env_myriad_audit.json` | Optional; F6 audit cross-arch numerical determinism (paper-2 scope per advisor 2026-05-14) |
| `vwa_snapshot_a100_lock.json` | A100 (VWA Docker self-hosted) | `results/provenance/vwa_a100_lock.json` | `scripts/provenance/snapshot_vwa.sh` runs at fire event; SBOM 4-match verified my SSH 2026-05-18 |

### 4. Run manifest + cell registry (per `osf_lock_manifest.md §2.1`)

| Artifact | Path | Status |
|---|---|---|
| `run_manifest.yaml` (paper-grade cells registry) | `results/phantom_paper/run_manifest.yaml` | Pre-fire 0 paper-grade cells; populated incrementally as Pass-1 baseline fires; OSF DOI mint event check = all 6 cells × Pass-1 marked `grade: paper-grade` |

### 5. Reusable patch artifacts (per `osf_lock_manifest.md §2.5` B-1604)

| Patch family | Code path | Commit SHA |
|---|---|---|
| B-440 + B-448 GRL walk-up click | `external/visualwebarena/browser_env/actions.py` + `p79/envs/vwa_wrapper.py` | TBD at OSF mint event (relevant for Track A workshop; paper-1 §3.5.2 GRL evidence subset) |
| B-91 LLM-judge polarity fix | VWA submodule `external/visualwebarena/p79-patches` branch `eb5cbd8` + `evaluation_harness/helper_functions.py:612-613` | Embedded in submodule HEAD `ac33d2f` (11-commit chain ending at fire-event lock 2026-05-18; `f0c835b` original B-91 commit preserved as 5th in chain; `f883a11` + `ac33d2f` 10th + 11th .gitignore symmetry restore); tree-hash chain `752caebd...` matches lock |
| B-535 N/A task-load exclusion | `p79/experiment/tasks.py` + `configs/exp_v2_base.yaml` | TBD at OSF mint event (relevant for Track B workshop; paper-1 §8.2 disclosure subset) |

---

## Pre-fire OSF deposit ready-state

**Pre-fire NOW (deterministic, DGX-side; substance-lock landed in advance per Q3=A doctrine)**:
- ✅ All `pre_run/` docs ready + walkthroughs written + substance-lock landed at commit c741c3a/72b93c9 + tag `preregistration-locked-q3a` (2026-05-18T11:30Z)
- ✅ `paper_drafts/` section 1-8 + paper.bib current
- ✅ A100 pre-fire substrate verified: B-1427 probe (B0 proxy production path) PASS + B-1428 snapshot (env + VWA SBOM match_lock=True) PASS + NLTK punkt word_tokenize PASS + mechanism §5 archive restored + smoke residual archived
- ⏭ `env_snapshot_a100_lock.json` + `vwa_a100_lock.json` at Fire-3 PID-alive (single capture window, NOT pre-fire snapshot)

**Fire-3 launch + DOI 1 canonical witness capture (single atomic 5-min window per B-1650 doctrine restoration; OSF submission immediately follows, NOT after Pass-1 data complete)**:

1. **Launch on A100**: `ssh condense-a100 'cd ~/workspace/p79 && RESET_BEFORE=1 setsid nohup bash scripts/queues/queue_phase1_paper_grade.sh launch > logs/phase1a_pass1_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &'`. Capture Fire-3 PID.

2. **Within 5 min of PID-alive + BEFORE first `<site>_task_<N>_summary_v2.json` creation on disk**: capture canonical empirical witness via `bash scripts/maintenance/capture_doi1_witness.sh` on A100 (B-1675 regression-tested 9-case script). Output = `docs/checkpoints/pre_run/osf_deposit_DOI1_<NEW-TS>/pre_run/artifact_existence_check_doi1_canonical_<fire3-UTC-TS>.txt` with all canonical-pattern counts (summary_v2 / steps_v2 / condition_summary_v2 / aggregate CSV) = 0 + Git HEAD SHA + tag SHA + capture UTC.

3. **Verify counts all zero** (sanity check the witness script output; non-zero count means we missed the 5-min window and the witness anchor is invalid).

4. **At same A100 capture moment**: `python scripts/provenance/snapshot_env.py --strict results/provenance/env_a100_fire3_lock.json && bash scripts/provenance/snapshot_vwa.sh results/provenance/vwa_a100_fire3_lock.json` (single launch snapshot, NOT separate from Fire-3 PID).

5. **Pull artifacts from A100 → DGX**, regenerate canonical `DOI_1_README.md` filling all TBD fields (file name + SHA-256 + captured UTC + Fire-3 PID + 4 canonical counts + known-positive probe), regenerate deposit `MANIFEST_SHA256.txt`.

6. **Commit + push** the canonical witness + filled README + new manifest. Tag `preregistration-doi1-witnessed-<fire3-UTC-TS>` on the witness commit.

7. **OSF DOI 1 submission (you, ~15 min user-side via OSF UI; happens within hours of Fire-3 launch, NOT after Pass-1 data complete)**:
   - Visit `https://osf.io/registries/`
   - Create new OSF preregistration linked to GitHub tag URL `https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents/tree/preregistration-doi1-witnessed-<fire3-UTC-TS>`
   - Upload `docs/checkpoints/pre_run/osf_deposit_DOI1_<NEW-TS>/` bundle in full (paper_drafts/ + pre_run/ + MANIFEST_SHA256.txt + retracted/...VOID_RETRACTION_ONLY.txt for forensic transparency)
   - **OSF submission timestamp = the cryptographic anchor** for DOI 1 (DOI string assignment auto-occurs 0-48h later per help.osf.io/article/330; submission timestamp is what reviewers cite as pre-canonical-outcome-creation evidence, NOT the DOI assignment timestamp)
   - Backfill `osf_lock_manifest.md §2` `<TBD>` cells with the OSF GUID + assigned DOI string post-approval + record in `paper_planning.md §19 decision log`

8. Setup watchdog + done-monitor for Pass-1 + Pass-2 fire (~1-2 weeks Pass-1 + 3-5 days Pass-2 wallclock; this happens AFTER OSF DOI 1 is already submitted, NOT before).

**Post-Phase-1a-Pass-1+Pass-2+analysis-frozen (DOI 2 reproducibility bundle, ~2-3 weeks post Fire-3)**:
- DOI 2 has its own deposit dir (NEW timestamp) + DOI_2_README explicit `cited_by DOI 1`
- DOI 2 bundle includes: 42 condition_summary_v2.json + episode summaries + steps JSONL + aggregate CSV + figures + analysis scripts + finalized paper §1-§8 + paper.bib
- Mint trigger = Pass-1 + Pass-2 complete + analysis scripts frozen + paper finalized; per `osf_lock_manifest.md §3b` 8-step workflow

**(Optional)** advisor email send (per advisor email draft `advisor_sync/advisor_email_draft_2026-05-18.md`) — pre-fire / at-fire / post-fire all paper-grade equivalent per B-1570 doctrine; OSF DOI 1 supersedes advisor email as cryptographic external witness per B-1650 doctrine restoration.

---

## Cross-references

- `osf_lock_manifest.md §3` 8-step DOI workflow (B-1570 updated header 2026-05-18)
- `preregistration.md §6 §(a)` 14 commit decisions (substance locked at master HEAD)
- `pre_rerun_audit_walkthrough_2026-05-18.md` (11-layer verification)
- `reeval_audit_protocol_walkthrough_2026-05-18.md` (Protocol B walkthrough)
- `advisor_sync/advisor_email_draft_2026-05-18.md` (optional witness email)
- 实验笔记 §200-§222 cross-AI audit chronicle
- master_bug_catalog ## A2.x sections
