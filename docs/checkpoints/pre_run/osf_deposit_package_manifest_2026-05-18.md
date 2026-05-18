# OSF Deposit Package Manifest — Phase 1a Pass-1 launch event 2026-05-18

> **Purpose**: Pre-fire collection of artifacts ready for OSF DOI mint
> (Step 7 of `osf_lock_manifest.md §3` 8-step workflow). Per B-1570
> doctrine shift 2026-05-18, OSF DOI mint is POST-Phase-1a-fire-data-complete;
> this manifest pre-stages the artifact bundle so the post-fire mint
> event is a 30-minute "upload + click mint" operation, not a scramble.
>
> **Workflow timing**:
> - **Pre-fire (this manifest)**: artifacts pre-collected + verified
> - **At Phase 1a Pass-1 launch**: git tag `preregistration-locked` created;
>   fire-time Git SHA captured into this manifest
> - **Post-Phase-1a-Pass-1-data-complete**: user uploads bundle to OSF +
>   mints DOI + backfills `osf_lock_manifest.md §2` `<TBD>` cells
>
> **Status**: 🟢 Pre-fire bundle ready 2026-05-18 (subject to fire-time
> Git SHA + paper_drafts snapshot copy at fire event).

---

## Required artifacts for OSF deposit (per `osf_lock_manifest.md §3` Step 7)

### 1. Preregistration document + companion lock docs

| Artifact | Path | Status | Notes |
|---|---|---|---|
| `preregistration.md` (locked) | `docs/checkpoints/pre_run/preregistration.md` | ⏸ frontmatter `status: draft` — flip to `locked` at fire event | Contains §1-§7 + Appendix A decision log + §6 §(a) 14 commit decisions |
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

**Pre-fire NOW (deterministic, my DGX-side)**:
- ✅ All `pre_run/` docs ready + walkthroughs written
- ✅ `paper_drafts/` section 1-8 + paper.bib current
- ⏭ `paper_drafts_locked/` snapshot at fire event
- ⏭ Git tag `preregistration-locked` at fire event
- ⏭ `env_snapshot_a100_lock.json` + `vwa_a100_lock.json` at fire event (immediate pre-`launch` commit)

**Fire event (~1 min total deterministic)**:
1. `cd /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents`
2. `cp -r docs/checkpoints/paper_drafts docs/checkpoints/paper_drafts_locked`
3. `ssh condense-a100 'cd ~/workspace/p79 && python scripts/provenance/snapshot_env.py results/provenance/env_a100_lock.json && bash scripts/provenance/snapshot_vwa.sh results/provenance/vwa_a100_lock.json'`
4. Pull artifacts back / commit on DGX side
5. Flip `preregistration.md` frontmatter `status: draft → locked` + fill `registered_at` + `registered_git_sha`
6. `git commit + git tag -a preregistration-locked -m "Phase 1a Pass-1 launch <date+sha>" + git push --tags`
7. **Fire**: `ssh condense-a100 'cd ~/workspace/p79 && setsid nohup bash scripts/queues/queue_phase1_paper_grade.sh launch > logs/phase1a_pass1_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &'`
8. Setup watchdog + done-monitor

**Post-Phase-1a-Pass-1-data-complete (you, ~30 min user-side via OSF UI)**:
1. Visit `https://osf.io/registries/`
2. Create new OSF preregistration linked to GitHub tag URL `https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents/tree/preregistration-locked`
3. Upload `docs/checkpoints/pre_run/*.md` bundle (all 15 docs listed in §1 above) + `paper_drafts_locked/section1-8 + paper.bib` + `results/provenance/env_a100_lock.json` + `vwa_a100_lock.json`
4. Mint DOI (OSF assigns ~1 min after click)
5. Backfill `osf_lock_manifest.md §2` `<TBD>` cells with the assigned DOI + record in `paper_planning.md §19 decision log`
6. Commit + push as final lock proof

**(Optional)** advisor email send (per advisor email draft `advisor_sync/advisor_email_draft_2026-05-18.md`) — pre-fire / at-fire / post-fire all paper-grade equivalent per B-1570 doctrine.

---

## Cross-references

- `osf_lock_manifest.md §3` 8-step DOI workflow (B-1570 updated header 2026-05-18)
- `preregistration.md §6 §(a)` 14 commit decisions (substance locked at master HEAD)
- `pre_rerun_audit_walkthrough_2026-05-18.md` (11-layer verification)
- `reeval_audit_protocol_walkthrough_2026-05-18.md` (Protocol B walkthrough)
- `advisor_sync/advisor_email_draft_2026-05-18.md` (optional witness email)
- 实验笔记 §200-§222 cross-AI audit chronicle
- master_bug_catalog ## A2.x sections
