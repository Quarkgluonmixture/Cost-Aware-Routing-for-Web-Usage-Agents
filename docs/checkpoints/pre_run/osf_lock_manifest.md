# OSF DOI Lock Manifest — Paper §3 / Appendix D

**Purpose**: Codify the artefacts whose SHA-256 / git ref get frozen at the
moment of OSF preregistration DOI minting. Substance-lock decisions
(**H1 FE superiority δ=1.0pp / H3 axis-1+axis-2 gates / H10 two-layer operational deployment gate / K-of-N transparency-only (no thresholds) / TOST retired per /stress A2.3c B-1051**)
are RESOLVED via §A2 14/16 audit cascade closure 2026-05-18 (B-1265 /stress A2.6a P0-6-B* 2026-05-18 + B-1550 /stress A2.8 P0-2-AB* 2026-05-18 operational-gate reframing — supersedes stale "K_h1 / K_h3 / TOST δ thresholds" phrasing + stale "advisor email reply gates lock" phrasing per B-1570 /stress A2.8 followup doctrine-shift propagation 2026-05-18; original pre-2026-05-14 framework had advisor email as primary lock gate but 2026-05-14 sync 收口 reclassified student scope to "experiment execution; prose advisor-side", and §A2 audit cascade self-decision substitutes advisor pre-lock email per 实验笔记 §209 "Advisor sync triage: 0 项真需要 advisor pre-sync"). This checklist
drives the 8-step DOI workflow (笔记 §110 + `paper_planning.md §19` decision log + `osf_lock_manifest.md §3`; ADVISOR_SYNC.md retired 2026-05-15, commit `f64bc9d`, replaced by `_status/issues/issue_advisor_sync_2026-05-14.md` frontmatter + Bases view).

**Status**: 🟢 **Substance-locked since 2026-05-18T11:30Z** at canonical anchor (see below) + ✅ **DOI 1 MINTED 2026-05-18T23:10:06Z UTC** at `10.17605/OSF.IO/9QCWU` (Registration GUID `9qcwu`, parent project `kv9sf`; archive auto-completed instantly). Canonical empirical witness captured pre-launch at 2026-05-18T21:16:28Z per attempt #4 strategy (B-1751-fu) — full-file SHA `6056b905...` / content-only SHA `011fa4c0...`. **DOI 2 still pending** (~2-3 weeks post Phase 1a fire start 2026-05-18T13:28:06Z); remaining TBD fields below (Code+manifest SHAs §2.1 + Patch SHAs §2.5) are post-DOI-mint backfill candidates for a separate follow-up commit (each row requires per-artifact `git log` last-touch lookup; not in DOI 1 mint scope).

**Canonical DOI 1 substance-lock anchor (single-source-of-truth for all OSF-facing docs)**:
```
Current canonical:
  Git SHA       : 72b93c939526fb0b9a733013852bd90136d601aa   (master HEAD post Q3=A doctrine fix + B-### renumber)
  Git tag       : preregistration-locked-q3a                 (annotated, at 72b93c9)

Legacy anchors retained for audit trail ONLY (NOT for citation by OSF DOI 1):
  Git SHA       : ef609a3863adc9b3698789b96a1ee9f709e1c832   (pre-doctrine-fix substance lock; superseded by Q3=A wave)
  Git tag       : preregistration-locked                     (legacy, at ef609a3; retained on origin for git-history continuity)
  Frontmatter   : 88521b9e254955ab156dc8115da826f171ef5990   (earlier prereg `prereg_registered_git_sha` field value pre-doctrine-fix)
```

**Two-DOI split (locked 2026-05-18 ~14:00 UTC per /stress doctrine restoration B-1650~B-1655 — restores original 2026-05-05 advisor sync §F.1 outcome decision "学生 lean DOI 时间戳 < data unblinding 时间戳", which had drifted into stale "post-Pass-1-data-complete" single-DOI framing between B-1570 doctrine shift (2026-05-18 ~09:30 UTC, advisor-email-retire) and this correction)**:

- **DOI 1 — Pre-canonical-outcome-creation witness for Fire-3** (mint blocker quintuple, status post Q3=A doctrine fix + A100 pre-fire verification 2026-05-18 evening):
  - ✅ B2 SHA `093f9f388b31de276ce2de164bdc2081324b9767` locked + A100-side verified (`probe_b0_production_path.py` GATE PASS + `snapshot_env.py --strict` vwa.match_lock=True; phase1_plan §B1 items 7+8 = B-1427 + B-1428).
  - ✅ preregistration substance-locked 2026-05-18T11:30Z at canonical anchor above (current `72b93c9` + tag `preregistration-locked-q3a`; legacy `ef609a3` + tag `preregistration-locked` retained for audit trail only per Q3=A doctrine wave B-1700~B-1709, renumbered to B-1750~B-1759 at 72b93c9).
  - ⚠️ **Fire-2-era empirical witness RETRACTED** 2026-05-18 ~14:45 UTC per /stress B-1670~B-1679 pattern bug catch (`*_summary.json` should be `*_summary_v2.json` per canonical schema; void file relocated to `retracted/artifact_existence_check_doi1_20260518T135722Z_VOID_RETRACTION_ONLY.txt` subdir per Q3=A cleanup — not co-located with canonical witness chain to avoid path-level ambiguity).
  - ✅ interim corrected scan landed (`artifact_existence_check_doi1_interim_20260518T144258Z.txt` at root level, SHA-256 `7563f0d55b651b604746ef0498fba3439ad7d7e130af97f0adda55e2bc7f1bf8`, post-fire-2-cleanup pre-fire-3, all-zero canonical-pattern counts).
  - ✅ NLTK punkt evaluator substrate verified on A100 fire host (`word_tokenize` end-to-end PASS 2026-05-18 evening; B-486 Fire-2 RC resolved).
  - ⏳ canonical DOI 1 witness (`artifact_existence_check_doi1_canonical_<fire3-UTC-TS>.txt`) pending Fire-3 PID-alive capture via `scripts/maintenance/capture_doi1_witness.sh` (B-1675 regression-tested 9-case script).
  - → **DOI 1 MINTABLE AT FIRE-3 LAUNCH** (5-min capture window post-PID-alive, pre-first-episode-summary). See §3a for 8-step workflow.

- **DOI 2 — Phase 1a reproducibility bundle** (mint blocker quadruple: Pass-1 baseline 36 conditions complete + Pass-2 learned router 6 conditions complete + analysis scripts frozen + paper §1-§8 finalized).
  - ⏳ pending Pass-1 + Pass-2 + analysis-frozen + paper-final (~2-3 weeks post Phase 1a fire start 2026-05-18T13:28:06Z).
  - DOI 2 README will explicitly cite_by DOI 1 (immutable forward reference; bidirectional reference defeats OSF immutable-registration semantics). See §3b for 8-step workflow.

**Advisor email reply RECLASSIFIED 2026-05-18 per /stress A2.8 followup B-1570 as optional post-fire collateral** (was hard gate pre-2026-05-14 sync 收口; current doctrine: §A2 14/16 audit cascade substantively closes all 14 commit decisions, advisor batch sign-off retained as optional reproducibility audit-trail completeness for OSF DOI mint). OSF DOI 1 supersedes advisor email as cryptographic external witness (public registry + immutable timestamp + content hash + machine-verifiable, strict superset over private + human-attested + non-cryptographic email per OSF help.osf.io/article/330).

---

## §1 Pre-lock checklist (everything must be done before OSF DOI mint)

- [x] **Substance-lock RESOLVED via §A2 14/16 audit cascade** (B-1570 /stress A2.8 followup 2026-05-18 doctrine-shift propagation — replaces pre-2026-05-14 "Advisor email reply received" hard-gate language; advisor batch sign-off now optional post-fire collateral): all 14 commit decisions substance-locked via Git SHA + audit-trail refs (master_bug_catalog ## A2.1~## A2.9 + 实验笔记 §200-§221). Specifically: **H1 FE superiority δ=1.0pp + bootstrap percentile primary gate** ✅ /stress A2.3a B-941~B-958 + A2.4a B-1009 + A2.3d B-1301 / **H3 axis-1+axis-2 FE gates** ✅ A2.3c B-1057 / **H10 two-layer operational deployment gate** (cell-level 95% paired-bootstrap + grid-level ≥5/6 fixed-cell operational robustness criterion, NOT binomial significance test) ✅ A2.8 B-1550 / **K-of-N transparency-only retired-as-gate** (fake-precision ⌈0.75×6⌉=⌈0.67×6⌉=5 argument) ✅ §2.4 + §H1 / **TOST framework retired** ✅ A2.3a B-957 + A2.3c B-1051 / **Phase 1a 42-condition 6-cell scope** (36 Pass-1 baseline + 6 Pass-2 learned router) ✅ A2.6a B-1264 sweep. Advisor email batch sign-off retained as optional post-fire collateral per §6 §(a) reframe 2026-05-18.
- [ ] `preregistration.md` final text edit committed + pushed (incl. 2026-05-13
      codex stress audit propagation: K-of-N transparency-only + 24/4 scope +
      drop-one H1 formula + outcome-independent smoke gate + Appendix A 2026-05-13)
- [ ] `run_manifest.yaml` archived rows verified (Phase 1a 42-condition scope = 36 baseline (2 sites × 3 models × 6 modes) + 6 router (2 sites × 3 models × 1 learned router/cell);
      all `grade=archived` for pre-fix cells, `grade=paper-grade` for Phase 1a post-fix rerun cells on A100 self-host;
      Phase 1b shop deferred rows separately tagged — B-132 cleanup 2026-05-15 removed dangling "× 2 models × 6 modes" residual from Batch 4 edit)
- [ ] All paper draft sections section1-8 + paper.bib (57 entries) snapshot to
      `docs/checkpoints/paper_drafts_locked/` directory (immutable copy)
- [ ] `env_snapshot.json` of latest run on each machine (DGX, A100, Myriad if
      used) committed under `results/provenance/env_lock_<hostname>.json`
- [ ] `vwa_snapshot_<host>.json` for any VWA-using cells committed
- [ ] No untracked / uncommitted files in repo (clean `git status`)
- [ ] Repo pushed to GitHub master (DOI cites GitHub commit URL)

---

## §2 Locked artefacts — fields populate at lock moment

### 2.1 Code + manifest SHAs

| Artefact | Path | Git ref @ lock | Captured |
|---|---|---|---|
| Repository HEAD | `master` branch | `<TBD>` | TBD |
| Pre-registration text | `docs/checkpoints/pre_run/preregistration.md` | `<TBD commit-SHA>` | TBD |
| Run manifest YAML | `results/phantom_paper/run_manifest.yaml` | `<TBD commit-SHA>` | TBD |
| Paper drafts (locked snapshot) | `docs/checkpoints/paper_drafts_locked/` | `<TBD commit-SHA>` | TBD |
| Bibliography (57 entries) | `docs/checkpoints/paper_drafts/paper.bib` | `<TBD commit-SHA>` | TBD |
| Power analysis script (B-954 /stress A2.3a P1-7-B 2026-05-17) | `scripts/analysis/power_analysis.py` | `<TBD commit-SHA — after A2.3a fix landed>` | TBD |
| Power analysis output (B-954) | `docs/analysis/cross_sites/power_analysis.md` | `<TBD commit-SHA>` | TBD; generated via `.venv/bin/python3 scripts/analysis/power_analysis.py --baseline-sr 0.10 --output docs/analysis/cross_sites/power_analysis.md` (one-sided z_α=1.645 per prereg §2.5 default) |
| FE-pool empirical SE source (B-954) | `results/phantom_paper/meta_phantom_lift.csv` (`4psom_vs_3` row: `theta_fe=2.336`, `se_fe=0.529`, k_cells=3) | `<regenerable via aggregate_phantom_meta.py; gitignored but reproducible from condition_summary_v2.json>` | empirical SE_FE = 0.529pp at k=3 archive ground truth for power section §2.4 |
| **VWA submodule HEAD** | `external/visualwebarena` (branch `p79-patches`) | `1c3a615308fd9f17c73a9d33a96cf29ec6807d48` | re-locked 2026-05-17 (/stress A1.25 GRL Chunks 1+4 — supersedes 2026-05-16 A1.18 lock at `eb5cbd8`) |
| **VWA upstream base** | `external/visualwebarena` (upstream `main`) | `89f5af29305c3d1e9f97ce4421462060a70c9a03` | locked 2026-05-16 |
| **VWA patch-bundle diff SHA-256** | `git diff 89f5af2..HEAD` over submodule | `f1315dc49a33c4b5e8d7d3958974d26f4e6ad330b15b8ce01a6eb8b80a958b1a` | re-locked 2026-05-17; 8 commits enumerated in `locked_versions.md` + paper §4.X.11 disclosure table (post A1.25 GRL Chunks 1+4) |
| ~~Mechanistic 24+15 candidates~~ | RETIRED 2026-05-15 (B-124 phantom-limb purge per gemini Mode C P1-5) — mechanism §5 deferred to paper-2 per advisor 2026-05-14; this lock-artifact moves to paper-2 OSF manifest, NOT included in paper-1 DOI | — | retired |

### 2.2 Hypothesis thresholds (advisor email confirmed) — REVISED 2026-05-13

| Threshold | Pre-reg value | Advisor confirmed? | Notes |
|---|---|---|---|
| H1 PRIMARY gate (P-SoM drop-one oracle ceiling lift) | **Fixed-effects inverse-variance pooled** average θ_FE over the **6 *planned* (site, model) cells** + one-sided superiority test (H0: θ_FE ≤ +1.0pp vs H1: θ_FE > +1.0pp) rejected at α=0.05 (single test m=1; prior H1(i) meta-≠-0 + magnitude folded in per decision "3A" 2026-05-14) | ⏳ pending | Drop-one = oracle SR over {6 modes} − oracle SR over {5 modes drop P-SoM} per task, paired bootstrap per cell, pooled via FE inverse-variance weighting over 6 cells (decision "3A" 2026-05-14 — NOT DerSimonian-Laird; the cells are the design not a population, so no τ²). DerSimonian-Laird retired 2026-05-14; cell count expanded 4 → 6 per B2 addition 2026-05-14. K-of-N retired as gate 2026-05-14 (transparency-only count). See preregistration.md §2 + §4 + Appendix A. |
| H3 PRIMARY gate axis-1 (P-text \ P-SoM) | **Fixed-effects** pooled axis-1 meta over **6 cells**, FE CI excludes 0 (one-sided, α=0.05, m=1 within axis-1 sub-family) | ⏳ pending | Per-cell bootstrap CI on unique-task count, then FE pooled meta over 6 cells (decision "3A" 2026-05-14, NOT DerSimonian-Laird). Cell count expanded 4 → 6 per B2 addition 2026-05-14. |
| H3 PRIMARY gate axis-2 (P-prompt \ P-SoM) | Same as axis-1 | ⏳ pending | Requires P-prompt mode (re-included 2026-05-13) |
| K-of-N transparency count (NOT a gate, NO threshold) | Report n of 6 cells with per-cell CI > 0 + n individually Holm-sig, for H1 + each H3 axis (decision "3A" 2026-05-14: K_h1=0.75 / K_h3=0.67 ratios retired — at both N=4 and N=6 the ratios are indistinguishable, ⌈0.75×6⌉=⌈0.67×6⌉=5; fake precision) | ⏳ pending | Reclassified gate → transparency 2026-05-13 (power analysis dysfunction at <7pp effects); cell count expanded 4 → 6 per B2 addition 2026-05-14. Reported alongside pooled FE meta as per-cell consistency check; no decision rule. |
| ~~K_h3 transparency ratio~~ (RETIRED — folded into K-of-N transparency count row above) | — | — | Retired 2026-05-14 (same fake-precision argument as K_h1 at both N=4 and N=6) |
| H1 superiority threshold δ (SR-margin) | 1.0pp | ⏳ pending | SR percentage-point margin for H1 one-sided FE superiority test (primary; m=1 single test per §3); distinct from H2(a) cost ±10% relative margin. **Updated 2026-05-18 per /stress A2.6a B-1265 P0-6-B*** — supersedes stale "H1(ii) ... — also TOST informational margin" phrasing that referenced retired TOST framework per `preregistration.md §2.4` TOST retirement 2026-05-17 (B-957 + B-1051 sibling-propagation sweep). |
| Cell scope (Phase 1a operational) | **42 conditions** = **Pass-1 baseline 36** (2 sites × 3 models × 6 modes) + **Pass-2 learned router 6** (2 sites × 3 models × 1 router cond/cell, `obs_mode="learned"` sentinel) | ⏳ pending | A1.21 P1-5 fix 2026-05-17 (B-533, codex F10): aligned with prereg §4 L438 "42 conditions" scope expansion 2026-05-16 (B-264+B-267 /stress A1.7 cross-AI cycle for §2 H10 Pass-2 router parity with gating-hypothesis registration). Replaces prior "36 operational" entry which was Pass-1-only and mismatched preregistration since 2026-05-16 |
| Cell scope (Phase 1a statistical) | **6 cells** = (site, model) tuples: (cls, B0), (cls, B1), (cls, B2), (red, B0), (red, B1), (red, B2) | ⏳ pending | One drop-one number per cell; pooled FE meta input (decision "3A" 2026-05-14, FE not DL); k=4 → k=6 per B2 addition 2026-05-14 |
| Cell scope (Phase 1b deferred) | **+21 conditions** = shop × {B0, B1, B2} × (6 baseline + 1 learned router). N_cells statistical becomes 9 (= 3 sites × 3 models) when Phase 1b lands. | ⏳ pending | Main-paper expansion lever; not part of Phase 1a workshop submission. **Updated 2026-05-18 per /stress A2.6a B-1263 P0-4-B*** — supersedes stale "+12 conditions = shop × B0+B1 × 6 modes" phrasing (which omitted B2 and Pass-2 learned router); consolidates with `preregistration.md §4 N_conditions Phase 1b row + Cell inclusion Phase 1b row` + `phase1_plan.md:42` to single +21 source. |

### 2.3 Environment fingerprints (per-machine snapshots)

| Machine | env_snapshot.json | vwa_snapshot.json | Locked at |
|---|---|---|---|
| **A100 `a100-jiaming-test` (paper-1 CANONICAL Phase 1a + Phase 1b post-2026-05-15)** | `results/provenance/env_lock_a100.json` | `results/provenance/vwa_a100_self_host.json` | TBD (gates OSF lock) |
| DGX `spark-9ea3` (pre-2026-05-15 archive reference only — NOT paper-1 canonical, retained for §139.8 FP sensitivity ladder + Appendix D contamination disclosure) | `results/provenance/env_lock_dgx.json` | `results/provenance/vwa_dgx_via_quark.json` | TBD (archive baseline; cross-host comparison NOT made per §7 Infrastructure migration note) |
| Myriad (cross-arch numerical-determinism check, optional — paper-2 / Appendix F6) | `results/provenance/env_lock_myriad.json` | N/A (no VWA use case) | TBD if used |
| ~~A100 "Phase 2 mechanistic"~~ | RETIRED 2026-05-15 (B-132): mechanism §5 deferred to paper-2 per advisor 2026-05-14; mechanism-specific A100 provenance moves to paper-2 OSF manifest | — | retired |

Each `env_snapshot.json` captures: torch / transformers / Python / git commit /
HuggingFace model revision SHA (Qwen3-VL-4B + Llama-3.2-Vision if used) /
GPU compute capability / hostname / nvidia-smi output.

### 2.4 Witness chain (4-layer paper-grade; doctrine shift 2026-05-18 per /stress A2.8 followup B-1570 + B-1650~B-1655 two-DOI doctrine restoration 2026-05-18 ~14:00 UTC)

| Layer | Mechanism | Status |
|---|---|---|
| **Git** | Current canonical substance-lock: `git tag preregistration-locked-q3a` on master at `72b93c939526fb0b9a733013852bd90136d601aa` (substance-lock 2026-05-18T11:30Z post Q3=A doctrine fix wave B-1750~B-1759); legacy `preregistration-locked` tag at `ef609a3863adc9b3698789b96a1ee9f709e1c832` retained for audit-trail only. Pre-DOI witness anchor: `git tag preregistration-doi1-witnessed-20260518T211628Z` at commit `5edac3b` (canonical witness file capture at 2026-05-18T21:16:28Z). Post-DOI promotion tag: `git tag preregistration-doi1-minted-osf-9QCWU` at commit `5edac3b` (same anchor, post-DOI-mint name promotion per §3a step 8). | ✅ Layer 1 substance-locked + witnessed + DOI-minted: preregistration-locked-q3a @ 72b93c9 substance-locked; B2 SHA `093f9f388b31de276ce2de164bdc2081324b9767` ✅ landed per B-1603 /stress 深入审 Mode A; preregistration-doi1-witnessed-20260518T211628Z @ 5edac3b ✅ witness-anchored; preregistration-doi1-minted-osf-9QCWU @ 5edac3b ✅ minted 2026-05-18T23:10:06Z UTC. |
| **§A2 audit cascade refs** (was "Advisor email" pre-2026-05-14 sync 收口 — reclassified 2026-05-18 per B-1570 doctrine-shift propagation) | §A2 14/16 audit cascade Git SHA refs at master HEAD substantively close all 14 commit decisions via per-decision B-### audit-trail (master_bug_catalog ## A2.1~A2.9 + 实验笔记 §200-§221); advisor batch sign-off retained as optional post-fire collateral | ✅ Layer 2 substantively complete via §A2 cascade (formal advisor batch sign-off optional post-fire) |
| **OSF DOI 1 — Pre-canonical-outcome-creation witness for Fire-3** (paper-1 cryptographic external witness; supersedes advisor email per B-1570 + B-1650; supersedes fire-2-era witness per B-1670 retraction) | **DOI: `10.17605/OSF.IO/9QCWU`**; OSF Registration GUID `9qcwu` (URL https://osf.io/9qcwu); OSF parent project GUID `kv9sf` (URL https://osf.io/kv9sf). Submitted **2026-05-18T23:10:06Z UTC** (per OSF API `https://api.osf.io/v2/registrations/9qcwu/` date_registered field; UI display "May 19, 2026, 12:10 AM" = BST local time). Archive auto-completed instantly (no 48h wait). License: CC-By Attribution 4.0 International. Canonical empirical witness `artifact_existence_check_doi1_canonical_20260518T211628Z.txt` captured **pre-launch** at 2026-05-18T21:16:28Z (full-file SHA-256 `6056b905e25b0880e613eeb43a091f3281e107073736608755331185ffcadbe3`; content-only SHA-256 `011fa4c07d76798a57070d9aea8b1653ca50dfe432b9c4dee689b96ee9cc6691` per DOI_1_README §"Witness file hash convention"). Tier 1 strictest: all canonical-pattern counts = 0 + any_files_in_run_dirs = 0 at capture moment (no Fire-3 run_dir even existed). Witness strategy revised pre-launch attempt #4 per B-1751-fu (NOT Fire-3 PID-alive — strategy revised to side-step ~5-15 sec runner-startup window per attempt #3 evidence: tier 2 landed because witness captured ~2 min after runner started writing first *_steps_v2.jsonl). Bundle uploaded: preregistration.md + osf_lock_manifest.md + locked_versions.md + model/dataset/ethics/evaluator/compute/neurips/negative-results/release-redaction/topvenue + paper_drafts (frozen pre-outcome) + DOI_1_README.md + canonical witness + interim retraction witness (audit-trail bridge) + retracted/Fire-2-era VOID witness (forensic-only). env_snapshot.json deferred to DOI 2 per pre-launch strategy revision (no post-launch substrate snapshot at DOI 1 mint). Sign-off chain: Row A 2026-05-18T22:52Z PASS + Row B 2026-05-18T22:51Z PASS (release_redaction_checklist.md) → commit 50be2ff → OSF UI navigation → submit 23:10:06Z. | ✅ **Layer 3 MINTED** 2026-05-18T23:10:06Z UTC at DOI `10.17605/OSF.IO/9QCWU` (Registration GUID 9qcwu); witness pattern retraction landed 2026-05-18 ~14:45 UTC per B-1670; pre-launch witness capture landed 21:16:28Z per attempt #4; OSF archive complete + DOI minted instantly (no 48h wait). |
| **OSF DOI 2 — Phase 1a reproducibility bundle** (paper-1 reproducibility deposit; cited_by DOI 1) | OSF deposit minted post-Pass-1+Pass-2+analysis-frozen+paper-final (~2-3 weeks post Phase 1a fire 2026-05-18T13:28:06Z). Bundle = 42 condition_summary_v2.json + per-episode summary + steps JSONLs + aggregate outputs + figures + frozen analysis scripts + paper drafts post-data finalized + DOI_2_README.md with mandatory `cited_by` field referencing DOI 1 | ⏸ Layer 4 pending Pass-1 + Pass-2 + analysis-frozen + paper-final per §3b 8-step workflow |

### §2.5 Reusable patch artifacts — cross-submission provenance witness (B-1624 /stress A2.6c P1-8-B* codex Mode B reproducibility-auditor unique OOB 2026-05-18)

> **Driver**: codex Mode B unique OOB caught that `paper_planning.md §16.0` Multi-submission matrix CLAIMS reused-artifact non-overlap across Paper-1 Workshop / Paper-1 Main / Track A / Track B submissions, but OSF manifest didn't expose the underlying patch families as separate witnesses. `grep -Ec 'B-440|B-448|B-91|B-535|Workshop A|Track A|Track B|cross-paper FP|reused artifacts' osf_lock_manifest.md` returned 0 pre-fix. Without explicit OSF provenance, external replicators couldn't verify workshop artifact subset ≠ paper-1 artifact subset, defeating the salami-slicing-defense audit trail at the OSF layer.

| Patch family | Code path | Commit SHA | Diff hash | Paper-1 Workshop | Paper-1 Main | Track A Workshop | Track B Workshop |
|---|---|---|---|:---:|:---:|:---:|:---:|
| **B-440 + B-448 GRL walk-up click ON_TARGET grounding** | `external/visualwebarena/browser_env/actions.py` (walk-up click implementation) + `p79/envs/vwa_wrapper.py` (wrapper integration) | `<TBD lock at OSF mint>` | `<TBD>` | ✗ (not reused) | ✓ (paper-1 §3.5.2 GRL evidence subset) | ✓ (workshop A hero — cross-benchmark methodology evaluation) | ✗ (not reused) |
| **B-91 LLM-judge polarity fix** | VWA submodule `external/visualwebarena/p79-patches` branch `eb5cbd8` + `evaluation_harness/helper_functions.py:612-613` source patch | `eb5cbd8` (VWA submodule HEAD) | tree-hash chain `752caebdc6bd84761b2f308331f21241a9b4a28de65b46ff0007ef27d8c72778` (per prereg §7 SBOM lock) | ✓ (paper-1 §8.2 disclosure subset) | ✓ (paper-1 §8.2 disclosure subset) | ✗ (not reused) | ✓ (workshop B hero — cross-paper FP family taxonomy + B-91 remediation protocol) |
| **B-535 N/A task-load exclusion** | `p79/experiment/tasks.py` (`exclude_na_tasks: true` default) + `configs/exp_v2_base.yaml` | `<TBD lock at OSF mint>` | `<TBD>` | ✓ (paper-1 §8.2 disclosure subset) | ✓ (paper-1 §8.2 disclosure subset) | ✗ (not reused) | ✓ (workshop B disclosure — part of FP architecture restructure narrative subset) |

**Eligibility verdict per submission**:

- **Paper-1 Workshop** (workshop_R1 = H1 + H2(a) only) reuses: B-91 + B-535 (as §8.2 LLM-judge disclosure subset). Phase 1a-only data (cls + red 6 cells); excludes Phase 1b shop expansion + H10 router substrate.
- **Paper-1 Main** reuses: ALL — full paper-1 substrate including B-440 + B-448 (§3.5.2 GRL evidence) + B-91 + B-535 (§8.2 LLM-judge disclosure) + Phase 1b shop + H10 learned router + B-1284 cross-family claim-tier gate B2-outcome evidence.
- **Track A Workshop** reuses: B-440 + B-448 only (workshop hero = cross-benchmark methodology evaluation of GRL walk-up click ON_TARGET grounding); NOT B-91 or B-535; NOT phantom-space hero / cost-aware routing / P-SoM drop-in.
- **Track B Workshop** reuses: B-91 + B-535 only (workshop hero = cross-paper evaluator FP family taxonomy + B-91 remediation protocol); NOT B-440 or B-448; NOT phantom-space hero / cost-aware routing / P-SoM drop-in.

**OSF audit verifiability**: an external replicator can verify each submission's claimed patch-artifact subset by checking out the corresponding git commit SHA + verifying the patch family code path + comparing against the submission's §1 prose non-overlap paragraph. The "Eligibility per submission" column above is the contractual audit trail for the salami-slicing defense.

---

## §3 OSF DOI workflows — two-DOI split (B-1650~B-1655 doctrine restoration 2026-05-18 ~14:00 UTC)

> **Doctrine-shift note (/stress doctrine restoration B-1650~B-1655 2026-05-18 ~14:00 UTC, supersedes B-1570 single-DOI "post Pass-1 data complete" framing that drifted from 2026-05-05 advisor sync §F.1 outcome decision)**: pre-correction §3 was a single 8-step workflow gated on Phase 1a Pass-1 baseline data complete (B-1570 doctrine retired advisor email gate; this correction restores the §F.1 pre-data DOI ordering invariant — DOI public-ledger timestamp must precede outcome unblinding, NOT lag it). Pre-correction §3 is now SPLIT into §3a (DOI 1 pre-outcome-creation witness, MINTABLE TODAY 2026-05-18) + §3b (DOI 2 reproducibility bundle, mint post Pass-1+Pass-2+analysis-frozen+paper-final). Step 1 advisor batch sign-off remains optional collateral; can land before OR after EITHER DOI mint as audit-trail reproducibility completeness.

### §3a Eight-step DOI 1 workflow — Pre-outcome-creation witness (MINTABLE NOW 2026-05-18)

Mint blocker triple: ✅ zero Phase 1a outcome artifacts at submission time (UTC 2026-05-18T13:57:22Z empirical check per `retracted/artifact_existence_check_doi1_20260518T135722Z_VOID_RETRACTION_ONLY.txt`) + ✅ B2 HF SHA `093f9f388b31de276ce2de164bdc2081324b9767` pinned + ✅ preregistration substance-locked (Git tag `preregistration-locked` at SHA `ef609a3863adc9b3698789b96a1ee9f709e1c832`, status: locked 2026-05-18T11:30Z per A2.8 cascade closure).

1. **(operator)** Verify outcome-artifact-zero check still PASSES (re-run `artifact_existence_check` SSH probe to A100; if any episode_summary_v2.json has landed since `retracted/artifact_existence_check_doi1_20260518T135722Z_VOID_RETRACTION_ONLY.txt` capture, downgrade DOI 1 wording from `pre-outcome-creation` → `pre-outcome-inspection` and update `DOI_1_README.md` empirical status table accordingly)
2. **(operator)** Stage DOI 1 bundle: copy/symlink all `docs/checkpoints/pre_run/*.md` + `paper_drafts/section{1..8}_*.md @ ef609a3` + `paper_drafts/paper.bib @ ef609a3` + `env_snapshot.json` (A100 pre-fire) + `DOI_1_README.md` + `artifact_existence_check_doi1_<UTC-TS>.txt` into `docs/checkpoints/pre_run/osf_deposit_DOI1_<UTC-TS>/` (frozen-at-`ef609a3` snapshot folder, git committable)
3. **(operator)** Verify Git tag `preregistration-locked` is pushed to GitHub `origin/master` (DOI 1 OSF page will cite the GitHub tag URL https://github.com/<user>/<repo>/tree/preregistration-locked); commit + push if local-only
4. **(operator at OSF.io)** Create new OSF project: title `Phase 1a Pre-outcome Analysis Lock — phantom-SoM pre-registration witness`; tags: pre-registration, hypothesis-lock, analysis-plan-lock, pre-outcome-witness, OSF DOI 1
5. **(operator at OSF.io)** Upload DOI 1 bundle from staging folder; verify all files present per `DOI_1_README.md` "Bundle contents" table
6. **(operator at OSF.io)** Submit public registration — captures submission UTC timestamp as cryptographic pre-outcome-creation anchor; record `osf_submitted_at_utc` + `osf_guid` in `preregistration.md` frontmatter + `DOI_1_README.md` frontmatter
7. **(0-48h auto, OR manual approve)** OSF admin approval → DOI string assigned (`10.17605/OSF.IO/XXXXX`); record `osf_doi_1_pre_outcome_witness` in `preregistration.md` frontmatter + `DOI_1_README.md` frontmatter
8. **(operator)** Backfill: paper §1 + §4 + Appendix D footnotes cite DOI 1; Git tag promote `preregistration-locked` → `preregistration-doi1-witnessed-<DOI>` + push; commit `doctrine(osf-witness): DOI 1 minted — <DOI>` as final pre-outcome-creation lock proof

After Step 8, **no edits permitted** to any file referenced in §2 / §2.1 / §2.4 Layer 3 except via new commits explicitly noted as "post-lock amendment" in `paper_planning.md` §19 decision log + paper §3 disclosure paragraph.

### §3b Eight-step DOI 2 workflow — Phase 1a reproducibility bundle (mint trigger: Pass-1 + Pass-2 + analysis-frozen + paper-final, ~2-3 weeks post fire)

Mint blocker quadruple: ⏳ Pass-1 baseline 36 conditions complete (all 36 `condition_summary_v2.json` present per `make active` cell snapshot) + ⏳ Pass-2 learned router 6 conditions complete (all 6 `obs_mode="learned"` sentinel conditions present) + ⏳ analysis scripts frozen at commit SHA (`scripts/analysis/aggregate_phantom_lift.py` + `scripts/analysis/figures/*.py` SHA-locked) + ⏳ paper §1-§8 finalized post-codex round.

1. **(operator)** Verify ALL 4 mint blockers PASS: re-run `artifact_existence_check` SSH probe to A100 with `episode_summary_count + condition_summary_count` expected = 42 + per-cell aggregates present + paper drafts post-codex-round signed off
2. **(operator)** Stage DOI 2 bundle: copy all 42 `condition_summary_v2.json` + `episodes/<site>_task_<N>_summary_v2.json` + `episodes/<site>_task_<N>_steps_v2.jsonl` (canonical schema per `p79/experiment/logger_v2.py:111` + `analysis.py:209`; correct after B-1670 retraction wave 2026-05-18) + `results/phantom_paper/*.csv` + `results/phantom_paper/figures/*.png` + frozen analysis scripts + final paper drafts + `run_registry/*.json` + `analysis_walkthrough.md` + `DOI_2_README.md` (with mandatory `cited_by: DOI 1`) into `docs/checkpoints/pre_run/osf_deposit_DOI2_<UTC-TS>/` folder
3. **(operator)** Git commit + tag `phase-1a-data-complete` at HEAD; push to GitHub `origin/master`
4. **(operator at OSF.io)** Create new OSF project: title `Phase 1a Reproducibility Bundle — phantom-SoM full outcome + analysis + paper deposit`; tags: reproducibility, data-deposit, OSF DOI 2; **explicit `cited_by` reference to DOI 1** (forward-only cross-link per OSF immutable-registration semantics)
5. **(operator at OSF.io)** Upload DOI 2 bundle from staging folder
6. **(operator at OSF.io)** Submit public registration — DOI 2 submission UTC timestamp is post-data (NOT pre-outcome-creation; this DOI's role is reproducibility, not pre-reg)
7. **(0-48h auto, OR manual approve)** OSF admin approval → DOI 2 string assigned
8. **(operator)** Backfill: paper §1 + §4 + Appendix D footnotes cite BOTH DOI 1 (pre-reg) + DOI 2 (reproducibility); Git tag promote `phase-1a-data-complete` → `phase-1a-doi2-deposited-<DOI>` + push; commit `doctrine(osf-witness): DOI 2 minted — <DOI>` as final reproducibility deposit proof

After Step 8, the full paper-1 cryptographic witness chain is complete (Git tag + §A2 cascade + DOI 1 pre-outcome + DOI 2 reproducibility), and any paper version is replayable from `git checkout phase-1a-doi2-deposited-<DOI>` + DOI 2 OSF bundle.

### §3c OSF approval-lag operational note (B-1650 OSF help.osf.io/article/330 reference)

OSF public registration uses a 2-stage approval model:
- **Stage A — Submission** (operator action): captures cryptographic submission UTC timestamp as the witness anchor; registration content immutable from this moment forward
- **Stage B — Admin approval** (default auto 48h, OR manual project admin click): assigns DOI string `10.17605/OSF.IO/XXXXX` to the registration

Paper §4 / Appendix D citation form depends on stage:
- **Interim (pre-approval, 0-48h post-submission)**: `OSF registration GUID osf.io/xxxxx, submitted YYYY-MM-DDTHH:MM:SSZ UTC, pre-outcome-creation witness; DOI <pending OSF admin approval, default auto-approve 48h>.`
- **Final (post-approval)**: `OSF DOI 10.17605/OSF.IO/XXXXX, submitted YYYY-MM-DDTHH:MM:SSZ UTC, pre-outcome-creation witness (empirical artifact-zero check at submission); registered Git tag preregistration-locked at SHA <ef609a3>.`

The DOI string is a convenience for citation; the cryptographic anchor is the OSF submission UTC timestamp on the immutable registration page — verifiable by any third party visiting osf.io/xxxxx, before AND after DOI assignment.

---

## §4 Post-lock change discipline

If advisor / reviewer pushes for threshold change post-lock:
- **NEW** preregistration document added (`preregistration_v2.md`) — DOI'd
  separately
- v1 + v2 both cited in paper §3
- Each pre-spec block in paper §5 footnoted with "matches v1 OSF lock at
  YYYY-MM-DD" or "amended in v2 OSF lock"

This avoids the "moving goalpost" critique while enabling honest threshold
revision when data warrants.

---

## §5 References

- 笔记 §110 (5/5 advisor sync outcomes)
- 笔记 §111 (Stage 2 mechanistic pilot results)
- 笔记 §113 (24+15 candidates curation)
- 笔记 §114 (this provenance hardening)
- `docs/checkpoints/paper_planning.md` §19 (decision log)
- `docs/checkpoints/_status/issues/issue_advisor_sync_2026-05-14.md` + `docs/checkpoints/paper_planning.md §19` decision log (ADVISOR_SYNC.md retired 2026-05-15, commit `f64bc9d`; OSF DOI workflow detail consolidated into `osf_lock_manifest.md §3` + paper_planning §19)
- `docs/checkpoints/advisor_sync_5_5_followup.md` (Q1-Q11 pending email)
- `scripts/provenance/snapshot_env.py` / `snapshot_vwa.sh` /
  `numerical_determinism_check.py`
