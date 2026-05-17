# OSF DOI Lock Manifest — Paper §3 / Appendix D

**Purpose**: Codify the artefacts whose SHA-256 / git ref get frozen at the
moment of OSF preregistration DOI minting. Once advisor's email reply arrives
(post 5/5 sync) confirming K_h1 / K_h3 / TOST δ thresholds, this checklist
drives the 8-step DOI workflow (笔记 §110 + `paper_planning.md §19` decision log + `osf_lock_manifest.md §3`; ADVISOR_SYNC.md retired 2026-05-15, commit `f64bc9d`, replaced by `_status/issues/issue_advisor_sync_2026-05-14.md` frontmatter + Bases view).

**Status**: 🟡 Draft — fields populate at lock moment.
**Lock blocker**: ⏳ advisor email reply (Q1-Q11 in `advisor_sync_5_5_followup.md`)

---

## §1 Pre-lock checklist (everything must be done before OSF DOI mint)

- [ ] Advisor email reply received (K_h1=0.75 transparency / K_h3=0.67 transparency /
      TOST δ=1.0pp SR-margin / Phase 1a 42-condition 6-cell scope (36 baseline + 6 router; A1.21 P1-5 align 2026-05-17) confirmed
      OR alternative noted)
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
| H1(ii) superiority threshold δ (SR-margin) — also TOST informational margin | 1.0pp | ⏳ pending | SR percentage-point margin for H1(ii) one-sided superiority test (primary) + TOST equivalence informational secondary; distinct from H2(a) cost ±10% relative margin |
| Cell scope (Phase 1a operational) | **42 conditions** = **Pass-1 baseline 36** (2 sites × 3 models × 6 modes) + **Pass-2 learned router 6** (2 sites × 3 models × 1 router cond/cell, `obs_mode="learned"` sentinel) | ⏳ pending | A1.21 P1-5 fix 2026-05-17 (B-533, codex F10): aligned with prereg §4 L438 "42 conditions" scope expansion 2026-05-16 (B-264+B-267 /stress A1.7 cross-AI cycle for §2 H10 Pass-2 router parity with gating-hypothesis registration). Replaces prior "36 operational" entry which was Pass-1-only and mismatched preregistration since 2026-05-16 |
| Cell scope (Phase 1a statistical) | **6 cells** = (site, model) tuples: (cls, B0), (cls, B1), (cls, B2), (red, B0), (red, B1), (red, B2) | ⏳ pending | One drop-one number per cell; pooled FE meta input (decision "3A" 2026-05-14, FE not DL); k=4 → k=6 per B2 addition 2026-05-14 |
| Cell scope (Phase 1b deferred) | +12 conditions = shop × B0+B1 × 6 modes | ⏳ pending | Main-paper expansion lever; not part of Phase 1a workshop submission |

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

### 2.4 Witness chain (3-layer paper-grade)

| Layer | Mechanism | Status |
|---|---|---|
| **Git** | `git tag preregistration-locked` on master at OSF DOI mint commit | TBD |
| **Email** | Advisor reply email PDF saved to `docs/reference/advisor_email_<date>.pdf` (Gmail message-id recorded in this manifest) | TBD |
| **OSF** | OSF preregistration page DOI: `<TBD>` (e.g., `10.17605/OSF.IO/XXXXX`) | TBD |

---

## §3 Eight-step DOI workflow (post advisor email reply)

1. **Receive advisor reply** → save email PDF + extract Gmail message-id
2. **Update `preregistration.md`** with confirmed thresholds + decision log entry
3. **Run `python3 scripts/provenance/snapshot_env.py`** on **A100 (paper-1 canonical, mandatory)** + DGX (archive reference) + Myriad (optional cross-arch),
   commit results under `results/provenance/env_lock_<host>.json` (B-132 2026-05-15: A100 is canonical, DGX is archive-only per §7 Infrastructure migration note)
4. **Run `bash scripts/provenance/snapshot_vwa.sh`** on each VWA-bearing host,
   commit results
5. **Snapshot paper drafts** → `cp -r paper_drafts paper_drafts_locked` + commit
6. **Tag git** → `git tag -a preregistration-locked -m "OSF DOI mint $(date)"`
   + `git push origin preregistration-locked`
7. **Mint OSF DOI** at https://osf.io/registries/ — link OSF page to the tagged
   commit URL on GitHub (https://github.com/<user>/<repo>/tree/preregistration-locked)
8. **Backfill this manifest** — populate all `<TBD>` cells with actual SHAs +
   timestamps + DOI; commit as final lock proof

After Step 8, **no edits permitted** to any file referenced in §2 except via
new commits explicitly noted as "post-lock amendment" in `paper_planning.md` §19
decision log + paper §3 disclosure paragraph.

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
