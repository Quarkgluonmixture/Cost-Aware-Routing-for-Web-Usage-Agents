# OSF DOI Lock Manifest — Paper §3 / Appendix D

**Purpose**: Codify the artefacts whose SHA-256 / git ref get frozen at the
moment of OSF preregistration DOI minting. Once advisor's email reply arrives
(post 5/5 sync) confirming K_h1 / K_h3 / TOST δ thresholds, this checklist
drives the 8-step DOI workflow (笔记 §110 + ADVISOR_SYNC §F).

**Status**: 🟡 Draft — fields populate at lock moment.
**Lock blocker**: ⏳ advisor email reply (Q1-Q11 in `advisor_sync_5_5_followup.md`)

---

## §1 Pre-lock checklist (everything must be done before OSF DOI mint)

- [ ] Advisor email reply received (K_h1=0.75, K_h3=0.67, TOST δ=1.0pp confirmed
      OR alternative thresholds noted)
- [ ] `preregistration.md` final text edit committed + pushed
- [ ] `run_manifest.yaml` archived rows verified (16-cell scope, all
      `grade=archived` for pre-rerun cells, `grade=paper-grade` post-rerun cells)
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
| Mechanistic 24+15 candidates | `results/mechanistic/archive_subset_b1_cls/manifest.json` | `<TBD commit-SHA>` | TBD |

### 2.2 Hypothesis thresholds (advisor email confirmed)

| Threshold | Pre-reg value | Advisor confirmed? | Notes |
|---|---|---|---|
| K_h1 (Phantom-SoM ≥ best DOM/SoM/Vision) | ≥ 12/16 cells | ⏳ pending | Cell scope = 16 (B0×{cls,red,shop} × 3 phantom-flavours × 2 if cell=site×phantom_axis) |
| K_h3 (drop-one oracle pp lift) | ≥ 11/16 cells | ⏳ pending | Original pre-reg was 14-cell with K_h3≥10/14 |
| TOST equivalence δ | 1.0pp | ⏳ pending | Equivalence margin for "cost ≈ DOM" claim |

### 2.3 Environment fingerprints (per-machine snapshots)

| Machine | env_snapshot.json | vwa_snapshot.json | Locked at |
|---|---|---|---|
| DGX `spark-9ea3` (Phase 1 baseline) | `results/provenance/env_lock_dgx.json` | `results/provenance/vwa_dgx_via_quark.json` | TBD |
| A100 `condense` (Phase 2 rerun + mechanistic) | `results/provenance/env_lock_a100.json` | `results/provenance/vwa_a100_self_host.json` | TBD |
| Myriad (cross-arch backup, optional) | `results/provenance/env_lock_myriad.json` | N/A (no VWA use case) | TBD if used |

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
3. **Run `python3 scripts/provenance/snapshot_env.py`** on DGX + A100 (+ Myriad
   if used), commit results under `results/provenance/env_lock_<host>.json`
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
- `docs/checkpoints/ADVISOR_SYNC.md` §F (OSF DOI workflow detail)
- `docs/checkpoints/advisor_sync_5_5_followup.md` (Q1-Q11 pending email)
- `scripts/provenance/snapshot_env.py` / `snapshot_vwa.sh` /
  `numerical_determinism_check.py`
