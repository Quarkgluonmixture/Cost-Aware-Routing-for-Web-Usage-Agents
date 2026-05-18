# Advisor Pre-Registration Witness Email — Draft 2026-05-18

> **Status**: Courtesy notification (optional post-fire collateral per
> /stress A2.8 followup B-1570 doctrine shift 2026-05-18 — advisor email
> reply is **no longer the gating event** for Phase 1a Pass-1 launch nor
> for OSF DOI mint per `osf_lock_manifest.md §3` updated header). The 14
> commit decisions in `preregistration.md §6 §(a)` are substance-locked
> via §A2 audit cascade Git refs (each lock decision cross-references a
> B-### audit fix landed in master with commit SHA).
>
> **Why send anyway**: completeness of OSF DOI mint reproducibility
> audit-trail; advisor signature populates `witnessed_by` field in
> `preregistration.md` frontmatter at OSF DOI mint time (paper §6 §(b)
> external witness layer).
>
> **Action**: review email below + send to advisor at your convenience.
> If advisor replies → save as `.witness/preregistration_witness.eml`
> (gitignored, local-only) + paste advisor sign-off date into
> `osf_lock_manifest.md §1` and `preregistration.md` frontmatter
> `witnessed_by:` field at OSF DOI mint time.

---

## Email body (suggested)

**Subject**: P79 — Phantom-SoM Pre-Registration Witness Request (paper-1 Phase 1a Pass-1 launch 2026-05-18)

Dear [Advisor Name],

I am about to fire the Phase 1a Pass-1 baseline rerun for paper-1 ("Cost-Aware
Routing for Web Usage Agents", aka phantom-SoM hero hypothesis) on the
UCL Condense A100 self-hosted VWA Docker stack. Per our 2026-05-14 sync
("advisor sign-off is optional collateral for paper-1; student focus =
experiment execution"), I am sending this as courtesy notification —
the pre-registration is substance-locked via Git refs at master HEAD
already, so a reply is not required to unblock the launch. I want to
flag it nonetheless so you can review pre-fire if you wish.

**Pre-registration commitment (locked at Git SHA `<FIRE_TIME_GIT_SHA>`)**:

The 14 commit decisions in `docs/checkpoints/pre_run/preregistration.md
§6 §(a)` are locked in as of master HEAD. Summary:

1. **Estimand** = fixed-effects average over 6 planned (site, model) cells
   per decision "3A" 2026-05-14; expanded to k=6 cells with B2 addition
   2026-05-14. No τ²/REML — 6 cells = study design not population sample.
2. **K-of-N** = pure transparency count, NO threshold (2026-05-14).
3. **δ** = 1.0pp for H1 one-sided fixed-effects superiority test.
4. **Cell inclusion**: Phase 1a = cls + red × B0+B1+B2 × 6 modes; Phase 1b
   shop deferred to post-workshop expansion.
5. **Witness mechanism**: Git + advisor email (optional collateral) + OSF DOI.
6. **N_conditions** = 42 operational (36 baseline + 6 H10 learned router).
7. **Smoke-gate** = outcome-independent (no SR-based restart).
8. **Router H7/H8** = paper-2 forward stub; paper-1 H10 estimand locked
   2026-05-18 per A2.8 B-1550 (two-layer operational deployment criterion).
9. **Train/test split** = paper-2 router scope (5-fold site-stratified CV vs LOSO).
10. **B2 baseline** = Gemma3-VL `google/gemma-3-4b-it` (4B parity, cross-family
    robustness check; HF SHA `093f9f388b31de276ce2de164bdc2081324b9767`
    locked 2026-05-18 per B-1603).
11. **Host migration** = A100 Condenser canonical, DGX archive-only post 2026-05-15.
12. **Paper-1 router scope flip** = router promoted paper-2→paper-1 per advisor 2026-05-14.
13. **Mechanism §5 paper-2 deferral** = patching/probes/SAE deferred entirely to paper-2.
14. **Power re-derivation at k=6** = closed empirically per /stress A2.3a
    B-941~B-958 2026-05-17 (FE-pool projected 97-100% across H1 + H3(i) + H3(ii)).

**H-list locked**: H1 + H2(a) + H3(i)/(ii) + H10 (primary gating), with
H5/H6 post-hoc disclosure + H7/H8 paper-2 deferred + H9/H11 paper-2 deferred
per v7 amendment 2026-05-16.

**§A2 audit cascade closure** (16 design-layer audits, 12+ closed 2026-05-18):
A2.1 + A2.2 + A2.3a/b/c/d + A2.4a/b + A2.5 + A2.6a/b + A2.7 + A2.8 + A2.9 closed;
A2.6c + A2.10 in flight (Submission-ready scope, NOT launch gate).
Cross-AI audit substrate (Claude + codex + gemini /stress) used as
methodology-quality check; 2026-05-18 NeurIPS 2025 LLM Use Disclosure
documented in `pre_run/ethics_license_coi_statements.md §"LLM Use Disclosure"`.

**Fire plan**:

- Phase 1a Pass-1 baseline (36 conditions) sequential per-site via
  `queue_phase1_paper_grade.sh launch`
- A100 wallclock estimate: 1-2 weeks
- Pass-2 learned router (6 conditions) launched sequentially after Pass-1
  data lands
- Phase 1b shopping expansion (21 conditions) deferred to post-workshop main paper

If you have any concern about substance of the 14 commit decisions or the
hypothesis-tier gates, please reply pre-fire or within reasonable window
post-fire and I will mark a v2 preregistration document per Protocol A §4
post-lock change discipline. Otherwise no reply is required for the
internal-witness chain.

**A one-line confirmation reply** (if you choose to provide it) suffices:
"I witness the P79 phantom-SoM pre-registration of gating hypotheses
H1-H3 + H10 and the 14 lock decisions as of Git SHA `<FIRE_TIME_GIT_SHA>`
on 2026-05-18." — I will archive this in `.witness/preregistration_witness.eml`
(gitignored, local-only) for OSF DOI mint reproducibility audit trail
completeness.

**OSF DOI mint timeline**: Phase 1a Pass-1 baseline data complete →
OSF page upload (`preregistration.md` + `paper_drafts_locked/` snapshot +
locked artifact bundle per `osf_lock_manifest.md §3` 8-step workflow) →
DOI assignment → backfill into `osf_lock_manifest.md §2` <TBD> cells.
Expected ~1 week post-Pass-1-complete.

Best regards,
Jiaming

---

## Email metadata

| Field | Value |
|---|---|
| To | `<advisor email>` |
| Cc | `<lab pre-submission audit cc>` (optional) |
| Subject | P79 — Phantom-SoM Pre-Registration Witness Request (paper-1 Phase 1a Pass-1 launch 2026-05-18) |
| Send date | 2026-05-18 (pre-fire) OR within 24h post-fire (courtesy) |
| Reply expected | optional — not blocking |
| Archive location post-reply | `.witness/preregistration_witness.eml` (gitignored, local) |

## Cross-references in this email

- `docs/checkpoints/pre_run/preregistration.md` §6 §(a) — 14 commit decisions canonical
- `docs/checkpoints/pre_run/osf_lock_manifest.md` §3 — 8-step DOI workflow + B-1570 doctrine shift
- `docs/checkpoints/_status/issues/issue_advisor_sync_2026-05-14.md` — sync 收口 record
- `docs/checkpoints/实验笔记.md` §200-§221 — §A2 audit cascade chronicle
- `docs/reference/master_bug_catalog.md ## A2.x` sections — per-audit fix tables
- Git tag `preregistration-locked` (created post-prep at Phase 1a fire start)

## Note on send timing

Per current doctrine, advisor email can be sent (a) pre-fire as substance
notification, (b) at-fire-start with Git SHA reference, or (c) post-fire
within ~24h as audit-trail collateral. Substance is identical across the
three options. Recommendation: send (b) at-fire-start so the Git SHA in
the email body matches the immutable Git tag created at fire start.
