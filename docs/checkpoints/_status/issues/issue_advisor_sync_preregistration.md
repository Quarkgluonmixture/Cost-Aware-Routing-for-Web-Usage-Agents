---
type: issue
category: blocker
status: active
priority: high
action: wait advisor email reply on advisor_sync_5_5_followup.md Q1-Q11 (K_h1/K_h3/TOST + 7 other) → flip preregistration locked → OSF DOI upload
created: 2026-05-03
updated: 2026-05-06
---

# Advisor sync + pre-registration lock

Pre-registration framework reframed 2026-05-03 (Hero + Structural + Framing-rule R1-R5) — `docs/checkpoints/preregistration.md` is `status: draft` pending advisor email witness reply.

## 5/5 sync outcome (partial)

Sync **happened 2026-05-05** (Teams, ~30 min, 网卡断在 threshold detail). Transcript: `docs/reference/transcript.md`. 

✅ **Confirmed in sync** (transcript-grounded, see `advisor_sync_5_5_outcomes.md §A`):
- Early-stop A 全 cancel
- Manifest 全 archive + 16-cell rerun
- Paper 拆开发 (3 vs 4 papers exact count pending Q1)
- VWA bug → ACL position paper / survey
- Routing paper = benchmark study
- Mechanistic interp = new paper-worthy direction (advisor 5/5 push)
- Workshop submission node
- Compute paths (now superseded by 5/6 A100 allocation)
- Pre-registration witness mechanism framework (git + email + OSF)
- Environment 3-layer framework

## Pending advisor email reply (~2-5 d)

Follow-up doc: `advisor_sync_5_5_followup.md` Q1-Q11 sent 5/5 evening:
- Q1 ⭐ Mechanistic nested in Phantom (3 papers) vs independent (4 papers)
- Q2 Phantom paper venue (D&B / TMLR / MLSys benchmark study)
- Q3 Mechanistic scope (B1-only vs cross-arch) — **likely auto-resolved by A100 + Llama-4 affordable, see 笔记 §112**
- Q4 Environment 3-layer mapping (Routing / Phantom / Mechanistic)
- Q5 Workshop names + node
- Q6 Pre-reg mechanism final OK
- **Part 3 #1 ⭐⭐⭐ Threshold lock**: K_h1=0.75 / K_h3=0.67 / TOST δ=1.0pp witness
- Part 3 #2 Train/test split (5-fold lean vs LOSO)
- Part 3 #3 Mechanistic scope (B1-only vs cross-arch) — paired with Q3

## Email reply 进来后 action plan (8 步, 见 `outcomes.md §F.5`)

1. Migrate outcomes §C → §A
2. Flip `preregistration.md` `status: draft` → `status: locked`, fill `registered_at` + `registered_git_sha` + `witnessed_by`
3. preregistration decision log 加 entry
4. paper_planning §19 decision log 加 entries
5. 笔记 §113 chronicle (advisor email reply)
6. git commit + push
7. `.witness/preregistration_witness.eml` 落地 (gitignored)
8. OSF DOI 上传 → paper §1 footnote cite

## Blocks

- T0e: `preregistration.md` flip status:locked
- 16-cell phantom rerun launch (`issue_14cell_phantom_rerun.md`) — gated on threshold lock OR student-launch-without-witness decision
- OSF DOI upload (`outcomes.md §F`)
- Final paper hook commit (R1-R5 framing rule outcome)
- Paper §1 footnote audit-trail prose

## Refs

- `docs/checkpoints/preregistration.md` (status:draft, frontmatter `data_lock_until: <pending 16-cell rerun completion>`)
- `docs/checkpoints/advisor_sync_5_5_followup.md` (Q1-Q11 邮件主体, sent 5/5 evening)
- `docs/checkpoints/advisor_sync_5_5_outcomes.md` (sync registry §A-§F + OSF workflow)
- `docs/reference/transcript.md` (5/5 sync 完整逐字稿)
- `docs/checkpoints/实验笔记.md §110 §111 §112` (5/5 sync chronicle + mechanistic + A100)
- `docs/reference/EVIDENCE_LAYER_AUDIT.md` §2 (template + meta-rationale)
