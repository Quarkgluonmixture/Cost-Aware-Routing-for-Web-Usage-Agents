---
title: Drop B2 (Gemma3-VL) reddit conditions — Phase 1a matrix 6→5 cells, reddit Pass-1 = B0+B1 only
status: WITNESS — canonical chain-def change (build_red_chain: 18→12 reddit conditions); current running orchestrator intercept PENDING (auto-mode classifier blocked the kill-guard daemon → operator-authorized intercept required, see §4)
parent_doi: 10.17605/OSF.IO/9QCWU
parent_lock_tag: preregistration-locked @ ef609a3
prior_amendments:
  - AMENDMENT_01_PROTOCOL_RESET_20260521
  - AMENDMENT_02_GATE_LADDER_20260523
  - AMENDMENT_03_IMPLEMENTATION_ALIGNMENT_20260524
  - AMENDMENT_04_ANALYSIS_ALIGNMENT_20260524
  - AMENDMENT_05_COORDINATE_CONTRACT_20260525
  - AMENDMENT_06_REPRODUCIBILITY_SENSITIVITY_20260525
  - AMENDMENT_07_SOM_IDENTIFIER_CONTRACT_20260525
  - PROTOCOL_NOTE_01_SESSION_LOST_PAPER_GRADE_20260527
  - PROTOCOL_NOTE_02_TRANSIENT_PREFLIGHT_RETRY_20260621
  # PROTOCOL_NOTE_03 reserved (resume-on-abort policy witness, B-1882, pending)
  - PROTOCOL_NOTE_04_REDDIT_IDENTITY_RESET_20260625
witness_tag: protocol-note-05-drop-b2-reddit-20260628   # set at finalizing commit
osf_deposit: RECOMMENDED at next advisor sync — this REDUCES the preregistered matrix (6→5 cells); advisor-confirm advised
decided_by: user 2026-06-28 ("把 b2 停掉 … 直接进入 router"; chose mechanism A — boundary intercept; did not defer to advisor)
cross_ai_audit: NOT run — this is a scope/scheduling decision (drop a cell), not a new measurement claim or pipeline change to existing cells
---

# PROTOCOL_NOTE_05 — Drop B2 (Gemma3-VL) reddit conditions

## 0. Decision

Phase 1a reddit Pass-1 baseline is reduced from **18 conditions (B0/B1/B2 × 6 modes)**
to **12 conditions (B0/B1 × 6 modes)** — the **6 B2 (Gemma3-VL = `google/gemma-3-4b-it`)
reddit conditions are dropped**. The Phase 1a statistical matrix goes from **6 cells**
`(cls/reddit) × (B0/B1/B2)` to **5 cells** (reddit loses its B2 cell). **B2 cls (6
conditions) is unaffected and already bound-clean.**

Decided by **user 2026-06-28** ("把 b2 停掉 … 直接进入 router"), without an advisor gate
(same posture as PROTOCOL_NOTE_04's estimand decision).

## 1. Rationale

- **Time**: B2 reddit ≈ 40h (the last 6 of the ~4-day reddit chain). D4 (Pass-1 all 36
  @06-26) already missed; dropping B2 reddit reaches the **router pass (paper §6 core
  contribution) ~1.7 days sooner**.
- **Low scientific cost on these benchmarks**: B2/Gemma is a settled capability **floor**
  (cls ~1% = real floor, B-1876; reddit prior aborted runs ~0). Its role is **cross-family
  replication BREADTH** (2026-06-16 framing: cross-model = breadth, NOT controlled ablation
  — the router only ever compares representations *within* a model). Dropping B2 reddit
  removes reddit's cross-family control; this aligns with the standing **"换槽" plan**
  (replace B2 slot with MiMo-VL B3, demote Gemma to §8 floor disclosure). reddit cross-family
  is therefore **deferred to MiMo B3**, not abandoned.

## 2. What the paper must disclose (§8)

- Phase 1a reddit = **2 models (B0 Qwen3-VL-235B, B1 Qwen3-VL-4B — both Qwen family)**; no
  cross-family control on reddit (cls retains 3 models incl. B2 Gemma).
- Matrix asymmetry: **cls 6 cells / reddit 5... → actually cls 3-model, reddit 2-model**
  (Phase 1a = 18 cls + 12 reddit = 30 conditions / 5 cells).
- B2 cls retained as cross-family floor evidence (§8 floor disclosure).
- **Reversible**: re-adding the 6 `... B2 reddit` lines in `build_red_chain` restores the
  cell; B2 reddit can be generated later if the advisor wants the symmetric matrix.

## 3. Implementation — primary protection (DONE)

`scripts/queues/queue_phase1_paper_grade.sh::build_red_chain` edited: heredoc 18→12
conditions (B2 reddit lines removed; explanatory comment + reversibility note added).
Synced to A100 (`/home/ubuntu/workspace/p79`, verified 12 command lines / 0 B2). Any future
`launch red` / reboot-relaunch now generates a B2-less reddit chain. **cls chain
(`build_cls_chain`) keeps B2 unchanged.**

## 4. Implementation — current-orchestrator intercept (PENDING operator authorization)

The reddit orchestrator launched 2026-06-25 (`queue_chain.sh`, started before this note)
has all 18 conditions baked into its **argv**; bash buffers the `for cmd in "$@"` loop, so
editing the script does **not** change the running loop. Mechanism **A** (user-chosen) =
intercept at the B1→B2 boundary by killing the orchestrator once B0+B1 reddit (12 cond)
complete. Runners are **detached** (orchestrator polls via `pgrep`, not `wait`), so killing
the orchestrator does **not** disturb any in-flight condition.

A background kill-guard (`scripts/maintenance/b2_reddit_drop_guard.sh`, triggers on a B2
reddit runner appearing → kills orchestrator + B2 runner + archives partial + ntfy) was
written but its deployment was **blocked by the Claude Code auto-mode safety classifier**
(autonomous kill-daemon on shared infra requires explicit operator authorization). Options
to enact the intercept (operator decides): (a) run the guard manually / add a Bash
permission rule; (b) manual boundary kill when B1 reddit nears completion (no daemon);
(c) kill current orchestrator + relaunch the B2-less chain now (one-time; runner detached =
safe; deviates from "don't touch fire now"). **Until enacted, the current orchestrator will
run B2 reddit if it reaches the boundary unattended.**

## 5. Router

Pass-2 router (paper §6) runs after reddit baseline (single-chain-per-host rule). With B2
dropped, reddit router cells = B0/B1 (cls router keeps all 3 models). P0-2 gate (Stage 1→3
on landed Pass-1 data) still applies before router launch.
