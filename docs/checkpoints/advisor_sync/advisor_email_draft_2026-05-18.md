# Email 1 — Phase 1a Fire Notification (informal FYI) — Draft 2026-05-18

> **2-email logic clarification 2026-05-18**: 这是 **唯一的** advisor email,**不是 formal witness 邮件**。`Email 2` (formal pre-registration witness reply ask) 在 OSF DOI mint planning 之后已**冗余** — OSF DOI 是 cryptographically stronger external witness (public registry + immutable timestamp + content hash + machine-verifiable),strictly supersedes advisor email witness (per `preregistration.md §6 §(b)` external witness layer + B-1570 doctrine 2026-05-18 codification). 这封 Email 1 仅做 supervision loop courtesy + collegial FYI,**不 ask** witness 回复。Advisor 若自愿回 "good luck",截图归档锦上添花,但 paper-grade integrity 完全靠 Git tag + OSF DOI 双 layer。
>
> **Recipients**: Maria (primary advisor) + Zekun (co-supervisor)
> **Tone**: informal, light technical (FYI, not action-required)
> **Send timing**: 同 fire event (i.e., I send fire signal → Claude execute fire sequence → I send this email)
> **Reply expected**: none required;casual "good luck" 也 OK

---

## Subject

`P79 phantom-SoM — Phase 1a Pass-1 baseline 实验启动 FYI`

## Body

Hi Maria + Zekun,

Just a quick FYI — I'm firing the Phase 1a Pass-1 baseline rerun for the P79
phantom-SoM paper today (2026-05-18). Thought you'd want to know it's
underway since it's the first paper-grade experiment for paper-1.

**What's running**:

- Two VWA sites (classifieds + reddit); shop deferred to Phase 1b post-workshop
- Three baselines:
  - **B0** Qwen3-VL-235B-A22B (via HolisticAI proxy API)
  - **B1** Qwen3-VL-4B (local on UCL Condense A100)
  - **B2** Gemma3-VL `google/gemma-3-4b-it` (local, added per our 2026-05-14
    sync as cross-family 4B-parity robustness check vs B1)
- Six observation modes per cell: DOM / SoM / Vision / phantom-text /
  phantom-prompt / phantom-SoM
- 36 conditions total (= 2 sites × 3 baselines × 6 modes); learned-router
  Pass-2 (6 conditions) runs sequentially after Pass-1 completes
- **Estimated wallclock**: ~1-2 weeks on the A100 (sequential per-site to
  avoid shared-account collision across baselines)

**Where it's running**:

- UCL Condense A100 VM (`a100-jiaming-test`, A100-PCIE-40GB dedicated allocation)
  with VWA Docker stack self-hosted on the VM (canonical paper-grade host
  per our 2026-05-15 migration from DGX→quark→Tailscale stack)

**Pre-fire prep that landed** (since 2026-05-14 sync):

- Cross-AI audit cascade closure — used `/stress` workflow (Claude + codex CLI
  + Gemini CLI as three independent reviewers) for 14/14 §A2 design-layer
  audits (research question framing / comparison design / power & sample /
  evidence-claim coupling / router operationalization / external validity /
  confound register / preregistration / reporting & ethics / prose↔code
  integrity); all closed with substantive fixes per `master_bug_catalog.md`
  + `实验笔记 §200-§224`. This was the bulk of my work post our sync.
- Pre-registration is locked at Git tag `preregistration-locked`
  (commit SHA `<FILL_AT_FIRE_TIME>` 2026-05-18). Covers 14 commit decisions
  in `preregistration.md §6 §(a)` — FE-pool estimand over 6 cells (decision
  "3A"), δ=1.0pp threshold, K-of-N transparency-only, smoke-gate rules,
  3-axis cost-latency canonical estimand (cost = raw billed / latency =
  retry-adjusted / raw-latency = sensitivity), B2 cross-family claim-tier
  gate, etc.

**OSF DOI**: minting ~1-2 weeks after Pass-1 baseline data completes
(per `osf_lock_manifest.md §3` 8-step workflow). DOI will be the
permanent external witness for the pre-registration; the local Git tag
covers interim period.

**Paper writing**: following our 2026-05-14 directive ("paper writing
交给 advisor, student focus = experiment execution"), I held paper §1-§8
prose finalization for post-data. Codex prose round queued.

**No action required** — this is purely FYI. I'll send a follow-up
when Pass-1 baseline data completes (around 2026-06-01 ± a few days,
depending on per-cell wallclock).

Best,
Jiaming

P.S. If anything in the methodology / scope / framing reads off and
you want me to course-correct mid-flight, please let me know. The
fire is paused-able at any cell boundary if there's an integrity
concern.

---

## Send checklist

- [ ] Replace `<FILL_AT_FIRE_TIME>` with actual Git SHA from `preregistration-locked` tag (Claude will paste the exact SHA into chat at fire event)
- [ ] Verify Maria + Zekun email addresses
- [ ] Send via standard email client
- [ ] (optional) Save sent copy to `.witness/email1_fire_notification_2026-05-18.sent.eml` for record (gitignored, local-only)
- [ ] (optional, only if advisor replies with substantive note) Save reply to `.witness/email1_fire_notification_2026-05-18.reply.eml` (gitignored)

## Cross-references

- `osf_lock_manifest.md §3` — 8-step DOI workflow (B-1570 updated 2026-05-18)
- `preregistration.md §6` — Internal witness (Git refs) + External witness (OSF DOI) two-layer
- `osf_deposit_package_manifest_2026-05-18.md` — Bundle pre-staged for OSF deposit
- 实验笔记 §200-§224 — Cross-AI audit cascade chronicle
- `_status/issues/issue_advisor_sync_2026-05-14.md` — 2026-05-14 sync 收口 record

## Email 2 retirement note

Prior version of this draft contained a formal pre-registration witness
section requesting a 1-line confirmation reply ("I witness pre-registration
of phantom-SoM gating hypotheses (H1-H3 + H10) and the 14 lock decisions
as of Git SHA <SHA> on <date>"). That section was **retired 2026-05-18**
per user analysis: with OSF DOI mint planned (~1-2 weeks post-Pass-1-data),
the advisor email witness is structurally redundant. The OSF DOI is a
cryptographically stronger witness (public + immutable + content-hashed
+ machine-verifiable) than advisor email (private + human-attested +
non-cryptographic). `witnessed_by:` field in `preregistration.md`
frontmatter will be populated with "Git tag `preregistration-locked` +
OSF DOI <to-be-assigned>" at OSF mint event, replacing the original
"witnessed_by: <advisor name>" plan. Per B-1570 doctrine 2026-05-18,
advisor email is now strictly optional collateral.
