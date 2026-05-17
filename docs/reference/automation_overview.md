# P79 Automation + Notification + Cleanup Architecture

**Last updated**: 2026-05-09 after audit (A)-(G).
**Audience**: self-only (paper-grade rerun protocol).

This is the canonical map of every auto-firing thing in the P79 repo. If
something happens without you typing it, it lives here.

## Overview — 6 layers

```
┌─────────────────────────────────────────────────────────────┐
│                    YOU TYPE: make launch                     │
│                              │                               │
│  ┌───────────────────────────┴────────────────────────────┐  │
│  │ L1 LAUNCH PROTOCOL (one-shot)                           │  │
│  │   launch.sh → cell-md auto-create → glm pre-launch     │  │
│  │   check → nohup queue script → 30s post-hook refresh   │  │
│  └─────────────────────┬───────────────────────────────────┘  │
│                        │                                       │
│  ┌─────────────────────┴───────────────────────────────────┐  │
│  │ L2 RUNNER + WATCHDOG (continuous, paired with run)       │  │
│  │   experiment_watchdog.py 6-layer auto-clean              │  │
│  │   (REPORT / IDLE / SESSION / COMPLETE / AUTO-CLEANUP)    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ L3 CRON SIDECAR (6 jobs, daemon-side)                    │  │
│  │   error-scan / glm-update-cells / glm-playbook /         │  │
│  │   check-links / myriad-watcher                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ L4 NTFY (single topic: p79-exp-dgx-spark)                │  │
│  │   alerting + heartbeat from L1+L2+L3 → your phone        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ L5 OBSIDIAN DATA LAYER (frontmatter single-source)       │  │
│  │   _status/{cells,issues,codex,section}/*.md →            │  │
│  │   4 Bases views + Git plugin sync                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ L6 LIVE STATE SURFACES                                   │  │
│  │   make active (秒) / cells.base (10min) /                │  │
│  │   PLAYBOOK §1+§2 (2h GLM narrative) / next_steps.md      │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

## L1 — Launch Protocol (`make launch`)

```bash
make launch BASELINE=B0 SITE=reddit MODE=phantom_som
```

Driver: `Makefile launch:` → `scripts/maintenance/launch.sh`. Steps:

| # | Action | Script | What can abort |
|---|---|---|---|
| 1 | Cell-md auto-create if missing | `launch.sh` inline | — |
| 2 | `make pre-launch-check` (C10 + F audit) | `Makefile` | git dirty / VWA SHA mismatch / HF model missing / Playwright ver / disk free / seed / **pytest** |
| 3 | GLM review of queue + active runs | `glm_pre_launch_check.py` (F37 fail-closed) | rc=2 BLOCK abort / rc=1 WARN interactive y/N |
| 4 | nohup queue script `RESET_BEFORE=1` | `scripts/queues/queue_*.sh` | site reset failure logged but non-fatal |
| 5 | Post-launch hook (sleep 30 + dual GLM refresh) | inline | (best-effort) |

**Exit codes**: 0 OK / 64 usage / 65 unknown mode / 2 hard-rule violation.

**Hard rules enforced** (paper-grade hygiene):
- Same site B0 XOR B1 (no cross-baseline contamination via shared user
  accounts). Detected by `glm_pre_launch_check` `pgrep -af run_experiment`.
- `RESET_BEFORE=1` mandatory per cell. Site reset via SSH chain
  DGX → quark Tailscale → Windows PowerShell `reset_vwa.ps1`.
- Queue script idempotent (`pgrep -f` skip if same site+mode running).

---

## L2 — Runner-side Watchdog (`experiment_watchdog.py`)

Started by queue script alongside runner. Per-condition lifecycle. Pushes
to ntfy topic. Functions invoked:

| Trigger | Action | Auto-clean? |
|---|---|---|
| `--report-interval-mins` (e.g. 60min) | post status (raw/adj SR + counts) | no |
| `--idle-alert-mins` (no new episode for N min) | ntfy priority=high | no |
| 3 consecutive episodes lacking `link 'Logout'` | `_auto_refresh_auth(site)` Playwright re-login + write `.auth/<site>_state.json` + purge contaminated episode summaries (B-854 2026-05-17: prose updated from "digests" → "episode summaries"; post-A1.15 Chunk a digest pipeline retired) | **yes** |
| `--notify-completion` (per-condition done) | post-condition analysis + paper figures regen | no |
| Episode tagged `not_logged_in` / corrupt summary | delete summary file → runner retries | **yes** |
| Cross-restart contamination tracking | persist state file → resume cleanup | **yes** |

**6-layer Cross-Component Auto-Clean Pipeline** (笔记 §95 + §107; reframed B-766 A1.15 cold-start 2026-05-17): detect → alert → refresh → cleanup → resume → verify. Layers 1-4 + 6 are watchdog-side explicit code (`experiment_watchdog.py:275-318` + `:1700-1816`); **layer 5 (resume) is runner-side explicit code** (`p79/experiment/runner/main.py:762 if self.resume and summary_file.exists()` — runner re-runs task when watchdog deletes its `summary_v2.json`). Layer 6 (verify) re-uses layer 1 code on next task's step_000 DOM check, so verify is delayed (not immediate-after-refresh) but explicit. Edge cases (layer 5 runner-already-exited / layer 6 no-next-task) disclosed in paper §4.X.15 with post-data Supp Table S-layer56-edge planned. State file `--state-file` survives watchdog restarts so contaminated-episode tracking is not lost on signal/SIGTERM (B-393 + B-762 `--recover-and-quarantine` partial-state recovery).

---

## L3 — Cron Sidecar (6 jobs)

Defined in `scripts/maintenance/crontab.txt`. Activate with
`crontab scripts/maintenance/crontab.txt`. All wrapped by
`notify_on_fail.sh` so non-zero exit → ntfy.

| Cron | Cadence | What | Output |
|---|---|---|---|
| `error-scan` | `*/5 min` | Scan `logs/{B*,queue*,watchdog*}.log` + `logs/cron/*.log` for traceback / OOM / NOT_LOGGED_IN / **B-22 magento_redirect_loop** / **B-81h cutlass_kernel_miss** / **sm_121 nvrtc_arch** / **F23 fp_adjust_error** + **disk-free probe (<50GB)** + **tailscale BackendState** | `logs/cron/error_scan.json` → §2.5 |
| `glm-update-cells` | `*/10 min` | Sync `_status/cells/cell_*.md` frontmatter from `condition_summary_v2.json`. Re-run detect via `last_run_id` change → archive history. PID liveness check (dead PID auto-clear). `cell_changelog.jsonl`. **status active→done flips trigger `make analysis FAST=1` (audit B)** | `logs/cron/cell_changelog.jsonl` |
| ~~`glm-refresh-playbook-s2`~~ | ~~`15,45 min`~~ — **RETIRED 2026-05-13** (crontab.txt comment) | ~~Fast §2 refresh.~~ Manual via `make glm-refresh-playbook --section 2` only. | ~~`PLAYBOOK.md §2`~~ |
| ~~`glm-refresh-playbook`~~ | ~~`0 */2 hour`~~ — **RETIRED 2026-05-13** (crontab.txt) + Makefile post-hook trim 2026-05-17 (A1.15b B-842, commit `d6dd949`) | ~~Full §1+§2 GLM call.~~ Manual via `make glm-refresh-playbook` only; no cron + no post-hook auto-trigger. PLAYBOOK retire planned. (B-854 doc-drift fix 2026-05-17) | ~~`PLAYBOOK.md §1+§2`~~ |
| `check-links` | `0 0 * * 0` (weekly Sun) | Scan `docs/` for broken wikilinks + path refs | `logs/cron/dead_links_<date>.log` |
| `myriad-watcher` | `*/5 min` | SSH chain DGX → quark → Myriad qstat → diff state → ntfy on **NEW/CHG/GONE** events. **3 consecutive SSH failures → ntfy high (F36)**. **GONE event with matching `GONE_HOOKS` prefix → fire `auto_pull_myriad_cell.sh` (audit A)** | `logs/cron/myriad_state.json` + `logs/cron/myriad_watcher.log` |

### Auto-pull flow (audit A+C, post-2026-05-09)

```
Myriad job 336424 finishes (state r → gone)
  ↓ cron tick (within 5 min)
myriad_watcher.py detects GONE event
  ↓ GONE_HOOKS["cellg_rev_"] matches → spawn:
auto_pull_myriad_cell.sh 336424 cellg_rev_ stage2c_cellg_rev_reddit_reverse_myriad …
  ↓ Phase 1
  SCP via DGX→quark→Myriad chain (cat | base64 -w0)
    → results/mechanistic/<remote_dir>/{env_snapshot, results.json, summary.md, manifest.json, curves.png, condition_summary_v2.json}
  ↓ Phase 2
  python3 scripts/analysis/validate_run.py --run-dir <local> --strict
    → write validation_report.json
    → rc=0 ✅ pass / rc=1 ❌ FAIL → cell-md status=quarantined
  ↓ Phase 3
  Update cell_<id>.md frontmatter (quarantined flag if applicable)
  ↓ Phase 4
  nohup make analysis FAST=1 & (audit B chain — figures regen)
  ↓ Phase 5
  push_ntfy "Cell pulled: cellg_rev_ | files=N validate=…"
```

---

## L4 — NTFY (`p79-exp-dgx-spark` topic)

| Source | Trigger | Priority |
|---|---|---|
| `experiment_watchdog.py` | REPORT / IDLE / COMPLETE / SESSION / DIGEST | varies |
| `myriad_watcher.py` | NEW/CHG/GONE state events + auto_pull dispatch summary | default |
| `myriad_watcher.py` | 3 consecutive SSH failures (F36) | high |
| `glm_playbook_refresh.py` | 3 consecutive GLM-API failures (audit D) | high |
| `error_scan.py` | disk free <50GB (2 consecutive ticks) | high |
| `error_scan.py` | tailscale BackendState != Running (3 consecutive ticks) | high |
| `glm_pre_launch_check.py` | (rc=1 WARN / rc=2 BLOCK on launch) | — (synchronous) |
| `auto_pull_myriad_cell.sh` | per-cell pull complete (success or partial) | default |
| `auto_pull_myriad_cell.sh` | 0 files pulled / SSH chain dead | high |
| `notify_on_fail.sh` | any cron job non-zero exit | high |

**Subscriber side**: phone app (or ntfy.sh web) subscribed to topic.

**Rate-limited**: each consecutive-fail counter (`SSH_FAIL_FILE`,
`GLM_FAIL_FILE`, `DISK_FAIL_FILE`, `TAILSCALE_FAIL_FILE`) prevents
notification storms — single ntfy after N consecutive ticks, reset on
recovery.

---

## L5 — Obsidian Data Layer

**Vault root**: `docs/`. Git plugin (Windows side) auto-pull every 10 min.

```
docs/
├── _status/cells/cell_*.md         (cron-managed: progress / sr_raw / pid / history / last_run_id)
├── _status/codex/codex_*.md        (codex lifecycle, manual)
├── _status/issues/issue_*.md       (issue triage, manual)
├── _status/section{1..8}_*.md      (paper section status, manual)
├── cells.base                       (Bases: status icons + progress bar + cell labels)
├── status.base                      (Bases: section status board)
├── codex.base                       (Bases: codex task lifecycle)
├── issues.base                      (Bases: issue ledger)
├── checkpoints/PLAYBOOK.md          (gitignored, GLM-managed §1+§2; manual §3-§10)
├── checkpoints/{next_steps, paper_planning, ADVISOR_SYNC, 实验笔记}.md
├── checkpoints/paper_drafts/section{1..8}_*.md + paper.bib
├── checkpoints/_status/             (mirror of _status/, kept for backward-compat)
├── checkpoints/phantom_space.canvas
├── checkpoints/paper_section2_framework.canvas
└── checkpoints/experiment_matrix.canvas
```

### Cell-md schema (auto-managed fields)

| Field | Type | Source |
|---|---|---|
| `type` | "cell" | manual create |
| `baseline` / `site` / `mode` | str | manual create |
| `status` | pending → active → done → quarantined | cron auto-flip |
| `progress` | 0-100 (% N done) | cron |
| `n` / `expected_n` | int | manual / cron |
| `sr_raw` | float | cron |
| `sr_adj` | float | manual after `make analysis` |
| `last_run_id` | str | cron (re-run trigger) |
| `pid` | int | cron pgrep + liveness check |
| `history` | list | cron append on done flip |
| `finalized_at` | ISO date | cron on done flip |
| `quarantine_reason` / `quarantine_at` | str | auto_pull post-validate-strict (audit C) |
| `blocker` / `eta` | str | manual |

### Sync transport

- **Live state (Tailscale scp, ~1min latency)**: `PLAYBOOK.md` +
  `_status/cells/*.md` + `results/phantom_paper/{auroc_cross_condition,phantom_lift}.md`.
- **Source of truth (git, ~10min latency)**: paper drafts / 实验笔记 /
  paper_planning / next_steps / ADVISOR_SYNC / code / `_status/{issues,codex,section}/*.md`.
- **Force pull**: Windows `Ctrl+P` → "Obsidian Git: Pull"; PowerShell
  `Start-ScheduledTask -TaskName "Pull PLAYBOOK from DGX"`.

---

## L6 — Live State Surfaces

| Granularity | Where |
|---|---|
| Right-now (sec) | `make active` (ps + episode mtime scan) |
| ~5min snapshot (errors) | `PLAYBOOK §2.5` (cron `error-scan`) |
| ~10min snapshot (cells) | `cells.base` Obsidian (cron `glm-update-cells`) |
| Today narrative + 瓶颈 | `PLAYBOOK §1` (GLM 2h cron) |
| Cron health + cell changelog | `PLAYBOOK §2` (GLM 30min/2h cron) |
| Forward actions / chains / horizon | `next_steps.md` (manual) |
| Paper strategy / theory / decision log | `paper_planning.md` (manual, weekly) |
| Append-only chronicle | `实验笔记.md` (manual, append-only) |
| Pre-meeting decisions | `ADVISOR_SYNC.md` (manual, per-meeting) |

---

## What's NOT automated (manual brain decisions)

| Trigger | File | Action |
|---|---|---|
| New issue | 1 | `_status/issues/issue_*.md` frontmatter |
| New finding | 1 | append `实验笔记.md §X` chronicle (`#finding` tag) |
| Cross-X pattern | +1 | `paper_planning.md §3` |
| Framework decision | +1 | `paper_planning.md §19` + `ADVISOR_SYNC.md §2` |
| Paper prose update | manual | `paper_drafts/` codex round |
| **manifest grade promotion** | manual | `results/phantom_paper/run_manifest.yaml` mark new cells `paper-grade` (16-cell rerun prerequisite) |

---

## Audit (A)-(G) summary (2026-05-09)

| # | Gap | Fix location |
|---|---|---|
| **A** | Myriad cell completion 数据滞留 | `myriad_watcher.GONE_HOOKS` + `auto_pull_myriad_cell.sh` |
| **B** | Cell finalize 不重生 figures | `glm_cell_autoupdate` 在 status 翻转时 `make analysis FAST=1` |
| **C** | `validate_run.py --strict` 没 cron | `auto_pull_myriad_cell.sh` Phase 2 → `status: quarantined` on fail |
| **D** | GLM API fail 静默 | `glm_playbook_refresh.GLM_FAIL_FILE` consecutive counter + ntfy high |
| **E** | 磁盘 + Tailscale 健康度无监控 | `error_scan._check_disk()` + `_check_tailscale()` + ntfy thresholds |
| **F** | pre-launch-check 不跑 pytest | `Makefile pre-launch-check` 加 `timeout 60 pytest tests/ -x` |
| **G** | error-scan 缺 P79-specific patterns | `error_scan.PATTERNS` 加 magento_redirect_loop / cutlass_kernel_miss / nvrtc_arch / fp_adjust_error |

---

## Open follow-ups (not done in (A)-(G))

| H | A1 prereg lock 未 enforce 在 pre-launch | `make pre-launch-check` 加 `grep status:.*locked preregistration.md` |
| I | artifacts 不 prune | weekly cron 删 N>30 天的 `artifacts/{step,dom,screenshot}.{html,jpg,png}` |
| J | ntfy 端到端 heartbeat | weekly cron 推 "🟢 ntfy alive" 一条 |

These are nice-to-have; not blockers for 16-cell rerun.

---

## See also

- **`docs/reference/launch_checklist.md`** — the human-runnable
  step-by-step protocol for a 16-cell paper-grade rerun, including
  the manifest grade promotion step that this overview only points at.
- `docs/checkpoints/PLAYBOOK.md` §6 cron sidecar table (live status)
- `docs/reference/glm_quark_myriad_sync.md` — alt: GLM-driven SCP from quark
- `docs/reference/DGX_SPARK_MACHINE_QUIRKS.md` — DGX-specific environment
- `docs/checkpoints/实验笔记.md` §95 + §107 — watchdog 6-layer history
- `Makefile` — `make help` for all manual targets
