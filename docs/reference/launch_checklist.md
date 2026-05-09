# Launch Protocol Checklist (16-cell paper-grade rerun)

**Last updated**: 2026-05-09
**Audience**: self-only
**See also**: `docs/reference/automation_overview.md` for the full
automation architecture; this file is the **human-runnable checklist**
that walks you through a paper-grade rerun.

This checklist is the **single source of truth** for what a paper-grade
launch requires. The `make pre-launch-check` target enforces a subset
programmatically; everything below covers the gaps it can't (yet) check.

---

## 0. Pre-flight (one-time per rerun batch)

Run these **before** the first `make launch` of the batch.

- [ ] **Pull latest from origin** (paper-grade reruns must not start on
      a stale branch):
      ```bash
      git pull origin master
      git status --porcelain --untracked-files=all   # must be empty
      ```
- [ ] **Preregistration locked** (audit A1): frontmatter `status: locked`
      + git tag `prereg-<date>` + advisor witness email + OSF DOI.
      ```bash
      grep "^status:" docs/checkpoints/pre_run/preregistration.md
      git tag --list 'prereg-*'
      ```
      If still draft: do not start a paper-grade rerun. Run with
      `RESET=0 FORCE_NO_CHECK=1` only for in-flight smoke tests, never
      paper-claim cells.
- [ ] **Locked versions match** (audit A5+F8):
      `docs/checkpoints/pre_run/locked_versions.md` records the
      Playwright + Chromium + VWA SHA + HF model SHA. `make
      pre-launch-check` verifies these programmatically; if you
      bumped any, update the lock doc first and re-witness.
- [ ] **Quark + Tailscale + Cisco**: VWA Docker stack reachable from
      DGX:
      ```bash
      curl -sI http://100.95.81.103:9999/  # reddit, expect 200
      curl -sI http://100.95.81.103:9980/  # cls
      curl -sI http://100.95.81.103:7770/  # shopping
      ```
      If any fail: user opens quark + Cisco AnyConnect + checks Docker.
      Cron `myriad_watcher` will ntfy after 3 SSH chain failures (F36).
- [ ] **GLM API key present**: `cat .auth/glm` has valid key. Cron
      `glm-playbook` will ntfy after 3 consecutive API failures
      (audit D).
- [ ] **Disk free > 50 GB**: `df -BG .` — cron `error_scan` ntfies
      below threshold (audit E). 16 cells × ~10 GB each = ~160 GB
      headroom needed; if low, prune `logs/` or archive old `results/`.

---

## 1. Manifest grade promotion (PER CELL, before launch)

🚨 **The hidden prerequisite that's NOT in `make pre-launch-check`**.

Per `audit_overview.md`: the registry filter (post-F01) defaults to
`['paper-grade']`. A cell whose run is launched but **not entered in
`results/phantom_paper/run_manifest.yaml`** with `grade: paper-grade`
will produce data the analysis pipeline ignores.

### What to do

For each cell about to launch:

```bash
# 1. Resolve canonical run_dir name (your queue script will create it
#    in results/visualwebarena/phase1/<RUN_ID>):
RUN_ID="B0_phantom_som_reddit_$(date +%Y%m%d)"

# 2. Add manifest entry BEFORE kicking off the launch:
cat >> results/phantom_paper/run_manifest.yaml <<EOF
  - baseline: B0
    site: reddit
    mode: P-SoM
    run_dir: ${RUN_ID}
    condition_subdir: phase1_phantom_som_router_0
    expected_n: 210
    grade: paper-grade
    notes: "16-cell rerun batch 2026-05-09"
EOF
```

### Why must it be `paper-grade` (not `paper-grade-pre-bug`)?

Pre-bug is a sensitivity-check tier; main paper figures use
`paper-grade` only. Mark cells `paper-grade` if they were launched
with **all of**:
- Phase A bug fixes committed (commit ≥ `3c15cd7`).
- Lock SHA chain verified (`make pre-launch-check` rc=0).
- `RESET_BEFORE=1` (no contaminated site state from previous cells).
- Watchdog auto-clean enabled.

Drop to `paper-grade-pre-bug` if any of those conditions weren't met,
or for the legacy 2026-04 cells.

### Verify after launch

```bash
.venv/bin/python3 -c "
from scripts.analysis.lib.run_registry import get_cells
print(get_cells(baseline='B0', site='reddit', mode='phantom_som'))
"
# Should print [CellSpec(...)]; if empty, manifest update missed.
```

---

## 2. The launch command (per cell)

Pre-flight done + manifest entry added → run:

```bash
make launch BASELINE=B0 SITE=reddit MODE=phantom_som
```

What `launch.sh` does (audit-traced flow):

1. **Auto-create cell-md** at `_status/cells/cell_b0_red_psom.md` (if
   missing) with frontmatter `status: pending`.
2. **`make pre-launch-check`** (C10 + F audit):
   - Git working tree clean (no untracked, no diff).
   - VWA submodule SHA = lock.
   - HF Qwen3-VL-4B revision present.
   - Playwright 1.58.0.
   - Disk free > 20 GB.
   - Seed = 42 in base config.
   - **pytest passes (60s timeout, fail-fast)** ← audit F.
3. **`glm_pre_launch_check.py`**: GLM reads queue script + active
   processes + recent logs. rc=2 BLOCK abort / rc=1 WARN interactive
   y/N / rc=0 OK. Fail-closed when GLM unreachable (audit F37); set
   `P79_ALLOW_GLM_FAIL=1` only for dev runs.
4. **`nohup queue_phantom_som.sh B0 reddit`** in background with
   `RESET_BEFORE=1`. PID printed.
5. **30-second post-hook**: dual GLM refresh
   (`glm-update-cells` + `glm-refresh-playbook`) so PLAYBOOK §1
   narrative reflects the new run within ~30 seconds.

---

## 3. During the run (continuous monitoring)

| Frequency | What | Where |
|---|---|---|
| Real-time | Active processes + episode mtime | `make active` |
| Every 5 min | Errors / OOM / NOT_LOGGED_IN / Magento 302 / cutlass / sm_121 / fp_adjust_error | cron `error-scan` → PLAYBOOK §2.5 |
| Every 5 min | Disk free + Tailscale BackendState (audit E) | `error_scan.json system_health` |
| Every 5 min | Myriad qstat + state diff (if Stage 2 cells running) | cron `myriad-watcher` |
| Every 10 min | Cell frontmatter sync (progress / pid / sr_raw / history) | cron `glm-update-cells` |
| Every 30 min | PLAYBOOK §2 fast refresh (cron health + dead-links + ntfy fails) | cron `glm-refresh-playbook-s2` |
| Every 2 hours | PLAYBOOK §1 narrative + §2 board | cron `glm-refresh-playbook` |
| Continuous | REPORT / IDLE / SESSION-HEALTH watchdog | runner-paired `experiment_watchdog.py` |

### When to expect an alert

Single ntfy topic: `p79-exp-dgx-spark`. Alerts:

- **High priority**: IDLE (no episode in N min), 3-consec SSH chain
  fail (F36), 3-consec GLM API fail (audit D), disk free <50 GB
  (E), tailscale non-running (E), 0-files SCP failure (auto_pull
  audit A), watchdog auth refresh failed.
- **Default priority**: per-condition COMPLETE, Myriad state changes
  (NEW/CHG/GONE), per-cell auto_pull complete.

### When to act manually

| Symptom | Action |
|---|---|
| 3 consec NOT_LOGGED_IN despite watchdog auth refresh | Quark side: re-launch Docker reddit/cls/shop containers; re-export auth via Playwright record |
| Disk free crashing | Prune `logs/B0_*.log` (>30 days), archive `results/visualwebarena/phase1/<old_run>/artifacts/` to S3 |
| Myriad SSH chain dead 30 min+ | Quark side: reconnect Cisco AnyConnect; check `tailscale status` |
| Cell quarantined (audit C validate-strict fail) | Inspect `validation_report.json`; common: missing summary fields, low SR vs prior cell, unexpected fp_reason |

---

## 4. Per-cell completion (autonomous, audit B chain)

When a cell's runner finishes (episodes ≥ expected_n):

1. **`glm-update-cells` cron** (next 10-min tick) detects active→done
   flip → frontmatter `status: done`, `finalized_at`, history append.
2. **Audit B trigger**: `nohup make analysis FAST=1 &` fires →
   aggregator (`aggregate_phantom_lift.py` + `aggregate_phantom_meta.py`
   + `sr_fp_per_mode.py`) + figures regen. Logged to
   `logs/cron/post_finalize_analysis.log`.
3. **Manual brain decision** (NOT automated):
   - Append `实验笔记.md §X` chronicle entry with finding (`#finding`).
   - Update `paper_planning.md §3` if cross-X pattern emerges.
   - Update `paper_drafts/` if section-level prose needs revision.

For Stage 2 mechanism cells (Myriad-launched):

1. **Myriad job finishes** (state r → gone in `qstat`).
2. Cron `myriad-watcher` (next 5-min tick) detects GONE event →
   ntfy + dispatches `auto_pull_myriad_cell.sh` (audit A).
3. **Auto-pull script**: SCP via SSH chain → `validate_run.py
   --strict` → cell-md frontmatter update (`quarantined` if validate
   fails) → `make analysis FAST=1` → ntfy "Cell pulled".

---

## 5. Post-batch (after all 16 cells done)

- [ ] All 16 cell-md `status: done` (none `quarantined`):
      ```bash
      grep -l "status: quarantined" docs/checkpoints/_status/cells/cell_*.md
      ```
      If any quarantined: investigate `validation_report.json`, fix
      bug, re-launch with explicit `RESET=1`.
- [ ] **`make analysis`** (full, not FAST=1) regenerates all
      aggregators + figures. Should produce **paper-grade** numbers
      (not pre-bug; the F01 + F40 chain enforces this).
- [ ] **`scripts/analysis/sensitivity_loo_meta.py`**: re-run; verify
      P-text drop-in arm robustness post-rerun (was FRAGILE on 3 cells;
      may stabilize on 16).
- [ ] **`scripts/analysis/power_analysis.py`** with observed SR:
      regenerate `docs/analysis/cross_sites/power_analysis.md`.
- [ ] **Update `section8_limitations.md`** §8.5 + §8.6 with final
      numbers (mechanism cells F/G results + sparse-mechanism caveat
      reaffirmed).
- [ ] Append `实验笔记.md §X` chronicle: full 16-cell summary.
- [ ] **Push 16-cell rerun commit batch** + tag `paper-grade-rerun-<date>`.

---

## 6. Rollback / abort

If contamination detected mid-batch (cross-baseline same site, auth
drift not auto-recovered, etc.):

```bash
# 1. Kill runner + watchdog cleanly:
pgrep -af "run_experiment.*<SITE>" | awk '{print $1}' | xargs kill -TERM

# 2. Mark cell as dirty:
.venv/bin/python3 scripts/maintenance/mark_cell_dirty.py \
    --cell docs/checkpoints/_status/cells/cell_<id>.md \
    --reason "contamination" --commit-sha $(git rev-parse HEAD)

# 3. Reset site:
RESET_BEFORE=1 bash scripts/maintenance/reset_vwa_sites.sh <SITE>

# 4. Update manifest entry: grade: archived → keep history; or remove
#    entirely if never paper-grade.

# 5. Re-launch with fresh queue (skip-restart guard cleared):
make launch BASELINE=<bx> SITE=<sx> MODE=<mx> FORCE_RESTART=1
```

---

## Quick reference

```bash
# Pre-flight
make pre-launch-check                              # programmatic gates
git pull && git status --porcelain --untracked-files=all   # clean tree

# Manifest update
$EDITOR results/phantom_paper/run_manifest.yaml

# Launch
make launch BASELINE=B0 SITE=reddit MODE=phantom_som

# Monitor
make active                                        # real-time
tail -f logs/cron/error_scan.log                   # 5min
ntfy.sh/p79-exp-dgx-spark                          # phone

# Post-cell
# (audit B chain auto-fires; if needed:)
make analysis FAST=1

# Post-batch
make analysis                                      # full
.venv/bin/python3 scripts/analysis/sensitivity_loo_meta.py
.venv/bin/python3 scripts/analysis/power_analysis.py --baseline-sr <observed>
```

---

## Audit B/C/D/E/F/G chain in this checklist

- **A** (auto-pull) → §4 Stage 2 mechanism cells
- **B** (analysis trigger on done) → §4 per-cell completion
- **C** (validate-strict gate) → §4 + §5 quarantine handling
- **D** (GLM fail counter) → §3 high-priority alerts
- **E** (disk + tailscale) → §0 pre-flight + §3 alert table
- **F** (pytest gate) → §2 launch command flow
- **G** (P79-specific patterns) → §3 error monitoring

Open follow-ups (not in (A)-(G)):
- **H** A1 prereg lock enforcement in `make pre-launch-check` (currently §0 manual check)
- **I** weekly artifacts prune cron
- **J** weekly ntfy heartbeat
