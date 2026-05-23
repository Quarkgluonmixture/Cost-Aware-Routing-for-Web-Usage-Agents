#!/usr/bin/env bash
# fire6_monitor.sh — Phase-1 paper-grade fire health monitor (cron + ntfy).
#
# Zero Claude-quota by design: pure shell + ntfy push. Run ON the fire host
# (A100) from the repo root. Two modes (registered as two cron lines):
#
#   healthcheck  (every 30 min) — ntfy ONLY on anomaly; silent when healthy.
#   heartbeat    (daily)        — ntfy a one-line progress summary (reassurance).
#
# Anomalies detected (healthcheck):
#   - orchestrator process gone (Pass-1 either COMPLETED normally OR died — the
#     ntfy says "verify via status"; either way the operator wants to know).
#   - orchestrator up but NO step JSONL written in the last 60 min (stall).
#   - a B1/B2 (local-model) runner is up but GPU < 500 MiB (model load stuck).
#     (B0 = proxy API → GPU idle is NORMAL; never alerts during the B0 phase.)
#   - FATAL / quota-exhaust / PaperGradeAbort / non-zero chain rc in the fire or
#     active runner log.
#
# Usage (cron, on A100):
#   */30 * * * *  cd /home/ubuntu/workspace/p79 && bash scripts/maintenance/fire6_monitor.sh healthcheck
#   5 9 * * *     cd /home/ubuntu/workspace/p79 && bash scripts/maintenance/fire6_monitor.sh heartbeat
set -uo pipefail
cd "$(dirname "$0")/../.." 2>/dev/null || cd /home/ubuntu/workspace/p79
NTFY="${NTFY_TOPIC:-p79-exp-dgx-spark}"
URL="https://ntfy.sh/${NTFY}"
MODE="${1:-healthcheck}"
# Auto-detect the NEWEST fire log across BOTH naming families — fire6_phase1a*.log
# (B-1803 task-4 re-fire era) AND fire6_relaunch_*.log (RESUME_MISSING relaunch,
# B-1825). B-1827 (Fire-6 relaunch, 2026-05-22): the prior glob only matched
# fire6_phase1a* → it pinned the OLD aborted log (whose rc=1 FAIL is permanent) and
# SILENTLY missed the live fire6_relaunch_* log entirely — both a false-positive
# "fatal/abort in fire log" every tick AND a real monitoring blind spot (the live
# relaunch was not being watched at all). Glob both; newest by mtime wins.
# B-1840 (2026-05-23): glob MUST include the live orchestrator chain log
# (queue_phase1_{cls,red}_*.log written by queue_chain.sh). Pre-fix glob only matched
# the launcher-era fire6_phase1a*/fire6_relaunch_* family → ls -t pinned a STALE aborted
# relaunch log (e.g. fire6_relaunch_20260522_124915.log line 61 'rc=1 cascade halt') and
# grep rc=[1-9] false-positived "FATAL/abort in fire log" every 30-min tick. Newest-by-mtime
# wins → during a live fire the today chain log is selected; old globs remain as fallback.
FIRELOG="$(ls -t logs/queue_phase1_cls_*.log logs/queue_phase1_red_*.log logs/fire6_phase1a*.log logs/fire6_relaunch_*.log 2>/dev/null | head -1)"
[ -z "$FIRELOG" ] && FIRELOG="logs/fire6_phase1a.log"
RESULTS="results/visualwebarena/phase1"

# B-1840 (2026-05-23): the real long-lived orchestrator is queue_chain.sh —
# `queue_phase1_paper_grade.sh launch` is a launcher that spawns queue_chain.sh then
# EXITS, so matching ONLY the launcher false-positived "orchestrator GONE" every 30-min
# tick mid-fire (§0 line 53 canary-era same naming-drift root cause). Match queue_chain.sh
# (live fire) OR the launcher (brief launch window). orch DOWN at fire END (chain exits
# after 18 conditions) is still a real "COMPLETED or DIED" signal — intended.
_orch_up()      { pgrep -f 'queue_chain.sh' >/dev/null 2>&1 || pgrep -f 'queue_phase1_paper_grade.sh launch' >/dev/null 2>&1; }
_recent_step()  { find "$RESULTS" -name '*steps*.jsonl' -newermt '-60 min' ! -path '*smoke*' 2>/dev/null | head -1; }
_local_runner() { pgrep -af 'run_experiment.py' 2>/dev/null | grep -qE '/exp_v2_B[12]_'; }
_gpu_mib()      { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 0; }
_cur_cond()     { pgrep -af 'run_experiment.py.*--config' 2>/dev/null | grep -oE 'exp_v2_B[012]_[a-z_]+_(classifieds|reddit|shopping)' | head -1; }
_n_done()       { find "$RESULTS" -name 'condition_summary_v2.json' -path '*_2*' ! -path '*smoke*' 2>/dev/null | wc -l | tr -d ' '; }

if [[ "$MODE" == "heartbeat" ]]; then
  orch=$(_orch_up && echo up || echo DOWN)
  step=$(_recent_step >/dev/null && echo '<60min' || echo STALE)
  msg="Fire-6 daily heartbeat: orch=${orch} cur=$(_cur_cond) cond_done=$(_n_done) last_step=${step} gpu=$(_gpu_mib)MiB"
  curl -s -m 15 -H "Title: Fire-6 heartbeat" -d "$msg" "$URL" >/dev/null 2>&1
  exit 0
fi

# ---- healthcheck (anomaly-only) ----
ALERT=""
if ! _orch_up; then
  ALERT+="orchestrator GONE (queue_chain.sh) — Pass-1 COMPLETED or DIED, verify: pgrep -af queue_chain.sh + chain log tail; "
else
  if [[ -z "$(_recent_step)" ]]; then
    ALERT+="orch up but NO step in 60min (stall?); "
  fi
  if _local_runner && [[ "$(_gpu_mib)" -lt 500 ]]; then
    ALERT+="B1/B2 runner up but GPU<500MiB (model load stuck?); "
  fi
fi
if tail -300 "$FIRELOG" 2>/dev/null | grep -qiE 'FATAL|quota exhaust|PaperGradeAbort|non-zero|rc=[1-9]'; then
  ALERT+="FATAL/abort in fire log; "
fi
RL=$(ls -t logs/*_runner.log 2>/dev/null | head -1)
if [[ -n "${RL:-}" ]] && tail -150 "$RL" 2>/dev/null | grep -qiE 'PaperGradeAbort|quota exhaust|CUDA out of memory'; then
  ALERT+="runner-log fatal ($(basename "$RL")); "
fi

if [[ -n "$ALERT" ]]; then
  curl -s -m 15 -H "Priority: high" -H "Title: Fire-6 ALERT" -d "$ALERT" "$URL" >/dev/null 2>&1
fi
exit 0
