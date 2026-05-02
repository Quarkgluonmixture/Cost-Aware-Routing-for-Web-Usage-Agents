---
type: issue
category: backlog
status: backlog
priority: medium
action: watchdog stale-runner heartbeat (episode mtime > N min → SIGTERM + queue idempotent re-spawn)
---

# GPU contention deadlock detection

B1 cls stalled 4h 04-29 05:01-09:00, kernel wait 不报 progress. runner state R but ep_pol kernel wait → no progress detection. 当前 workaround: manual stop+restart `bash queue_phantom_som.sh B1 classifieds`.
