# scripts/maintenance/ — Category Index

33 maintenance scripts categorized by function. Top-level files preserved
to avoid breaking external references; only `glm/` moved into subdir
(see "Reorg history" below).

## 🤖 GLM Sidecar Automation (`glm/` subdir)

| File | Purpose | Trigger |
|---|---|---|
| `glm/glm_diagnosis_sidecar.py` | Deep failure attribution per episode (existing) | watchdog post-condition |
| `glm/glm_batch_digest.py` | Batch failure digest (existing) | post-condition |
| `glm/glm_cell_autoupdate.py` | Sync `_status/cells/*.md` frontmatter from condition_summary | `make glm-update-cells` / cron |
| `glm/glm_playbook_refresh.py` | Synthesize PLAYBOOK §6 critical path | `make glm-refresh-playbook` / cron |
| `glm/glm_pre_launch_check.py` | GLM reviews queue launch for hard-rule violations | manual / queue script hook |

## 🔬 One-off Probes / Audits (top-level, frequently cited in docs)

| File | Purpose |
|---|---|
| `probe_b01_b13_self_verify.py` | B-01/B-13 audit verification |
| `probe_b08_b06_self_replay.py` | B-08/B-06 self-replay |
| `probe_b37_api_determinism.py` | B-37 5-call API determinism (paper §4) |
| `probe_som_occlusion.py` | §100 SoM ground truth |
| `probe_tier10_dispatch_target.py` | Tier 10 dispatch effective-target |

## 🚦 Watchdog / Runtime

| File | Purpose |
|---|---|
| `experiment_watchdog.py` | Auto-clean + post-condition pipeline daemon |
| `active_processes.py` | `make active` real-time scan |
| `restart_watchdog.sh` | Restart watchdog after change |
| `trigger_watchdog_status.sh` | Manual watchdog trigger |
| `wait_for_reddit_then_rederive.sh` | Reddit completion → auto rederive |

## 📊 Data Operations

| File | Purpose |
|---|---|
| `clear_tasks.py` | Delete task results (summary/steps/artifacts/digest) — use this NOT `rm` |
| `rederive_episode_summary.py` | Rebuild episode summary from steps |
| `reeval_phase1.py` | Re-evaluate phase1 results |
| `digest_enrich.py` | Augment failure digest |
| `cleanup_logs.py` | Log rotation |
| `split_wa_tasks.py` | WA 480-task split (B0+B1) |
| `dead_link_check.py` | Markdown link validation |

## 🌐 Site Setup / Reset

| File | Purpose |
|---|---|
| `reset_vwa_sites.sh` | DGX → quark Powershell + defensive curl |
| `create_b1_classifieds_stub.py` | B1 classifieds stub creation |
| `retry_b1_single_task.sh` | Retry single B1 task |

## 🖼 Gallery / Assets

| File | Purpose |
|---|---|
| `annotate_screenshots.py` | SoM mark annotation |
| `generate_gallery.py` | Per-run gallery HTML |
| `refresh_gallery.sh` | Regenerate gallery |

## 🔄 Cross-Host Sync

| File | Purpose |
|---|---|
| `rsync_results_to_hub.sh` | DGX → hub |
| `rsync_results_from_hub.sh` | Hub → DGX |

## 🧪 Misc / Smoke

| File | Purpose |
|---|---|
| `check_disk_usage.sh` | Disk usage report |
| `run_one_vwa_episode.py` | Single-episode runner (dev) |
| `smoke_test_vwa.py` | VWA smoke test |
| `crontab.txt` | Opt-in cron config (`crontab crontab.txt` to install) |

## Reorg history

- **2026-05-02**: Added `glm/` subdir, moved 5 GLM scripts (low risk: only Makefile referenced these). `probe_*` kept top-level (referenced in 5 docs: 实验笔记 / B0_B1_findings / PHANTOM_SOM_CODE_TOUR / PAPER_STRATEGY_OPEN_QUESTIONS / master_bug_catalog — moving cost > benefit).

## Future reorg candidates (low priority)

If file count grows >50:
- `probes/` subdir (requires updating 5 doc references)
- `data/` subdir (requires updating Makefile + watchdog refs)
- `assets/` subdir
