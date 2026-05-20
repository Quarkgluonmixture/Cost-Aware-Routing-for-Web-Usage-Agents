# Cross-System Docker Audit — quark (prod) vs A100 self-host (paper-grade target)

**Date**: 2026-05-15
**Scope**: pre-fire (Phase 1a clean-run readiness)
**Goal**: Find silent contamination vectors where A100 self-host docker stack will produce different paper-grade numbers than quark Windows docker stack, beyond the obvious (already-fixed) ZIM version mismatch.

## Production baseline (quark)

- Windows host, Docker Desktop
- Tailscale IP `100.95.81.103`
- Containers (from MEMORY + COMPUTE_INFRASTRUCTURE.md):
  - `vwa-shopping` (Magento 2) port 7770, also 7780 (admin)
  - `vwa-reddit` (Postmill) port 9999
  - `classifieds` + `classifieds_db` (OSClass + MySQL) port 9980
  - `vwa-wikipedia` (kiwix-serve 3.3.0) port 8888 mounting `wikipedia_en_all_maxi_2025-08.zim`
  - `homepage` port 4399 (static index)
- Cookies in `.auth/`:
  - blake.sullivan@gmail.com (classifieds)
  - MarvelsGrantMan136 (reddit)
  - emma.lopez@gmail.com (shopping)
  - admin/admin1234 (shopping_admin, 笔记 §103)
- All cookies generated against `100.95.81.103:<port>` host

## A100 self-host current state (2026-05-15 23:30 UTC)

- Ubuntu 22.04, A100-PCIE-40GB
- Docker 29.x (containerd-snapshotter, 笔记 §140.11)
- Containers ACTIVE on A100:
  - `vwa-reddit` (postmill-populated-exposed-withimg:latest, same hash as prod) up 6h, port 9999
  - `classifieds` (jykoh/classifieds:latest, same hash) up 7h, port 9980
  - `classifieds_db` (mysql:8.1) up 10h
- Containers NOT YET STARTED on A100:
  - shopping: image `shopping_final_0712:latest` (67.6GB on-disk) loaded but `docker run` not done
  - wikipedia: zim 19GB / 95GB downloading (wget PID 17987 from kiwix.org, ~50 MB/s, ETA 30 min); `docker run kiwix-serve` not done
  - homepage: image not present, container not running
- Cookies in `.auth/` (verified): domain = `localhost` ✅ (matches A100 expectation)
- TZ = UTC; LC_ALL = C.UTF-8

## P79 contract assumptions (queue scripts, runner)

- `scripts/vwa_env_remote.sh` (per-machine, gitignored) exports `__SITE__` URLs:
  - A100 version: `VWA_REMOTE_HOST=localhost` → `CLASSIFIEDS=http://localhost:9980`, etc.
  - DGX version: `VWA_REMOTE_HOST=100.95.81.103` → `http://100.95.81.103:9980`
- Task placeholders (`__CLASSIFIEDS__` / `__REDDIT__` / `__SHOPPING__` / `__WIKIPEDIA__` / `__HOMEPAGE__`) → env-var substitution at task-load
- `WIKIPEDIA_ZIM_VERSION` env var pinned to `wikipedia_en_all_maxi_2025-08` in 5 queue scripts (笔记 §81 fix)
- VWA `tasks.py` `_placeholder_mapping()` reads these env vars, rewrites URLs in task config raw.json before runner consumes
- Cookies loaded from `.auth/<site>_state.json` (Playwright browser context state)
- Auth refresh: `p79/utils/auth_refresh.py` calls site login URL, writes `.auth/<site>_state.json`

## Phase 1a actual scope (CLAUDE.md hard rule)

- 2 sites: classifieds (234 tasks) + reddit (210 tasks)
- 6 modes: dom / som / vision / phantom_som / phantom_text / phantom_prompt
- 3 baselines (advisor 2026-05-14): B0 (Qwen3-VL-235B-A22B proxy) / B1 (Qwen3-VL-4B local) / B2 (Gemma3-VL-4B local)
- Phase 1a does NOT run shopping (R3 framing per CLAUDE.md)
- Cross-link refs from cls+reddit raw.json:
  - `__HOMEPAGE__`: 0 refs in both → A100 not running homepage container is OK
  - `__WIKIPEDIA__`: 58 refs in reddit, 0 in cls → wiki ZIM is critical for reddit
  - `__SHOPPING__` cross-refs from reddit: NOT YET CHECKED — could be a contamination vector

## Already-fixed issues (this session, commits b386bb1 + fb97c14 + earlier)

- ✅ Wikipedia ZIM version: pinned to 2025-08 across setup_vwa.sh / start_vwa_docker.sh / a100_self_host_vwa.sh / import_vwa_assets.sh (笔记 §81 verbatim reproduce risk eliminated)
- ✅ Source switch: metis (only 2022-05 mirror) → kiwix.org (has 2025-08, 12× faster download)
- ✅ Repo-root `.gitignore` protects `/shopping_final_*.tar` `/postmill-populated-*.tar` `/wikipedia_en_all_*.zim` `/classifieds_docker_compose.zip` from accidental commit if setup interrupts
- ✅ a100_self_host_vwa.sh known-issues header documents docker 29 deadlock + VWA requirements.txt destructive + auth_refresh path + ZIM version
- ✅ Cookies regenerated locally on A100 against `localhost`, domain field verified `localhost` for all 3 sites
- ✅ Watchdog + run_registry + 6 aggregators support 3-baseline (B0/B1/B2 alphabetical) — no B2 silent skip in cross-run analysis
- ✅ 36 WA configs + 12 B2 phantom WA configs generated (per advisor 2026-05-14 6-mode parity)
- ✅ Queue collision guards (queue_chain.sh + queue_phase1_paper_grade.sh) generalized to 3-way B0/B1/B2

## My self-audit findings (not exhaustive)

### Confirmed risks

**R1 — Magento base_url image-baked metis hostname** (笔记 §75 + §103)
- Magento image stores `web/unsecure/base_url` = `http://metis.lti.cs.cmu.edu:7770/` (from VWA original deployment)
- On first `docker run vwa-shopping`, hitting any URL 302-redirects to metis.lti.cs.cmu.edu → A100 cannot resolve → connection refused
- Quark fix (笔记 §75/§103): `docker exec vwa-shopping magento config:set web/{secure,unsecure}/base_url http://100.95.81.103:7770/`
- A100 needs same fix targeting `http://localhost:7770/`
- **Phase 1a impact**: 0 (shopping not in 1a scope), but Phase 1b will be blocked
- **Status**: Container not yet `docker run` on A100, so bug not exposed yet

**R2 — TZ / locale mismatch**
- A100 TZ = UTC, LANG=C.UTF-8
- Quark Windows likely BST or Windows-default locale
- Possible impacts:
  - Postmill "newest posts" ordering: Postmill stores `created_at` UTC, displays in browser TZ → A100 browser TZ unknown (Playwright default UTC?), quark browser TZ likely Windows local → display order may differ for tasks asking "latest 3 posts" if posted at boundary
  - Locale string formatting (date format dd/MM vs MM/dd) in OSClass listing detail pages
- **Phase 1a impact**: unknown — could affect a small subset of timestamp-sensitive tasks
- **Status**: NOT yet measured

### Negative findings (audit confirms NOT a risk)

**N1 — Postmill site_url auto-derived from request host**
- Curl `http://localhost:9999/` returned `<link rel="canonical" href="http://localhost:9999/">` — Postmill uses request.host, not DB-baked URL
- Cross-platform: same image hash will canonicalize against whatever host the request arrives on → A100 produces `localhost:9999` URLs, quark produces `100.95.81.103:9999` URLs, both internally consistent
- Conclusion: NO postmill-equivalent of Magento §75 bug

**N2 — Image hashes match prod**
- jykoh/classifieds:latest sha256:a2a794da92f6 (2 years old)
- postmill-populated-exposed-withimg:latest sha256:0a0c002b4dd0 (2 years old)
- shopping_final_0712:latest sha256:ccff8c1772be (2 years old)
- All from VWA upstream → identical to quark side (pulled from same origin) → deterministic image-level config drift eliminated

**N3 — Cookies domain match A100 host**
- `.auth/classifieds_state.json` `.auth/reddit_state.json` `.auth/shopping_state.json` all have `domain: "localhost"`
- Playwright will replay cookies on localhost requests → match A100 site domain

**N4 — HOMEPAGE 4399 missing is safe for Phase 1a**
- grep `__HOMEPAGE__` cls/reddit raw.json = 0 ref both sides
- A100 not running homepage container is acceptable for 1a; flag for 1b shopping check

## Open question regions (audit-incomplete, what I'd like adversarial reviewers to attack)

**Q1 — OSClass URL rewrite + image links**
- OSClass `oc_t_preference.webSiteUrl` is EMPTY on A100 — does OSClass fall back to request.host (like Postmill) or to some default that breaks on localhost?
- Image upload URLs in `oc_t_item_resource`: are they relative or absolute? Absolute = baked at upload time = quark-era URL = broken on A100

**Q2 — Postmill image static URLs (avatars, thumbnails, post images)**
- `sites` table in postmill DB unknown content — does it have stored absolute URLs?
- `images` table — relative path or absolute URL?
- Inline post body markdown rendering: `[link](http://100.95.81.103:9999/...)` references in seeded posts will 404 on A100

**Q3 — Postmill ActionCable / WebSocket**
- Postmill Rails uses ActionCable WebSocket for real-time updates (vote count, new comments)
- WS endpoint default `ws://localhost:9999/cable` — request-host-derived? or CSP-pinned?
- If pinned to quark IP, A100 browser will fail WS connection → some interactive features broken → some tasks may fail differently

**Q4 — CORS / CSP / fetch metadata across sites**
- Reddit task referencing `__WIKIPEDIA__` link → browser fetches localhost:8888 from origin localhost:9999 — CORS-OK on same hostname?
- But quark side: fetch from 100.95.81.103:9999 to 100.95.81.103:8888 → same origin? Tailscale uses same hostname → consistent
- A100 localhost-to-localhost should be same-origin → OK in theory, but if any preflight is cached against quark IP origin it would 404

**Q5 — Auto-login email validation domain**
- Cookie generation via `p79/utils/auth_refresh.py` → POSTs to login form with `blake.sullivan@gmail.com` etc.
- Login form may verify against MX records of `@gmail.com` (online) — A100 has internet, but if any classifier/MX-check happens, it might differ from prod path
- More plausible: login form just stores the email string; cross-platform identical

**Q6 — Reset wrapper differs by site**
- `scripts/maintenance/reset_vwa_sites.sh` mode=local:
  - cls: curl `http://localhost:9980/reset?token=<CLASSIFIEDS_RESET_TOKEN>` (~0.1s)
  - reddit: `docker stop && docker rm vwa-reddit && docker run --name vwa-reddit -p 9999:80 -d postmill-populated-exposed-withimg:latest` + 60-iter HTTP 200 wait (~58s) + 3s settle
  - shopping: stub (no-op) — pending Phase 1b SQL restore implementation
- Quark mode=remote: SSH through quark Tailscale to docker on Windows side
- Risk: reddit `docker rm vwa-reddit` then `docker run` creates NEW container ID each time → some quark watchers may track container ID, breaking; A100 watchdog uses container NAME (not ID) so this should be OK; but worth verifying watchdog auth_refresh path doesn't pin container ID

**Q7 — Image upload directories persistence**
- Classifieds `oc-content/uploads/` is mounted volume? bind mount?
- If user `blake.sullivan` is supposed to have N uploaded images at task-start, are those baked in image or in volume? `docker rm` of `classifieds` container would lose them if volume.
- Reset wrapper for cls is just `curl /reset` endpoint — does OSClass /reset actually restore image dir? Or only DB?

**Q8 — MySQL initialization race**
- 笔记 §140.11 entry 4: A100's first compose-up skipped osclass_craigslist.sql init → manual `docker exec mysql source /docker-entrypoint-initdb.d/...` needed
- Was this idempotent? If reset wrapper `curl /reset` doesn't re-trigger init, and we later `docker rm classifieds_db` accidentally → init must be re-run manually
- Quark side: init guaranteed by Docker Desktop volume persistence

**Q9 — Concurrent run guard ineffective across machines**
- `pgrep -f "run_experiment.*reddit"` checks LOCAL processes
- DGX runner + A100 runner BOTH hitting `__REDDIT__` is fine if they hit DIFFERENT containers (DGX→quark, A100→localhost)
- BUT if A100 self-host reddit container and quark reddit container have DIFFERENT internal state (different posts/comments accumulated), the SAME paper-grade clean-run sampling on DGX vs A100 produces DIFFERENT data → cross-machine reproducibility broken
- Reset wrapper should reset to identical container state, but if quark accumulates non-reset side effects (cron'd auth refresh that posts comments?) drift accumulates

**Q10 — kiwix-serve container args do NOT match new ZIM filename**
- `start_vwa_docker.sh` now passes `wikipedia_en_all_maxi_2025-08.zim` to kiwix-serve CLI
- BUT kiwix-serve URL serving pattern: `http://localhost:8888/viewer#wikipedia_en_all_maxi_2025-08/A/<title>` — the version string appears in URL path
- Task config raw.json has `__WIKIPEDIA__/<...>` placeholders — does P79 placeholder rewrite include the ZIM version segment or just the host:port?
- Critical: if placeholder = `http://__WIKIPEDIA_HOST__:__WIKIPEDIA_PORT__/viewer#wikipedia_en_all_maxi_<old_version>/...` then env var pin of WIKIPEDIA_ZIM_VERSION must replace the path segment too. Check `tasks.py:_placeholder_mapping` rewrite logic.

**Q11 — Playwright user-agent fingerprint**
- Playwright default UA includes Playwright signature
- Postmill / Magento / OSClass may have anti-bot detection
- Quark side proven to work, A100 side first paper-grade run — could trigger different anti-bot pathway

**Q12 — Browser viewport / device DPI differ from quark**
- VWA paper specifies viewport 1280×720
- Playwright on A100 (headless) vs Playwright on DGX (same Playwright? but rendering against quark remote site)
- If A100 rendering different DPI → SoM bbox coordinates differ → som mode SR may shift
- For paper-grade cross-baseline ablation this is invariant per-baseline so doesn't bias, but a baseline-mode interaction might exist

## Files to read (audit input pointers)

### Production-validated quark side (DGX)
- `scripts/vwa_env_remote.sh` (gitignored, quark version uses 100.95.81.103)
- `scripts/queues/queue_baseline.sh` (5 queue scripts, all 5 hardcode WIKIPEDIA_ZIM_VERSION)
- `scripts/maintenance/reset_vwa_sites.sh` (mode=remote SSH chain)
- `p79/utils/auth_refresh.py` (account creds + cookie generation)
- `p79/experiment/tasks.py` `_placeholder_mapping()` (env var substitution)
- `p79/experiment/config.py` (DEFAULT_CONFIG including include_sites)
- `external/visualwebarena/config_files/vwa/test_{classifieds,reddit,shopping}.raw.json` (task templates)
- `docs/checkpoints/实验笔记.md` §75 (Magento 302 metis bug), §81 (Kiwix ZIM version), §103 (Magento auth bug paper-grade fix)
- `docs/reference/master_bug_catalog.md`

### A100 self-host side
- `scripts/setup/a100_self_host_vwa.sh` (NEW this session, 30-line known-issues header)
- `scripts/vwa/setup_vwa.sh` (4 wget functions, mirrors swapped, ZIM 2025-08)
- `scripts/vwa/start_vwa_docker.sh` (docker run commands, 7780 admin port logic)
- `scripts/maintenance/reset_vwa_sites.sh` mode=local (cls curl 0.1s / reddit docker-rm-run 58s / shop stub)
- On A100: `scripts/vwa_env_remote.sh` (VWA_REMOTE_HOST=localhost)
- On A100: `.auth/{classifieds,reddit,shopping}_state.json` (Playwright session state)

### Live state right now
- A100 containers: vwa-reddit up 6h, classifieds + classifieds_db up 7-10h, vwa-shopping NOT started, vwa-wikipedia NOT started
- A100 wget PID 17987 downloading wikipedia_en_all_maxi_2025-08.zim from kiwix.org (19GB / 95GB, ~50 MB/s, ETA 30 min)
- DGX: results/visualwebarena/phase1/ all archive (May 4 or earlier), no active paper-grade run
- 14 commits unpushed since last push

## Attack invitation

This is **pre-fire scope**. Find ≥7 findings, ≥3 OOB attacks. Specifically wanted:
- A specific bug where the A100 paper-grade clean-run will produce SR numbers that differ from quark by ≥0.5pp due to a contract-drift vector NOT enumerated above
- A bug where the difference is "silent" (no error, no warning, just different SR distribution)
- A bug that would survive both a normal code review AND would survive a normal stress audit of the paper draft (i.e., something only audit-from-deployment angle catches)
