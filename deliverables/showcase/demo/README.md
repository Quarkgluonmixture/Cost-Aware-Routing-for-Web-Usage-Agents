# Look, read, or both? — the board demo (16 Sep)

Two builds, same page:

| | what it is | when to use it |
|---|---|---|
| **`demo_portable.html`** | one file, ~12 MB, everything inlined | **the venue.** Copy it anywhere — USB stick, a borrowed laptop, email to yourself. Nothing else needs to travel with it. |
| `demo/index.html` | page + `data.js` + `frames/` | editing and rebuilding |

Either one: double-click to open. The three recorded tasks need no server, no network
and no site. The fourth tab (*try your own*) is live and does need them — see **Live**.

```
←  →     step            space   play / pause          1 2 3   task          4   try your own
```

It auto-plays and loops through the three tasks on its own, so it can be left
running while nobody is at the board. The live tab drifts back to the replay after
two idle minutes for the same reason.

## What a visitor is looking at

The same task, replayed in three representations of the same web page:

| lane | what the agent got | colour |
|---|---|---|
| **LOOK** | screenshot only | blue |
| **READ** | accessibility tree only, no image | orange |
| **BOTH** | screenshot with numbered marks | green |

Three tasks, one per shape of the result:

| task | LOOK | READ | BOTH | why |
|---|---|---|---|---|
| 130 | ✓ 2 | ✗ 9 | ✓ 3 | "…image taken during a sunset" — the text tree never says *sunset* |
| 76 | ✗ 26 | ✓ 12 | ✗ 7 | "…change the price to $85.50" — two form fields; LOOK never edits them |
| 17 | ✗ 9 | ✗ 8 | ✓ 6 | "cheapest bike with **red handlebars** between **$900–950**" — one half must be seen, the other filtered |

### Three meters under every lane

| meter | what it is | measured? |
|---|---|---|
| **cost** | billed API dollars, summed step by step | yes — from the step records |
| **time** | the run's recorded wall-clock (model + page loads), not the replay's pace | yes |
| **CO₂e ≈** | an estimated **range**: solid bar to the low end, pale to the high end | **no — estimated, see below** |

Each row is scaled to the largest final value among the three lanes of that task, so
the lane that spends the most fills its row and the other two are read against it.
The build asserts that every meter ends on the episode summary's own totals (cost,
seconds, tokens), so a lane can't show numbers from a different run than it names.

### The arrow: which view a learned choice picked

A line under the task says which view the project's learned router picked for it, and
that lane wears a ring — **green** if it solved the task in the run shown, **red** if not.

| task | learned choice | outcome |
|---|---|---|
| 130 | READ | ✗ — LOOK and BOTH solved it |
| 76 | READ | ✓ |
| 17 | a fourth view (text tree read with the marked-screenshot instructions), not one of the lanes | its own two recorded runs split: ✓ once, ✗ once |

The pick is read from the **fold-held-out** replay
(`results/phantom_paper/l1_router_offline_20260715/router_offline_replay.json`): the
router that made each pick was trained without that task, and the build refuses an
in-sample pick. The three tasks were chosen for how the *views* differ, not for how
the router does, so they are an illustration — the router's score is the poster's
**0 of 8**. Nothing here is picked to flatter it.

### The CO₂e row

The model (Qwen3-VL-235B-A22B) is served from **AWS eu-west-2 (London)**, so there is
no local energy draw to measure; every run records `energy.source = disabled`. The row
is computed from the step records' input and output tokens:

```
grams = (input_tokens × J_in + output_tokens × J_out) × facility factor ÷ 3.6e6 × grid g/kWh
```

| constant | low | high | source |
|---|---|---|---|
| J per **output** token | 1.0 | 8.0 | **measured on this model**: ML.ENERGY Leaderboard v3 (companion to Chung et al. 2025, arXiv 2505.06371) — Qwen3-VL-235B-A22B BF16, 8×H100: 7.26 J at batch 8, ~1.1 J at batch 192–256 |
| J per **input** token | 0.05 | 0.4 | no prefill measurement exists for this model; scaled from measured dense prefill (Chung et al. 2025, App. B.3: Llama-3.1-8B 0.017 J, 70B 0.18 J) to ~22B active. The high end is a **judgement** (MoE overhead, vision encoder) |
| facility factor | 1.23 | 1.97 | host + idle 1.0–1.6 (Google 2025, arXiv 2508.15734, Table 1) × AWS London PUE 1.23 |
| grid | 131 | 144 g/kWh | UK DESNZ 2026 conversion factor, at generation / with T&D losses |

What would make it wrong by more than 3×: the serving batch (≤8 concurrent requests
can exceed the high end); non-GPU hardware (Trainium / Inferentia are unmeasured);
prompt caching (drives input energy below the low end); market-based rather than
location-based carbon accounting (can be near zero). Token count is a workload
indicator, not an energy measurement (Fernandez et al., ACL 2025, arXiv 2504.17674) —
which is why the page never shows one number here. All four arXiv ids above were
confirmed through the arXiv API on 2026-09-10.

## Honesty rules this demo keeps

- **Every lane is one recorded run**, labelled as such on the page. Nothing in the
  replay is live and nothing is re-simulated. The run id of each lane is printed
  bottom-right.
- **The three tasks are outcome-stable**: each one's three-way result is identical on
  the canonical run and on its replicate. A task whose outcome moves between reruns
  would be showing run-to-run noise, not a finding (`serving_mode_floor.md` puts the
  B0 floor at 12–14% of tasks flipping per rerun, so this filter is not optional).
- **A blank reasoning box is real.** Some steps carry a tool call with no reasoning
  text; the lane says "the agent stated no reason" and shows what it did instead of
  inventing prose.
- **Only BOTH shows an element number.** SoM numbers elements 1..K and prints those
  numbers on the frame, so `[36]` is something you can find in the picture. DOM refers
  to elements by the accessibility tree's own node id (e.g. `1433`), which appears
  nowhere in the image — showing it would imply a correspondence that does not exist.
- **LOOK's click is scaled, not assumed.** It arrives in Qwen's 0–1000 normalised
  space; the build asserts `coordinate_type == "qwen_0_1000"` before scaling to pixels,
  so a future run that changes the convention fails the build instead of silently
  drawing the click in the wrong place (B-1860).
- **Live runs are never scored.** A typed task has no answer key; the live lane shows
  what the agent said when it finished and leaves the verdict to the visitor.

## Live — *try your own*

```
visitor ── quark (venue laptop) ───────── DGX (spark-9ea3) ─────────────────────── AWS eu-west-2
           browser: demo page  ssh tunnel  live/server.py                            the model
                                           3 × p79 runner (LOOK / READ / BOTH)
                                           classifieds site: live/site-compose.yml
                                             (own web + db, 127.0.0.1:9981)
```

Everything live runs on DGX: the repo, the venv, the model key, Playwright, and the
site itself. quark only shows the page and holds one SSH tunnel — quark cannot reach
DGX's Tailscale address (DGX sits in another tailnet) but can reach it with `ssh spark`.

**Why the site is on DGX, and which site it is.** DGX is aarch64 with no x86
emulation, so the official `jykoh/classifieds` (amd64-only) cannot run there; the
community arm64 rebuild `ghcr.io/bgrins/vwa_classifieds_{web,db}` can, and was
already on the machine. It is not the image the recorded runs used, so it was checked
against them on 2026-09-10: item `19604` is "Indestructible Triumph 22' center console
with Honda Motor" at 23750.00 (task 130), and task 17's two reference items are the
Cannondale Six13 and the salsa mukluk 3 — the same listings the recorded lanes show.
One visible difference: its page header reads "Classifieds" in text where the recorded
frames show the OsClass logo — a visitor comparing tabs closely may notice.
DGX is shared: another project runs its own copy of this site (`classifieds_db` +
a web container on host port 9980). This stack has its own project name, container
names, network and database and binds only `127.0.0.1:9981`, so neither can touch the
other. quark's docker (official image, port 9980) stays the backup:
`LIVE_CLASSIFIEDS=http://100.95.81.103:9980` before starting the server.

**Runbook (16 Sep, ~10 min):**

1. **DGX:** `docker compose -f deliverables/showcase/demo/live/site-compose.yml up -d`;
   check `curl -s -o /dev/null -w '%{http_code}' http://localhost:9981/` prints 200.
2. **DGX:** `.venv/bin/python3 deliverables/showcase/demo/live/server.py`
   (env: `LIVE_PORT` 8799, `LIVE_MAX_STEPS` 12).
3. **quark:** `ssh -N -L 8799:localhost:8799 spark` and leave it open.
   If it prints `bind [127.0.0.1]:8799: Permission denied`, that is **not** a failure:
   VS Code's Remote-SSH session to DGX has already auto-forwarded port 8799 to quark's
   `127.0.0.1`, and ssh has bound `::1` instead — both reach the server. Check with
   `curl.exe -s http://localhost:8799/health` (PowerShell: no `&` between commands).
   Don't rely on the VS Code forward alone on the day: it disappears when VS Code
   disconnects, while the ssh tunnel then binds both addresses by itself.
4. **quark:** open **`http://localhost:8799/`** in the browser — the live server
   serves the page itself — press `4`, run one suggestion as a test. The status line
   says *ready* when the server answers and *offline* when it does not.
   **Offline fallback:** if the tunnel or DGX is down, double-click a local copy of
   `demo_portable.html` instead; the three recorded tasks play with no network.

> ⚠️ **Never open the page through VS Code's Live Server** (`localhost:5500–5502`).
> Live Server reloads the whole page whenever any file in the workspace changes, and a
> live session writes files every step (DGX's cron jobs also write logs every few
> minutes) — so the page reloads itself over and over, the replay restarts, and the
> live tab loses its session. This is what "一直闪" was on 2026-09-10.

What the server does per session: logs in to the site once (reused for 15 min; the
site's sessions expire at ~24), writes a one-task config, and starts the ordinary
runner three times with `output_root = demo/live/runs/` and `P79_PAPER_GRADE=0`
(`live/run_lane.sh`). One session at a time — three lanes on one site and one account
is already the collision the paper-grade launch rules forbid; a second visitor gets
the running session instead of a new one. Each lane stops at 12 steps; a session is
killed after 8 minutes.

**What the live tab does on screen:**

- **Before a task**, each lane shows the site's start page as that view first sees it
  (`live/idle/*.png`, copied from step 0 of a real live session; BOTH's copy carries the
  numbered marks), not a black stage. Offline, it falls back to black.
- **Each lane runs at its own pace.** A lane shows its steps as they arrive (at most one
  per 0.8 s, so a burst still reads step by step), and between steps its corner says what
  it is doing, with a running count of seconds — "opening the site · 12 s", "working on
  step 3 · 6 s" — so a slow model call does not look like a freeze. (A version that held
  all three lanes until the slowest had its first step was dropped: the screen sat still
  for 20–30 s while three agents cold-started, which read as stuck.)
- **Expect ~25 s before the first step.** Measured on 2026-09-10: three runner processes
  start (~7 s, Python + torch imports), each opens a browser and the start page
  (~5–8 s), then step 0 — look, one model call, act (5–10 s); a step appears only after
  its action, because its screenshot is written after the step's timing window. The one
  avoidable part, the ~5 s site login, is now done by the server in the background (at
  start-up and every 10 min), so a Run never waits for it. Say so when you press Run.
- **After a run, the learned choice.** Each lane gets ✓ / ✗ buttons and the visitor
  judges each answer. The line under the task names the view the project's learned
  router picks for the typed task; that lane wears a neutral ring until its answer is
  judged, then green or red. The colour is the visitor's verdict on the router's pick —
  a typed task has no answer key. How the pick is made (`live/router_pick.py`): all five
  fold models vote (a typed task belongs to no fold), each with its own threshold; the
  features are the typed intent plus the READ lane's first page; the one feature with no
  live value, the task's annotated difficulty, is set to the classifieds median (medium),
  and the page says so. Check: on recorded tasks 130 / 76 / 17 the vote gives READ /
  READ / P-prompt — the same picks as the fold-held-out replay on those tabs.
- **No drifting back.** The live tab stays until someone leaves it; `?idle=<seconds>`
  turns an automatic return to the recorded tasks back on (any input restarts the count).
  In the live tab, space does nothing; `1` `2` `3`, the task tabs and the *Replay* button
  go back on purpose.

**Limits to know before you open it to visitors:**

- **Writes are real.** A task like "post an ad" or "change my listing" changes the
  live site, and the three lanes can trip over each other's changes. The suggestions
  are read-only on purpose. Reset = `docker compose -f … down` then `up -d` (state
  lives only in the db container).
- **Live answers say nothing about the paper's numbers**, which all come from recorded
  A100 runs on the official image.
- **Cost:** about $0.004 per step, so a full three-lane session is ~$0.15.

## Rebuilding

```bash
.venv/bin/python3 deliverables/showcase/demo/build_demo_data.py   # data/ + data.js + frames/
.venv/bin/python3 deliverables/showcase/demo/build_portable.py    # -> ../demo_portable.html
```

`frames/` is committed on purpose: its inputs live under `results/`, which is
gitignored and is not present on a laptop taken to the venue. LOOK and READ read from
`results/repro_replicates/`; BOTH reads the SoM arm's replicate under
`results/visualwebarena/phase1/`, whose `artifacts/` is **not on DGX**.

> ⚠️ **Artifacts pulled by hand into `results/` on DGX are deleted within 15 minutes.**
> The DGX cron job `sync_a100_results.sh` runs every 15 min with
> `--exclude='artifacts/' --delete-after --delete-excluded`, and it also resets the
> parent directory's mtime — which is why the pulled directory once "vanished without
> a trace" (笔记 §501.6, root cause found 2026-09-10). Same class as B-1929.

So the build does not need them: when a frame's source artifact is missing it reuses
the committed frame, but **only if** the previous `data.js` names the same run for
that lane and lists that exact frame — otherwise it stops and asks for the artifacts.
To re-pull them (e.g. to change the task set), rsync to a directory outside `results/`
and point `LANES["BOTH"]` there, or build within the 15-minute window.

### The portable build does not trade away quality

Frames are re-encoded to **lossless WebP** — not to a lossy format, and not resized.
Three checks stand behind that word:

1. the encoder output is decoded again and compared to the source PNG **per frame**,
   and the build aborts if a single pixel differs;
2. measured result: 15.0 MB of PNG becomes 8.7 MB of WebP (58%) with identical pixels,
   which is what keeps the inlined file near 12 MB rather than ~20 MB;
3. both builds are then rendered side by side and their frames diffed on screen —
   LOOK, READ and BOTH all come back `pixel identical` at 1280x720.

The data URIs live in one JS object rather than in 82 `<img src="data:...">`
attributes: same bytes, but the HTML parser sees a single string literal instead of
~12 MB of attribute text. Cold open of the single file measures ~3 s to fully decoded.

`index.html` is shared by both builds — it reads `window.FRAMES` when the portable
build defines it and falls back to the relative paths otherwise, so there is no second
copy of the page to keep in sync.
