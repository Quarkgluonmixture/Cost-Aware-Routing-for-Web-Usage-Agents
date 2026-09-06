# Look, read, or both? — the board demo (16 Sep)

Two builds, same page:

| | what it is | when to use it |
|---|---|---|
| **`demo_portable.html`** | one file, 11.7 MB, everything inlined | **the venue.** Copy it anywhere — USB stick, a borrowed laptop, email to yourself. Nothing else needs to travel with it. |
| `demo/index.html` | page + `data.js` + `frames/` | editing and rebuilding |

Either one: double-click to open. No server, no network, no site.

```
←  →     step            space   play / pause          1 2 3   task
```

It auto-plays and loops through the three tasks on its own, so it can be left
running while nobody is at the board.

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

The dollar figure under each lane is that lane's **running API cost**, so task 130
shows the point of the whole project in one frame: LOOK solves it for $0.007 while
READ spends $0.041 and fails.

## Honesty rules this demo keeps

- **Every lane is one recorded run**, labelled as such on the page. Nothing is live
  and nothing is re-simulated. The run id of each lane is printed bottom-right.
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

## Rebuilding

```bash
.venv/bin/python3 deliverables/showcase/demo/build_demo_data.py   # data/ + data.js + frames/
.venv/bin/python3 deliverables/showcase/demo/build_portable.py    # -> ../demo_portable.html
```

### The portable build does not trade away quality

Frames are re-encoded to **lossless WebP** — not to a lossy format, and not resized.
Three checks stand behind that word:

1. the encoder output is decoded again and compared to the source PNG **per frame**,
   and the build aborts if a single pixel differs;
2. measured result: 15.0 MB of PNG becomes 8.7 MB of WebP (58%) with identical pixels,
   which is what keeps the inlined file at 11.7 MB rather than ~20 MB;
3. both builds are then rendered side by side and their frames diffed on screen —
   LOOK, READ and BOTH all come back `pixel identical` at 1280x720.

The data URIs live in one JS object rather than in 82 `<img src="data:...">`
attributes: same bytes, but the HTML parser sees a single string literal instead of
~12 MB of attribute text. Cold open of the single file measures ~3 s to fully decoded.

`index.html` is shared by both builds — it reads `window.FRAMES` when the portable
build defines it and falls back to the relative paths otherwise, so there is no second
copy of the page to keep in sync.

Regenerates `data/`, `data.js` and `frames/`. `frames/` is committed on purpose: its
inputs live under `results/`, which is gitignored and is not present on a laptop taken
to the venue. LOOK and READ read from `results/repro_replicates/`; BOTH reads the SoM
arm's replicate under `results/visualwebarena/phase1/`, whose `artifacts/` may need to
be pulled from the A100 first (they are not part of the normal results sync):

```bash
rsync -a condense-a100:/mnt/scratch/p79_results_active_visualwebarena/phase1/<run>/phase1_som_router_0/artifacts/classifieds_task_{130,76,17} \
      results/visualwebarena/phase1/<run>/phase1_som_router_0/artifacts/
```
