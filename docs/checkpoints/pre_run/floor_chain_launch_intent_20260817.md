# Launch intent — 8-cell noise-floor replicate chain (declared 2026-08-17, BEFORE fire)

**Why this file exists.** codex Mode B Finding 2 (`/stress` 2026-08-17): every cell in this
chain is a same-condition replicate, and replicates are only recognised as such by being
listed in `CLEAN_PAIRS` (`scripts/analysis/aggregate_noise_floor_inventory.py`). Registering
them *after* the runs land makes registration a post-hoc choice — a cell whose floor came
out inconveniently could quietly stay unregistered and be re-labelled contamination. That is
a selective-reporting hazard, and it is invisible in the final artifact.

So the intent is declared here first, and committed before launch. When the runs land, the
registration must match this list exactly: **every cell below gets registered, whatever its
number says.** A cell that lands and is NOT registered requires a written reason in this
file, in the same commit that omits it.

This is also the fix for the concrete failure it describes: `R28065` ran a full 224 episodes
on 2026-08-16 with no such declaration, and `validate_fire_manifest.py` correctly called it a
COMPLETE ghost, raising the fail-closed halt marker at 08:04Z.

## Canonical (arm A) runs these replicate against

Each cell is arm B of a same-condition pair; arm A is the already-bound authoritative run in
`docs/checkpoints/pre_run/fire_manifest.json`.

| # | Cell (condition) | Cost | Purpose |
|---|---|---|---|
| 1 | B0 × classifieds × phantom_text | paid | first clean floor for a **phantom** arm |
| 2 | B0 × classifieds × phantom_prompt | paid | ditto |
| 3 | B0 × classifieds × phantom_som | paid | ditto |
| 4 | B1 × classifieds × vision | free | first B1 floor with any power |
| 5 | B1 × classifieds × dom | free | descriptive |
| 6 | B1 × classifieds × phantom_som | free | descriptive |
| 7 | B1 × classifieds × phantom_text | free | descriptive |
| 8 | B1 × classifieds × phantom_prompt | free | descriptive |

Already landed and registered under this same policy: **B1 × classifieds × som (R28065)**,
registered in `CLEAN_PAIRS` as `B1.cls.som` on 2026-08-17.

## Power, declared up front so the results cannot be oversold later

Floor measurability scales as `d ≈ n × SR × 0.59` (§468 / B-1972). Under that rule:

- **Cells 1-3 (B0 phantom arms): d ≈ 21-26** — the only cells in this chain powered to
  report an interval. These are the reason the chain is worth paying for.
- **Cells 4-8 (B1): d ≈ 8-10**, below the d<10 reporting bar this project set for itself.
  They are inventory and descriptive context, **not** interval evidence.

Declaring this before the data exists is the point: it removes the option of deciding, after
seeing the numbers, which cells were "the real measurement".

## What still is not solved by this file

A per-cell run ID cannot be reserved in advance here — run IDs are minted at launch with a
timestamp and nonce. This file pins the *intent* (which conditions, what they replicate, how
they may be read); it does not pin the identifiers. codex's stronger proposal — reserve IDs
before spawning and record a launch nonce — remains open, and is the durable fix.
