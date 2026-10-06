---
type: analysis
status: complete
created: 2026-10-07
purpose: is the shopping category grid served in the same order across runs
producer: scripts/analysis/shopping_grid_order_drift.py
---

# Shopping grid order across runs

Regenerate: `python scripts/analysis/shopping_grid_order_drift.py` (needs step-0 `observation_dom.txt` artifacts, i.e. the merged run store).

- Runs with step-0 artifacts: **9**; task×page groups seen by ≥2 runs: **184**.
- Scored tasks whose start grid's first row differs between runs: **152 / 432**.
- Of those, intent names a position (order-sensitive): **42**.

## Catalog states (clustered from the data, not assumed)

Runs whose first rows agree on ≥ 85% of shared pages are one state.

| state | runs |
|---|---|
| S1 | `B0_dom_shopping_20260804_003607_264370398_3845634_R3561`, `B0_vision_shopping_20260807_191852_632106648_362979_R23934`, `B1_phantom_prompt_shopping_20260911`, `B1_phantom_som_shopping_20260814`, `B1_phantom_text_shopping_20260908` |
| S2 | `B0_som_shopping_20260806_113115_297007393_109097_R12449`, `B1_vision_shopping_20260906` |
| S3 | `B1_dom_shopping_20260809` |
| S4 | `B1_som_shopping_20260812` |

## Success on order-sensitive tasks vs the rest

| run | state | order-sensitive | rest |
|---|---|---:|---:|
| `B0_dom_shopping_20260804_003607_264370398_3845634_R3561` | S1 | 1/42 | 12.3% |
| `B0_vision_shopping_20260807_191852_632106648_362979_R23934` | S1 | 0/42 | 15.9% |
| `B1_phantom_prompt_shopping_20260911` | S1 | 0/42 | 5.9% |
| `B1_phantom_som_shopping_20260814` | S1 | 1/42 | 4.9% |
| `B1_phantom_text_shopping_20260908` | S1 | 1/42 | 7.2% |
| `B0_som_shopping_20260806_113115_297007393_109097_R12449` | S2 | 1/42 | 16.2% |
| `B1_vision_shopping_20260906` | S2 | 1/42 | 5.9% |
| `B1_dom_shopping_20260809` | S3 | 2/42 | 4.9% |
| `B1_som_shopping_20260812` | S4 | 1/42 | 8.2% |

## Reading

- The order is a property of the run, not of the episode: inside a state, runs agree on essentially every shared page; across states most pages differ. All runs were on the same host and no snapshot records a container identity, so *why* a run lands in a state (reset / reindex / container generation) is not determinable offline.
- Order-sensitive tasks are near-zero in every state, so no state is visibly the one the reference answers were written against. Whether these tasks are hard or unanswerable as served cannot be told apart here; it needs the live site.
- Consequence for mode comparisons on shopping: the B1 arms do not share one state, so a per-task mode contrast on an order-sensitive task compares different pages. With near-zero success there, the effect on SR differences is at most a task or two; the larger consequence is that these tasks should be named as a scope limit, not counted as agent failures.
