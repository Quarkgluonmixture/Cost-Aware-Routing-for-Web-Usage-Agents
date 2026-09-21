# Evidence snapshot 2026-09-20 — small-text layer off the Condenser A100

## Why this exists

`results/*` is gitignored. Every aggregator under `scripts/analysis/` reads
per-episode records from run directories that only ever existed on the paper-grade
fire host (UCL Condenser A100 VM, `/mnt/scratch/p79_results_active_visualwebarena`).
Access to that host is tied to a UCL account and a 7-day SSH certificate, so a
checkout on any other machine could not recompute a single paper number.

This snapshot moves the *recomputable substrate* into git. It holds only small text
files — no screenshots, no trajectories, no raw model I/O.

## What is here

| file | what it is |
| --- | --- |
| `p79_smalltext_20260920.tgz` | 20,615 files, 8.9 MB gz. Every `condition_meta.json`, `condition_summary_v2.json` and `episodes/*_summary_v2.json` under the three result roots (VWA phase1, WA phase1, `p79_archives`). |
| `BUNDLE_MANIFEST.json` | Manifest-driven verification record produced by `scripts/maintenance/export_evidence_bundle.py`: per-condition status, episode counts, byte sizes, and the sha256 of the `run_manifest.yaml` it was taken against. **36/36 paper-grade conditions verified present.** |

Alongside it, at its canonical path:

- `results/phantom_paper/per_task_sr.csv` — the task x mode outcome matrix
  (1287 rows = 6 complete cells: classifieds/reddit x B0/B1/B2, 224/205 tasks x 6
  modes, plus cost columns). Consumed by `aggregate_fusion_premium.py`,
  `aggregate_label_instability.py`, `aggregate_noise_floor_inventory.py` and
  `aggregate_phase1_full_prereg_decision.py`.
- `results/phantom_paper/per_task_sr.csv.provenance.json` — its sidecar.

## Restoring

The tarball's paths are relative to the repo root for the two `results/` roots, and
absolute for `/mnt/scratch/p79_archives`. To put the repo-relative half back in place:

```bash
tar xzf results/evidence_snapshots/20260920/p79_smalltext_20260920.tgz \
    -C . --wildcards 'results/*'
```

Then the aggregators run without the fire host. Regenerating the matrix itself:

```bash
python3 scripts/analysis/generate_per_task_sr.py --out results/phantom_paper/per_task_sr.csv
```

## Known gap

`per_task_sr.csv` covers **6 cells, not the 8 the paper reports**. The two WebArena
cells are not registered in `results/phantom_paper/run_manifest.yaml` — that manifest
only tracks VisualWebArena — so the generator cannot see them, and whatever produced
the `*_with_wa.json` products read WA runs through some other, unregistered path.
The WA run directories themselves (25 of them under `results/webarena/phase1`) *are*
inside the tarball, so nothing is lost; what is missing is the registry entry that
says which of them are paper-grade. Finding or rebuilding that registry is the open
item before any 8-cell number can be recomputed from this snapshot.
