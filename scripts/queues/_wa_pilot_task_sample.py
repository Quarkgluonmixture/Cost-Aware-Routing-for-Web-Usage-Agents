#!/usr/bin/env python3
"""WA cross-benchmark pilot — stratified task sampler (prereg §8.8 B-1296/B-1627).

Registered at this exact path by `docs/checkpoints/paper_drafts/section8_limitations.md`
§8.8 mini-prereg ("sampling script registered at `scripts/queues/_wa_pilot_task_sample.py`,
to be landed before the pilot fires"). This file lands it.

Spec as registered
------------------
  Task pool  : stratified sample, 10 tasks per WA site, fixed seed=42.
  Estimand   : task-set Jaccard across the mode universe (mean pairwise Jaccard).
  Decision   : upper 95% CI <= 0.5 generalizable / lower > 0.7 VWA-specific /
               else inconclusive (tri-state, pre-registered).

Site-coverage deviation (documented, NOT silent)
------------------------------------------------
§8.8 prose lists 5 sites (shopping / reddit / shopping_admin / wiki / gitlab). Only
THREE are runnable in this repo and on the paper-grade host:

  * `scripts/queues/queue_baseline.sh` WA site whitelist == {reddit, shopping,
    shopping_admin} — no gitlab/map/wiki branch has ever existed.
  * `external/visualwebarena/config_files/wa/` ships exactly three split files
    (test_reddit / test_shopping / test_shopping_admin) + the 812-task union.
  * The A100 paper-grade host runs docker containers for postmill (reddit),
    magento (shopping + shopping_admin share one container), and a read-only
    kiwix wikipedia. No gitlab image, no map service.
  * "wiki" is not a standalone WA site at all: `wikipedia` appears only inside
    cross-site task tuples (map+wikipedia 17 tasks, gitlab+wikipedia 6), so a
    10-task wiki stratum is not constructible even with a gitlab container.

The §8.8 5-site list is therefore a prose-layer error of the same class as the
already-corrected "classifieds mis-listed" note in that same sentence.

Per-site quota (10) is held at the registered value rather than rescaled to keep
n=50, so that landing a gitlab container later EXTENDS the sample (30 -> 40 -> 50)
instead of invalidating the strata already drawn. Sampling is seeded per site
(seed 42 + site name), so each site's draw is independent of which other sites
are present — adding gitlab later cannot perturb the reddit/shopping/admin draws.

Usage
-----
  python3 scripts/queues/_wa_pilot_task_sample.py                  # print summary
  python3 scripts/queues/_wa_pilot_task_sample.py --emit-yaml      # task_ids block
  python3 scripts/queues/_wa_pilot_task_sample.py --json out.json  # machine-readable
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from p79.experiment.tasks import _is_na_task  # noqa: E402  (needs REPO on path)
WA_CONFIG_DIR = REPO / "external" / "visualwebarena" / "config_files" / "wa"

SEED = 42
QUOTA_PER_SITE = 10

# Runnable strata only. Extend this tuple (not the quota) when a site's container
# and split file both land; see module docstring.
SITES = ("reddit", "shopping", "shopping_admin")


def _site_seed(site: str) -> int:
    """Per-site deterministic seed derived from the registered global seed.

    Deriving per site (rather than drawing all sites from one shared RNG stream)
    makes each stratum independent of the presence/order of the others, so a
    later gitlab stratum extends the sample without perturbing existing draws.
    """
    digest = hashlib.sha256(f"{SEED}:{site}".encode()).hexdigest()
    return int(digest[:16], 16)


def load_site_tasks(site: str) -> list[dict]:
    path = WA_CONFIG_DIR / f"test_{site}.raw.json"
    if not path.exists():
        raise FileNotFoundError(f"WA split file missing for site '{site}': {path}")
    with path.open() as fh:
        return json.load(fh)


def sample_site(site: str, quota: int = QUOTA_PER_SITE) -> dict:
    raw = load_site_tasks(site)
    # Draw from the SCORED pool, not the raw pool. N/A tasks (reference answer
    # fuzzy_match == "N/A") are a pre-registered exclusion (§139.8) that the
    # runner applies at load time via `_is_na_task`. Sampling before that filter
    # silently under-fills the stratum — the first draw here lost 3 of 30 that
    # way (shopping 368, shopping_admin 115/247). Reusing the runner's own
    # predicate keeps the sampler and the loader on one definition of "task".
    tasks = [t for t in raw if not _is_na_task(t)]
    # Sort by task_id first: json file order is not a stable contract, but task_id is.
    tasks = sorted(tasks, key=lambda t: int(t["task_id"]))
    rng = random.Random(_site_seed(site))
    if len(tasks) < quota:
        raise ValueError(f"site '{site}' has {len(tasks)} scored tasks < quota {quota}")
    drawn = rng.sample(tasks, quota)
    ids = sorted(int(t["task_id"]) for t in drawn)
    return {
        "site": site,
        "pool_size_raw": len(raw),
        "pool_size_scored": len(tasks),
        "n_excluded_na": len(raw) - len(tasks),
        "quota": quota,
        "task_ids": ids,
        "intents": {int(t["task_id"]): t["intent"] for t in drawn},
    }


def build_sample(sites: tuple[str, ...] = SITES) -> dict:
    strata = [sample_site(s) for s in sites]
    total = sum(len(s["task_ids"]) for s in strata)
    return {
        "seed": SEED,
        "quota_per_site": QUOTA_PER_SITE,
        "sites": list(sites),
        "n_total": total,
        "n_registered_target": 50,
        "strata": strata,
    }


def emit_yaml(sample: dict) -> str:
    lines = ["task:", "  task_ids:"]
    for stratum in sample["strata"]:
        ids = ", ".join(str(i) for i in stratum["task_ids"])
        lines.append(f"    {stratum['site']}: [{ids}]")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit-yaml", action="store_true", help="print a task_ids YAML block")
    ap.add_argument("--json", type=Path, help="write the full sample as JSON")
    ap.add_argument("--sites", nargs="+", default=list(SITES), help="override strata")
    args = ap.parse_args()

    sample = build_sample(tuple(args.sites))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(sample, indent=2, ensure_ascii=False))
        print(f"wrote {args.json}")

    if args.emit_yaml:
        print(emit_yaml(sample))
        return

    print(f"WA pilot stratified sample — seed={sample['seed']}, "
          f"quota={sample['quota_per_site']}/site")
    print(f"  n_total = {sample['n_total']} "
          f"(registered target {sample['n_registered_target']}; "
          f"{sample['n_registered_target'] - sample['n_total']} pending gitlab/map strata)")
    for stratum in sample["strata"]:
        print(f"\n  [{stratum['site']}] scored pool={stratum['pool_size_scored']} "
              f"(raw {stratum['pool_size_raw']} - {stratum['n_excluded_na']} N/A) "
              f"-> {stratum['task_ids']}")
        for tid in stratum["task_ids"]:
            intent = stratum["intents"][tid]
            snippet = intent if len(intent) <= 88 else intent[:85] + "..."
            print(f"      {tid:>4}  {snippet}")


if __name__ == "__main__":
    main()
