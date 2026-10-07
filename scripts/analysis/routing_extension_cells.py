#!/usr/bin/env python3
"""Representation routing on the extension cells: GPT-5.6 (B5) on classifieds and both shopping
cells, each with only the arms and tasks a cross-mode reading can trust — 2026-10-07,
post_hoc_exploratory (笔记 §547)

Why this exists. Every routing product so far (§539–§546) reads the 8 cross-mode units
(VWA classifieds/reddit × B0/B1/B2 + WA reddit × B0/B1). Three paper-grade cells sat outside:
  cls_B5   GPT-5.6, the strongest backbone — the "is the baseline strong enough" attack surface
           has no routing-level reading without it. Its vision run is the broken coordinate
           contract (B-1997, §508) and is not in the manifest; the other five arms are fine.
  shop_B0  three arms (DOM / SoM / Vision; the phantom arms were never run — paid).
  shop_B1  six arms.
Shopping was out of the cross-mode products because of B-2002 (a TYPE into a filled search box
submits old + new query; 43% of such submissions, far heavier on text arms), the catalog-grid
order drift (§534) and B-2003 (wishlist not reset). A router on raw shopping would partly learn
to route around the harness.

What this does about it:
  * B5: route among its five valid arms. No task is dropped.
  * shopping: two versions of each cell. `clean` drops, as whole task rows, every task on which
    ANY of the cell's arms has a B-2002-affected episode (detector in lib/shopping_contamination,
    which reproduces the catalog's per-arm counts exactly), the 42 grid-order-sensitive tasks and
    the 4 B-2003 tasks — so no arm is compared on a task where the harness, not the
    representation, moved its outcome. `all` keeps every scored task and is a sensitivity
    reading only. Dropping whole rows keeps the comparison across arms on one task set; it does
    change WHICH tasks are read (contaminated tasks are search tasks), which the report states.

Per cell, the readings of the 8-cell suite, unchanged in method:
  frontier     §539: six-head (one head per available arm) / triage curves vs the fixed modes and
               their random mixtures, label-shuffle null (B=1000) on the max excess. Holm across
               the three primary variants (cls_B5, shop_B0 clean, shop_B1 clean).
  deployable   §544: operating point chosen on training folds only; task bootstrap (B=1000),
               95% upper bound.
  template     §546: the deployable excess with folds grouped by intent_template_id.
Features: the matched 18 of the 8-cell suite (step-0 page statistics read from the cell's own
canonical runs, task-config annotations), so the extension rows are read on the same columns.
Cost: `total_billed_cost_usd` (B5 and B0 are API invoices; B1 is the token-priced estimate).
Nothing here enters the 8-cell pooled numbers.

Usage:
  python scripts/analysis/routing_extension_cells.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis import representation_routing_frontier as rrf  # noqa: E402
from scripts.analysis import router_triage_learnability as rt  # noqa: E402
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402
from scripts.analysis.lib.episode_rows import load_task_rows  # noqa: E402
from scripts.analysis.lib.run_registry import get_all_cells  # noqa: E402
from scripts.analysis.lib.shopping_contamination import (  # noqa: E402
    B2003_TASKS, episode_b2002, grid_order_sensitive_tasks, start_urls,
)
from scripts.analysis.routing_crossrun_template_validation import crossfit as crossfit_eval  # noqa: E402
from scripts.analysis.routing_crossrun_template_validation import folds_by  # noqa: E402
from scripts.analysis.routing_gain_upper_bounds import crossfit, group_folds  # noqa: E402

N_SHUFFLE = 1000
N_BOOT = 1000
SEED_BOOT = 7
OUT_MD = REPO / "docs/analysis/cross_sites/routing_extension_cells.md"
OUT_JSON = REPO / "docs/analysis/cross_sites/routing_extension_cells.json"
VARIANTS = [  # (variant id, baseline, site, modes, clean, primary)
    ("cls_B5", "B5", "classifieds", ("DOM", "SoM", "P-text", "P-prompt", "P-SoM"), False, True),
    ("shop_B0_clean", "B0", "shopping", ("DOM", "SoM", "Vision"), True, True),
    ("shop_B1_clean", "B1", "shopping", rt.SIX_MODES, True, True),
    ("shop_B0_all", "B0", "shopping", ("DOM", "SoM", "Vision"), False, False),
    ("shop_B1_all", "B1", "shopping", rt.SIX_MODES, False, False),
]


# ---------------------------------------------------------------------------- data

def manifest_dirs(baseline: str, site: str, modes) -> dict[str, Path]:
    cells = [c for c in get_all_cells(include_extension=True) if c.baseline == baseline and c.site == site]
    by_mode = {c.mode: Path(c.episodes_dir) for c in cells}
    missing = [m for m in modes if m not in by_mode]
    if missing:
        raise KeyError(f"{baseline}·{site}: no paper-grade manifest entry for {missing}")
    return {m: by_mode[m] for m in modes}


def contaminated_tasks(dirs: dict[str, Path]) -> tuple[set[int], dict[str, int]]:
    hit: set[int] = set()
    per_arm = {}
    for m, ep in dirs.items():
        su = start_urls(ep.parents[1] / "task_configs")
        k = 0
        for sp in ep.glob("*_steps_v2.jsonl"):
            t = int(sp.name.split("_task_")[1].split("_steps")[0])
            if episode_b2002(sp, su.get(t))["appended"] > 0:
                hit.add(t)
                k += 1
        per_arm[m] = k
    return hit, per_arm


def build(variant: tuple) -> dict:
    vid, bl, site, modes, clean, _ = variant
    dirs = manifest_dirs(bl, site, modes)
    rows = {m: load_task_rows(d) for m, d in dirs.items()}
    universe, _ = expected_scored_ids(site)
    tids = sorted(t for t in universe if all(t in rows[m] for m in modes))
    dropped = {}
    if clean:
        hit, per_arm = contaminated_tasks(dirs)
        grid = grid_order_sensitive_tasks()
        dropped = {"b2002_tasks": len(hit & set(tids)), "b2002_episodes_per_arm": per_arm,
                   "grid_order_tasks": len(grid & set(tids)), "b2003_tasks": len(B2003_TASKS & set(tids))}
        tids = [t for t in tids if t not in hit and t not in grid and t not in B2003_TASKS]
        dropped["union_dropped"] = len([t for t in universe if all(t in rows[m] for m in modes)]) - len(tids)
    runs = rt.find_pass1_runs(bl, site)
    X, keep = [], []
    for t in tids:
        fr = rt._feature_row(runs, site, t)
        if fr is None:
            continue
        X.append(fr[0] + fr[1])
        keep.append(t)
    X = np.asarray(X, dtype=float)[:, rrf.FEAT_IDX]
    S = np.array([[bool(rows[m][t].get("success")) for m in modes] for t in keep], dtype=float)
    C = np.array([[float(rows[m][t][rt.COST_FIELD]) for m in modes] for t in keep], dtype=float)
    cfg_dir = dirs[modes[0]].parents[1] / "task_configs"
    tmpl = np.array([int(json.loads((cfg_dir / f"{site}_task_{t}.json").read_text(encoding="utf-8"))
                         ["intent_template_id"]) for t in keep])
    return {"variant": vid, "site": site, "baseline": bl, "modes": list(modes), "task_ids": keep,
            "X": X, "y": S.max(1).astype(int), "S": S, "C": C, "templates": tmpl,
            "n_universe": len(universe), "dropped": dropped, "n_no_feature": len(tids) - len(keep)}


# ---------------------------------------------------------------------------- per variant

def run_variant(variant: tuple, n_shuffle: int, n_boot: int) -> dict:
    t0 = time.time()
    cell = build(variant)
    X, y, S, C = cell["X"], cell["y"], cell["S"], cell["C"]
    n = len(y)
    fr = rrf.evaluate(cell, n_shuffle)          # frontier (§539)
    fr.pop("_null_gain")
    for k in ("cell_id", "site"):
        fr.pop(k, None)
    # deployable (§544)
    dep = crossfit(X, y, S, C, rrf.fold_split(n))
    rng = np.random.default_rng(SEED_BOOT)
    boot = {"six_head": [], "triage": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        st = crossfit(X[idx], y[idx], S[idx], C[idx], group_folds(idx))
        for k in boot:
            boot[k].append(st[k])
    # template-held-out (§546)
    tmpl = crossfit_eval(X, S, C, S, C, folds_by(cell["templates"]))
    out = {
        "variant": cell["variant"], "baseline": cell["baseline"], "site": cell["site"],
        "modes": cell["modes"], "primary": variant[5], "n_tasks": n, "n_universe": cell["n_universe"],
        "n_templates": int(len(np.unique(cell["templates"]))), "dropped": cell["dropped"],
        "n_no_feature": cell["n_no_feature"],
        "sr_by_mode": {m: float(100 * S[:, j].mean()) for j, m in enumerate(cell["modes"])},
        "frontier": {k: fr[k] for k in ("fixed_modes", "fixed_hull", "six_head", "triage", "oracle",
                                        "frontier_gain", "n_shuffle")},
        "deployable": {"observed": dep,
                       "lower05": {k: float(np.quantile(v, 0.05)) for k, v in boot.items()},
                       "upper95": {k: float(np.quantile(v, 0.95)) for k, v in boot.items()}},
        "template_folds": tmpl,
    }
    print(f"{out['variant']}: n={n} six max {fr['six_head']['summary']['max_excess_pp']:+.2f} "
          f"(p {fr['six_head']['summary']['null_p']:.3f}) triage max {fr['triage']['summary']['max_excess_pp']:+.2f} "
          f"(p {fr['triage']['summary']['null_p']:.3f}) · deployable {dep['six_head']:+.2f}/{dep['triage']:+.2f} "
          f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    return out


# ---------------------------------------------------------------------------- render

def render(p: dict) -> str:
    vs = p["variants"]
    L = [
        "# Representation routing on the extension cells — GPT-5.6 (B5) and shopping",
        "",
        "Generated by `scripts/analysis/routing_extension_cells.py`. `post_hoc_exploratory=True`. Chronicle: "
        "实验笔记 §547. Same methods as the 8-cell suite: frontier + label-shuffle null (§539), deployable "
        "cross-fitted router with a task-bootstrap upper bound (§544), template-held-out folds (§546). "
        "Nothing here enters the 8-cell pooled numbers.",
        "",
        "- **cls_B5**: GPT-5.6, five arms; Vision excluded (broken coordinate contract, B-1997).",
        "- **shopping `clean`**: whole task rows dropped where ANY arm of the cell has a B-2002-affected "
        "episode (search box submits old + new query), plus the 42 grid-order-sensitive and 4 B-2003 "
        "(wishlist) tasks. The dropped tasks are mostly search tasks, so `clean` reads a different task "
        "mix, not just a cleaner one. **`all`** keeps every scored task — sensitivity only.",
        "",
        "## 1. Who was dropped",
        "",
        "| variant | arms | tasks read | scored universe | B-2002 tasks | grid-order | B-2003 | dropped (union) | templates |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for v in vs:
        d = v["dropped"]
        L.append(f"| {v['variant']} | {len(v['modes'])} | {v['n_tasks']} | {v['n_universe']} "
                 f"| {d.get('b2002_tasks', '—')} | {d.get('grid_order_tasks', '—')} | {d.get('b2003_tasks', '—')} "
                 f"| {d.get('union_dropped', 0)} | {v['n_templates']} |")
    L += ["", "SR by arm on the tasks read (%):", "", "| variant | " + " | ".join(rt.SIX_MODES) + " |",
          "|---|" + "---|" * len(rt.SIX_MODES)]
    for v in vs:
        L.append(f"| {v['variant']} | " + " | ".join(f"{v['sr_by_mode'][m]:.2f}" if m in v["sr_by_mode"] else "—"
                                                   for m in rt.SIX_MODES) + " |")
    L += ["", "## 2. Frontier (§539 method): max excess over fixed modes + random mixtures", "",
          f"Label-shuffle null B={p['n_shuffle']}; Holm across the three primary variants (cls_B5, "
          "shop_B0_clean, shop_B1_clean).", "",
          "| variant | frontier modes | oracle max | six-head max (p) | Holm | triage max (p) | Holm |",
          "|---|---|---|---|---|---|---|"]
    for v in vs:
        f = v["frontier"]
        s6, st = f["six_head"]["summary"], f["triage"]["summary"]
        hull = " → ".join(h["mode"] for h in f["fixed_hull"])
        hb = lambda k: ("pass" if v["holm"][k] else "—") if v["primary"] else "(sens.)"
        L.append(f"| {v['variant']} | {hull} | +{f['oracle']['summary']['max_excess_pp']:.2f} "
                 f"| {s6['max_excess_pp']:+.2f} ({s6['null_p']:.4f}) | {hb('six_head')} "
                 f"| {st['max_excess_pp']:+.2f} ({st['null_p']:.4f}) | {hb('triage')} |")
    L += ["", "## 3. Deployable router (§544 / §546 methods)", "",
          f"Operating point chosen on training folds only. Task bootstrap B={p['n_boot']}: [5%, **95%**]. "
          "Template folds: the same estimand with folds grouped by intent_template_id.", "",
          "| variant | six-head | [5%, **95%**] | template folds | triage | [5%, **95%**] | template folds |",
          "|---|---|---|---|---|---|---|"]
    for v in vs:
        d = v["deployable"]
        L.append(f"| {v['variant']} | {d['observed']['six_head']:+.2f} | [{d['lower05']['six_head']:+.2f}, "
                 f"**{d['upper95']['six_head']:.2f}**] | {v['template_folds']['six_head']:+.2f} "
                 f"| {d['observed']['triage']:+.2f} | [{d['lower05']['triage']:+.2f}, **{d['upper95']['triage']:.2f}**] "
                 f"| {v['template_folds']['triage']:+.2f} |")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--n-shuffle", type=int, default=N_SHUFFLE)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", type=Path, default=OUT_MD)
    ap.add_argument("--json-out", type=Path, default=OUT_JSON)
    args = ap.parse_args()
    if (args.n_shuffle, args.n_boot) != (N_SHUFFLE, N_BOOT) and (args.out == OUT_MD or args.json_out == OUT_JSON):
        raise SystemExit("non-default B must not overwrite the tracked product")
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        vs = list(ex.map(run_variant, VARIANTS, [args.n_shuffle] * len(VARIANTS), [args.n_boot] * len(VARIANTS)))
    prim = [v for v in vs if v["primary"]]
    for k in ("six_head", "triage"):
        verdict = rrf.holm({v["variant"]: v["frontier"][k]["summary"]["null_p"] for v in prim})
        for v in vs:
            v.setdefault("holm", {})[k] = bool(verdict.get(v["variant"], False))
    payload = {"post_hoc_exploratory": True, "producer": "scripts/analysis/routing_extension_cells.py",
               "n_shuffle": args.n_shuffle, "n_boot": args.n_boot, "features": rrf.FEATURES,
               "cost_field": rt.COST_FIELD, "variants": vs}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out.write_text(render(payload), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
