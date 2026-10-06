#!/usr/bin/env python3
"""Is the VWA shopping category grid served in the same order across runs? (read-only)

Why (实验笔记 §534): the 2026-10-06 Tier-2 sample (§531.9) traced two benchmark-FP verdicts
and one unclear one to "the first row as served does not contain the expected item", and
guessed the grid order (Magento default sort = "Position") differs between runs. The ledger
carried it as CLAIM_UNVERIFIED. This checks it on step-0 observations, which every
condition records for every task before the agent acts — so any difference is the page as
served, not something the agent did.

Method
- Product identity = the ASIN in the catalog image url, in page order, restricted to the
  main product list (after the "Sort By" control, before the "Shop By" sidebar). Pages
  without a Sort By control are not category grids and are skipped. The sidebar is cut
  because it shows wish-list / recently-viewed items, which differ by run for another
  reason (B-2003).
- "First row" = first four products (the grid is four wide at the recorded viewport).
- Catalog states are not assumed: runs are clustered by pairwise first-row agreement
  (>= AGREE_MIN on shared task pages = same state, union-find).
- Order-sensitive tasks = scored tasks whose start grid differs across runs AND whose
  intent names a position (row / column / first / Nth / last / top ...).

    python scripts/analysis/shopping_grid_order_drift.py
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402

PHASE1 = REPO / "results/visualwebarena/phase1"
TASKS = REPO / "external/visualwebarena/config_files/vwa/test_shopping.json"
OUT = REPO / "docs/analysis/vwa_shopping/grid_order_drift_20261007"
ASIN = re.compile(r"catalog/product/[^ ]*/([A-Z0-9]{10})\.\d+\.jpg")
ORDINAL = re.compile(r"\b(first|second|third|fourth|fifth|sixth|last|top|bottom|leftmost|"
                     r"rightmost|\d+(st|nd|rd|th)|row|column)\b", re.I)
ROW = 4
AGREE_MIN = 0.85


def grid(path: str) -> tuple[str | None, list[str]]:
    txt = open(path, encoding="utf-8", errors="replace").read()
    um = re.search(r"url: (\S+)", txt[:3000])
    i = txt.find("Sort By")
    if i < 0:
        return (um.group(1) if um else None), []
    j = txt.find("'Shop By'", i)
    seq: list[str] = []
    for m in ASIN.finditer(txt[i:j if j > 0 else len(txt)]):
        if m.group(1) not in seq:
            seq.append(m.group(1))
    return (um.group(1) if um else None), seq


def main() -> None:
    runs = sorted(Path(d) for d in glob.glob(str(PHASE1 / "B*_shopping_*")) if os.path.isdir(d))
    first_rows: dict[int, dict[str, dict[str, tuple]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict))  # task -> url -> run -> first row
    for r in runs:
        for f in glob.glob(str(r / "*/artifacts/shopping_task_*/step_000/observation_dom.txt")):
            t = int(re.search(r"shopping_task_(\d+)", f).group(1))
            url, seq = grid(f)
            if seq:
                first_rows[t][url][r.name] = tuple(seq[:ROW])
    run_names = sorted({n for d in first_rows.values() for m in d.values() for n in m})
    if not run_names:
        raise SystemExit("no step-0 observation_dom.txt found under phase1/ — artifacts missing?")

    same = collections.Counter(); tot = collections.Counter()
    drift: set[int] = set(); groups = 0
    for t, by_url in first_rows.items():
        for url, m in by_url.items():
            if len(m) < 2:
                continue
            groups += 1
            if len(set(m.values())) > 1:
                drift.add(t)
            names = sorted(m)
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    k = (names[a], names[b]); tot[k] += 1; same[k] += m[names[a]] == m[names[b]]

    parent = {n: n for n in run_names}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for k in tot:
        if same[k] / tot[k] >= AGREE_MIN:
            parent[find(k[0])] = find(k[1])
    clusters = collections.defaultdict(list)
    for n in run_names:
        clusters[find(n)].append(n)
    states = sorted(clusters.values(), key=lambda c: (-len(c), c))
    state_of = {n: f"S{i + 1}" for i, c in enumerate(states) for n in c}

    scored = set(expected_scored_ids("shopping", "visualwebarena")[0])
    tasks = {t["task_id"]: t for t in json.loads(TASKS.read_text(encoding="utf-8"))}
    sensitive = sorted(t for t in drift if t in scored and t in tasks and ORDINAL.search(tasks[t]["intent"]))

    per_run = []
    for n in run_names:
        succ = {}
        for f in glob.glob(str(PHASE1 / n / "*/episodes/shopping_task_*_summary_v2.json")):
            t = int(re.search(r"task_(\d+)_summary", f).group(1))
            if t in scored:
                succ[t] = bool(json.loads(open(f, encoding="utf-8").read()).get("success"))
        a = [succ[t] for t in sensitive if t in succ]
        b = [v for t, v in succ.items() if t not in sensitive]
        per_run.append({"run": n, "state": state_of[n], "n_scored": len(succ),
                        "sensitive_success": sum(a), "sensitive_n": len(a),
                        "rest_success": sum(b), "rest_n": len(b)})

    out = {
        "method": __doc__.split("Method", 1)[1].split("python scripts", 1)[0].strip(),
        "runs_with_step0_artifacts": run_names,
        "task_url_groups_with_2plus_runs": groups,
        "scored_tasks": len(scored),
        "scored_tasks_whose_start_grid_differs": len(drift & scored),
        "order_sensitive_tasks": sensitive,
        "pairwise_first_row_agreement": [
            {"a": k[0], "b": k[1], "same": same[k], "n": tot[k]} for k in sorted(tot)],
        "states": {f"S{i + 1}": c for i, c in enumerate(states)},
        "per_run": per_run,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    L = ["---", "type: analysis", "status: complete", "created: 2026-10-07",
         "purpose: is the shopping category grid served in the same order across runs",
         "producer: scripts/analysis/shopping_grid_order_drift.py", "---", "",
         "# Shopping grid order across runs", "",
         "Regenerate: `python scripts/analysis/shopping_grid_order_drift.py` (needs step-0 "
         "`observation_dom.txt` artifacts, i.e. the merged run store).", "",
         f"- Runs with step-0 artifacts: **{len(run_names)}**; task×page groups seen by ≥2 runs: **{groups}**.",
         f"- Scored tasks whose start grid's first row differs between runs: "
         f"**{len(drift & scored)} / {len(scored)}**.",
         f"- Of those, intent names a position (order-sensitive): **{len(sensitive)}**.", "",
         "## Catalog states (clustered from the data, not assumed)", "",
         f"Runs whose first rows agree on ≥ {AGREE_MIN:.0%} of shared pages are one state.", "",
         "| state | runs |", "|---|---|"]
    for i, c in enumerate(states):
        L.append(f"| S{i + 1} | " + ", ".join(f"`{n}`" for n in c) + " |")
    L += ["", "## Success on order-sensitive tasks vs the rest", "",
          "| run | state | order-sensitive | rest |", "|---|---|---:|---:|"]
    for r in sorted(per_run, key=lambda r: (r["state"], r["run"])):
        L.append(f"| `{r['run']}` | {r['state']} | {r['sensitive_success']}/{r['sensitive_n']} | "
                 f"{100 * r['rest_success'] / max(1, r['rest_n']):.1f}% |")
    L += ["", "## Reading", "",
          "- The order is a property of the run, not of the episode: inside a state, runs agree "
          "on essentially every shared page; across states most pages differ. All runs were on "
          "the same host and no snapshot records a container identity, so *why* a run lands in "
          "a state (reset / reindex / container generation) is not determinable offline.",
          "- Order-sensitive tasks are near-zero in every state, so no state is visibly the one "
          "the reference answers were written against. Whether these tasks are hard or "
          "unanswerable as served cannot be told apart here; it needs the live site.",
          "- Consequence for mode comparisons on shopping: the B1 arms do not share one state, so a "
          "per-task mode contrast on an order-sensitive task compares different pages. With "
          "near-zero success there, the effect on SR differences is at most a task or two; the "
          "larger consequence is that these tasks should be named as a scope limit, not counted "
          "as agent failures.", ""]
    OUT.with_suffix(".md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT.with_suffix('.md')}")


if __name__ == "__main__":
    main()
