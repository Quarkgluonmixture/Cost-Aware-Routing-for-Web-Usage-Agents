#!/usr/bin/env python3
"""One table of every run/condition we have, with completeness, SR and registry status.

Input: `conditions.jsonl` from `inventory_all_runs.py` (all copies on all hosts).
For each unique (run_id, condition_id) the most complete copy is read (ties broken by
source priority local > a100_ws > a100_scratch > a100_archives > dgx), and:

  - completeness against the COLLECTION set (`collected_task_ids`)
  - SR over the canonical SCORED set (`expected_scored_ids`, AMENDMENT_08/09/10) —
    reported only for complete runs; partial runs get `sr_partial` over what exists
  - registry status: `run_manifest.yaml` section + grade, `CLEAN_PAIRS` role
  - how often the run is cited in docs/ (full run_id, and its R-token)

Descriptive only: this is a coverage map for the evidence layer, not a paper-facing estimand.

Usage:
    python scripts/maintenance/run_inventory_report.py --inv docs/analysis/run_inventory
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

SRC_PRIORITY = ["local", "a100_ws", "a100_scratch", "a100_archives", "dgx"]
SRC_ROOT = {
    "local": "C:/Workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results",
    "a100_ws": "E:/a100-condenser-backup/home-ubuntu/workspace/p79/results",
    "a100_scratch": "E:/a100-condenser-backup/mnt-scratch/p79_results_active_visualwebarena",
    "a100_archives": "E:/a100-condenser-backup/mnt-scratch/p79_archives",
    "dgx": "E:/dgx-jiaming-backup/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results",
}
RUN_RE = re.compile(r"^(?P<b>B\d+)_(?P<mode>.+?)_(?P<site>wa_reddit|wa_shopping_admin|wa_shopping|classifieds|reddit|shopping)_")


def load_manifest():
    import yaml
    m = yaml.safe_load(open(REPO / "results/phantom_paper/run_manifest.yaml", encoding="utf-8"))
    idx = {}
    for section in ("cells", "in_flight", "archived", "extension"):
        for e in m.get(section) or []:
            key = (Path(str(e.get("run_dir"))).name, e.get("condition_subdir"))
            idx[key] = f"{section}:{e.get('grade')}"
    return idx


def load_clean_pairs():
    src = (REPO / "scripts/analysis/aggregate_noise_floor_inventory.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CLEAN_PAIRS" for t in node.targets):
            pairs = ast.literal_eval(node.value)
            break
    else:
        return {}
    roles = {}
    for label, a, b in pairs:
        for role, p in (("canonical", a), ("replicate", b)):
            parts = Path(p).parts
            roles[(parts[-2], parts[-1])] = f"{label}:{role}"
    return roles


def doc_corpus() -> str:
    chunks = []
    for p in (REPO / "docs").rglob("*.md"):
        if "run_inventory" in p.parts:
            continue
        try:
            chunks.append(p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    return "\n".join(chunks)


def read_outcomes(cond_dir: Path):
    out = {}
    ep = cond_dir / "episodes"
    for f in ep.glob("*_summary_v2.json"):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out[int(d.get("task_id"))] = (bool(d.get("success")), d.get("benchmark"), d.get("benchmark_site"),
                                      d.get("total_billed_cost_usd"), d.get("cost_unit_basis"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv", required=True)
    args = ap.parse_args()
    inv = Path(args.inv)
    rows = [json.loads(l) for l in open(inv / "conditions.jsonl", encoding="utf-8")]

    from lib.canonical_task_universe import collected_task_ids, expected_scored_ids  # noqa: E402

    manifest = load_manifest()
    pairs = load_clean_pairs()
    corpus = doc_corpus()

    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["run_id"], r["condition_id"])].append(r)

    out = []
    universe_cache = {}
    for (run_id, cond_id), copies in sorted(groups.items()):
        best = sorted(copies, key=lambda r: (-r["n_summary"], SRC_PRIORITY.index(r["source"])))[0]
        cond_dir = Path(SRC_ROOT[best["source"]]) / best["relpath"]
        outcomes = read_outcomes(cond_dir)
        bench = site = None
        for _, (_, b, s, _, _) in outcomes.items():
            bench, site = b, s
            break
        m = RUN_RE.match(run_id.lstrip("_"))
        rec = {
            "run_id": run_id,
            "condition_id": cond_id,
            "baseline": m.group("b") if m else None,
            "mode": best.get("observation_mode"),
            "benchmark": bench or best.get("benchmark"),
            "site": site,
            "model_name": best.get("model_name"),
            "sources": sorted({c["source"] for c in copies}),
            "best_source": best["source"],
            "n_episodes": len(outcomes),
            "max_artifact_dirs": max(c["n_artifact_dirs"] for c in copies),
            "copies_agree": len({c["episode_fingerprint"] for c in copies if c["n_summary"] == best["n_summary"]}) == 1,
            "manifest": manifest.get((run_id, cond_id)),
            "clean_pair": pairs.get((run_id, cond_id)),
            "is_archive_name": run_id.startswith("_archive") or best["source"] == "a100_archives",
            "doc_refs_run_id": corpus.count(run_id),
        }
        rtok = re.search(r"_(R\d{2,6})(?:_|$)", run_id)
        rec["r_token"] = rtok.group(1) if rtok else None
        rec["doc_refs_r_token"] = len(re.findall(rf"\b{rtok.group(1)}\b", corpus)) if rtok else None
        if bench and site:
            key = (site, bench)
            if key not in universe_cache:
                try:
                    universe_cache[key] = (collected_task_ids(site, bench), expected_scored_ids(site, bench)[0])
                except (FileNotFoundError, ValueError) as e:
                    universe_cache[key] = None
                    print(f"WARN universe {key}: {e}", file=sys.stderr)
            uni = universe_cache[key]
            if uni:
                collected, scored = uni
                have = set(outcomes)
                rec["n_collected_expected"] = len(collected)
                rec["completeness"] = round(len(have & collected) / len(collected), 4)
                sc_have = have & scored
                succ = sum(1 for t in sc_have if outcomes[t][0])
                rec["n_scored"] = len(scored)
                rec["n_success_scored"] = succ
                if scored <= have:
                    rec["sr"] = round(succ / len(scored), 4)
                elif sc_have:
                    rec["sr_partial"] = round(succ / len(sc_have), 4)
        out.append(rec)

    with open(inv / "run_inventory.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} rows -> {inv / 'run_inventory.json'}", file=sys.stderr)
    (inv / "run_matrix.md").write_text(render_matrix(out), encoding="utf-8")
    print(f"wrote {inv / 'run_matrix.md'}", file=sys.stderr)
    return 0


MODES = ["dom", "som", "vision", "phantom_text", "phantom_prompt", "phantom_som"]
MODE_LABEL = {"dom": "DOM", "som": "SoM", "vision": "Vision", "phantom_text": "P-text",
              "phantom_prompt": "P-prompt", "phantom_som": "P-SoM"}


def status_tag(r) -> str:
    if r["manifest"]:
        sec = r["manifest"].split(":")[0]
        return {"cells": "P", "extension": "X", "in_flight": "I", "archived": "A"}.get(sec, "?")
    if r["clean_pair"]:
        return "R"
    return "U"


def render_matrix(rows) -> str:
    """Site × baseline × mode table of FULL runs only (completeness == 1, not archive/smoke/diag)."""
    full = [r for r in rows if r.get("completeness") == 1 and not r["is_archive_name"]
            and "smoke" not in r["run_id"] and not r["run_id"].startswith("diag_")]
    cells = collections.defaultdict(list)
    for r in full:
        site = f"{'WA' if r['benchmark'] == 'webarena' else 'VWA'}·{r['site']}"
        cells[(site, r["baseline"], r["mode"])].append(r)
    keys = sorted({(s, b) for s, b, _ in cells}, key=lambda k: (k[0], int(k[1][1:])))
    lines = [
        "# Run matrix — full runs only",
        "",
        "Generated by `scripts/maintenance/run_inventory_report.py`. Each cell lists every complete run of",
        "that (site, baseline, mode): `SR%` over the canonical scored set, then a registry tag —",
        "**P** run_manifest `cells` · **X** `extension` · **R** CLEAN_PAIRS replicate · **U** unregistered",
        "(shopping / WA are discovered by glob, so U there is not by itself a defect). Two entries in one",
        "cell = a same-condition replicate pair exists on disk.",
        "",
        "| site | baseline | " + " | ".join(MODE_LABEL[m] for m in MODES) + " |",
        "|---|---|" + "---|" * len(MODES),
    ]
    for site, b in keys:
        tds = []
        for m in MODES:
            rs = sorted(cells.get((site, b, m), []), key=lambda r: r["run_id"])
            tds.append("<br>".join(f"{100 * r['sr']:.1f} {status_tag(r)}" for r in rs) or "—")
        lines.append(f"| {site} | {b} | " + " | ".join(tds) + " |")
    lines += ["", f"Full runs counted: {len(full)}.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
