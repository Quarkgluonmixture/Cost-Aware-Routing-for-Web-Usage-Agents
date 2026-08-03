#!/usr/bin/env python3
"""Which recorded fields does the evidence layer actually consume? — 2026-08-03

Why this exists
---------------
`confidence.verbalized` — the model's own stated confidence — is written on 75-100% of
steps in every cell. `analyze_confidence_calibration` reads it and `mechanism_per_task`
E3 reports it as often the STRONGEST single signal available. And
`aggregate_confidence_cascade`, the product carrying the "confidence-triggered escalation
fails" result, did not have it in its signal list at all.

The existing §G1 sweep asked "which fields does nobody read?" and answered it. It could
not have caught this one, because the answer for `verbalized` was "several scripts read
it". The defect is a different shape:

    a field that IS consumed, but not by the product whose conclusion it would change.

So this tool does not produce a verdict. It produces the matrix — every recorded field
against every product that names it — and sorts it so the thin rows surface. Reading the
matrix is a human judgement, the same way the "live features" disclosure column is what
actually surfaced the three dead router features (B-1928) rather than any code review.

Three classes it separates:
  DEAD     0% populated. Nobody can use it. (§G1 already listed these.)
  ORPHAN   populated, referenced by no analysis script. Recorded for nothing.
  THIN     populated, referenced by 1-2 scripts. The `verbalized` shape — check by hand
           whether the products missing it would change if they had it.

Usage
-----
    .venv/bin/python3 scripts/analysis/audit_field_consumption.py
    .venv/bin/python3 scripts/analysis/audit_field_consumption.py --max-consumers 3
    .venv/bin/python3 scripts/analysis/audit_field_consumption.py --json-out <path>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SCRIPT_DIRS = [REPO / "scripts/analysis", REPO / "scripts/analysis/figures"]
# Names too generic to attribute a mention to: a grep for "total" hits everything.
GENERIC = {"total", "input", "output", "model", "source", "enabled", "text", "type",
           "seed", "value", "count", "name", "id", "step", "index", "reason", "url",
           "success", "error", "status", "mode", "site", "task_id", "run_id"}


def flatten(obj: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def populated(v) -> bool:
    return v is not None and v != "" and v != [] and v != {}


def sample_records(per_cell: int) -> tuple[dict, dict, int, int]:
    """(step_stats, ep_stats, n_steps, n_eps) — populated counts per flattened field."""
    from scripts.analysis.lib.run_registry import get_cells
    step_hits: dict[str, int] = defaultdict(int)
    step_vals: dict[str, set] = defaultdict(set)
    ep_hits: dict[str, int] = defaultdict(int)
    ep_vals: dict[str, set] = defaultdict(set)
    n_steps = n_eps = 0
    for bl in ("B0", "B1", "B2"):
        for site in ("classifieds", "reddit"):
            for cell in get_cells(baseline=bl, site=site):
                eps = sorted(Path(cell.episodes_dir).glob("*steps_v2.jsonl"))[:per_cell]
                for sp in eps:
                    for line in sp.read_text().splitlines():
                        if not line.strip():
                            continue
                        try:
                            rec = flatten(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                        n_steps += 1
                        for k, v in rec.items():
                            if populated(v):
                                step_hits[k] += 1
                                if len(step_vals[k]) < 6 and not isinstance(v, (list, dict)):
                                    step_vals[k].add(str(v)[:24])
                    sm = sp.with_name(sp.name.replace("_steps_v2.jsonl", "_summary_v2.json"))
                    if not sm.exists():
                        continue
                    try:
                        rec = flatten(json.loads(sm.read_text()))
                    except (OSError, json.JSONDecodeError):
                        continue
                    n_eps += 1
                    for k, v in rec.items():
                        if populated(v):
                            ep_hits[k] += 1
                            if len(ep_vals[k]) < 6 and not isinstance(v, (list, dict)):
                                ep_vals[k].add(str(v)[:24])
    return ({k: (v, step_vals[k]) for k, v in step_hits.items()},
            {k: (v, ep_vals[k]) for k, v in ep_hits.items()}, n_steps, n_eps)


def build_index() -> dict[str, str]:
    return {p.name: p.read_text(errors="ignore")
            for d in SCRIPT_DIRS if d.is_dir()
            for p in sorted(d.glob("*.py"))}


def consumers(field: str, index: dict[str, str], self_name: str) -> list[str] | None:
    """Scripts naming the field. None means 'cannot be attributed' — the leaf is a word
    like `total` or `success` that appears everywhere, so absence of a match proves
    nothing and a match proves nothing either. Those are reported separately rather than
    counted as unconsumed, which is what made the first run call `task_id` an orphan.

    A field is matched on its leaf, its full dotted path, or `parent.leaf` — products
    reach these three ways (`r["confidence"]["verbalized"]`, `r["latency_ms.total"]`,
    `rec.latency_ms["total"]`), and leaf-only matching under-counts."""
    leaf = field.split(".")[-1]
    if leaf in GENERIC:
        return None
    forms = {leaf, field}
    if "." in field:
        forms.add(".".join(field.split(".")[-2:]))
    pat = re.compile("|".join(rf'(?:["\']){re.escape(f)}(?:["\'])|\.{re.escape(f)}\b'
                              for f in sorted(forms)))
    return [n for n, src in index.items() if n != self_name and pat.search(src)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-cell", type=int, default=6, help="episodes sampled per cell")
    ap.add_argument("--max-consumers", type=int, default=2,
                    help="flag fields with at most this many consuming scripts")
    ap.add_argument("--min-populated", type=float, default=5.0,
                    help="percent below which a field counts as DEAD, not THIN")
    ap.add_argument("--json-out", type=Path)
    a = ap.parse_args()

    step_stats, ep_stats, n_steps, n_eps = sample_records(a.per_cell)
    index = build_index()
    self_name = Path(__file__).name

    rows = []
    for kind, stats, denom in (("step", step_stats, n_steps), ("episode", ep_stats, n_eps)):
        for field, (hits, vals) in stats.items():
            pct = 100.0 * hits / denom if denom else 0.0
            cons = consumers(field, index, self_name)
            if cons is None:
                klass, n = "GENERIC", -1
            elif pct < a.min_populated:
                klass, n = "DEAD", len(cons)
            elif not cons:
                klass, n = "ORPHAN", 0
            elif len(cons) <= a.max_consumers:
                klass, n = "THIN", len(cons)
            else:
                klass, n = "OK", len(cons)
            rows.append({"kind": kind, "field": field, "populated_pct": round(pct, 1),
                         "n_consumers": n, "consumers": sorted(cons or []),
                         "sample_values": sorted(vals)[:4], "klass": klass})
    rows.sort(key=lambda r: (r["n_consumers"], -r["populated_pct"]))

    by = defaultdict(list)
    for r in rows:
        by[r["klass"]].append(r)

    print(f"[audit] sampled {n_steps} steps / {n_eps} episodes over "
          f"{a.per_cell} episodes per cell; {len(index)} analysis scripts indexed")
    print(f"[audit] {len(rows)} distinct fields — "
          + " · ".join(f"{k} {len(by[k])}" for k in ("OK", "THIN", "ORPHAN", "DEAD", "GENERIC")))

    for klass, blurb in (
        ("ORPHAN", "populated, and NO analysis script names them — recorded for nothing"),
        ("THIN", f"populated, named by ≤{a.max_consumers} script(s) — the `verbalized` "
                 "shape: check by hand whether a product that omits it would change"),
    ):
        sel = [r for r in by[klass] if r["populated_pct"] >= a.min_populated]
        if not sel:
            continue
        print(f"\n=== {klass} — {blurb} ===")
        for r in sel:
            vals = ", ".join(r["sample_values"])
            print(f"  {r['field']:52} {r['populated_pct']:5.1f}%  "
                  f"[{r['kind']}]   值样例: {vals[:52]}")
            if r["consumers"]:
                print(f"    {'':50} 消费者: {', '.join(r['consumers'])}")

    dead = [r for r in by["DEAD"]]
    if dead:
        print(f"\n=== DEAD — <{a.min_populated}% populated ({len(dead)}) ===")
        print("  " + ", ".join(sorted(r["field"] for r in dead)))

    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(
            {"n_steps": n_steps, "n_episodes": n_eps, "per_cell": a.per_cell,
             "n_scripts_indexed": len(index), "fields": rows},
            ensure_ascii=False, indent=1))
        print(f"\n[audit] wrote {a.json_out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
