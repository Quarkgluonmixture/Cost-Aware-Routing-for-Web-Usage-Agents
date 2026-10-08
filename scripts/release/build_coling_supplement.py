#!/usr/bin/env python3
"""Build the anonymous supplementary material for the COLING 2027 (ARR October 2026) submission.

Writes deliverables/coling2027/supplement/ and deliverables/coling2027/coling2027_supplement.zip
(both gitignored; rebuild with this script). Contents:

  data/episodes.csv   one row per (cell, mode, run, task): success, cost in USD, wall-clock and
                      GPU seconds — the inputs every routing / null-model product reads
  data/features.csv   the 18 pre-run router features per (cell, task)
  data/tasks.csv      scored task ids, intent template, intent-rule flag, shopping clean-set flags
  prompts/            the per-mode system prompts, as the agents build them
  configs/            run configurations, comments stripped, API endpoints redacted
  products/           every analysis product the paper cites (JSON + Markdown)
  code/               the scripts that wrote those products, with their in-repo imports
  README.md           layout, how each paper number maps to a product field, what is not included

Anonymisation is enforced, not hoped for: every text file is rewritten through `redact`, then
the whole tree is scanned for identity tokens and the build fails on any hit.

Usage:
  python scripts/release/build_coling_supplement.py
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis import representation_routing_frontier as rrf  # noqa: E402
from scripts.analysis import router_triage_learnability as rt  # noqa: E402
from scripts.analysis import routing_extension_cells as rec  # noqa: E402
from scripts.analysis import visual_intent_routing as vir  # noqa: E402
from scripts.analysis.aggregate_phantom_lift import CELLS  # noqa: E402
from scripts.analysis.lib.episode_rows import load_task_rows  # noqa: E402
from scripts.analysis.lib.replicate_pairs import clean_pairs  # noqa: E402
from scripts.analysis.routing_crossrun_template_validation import templates  # noqa: E402

OUT = REPO / "deliverables/coling2027/supplement"
ZIP = REPO / "deliverables/coling2027/coling2027_supplement.zip"
CS = REPO / "docs/analysis/cross_sites"

PRODUCTS = [
    "routing_three_arm", "routing_three_arm_wallclock", "routing_three_arm_gpu_time",
    "representation_routing_frontier", "representation_routing_frontier_wallclock",
    "representation_routing_frontier_gpu_time", "routing_gain_upper_bounds", "routing_extension_cells",
    "routing_crossrun_template_validation", "task_mode_interaction_null", "visual_intent_routing",
    "fusion_premium", "leakage_sensitivity", "noise_floor_inventory", "reddit_sidebar_leakage_audit_with_wa",
]
ENTRY_SCRIPTS = [
    "scripts/analysis/routing_three_arm.py", "scripts/analysis/representation_routing_frontier.py",
    "scripts/analysis/routing_gain_upper_bounds.py", "scripts/analysis/routing_extension_cells.py",
    "scripts/analysis/routing_crossrun_template_validation.py", "scripts/analysis/task_mode_interaction_null.py",
    "scripts/analysis/visual_intent_routing.py", "scripts/analysis/aggregate_noise_floor_inventory.py",
    "scripts/analysis/figures/coling/make_figures.py",
]
MODE_FILE = {"DOM": "dom", "SoM": "som", "Vision": "vision", "P-text": "phantom_text",
             "P-prompt": "phantom_prompt", "P-SoM": "phantom_som"}
PAIR_MODE = {"dom": "DOM", "som": "SoM", "vision": "Vision", "ptext": "P-text", "pprompt": "P-prompt",
             "psom": "P-SoM"}
PAIR_SITE = {"cls": ("visualwebarena", "classifieds"), "red": ("visualwebarena", "reddit"),
             "wared": ("webarena", "reddit")}

# Identity tokens: the build fails if any survives redaction (case-insensitive).
BANNED = ["quarkgluon", "quark", "jiaming", "ucab352", "ucl.ac.uk", "condenser", "spark-9ea3",
          "administrator", "wubbalabbadubdub", "i5xpracyci", "amazonaws.com/model-api", "holistic", "zekun",
          "/home/ubuntu", "c:\\users", "c:/users", "c:\\workspace", "c:/workspace", "e:\\", "a100-"]
REDACT = [
    (re.compile(r"https?://[a-z0-9]+\.execute-api\.[a-z0-9-]+\.amazonaws\.com\S*", re.I), "<redacted-api-endpoint>"),
    (re.compile(re.escape(str(REPO)).replace(r"\\", r"[\\/]"), re.I), "<repo>"),
    (re.compile(r"C:[\\/]+Workspace[\\/]+Cost-Aware-Routing-for-Web-Usage-Agents", re.I), "<repo>"),
    # a drive letter, not the "p:" of "http:" (2026-10-08: the case-insensitive first version
    # rewrote every http:// URL in the prompts)
    (re.compile(r"(?<![A-Za-z0-9])[C-Z]:[\\/]+[^\s\"'`|)]*"), "<local-path>"),
    (re.compile(r"/home/ubuntu/[^\s\"'`|)]*"), "<remote-path>"),
    (re.compile(r"/mnt/scratch/[^\s\"'`|)]*"), "<remote-path>"),
    (re.compile(r"github\.com/[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+"), "<anonymous-repository>"),
    (re.compile(r"a100-[a-z0-9-]+", re.I), "<gpu-host>"),
    (re.compile(r"spark-9ea3", re.I), "<gpu-host>"),
    (re.compile(r"\b(condenser|Condenser|CONDENSER)\b"), "<gpu-host>"),
    (re.compile(r"\bquark\b", re.I), "<workstation>"),
    (re.compile(r"\bjiaming\b", re.I), "<author>"),
    (re.compile(r"\bUCL\b"), "<institution>"),
    (re.compile(r"\bZekun\b", re.I), "<advisor>"),
]


def redact(text: str) -> str:
    for pat, rep in REDACT:
        text = pat.sub(rep, text)
    return text


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(redact(text), encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------- data

def export_data() -> dict:
    ep_rows, feat_rows, tmpl = [], [], {}
    feat_names = list(rrf.FEATURES)
    counts = {}
    specs = list(CELLS) + list(rt.WA_CELLS)
    for spec in specs:
        cells = {b: rrf.load_cell(spec, b) for b in ("billed", "wallclock")}
        local = spec["baseline"] in ("B1", "B2")
        if local:
            cells["gpu_time"] = rrf.load_cell(spec, "gpu_time")
        c = cells["billed"]
        bench = "webarena" if spec.get("_wa") else "visualwebarena"
        cid = f"{'wared' if spec.get('_wa') else rrf.SITE_KEY[spec['site']]}_{spec['baseline']}"
        tids = c["task_ids"]
        for j, m in enumerate(rrf.MODES):
            for i, t in enumerate(tids):
                ep_rows.append({"cell": cid, "benchmark": bench, "site": "reddit" if spec.get("_wa") else spec["site"],
                                "backbone": spec["baseline"], "mode": m, "run": "a", "task_id": t,
                                "success": int(c["S"][i, j]), "cost_usd": round(float(c["C"][i, j]), 8),
                                "wallclock_s": round(float(cells["wallclock"]["C"][i, j]), 3),
                                "gpu_s": round(float(cells["gpu_time"]["C"][i, j]), 3) if local else ""})
        for i, t in enumerate(tids):
            feat_rows.append({"cell": cid, "task_id": t, **{f: float(c["X"][i, k]) for k, f in enumerate(feat_names)}})
        for t, g in zip(tids, templates(spec, tids)):
            tmpl[(bench, "reddit" if spec.get("_wa") else spec["site"], t)] = int(g)
        counts[cid] = len(tids)
    # extension cells: B5 classifieds and shopping (all tasks; clean membership goes to tasks.csv)
    clean_ids = {}
    for v in rec.VARIANTS:
        vid, bl, site, modes, clean, _ = v
        b = rec.build(v)
        if clean:
            clean_ids[bl] = set(b["task_ids"])
            continue
        cid = vid.replace("_all", "")
        for j, m in enumerate(b["modes"]):
            for i, t in enumerate(b["task_ids"]):
                ep_rows.append({"cell": cid, "benchmark": "visualwebarena", "site": site, "backbone": bl, "mode": m,
                                "run": "a", "task_id": t, "success": int(b["S"][i, j]),
                                "cost_usd": round(float(b["C"][i, j]), 8), "wallclock_s": "", "gpu_s": ""})
        for i, t in enumerate(b["task_ids"]):
            feat_rows.append({"cell": cid, "task_id": t, **{f: float(b["X"][i, k]) for k, f in enumerate(feat_names)}})
            tmpl[("visualwebarena", site, t)] = int(b["templates"][i])
        counts[cid] = len(b["task_ids"])
    # registered same-condition reruns: arm b of every pair
    n_b = 0
    for label, _a, bdir in clean_pairs():
        bl, st, md = label.split(".")
        bench, site = PAIR_SITE[st]
        rows = load_task_rows(bdir / "episodes")
        for t, r in sorted(rows.items()):
            ep_rows.append({"cell": f"{st}_{bl}", "benchmark": bench, "site": site, "backbone": bl,
                            "mode": PAIR_MODE[md], "run": "b", "task_id": t, "success": int(bool(r.get("success"))),
                            "cost_usd": round(float(r.get(rt.COST_FIELD) or 0.0), 8),
                            "wallclock_s": round(float(r["total_latency_canonical_ms"]) / 1000, 3)
                            if r.get("total_latency_canonical_ms") else "", "gpu_s": ""})
            n_b += 1
    d = OUT / "data"
    d.mkdir(parents=True, exist_ok=True)
    cols = ["cell", "benchmark", "site", "backbone", "mode", "run", "task_id", "success", "cost_usd",
            "wallclock_s", "gpu_s"]
    with (d / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(ep_rows)
    with (d / "features.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "task_id"] + feat_names)
        w.writeheader()
        w.writerows(feat_rows)
    flags = {("visualwebarena", s): vir.ex_ante_set(s) for s in ("classifieds", "reddit", "shopping")}
    wa_universe = {t for (b, s, t) in tmpl if b == "webarena"}
    flags[("webarena", "reddit")] = vir.wa_ex_ante_set("B0", wa_universe)
    with (d / "tasks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "site", "task_id", "intent_template_id", "intent_rule_flag",
                    "shopping_clean_B0", "shopping_clean_B1"])
        for (b, s, t), g in sorted(tmpl.items()):
            shop = s == "shopping"
            w.writerow([b, s, t, g, int(t in flags[(b, s)]), int(t in clean_ids["B0"]) if shop else "",
                        int(t in clean_ids["B1"]) if shop else ""])
    return {"episodes": len(ep_rows), "rerun_rows": n_b, "cells": counts, "tasks": len(tmpl)}


# ---------------------------------------------------------------------------- prompts / configs

def export_prompts() -> int:
    from p79.agents._shared_vl_utils import build_mode_prompt_dispatch_table
    table = build_mode_prompt_dispatch_table()
    for mode, text in table.items():
        write(OUT / "prompts" / f"{mode}.txt", text)
    return len(table)


def export_configs() -> int:
    names = ["exp_v2_base.yaml"]
    for bl in ("B0", "B1", "B2", "B5"):
        for m in MODE_FILE.values():
            for site in ("classifieds", "reddit", "shopping", "wa_reddit"):
                f = REPO / "configs" / f"exp_v2_{bl}_{m}_{site}.yaml"
                if f.exists():
                    names.append(f.name)
    for n in names:
        data = yaml.safe_load((REPO / "configs" / n).read_text(encoding="utf-8"))   # drops comments
        write(OUT / "configs" / n, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return len(names)


# ---------------------------------------------------------------------------- products / code

def export_products() -> int:
    k = 0
    for p in PRODUCTS:
        for ext in (".json", ".md"):
            src = CS / f"{p}{ext}"
            if not src.exists():
                raise FileNotFoundError(src)
            write(OUT / "products" / src.name, src.read_text(encoding="utf-8"))
            k += 1
    return k


IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+(scripts\.analysis[\w.]*|p79[\w.]*)", re.M)


def _module_file(mod: str) -> Path | None:
    base = REPO / Path(*mod.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.exists():
            return cand
    return None


def export_code() -> int:
    todo = [REPO / s for s in ENTRY_SCRIPTS]
    seen: set[Path] = set()
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for mod in IMPORT_RE.findall(f.read_text(encoding="utf-8")):
            mf = _module_file(mod)
            if mf is not None and mf not in seen:
                todo.append(mf)
    for f in sorted(seen):
        write(OUT / "code" / f.relative_to(REPO), f.read_text(encoding="utf-8"))
    return len(seen)


README = """# Supplementary material — Text, Screenshots, or Both? Representation Value and Task Routing in Web Agents

Anonymous supplement for review. Everything the paper's numbers are computed from is here except
the raw trajectories (screenshots and page text, too large; released with the camera-ready version).

## Layout

- `data/episodes.csv` — one row per (cell, mode, run, task). `cell` = site × backbone
  (`cls` VWA-classifieds, `red` VWA-reddit, `wared` WebArena-reddit, `shop` VWA-shopping;
  B0 Qwen3-VL-235B-A22B, B1 Qwen3-VL-4B, B2 Gemma-3-4B, B5 GPT-5.6). `run` = `a` (the run every
  analysis reads) or `b` (a registered same-condition rerun). `cost_usd` = API charge (B0, B5) or
  the token-priced local-serving proxy (B1, B2); `wallclock_s` = episode wall-clock time;
  `gpu_s` = summed model-inference time (locally served backbones only).
  Rows: {episodes} ({rerun_rows} of them reruns).
- `data/features.csv` — the 18 pre-run router features (14 intent-keyword indicators, intent
  length, three first-page statistics) per (cell, task).
- `data/tasks.csv` — scored task ids per site, intent template, the intent-rule flag (Appendix D),
  and membership in the clean shopping sets (Appendix E). Task configurations themselves are the
  public VisualWebArena / WebArena files.
- `prompts/` — the system prompt of each observation mode (`phantom_*` = the screenshot-free
  controls: `phantom_som` = P-SoM, `phantom_text` = P-text, `phantom_prompt` = P-prompt;
  `phantom_dom` is a legacy alias of `phantom_text`). For GPT-5.6, whose API returns structured
  output, the JSON-only line of every prompt is extended with "Emit exactly ONE JSON object for the
  single next action — do not plan ahead or emit further objects."
- `configs/` — run configurations (comments stripped, API endpoints redacted).
- `products/` — every analysis product the paper cites; each `.md` names its producer and protocol.
- `code/` — the producer scripts and their in-repository imports. They read the raw run
  directories, which are not included; `data/` holds the per-episode values they extract.

## Where each number comes from

| paper | product |
|---|---|
| Fig. 1, Table 3 | `routing_three_arm` (`cells[].frontier.fixed_modes`), `routing_extension_cells` |
| §3 contrasts | `fusion_premium`; leakage-adjusted value: `leakage_sensitivity` |
| Fig. 2, §4 | `visual_intent_routing` (extension section) |
| §5, Fig. 3 | `routing_three_arm` (`null_cells`); six modes: `task_mode_interaction_null` |
| §5 persistence | `routing_crossrun_template_validation` |
| Fig. 4, Table 1 | `routing_three_arm`, `representation_routing_frontier`, `routing_gain_upper_bounds`, `routing_extension_cells` |
| Table 2 (sensitivity) | `routing_three_arm`, `_wallclock`, `_gpu_time` |
| rerun range 0–14% | `noise_floor_inventory` |
| reddit inherited state | `reddit_sidebar_leakage_audit_with_wa` |

Figures and tables are regenerated from the products by `code/scripts/analysis/figures/coling/make_figures.py`.

## Recomputing from `data/`

The frontier of a cell is the upper concave envelope of the fixed modes' (mean cost, success
rate) points (Appendix I). Mean cost and success per mode come directly from `episodes.csv`
(`run == "a"`); the router curves need `features.csv` and the fold split described in Appendix I
(five folds over distinct task ids, seed 42 permutation).
"""


def scan() -> list[str]:
    hits = []
    for f in OUT.rglob("*"):
        if not f.is_file():
            continue
        low = f.read_text(encoding="utf-8", errors="replace").lower()
        for tok in BANNED:
            if tok in low:
                i = low.index(tok)
                hits.append(f"{f.relative_to(OUT)}: {tok!r} … {low[max(0, i - 40):i + 40]!r}")
    return hits


def main() -> int:
    if OUT.exists():
        if OUT.resolve().parent != (REPO / "deliverables/coling2027").resolve():
            raise SystemExit(f"refusing to delete {OUT}")
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    stats = export_data()
    n_prompts, n_cfg, n_prod, n_code = export_prompts(), export_configs(), export_products(), export_code()
    write(OUT / "README.md", README.format(**stats))
    hits = scan()
    if hits:
        print("ANONYMITY SCAN FAILED:", *hits[:40], sep="\n  ", file=sys.stderr)
        return 1
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, Path("supplement") / f.relative_to(OUT))
    print(f"episodes {stats['episodes']} (reruns {stats['rerun_rows']}), tasks {stats['tasks']}, cells {stats['cells']}; "
          f"prompts {n_prompts}, configs {n_cfg}, products {n_prod}, code {n_code}; zip {ZIP.stat().st_size / 1e6:.1f} MB",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
