#!/usr/bin/env python3
"""Assemble the zero-preset evidence handoff package for other AIs (2026-10-08).

The authored entry point is docs/handoff/EVIDENCE_HANDOFF_20261008.md; everything else in the
package is copied or derived from the repository's sources of truth, so this script is the only
thing to rerun when they change. Output (gitignored): deliverables/handoff_build/
evidence_handoff_20261008/ and a .zip beside it.

Deliberately NOT included (zero preset): paper drafts, frame proposals, workshop papers, cross-AI
frame reviews. Reviews received from workshop referees ARE included (external feedback).

Secrets: private API endpoints and notification topics are redacted in every text file, then the
tree is scanned for credential patterns and the build fails on any hit.

Usage:
  python scripts/release/build_evidence_handoff.py
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BUILD = REPO / "deliverables/handoff_build"
OUT = BUILD / "evidence_handoff_20261008"
ZIP = BUILD / "evidence_handoff_20261008.zip"
MAIN = REPO / "docs/handoff/EVIDENCE_HANDOFF_20261008.md"

# Research-direction audits whose text delivers a framing verdict; withheld under zero preset.
WITHHELD_PRODUCTS = {"cu_pareto_operating_envelope_audit_2026-09-22.md",
                     "trajectory_adaptive_compute_audit_2026-09-22.md"}
TEXT_EXT = {".md", ".json", ".jsonl", ".csv", ".yaml", ".yml", ".txt", ".py", ".sh", ".log", ".tsv"}
REDACT = [
    (re.compile(r"https?://[a-z0-9]+\.execute-api\.[a-z0-9-]+\.amazonaws\.com\S*", re.I), "<redacted-api-endpoint>"),
    (re.compile(r"[a-z0-9]{10}\.execute-api\.[a-z0-9-]+\.amazonaws\.com", re.I), "<redacted-api-endpoint>"),
    (re.compile(r"ntfy\.sh/[A-Za-z0-9_-]+"), "ntfy.sh/<redacted-topic>"),
]
SECRET = re.compile(r"(?<![A-Za-z0-9-])sk-[A-Za-z0-9]{32,}|hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|"
                    r"github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|xox[bp]-[A-Za-z0-9-]{20,}|"
                    r"execute-api\.[a-z0-9-]+\.amazonaws|ntfy\.sh/(?!<redacted)[A-Za-z0-9_-]+")


def put(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in TEXT_EXT:
        t = src.read_text(encoding="utf-8", errors="replace")
        for pat, rep in REDACT:
            t = pat.sub(rep, t)
        dst.write_text(t, encoding="utf-8", newline="\n")
    else:
        shutil.copy2(src, dst)


def put_text(dst: Path, text: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    for pat, rep in REDACT:
        text = pat.sub(rep, text)
    dst.write_text(text, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------- sections

def products() -> int:
    root = REPO / "docs/analysis"
    n = 0
    idx = ["# Product index", "",
           "Every analysis report under the repository's `docs/analysis/` (copied into this folder with the "
           "same relative paths), with its title and opening lines. Cross-site result files (`.json`) sit "
           "beside their reports. Each report's own header states its producer script, estimator and caveats.", ""]
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix not in (".md", ".json"):
            continue
        if f.name in WITHHELD_PRODUCTS:
            continue
        put(f, OUT / "01_products" / f.relative_to(root))
        n += 1
        if f.suffix != ".md":
            continue
        t = f.read_text(encoding="utf-8", errors="replace")
        if t.startswith("---"):
            j = t.find("\n---", 3)
            t = t[j + 4:] if j > 0 else t
        lines = [ln for ln in t.splitlines() if ln.strip()]
        title = next((ln for ln in lines if ln.startswith("#")), lines[0] if lines else "")
        body, size = [], 0
        for ln in lines:
            if ln == title:
                continue
            body.append(ln[:300])
            size += len(ln)
            if size > 700:
                break
        idx += [f"## `{f.relative_to(root).as_posix()}` ({len(t) // 1000} KB)", "",
                title.lstrip("# ").strip(), "", *body, ""]
    put_text(OUT / "01_products" / "products_index.md", "\n".join(idx) + "\n")
    return n


def data() -> dict:
    from scripts.release import build_coling_supplement as sup
    sup.OUT = OUT / "02_data"
    sup.OUT.mkdir(parents=True, exist_ok=True)
    stats = sup.export_data()
    sup.export_prompts()
    sup.export_configs()
    readme = f"""# Data files

- `episodes.csv` — one row per (cell, mode, run, task). {stats['episodes']} rows, {stats['rerun_rows']} of them reruns.
  - `cell`: `cls` VWA-classifieds, `red` VWA-reddit, `wared` WebArena-reddit, `shop` VWA-shopping; suffix = backbone
    (B0 Qwen3-VL-235B-A22B API, B1 Qwen3-VL-4B local, B2 Gemma-3-4B local, B5 GPT-5.6 API).
  - `mode`: DOM, SoM, Vision, P-text, P-prompt, P-SoM (see 00_START_HERE.md §2.3).
  - `run`: `a` = the run every analysis reads; `b` = a registered same-condition rerun (24 pairs).
  - `success`: binary evaluator outcome. `cost_usd`: API charge (B0, B5) or token-priced local-serving proxy
    (B1, B2) — comparable within a cell only. `wallclock_s`: canonical episode wall-clock.
    `gpu_s`: summed model-inference seconds (B1/B2 run-a rows of the 8 core cells only).
  - Shopping and B5 rows cover every scored task; whether a shopping task is in the "clean" set is in `tasks.csv`.
  - Inherited-state successes on reddit (00_START_HERE.md §8.3) are NOT zeroed here; they are listed in
    `01_products/cross_sites/leakage_sensitivity.md` and `persistent_state_leakage_audit.md`.
- `features.csv` — the 18 pre-run router features per (cell, task).
- `tasks.csv` — scored task ids per site, intent template id, the intent-rule flag, shopping clean-set flags.
- `prompts/` — per-mode system prompts exactly as the agents build them.
- `configs/` — run configurations (YAML, comments stripped, API endpoints redacted).

Check: recomputing per-mode success rate and mean cost for every cell from `episodes.csv` (run a) reproduces
`01_products/cross_sites/routing_three_arm.json` and `routing_extension_cells.json` exactly.
"""
    put_text(OUT / "02_data" / "DATA_README.md", readme)
    return stats


def conclusions() -> int:
    src = REPO / "docs/reference/known/conclusions"
    k = 0
    for f in sorted(src.glob("*.md")):
        put(f, OUT / "03_conclusion_layer" / f.name)
        k += 1
    put(REPO / "docs/reference/known/ledger.jsonl", OUT / "03_conclusion_layer" / "ledger.jsonl")
    return k + 1


def inventory() -> int:
    src = REPO / "docs/analysis/run_inventory"
    names = ["README.md", "run_matrix.md", "run_inventory.json", "product_coverage.md", "product_coverage.json",
             "product_scope.yaml"]
    for n in names:
        put(src / n, OUT / "04_run_inventory" / n)
    return len(names)


def log() -> int:
    nb = REPO / "docs/checkpoints/实验笔记.md"
    put(nb, OUT / "05_experiment_log" / "experiment_notebook.md")
    heads = [ln for ln in nb.read_text(encoding="utf-8").splitlines()
             if ln.startswith("## ") and re.match(r"## (§?\d)", ln)]
    idx = ["# Chronology index", "",
           f"All {len(heads)} numbered section titles of the lab notebook (`experiment_notebook.md`, Chinese), "
           "in order. Tags in brackets: [finding] measured result · [bug] defect · [infra] infrastructure · "
           "[design] decision. Search the notebook for the section number to read it.", ""]
    idx += [f"- {h[3:]}" for h in heads]
    put_text(OUT / "05_experiment_log" / "chronology_index.md", "\n".join(idx) + "\n")
    return len(heads)


def prereg() -> int:
    k = 0
    put(REPO / "docs/checkpoints/pre_run/preregistration.md", OUT / "06_preregistration" / "preregistration.md")
    for f in sorted((REPO / "docs/prereg_amendments").iterdir()):
        if f.is_file():
            put(f, OUT / "06_preregistration" / "amendments" / f.name)
            k += 1
    for f in sorted((REPO / "docs/checkpoints/pre_run").glob("*")):
        if f.is_file() and ("intent" in f.name or f.name.startswith("budget_router_prospective")):
            put(f, OUT / "06_preregistration" / "run_intents_and_prospective_tests" / f.name)
            k += 1
    return k + 1


def router_pilot() -> int:
    src = REPO / "results/evidence_snapshots/20261006_router_pilot_20260909"
    k = 0
    for f in sorted(src.rglob("*")):
        if f.is_file() and f.name != "NEXT_SESSION_PROMPT.md":   # a session prompt, not evidence
            put(f, OUT / "07_router_pilot" / f.relative_to(src))
            k += 1
    return k


def feedback() -> int:
    src = REPO / "docs/checkpoints/_status/issues"
    names = ["issue_realm_reviews_2026-09-09.md", "issue_vlm4rwd_reviews_2026-09-29.md"]
    note = ("# External feedback\n\nReferee reviews received by two earlier workshop submissions built on subsets of "
            "this data (REALM @ EMNLP 2026, poster; VLM4RWD @ NeurIPS 2026, accepted). The submissions themselves "
            "are withheld (zero preset); the reviews are included as external evidence of what readers asked for. "
            "The files also contain the project's own triage notes on each point.\n")
    put_text(OUT / "08_external_feedback" / "README.md", note)
    for n in names:
        put(src / n, OUT / "08_external_feedback" / n)
    return len(names)


def defects() -> int:
    cat = REPO / "docs/reference/master_bug_catalog.md"
    put(cat, OUT / "09_harness_defects" / "master_bug_catalog.md")
    heads = [ln[4:] for ln in cat.read_text(encoding="utf-8").splitlines() if ln.startswith("### B-")]
    put_text(OUT / "09_harness_defects" / "bug_index.md",
             f"# Bug index\n\n{len(heads)} numbered entries of `master_bug_catalog.md` (title, severity, status).\n\n"
             + "\n".join(f"- {h}" for h in heads) + "\n")
    return len(heads)


def scan() -> list[str]:
    hits = []
    for f in OUT.rglob("*"):
        if f.is_file() and f.suffix.lower() in TEXT_EXT:
            t = f.read_text(encoding="utf-8", errors="replace")
            for m in SECRET.finditer(t):
                hits.append(f"{f.relative_to(OUT)}: {t[max(0, m.start() - 30):m.end() + 10]!r}")
    return hits


def main() -> int:
    if OUT.exists():
        if OUT.resolve().parent != BUILD.resolve():
            raise SystemExit(f"refusing to delete {OUT}")
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    put(MAIN, OUT / "00_START_HERE.md")
    counts = {"products": products(), "data": data(), "conclusion files": conclusions(),
              "inventory files": inventory(), "log sections": log(), "prereg files": prereg(),
              "router pilot files": router_pilot(), "feedback files": feedback(), "bug entries": defects()}
    hits = scan()
    if hits:
        print("SECRET SCAN FAILED:", *hits[:30], sep="\n  ", file=sys.stderr)
        return 1
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, Path(OUT.name) / f.relative_to(OUT))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"{counts}\nunpacked {size / 1e6:.1f} MB · zip {ZIP.stat().st_size / 1e6:.1f} MB → {ZIP}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
