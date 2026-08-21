#!/usr/bin/env python3
"""Export a de-identified, submission-ready snapshot of this repo.

Rationale
---------
The live repo cannot be anonymised in place: 1700+ commits carry a single
authorship, the repo name is the paper title, and public forks exist. The only
sound route is a *fresh* repo built from a curated file set with a single
anonymous commit and no inherited history.

This script is the reproducible half of that. It is deliberately re-runnable:
run it now for a dry-run, run it again after the final draft settles, and the
diff between the two outputs is the only thing you need to re-review.

Usage
-----
    python3 scripts/maintenance/export_anonymous_repo.py --out DIR [--init-git]
    python3 scripts/maintenance/export_anonymous_repo.py --report-only
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# 1. File selection.  Order matters: EXCLUDE wins over INCLUDE.
# --------------------------------------------------------------------------
# Everything a reviewer needs to (a) read the harness, (b) re-run the analysis,
# (c) audit the preregistration chain.  Nothing about who ran it or where.
INCLUDE_PREFIXES = (
    "p79/",
    "tests/",
    "configs/",
    "scripts/analysis/",
    "scripts/queues/",
    "scripts/mechanistic/",
    "scripts/vwa/",
    "scripts/provenance/",
    "scripts/setup/",
    "docs/checkpoints/pre_run/",
    "docs/prereg_amendments/",
    "results/mechanistic/",
    "results/provenance/",
)
INCLUDE_EXACT = {
    "pyproject.toml",
    "requirements-lock.txt",
    "Makefile",
    ".gitignore",
    "scripts/README.md",
    "scripts/run_experiment.py",
    "scripts/preflight_v2.sh",
    "scripts/vwa_env_remote.sh.example",
    # methodology reference worth shipping; machine-specific ones are excluded below
    "docs/reference/PHANTOM_SOM_CODE_TOUR.md",
    "docs/reference/condition_map.md",
    "docs/reference/analysis_templates.md",
    "docs/reference/master_bug_catalog.md",
    "docs/reference/master_bug_catalog_index.md",
    "docs/reference/paper_process_pitfalls.md",
    "docs/reference/GOTCHAS.md",
    "docs/reference/EVIDENCE_LAYER_AUDIT.md",
    "docs/reference/launch_checklist.md",
    "docs/reference/vwa_assets_manifest.json",
}

# Identity, infrastructure, personal chronicle, and unpublished prose.
EXCLUDE_PREFIXES = (
    "portfolio/",
    "deliverables/",
    "final_dissertation/",
    "assets/",
    "external/",
    "tools/",
    ".claude/",
    ".github/",
    ".githooks/",
    "docs/archive/",
    "docs/literature/",           # third-party DR outputs, not ours to redistribute
    "docs/analysis/",             # narrative digests; re-add selectively if cited
    "docs/checkpoints/_status/",
    "docs/checkpoints/canvas/",
    "docs/checkpoints/codex_prompts/",
    "docs/checkpoints/codex_outputs/",
    "docs/checkpoints/paper_drafts/",   # prose lives in the submission, not the repo
    "docs/checkpoints/process/",
    "docs/reference/condenser/",
    "docs/reference/deployment_baseline/",
    "docs/reference/known/",
    "scripts/maintenance/",       # cron sidecars / ntfy / host-specific ops
    "scripts/spike/",
    "scripts/myriad/",
    "configs/_deprecated/",
)
EXCLUDE_EXACT = {
    ".gitmodules",                # rewritten, see below
    ".nojekyll",
    "board-top.png",
    "GEMINI.md",
    "README.md",                  # rewritten, see below
    ".analysis_k6.done",
}
# Substrings that disqualify a path regardless of prefix rules.
EXCLUDE_SUBSTRINGS = (
    ".bak",
    # A hash-manifested deposit of a *named* OSF DOI.  Scrubbing it invalidates
    # its own MANIFEST_SHA256 (destroying the evidentiary value that is its
    # entire point); not scrubbing it points at the author.  It is therefore
    # non-anonymisable by construction and is withheld -- note that it also
    # carries a nested paper_drafts/ copy that slipped past the top-level
    # paper_drafts exclusion.
    "osf_deposit_",
    "实验笔记",
    "next_steps",
    "paper_planning",
    "ADVISOR",
    "advisor",
    "onepager",
)

# --------------------------------------------------------------------------
# 2. Scrub patterns.  (regex, replacement) applied to every text file kept.
# --------------------------------------------------------------------------
SCRUB = [
    # personal identity
    (r"(?i)\bjiaming\s+wei\b", "ANON AUTHOR"),
    (r"(?i)\bwei\s+jiaming\b", "ANON AUTHOR"),
    (r"(?i)\bquarkgluonmixture\b", "anon-author"),
    (r"(?i)\bucab352\b", "anonuser"),
    (r"(?i)\bjimmyenglish@126\.com\b", "anon@example.com"),
    (r"(?i)\bwubbalabbadubdub1@gmail\.com\b", "anon@example.com"),
    (r"/home/jiaming\b", "/home/anon"),
    (r"/clusterhome/jiaming\b", "/clusterhome/anon"),
    (r"(?i)\bjiaming\b", "anon"),
    # institution
    (r"\bmyriad\.rc\.ucl\.ac\.uk\b", "hpc.example.edu"),
    (r"(?i)\bUCL\b", "[INSTITUTION]"),
    (r"(?i)\bUniversity College London\b", "[INSTITUTION]"),
    # hosts / fleet identifiers
    # NB: no leading \b — these appear inside filenames like
    # "det_check_spark-9ea3.pt", where the preceding "_" is a word char and
    # \b never fires.  Use an explicit non-alphanumeric lookbehind instead.
    (r"(?<![A-Za-z0-9])spark-9ea3\b", "workstation-a"),
    (r"(?<![A-Za-z0-9])spark-9017\b", "cluster-node-1"),
    (r"(?<![A-Za-z0-9])spark-97a6\b", "cluster-node-2"),
    (r"(?<![A-Za-z0-9])a100-jiaming-test\b", "gpu-vm-a100"),
    (r"(?i)\bmyriad\b", "hpc-cluster"),
    (r"(?i)\bquark\b(?!gluon)", "home-host"),
    (r"(?i)\bcondenser\b", "cloud-vm"),
    (r"(?i)\bDGX[ -]?Spark\b", "workstation"),
    (r"(?i)\bHolistic[\s-]*AI\b", "[LAB]"),
    # network
    (r"\b100\.95\.81\.103\b", "10.0.0.1"),
    (r"\b10\.134\.51\.2\b", "10.0.0.2"),
    (r"\b[a-z0-9]{10}\.execute-api\.[a-z0-9-]+\.amazonaws\.com\b",
     "PROXY_ENDPOINT.example.com"),
    # the id also appears elided, as "i5xpracyci...execute-api"
    (r"\bi5xpracyci\b", "PROXYID"),
    # notification channel
    (r"ntfy\.sh/[A-Za-z0-9_-]+", "ntfy.sh/NTFY_TOPIC"),
    # topic names also appear as shell defaults, e.g. "${NTFY_TOPIC:-p79-...}",
    # which the URL pattern above never sees
    (r"\bp79-claude\b", "NTFY_TOPIC"),
    (r"\bp79-exp-[A-Za-z0-9_-]+", "NTFY_TOPIC"),
]

# Extension whitelists silently miss extensionless files.  The first export
# copied `Makefile` verbatim -- unscrubbed -- for exactly this reason.  Sniff
# the bytes instead: a NUL in the first 8 KiB means binary, everything else is
# treated as text.
# --------------------------------------------------------------------------
# 2b. Independent verification probes.
# --------------------------------------------------------------------------
# Written from scratch rather than derived from SCRUB.  A scan that reuses the
# scrubber's own patterns is circular -- it can only rediscover what the
# scrubber already knew about, and will report "clean" for every leak class
# nobody thought of.  These probes exist to disagree with SCRUB.
PROBES = [
    r"(?i)jiaming",
    r"(?i)ucab\d*",
    r"(?i)quarkgluon",
    r"126\.com",
    r"ucl\.ac\.uk",
    r"\bUCL\b",
    r"(?i)university college",
    r"(?<![A-Za-z0-9])spark-9",
    r"\b100\.95\.81\.\d+",
    r"\b10\.134\.51\.\d+",
    r"(?i)i5xp",
    r"(?i)holistic",
    r"(?i)\bmyriad\b",
    r"(?i)\bquark\b",
    r"(?i)\bcondenser\b",
    r"ntfy\.sh/(?!NTFY_TOPIC|\$|\{)",
    r"\bp79-(claude|exp)-?",
]

# Tokens that look identifying but must survive: they are upstream benchmark
# fixtures, and rewriting them would silently break reproduction.
INTENTIONALLY_KEPT = {
    "MarvelsGrantMan136": "VisualWebArena reddit fixture account",
    "blake.sullivan": "VisualWebArena classifieds fixture account",
    "emma.lopez": "VisualWebArena/WebArena shopping fixture account",
    "execute-api": "generic substring in proxy error classifiers (p79/experiment/metrics.py)",
}


def is_text(path: Path) -> bool:
    try:
        chunk = path.open("rb").read(8192)
    except OSError:
        return False
    return b"\0" not in chunk

ANON_README = """# Cost-Aware Routing for Web-Usage Agents

Anonymous code and analysis release accompanying the submission.
Author, institution, host names and network endpoints have been removed;
paths such as `/home/anon` and `PROXY_ENDPOINT.example.com` are placeholders.

## What is here

| Path | Contents |
|---|---|
| `p79/` | Experiment engine: agents, backends, runner, routing, metrics, schema v2 |
| `scripts/analysis/` | Aggregation, statistics, and figure generation |
| `scripts/queues/` | Per-condition launch wrappers (the only supported entry point) |
| `configs/` | Per-condition YAML; merged over `DEFAULT_CONFIG` |
| `docs/checkpoints/pre_run/` | Preregistration, model/dataset cards, locked versions |
| `docs/prereg_amendments/` | Timestamped amendments to the preregistration |
| `results/mechanistic/`, `results/provenance/` | Released artifact subsets |
| `tests/` | pytest suite including runner invariants |

## Environment

Benchmarks are [VisualWebArena](https://github.com/web-arena-x/visualwebarena)
(classifieds / reddit / shopping) and WebArena (reddit / shopping), served from
self-hosted Docker containers. The harness expects a patched VisualWebArena
checkout; see `PATCHES.md` for the patch set applied over upstream.

```bash
pip install -e ".[analysis,dev]"
bash scripts/preflight_v2.sh
```

## Running

Never invoke `scripts/run_experiment.py` directly — the queue wrappers own site
reset, credential handling, and idempotent skip:

```bash
RESET_BEFORE=1 bash scripts/queues/queue_baseline.sh B0 dom classifieds
python scripts/analysis/analyze_experiment.py --run_dir <run_dir>
```

## Withheld for anonymity

The preregistration is included (`docs/checkpoints/pre_run/preregistration.md`)
along with its amendment chain. The timestamped third-party deposit of that
preregistration is **not** included: it is a hash-manifested archive of a
named public DOI, so it cannot be de-identified without invalidating the very
manifest that gives it evidentiary value. The DOI will be cited in the
camera-ready version.

## Reproducibility notes

- Analysis reads JSONL through `p79.experiment.io_utils.read_jsonl_dedup`,
  which handles restart de-duplication and truncated lines.
- The primary statistical gate is a one-sided fixed-effect inverse-variance
  pooled superiority test; K-of-N counts are reported for transparency only.
  See `docs/checkpoints/pre_run/preregistration.md`.
- Ablation tables are generated, not hand-edited
  (`scripts/analysis/export_ablation_tables.py`).
"""

ANON_PATCHES = """# VisualWebArena patch set

The live repository consumes VisualWebArena as a submodule pointing at a
patched fork. To keep this release anonymous the submodule is not included.

Reproduce it with:

```bash
git clone https://github.com/web-arena-x/visualwebarena external/visualwebarena
cd external/visualwebarena && git checkout <BASE_SHA>
git apply ../../patches/visualwebarena-p79.patch
```

`BASE_SHA` and the patch bundle are listed in
`docs/checkpoints/pre_run/locked_versions.md`.
"""


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "-c", "core.quotePath=false", "ls-files", "-z"],
        cwd=REPO, capture_output=True, check=True,
    ).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def keep(path: str) -> bool:
    if any(s in path for s in EXCLUDE_SUBSTRINGS):
        return False
    if path in EXCLUDE_EXACT:
        return False
    if any(path.startswith(p) for p in EXCLUDE_PREFIXES):
        return False
    if path in INCLUDE_EXACT:
        return True
    return any(path.startswith(p) for p in INCLUDE_PREFIXES)


def scrub_text(text: str) -> tuple[str, int]:
    hits = 0
    for pattern, repl in SCRUB:
        text, n = re.subn(pattern, repl, text)
        hits += n
    return text, hits


def binary_probe(root: Path) -> list[tuple[str, str]]:
    """Probe *binary* payloads for identity.

    Text scrubbing never touches these, and the failure mode is not metadata
    fields with obvious names -- it is absolute source paths that tooling
    embeds silently (pdfTeX writes /PTEX.FileName into every included figure;
    PNG tEXt chunks can carry a capture path).  So probe for path fragments,
    not for "Author".
    """
    needles = (b"jiaming", b"Jiaming", b"ucab", b"Quarkgluon", b"quarkgluon",
               b"/home/", b"/clusterhome/", b"PTEX.FileName", b"ucl.ac.uk",
               b"spark-9", b"100.95.81")
    hits: list[tuple[str, str]] = []
    for f in sorted(root.rglob("*")):
        if not f.is_file() or ".git/" in str(f.relative_to(root)):
            continue
        if is_text(f):
            continue
        try:
            blob = f.read_bytes()
        except OSError:
            continue
        for n in needles:
            if n in blob:
                hits.append((str(f.relative_to(root)), n.decode()))
    return hits


def leak_scan(root: Path) -> list[tuple[str, str, int]]:
    """Re-grep the *output* with the independent PROBES. This, not the
    replacement count, is the check that matters."""
    residual: list[tuple[str, str, int]] = []
    for pattern in PROBES:
        rx = re.compile(pattern)
        for f in root.rglob("*"):
            if not f.is_file() or ".git/" in str(f.relative_to(root)):
                continue
            if not is_text(f):
                continue
            try:
                body = f.read_text("utf-8", errors="ignore")
            except OSError:
                continue
            n = len(rx.findall(body))
            if n:
                residual.append((pattern, str(f.relative_to(root)), n))
    return residual


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--init-git", action="store_true")
    args = ap.parse_args()

    files = tracked_files()
    kept = [f for f in files if keep(f)]
    dropped = [f for f in files if not keep(f)]

    print(f"tracked={len(files)}  kept={len(kept)}  dropped={len(dropped)}")
    by_dir: dict[str, int] = {}
    for f in kept:
        top = "/".join(f.split("/")[:2]) if "/" in f else f
        by_dir[top] = by_dir.get(top, 0) + 1
    print("\n-- kept by dir --")
    for d, n in sorted(by_dir.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {d}")

    if args.report_only:
        return 0
    if not args.out:
        print("error: --out required unless --report-only", file=sys.stderr)
        return 2

    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    total_hits = 0
    binaries = 0
    for rel in kept:
        src, dst = REPO / rel, out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if is_text(src):
            body = src.read_text("utf-8", errors="ignore")
            body, hits = scrub_text(body)
            total_hits += hits
            dst.write_text(body, encoding="utf-8")
            shutil.copymode(src, dst)
        else:
            binaries += 1
            shutil.copy2(src, dst)

    (out / "README.md").write_text(ANON_README, encoding="utf-8")
    (out / "PATCHES.md").write_text(ANON_PATCHES, encoding="utf-8")

    print(f"\nscrub replacements: {total_hits}   binaries copied verbatim: {binaries}")

    residual = leak_scan(out)
    print("\n-- residual leak scan --")
    if not residual:
        print("  clean")
    else:
        agg: dict[str, int] = {}
        for pat, path, n in residual:
            agg[pat] = agg.get(pat, 0) + n
        for pat, n in sorted(agg.items(), key=lambda kv: -kv[1]):
            sample = [p for pp, p, _ in residual if pp == pat][:3]
            print(f"  {n:5d}  {pat}")
            for s in sample:
                print(f"           e.g. {s}")

    bin_hits = binary_probe(out)
    print("\n-- binary payload probe (scrubbing cannot reach these) --")
    if not bin_hits:
        print(f"  clean across {binaries} binaries")
    else:
        for path, needle in bin_hits[:20]:
            print(f"  LEAK  {needle:16s} {path}")
        if len(bin_hits) > 20:
            print(f"  ... and {len(bin_hits) - 20} more")

    print("\n-- intentionally kept (upstream fixtures, do not scrub) --")
    for tok, why in INTENTIONALLY_KEPT.items():
        n = sum(1 for f in out.rglob("*")
                if f.is_file() and is_text(f)
                and tok in f.read_text("utf-8", errors="ignore"))
        print(f"  {n:5d} files  {tok:22s} {why}")

    manifest = out / "MANIFEST_SHA256.txt"
    lines = []
    for f in sorted(out.rglob("*")):
        if f.is_file() and f != manifest:
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            lines.append(f"{h}  {f.relative_to(out)}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nmanifest: {len(lines)} files -> {manifest}")

    if args.init_git:
        env = dict(os.environ,
                   GIT_AUTHOR_NAME="Anonymous", GIT_AUTHOR_EMAIL="anon@example.com",
                   GIT_COMMITTER_NAME="Anonymous", GIT_COMMITTER_EMAIL="anon@example.com")
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=out, check=True)
        subprocess.run(["git", "add", "-A"], cwd=out, check=True)
        subprocess.run(["git", "commit", "-q", "-m",
                        "Anonymous code and analysis release"],
                       cwd=out, env=env, check=True)
        print("git: initialised with a single anonymous commit")

    print(f"\noutput: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
