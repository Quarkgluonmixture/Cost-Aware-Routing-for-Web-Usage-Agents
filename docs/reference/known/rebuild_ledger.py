#!/usr/bin/env python3
"""Merge ledger chunks + repair artifact_exists false-negatives.

Why the repair pass exists: the chronicle records artifact paths as they were
when written. Several have since MOVED (e.g. docs/analysis/phantom_paper/* ->
results/phantom_paper/*). A subagent checking the literal recorded path gets
exists=false for a file that is present under a different directory.

That failure mode is worse than a missing entry: a ledger that says "this
measurement's artifact is gone" reads downstream as "not reproducible", which
invites exactly the redo the ledger exists to prevent.

So: for every exists=false record, search the repo by basename. If found,
flip to true and record where it actually is, keeping the original path.
"""
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents")
SCRATCH = Path(__file__).resolve().parent
OUT = SCRATCH / "ledger_merged.jsonl"

PATH_FIELDS = ("source_artifact", "path", "recorded_where")


def build_index() -> dict:
    """basename -> [repo-relative paths]. Tracked + untracked, excludes .git."""
    idx = defaultdict(list)
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs
                   if d not in (".git", "node_modules", "__pycache__", ".venv")]
        rel_root = Path(root).relative_to(REPO)
        for f in files:
            idx[f].append(str(rel_root / f))
    return idx


def extract_path(val) -> str | None:
    """Pull a filesystem-looking path out of a field value."""
    if not isinstance(val, str) or not val:
        return None
    tok = val.split()[0].split(":")[0].strip("`'\",;")
    if "/" not in tok and "." not in tok:
        return None
    if tok.startswith(("§", "http", "B-")):
        return None
    return tok


def main() -> int:
    chunks = sorted(SCRATCH.glob("ledger_chunk*.jsonl"))
    if not chunks:
        print("no chunks found", file=sys.stderr)
        return 1
    print(f"merging {len(chunks)} chunks: {[c.name for c in chunks]}")

    idx = build_index()
    print(f"repo file index: {len(idx)} distinct basenames")

    records, per_chunk = [], Counter()
    for c in chunks:
        for ln, line in enumerate(c.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  !! {c.name}:{ln} bad JSON: {e}", file=sys.stderr)
                continue
            r["_chunk"] = c.stem.replace("ledger_chunk", "")
            records.append(r)
            per_chunk[c.stem] += 1

    repaired = notfound = 0
    for r in records:
        if r.get("artifact_exists") is not False:
            continue
        cand = None
        for fld in PATH_FIELDS:
            p = extract_path(r.get(fld))
            if p:
                cand = p
                break
        if not cand:
            continue
        # literal path may already be right (agent could have checked wrong cwd)
        if (REPO / cand).exists():
            r["artifact_exists"] = True
            r["_repair"] = f"literal path exists: {cand}"
            repaired += 1
            continue
        hits = idx.get(Path(cand).name, [])
        if hits:
            r["artifact_exists"] = True
            r["_repair"] = (f"MOVED: recorded {cand} -> found at "
                            f"{hits[0]}" + (f" (+{len(hits)-1} more)" if len(hits) > 1 else ""))
            repaired += 1
        else:
            r["_repair"] = f"CONFIRMED ABSENT: {cand}"
            notfound += 1

    # ---- cross-chunk retraction linking (ONLY the orchestrator can do this) --
    # Each agent saw one § range and was told to leave superseded_by null unless
    # the retraction was inside its own range. So a claim retracted in a LATER
    # chunk still reads as live in the earlier one. Unlinked, the ledger would
    # serve retracted findings as current — the exact failure it exists to stop.
    def sect_key(s):
        """'§302.1' / '302.1' -> (302, 1) for ordering."""
        if not isinstance(s, str):
            return None
        t = s.strip().lstrip("§").split()[0].rstrip(".,;:")
        parts = t.split(".")
        try:
            return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
        except (ValueError, IndexError):
            return None

    # Token overlap was tried first and produced ~50% false positives: the text is
    # Chinese, so whitespace splitting yields giant tokens whose substring matches
    # are meaningless. Use only a hard signal — a § number the retraction ITSELF
    # names. A ledger that cries retraction on live findings is worse than one that
    # stays quiet, because the reader stops trusting every flag.
    import re as _re
    SECT_RE = _re.compile(r"§\s*(\d+)(?:\.(\d+))?")

    retractions = [r for r in records if r.get("type") == "RETRACTED"]
    by_section = defaultdict(list)
    for r in records:
        k = sect_key(r.get("source_section"))
        if k:
            by_section[k].append(r)

    linked = 0
    for ret in retractions:
        theirs = sect_key(ret.get("source_section"))
        if not theirs:
            continue
        cited = " ".join(str(ret.get(f, "")) for f in
                         ("former_claim", "why_dead", "replaced_by", "source_section"))
        targets = set()
        for m in SECT_RE.finditer(cited):
            k = (int(m.group(1)), int(m.group(2) or 0))
            if k < theirs:  # a retraction can only kill something EARLIER
                targets.add(k)
                if m.group(2) is None:  # bare "§302" also covers §302.x
                    targets.update(kk for kk in by_section if kk[0] == k[0] and kk < theirs)
        for k in targets:
            for victim in by_section.get(k, []):
                if victim.get("type") == "RETRACTED" or victim.get("superseded_by"):
                    continue
                victim.setdefault("_cross_chunk_flags", []).append(
                    f"named by RETRACTED {ret.get('source_section')}: "
                    f"{str(ret.get('former_claim', ''))[:110]}")
                linked += 1

    with OUT.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {OUT}  ({len(records)} records)")
    print("\nper chunk:", dict(per_chunk))
    print("\nby type:", dict(Counter(r.get("type") for r in records)))
    print(f"\nartifact repair: {repaired} false-negatives fixed, "
          f"{notfound} confirmed absent")
    print(f"cross-chunk retraction candidates flagged: {linked} "
          f"(from {len(retractions)} RETRACTED records) — REVIEW, not authoritative")
    return 0


if __name__ == "__main__":
    sys.exit(main())
