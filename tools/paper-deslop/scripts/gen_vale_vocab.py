#!/usr/bin/env python3
"""Generate the Vale vocabulary from terms.txt.

terms.txt is the single source of truth for the domain-term whitelist.
This script writes styles/config/vocabularies/Paper/accept.txt so Vale
treats whitelisted terms as accepted vocabulary. Rerun after editing
terms.txt and commit both files.
"""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
terms = [
    line.strip()
    for line in (root / "terms.txt").read_text().splitlines()
    if line.strip() and not line.strip().startswith("#")
]
out = root / "styles" / "config" / "vocabularies" / "Paper" / "accept.txt"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("".join(f"(?i){t}\n" for t in terms))
print(f"wrote {out.relative_to(root)} ({len(terms)} terms)")
