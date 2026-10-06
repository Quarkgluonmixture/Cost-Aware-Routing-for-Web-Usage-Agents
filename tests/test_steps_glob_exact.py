"""Every reader of step trajectories must glob the exact `*steps_v2.jsonl` suffix.

Canonical run dirs carry quarantined earlier attempts as `<stem>_steps_v2.stale_<ts>.jsonl`
(13 of them in 2026-10). A loose `*_steps*.jsonl` glob matches both, and `[0]` then reads
whichever the filesystem lists first: that is how `aggregate_confidence_cascade` read the
stale file for red_B1·vision task 33 on the DGX and reported 4.43% where the data give 3.94%
(实验笔记 §531.7). The only pattern allowed to see stale files is one that names them.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOTS = [Path("scripts"), Path("p79")]
GLOB = re.compile(r"""glob\(\s*f?["']([^"']*steps[^"']*\.jsonl)["']""")


def test_step_globs_cannot_match_stale_files():
    loose = []
    for root in ROOTS:
        for py in root.rglob("*.py"):
            for m in GLOB.finditer(py.read_text(encoding="utf-8", errors="replace")):
                pat = m.group(1)
                if "stale" in pat:
                    continue
                if not pat.endswith("steps_v2.jsonl"):
                    loose.append(f"{py}: {pat}")
    assert not loose, "step-file globs that also match `.stale_*` quarantine files:\n" + "\n".join(loose)
