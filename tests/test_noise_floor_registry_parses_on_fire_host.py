"""The replicate registry must stay readable by the fire host's Python, not just the dev host's.

`validate_fire_manifest.registered_replicate_run_ids()` reads CLEAN_PAIRS out of
`aggregate_noise_floor_inventory.py` with `ast.parse` + `ast.literal_eval`, and catches
SyntaxError by returning an EMPTY registry — fail-closed, so every deliberate
same-condition replicate is then reported as a ghost and aggregation halts.

That failure mode has no local symptom: the dev host (3.12) parses the file fine, the run
directories are all intact, and the only signal is a ghost report naming runs whose own
entries were never touched. Empirical 2026-08-17: rsyncing this file from the 3.12 dev
mirror to the 3.10 fire host turned `B0.cls.som` from "registered replicate" into a GHOST,
because two unrelated lines used PEP 701 same-quote f-string nesting (3.12+ only).

So the contract under test is not "the file is valid Python" — it is "the file is valid
Python *on the version the fire host runs*", plus "the registry the parser recovers is
non-empty and well-shaped".
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "scripts/analysis/aggregate_noise_floor_inventory.py"

# The A100 fire host (`a100-jiaming-test`) runs CPython 3.10.12; the DGX dev mirror runs
# 3.12.3, and only 3.12 exists here — so this cannot be checked by actually compiling.
FIRE_HOST_PYTHON = "3.10"

# `ast.parse(..., feature_version=(3, 10))` does NOT help: verified 2026-08-17, it happily
# parses `f"{d["k"]}"` because PEP 701 changed the tokenizer, which feature_version does not
# gate. So detect it in the source text instead.
#
# Matches an f-string whose interpolation reuses the OUTER quote before the string closes:
#   f" ... { ... "        ← the quote inside {...} terminates the literal on 3.10
# Deliberately does not match the legal forms:
#   f"{d['k']}"  (other quote)   f"{x:.2f}"  (no quote at all)
_SAME_QUOTE_NESTING = re.compile(r"""(?<![\w])[rRbB]?[fF][rR]?(?P<q>["'])[^"'{]*\{[^}"']*(?P=q)""")


def _source() -> str:
    return INVENTORY.read_text(encoding="utf-8")


def test_inventory_has_no_pep701_only_fstrings():
    """Heuristic, textual — see comment above for why compiling cannot be used here."""
    offenders = [
        (i, line.strip())
        for i, line in enumerate(_source().splitlines(), start=1)
        if _SAME_QUOTE_NESTING.search(line)
    ]
    assert not offenders, (
        f"{INVENTORY.name} uses same-quote f-string nesting (PEP 701), which parses on the "
        f"3.12 dev host but is a SyntaxError on the Python {FIRE_HOST_PYTHON} fire host:\n"
        + "\n".join(f"  line {i}: {t}" for i, t in offenders)
        + "\n\nWhy this matters more than a normal syntax slip: "
        "validate_fire_manifest.registered_replicate_run_ids() catches SyntaxError and "
        "returns an EMPTY registry (fail-closed), so EVERY registered replicate is then "
        "reported as a ghost and aggregation halts — while the offending line is nowhere "
        "in that report. Fix: use the other quote character inside {...}."
    )


def test_clean_pairs_survives_literal_eval_and_is_non_empty():
    """Mirror the exact recovery path validate_fire_manifest uses, not an import."""
    tree = ast.parse(_source())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "CLEAN_PAIRS" for t in node.targets):
            continue
        pairs = ast.literal_eval(node.value)  # ValueError here = registry unreadable
        assert pairs, "CLEAN_PAIRS is empty — every replicate would be judged a ghost"
        for entry in pairs:
            assert len(entry) == 3, f"entry must be (label, arm_a, arm_b): {entry!r}"
            label, arm_a, arm_b = entry
            assert isinstance(label, str) and label
            assert isinstance(arm_a, str) and arm_a
            assert isinstance(arm_b, str) and arm_b
            assert arm_a != arm_b, f"pair {label} names the same run twice"
        return
    pytest.fail("CLEAN_PAIRS assignment not found — the registry parser would recover nothing")
