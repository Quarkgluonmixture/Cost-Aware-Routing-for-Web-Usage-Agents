"""Figure-layer shared helpers — `/stress A1.20 P0-3 / Q2=A` (2026-05-17).

Avoid hard-coded PANELS lists in individual figure scripts. Pull cell topology
from `run_registry.get_cells(...)` so additions like B2 (Gemma3-VL 2026-05-14)
propagate automatically instead of triggering sibling-script propagation gaps
across all paper-§1 figure scripts.
"""
