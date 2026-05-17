"""Shared figure-panel topology helper.

`/stress A1.20 P0-3-ABC* (2026-05-17, 3-AI overlap Claude+Codex+Gemini)`:
12/26 figure scripts had hardcoded `PANELS = [...]` or `for baseline in ("B0", "B1")` —
B2 (Gemma3-VL added 2026-05-14) silent missing across all of them. Paper §1 prose
claims "B0/B1/B2 cross-family" while ~46% of figures show only 2-baseline. Sibling-
propagation reservoir → centralize via this helper.

Each figure script should call `paper_grade_panels(sites=...)` instead of literal
PANELS lists. New baseline addition → 1 file change in `run_registry.BASELINES` +
manifest, all figures pick it up.

Companion `expected_n_canonical(site)` helper unifies `/stress A1.20 P1-1-AB`
fix — hardcoded `234/210` vs canonical `scored_task_count(...) = 224/205` post-§139.8.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from scripts.analysis.lib.run_registry import (
        BASELINES,
        SITES,
        get_cells,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from scripts.analysis.lib.run_registry import (  # noqa: F401
        BASELINES,
        SITES,
        get_cells,
    )

# /stress A1.20 P1-1-AB (2026-05-17): canonical N from p79.experiment.analysis
# replaces hardcoded `expected=234/210` in fig0c/0d/0e/0f. `scored_task_count`
# returns post-§139.8 N/A-excluded values (cls=224 / red=205).
from p79.experiment.analysis import scored_task_count


@dataclass(frozen=True)
class PanelSpec:
    """One (baseline, site) figure panel spec, with cell topology resolved.

    Replaces ad-hoc `_panel(...)` dicts inside each figure script. `modes` is
    {mode -> episodes_dir}; empty dict for cells not yet in run_manifest (e.g.,
    B2 pre-Phase-1a-fire → renders placeholder panel rather than silent skip).
    """

    key: str
    title: str
    baseline: str
    site: str
    expected_n: int
    modes: dict[str, Path]
    is_placeholder: bool = False

    @property
    def is_complete_six_mode(self) -> bool:
        """True iff cell has all 6 paper modes (DOM/SoM/Vision/P-text/P-prompt/P-SoM)."""
        required = {"DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM"}
        return required.issubset(set(self.modes.keys()))


def expected_n_canonical(site: str) -> int:
    """Canonical paper-grade N per site (post-§139.8 N/A exclusion at task-load).

    Use everywhere figures need `expected_n`; do NOT hardcode 234/210/466 anymore.
    /stress A1.20 P1-1-AB fix (2026-05-17, A1.19 §139.8 catalog tail).
    """
    return scored_task_count(site, "visualwebarena", strict=True)


def paper_grade_panels(
    *,
    sites: Iterable[str] = ("classifieds", "reddit"),
    baselines: Iterable[str] | None = None,
    include_placeholders: bool = True,
) -> list[PanelSpec]:
    """Build panel list from `run_registry` + canonical N.

    Args:
      sites: site subset (default cls+red Phase 1a scope; pass `SITES` for full).
      baselines: baseline subset (default all from `BASELINES` registry, i.e.
        B0+B1+B2 post-2026-05-14). Pass explicit list to drop one.
      include_placeholders: if True, cells with 0 modes loaded emit a placeholder
        PanelSpec (is_placeholder=True) so figure can render "pending" tile rather
        than silently dropping panel. Default True (paper-grade transparency).

    Returns: sorted list[PanelSpec] in (baseline_idx, site_idx) canonical order.
    """
    if baselines is None:
        baselines = list(BASELINES)
    panels: list[PanelSpec] = []
    for baseline in baselines:
        for site in sites:
            cells = get_cells(baseline=baseline, site=site)
            modes = {cell.mode: cell.episodes_dir for cell in cells}
            expected = expected_n_canonical(site)
            is_placeholder = not modes
            if is_placeholder and not include_placeholders:
                continue
            panels.append(PanelSpec(
                key=f"{baseline.lower()}_{site[:3]}",
                title=f"{baseline} {site}",
                baseline=baseline,
                site=site,
                expected_n=expected,
                modes=modes,
                is_placeholder=is_placeholder,
            ))
    return panels
