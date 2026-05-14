"""Run manifest registry for paper analysis cells.

The manifest is the single source of truth for mapping paper cells
``(baseline, site, mode)`` to run directories and condition subdirectories.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal
import argparse
import warnings

import yaml


PAPER_MODES = ["DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM"]
LEGACY_MODE_ALIAS = {
    "dom": "DOM",
    "som": "SoM",
    "vision": "Vision",
    "phantom_dom": "P-text",
    "phantom_text": "P-text",
    "phantom_prompt": "P-prompt",
    "phantom_som": "P-SoM",
}
SITES = ["classifieds", "reddit", "shopping"]
BASELINES = ["B0", "B1"]
# §139.8: scored task counts (total − N/A tasks excluded at load time) from the
# single source of truth. Pre-exclusion counts were classifieds 234 / reddit
# 210 / shopping 466. A manifest entry's explicit `expected_n` still overrides.
from p79.experiment.analysis import scored_task_count as _scored_task_count
EXPECTED_N = {_s: _scored_task_count(_s, "visualwebarena") for _s in SITES}

Grade = Literal["paper-grade", "paper-grade-pre-bug", "in-flight", "archived"]
GRADES = ("paper-grade", "paper-grade-pre-bug", "in-flight", "archived")

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE1_ROOT = REPO_ROOT / "results/visualwebarena/phase1"
DEFAULT_MANIFEST = REPO_ROOT / "results/phantom_paper/run_manifest.yaml"
DEFAULT_GRADE_FILTER: list[Grade] = ["paper-grade"]
# F01 audit fix 2026-05-09: pre-bug cells excluded by default. Pass
# `grade=["paper-grade", "paper-grade-pre-bug"]` explicitly for Appendix-D
# robustness check; never as default.


@dataclass(frozen=True)
class CellSpec:
    """One (baseline x site x mode) cell, identified by full run + condition path."""

    baseline: str
    site: str
    mode: str
    run_dir: Path
    condition_subdir: str
    expected_n: int
    grade: Grade
    notes: str = ""

    @property
    def episodes_dir(self) -> Path:
        return self.run_dir / self.condition_subdir / "episodes"

    @property
    def condition_summary_path(self) -> Path:
        return self.run_dir / self.condition_subdir / "condition_summary_v2.json"

    @property
    def actual_n(self) -> int:
        """Live count of episode summary files."""
        if not self.episodes_dir.exists():
            return 0
        return len(list(self.episodes_dir.glob("*_summary_v2.json")))

    @property
    def is_complete(self) -> bool:
        return self.actual_n >= self.expected_n


def canonical_mode(mode: str) -> str:
    """Resolve legacy condition mode ids to paper-canonical mode names."""
    key = mode.strip().replace("-", "_").lower()
    return LEGACY_MODE_ALIAS.get(key, mode)


def _resolve_run_dir(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts[:1] == ("results",):
        return REPO_ROOT / path
    return PHASE1_ROOT / path


def _manifest_path(path: Path | None = None) -> Path:
    return DEFAULT_MANIFEST if path is None else Path(path)


@lru_cache(maxsize=4)
def _load_manifest_cached(path_str: str) -> dict:
    path = Path(path_str)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a mapping: {path}")
    return data


def load_manifest(path: Path | None = None) -> dict:
    """Load run_manifest.yaml.

    Default path: results/phantom_paper/run_manifest.yaml.
    """
    return _load_manifest_cached(str(_manifest_path(path).resolve()))


def _iter_manifest_entries(manifest: dict) -> list[dict]:
    entries: list[dict] = []
    for section in ("cells", "in_flight", "archived"):
        section_entries = manifest.get(section) or []
        if not isinstance(section_entries, list):
            raise ValueError(f"Manifest section {section!r} must be a list")
        for entry in section_entries:
            if isinstance(entry, str):
                if section != "archived":
                    raise ValueError(f"String entries are only supported in archived: {entry}")
                entry = {
                    "baseline": "?",
                    "site": "?",
                    "mode": "DOM",
                    "run_dir": entry,
                    "condition_subdir": "",
                    "expected_n": 0,
                    "grade": "archived",
                }
            if not isinstance(entry, dict):
                raise ValueError(f"Manifest entry must be a mapping: {entry!r}")
            entries.append(entry)
    return entries


def _entry_to_cell(entry: dict) -> CellSpec:
    grade = entry.get("grade")
    if grade not in GRADES:
        raise ValueError(f"Invalid grade {grade!r} for {entry.get('baseline')}/{entry.get('site')}/{entry.get('mode')}")

    mode = canonical_mode(str(entry.get("mode", "")))
    if mode not in PAPER_MODES:
        raise ValueError(f"Invalid mode {mode!r} for {entry.get('baseline')}/{entry.get('site')}")

    site = str(entry.get("site", ""))
    expected_n = int(entry.get("expected_n") or EXPECTED_N.get(site, 0))
    run_dir = _resolve_run_dir(entry["run_dir"])
    cell = CellSpec(
        baseline=str(entry.get("baseline", "")),
        site=site,
        mode=mode,
        run_dir=run_dir,
        condition_subdir=str(entry.get("condition_subdir", "")),
        expected_n=expected_n,
        grade=grade,
        notes=str(entry.get("notes", "")),
    )
    if grade in ("paper-grade", "paper-grade-pre-bug") and cell.actual_n == 0:
        warnings.warn(
            f"Paper cell has no episode summaries: {cell.baseline}/{cell.site}/{cell.mode} -> {cell.episodes_dir}",
            RuntimeWarning,
            stacklevel=2,
        )
    return cell


def _sort_key(cell: CellSpec) -> tuple:
    baseline_idx = BASELINES.index(cell.baseline) if cell.baseline in BASELINES else len(BASELINES)
    site_idx = SITES.index(cell.site) if cell.site in SITES else len(SITES)
    mode_idx = PAPER_MODES.index(cell.mode) if cell.mode in PAPER_MODES else len(PAPER_MODES)
    return (baseline_idx, cell.baseline, site_idx, cell.site, mode_idx, cell.mode, str(cell.run_dir))


def _all_cells_unfiltered(path: Path | None = None) -> list[CellSpec]:
    manifest = load_manifest(path)
    cells = [_entry_to_cell(entry) for entry in _iter_manifest_entries(manifest)]
    seen: set[tuple[str, str, str, Grade]] = set()
    for cell in cells:
        key = (cell.baseline, cell.site, cell.mode, cell.grade)
        if key in seen and cell.grade != "archived":
            raise ValueError(f"Duplicate manifest cell: {cell.baseline}/{cell.site}/{cell.mode}/{cell.grade}")
        seen.add(key)
    return sorted(cells, key=_sort_key)


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def get_all_cells(grade_filter: list[Grade] | None = None) -> list[CellSpec]:
    """All cells in manifest.

    grade_filter defaults to ['paper-grade'] only (post-F01 audit
    2026-05-09). Pass ['paper-grade', 'paper-grade-pre-bug'] explicitly
    for Appendix-D legacy robustness; ['archived'] / ['in-flight'] for
    sensitivity analyses.
    """
    grades = DEFAULT_GRADE_FILTER if grade_filter is None else grade_filter
    return [cell for cell in _all_cells_unfiltered() if cell.grade in grades]


def get_cells(
    *,
    baseline: str | list[str] | None = None,
    site: str | list[str] | None = None,
    mode: str | list[str] | None = None,
    grade: Grade | list[Grade] | None = None,
) -> list[CellSpec]:
    """Filter cells by any combination of fields. Returns sorted list."""
    baselines = set(_as_list(baseline) or [])
    sites = set(_as_list(site) or [])
    modes = {canonical_mode(m) for m in (_as_list(mode) or [])}
    grades = set(_as_list(grade) or DEFAULT_GRADE_FILTER)

    out = []
    for cell in _all_cells_unfiltered():
        if baselines and cell.baseline not in baselines:
            continue
        if sites and cell.site not in sites:
            continue
        if modes and cell.mode not in modes:
            continue
        if grades and cell.grade not in grades:
            continue
        out.append(cell)
    return sorted(out, key=_sort_key)


def get_cell(baseline: str, site: str, mode: str) -> CellSpec | None:
    """Get exactly one cell or None if not in manifest."""
    matches = get_cells(baseline=baseline, site=site, mode=mode, grade=list(GRADES))
    if not matches:
        return None
    return matches[0]


def get_run_dirs_paper_vwa() -> list[Path]:
    """Compatibility shim: return paper-grade VWA run dirs, deduped."""
    out: list[Path] = []
    seen: set[Path] = set()
    for cell in get_all_cells():
        if cell.grade == "archived":
            continue
        if cell.run_dir not in seen:
            out.append(cell.run_dir)
            seen.add(cell.run_dir)
    return out


def _report() -> str:
    cells = _all_cells_unfiltered()
    paper_cells = [cell for cell in cells if cell.grade in DEFAULT_GRADE_FILTER]
    paper_groups: dict[tuple[str, str], int] = {}
    for cell in paper_cells:
        paper_groups[(cell.baseline, cell.site)] = paper_groups.get((cell.baseline, cell.site), 0) + 1
    paper_detail = ", ".join(f"{baseline} {site}={count}" for (baseline, site), count in sorted(paper_groups.items()))
    lines: list[str] = [f"Paper-grade cells: {len(paper_cells)} ({paper_detail})"]
    for grade in GRADES:
        grade_cells = [cell for cell in cells if cell.grade == grade]
        if not grade_cells:
            continue
        groups: dict[tuple[str, str], int] = {}
        for cell in grade_cells:
            groups[(cell.baseline, cell.site)] = groups.get((cell.baseline, cell.site), 0) + 1
        detail = ", ".join(f"{baseline} {site}={count}" for (baseline, site), count in sorted(groups.items()))
        label = grade
        lines.append(f"{label}: {len(grade_cells)} ({detail})")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the analysis run registry")
    parser.add_argument("--report", action="store_true", help="Print cell coverage report")
    args = parser.parse_args()
    if args.report:
        print(_report())
    else:
        cells = get_all_cells()
        print(f"{len(cells)} cells loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
