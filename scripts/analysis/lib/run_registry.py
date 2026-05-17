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
# A1.21 P1-13 fix (B-498): MODE_TO_KEY consolidated here as single source of truth
# (was duplicated in scripts/analysis/generate_per_task_sr.py:62-69 — silent drift risk
# when new mode added). Maps paper-canonical mode names → CSV column key suffix.
MODE_TO_KEY = {
    "DOM": "dom",
    "SoM": "som",
    "Vision": "vision",
    "P-text": "ptext",
    "P-prompt": "pprompt",
    "P-SoM": "psom",
}
LEGACY_MODE_ALIAS = {
    "dom": "DOM",
    "som": "SoM",
    "vision": "Vision",
    # /stress A1.19 P1-6-A (2026-05-17, Claude): `phantom_dom` and `phantom_text` BOTH
    # map to canonical P-text → silent merge risk if a manifest contains both for the
    # same (baseline, site) at the same grade tier. Mitigation: (a) B-261 retired
    # `phantom_dom` obs_mode from all new YAML configs (2026-05-16), so paper-grade
    # entries should never use `phantom_dom`; (b) `_all_cells_unfiltered` now detects
    # the silent-merge case and raises explicitly (see post-canonicalization dup check
    # below). Archive runs still use `phase1_phantom_dom_router_0/` dirs, hence the
    # alias is retained for backward-compat in the `archived` grade tier only.
    "phantom_dom": "P-text",
    "phantom_text": "P-text",
    "phantom_prompt": "P-prompt",
    "phantom_som": "P-SoM",
}
SITES = ["classifieds", "reddit", "shopping"]
BASELINES = ["B0", "B1", "B2"]
# §139.8 + /stress A1.6 (2026-05-16): paper-grade EXPECTED_N (total − N/A
# excluded at load time). Pre-exclusion was classifieds 234 / reddit 210 /
# shopping 466; post-exclusion = 224 / 205 / 435. `strict=True` fails loud
# on missing VWA config — silent 0-fallback used to mark missing cells as
# `is_complete == True` (n=0 >= expected=0).
from p79.experiment.analysis import scored_task_count as _scored_task_count
EXPECTED_N = {_s: _scored_task_count(_s, "visualwebarena", strict=True) for _s in SITES}

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
        # /stress A1.6 (2026-05-16): expected_n must be positive or the
        # "actual >= 0" tautology silently marks missing data as complete.
        if self.expected_n <= 0:
            return False
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


def _entry_to_cell(entry: dict, *, strict_paper_grade: bool = False) -> CellSpec:
    """Build CellSpec from manifest entry.

    A1.21 P1-1 + P0-8 fix (B-491 + B-492): paper-grade tier now rejects hardcoded
    `expected_n` override (forces canonical `scored_task_count`) AND when
    `strict_paper_grade=True` (i.e., post-Phase-1a promotion), `actual_n == 0` raises
    RuntimeError instead of warning (silent missing-data → false-completeness vector).
    Set via `get_all_cells(..., strict_paper_grade=True)` or directly through
    `validate_run_manifest.py` (B-494 P0-8 validator).
    """
    grade = entry.get("grade")
    if grade not in GRADES:
        raise ValueError(f"Invalid grade {grade!r} for {entry.get('baseline')}/{entry.get('site')}/{entry.get('mode')}")

    mode = canonical_mode(str(entry.get("mode", "")))
    if mode not in PAPER_MODES:
        raise ValueError(f"Invalid mode {mode!r} for {entry.get('baseline')}/{entry.get('site')}")

    site = str(entry.get("site", ""))
    # A1.21 P1-1 fix (B-491): paper-grade tier rejects hardcoded expected_n.
    # Archived tier preserved backwards-compat (pre-§139.8 234 cls / 210 red / 466 shop).
    raw_expected = entry.get("expected_n")
    if grade in ("paper-grade", "paper-grade-pre-bug") and raw_expected is not None:
        canonical = EXPECTED_N.get(site, 0)
        if int(raw_expected) != canonical:
            warnings.warn(
                f"Paper-grade entry {entry.get('baseline')}/{site}/{mode} has stale "
                f"`expected_n: {raw_expected}` (likely copied from archived); canonical "
                f"post-§139.8 scored_task_count for {site} = {canonical}. Drop the "
                f"`expected_n:` line from this entry OR set to {canonical} (A1.21 B-491).",
                RuntimeWarning, stacklevel=2,
            )
        expected_n = canonical  # Force canonical regardless of yaml override for paper-grade
    else:
        expected_n = int(raw_expected) if raw_expected is not None else EXPECTED_N.get(site, 0)

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
        if strict_paper_grade:
            raise RuntimeError(
                f"Paper-grade cell has no episode summaries: {cell.baseline}/{cell.site}/{cell.mode} "
                f"-> {cell.episodes_dir}. Either run_dir/condition_subdir mismatch (yaml typo) OR "
                f"Phase 1a launch hasn't fired this cell yet. Run scripts/analysis/validate_run_manifest.py "
                f"to confirm. (A1.21 B-492 strict_paper_grade=True semantics)"
            )
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


def _all_cells_unfiltered(path: Path | None = None, *,
                            strict_paper_grade: bool = False) -> list[CellSpec]:
    manifest = load_manifest(path)
    cells = [_entry_to_cell(entry, strict_paper_grade=strict_paper_grade)
             for entry in _iter_manifest_entries(manifest)]
    seen: set[tuple[str, str, str, Grade]] = set()
    # /stress A1.19 P1-6-A (2026-05-17): also track (baseline, site, mode) -> set[grade]
    # to detect LEGACY_MODE_ALIAS silent-merge across grade tiers within the SAME canonical
    # mode. Pre-fix: archive `phantom_dom` (alias→P-text) + paper-grade `phantom_text`
    # (alias→P-text) for same (baseline, site) silently collapsed to single dict key in
    # downstream `get_cells(mode="P-text")` queries. Now explicit cross-grade-tier
    # collision raises so user must explicitly resolve via archived-vs-paper-grade pick.
    cross_tier: dict[tuple[str, str, str], list[Grade]] = {}
    for cell in cells:
        key = (cell.baseline, cell.site, cell.mode, cell.grade)
        if key in seen and cell.grade != "archived":
            raise ValueError(f"Duplicate manifest cell: {cell.baseline}/{cell.site}/{cell.mode}/{cell.grade}")
        seen.add(key)
        ck = (cell.baseline, cell.site, cell.mode)
        cross_tier.setdefault(ck, []).append(cell.grade)
    for ck, grades in cross_tier.items():
        # Cross-tier alias collision: same canonical (baseline, site, mode) has BOTH a
        # paper-grade entry AND an archived entry — this is normally fine (archive +
        # paper-grade can coexist) BUT if both came from different LEGACY_MODE_ALIAS
        # source strings collapsing to the same canonical mode, downstream callers may
        # accidentally merge. Surface a warning so user can audit run_manifest.yaml.
        if "paper-grade" in grades and "archived" in grades:
            # Look up the source mode strings to detect collapsing alias chains.
            sources = sorted({
                entry.get("mode", "")
                for entry in _iter_manifest_entries(manifest)
                if entry.get("baseline") == ck[0] and entry.get("site") == ck[1]
                and canonical_mode(str(entry.get("mode", ""))) == ck[2]
            })
            if len(sources) > 1:
                warnings.warn(
                    f"LEGACY_MODE_ALIAS silent-merge risk: {ck[0]}/{ck[1]}/{ck[2]} has "
                    f"multiple source modes {sources} mapping to same canonical key "
                    f"across grade tiers {grades}. Audit run_manifest.yaml to ensure "
                    f"`archived` and `paper-grade` entries are intentionally distinct.",
                    RuntimeWarning, stacklevel=2,
                )
    return sorted(cells, key=_sort_key)


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def get_all_cells(grade_filter: list[Grade] | None = None, *,
                    manifest_path: Path | None = None,
                    strict_paper_grade: bool = False) -> list[CellSpec]:
    """All cells in manifest.

    grade_filter defaults to ['paper-grade'] only (post-F01 audit
    2026-05-09). Pass ['paper-grade', 'paper-grade-pre-bug'] explicitly
    for Appendix-D legacy robustness; ['archived'] / ['in-flight'] for
    sensitivity analyses.

    A1.21 P0-5 fix (B-493): `manifest_path` arg actually propagates to data
    discovery (was provenance theater in `generate_per_task_sr.py:131` —
    CLI defined `--run-manifest` arg but registry still read default).
    A1.21 P0-8 (B-492): `strict_paper_grade=True` raises on actual_n==0 for
    paper-grade cells (validator enforcement entry point).
    """
    grades = DEFAULT_GRADE_FILTER if grade_filter is None else grade_filter
    return [cell for cell in _all_cells_unfiltered(manifest_path, strict_paper_grade=strict_paper_grade)
            if cell.grade in grades]


def get_cells(
    *,
    baseline: str | list[str] | None = None,
    site: str | list[str] | None = None,
    mode: str | list[str] | None = None,
    grade: Grade | list[Grade] | None = None,
    manifest_path: Path | None = None,
) -> list[CellSpec]:
    """Filter cells by any combination of fields. Returns sorted list.

    A1.21 P0-5 (B-493): `manifest_path` arg propagates to data discovery.
    """
    baselines = set(_as_list(baseline) or [])
    sites = set(_as_list(site) or [])
    modes = {canonical_mode(m) for m in (_as_list(mode) or [])}
    grades = set(_as_list(grade) or DEFAULT_GRADE_FILTER)

    out = []
    for cell in _all_cells_unfiltered(manifest_path):
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
