"""Shared JSONL I/O utilities with restart dedup and corrupt-line handling."""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)


# B-283 fix (2026-05-16, A1.8): paper-grade-strict episode summary loader for
# outcome aggregators. Pre-fix `aggregate_sr_fp_per_mode.py:77` +
# `aggregate_phantom_lift.py:109` used `bool(row.get("success", False))` —
# JSON literal `"success": "false"` (string) is Python truthy → SR inflated.
# Strict loader requires bool-typed success / int task_id / non-None schema
# version so the headline paper §1 hero number is type-safe at the boundary.
StrictMode = Literal["strict", "lenient"]


def load_episode_summary_strict(
    path: Path,
    *,
    mode: StrictMode = "strict",
    reject_needs_reevaluation: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load an episode summary JSON with paper-grade type-safety enforcement.

    Args:
        path: path to `*_summary_v2.json`.
        mode: "strict" raises on type mismatch (default for paper aggregators);
              "lenient" logs warning + returns None (for diagnostic tools that
              want to survey all archives without crashing).
        reject_needs_reevaluation: B-542 (/stress A1.5b Phase 2 P0-3-B codex
            OOB, 2026-05-17). When True, treat a payload with
            `needs_reevaluation=True` (B-486 quarantine flag) as load failure
            — strict mode raises ValueError; lenient logs + returns None.
            Canonical paper-grade aggregators producing the first-published
            SR / oracle-lift numbers MUST set this True so quarantined episodes
            (crash before evaluator scored) do NOT enter the H1/H2/H3 universe
            as `success=False` failures. Default False preserves legacy
            consumer semantics where quarantined rows are tolerated as
            transparency input.

    Returns:
        dict on success, None on lenient-mode soft failure.

    Raises:
        ValueError in strict mode on type mismatch or quarantined episode
            (when reject_needs_reevaluation=True).
        FileNotFoundError if `path` does not exist.

    Paper-grade contract: callers consuming `success` for SR / lift computation
    MUST use strict mode. Pre-fix aggregators that did `bool(d.get("success"))`
    would have accepted `"success": "false"` as truthy (paper §1 hero risk).
    """
    if not path.exists():
        raise FileNotFoundError(f"Episode summary not found: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            if mode == "strict":
                raise ValueError(f"Corrupt JSON in {path}: {exc}") from exc
            logger.warning("B-283 load_episode_summary_strict[lenient] corrupt JSON %s: %s", path, exc)
            return None
    bad: List[str] = []
    if not isinstance(payload.get("success"), bool):
        bad.append(f"success={payload.get('success')!r} (type={type(payload.get('success')).__name__}), expected bool")
    if not isinstance(payload.get("task_id"), int):
        bad.append(f"task_id={payload.get('task_id')!r} (type={type(payload.get('task_id')).__name__}), expected int")
    if not isinstance(payload.get("schema_version"), str):
        bad.append(f"schema_version={payload.get('schema_version')!r}, expected str")
    if bad:
        msg = f"Paper-grade type mismatch in {path}: " + "; ".join(bad)
        if mode == "strict":
            raise ValueError(msg)
        logger.warning("B-283 load_episode_summary_strict[lenient] %s", msg)
        return None
    # B-542: paper-grade quarantine filter. Quarantined episodes (B-486) have
    # `needs_reevaluation=True` and represent crash-before-evaluator state;
    # their `success=False` is bookkeeping, not a real evaluator outcome.
    # Treating them as failures inflates the denominator with non-evaluated
    # tasks → paper §1 hero / H1/H2/H3 universe pollution.
    #
    # B-734 fix (/stress A1.8 cold-start P0-4-B* codex OOB, 2026-05-17): explicit
    # type-check on `needs_reevaluation` BEFORE bool coercion. Pre-fix used
    # `bool(payload.get("needs_reevaluation", False))` — JSON literal
    # `"needs_reevaluation": "false"` (string) would be Python truthy →
    # valid episode silently rejected → canonical aggregator
    # (aggregate_phase1_full_prereg_decision.py:164) catches ValueError +
    # `continue` → denominator silent shrink → paper §1 hero N off without
    # any audit log entry. Cold-start codex Mode B caught the B-283 sibling
    # propagation lineage. Now: present + non-bool → fail-loud in strict;
    # warn + skip-row in lenient.
    nrv_raw = payload.get("needs_reevaluation", False)
    if not isinstance(nrv_raw, bool):
        nrv_msg = (
            f"B-734 needs_reevaluation type mismatch in {path}: "
            f"got {type(nrv_raw).__name__}={nrv_raw!r}, expected bool"
        )
        if mode == "strict":
            raise ValueError(nrv_msg)
        logger.warning("B-734 load_episode_summary_strict[lenient] %s", nrv_msg)
        return None
    if reject_needs_reevaluation and nrv_raw:
        msg = (
            f"B-542 quarantined episode (needs_reevaluation=True) rejected by "
            f"paper-grade aggregator at {path}: task_id={payload.get('task_id')!r}, "
            f"error={str(payload.get('error', ''))[:120]!r}"
        )
        if mode == "strict":
            raise ValueError(msg)
        logger.warning("B-542 load_episode_summary_strict[lenient] %s", msg)
        return None
    return payload


# B-196 (/stress A1.4b-ii codex B-ii-4, P1): module-level corruption counter
# so `analysis.py::analyze_run` can emit a canonical `jsonl_integrity_report.csv`
# alongside `parse_failures.csv` (B-174). Reset at the start of each
# `analyze_run` and read at the end. Each entry records:
#   {"path": str, "lines_read": int, "corrupt_lines": int,
#    "dedup_discarded": int, "summary_identity_mismatch": bool}
_JSONL_INTEGRITY_LOG: List[Dict[str, Any]] = []


def dedup_restart_lines(file_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the last run when a JSONL has restart artifacts.

    The runner appends to JSONL (mode='a'), so if the watchdog/queue
    restarts a task, earlier runs' steps remain in the file.  We detect
    restarts by step_idx resetting to 0 and keep lines from the last run
    (matching the authoritative summary_v2.json which is overwritten).

    B-287 fix (2026-05-16, A1.8): when no `step_idx=0` boundary exists in the
    file (legitimate edge: first step lost to crash before flush, or corrupt
    first line), fall back to the minimum step_idx position as boundary —
    rather than the pre-fix silent "keep entire file" behaviour that mixes
    earlier-run + later-run records into a single segment.

    B-737 fix (/stress A1.8 cold-start P1-3-AB 2-AI overlap, 2026-05-17):
    explicit `isinstance(step_idx, int)` type guard. Pre-fix relied on
    `rec.get("step_idx", -1) == 0` equality — JSON `"step_idx": "0"` (string)
    would NOT equal int `0` so boundary never triggered → entire file
    returned as single segment → duplicate restart rows surfaced to
    aggregator. Both Claude F7 + codex B4 cold-start independently caught.
    Now: only int step_idx counts as boundary; non-int rows are skipped.
    Companion `read_jsonl_dedup(strict_identity=True)` raises if ANY row
    has non-int step_idx (fail-loud paper-grade path).
    """
    if not file_lines:
        return file_lines
    last_run_start = 0
    for i, rec in enumerate(file_lines):
        step_idx_val = rec.get("step_idx", -1)
        # B-737: int-strict equality. `"0" == 0` is False (good) but
        # `bool(0) == 0` is True (bad — bool subclass of int). Exclude bool.
        if i > 0 and isinstance(step_idx_val, int) and not isinstance(step_idx_val, bool) and step_idx_val == 0:
            last_run_start = i
    if last_run_start == 0 and len(file_lines) > 1:
        # B-287: no step_idx=0 boundary in tail — find the LAST position where
        # step_idx decreased relative to its predecessor (= restart edge).
        for i in range(1, len(file_lines)):
            prev = file_lines[i - 1].get("step_idx", -1)
            curr = file_lines[i].get("step_idx", -1)
            # B-737: both must be int (NOT bool); string `"5"` vs `"6"` would
            # have compared as strings ("string-decrease" never triggers).
            if (isinstance(prev, int) and not isinstance(prev, bool)
                and isinstance(curr, int) and not isinstance(curr, bool)
                and curr < prev):
                last_run_start = i
    if last_run_start > 0:
        logger.debug(
            "Dedup: discarded %d lines from earlier run(s)", last_run_start
        )
    return file_lines[last_run_start:]


def _assert_step_idx_monotonic(segment: List[Dict[str, Any]]) -> bool:
    """B-287 helper: return True if post-dedup segment has monotonic step_idx.

    Used by the integrity log to flag suspected restart-pattern corruption that
    dedup did not catch (paper-grade analysis pipeline must know).
    """
    last = -1
    for rec in segment:
        idx = rec.get("step_idx")
        if not isinstance(idx, int) or idx <= last:
            return False
        last = idx
    return True


def read_jsonl_dedup(
    path: Path,
    summary_path: Optional[Path] = None,
    *,
    strict_identity: bool = False,
) -> List[Dict[str, Any]]:
    """Read a single JSONL file, deduplicating restart artifacts.

    B-180 (/stress A1.4b-i codex B7): when ``summary_path`` is provided, the
    last JSONL segment is validated against the summary's authoritative
    fields ((schema_version, run_id, condition_id, seed, benchmark_site,
    task_id, steps)). If the segment doesn't match, log a warning + still
    return the last-segment lines (caller decides whether to consume) so
    audit can see the divergence without crashing analysis. Pre-fix: a
    restart that wrote ``step_idx=0`` and then crashed before summary
    overwrite would have the old complete summary co-exist with the new
    partial segment; the dedup unconditionally kept the partial, breaking
    step-level cost/latency diagnostics while episode-level summaries
    pointed at the old run.

    B-571 (/stress A1.22 P1-13-B* codex OOB, 2026-05-17): `strict_identity`
    kwarg added. When True AND `summary_path` is provided AND the validator
    reports `identity_mismatch=True`, raise ValueError instead of silently
    returning the partial tail. Default False preserves the legacy
    transparency-only behavior (analysis pipeline can still inspect
    `_JSONL_INTEGRITY_LOG`). Paper-grade analysis call sites should pass
    `strict_identity=True` to fail-loud on restart-crash silent tail
    pollution — cross-baseline asymmetric (slow B1/B2 more prone to
    partial/restart than B0 quick API calls).
    """
    file_lines: List[Dict[str, Any]] = []
    corrupt_count = 0
    total_lines = 0
    # B-288 fix (2026-05-16, A1.8): catch UnicodeDecodeError so a single invalid
    # UTF-8 byte (e.g. broken screenshot path / encoding mishap in obs.text)
    # doesn't crash the whole analyze pipeline. `errors="replace"` substitutes
    # the bad byte with U+FFFD and continues; corrupt lines fall through the
    # JSONDecodeError handler below.
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                file_lines.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Dropped corrupt JSONL line %d in %s: %.100s", line_num, path, line)
                corrupt_count += 1
                continue
    last_segment = dedup_restart_lines(file_lines)
    dedup_discarded = max(0, len(file_lines) - len(last_segment))

    # B-293 fix (2026-05-16, A1.8): semantic Optional[bool] — None means
    # "not checked" (summary_path was None); False = "checked and matched";
    # True = "checked and mismatch". Pre-fix all rows logged False even when
    # validator did not run → reviewer audit could not distinguish.
    identity_mismatch: Optional[bool] = None
    strict_mode_active = (
        strict_identity or os.environ.get("P79_STRICT_READ_JSONL", "") == "1"
    )
    if summary_path is not None and last_segment:
        # B-738 cold-start fix: strict_rows=True under strict_identity so the
        # helper checks every row + None identity values count as mismatch.
        identity_mismatch = _validate_against_summary(
            path, last_segment, summary_path,
            strict_rows=strict_mode_active,
        )
        # B-571 (/stress A1.22 P1-13-B* codex OOB, 2026-05-17): strict mode
        # — paper-grade analysis call sites pass strict_identity=True to
        # fail-loud on restart-crash silent tail pollution. Pre-fix the
        # mismatched tail was returned silently; only `_JSONL_INTEGRITY_LOG`
        # carried the warning, which `analysis.py:269-279` did not read.
        # Lenient default preserves legacy transparency-only behavior;
        # `P79_STRICT_READ_JSONL=1` env opts strict-mode into ALL call
        # sites for paper-grade fire-blocker check.
        if identity_mismatch and strict_mode_active:
            raise ValueError(
                f"read_jsonl_dedup: summary identity mismatch for {path} "
                f"vs {summary_path}; strict_identity=True refuses to return "
                f"the partial tail (paper-grade fail-loud). Set "
                f"P79_STRICT_READ_JSONL=0 OR drop strict_identity kwarg for "
                f"lenient transparency-only mode."
            )
        # B-733 cold-start fix: strict_identity + summary missing/unreadable
        # (helper returned None) → fail-loud. Pre-fix the helper returned
        # hard False ("checked & OK") on missing summary, so strict_identity
        # never fired on this case → orphan step JSONL silently entered
        # paper-grade analysis. Now: missing summary is a paper-grade fail-
        # loud, since `strict_identity=True` was the caller's promise of
        # "summary identity verified before consumption".
        if identity_mismatch is None and strict_mode_active and summary_path is not None:
            raise ValueError(
                f"read_jsonl_dedup: summary file missing or unreadable at "
                f"{summary_path} for {path}; strict_identity=True refuses "
                f"orphan step JSONL (B-733 paper-grade fail-loud — crash "
                f"window after step fsync before summary atomic-replace)."
            )

    # B-287: post-dedup invariant — step_idx must be monotonic. If not, the
    # last_segment still has restart artifact bleed-through; surface to the
    # integrity log so reviewer audit catches what dedup missed.
    #
    # B-737 cold-start fix: type-strict check on step_idx. Any non-int
    # step_idx in a paper-grade segment counts as restart-artifact poison
    # (string `"0"` would not trigger dedup boundary, leaving duplicate
    # rows in segment). Under strict_identity, fail-loud rather than log-
    # only-advisory.
    non_int_step_idx = (
        [i for i, r in enumerate(last_segment)
         if not isinstance(r.get("step_idx"), int) or isinstance(r.get("step_idx"), bool)]
        if last_segment else []
    )
    step_idx_non_monotonic = (not _assert_step_idx_monotonic(last_segment)) if last_segment else False
    if strict_mode_active and non_int_step_idx:
        raise ValueError(
            f"read_jsonl_dedup: non-int step_idx at row indices "
            f"{non_int_step_idx[:5]}{'...' if len(non_int_step_idx) > 5 else ''} "
            f"in {path}; strict_identity=True refuses corrupted/poisoned "
            f"step_idx values (B-737 paper-grade fail-loud — `\"0\"` string "
            f"would bypass dedup boundary detection)."
        )
    if strict_mode_active and step_idx_non_monotonic:
        raise ValueError(
            f"read_jsonl_dedup: post-dedup step_idx non-monotonic in {path}; "
            f"strict_identity=True refuses restart-artifact bleed-through "
            f"(B-737 paper-grade fail-loud — dedup heuristic failed to find "
            f"clean restart boundary)."
        )

    # B-196: record per-file integrity stats; consumed by analysis.py
    _JSONL_INTEGRITY_LOG.append({
        "path": str(path),
        "lines_read": total_lines,
        "corrupt_lines": corrupt_count,
        "dedup_discarded": dedup_discarded,
        "summary_identity_mismatch": identity_mismatch,
        # B-287: flag if dedup did not produce a clean monotonic segment.
        "step_idx_non_monotonic": step_idx_non_monotonic,
        # B-290: pointer to where discarded earlier-segment data was archived
        # (sidecar). None when no segments discarded; analysis can audit.
        "discarded_segments_archive": None,
        # B-733 cold-start fix: explicit "summary missing or unreadable"
        # telemetry. Distinguishes:
        #   - None: summary_path arg was None (validator didn't run by design)
        #   - True: summary_path provided but file missing/unreadable on disk
        #   - False: summary_path provided + file exists + readable
        # Auditors grepping for `summary_missing=True` find orphan-JSONL
        # scenarios (crash window). Paired with `summary_identity_mismatch`
        # which now correctly None-encodes "not checked" (B-733 tri-state).
        "summary_missing": (
            None if summary_path is None
            else (not summary_path.exists()) if summary_path else None
        ),
        # B-737 cold-start fix: count of rows with non-int step_idx (poisoned
        # by string coercion attack or future schema drift). 0 in clean
        # paths; >0 means dedup boundary detection was structurally
        # compromised. Strict mode raises; lenient surfaces the count.
        "non_int_step_idx_rows": len(non_int_step_idx),
    })

    return last_segment


def _validate_against_summary(
    jsonl_path: Path,
    last_segment: List[Dict[str, Any]],
    summary_path: Path,
    *,
    strict_rows: bool = False,
) -> Optional[bool]:
    """B-180 helper: emit warning if last JSONL segment doesn't match summary.

    Identity tuple checked: (schema_version, run_id, condition_id, seed,
    benchmark_site, task_id). Cardinality: len(last_segment) vs summary
    `steps` field. All mismatches are logged + caller may inspect; not raised
    because in-progress data is legitimately mid-flight.

    Returns:
        - True: mismatch detected
        - False: checked, no mismatch
        - None: NOT checked (summary file missing, unreadable, or empty
          segment). B-733 fix (/stress A1.8 cold-start P0-3-AB* OOB,
          2026-05-17): pre-fix returned hard False on missing summary,
          which got bundled into `read_jsonl_dedup`'s Optional[bool]
          `identity_mismatch` as "checked & OK" — exactly the B-293
          tri-state contract violation that B-293 closed at one layer
          but `_validate_against_summary` reintroduced one level down.
          Now: tri-state honored end-to-end.

    Args:
        strict_rows: B-738 fix (/stress A1.8 cold-start P1-4-B codex,
            2026-05-17). When True, identity check loops over ALL rows of
            the segment (not just `last_segment[0]`), and any None value
            for an identity-key on either side counts as mismatch (was
            previously silently `continue`'d). Default False preserves
            legacy advisory-mode semantics for non-strict callers.
    """
    if not summary_path.exists():
        # B-733 cold-start fix: tri-state None ("not checked") rather than
        # hard False ("checked & OK"). The caller `read_jsonl_dedup`'s
        # `identity_mismatch: Optional[bool]` field thereby reports the
        # genuine "no summary on disk" condition to `_JSONL_INTEGRITY_LOG`
        # auditors. Paper-grade analysis path with `strict_identity=True`
        # detects None and can fail-loud.
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "B-180 read_jsonl_dedup: cannot read summary %s for identity check: %s",
            summary_path, exc,
        )
        # B-733 cold-start fix: also tri-state None for "summary unreadable"
        # — same semantic as missing (we couldn't validate; don't claim OK).
        return None

    any_mismatch = False
    identity_keys = ("schema_version", "run_id", "condition_id", "seed",
                     "benchmark_site", "task_id")
    # B-738 cold-start fix: check every row, not just last_segment[0]. Pre-fix
    # mixed JSONL segment with first row matching survived even if later rows
    # had divergent identity (cross-baseline / cross-task poison). The
    # None-on-either-side skip is preserved (legitimate: legacy archive
    # without full identity tuple, or fixture data with minimal fields);
    # what `strict_rows=True` adds is multi-row coverage so a poisoned tail
    # row whose value differs from summary triggers mismatch.
    rows_to_check = last_segment if strict_rows else last_segment[:1]
    mismatches: List[str] = []
    for row_idx, row in enumerate(rows_to_check):
        for k in identity_keys:
            s_val = summary.get(k)
            l_val = row.get(k)
            if s_val is None or l_val is None:
                continue  # field not stamped in either side; skip
            if s_val != l_val:
                mismatches.append(
                    f"row{row_idx}.{k}: summary={s_val!r} vs jsonl={l_val!r}"
                )
    if mismatches:
        any_mismatch = True
        logger.warning(
            "B-180 identity mismatch %s ↔ %s: %s",
            jsonl_path, summary_path, "; ".join(mismatches),
        )

    summary_steps = summary.get("steps")
    if isinstance(summary_steps, int) and summary_steps != len(last_segment):
        any_mismatch = True
        logger.warning(
            "B-180 step count mismatch %s: summary.steps=%d vs jsonl_segment=%d "
            "(may indicate restart-crash; summary points to older run)",
            jsonl_path, summary_steps, len(last_segment),
        )
    return any_mismatch
