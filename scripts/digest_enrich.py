#!/usr/bin/env python3
"""
Enrich digest JSONL with deterministic behavioral indicators extracted from
raw step logs and artifacts. Runs as a post-processor after glm_batch_digest.py.

Adds per-episode fields:
  - visual_observation_steps: steps where thought contains actual visual observation
  - visual_observation_ratio: visual_observation_steps / total_steps
  - task_keyword_echo_only: True if all "visual" words in thoughts are just task intent echoes
  - thought_references_marks: steps referencing element_id / mark ID in thought
  - thought_references_marks_ratio: thought_references_marks / total_steps
  - dom_has_description: whether any detail-page DOM contains product description text
  - avg_dom_length: average observation_dom.txt length across steps (proxy for info density)
  - detail_page_dom_lengths: list of DOM lengths on detail pages (url contains "page=item&id=")

Usage:
  # Directory mode: enrich all digest_*.jsonl files in-place
  python3 scripts/digest_enrich.py \
    --digest results/.../analysis/digest/ \
    --run-dir results/.../B1_3mode_classifieds_20260404_141103

  # Single file mode (backwards compat)
  python3 scripts/digest_enrich.py \
    --digest results/.../digest_dom.jsonl \
    --run-dir results/.../B1_3mode_classifieds_20260404_141103
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Visual observation detection
# ---------------------------------------------------------------------------

# Phrases that indicate the agent is describing what it SEES in the screenshot,
# not just echoing task keywords
_VISUAL_OBS_PHRASES = [
    r"\bi can see\b",
    r"\bthe image shows?\b",
    r"\bin the screenshot\b",
    r"\bthe thumbnail\b",
    r"\blooks? like\b",
    r"\bappears? to be\b",
    r"\bthe color is\b",
    r"\bit is (red|blue|white|black|green|yellow|silver|grey|gray|dark|orange)\b",
    r"\bis (red|blue|white|black|green|yellow|silver|grey|gray|dark|orange)\b",
    r"\bnot (red|blue|white|black|green|yellow|silver|grey|gray|dark|orange)\b",
    r"\bthe (bike|car|motorcycle|phone|item|xbox|listing|product) is (red|blue|white|black|silver|grey|gray|dark)\b",
    r"\bvisually\b",
    r"\bfrom the (image|photo|picture|screenshot)\b",
    r"\bin the (image|photo|picture)\b",
    r"\bimage (of|shows?|depicts?|displays?)\b",
]
_VISUAL_OBS_RE = re.compile("|".join(_VISUAL_OBS_PHRASES), re.IGNORECASE)

# Task intent color/visual keywords — used to detect echo-only
_TASK_VISUAL_KEYWORDS = re.compile(
    r"\b(red|blue|white|black|green|yellow|silver|grey|gray|dark|orange|purple|pink|brown|"
    r"color|image|photo|picture|thumbnail)\b",
    re.IGNORECASE,
)

# Mark / element_id references in thought
_MARK_REF_RE = re.compile(
    r"\b(element[_ ]?id|mark|ID|id=)\s*\d+|\[\d+\]|\bid\s+\d+\b",
    re.IGNORECASE,
)

# Detail page URL pattern for Classifieds
_DETAIL_PAGE_RE = re.compile(r"page=item&id=|/item/\d+", re.IGNORECASE)

# Product description indicators in DOM
_DESCRIPTION_INDICATORS = [
    "item_description",
    "listing-description",
    "description",
    "product-description",
    "ad-description",
]


def _extract_thoughts(steps_jsonl: Path) -> List[Dict[str, Any]]:
    """Extract (step_idx, thought, obs_url, action_type) from step log.

    Handles restart artifacts: if the watchdog/queue restarted a task,
    the JSONL may contain stale lines from earlier runs.  We detect
    step_idx resetting to 0 and only keep the last run.
    """
    try:
        from p79.experiment.io_utils import read_jsonl_dedup
        raw_steps = read_jsonl_dedup(steps_jsonl)
    except Exception:
        return []
    results = []
    for step in raw_steps:
        action = step.get("action", {})
        thought = ""
        if isinstance(action, dict):
            thought = str(action.get("thought", "") or "")
        elif isinstance(action, str):
            thought = action
        results.append({
            "step_idx": step.get("step_idx", len(results)),
            "thought": thought,
            "obs_url": str(step.get("obs_url", "") or ""),
            "action_type": str(step.get("action_type", "") or ""),
        })
    return results


def _has_visual_observation(thought: str) -> bool:
    """Does the thought contain actual visual observation (not just task keyword echo)?"""
    return bool(_VISUAL_OBS_RE.search(thought))


def _is_task_keyword_echo_only(thought: str, task_intent: str) -> bool:
    """Check if visual keywords in thought are ALL just echoes of the task intent."""
    thought_visual_words = set(
        w.lower() for w in _TASK_VISUAL_KEYWORDS.findall(thought)
    )
    if not thought_visual_words:
        return True  # no visual words at all
    intent_visual_words = set(
        w.lower() for w in _TASK_VISUAL_KEYWORDS.findall(task_intent)
    )
    # If all visual words in thought also appear in intent → echo only
    return thought_visual_words.issubset(intent_visual_words)


def _has_mark_reference(thought: str) -> bool:
    """Does the thought reference element IDs or mark IDs?"""
    return bool(_MARK_REF_RE.search(thought))


def _check_dom_description(artifact_dir: Path, steps: List[Dict]) -> Tuple[bool, List[int], float]:
    """
    Check if any detail-page DOM contains product description text.
    Returns (has_description, detail_page_dom_lengths, avg_dom_length).
    """
    dom_lengths: List[int] = []
    detail_dom_lengths: List[int] = []
    has_desc = False

    for s in steps:
        step_idx = s["step_idx"]
        dom_path = artifact_dir / f"step_{step_idx:03d}" / "observation_dom.txt"
        if not dom_path.exists():
            continue
        try:
            dom_text = dom_path.read_text(encoding="utf-8")
        except Exception:
            continue

        dom_len = len(dom_text)
        dom_lengths.append(dom_len)

        # Check if this is a detail page
        obs_url = s.get("obs_url", "")
        if _DETAIL_PAGE_RE.search(obs_url):
            detail_dom_lengths.append(dom_len)
            # Check for description section
            dom_lower = dom_text.lower()
            for indicator in _DESCRIPTION_INDICATORS:
                if indicator in dom_lower:
                    has_desc = True
                    break

    avg_dom = sum(dom_lengths) / len(dom_lengths) if dom_lengths else 0
    return has_desc, detail_dom_lengths, avg_dom


def enrich_one(
    record: Dict[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    """Add behavioral indicators to a single digest record."""
    condition_id = str(record.get("condition_id", "") or "")
    task_id = record.get("task_id")
    task_intent = str(record.get("task_intent", "") or "")
    obs_mode = str(record.get("observation_mode", "") or "").lower()

    if task_id is None:
        return record

    # Find step log
    ep_dir = run_dir / condition_id / "episodes"
    steps_files = list(ep_dir.glob(f"*task_{task_id}_steps_v2.jsonl"))
    if not steps_files:
        return record

    steps = _extract_thoughts(steps_files[0])
    if not steps:
        return record

    total = len(steps)

    # 1. Visual observation analysis
    visual_obs_steps = sum(1 for s in steps if _has_visual_observation(s["thought"]))

    # 2. Task keyword echo check
    all_echo = all(
        _is_task_keyword_echo_only(s["thought"], task_intent)
        for s in steps
        if _TASK_VISUAL_KEYWORDS.search(s["thought"])
    )
    # If no visual keywords at all, it's trivially echo-only
    has_any_visual_kw = any(_TASK_VISUAL_KEYWORDS.search(s["thought"]) for s in steps)

    # 3. Mark/element references
    mark_ref_steps = sum(1 for s in steps if _has_mark_reference(s["thought"]))

    # 4. DOM description check
    site = str(record.get("benchmark_site", "classifieds"))
    artifact_dir = run_dir / condition_id / "artifacts" / f"{site}_task_{task_id}"
    has_desc = False
    detail_dom_lengths: List[int] = []
    avg_dom = 0.0
    if artifact_dir.exists():
        has_desc, detail_dom_lengths, avg_dom = _check_dom_description(artifact_dir, steps)

    # Write enrichment fields
    record["_enrich"] = {
        "visual_observation_steps": visual_obs_steps,
        "visual_observation_ratio": round(visual_obs_steps / total, 3) if total else 0,
        "task_keyword_echo_only": all_echo if has_any_visual_kw else None,
        "thought_references_marks": mark_ref_steps,
        "thought_references_marks_ratio": round(mark_ref_steps / total, 3) if total else 0,
        "dom_has_description": has_desc,
        "detail_page_dom_lengths": detail_dom_lengths,
        "avg_dom_length": round(avg_dom),
        "total_steps_analyzed": total,
    }
    return record


def _enrich_one_file(digest_path: Path, run_dir: Path, output_path: Path) -> int:
    """Enrich a single digest JSONL file. Returns number of enriched records."""
    with digest_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"[enrich] Processing {len(records)} records from {digest_path.name}")

    enriched = 0
    for i, rec in enumerate(records):
        if rec.get("_dry_run") or rec.get("_glm_error"):
            continue
        records[i] = enrich_one(rec, run_dir)
        if "_enrich" in records[i]:
            enriched += 1

    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[enrich] Enriched {enriched}/{len(records)} → {output_path.name}")
    return enriched


def _print_summary(digest_path: Path) -> None:
    """Print summary stats for a single digest file."""
    with digest_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    enriched_recs = [r for r in records if "_enrich" in r]
    if not enriched_recs:
        return

    items = [r["_enrich"] for r in enriched_recs]
    mode = enriched_recs[0].get("observation_mode", "?")
    n = len(items)

    avg_vis = sum(e["visual_observation_ratio"] for e in items) / n
    avg_mark = sum(e["thought_references_marks_ratio"] for e in items) / n
    echo_only = sum(1 for e in items if e.get("task_keyword_echo_only") is True)
    echo_applicable = sum(1 for e in items if e.get("task_keyword_echo_only") is not None)
    no_desc = sum(1 for e in items if not e["dom_has_description"] and e["detail_page_dom_lengths"])
    has_detail = sum(1 for e in items if e["detail_page_dom_lengths"])

    print(f"\n[{mode}] {n} episodes:")
    print(f"  visual observation ratio: avg={avg_vis:.3f}")
    print(f"  mark reference ratio: avg={avg_mark:.3f}")
    if echo_applicable:
        print(f"  task keyword echo only: {echo_only}/{echo_applicable} ({100*echo_only/echo_applicable:.0f}%)")
    if has_detail:
        print(f"  detail pages missing description: {no_desc}/{has_detail} ({100*no_desc/has_detail:.0f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich digest JSONL with behavioral indicators")
    parser.add_argument("--digest", required=True, type=Path,
                        help="Path to digest file or directory containing digest_*.jsonl")
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory")
    parser.add_argument("--output", default=None, type=Path, help="Output path (default: overwrite in-place)")
    args = parser.parse_args()

    digest_path = args.digest.resolve()
    run_dir = args.run_dir.resolve()

    # Directory mode: process all digest_*.jsonl files
    if digest_path.is_dir():
        files = sorted(digest_path.glob("digest_*.jsonl"))
        if not files:
            print(f"ERROR: no digest_*.jsonl files found in {digest_path}")
            sys.exit(1)
        print(f"[enrich] Found {len(files)} digest files in {digest_path}")
        for f in files:
            out = (args.output / f.name) if args.output else f
            _enrich_one_file(f, run_dir, out)
        print("\n--- Summary ---")
        for f in files:
            out = (args.output / f.name) if args.output else f
            _print_summary(out)
        return

    # Single file mode (backwards compat)
    if not digest_path.exists():
        print(f"ERROR: digest file not found: {digest_path}")
        sys.exit(1)

    output_path = (args.output or digest_path).resolve()
    _enrich_one_file(digest_path, run_dir, output_path)
    _print_summary(output_path)


if __name__ == "__main__":
    main()
