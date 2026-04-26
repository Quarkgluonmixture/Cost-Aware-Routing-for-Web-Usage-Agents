#!/usr/bin/env python3
"""Generate an HTML gallery of annotated (or raw) screenshots for easy browsing.

Usage:
    # Single task
    python3 scripts/maintenance/generate_gallery.py --run-dir results/.../B1_xxx --task-id 17

    # Single condition, all tasks
    python3 scripts/maintenance/generate_gallery.py --run-dir results/.../B1_xxx --condition phase1_dom_router_0

    # Entire run
    python3 scripts/maintenance/generate_gallery.py --run-dir results/.../B1_xxx

Output: <run-dir>/gallery.html (open in browser)
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import html as html_mod
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_steps(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Read step JSONL with dedup."""
    try:
        from p79.experiment.io_utils import read_jsonl_dedup
        return read_jsonl_dedup(jsonl_path)
    except ImportError:
        lines = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return lines


def _img_to_data_uri(img_path: Path) -> Optional[str]:
    """Convert image to base64 data URI for embedding in HTML."""
    if not img_path.exists():
        return None
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _img_to_relative(img_path: Path, gallery_path: Path) -> Optional[str]:
    """Get relative path from gallery HTML to image."""
    if not img_path.exists():
        return None
    import os
    return os.path.relpath(str(img_path), str(gallery_path.parent))


# ---------------------------------------------------------------------------
# Action summary (adapted from annotate_screenshots._build_action_line)
# ---------------------------------------------------------------------------

def _build_action_summary(step: Dict[str, Any]) -> str:
    """Build a compact action summary string from a step record."""
    action = step.get("action", {})
    if isinstance(action, str):
        try:
            action = json.loads(action)
        except (json.JSONDecodeError, TypeError):
            return action[:120] if action else ""

    action_type = str(
        step.get("action_type") or action.get("action_type") or "unknown"
    ).lower()

    parts = [action_type.upper()]
    eid = action.get("element_id")
    coord = action.get("coordinate")

    if eid is not None:
        parts.append(f"[{eid}]")

    if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
        cx, cy = float(coord[0]), float(coord[1])
        # Flag mixed-format coords (one normalized, one pixel) with asterisk
        mixed = cx <= 1.0 < cy or cy <= 1.0 < cx
        parts.append(f"@({cx:.2f},{cy:.2f})" + ("*" if mixed else ""))

    if action_type == "type":
        text = str(action.get("text", "")).replace("\n", "\\n")
        if len(text) > 40:
            text = text[:37] + "..."
        parts.append(f"'{text}'")
    elif action_type == "finish":
        answer = str(action.get("answer", ""))
        if len(answer) > 40:
            answer = answer[:37] + "..."
        parts.append(f"ans='{answer}'")
    elif action_type == "scroll":
        delta = action.get("delta", [0, 0])
        if isinstance(delta, (list, tuple)) and len(delta) >= 2:
            if abs(delta[1]) >= abs(delta[0]):
                parts.append("DOWN" if delta[1] > 0 else "UP")
            else:
                parts.append("RIGHT" if delta[0] > 0 else "LEFT")
    elif action_type == "tab_focus":
        parts.append(f"tab={action.get('page_number', '?')}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Reason diagnostics CSV
# ---------------------------------------------------------------------------

def _load_reason_rows(source_run_dirs: List[Path]) -> Dict[str, Dict[str, Any]]:
    """Load reason diagnostics CSV for each source run dir.

    Returns dict keyed by ``{condition_id}__{site}_task_{tid}`` with fields:
    reason_bucket, task_type, adjusted_success, fp_reason.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for run_dir in source_run_dirs:
        csv_path = run_dir / "analysis" / "reason_diagnostics" / "episode_reason_rows.csv"
        if not csv_path.exists():
            continue
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cond = row.get("condition_id", "")
                    site = row.get("site", "")
                    tid = row.get("task_id", "")
                    if not (cond and site and tid):
                        continue
                    key = f"{cond}__{site}_task_{tid}"
                    adj = row.get("adjusted_success", "")
                    result[key] = {
                        "reason_bucket": row.get("reason_bucket", ""),
                        "task_type": row.get("task_type", ""),
                        "adjusted_success": (
                            True if adj == "True" else
                            False if adj == "False" else None
                        ),
                        "fp_reason": row.get("fp_reason", ""),
                    }
        except Exception:
            continue
    return result


# ---------------------------------------------------------------------------
# Reference answer formatting
# ---------------------------------------------------------------------------

def _format_reference_answer(eval_cfg: Dict[str, Any]) -> str:
    """Extract a human-readable reference answer string from task eval config."""
    if not eval_cfg:
        return ""
    ref = eval_cfg.get("reference_answers")
    ref_url = eval_cfg.get("reference_url", "")

    if isinstance(ref, dict):
        if "fuzzy_match" in ref:
            return str(ref["fuzzy_match"])
        if "exact_match" in ref:
            return str(ref["exact_match"])
        if "must_include" in ref:
            items = ref["must_include"]
            if isinstance(items, list):
                return ", ".join(str(x) for x in items)
            return str(items)

    if ref_url:
        return ref_url

    # program_html: summarize required_contents if available
    ph = eval_cfg.get("program_html", [])
    if isinstance(ph, list):
        parts = []
        for entry in ph:
            rc = entry.get("required_contents", {}) if isinstance(entry, dict) else {}
            # Check exact_match first, then fuzzy_match, then must_include
            em = rc.get("exact_match")
            if em:
                parts.append(str(em))
                continue
            fm = rc.get("fuzzy_match")
            if fm:
                parts.append(str(fm))
                continue
            mi = rc.get("must_include", [])
            if isinstance(mi, list) and mi:
                parts.extend(str(x) for x in mi)
        if parts:
            return ", ".join(parts)

    return ""


# ---------------------------------------------------------------------------
# Condition metadata
# ---------------------------------------------------------------------------

def _load_condition_labels(source_run_dirs: List[Path]) -> Dict[str, Dict[str, str]]:
    """Load condition labels and observation modes from condition_meta.json files."""
    labels: Dict[str, Dict[str, str]] = {}
    for source_run_dir in source_run_dirs:
        for cond_dir in sorted(source_run_dir.iterdir()):
            if not cond_dir.is_dir() or cond_dir.name in ("analysis", ".git", "_vwa"):
                continue
            default = {"label": cond_dir.name, "observation_mode": "unknown"}
            meta_path = cond_dir / "condition_meta.json"
            candidate = default
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    candidate = {
                        "label": meta.get("label", cond_dir.name),
                        "observation_mode": meta.get("observation_mode", "unknown"),
                    }
                except Exception:
                    candidate = default

            # Prefer non-unknown observation mode if discovered later.
            existing = labels.get(cond_dir.name)
            if existing is None or (
                existing.get("observation_mode") == "unknown"
                and candidate.get("observation_mode") != "unknown"
            ):
                labels[cond_dir.name] = candidate
    return labels


_KNOWN_SITE_ORDER = {
    "classifieds": 0,
    "reddit": 1,
    "shopping": 2,
    "shopping_admin": 3,
    "wikipedia": 4,
    # Cross-benchmark gallery: vwa sites first, then wa sites
    "vwa:classifieds": 10,
    "vwa:reddit": 11,
    "vwa:shopping": 12,
    "wa:shopping": 20,
    "wa:shopping_admin": 21,
    "wa:reddit": 22,
}
_RUN_FAMILY_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<site>classifieds|reddit|shopping_admin|shopping|wikipedia)_(?P<stamp>\d{8}(?:_\d{6})?)$"
)


def _parse_run_family(run_name: str) -> Optional[Dict[str, str]]:
    """Parse run name into {prefix, site, stamp} when it matches known naming."""
    m = _RUN_FAMILY_RE.match(run_name)
    if not m:
        return None
    return m.groupdict()


def _has_episode_data(run_dir: Path) -> bool:
    """Quick check: does this run dir contain at least one completed episode summary?"""
    for cond_dir in run_dir.iterdir():
        if not cond_dir.is_dir() or cond_dir.name in ("analysis", ".git", "_vwa"):
            continue
        eps_dir = cond_dir / "episodes"
        if not eps_dir.is_dir():
            continue
        if any(eps_dir.glob("*_summary_v2.json")):
            return True
    return False


def _discover_source_runs(run_dir: Path) -> tuple[List[Path], str]:
    """Find source run dirs for gallery and a concise display title."""
    family = _parse_run_family(run_dir.name)
    if not family:
        return [run_dir], run_dir.name

    prefix = family["prefix"]
    siblings: List[Path] = []
    for cand in run_dir.parent.iterdir():
        if not cand.is_dir():
            continue
        c_family = _parse_run_family(cand.name)
        if not c_family:
            continue
        if c_family["prefix"] != prefix:
            continue
        if _has_episode_data(cand):
            siblings.append(cand)

    if not siblings:
        return [run_dir], prefix

    siblings.sort(
        key=lambda p: (
            _KNOWN_SITE_ORDER.get((_parse_run_family(p.name) or {}).get("site", ""), 999),
            p.name,
        )
    )
    return siblings, prefix


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def _build_groups(
    episodes: List[Dict[str, Any]],
    condition_labels: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Group episodes by (site, condition) and compute per-group stats."""
    group_map: Dict[tuple, Dict[str, Any]] = OrderedDict()
    for ep in episodes:
        key = (ep["site"], ep["condition"])
        if key not in group_map:
            cl = condition_labels.get(ep["condition"], {})
            group_map[key] = {
                "site": ep["site"],
                "condition": ep["condition"],
                "condition_label": cl.get("label", ep["condition"]),
                "observation_mode": cl.get("observation_mode", "unknown"),
                "episodes": [],
            }
        group_map[key]["episodes"].append({
            "key": ep["key"],
            "task_id": ep["task_id"],
            "label": ep["label"],
            "intent": ep.get("intent", ""),
            "intent_image": ep.get("intent_image", ""),
            "success": ep["success"],
            "score": ep["score"],
            "total_steps": ep["total_steps"],
            "steps": ep["steps"],
            "eval_type": ep.get("eval_type", ""),
            "reference_answer": ep.get("reference_answer", ""),
            "final_answer": ep.get("final_answer", ""),
            "task_type": ep.get("task_type", ""),
            "reason_bucket": ep.get("reason_bucket", ""),
            "adjusted_success": ep.get("adjusted_success"),
            "fp_reason": ep.get("fp_reason", ""),
        })

    groups = []
    for group in group_map.values():
        total = len(group["episodes"])
        success = sum(1 for e in group["episodes"] if e.get("success") is True)
        fail = sum(1 for e in group["episodes"] if e.get("success") is False)
        group["stats"] = {
            "total": total,
            "success": success,
            "fail": fail,
            "success_rate": round(success / total, 3) if total > 0 else 0,
        }
        groups.append(group)
    return groups


# ---------------------------------------------------------------------------
# Task intents from VWA config
# ---------------------------------------------------------------------------

_CONFIG_FILES_BASE = Path(__file__).resolve().parent.parent.parent / "external" / "visualwebarena" / "config_files"

def _load_task_intents() -> Dict[str, Dict[str, str]]:
    """Load task intents + image paths from VWA and WA config files.

    Returns {'{site}_task_{id}': {'intent': ..., 'image': ...}}.
    """
    info: Dict[str, Dict[str, str]] = {}
    vwa_root = _CONFIG_FILES_BASE.parent  # external/visualwebarena
    # Scan both vwa/ and wa/ config directories
    for subdir_name in ("vwa", "wa"):
        config_base = _CONFIG_FILES_BASE / subdir_name
        if not config_base.exists():
            continue
        for site_dir in config_base.iterdir():
            if not site_dir.is_dir() or not site_dir.name.startswith("test_"):
                continue
            site = site_dir.name.replace("test_", "")
            for cfg_path in site_dir.glob("*.json"):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    tid = cfg.get("task_id")
                    intent = cfg.get("intent", "")
                    if tid is not None and intent:
                        entry: Dict[str, str] = {"intent": intent}
                        img_rel = cfg.get("image", "")
                        # image field may be a list (multi-image tasks); take first element
                        if isinstance(img_rel, list):
                            img_rel = img_rel[0] if img_rel else ""
                        if img_rel:
                            img_abs = (vwa_root / img_rel).resolve()
                            if img_abs.exists():
                                entry["image"] = str(img_abs)
                        # Eval metadata
                        eval_cfg = cfg.get("eval", {})
                        if isinstance(eval_cfg, dict):
                            et = eval_cfg.get("eval_types", [])
                            entry["eval_type"] = et[0] if isinstance(et, list) and et else ""
                            entry["reference_answer"] = _format_reference_answer(eval_cfg)
                        info[f"{site}_task_{tid}"] = entry
                except Exception:
                    continue
        # Also load from raw.json files (WA per-site split files)
        for raw_path in config_base.glob("test_*.raw.json"):
            site = raw_path.stem.replace("test_", "").replace(".raw", "")
            try:
                with open(raw_path, "r", encoding="utf-8") as f:
                    tasks = json.load(f)
                if not isinstance(tasks, list):
                    continue
                for cfg in tasks:
                    tid = cfg.get("task_id")
                    intent = cfg.get("intent", "")
                    if tid is None or not intent:
                        continue
                    key = f"{site}_task_{tid}"
                    raw_entry: Dict[str, str] = {"intent": intent}
                    eval_cfg = cfg.get("eval", {})
                    if isinstance(eval_cfg, dict):
                        et = eval_cfg.get("eval_types", [])
                        raw_entry["eval_type"] = et[0] if isinstance(et, list) and et else ""
                        raw_entry["reference_answer"] = _format_reference_answer(eval_cfg)
                    if key not in info:
                        info[key] = raw_entry
                    # WA tasks also stored under wa: prefix to avoid
                    # VWA/WA collision on shared site+task_id combos
                    if subdir_name == "wa":
                        info[f"wa:{site}_task_{tid}"] = raw_entry
            except Exception:
                continue
    return info


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def _collect_episodes(
    source_run_dirs: List[Path],
    condition_filter: Optional[str],
    task_id_filter: Optional[int],
    gallery_path: Path,
    embed: bool,
    *,
    prefix_site_with_benchmark: bool = False,
) -> List[Dict[str, Any]]:
    """Collect episodes with their steps and image sources.

    If *prefix_site_with_benchmark* is True, site names are prefixed with
    ``vwa:`` or ``wa:`` based on the source_run_dir path.  Used for
    cross-benchmark aggregate galleries.
    """
    intents = _load_task_intents()
    reason_rows = _load_reason_rows(source_run_dirs)
    episodes = []
    for source_run_dir in source_run_dirs:
        condition_dirs = sorted(source_run_dir.iterdir())
        for cond_dir in condition_dirs:
            if not cond_dir.is_dir() or cond_dir.name in ("analysis", ".git", "_vwa"):
                continue
            if condition_filter and cond_dir.name != condition_filter:
                continue
            episodes_dir = cond_dir / "episodes"
            artifacts_dir = cond_dir / "artifacts"
            if not episodes_dir.exists():
                continue

            # Determine benchmark prefix for this run dir (once per run dir)
            _is_wa_run = any(p == "webarena" for p in source_run_dir.parts)
            _bm_prefix = "wa" if _is_wa_run else "vwa"

            for jsonl_path in sorted(episodes_dir.glob("*_steps_v2.jsonl")):
                stem = jsonl_path.stem.replace("_steps_v2", "")
                parts = stem.rsplit("_task_", 1)
                if len(parts) != 2:
                    continue
                raw_site = parts[0]  # original site name (used for file paths)
                site = f"{_bm_prefix}:{raw_site}" if prefix_site_with_benchmark else raw_site
                try:
                    task_id = int(parts[1])
                except ValueError:
                    continue
                if task_id_filter is not None and task_id != task_id_filter:
                    continue

                steps = _read_steps(jsonl_path)
                if not steps:
                    continue

                # Read summary — skip orphan steps files that have no summary yet
                # File paths always use raw_site (without benchmark prefix)
                summary_path = episodes_dir / f"{raw_site}_task_{task_id}_summary_v2.json"
                if not summary_path.exists():
                    continue
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                except Exception:
                    summary = None

                # Collect steps
                task_artifact_dir = artifacts_dir / f"{raw_site}_task_{task_id}"
                step_data = []
                for step in steps:
                    step_idx = step.get("step_idx", len(step_data))
                    step_dir = task_artifact_dir / f"step_{step_idx:03d}"
                    annotated = step_dir / "screenshot_annotated.png"
                    raw = step_dir / "screenshot.png"
                    img_path = annotated if annotated.exists() else raw

                    if embed:
                        img_src = _img_to_data_uri(img_path)
                    else:
                        img_src = _img_to_relative(img_path, gallery_path)

                    # Extract thought from action dict (primary) or top-level (fallback)
                    action = step.get("action", {})
                    if isinstance(action, dict):
                        thought = str(action.get("thought", "") or "").strip()
                    else:
                        thought = ""
                    if not thought:
                        thought = str(step.get("thought", "") or "").strip()

                    step_data.append({
                        "step_idx": step_idx,
                        "action_summary": _build_action_summary(step),
                        "thought": thought[:200],
                        "reward": step.get("reward"),
                        "img_path": img_src,
                    })

                ep_key = f"{source_run_dir.name}__{cond_dir.name}__{raw_site}_task_{task_id}"
                label = f"{site}_task_{task_id}"
                # Prefer wa:-prefixed intent for WA runs to avoid VWA/WA collision
                raw_label = f"{raw_site}_task_{task_id}"
                task_info = intents.get(f"wa:{raw_label}", {}) if _is_wa_run else {}
                if not task_info:
                    task_info = intents.get(raw_label, {})
                intent_text = task_info.get("intent", "") if isinstance(task_info, dict) else str(task_info)
                intent_img_abs = task_info.get("image", "") if isinstance(task_info, dict) else ""
                # Convert intent image to relative path (or base64 if --embed)
                intent_img_src = ""
                if intent_img_abs:
                    if embed:
                        intent_img_src = _img_to_data_uri(Path(intent_img_abs)) or ""
                    else:
                        # Use _vwa symlink so path stays inside HTTP server root
                        vwa_root = _CONFIG_FILES_BASE.resolve().parent.parent
                        img_abs = Path(intent_img_abs)
                        try:
                            rel_in_vwa = img_abs.resolve().relative_to(vwa_root.resolve())
                            intent_img_src = f"_vwa/{rel_in_vwa}"
                        except ValueError:
                            intent_img_src = _img_to_relative(
                                img_abs, gallery_path
                            ) or ""
                # Eval metadata from task config
                eval_type = task_info.get("eval_type", "") if isinstance(task_info, dict) else ""
                reference_answer = task_info.get("reference_answer", "") if isinstance(task_info, dict) else ""

                # Extract agent's final answer from last FINISH action
                final_answer = ""
                for step in reversed(steps):
                    act = step.get("action", {})
                    if isinstance(act, str):
                        try:
                            act = json.loads(act)
                        except (json.JSONDecodeError, TypeError):
                            continue
                    if isinstance(act, dict):
                        at = str(act.get("action_type", "") or step.get("action_type", "")).lower()
                        if at == "finish":
                            final_answer = str(act.get("answer", ""))
                            break

                # Reason diagnostics from CSV
                reason_key = f"{cond_dir.name}__{raw_site}_task_{task_id}"
                reason_info = reason_rows.get(reason_key, {})

                episodes.append({
                    "key": ep_key,
                    "run_id": source_run_dir.name,
                    "condition": cond_dir.name,
                    "site": site,
                    "task_id": task_id,
                    "label": label,
                    "intent": intent_text,
                    "intent_image": intent_img_src,
                    "steps": step_data,
                    "success": summary.get("success") if summary else None,
                    "score": summary.get("score") if summary else None,
                    "total_steps": len(step_data),
                    "eval_type": eval_type,
                    "reference_answer": reference_answer,
                    "final_answer": final_answer,
                    "task_type": reason_info.get("task_type", ""),
                    "reason_bucket": reason_info.get("reason_bucket", ""),
                    "adjusted_success": reason_info.get("adjusted_success"),
                    "fp_reason": reason_info.get("fp_reason", ""),
                })

    episodes.sort(
        key=lambda e: (
            _KNOWN_SITE_ORDER.get(e["site"], 999),
            e["condition"],
            e["task_id"],
            e["run_id"],
        )
    )
    return episodes


# ---------------------------------------------------------------------------
# HTML template (v2 — dual-view architecture)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE_V2 = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>Episode Gallery &mdash; {title}</title>
<style>
*{{ margin:0; padding:0; box-sizing:border-box; }}
body{{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#1a1a2e; color:#e0e0e0;
}}

/* ========== Home View ========== */
#home-view{{ padding:20px 30px; max-width:1400px; margin:0 auto; }}
.home-header{{ display:flex; align-items:baseline; gap:16px; margin-bottom:18px; }}
.home-header h1{{ font-size:22px; }}
.home-header .gen-time{{ font-size:12px; color:#666; }}

.group-card{{
  margin-bottom:14px; background:#16213e; border-radius:8px;
  border:1px solid #2a2a4a; overflow:hidden;
  box-shadow:0 2px 8px rgba(0,0,0,.3);
}}
.group-header{{
  padding:12px 16px; display:flex; align-items:center; gap:12px;
  cursor:pointer; user-select:none; transition:background .15s;
}}
.group-header:hover{{ background:#1a3a5c; }}
.group-header h3{{ font-size:15px; }}
.site-badge{{
  padding:2px 8px; border-radius:4px; font-size:11px;
  font-weight:700; text-transform:uppercase;
}}
.site-shopping{{ background:#1b5e20; color:#a5d6a7; }}
.site-reddit{{ background:#bf360c; color:#ffab91; }}
.site-wikipedia{{ background:#0d47a1; color:#90caf9; }}
.site-classifieds{{ background:#4a148c; color:#ce93d8; }}
.site-shopping_admin{{ background:#2e7d32; color:#c8e6c9; }}
.site-vwa-classifieds{{ background:#4a148c; color:#ce93d8; }}
.site-vwa-reddit{{ background:#bf360c; color:#ffab91; }}
.site-vwa-shopping{{ background:#1b5e20; color:#a5d6a7; }}
.site-wa-shopping{{ background:#33691e; color:#dce775; }}
.site-wa-shopping_admin{{ background:#558b2f; color:#e6ee9c; }}
.site-wa-reddit{{ background:#d84315; color:#ffcc80; }}

.stats-bar{{
  display:flex; align-items:center; gap:10px;
  margin-left:auto; font-size:13px; color:#aaa;
}}
.stats-bar .s{{ white-space:nowrap; }}
.stats-bar .s.ok{{ color:#4caf50; }}
.stats-bar .s.no{{ color:#f44336; }}
.progress-bar{{ width:80px; height:6px; background:#333; border-radius:3px; overflow:hidden; }}
.progress-fill{{ height:100%; background:linear-gradient(90deg,#2e7d32,#66bb6a); border-radius:3px; }}
.group-toggle{{ font-size:12px; color:#888; min-width:14px; text-align:center; }}

.ep-table{{ display:none; width:100%; border-collapse:collapse; }}
.ep-table.expanded{{ display:table; }}
.ep-table th{{
  text-align:left; padding:5px 12px; font-size:12px; color:#666;
  border-bottom:1px solid #333; background:#0f1a30;
}}
.ep-table td{{ padding:5px 12px; font-size:13px; border-bottom:1px solid #1e1e3e; }}
.ep-table tr.ep-row{{ cursor:pointer; transition:background .2s,border-color .2s; border-left:3px solid transparent; }}
.ep-table tr.ep-row:hover{{ background:#1a3a5c; border-left-color:#64b5f6; }}
.ep-table tr.ep-row.last-viewed{{ background:#2a3a1a; border-left-color:#ffa726; }}
.ep-table tr.ep-row.last-viewed:hover{{ background:#35451f; border-left-color:#ffa726; }}

.badge{{
  display:inline-block; padding:3px 10px; border-radius:10px;
  font-size:11px; font-weight:600; text-shadow:0 1px 2px rgba(0,0,0,.3);
}}
.badge.success{{ background:#1b5e20; color:#a5d6a7; }}
.badge.fail{{ background:#b71c1c; color:#ef9a9a; }}
.badge.unknown{{ background:#333; color:#888; }}

/* ========== Episode View ========== */
#episode-view{{ display:none; }}

.ep-top-bar{{
  position:sticky; top:0; z-index:100;
  background:#16213e; padding:2px 12px;
  border-bottom:1px solid #333;
}}
.ep-row1{{
  display:flex; align-items:center; gap:4px; flex-wrap:nowrap;
}}
.ep-row2{{
  display:flex; align-items:center; gap:6px; margin-top:2px;
  font-size:13px; color:#ccc; line-height:1.3;
  height:20px; overflow:hidden; white-space:nowrap;
}}
.ep-row2 .ep-ref{{ color:#80cbc4; flex-shrink:0; }}
.ep-row2 .ep-ans{{ color:#ef9a9a; flex-shrink:0; }}
.ep-row2 .ep-ans.match{{ color:#a5d6a7; }}
.ep-ref-val,.ep-ans-val{{
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  display:inline-block; vertical-align:middle; cursor:help;
  flex-shrink:0;
}}
.ep-ref-val{{ max-width:250px; }}
.ep-ans-val{{ max-width:200px; }}
.ep-sep{{ color:#555; margin:0 2px; flex-shrink:0; }}
.ep-intent-span{{
  flex:1 1 0; min-width:0; overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap; color:#e0e0e0;
}}
.back-btn{{
  background:none; border:1px solid #555; color:#ccc;
  padding:3px 8px; border-radius:4px; cursor:pointer; font-size:12px;
  flex-shrink:0;
}}
.back-btn:hover{{ background:#1a3a5c; }}
.ep-title{{ font-size:14px; font-weight:600; white-space:nowrap; flex-shrink:0; }}
.nav-btn{{
  background:#1a3a5c; border:none; color:#ccc;
  padding:3px 8px; border-radius:4px; cursor:pointer; font-size:12px;
  flex-shrink:0;
}}
.nav-btn:hover{{ background:#2a5a8c; }}
.nav-btn:disabled{{ opacity:.3; cursor:default; }}
.ep-spacer{{ flex:1; }}

.step-dot{{
  width:20px; height:20px; border-radius:3px;
  border:1px solid #444; display:flex; align-items:center;
  justify-content:center; font-size:9px; cursor:pointer;
  transition:all .12s; color:#aaa;
}}
.step-dot:hover{{ background:#1a3a5c; border-color:#64b5f6; }}
.step-dot.active{{ background:#1a3a5c; border-color:#64b5f6; color:#fff; }}
.step-dot.reward{{ border-color:#4caf50; background:rgba(76,175,80,.15); }}
.step-dot.reward.active{{ border-color:#66bb6a; background:#1a3a5c; }}

.steps-area{{ padding:0 16px 80px; margin:0 auto; }}

.step-card{{
  margin:2px 0 16px; background:#16213e; border-radius:6px;
  overflow:hidden; border:1px solid #2a2a4a;
  scroll-margin-top:48px;
  box-shadow:0 1px 4px rgba(0,0,0,.2);
}}
.step-card img{{
  width:100%; max-height:calc(100vh - 48px); display:block; cursor:pointer;
  object-fit:contain; background:#0a0a1a;
}}
img.zoomed{{
  position:fixed; top:0; left:0; width:100vw; height:100vh;
  object-fit:contain; z-index:1000; background:rgba(0,0,0,.92);
  cursor:zoom-out;
}}
.no-img{{ padding:40px; color:#555; text-align:center; }}
.kb-hints{{ text-align:center; padding:16px 0 8px; font-size:12px; color:#555; }}
.intent-has-img{{ cursor:help; border-bottom:1px dashed #555; }}

/* ---- eval & reason badges ---- */
.eval-badge{{
  display:inline-block; padding:1px 6px; border-radius:8px;
  font-size:10px; font-weight:600; background:#333; color:#aaa;
  vertical-align:middle; margin-right:4px;
}}
.eval-badge.string_match{{ background:#1a237e; color:#9fa8da; }}
.eval-badge.url_match{{ background:#004d40; color:#80cbc4; }}
.eval-badge.program_html{{ background:#4e342e; color:#bcaaa4; }}

.reason-badge{{
  display:inline-block; padding:1px 6px; border-radius:8px;
  font-size:10px; font-weight:600; vertical-align:middle;
}}
.reason-success{{ background:#1b5e20; color:#a5d6a7; }}
.reason-early{{ background:#e65100; color:#ffcc80; }}
.reason-mismatch{{ background:#f9a825; color:#333; }}
.reason-stuck{{ background:#b71c1c; color:#ef9a9a; }}
.reason-maxsteps{{ background:#4a148c; color:#ce93d8; }}
.reason-default{{ background:#424242; color:#bbb; }}

.ref-text{{ font-size:11px; color:#888; max-width:180px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; display:inline-block; vertical-align:middle; }}
.fp-indicator{{ color:#ff9800; font-size:10px; margin-left:2px; }}
#intent-tooltip{{
  display:none; position:fixed; z-index:500; padding:4px;
  background:#0f1a30; border:1px solid #2a2a4a; border-radius:6px;
  box-shadow:0 4px 16px rgba(0,0,0,.6); pointer-events:none;
}}
#intent-tooltip img{{ max-width:400px; max-height:320px; border-radius:4px; display:block; }}
</style>
</head>
<body>

<div id="home-view"></div>
<div id="episode-view"></div>
<div id="intent-tooltip"><img></div>

<script type="application/json" id="gallery-data">
{data_json}
</script>

<script>
(function(){{
'use strict';

var D=JSON.parse(document.getElementById('gallery-data').textContent);
var GROUPS=D.groups, ORDER=D.episode_order, IDX=D.episode_index;
var SKEY='gallery_v2_'+(D.state_key||D.title);

/* ---- state ---- */
var S={{view:'home',epKey:null,step:0,scrollY:0,eg:{{}}}};
function save(){{ try{{localStorage.setItem(SKEY,JSON.stringify(S));}}catch(e){{}} }}
function load(){{ try{{var s=localStorage.getItem(SKEY);if(s)Object.assign(S,JSON.parse(s));}}catch(e){{}} }}

/* ---- helpers ---- */
function ep(k){{ var l=IDX[k]; return l?GROUPS[l[0]].episodes[l[1]]:null; }}
function oi(k){{ return ORDER.indexOf(k); }}
function esc(s){{ return s?s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'):''; }}
function escA(s){{ return s?s.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/'/g,'&#39;'):''; }}
function hlAct(s){{
  if(!s) return '';
  var sp=s.indexOf(' ');
  if(sp<0) return '<span class="kw">'+esc(s)+'</span>';
  return '<span class="kw">'+esc(s.substring(0,sp))+'</span> '+esc(s.substring(sp+1));
}}

function reasonClass(r){{
  if(!r) return 'reason-default';
  if(r==='success') return 'reason-success';
  if(r.indexOf('early_finish')>=0) return 'reason-early';
  if(r.indexOf('eval_mismatch')>=0) return 'reason-mismatch';
  if(r.indexOf('no_progress')>=0||r.indexOf('stuck')>=0||r.indexOf('incomplete')>=0) return 'reason-stuck';
  if(r.indexOf('max_steps')>=0) return 'reason-maxsteps';
  return 'reason-default';
}}
function shortReason(r){{
  if(!r) return '';
  return r.replace(/^fail_/,'').replace(/_/g,' ');
}}

var $h=document.getElementById('home-view');
var $e=document.getElementById('episode-view');

/* ======== Home View ======== */
function renderHome(){{
  var h='<div class="home-header"><h1>'+esc(D.title)+'</h1>'
    +'<span class="gen-time">'+esc(D.generated_at)+'</span>'
    +'<span class="gen-time">'+ORDER.length+' episodes</span></div>';
  GROUPS.forEach(function(g,gi){{
    var sr=(g.stats.success_rate*100).toFixed(1);
    var sc='site-'+g.site.replace(/:/g,'-');
    var lgi=S.lastEp&&IDX[S.lastEp]?IDX[S.lastEp][0]:-1;
    var ex=S.eg[gi]||(gi===lgi);
    h+='<div class="group-card">'
      +'<div class="group-header" data-gi="'+gi+'">'
      +'<span class="site-badge '+sc+'">'+esc(g.site)+'</span>'
      +'<h3>'+esc(g.condition_label)+'</h3>'
      +'<span style="color:#666;font-size:12px">'+esc(g.observation_mode)+'</span>'
      +'<div class="stats-bar">'
      +'<span class="s">'+g.stats.total+'</span>'
      +'<span class="s ok">'+g.stats.success+' pass</span>'
      +'<span class="s no">'+g.stats.fail+' fail</span>'
      +'<div class="progress-bar"><div class="progress-fill" style="width:'+sr+'%"></div></div>'
      +'<span class="s">'+sr+'%</span>'
      +'</div>'
      +'<span class="group-toggle">'+(ex?'&#9660;':'&#9654;')+'</span>'
      +'</div>';
    h+='<table class="ep-table'+(ex?' expanded':'')+'" data-gi="'+gi+'">'
      +'<thead><tr><th style="width:130px">Task</th><th>Intent</th><th style="width:70px">Status</th><th style="width:50px">Steps</th><th style="width:50px">Score</th><th style="width:200px">Eval / Ref</th><th style="width:120px">Reason</th></tr></thead><tbody>';
    g.episodes.forEach(function(e){{
      var c=e.success===true?'success':e.success===false?'fail':'unknown';
      var sl=e.success===true?'PASS':e.success===false?'FAIL':'&mdash;';
      var sc2=e.score!=null?e.score.toFixed(2):'&mdash;';
      var it=e.intent||'';
      var it60=it.length>60?it.substring(0,57)+'...':it;
      var lv=S.lastEp===e.key?' last-viewed':'';
      var etb=e.eval_type?'<span class="eval-badge '+esc(e.eval_type)+'">'+esc(e.eval_type)+'</span>':'';
      var ra=e.reference_answer||'';
      var ra30=ra.length>30?ra.substring(0,27)+'...':ra;
      var rb=e.reason_bucket||'';
      var fpLabel=e.fp_reason==='na_fp'?'N-FP':e.fp_reason==='eval_fp'?'E-FP':e.fp_reason?'FP':'';
      var fpTag=fpLabel?'<span class="fp-indicator" title="'+escA(e.fp_reason)+'">'+fpLabel+'</span>':'';
      h+='<tr class="ep-row'+lv+'" data-key="'+escA(e.key)+'">'
        +'<td>'+esc(e.label)+'</td>'
        +'<td style="color:#bbb;font-size:12px" title="'+escA(it)+'">'+(e.intent_image?'<span class="intent-has-img" data-img="'+escA(e.intent_image)+'">'+esc(it60)+'</span>':esc(it60))+'</td>'
        +'<td><span class="badge '+c+'">'+sl+'</span>'+fpTag+'</td>'
        +'<td>'+e.total_steps+'</td>'
        +'<td>'+sc2+'</td>'
        +'<td>'+etb+(ra?'<span class="ref-text" title="'+escA(ra)+'">'+esc(ra30)+'</span>':'')+'</td>'
        +'<td>'+(rb?'<span class="reason-badge '+reasonClass(rb)+'" title="'+escA(rb)+'">'+esc(shortReason(rb))+'</span>':'')+'</td>'
        +'</tr>';
    }});
    h+='</tbody></table></div>';
  }});
  h+='<div class="kb-hints">&#8592; &#8594; switch episode &middot; &#8593; &#8595; switch step &middot; Esc back &middot; click screenshot to zoom</div>';
  $h.innerHTML=h;

  /* bind group toggles */
  $h.querySelectorAll('.group-header').forEach(function(hdr){{
    hdr.addEventListener('click',function(){{
      var gi=parseInt(hdr.dataset.gi);
      var t=$h.querySelector('.ep-table[data-gi="'+gi+'"]');
      var tg=hdr.querySelector('.group-toggle');
      var x=t.classList.toggle('expanded');
      tg.innerHTML=x?'&#9660;':'&#9654;';
      S.eg[gi]=x||undefined;
      if(!x) delete S.eg[gi];
      save();
    }});
  }});
  /* bind episode rows */
  $h.querySelectorAll('.ep-row').forEach(function(r){{
    r.addEventListener('click',function(){{ goEp(r.dataset.key,0); }});
  }});
}}

/* ======== Episode View ======== */
function renderEp(k){{
  $tt.style.display='none';
  var e=ep(k); if(!e) return;
  var o=oi(k), hp=o>0, hn=o<ORDER.length-1;
  var c=e.success===true?'success':e.success===false?'fail':'unknown';
  var sl=e.success===true?'PASS':e.success===false?'FAIL':'&mdash;';
  var gm=GROUPS[IDX[k][0]].observation_mode||'';
  var mc=gm==='dom'?'#5b9bd5':gm==='som'?'#ed7d31':'#70ad47';
  /* ---- Row 1: task label + badges + nav ---- */
  var h='<div class="ep-top-bar">'
    +'<div class="ep-row1">'
    +'<button class="back-btn" id="eb">&#8592; Home</button>'
    +'<span class="ep-title">'+esc(e.label)+'</span>'
    +'<span style="background:'+mc+';color:#fff;padding:1px 8px;border-radius:8px;font-size:11px;font-weight:600">'+esc(gm.toUpperCase())+'</span>'
    +'<span class="badge '+c+'">'+sl+'</span>';
  if(e.reason_bucket) h+='<span class="reason-badge '+reasonClass(e.reason_bucket)+'" title="'+escA(e.reason_bucket)+'">'+esc(shortReason(e.reason_bucket))+'</span>';
  if(e.fp_reason){{
    var fpl=e.fp_reason==='na_fp'?'N-FP':e.fp_reason==='eval_fp'?'E-FP':'FP';
    h+='<span class="fp-indicator" title="'+escA(e.fp_reason)+'">'+fpl+'</span>';
  }}
  /* ---- spacer + step dots + nav (all inline in row1) ---- */
  var si=Math.min(S.step, e.steps.length-1);
  if(si<0) si=0;
  S.step=si;
  h+='<span class="ep-spacer"></span>';
  e.steps.forEach(function(s,i){{
    var dc='step-dot'+(i===si?' active':'')+(s.reward!=null&&s.reward>0?' reward':'');
    h+='<div class="'+dc+'" data-si="'+i+'">'+i+'</div>';
  }});
  h+='<button class="nav-btn" id="enp"'+(hp?'':' disabled')+'>&#8592; Prev</button>'
    +'<span style="color:#666;font-size:11px">'+(o+1)+'/'+ORDER.length+'</span>'
    +'<button class="nav-btn" id="enn"'+(hn?'':' disabled')+'>Next &#8594;</button>'
    +'</div>';

  /* ---- Row 2: eval + ref + ans + intent (single line) ---- */
  h+='<div class="ep-row2">';
  if(e.eval_type) h+='<span class="eval-badge '+esc(e.eval_type)+'">'+esc(e.eval_type)+'</span>';
  if(e.reference_answer) h+='<span class="ep-ref">Ref:</span> <span class="ep-ref-val" title="'+escA(e.reference_answer)+'">'+esc(e.reference_answer)+'</span>';
  if(e.final_answer){{
    var ansCls=e.success===true?'ep-ans match':'ep-ans';
    h+='<span class="'+ansCls+'">Ans:</span> <span class="ep-ans-val '+ansCls+'" title="'+escA(e.final_answer)+'">'+esc(e.final_answer)+'</span>';
  }}
  if(e.intent){{
    if(e.eval_type||e.reference_answer||e.final_answer) h+='<span class="ep-sep">|</span>';
    var ic=e.intent_image?' intent-has-img':'';
    h+='<span class="ep-intent-span'+ic+'" title="'+escA(e.intent)+'"'+(e.intent_image?' data-img="'+escA(e.intent_image)+'"':'')+'>'+esc(e.intent)+'</span>';
  }}
  h+='</div>';
  h+='</div>';

  /* step cards */
  h+='<div class="steps-area">';
  e.steps.forEach(function(s,i){{
    h+='<div class="step-card" id="sc'+i+'">';
    if(s.img_path)
      h+='<img src="'+escA(s.img_path)+'"'+(i>2?' loading="lazy"':'')+' style="min-height:200px">';
    else
      h+='<div class="no-img">No screenshot</div>';
    h+='</div>';
  }});
  h+='</div>';
  $e.innerHTML=h;

  /* bind */
  document.getElementById('eb').addEventListener('click',goHome);
  if(hp) document.getElementById('enp').addEventListener('click',function(){{ goEp(ORDER[o-1],0); }});
  if(hn) document.getElementById('enn').addEventListener('click',function(){{ goEp(ORDER[o+1],0); }});
  $e.querySelectorAll('.step-dot').forEach(function(d){{
    d.addEventListener('click',function(){{ scrollStep(parseInt(d.dataset.si)); }});
  }});
}}

/* ---- view switching ---- */
function goHome(){{
  $e.style.display='none'; $h.style.display='block';
  S.view='home'; S.epKey=null; save();
  renderHome();
  window.scrollTo(0,S.scrollY||0);
}}
function goEp(k,si){{
  if(!ep(k)) return;
  S.scrollY=window.scrollY; S.lastEp=k;
  S.view='episode'; S.epKey=k; S.step=si||0; save();
  try{{
    renderEp(k);
    $h.style.display='none'; $e.style.display='block';
    if(si>0) requestAnimationFrame(function(){{ scrollStep(si); }});
    else window.scrollTo(0,0);
    preloadAdjacent(k);
  }}catch(err){{
    $e.innerHTML='<div style="padding:40px;color:#f44"><h2>Render error</h2><pre>'+esc(err.message+'\\n'+err.stack)+'</pre></div>';
    $e.style.display='block'; $h.style.display='none';
    console.error('goEp error:',err);
  }}
}}
/* preload images — no dedup cache so idle browser eviction is harmless */
function preloadEp(k){{
  var e=ep(k); if(!e) return;
  e.steps.forEach(function(s){{
    if(s.img_path){{ var img=new Image(); img.src=s.img_path; }}
  }});
}}
function preloadAdjacent(k){{
  preloadEp(k);
  var o=oi(k);
  if(o>0) preloadEp(ORDER[o-1]);
  if(o<ORDER.length-1) preloadEp(ORDER[o+1]);
}}

function scrollStep(i){{
  S.step=i; save();
  var c=document.getElementById('sc'+i);
  if(c) c.scrollIntoView({{behavior:'smooth',block:'start'}});
  $e.querySelectorAll('.step-dot').forEach(function(d){{
    d.classList.toggle('active',parseInt(d.dataset.si)===i);
  }});
}}

/* ---- intent image tooltip ---- */
var $tt=document.getElementById('intent-tooltip');
var $ttImg=$tt.querySelector('img');
document.addEventListener('mouseover',function(ev){{
  var el=ev.target.closest('.intent-has-img');
  if(!el) return;
  var src=el.dataset.img;
  if(!src) return;
  $ttImg.src=src;
  $tt.style.display='block';
  var r=el.getBoundingClientRect();
  var x=r.left, y=r.bottom+6;
  if(y+330>window.innerHeight) y=r.top-330;
  if(x+410>window.innerWidth) x=window.innerWidth-420;
  if(x<0) x=4;
  $tt.style.left=x+'px'; $tt.style.top=y+'px';
}});
document.addEventListener('mouseout',function(ev){{
  var el=ev.target.closest('.intent-has-img');
  if(el) $tt.style.display='none';
}});

/* ---- image zoom ---- */
document.addEventListener('click',function(ev){{
  if(ev.target.tagName==='IMG'&&ev.target.closest('.step-card'))
    ev.target.classList.toggle('zoomed');
}});

/* ---- keyboard ---- */
document.addEventListener('keydown',function(ev){{
  if(ev.target.tagName==='INPUT'||ev.target.tagName==='TEXTAREA') return;
  if(S.view==='episode'){{
    var e=ep(S.epKey); if(!e) return;
    var o=oi(S.epKey);
    if(ev.key==='Escape'){{
      var z=document.querySelector('.zoomed');
      if(z) z.classList.remove('zoomed'); else goHome();
      ev.preventDefault();
    }} else if(ev.key==='ArrowLeft'||ev.key==='k'){{
      if(o>0) goEp(ORDER[o-1],0); ev.preventDefault();
    }} else if(ev.key==='ArrowRight'||ev.key==='j'){{
      if(o<ORDER.length-1) goEp(ORDER[o+1],0); ev.preventDefault();
    }} else if(ev.key==='ArrowUp'){{
      if(S.step>0) scrollStep(S.step-1); ev.preventDefault();
    }} else if(ev.key==='ArrowDown'){{
      if(S.step<e.steps.length-1) scrollStep(S.step+1); ev.preventDefault();
    }}
  }}
}});

/* ---- scroll sync (episode view) ---- */
var sT=null;
window.addEventListener('scroll',function(){{
  if(S.view!=='episode') return;
  clearTimeout(sT);
  sT=setTimeout(function(){{
    var e=ep(S.epKey); if(!e) return;
    var vt=window.scrollY+120;
    for(var i=e.steps.length-1;i>=0;i--){{
      var c=document.getElementById('sc'+i);
      if(c&&c.offsetTop<=vt){{
        S.step=i;
        $e.querySelectorAll('.step-dot').forEach(function(d){{
          d.classList.toggle('active',parseInt(d.dataset.si)===i);
        }});
        save(); break;
      }}
    }}
  }},150);
}});

/* ---- auto-refresh (fetch new data without full reload to preserve JS state) ---- */
var _refreshCount=0;
setInterval(function(){{
  _refreshCount++;
  var rid=_refreshCount;
  console.log('[gallery] auto-refresh #'+rid+' fetching...');
  fetch(location.pathname+'?_='+Date.now())
    .then(function(r){{
      if(!r.ok){{ console.warn('[gallery] #'+rid+' fetch status '+r.status); return null; }}
      return r.text();
    }})
    .then(function(html){{
      if(!html){{ console.warn('[gallery] #'+rid+' empty response'); return; }}
      var p=new DOMParser().parseFromString(html,'text/html');
      var el=p.getElementById('gallery-data');
      if(!el){{ console.warn('[gallery] #'+rid+' gallery-data element not found'); return; }}
      try{{
        var nd=JSON.parse(el.textContent);
        var oldLen=ORDER.length, newLen=nd.episode_order.length;
        D=nd; GROUPS=D.groups; ORDER=D.episode_order; IDX=D.episode_index;
        if(S.view==='home'){{ renderHome(); }}
        else if(S.view==='episode'&&S.epKey&&ep(S.epKey)){{
          var sy=window.scrollY; renderEp(S.epKey); window.scrollTo(0,sy);
        }}
        save();
        console.log('[gallery] #'+rid+' OK: '+oldLen+'→'+newLen+' episodes, gen='+D.generated_at);
      }}catch(e){{ console.error('[gallery] #'+rid+' parse/render error:',e); }}
    }}).catch(function(e){{ console.error('[gallery] #'+rid+' fetch error:',e); }});
}},60000);

/* ---- init ---- */
load();
renderHome();
if(S.view==='episode'&&S.epKey&&ep(S.epKey)){{
  goEp(S.epKey,S.step);
}} else {{
  S.view='home'; $h.style.display='block'; $e.style.display='none';
  if(S.scrollY) requestAnimationFrame(function(){{ window.scrollTo(0,S.scrollY); }});
}}

}})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def _ensure_intent_images_symlink(run_dir: Path) -> None:
    """Create a symlink inside run_dir so the HTTP server can reach intent images."""
    src = Path(__file__).resolve().parent.parent.parent / "external" / "visualwebarena"
    link = run_dir / "_vwa"
    if link.is_symlink():
        if link.resolve() == src.resolve():
            return
        link.unlink()
    elif link.exists():
        return  # real dir/file — don't touch
    if src.exists():
        link.symlink_to(src)


def generate_gallery(
    run_dir: Path,
    condition: Optional[str],
    task_id: Optional[int],
    embed: bool,
) -> Path:
    gallery_path = run_dir / "gallery.html"
    if not embed:
        _ensure_intent_images_symlink(run_dir)
    source_run_dirs, base_title = _discover_source_runs(run_dir)
    episodes = _collect_episodes(source_run_dirs, condition, task_id, gallery_path, embed)
    if not episodes:
        print("No episodes found.")
        raise SystemExit(1)

    condition_labels = _load_condition_labels(source_run_dirs)
    groups = _build_groups(episodes, condition_labels)

    # Build global ordering and O(1) index
    episode_order: List[str] = []
    episode_index: Dict[str, List[int]] = {}
    for gi, group in enumerate(groups):
        for ei, ep in enumerate(group["episodes"]):
            episode_order.append(ep["key"])
            episode_index[ep["key"]] = [gi, ei]

    title = base_title
    if condition:
        title += f" / {condition}"
    if task_id is not None:
        title += f" / task_{task_id}"

    data = {
        "title": title,
        "state_key": run_dir.name,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "groups": groups,
        "episode_order": episode_order,
        "episode_index": episode_index,
    }

    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # Prevent </script> in JSON from closing the script tag
    data_json = data_json.replace("</", "<\\/")

    html_content = _HTML_TEMPLATE_V2.format(
        title=html_mod.escape(title),
        data_json=data_json,
    )

    gallery_path.write_text(html_content, encoding="utf-8")
    print(f"Gallery: {gallery_path}  ({len(episodes)} episodes)")
    return gallery_path


def generate_aggregate_gallery(
    phase_dir: Path,
    prefix_filter: Optional[str],
    condition: Optional[str],
    task_id: Optional[int],
    embed: bool,
) -> Path:
    """Generate a single gallery aggregating all matching run dirs under *phase_dir*.

    Args:
        phase_dir: e.g. results/visualwebarena/phase1/
        prefix_filter: only include runs whose prefix matches (e.g. "B1_3mode").
                       None = include all.
    """
    # Place aggregate gallery in a subdirectory named after the prefix
    # so the URL is e.g. http://localhost:8765/B1_3mode/gallery.html
    if prefix_filter:
        gallery_dir = phase_dir / prefix_filter
        gallery_dir.mkdir(parents=True, exist_ok=True)
    else:
        gallery_dir = phase_dir
    gallery_path = gallery_dir / "gallery.html"
    source_run_dirs: List[Path] = []
    for cand in sorted(phase_dir.iterdir()):
        if not cand.is_dir() or cand.is_symlink():
            continue
        if cand.name.startswith(".") or cand.name in ("analysis",):
            continue
        family = _parse_run_family(cand.name)
        if prefix_filter and (not family or family["prefix"] != prefix_filter):
            continue
        if _has_episode_data(cand):
            source_run_dirs.append(cand)

    if not source_run_dirs:
        print(f"No run dirs with episode data found in {phase_dir}")
        raise SystemExit(1)

    print(f"Aggregating {len(source_run_dirs)} run dirs:")
    for d in source_run_dirs:
        print(f"  {d.name}")

    # Ensure each source has a _vwa symlink for intent images
    if not embed:
        for d in source_run_dirs:
            _ensure_intent_images_symlink(d)
        # Also ensure a _vwa symlink in the aggregate gallery dir
        _ensure_intent_images_symlink(gallery_dir)

    episodes = _collect_episodes(source_run_dirs, condition, task_id, gallery_path, embed)
    if not episodes:
        print("No episodes found.")
        raise SystemExit(1)

    condition_labels = _load_condition_labels(source_run_dirs)
    groups = _build_groups(episodes, condition_labels)

    episode_order: List[str] = []
    episode_index: Dict[str, List[int]] = {}
    for gi, group in enumerate(groups):
        for ei, ep in enumerate(group["episodes"]):
            episode_order.append(ep["key"])
            episode_index[ep["key"]] = [gi, ei]

    title = prefix_filter or "B1 Aggregate"
    if condition:
        title += f" / {condition}"
    if task_id is not None:
        title += f" / task_{task_id}"

    data = {
        "title": title,
        "state_key": f"aggregate_{prefix_filter or 'all'}",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "groups": groups,
        "episode_order": episode_order,
        "episode_index": episode_index,
    }

    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    data_json = data_json.replace("</", "<\\/")

    html_content = _HTML_TEMPLATE_V2.format(
        title=html_mod.escape(title),
        data_json=data_json,
    )

    gallery_path.write_text(html_content, encoding="utf-8")
    print(f"Gallery: {gallery_path}  ({len(episodes)} episodes)")
    return gallery_path


def generate_combined_gallery(
    phase_dirs: List[Path],
    prefix_filter: str,
    output_dir: Path,
    condition: Optional[str],
    task_id: Optional[int],
    embed: bool,
) -> Path:
    """Generate a cross-benchmark gallery merging VWA + WA runs.

    Scans each *phase_dir* for run dirs matching *prefix_filter* (via
    ``_parse_run_family``), collects episodes with ``vwa:``/``wa:`` site
    prefixes, and writes ``gallery.html`` into *output_dir*.

    Args:
        phase_dirs: e.g. [results/visualwebarena/phase1, results/webarena/phase1]
        prefix_filter: e.g. "B1_3mode" — also matches "B1_wa_3mode" by
                       stripping the ``_wa`` infix for comparison.
        output_dir: where to write gallery.html
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = output_dir / "gallery.html"

    # Collect run dirs across all phase_dirs
    source_run_dirs: List[Path] = []
    # Normalize prefix for matching: B1_3mode matches B1_3mode_* and B1_wa_3mode_*
    base_prefix = prefix_filter.replace("_wa_", "_")  # B1_wa_3mode -> B1_3mode
    for phase_dir in phase_dirs:
        if not phase_dir.is_dir():
            continue
        for cand in sorted(phase_dir.iterdir()):
            if not cand.is_dir() or cand.is_symlink():
                continue
            if cand.name.startswith(".") or cand.name in ("analysis",):
                continue
            family = _parse_run_family(cand.name)
            if not family:
                continue
            # Match: exact prefix OR wa variant (B1_wa_3mode for B1_3mode)
            fam_base = family["prefix"].replace("_wa_", "_")
            if fam_base != base_prefix and family["prefix"] != prefix_filter:
                continue
            if _has_episode_data(cand):
                source_run_dirs.append(cand)

    if not source_run_dirs:
        print(f"No run dirs with episode data found for prefix={prefix_filter!r} in {phase_dirs}")
        raise SystemExit(1)

    print(f"Combined gallery: {len(source_run_dirs)} run dirs:")
    for d in source_run_dirs:
        print(f"  {d}")

    # Ensure _vwa symlinks
    if not embed:
        for d in source_run_dirs:
            _ensure_intent_images_symlink(d)
        _ensure_intent_images_symlink(output_dir)

    episodes = _collect_episodes(
        source_run_dirs, condition, task_id, gallery_path, embed,
        prefix_site_with_benchmark=True,
    )
    if not episodes:
        print("No episodes found.")
        raise SystemExit(1)

    condition_labels = _load_condition_labels(source_run_dirs)
    groups = _build_groups(episodes, condition_labels)

    episode_order: List[str] = []
    episode_index: Dict[str, List[int]] = {}
    for gi, group in enumerate(groups):
        for ei, ep in enumerate(group["episodes"]):
            episode_order.append(ep["key"])
            episode_index[ep["key"]] = [gi, ei]

    title = f"{prefix_filter} (VWA + WA)"
    if condition:
        title += f" / {condition}"
    if task_id is not None:
        title += f" / task_{task_id}"

    data = {
        "title": title,
        "state_key": f"combined_{prefix_filter}",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "groups": groups,
        "episode_order": episode_order,
        "episode_index": episode_index,
    }

    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    data_json = data_json.replace("</", "<\\/")

    html_content = _HTML_TEMPLATE_V2.format(
        title=html_mod.escape(title),
        data_json=data_json,
    )

    gallery_path.write_text(html_content, encoding="utf-8")
    print(f"Gallery: {gallery_path}  ({len(episodes)} episodes)")
    return gallery_path


def main():
    parser = argparse.ArgumentParser(description="Generate screenshot gallery HTML")
    parser.add_argument("--run-dir", default=None, help="Run directory (single run mode)")
    parser.add_argument("--phase-dir", default=None,
                        help="Phase directory to aggregate all runs (e.g. results/visualwebarena/phase1)")
    parser.add_argument("--phase-dirs", nargs="+", default=None,
                        help="Multiple phase directories for cross-benchmark combined gallery")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for combined gallery (required with --phase-dirs)")
    parser.add_argument("--prefix", default=None,
                        help="Filter runs by prefix when using --phase-dir/--phase-dirs (e.g. B1_3mode)")
    parser.add_argument("--condition", default=None, help="Filter to condition_id")
    parser.add_argument("--task-id", type=int, default=None, help="Filter to task_id")
    parser.add_argument(
        "--embed", action="store_true",
        help="Embed images as base64 (larger file but self-contained)",
    )
    args = parser.parse_args()
    if args.phase_dirs:
        if not args.prefix:
            parser.error("--prefix is required with --phase-dirs")
        if not args.output_dir:
            parser.error("--output-dir is required with --phase-dirs")
        generate_combined_gallery(
            [Path(p) for p in args.phase_dirs],
            args.prefix,
            Path(args.output_dir),
            args.condition, args.task_id, args.embed,
        )
    elif args.phase_dir:
        generate_aggregate_gallery(
            Path(args.phase_dir), args.prefix, args.condition, args.task_id, args.embed,
        )
    elif args.run_dir:
        generate_gallery(Path(args.run_dir), args.condition, args.task_id, args.embed)
    else:
        parser.error("Either --run-dir, --phase-dir, or --phase-dirs is required")


if __name__ == "__main__":
    main()
