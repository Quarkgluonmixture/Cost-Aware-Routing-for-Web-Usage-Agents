#!/usr/bin/env python3
"""Generate an HTML gallery of annotated (or raw) screenshots for easy browsing.

Usage:
    # Single task
    python3 scripts/generate_gallery.py --run-dir results/.../B1_xxx --task-id 17

    # Single condition, all tasks
    python3 scripts/generate_gallery.py --run-dir results/.../B1_xxx --condition phase1_dom_router_0

    # Entire run
    python3 scripts/generate_gallery.py --run-dir results/.../B1_xxx

Output: <run-dir>/gallery.html (open in browser)
"""
from __future__ import annotations

import argparse
import base64
import json
import html as html_mod
from pathlib import Path
from typing import List, Dict, Any, Optional


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
    try:
        return str(img_path.relative_to(gallery_path.parent))
    except ValueError:
        return str(img_path)


def _collect_episodes(
    run_dir: Path,
    condition_filter: Optional[str],
    task_id_filter: Optional[int],
) -> List[Dict[str, Any]]:
    """Collect episodes with their steps and screenshot paths."""
    episodes = []
    condition_dirs = sorted(run_dir.iterdir())
    for cond_dir in condition_dirs:
        if not cond_dir.is_dir() or cond_dir.name in ("analysis", ".git"):
            continue
        if condition_filter and cond_dir.name != condition_filter:
            continue
        episodes_dir = cond_dir / "episodes"
        artifacts_dir = cond_dir / "artifacts"
        if not episodes_dir.exists():
            continue

        for jsonl_path in sorted(episodes_dir.glob("*_steps_v2.jsonl")):
            # Parse task info from filename: {site}_task_{id}_steps_v2.jsonl
            stem = jsonl_path.stem.replace("_steps_v2", "")
            parts = stem.rsplit("_task_", 1)
            if len(parts) != 2:
                continue
            site = parts[0]
            try:
                task_id = int(parts[1])
            except ValueError:
                continue
            if task_id_filter is not None and task_id != task_id_filter:
                continue

            steps = _read_steps(jsonl_path)
            if not steps:
                continue

            # Read summary if available
            summary_path = episodes_dir / f"{site}_task_{task_id}_summary_v2.json"
            summary = None
            if summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                except Exception:
                    pass

            # Collect screenshot paths
            task_artifact_dir = artifacts_dir / f"{site}_task_{task_id}"
            step_data = []
            for step in steps:
                step_idx = step.get("step_idx", len(step_data))
                step_dir = task_artifact_dir / f"step_{step_idx:03d}"
                # Prefer annotated screenshot
                annotated = step_dir / "screenshot_annotated.png"
                raw = step_dir / "screenshot.png"
                img_path = annotated if annotated.exists() else raw
                step_data.append({
                    "step_idx": step_idx,
                    "action": step.get("action_str", ""),
                    "thought": (step.get("thought") or "")[:200],
                    "reward": step.get("reward"),
                    "img_path": img_path,
                    "has_annotated": annotated.exists(),
                })

            episodes.append({
                "condition": cond_dir.name,
                "site": site,
                "task_id": task_id,
                "label": f"{site}_task_{task_id}",
                "steps": step_data,
                "success": summary.get("success") if summary else None,
                "score": summary.get("score") if summary else None,
                "total_steps": len(step_data),
            })

    episodes.sort(key=lambda e: (e["condition"], e["site"], e["task_id"]))
    return episodes


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Episode Gallery — {title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #1a1a2e; color: #e0e0e0; }}

.sidebar {{
  position: fixed; left: 0; top: 0; bottom: 0; width: 280px;
  background: #16213e; overflow-y: auto; padding: 12px;
  border-right: 1px solid #333;
}}
.sidebar h2 {{ font-size: 14px; color: #888; margin: 8px 0; text-transform: uppercase; }}
.task-link {{
  display: block; padding: 8px 10px; margin: 2px 0; border-radius: 6px;
  text-decoration: none; color: #ccc; font-size: 13px;
  transition: background 0.15s;
}}
.task-link:hover {{ background: #1a3a5c; }}
.task-link.success {{ border-left: 3px solid #4caf50; }}
.task-link.fail {{ border-left: 3px solid #f44336; }}
.task-link.unknown {{ border-left: 3px solid #888; }}
.task-link .meta {{ font-size: 11px; color: #888; }}

.main {{ margin-left: 280px; padding: 20px 30px; }}

.episode-section {{
  margin-bottom: 60px;
  scroll-margin-top: 20px;
}}
.episode-header {{
  position: sticky; top: 0; z-index: 10;
  background: #1a1a2e; padding: 10px 0; border-bottom: 2px solid #333;
  display: flex; align-items: center; gap: 12px;
}}
.episode-header h2 {{ font-size: 18px; }}
.badge {{
  padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;
}}
.badge.success {{ background: #1b5e20; color: #a5d6a7; }}
.badge.fail {{ background: #b71c1c; color: #ef9a9a; }}

.step-card {{
  margin: 16px 0; background: #16213e; border-radius: 8px;
  overflow: hidden; border: 1px solid #2a2a4a;
  scroll-margin-top: 60px;
}}
.step-info {{
  padding: 8px 14px; font-size: 13px; color: #aaa;
  display: flex; gap: 16px; align-items: center;
  background: #0f1a30;
}}
.step-info .step-num {{ font-weight: 700; color: #64b5f6; min-width: 55px; }}
.step-info .action {{ color: #e0e0e0; font-family: monospace; font-size: 12px; }}
.step-card img {{
  width: 100%; display: block; cursor: pointer;
  aspect-ratio: 16 / 9;
  object-fit: contain;
  background: #0a0a1a;
}}
.step-card img.zoomed {{
  position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
  object-fit: contain; z-index: 1000; background: rgba(0,0,0,0.9);
  cursor: zoom-out;
}}

.kbd {{ background: #333; padding: 2px 6px; border-radius: 3px; font-size: 11px; }}

.controls {{
  position: fixed; bottom: 20px; right: 20px; z-index: 100;
  display: flex; gap: 8px;
}}
.controls button {{
  padding: 8px 14px; border: none; border-radius: 6px;
  background: #1a3a5c; color: #fff; cursor: pointer; font-size: 13px;
}}
.controls button:hover {{ background: #2a5a8c; }}
</style>
</head>
<body>

<div class="sidebar">
  <h2>Episodes ({episode_count})</h2>
  {sidebar_links}
</div>

<div class="main">
  {episode_sections}
</div>

<div class="controls">
  <button onclick="stepPrev()" title="Prev Step (&#8593;)">&#9650; Prev Step</button>
  <button onclick="stepNext()" title="Next Step (&#8595;)">&#9660; Next Step</button>
  <button onclick="scrollPrev()" title="Prev Task (&#8592; / k)">&#9664; Prev Task</button>
  <button onclick="scrollNext()" title="Next Task (&#8594; / j)">&#9654; Next Task</button>
</div>

<script>
// Click to zoom
document.addEventListener('click', e => {{
  if (e.target.tagName === 'IMG' && e.target.closest('.step-card')) {{
    e.target.classList.toggle('zoomed');
  }}
}});
// Sidebar click — sync currentIdx and currentStepIdx
document.querySelectorAll('.task-link').forEach(link => {{
  link.addEventListener('click', e => {{
    e.preventDefault();
    const targetId = link.getAttribute('href').slice(1);
    const idx = Array.from(sections).findIndex(s => s.id === targetId);
    if (idx >= 0) scrollToSection(idx);
  }});
}});
// Keyboard nav — tasks
const sections = document.querySelectorAll('.episode-section');
const allSteps = document.querySelectorAll('.step-card');
let currentIdx = 0;
let currentStepIdx = 0;
function scrollToSection(i) {{
  if (i >= 0 && i < sections.length) {{
    currentIdx = i;
    sections[i].scrollIntoView({{ behavior: 'smooth' }});
    // Reset step index to first step of this section
    const firstStep = sections[i].querySelector('.step-card');
    if (firstStep) currentStepIdx = Array.from(allSteps).indexOf(firstStep);
  }}
}}
function scrollPrev() {{ scrollToSection(currentIdx - 1); }}
function scrollNext() {{ scrollToSection(currentIdx + 1); }}
// Step nav
function stepNext() {{
  if (currentStepIdx < allSteps.length - 1) {{
    currentStepIdx++;
    allSteps[currentStepIdx].scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    // Update task index to match current step
    const parent = allSteps[currentStepIdx].closest('.episode-section');
    if (parent) {{ const ti = Array.from(sections).indexOf(parent); if (ti >= 0) currentIdx = ti; }}
  }}
}}
function stepPrev() {{
  if (currentStepIdx > 0) {{
    currentStepIdx--;
    allSteps[currentStepIdx].scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    // Update task index to match current step
    const parent = allSteps[currentStepIdx].closest('.episode-section');
    if (parent) {{ const ti = Array.from(sections).indexOf(parent); if (ti >= 0) currentIdx = ti; }}
  }}
}}
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft' || e.key === 'k') scrollPrev();
  if (e.key === 'ArrowRight' || e.key === 'j') scrollNext();
  if (e.key === 'ArrowDown') {{ e.preventDefault(); stepNext(); }}
  if (e.key === 'ArrowUp') {{ e.preventDefault(); stepPrev(); }}
  if (e.key === 'Escape') {{
    document.querySelectorAll('.zoomed').forEach(el => el.classList.remove('zoomed'));
  }}
}});
// Sync currentIdx + currentStepIdx from scroll position
function syncFromScroll() {{
  const viewTop = window.scrollY + 80;
  // Find current section
  for (let i = sections.length - 1; i >= 0; i--) {{
    if (sections[i].offsetTop <= viewTop) {{ currentIdx = i; break; }}
  }}
  // Find current step
  for (let i = allSteps.length - 1; i >= 0; i--) {{
    if (allSteps[i].offsetTop <= viewTop) {{ currentStepIdx = i; break; }}
  }}
}}
let scrollTimer = null;
window.addEventListener('scroll', () => {{
  clearTimeout(scrollTimer);
  scrollTimer = setTimeout(syncFromScroll, 100);
}});
// Auto-refresh: reload every 60s, preserving scroll position
setInterval(() => {{
  sessionStorage.setItem('gallery_scroll', String(window.scrollY));
  location.reload();
}}, 60000);
// Restore scroll position after reload (auto or manual)
window.addEventListener('DOMContentLoaded', () => {{
  const saved = sessionStorage.getItem('gallery_scroll');
  if (saved) {{
    const y = parseInt(saved);
    sessionStorage.removeItem('gallery_scroll');
    // Wait for images to reserve space (aspect-ratio), then scroll + sync
    requestAnimationFrame(() => {{
      window.scrollTo(0, y);
      syncFromScroll();
    }});
  }}
}});
</script>

</body>
</html>
"""


def generate_gallery(
    run_dir: Path,
    condition: Optional[str],
    task_id: Optional[int],
    embed: bool,
) -> Path:
    episodes = _collect_episodes(run_dir, condition, task_id)
    if not episodes:
        print("No episodes found.")
        raise SystemExit(1)

    gallery_path = run_dir / "gallery.html"

    # Build sidebar
    sidebar_parts = []
    for ep in episodes:
        status = "unknown"
        if ep["success"] is True:
            status = "success"
        elif ep["success"] is False:
            status = "fail"
        score_str = f"score={ep['score']}" if ep["score"] is not None else ""
        sidebar_parts.append(
            f'<a class="task-link {status}" href="#{ep["label"]}">'
            f'{ep["label"]}'
            f'<div class="meta">{ep["condition"]} | {ep["total_steps"]} steps {score_str}</div>'
            f'</a>'
        )

    # Build episode sections
    section_parts = []
    for ep in episodes:
        badge = ""
        if ep["success"] is True:
            badge = '<span class="badge success">SUCCESS</span>'
        elif ep["success"] is False:
            badge = '<span class="badge fail">FAIL</span>'

        steps_html = []
        for s in ep["steps"]:
            if embed:
                img_src = _img_to_data_uri(s["img_path"])
            else:
                img_src = _img_to_relative(s["img_path"], gallery_path)
            img_tag = f'<img src="{img_src}" loading="lazy">' if img_src else '<div style="padding:40px;color:#666">No screenshot</div>'
            action_esc = html_mod.escape(s["action"][:120])
            steps_html.append(
                f'<div class="step-card">'
                f'<div class="step-info">'
                f'<span class="step-num">Step {s["step_idx"]}</span>'
                f'<span class="action">{action_esc}</span>'
                f'</div>'
                f'{img_tag}'
                f'</div>'
            )

        section_parts.append(
            f'<div class="episode-section" id="{ep["label"]}">'
            f'<div class="episode-header">'
            f'<h2>{ep["label"]}</h2>{badge}'
            f'<span style="color:#666;font-size:13px">{ep["condition"]}</span>'
            f'</div>'
            f'{"".join(steps_html)}'
            f'</div>'
        )

    title = run_dir.name
    if condition:
        title += f" / {condition}"
    if task_id is not None:
        title += f" / task_{task_id}"

    html_content = _HTML_TEMPLATE.format(
        title=html_mod.escape(title),
        episode_count=len(episodes),
        sidebar_links="\n  ".join(sidebar_parts),
        episode_sections="\n".join(section_parts),
    )

    gallery_path.write_text(html_content, encoding="utf-8")
    print(f"Gallery: {gallery_path}  ({len(episodes)} episodes)")
    return gallery_path


def main():
    parser = argparse.ArgumentParser(description="Generate screenshot gallery HTML")
    parser.add_argument("--run-dir", required=True, help="Run directory")
    parser.add_argument("--condition", default=None, help="Filter to condition_id")
    parser.add_argument("--task-id", type=int, default=None, help="Filter to task_id")
    parser.add_argument(
        "--embed", action="store_true",
        help="Embed images as base64 (larger file but self-contained)",
    )
    args = parser.parse_args()
    generate_gallery(Path(args.run_dir), args.condition, args.task_id, args.embed)


if __name__ == "__main__":
    main()
