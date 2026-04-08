#!/usr/bin/env python3
"""
Post-process step screenshots with agent action overlays for manual inspection.

Draws on each screenshot:
  - Prominent colored banner (action type + element description + typed text)
  - Thought excerpt (second line)
  - Highlight box around target element (if element_bbox in step record)
  - Crosshair at click/type coordinates (SoM/Vision modes)
  - Scroll direction arrow
  - Colored left-side strip for quick visual scanning

Output: screenshot_annotated.png alongside the original screenshot.png

Usage:
  # Annotate a single episode
  python3 scripts/annotate_screenshots.py \
    --run-dir results/visualwebarena/phase1/B1_3mode_classifieds_20260404_141103 \
    --condition phase1_dom_router_0 \
    --task-id 12

  # Annotate all episodes in a condition
  python3 scripts/annotate_screenshots.py \
    --run-dir results/visualwebarena/phase1/B1_3mode_classifieds_20260404_141103 \
    --condition phase1_dom_router_0

  # Annotate entire run
  python3 scripts/annotate_screenshots.py \
    --run-dir results/visualwebarena/phase1/B1_3mode_classifieds_20260404_141103

  # Dry-run: list what would be annotated
  python3 scripts/annotate_screenshots.py --run-dir ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install pillow", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Colors & style
# ---------------------------------------------------------------------------

_COLORS: Dict[str, Tuple[int, int, int]] = {
    "click": (0, 220, 80),        # bright green
    "type": (50, 140, 255),       # bright blue
    "scroll": (255, 180, 0),      # orange
    "wait": (160, 160, 160),      # gray
    "back": (160, 160, 160),
    "forward": (160, 160, 160),
    "finish": (255, 60, 60),      # red
    "stop": (255, 60, 60),
    "tab_focus": (180, 100, 255), # purple
}
_DEFAULT_COLOR = (200, 200, 200)

# Layout constants
_BANNER_LINE_HEIGHT = 28
_BANNER_PADDING = 6
_SIDE_STRIP_WIDTH = 6
_FONT_SIZE_ACTION = 18
_FONT_SIZE_THOUGHT = 14
_CROSSHAIR_RADIUS = 22
_CROSSHAIR_WIDTH = 3
_BBOX_WIDTH = 3


def _get_color(action_type: str) -> Tuple[int, int, int]:
    return _COLORS.get(action_type.lower(), _DEFAULT_COLOR)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# AXTree element lookup
# ---------------------------------------------------------------------------

_ELEMENT_ID_RE = re.compile(r"^\s*\[(\d+)\]\s*(.+)$")


def _lookup_element_desc(dom_text: str, element_id: int, max_len: int = 60) -> Optional[str]:
    target = f"[{element_id}]"
    for line in dom_text.splitlines():
        if target not in line:
            continue
        m = _ELEMENT_ID_RE.match(line.strip())
        if m and int(m.group(1)) == element_id:
            desc = m.group(2).strip()
            if len(desc) > max_len:
                desc = desc[:max_len - 3] + "..."
            return desc
    return None


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _draw_highlight_box(
    draw: ImageDraw.ImageDraw,
    bbox: List[float],
    color: Tuple[int, int, int],
    width: int = _BBOX_WIDTH,
) -> None:
    """Draw a highlight rectangle around an element. bbox = [x, y, w, h] in pixels."""
    x, y, w, h = bbox
    # Semi-transparent fill
    fill_color = color + (40,)
    draw.rectangle([x, y, x + w, y + h], fill=fill_color, outline=color, width=width)


def _draw_crosshair(
    draw: ImageDraw.ImageDraw,
    x: int, y: int,
    color: Tuple[int, int, int],
    radius: int = _CROSSHAIR_RADIUS,
    width: int = _CROSSHAIR_WIDTH,
) -> None:
    # Semi-transparent filled circle
    fill = color + (50,)
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill, outline=color, width=width)
    # Cross lines
    ext = radius + 6
    draw.line([x - ext, y, x + ext, y], fill=color, width=width)
    draw.line([x, y - ext, x, y + ext], fill=color, width=width)


def _draw_scroll_arrow(
    draw: ImageDraw.ImageDraw,
    img_w: int, img_h: int,
    delta: List[float],
    color: Tuple[int, int, int],
) -> None:
    cx = int(img_w * 0.5)
    cy = int(img_h * 0.5)
    dx = delta[0] if len(delta) > 0 else 0
    dy = delta[1] if len(delta) > 1 else 0

    arrow_len = 80
    end_x = cx + int(dx * arrow_len)
    end_y = cy + int(dy * arrow_len)

    # Wide semi-transparent shaft
    draw.line([cx, cy, end_x, end_y], fill=color + (180,), width=6)

    # Arrowhead
    angle = math.atan2(end_y - cy, end_x - cx)
    head_len = 20
    for offset in [2.5, -2.5]:
        wx = end_x - int(head_len * math.cos(angle + offset))
        wy = end_y - int(head_len * math.sin(angle + offset))
        draw.line([end_x, end_y, wx, wy], fill=color + (180,), width=6)

    # Label
    direction = ""
    if abs(dy) >= abs(dx):
        direction = "SCROLL DOWN" if dy > 0 else "SCROLL UP"
    else:
        direction = "SCROLL RIGHT" if dx > 0 else "SCROLL LEFT"
    font = _load_font(20)
    bbox = font.getbbox(direction)
    tw = bbox[2] - bbox[0]
    lx = cx - tw // 2
    ly = cy - 50 if dy > 0 else cy + 30
    draw.rectangle([lx - 4, ly - 2, lx + tw + 4, ly + 24], fill=(0, 0, 0, 180))
    draw.text((lx, ly), direction, fill=color, font=font)


def _draw_banner(
    draw: ImageDraw.ImageDraw,
    img_w: int,
    lines: List[Tuple[str, Tuple[int, int, int], ImageFont.FreeTypeFont]],
) -> int:
    """Draw multi-line banner at top. Returns total banner height."""
    total_h = _BANNER_PADDING
    for text, color, font in lines:
        total_h += _BANNER_LINE_HEIGHT

    # Background
    draw.rectangle([0, 0, img_w, total_h + _BANNER_PADDING], fill=(0, 0, 0, 210))

    y = _BANNER_PADDING
    for text, color, font in lines:
        draw.text((12, y), text, fill=color, font=font)
        y += _BANNER_LINE_HEIGHT

    return total_h + _BANNER_PADDING


def _draw_side_strip(
    draw: ImageDraw.ImageDraw,
    img_h: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw a colored strip on the left edge for quick visual scanning."""
    draw.rectangle([0, 0, _SIDE_STRIP_WIDTH, img_h], fill=color + (200,))


def _draw_label_at(
    draw: ImageDraw.ImageDraw,
    x: int, y: int,
    text: str,
    color: Tuple[int, int, int],
    font: ImageFont.FreeTypeFont,
) -> None:
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = 4
    lx = x + _CROSSHAIR_RADIUS + 8
    ly = y - th // 2
    draw.rectangle([lx - pad, ly - pad, lx + tw + pad, ly + th + pad], fill=(0, 0, 0, 200))
    draw.text((lx, ly), text, fill=color, font=font)


# ---------------------------------------------------------------------------
# Core annotation
# ---------------------------------------------------------------------------

def _build_action_line(action: Dict[str, Any], action_type: str, dom_text: str = "") -> str:
    parts = [action_type.upper()]
    eid = action.get("element_id")
    coord = action.get("coordinate")

    if eid is not None:
        desc = _lookup_element_desc(dom_text, eid) if dom_text else None
        if desc:
            parts.append(f"[{eid}] {desc}")
        else:
            parts.append(f"[{eid}]")

    if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
        parts.append(f"@ ({coord[0]:.2f}, {coord[1]:.2f})")

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


def _extract_thought(action: Dict[str, Any]) -> Optional[str]:
    thought = str(action.get("thought", "") or "")
    thought = thought.strip().replace("\n", " ")
    if not thought:
        return None
    return thought


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    if not words:
        return [text]
    lines: List[str] = []
    current = words[0]
    for w in words[1:]:
        candidate = current + " " + w
        bbox = font.getbbox(candidate)
        if (bbox[2] - bbox[0]) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines


def annotate_step(
    screenshot_path: Path,
    step_record: Dict[str, Any],
    action: Dict[str, Any],
    action_type: str,
    step_idx: int,
    output_path: Path,
    font_action: ImageFont.FreeTypeFont,
    font_thought: ImageFont.FreeTypeFont,
    dom_text: str = "",
) -> None:
    img = Image.open(screenshot_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    img_w, img_h = img.size
    color = _get_color(action_type)

    # Side strip
    _draw_side_strip(draw, img_h, color)

    # Build banner lines
    action_line = f"Step {step_idx}  |  {_build_action_line(action, action_type, dom_text)}"
    banner_lines: List[Tuple[str, Tuple[int, int, int], ImageFont.FreeTypeFont]] = [
        (action_line, color, font_action),
    ]
    thought = _extract_thought(action)
    if thought:
        # Wrap thought to fit image width (leave margin for side strip + padding)
        thought_max_w = img_w - 24 - _SIDE_STRIP_WIDTH
        wrapped = _wrap_text(f"  > {thought}", font_thought, thought_max_w)
        for wline in wrapped:
            banner_lines.append((wline, (200, 200, 200), font_thought))

    banner_h = _draw_banner(draw, img_w, banner_lines)

    # Element bbox highlight (from step record, saved by runner)
    element_bbox = step_record.get("element_bbox")
    eid = action.get("element_id")
    if element_bbox and isinstance(element_bbox, (list, tuple)) and len(element_bbox) == 4:
        bx, by, bw, bh = element_bbox
        # Clamp bbox below banner so it doesn't overlap
        if by < banner_h:
            overlap = banner_h - by
            by = banner_h
            bh = max(0, bh - overlap)
        if bh > 0:
            _draw_highlight_box(draw, [bx, by, bw, bh], color)
            if eid is not None:
                _draw_label_at(draw, int(bx) - _CROSSHAIR_RADIUS - 8, int(by), f"[{eid}]", color, font_action)

    # Coordinate-based crosshair (SoM/Vision modes)
    coord = action.get("coordinate")
    has_coord = coord and isinstance(coord, (list, tuple)) and len(coord) >= 2
    if action_type in ("click", "type") and has_coord:
        px = int(coord[0] * img_w)
        py = int(coord[1] * img_h)
        _draw_crosshair(draw, px, py, color)
        if action_type == "type":
            text = str(action.get("text", "")).replace("\n", "\\n")
            if len(text) > 25:
                text = text[:22] + "..."
            _draw_label_at(draw, px, py, f"TYPE '{text}'", color, font_action)

    # Scroll arrow
    if action_type == "scroll":
        delta = action.get("delta", [0, 0])
        if isinstance(delta, (list, tuple)) and len(delta) >= 2:
            _draw_scroll_arrow(draw, img_w, img_h, delta, color)

    # Composite
    result = Image.alpha_composite(img, overlay).convert("RGB")
    result.save(output_path, "PNG")


# ---------------------------------------------------------------------------
# Episode / condition / run iteration
# ---------------------------------------------------------------------------

def _find_step_logs(
    run_dir: Path,
    condition: Optional[str] = None,
    task_id: Optional[int] = None,
) -> List[Path]:
    if condition:
        pattern_base = run_dir / condition / "episodes"
    else:
        pattern_base = run_dir

    results = []
    for p in sorted(pattern_base.rglob("*_steps_v2.jsonl")):
        if task_id is not None:
            if f"_task_{task_id}_steps_v2" not in p.name:
                continue
        results.append(p)
    return results


def _resolve_screenshot(step_record: Dict, run_dir: Path, step_dir: Path) -> Optional[Path]:
    ap = step_record.get("artifact_paths", {})
    if ap and ap.get("screenshot"):
        p = Path(ap["screenshot"])
        if p.exists():
            return p
        p2 = run_dir / ap["screenshot"]
        if p2.exists():
            return p2
    conv = step_dir / "screenshot.png"
    if conv.exists():
        return conv
    return None


def annotate_episode(
    step_log: Path,
    run_dir: Path,
    font_action: ImageFont.FreeTypeFont,
    font_thought: ImageFont.FreeTypeFont,
    dry_run: bool = False,
) -> int:
    try:
        from p79.experiment.io_utils import read_jsonl_dedup
        steps = read_jsonl_dedup(step_log)
    except Exception:
        steps = []
        with step_log.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        steps.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    ep_name = step_log.stem.replace("_steps_v2", "")
    condition_dir = step_log.parent.parent
    artifact_base = condition_dir / "artifacts" / ep_name

    count = 0
    for step in steps:
        step_idx = step.get("step_idx", 0)
        action = step.get("action", {})
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except json.JSONDecodeError:
                action = {"action_type": "unknown", "thought": action}

        action_type = str(
            step.get("action_type") or action.get("action_type") or "unknown"
        ).lower()

        step_dir = artifact_base / f"step_{step_idx:03d}"
        screenshot = _resolve_screenshot(step, run_dir, step_dir)
        if screenshot is None:
            continue

        # Load DOM text for element_id lookup
        dom_text = ""
        dom_path = step_dir / "observation_dom.txt"
        if dom_path.exists():
            try:
                dom_text = dom_path.read_text(encoding="utf-8")
            except Exception:
                pass

        output = screenshot.parent / "screenshot_annotated.png"
        if dry_run:
            print(f"  [dry-run] {output.relative_to(run_dir)}")
            count += 1
            continue

        try:
            annotate_step(
                screenshot_path=screenshot,
                step_record=step,
                action=action,
                action_type=action_type,
                step_idx=step_idx,
                output_path=output,
                font_action=font_action,
                font_thought=font_thought,
                dom_text=dom_text,
            )
            count += 1
        except Exception as e:
            print(f"  [WARN] step {step_idx}: {e}", file=sys.stderr)

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate step screenshots with agent action overlays"
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory")
    parser.add_argument("--condition", default=None, help="Filter to condition_id")
    parser.add_argument("--task-id", default=None, type=int, help="Filter to task_id")
    parser.add_argument("--dry-run", action="store_true", help="List files without writing")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"ERROR: run_dir not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    font_action = _load_font(_FONT_SIZE_ACTION)
    font_thought = _load_font(_FONT_SIZE_THOUGHT)
    step_logs = _find_step_logs(run_dir, args.condition, args.task_id)

    if not step_logs:
        print("No step logs found matching filters.")
        sys.exit(0)

    print(f"Found {len(step_logs)} episode(s) to annotate")
    total = 0
    for i, log_path in enumerate(step_logs):
        rel = log_path.relative_to(run_dir)
        print(f"[{i+1}/{len(step_logs)}] {rel}")
        count = annotate_episode(log_path, run_dir, font_action, font_thought, dry_run=args.dry_run)
        total += count

    verb = "would annotate" if args.dry_run else "annotated"
    print(f"\nDone. {verb} {total} screenshots across {len(step_logs)} episodes.")


if __name__ == "__main__":
    main()
