#!/usr/bin/env python3
"""
GLM batch digest — pre-process failed episodes into compact JSONL for Claude analysis.

Pipeline:
  GLM reads screenshots + step logs → outputs per-episode digest
  Claude reads digest.jsonl → fast attribution without viewing images

Output is auto-split by observation mode into the --output directory:
  digest_dom.jsonl, digest_som.jsonl, digest_vision.jsonl

Usage:
  python3 scripts/maintenance/glm_batch_digest.py \
    --run-dir results/.../B1_3mode_classifieds_20260404_141103 \
    --output results/.../analysis/digest/ \
    --glm-config .auth/glm \
    --condition phase1_dom_router_0 \
    --delay-secs 2 --max-images 5

  # dry-run (no GLM calls, only deterministic fallback)
  python3 scripts/maintenance/glm_batch_digest.py \
    --run-dir results/.../B1_3mode_classifieds_20260404_141103 \
    --output /tmp/test_digest/ --dry-run
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Imports: GLM client helpers + sidecar-specific helpers
# ---------------------------------------------------------------------------
# B-855 (A1.15b Chunk δ P1-6): GLM client helpers moved to glm_client.py.
# Pre-fix used importlib boilerplate to pull these from
# glm_diagnosis_sidecar — paid full module load (1996 LOC) just for 3
# GLM API helpers. Now: direct `from glm_client import` for the 3 GLM
# helpers; importlib still used for sidecar-specific helpers (episode
# artifact / SoM marks / case evidence) which legitimately belong in
# diagnosis_sidecar.
import sys as _sys
_GLM_DIR = Path(__file__).parent
if str(_GLM_DIR) not in _sys.path:
    _sys.path.insert(0, str(_GLM_DIR))
from glm_client import (  # noqa: E402
    _call_glm_chat,
    _candidate_glm_urls,
    _load_glm_config,
)

_SIDECAR_PATH = _GLM_DIR / "glm_diagnosis_sidecar.py"

def _load_sidecar():
    spec = importlib.util.spec_from_file_location("glm_sidecar", _SIDECAR_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

sidecar = _load_sidecar()

# Re-export sidecar-specific helpers (diagnosis-domain logic stays in sidecar)
_fallback_episode_diagnosis = sidecar._fallback_episode_diagnosis
_find_episode_artifact_dir = sidecar._find_episode_artifact_dir
_load_som_marks_steps = sidecar._load_som_marks_steps
_load_som_image_b64 = sidecar._load_som_image_b64
_case_evidence = sidecar._case_evidence
_read_csv_rows = sidecar._read_csv_rows
_to_int = sidecar._to_int
_to_optional_bool = sidecar._to_optional_bool

# ---------------------------------------------------------------------------
# Data extraction (no incremental filtering — read ALL failed cases)
# ---------------------------------------------------------------------------

def _extract_all_failed_cases(
    episode_rows_csv: Path,
    condition_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read all failed cases from episode_reason_rows.csv (no prev-max gating)."""
    rows = _read_csv_rows(episode_rows_csv)
    failed: List[Dict[str, Any]] = []
    for r in rows:
        condition_id = str(r.get("condition_id", "") or "").strip()
        if condition_filter and condition_id != condition_filter:
            continue
        task_id = _to_int(r.get("task_id"))
        if task_id is None:
            continue
        success = _to_optional_bool(r.get("success"))
        if success is True:
            continue

        # Parse JSON list fields
        all_step_thoughts = _parse_json_list(r.get("all_step_thoughts_json", ""), _parse_thought_item)
        select_events = _parse_json_list(r.get("select_events_json", ""), _parse_select_item)
        search_queries = _parse_json_list(r.get("search_queries_json", ""), _parse_search_item)

        ref_answers_raw = str(r.get("reference_answers_json", "") or "").strip()
        reference_answers: Any = None
        if ref_answers_raw:
            try:
                reference_answers = json.loads(ref_answers_raw)
            except Exception:
                reference_answers = ref_answers_raw

        failed.append({
            "task_id": task_id,
            "condition_id": condition_id,
            "reason_bucket": str(r.get("reason_bucket", "") or "").strip(),
            "task_intent": str(r.get("task_intent", "") or "").strip(),
            "steps": _to_int(r.get("steps")),
            "thought_at_step_0": str(r.get("thought_at_step_0", "") or "").strip(),
            "all_step_thoughts": all_step_thoughts,
            "final_answer": str(r.get("final_answer", "") or "").strip(),
            "final_answer_excerpt": str(r.get("final_answer_excerpt", "") or "").strip(),
            "reference_url": str(r.get("reference_url", "") or "").strip(),
            "reference_answers": reference_answers,
            "observation_mode": str(r.get("observation_mode", "") or "").strip(),
            "answer_in_intent_price_range": _to_optional_bool(r.get("answer_in_intent_price_range")),
            "final_url_match": _to_optional_bool(r.get("final_url_match")),
            "ever_visited_reference_url": _to_optional_bool(r.get("ever_visited_reference_url")),
            "target_item_ever_visible": _to_optional_bool(r.get("target_item_ever_visible")),
            "hit_max_steps": _to_optional_bool(r.get("hit_max_steps")),
            "early_finish": _to_optional_bool(r.get("early_finish")),
            "stuck_first_step": _to_int(r.get("stuck_first_step")),
            "page_unchanged_streak_max_len": _to_int(r.get("page_unchanged_streak_max_len")),
            "page_unchanged_streak_max_pos": _to_int(r.get("page_unchanged_streak_max_pos")),
            "action_type_sequence": str(r.get("action_type_sequence", "") or "").strip(),
            "page_type_sequence": str(r.get("page_type_sequence", "") or "").strip(),
            "final_action_type": str(r.get("final_action_type", "") or "").strip(),
            "task_type": str(r.get("task_type", "") or "").strip(),
            "loop_pattern": str(r.get("loop_pattern", "") or "").strip(),
            "click_back_pairs": _to_int(r.get("click_back_pairs")),
            "max_search_query_repeat": _to_int(r.get("max_search_query_repeat")),
            "most_repeated_search_query": str(r.get("most_repeated_search_query", "") or "").strip(),
            "unique_search_queries": _to_int(r.get("unique_search_queries")),
            "select_events": select_events,
            "search_queries": search_queries,
            "stuck_subtype": str(r.get("stuck_subtype", "") or "").strip(),
            "unreachable_subtype": str(r.get("unreachable_subtype", "") or "").strip(),
            "degraded_som_steps": _to_int(r.get("degraded_som_steps")) or 0,
        })
    failed.sort(key=lambda x: (str(x.get("condition_id", "")), int(x.get("task_id") or 0)))
    return failed


def _parse_json_list(raw: Any, item_parser) -> List[Dict[str, Any]]:
    s = str(raw or "").strip()
    if not s:
        return []
    try:
        maybe = json.loads(s)
        if not isinstance(maybe, list):
            return []
        return [item_parser(item) for item in maybe if isinstance(item, dict)]
    except Exception:
        return []


def _parse_thought_item(item: Dict) -> Dict[str, Any]:
    thought = str(item.get("thought", "") or "").strip()
    if not thought:
        return {}
    return {
        "step_idx": _to_int(item.get("step_idx")) if _to_int(item.get("step_idx")) is not None else -1,
        "thought": thought,
    }


def _parse_select_item(item: Dict) -> Dict[str, Any]:
    return {
        "step_idx": _to_int(item.get("step_idx")) if _to_int(item.get("step_idx")) is not None else -1,
        "element_id": str(item.get("element_id", "") or "").strip(),
        "option": str(item.get("option", "") or "").strip(),
        "page_changed": _to_optional_bool(item.get("page_changed")),
        "action_success": _to_optional_bool(item.get("action_success")),
        "obs_url": str(item.get("obs_url", "") or "").strip(),
    }


def _parse_search_item(item: Dict) -> Dict[str, Any]:
    return {
        "step_idx": _to_int(item.get("step_idx")) if _to_int(item.get("step_idx")) is not None else -1,
        "query": str(item.get("query", "") or "").strip(),
        "element_id": str(item.get("element_id", "") or "").strip(),
        "page_changed": _to_optional_bool(item.get("page_changed")),
        "obs_url": str(item.get("obs_url", "") or "").strip(),
    }


# ---------------------------------------------------------------------------
# Key step selection + artifact loading
# ---------------------------------------------------------------------------

def _compute_key_steps(
    case: Dict[str, Any],
    max_images: int = 5,
    failed_step_indices: Optional[List[int]] = None,
) -> List[int]:
    """Select key step indices for an episode: start, stuck, mid, last, + failed steps."""
    steps = _to_int(case.get("steps")) or 1
    last = max(steps - 1, 0)
    stuck = _to_int(case.get("stuck_first_step"))

    candidates: List[int] = [0]
    if stuck is not None and stuck > 0 and stuck != last:
        candidates.append(stuck)
    mid = steps // 2
    if mid > 0 and mid not in candidates and mid != last:
        candidates.append(mid)
    if last > 0 and last not in candidates:
        candidates.append(last)

    # Deduplicate preserving order
    seen: Set[int] = set()
    result: List[int] = []
    for idx in candidates:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)

    # Fill remaining slots with failed steps (maximize distance from existing)
    if failed_step_indices and len(result) < max_images:
        remaining = [i for i in failed_step_indices if i not in seen and 0 <= i < steps]
        while remaining and len(result) < max_images:
            # Pick the failed step farthest from any existing key step
            best_idx = max(remaining, key=lambda fi: min(abs(fi - r) for r in result))
            result.append(best_idx)
            seen.add(best_idx)
            remaining.remove(best_idx)

    return result[:max(1, max_images)]


def _load_raw_screenshot_b64(episode_dir: Path, step_idx: int) -> Optional[str]:
    """Load raw screenshot.png as base64."""
    path = episode_dir / f"step_{step_idx:03d}" / "screenshot.png"
    if not path.exists():
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None


def _load_annotated_screenshot_b64(episode_dir: Path, step_idx: int) -> Optional[str]:
    """Load screenshot_annotated.png as base64, fallback to screenshot.png."""
    step_dir = episode_dir / f"step_{step_idx:03d}"
    annotated = step_dir / "screenshot_annotated.png"
    if annotated.exists():
        try:
            return base64.b64encode(annotated.read_bytes()).decode("utf-8")
        except Exception:
            pass
    # Fallback to raw screenshot
    return _load_raw_screenshot_b64(episode_dir, step_idx)


def _load_reference_images_b64(site: Optional[str], task_id: int) -> List[str]:
    """Load task reference images from VWA task JSON, return list of base64 strings.

    B-846 (A1.15b Chunk β P1-3): Pre-fix this resolved to
    `Path(__file__).parent.parent.parent / "external" / "visualwebarena"`
    which expands to `scripts/external/visualwebarena` (path nonexistent),
    not repo-root `external/visualwebarena` (path exists). Codex Mode B
    spot-check confirmed `parents[2]` resolves wrong, `parents[3]` resolves
    correct. Paper §3 prose claims GLM uses task reference images for
    visual-match diagnosis ← falsified by current code path: site is also
    always "" from _extract_site live-sidecar (see B-846 sibling fix
    below), so reference image lookup never fires regardless of path.

    Fix: use `parents[3]` (resolves to repo root since this file is at
    `scripts/maintenance/glm/glm_batch_digest.py`, parents[3] = repo).
    Paired with `_extract_site` fix to actually pass a site name.
    """
    if not site:
        return []
    vwa_root = Path(__file__).resolve().parents[3] / "external" / "visualwebarena"
    # Try both test_{site}.json and test_{site}.raw.json
    for fname in [f"test_{site}.raw.json", f"test_{site}.json"]:
        cfg_path = vwa_root / "config_files" / "vwa" / fname
        if not cfg_path.exists():
            continue
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                tasks = json.load(f)
        except Exception:
            continue
        for t in tasks:
            if t.get("task_id") != task_id:
                continue
            raw_image = t.get("image")
            if not raw_image:
                return []
            paths = [raw_image] if isinstance(raw_image, str) else list(raw_image)
            result = []
            for p in paths:
                img_path = vwa_root / p
                if img_path.exists():
                    try:
                        result.append(base64.b64encode(img_path.read_bytes()).decode("utf-8"))
                    except Exception:
                        pass
            return result
    return []


def _load_dom_snippet(episode_dir: Path, step_idx: int, max_chars: int = 0) -> Optional[str]:
    """Load observation_dom.txt for a step. max_chars=0 means no truncation."""
    path = episode_dir / f"step_{step_idx:03d}" / "observation_dom.txt"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return text[:max_chars] if max_chars > 0 else text
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Action execution summary (from step JSONL)
# ---------------------------------------------------------------------------

def _find_step_jsonl(run_dir: Path, condition_id: str, task_id: int) -> Optional[Path]:
    """Locate {site}_task_{task_id}_steps_v2.jsonl under episodes/."""
    episodes_dir = run_dir / condition_id / "episodes"
    if not episodes_dir.exists():
        return None
    for f in episodes_dir.iterdir():
        if f.name.endswith(f"_task_{task_id}_steps_v2.jsonl") and f.is_file():
            return f
    return None


def _compute_action_execution_summary(step_jsonl_path: Path) -> Dict[str, Any]:
    """Compute action execution statistics from step JSONL (all modes)."""
    # Import read_jsonl_dedup from p79
    from p79.experiment.io_utils import read_jsonl_dedup

    steps = read_jsonl_dedup(step_jsonl_path)
    total_steps = len(steps)

    click_total = 0
    click_failed = 0
    type_total = 0
    type_failed = 0
    scroll_total = 0
    parse_error_count = 0
    page_changed_count = 0
    failed_step_indices: List[int] = []
    pixel_coordinate_leak = False
    consecutive_fail = 0
    max_consecutive_fail_streak = 0

    for rec in steps:
        atype = str(rec.get("action_type", "") or "").lower()
        success = rec.get("action_success")
        step_idx = rec.get("step_idx", -1)
        err_cat = rec.get("error_category") or ""

        if atype == "click":
            click_total += 1
            if success is False:
                click_failed += 1
        elif atype == "type":
            type_total += 1
            if success is False:
                type_failed += 1
        elif atype == "scroll":
            scroll_total += 1

        if err_cat == "parse_error":
            parse_error_count += 1

        if rec.get("page_changed") is True:
            page_changed_count += 1

        if success is False:
            failed_step_indices.append(step_idx)
            consecutive_fail += 1
            max_consecutive_fail_streak = max(max_consecutive_fail_streak, consecutive_fail)
        else:
            consecutive_fail = 0

        # Check for pixel coordinate leak (coordinate > 1.0 in normalized mode)
        action_obj = rec.get("action") or {}
        coord = action_obj.get("coordinate")
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            if any(isinstance(c, (int, float)) and c > 1.0 for c in coord[:2]):
                pixel_coordinate_leak = True

    def _rate(num: int, den: int) -> str:
        return f"{num / den * 100:.1f}%" if den > 0 else "N/A"

    return {
        "total_steps": total_steps,
        "click_total": click_total,
        "click_failed": click_failed,
        "click_fail_rate": _rate(click_failed, click_total),
        "type_total": type_total,
        "type_failed": type_failed,
        "type_fail_rate": _rate(type_failed, type_total),
        "scroll_total": scroll_total,
        "parse_error_count": parse_error_count,
        "parse_error_rate": _rate(parse_error_count, total_steps),
        "page_change_rate": _rate(page_changed_count, total_steps),
        "max_consecutive_fail_streak": max_consecutive_fail_streak,
        "pixel_coordinate_leak": pixel_coordinate_leak,
        "failed_step_indices": failed_step_indices,
    }


# ---------------------------------------------------------------------------
# GLM digest prompt + call
# ---------------------------------------------------------------------------

_DIGEST_SYSTEM_PROMPT_BASE = """\
你是实验失败归因助手。对每个失败 episode，输出结构化诊断 JSON（中文）。

你需要输出以下字段（严格 JSON 对象）。
**重要**：优先输出分析结论字段，screenshot_descriptions 放在最后（即使 token 不够也能保留关键结论）。
{
  "task_id": 123,
  "category": "失败类别（如：搜索循环/导航循环/执行停滞/事实推理错误/目标不可达/过早结束/答案对齐错误）",
  "root_cause": "≤60字的具体根因",
  "is_scaffolding_issue": "是|否",
  "confidence": "high|medium|low",
  "evidence": "至少一个具体证据点",
  "thought_summary": "2-3句话压缩全程思维链，概括agent的决策路径和失败点",
  "key_actions_compressed": "≤15个语义块的动作序列，如 SEARCH(used boat)→CLICK(listing)→BACK→FINISH",
  "screenshot_descriptions": {"0": "30-50字描述step 0截图内容", "15": "...", ...}
}

规则：
1) screenshot_descriptions：对每个提供的关键步骤截图（或DOM片段），写30-50字中文描述页面状态，
   包含可见的关键元素（搜索框内容、列表数量、价格、按钮状态等）。
   若无图片/DOM则写"无可用截图"。
2) thought_summary：压缩全部 step thoughts 为2-3句话，概括 agent 的整体决策路径、
   在哪个阶段卡住、为什么没有收敛。不要逐步复述。
3) key_actions_compressed：将完整动作序列压缩为≤15个语义块，
   格式如：SEARCH(keyword)→CLICK(target)→SCROLL→BACK→FINISH。
   合并连续相同动作（如 SCROLL×3），只保留关键转折点。
4) category/root_cause/evidence/is_scaffolding_issue 的判断规则与确定性 fallback 一致。
5) 参考提供的 fallback_diagnosis，可修正也可保留，但必须有自己的判断依据。
6) root_cause ≤60字，evidence 必须引用具体步骤/搜索词/数值。
7) 若消息附有图片，优先基于图片内容填写 screenshot_descriptions。
8) observation_mode=dom 时没有页面截图，但会附带任务参考图片（标注为"[任务参考图片]"）。
   根据参考图片判断任务是否需要视觉信息（图片匹配、颜色识别等），结合 DOM snippet 描述页面状态。
9) 若 unreachable_subtype=visual_dom_only 或 location_filter*，is_scaffolding_issue=是。
   若 unreachable_subtype=visual_has_ref_image，is_scaffolding_issue=否（参考图片已可见，失败属于模型能力不足）。
10) 若 stuck_subtype=account_loop 或 scroll_static，is_scaffolding_issue=是。
"""

_DIGEST_DOM_ADDENDUM = """
=== DOM 模式专属归因规则 ===

本 episode 使用 DOM（纯文本）观测模式：agent 仅接收 AXTree（Accessibility Tree）文本，
无页面截图。所有交互通过 element_id 完成。你需要额外判断 DOM 模式特有的失败模式。

**DOM 模式已知的结构性限制**：
D1) element_id 失效：agent 引用的 element_id 在当前 DOM 中不存在或已过期，
    导致操作无效（action_success=false）。
    → 参考 action_execution_summary 中的 click_fail_rate / type_fail_rate 判断严重程度。
D2) AXTree 信息过载：DOM 过长时可能被截断，关键元素丢失。
    agent 的 thought 描述了操作目标但找不到对应 element_id → AXTree 截断问题。
D3) 无视觉信息：任务需要视觉属性（颜色、外观、图片匹配）时，DOM 模式无法获取。
    → 若 unreachable_subtype=visual_dom_only，is_scaffolding_issue=是（已在通用规则 9 覆盖）。
D4) 无空间感知：DOM 是线性文本，无法表达 2D 布局。
    任务要求空间定位（如"第X行第Y列"）时，agent 无法从 AXTree 推断位置。
D5) 参考 payload 中的 action_execution_summary 字段（若存在）判断 element_id 问题：
    - click_fail_rate 或 type_fail_rate > 30% → element_id 定位能力不足
    - parse_error_rate > 5% → 输出格式不稳定，可能是基础设施问题

**额外输出字段**（DOM 模式必填）：
在 JSON 中增加：
  "dom_element_id_issue": "是|否" — agent 是否频繁引用无效的 element_id
  "dom_failure_type": "element_id失效|AXTree截断|视觉信息缺失|空间感知缺失|不适用"
    — 若为 DOM 模式特有问题，选择具体子类型；否则填"不适用"
"""

_DIGEST_SOM_ADDENDUM = """
=== SoM 模式专属归因规则 ===

本 episode 使用 SoM（Set-of-Marks）观测模式：截图上叠加青色（#00BCD4）编号标注框，
agent 同时接收标注截图 + [SOM_MARKS] 文本索引。你需要额外判断 SoM 表征是否有效。

**SoM 已知的结构性缺陷（应判为 is_scaffolding_issue=是）**：
S1) 标注遮挡：青色 mark 框覆盖缩略图关键区域（如颜色、车型、商品外观），
    导致模型即使看了截图也无法准确识别视觉属性。
    → 检查截图中 mark 框是否遮挡了任务所需的视觉信息。
S2) 标注色混淆：统一青色标注在涉及颜色识别的任务中，可能与目标颜色混淆
    （如红色物品被青色框覆盖后难以辨认）。
S3) 空间布局丢失：SoM marks 列表是线性序号，无法表达 2D grid 位置关系，
    任务要求"第X行第Y个"时，agent 无法从 mark 列表定位。
S4) text-over-vision 偏差：agent 的 thought 中仅引用 SoM marks 文本标签（商品标题等）
    做决策，完全没有利用截图中的视觉信息（颜色、外观）。
    → 这是 SoM 表征的结构性问题：mark 文字提供了"捷径"，4B 模型会忽略视觉。
S5) viewport 外 ID 幻觉：深度滚动后 agent 引用了不可见元素的 mark ID。
S6) 参考 payload 中的 action_execution_summary 字段（若存在）辅助判断：
    - parse_error_rate > 10% → 输出格式不稳定，可能是多模态输入导致的基础设施问题，is_scaffolding_issue=是
    - click_fail_rate > 30% → mark ID 引用可能存在系统性问题

**关键判断：区分"SoM 表征失效"vs"模型能力不足"**：
- 若任务需要视觉属性（颜色、外观、图片匹配）且 agent 的 thought 没有提及任何视觉观察
  → is_scaffolding_issue=是（SoM 未能引导模型利用视觉信息）
- 若 agent 的 thought 明确描述了视觉观察但判断错误（如"看到银色"但实际是红色）
  → 需要进一步判断：检查截图中 mark 框是否遮挡了该区域
    - 若遮挡 → is_scaffolding_issue=是
    - 若未遮挡，模型确实看到了但判断错误 → is_scaffolding_issue=否（模型能力问题）
- 若任务不涉及视觉属性（纯文本/价格/导航）→ 按通用规则判断

**额外输出字段**（SoM 模式必填）：
在 JSON 中增加：
  "som_visual_used": "是|否" — agent 是否在 thought 中引用了视觉信息（颜色、外观等）
  "som_mark_occlusion": "是|否|不适用" — 截图中 mark 框是否遮挡了任务所需的关键视觉区域
  "som_failure_type": "标注遮挡|颜色混淆|空间布局丢失|text_over_vision|ID幻觉|不适用"
    — 若为 SoM 表征问题，选择具体子类型；否则填"不适用"
"""


_DIGEST_VISION_ADDENDUM = """
=== Vision 模式专属归因规则 ===

本 episode 使用 Vision（纯截图）观测模式：agent 仅接收原始页面截图，无 AXTree、无 SoM 标注。
所有交互通过坐标点击完成。你需要额外判断 Vision 模式特有的失败模式。

**Vision 模式已知的结构性限制**：
V1) 坐标偏移：agent 输出的点击坐标偏离目标元素，导致误操作或无效操作。
    → 截图已带标注（红色十字线=点击位置，顶部 banner=动作类型），可直接观察坐标是否偏移。
    若十字线明显未落在目标元素上 → 坐标偏移确认。
V2) 信息充分幻觉：agent 在 thought 中声称看到了某信息但实际截图中不可见
    （如声称看到价格但页面只显示标题）。
V3) 过早放弃：agent 在截图可见目标元素的情况下选择放弃或提交错误答案，
    可能因为缺乏 DOM 结构辅助定位。
V4) 导航能力不足：缺乏 DOM 元素 ID 辅助，agent 无法有效切换页面或使用筛选器，
    导致导航效率低于 DOM/SoM 模式。
V5) 无跨步自纠正能力：misclick 后 agent 以完全相同的坐标重复点击，不会微调。
V6) 参考 payload 中的 action_execution_summary 字段（若存在）判断坐标问题严重程度：
    - click_fail_rate > 40% → 高度怀疑系统性坐标精度问题
    - max_consecutive_fail_streak >= 3 → 连续失败说明 agent 无法自纠正
    - failed_step_indices 对应的截图可用于交叉验证偏移

**关键判断：区分"坐标精度问题"vs"视觉理解问题"**：
- 若 agent 的 thought 正确描述了目标但 action 无效 → 坐标精度问题
- 若 agent 的 thought 对视觉内容描述错误 → 视觉理解问题
- 若任务需要文本信息但 agent 无法准确读取 → Vision 模式信息获取限制

**额外输出字段**（Vision 模式必填）：
在 JSON 中增加：
  "vision_coordinate_issue": "是|否" — agent 是否存在坐标偏移问题
  "vision_failure_type": "坐标偏移|信息充分幻觉|过早放弃|导航能力不足|不适用"
    — 若为 Vision 模式特有问题，选择具体子类型；否则填"不适用"
"""


# B-845 (A1.15b Chunk β P1-2): phantom mode central normalize. Pre-fix,
# `_get_system_prompt` / mode-specific fields write / SoM-image-load all
# only branched on `obs_mode in {dom, som, vision}`. Phantom modes
# (`phantom_som`, `phantom_dom`, `phantom_text`, `phantom_prompt`) all
# fell through to the base prompt + no mode-specific fields + no
# annotated screenshots. Codex Mode B audit confirmed via disk CSV
# spot-check that archived obs_mode values include phantom_*. On a 36-
# condition Phase 1a, half the conditions get generic GLM diagnosis →
# paper §3 phantom hero claim's failure narratives are noise.
#
# Fix: normalize phantom_* into canonical {dom, som, vision} digest
# bucket. `phantom_som` shares SoM-text format (marks + image when
# present) → som-like. `phantom_dom`/`phantom_text` use AXTree text
# stripped of hierarchy → dom-like (no annotated image). `phantom_prompt`
# uses SoM prompt with AXTree text → dom-like (no SoM image). See
# memory `project_phantom_space_axes_format_not_information` for the
# format/info axis discussion that justifies this mapping.
_PHANTOM_TO_CANONICAL = {
    "phantom_som": "som",     # SoM text + image (when annotated avail)
    "phantom_dom": "dom",     # AXTree flat-list, no image
    "phantom_text": "dom",    # AXTree flat-list, no image
    "phantom_prompt": "dom",  # SoM-prompt + AXTree text, no image
}


def _normalize_obs_mode(obs_mode: str) -> str:
    """Normalize raw obs_mode to canonical {dom, som, vision} digest bucket.

    Phantom modes get mapped to the diagnostic family that shares the
    SAME observation surface (text format + image presence), so GLM
    digest prompts + mode-specific fields apply consistently across
    canonical+phantom siblings. Original obs_mode is preserved in the
    case record for downstream (paper §3 phantom-vs-canonical analysis);
    only the DIGEST decision branches use the canonical mode.
    """
    m = str(obs_mode or "").strip().lower()
    return _PHANTOM_TO_CANONICAL.get(m, m)


def _get_system_prompt(obs_mode: str) -> str:
    """Return the appropriate system prompt based on observation mode.

    Phantom modes are normalized via `_normalize_obs_mode` before dispatch.
    """
    canonical = _normalize_obs_mode(obs_mode)
    if canonical == "dom":
        return _DIGEST_SYSTEM_PROMPT_BASE + _DIGEST_DOM_ADDENDUM
    if canonical == "som":
        return _DIGEST_SYSTEM_PROMPT_BASE + _DIGEST_SOM_ADDENDUM
    if canonical == "vision":
        return _DIGEST_SYSTEM_PROMPT_BASE + _DIGEST_VISION_ADDENDUM
    return _DIGEST_SYSTEM_PROMPT_BASE


def _build_digest_payload(
    case: Dict[str, Any],
    fallback: Dict[str, Any],
    dom_snippets: Dict[int, str],
) -> Dict[str, Any]:
    """Build the text payload for GLM (excluding images which are added separately)."""
    thought_trace = case.get("all_step_thoughts") or []
    search_queries = case.get("search_queries") or []

    return {
        "task_id": case.get("task_id"),
        "task_intent": case.get("task_intent", ""),
        "task_type": case.get("task_type", ""),
        "observation_mode": case.get("observation_mode", ""),
        "steps": case.get("steps"),
        "reason_bucket": case.get("reason_bucket", ""),
        "stuck_subtype": case.get("stuck_subtype", ""),
        "unreachable_subtype": case.get("unreachable_subtype", ""),
        "degraded_som_steps": int(case.get("degraded_som_steps") or 0),
        "thought_at_step_0": case.get("thought_at_step_0", ""),
        "all_step_thoughts": thought_trace,
        "final_answer": case.get("final_answer", "") or case.get("final_answer_excerpt", ""),
        "reference_url": case.get("reference_url", ""),
        "reference_answers": case.get("reference_answers"),
        "target_item_ever_visible": case.get("target_item_ever_visible"),
        "action_type_sequence": case.get("action_type_sequence", ""),
        "page_type_sequence": case.get("page_type_sequence", ""),
        "loop_pattern": case.get("loop_pattern", ""),
        "click_back_pairs": case.get("click_back_pairs"),
        "max_search_query_repeat": case.get("max_search_query_repeat"),
        "most_repeated_search_query": case.get("most_repeated_search_query", ""),
        "search_queries": search_queries,
        "stuck_first_step": case.get("stuck_first_step"),
        "page_unchanged_streak_max_len": case.get("page_unchanged_streak_max_len"),
        "page_unchanged_streak_max_pos": case.get("page_unchanged_streak_max_pos"),
        "hit_max_steps": case.get("hit_max_steps"),
        "early_finish": case.get("early_finish"),
        "final_action_type": case.get("final_action_type"),
        "select_events": (case.get("select_events") or [])[-12:],
        "dom_snippets_by_step": {str(k): v for k, v in dom_snippets.items()},
        "fallback_diagnosis": {
            "category": fallback.get("category", ""),
            "root_cause": fallback.get("root_cause", ""),
            "is_scaffolding_issue": fallback.get("is_scaffolding_issue", "否"),
            "evidence": fallback.get("evidence", ""),
        },
    }


def _glm_digest_one(
    glmm: Dict[str, str],
    case: Dict[str, Any],
    run_dir: Path,
    max_images: int = 5,
    site: Optional[str] = None,
) -> Dict[str, Any]:
    """Run GLM digest for a single episode. Returns the digest dict."""
    task_id = _to_int(case.get("task_id"))
    if task_id is None:
        task_id = -1
    condition_id = str(case.get("condition_id", "") or "").strip()
    obs_mode = str(case.get("observation_mode", "") or "").strip().lower()

    fallback = _fallback_episode_diagnosis(case)

    # B-845 (Chunk β P1-2): canonical mode for digest branching;
    # phantom_som/dom/text/prompt mapped here. Original obs_mode preserved
    # in case record for downstream paper §3 phantom-vs-canonical analysis.
    canonical_mode = _normalize_obs_mode(obs_mode)

    # Compute action_execution_summary for all modes
    action_summary: Optional[Dict[str, Any]] = None
    failed_indices: Optional[List[int]] = None
    step_jsonl = _find_step_jsonl(run_dir, condition_id, task_id)
    if step_jsonl is not None:
        try:
            action_summary = _compute_action_execution_summary(step_jsonl)
            # Vision mode: use failed steps for key_steps selection
            if canonical_mode == "vision":
                failed_indices = action_summary.get("failed_step_indices")
        except Exception as e:
            print(f"[batch-digest] WARNING: action summary failed task_id={task_id}: {e}")

    key_steps = _compute_key_steps(
        case, max_images=max_images, failed_step_indices=failed_indices,
    )

    # Load artifacts
    ep_dir = _find_episode_artifact_dir(run_dir, condition_id, task_id)

    dom_snippets: Dict[int, str] = {}
    images_b64: Dict[int, str] = {}
    som_marks: Dict[int, str] = {}

    if ep_dir is not None:
        for idx in key_steps:
            dom = _load_dom_snippet(ep_dir, idx)
            if dom:
                dom_snippets[idx] = dom

        if canonical_mode == "som":
            # SoM mode: use annotated SoM screenshots + SoM marks text
            # phantom_som also lands here per _PHANTOM_TO_CANONICAL.
            som_marks = _load_som_marks_steps(ep_dir, key_steps)
            for idx in key_steps[:max_images]:
                b64 = _load_som_image_b64(ep_dir, idx)
                if b64:
                    images_b64[idx] = b64
        elif canonical_mode == "vision":
            # Vision mode: use annotated screenshots (with crosshair + banner)
            for idx in key_steps[:max_images]:
                b64 = _load_annotated_screenshot_b64(ep_dir, idx)
                if b64:
                    images_b64[idx] = b64
        # dom mode: no step screenshots, but load task reference images
        # so GLM can see what visual task the agent was supposed to match.
        # phantom_dom/text/prompt also land here per _PHANTOM_TO_CANONICAL
        # — they share the no-annotated-image surface with canonical DOM.
        if canonical_mode == "dom":
            ref_imgs = _load_reference_images_b64(site, task_id)
            for ri, b64 in enumerate(ref_imgs):
                # Use negative indices to distinguish from step screenshots
                images_b64[-(ri + 1)] = b64

    # Build payload
    payload = _build_digest_payload(case, fallback, dom_snippets)
    if som_marks:
        payload["som_marks_by_step"] = {str(k): v for k, v in som_marks.items()}
    if action_summary is not None:
        payload["action_execution_summary"] = action_summary

    payload_text = json.dumps(payload, ensure_ascii=False)

    # Build messages
    if images_b64:
        user_content: Any = [{"type": "text", "text": payload_text}]
        ref_keys = sorted([k for k in images_b64 if k < 0])
        step_keys = sorted([k for k in images_b64 if k >= 0])
        if ref_keys:
            user_content.append({"type": "text", "text": "[任务参考图片 — agent 需要根据此图匹配目标]"})
            for rk in ref_keys:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{images_b64[rk]}"},
                })
        if step_keys:
            user_content.append({"type": "text", "text": "[关键步骤截图]"})
        for sidx in step_keys:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{images_b64[sidx]}"},
            })
        # Vision model with fallback chain
        vision_models = []
        primary_vm = glmm.get("vision_model") or "GLM-4.6V"
        vision_models.append(primary_vm)
        for fb in ["GLM-4.6V", "GLM-5V-Turbo"]:
            if fb not in vision_models:
                vision_models.append(fb)
        glmm_use = dict(glmm)
        glmm_use["model"] = vision_models[0]
    else:
        user_content = payload_text
        glmm_use = glmm
        vision_models = []

    system_prompt = _get_system_prompt(obs_mode)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Call GLM with retry + vision fallback
    MAX_RETRIES = 3
    RETRY_SLEEP_S = 12
    last_exc: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = _call_glm_chat(glmm_use, messages).strip()
            # Strip markdown code fence
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()

            parsed = _try_parse_json(raw)
            if not parsed:
                raise ValueError(f"GLM returned unparseable response: {raw[:200]!r}")

            record = _build_digest_record(case, fallback, parsed, vision_models, glmm_use)
            if action_summary is not None:
                record["action_execution_summary"] = action_summary
            return record

        except Exception as e:
            last_exc = e
            print(
                f"[batch-digest] WARNING: attempt {attempt}/{MAX_RETRIES} failed "
                f"task_id={task_id}: {type(e).__name__}: {e}"
            )
            # Vision model fallback on 429 (rate limit) or 404 (model not found)
            err_str = str(e)
            if vision_models and ("429" in err_str or "404" in err_str or "rate" in err_str.lower()):
                cur_vm = glmm_use.get("model", "")
                try:
                    cur_idx = vision_models.index(cur_vm)
                except ValueError:
                    cur_idx = -1
                if cur_idx + 1 < len(vision_models):
                    next_vm = vision_models[cur_idx + 1]
                    print(f"[batch-digest] vision model fallback: {cur_vm} → {next_vm}")
                    glmm_use["model"] = next_vm
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_S)

    raise RuntimeError(
        f"GLM digest failed after {MAX_RETRIES} retries for task_id={task_id}: {last_exc}"
    )


def _repair_truncated_json(s: str) -> Optional[Dict[str, Any]]:
    """Attempt to repair a truncated JSON object by closing open structures.

    GLM thinking models (e.g. glm-5.1) may exhaust the token budget on
    reasoning_content, leaving the content JSON truncated.  This function
    truncates back to the last complete key-value pair, then closes all
    open structures so we recover the fields that were fully generated.
    """
    start = s.find("{")
    if start < 0:
        return None
    s = s[start:]

    # Strategy: find the last complete key-value pair boundary, truncate
    # there, then close structures.  A complete pair ends with one of:
    #   "value",   "value"}   123,   true,   null,   ],   },
    # We scan for the last comma or closing brace/bracket that is outside
    # any string.
    in_string = False
    escape_next = False
    depth = 0
    # Track position of last comma at depth=1 (top-level key-value separator)
    last_top_comma = -1

    for i, ch in enumerate(s):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            depth += 1
            continue
        if ch in "}]":
            depth -= 1
            continue
        if ch == "," and depth == 1:
            last_top_comma = i

    if last_top_comma <= 0:
        return None

    # Truncate at the last top-level comma, then close the outer object
    truncated = s[:last_top_comma] + "}"

    try:
        obj = json.loads(truncated)
        if isinstance(obj, dict):
            obj["_repaired"] = True
            return obj
    except Exception:
        pass

    # Fallback: try closing all open structures naively
    repair = s
    if in_string:
        repair += '"'
    # Re-scan for open structures
    in_str2 = False
    esc2 = False
    stack: list[str] = []
    for ch in repair:
        if esc2:
            esc2 = False
            continue
        if ch == "\\":
            if in_str2:
                esc2 = True
            continue
        if ch == '"':
            in_str2 = not in_str2
            continue
        if in_str2:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    for bracket in reversed(stack):
        repair += "}" if bracket == "{" else "]"
    try:
        obj = json.loads(repair)
        if isinstance(obj, dict):
            obj["_repaired"] = True
            return obj
    except Exception:
        pass
    return None


def _try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    l = raw.find("{")
    r = raw.rfind("}")
    if l >= 0 and r > l:
        try:
            obj = json.loads(raw[l : r + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # Last resort: try to repair truncated JSON
    repaired = _repair_truncated_json(raw)
    if repaired is not None:
        print("[batch-digest] NOTE: repaired truncated JSON response")
        return repaired
    return None


def _build_digest_record(
    case: Dict[str, Any],
    fallback: Dict[str, Any],
    parsed: Dict[str, Any],
    vision_models: List[str],
    glmm_use: Dict[str, str],
) -> Dict[str, Any]:
    """Assemble the final digest record from GLM response + case metadata."""
    # Extract GLM fields with fallback defaults
    screenshot_descs = parsed.get("screenshot_descriptions") or {}
    if not isinstance(screenshot_descs, dict):
        screenshot_descs = {}

    thought_summary = str(parsed.get("thought_summary", "") or "").strip()
    key_actions = str(parsed.get("key_actions_compressed", "") or "").strip()
    category = str(parsed.get("category", "") or "").strip() or fallback.get("category", "综合失败")
    root_cause = str(parsed.get("root_cause", "") or "").strip() or fallback.get("root_cause", "")
    confidence = str(parsed.get("confidence", "") or "").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    issue = str(parsed.get("is_scaffolding_issue", "") or "").strip()
    issue_low = issue.lower()
    if issue in {"是", "否"}:
        issue_cn = issue
    elif issue_low in {"yes", "true", "1", "y"}:
        issue_cn = "是"
    elif issue_low in {"no", "false", "0", "n"}:
        issue_cn = "否"
    else:
        issue_cn = str(fallback.get("is_scaffolding_issue", "否"))

    evidence = str(parsed.get("evidence", "") or "").strip()
    if not evidence:
        evidence = _case_evidence(case)

    _tid = _to_int(case.get("task_id"))
    record = {
        "task_id": _tid if _tid is not None else -1,
        "condition_id": str(case.get("condition_id", "") or "").strip(),
        "site": _extract_site(case),
        "task_intent": str(case.get("task_intent", "") or "").strip(),
        "observation_mode": str(case.get("observation_mode", "") or "").strip(),
        "steps": _to_int(case.get("steps")) or 0,
        "reason_bucket": str(case.get("reason_bucket", "") or "").strip(),
        "stuck_subtype": str(case.get("stuck_subtype", "") or "").strip(),
        "degraded_som_steps": int(case.get("degraded_som_steps") or 0),
        "screenshot_descriptions": screenshot_descs,
        "thought_summary": thought_summary,
        "key_actions_compressed": key_actions,
        "category": category,
        "root_cause": root_cause,
        "is_scaffolding_issue": issue_cn,
        "confidence": confidence,
        "evidence": evidence,
    }
    if vision_models:
        record["_vision_model_used"] = glmm_use.get("model", "")

    # Mode-specific fields. B-845 (Chunk β P1-2): branch on canonical
    # mode to include phantom_* under their parent family. Original
    # obs_mode kept in record["observation_mode"] for downstream analysis.
    obs_mode = str(case.get("observation_mode", "") or "").strip().lower()
    canonical_mode = _normalize_obs_mode(obs_mode)
    if canonical_mode == "dom":
        record["dom_element_id_issue"] = str(parsed.get("dom_element_id_issue", "") or "").strip() or "否"
        record["dom_failure_type"] = str(parsed.get("dom_failure_type", "") or "").strip() or "不适用"
    elif canonical_mode == "som":
        record["som_visual_used"] = str(parsed.get("som_visual_used", "") or "").strip() or "否"
        record["som_mark_occlusion"] = str(parsed.get("som_mark_occlusion", "") or "").strip() or "不适用"
        record["som_failure_type"] = str(parsed.get("som_failure_type", "") or "").strip() or "不适用"
    elif canonical_mode == "vision":
        record["vision_coordinate_issue"] = str(parsed.get("vision_coordinate_issue", "") or "").strip() or "否"
        record["vision_failure_type"] = str(parsed.get("vision_failure_type", "") or "").strip() or "不适用"

    return record


def _extract_site(case: Dict[str, Any]) -> str:
    """Extract site name from `run_dir` field or fallback heuristics.

    B-846 (A1.15b Chunk β P1-3): Pre-fix always returned `""` which made
    `_load_reference_images_b64` always early-exit at `if not site`.
    Combined with parents[2] path bug, DOM visual-match diagnosis path
    was double-dead (wrong path + no site).

    Fix: walk known fields where site can be inferred:
      1. case["site"] if explicitly set (caller may pass)
      2. case["run_dir"] basename matches `_<site>_<8-digit>` pattern
         (same anchored pattern as glm_cell_autoupdate.py:135)
      3. condition_id has no site segment (e.g. `phase1_dom_router_0`);
         fall back to "" only when none of the above matches.

    Returns canonical site name in {classifieds, reddit, shopping,
    shopping_admin} or "" if not inferable. Caller must handle "" by
    skipping reference-image load (current behavior preserved).
    """
    explicit = str(case.get("site", "") or "").strip().lower()
    if explicit in {"classifieds", "reddit", "shopping", "shopping_admin"}:
        return explicit
    run_dir = str(case.get("run_dir", "") or "")
    if run_dir:
        # Match the same anchored pattern as glm_cell_autoupdate.py:135
        # `_<site>_<8-digit>` to avoid shopping/shopping_admin substring
        # collision. Order longest-first per longest-prefix-match policy.
        import re as _re
        for site in ("shopping_admin", "classifieds", "reddit", "shopping"):
            if _re.search(rf"_{_re.escape(site)}_\d{{8}}", run_dir):
                return site
    # condition_id pattern: phase1_<mode>_router_0 has no site segment;
    # cannot extract from there. Falls through to "" → reference image
    # load early-exits (preserved fallback behavior).
    return ""


def _build_dry_run_record(case: Dict[str, Any]) -> Dict[str, Any]:
    """Build a digest record using only deterministic fallback (no GLM)."""
    fallback = _fallback_episode_diagnosis(case)
    _tid = _to_int(case.get("task_id"))
    record = {
        "task_id": _tid if _tid is not None else -1,
        "condition_id": str(case.get("condition_id", "") or "").strip(),
        "site": _extract_site(case),
        "task_intent": str(case.get("task_intent", "") or "").strip(),
        "observation_mode": str(case.get("observation_mode", "") or "").strip(),
        "steps": _to_int(case.get("steps")) or 0,
        "reason_bucket": str(case.get("reason_bucket", "") or "").strip(),
        "stuck_subtype": str(case.get("stuck_subtype", "") or "").strip(),
        "degraded_som_steps": int(case.get("degraded_som_steps") or 0),
        "screenshot_descriptions": {},
        "thought_summary": "",
        "key_actions_compressed": str(case.get("action_type_sequence", "") or "")[:120],
        "category": fallback.get("category", "综合失败"),
        "root_cause": fallback.get("root_cause", ""),
        "is_scaffolding_issue": fallback.get("is_scaffolding_issue", "否"),
        "confidence": fallback.get("confidence", "medium"),
        "evidence": fallback.get("evidence", ""),
        "_dry_run": True,
    }
    # Mode-specific default fields
    obs_mode = str(case.get("observation_mode", "") or "").strip().lower()
    if obs_mode == "dom":
        record["dom_element_id_issue"] = "否"
        record["dom_failure_type"] = "不适用"
    elif obs_mode == "som":
        record["som_visual_used"] = "否"
        record["som_mark_occlusion"] = "不适用"
        record["som_failure_type"] = "不适用"
    elif obs_mode == "vision":
        record["vision_coordinate_issue"] = "否"
        record["vision_failure_type"] = "不适用"
    return record


# ---------------------------------------------------------------------------
# Resume support: read already-processed (condition_id, task_id) from output
# ---------------------------------------------------------------------------

def _output_path_for_mode(output_dir: Path, obs_mode: str) -> Path:
    """Return mode-specific digest file path: digest_dom.jsonl, digest_som.jsonl, etc."""
    mode = obs_mode.lower() if obs_mode else "unknown"
    return output_dir / f"digest_{mode}.jsonl"


def _load_done_keys(output_dir: Path) -> Set[Tuple[str, int]]:
    """Load (condition_id, task_id) pairs already processed across all digest_*.jsonl files."""
    done: Set[Tuple[str, int]] = set()
    if not output_dir.exists():
        return done
    for jsonl_file in output_dir.glob("digest_*.jsonl"):
        try:
            with jsonl_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        cid = str(obj.get("condition_id", "") or "")
                        tid = _to_int(obj.get("task_id"))
                        if tid is not None:
                            done.add((cid, tid))
                    except Exception:
                        pass
        except Exception:
            pass
    return done


def _append_jsonl(output_path: Path, record: Dict[str, Any]) -> None:
    """Append one JSON record to the output JSONL file.

    B-856 (A1.15b Chunk δ P1-10): Mandatory exclusive file lock via
    fcntl.flock(LOCK_EX). Pre-fix had no lock — two concurrent operator
    runs (e.g. `glm_batch_digest.py` re-run while a previous run still
    finishing) both computed `done_keys` once at startup, then BOTH
    appended records for the same (condition_id, task_id) → duplicate
    JSONL rows → downstream count-based aggregators double-counted +
    paper §3 phantom-mode failure narratives potentially doubled.

    Optional fcntl was only used when `--state-file` was passed
    (separate codepath); the general digest append path was
    unprotected. Now: every `_append_jsonl` acquires LOCK_EX on the
    output file before write, releases on file close. Blocks until
    lock acquired (waits for concurrent writer); blast-radius is
    serialized appends on single host (acceptable cost).
    """
    import fcntl as _fcntl  # local import keeps stdlib top-level minimal
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        try:
            _fcntl.flock(f.fileno(), _fcntl.LOCK_EX)
        except OSError as _lock_err:
            # Filesystem doesn't support flock (rare — e.g. NFS edge case).
            # Fail-loud: paper-grade concurrency safety is required; refuse
            # to write rather than silently allow duplicate rows.
            raise RuntimeError(
                f"flock(LOCK_EX) failed on {output_path}: {_lock_err}. "
                f"Mandatory lock for digest append; check filesystem "
                f"supports advisory locks."
            ) from _lock_err
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        # Lock released on context-manager close.


# ---------------------------------------------------------------------------
# Find episode_reason_rows.csv
# ---------------------------------------------------------------------------

def _find_episode_csv(run_dir: Path, condition: Optional[str]) -> Optional[Path]:
    """Locate episode_reason_rows.csv in the run directory tree."""
    # Check common locations
    candidates = [
        run_dir / "episode_reason_rows.csv",
        run_dir / "analysis" / "episode_reason_rows.csv",
    ]
    # Also check inside condition directories
    if condition:
        candidates.append(run_dir / condition / "episode_reason_rows.csv")
    # Check parent (run dir might be a condition dir)
    candidates.append(run_dir.parent / "episode_reason_rows.csv")

    # Search recursively as last resort
    for c in candidates:
        if c.exists():
            return c

    # glob search
    found = list(run_dir.rglob("episode_reason_rows.csv"))
    if found:
        return found[0]
    return None


# ---------------------------------------------------------------------------
# CLI + main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GLM batch digest: pre-process failed episodes into compact JSONL"
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory")
    parser.add_argument("--condition", default=None, help="Filter to specific condition_id")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory (auto-splits into digest_dom.jsonl / digest_som.jsonl / digest_vision.jsonl)")
    parser.add_argument("--glm-config", default=None, type=Path, help="GLM config file (.auth/glm)")
    parser.add_argument("--delay-secs", default=2.0, type=float, help="Delay between GLM calls (seconds)")
    parser.add_argument("--max-images", default=5, type=int, help="Max screenshots per episode")
    parser.add_argument("--max-cases", default=0, type=int, help="Max cases to process (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="No GLM calls, only deterministic fallback")
    parser.add_argument("--site", default=None, help="Override site name in output records")
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"[batch-digest] ERROR: run-dir does not exist: {run_dir}")
        sys.exit(1)

    # Find CSV
    csv_path = _find_episode_csv(run_dir, args.condition)
    if csv_path is None:
        print(f"[batch-digest] ERROR: cannot find episode_reason_rows.csv under {run_dir}")
        print("  Run analyze_reason_diagnostics.py first to generate it.")
        sys.exit(1)
    print(f"[batch-digest] Using CSV: {csv_path}")

    # Load GLM config (unless dry-run)
    glmm: Optional[Dict[str, str]] = None
    if not args.dry_run:
        if args.glm_config is None:
            print("[batch-digest] ERROR: --glm-config required when not using --dry-run")
            sys.exit(1)
        glmm = _load_glm_config(args.glm_config)
        print(f"[batch-digest] GLM model: {glmm['model']}")

    # Extract all failed cases
    all_cases = _extract_all_failed_cases(csv_path, condition_filter=args.condition)
    print(f"[batch-digest] Found {len(all_cases)} failed cases")

    # Output directory (mode-specific files)
    output_dir = args.output.resolve()
    # Backwards compat: if user passed a .jsonl file path, use its parent dir
    if output_dir.suffix == ".jsonl":
        output_dir = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip already processed (scans all digest_*.jsonl in output_dir)
    done_keys = _load_done_keys(output_dir)
    if done_keys:
        print(f"[batch-digest] Resuming: {len(done_keys)} already processed, skipping")

    pending = [
        c for c in all_cases
        if (str(c.get("condition_id", "")), _to_int(c.get("task_id")) if _to_int(c.get("task_id")) is not None else -1) not in done_keys
    ]
    if args.max_cases > 0:
        pending = pending[:args.max_cases]
    print(f"[batch-digest] Processing {len(pending)} cases" + (" (dry-run)" if args.dry_run else ""))

    if not pending:
        print("[batch-digest] Nothing to process.")
        return

    # Process
    success_count = 0
    fail_count = 0
    for i, case in enumerate(pending):
        _tid = _to_int(case.get("task_id"))
        task_id = _tid if _tid is not None else -1
        cond_id = str(case.get("condition_id", "") or "")
        obs_mode = str(case.get("observation_mode", "") or "")
        progress = f"[{i+1}/{len(pending)}]"

        out_file = _output_path_for_mode(output_dir, obs_mode)

        if args.dry_run:
            record = _build_dry_run_record(case)
            if args.site:
                record["site"] = args.site
            _append_jsonl(out_file, record)
            print(f"{progress} {cond_id}/task_{task_id} ({obs_mode}) → fallback")
            success_count += 1
            continue

        # GLM call
        print(f"{progress} {cond_id}/task_{task_id} ({obs_mode}) ...", end=" ", flush=True)
        try:
            record = _glm_digest_one(
                glmm=glmm,
                case=case,
                run_dir=run_dir,
                max_images=args.max_images,
                site=args.site,
            )
            if args.site:
                record["site"] = args.site
            _append_jsonl(out_file, record)
            cat = record.get("category", "?")
            print(f"→ {cat}")
            success_count += 1
        except Exception as e:
            print(f"→ FAILED: {e}")
            # Write fallback record on permanent failure
            fb_record = _build_dry_run_record(case)
            fb_record["_glm_error"] = str(e)[:200]
            if args.site:
                fb_record["site"] = args.site
            _append_jsonl(out_file, fb_record)
            fail_count += 1

        # Rate limit delay
        if i < len(pending) - 1:
            time.sleep(args.delay_secs)

    print(f"\n[batch-digest] Done. success={success_count} failed={fail_count}")
    print(f"[batch-digest] Output dir: {output_dir}")
    for f in sorted(output_dir.glob("digest_*.jsonl")):
        n = sum(1 for _ in f.open())
        print(f"  {f.name}: {n} records")


if __name__ == "__main__":
    main()
