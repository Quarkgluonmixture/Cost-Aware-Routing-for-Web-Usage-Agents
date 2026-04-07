#!/usr/bin/env python3
"""
GLM batch digest — pre-process failed episodes into compact JSONL for Claude analysis.

Pipeline:
  GLM reads screenshots + step logs → outputs per-episode digest
  Claude reads digest.jsonl → fast attribution without viewing images

Usage:
  python3 scripts/glm_batch_digest.py \
    --run-dir results/.../B1_3mode_classifieds_20260404_141103 \
    --output analysis/dom_digest.jsonl \
    --glm-config .auth/glm \
    --delay-secs 2 --max-images 3

  # dry-run (no GLM calls, only deterministic fallback)
  python3 scripts/glm_batch_digest.py \
    --run-dir results/.../B1_3mode_classifieds_20260404_141103 \
    --output /tmp/test_digest.jsonl --dry-run
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
# Import helpers from glm_diagnosis_sidecar.py via importlib
# ---------------------------------------------------------------------------
_SIDECAR_PATH = Path(__file__).parent / "glm_diagnosis_sidecar.py"

def _load_sidecar():
    spec = importlib.util.spec_from_file_location("glm_sidecar", _SIDECAR_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

sidecar = _load_sidecar()

# Re-export frequently used helpers
_load_glm_config = sidecar._load_glm_config
_call_glm_chat = sidecar._call_glm_chat
_candidate_glm_urls = sidecar._candidate_glm_urls
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

def _compute_key_steps(case: Dict[str, Any], max_images: int = 3) -> List[int]:
    """Select key step indices for an episode: start, stuck, mid, last."""
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

    # Deduplicate preserving order, cap at max_images
    seen: Set[int] = set()
    result: List[int] = []
    for idx in candidates:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
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


def _load_dom_snippet(episode_dir: Path, step_idx: int, max_chars: int = 800) -> Optional[str]:
    """Load first max_chars of observation_dom.txt for a step."""
    path = episode_dir / f"step_{step_idx:03d}" / "observation_dom.txt"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return text[:max_chars]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GLM digest prompt + call
# ---------------------------------------------------------------------------

_DIGEST_SYSTEM_PROMPT = """\
你是实验失败归因助手。对每个失败 episode，输出结构化诊断 JSON（中文）。

你需要输出以下字段（严格 JSON 对象）：
{
  "task_id": 123,
  "screenshot_descriptions": {"0": "30-50字描述step 0截图内容", "15": "...", ...},
  "thought_summary": "2-3句话压缩全程思维链，概括agent的决策路径和失败点",
  "key_actions_compressed": "≤15个语义块的动作序列，如 SEARCH(used boat)→CLICK(listing)→BACK→FINISH",
  "category": "失败类别（如：搜索循环/导航循环/执行停滞/事实推理错误/目标不可达/过早结束/答案对齐错误）",
  "root_cause": "≤60字的具体根因",
  "is_scaffolding_issue": "是|否",
  "confidence": "high|medium|low",
  "evidence": "至少一个具体证据点"
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
8) observation_mode=dom 时没有截图，根据 DOM snippet 描述页面状态。
9) 若 unreachable_subtype=visual_dom_only 或 location_filter*，is_scaffolding_issue=是。
10) 若 stuck_subtype=account_loop 或 scroll_static，is_scaffolding_issue=是。
"""


def _build_digest_payload(
    case: Dict[str, Any],
    fallback: Dict[str, Any],
    dom_snippets: Dict[int, str],
) -> Dict[str, Any]:
    """Build the text payload for GLM (excluding images which are added separately)."""
    thought_trace = case.get("all_step_thoughts") or []
    if isinstance(thought_trace, list) and len(thought_trace) > 20:
        # Keep first 3 + last 17 to preserve start context and recent trajectory
        thought_trace = thought_trace[:3] + thought_trace[-17:]
    search_queries = case.get("search_queries") or []
    if isinstance(search_queries, list) and len(search_queries) > 16:
        search_queries = search_queries[-16:]

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
    max_images: int = 3,
) -> Dict[str, Any]:
    """Run GLM digest for a single episode. Returns the digest dict."""
    task_id = _to_int(case.get("task_id"))
    if task_id is None:
        task_id = -1
    condition_id = str(case.get("condition_id", "") or "").strip()
    obs_mode = str(case.get("observation_mode", "") or "").strip().lower()

    fallback = _fallback_episode_diagnosis(case)
    key_steps = _compute_key_steps(case, max_images=max_images)

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

        if obs_mode == "som":
            # SoM mode: use annotated SoM screenshots + SoM marks text
            som_marks = _load_som_marks_steps(ep_dir, key_steps)
            for idx in key_steps[:max_images]:
                b64 = _load_som_image_b64(ep_dir, idx)
                if b64:
                    images_b64[idx] = b64
        elif obs_mode == "vision":
            # Vision mode: use raw screenshots
            for idx in key_steps[:max_images]:
                b64 = _load_raw_screenshot_b64(ep_dir, idx)
                if b64:
                    images_b64[idx] = b64
        # dom mode: no images

    # Build payload
    payload = _build_digest_payload(case, fallback, dom_snippets)
    if som_marks:
        payload["som_marks_by_step"] = {str(k): v for k, v in som_marks.items()}

    payload_text = json.dumps(payload, ensure_ascii=False)

    # Build messages
    if images_b64:
        user_content: Any = [{"type": "text", "text": payload_text}]
        for sidx in sorted(images_b64.keys()):
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

    messages = [
        {"role": "system", "content": _DIGEST_SYSTEM_PROMPT},
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

            return _build_digest_record(case, fallback, parsed, vision_models, glmm_use)

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
    return record


def _extract_site(case: Dict[str, Any]) -> str:
    """Extract site name from condition_id or other fields."""
    cid = str(case.get("condition_id", "") or "")
    # condition_id pattern: phase1_<mode>_router_0
    # site is usually in the run_dir name, not condition. Fall back to empty.
    return ""


def _build_dry_run_record(case: Dict[str, Any]) -> Dict[str, Any]:
    """Build a digest record using only deterministic fallback (no GLM)."""
    fallback = _fallback_episode_diagnosis(case)
    _tid = _to_int(case.get("task_id"))
    return {
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


# ---------------------------------------------------------------------------
# Resume support: read already-processed (condition_id, task_id) from output
# ---------------------------------------------------------------------------

def _load_done_keys(output_path: Path) -> Set[Tuple[str, int]]:
    """Load (condition_id, task_id) pairs already in the output JSONL."""
    done: Set[Tuple[str, int]] = set()
    if not output_path.exists():
        return done
    try:
        with output_path.open("r", encoding="utf-8") as f:
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
    """Append one JSON record to the output JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    parser.add_argument("--glm-config", default=None, type=Path, help="GLM config file (.auth/glm)")
    parser.add_argument("--delay-secs", default=2.0, type=float, help="Delay between GLM calls (seconds)")
    parser.add_argument("--max-images", default=3, type=int, help="Max screenshots per episode")
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

    # Resume: skip already processed
    output_path = args.output.resolve()
    done_keys = _load_done_keys(output_path)
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

        if args.dry_run:
            record = _build_dry_run_record(case)
            if args.site:
                record["site"] = args.site
            _append_jsonl(output_path, record)
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
            )
            if args.site:
                record["site"] = args.site
            _append_jsonl(output_path, record)
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
            _append_jsonl(output_path, fb_record)
            fail_count += 1

        # Rate limit delay
        if i < len(pending) - 1:
            time.sleep(args.delay_secs)

    print(f"\n[batch-digest] Done. success={success_count} failed={fail_count}")
    print(f"[batch-digest] Output: {output_path}")


if __name__ == "__main__":
    main()
