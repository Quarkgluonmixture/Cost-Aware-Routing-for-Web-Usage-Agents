#!/usr/bin/env python3
"""
GLM diagnosis sidecar — deep failure attribution using GLM.

- Watches episode progress for a run directory.
- Every N newly completed episodes, runs incremental reason diagnostics.
- Uses GLM to translate bucket stats into concise human-readable conclusions.
- Optionally pushes results to ntfy.
"""

from __future__ import annotations

import argparse
import base64
import csv
import fcntl
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_glm_config(cfg_path: Path) -> Dict[str, str]:
    lines: List[str] = []
    for raw in cfg_path.read_text(encoding="utf-8").splitlines():
        t = raw.strip()
        if not t or t.startswith("#"):
            continue
        lines.append(t)
    if len(lines) < 3:
        raise ValueError(f"GLM config invalid: need 3 lines (endpoint/model/api_key), got {len(lines)}")
    return {"endpoint": lines[0], "model": lines[1], "api_key": lines[2]}


def _candidate_glm_urls(endpoint: str) -> List[str]:
    ep = endpoint.rstrip("/")
    if ep.endswith("/chat/completions"):
        return [ep]
    return [f"{ep}/chat/completions", ep]


def _call_glm_chat(glmm: Dict[str, str], messages: Sequence[Dict[str, Any]], timeout_s: int = 120) -> str:
    payload_variants = [
        {
            "model": glmm["model"],
            "messages": list(messages),
            "temperature": 0.1,
            "max_tokens": 32768,
        },
    ]
    last_err: Optional[Exception] = None
    for url in _candidate_glm_urls(glmm["endpoint"]):
        for payload in payload_variants:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {glmm['api_key']}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices") or []
                if choices:
                    msg_obj = choices[0].get("message") or {}
                    msg = msg_obj.get("content")
                    if isinstance(msg, str) and msg.strip():
                        return msg.strip()
                    # GLM thinking models (e.g. glm-4.6) may put the answer in
                    # reasoning_content with content="" or missing.
                    reasoning = msg_obj.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning.strip():
                        # Try to extract JSON block first
                        r_start = reasoning.rfind("{")
                        r_end = reasoning.rfind("}")
                        if r_start >= 0 and r_end > r_start:
                            return reasoning[r_start : r_end + 1]
                        # Otherwise return the full reasoning text
                        return reasoning.strip()
                text = data.get("output_text") or data.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
                # If response lacks assistant visible text, treat as failed variant.
                last_err = RuntimeError("response has no assistant content")
            except Exception as e:  # noqa: BLE001
                last_err = e
    raise RuntimeError(f"GLM request failed: {last_err}")


def _post_ntfy(
    topic: str,
    title: str,
    body: str,
    priority: str = "default",
    timeout_s: int = 15,
) -> tuple[bool, Optional[int], str]:
    url = f"https://ntfy.sh/{topic}"
    req = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        method="POST",
        headers={"Title": title, "Priority": priority},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s):
            return True, None, ""
    except urllib.error.HTTPError as e:
        msg = f"HTTP {e.code}: {e.reason}"
        print(f"[live-diag] WARNING: ntfy push failed: {msg}")
        return False, int(e.code), msg
    except Exception as e:  # noqa: BLE001
        print(f"[live-diag] WARNING: ntfy push failed: {e}")
        return False, None, str(e)


def _load_state(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: Optional[Path], state: Dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_episode_summary_paths(run_dir: Path, condition: Optional[str]) -> List[Path]:
    paths: List[Path] = []
    if condition:
        roots = [run_dir / condition]
    else:
        roots = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("phase")]
    for root in roots:
        ep_dir = root / "episodes"
        if not ep_dir.exists():
            continue
        paths.extend(ep_dir.glob("*_summary_v2.json"))
    return sorted(paths)


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return _read_json(path)
    except Exception:
        return None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({str(k): str(v or "") for k, v in row.items()})
    except Exception:
        return []
    return rows


def _to_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _to_optional_bool(v: Any) -> Optional[bool]:
    s = str(v or "").strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def _extract_new_failed_cases(
    episode_rows_csv: Path,
    prev_task_max_by_condition: Dict[str, int],
    condition_filter: Optional[str],
    max_cases: int,
) -> List[Dict[str, Any]]:
    rows = _read_csv_rows(episode_rows_csv)
    failed: List[Dict[str, Any]] = []
    for r in rows:
        condition_id = str(r.get("condition_id", "") or "").strip()
        if condition_filter and condition_id != condition_filter:
            continue
        task_id = _to_int(r.get("task_id"))
        if task_id is None:
            continue
        prev_max = int(prev_task_max_by_condition.get(condition_id, -1))
        if task_id <= prev_max:
            continue
        success = _to_optional_bool(r.get("success"))
        if success is True:
            continue
        all_raw = str(r.get("all_step_thoughts_json", "") or "").strip()
        all_step_thoughts: List[Dict[str, Any]] = []
        if all_raw:
            try:
                maybe = json.loads(all_raw)
                if isinstance(maybe, list):
                    for item in maybe:
                        if not isinstance(item, dict):
                            continue
                        thought = str(item.get("thought", "") or "").strip()
                        if not thought:
                            continue
                        all_step_thoughts.append(
                            {
                                "step_idx": _to_int(item.get("step_idx")) if _to_int(item.get("step_idx")) is not None else -1,
                                "thought": thought,
                            }
                        )
            except Exception:
                pass
        select_raw = str(r.get("select_events_json", "") or "").strip()
        select_events: List[Dict[str, Any]] = []
        if select_raw:
            try:
                maybe = json.loads(select_raw)
                if isinstance(maybe, list):
                    for item in maybe:
                        if not isinstance(item, dict):
                            continue
                        step_idx = _to_int(item.get("step_idx"))
                        element_id = str(item.get("element_id", "") or "").strip()
                        option = str(item.get("option", "") or "").strip()
                        page_changed = _to_optional_bool(item.get("page_changed"))
                        action_success = _to_optional_bool(item.get("action_success"))
                        obs_url = str(item.get("obs_url", "") or "").strip()
                        select_events.append(
                            {
                                "step_idx": step_idx if step_idx is not None else -1,
                                "element_id": element_id,
                                "option": option,
                                "page_changed": page_changed,
                                "action_success": action_success,
                                "obs_url": obs_url,
                            }
                        )
            except Exception:
                pass

        search_raw = str(r.get("search_queries_json", "") or "").strip()
        search_queries: List[Dict[str, Any]] = []
        if search_raw:
            try:
                maybe = json.loads(search_raw)
                if isinstance(maybe, list):
                    for item in maybe:
                        if not isinstance(item, dict):
                            continue
                        search_queries.append(
                            {
                                "step_idx": _to_int(item.get("step_idx")) if _to_int(item.get("step_idx")) is not None else -1,
                                "query": str(item.get("query", "") or "").strip(),
                                "element_id": str(item.get("element_id", "") or "").strip(),
                                "page_changed": _to_optional_bool(item.get("page_changed")),
                                "obs_url": str(item.get("obs_url", "") or "").strip(),
                            }
                        )
            except Exception:
                pass
        ref_answers_raw = str(r.get("reference_answers_json", "") or "").strip()
        reference_answers: Any = None
        if ref_answers_raw:
            try:
                reference_answers = json.loads(ref_answers_raw)
            except Exception:
                reference_answers = ref_answers_raw

        failed.append(
            {
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
            }
        )
    failed.sort(key=lambda x: int(x.get("task_id") or -1), reverse=True)
    return failed[: max(1, int(max_cases))]


def _task_max_by_condition_from_episode_csv(episode_rows_csv: Path) -> Dict[str, int]:
    rows = _read_csv_rows(episode_rows_csv)
    out: Dict[str, int] = {}
    for r in rows:
        cid = str(r.get("condition_id", "") or "").strip()
        tid = _to_int(r.get("task_id"))
        if not cid or tid is None:
            continue
        prev = out.get(cid)
        if prev is None or tid > prev:
            out[cid] = tid
    return out


def _case_evidence(case: Dict[str, Any]) -> str:
    parts: List[str] = []
    loop_pattern = str(case.get("loop_pattern", "") or "").strip()
    if loop_pattern and loop_pattern != "none":
        parts.append(f"loop={loop_pattern}")
    click_back_pairs = _to_int(case.get("click_back_pairs"))
    if click_back_pairs is not None and click_back_pairs > 0:
        parts.append(f"click_back_pairs={click_back_pairs}")
    max_search_repeat = _to_int(case.get("max_search_query_repeat"))
    repeated_q = str(case.get("most_repeated_search_query", "") or "").strip()
    if max_search_repeat is not None and max_search_repeat > 1 and repeated_q:
        parts.append(f"重复搜索={repeated_q[:24]}×{max_search_repeat}")
    tiv = _to_optional_bool(case.get("target_item_ever_visible"))
    if tiv is not None:
        parts.append("目标item曾可见=是" if tiv else "目标item曾可见=否")
    up_len = _to_int(case.get("page_unchanged_streak_max_len"))
    up_pos = _to_int(case.get("page_unchanged_streak_max_pos"))
    if up_len is not None and up_len > 0:
        if up_pos is not None and up_pos >= 0:
            parts.append(f"连续无变化={up_len}步(起于step{up_pos})")
        else:
            parts.append(f"连续无变化={up_len}步")
    elif _to_int(case.get("stuck_first_step")) is not None and int(case.get("stuck_first_step")) >= 0:
        parts.append(f"首次停滞=step{int(case.get('stuck_first_step'))}")
    final_answer = str(case.get("final_answer", "") or "").strip() or str(case.get("final_answer_excerpt", "") or "").strip()
    if final_answer:
        parts.append(f"最终答案片段={final_answer[:40]}")
    select_events = case.get("select_events") or []
    if isinstance(select_events, list) and select_events:
        last = select_events[-1] if isinstance(select_events[-1], dict) else {}
        step_idx = _to_int(last.get("step_idx"))
        element_id = str(last.get("element_id", "") or "").strip()
        option = str(last.get("option", "") or "").strip()
        changed = _to_optional_bool(last.get("page_changed"))
        select_bits: List[str] = []
        if step_idx is not None and step_idx >= 0:
            select_bits.append(f"step{step_idx}")
        if element_id:
            select_bits.append(f"id={element_id}")
        if option:
            select_bits.append(f"值={option[:20]}")
        if changed is not None:
            select_bits.append("页面变化=是" if changed else "页面变化=否")
        if select_bits:
            parts.append("最近select=" + ",".join(select_bits))
    return "；".join(parts[:5])


def _task_intent_to_short_zh(intent: str) -> str:
    """Convert English VWA task_intent to short Chinese summary (≤15 chars)."""
    import re as _re
    s = (intent or "").strip()
    if not s:
        return ""

    def _q(m: "_re.Match") -> str:  # type: ignore[name-defined]
        return m.group(1).strip('"\'') if m else ""

    # Navigate to most/least/newest/oldest listing
    m = _re.search(r'[Nn]avigate to the (most |least |newest |oldest )(expensive |cheap\S* |recent |old\S* )?', s, _re.I)
    if m:
        qualifier = (m.group(1) + (m.group(2) or "")).strip().lower()
        mapping = {
            "most expensive": "最贵", "most": "最贵",
            "least expensive": "最便宜", "cheapest": "最便宜", "least": "最便宜",
            "newest": "最新", "most recent": "最新",
            "oldest": "最旧",
        }
        adj = mapping.get(qualifier, qualifier[:6])
        # try to get search keyword
        km = _re.search(r'[Ss]earch(?:ing)? for ["\']?([^"\']+?)["\']? and', s)
        kw = _q(km)[:8] if km else ""
        return f"搜{kw}找{adj}listing" if kw else f"找{adj}listing"

    # Search for X and navigate to ...
    m = _re.search(r'[Ss]earch for ["\']?([^"\']+?)["\']? and (navigate|find|go)', s, _re.I)
    if m:
        kw = m.group(1).strip()[:10]
        rest = s[m.end():].strip().lower()
        if any(x in rest for x in ("most expensive", "highest price")):
            return "搜" + kw + "找最贵"
        if any(x in rest for x in ("least expensive", "cheapest", "lowest price")):
            return "搜" + kw + "找最便宜"
        if any(x in rest for x in ("newest", "most recent", "latest")):
            return "搜" + kw + "找最新"
        if any(x in rest for x in ("oldest",)):
            return "搜" + kw + "找最旧"
        return "搜" + kw + "并导航"

    # What is the email/phone/price of ...
    m = _re.search(r'[Ww]hat is the (email|phone|price|title|name|location|address|seller)', s, _re.I)
    if m:
        field = {"email": "邮箱", "phone": "电话", "price": "价格", "title": "标题",
                 "name": "名称", "location": "地点", "address": "地址", "seller": "卖家信息"}.get(m.group(1).lower(), m.group(1))
        return f"查询{field}"

    # Add/give a N star rating
    m = _re.search(r'(?:Add|Give|Leave) (?:a )?(\d+)[- ]star', s, _re.I)
    if m:
        return f"打{m.group(1)}星评价"

    # Message/contact the seller
    if _re.search(r'[Mm]essage|[Cc]ontact|[Ss]end.*seller', s, _re.I):
        return "给卖家发消息"

    # Find the most/least expensive X
    m = _re.search(r'[Ff]ind (?:the )?(most|least) (expensive|cheap\S+) (\w[\w\s]{0,12})', s, _re.I)
    if m:
        adj = "最贵" if m.group(1).lower() == "most" else "最便宜"
        obj = m.group(3).strip()[:8]
        return f"找{adj}{obj}"

    # Fallback: first 12 chars of English (better than 40-char truncation)
    return s[:12]


def _fallback_episode_diagnosis(case: Dict[str, Any]) -> Dict[str, Any]:
    task_id = int(case.get("task_id") or -1)
    condition_id = str(case.get("condition_id", "") or "").strip()
    reason_bucket = str(case.get("reason_bucket", "") or "")
    final_answer = str(case.get("final_answer", "") or "").strip() or str(case.get("final_answer_excerpt", "") or "").strip()
    price_match = case.get("answer_in_intent_price_range", None)
    final_url_match = case.get("final_url_match", None)
    ever_visited = case.get("ever_visited_reference_url", None)
    target_item_visible = case.get("target_item_ever_visible", None)
    hit_max_steps = bool(case.get("hit_max_steps", False))
    early_finish = bool(case.get("early_finish", False))
    final_action_type = str(case.get("final_action_type", "") or "").strip().lower()
    task_type = str(case.get("task_type", "") or "").strip()
    observation_mode = str(case.get("observation_mode", "") or "").strip().lower()
    loop_pattern = str(case.get("loop_pattern", "") or "").strip()
    most_repeated_q = str(case.get("most_repeated_search_query", "") or "").strip()
    max_search_repeat = _to_int(case.get("max_search_query_repeat")) or 0

    evidence = _case_evidence(case)

    if reason_bucket == "fail_max_steps_target_unreachable" or target_item_visible is False:
        unreachable_subtype = str(case.get("unreachable_subtype", "") or "").strip()
        if unreachable_subtype == "visual_dom_only":
            category = "视觉属性DOM不可达"
            root_cause = (
                "任务要求匹配图片中的视觉属性（颜色/品牌/款式），"
                "但 DOM-only 模式无法感知 listing 图片内容，导致目标结构性不可达。"
            )
            confidence = "high"
        elif unreachable_subtype in ("location_filter_keyword", "location_filter"):
            category = "地点过滤不可达"
            root_cause = (
                "地点约束被当作关键词输入搜索框，而非通过地点筛选 UI 过滤，"
                "导致无法将结果限定到目标州/城市，目标 listing 始终不在可见结果集中。"
                if unreachable_subtype == "location_filter_keyword"
                else "任务要求特定地区 listing，但 DOM 中未找到可操作的地点过滤入口。"
            )
            confidence = "high"
        else:
            category = "目标不可达"
            root_cause = "目标item在已观测页面中始终未出现，导致无法命中目标并耗尽步数。"
            confidence = "high"
    elif reason_bucket == "fail_max_steps_click_back_loop" or loop_pattern == "click_back_loop":
        _us = str(case.get("unreachable_subtype", "") or "").strip()
        category = "导航循环"
        confidence = "high"
        if _us == "visual_dom_only":
            root_cause = "任务要求判断 listing 图片的视觉属性（颜色/款式/场景），DOM-only 模式不可见，Agent 反复进入同一页面验证失败后 back，形成循环。"
        else:
            root_cause = "反复 click→back 循环，关键路径未收敛，最终耗尽步数。"
    elif reason_bucket == "fail_max_steps_search_repeat" or loop_pattern == "search_repeat_loop":
        _us = str(case.get("unreachable_subtype", "") or "").strip()
        category = "搜索循环"
        confidence = "high"
        if _us == "visual_dom_only":
            root_cause = (
                f"任务要求判断 listing 图片的视觉属性（如封面图案/拍摄方式），"
                f"DOM-only 模式不可见，Agent 反复使用同一搜索词「{most_repeated_q[:20]}」无效，形成搜索循环。"
                if most_repeated_q
                else "任务要求判断 listing 图片的视觉属性，DOM-only 模式不可见，Agent 搜索无效导致循环。"
            )
        elif most_repeated_q:
            root_cause = f"重复使用同一搜索词「{most_repeated_q[:20]}」{max_search_repeat}次，未有效探索新路径。"
        else:
            root_cause = "搜索词重复导致流程未推进，最终耗尽步数。"
    elif price_match is False and final_action_type in {"finish", "stop"}:
        category = "事实推理错误"
        root_cause = f"最终答案中的金额/价格判断与约束不一致（答案片段：{final_answer}）。"
        confidence = "high"
    elif "wrong_url" in reason_bucket:
        category = "导航失败"
        if ever_visited is False:
            root_cause = "结束前未成功进入目标详情页，属于目标页面未命中。"
        elif final_url_match is False:
            root_cause = "曾接近目标但结束时不在目标页，属于收尾阶段导航偏移。"
        else:
            root_cause = "最终停留页面与目标不一致，导致结果判定失败。"
        confidence = "high"
    elif reason_bucket in {"fail_incomplete_or_stuck", "fail_no_progress"}:
        stuck_subtype = str(case.get("stuck_subtype", "") or "").strip()
        category = "执行停滞"
        confidence = "high"
        if stuck_subtype == "account_loop":
            root_cause = "Agent 陷入账号/登录页面循环，未能进入正常任务流程（认证墙脚手架缺陷）。"
        elif stuck_subtype == "scroll_static":
            root_cause = "Agent 反复滚动页面但未执行任何 click/type，未找到可交互元素。"
        elif stuck_subtype == "search_no_result":
            action_seq = str(case.get("action_type_sequence", "") or "")[:60]
            root_cause = f"多次重复相同搜索词，结果未变化，任务停滞（动作序列={action_seq}）。"
        elif stuck_subtype == "nav_loop":
            root_cause = "频繁 click→back 往返，关键路径未收敛，最终步数耗尽。"
        elif stuck_subtype == "target_unreachable":
            root_cause = "目标 item 在整个流程中始终未出现在可见页面，任务结构性不可达。"
        elif stuck_subtype == "target_visible_not_entered":
            root_cause = "目标 item 曾出现在结果中，但 Agent 未进入详情页完成关键交互。"
        elif stuck_subtype == "page_reading_mismatch":
            root_cause = "页面阅读类任务：Agent 的感知结果与评测标准不一致，未完成信息提取。"
        else:
            page_type_seq = str(case.get("page_type_sequence", "") or "")[:40]
            root_cause = f"关键交互未完成，任务流程卡住（页面轨迹={page_type_seq}）。"
    elif reason_bucket == "fail_max_steps" or hit_max_steps:
        category = "流程超时"
        root_cause = "在步数上限前未完成关键路径，流程被截断。"
        confidence = "high"
    elif reason_bucket == "fail_finish_eval_mismatch":
        category = "答案对齐错误"
        root_cause = f"最终答案与评测标准不一致（答案片段：{final_answer}）。"
        confidence = "medium"
    elif early_finish:
        category = "过早结束"
        root_cause = "在关键信息不足时提前 finish。"
        confidence = "medium"
    else:
        category = "综合失败"
        root_cause = f"最终答案未满足成功条件（答案片段：{final_answer}）。"
        confidence = "medium"
    scaffolding_issue = "否"
    _unreachable_subtype = str(case.get("unreachable_subtype", "") or "").strip()
    _stuck_subtype = str(case.get("stuck_subtype", "") or "").strip()
    if reason_bucket in {
        "fail_benchmark_noise",
        "fail_env_error",
        "fail_summary_error",
        "fail_max_steps_click_back_loop",
    }:
        # fail_max_steps_target_unreachable is NOT unconditionally a scaffold issue:
        # in SoM/vision modes the agent can see images, so target-unreachable is a
        # model capability failure. Only classify as scaffold via unreachable_subtype.
        scaffolding_issue = "是"
    elif _unreachable_subtype in ("visual_dom_only", "location_filter_keyword", "location_filter"):
        scaffolding_issue = "是"
    elif _stuck_subtype in ("account_loop", "scroll_static"):
        scaffolding_issue = "是"
    elif task_type == "collection" and loop_pattern == "click_back_loop":
        scaffolding_issue = "是"
    else:
        seq = str(case.get("action_type_sequence", "") or "")
        has_type = "type×" in seq
        scroll_only_like = ("scroll×" in seq) and (not has_type) and ("click×" not in seq)
        if task_type == "page_reading" and scroll_only_like and (hit_max_steps or early_finish):
            scaffolding_issue = "是"
        # Scroll-only page_reading without finishing: scaffold issue in all observation modes
        # (DOM: no visual output; SoM degraded: effectively DOM; vision: agent can't locate submit)
        if task_type == "page_reading" and scroll_only_like:
            scaffolding_issue = "是"
    return {
        "task_id": task_id,
        "condition_id": condition_id,
        "task_intent": _task_intent_to_short_zh(str(case.get("task_intent", "") or "").strip()),
        "category": category,
        "root_cause": root_cause,
        "confidence": confidence,
        "is_scaffolding_issue": scaffolding_issue,
        "evidence": evidence,
    }


def _extract_visited_item_ids(case: Dict[str, Any]) -> List[str]:
    """Extract item ids visited during the episode from search_queries obs_url and select_events."""
    import re as _re
    id_pattern = _re.compile(r"[?&]id=(\d+)")
    ids: List[str] = []
    seen: set = set()
    # From search_queries obs_url
    for q in (case.get("search_queries") or []):
        url = str(q.get("obs_url", "") or "")
        for m in id_pattern.finditer(url):
            v = m.group(1)
            if v not in seen:
                ids.append(v)
                seen.add(v)
    # From select_events obs_url
    for e in (case.get("select_events") or []):
        url = str(e.get("obs_url", "") or "")
        for m in id_pattern.finditer(url):
            v = m.group(1)
            if v not in seen:
                ids.append(v)
                seen.add(v)
    return ids


def _find_episode_artifact_dir(run_dir: Path, condition_id: str, task_id: int) -> Optional[Path]:
    """Find the artifact directory for a given condition + task_id."""
    artifacts_dir = run_dir / condition_id / "artifacts"
    if not artifacts_dir.exists():
        return None
    for d in artifacts_dir.iterdir():
        if d.is_dir() and d.name.endswith(f"_task_{task_id}"):
            return d
    return None


def _load_som_marks_steps(episode_dir: Path, step_indices: Sequence[int]) -> Dict[int, str]:
    """Load observation_som.txt for specified step indices."""
    result: Dict[int, str] = {}
    for idx in step_indices:
        path = episode_dir / f"step_{idx:03d}" / "observation_som.txt"
        if path.exists():
            try:
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    result[idx] = text
            except Exception:
                pass
    return result


def _load_som_image_b64(episode_dir: Path, step_idx: int) -> Optional[str]:
    """Load annotated SoM screenshot as base64 string."""
    path = episode_dir / "som" / f"step_{step_idx:03d}_som.png"
    if path.exists():
        try:
            return base64.b64encode(path.read_bytes()).decode("utf-8")
        except Exception:
            pass
    return None


def _glm_episode_diagnosis_one(
    glmm: Optional[Dict[str, str]],
    case: Dict[str, Any],
    som_marks_by_step: Optional[Dict[int, str]] = None,
    som_images_by_step: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    fallback = _fallback_episode_diagnosis(case)
    if not glmm:
        return fallback
    thought_trace = case.get("all_step_thoughts") or []
    if isinstance(thought_trace, list) and len(thought_trace) > 16:
        thought_trace = thought_trace[-16:]
    search_queries = case.get("search_queries") or []
    if isinstance(search_queries, list) and len(search_queries) > 16:
        search_queries = search_queries[-16:]
    payload = {
        "task_id": case.get("task_id"),
        "task_intent": case.get("task_intent", ""),
        "task_type": case.get("task_type", ""),
        "observation_mode": case.get("observation_mode", ""),
        "thought_at_step_0": case.get("thought_at_step_0", ""),
        "all_step_thoughts": thought_trace,
        "final_answer": case.get("final_answer", "") or case.get("final_answer_excerpt", ""),
        "reference_url": case.get("reference_url", ""),
        "reference_answers": case.get("reference_answers", None),
        "target_item_ever_visible": case.get("target_item_ever_visible", None),
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
        "final_action_type": case.get("final_action_type"),
        "reason_bucket": case.get("reason_bucket"),
        "stuck_subtype": case.get("stuck_subtype", ""),
        "unreachable_subtype": case.get("unreachable_subtype", ""),
        "degraded_som_steps": int(case.get("degraded_som_steps") or 0),
        "select_events": (case.get("select_events") or [])[-12:],
        "visited_item_ids": _extract_visited_item_ids(case),
        "som_marks_by_step": {str(k): v for k, v in (som_marks_by_step or {}).items()},
    }
    _system_content = (
        "你是实验失败归因助手。请基于单个失败 episode 的上下文输出结构化诊断（中文，严格 JSON）。\n"
        "输入字段包括：task_intent、task_type、observation_mode、thought_at_step_0、all_step_thoughts、final_answer、reference_answers、reference_url、select_events、search_queries、loop_pattern。\n"
        "som_marks_by_step（若存在）：各关键步骤中 agent 实际收到的 SoM 标注文本（[SOM_MARKS]...块），"
        "可用于核查目标元素是否出现在标注中、agent 选择了哪个 element_id 等。\n"
        "若消息中附有图片，图片为对应步骤的 SoM 标注截图（带编号框），"
        "可直接观察页面布局、标注覆盖情况及目标元素是否可见。\n"
        "输出必须严格为 JSON 对象："
        '{"task_id":123,"task_summary":"...","category":"...","root_cause":"...","is_scaffolding_issue":"是|否","evidence":"..."}\n'
        "要求：\n"
        "0) task_summary：≤12字中文，用口语概括任务目标核心操作，如「找最贵的船并打五星」「给帖子添加评论」「查找红色自行车价格」；不要包含条件细节；\n"
        "1) task_id 必须保留；\n"
        "2) 必须结合 all_step_thoughts 给出结论，不要只复述 bucket；\n"
        "3) root_cause 要简短具体（<=60字），不要包含任务要求原文；\n"
        "4) evidence 必须给出至少一个具体证据点（如某步 thought 的关键词/决策逻辑、重复搜索词及其失败原因、连续无变化区间、某次select值/id）；"
        "禁止原样输出 action_type_sequence 或 page_type_sequence 的压缩字符串（如 clickx4|waitx1 或 other|search|detail 等），这些对人类无意义；\n"
        "5) 不要给建议，不要输出 markdown；\n"
        "6) 禁止使用泛化模板句「思维与动作重复，且页面长期无变化，关键交互未完成」；\n"
        "7) 若 loop_pattern=click_back_loop，必须回答：agent 每次进入详情页后为什么选择返回？"
        "具体分析：(a) agent 进入详情页后 thought 里在验证什么属性？(b) 它读到了什么、缺少什么信息导致判断失败？"
        "(c) 若 target_item_ever_visible=false 则说明目标根本未出现在结果中，agent 只能盲目尝试；"
        "(d) 若 visited_item_ids 中出现重复 id 则说明 agent 缺乏访问历史记忆，反复进入同一页面。"
        "禁止仅描述循环发生而不解释决策失败原因；\n"
        "8) 若 loop_pattern=search_repeat_loop，必须引用重复搜索词证据；\n"
        "9) 若 reference_answers 提供了价格/答案，evidence 必须优先引用 reference_answers，不得编造具体数字；\n"
        "10) 若 task_type=page_reading，应优先判断「是否读取初始页面信息」，不要默认要求搜索。\n"
        "11) 若 task_type=page_reading 且动作几乎只有 scroll/无 type（任何 observation_mode 均适用），可判定 is_scaffolding_issue=是。\n"
        "12) 若 target_item_ever_visible=false，必须进一步分析目标为何未出现，需覆盖以下三点："
        "(a) agent 实际使用了哪些搜索词（引用 search_queries 字段），这些词是否能合理匹配目标商品？"
        "(b) 若搜索词本身无效（如过于泛化、含模板占位符、语义偏差），归因为模型选词错误；"
        "(c) 若搜索词合理但目标仍未出现，分析是否因 DOM/accessibility tree 缺失关键属性（颜色/品牌/图片内容）"
        "导致结构性不可达，这种情况应判定 is_scaffolding_issue=是。"
        "禁止仅写目标未出现而不解释搜索词与目标的语义匹配关系。\n"
        "13) 若 stuck_subtype=account_loop，必须判定 is_scaffolding_issue=是，"
        "说明 agent 触碰了认证墙（始终停留在 account/login 页面），无法进入任务所需功能入口，这是脚手架鉴权配置缺陷。\n"
        "14) 若 stuck_subtype=scroll_static，说明 agent 在页面上只找到了 scroll 操作，"
        "没有找到可点击/输入的交互元素，说明页面可访问性不足，应判定 is_scaffolding_issue=是。\n"
        "15) 若 unreachable_subtype=visual_dom_only 或 location_filter_keyword 或 location_filter，"
        "必须判定 is_scaffolding_issue=是："
        "visual_dom_only 意味着任务需要图片内容匹配但观测模式无法感知 listing 图片"
        "（DOM-only 模式天然不可见；SoM 模式若 degraded_som_steps>0 则也等价于 DOM-only）；"
        "location_filter_keyword/location_filter 意味着任务要求按地区筛选但地区 UI 入口在 DOM 中不可操作，"
        "agent 只能把地名当搜索词，导致结果集无法限定到目标地区。\n"
        "16) 观测模式特异性规则：\n"
        "- observation_mode=dom：看不到图片，颜色/视觉属性类任务结构性不可达，归 is_scaffolding_issue=是；\n"
        "- observation_mode=som：有截图+标注框，视觉任务原则上可完成；"
        "若 degraded_som_steps>0（SoM 标注失效步骤数），说明部分步骤退化为 DOM-only，"
        "视觉任务在这些步骤中不可达；degraded_som_steps=0 的视觉任务失败属于模型能力问题；\n"
        "- observation_mode=vision：纯截图模式，视觉任务理论上可完成；失败通常为模型推理能力不足，"
        "is_scaffolding_issue 默认为否，除非有其他明确的脚手架缺陷证据。\n"
        "17) 若提供了 som_marks_by_step，必须结合其中的元素列表辅助分析："
        "检查目标商品是否出现在 SoM marks 中（若未出现则 target_item 在感知层不可达）；"
        "核对 agent thought 引用的 element_id 是否与 marks 中对应的元素类型/标签一致；"
        "若图片也已提供，结合截图确认 agent 在视觉上是否能看到目标区域。"
    )
    _payload_text = json.dumps(payload, ensure_ascii=False)
    # Build multimodal user content if vision images are present
    _images = som_images_by_step or {}
    if _images:
        _user_content: Any = [{"type": "text", "text": _payload_text}]
        for _sidx in sorted(_images.keys()):
            _user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_images[_sidx]}"},
            })
        # Use vision-capable model when images are present
        _glmm_use = dict(glmm)
        _glmm_use["model"] = glmm.get("vision_model") or "GLM-4.6V"
    else:
        _user_content = _payload_text
        _glmm_use = glmm
    messages = [
        {"role": "system", "content": _system_content},
        {"role": "user", "content": _user_content},
    ]
    _MAX_RETRIES = 3
    _RETRY_SLEEP_S = 12
    last_exc: Optional[Exception] = None
    for _attempt in range(1, _MAX_RETRIES + 1):
        try:
            raw = _call_glm_chat(_glmm_use, messages).strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()
            parsed: Optional[Dict[str, Any]] = None
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    parsed = obj
            except Exception:
                l = raw.find("{")
                r = raw.rfind("}")
                if l >= 0 and r > l:
                    try:
                        obj = json.loads(raw[l : r + 1])
                        if isinstance(obj, dict):
                            parsed = obj
                    except Exception:
                        parsed = None
            if not parsed:
                raise ValueError(f"GLM returned unparseable response: {raw[:200]!r}")
            task_id = _to_int(parsed.get("task_id"))
            task_summary = str(parsed.get("task_summary", "") or "").strip()
            category = str(parsed.get("category", "") or "").strip()
            root_cause = str(parsed.get("root_cause", "") or "").strip()
            confidence = str(parsed.get("confidence", "") or "").strip().lower()
            issue = str(parsed.get("is_scaffolding_issue", "") or parsed.get("is_agent_capability_issue", "") or "").strip()
            evidence = str(parsed.get("evidence", "") or "").strip()
            if task_id is None or not category or not root_cause:
                raise ValueError(f"GLM response missing required fields: {parsed}")
            issue_norm = issue.lower()
            if issue in {"是", "否"}:
                issue_cn = issue
            elif issue_norm in {"yes", "true", "1", "y"}:
                issue_cn = "是"
            elif issue_norm in {"no", "false", "0", "n"}:
                issue_cn = "否"
            else:
                issue_cn = str(fallback.get("is_scaffolding_issue", "否"))
            if confidence not in {"high", "medium", "low"}:
                confidence = "medium"
            if not evidence:
                evidence = _case_evidence(case)
            return {
                "task_id": task_id,
                "condition_id": str(case.get("condition_id", "") or "").strip(),
                "task_intent": task_summary or _task_intent_to_short_zh(str(case.get("task_intent", "") or "").strip()),
                "category": category,
                "root_cause": root_cause,
                "confidence": confidence,
                "is_scaffolding_issue": issue_cn,
                "evidence": evidence,
            }
        except Exception as e:  # noqa: BLE001
            last_exc = e
            task_id_hint = case.get("task_id", "?")
            print(
                f"[live-diag] WARNING: GLM episode diagnosis attempt {_attempt}/{_MAX_RETRIES} "
                f"failed task_id={task_id_hint}: {type(e).__name__}: {e}"
            )
            if _attempt < _MAX_RETRIES:
                time.sleep(_RETRY_SLEEP_S)
    # All retries exhausted — raise so caller decides what to do (no silent fallback)
    task_id_hint = case.get("task_id", "?")
    raise RuntimeError(
        f"GLM episode diagnosis failed after {_MAX_RETRIES} retries for task_id={task_id_hint}: {last_exc}"
    )


def _glm_episode_diagnosis(
    glmm: Optional[Dict[str, str]],
    cases: Sequence[Dict[str, Any]],
    run_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, case in enumerate(cases):
        if i > 0:
            time.sleep(2)  # prevent rapid-fire rate limiting
        # Determine key step indices: step 0, stuck start, last step
        _thoughts = case.get("all_step_thoughts") or []
        _thought_idxs = [
            int(t.get("step_idx", -1)) for t in _thoughts
            if isinstance(t, dict) and _to_int(t.get("step_idx")) is not None and int(t.get("step_idx", -1)) >= 0
        ]
        _last_step = max(_thought_idxs) if _thought_idxs else None
        _stuck_step = _to_int(case.get("stuck_first_step"))
        _key_steps: List[int] = [0]
        if _stuck_step is not None and _stuck_step > 0:
            _key_steps.append(_stuck_step)
        if _last_step is not None and _last_step > 0 and _last_step not in _key_steps:
            _key_steps.append(_last_step)
        # Load SoM artifacts
        _som_marks: Dict[int, str] = {}
        _som_images: Dict[int, str] = {}
        if run_dir is not None:
            _task_id = _to_int(case.get("task_id"))
            _cond_id = str(case.get("condition_id", "") or "").strip()
            if _task_id is not None and _cond_id:
                _ep_dir = _find_episode_artifact_dir(run_dir, _cond_id, _task_id)
                if _ep_dir is not None:
                    _som_marks = _load_som_marks_steps(_ep_dir, _key_steps)
                    if str(case.get("observation_mode", "")) == "som":
                        for _s in _key_steps:
                            _b64 = _load_som_image_b64(_ep_dir, _s)
                            if _b64:
                                _som_images[_s] = _b64
        try:
            out.append(_glm_episode_diagnosis_one(
                glmm=glmm,
                case=case,
                som_marks_by_step=_som_marks or None,
                som_images_by_step=_som_images or None,
            ))
        except Exception as e:  # noqa: BLE001
            task_id_hint = case.get("task_id", "?")
            print(f"[live-diag] ERROR: GLM episode diagnosis permanently failed task_id={task_id_hint}: {e}")
            out.append({
                "task_id": _to_int(case.get("task_id")),
                "condition_id": str(case.get("condition_id", "") or "").strip(),
                "task_intent": str(case.get("task_intent", "") or "")[:60].strip(),
                "category": "GLM_FAILED",
                "root_cause": f"GLM诊断失败，请手动分析（错误：{type(e).__name__}）",
                "confidence": "low",
                "is_scaffolding_issue": "?",
                "evidence": "",
            })
    return out


def _confidence_to_zh(conf: str) -> str:
    c = str(conf or "").strip().lower()
    if c == "high":
        return "高"
    if c == "low":
        return "低"
    return "中"


def _category_to_zh(cat: str) -> str:
    c = str(cat or "").strip()
    low = c.lower()
    mapping = {
        "execution loop": "执行循环",
        "execution stall": "执行停滞",
        "execution failure": "执行失败",
        "navigation loop": "导航循环",
        "search_strategy_failure": "搜索策略失败",
        "reasoning error": "推理错误",
        "factual reasoning error": "事实推理错误",
        "perception error": "感知错误",
        "navigation failure": "导航失败",
        "answer mismatch": "答案不匹配",
        "fail_finish_eval_mismatch": "答案对齐错误",
        "glm_failed": "⚠GLM诊断失败",
    }
    return mapping.get(low, c or "综合失败")


def _format_episode_report_lines(items: Sequence[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for x in items:
        if not isinstance(x, dict):
            continue
        task_id = _to_int(x.get("task_id"))
        condition_id = str(x.get("condition_id", "") or "").strip()
        task_intent = str(x.get("task_intent", "") or "").strip()
        category = _category_to_zh(str(x.get("category", "") or "").strip())
        root = str(x.get("root_cause", "") or "").strip()
        evidence = str(x.get("evidence", "") or "").strip()
        scaffolding_issue = str(x.get("is_scaffolding_issue", "否") or "否").strip()
        if scaffolding_issue not in {"是", "否", "?"}:
            scaffolding_issue = "否"
        if task_id is None:
            continue
        prefix = f"{condition_id}/" if condition_id else ""
        lines.append(f"- {prefix}task_{task_id} | {category} | 脚手架缺陷:{scaffolding_issue}")
        if task_intent:
            lines.append(f"  目标: {task_intent}")
        lines.append(f"  诊断: {root}")
        if evidence:
            lines.append(f"  证据: {evidence}")
    return lines


def _episode_count(run_dir: Path, condition: Optional[str]) -> int:
    return len(_iter_episode_summary_paths(run_dir, condition))


def _episode_count_by_condition(run_dir: Path) -> Dict[str, int]:
    """Count episodes per condition directory (phase*)."""
    counts: Dict[str, int] = {}
    for cond_dir in run_dir.iterdir():
        if not cond_dir.is_dir() or not cond_dir.name.startswith("phase"):
            continue
        ep_dir = cond_dir / "episodes"
        if not ep_dir.exists():
            continue
        n = len(list(ep_dir.glob("*_summary_v2.json")))
        if n > 0:
            counts[cond_dir.name] = n
    return counts


def _max_task_id(run_dir: Path, condition: Optional[str]) -> Optional[int]:
    max_tid: Optional[int] = None
    for p in _iter_episode_summary_paths(run_dir, condition):
        name = p.name
        # <site>_task_<id>_summary_v2.json
        try:
            tid = int(name.split("_task_")[1].split("_summary_v2.json")[0])
        except Exception:
            continue
        if max_tid is None or tid > max_tid:
            max_tid = tid
    return max_tid


def _cleanup_old_output_dirs(out_root: Path, keep_dir: Path, retain_dirs: int = 1) -> int:
    if retain_dirs < 1:
        retain_dirs = 1
    all_dirs = [p for p in out_root.iterdir() if p.is_dir()]
    keep_dir = keep_dir.resolve()
    ordered: List[Path] = [keep_dir]
    ordered.extend(sorted((p.resolve() for p in all_dirs if p.resolve() != keep_dir), key=lambda p: p.stat().st_mtime, reverse=True))
    keep_set = {p for p in ordered[:retain_dirs]}
    deleted = 0
    for d in all_dirs:
        rp = d.resolve()
        if rp in keep_set:
            continue
        try:
            shutil.rmtree(rp)
            deleted += 1
        except Exception as e:  # noqa: BLE001
            print(f"[live-diag] WARNING: failed to remove old output dir {rp}: {e}")
    return deleted


def _run_diagnostics(
    py_bin: str,
    script_path: Path,
    run_dir: Path,
    condition: Optional[str],
    task_max: Optional[int],
    out_dir: Path,
    report_language: str,
    samples_per_bucket: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        py_bin,
        str(script_path),
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(out_dir),
        "--report",
        "--report-language",
        report_language,
        "--samples-per-bucket",
        str(samples_per_bucket),
        "--skip-similarity",
    ]
    if condition:
        cmd += ["--condition", condition]
    if task_max is not None:
        cmd += ["--task-max", str(task_max)]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _top_bucket_lines(summary_json: Dict[str, Any], top_k: int = 5) -> List[str]:
    buckets = summary_json.get("reason_buckets_global") or {}
    if not isinstance(buckets, dict):
        return []
    items = sorted(
        ((str(k), int(v)) for k, v in buckets.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return [f"{k}: {v}" for k, v in items[:top_k]]


def _glm_conclusion(
    glmm: Optional[Dict[str, str]],
    run_id: str,
    condition: Optional[str],
    episodes: int,
    task_max: Optional[int],
    bucket_map: Dict[str, Any],
) -> Dict[str, Any]:
    def _cnt(key: str) -> int:
        try:
            return int(bucket_map.get(key, 0) or 0)
        except Exception:
            return 0

    def _wrong_url_total() -> int:
        # Backward-compatible: prefer new sub-buckets, fallback to legacy single bucket.
        subtype_total = (
            _cnt("fail_finish_wrong_url_not_found")
            + _cnt("fail_finish_wrong_url_left_target")
            + _cnt("fail_finish_wrong_url_price_mismatch")
        )
        if subtype_total > 0:
            return subtype_total
        return _cnt("fail_finish_wrong_url")

    wrong_not_found = _cnt("fail_finish_wrong_url_not_found")
    wrong_left = _cnt("fail_finish_wrong_url_left_target")
    wrong_price = _cnt("fail_finish_wrong_url_price_mismatch")
    stuck = _cnt("fail_incomplete_or_stuck")
    no_progress = _cnt("fail_no_progress")
    max_steps = (
        _cnt("fail_max_steps")
        + _cnt("fail_max_steps_click_back_loop")
        + _cnt("fail_max_steps_search_repeat")
        + _cnt("fail_max_steps_target_unreachable")
    )
    loop_click_back = _cnt("fail_max_steps_click_back_loop")
    loop_search_repeat = _cnt("fail_max_steps_search_repeat")
    target_unreachable = _cnt("fail_max_steps_target_unreachable")
    finish_mismatch = _cnt("fail_finish_eval_mismatch")
    success = _cnt("success")

    denom = max(1, int(episodes))
    fallback_summary = (
        "主要失败集中在三类：未稳定到达目标详情页即结束、流程卡住或步数耗尽、"
        "以及最终答案与评测标准不一致。"
        f" 当前成功率为 {success/denom:.1%}。"
    )
    fallback_diag: List[Dict[str, str]] = []
    if stuck + no_progress > 0:
        fallback_diag.append(
            {
                "category": "流程停滞",
                "root_cause": "关键动作未完成，反复停留在同一流程或页面。",
                "confidence": "high",
            }
        )
    if _wrong_url_total() > 0:
        fallback_diag.append(
            {
                "category": "目标页定位失败",
                "root_cause": "未找到目标详情页、离开目标页后结束，或价格理解偏差导致收尾错误。",
                "confidence": "high",
            }
        )
    if max_steps + finish_mismatch > 0:
        fallback_diag.append(
            {
                "category": "收尾判定失败",
                "root_cause": "达到步数上限被截断，或最终答案与评测标准不一致。",
                "confidence": "medium",
            }
        )
    if loop_click_back + loop_search_repeat > 0:
        fallback_diag.append(
            {
                "category": "循环失败",
                "root_cause": "存在 click/back 或重复搜索词循环，导致任务不收敛。",
                "confidence": "high",
            }
        )
    if target_unreachable > 0:
        fallback_diag.append(
            {
                "category": "目标不可达",
                "root_cause": "目标item在观测页面中未出现，导致无法命中目标。",
                "confidence": "high",
            }
        )
    fallback = {
        "semantic_summary": fallback_summary,
        "failure_diagnosis": fallback_diag[:3],
    }
    if not glmm:
        return fallback
    payload = {
        "run_id": run_id,
        "condition": condition,
        "episodes_analyzed": episodes,
        "task_max": task_max,
        "reason_buckets_global": bucket_map,
        "bucket_hints_zh": {
            "fail_finish_wrong_url_not_found": "没找到目标详情页就结束",
            "fail_finish_wrong_url_left_target": "到过目标页但离开后结束",
            "fail_finish_wrong_url_price_mismatch": "价格/金额理解错误导致收尾错误",
            "fail_incomplete_or_stuck": "流程卡住，没完成关键动作",
            "fail_no_progress": "重复操作导致页面几乎无变化",
            "fail_max_steps": "达到步数上限被截断",
            "fail_max_steps_click_back_loop": "click/back 交替循环导致步数耗尽",
            "fail_max_steps_search_repeat": "重复搜索同一词导致步数耗尽",
            "fail_max_steps_target_unreachable": "目标item在可见DOM中未出现，导致不可达",
            "fail_finish_eval_mismatch": "有答案但与评测标准不一致",
        },
    }
    messages = [
        {
            "role": "system",
            "content": (
                "你是实验归因分析助手。请输出结构化诊断（中文，严格 JSON）。\n"
                "要求：\n"
                "1) 严格返回 JSON，不要 markdown，不要解释文字；\n"
                "2) JSON 结构必须为："
                '{"semantic_summary":"...","failure_diagnosis":[{"category":"...","root_cause":"...","confidence":"high|medium|low"}]}'
                "\n"
                "3) failure_diagnosis 最多 3 条，按重要性排序；\n"
                "4) 不要输出 bucket 英文名，不要建议或行动项；\n"
                "5) semantic_summary 用 2-4 句中文，聚焦失败语义。"
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        raw = _call_glm_chat(glmm, messages).strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        parsed: Optional[Dict[str, Any]] = None
        try:
            maybe = json.loads(raw)
            if isinstance(maybe, dict):
                parsed = maybe
        except Exception:
            l = raw.find("{")
            r = raw.rfind("}")
            if l >= 0 and r > l:
                try:
                    maybe = json.loads(raw[l : r + 1])
                    if isinstance(maybe, dict):
                        parsed = maybe
                except Exception:
                    parsed = None
        if not parsed:
            return fallback
        summary = str(parsed.get("semantic_summary", "") or "").strip()
        if not summary:
            summary = fallback_summary
        fd = parsed.get("failure_diagnosis")
        items: List[Dict[str, str]] = []
        if isinstance(fd, list):
            for it in fd[:3]:
                if not isinstance(it, dict):
                    continue
                category = str(it.get("category", "") or "").strip()
                root_cause = str(it.get("root_cause", "") or "").strip()
                confidence = str(it.get("confidence", "") or "").strip().lower()
                if confidence not in {"high", "medium", "low"}:
                    confidence = "medium"
                if category and root_cause:
                    items.append(
                        {
                            "category": category,
                            "root_cause": root_cause,
                            "confidence": confidence,
                        }
                    )
        if not items:
            items = fallback_diag[:3]
        return {"semantic_summary": summary, "failure_diagnosis": items}
    except Exception as e:  # noqa: BLE001
        print(f"[live-diag] WARNING: GLM conclusion failed run_id={run_id} episodes={episodes}: {type(e).__name__}: {e}")
        return fallback


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live incremental reason-diagnostics sidecar")
    p.add_argument("--run-dir", required=True, help="Run dir, e.g. results/.../<run_id>")
    p.add_argument("--condition", default=None, help="Optional condition id, e.g. phase1_dom_router_0")
    p.add_argument("--poll-secs", type=int, default=60, help="Polling interval")
    p.add_argument("--interval-episodes", type=int, default=10, help="Trigger diagnostics every N new episodes")
    p.add_argument("--py-bin", default=".venv/bin/python", help="Python bin used to run diagnostics")
    p.add_argument("--diag-script", default="scripts/analysis/analyze_reason_diagnostics.py")
    p.add_argument("--report-language", default="zh", choices=["zh", "en"])
    p.add_argument("--samples-per-bucket", type=int, default=5)
    p.add_argument("--out-root", default=None, help="Output root for incremental diagnostics (default: <run_dir>/analysis/reason_diagnostics_live)")
    p.add_argument("--glm-config", default=".auth/glm", help="Path to glm config file")
    p.add_argument("--disable-glm", action="store_true")
    p.add_argument("--ntfy-topic", default=None, help="Optional ntfy topic")
    p.add_argument("--label", default=None, help="Short label used in ntfy title, e.g. 'classifieds' (overrides condition/run_id fallback)")
    p.add_argument("--state-file", default=None, help="Optional state json file")
    p.add_argument(
        "--ntfy-cooldown-secs",
        type=int,
        default=600,
        help="Cooldown after ntfy 429 errors before next push attempt (default: 600)",
    )
    p.add_argument(
        "--retain-output-dirs",
        type=int,
        default=1,
        help="How many newest diagnostics output dirs to keep (default: 1)",
    )
    p.add_argument("--once", action="store_true", help="Run one scan and exit")
    p.add_argument(
        "--episode-diagnosis-max-cases",
        type=int,
        default=5,
        help="Max number of newly failed episodes to diagnose per trigger",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    out_root = Path(args.out_root).resolve() if args.out_root else (run_dir / "analysis" / "reason_diagnostics_live")
    out_root.mkdir(parents=True, exist_ok=True)
    diag_script_path = Path(args.diag_script).resolve()
    if not diag_script_path.exists():
        raise SystemExit(f"diag_script not found: {diag_script_path}")

    state_file = Path(args.state_file).resolve() if args.state_file else None

    # Exclusive flock: only one sidecar per state file may run at a time.
    # A second instance (manually started or orphaned) exits immediately.
    _lock_fd = None
    if state_file:
        _lock_path = state_file.with_suffix(".lock")
        _lock_path.parent.mkdir(parents=True, exist_ok=True)
        _lock_fd = open(_lock_path, "w")  # noqa: SIM115
        try:
            fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            _lock_fd.write(str(__import__("os").getpid()))
            _lock_fd.flush()
        except BlockingIOError:
            print(
                f"[live-diag] Another sidecar instance holds the lock ({_lock_path}). "
                "Exiting to avoid competition."
            )
            _lock_fd.close()
            return 1

    state = _load_state(state_file)
    # Per-condition trigger counts (backward compat: migrate legacy scalar).
    raw_ltc_map = state.get("last_trigger_count_by_condition", {})
    last_trigger_count_by_condition: Dict[str, int] = {}
    if isinstance(raw_ltc_map, dict):
        for k, v in raw_ltc_map.items():
            tv = _to_int(v)
            if tv is not None:
                last_trigger_count_by_condition[str(k)] = tv
    # Backward compat: old scalar last_trigger_count → assign to --condition if given.
    legacy_ltc = _to_int(state.get("last_trigger_count"))
    if legacy_ltc is not None and args.condition and args.condition not in last_trigger_count_by_condition:
        last_trigger_count_by_condition[args.condition] = legacy_ltc
    ntfy_cooldown_until = float(state.get("ntfy_cooldown_until", 0) or 0.0)
    pending_ntfy_queue: List[Dict[str, Any]] = []
    raw_queue = state.get("pending_ntfy_queue")
    if isinstance(raw_queue, list):
        for item in raw_queue:
            if isinstance(item, dict):
                pending_ntfy_queue.append(item)
    # Backward compatibility: old single pending message.
    old_pending = state.get("pending_ntfy")
    if isinstance(old_pending, dict):
        if not pending_ntfy_queue:
            pending_ntfy_queue.append(old_pending)
    raw_task_max_map = state.get("last_task_max_by_condition", {})
    last_task_max_by_condition: Dict[str, int] = {}
    if isinstance(raw_task_max_map, dict):
        for k, v in raw_task_max_map.items():
            cid = str(k or "").strip()
            tv = _to_int(v)
            if cid and tv is not None:
                last_task_max_by_condition[cid] = tv
    # Backward compatibility with old single-value state.
    legacy_last_task_max = _to_int(state.get("last_task_max"))
    if legacy_last_task_max is not None and args.condition:
        last_task_max_by_condition.setdefault(str(args.condition), legacy_last_task_max)

    # Track the max task_id per condition for which ntfy was successfully sent.
    # Used on startup to detect un-pushed episodes and queue backfill batches.
    last_ntfy_task_max_by_condition: Dict[str, int] = {}
    raw_ntfy_max = state.get("last_ntfy_task_max_by_condition", {})
    if isinstance(raw_ntfy_max, dict):
        for k, v in raw_ntfy_max.items():
            cid = str(k or "").strip()
            tv = _to_int(v)
            if cid and tv is not None:
                last_ntfy_task_max_by_condition[cid] = tv

    glmm: Optional[Dict[str, str]] = None
    if not args.disable_glm:
        cfg_path = Path(args.glm_config).resolve()
        if cfg_path.exists():
            try:
                glmm = _load_glm_config(cfg_path)
                print(f"[live-diag] GLM enabled: model={glmm['model']} endpoint={glmm['endpoint']}")
            except Exception as e:  # noqa: BLE001
                print(f"[live-diag] GLM config invalid ({cfg_path}): {e}. fallback only.")
        else:
            print(f"[live-diag] GLM config not found ({cfg_path}). fallback only.")
    else:
        print("[live-diag] GLM disabled.")

    print(
        f"[live-diag] watch run_id={run_dir.name} condition={args.condition or '*'} "
        f"poll={args.poll_secs}s interval={args.interval_episodes}"
    )

    # GLM retry queue: when GLM API fails, failed cases are queued here for
    # periodic retry (every GLM_RETRY_INTERVAL_S seconds).  Each entry:
    #   {condition, out_dir, failed_cases, queued_at, last_retry_at, attempt}
    # Persisted in state["glm_retry_queue"].
    glm_retry_queue: List[Dict[str, Any]] = []
    _raw_rq = state.get("glm_retry_queue")
    if isinstance(_raw_rq, list):
        for _qi in _raw_rq:
            if isinstance(_qi, dict) and _qi.get("condition") and _qi.get("failed_cases"):
                glm_retry_queue.append(_qi)
    if glm_retry_queue:
        print(f"[live-diag] restored {len(glm_retry_queue)} GLM retry queue entries from state")
    GLM_RETRY_INTERVAL_S = 300  # retry every 5 minutes

    # ── Startup backfill: detect un-pushed episodes and queue them ──
    # Compare last_ntfy_task_max vs last_task_max to find episodes that were
    # analyzed but never successfully pushed via ntfy.  Split into batches of
    # ~interval_episodes and queue as retry entries so each batch gets its own
    # ntfy push.
    if glmm and args.ntfy_topic and not glm_retry_queue:
        _last_out_dir_str = state.get("last_output_dir")
        _last_out_dir = Path(_last_out_dir_str) if _last_out_dir_str else None
        _backfill_csv = _last_out_dir / "episode_reason_rows.csv" if _last_out_dir and _last_out_dir.exists() else None
        if _backfill_csv and _backfill_csv.exists():
            for _bf_cid, _bf_analyzed_max in last_task_max_by_condition.items():
                if args.condition and _bf_cid != args.condition:
                    continue
                _bf_pushed_max = last_ntfy_task_max_by_condition.get(_bf_cid, -1)
                if _bf_pushed_max >= _bf_analyzed_max:
                    continue  # no gap
                # Extract all failed cases in the un-pushed range
                _bf_cases = _extract_new_failed_cases(
                    episode_rows_csv=_backfill_csv,
                    prev_task_max_by_condition={_bf_cid: _bf_pushed_max},
                    condition_filter=_bf_cid,
                    max_cases=999,  # get all, we'll chunk them
                )
                if not _bf_cases:
                    print(f"[live-diag] backfill: no un-pushed failed cases for {_bf_cid} (pushed_max={_bf_pushed_max}, analyzed_max={_bf_analyzed_max})")
                    continue
                # Sort by task_id ascending for chronological batching
                _bf_cases.sort(key=lambda x: int(x.get("task_id") or 0))
                _batch_size = max(1, int(args.interval_episodes))
                _bf_batches: List[List[Dict[str, Any]]] = []
                for _i in range(0, len(_bf_cases), _batch_size):
                    _bf_batches.append(_bf_cases[_i : _i + _batch_size])
                print(
                    f"[live-diag] backfill: {_bf_cid} has {len(_bf_cases)} un-pushed failed cases "
                    f"(pushed_max={_bf_pushed_max}, analyzed_max={_bf_analyzed_max}), "
                    f"queuing {len(_bf_batches)} batches of ~{_batch_size}"
                )
                for _bi, _batch in enumerate(_bf_batches):
                    _task_ids_in_batch = [c.get("task_id") for c in _batch]
                    glm_retry_queue.append({
                        "condition": _bf_cid,
                        "out_dir": str(_last_out_dir),
                        "failed_cases": _batch,
                        "queued_at": time.time(),
                        "last_retry_at": 0,  # eligible for immediate retry
                        "attempt": 0,
                        "backfill": True,
                        "backfill_task_range": f"{min(_task_ids_in_batch)}-{max(_task_ids_in_batch)}",
                    })
                state["glm_retry_queue"] = glm_retry_queue
                _save_state(state_file, state)

    while True:
        # ── GLM retry queue processing ──
        if glm_retry_queue and glmm:
            _new_queue: List[Dict[str, Any]] = []
            for _qi in glm_retry_queue:
                _elapsed = time.time() - float(_qi.get("last_retry_at") or 0)
                if _elapsed < GLM_RETRY_INTERVAL_S:
                    _new_queue.append(_qi)
                    continue
                _qi_cond = str(_qi["condition"])
                _qi_out_dir = Path(str(_qi["out_dir"]))
                _qi_attempt = int(_qi.get("attempt") or 0) + 1
                _qi_cases = _qi.get("failed_cases") or []
                if not _qi_cases:
                    continue
                print(f"[live-diag] GLM retry: condition={_qi_cond} cases={len(_qi_cases)} attempt={_qi_attempt}")
                _retry_results = _glm_episode_diagnosis(glmm=glmm, cases=_qi_cases, run_dir=run_dir)
                _has_any_failed = any(r.get("category") == "GLM_FAILED" for r in _retry_results)
                if _has_any_failed:
                    # Any degraded → keep entire batch in queue, no ntfy
                    _n_still = sum(1 for r in _retry_results if r.get("category") == "GLM_FAILED")
                    print(f"[live-diag] GLM retry: {_n_still}/{len(_qi_cases)} still failed, entire batch stays in queue (attempt={_qi_attempt})")
                    _qi["attempt"] = _qi_attempt
                    _qi["last_retry_at"] = time.time()
                    _new_queue.append(_qi)
                else:
                    # All succeeded → append to report + send ntfy
                    _is_backfill = bool(_qi.get("backfill"))
                    _bf_range = str(_qi.get("backfill_task_range", "")) if _is_backfill else ""
                    _tag = f"backfill tasks {_bf_range}" if _bf_range else f"retry attempt={_qi_attempt}"
                    print(f"[live-diag] GLM {_tag} OK: all {len(_qi_cases)} succeeded")
                    if _qi_out_dir.exists():
                        _report_path = _qi_out_dir / "failure_report.md"
                        if _report_path.exists():
                            _all_lines = _format_episode_report_lines(_retry_results)
                            if _all_lines:
                                _old = _report_path.read_text(encoding="utf-8")
                                _section = f"## GLM {'Backfill' if _is_backfill else 'Retry'} ({_tag})"
                                _patch = f"\n\n{_section}\n" + "\n".join(_all_lines)
                                _report_path.write_text(_old + _patch, encoding="utf-8")
                    if args.ntfy_topic:
                        _all_lines = _format_episode_report_lines(_retry_results)
                        if _all_lines:
                            _title_tag = f"tasks {_bf_range}" if _bf_range else f"retry OK ({len(_retry_results)})"
                            _ok, _, _ = _post_ntfy(
                                topic=args.ntfy_topic,
                                title=f"P79 [{args.label or _qi_cond}] {_title_tag}",
                                body=f"condition={_qi_cond}\n" + "\n".join(_all_lines),
                                priority="default",
                            )
                    # Update ntfy task max for backfill batches
                    if _is_backfill and _qi_cases:
                        _batch_max_tid = max(int(c.get("task_id") or -1) for c in _qi_cases)
                        _prev_ntfy_max = last_ntfy_task_max_by_condition.get(_qi_cond, -1)
                        if _batch_max_tid > _prev_ntfy_max:
                            last_ntfy_task_max_by_condition[_qi_cond] = _batch_max_tid
                            state["last_ntfy_task_max_by_condition"] = last_ntfy_task_max_by_condition
            glm_retry_queue = _new_queue
            # Persist queue (bounded)
            if len(glm_retry_queue) > 10:
                glm_retry_queue = glm_retry_queue[-10:]
            state["glm_retry_queue"] = glm_retry_queue if glm_retry_queue else []
            _save_state(state_file, state)

        # Determine which conditions need a trigger — per-condition episode counting.
        if args.condition:
            # Single condition mode: only check the specified condition.
            _cond_counts = {args.condition: _episode_count(run_dir, args.condition)}
        else:
            _cond_counts = _episode_count_by_condition(run_dir)

        # Find conditions with enough new episodes to trigger.
        _triggered_conditions: List[str] = []
        for _cid, _cnt in _cond_counts.items():
            _prev = last_trigger_count_by_condition.get(_cid, 0)
            if _cnt > 0 and (_cnt - _prev) >= int(args.interval_episodes):
                _triggered_conditions.append(_cid)

        for _trigger_cond in _triggered_conditions:
            _cond_done = _cond_counts[_trigger_cond]
            task_max = _max_task_id(run_dir, _trigger_cond)
            tag = f"{_trigger_cond}_upto_{task_max:04d}" if task_max is not None else f"{_trigger_cond}_count_{_cond_done:04d}"
            out_dir = out_root / tag
            print(f"[live-diag] trigger: condition={_trigger_cond} episodes={_cond_done}, last={last_trigger_count_by_condition.get(_trigger_cond, 0)}, out={out_dir}")

            proc = _run_diagnostics(
                py_bin=args.py_bin,
                script_path=diag_script_path,
                run_dir=run_dir,
                condition=_trigger_cond,
                task_max=task_max,
                out_dir=out_dir,
                report_language=args.report_language,
                samples_per_bucket=int(args.samples_per_bucket),
            )
            if proc.returncode != 0:
                print(f"[live-diag] diagnostics failed rc={proc.returncode} condition={_trigger_cond}")
                if proc.stdout.strip():
                    print(proc.stdout.strip()[-1000:])
                if proc.stderr.strip():
                    print(proc.stderr.strip()[-1000:])
                # Prevent rapid-fire retrigger loops when diagnostics repeatedly fail.
                last_trigger_count_by_condition[_trigger_cond] = _cond_done
                state["last_trigger_count_by_condition"] = last_trigger_count_by_condition
                state["last_error"] = f"diagnostics_failed_rc={proc.returncode}_cond={_trigger_cond}"
                state["last_error_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                _save_state(state_file, state)
            else:
                summary_path = out_dir / "reason_diagnostics_summary.json"
                summary_json = _safe_load_json(summary_path) or {}
                bucket_map = summary_json.get("reason_buckets_global") or {}
                # Compute success_rate for the triggered condition from CSV.
                _csv_rows = _read_csv_rows(out_dir / "episode_reason_rows.csv")
                _cond_rows = [
                    r for r in _csv_rows
                    if str(r.get("condition_id", "") or "") == _trigger_cond
                ]
                episodes = len(_cond_rows)
                success_count = sum(
                    1 for r in _cond_rows if _to_optional_bool(r.get("success")) is True
                )
                success_rate = (success_count / episodes) if episodes > 0 else 0.0

                # Determine the active mode(s) for this trigger
                from collections import defaultdict as _defaultdict
                _mode_stats: Dict[str, Dict[str, int]] = _defaultdict(lambda: {"total": 0, "success": 0})
                _active_mode_stats: Dict[str, Dict[str, int]] = _defaultdict(lambda: {"total": 0, "success": 0})
                for _r in _cond_rows:
                    _mode = str(_r.get("observation_mode", "") or "?").strip() or "?"
                    _cid = str(_r.get("condition_id", "") or "").strip()
                    _tid = _to_int(_r.get("task_id"))
                    _ok = _to_optional_bool(_r.get("success")) is True
                    _mode_stats[_mode]["total"] += 1
                    if _ok:
                        _mode_stats[_mode]["success"] += 1
                    # "active" = new episodes since last trigger
                    _prev_max = last_task_max_by_condition.get(_cid, -1)
                    if _tid is not None and _tid > _prev_max:
                        _active_mode_stats[_mode]["total"] += 1
                        if _ok:
                            _active_mode_stats[_mode]["success"] += 1
                # Single success line: show active mode(s) with cumulative totals
                _shown_stats = _active_mode_stats if _active_mode_stats else _mode_stats
                _success_line = "  ".join(
                    f"{m}: {_mode_stats[m]['success']}/{_mode_stats[m]['total']} ({_mode_stats[m]['success']/_mode_stats[m]['total']:.1%})"
                    for m in sorted(_shown_stats)
                ) or f"{success_count}/{episodes}"

                top_lines = _top_bucket_lines(summary_json, top_k=5)
                _glm_episodes = episodes
                _glm_bucket_map = bucket_map if isinstance(bucket_map, dict) else {}

                glm_diag = _glm_conclusion(
                    glmm=glmm,
                    run_id=run_dir.name,
                    condition=_trigger_cond,
                    episodes=_glm_episodes,
                    task_max=task_max,
                    bucket_map=_glm_bucket_map,
                )
                glm_text = str(glm_diag.get("semantic_summary", "") or "").strip()
                failure_diag = glm_diag.get("failure_diagnosis")
                if not isinstance(failure_diag, list):
                    failure_diag = []
                new_failed_cases = _extract_new_failed_cases(
                    episode_rows_csv=out_dir / "episode_reason_rows.csv",
                    prev_task_max_by_condition=last_task_max_by_condition,
                    condition_filter=_trigger_cond,
                    max_cases=int(args.episode_diagnosis_max_cases),
                )
                episode_diagnoses = _glm_episode_diagnosis(glmm=glmm, cases=new_failed_cases, run_dir=run_dir)
                # If ANY case has GLM_FAILED, queue the ENTIRE batch for retry
                # and skip ntfy — user should never see degraded/partial results.
                _has_glm_failed = any(
                    d.get("category") == "GLM_FAILED" for d in episode_diagnoses
                )
                if _has_glm_failed:
                    glm_retry_queue.append({
                        "condition": _trigger_cond,
                        "out_dir": str(out_dir),
                        "failed_cases": new_failed_cases,
                        "queued_at": time.time(),
                        "last_retry_at": time.time(),
                        "attempt": 0,
                    })
                    _n_failed = sum(1 for d in episode_diagnoses if d.get("category") == "GLM_FAILED")
                    print(f"[live-diag] {_n_failed}/{len(new_failed_cases)} GLM-failed; entire batch queued for retry, ntfy deferred (condition={_trigger_cond})")
                    # Skip ntfy push — will be sent when retry succeeds for all
                    episode_diag_lines: List[str] = []
                else:
                    episode_diag_lines = _format_episode_report_lines(episode_diagnoses)
                msg_lines = [
                    f"run_id={run_dir.name}",
                    f"condition={_trigger_cond}",
                    f"episodes={episodes}",
                    f"success={_success_line}",
                    f"task_max={task_max}",
                    "top_buckets:",
                    *[f"- {x}" for x in top_lines],
                    "glm_summary:",
                    glm_text,
                    "failure_diagnosis:",
                    *[
                        f"- [{str(x.get('confidence', 'medium')).lower()}] {str(x.get('category', '')).strip()}: {str(x.get('root_cause', '')).strip()}"
                        for x in failure_diag[:3]
                        if isinstance(x, dict)
                    ],
                    "per_task_failure_report:",
                    *(episode_diag_lines or ["- 本轮新增任务未发现失败"]),
                    f"report={out_dir / 'failure_report.md'}",
                ]
                msg = "\n".join(msg_lines)
                print(f"[live-diag][OK]\n{msg}")
                if args.ntfy_topic and not _has_glm_failed:
                    _cond_label = args.label or _trigger_cond or run_dir.name
                    title = f"P79 [{_cond_label}] {_success_line}"
                    ntfy_body = "\n".join(
                        [
                            f"condition={_trigger_cond}",
                            f"success={_success_line}",
                            "per_task_failure_report:",
                            *(episode_diag_lines or ["- 本轮新增任务未发现失败"]),
                        ]
                    )
                    current_msg = {
                        "title": title,
                        "body": ntfy_body,
                        "priority": "default",
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    def _queue_contains(msg: Dict[str, Any]) -> bool:
                        t = str(msg.get("title", "") or "")
                        b = str(msg.get("body", "") or "")
                        for it in pending_ntfy_queue:
                            if str(it.get("title", "") or "") == t and str(it.get("body", "") or "") == b:
                                return True
                        return False
                    now = time.time()
                    if now < ntfy_cooldown_until:
                        remain = int(ntfy_cooldown_until - now)
                        print(f"[live-diag] ntfy cooldown active, skip push ({remain}s left)")
                        if not _queue_contains(current_msg):
                            pending_ntfy_queue.append(current_msg)
                    else:
                        if pending_ntfy_queue:
                            # Always replay oldest pending first.
                            replay = pending_ntfy_queue[0]
                            ok, status_code, _ = _post_ntfy(
                                topic=args.ntfy_topic,
                                title=str(replay.get("title", title) or title),
                                body=str(replay.get("body", "") or ""),
                                priority=str(replay.get("priority", "default") or "default"),
                            )
                            if ok:
                                print("[live-diag] pending ntfy message delivered.")
                                pending_ntfy_queue.pop(0)
                                ntfy_cooldown_until = 0.0
                                state.pop("ntfy_cooldown_until", None)
                                if not pending_ntfy_queue:
                                    if episode_diag_lines:
                                        ok2, status_code2, _ = _post_ntfy(
                                            topic=args.ntfy_topic,
                                            title=title,
                                            body=ntfy_body,
                                            priority="default",
                                        )
                                        if not ok2:
                                            if not _queue_contains(current_msg):
                                                pending_ntfy_queue.append(current_msg)
                                            if status_code2 == 429:
                                                ntfy_cooldown_until = time.time() + max(60, int(args.ntfy_cooldown_secs))
                                                state["ntfy_cooldown_until"] = ntfy_cooldown_until
                                                print(
                                                    "[live-diag] ntfy 429 received on current message; "
                                                    f"cooldown set to {int(args.ntfy_cooldown_secs)}s"
                                                )
                                    else:
                                        if not _queue_contains(current_msg):
                                            pending_ntfy_queue.append(current_msg)
                                        print("[live-diag] Deferring current ntfy (no new episode diagnoses).")
                                else:
                                    if not _queue_contains(current_msg):
                                        pending_ntfy_queue.append(current_msg)
                            elif status_code == 429:
                                ntfy_cooldown_until = time.time() + max(60, int(args.ntfy_cooldown_secs))
                                state["ntfy_cooldown_until"] = ntfy_cooldown_until
                                print(
                                    "[live-diag] ntfy 429 received while replaying pending; "
                                    f"cooldown set to {int(args.ntfy_cooldown_secs)}s"
                                )
                                if not _queue_contains(current_msg):
                                    pending_ntfy_queue.append(current_msg)
                            else:
                                print("[live-diag] pending ntfy replay failed; will retry later.")
                                if not _queue_contains(current_msg):
                                    pending_ntfy_queue.append(current_msg)
                        else:
                            ok, status_code, _ = _post_ntfy(
                                topic=args.ntfy_topic,
                                title=title,
                                body=ntfy_body,
                                priority="default",
                            )
                            if not ok:
                                if not _queue_contains(current_msg):
                                    pending_ntfy_queue.append(current_msg)
                                if status_code == 429:
                                    ntfy_cooldown_until = time.time() + max(60, int(args.ntfy_cooldown_secs))
                                    state["ntfy_cooldown_until"] = ntfy_cooldown_until
                                    print(
                                        "[live-diag] ntfy 429 received; "
                                        f"cooldown set to {int(args.ntfy_cooldown_secs)}s"
                                    )
                            elif ok:
                                ntfy_cooldown_until = 0.0
                                state.pop("ntfy_cooldown_until", None)
                    # Persist queue (bounded) + legacy mirror key.
                    if len(pending_ntfy_queue) > 12:
                        pending_ntfy_queue = pending_ntfy_queue[-12:]
                    if pending_ntfy_queue:
                        state["pending_ntfy_queue"] = pending_ntfy_queue
                        state["pending_ntfy"] = pending_ntfy_queue[0]
                    else:
                        state.pop("pending_ntfy_queue", None)
                        state.pop("pending_ntfy", None)
                deleted = _cleanup_old_output_dirs(out_root=out_root, keep_dir=out_dir, retain_dirs=int(args.retain_output_dirs))
                if deleted > 0:
                    print(f"[live-diag] cleaned old outputs: deleted={deleted}, keep={out_dir}")
                last_trigger_count_by_condition[_trigger_cond] = _cond_done
                state["last_trigger_count_by_condition"] = last_trigger_count_by_condition
                state["last_output_dir"] = str(out_dir)
                state["last_task_max"] = task_max
                current_task_max_by_condition = _task_max_by_condition_from_episode_csv(out_dir / "episode_reason_rows.csv")
                for cid, mx in current_task_max_by_condition.items():
                    prev = last_task_max_by_condition.get(cid)
                    if prev is None or mx > prev:
                        last_task_max_by_condition[cid] = mx
                state["last_task_max_by_condition"] = last_task_max_by_condition
                # Track ntfy task max: when push succeeded (no GLM failure, no pending queue),
                # record the task_max so backfill can detect un-pushed episodes on next restart.
                if not _has_glm_failed and not pending_ntfy_queue:
                    _ntfy_max = current_task_max_by_condition.get(_trigger_cond)
                    if _ntfy_max is not None:
                        _prev_ntfy = last_ntfy_task_max_by_condition.get(_trigger_cond, -1)
                        if _ntfy_max > _prev_ntfy:
                            last_ntfy_task_max_by_condition[_trigger_cond] = _ntfy_max
                            state["last_ntfy_task_max_by_condition"] = last_ntfy_task_max_by_condition
                state.pop("last_error", None)
                state.pop("last_error_at", None)
                state["glm_retry_queue"] = glm_retry_queue if glm_retry_queue else []
                _save_state(state_file, state)

        if args.once:
            break
        time.sleep(max(1, int(args.poll_secs)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
