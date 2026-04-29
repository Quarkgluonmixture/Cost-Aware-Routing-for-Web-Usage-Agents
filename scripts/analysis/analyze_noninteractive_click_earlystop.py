#!/usr/bin/env python3
"""Per-run diagnostic; not part of the 4-layer evidence framework.

分析 B0/B1 classifieds SoM 模式中，agent 点击非交互元素导致早停的比例。

对每个 SoM episode 的每一步：
  - 读 artifacts/<task>/step_NNN/observation_som.txt 获取 element_id → role 映射
  - 读 steps JSONL 获取 action（action_type=click 时的 element_id）
  - 判断该 element_id 对应的 role 是否为非交互元素

统计：
  - 总 episode 数
  - 包含至少一次非交互点击的 episode 数 + 比例
  - 早停 episode 数（cycle early stop 或 URL stuck，判据: !agent_finished && steps<30）
  - 截断 episode 数（steps==30 && !agent_finished）
  - 早停且包含非交互点击的 episode 数 + 比例
  - 早停 episode 中最后 N 步全为非交互点击的比例
"""

import json
import glob
import os
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

# ── 角色分类 ──────────────────────────────────────────────

INTERACTIVE_ROLES = {
    "link", "button", "searchbox", "textbox", "combobox", "checkbox",
    "radio", "menuitem", "tab", "DisclosureTriangle", "switch",
    "spinbutton", "slider", "option", "listbox", "menu",
    "menuitemcheckbox", "menuitemradio", "progressbar", "scrollbar",
    "treeitem", "gridcell", "columnheader", "rowheader", "row",
}

NON_INTERACTIVE_ROLES = {
    "StaticText", "heading", "sectionheader", "article", "time",
    "image", "complementary", "separator", "generic", "RootWebArea",
    "group", "banner", "contentinfo", "main", "navigation", "region",
    "list", "listitem", "paragraph", "blockquote", "figure",
    "figcaption", "table", "cell", "status", "tooltip", "alert",
    "dialog", "form", "Section", "none", "presentation", "math",
    "Iframe", "iframe", "webview", "document", "application",
    "directory", "feed", "log", "marquee", "note", "tabpanel",
    "timer", "toolbar", "tree", "treegrid",
}


def parse_som_marks(som_text: str) -> dict:
    """解析 observation_som.txt，返回 {element_id: role} 映射。"""
    id_to_role = {}
    # 格式: [id=N] role 'label' ...
    for line in som_text.split("\n"):
        m = re.match(r"\[id=(\d+)\]\s+(\S+)", line)
        if m:
            eid = int(m.group(1))
            role = m.group(2)
            id_to_role[eid] = role
    return id_to_role


def extract_click_element_id(action) -> int | None:
    """从 action 字段提取 click 的 element_id，返回 None 如果非 click 或无法解析。"""
    if isinstance(action, dict):
        if action.get("action_type") == "click":
            eid = action.get("element_id")
            if eid is not None:
                return int(eid)
    elif isinstance(action, str):
        # 尝试解析字符串格式 "click [id]"
        m = re.match(r"click\s*\[(\d+)\]", action, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def is_interactive(role: str) -> bool:
    """判断 role 是否为可交互元素。"""
    if role in INTERACTIVE_ROLES:
        return True
    if role in NON_INTERACTIVE_ROLES:
        return False
    # 未知 role 默认为非交互，但记录
    return False


def analyze_condition(condition_dir: str, label: str):
    """分析单个 condition 目录。"""
    episodes_dir = os.path.join(condition_dir, "episodes")
    artifacts_dir = os.path.join(condition_dir, "artifacts")

    # 收集所有 summary 文件
    summary_files = sorted(glob.glob(os.path.join(episodes_dir, "*_summary_v2.json")))
    if not summary_files:
        print(f"[{label}] 未找到 summary 文件")
        return None

    results = {
        "label": label,
        "total_episodes": 0,
        "episodes_with_noninteractive_click": 0,
        "agent_finished_episodes": 0,
        "truncated_episodes": 0,  # steps>=30, !agent_finished
        "early_stop_episodes": 0,  # steps<30, !agent_finished (cycle/url_stuck)
        "early_stop_with_noninteractive_click": 0,
        "early_stop_last_N_all_noninteractive": 0,
        "truncated_with_noninteractive_click": 0,
        "total_click_steps": 0,
        "total_noninteractive_clicks": 0,
        "noninteractive_role_counter": Counter(),
        "unknown_roles_clicked": Counter(),
        "early_stop_tasks_detail": [],
        # 按终止类型分的非交互点击率
        "finished_noninteractive_click_eps": 0,
    }

    for sf in summary_files:
        with open(sf) as f:
            summary = json.load(f)

        task_id = summary["task_id"]
        steps_count = summary["steps"]
        agent_finished = summary.get("agent_finished", False)

        results["total_episodes"] += 1

        # 判断终止类型
        if agent_finished:
            termination = "agent_finished"
            results["agent_finished_episodes"] += 1
        elif steps_count >= 30:
            termination = "truncated"
            results["truncated_episodes"] += 1
        else:
            termination = "early_stop"
            results["early_stop_episodes"] += 1

        # 读取 steps
        steps_file = sf.replace("_summary_v2.json", "_steps_v2.jsonl")
        if not os.path.exists(steps_file):
            continue

        with open(steps_file) as f:
            step_records = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        step_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # 分析每步的点击
        task_artifact_dir = os.path.join(artifacts_dir, f"classifieds_task_{task_id}")
        has_noninteractive_click = False
        noninteractive_click_steps = []  # (step_idx, role)
        total_clicks_this_ep = 0

        for step in step_records:
            step_idx = step.get("step_idx", 0)
            action_type = step.get("action_type", "")
            action = step.get("action", {})

            # 只关心 click 动作
            if action_type != "click":
                continue

            eid = extract_click_element_id(action)
            if eid is None:
                continue

            total_clicks_this_ep += 1
            results["total_click_steps"] += 1

            # 读取对应步骤的 observation_som.txt
            som_path = os.path.join(task_artifact_dir, f"step_{step_idx:03d}", "observation_som.txt")
            if not os.path.exists(som_path):
                continue

            with open(som_path) as sf2:
                som_text = sf2.read()
            id_to_role = parse_som_marks(som_text)

            if eid in id_to_role:
                role = id_to_role[eid]
                if not is_interactive(role):
                    has_noninteractive_click = True
                    noninteractive_click_steps.append((step_idx, role))
                    results["total_noninteractive_clicks"] += 1
                    results["noninteractive_role_counter"][role] += 1
                    if role not in INTERACTIVE_ROLES and role not in NON_INTERACTIVE_ROLES:
                        results["unknown_roles_clicked"][role] += 1
            else:
                # element_id 不在 SoM marks 中（可能是幻觉 ID）
                has_noninteractive_click = True
                noninteractive_click_steps.append((step_idx, f"PHANTOM_ID_{eid}"))
                results["total_noninteractive_clicks"] += 1
                results["noninteractive_role_counter"][f"PHANTOM_ID"] += 1

        if has_noninteractive_click:
            results["episodes_with_noninteractive_click"] += 1

        # 按终止类型统计
        if termination == "early_stop":
            if has_noninteractive_click:
                results["early_stop_with_noninteractive_click"] += 1

            # 检查最后 N 步是否全是非交互点击
            # 策略：检查最后 3 步中的 click 动作是否全部为非交互
            last_n = 3
            last_steps = step_records[-last_n:] if len(step_records) >= last_n else step_records
            last_click_roles = []
            for step in last_steps:
                if step.get("action_type") != "click":
                    continue
                eid = extract_click_element_id(step.get("action", {}))
                if eid is None:
                    continue
                sidx = step.get("step_idx", 0)
                som_path = os.path.join(task_artifact_dir, f"step_{sidx:03d}", "observation_som.txt")
                if os.path.exists(som_path):
                    with open(som_path) as sf2:
                        id_to_role = parse_som_marks(sf2.read())
                    if eid in id_to_role:
                        last_click_roles.append(is_interactive(id_to_role[eid]))
                    else:
                        last_click_roles.append(False)  # phantom ID = non-interactive
                else:
                    last_click_roles.append(None)

            # 如果最后 N 步有 click 且全部为非交互
            if last_click_roles and all(r is False for r in last_click_roles):
                results["early_stop_last_N_all_noninteractive"] += 1

            results["early_stop_tasks_detail"].append({
                "task_id": task_id,
                "steps": steps_count,
                "has_noninteractive_click": has_noninteractive_click,
                "noninteractive_clicks": len(noninteractive_click_steps),
                "total_clicks": total_clicks_this_ep,
                "last_N_all_noninteractive": bool(last_click_roles and all(r is False for r in last_click_roles)),
                "noninteractive_roles": [r for _, r in noninteractive_click_steps],
            })

        elif termination == "truncated":
            if has_noninteractive_click:
                results["truncated_with_noninteractive_click"] += 1

        elif termination == "agent_finished":
            if has_noninteractive_click:
                results["finished_noninteractive_click_eps"] += 1

    return results


def print_results(r):
    """打印单个 condition 的分析结果。"""
    label = r["label"]
    total = r["total_episodes"]
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  总 episode 数:                        {total}")
    print(f"  Agent 正常完成:                       {r['agent_finished_episodes']} ({r['agent_finished_episodes']/total*100:.1f}%)")
    print(f"  截断 (30 步上限):                     {r['truncated_episodes']} ({r['truncated_episodes']/total*100:.1f}%)")
    print(f"  早停 (cycle/url_stuck):               {r['early_stop_episodes']} ({r['early_stop_episodes']/total*100:.1f}%)")

    print(f"\n  --- 非交互点击统计 ---")
    print(f"  总 click 步数:                        {r['total_click_steps']}")
    print(f"  非交互 click 步数:                    {r['total_noninteractive_clicks']} ({r['total_noninteractive_clicks']/max(r['total_click_steps'],1)*100:.1f}%)")
    print(f"  含非交互点击的 episode:               {r['episodes_with_noninteractive_click']} ({r['episodes_with_noninteractive_click']/total*100:.1f}%)")

    print(f"\n  --- 早停 episode 分析 ---")
    es = r["early_stop_episodes"]
    if es > 0:
        print(f"  早停且含非交互点击:                   {r['early_stop_with_noninteractive_click']} ({r['early_stop_with_noninteractive_click']/es*100:.1f}%)")
        print(f"  早停且最后 3 步 click 全为非交互:     {r['early_stop_last_N_all_noninteractive']} ({r['early_stop_last_N_all_noninteractive']/es*100:.1f}%)")
    else:
        print(f"  （无早停 episode）")

    print(f"\n  --- 截断 episode 分析 ---")
    tr = r["truncated_episodes"]
    if tr > 0:
        print(f"  截断且含非交互点击:                   {r['truncated_with_noninteractive_click']} ({r['truncated_with_noninteractive_click']/tr*100:.1f}%)")

    print(f"\n  --- 正常完成 episode 分析 ---")
    af = r["agent_finished_episodes"]
    if af > 0:
        print(f"  正常完成且含非交互点击:               {r['finished_noninteractive_click_eps']} ({r['finished_noninteractive_click_eps']/af*100:.1f}%)")

    print(f"\n  --- 被点击的非交互 role 分布 ---")
    for role, count in r["noninteractive_role_counter"].most_common(15):
        print(f"    {role:30s} {count:>5d}")

    if r["unknown_roles_clicked"]:
        print(f"\n  --- 未分类 role（被归入非交互） ---")
        for role, count in r["unknown_roles_clicked"].most_common():
            print(f"    {role:30s} {count:>5d}")


def print_comparison(r_b0, r_b1):
    """打印 B0 vs B1 对比表格。"""
    print(f"\n{'='*80}")
    print(f"  B0 vs B1 对比表格（SoM classifieds）")
    print(f"{'='*80}")

    def fmt(val, total, show_pct=True):
        if show_pct and total > 0:
            return f"{val} ({val/total*100:.1f}%)"
        return str(val)

    rows = [
        ("总 episode 数", r_b0["total_episodes"], r_b1["total_episodes"], False),
        ("Agent 正常完成", r_b0["agent_finished_episodes"], r_b1["agent_finished_episodes"], True),
        ("截断 (30 步)", r_b0["truncated_episodes"], r_b1["truncated_episodes"], True),
        ("早停 (cycle/stuck)", r_b0["early_stop_episodes"], r_b1["early_stop_episodes"], True),
        ("", None, None, None),
        ("总 click 步数", r_b0["total_click_steps"], r_b1["total_click_steps"], False),
        ("非交互 click", r_b0["total_noninteractive_clicks"], r_b1["total_noninteractive_clicks"], True),
        ("含非交互 click 的 ep", r_b0["episodes_with_noninteractive_click"], r_b1["episodes_with_noninteractive_click"], True),
        ("", None, None, None),
        ("早停含非交互 click", r_b0["early_stop_with_noninteractive_click"], r_b1["early_stop_with_noninteractive_click"], True),
        ("早停末 3 步全非交互", r_b0["early_stop_last_N_all_noninteractive"], r_b1["early_stop_last_N_all_noninteractive"], True),
    ]

    header = f"  {'指标':<28s} {'B0 (235B API)':>20s}  {'B1 (4B local)':>20s}"
    print(header)
    print(f"  {'-'*28} {'-'*20}  {'-'*20}")

    for row in rows:
        if row[1] is None:
            print()
            continue
        metric, v0, v1, use_pct = row
        if use_pct:
            # 用各自的总量做分母
            if "早停" in metric:
                t0, t1 = r_b0["early_stop_episodes"], r_b1["early_stop_episodes"]
            elif "截断" in metric:
                t0, t1 = r_b0["truncated_episodes"], r_b1["truncated_episodes"]
            elif "click 步" in metric or "非交互 click" == metric:
                t0, t1 = r_b0["total_click_steps"], r_b1["total_click_steps"]
            else:
                t0, t1 = r_b0["total_episodes"], r_b1["total_episodes"]
            s0 = fmt(v0, t0)
            s1 = fmt(v1, t1)
        else:
            s0 = str(v0)
            s1 = str(v1)
        print(f"  {metric:<28s} {s0:>20s}  {s1:>20s}")

    # 非交互 click 比例（以 click 步数为分母）
    nic0 = r_b0["total_noninteractive_clicks"]
    tc0 = max(r_b0["total_click_steps"], 1)
    nic1 = r_b1["total_noninteractive_clicks"]
    tc1 = max(r_b1["total_click_steps"], 1)
    print(f"\n  {'非交互 click 率 (click步)':<28s} {nic0/tc0*100:>19.1f}%  {nic1/tc1*100:>19.1f}%")

    # 非交互 click 在早停中的角色
    es0 = max(r_b0["early_stop_episodes"], 1)
    es1 = max(r_b1["early_stop_episodes"], 1)
    enic0 = r_b0["early_stop_with_noninteractive_click"]
    enic1 = r_b1["early_stop_with_noninteractive_click"]
    print(f"  {'早停中非交互 click 占比':<28s} {enic0/es0*100:>19.1f}%  {enic1/es1*100:>19.1f}%")

    ln0 = r_b0["early_stop_last_N_all_noninteractive"]
    ln1 = r_b1["early_stop_last_N_all_noninteractive"]
    print(f"  {'早停末段全非交互 click':<28s} {ln0/es0*100:>19.1f}%  {ln1/es1*100:>19.1f}%")

    # Role 分布对比
    print(f"\n  --- 被点击的非交互 role 对比 (Top 10) ---")
    all_roles = set(r_b0["noninteractive_role_counter"].keys()) | set(r_b1["noninteractive_role_counter"].keys())
    combined = []
    for role in all_roles:
        c0 = r_b0["noninteractive_role_counter"].get(role, 0)
        c1 = r_b1["noninteractive_role_counter"].get(role, 0)
        combined.append((role, c0, c1, c0 + c1))
    combined.sort(key=lambda x: -x[3])

    print(f"  {'Role':<30s} {'B0':>8s} {'B1':>8s} {'Total':>8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    for role, c0, c1, ct in combined[:15]:
        print(f"  {role:<30s} {c0:>8d} {c1:>8d} {ct:>8d}")


def print_early_stop_details(r, label):
    """打印早停 episode 的详细信息。"""
    details = r.get("early_stop_tasks_detail", [])
    if not details:
        return
    print(f"\n  --- {label} 早停 episode 详情 ---")
    print(f"  {'Task':>6s} {'Steps':>6s} {'Clicks':>7s} {'NonInt':>7s} {'末段全非交互':>14s} {'非交互 roles'}")
    print(f"  {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*14} {'-'*30}")
    for d in sorted(details, key=lambda x: x["task_id"]):
        roles_str = ", ".join(d["noninteractive_roles"][:5])
        if len(d["noninteractive_roles"]) > 5:
            roles_str += f" (+{len(d['noninteractive_roles'])-5})"
        last_n = "是" if d["last_N_all_noninteractive"] else "否"
        print(f"  {d['task_id']:>6d} {d['steps']:>6d} {d['total_clicks']:>7d} {d['noninteractive_clicks']:>7d} {last_n:>14s} {roles_str}")


def main():
    base = "/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results/visualwebarena/phase1"

    b0_dir = os.path.join(base, "B0_3mode_classifieds_20260413/phase1_som_router_0")
    b1_dir = os.path.join(base, "B1_3mode_classifieds_20260413/phase1_som_router_0")

    print("分析 B0 (235B API) SoM classifieds ...")
    r_b0 = analyze_condition(b0_dir, "B0 (235B API) SoM classifieds")

    print("分析 B1 (4B local) SoM classifieds ...")
    r_b1 = analyze_condition(b1_dir, "B1 (4B local) SoM classifieds")

    if r_b0 is None or r_b1 is None:
        print("数据不完整，无法分析")
        sys.exit(1)

    print_results(r_b0)
    print_results(r_b1)
    print_comparison(r_b0, r_b1)
    print_early_stop_details(r_b0, "B0")
    print_early_stop_details(r_b1, "B1")


if __name__ == "__main__":
    main()
