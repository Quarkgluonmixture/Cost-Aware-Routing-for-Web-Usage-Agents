#!/usr/bin/env python3
"""Generate the deterministic (Tier-1) half of each reddit /diag digest.

Tier-2/Tier-3 prose is written by hand on top of this; this script only emits
what is mechanically derivable from the Tier-1 scan so the numbers in every
digest come from one place and stay reproducible.
"""
import json, collections, sys
from pathlib import Path

REPO = Path("/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents")
SCAN = Path("/tmp/diag_v8")
OUT = REPO / "docs/analysis/vwa_reddit"

# Canonical (model, mode, site) → run-dir map lives in diag_rescan_all so the scan
# outputs and the digests cannot drift apart. Keys are `<model>_<mode>_<site>`, which
# is also the digest basename and the scan filename stem.
import importlib.util as _ilu
_rs_spec = _ilu.spec_from_file_location("diag_rescan_all", REPO / "scripts" / "analysis" / "diag_rescan_all.py")
_rs = _ilu.module_from_spec(_rs_spec)
sys.modules["diag_rescan_all"] = _rs
_rs_spec.loader.exec_module(_rs)
RUNS = dict(_rs.CANONICAL)
RUNS.update(_rs._discover_cls())

MODEL_LABEL = {"B0":"Qwen3-VL-235B-A22B (proxy)","B1":"Qwen3-VL-4B (local)","B2":"Gemma3-VL `google/gemma-3-4b-it` (local)"}
RULE_NAME = {
 "P1":"元素中心越界","P2":"容器节点误点","P3":"Thought-Action 解耦","P4":"根节点误操作",
 "P5":"感知缺失循环","P6":"视觉任务 DOM 必败(dom)","P7":"sCity=州名","P8":"select 反馈缺失(scaffold)",
 "P10":"跨步数值记忆失败","P11":"最新+地点组合","P12":"从不翻页","P13":"搜索代替浏览",
 "P14":"URL 自环","P15":"gallery 行位置(dom)","P16":"视觉图像内容(dom)","P17":"click-back 振荡",
 "P18":"cheapest 漏价格排序","P19":"url_match 过早 finish","P20":"评测目标页从未访问",
 "P21":"dom 视觉幻觉(dom)","P22":"图上数字 dom 不可读","P23":"oldest 误用价格排序",
 "P24":"不确定仍 finish","P25":"跨站任务跳过其中一站","P27":"找不到即放弃","P28":"benchmark-FP 货币 tokenize",
 "P29":"benchmark-FP 语义 yes/no","P30":"到达正确 item 后离开","P31":"budget 耗尽未完成",
 "P32":"文本误入价格 filter","P33":"导航至裸图片 URL 幻觉","P34":"VISUAL_BLIND_IMAGE_TASK",
 "P35":"MUTATION_MISSING","P36":"WALK_FAIL_DEGENERATE","P37":"URL_HALLUCINATION",
 "P38":"DOM_URL_AS_IMAGE","P39":"SUCCESS_NO_MUTATION(success侧)","P40":"LUCKY_NUMERIC_FP(success侧)",
 # reddit discover batch v8 (2026-07-27)
 "P41":"PASSIVE_MUST_EXCLUDE_FP(success侧, B-1889)","P42":"MULTI_SITE_SINGLE_SITE_GROUNDING(success侧, B-1892)",
 "P43":"PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT(中性标签)","P44":"HALLUCINATED_ELEMENT_REF",
 "P45":"IDENTICAL_FAILED_ACTION_STREAK","P46":"COMMENT_INTENT_NO_TYPE",
}

def rid(h): return h.get("rule_id") if isinstance(h, dict) else str(h)

def build(key):
    d = json.load(open(SCAN / f"{key}.json"))
    res = d["results"]
    parts = key.split("_")
    model, site = parts[0], parts[-1]
    mode = "_".join(parts[1:-1])
    n = len(res)
    succ = [e for e in res if e["success"]]
    fh = [e for e in res if not e["success"] and e["hits"]]
    fn = [e for e in res if not e["success"] and not e["hits"]]
    sh = [e for e in succ if e["hits"]]
    failed_rules = collections.Counter()
    failed_eps = collections.Counter()
    for e in res:
        if e["success"]: continue
        seen = set()
        for h in e["hits"]:
            failed_rules[rid(h)] += 1
            seen.add(rid(h))
        for r in seen: failed_eps[r] += 1
    succ_rules = collections.Counter()
    for e in succ:
        for h in e["hits"]: succ_rules[rid(h)] += 1
    t160 = any(e.get("task_id") == 160 and e["success"] for e in res)
    return dict(key=key, model=model, mode=mode, run=RUNS[key], n=n,
                succ=len(succ), sr=100*len(succ)/n, fh=len(fh), fn=len(fn), sh=len(sh),
                failed_rules=failed_rules, failed_eps=failed_eps, succ_rules=succ_rules,
                ruleset=d.get("ruleset_version","?"), t160=t160,
                nohit_ids=sorted(e.get("task_id") for e in fn),
                succ_ids=sorted(e.get("task_id") for e in succ))

def tier1_section(s):
    L = []
    L.append("## 1. Header\n")
    L.append(f"| 字段 | 值 |\n|---|---|")
    L.append(f"| **Run** | `{s['run']}` |")
    L.append(f"| **Condition** | `phase1_{s['mode']}_router_0` |")
    L.append(f"| **Site / Mode / Model** | reddit / `{s['mode']}` / {s['model']} = {MODEL_LABEL[s['model']]} |")
    L.append(f"| **Episodes** | {s['n']} |")
    L.append(f"| **SR** | **{s['sr']:.2f}%** ({s['succ']} success / {s['n']-s['succ']} failed) |")
    L.append(f"| **ruleset_version** | `{s['ruleset']}` |")
    L.append(f"| **Tier-1 三子集** | failed+hit {s['fh']} · **failed-NO-hit {s['fn']}** · success+hit {s['sh']} |")
    L.append("")
    L.append("## 2. Tier-1 规则分布（failed 侧）\n")
    L.append("| 规则 | 含义 | step-level hits | 命中 episode 数 |\n|---|---|---|---|")
    for r, c in s["failed_rules"].most_common():
        L.append(f"| `{r}` | {RULE_NAME.get(r,'?')} | {c} | {s['failed_eps'][r]} |")
    if not s["failed_rules"]: L.append("| — | 无命中 | 0 | 0 |")
    L.append("")
    if s["succ_rules"]:
        L.append("**success 侧 fire 的规则（presence-only 误报审计对象）**: " +
                 ", ".join(f"`{r}`×{c}" for r, c in s["succ_rules"].most_common()) + "\n")
    else:
        L.append("**success 侧 fire 的规则**: 无（success 侧 0 命中）\n")
    L.append(f"**failed-NO-hit episode（deterministic 盲区）**: {s['nohit_ids']}\n")
    L.append(f"**success episode**: {s['succ_ids']}\n")
    return "\n".join(L)

if __name__ == "__main__":
    keys = sys.argv[1:] or list(RUNS)
    for k in keys:
        s = build(k)
        print(f"=== {k} ===")
        print(tier1_section(s))
        print()
