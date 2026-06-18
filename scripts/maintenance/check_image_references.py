#!/usr/bin/env python3
"""check_image_references.py — preflight gate: paper-grade evaluator 本地 reference image 可达性

B-1878 (2026-06-18): reddit Phase 1a fire (R15710) task28 撞 VWA `PageImageEvaluator`
`Image.open('coco_images/000000515982.jpg')` → FileNotFoundError → paper-grade
`EvaluatorUnavailableError` (拒绝吸收成 agent score=0) → chain abort + orchestrator DOWN.
根因 = DGX→A100 self-hosted 迁移漏建 `coco_images` symlink + 漏配 `media/` reference;
cls 18 condition 全过 = cls 本地 reference 数 0; reddit 首撞 = 唯一有本地 reference 的站点 (2 张).
症结: preflight 旧有只查 evaluator INIT (B-793) 与 import,不查 reference image 是否在盘
→ fire 跑到 task-N 才 abort (同 B-1660 NLTK / B-679 OpenAI key 的 "silent preflight pass →
first task evaluator crash" 体例).

本 gate 复现 runner 的 reference 解析路径 (p79.experiment.tasks._replace_placeholders):
`__REDDIT__`/`__SHOPPING__`/`__CLASSIFIEDS__` → base URL (评测走 requests.get, 不碰本地盘);
裸相对路径 (`coco_images/...` / `media/...`, 无 placeholder) 保持原样 → evaluator
`Image.open` 当作 CWD-相对本地文件读 → 必须可达. 扫 VWA raw config 的
`eval.page_image_query[].eval_fuzzy_image_match` (` |OR| ` 分隔), placeholder 替换后,
对每个非-http 项 `os.path.exists` (相对当前 CWD = runner 评测时 CWD).

exit 0 = 所有本地 reference 可达 (或目标站点无本地 reference);
exit 2 = 有缺失 (stderr 列出具体路径 + 修复提示);
exit 3 = raw config / P79 import 异常.

用法:
  python3 scripts/maintenance/check_image_references.py                 # 扫全 3 站
  python3 scripts/maintenance/check_image_references.py --site reddit   # 单站
  # preflight_v2.sh check_image_references() 在 PAPER_GRADE_PREFLIGHT=1 时调用本脚本.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# repo root 上 sys.path 以 import p79 (脚本在 scripts/maintenance/ 下, 上溯 2 层)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_CFG_DIR = "external/visualwebarena/config_files/vwa"
SITE_RAW = {
    "classifieds": "test_classifieds.raw.json",
    "reddit": "test_reddit.raw.json",
    "shopping": "test_shopping.raw.json",
}


def _load_p79_placeholder_tools():
    """复用 runner canonical 替换逻辑, 保证 gate 与 fire 同源 (避免 gate-vs-runner 漂移)."""
    from p79.experiment.tasks import _replace_placeholders, _placeholder_mapping

    return _replace_placeholders, _placeholder_mapping()


def scan_site(cfg_dir: str, site: str, replace, mapping):
    """返回 (task_count, local_refs:set, missing:list). raw 缺失 → (None, set(), [])."""
    path = os.path.join(cfg_dir, SITE_RAW[site])
    if not os.path.exists(path):
        return None, set(), []
    tasks = json.load(open(path))
    if isinstance(tasks, dict):
        tasks = [tasks]
    local_refs: set[str] = set()
    for task in tasks:
        for query in (task.get("eval", {}).get("page_image_query") or []):
            raw_val = query.get("eval_fuzzy_image_match")
            if not raw_val:
                continue
            for item in replace(raw_val, mapping).split(" |OR| "):
                item = item.strip()
                if item and not item.startswith("http"):
                    local_refs.add(item)
    missing = [ref for ref in sorted(local_refs) if not os.path.exists(ref)]
    return len(tasks), local_refs, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", choices=sorted(SITE_RAW), action="append",
                    help="只扫指定站点 (可重复); 默认全 3 站")
    ap.add_argument("--config-dir", default=DEFAULT_CFG_DIR,
                    help=f"VWA raw config 目录 (默认 {DEFAULT_CFG_DIR})")
    args = ap.parse_args()

    try:
        replace, mapping = _load_p79_placeholder_tools()
    except Exception as exc:  # noqa: BLE001 — preflight gate 须报清楚而非裸 traceback
        print(f"[check_image_references] P79 import 失败: {type(exc).__name__}: {exc!r}", file=sys.stderr)
        return 3

    sites = args.site or list(SITE_RAW)
    total_missing: list[str] = []
    print(f"[check_image_references] CWD={os.getcwd()} config_dir={args.config_dir}")
    for site in sites:
        try:
            n, refs, missing = scan_site(args.config_dir, site, replace, mapping)
        except Exception as exc:  # noqa: BLE001
            print(f"[check_image_references] {site}: 扫描异常 {type(exc).__name__}: {exc!r}", file=sys.stderr)
            return 3
        if n is None:
            print(f"  {site}: raw config 不在 {args.config_dir}/{SITE_RAW[site]} — SKIP (非 fatal)")
            continue
        status = "OK" if not missing else f"MISSING {len(missing)}"
        # no silent caps: 明确报扫了多少 task / 多少本地 ref / 缺几个
        print(f"  {site}: {n} tasks | 本地 reference={len(refs)} | 缺失={len(missing)} [{status}]")
        for ref in missing:
            print(f"      ✗ MISSING (CWD-相对不可达): {ref}", file=sys.stderr)
        total_missing.extend(missing)

    if total_missing:
        print(file=sys.stderr)
        print("[check_image_references] FAIL — paper-grade evaluator 会在跑到对应 task 时", file=sys.stderr)
        print("  抛 FileNotFoundError → EvaluatorUnavailableError → chain abort (B-1878)。", file=sys.stderr)
        print("  修复 (A100, cd 到 runner CWD):", file=sys.stderr)
        print("    coco_images/* → ln -sfn external/visualwebarena/coco_images coco_images", file=sys.stderr)
        print("    media/.../*.jpg → curl http://localhost:7770/<相对路径> -o <相对路径> (Magento 按需生成)", file=sys.stderr)
        return 2

    print("[check_image_references] PASS — 所有站点本地 reference image 均 CWD 可达")
    return 0


if __name__ == "__main__":
    sys.exit(main())
