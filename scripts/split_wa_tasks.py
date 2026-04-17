#!/usr/bin/env python3
"""One-time script: split WA test_webarena.raw.json into per-site files.

Outputs:
  external/visualwebarena/config_files/wa/test_shopping.raw.json      (192 = 187 single + 5 cross)
  external/visualwebarena/config_files/wa/test_shopping_admin.raw.json (182)
  external/visualwebarena/config_files/wa/test_reddit.raw.json         (106)

Cross-site tasks (5 shopping+reddit) are assigned to shopping (primary site).
"""
from __future__ import annotations

import json
from pathlib import Path

WA_DIR = Path(__file__).resolve().parent.parent / "external" / "visualwebarena" / "config_files" / "wa"
RAW_FILE = WA_DIR / "test_webarena.raw.json"
TARGET_SITES = {"shopping", "shopping_admin", "reddit"}


def main() -> None:
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    shopping: list = []
    shopping_admin: list = []
    reddit: list = []

    for t in tasks:
        sites = set(t.get("sites", []))
        if not sites or not sites.issubset(TARGET_SITES):
            continue
        if sites == {"shopping"} or (sites == {"shopping", "reddit"}):
            # Single-site shopping + cross-site shopping+reddit → shopping file
            shopping.append(t)
        elif sites == {"shopping_admin"}:
            shopping_admin.append(t)
        elif sites == {"reddit"}:
            reddit.append(t)

    for name, data in [
        ("test_shopping.raw.json", shopping),
        ("test_shopping_admin.raw.json", shopping_admin),
        ("test_reddit.raw.json", reddit),
    ]:
        out_path = WA_DIR / name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"{name}: {len(data)} tasks")

    print(f"Total: {len(shopping) + len(shopping_admin) + len(reddit)} tasks")


if __name__ == "__main__":
    main()
