#!/usr/bin/env python3
"""日志清理脚本

用法:
    # 清理所有日志（使用默认配置）
    python scripts/utils/cleanup_logs.py

    # 只显示将要删除的内容（不实际删除）
    python scripts/utils/cleanup_logs.py --dry-run

    # 自定义清理策略
    python scripts/utils/cleanup_logs.py --max-age 7 --max-size 500 --max-count 50

    # 清理指定目录
    python scripts/utils/cleanup_logs.py --dir logs
    python scripts/utils/cleanup_logs.py --dir results

    # 显示磁盘使用情况
    python scripts/utils/cleanup_logs.py --usage-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from p79.utils.log_cleanup import (
    LogCleanupConfig,
    cleanup_logs,
    cleanup_results,
    cleanup_all,
    print_disk_usage,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="清理日志和临时文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="日志文件最大保留天数（默认: 30）",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1000,
        help="日志目录最大总大小（MB）（默认: 1000）",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=100,
        help="日志文件最大数量（默认: 100）",
    )
    parser.add_argument(
        "--max-run-age",
        type=int,
        default=90,
        help="实验结果目录最大保留天数（默认: 90）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将要删除的内容，不实际删除",
    )
    parser.add_argument(
        "--dir",
        type=str,
        choices=["logs", "results", "all"],
        default="all",
        help="要清理的目录（默认: all）",
    )
    parser.add_argument(
        "--usage-only",
        action="store_true",
        help="只显示磁盘使用情况，不进行清理",
    )
    
    args = parser.parse_args()
    
    config = LogCleanupConfig(
        max_log_age_days=args.max_age,
        max_log_size_mb=args.max_size,
        max_log_count=args.max_count,
        dry_run=args.dry_run,
    )
    
    if args.usage_only:
        # 只显示磁盘使用情况
        if args.dir in ("logs", "all"):
            logs_dir = project_root / "logs"
            if logs_dir.exists():
                print_disk_usage(logs_dir)
        
        if args.dir in ("results", "all"):
            results_dir = project_root / "results"
            if results_dir.exists():
                print_disk_usage(results_dir)
        
        return 0
    
    if args.dir == "all":
        # 清理所有
        cleanup_all(project_root, config)
    elif args.dir == "logs":
        # 只清理日志
        logs_dir = project_root / "logs"
        if logs_dir.exists():
            print_disk_usage(logs_dir)
            cleanup_logs(logs_dir, config, pattern="*.log")
    elif args.dir == "results":
        # 只清理实验结果
        results_dir = project_root / "results"
        if results_dir.exists():
            print_disk_usage(results_dir)
            cleanup_results(results_dir, config, max_run_age_days=args.max_run_age)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
