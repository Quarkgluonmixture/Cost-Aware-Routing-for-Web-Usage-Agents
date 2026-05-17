"""日志清理和轮转工具模块

提供日志文件的自动清理和轮转功能，防止日志目录无限增长。
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Optional, List, Tuple

import logging

logger = logging.getLogger(__name__)


class LogCleanupConfig:
    """日志清理配置"""
    def __init__(
        self,
        max_log_age_days: int = 30,
        max_log_size_mb: Optional[int] = None,
        max_log_count: Optional[int] = None,
        dry_run: bool = True,
    ):
        # B-724 (/stress A1.11 P1-8 AB* OOB, 2026-05-17): dry_run default changed to
        # True. Pre-fix `LogCleanupConfig()` paired with `cleanup_logs()` /
        # `cleanup_results()` recursively rmtree'd results/run_* > max_run_age_days,
        # nuking paper-grade历史 forensic evidence. `cleanup_all()` had a B-213 second
        # safety gate, but `cleanup_logs` / `cleanup_results` direct callers (notebook,
        # interactive shell, maintenance script with explicit config) bypassed it.
        # Explicit opt-in to destructive ops: pass `dry_run=False` AND use
        # `cleanup_logs(..., confirmed=True)` / `cleanup_results(..., confirmed=True)`.
        self.max_log_age_days = max_log_age_days
        self.max_log_size_mb = max_log_size_mb
        self.max_log_count = max_log_count
        self.dry_run = dry_run


def get_log_size_mb(path: Path) -> float:
    """获取文件或目录的大小（MB）"""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total / (1024 * 1024)
    return 0.0


def get_file_age_days(path: Path) -> float:
    """获取文件的年龄（天）"""
    mtime = path.stat().st_mtime
    return (time.time() - mtime) / (24 * 60 * 60)


def cleanup_logs(
    log_dir: Path,
    config: LogCleanupConfig,
    pattern: str = "*.log",
    *,
    confirmed: bool = False,
) -> Tuple[List[Path], List[Path], float]:
    """清理日志文件

    B-724 (/stress A1.11 P1-8 AB*, 2026-05-17): `confirmed=True` required to actually
    delete even when `config.dry_run=False`. Mirrors `cleanup_all` safety gate.
    B-723 (/stress A1.11 P1-7 AC*, 2026-05-17): `unlink(missing_ok=True)` so TOCTOU
    file-already-deleted between glob and unlink does not crash the whole cleanup
    sweep (was: FileNotFoundError → cleanup_logs raises → 后续 cleanup skip → 磁盘满).

    Args:
        log_dir: 日志目录路径
        config: 清理配置
        pattern: 文件匹配模式
        confirmed: 必须显式 True 才会真删 (paper-grade safety gate)

    Returns:
        (已删除的文件列表, 保留的文件列表, 释放的空间MB)
    """
    deleted_files: List[Path] = []
    kept_files: List[Path] = []
    freed_space_mb = 0.0

    if not log_dir.exists():
        logger.info(f"日志目录不存在: {log_dir}")
        return deleted_files, kept_files, freed_space_mb

    # B-724 safety gate: even if config.dry_run=False, require confirmed=True kwarg.
    _effective_dry_run = config.dry_run or not confirmed
    if not config.dry_run and not confirmed:
        logger.warning(
            "cleanup_logs called with dry_run=False but confirmed=False — forcing dry_run "
            "(B-724 A1.11 P1-8 safety gate). Pass confirmed=True to actually delete."
        )

    # 获取所有匹配的日志文件
    log_files = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime)

    # 按年龄清理
    if config.max_log_age_days > 0:
        for log_file in log_files[:]:
            age_days = get_file_age_days(log_file)
            if age_days > config.max_log_age_days:
                size_mb = get_log_size_mb(log_file)
                deleted_files.append(log_file)
                freed_space_mb += size_mb
                if not _effective_dry_run:
                    log_file.unlink(missing_ok=True)
                    logger.info(f"删除过期日志: {log_file} (年龄: {age_days:.1f}天, 大小: {size_mb:.2f}MB)")
                else:
                    logger.info(f"[DRY RUN] 将删除过期日志: {log_file} (年龄: {age_days:.1f}天, 大小: {size_mb:.2f}MB)")
                log_files.remove(log_file)

    # 按数量清理
    if config.max_log_count is not None and len(log_files) > config.max_log_count:
        # 删除最旧的文件
        files_to_delete = log_files[:len(log_files) - config.max_log_count]
        for log_file in files_to_delete:
            size_mb = get_log_size_mb(log_file)
            deleted_files.append(log_file)
            freed_space_mb += size_mb
            if not _effective_dry_run:
                log_file.unlink(missing_ok=True)
                logger.info(f"删除多余日志: {log_file} (大小: {size_mb:.2f}MB)")
            else:
                logger.info(f"[DRY RUN] 将删除多余日志: {log_file} (大小: {size_mb:.2f}MB)")
            log_files.remove(log_file)

    # 按总大小清理
    if config.max_log_size_mb is not None:
        total_size_mb = sum(get_log_size_mb(f) for f in log_files)
        while total_size_mb > config.max_log_size_mb and log_files:
            # 删除最旧的文件
            oldest_file = log_files.pop(0)
            size_mb = get_log_size_mb(oldest_file)
            deleted_files.append(oldest_file)
            freed_space_mb += size_mb
            total_size_mb -= size_mb
            if not _effective_dry_run:
                oldest_file.unlink(missing_ok=True)
                logger.info(f"删除超大小限制日志: {oldest_file} (大小: {size_mb:.2f}MB)")
            else:
                logger.info(f"[DRY RUN] 将删除超大小限制日志: {oldest_file} (大小: {size_mb:.2f}MB)")

    kept_files = log_files

    logger.info(f"日志清理完成: 删除 {len(deleted_files)} 个文件, 保留 {len(kept_files)} 个文件, 释放 {freed_space_mb:.2f}MB")

    return deleted_files, kept_files, freed_space_mb


def cleanup_results(
    results_dir: Path,
    config: LogCleanupConfig,
    max_run_age_days: int = 90,
    *,
    confirmed: bool = False,
) -> Tuple[List[Path], float]:
    """清理旧的实验结果目录

    B-724 (/stress A1.11 P1-8 AB*, 2026-05-17): `confirmed=True` required to actually
    rmtree even when `config.dry_run=False`. Most destructive helper — recursive
    rmtree of `results/.../run_*` > max_run_age_days = paper-grade历史 forensic
    evidence elimination. Mirrors `cleanup_all` safety gate.

    Args:
        results_dir: 结果目录路径
        config: 清理配置
        max_run_age_days: 实验运行目录的最大年龄（天）
        confirmed: 必须显式 True 才会真删 (paper-grade safety gate)

    Returns:
        (已删除的目录列表, 释放的空间MB)
    """
    deleted_dirs: List[Path] = []
    freed_space_mb = 0.0

    if not results_dir.exists():
        logger.info(f"结果目录不存在: {results_dir}")
        return deleted_dirs, freed_space_mb

    _effective_dry_run = config.dry_run or not confirmed
    if not config.dry_run and not confirmed:
        logger.warning(
            "cleanup_results called with dry_run=False but confirmed=False — forcing "
            "dry_run (B-724 A1.11 P1-8 safety gate). Pass confirmed=True to actually rmtree."
        )

    # 获取所有运行目录
    for run_dir in results_dir.rglob("run_*"):
        if not run_dir.is_dir():
            continue

        age_days = get_file_age_days(run_dir)
        if age_days > max_run_age_days:
            size_mb = get_log_size_mb(run_dir)
            deleted_dirs.append(run_dir)
            freed_space_mb += size_mb
            if not _effective_dry_run:
                # B-723 (/stress A1.11 P1-7 AC*, 2026-05-17): tolerate concurrent
                # removal between rglob and rmtree (another watchdog/cleanup process).
                shutil.rmtree(run_dir, ignore_errors=True)
                logger.info(f"删除过期实验结果: {run_dir} (年龄: {age_days:.1f}天, 大小: {size_mb:.2f}MB)")
            else:
                logger.info(f"[DRY RUN] 将删除过期实验结果: {run_dir} (年龄: {age_days:.1f}天, 大小: {size_mb:.2f}MB)")

    logger.info(f"实验结果清理完成: 删除 {len(deleted_dirs)} 个目录, 释放 {freed_space_mb:.2f}MB")

    return deleted_dirs, freed_space_mb


def print_disk_usage(path: Path) -> None:
    """打印目录的磁盘使用情况"""
    if not path.exists():
        logger.info(f"路径不存在: {path}")
        return
    
    total_size_mb = get_log_size_mb(path)
    file_count = 0
    dir_count = 0
    
    if path.is_file():
        file_count = 1
    elif path.is_dir():
        for item in path.rglob("*"):
            if item.is_file():
                file_count += 1
            elif item.is_dir():
                dir_count += 1
    
    logger.info(f"磁盘使用情况 - {path}:")
    logger.info(f"  总大小: {total_size_mb:.2f} MB")
    logger.info(f"  文件数: {file_count}")
    logger.info(f"  目录数: {dir_count}")


def cleanup_all(
    project_root: Path,
    config: Optional[LogCleanupConfig] = None,
    *,
    confirmed: bool = False,
) -> None:
    """清理所有日志和临时文件

    Args:
        project_root: 项目根目录
        config: 清理配置，如果为None则使用默认配置 (dry_run=True 默认)
        confirmed: 必须显式 ``confirmed=True`` 才会执行实际删除。否则强制
            dry_run 即使 config.dry_run=False。

    B-213 fix (2026-05-16, A1.5 Item 4): default behavior is now NON-DESTRUCTIVE.
    Pre-fix: default config.dry_run=False + max_run_age_days=90 →误调一次
    ``cleanup_all(project_root)`` 永久 ``shutil.rmtree(results/.../run_*)`` > 90d,
    paper-1 archive baseline 数据会灭失。Post-fix: 默认 config.dry_run=True;
    caller 必须显式 ``cleanup_all(root, config, confirmed=True)`` 才真删。
    CLI ``scripts/maintenance/cleanup_logs.py`` 加 ``--confirm`` flag 才会传 True。
    """
    if config is None:
        config = LogCleanupConfig(
            max_log_age_days=30,
            max_log_size_mb=1000,  # 1GB
            max_log_count=100,
            dry_run=True,  # B-213: default to dry-run; opt-in to destructive via confirmed=True
        )

    # B-213 safety gate: even if caller passed dry_run=False explicitly,
    # require confirmed=True kwarg as a second safeguard.
    if not config.dry_run and not confirmed:
        logger.warning(
            "cleanup_all called with dry_run=False but confirmed=False — "
            "forcing dry_run=True (B-213 safety gate, A1.5 Item 4). "
            "Pass confirmed=True to actually delete."
        )
        config.dry_run = True

    logger.info("=" * 60)
    logger.info("开始清理日志和临时文件 (dry_run=%s)", config.dry_run)
    logger.info("=" * 60)
    
    # 清理 logs/ 目录
    logs_dir = project_root / "logs"
    if logs_dir.exists():
        logger.info(f"\n清理日志目录: {logs_dir}")
        print_disk_usage(logs_dir)
        # B-724 (/stress A1.11 P1-8): propagate `confirmed` to per-helper safety gates.
        cleanup_logs(logs_dir, config, pattern="*.log", confirmed=confirmed)

    # 清理 results/ 目录中的旧实验
    results_dir = project_root / "results"
    if results_dir.exists():
        logger.info(f"\n清理实验结果目录: {results_dir}")
        print_disk_usage(results_dir)
        cleanup_results(results_dir, config, max_run_age_days=90, confirmed=confirmed)

    # 清理临时文件
    temp_patterns = [
        "temp_task_*.json",
        "temp_smoke_config.json",
        "episode_*.jsonl",
    ]

    logger.info(f"\n清理临时文件")
    for pattern in temp_patterns:
        for temp_file in project_root.glob(pattern):
            if temp_file.is_file():
                size_mb = get_log_size_mb(temp_file)
                if not config.dry_run:
                    # B-723 (/stress A1.11 P1-7 AC*, 2026-05-17): tolerate concurrent
                    # removal between glob and unlink (another cleanup process).
                    temp_file.unlink(missing_ok=True)
                    logger.info(f"删除临时文件: {temp_file} ({size_mb:.2f}MB)")
                else:
                    logger.info(f"[DRY RUN] 将删除临时文件: {temp_file} ({size_mb:.2f}MB)")
    
    logger.info("=" * 60)
    logger.info("清理完成")
    logger.info("=" * 60)
