#!/usr/bin/env python3
"""Scan running run_experiment + experiment_watchdog processes and emit a
markdown table with episode count, adjusted SR, throughput, ETA, and stale
watchdog flags.

Replaces the manually-maintained §1 Active Processes table in next_steps.md
which goes stale when runners restart or directories get renamed.

Usage:
    python3 scripts/maintenance/active_processes.py            # default markdown
    python3 scripts/maintenance/active_processes.py --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_VWA = ROOT / "results/visualwebarena/phase1"
RESULTS_WA = ROOT / "results/webarena/phase1"

# §139.8 + /stress A1.6 (2026-05-16): scored task counts (total − N/A
# excluded at load) from single source of truth. Pre-exclusion counts were
# vwa 234/210/466, wa 106/192/182. `strict=True` fails loud on missing
# config — silent 0-fallback used to make `make active` print 0/0 progress
# masking the underlying CLI failure.
from p79.experiment.analysis import scored_task_count as _scored_task_count

EXPECTED_N = {
    # VWA
    "classifieds": _scored_task_count("classifieds", "visualwebarena", strict=True),
    "reddit": _scored_task_count("reddit", "visualwebarena", strict=True),
    "shopping": _scored_task_count("shopping", "visualwebarena", strict=True),
    # WA (cross-bench, prefix wa_*)
    "wa_reddit": _scored_task_count("reddit", "webarena", strict=True),
    "wa_shopping": _scored_task_count("shopping", "webarena", strict=True),
    "wa_shopping_admin": _scored_task_count("shopping_admin", "webarena", strict=True),
}


@dataclass
class Proc:
    pid: int
    etime: str           # `[[DD-]HH:]MM:SS` from ps
    etimes: int          # elapsed seconds
    cmd: str
    run_id: str | None = None
    run_dir: Path | None = None
    condition: str | None = None
    config: str | None = None
    role: str = "?"      # runner | watchdog


@dataclass
class Cell:
    runner: Proc | None = None
    watchdog: Proc | None = None
    run_id: str = ""
    run_dir: Path | None = None
    condition: str = ""
    expected_n: int | None = None
    n_done: int = 0
    n_succ: int = 0
    n_adj_succ: int = 0
    last_ep_mtime: float | None = None
    throughput_per_h: float | None = None
    eta_human: str = "?"
    flags: list[str] = field(default_factory=list)


@dataclass
class Chain:
    pid: int
    etime: str
    cells: list[str]                  # original argv: ["queue_phantom_text.sh B1 classifieds", ...]
    log_path: Path | None = None
    current_idx: int = 0              # 0-indexed; last "[N/M]" line tells us
    current_elapsed_secs: int | None = None

    def etime_seconds(self) -> int:
        # Parse [[DD-]HH:]MM:SS
        s = self.etime
        days = 0
        if "-" in s:
            d, s = s.split("-", 1)
            days = int(d)
        parts = [int(p) for p in s.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        h, m, sec = parts
        return days * 86400 + h * 3600 + m * 60 + sec


def parse_ps() -> tuple[list[Proc], list[Chain]]:
    out = subprocess.run(
        ["ps", "-eo", "pid,etime,etimes,cmd"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()[1:]
    procs: list[Proc] = []
    chains: list[Chain] = []
    for line in out:
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        pid_s, etime, etimes, cmd = parts
        if "grep" in cmd:
            continue
        is_runner_or_watchdog = "run_experiment" in cmd or "experiment_watchdog" in cmd
        is_chain = "queue_chain.sh" in cmd
        if not (is_runner_or_watchdog or is_chain):
            continue
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if is_chain:
            cells = parse_chain_argv(pid)
            if cells:
                chains.append(Chain(pid=pid, etime=etime, cells=cells))
            continue
        p = Proc(pid=pid, etime=etime, etimes=int(etimes), cmd=cmd)
        if "experiment_watchdog" in cmd:
            p.role = "watchdog"
            m = re.search(r"--run-dir\s+(\S+)", cmd)
            if m:
                p.run_dir = Path(m.group(1))
                p.run_id = p.run_dir.name
            m = re.search(r"--condition\s+(\S+)", cmd)
            if m:
                p.condition = m.group(1)
        elif "run_experiment.py" in cmd:
            p.role = "runner"
            m = re.search(r"--run_id\s+(\S+)", cmd)
            if m:
                p.run_id = m.group(1)
            m = re.search(r"--config\s+(\S+)", cmd)
            if m:
                p.config = m.group(1)
        procs.append(p)
    return procs, chains


def parse_chain_argv(pid: int) -> list[str]:
    """Read /proc/PID/cmdline (NUL-separated argv) — preserves quoted chain cells."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    args = [a for a in raw.split(b"\0") if a]
    out: list[str] = []
    seen_script = False
    for a in args:
        s = a.decode("utf-8", errors="replace")
        if not seen_script:
            if s.endswith("queue_chain.sh"):
                seen_script = True
            continue
        if s.startswith("--"):
            continue
        if s.startswith("queue_") and ".sh" in s:
            out.append(s)
    return out


def find_chain_log(chain: Chain) -> Path | None:
    """Pick the queue_chain log file most likely matching this PID — by mtime
    closest to (but not earlier than) chain start time."""
    candidates = sorted(
        (ROOT / "logs").glob("queue_chain*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    cutoff = time.time() - chain.etime_seconds() - 60
    for p in candidates:
        if p.stat().st_mtime >= cutoff:
            return p
    return candidates[0] if candidates else None


def parse_chain_log_state(log_path: Path) -> tuple[int | None, int | None]:
    """Return (current_index_1based, elapsed_secs_in_step) from last `[N/M]` line."""
    if not log_path.is_file():
        return None, None
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None, None
    last = None
    for m in re.finditer(r"\[(\d+)/(\d+)\][^\[]*?(\d+)s elapsed", text):
        last = m
    if last is None:
        return None, None
    return int(last.group(1)), int(last.group(3))


def detect_site(run_id: str) -> str | None:
    rid = run_id.lower()
    for site in ("wa_shopping_admin", "wa_shopping", "wa_reddit",
                 "shopping", "reddit", "classifieds"):
        if site in rid:
            return site
    return None


def find_run_dir(run_id: str) -> Path | None:
    for base in (RESULTS_VWA, RESULTS_WA):
        cand = base / run_id
        if cand.is_dir():
            return cand
    return None


def find_condition_dir(run_dir: Path) -> Path | None:
    if not run_dir.is_dir():
        return None
    for child in sorted(run_dir.iterdir()):
        if child.is_dir() and (child / "episodes").is_dir():
            return child
    return None


def episode_stats(cond_dir: Path) -> tuple[int, int, int, float | None]:
    ep_dir = cond_dir / "episodes"
    files = sorted(ep_dir.glob("*_summary_v2.json"))
    n = succ = adj = 0
    last_mtime: float | None = None
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        n += 1
        if data.get("success"):
            succ += 1
        # §139.8: adjusted_success retired — `success` is canonical
        if data.get("success", False):
            adj += 1
        m = f.stat().st_mtime
        last_mtime = m if last_mtime is None else max(last_mtime, m)
    return n, succ, adj, last_mtime


def estimate_throughput(cond_dir: Path, window_min: int = 60) -> float | None:
    """Episodes/hour based on summary_v2.json mtimes in the last `window_min`."""
    ep_dir = cond_dir / "episodes"
    cutoff = time.time() - window_min * 60
    recent = [f for f in ep_dir.glob("*_summary_v2.json") if f.stat().st_mtime >= cutoff]
    if not recent:
        return None
    return len(recent) * (60 / window_min)


def fmt_eta(remaining: int, throughput_per_h: float | None) -> str:
    if not throughput_per_h or throughput_per_h <= 0:
        return "?"
    hours = remaining / throughput_per_h
    if hours < 1:
        return f"~{int(round(hours * 60))}min"
    if hours < 48:
        return f"~{hours:.1f}h"
    return f"~{hours/24:.1f}d"


def build_cells(procs: list[Proc]) -> list[Cell]:
    """B-911 (/stress A2.2 P1-12-B codex F6, 2026-05-17): multi-PID collision
    detection. Pre-fix `c.runner = r` / `c.watchdog = w` OVERWROTE any prior
    Proc for the same run_id — operator viewing `make active` saw only the
    last-iterated PID even when 2 runners or 2 watchdogs raced on the same
    RUN_ID (which is the most dangerous multi-session collision state).
    Combined with the watchdog flock B-907 the actual race is now closed at
    the kernel layer, but `make active` still serves as the operator-visible
    audit signal — if a duplicate ever does slip through (e.g. operator
    `rm .locks/watchdog_${RUN_ID}.lock` force-release + manual retry), the
    operator should SEE it in the table, not have it silently collapsed.

    Records duplicate count + appends DUPLICATE-RUNNER / DUPLICATE-WATCHDOG
    flag while keeping `c.runner` / `c.watchdog` as the most-recent Proc
    (back-compat: existing render paths still read `c.runner.pid` etc).
    """
    runners = [p for p in procs if p.role == "runner"]
    watchdogs = [p for p in procs if p.role == "watchdog"]

    cells: dict[str, Cell] = {}
    runner_counts: dict[str, int] = {}
    runner_pids: dict[str, list[int]] = {}
    for r in runners:
        if not r.run_id:
            continue
        c = cells.setdefault(r.run_id, Cell(run_id=r.run_id))
        runner_counts[r.run_id] = runner_counts.get(r.run_id, 0) + 1
        runner_pids.setdefault(r.run_id, []).append(r.pid)
        c.runner = r
        c.run_dir = find_run_dir(r.run_id)

    watchdog_counts: dict[str, int] = {}
    watchdog_pids: dict[str, list[int]] = {}
    for w in watchdogs:
        if not w.run_id:
            continue
        c = cells.setdefault(w.run_id, Cell(run_id=w.run_id))
        watchdog_counts[w.run_id] = watchdog_counts.get(w.run_id, 0) + 1
        watchdog_pids.setdefault(w.run_id, []).append(w.pid)
        c.watchdog = w
        if c.run_dir is None and w.run_dir is not None:
            c.run_dir = w.run_dir
        if not c.condition and w.condition:
            c.condition = w.condition

    # B-911: append DUPLICATE flags for any run_id with > 1 runner or watchdog.
    for run_id, c in cells.items():
        if runner_counts.get(run_id, 0) > 1:
            pids = ",".join(str(p) for p in runner_pids[run_id])
            c.flags.append(f"DUPLICATE-RUNNER(n={runner_counts[run_id]}, pids={pids})")
        if watchdog_counts.get(run_id, 0) > 1:
            pids = ",".join(str(p) for p in watchdog_pids[run_id])
            c.flags.append(f"DUPLICATE-WATCHDOG(n={watchdog_counts[run_id]}, pids={pids})")

    for c in cells.values():
        # Stale watchdog detection (dir gone or renamed)
        if c.watchdog and c.watchdog.run_dir and not c.watchdog.run_dir.is_dir():
            c.flags.append("STALE-WATCHDOG (run_dir missing)")
        if c.watchdog and not c.runner:
            c.flags.append("ZOMBIE-WATCHDOG (no runner)")

        site = detect_site(c.run_id)
        if site:
            c.expected_n = EXPECTED_N.get(site)

        if not c.run_dir or not c.run_dir.is_dir():
            continue
        cond_dir = (c.run_dir / c.condition) if c.condition else find_condition_dir(c.run_dir)
        if cond_dir is None or not cond_dir.is_dir():
            continue
        c.condition = cond_dir.name
        c.n_done, c.n_succ, c.n_adj_succ, c.last_ep_mtime = episode_stats(cond_dir)
        c.throughput_per_h = estimate_throughput(cond_dir, window_min=60)
        if c.expected_n:
            remaining = max(0, c.expected_n - c.n_done)
            c.eta_human = fmt_eta(remaining, c.throughput_per_h)

    return sorted(cells.values(), key=lambda c: c.run_id)


def hydrate_chains(chains: list[Chain]) -> None:
    for ch in chains:
        ch.log_path = find_chain_log(ch)
        if ch.log_path:
            idx_1based, elapsed = parse_chain_log_state(ch.log_path)
            if idx_1based is not None:
                ch.current_idx = idx_1based - 1
            if elapsed is not None:
                ch.current_elapsed_secs = elapsed


def render_markdown(cells: list[Cell], chains: list[Chain]) -> str:
    lines = []
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    lines.append(f"# Active Processes — generated {now}\n")
    if not cells and not chains:
        lines.append("_No active run_experiment / experiment_watchdog / queue_chain processes._")
        return "\n".join(lines)

    if cells:
        lines.append("## Runners + watchdogs\n")
        lines.append("| run_id | runner PID | watchdog PID | progress | adj-SR | rate (ep/h) | ETA | flags |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in cells:
            runner_pid = c.runner.pid if c.runner else "—"
            watchdog_pid = c.watchdog.pid if c.watchdog else "—"
            if c.expected_n:
                progress = f"{c.n_done}/{c.expected_n} ({100*c.n_done/c.expected_n:.0f}%)"
            else:
                progress = f"{c.n_done}"
            adj_sr = (
                f"{c.n_adj_succ}/{c.n_done} ({100*c.n_adj_succ/c.n_done:.1f}%)"
                if c.n_done else "—"
            )
            rate = f"{c.throughput_per_h:.1f}" if c.throughput_per_h else "—"
            flags = ", ".join(c.flags) if c.flags else "✅"
            lines.append(
                f"| `{c.run_id}` | {runner_pid} | {watchdog_pid} | {progress} | {adj_sr} | {rate} | {c.eta_human} | {flags} |"
            )

        elapsed_lines = [f"- `{c.run_id}` runner elapsed: {c.runner.etime}" for c in cells if c.runner]
        if elapsed_lines:
            lines.append("")
            lines.append("**Runner elapsed:** " + " | ".join(e.lstrip("- ") for e in elapsed_lines))

    if chains:
        lines.append("\n## Queue chains\n")
        for ch in chains:
            total = len(ch.cells)
            cur = ch.current_idx
            elapsed_step = (
                f"{ch.current_elapsed_secs/3600:.1f}h"
                if ch.current_elapsed_secs and ch.current_elapsed_secs >= 3600
                else (f"{ch.current_elapsed_secs/60:.0f}min" if ch.current_elapsed_secs else "?")
            )
            log_label = ch.log_path.name if ch.log_path else "?"
            lines.append(f"**chain PID={ch.pid}** (elapsed {ch.etime}, log `{log_label}`)")
            for i, cell in enumerate(ch.cells):
                if i < cur:
                    state = "✅ done"
                elif i == cur:
                    state = f"▶️ running ({elapsed_step} in step)"
                else:
                    state = "⏳ pending"
                lines.append(f"  - [{i+1}/{total}] `{cell}` — {state}")

    return "\n".join(lines)


def render_json(cells: list[Cell], chains: list[Chain]) -> str:
    cell_out = []
    for c in cells:
        cell_out.append({
            "run_id": c.run_id,
            "runner_pid": c.runner.pid if c.runner else None,
            "runner_etime": c.runner.etime if c.runner else None,
            "watchdog_pid": c.watchdog.pid if c.watchdog else None,
            "run_dir": str(c.run_dir) if c.run_dir else None,
            "condition": c.condition,
            "expected_n": c.expected_n,
            "n_done": c.n_done,
            "n_succ": c.n_succ,
            "n_adj_succ": c.n_adj_succ,
            "throughput_per_h": c.throughput_per_h,
            "eta_human": c.eta_human,
            "flags": c.flags,
        })
    chain_out = []
    for ch in chains:
        chain_out.append({
            "pid": ch.pid,
            "etime": ch.etime,
            "cells": [
                {
                    "spec": cell,
                    "state": "done" if i < ch.current_idx
                             else ("running" if i == ch.current_idx else "pending"),
                }
                for i, cell in enumerate(ch.cells)
            ],
            "current_idx": ch.current_idx,
            "current_elapsed_secs": ch.current_elapsed_secs,
            "log_path": str(ch.log_path) if ch.log_path else None,
        })
    return json.dumps(
        {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
         "cells": cell_out, "chains": chain_out},
        indent=2,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = ap.parse_args()
    procs, chains = parse_ps()
    cells = build_cells(procs)
    hydrate_chains(chains)
    print(render_json(cells, chains) if args.json else render_markdown(cells, chains))


if __name__ == "__main__":
    main()
