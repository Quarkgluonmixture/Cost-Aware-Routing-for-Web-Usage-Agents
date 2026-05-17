from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# B-718 (/stress A1.11 P0-2 B* OOB, 2026-05-17): bootstrap repo root into sys.path
# BEFORE any p79.* imports. Pre-fix absolute-path invocation
# (`python3 /full/path/p79/cli/run_experiment.py --help`) died with
# `ModuleNotFoundError: No module named 'p79'` — codex verified by reproduction.
# Queue scripts going through `pip install -e .` were unaffected, but manual
# recovery + watchdog absolute-path invocations would silently break.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from p79.experiment.config import load_experiment_config  # noqa: E402
from p79.experiment.runner import ExperimentRunner  # noqa: E402
from p79.utils.asyncio_workarounds import install_asyncio_target_closed_warning_filter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified experiment runner for P79")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--run_id", default=None, help="Optional run ID override")
    parser.add_argument("--phase", default=None, choices=["phase1", "phase2", "phase3"], help="Optional phase override")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional max step override")
    parser.add_argument("--log_path", default=None, help="Path to the log file for this run (stored in run_meta.json)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    install_asyncio_target_closed_warning_filter()

    cfg = load_experiment_config(args.config)
    if args.run_id:
        cfg["experiment"]["run_id"] = args.run_id
    else:
        # B-722 (/stress A1.11 P1-6 AC, 2026-05-17): ms-precision + PID so concurrent
        # queue dispatch (24 conditions × 3 baselines launching same-second) does not
        # collide on POSIX-second timestamp → output_root race + env_snapshot互覆盖.
        cfg["experiment"]["run_id"] = cfg["experiment"].get(
            "run_id", f"run_{int(time.time() * 1000)}_{os.getpid()}"
        )

    if args.log_path:
        cfg["experiment"]["log_path"] = args.log_path

    if args.phase:
        cfg["experiment"]["phase"] = args.phase
    if args.max_steps is not None:
        cfg.setdefault("runtime", {})["max_steps"] = args.max_steps

    runner = ExperimentRunner(cfg)

    # Paper-grade provenance: dump env snapshot BEFORE runner.run() so crashed
    # runs still have provenance (B-119 fix 2026-05-15 per codex Mode B P1-2c —
    # previously snapshot ran post-run, crashed runs landed with no env_snapshot.json).
    # output_root is set in ExperimentRunner.__init__, available before run().
    # B-721 (/stress A1.11 P1-5 AC, 2026-05-17): split fail semantics by P79_PAPER_GRADE
    # env. Paper-grade fire (queue scripts that set `P79_PAPER_GRADE=1`) MUST fail loud —
    # missing env_snapshot.json breaks prereg §7 reproducibility audit. Dev mode keeps
    # warning-and-continue for ergonomics.
    try:
        from scripts.provenance.snapshot_env import capture_env_snapshot
        runner.output_root.mkdir(parents=True, exist_ok=True)
        snap_path = runner.output_root / "env_snapshot.json"
        capture_env_snapshot(snap_path, extra={"run_id": cfg["experiment"]["run_id"], "config_path": args.config})
        logging.info("Env snapshot dumped pre-run: %s", snap_path)
    except Exception as e:
        if os.environ.get("P79_PAPER_GRADE", "0") == "1":
            logging.error(
                "Env snapshot FAILED in paper-grade mode (P79_PAPER_GRADE=1) — refusing to run: %s",
                e,
            )
            raise SystemExit(2) from e
        logging.warning("Env snapshot failed (non-fatal, dev mode): %s", e)

    out_dir = runner.run()
    logging.info("Experiment completed: %s", out_dir)


if __name__ == "__main__":
    main()
