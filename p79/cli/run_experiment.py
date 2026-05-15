from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from p79.experiment.config import load_experiment_config
from p79.experiment.runner import ExperimentRunner
from p79.utils.asyncio_workarounds import install_asyncio_target_closed_warning_filter

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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
        cfg["experiment"]["run_id"] = cfg["experiment"].get("run_id", f"run_{int(time.time())}")

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
    try:
        from scripts.provenance.snapshot_env import capture_env_snapshot
        runner.output_root.mkdir(parents=True, exist_ok=True)
        snap_path = runner.output_root / "env_snapshot.json"
        capture_env_snapshot(snap_path, extra={"run_id": cfg["experiment"]["run_id"], "config_path": args.config})
        logging.info("Env snapshot dumped pre-run: %s", snap_path)
    except Exception as e:
        logging.warning("Env snapshot failed (non-fatal): %s", e)

    out_dir = runner.run()
    logging.info("Experiment completed: %s", out_dir)


if __name__ == "__main__":
    main()
