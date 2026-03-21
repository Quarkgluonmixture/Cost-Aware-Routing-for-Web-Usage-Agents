from __future__ import annotations

import argparse
import logging
import time

from p79.experiment.config import load_experiment_config
from p79.experiment.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified experiment runner for P79")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--run_id", default=None, help="Optional run ID override")
    parser.add_argument("--phase", default=None, choices=["phase1", "phase2", "phase3"], help="Optional phase override")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional max step override")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = load_experiment_config(args.config)
    if args.run_id:
        cfg["experiment"]["run_id"] = args.run_id
    else:
        cfg["experiment"]["run_id"] = cfg["experiment"].get("run_id", f"run_{int(time.time())}")

    if args.phase:
        cfg["experiment"]["phase"] = args.phase
    if args.max_steps is not None:
        cfg.setdefault("runtime", {})["max_steps"] = args.max_steps

    runner = ExperimentRunner(cfg)
    out_dir = runner.run()
    logging.info("Experiment completed: %s", out_dir)


if __name__ == "__main__":
    main()
