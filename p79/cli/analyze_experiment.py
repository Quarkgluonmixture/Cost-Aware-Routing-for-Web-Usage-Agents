from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# B-718 (/stress A1.11 P0-2 B* OOB, 2026-05-17): bootstrap repo root into sys.path
# BEFORE p79.* imports. Pre-fix this file had ZERO bootstrap — absolute-path
# invocation died with `ModuleNotFoundError: No module named 'p79'`. Watchdog
# auto-analysis path + manual recovery dispatch through scripts/analysis shim
# OR via pyproject entry point, but absolute-path invocation is the cold-start
# default if pip install -e . has not been run.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from p79.experiment.analysis import analyze_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze unified P79 experiment outputs")
    parser.add_argument("--run_dir", required=True, help="Path to run directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        out = analyze_run(args.run_dir)
    except RuntimeError as exc:
        logging.error("%s", exc)
        raise SystemExit(2) from exc
    logging.info("Analysis outputs written to: %s", out)


if __name__ == "__main__":
    main()
