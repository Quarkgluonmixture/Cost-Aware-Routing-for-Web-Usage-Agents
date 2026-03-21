from __future__ import annotations

import argparse
import logging

from p79.experiment.analysis import analyze_run


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
