"""Per-run diagnostic wrapper; not part of the 4-dimension evidence framework."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from p79.cli.analyze_experiment import main


if __name__ == "__main__":
    main()
