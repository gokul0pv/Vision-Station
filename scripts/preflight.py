"""Check that everything Vision Station needs is installed.

    python scripts/preflight.py

The checks themselves are vision_station.core.preflight, which the
application also runs (Settings > General > Check installation).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the project root

from vision_station.core.preflight import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
