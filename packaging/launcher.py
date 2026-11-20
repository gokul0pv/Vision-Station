"""VisionStation.exe starts here (see vision_station.spec)."""
import os
import sys

# Networks on the graphics card when this PC's card can carry them, else the
# processor -- chosen from the PC itself (core/system_profile.py); set
# VS_DEVICE=CPU or GPU before starting to choose by hand.
from vision_station.core import system_profile  # noqa: E402
system_profile.apply_device()

from vision_station.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
