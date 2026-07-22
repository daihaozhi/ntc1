"""Export NTC model to runtime assets (thin CLI wrapper).

Usage:
    python scripts/export.py --checkpoint model_best.pth --output_dir exported/ --texture_resolution 4096
"""

import sys
from pathlib import Path

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.exporter import main

if __name__ == "__main__":
    main()
