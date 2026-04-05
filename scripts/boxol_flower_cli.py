#!/usr/bin/env python3
"""
Thin shim: delegates to boxol_flower.cli so this script can be run directly
from the repo root without installing the package.

Usage::

    PYTHONPATH=src python scripts/boxol_flower_cli.py --headless
    PYTHONPATH=src python scripts/boxol_flower_cli.py --gui
"""

from __future__ import annotations

import os
import sys

# Ensure src/ is on the path when running this script directly.
_src = os.path.join(os.path.dirname(__file__), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from boxol_flower.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
