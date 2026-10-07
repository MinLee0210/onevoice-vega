"""Make the qaihub helper module importable from the tests.

The scripts under ``scripts/qaihub/`` do ``import common`` and are meant to
be run from that directory; this adds it to ``sys.path`` so the test
suite can import the same module.
"""

from __future__ import annotations

import sys
from pathlib import Path

QAIHUB_DIR = Path(__file__).resolve().parents[1] / "scripts" / "qaihub"
if str(QAIHUB_DIR) not in sys.path:
    sys.path.insert(0, str(QAIHUB_DIR))
