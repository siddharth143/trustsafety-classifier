import sys
from pathlib import Path

# Automatically add src/ and scripts/ to sys.path for all tests
_BASE_DIR = Path(__file__).resolve().parent.parent
_SRC_DIR = _BASE_DIR / "src"
_SCRIPTS_DIR = _BASE_DIR / "scripts"

for p in [str(_SRC_DIR), str(_SCRIPTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)
