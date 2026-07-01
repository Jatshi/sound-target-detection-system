from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.executable).resolve().parent
    if not (root / "scripts" / "run_desktop.py").exists():
        root = Path(__file__).resolve().parents[1]
    python = os.environ.get("SOUNDDET_PYTHON", r"python")
    script = root / "scripts" / "run_desktop.py"
    if not script.exists():
        print(f"Cannot find desktop entry: {script}", file=sys.stderr)
        return 2
    return subprocess.call([python, str(script)], cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())
