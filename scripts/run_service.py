from __future__ import annotations

import sys
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sounddet.config import load_config


def main() -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the Sound Target Detection FastAPI service.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    os.environ["SOUNDDET_CONFIG"] = args.config
    uvicorn.run("sounddet.service:app", host=args.host or cfg.service_host, port=args.port or cfg.service_port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
