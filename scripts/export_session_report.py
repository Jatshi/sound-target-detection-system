from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sounddet.config import load_config, resolve_app_path
from sounddet.event_store import EventStore
from sounddet.reporting import export_session_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a session report package.")
    parser.add_argument("session_id", nargs="?")
    parser.add_argument("--session-id", dest="session_id_opt", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    session_id = args.session_id_opt or args.session_id
    if not session_id:
        parser.error("session_id is required")
    cfg = load_config()
    store = EventStore(cfg)
    out = Path(args.out) if args.out else resolve_app_path(cfg, cfg.report_dir) / session_id
    package = export_session_report(store, session_id, out)
    print(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
