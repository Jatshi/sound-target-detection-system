from __future__ import annotations

import subprocess
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standard sound detection regression checks.")
    parser.add_argument("--quick", action="store_true", help="Run only the quick online replay evaluation stage.")
    args = parser.parse_args()
    run([PY, "scripts/smoke_model.py"])
    run([PY, "scripts/diagnose_offline_consistency.py", "--dataset", "OOD-B", "--model", "neurocap_sound_only", "--n", "128"])
    run([PY, "scripts/run_streaming_trial_eval.py", "--quick"])
    if not args.quick:
        run([PY, "scripts/summarize_online_results.py"])
    print("Regression evaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

