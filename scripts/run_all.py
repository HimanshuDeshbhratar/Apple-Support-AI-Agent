"""One-command pipeline for reproducing headline results (<15 min)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n==>", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.check_call(cmd, cwd=ROOT, env=env)


def main() -> None:
    py = sys.executable
    raw = ROOT / "data" / "raw" / "train.csv"
    if not raw.exists():
        run([py, "scripts/download_data.py"])
    run([py, "-m", "src.prepare_data"])
    run([py, "-m", "src.train"])
    run([py, "scripts/build_golden_set.py", "--n", "200"])
    # Refresh auto-judge side of the fixed human panel (if present)
    judge_panel = ROOT / "data" / "golden" / "judge_agreement.jsonl"
    if judge_panel.exists():
        run([py, "scripts/refresh_judge_agreement.py"])
    run([py, "-m", "src.evaluate"])
    print("\nDone. See results/RESULTS.md and results/metrics.json")


if __name__ == "__main__":
    main()
