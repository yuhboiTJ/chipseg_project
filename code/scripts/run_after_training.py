"""
Convenience wrapper: after training completes, run evaluation, finalize the
report with the real numbers, and re-execute the notebook in place so the
saved version has all the cell outputs.

Run from code/:
    python scripts/run_after_training.py
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent  # code/


def run(cmd, cwd=None):
    print()
    print("=" * 60)
    print("running:", " ".join(cmd))
    print("=" * 60)
    result = subprocess.run(cmd, cwd=cwd or ROOT)
    if result.returncode != 0:
        print(f"command failed with exit {result.returncode}")
        sys.exit(result.returncode)


def main():
    py = sys.executable
    run([py, "scripts/evaluate.py"])
    run([py, "scripts/finalize_report.py"])
    # execute the notebook end-to-end so saved cells contain outputs
    nb = ROOT / "notebooks" / "01_full_pipeline.ipynb"
    run([py, "-m", "jupyter", "nbconvert",
         "--to", "notebook", "--execute",
         "--inplace",
         "--ExecutePreprocessor.timeout=600",
         str(nb)])
    print()
    print("all post-training steps complete.")


if __name__ == "__main__":
    main()
