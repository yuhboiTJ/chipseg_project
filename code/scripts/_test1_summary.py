"""
Read outputs/test1_predictions.csv and produce:
  - per-chip mean and std across backgrounds
  - overall consistency stats (CoV by chip)
  - figures/test1_per_chip.png (bar chart with std error bars)
"""
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH  = ROOT / "outputs" / "test1_predictions.csv"
OUT_FIG   = ROOT / "outputs" / "figures" / "test1_per_chip.png"
OUT_JSON  = ROOT / "outputs" / "test1_summary.json"


def main():
    rows = list(csv.DictReader(CSV_PATH.open()))
    by_chip = defaultdict(list)
    for r in rows:
        by_chip[r["chip_dir"]].append(float(r["predicted_seedable_mm2"]))

    summary = {}
    for c in sorted(by_chip):
        vals = np.array(by_chip[c])
        summary[c] = {
            "n": int(len(vals)),
            "mean_mm2": float(vals.mean()),
            "std_mm2":  float(vals.std()),
            "min_mm2":  float(vals.min()),
            "max_mm2":  float(vals.max()),
            "cov":      float(vals.std() / vals.mean()) if vals.mean() else 0.0,
        }

    means = np.array([summary[c]["mean_mm2"] for c in sorted(summary)])
    stds  = np.array([summary[c]["std_mm2"] for c in sorted(summary)])
    covs  = np.array([summary[c]["cov"] for c in sorted(summary)])

    overall = {
        "n_chips":           int(len(summary)),
        "n_images":          int(sum(s["n"] for s in summary.values())),
        "mean_per_chip_std": float(stds.mean()),
        "mean_per_chip_cov": float(covs.mean()),
        "min_pred_mm2":      float(means.min()),
        "max_pred_mm2":      float(means.max()),
        "per_chip":          summary,
    }

    OUT_JSON.write_text(json.dumps(overall, indent=2))

    # bar chart
    chips = sorted(summary)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(chips, means, yerr=stds, capsize=4, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("predicted seedable area (mm^2)")
    ax.set_title(f"Testing_dataset_1: per-chip predicted area (mean over backgrounds, error = 1 std)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130)
    plt.close(fig)

    print(f"chips:           {overall['n_chips']}")
    print(f"images:          {overall['n_images']}")
    print(f"per-chip std:    mean = {overall['mean_per_chip_std']:.2f} mm^2")
    print(f"per-chip CoV:    mean = {100 * overall['mean_per_chip_cov']:.1f}%")
    print(f"per-chip mean range: {overall['min_pred_mm2']:.2f} to {overall['max_pred_mm2']:.2f} mm^2")
    print(f"\nwrote {OUT_FIG}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
