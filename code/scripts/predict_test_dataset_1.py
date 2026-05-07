"""
Run inference on Testing_dataset_1 and write predicted seedable areas to a CSV
plus a few qualitative overlays. Testing_dataset_1 has no ground truth so this
is purely qualitative: it lets us see whether the model still produces sane
masks on chips and lighting it has never seen.

Run from code/:
    python scripts/predict_test_dataset_1.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from src.eval import load_model, predict_chip_mask
from src.stage2_classical import remove_defects
from src.visualize import overlay_prediction


TEST_DIR = "../Testing_dataset_1"
OUT_FIG_DIR = "outputs/figures/test1_overlays"
OUT_CSV = "outputs/test1_predictions.csv"
METRICS_PATH = "outputs/metrics.json"
CKPT = "outputs/checkpoints/best.pt"


def main():
    metrics = json.loads(Path(METRICS_PATH).read_text())
    mm2_per_pixel = metrics["calibration"]["mm2_per_pixel_mean"]
    if mm2_per_pixel is None:
        raise RuntimeError("calibration not available, run evaluate.py first")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(CKPT, device=device)

    Path(OUT_FIG_DIR).mkdir(parents=True, exist_ok=True)
    rows = []
    overlay_count = 0

    for img_path in sorted(Path(TEST_DIR).rglob("*.png")):
        chip_dir = img_path.parent.name  # e.g. C01
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        chip_mask = predict_chip_mask(model, img_bgr, device=device)
        seedable, defects = remove_defects(img_bgr, chip_mask)
        seed_pix = int((seedable > 127).sum())
        seed_mm2 = seed_pix * mm2_per_pixel
        rows.append({
            "chip_dir": chip_dir,
            "image": img_path.name,
            "seedable_pixels": seed_pix,
            "predicted_seedable_mm2": round(seed_mm2, 3),
        })
        # save first 2 overlays per chip for inspection
        if overlay_count < 24:
            out_overlay = Path(OUT_FIG_DIR) / f"test1_{img_path.stem}.png"
            overlay_prediction(img_bgr, None, chip_mask, seedable, str(out_overlay))
            overlay_count += 1

    # write csv
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"processed {len(rows)} images from Testing_dataset_1")
    print(f"wrote {OUT_CSV}")
    print(f"saved {overlay_count} overlay png(s) to {OUT_FIG_DIR}")

    # per-chip summary
    per_chip = {}
    for r in rows:
        per_chip.setdefault(r["chip_dir"], []).append(r["predicted_seedable_mm2"])
    print("\nper-chip predicted seedable area (mean across backgrounds):")
    for chip in sorted(per_chip):
        vals = per_chip[chip]
        print(f"  {chip}: {np.mean(vals):6.2f} mm^2  +/- {np.std(vals):.2f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
