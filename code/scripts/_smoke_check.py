"""one-off smoke check: load masks, run calibration, print summary."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.dataset import list_labeled_pairs
from src.calibration import fit_calibration


pairs = list_labeled_pairs("../Training_dataset_2", "../Training_dataset_2_ground_truth_masks")
print("pairs:", len(pairs))

print("\nmask file inspection (first 12):")
for img_path, mask_path in pairs[:12]:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path)
    if mask is None:
        print(f"  FAILED to load: {mask_path}")
        continue
    nname = Path(mask_path).name
    binary = int((mask > 127).sum())
    img_shape = None if img is None else img.shape[:2]
    print(f"  {nname:30s} mask={mask.shape} dtype={mask.dtype} img={img_shape} white_px={binary}")

calib = fit_calibration(pairs, "../Training_set_2_ground_truth_areas")
print(f"\nchips with calibration: {calib['n']}")
if calib["n"]:
    print(f"mean   mm^2/px: {calib['mean']:.6e}")
    print(f"median mm^2/px: {calib['median']:.6e}")
    print(f"std    mm^2/px: {calib['std']:.6e}")
    cov = 100 * calib["std"] / calib["mean"]
    print(f"CoV: {cov:.2f}%")
    print("\nper-chip:")
    for r in calib["per_chip"]:
        print(f"  C{r['chip_id']:02d}  area={r['area_mm2']:6.2f} mm^2  ratio={r['mm2_per_pixel']:.4e}")
