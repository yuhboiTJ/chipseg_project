"""
Pixel to mm^2 calibration. The 24 ground truth area values live in
Training_set_2_ground_truth_areas/ as filenames like c01_z1_17.37.

We back-derive a per-image conversion factor from each user-labeled mask:
    mm2_per_pixel = ground_truth_area_mm2 / mask_pixel_count_in_native_resolution

Note: mask_pixel_count must be measured in the original image resolution,
not the resized 384x512. The trainer resizes for the network but here we
load the raw mask file at native size.
"""

import re
from pathlib import Path

import cv2
import numpy as np


AREA_FILE_RE = re.compile(r"^c(\d{1,2})_z(\d+)_(\d+\.\d+)$", re.IGNORECASE)


def load_area_ground_truth(areas_root):
    """
    Returns dict: {chip_id: {"zoom": int, "area_mm2": float}}.
    Skips junk like FINDER.DAT.
    """
    areas_root = Path(areas_root)
    out = {}
    for entry in areas_root.iterdir():
        if not entry.is_file():
            continue
        m = AREA_FILE_RE.match(entry.name)
        if not m:
            continue
        chip_id = int(m.group(1))
        zoom = int(m.group(2))
        area = float(m.group(3))
        out[chip_id] = {"zoom": zoom, "area_mm2": area}
    return out


def chip_id_from_path(path):
    name = Path(path).name
    m = re.search(r"[Cc](\d{1,2})_", name)
    return int(m.group(1)) if m else None


def per_image_calibration(mask_path, area_mm2):
    """mm^2 per pixel for a single labeled image at native resolution."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"cannot read mask {mask_path}")
    pix = int((mask > 127).sum())
    if pix == 0:
        return None
    return area_mm2 / pix


def fit_calibration(labeled_pairs, areas_root):
    """
    For each (image_path, mask_path) where the chip has a known ground truth area,
    compute mm^2 per pixel. Returns a dict with mean, std, n, per_chip list.
    """
    areas = load_area_ground_truth(areas_root)
    per_chip = []
    for img_path, mask_path in labeled_pairs:
        cid = chip_id_from_path(img_path)
        if cid is None or cid not in areas:
            continue
        gt = areas[cid]["area_mm2"]
        ratio = per_image_calibration(mask_path, gt)
        if ratio is None:
            continue
        per_chip.append({
            "chip_id": cid,
            "area_mm2": gt,
            "mm2_per_pixel": ratio,
            "image": str(img_path),
            "mask": str(mask_path),
        })
    if not per_chip:
        return {"mean": None, "std": None, "n": 0, "per_chip": []}
    ratios = np.array([p["mm2_per_pixel"] for p in per_chip], dtype=np.float64)
    return {
        "mean": float(ratios.mean()),
        "std": float(ratios.std()),
        "median": float(np.median(ratios)),
        "n": len(per_chip),
        "per_chip": per_chip,
    }
