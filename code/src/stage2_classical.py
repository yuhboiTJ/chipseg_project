"""
Stage 2 inside the predicted chip mask.

Two things to clean up after stage 1:
  1. Shadows. The chip casts a shadow on the background paper. The U-Net
     sometimes pulls the shadow into the chip mask because the shadow is
     darker than the surrounding paper. The shadow keeps the paper's hue
     though, so it is colored, while the chip itself is achromatic
     (gray/black silicon). We use saturation in HSV to peel off the shadow.
  2. Bright specks/edge debris on the chip surface that are not seedable.
     Found by Otsu thresholding the brightness of the now-refined chip
     pixels.

This step runs in milliseconds and does not need additional labeled data.
"""

import cv2
import numpy as np


def _largest_component(binary):
    """Keep only the largest connected component of a binary image (uint8 0/255)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary
    sizes = stats[1:, cv2.CC_STAT_AREA]
    if len(sizes) == 0:
        return binary
    largest = 1 + int(np.argmax(sizes))
    out = np.zeros_like(binary)
    out[labels == largest] = 255
    return out


def remove_shadow(image_bgr, chip_mask, sat_percentile=70, val_offset=20):
    """
    Refine the predicted chip mask by removing shadow regions.

    Shadow pixels in the predicted mask have non-trivial saturation (they
    keep the paper's color) and are noticeably brighter than the chip
    surface. Within the predicted mask we keep pixels whose saturation is
    in the lower part of the chip-pixel distribution AND whose brightness
    is at most val_offset above the median chip brightness, then keep only
    the largest connected component to drop disconnected shadow blobs.
    """
    chip_bool = chip_mask > 127
    if not chip_bool.any():
        return np.zeros_like(chip_mask)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    val = hsv[..., 2]

    chip_sat = sat[chip_bool]
    chip_val = val[chip_bool]

    sat_thresh = float(np.percentile(chip_sat, sat_percentile))
    sat_thresh = max(sat_thresh, 30.0)  # do not be too aggressive

    val_med = float(np.median(chip_val))
    val_thresh = val_med + val_offset

    refined = chip_bool & (sat <= sat_thresh) & (val <= val_thresh)
    refined_u8 = (refined.astype(np.uint8)) * 255

    # close small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_u8 = cv2.morphologyEx(refined_u8, cv2.MORPH_CLOSE, kernel)

    refined_u8 = _largest_component(refined_u8)
    return refined_u8


def find_extreme_defects(image_bgr, chip_mask, top_percent=2.0, min_area=15, max_area_frac=0.05):
    """
    Find unambiguously bright defect pixels inside the chip mask.
    Conservative: only the top top_percent brightest pixels, only
    components with area >= min_area, and one component cannot exceed
    max_area_frac of the chip (which would mean we are catching a regular
    surface pattern, not a defect).
    """
    chip_bool = chip_mask > 127
    if not chip_bool.any():
        return np.zeros_like(chip_mask)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    chip_pixels = gray[chip_bool]
    if len(chip_pixels) < 100:
        return np.zeros_like(chip_mask)

    cutoff = int(np.percentile(chip_pixels, 100.0 - top_percent))
    bright = ((gray > cutoff) & chip_bool).astype(np.uint8) * 255

    chip_area = int(chip_bool.sum())
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    cleaned = np.zeros_like(bright)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < min_area:
            continue
        if a > chip_area * max_area_frac:
            continue
        cleaned[labels == i] = 255
    return cleaned


def remove_defects(image_bgr, chip_mask, drop_defects=False):
    """
    Stage 2 = shadow refinement of the U-Net's predicted chip mask.

    Defect removal is off by default because the training masks already
    include or exclude on-chip features per the labeler's judgement, and
    the U-Net learned that distinction. Aggressive defect removal here
    would unlearn it (e.g. peel off a regular grid of bright wells on a
    larger chip).

    Pass drop_defects=True to also remove very-bright small specks (top 2
    percent of intensity, small connected components only). Useful for
    images with obvious edge debris.

    Returns (seedable_mask uint8 binary, defect_mask uint8 binary).
    """
    if image_bgr.dtype != np.uint8:
        raise ValueError("expected uint8 image")
    if chip_mask.shape[:2] != image_bgr.shape[:2]:
        raise ValueError(f"shape mismatch: image {image_bgr.shape}, mask {chip_mask.shape}")

    chip_bool_in = chip_mask > 127
    if not chip_bool_in.any():
        empty = np.zeros_like(chip_mask)
        return empty, empty

    chip_refined = remove_shadow(image_bgr, chip_mask)
    if (chip_refined > 127).sum() == 0:
        chip_refined = chip_mask  # safety fallback

    if drop_defects:
        defects = find_extreme_defects(image_bgr, chip_refined)
        seedable = cv2.bitwise_and(chip_refined, cv2.bitwise_not(defects))
    else:
        defects = np.zeros_like(chip_mask)
        seedable = chip_refined

    return seedable, defects


def predicted_seedable_area_mm2(image_bgr, chip_mask, mm2_per_pixel, drop_defects=False):
    """Convenience: returns seedable area in mm^2 plus the masks."""
    seedable, defects = remove_defects(image_bgr, chip_mask, drop_defects=drop_defects)
    pix = int((seedable > 127).sum())
    return pix * mm2_per_pixel, seedable, defects
