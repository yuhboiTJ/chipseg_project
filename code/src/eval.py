"""
Evaluation routines.
1. Mask quality on the labeled validation set (IoU and Dice).
2. End to end seedable area in mm^2 vs ground truth area (MAE and R^2),
   averaged across all images of each chip.
"""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import r2_score

from .dataset import (
    eval_augmentations, list_labeled_pairs, IMG_SIZE,
)
from .calibration import (
    load_area_ground_truth, chip_id_from_path,
)
from .stage2_classical import remove_defects
from .model import UNet, make_model


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = ckpt.get("model_type", "scratch")
    base = ckpt.get("base_channels", 32)
    encoder = ckpt.get("encoder_name", "resnet34")
    model = make_model(
        model_type=model_type,
        base_channels=base,
        encoder_name=encoder,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def _preprocess(img_rgb):
    h, w = IMG_SIZE
    transform = eval_augmentations()
    out = transform(image=img_rgb, mask=np.zeros(img_rgb.shape[:2], dtype=np.uint8))
    return out["image"].unsqueeze(0).float()


@torch.no_grad()
def predict_chip_mask(model, image_bgr, device="cpu", threshold=0.5):
    """Run inference on one image. Returns chip mask at native (input) resolution, uint8 0/255."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    native_h, native_w = image_bgr.shape[:2]
    x = _preprocess(img_rgb).to(device)
    logits = model(x)
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    pred_small = (probs > threshold).astype(np.uint8) * 255
    pred = cv2.resize(pred_small, (native_w, native_h), interpolation=cv2.INTER_NEAREST)
    return pred


def mask_metrics_on_pairs(model, pairs, device="cpu"):
    """Compute per-image IoU and Dice for predicted chip mask vs the user's GT mask."""
    rows = []
    for img_path, mask_path in pairs:
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_bin = (gt > 127).astype(np.uint8)
        pred = predict_chip_mask(model, image_bgr, device=device)
        pred_bin = (pred > 127).astype(np.uint8)
        inter = int((pred_bin & gt_bin).sum())
        union = int((pred_bin | gt_bin).sum())
        gt_sum = int(gt_bin.sum())
        pr_sum = int(pred_bin.sum())
        iou = inter / union if union > 0 else 0.0
        dice = (2.0 * inter) / (pr_sum + gt_sum) if (pr_sum + gt_sum) > 0 else 0.0
        rows.append({
            "image": Path(img_path).name,
            "iou": iou,
            "dice": dice,
            "gt_pixels": gt_sum,
            "pred_pixels": pr_sum,
        })
    return rows


def area_metrics_per_chip(model, images_root, areas_root, mm2_per_pixel, device="cpu"):
    """
    For every image in images_root, predict chip mask, run stage 2, compute seedable
    area in mm^2. Then group by chip id, average, and compare to the ground truth.
    Returns dict with per-chip rows and aggregate metrics.
    """
    images_root = Path(images_root)
    areas = load_area_ground_truth(areas_root)

    by_chip = defaultdict(list)
    for img_path in sorted(images_root.rglob("*.png")):
        cid = chip_id_from_path(img_path)
        if cid is None:
            continue
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        chip_mask = predict_chip_mask(model, image_bgr, device=device)
        seedable, _defects = remove_defects(image_bgr, chip_mask)
        pix = int((seedable > 127).sum())
        area_pred = pix * mm2_per_pixel
        by_chip[cid].append({
            "image": Path(img_path).name,
            "pred_pixels": pix,
            "pred_area_mm2": area_pred,
        })

    rows = []
    for cid in sorted(by_chip.keys()):
        if cid not in areas:
            continue
        preds = [r["pred_area_mm2"] for r in by_chip[cid]]
        gt = areas[cid]["area_mm2"]
        mean_pred = float(np.mean(preds))
        std_pred = float(np.std(preds))
        rows.append({
            "chip_id": cid,
            "n_images": len(preds),
            "gt_area_mm2": gt,
            "pred_area_mm2_mean": mean_pred,
            "pred_area_mm2_std": std_pred,
            "abs_error_mm2": abs(mean_pred - gt),
            "rel_error": abs(mean_pred - gt) / gt if gt > 0 else 0.0,
        })

    if not rows:
        return {"per_chip": [], "mae_mm2": None, "r2": None, "rmse_mm2": None}

    gts = np.array([r["gt_area_mm2"] for r in rows])
    preds = np.array([r["pred_area_mm2_mean"] for r in rows])
    errors = preds - gts
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    try:
        r2 = float(r2_score(gts, preds))
    except Exception:
        r2 = None
    return {
        "per_chip": rows,
        "mae_mm2": mae,
        "rmse_mm2": rmse,
        "r2": r2,
        "n_chips": len(rows),
    }


def write_metrics_json(out_path, mask_rows, area_summary, calibration_summary, training_summary):
    iou_vals = [r["iou"] for r in mask_rows] if mask_rows else []
    dice_vals = [r["dice"] for r in mask_rows] if mask_rows else []
    payload = {
        "training": training_summary,
        "calibration": {
            "mm2_per_pixel_mean": calibration_summary.get("mean"),
            "mm2_per_pixel_std": calibration_summary.get("std"),
            "mm2_per_pixel_median": calibration_summary.get("median"),
            "n_calibration_chips": calibration_summary.get("n"),
        },
        "mask_quality_val": {
            "n_images": len(mask_rows),
            "iou_mean": float(np.mean(iou_vals)) if iou_vals else None,
            "iou_std": float(np.std(iou_vals)) if iou_vals else None,
            "dice_mean": float(np.mean(dice_vals)) if dice_vals else None,
            "dice_std": float(np.std(dice_vals)) if dice_vals else None,
            "per_image": mask_rows,
        },
        "area_quality_per_chip": area_summary,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload
