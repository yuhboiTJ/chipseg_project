"""
Plotting helpers for figures used in the report and notebook.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(history_csv, out_path):
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE + Dice loss")
    axes[0].set_title("Training and validation loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(df["epoch"], df["val_dice"], label="val Dice")
    axes[1].plot(df["epoch"], df["val_iou"], label="val IoU")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_title("Validation Dice and IoU")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_area_scatter(per_chip_rows, out_path):
    if not per_chip_rows:
        return
    gts = np.array([r["gt_area_mm2"] for r in per_chip_rows])
    preds = np.array([r["pred_area_mm2_mean"] for r in per_chip_rows])

    lo = float(min(gts.min(), preds.min()) * 0.9)
    hi = float(max(gts.max(), preds.max()) * 1.1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x")
    ax.scatter(gts, preds, s=40, alpha=0.8, edgecolor="black", linewidth=0.6)

    for r in per_chip_rows:
        ax.annotate(f"C{r['chip_id']:02d}",
                    (r["gt_area_mm2"], r["pred_area_mm2_mean"]),
                    fontsize=7, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("ground truth area (mm^2)")
    ax.set_ylabel("predicted seedable area (mm^2)")
    ax.set_title("Predicted vs ground truth area (per chip)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def overlay_prediction(image_bgr, gt_mask, pred_chip_mask, seedable_mask, out_path):
    """
    Save a 4-panel figure: original, GT mask, predicted chip mask, predicted seedable mask
    overlaid on the original image.
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("input")
    axes[0].axis("off")

    if gt_mask is not None:
        axes[1].imshow(img_rgb)
        axes[1].imshow(gt_mask > 127, alpha=0.4, cmap="Greens")
        axes[1].set_title("manual mask (gt)")
    else:
        axes[1].imshow(img_rgb)
        axes[1].set_title("no gt mask for this image")
    axes[1].axis("off")

    axes[2].imshow(img_rgb)
    axes[2].imshow(pred_chip_mask > 127, alpha=0.4, cmap="Blues")
    axes[2].set_title("predicted chip mask")
    axes[2].axis("off")

    axes[3].imshow(img_rgb)
    axes[3].imshow(seedable_mask > 127, alpha=0.4, cmap="Reds")
    axes[3].set_title("predicted seedable area")
    axes[3].axis("off")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_calibration_distribution(calibration, out_path):
    per_chip = calibration.get("per_chip", [])
    if not per_chip:
        return
    ratios = np.array([p["mm2_per_pixel"] for p in per_chip])
    chip_ids = [p["chip_id"] for p in per_chip]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(ratios)), ratios * 1e6, color="steelblue")
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([f"C{c:02d}" for c in chip_ids], rotation=45, fontsize=7)
    ax.set_ylabel("mm^2 per pixel  (x1e-6)")
    ax.set_title("Per-chip calibration factor (lower variance is better)")
    ax.axhline(calibration["mean"] * 1e6, color="red", linestyle="--",
               label=f"mean = {calibration['mean']*1e6:.2f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
