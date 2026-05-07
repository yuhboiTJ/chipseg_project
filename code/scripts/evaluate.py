"""
CLI: evaluate a trained checkpoint.
Computes IoU/Dice on the labeled validation split, fits the pixel to mm^2
calibration, and computes per-chip area MAE/R^2 against the ground truth.
Writes outputs/metrics.json plus figures into outputs/figures/.

Run from code/:
    python scripts/evaluate.py
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import torch

from src.dataset import list_labeled_pairs, split_pairs_by_chip, DEFAULT_VAL_CHIPS
from src.calibration import fit_calibration
from src.eval import load_model, mask_metrics_on_pairs, area_metrics_per_chip, write_metrics_json
from src.visualize import (
    plot_training_curves, plot_area_scatter, plot_calibration_distribution,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="../Training_dataset_2")
    p.add_argument("--masks", default="../Training_dataset_2_ground_truth_masks")
    p.add_argument("--areas", default="../Training_set_2_ground_truth_areas")
    p.add_argument("--ckpt", default="outputs/checkpoints/best.pt")
    p.add_argument("--output", default="outputs")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.output)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    pairs = list_labeled_pairs(args.images, args.masks)
    if not pairs:
        raise RuntimeError("no labeled pairs found, cannot evaluate.")

    # match the split used during training. read the val chips from
    # training_summary.json if it exists, otherwise use the default.
    val_chips = DEFAULT_VAL_CHIPS
    summary_path = Path(args.output) / "training_summary.json"
    if summary_path.exists():
        try:
            ts = json.loads(summary_path.read_text())
            if "val_chips" in ts:
                val_chips = tuple(ts["val_chips"])
        except Exception:
            pass
    train_pairs, val_pairs = split_pairs_by_chip(pairs, val_chip_ids=val_chips)
    print(f"chip-level split. val chips = {sorted(val_chips)}    train: {len(train_pairs)}    val: {len(val_pairs)}")

    print(f"calibration on {len(pairs)} labeled images...")
    calibration = fit_calibration(pairs, args.areas)
    print(f"  mean mm^2/px = {calibration['mean']}    n = {calibration['n']}")
    if calibration["mean"] is None:
        raise RuntimeError("calibration failed: no labeled chips matched a ground truth area.")

    plot_calibration_distribution(calibration,
                                  figs_dir / "calibration_per_chip.png")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading model ({device})...")
    model, ckpt_meta = load_model(args.ckpt, device=device)

    print(f"evaluating mask quality on {len(val_pairs)} val images...")
    mask_rows = mask_metrics_on_pairs(model, val_pairs, device=device)
    mean_iou = sum(r["iou"] for r in mask_rows) / max(1, len(mask_rows))
    mean_dice = sum(r["dice"] for r in mask_rows) / max(1, len(mask_rows))
    print(f"  mean IoU  = {mean_iou:.4f}")
    print(f"  mean Dice = {mean_dice:.4f}")

    print("evaluating end-to-end area prediction across all chips...")
    area_summary = area_metrics_per_chip(
        model, args.images, args.areas,
        mm2_per_pixel=calibration["mean"], device=device,
    )
    print(f"  area MAE  = {area_summary['mae_mm2']:.3f} mm^2")
    print(f"  area RMSE = {area_summary['rmse_mm2']:.3f} mm^2")
    print(f"  area R^2  = {area_summary['r2']}")

    plot_area_scatter(area_summary["per_chip"], figs_dir / "area_scatter.png")
    plot_training_curves(out_dir / "training_history.csv",
                         figs_dir / "training_curves.png")

    training_summary_path = out_dir / "training_summary.json"
    training_summary = {}
    if training_summary_path.exists():
        training_summary = json.loads(training_summary_path.read_text())

    metrics = write_metrics_json(out_dir / "metrics.json",
                                 mask_rows, area_summary, calibration, training_summary)
    print(f"wrote {out_dir / 'metrics.json'}")
    print("done.")


if __name__ == "__main__":
    main()
