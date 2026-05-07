"""
Run a trained model on a single image and print the predicted seedable area.
Saves a side-by-side overlay PNG.

Usage from code/:
    python scripts/predict_image.py --image ../Training_dataset_2/C03/C03_Bg2_z1.png
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import torch

from src.eval import load_model, predict_chip_mask
from src.stage2_classical import remove_defects
from src.visualize import overlay_prediction


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--ckpt", default="outputs/checkpoints/best.pt")
    p.add_argument("--metrics", default="outputs/metrics.json",
                   help="used to read mm^2 per pixel calibration")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"cannot read image {args.image}")

    metrics = json.loads(Path(args.metrics).read_text())
    mm2_per_pixel = metrics["calibration"]["mm2_per_pixel_mean"]
    if mm2_per_pixel is None:
        raise RuntimeError("calibration not available in metrics.json. run evaluate.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.ckpt, device=device)

    chip_mask = predict_chip_mask(model, image_bgr, device=device)
    seedable, defects = remove_defects(image_bgr, chip_mask)
    pix = int((seedable > 127).sum())
    area_mm2 = pix * mm2_per_pixel

    print(f"image:           {args.image}")
    print(f"chip pixels:     {int((chip_mask > 127).sum())}")
    print(f"defect pixels:   {int((defects > 127).sum())}")
    print(f"seedable pixels: {pix}")
    print(f"seedable area:   {area_mm2:.2f} mm^2")

    out_path = args.out or f"outputs/predictions/{Path(args.image).stem}_pred.png"
    overlay_prediction(image_bgr, None, chip_mask, seedable, out_path)
    print(f"saved overlay to {out_path}")


if __name__ == "__main__":
    main()
