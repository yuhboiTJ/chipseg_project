"""
Save side-by-side prediction overlays for a few representative images.
Picks one labeled chip (so we have a GT mask to show) and a couple of
unlabeled chips to demonstrate generalization.

Run from code/:
    python scripts/save_sample_overlays.py
"""

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
from src.dataset import list_labeled_pairs


def main():
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "outputs/checkpoints/best.pt"
    model, _ = load_model(ckpt, device=device)

    pairs = list_labeled_pairs("../Training_dataset_2",
                               "../Training_dataset_2_ground_truth_masks")
    pair_map = {Path(img).stem: mask for img, mask in pairs}

    targets = [
        # one labeled (in-distribution)
        "../Training_dataset_2/C01/C01_Bg1_z1.png",
        "../Training_dataset_2/C04/C04_Bg2_z1.png",
        # two unlabeled chips (the model never saw these chips)
        "../Training_dataset_2/C12/C12_Bg1_z1.png",
        "../Training_dataset_2/C20/C20_Bg3_z1.png",
        "../Training_dataset_2/C24/C24_Bg5_z1.png",
    ]

    saved = []
    for t in targets:
        if not Path(t).exists():
            print(f"skip missing: {t}")
            continue
        img_bgr = cv2.imread(t)
        if img_bgr is None:
            print(f"failed to read: {t}")
            continue

        chip = predict_chip_mask(model, img_bgr, device=device)
        seedable, _defects = remove_defects(img_bgr, chip)

        stem = Path(t).stem
        gt_path = pair_map.get(stem)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if gt_path else None

        out_path = out_dir / f"sample_overlay_{stem}.png"
        overlay_prediction(img_bgr, gt_mask, chip, seedable, out_path)
        saved.append(str(out_path))
        print(f"saved {out_path}")

    print(f"\ntotal saved: {len(saved)}")


if __name__ == "__main__":
    main()
