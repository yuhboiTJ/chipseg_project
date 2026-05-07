"""
CLI: train the U-Net.

Run from the code/ directory:
    python -m scripts.train_model
or:
    python scripts/train_model.py
"""

import argparse
import sys
from pathlib import Path

# allow running directly without installing the package
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import run_training


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="../Training_dataset_2",
                   help="root folder of training images")
    p.add_argument("--masks", default="../Training_dataset_2_ground_truth_masks",
                   help="folder containing user-labeled masks")
    p.add_argument("--output", default="outputs",
                   help="output directory for checkpoints and logs")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-channels", type=int, default=32,
                   help="U-Net base channels. 32 is small (CPU friendly), 64 is standard.")
    p.add_argument("--model-type", choices=["scratch", "pretrained"], default="scratch",
                   help="scratch = our small from-scratch U-Net, pretrained = "
                        "smp.Unet with an ImageNet-pretrained encoder")
    p.add_argument("--encoder", default="resnet34",
                   help="encoder name for --model-type pretrained "
                        "(resnet34, resnet18, mobilenet_v2, efficientnet-b0, ...)")
    args = p.parse_args()

    summary, ckpt = run_training(
        images_root=args.images,
        masks_root=args.masks,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        base_channels=args.base_channels,
        model_type=args.model_type,
        encoder_name=args.encoder,
    )
    print("done.")
    print(f"  best val dice: {summary['best_val_dice']:.4f}")
    print(f"  checkpoint:    {ckpt}")


if __name__ == "__main__":
    main()
