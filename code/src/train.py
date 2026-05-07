"""
Training loop for the U-Net. Combined BCE + Dice loss, Adam, plateau LR scheduler.
Saves the best-by-val-Dice checkpoint and a per-epoch metrics CSV.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    MicrochipDataset, list_labeled_pairs,
    split_pairs, split_pairs_by_chip, DEFAULT_VAL_CHIPS,
    train_augmentations, eval_augmentations,
)
from .model import UNet, count_parameters, make_model


def dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce + dice_loss(logits, target)


def dice_score(logits, target, eps=1e-6):
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > 0.5).float()
        dims = (1, 2, 3)
        intersection = (preds * target).sum(dim=dims)
        union = preds.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * intersection + eps) / (union + eps)
        return dice.mean().item()


def iou_score(logits, target, eps=1e-6):
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > 0.5).float()
        dims = (1, 2, 3)
        intersection = (preds * target).sum(dim=dims)
        union = preds.sum(dim=dims) + target.sum(dim=dims) - intersection
        iou = (intersection + eps) / (union + eps)
        return iou.mean().item()


def run_training(
    images_root,
    masks_root,
    output_dir,
    epochs=50,
    batch_size=4,
    lr=1e-3,
    val_frac=0.2,
    seed=42,
    num_workers=0,
    device=None,
    base_channels=32,
    val_chips=None,
    model_type="scratch",
    encoder_name="resnet34",
):
    output_dir = Path(output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    pairs = list_labeled_pairs(images_root, masks_root)
    if len(pairs) == 0:
        raise RuntimeError(
            f"no labeled (image, mask) pairs found.\n"
            f"  images_root = {images_root}\n"
            f"  masks_root  = {masks_root}\n"
            f"check that masks were saved with the *_mask.png suffix."
        )
    print(f"found {len(pairs)} labeled pairs")

    if val_chips is None:
        val_chips = DEFAULT_VAL_CHIPS
    train_pairs, val_pairs = split_pairs_by_chip(pairs, val_chip_ids=val_chips)
    print(f"chip-level split. val chips = {sorted(val_chips)}")
    print(f"train: {len(train_pairs)}    val: {len(val_pairs)}")

    train_ds = MicrochipDataset(train_pairs, transform=train_augmentations())
    val_ds = MicrochipDataset(val_pairs, transform=eval_augmentations())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = make_model(
        model_type=model_type,
        base_channels=base_channels,
        encoder_name=encoder_name,
    ).to(device)
    print(f"model_type: {model_type}" +
          (f"  encoder: {encoder_name}" if model_type == "pretrained" else ""))
    print(f"params: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    history = []
    best_val_dice = -1.0
    ckpt_path = output_dir / "checkpoints" / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_losses = []
        for img, mask, _ in train_loader:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            logits = model(img)
            loss = bce_dice_loss(logits, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, val_dices, val_ious = [], [], []
        with torch.no_grad():
            for img, mask, _ in val_loader:
                img = img.to(device)
                mask = mask.to(device)
                logits = model(img)
                val_losses.append(bce_dice_loss(logits, mask).item())
                val_dices.append(dice_score(logits, mask))
                val_ious.append(iou_score(logits, mask))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        val_dice = float(np.mean(val_dices)) if val_dices else 0.0
        val_iou = float(np.mean(val_ious)) if val_ious else 0.0
        elapsed = time.time() - t0

        scheduler.step(val_dice)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
        })
        print(f"epoch {epoch:3d}/{epochs}  "
              f"train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
              f"val_dice {val_dice:.4f}  val_iou {val_iou:.4f}  "
              f"lr {optimizer.param_groups[0]['lr']:.2e}  ({elapsed:.1f}s)")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "base_channels": base_channels,
                "model_type": model_type,
                "encoder_name": encoder_name,
            }, ckpt_path)

    df = pd.DataFrame(history)
    df.to_csv(output_dir / "training_history.csv", index=False)

    summary = {
        "n_pairs": len(pairs),
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "split_strategy": "chip_level",
        "val_chips": sorted(int(c) for c in val_chips),
        "best_val_dice": best_val_dice,
        "final_val_dice": history[-1]["val_dice"],
        "final_val_iou": history[-1]["val_iou"],
        "epochs": epochs,
        "device": device,
        "model_type": model_type,
        "encoder_name": encoder_name if model_type == "pretrained" else None,
        "param_count": count_parameters(model),
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary, ckpt_path
