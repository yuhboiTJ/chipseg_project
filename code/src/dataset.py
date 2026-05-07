"""
PyTorch dataset for the microchip images plus the hand-labeled binary masks.
Images: Training_dataset_2/C{NN}/C{NN}_Bg{N}_z1.png
Masks:  Training_dataset_2_ground_truth_masks/.../<stem>_mask.{png,jpg,tif}

The matching is forgiving because masks were drawn by hand in ImageJ and the
filenames came out a bit varied (different extensions, typed 'co1' instead of
'c01', sometimes no z1 suffix, sometimes nested in per-chip subfolders).
We match images to masks by extracting (chip_number, background_number) from
each filename and pairing on that key.
"""

import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

# silence the noisy TIFF warnings from imagej-saved masks
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


IMG_SIZE = (288, 384)  # H, W. Native is 576x720. Smaller is faster on CPU.

MASK_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _parse_chip_bg(stem):
    """
    Pull (chip_id, bg_id) from a filename stem.
    Accepts: C01_Bg1_z1, c01_bg1, co1_bg1 (typo: o for 0), C24_Bg7, etc.
    Returns None if it cannot find both.
    """
    # chip part: c followed by digits, possibly with letter o/O typed instead of 0
    # bg part: bg followed by digits
    m = re.match(r"^[cC]([0-9oO]+)[_\s]*[bB][gG](\d+)", stem)
    if not m:
        return None
    chip_str = m.group(1).replace("o", "0").replace("O", "0")
    try:
        chip_id = int(chip_str)
        bg_id = int(m.group(2))
    except ValueError:
        return None
    return chip_id, bg_id


def list_labeled_pairs(images_root, masks_root):
    """
    Returns a list of (image_path, mask_path) for every image whose chip and
    background number can be matched to a mask file. The masks tree is walked
    recursively. If multiple mask files match the same (chip, bg) the first
    one found is used.
    """
    images_root = Path(images_root)
    masks_root = Path(masks_root)

    mask_by_key = {}
    if masks_root.exists():
        for mp in sorted(masks_root.rglob("*")):
            if not mp.is_file():
                continue
            if mp.suffix.lower() not in MASK_EXTS:
                continue
            stem = mp.stem
            # strip a trailing _mask if present
            low = stem.lower()
            if low.endswith("_mask"):
                stem = stem[: -len("_mask")]
            key = _parse_chip_bg(stem)
            if key is not None and key not in mask_by_key:
                mask_by_key[key] = str(mp)

    pairs = []
    skipped_empty = []
    for img_path in sorted(images_root.rglob("*.png")):
        key = _parse_chip_bg(img_path.stem)
        if key is None:
            continue
        if key not in mask_by_key:
            continue
        mask_path = mask_by_key[key]
        # skip blank masks (labeling miss)
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None or (m > 127).sum() == 0:
            skipped_empty.append(mask_path)
            continue
        pairs.append((str(img_path), mask_path))
    if skipped_empty:
        print(f"warning: skipped {len(skipped_empty)} empty mask file(s):")
        for s in skipped_empty:
            print(f"  {s}")
    return pairs


def chip_id_from_filename(name):
    """Returns chip id (1..24) from a filename, or None."""
    stem = Path(name).stem
    parsed = _parse_chip_bg(stem)
    return parsed[0] if parsed else None


def train_augmentations(img_size=IMG_SIZE):
    h, w = img_size
    return A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # no RandomRotate90 here: the input is rectangular (384x512) so a 90 deg rotate
        # swaps H and W and breaks batching. The affine rotate below covers small rotations.
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-20, 20), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def eval_augmentations(img_size=IMG_SIZE):
    h, w = img_size
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class MicrochipDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform if transform is not None else eval_augmentations()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read image {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask {mask_path}")
        mask = (mask > 127).astype(np.uint8)

        out = self.transform(image=img, mask=mask)
        image_t = out["image"].float()
        mask_t = out["mask"].float().unsqueeze(0)  # 1xHxW
        return image_t, mask_t, os.path.basename(img_path)


def split_pairs(pairs, val_frac=0.2, seed=42):
    """Random deterministic 80/20 split of (image, mask) pairs.
    Note: this can place different backgrounds of the same chip on opposite sides
    of the split, which leaks chip identity. Prefer split_pairs_by_chip when you
    have at least a dozen distinct chips labeled.
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(pairs) * val_frac)))
    val_idx = set(idx[:n_val].tolist())
    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in val_idx]
    val_pairs = [pairs[i] for i in range(len(pairs)) if i in val_idx]
    return train_pairs, val_pairs


# default validation chips for chip-level split.
# chosen to span the area range (mm^2): C20=5.49, C18=10.22, C12=18.68, C07=28.43
DEFAULT_VAL_CHIPS = (7, 12, 18, 20)


def split_pairs_by_chip(pairs, val_chip_ids=DEFAULT_VAL_CHIPS):
    """Hold out every mask of every chip in val_chip_ids for validation.
    No mask of a val chip ever appears in training, so the val metric measures
    true generalization to unseen chips.
    """
    val_set = set(int(c) for c in val_chip_ids)
    train_pairs, val_pairs = [], []
    for img, mask in pairs:
        cid_bg = _parse_chip_bg(Path(img).stem)
        if cid_bg is None:
            train_pairs.append((img, mask))
            continue
        cid = cid_bg[0]
        if cid in val_set:
            val_pairs.append((img, mask))
        else:
            train_pairs.append((img, mask))
    return train_pairs, val_pairs
