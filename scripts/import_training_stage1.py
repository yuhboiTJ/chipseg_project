# scripts/import_training_stage1.py
import os, re, glob, shutil, argparse, cv2, numpy as np

# Regex for names like C01_Bg1_Z1.png or C01_BG1_Z1_Mask.png (case-insensitive)
NAME_RE = re.compile(r"^c(\d+)_b[gG](\d+)_z(\d+)", re.I)

def parse_name(stem:str):
    m = NAME_RE.match(stem)
    if not m: return None
    chip = int(m.group(1))
    bg   = int(m.group(2))
    z    = int(m.group(3))
    return chip, bg, z

def split_for_chip(chip:int):
    if chip in (1,2,3,4): return "train"
    if chip == 5:         return "val"
    if chip == 6:         return "test"
    return None

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def binarize_mask(mask):
    # Accept grayscale or 3-channel; return uint8 0/255
    if mask is None: return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # If already binary, leave it; else threshold
    unique = np.unique(mask)
    if np.array_equal(unique, [0]) or np.array_equal(unique, [255]) or np.array_equal(unique, [0,255]):
        return (mask > 127).astype(np.uint8) * 255
    _, th = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return th

def find_mask(mask_dir, chip, bg, z):
    # Try several case patterns
    candidates = []
    base = f"C{chip:02d}_Bg{bg}_Z{z}"
    pats = [
        f"C{chip:02d}_Bg{bg}_Z{z}_mask.*",
        f"C{chip:02d}_Bg{bg}_Z{z}_Mask.*",
        f"C{chip:02d}_BG{bg}_Z{z}_mask.*",
        f"C{chip:02d}_BG{bg}_Z{z}_Mask.*",
    ]
    for pat in pats:
        candidates += glob.glob(os.path.join(mask_dir, pat))
    return candidates[0] if candidates else None

def import_one_chip(src_img_dir, src_mask_dir, out_root):
    copied, issues = 0, []
    for img_path in glob.glob(os.path.join(src_img_dir, "*.*")):
        if os.path.basename(img_path).lower().endswith((".png",".jpg",".jpeg",".tif",".tiff")):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            parsed = parse_name(stem)
            if not parsed:
                issues.append(f"SKIP (bad name): {img_path}")
                continue
            chip, bg, z = parsed
            split = split_for_chip(chip)
            if not split:
                issues.append(f"SKIP (chip out of 1-6): {img_path}")
                continue

            # Dest names
            base = f"C{chip:02d}_Bg{bg}_Z{z}"
            dst_img = os.path.join(out_root, split, "images", f"{base}.png")
            dst_msk = os.path.join(out_root, split, "masks",  f"{base}_mask.png")
            ensure_dir(os.path.dirname(dst_img))
            ensure_dir(os.path.dirname(dst_msk))

            # Copy/convert image to PNG if needed
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                issues.append(f"READ_FAIL image: {img_path}")
                continue
            cv2.imwrite(dst_img, img)

            # Locate mask
            mask_path = find_mask(src_mask_dir, chip, bg, z)
            if not mask_path:
                issues.append(f"MISSING_MASK for {base} (looked in {src_mask_dir})")
                continue

            msk = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if msk is None:
                issues.append(f"READ_FAIL mask: {mask_path}")
                continue
            msk = binarize_mask(msk)

            # Size check; if mismatch, resize mask to image with nearest neighbor
            H, W = img.shape[:2]
            if msk.shape[:2] != (H, W):
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                issues.append(f"RESIZED mask to match image: {base}")

            cv2.imwrite(dst_msk, msk)
            copied += 1
    return copied, issues

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="Path to Training_Stage1 (the folder that contains 'Raw Images' and 'Masks')")
    args = ap.parse_args()

    src = args.source
    raw_root   = os.path.join(src, "Raw Images")
    masks_root = os.path.join(src, "Masks")

    # Source subfolders
    src_map = {
        1: (os.path.join(raw_root, "C01"), os.path.join(masks_root, "C01_Mask")),
        2: (os.path.join(raw_root, "C02"), os.path.join(masks_root, "C02_Mask")),
        3: (os.path.join(raw_root, "C03"), os.path.join(masks_root, "C03_Mask")),
        4: (os.path.join(raw_root, "C04"), os.path.join(masks_root, "C04_Mask")),
        5: (os.path.join(raw_root, "C05"), os.path.join(masks_root, "C05_Mask_Validate")),
        6: (os.path.join(raw_root, "C06"), os.path.join(masks_root, "C06_Mask_Test")),
    }

    out_root = os.path.join("data", "stage1")
    ensure_dir(out_root)

    total, all_issues = 0, []
    for chip in range(1,7):
        img_dir, msk_dir = src_map[chip]
        if not os.path.isdir(img_dir):
            all_issues.append(f"MISSING dir: {img_dir}")
            continue
        if not os.path.isdir(msk_dir):
            all_issues.append(f"MISSING dir: {msk_dir}")
            continue
        copied, issues = import_one_chip(img_dir, msk_dir, out_root)
        total += copied
        all_issues += issues
        print(f"Chip C{chip:02d}: copied {copied} pairs")

    print(f"\nDONE. Copied {total} image/mask pairs.")
    if all_issues:
        print("\nIssues:")
        for s in all_issues:
            print(" -", s)

if __name__ == "__main__":
    main()
