# scripts/debug_masks.py
import glob, os, cv2, numpy as np

MASK_GLOB = r"data/stage1/train/masks\*.*"   # png/jpg ok

paths = sorted(glob.glob(MASK_GLOB))
print(f"Found {len(paths)} mask files")

empty = []
mostly_bg = []
mostly_fg = []
examples = []

for p in paths:
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m is None:
        print("Unreadable:", p)
        continue
    if m.ndim == 3:
        # if RGB/RGBA, take one channel
        m = m[...,0]
    # values expected 0..255
    pos = (m > 127).sum()
    total = m.size
    frac = pos / float(total)
    if pos == 0:
        empty.append(p)
    elif frac < 0.01:
        mostly_bg.append((p, frac))
    elif frac > 0.99:
        mostly_fg.append((p, frac))
    if len(examples) < 5:
        examples.append((p, np.unique(m)))

print("\nExamples (file, unique pixel values):")
for p, uniq in examples:
    print(" ", os.path.basename(p), uniq)

print(f"\nEmpty masks (0 pixels >127): {len(empty)}")
if empty[:10]:
    print("  sample:", empty[:10])

print(f"Mostly background (<1% foreground): {len(mostly_bg)}")
if mostly_bg[:5]:
    print("  sample:", mostly_bg[:5])

print(f"Mostly foreground (>99% foreground): {len(mostly_fg)}")
if mostly_fg[:5]:
    print("  sample:", mostly_fg[:5])
