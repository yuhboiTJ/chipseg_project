import csv, numpy as np, cv2, pathlib as P
from collections import defaultdict

mf = P.Path("data/stage1/manifest.csv")
if not mf.exists(): raise SystemExit("Run scripts/make_manifest.py first.")
rows = list(csv.DictReader(mf.open()))
problems, by_chip, by_bg = [], defaultdict(list), defaultdict(list)

for r in rows:
    img = cv2.imread(r["image"], cv2.IMREAD_COLOR)
    msk = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
    if img is None: problems.append(("missing_img", r["image"])); continue
    if msk is None: problems.append(("missing_mask", r["mask"])); continue
    if img.shape[:2] != msk.shape[:2]: problems.append(("shape_mismatch", r["image"])); continue
    if set(np.unique(msk).tolist()) - {0,255}: problems.append(("non_binary_mask", r["mask"]))
    frac = (msk>0).sum()/msk.size
    if r["chip"]!="": by_chip[r["chip"]].append(frac)
    if r["bg"]  !="": by_bg[r["bg"]].append(frac)

print("QC problems:", len(problems))
for t,p in problems[:30]: print(" -", t, ":", p)
print("\nChip coverage (mean frac):")
for k in sorted(by_chip): print(f"  Chip {k}: {np.mean(by_chip[k]):.3f} (n={len(by_chip[k])})")
print("Background balance (mean frac):")
for k in sorted(by_bg): print(f"  Bg {k}: {np.mean(by_bg[k]):.3f} (n={len(by_bg[k])})")
