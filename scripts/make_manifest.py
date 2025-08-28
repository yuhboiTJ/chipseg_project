import re, csv, pathlib as P
ROOT = P.Path("data/stage1")
pat = re.compile(r"C(?P<chip>\d+)_Bg(?P<bg>\d+)_Z(?P<zoom>[^.]+)\.png$", re.I)

def gather(split):
    img_dir, mask_dir = ROOT/split/"images", ROOT/split/"masks"
    for p in sorted(img_dir.glob("*.png")):
        m = pat.search(p.name)
        chip = int(m["chip"]) if m else ""
        bg   = int(m["bg"]) if m else ""
        zoom = m["zoom"] if m else ""
        yield dict(split=split, image=str(p),
                   mask=str(mask_dir/p.name.replace(".png","_mask.png")),
                   chip=chip, bg=bg, zoom=zoom)

rows = [*gather("train"), *gather("val"), *gather("test")]
if not rows:
    raise SystemExit("No PNGs yet. Put images in data/stage1/*/images/ first.")
out = ROOT/"manifest.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"Wrote {out} ({len(rows)} rows)")
