"""quick: print per-chip predicted areas and compare to ground truth."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
m = json.loads((ROOT / "outputs" / "metrics.json").read_text())

print(f"{'C':>4s} {'gt_mm2':>8s} {'pred_mean':>10s} {'pred_std':>10s} {'abs_err':>10s} {'pct_err':>9s}")
for r in m['area_quality_per_chip']['per_chip']:
    pct = 100 * r['abs_error_mm2'] / r['gt_area_mm2'] if r['gt_area_mm2'] else 0
    print(f"  C{r['chip_id']:02d} {r['gt_area_mm2']:>8.2f} {r['pred_area_mm2_mean']:>10.2f} "
          f"{r['pred_area_mm2_std']:>10.2f} {r['abs_error_mm2']:>10.2f} {pct:>8.1f}%")

# also break down by under vs over
preds = [(r['chip_id'], r['gt_area_mm2'], r['pred_area_mm2_mean']) for r in m['area_quality_per_chip']['per_chip']]
overpredicted = [p for p in preds if p[2] > p[1]]
underpredicted = [p for p in preds if p[2] < p[1]]
print()
print(f"overpredicted: {len(overpredicted)} chips")
print(f"underpredicted: {len(underpredicted)} chips")
print(f"mean over-prediction: {sum(p[2]-p[1] for p in overpredicted)/max(1,len(overpredicted)):.2f} mm^2")
print(f"mean under-prediction: {sum(p[1]-p[2] for p in underpredicted)/max(1,len(underpredicted)):.2f} mm^2")
