"""
Read outputs/metrics.json (produced by evaluate.py) and substitute the values
into REPORT.txt where the placeholders __KEY__ live.

Run from code/:
    python scripts/finalize_report.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # code/
PROJECT_ROOT = ROOT.parent  # microchip segmentation project

REPORT_PATH = PROJECT_ROOT / "REPORT.txt"
METRICS_PATH = ROOT / "outputs" / "metrics.json"


def fmt_num(x, fmt="{:.4f}"):
    if x is None:
        return "n/a"
    return fmt.format(x)


def main():
    if not METRICS_PATH.exists():
        print(f"missing {METRICS_PATH}, run evaluate.py first", file=sys.stderr)
        sys.exit(1)
    if not REPORT_PATH.exists():
        print(f"missing {REPORT_PATH}", file=sys.stderr)
        sys.exit(1)

    metrics = json.loads(METRICS_PATH.read_text())

    cal = metrics.get("calibration", {})
    mq = metrics.get("mask_quality_val", {})
    aq = metrics.get("area_quality_per_chip", {})
    tr = metrics.get("training", {})

    subs = {
        "__VAL_DICE__":   fmt_num(mq.get("dice_mean")),
        "__VAL_IOU__":    fmt_num(mq.get("iou_mean")),
        "__AREA_MAE__":   fmt_num(aq.get("mae_mm2"), "{:.3f}"),
        "__AREA_RMSE__":  fmt_num(aq.get("rmse_mm2"), "{:.3f}"),
        "__AREA_R2__":    fmt_num(aq.get("r2")),
        "__N_CHIPS__":    str(aq.get("n_chips", "n/a")),
        "__CALIB_MEAN__":   fmt_num(cal.get("mm2_per_pixel_mean"), "{:.6e}"),
        "__CALIB_MEDIAN__": fmt_num(cal.get("mm2_per_pixel_median"), "{:.6e}"),
        "__CALIB_STD__":    fmt_num(cal.get("mm2_per_pixel_std"), "{:.6e}"),
        "__CALIB_COV__":   fmt_num(
            (100 * cal.get("mm2_per_pixel_std") / cal.get("mm2_per_pixel_mean"))
            if cal.get("mm2_per_pixel_mean") else None,
            "{:.2f}"
        ),
        "__EPOCHS__":      str(tr.get("epochs", "n/a")),
    }

    text = REPORT_PATH.read_text(encoding="utf-8")
    n_replaced = 0
    for k, v in subs.items():
        if k in text:
            text = text.replace(k, v)
            n_replaced += 1

    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"updated {REPORT_PATH}, replaced {n_replaced} placeholders")
    for k, v in subs.items():
        print(f"  {k} -> {v}")


if __name__ == "__main__":
    main()
