"""
Compare scratch and pretrained models. Reads metrics.json from both runs and
writes:
  - outputs/figures/area_scatter_comparison.png  (side-by-side scatter)
  - outputs/model_comparison.json                (table of headline numbers)
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

SCRATCH    = ROOT / "outputs"
PRETRAINED = ROOT / "outputs_pretrained"


def load(out_dir):
    m = json.loads((out_dir / "metrics.json").read_text())
    s = json.loads((out_dir / "training_summary.json").read_text())
    return m, s


def main():
    ms, ss = load(SCRATCH)
    mp, sp = load(PRETRAINED)

    # build the comparison row
    def pull(m, s, name):
        return {
            "name": name,
            "model_type":     s.get("model_type"),
            "encoder":        s.get("encoder_name"),
            "param_count":    s.get("param_count"),
            "epochs":         s.get("epochs"),
            "best_val_dice":  s.get("best_val_dice"),
            "val_dice_eval":  m["mask_quality_val"]["dice_mean"],
            "val_iou_eval":   m["mask_quality_val"]["iou_mean"],
            "area_mae":       m["area_quality_per_chip"]["mae_mm2"],
            "area_rmse":      m["area_quality_per_chip"]["rmse_mm2"],
            "area_r2":        m["area_quality_per_chip"]["r2"],
            "n_chips":        m["area_quality_per_chip"]["n_chips"],
        }

    rows = [pull(ms, ss, "scratch"), pull(mp, sp, "pretrained")]

    print(f"{'metric':<22s} {'scratch':>14s} {'pretrained':>14s}")
    print("-" * 55)
    for k in ["model_type", "encoder", "param_count", "epochs", "best_val_dice",
              "val_dice_eval", "val_iou_eval", "area_mae", "area_rmse", "area_r2"]:
        a = rows[0].get(k)
        b = rows[1].get(k)
        def fmt(v):
            if v is None:
                return "n/a"
            if isinstance(v, float):
                return f"{v:.4f}"
            if isinstance(v, int) and k == "param_count":
                return f"{v:,}"
            return str(v)
        print(f"{k:<22s} {fmt(a):>14s} {fmt(b):>14s}")

    # write the comparison json
    payload = {
        "scratch":    rows[0],
        "pretrained": rows[1],
    }
    (ROOT / "outputs" / "model_comparison.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {ROOT / 'outputs' / 'model_comparison.json'}")

    # side-by-side scatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, m, label in [(axes[0], ms, "scratch"), (axes[1], mp, "pretrained")]:
        rows_p = m["area_quality_per_chip"]["per_chip"]
        gts = np.array([r["gt_area_mm2"] for r in rows_p])
        preds = np.array([r["pred_area_mm2_mean"] for r in rows_p])
        lo = float(min(gts.min(), preds.min()) * 0.9)
        hi = float(max(gts.max(), preds.max()) * 1.1)
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x")
        ax.scatter(gts, preds, s=40, alpha=0.8, edgecolor="black", linewidth=0.6)
        for r in rows_p:
            ax.annotate(f"C{r['chip_id']:02d}",
                        (r["gt_area_mm2"], r["pred_area_mm2_mean"]),
                        fontsize=7, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("ground truth area (mm^2)")
        if label == "scratch":
            ax.set_ylabel("predicted seedable area (mm^2)")
        ax.set_title(
            f"{label}\nMAE={m['area_quality_per_chip']['mae_mm2']:.2f}  "
            f"R^2={m['area_quality_per_chip']['r2']:.3f}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    fig.suptitle("Predicted vs ground truth area: scratch (1.94M params) vs pretrained ResNet34 (24.4M)")
    fig.tight_layout()
    out_fig = ROOT / "outputs" / "figures" / "area_scatter_comparison.png"
    fig.savefig(out_fig, dpi=130)
    plt.close(fig)
    print(f"wrote {out_fig}")


if __name__ == "__main__":
    main()
