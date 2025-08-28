# scripts/quick_infer_stage1.py
import os, json, argparse, numpy as np, cv2

# Try tflite-runtime first (Pi-style); fall back to TF's interpreter on desktop
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

IMG_SIZE = 384

def load_threshold(json_path="models/stage1_eval.json", default=0.5):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return float(data.get("best_threshold", default))
    except Exception:
        return float(default)

def preprocess(img):
    """Letterbox to IMG_SIZE with black padding (matches training resize_with_pad)."""
    H, W = img.shape[:2]
    s = min(IMG_SIZE / H, IMG_SIZE / W)
    nh, nw = int(H * s), int(W * s)
    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh), cv2.INTER_AREA)
    x = canvas.astype(np.float32) / 255.0
    return x, (H, W), (nh, nw)

def restore(mask_u8, H, W, nh, nw):
    """Undo letterbox back to original HxW."""
    return cv2.resize(mask_u8[:nh, :nw], (W, H), cv2.INTER_NEAREST)

def run(model_path, image_path, outdir="outputs", thresh=None):
    os.makedirs(outdir, exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {image_path}")

    x, (H, W), (nh, nw) = preprocess(img)

    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    # Match input dtype (uint8 for quant; float32 otherwise)
    xin = (x * 255).astype(np.uint8) if str(inp["dtype"]).endswith("uint8") else x.astype(np.float32)
    interp.set_tensor(inp["index"], xin[None, ...])
    interp.invoke()

    y = interp.get_tensor(out["index"])[0]
    y = (y.astype(np.float32) / 255.0) if str(y.dtype).endswith("uint8") else y.astype(np.float32)
    if y.ndim == 3:
        y = y[..., 0]

    if thresh is None:
        thresh = load_threshold()

    mask = (y > float(thresh)).astype(np.uint8) * 255
    mask_full = restore(mask, H, W, nh, nw)

    # Save outputs
    out_mask = os.path.join(outdir, "out_mask.png")
    out_vis  = os.path.join(outdir, "out_vis.png")
    cv2.imwrite(out_mask, mask_full)

    overlay = img.copy()
    overlay[mask_full > 0] = (0.3 * overlay[mask_full > 0] + 0.7 * np.array([0, 255, 0])).astype(np.uint8)
    vis = cv2.addWeighted(img, 0.7, overlay, 0.3, 0.0)
    cv2.imwrite(out_vis, vis)

    print(f"Wrote:\n  {out_mask}\n  {out_vis}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to a PNG/JPG to segment")
    ap.add_argument("--model", default="models/stage1_float32.tflite", help="Path to .tflite model")
    ap.add_argument("--thresh", type=float, default=None, help="Override threshold (else uses stage1_eval.json)")
    ap.add_argument("--outdir", default="outputs", help="Folder to save PNGs")
    args = ap.parse_args()
    run(args.model, args.image, args.outdir, args.thresh)
