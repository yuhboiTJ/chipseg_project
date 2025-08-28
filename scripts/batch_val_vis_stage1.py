# scripts/batch_val_vis_stage1.py
# Batch visualize val set: green overlay + raw predicted masks

import os, glob, json
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE   = 384
MODEL_PATH = "models/stage1_edges_best.h5"
EVAL_PATH  = "models/stage1_edges_eval.json"
IN_GLOB    = "data/stage1/val/images/*.png"
OUT_DIR    = "outputs/c05_val_vis"
MASK_DIR   = "outputs/c05_val_masks"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

# Threshold default / optional override from eval.json
thr = 0.5
if os.path.exists(EVAL_PATH):
    try:
        with open(EVAL_PATH, "r") as f:
            j = json.load(f)
            thr = float(j.get("best_threshold", thr))
    except Exception:
        pass

# -----------------------------
# Edge-channel functions (with grayscale→RGB via repeat)
# -----------------------------
def _to_4d(x):
    x = tf.convert_to_tensor(x)
    return (tf.expand_dims(x, 0), True) if x.shape.rank == 3 else (x, False)

def _ensure_rgb01(x):
    x = tf.cast(x, tf.float32)
    x = tf.where(tf.reduce_max(x) > 1.0, x / 255.0, x)
    if x.shape.rank is not None:
        if x.shape[-1] == 1:
            x = tf.repeat(x, 3, axis=-1)
    else:
        ch = tf.shape(x)[-1]
        x = tf.cond(tf.equal(ch, 1), lambda: tf.repeat(x, 3, axis=-1), lambda: x)
    return x

def _gray01_from_any(x):
    return tf.image.rgb_to_grayscale(_ensure_rgb01(x))

def sobel_mag_tf(img):
    x4, squeeze = _to_4d(img)
    gray = _gray01_from_any(x4)
    se = tf.image.sobel_edges(gray)
    gx, gy = se[..., 0], se[..., 1]
    mag = tf.sqrt(gx * gx + gy * gy)
    mag = mag / (tf.reduce_max(mag, axis=[1,2,3], keepdims=True) + 1e-6)
    return tf.squeeze(mag, 0) if squeeze else mag

def local_contrast_tf(img, k=5):
    x4, squeeze = _to_4d(img)
    gray = _gray01_from_any(x4)
    kernel = tf.ones((k, k, 1, 1), tf.float32) / float(k * k)
    mean = tf.nn.depthwise_conv2d(gray, kernel, strides=[1,1,1,1], padding='SAME')
    dev  = tf.abs(gray - mean)
    dev  = dev / (tf.reduce_max(dev, axis=[1,2,3], keepdims=True) + 1e-6)
    return tf.squeeze(dev, 0) if squeeze else dev

def add_edge_channels_tf(img_rgb01):
    e  = sobel_mag_tf(img_rgb01)
    lc = local_contrast_tf(img_rgb01)
    return tf.concat([img_rgb01, e, lc], axis=-1)

# -----------------------------
# Model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# -----------------------------
# Inference util: load → resize → add channels
# -----------------------------
def load_for_model(path):
    b = tf.io.read_file(path)
    img = tf.image.decode_png(b, channels=3)
    img = tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    x5  = add_edge_channels_tf(img)
    return x5[None, ...]   # [1,H,W,5]

# -----------------------------
# Run
# -----------------------------
for p in sorted(glob.glob(IN_GLOB)):
    name = os.path.splitext(os.path.basename(p))[0]
    x = load_for_model(p)
    pred = model.predict(x, verbose=0)[0, ..., 0]     # [H,W] float
    m = (pred > thr).astype(np.uint8) * 255

    # save raw mask
    Image.fromarray(m).save(os.path.join(MASK_DIR, f"{name}_pred.png"))

    # green overlay on resized original
    src = (x[0][..., :3].numpy() * 255).astype(np.uint8)  # RGB
    overlay = src.copy()
    overlay[..., 1] = np.maximum(overlay[..., 1], m)      # boost G
    overlay[..., 0] = (overlay[..., 0] * (1 - (m > 0) * 0.3)).astype(np.uint8)

    Image.fromarray(overlay).save(os.path.join(OUT_DIR, f"{name}_vis.png"))

print(f"Done. Wrote overlays -> {OUT_DIR} and masks -> {MASK_DIR}")
