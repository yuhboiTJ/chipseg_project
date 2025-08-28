# scripts/train_stage1_edges.py
# Stage-1 (chip vs background) with edge/contrast channels
# Stable training: BCE + Dice (no boundary loss)

import os, glob, json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

# -----------------------------
# Config / paths
# -----------------------------
IMG_SIZE = 384
BATCH    = 8
EPOCHS   = 50
LR       = 1e-4

TRAIN_IMG_DIR = "data/stage1/train/images"
TRAIN_MSK_DIR = "data/stage1/train/masks"
VAL_IMG_DIR   = "data/stage1/val/images"
VAL_MSK_DIR   = "data/stage1/val/masks"

OUT_DIR   = "models"
os.makedirs(OUT_DIR, exist_ok=True)
BEST_PATH = os.path.join(OUT_DIR, "stage1_edges_best.h5")

# -----------------------------
# Pairing utilities
# -----------------------------
def mask_path_for(img_path, masks_dir):
    name = os.path.splitext(os.path.basename(img_path))[0]
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    cands = [os.path.join(masks_dir, f"{name}_mask{e}") for e in exts] + \
            [os.path.join(masks_dir, f"{name}{e}")      for e in exts]
    for p in cands:
        if tf.io.gfile.exists(p):
            return p
    return None

def list_pairs(images_dir, masks_dir):
    imgs = []
    for ext in ["*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"]:
        imgs += glob.glob(os.path.join(images_dir, ext))
    imgs = sorted(imgs)
    pairs = []
    for ip in imgs:
        mp = mask_path_for(ip, masks_dir)
        if mp and tf.io.gfile.exists(mp):
            pairs.append((ip, mp))
    return pairs

train_pairs = list_pairs(TRAIN_IMG_DIR, TRAIN_MSK_DIR)
val_pairs   = list_pairs(VAL_IMG_DIR,   VAL_MSK_DIR)
print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")
assert train_pairs and val_pairs, "No image/mask pairs found."

# -----------------------------
# I/O helpers
# -----------------------------
def read_png_rgb(path):
    b = tf.io.read_file(path)
    return tf.image.decode_png(b, channels=3)      # [H,W,3] uint8

def read_png_mask1(path):
    b = tf.io.read_file(path)
    return tf.image.decode_png(b, channels=1)      # [H,W,1] uint8

def resize_pad(x):
    return tf.image.resize_with_pad(x, IMG_SIZE, IMG_SIZE)

def binarize_mask(mask):
    mask = tf.cast(mask, tf.float32)
    # your masks are 0/255 → threshold cleanly
    mask = tf.where(mask > 127.5, 1.0, 0.0)
    return mask

# -----------------------------
# Rank-agnostic edge channels (TF-only)
# -----------------------------
def _to_4d(x):
    x = tf.convert_to_tensor(x)
    return (tf.expand_dims(x, 0), True) if x.shape.rank == 3 else (x, False)

def _ensure_rgb01(x):
    """Accept 3D/4D; return float32 in [0,1], RGB (tiles grayscale via repeat)."""
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
    """Normalized Sobel magnitude in [0,1]; accepts [H,W,C] or [B,H,W,C]."""
    x4, squeeze = _to_4d(img)
    gray = _gray01_from_any(x4)                 # [B,H,W,1]
    se = tf.image.sobel_edges(gray)             # [B,H,W,1,2]
    gx, gy = se[..., 0], se[..., 1]
    mag = tf.sqrt(gx * gx + gy * gy)
    mag = mag / (tf.reduce_max(mag, axis=[1,2,3], keepdims=True) + 1e-6)
    return tf.squeeze(mag, 0) if squeeze else mag

def local_contrast_tf(img, k=5):
    """Local mean abs-dev in [0,1]; accepts [H,W,C] or [B,H,W,C]."""
    x4, squeeze = _to_4d(img)
    gray = _gray01_from_any(x4)
    kernel = tf.ones((k, k, 1, 1), tf.float32) / float(k * k)
    mean = tf.nn.depthwise_conv2d(gray, kernel, strides=[1,1,1,1], padding='SAME')
    dev  = tf.abs(gray - mean)
    dev  = dev / (tf.reduce_max(dev, axis=[1,2,3], keepdims=True) + 1e-6)
    return tf.squeeze(dev, 0) if squeeze else dev

def add_edge_channels_tf(img_rgb01):
    """Input [H,W,3] float in [0,1] → Output [H,W,5]."""
    e  = sobel_mag_tf(img_rgb01)
    lc = local_contrast_tf(img_rgb01)
    return tf.concat([img_rgb01, e, lc], axis=-1)

# -----------------------------
# Augmentations
# -----------------------------
def augment(img01, mask):
    if tf.random.uniform(()) < 0.5:
        img01 = tf.image.flip_left_right(img01); mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) < 0.5:
        img01 = tf.image.flip_up_down(img01);     mask = tf.image.flip_up_down(mask)
    k = tf.random.uniform((), maxval=4, dtype=tf.int32)
    img01 = tf.image.rot90(img01, k); mask = tf.image.rot90(mask, k)
    img01 = tf.image.random_brightness(img01, max_delta=0.08)
    img01 = tf.image.random_contrast(img01, 0.9, 1.1)
    img01 = tf.clip_by_value(img01, 0.0, 1.0)
    return img01, mask

# -----------------------------
# Datasets (only images go through add_edge_channels_tf)
# -----------------------------
def _parse_train(ip, mp):
    img = read_png_rgb(ip)
    msk = read_png_mask1(mp)
    img = resize_pad(img); msk = resize_pad(msk)
    img = tf.cast(img, tf.float32) / 255.0
    msk = binarize_mask(msk)
    img, msk = augment(img, msk)
    img = add_edge_channels_tf(img)         # -> [H,W,5]
    return img, msk

def _parse_val(ip, mp):
    img = read_png_rgb(ip)
    msk = read_png_mask1(mp)
    img = resize_pad(img); msk = resize_pad(msk)
    img = tf.cast(img, tf.float32) / 255.0
    msk = binarize_mask(msk)
    img = add_edge_channels_tf(img)
    return img, msk

def make_ds(pairs, parser, shuffle=False):
    ips = [p[0] for p in pairs]; mps = [p[1] for p in pairs]
    ds = tf.data.Dataset.from_tensor_slices((ips, mps))
    if shuffle:
        ds = ds.shuffle(len(ips), reshuffle_each_iteration=True)
    ds = ds.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

ds_train = make_ds(train_pairs, _parse_train, shuffle=True)
ds_val   = make_ds(val_pairs,   _parse_val,   shuffle=False)

# -----------------------------
# Model (compact U-Net, 5-ch input)
# -----------------------------
def conv_blk(x, f):
    x = L.Conv2D(f, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    x = L.Conv2D(f, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    return x

def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 5)):
    inputs = L.Input(shape=input_shape)
    c1 = conv_blk(inputs, 32);  p1 = L.MaxPool2D()(c1)
    c2 = conv_blk(p1, 64);      p2 = L.MaxPool2D()(c2)
    c3 = conv_blk(p2, 128);     p3 = L.MaxPool2D()(c3)
    c4 = conv_blk(p3, 256);     p4 = L.MaxPool2D()(c4)
    bn = conv_blk(p4, 384)
    u4 = L.UpSampling2D()(bn);  u4 = L.Concatenate()([u4, c4]); u4 = conv_blk(u4, 256)
    u3 = L.UpSampling2D()(u4);  u3 = L.Concatenate()([u3, c3]); u3 = conv_blk(u3, 128)
    u2 = L.UpSampling2D()(u3);  u2 = L.Concatenate()([u2, c2]); u2 = conv_blk(u2, 64)
    u1 = L.UpSampling2D()(u2);  u1 = L.Concatenate()([u1, c1]); u1 = conv_blk(u1, 32)
    out = L.Conv2D(1, 1, activation="sigmoid")(u1)
    return keras.Model(inputs, out, name="unet_edges")

model = build_unet()
model.summary()

# -----------------------------
# Losses & metrics (stable)
# -----------------------------
bce = keras.losses.BinaryCrossentropy()

def dice_coef(y_true, y_pred, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    inter = tf.reduce_sum(y_true*y_pred, axis=[1,2,3])
    denom = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])
    return tf.reduce_mean((2.*inter + eps)/(denom + eps))

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

@tf.function
def iou_metric(y_true, y_pred, eps=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true*y_pred, axis=[1,2,3])
    union = tf.reduce_sum(tf.clip_by_value(y_true + y_pred, 0, 1), axis=[1,2,3])
    return tf.reduce_mean((inter + eps)/(union + eps))

def total_loss(y_true, y_pred):
    # stable combo
    return 0.7*bce(y_true, y_pred) + 0.3*dice_loss(y_true, y_pred)

model.compile(optimizer=keras.optimizers.Adam(LR),
              loss=total_loss,
              metrics=[dice_coef, iou_metric])

# -----------------------------
# Train
# -----------------------------
cbs = [
    keras.callbacks.ModelCheckpoint(BEST_PATH, monitor="val_dice_coef",
                                    mode="max", save_best_only=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
]

hist = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=cbs)

# Save last (best already saved by checkpoint)
model.save(os.path.join(OUT_DIR, "stage1_edges_last.h5"))

best_dice = float(max(hist.history.get("val_dice_coef", [0.0])))
with open(os.path.join(OUT_DIR, "stage1_edges_eval.json"), "w") as f:
    json.dump({"best_val_dice": best_dice, "note": "stage1 edge-aware (BCE+Dice)"}, f, indent=2)

print("\nDone. Best model:", BEST_PATH)
