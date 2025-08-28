import os, glob, random, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, callbacks
from tensorflow.keras.metrics import BinaryIoU

SEED=13; random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
IMG_SIZE=384

# Precision policy: use mixed_float16 only if a GPU exists
from tensorflow.keras import mixed_precision
try:
    if tf.config.list_physical_devices("GPU"):
        mixed_precision.set_global_policy("mixed_float16")
    else:
        mixed_precision.set_global_policy("float32")
except Exception:
    pass

def list_pairs(img_dir, mask_dir):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    out=[]
    for p in imgs:
        m = os.path.join(mask_dir, os.path.basename(p).replace(".png","_mask.png"))
        if os.path.exists(m): out.append((p,m))
    return out

def read_png(path, ch):
    b = tf.io.read_file(path)
    return tf.image.decode_png(b, channels=ch)

def preprocess_image(img):
    img = tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE, method='bilinear')
    return tf.cast(img, tf.float32)/255.0

def preprocess_mask(mask):
    mask = tf.image.resize_with_pad(mask, IMG_SIZE, IMG_SIZE, method='nearest')
    mask = tf.cast(mask>127, tf.float32)[..., :1]
    return mask

def aug_pair(img, mask):
    k = tf.random.uniform([],0,4,dtype=tf.int32)
    img = tf.image.rot90(img,k); mask = tf.image.rot90(mask,k)
    if tf.random.uniform([])>0.5:
        img=tf.image.flip_left_right(img); mask=tf.image.flip_left_right(mask)
    if tf.random.uniform([])>0.5:
        img=tf.image.flip_up_down(img); mask=tf.image.flip_up_down(mask)
    img=tf.image.random_brightness(img,0.08)
    img=tf.image.random_contrast(img,0.9,1.1)
    return tf.clip_by_value(img,0,1), mask

def make_ds(pairs, training=True, batch=8):
    if not pairs: return tf.data.Dataset.from_tensors((tf.zeros([IMG_SIZE,IMG_SIZE,3]), tf.zeros([IMG_SIZE,IMG_SIZE,1]))).batch(1)
    img_paths, msk_paths = zip(*pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(img_paths), list(msk_paths)))
    def _load(i,m):
        return preprocess_image(read_png(i,3)), preprocess_mask(read_png(m,1))
    if training: ds = ds.shuffle(len(pairs), seed=SEED)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training: ds = ds.map(aug_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

def conv_block(x,f):
    x=layers.Conv2D(f,3,padding='same',activation='relu')(x); x=layers.BatchNormalization()(x)
    x=layers.Conv2D(f,3,padding='same',activation='relu')(x); x=layers.BatchNormalization()(x)
    return x

def unet(input_shape=(IMG_SIZE,IMG_SIZE,3), base=32):
    i=layers.Input(input_shape)
    c1=conv_block(i,base);    p1=layers.MaxPool2D()(c1)
    c2=conv_block(p1,base*2); p2=layers.MaxPool2D()(c2)
    c3=conv_block(p2,base*4); p3=layers.MaxPool2D()(c3)
    c4=conv_block(p3,base*8); p4=layers.MaxPool2D()(c4)
    bn=conv_block(p4,base*16)
    u4=layers.UpSampling2D()(bn); u4=layers.Concatenate()([u4,c4]); u4=conv_block(u4,base*8)
    u3=layers.UpSampling2D()(u4);  u3=layers.Concatenate()([u3,c3]); u3=conv_block(u3,base*4)
    u2=layers.UpSampling2D()(u3);  u2=layers.Concatenate()([u2,c2]); u2=conv_block(u2,base*2)
    u1=layers.UpSampling2D()(u2);  u1=layers.Concatenate()([u1,c1]); u1=conv_block(u1,base)
    o=layers.Conv2D(1,1,activation='sigmoid', dtype='float32')(u1)
    return models.Model(i,o)

def dice_coef(y_true,y_pred,eps=1e-6):
    y_true=tf.reshape(tf.cast(y_true,tf.float32),[-1])
    y_pred=tf.reshape(tf.cast(y_pred,tf.float32),[-1])
    inter=tf.reduce_sum(y_true*y_pred)
    return (2.*inter+eps)/(tf.reduce_sum(y_true)+tf.reduce_sum(y_pred)+eps)

def dice_loss(y_true,y_pred): return 1.0-dice_coef(y_true,y_pred)

def bce_dice(y_true,y_pred):
    y_true=tf.cast(y_true,tf.float32); y_pred=tf.cast(y_pred,tf.float32)
    return 0.5*losses.binary_crossentropy(y_true,y_pred)+0.5*dice_loss(y_true,y_pred)

if __name__=="__main__":
    tr = list_pairs("data/stage1/train/images","data/stage1/train/masks")
    va = list_pairs("data/stage1/val/images","data/stage1/val/masks")
    print(f"Train {len(tr)} | Val {len(va)}")
    train_ds = make_ds(tr, True,  batch=8)
    val_ds   = make_ds(va, False, batch=8)

    model = unet()
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss=bce_dice,
                  metrics=[dice_coef, BinaryIoU(threshold=0.5)])
    cbs=[
        callbacks.ModelCheckpoint("models/stage1_best.h5", monitor="val_dice_coef",
                                  save_best_only=True, mode="max", verbose=1),
        callbacks.EarlyStopping(monitor="val_dice_coef", mode="max", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_dice_coef", mode="max", factor=0.5, patience=4, verbose=1)
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=cbs)
    model.save("models/stage1_final.h5")
