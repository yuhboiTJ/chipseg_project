import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("float32")  # clean float32 graph

model = tf.keras.models.load_model("models/stage1_best.h5", compile=False)

def safe_convert(converter, outpath):
    try:
        tfl = converter.convert()                 # convert first...
        with open(outpath, "wb") as f:            # ...only then write to disk
            f.write(tfl)
        print(f"Wrote {outpath} ({len(tfl)} bytes)")
    except Exception as e:
        print(f"[WARN] Export failed for {outpath}: {e}")

# 1) float32 (reference)
conv = tf.lite.TFLiteConverter.from_keras_model(model)
safe_convert(conv, "models/stage1_float32.tflite")

# 2) float16 weights (smaller, float I/O)
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.target_spec.supported_types = [tf.float16]
safe_convert(conv, "models/stage1_f16.tflite")
