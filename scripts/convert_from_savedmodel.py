import tensorflow as tf
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("float32")

model = tf.keras.models.load_model("models/stage1_best.h5", compile=False)

# Export Keras model to a TensorFlow SavedModel directory
model.export("models/sm_stage1")

# Convert SavedModel -> TFLite (float32)
converter = tf.lite.TFLiteConverter.from_saved_model("models/sm_stage1")
tfl = converter.convert()
open("models/stage1_float32.tflite", "wb").write(tfl)
print("Wrote models/stage1_float32.tflite (size:", len(tfl), "bytes)")
