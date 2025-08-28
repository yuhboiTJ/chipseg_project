import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model("models/sm_stage1")
converter.target_spec.supported_types = [tf.float16]
tfl = converter.convert()
open("models/stage1_f16.tflite","wb").write(tfl)
print("Wrote models/stage1_f16.tflite (bytes:", len(tfl), ")")
