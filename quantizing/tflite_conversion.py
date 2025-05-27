import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('export_dir/0')
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('model_flex_quant.tflite', 'wb') as f:
    f.write(tflite_model)
