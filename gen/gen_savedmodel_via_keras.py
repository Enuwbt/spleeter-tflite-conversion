import os
import tensorflow as tf

trained_ckpt = r'../pretrained_models/2stems/model'
export_dir = os.path.join('../export_dir', '1')

graph = tf.Graph()
sess = tf.compat.v1.Session(graph=graph)
with graph.as_default():
    loader = tf.compat.v1.train.import_meta_graph(trained_ckpt + '.meta')
    loader.restore(sess, trained_ckpt)

    input_tensor = graph.get_tensor_by_name('waveform:0')
    vocals_tensor = graph.get_tensor_by_name('strided_slice_13:0')
    accomp_tensor = graph.get_tensor_by_name('strided_slice_23:0')

def separate_fn(waveform_np):
    vocals_np, accomp_np = sess.run(
        [vocals_tensor, accomp_tensor],
        feed_dict={input_tensor: waveform_np}
    )
    return vocals_np, accomp_np

waveform_input = tf.keras.Input(
    shape=(None, 2),
    dtype=tf.float32,
    name='waveform'
)

vocals_out, accomp_out = tf.keras.layers.Lambda(
    lambda x: tf.py_function(
        func=separate_fn,
        inp=[x],
        Tout=[tf.float32, tf.float32]
    ),
    name='separate_layer'
)(waveform_input)

model = tf.keras.Model(
    inputs=waveform_input,
    outputs={
        'vocals': vocals_out,
        'accompaniment': accomp_out
    },
    name='separator_model'
)

model.save(
    export_dir,
    overwrite=True,
    save_format='tf',
    include_optimizer=False
)

print(f"SavedModel を '{export_dir}' にエクスポートしました。")
