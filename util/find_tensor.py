import os, tensorflow as tf
from pathlib import Path

current = Path(__file__).resolve().parent
parent = current.parent

print(f'{parent=}')

trained_checkpoint_prefix = parent / "pretrained_models" / "2stems" / "model"
export_dir = parent / "export_dir" / "0"


graph = tf.Graph()

with tf.compat.v1.Session(graph=graph) as sess:
    
    loader = tf.compat.v1.train.import_meta_graph(str(trained_checkpoint_prefix) + '.meta')
    loader.restore(sess, str(trained_checkpoint_prefix))

    print("===== strided_slice テンソル一覧 (形状付き) =====")
    for op in graph.get_operations():
        if 'strided_slice' in op.name:
            try:
                t = graph.get_tensor_by_name(op.name + ":0")
                print(f"{op.name:60s} -> {t.shape.as_list()}")
            except Exception:
                pass
    print("===============================================")