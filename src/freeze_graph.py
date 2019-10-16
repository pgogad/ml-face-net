import os
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model

import keras

BASE_PATH = os.path.dirname(__file__)
MODEL = os.path.join(BASE_PATH, 'data', 'model', 'facenet_keras.h5')

keras.backend.clear_session()


def freeze_graph(graph, session, output, save_pb_dir=os.path.join(BASE_PATH, 'data', 'model'),
                 save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.compat.v1.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.compat.v1.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen


keras.backend.set_learning_phase(0)

model = load_model(MODEL)
session = keras.backend.get_session()

input_names = [t.op.name for t in model.inputs]
output_names = [t.op.name for t in model.outputs]

# Prints input and output nodes names, take notes of them.
print(input_names, output_names)
frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs])
