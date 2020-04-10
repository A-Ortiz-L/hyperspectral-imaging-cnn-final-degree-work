from logging import getLogger
import tensorflow as tf
from tensorflow.python.framework import graph_io
from src.entity.keras_model import KerasModel
import logging
from datetime import datetime
from tensorflow.python.util import deprecation
import os
import numpy as np

logging.basicConfig(
    filename=f'/app/log/events_{datetime.now()}.log',
    filemode='a',
    level=10,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
log = getLogger(__name__)
TRAIN = False
if __name__ == '__main__':
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)
    s = tf.compat.v1.keras.backend.get_session()

    keras = KerasModel()
    keras.load_model()
    if TRAIN:
        keras.train_model()
    model = keras.model
    tf.keras.models.save_model(
        model,
        filepath='/app/data/model/',
        save_format='tf'
    )

    graph = keras.save_model_tf(session=s, output_names=[out.op.name for out in model.outputs])
    graph_io.write_graph(graph, "model", "model.pb", as_text=True)
