from logging import getLogger
import tensorflow as tf
from tensorflow.python.framework import graph_io
from src.entity.KerasModel import KerasModel
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

if __name__ == '__main__':
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)
    s = tf.compat.v1.keras.backend.get_session()

    keras = KerasModel()
    # keras.train_model()
    model = keras.model

    for file in os.listdir('/app/data/damaged/'):
        f = keras.prepare_file('/app/data/damaged/' + file)
        f = np.float32(f)
        res = model.predict(f)
        print(res)
        predict = 1 if res[0][0] >= 0.5 else 0
        print(predict)
    for file in os.listdir('/app/data/undamaged/'):
        f = keras.prepare_file('app/data/undamaged' + file)
        res = model.predict(f)
        predict = 1 if res[0][0] >= 0.5 else 0
        print(predict)
    graph = keras.freeze_session(session=s, output_names=[out.op.name for out in model.outputs])
    graph_io.write_graph(graph, "model", "model.pb", as_text=True)
