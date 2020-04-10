import tensorflow as tf
import numpy as np
import os
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import pickle
from config.cfg import data_dir, pickle_dir
import time
from logging import getLogger
from sklearn.model_selection import train_test_split

log = getLogger(__name__)


class KerasModel:
    def __init__(self):
        self.sess = tf.compat.v1.Session()
        self.categories = ["damaged", "undamaged"]
        self.training_data = []
        self.model = Sequential()

    @staticmethod
    def save_pickle(pickle_route: str, x: np.array):
        f = open(pickle_route, "wb")
        pickle.dump(x, f)
        f.close()

    def relu_layer(self):
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(3, 3)))
        self.model.add(Dropout(0.50))

    def train_model(self):
        self.create_training_data()
        random.shuffle(self.training_data)
        x = []
        y = []
        for features, label in self.training_data:
            x.append(features)
            y.append(label)
        x = np.array(x).reshape(-1, 128, 128, 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)
        self.save_pickle(f'{pickle_dir}x.pickle', x_train)
        self.save_pickle(f'{pickle_dir}y.pickle', y_train)
        x = np.array(x_train).astype(np.float32) / 255.0
        y = np.array(y_train).astype(np.float32)

        start = time.time()
        self.model.add(Conv2D(64, (3, 3), input_shape=x.shape[1:]))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(3, 3)))
        self.model.add(Dropout(0.50))
        self.relu_layer()
        self.relu_layer()
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation("relu"))
        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Dense(1, activation="sigmoid"))
        optimizer = Adam(learning_rate=0.0008)
        self.model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        self.model.fit(x, y, batch_size=32, epochs=200, validation_split=0.2, shuffle=True)
        end = time.time()
        log.info(f'Training complete in {start - end}')
        _, accuracy = self.model.evaluate(x, y)
        log.info('Accuracy: %.2f' % (accuracy * 100))
        self.save_model_h5(self.model)

    def process_image(self, file_route):
        start = time.time()
        img_array = cv2.imread(file_route, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (128, 128))
        img = new_array.reshape(-1, 128, 128, 1) / 255.0
        res = self.model.predict(np.float32(img))
        predict = True if res[0][0] >= 0.5 else False
        return predict, (time.time() - start)

    def create_training_data(self):
        for category in self.categories:
            path = os.path.join(data_dir, category)
            class_num = self.categories.index(category)
            # index as 0 and 1 to damaged and undamaged
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                cv2.resize(img_array, (128, 128))
                self.training_data.append([img_array, class_num])

    def load_model(self):

        self.model = tf.keras.models.load_model(f'{pickle_dir}128CNN.h5')
        self.model.load_weights(f'{pickle_dir}128CNN_weights.h5')

    @staticmethod
    def save_model_h5(model):
        log.info(f'Saving model on {pickle_dir}')
        model.save(f'{pickle_dir}128CNN.h5', include_optimizer=True)
        model.save_weights(f'{pickle_dir}128CNN_weights.h5')

    @staticmethod
    def save_model_tf(session, keep_var_names=None, output_names=None, clear_devices=True):
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(
                set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.compat.v1.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
                session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph
