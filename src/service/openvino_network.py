from openvino.inference_engine import IENetwork, IEPlugin
import cv2
from config.cfg import pickle_dir
import time
from typing import Tuple


class OpenVinoNetwork:
    def __init__(self):
        self.plugin = IEPlugin(device='CPU')
        self.net = IENetwork(model=f'{pickle_dir}model.xml',
                             weights=f'{pickle_dir}model.bin')
        self.exec_net = self.plugin.load(network=self.net)

        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        self.net.batch_size = 1

    def process_image(self, image_path, shape) -> Tuple[bool, float]:
        start = time.time()
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (shape, shape))
        image = image.reshape(shape, shape) / 255.0
        res = self.exec_net.infer(inputs={self.input_blob: image})
        res = res[self.out_blob]
        res = False if res < 0.5 else True
        return res, time.time() - start
