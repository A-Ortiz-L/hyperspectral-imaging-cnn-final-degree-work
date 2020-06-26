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
        for f in self.net.inputs:
            self.net.inputs[f].precision = 'FP16'
        self.net.batch_size = 1
        self.image_shape = 128

    def process_image(self, image_path) -> Tuple[bool, float]:

        image = self.shape_image(image_path)
        start = time.time()
        res = self.network_request(image)
        return res, time.time() - start

    def shape_image(self, file_route):
        image = cv2.imread(file_route, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.image_shape, self.image_shape))
        image = image.reshape(self.image_shape, self.image_shape) / 255.0
        return image

    def network_request(self, image) -> bool:
        res = self.exec_net.infer(inputs={self.input_blob: image})
        res = res[self.out_blob]
        res = False if res < 0.5 else True
        return res
