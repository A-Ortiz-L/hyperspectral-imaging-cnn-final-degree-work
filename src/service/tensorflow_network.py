import numpy as np
import time
import cv2
import requests
import json
from typing import Tuple
import os


class TensorflowNetwork:
    def __init__(self):
        self.model_uri = 'http://localhost:8501/v1/models/model:predict'
        self.init_tensorflow_serve()

    @staticmethod
    def shape_image(file_route):
        img_array = cv2.imread(file_route, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (128, 128))
        img = new_array.reshape(-1, 128, 128, 1) / 255.0
        img = np.float32(img).tolist()
        return img

    def process_image(self, file_route) -> Tuple[bool, float]:
        start = time.time()
        image = self.shape_image(file_route)
        predict = self.network_request(image)
        return predict, (time.time() - start)

    def network_request(self, image) -> bool:
        headers = {"content-type": "application/json"}
        data = json.dumps({"signature_name": "serving_default", "instances": image})
        res = requests.post(self.model_uri, data=data,
                            headers=headers)
        predictions = json.loads(res.text)['predictions']
        predict = True if predictions[0][0] >= 0.5 else False
        return predict

    @staticmethod
    def init_tensorflow_serve():
        os.system('tensorflow_model_server '
                  '--rest_api_port=8501 --model_name=model '
                  '--model_base_path=/app/data/models &')
