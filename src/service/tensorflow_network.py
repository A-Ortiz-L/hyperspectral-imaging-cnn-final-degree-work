import numpy as np
import time
import cv2
import requests
import json


class TensorflowNetwork:
    def __init__(self):
        pass

    @staticmethod
    def process_image(file_route):
        start = time.time()
        img_array = cv2.imread(file_route, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (128, 128))
        img = new_array.reshape(-1, 128, 128, 1) / 255.0
        img = np.float32(img).tolist()

        headers = {"content-type": "application/json"}
        data = json.dumps({"signature_name": "serving_default", "instances": img})
        res = requests.post('http://localhost:8501/v1/models/model:predict', data=data,
                            headers=headers)
        predictions = json.loads(res.text)['predictions']
        predict = True if predictions[0][0] >= 0.5 else False
        return predict, (time.time() - start)
