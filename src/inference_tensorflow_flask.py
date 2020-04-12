from src.service.api import Api
from flask import Flask, request
from src.service.tensorflow_network import TensorflowNetwork

import os

os.system('tensorflow_model_server '
          '--rest_api_port=8501 --model_name=model '
          '--model_base_path=/app/data/models &')
app = Flask(__name__)
api = Api(
    docker=True,
    cloud=True,
    processor_unit='CPU',
    inference_engine='tensorflow',
    net=TensorflowNetwork(),
    web_engine='Flask'
)


@app.route('/', methods=['GET'])
def hello():
    return 'Hello friend'


@app.route('/imageRemote', methods=['POST'])
def remote_image():
    res = api.remote_image_request(request.json)
    return {
        'result': res
    }


@app.route('/image', methods=['POST'])
def cloud_storage_handler():
    item = request.json
    api.cloud_storage_request(item)
    return '200'
