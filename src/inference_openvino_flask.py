from src.service.openvino_network import OpenVinoNetwork
from src.service.api import Api
from flask import Flask, request

app = Flask(__name__)
api = Api(
    docker=True,
    cloud=True,
    processor_unit='CPU',
    inference_engine='openvino',
    net=OpenVinoNetwork(),
    web_engine='Flask'
)


@app.route('/', methods=['GET'])
def hello():
    return 'Hello my friend'


@app.route('/image', methods=['POST'])
def default():
    item = request.json
    api.process_request(item)
    return '200'
