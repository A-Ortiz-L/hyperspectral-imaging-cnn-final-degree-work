from src.service.openvino_network import OpenVinoNetwork
from src.service.api import Api
from fastapi import FastAPI

app = FastAPI()
api = Api(
    docker=True,
    cloud=True,
    processor_unit='CPU',
    inference_engine='openvino',
    net=OpenVinoNetwork(),
    web_engine='FastApi'
)


@app.get('/')
def hello():
    return 'Hello World'


@app.post('/remoteImage')
def remote_image(item: dict):
    res = api.remote_image_request(item)
    return {
        'result': res
    }


@app.post('/image')
def cloud_storage_handler(item: dict):
    api.cloud_storage_request(item)
    return '200'
