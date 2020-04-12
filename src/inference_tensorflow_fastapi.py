from src.service.api import Api
from fastapi import FastAPI
from src.service.tensorflow_network import TensorflowNetwork

app = FastAPI()
api = Api(
    docker=True,
    cloud=True,
    processor_unit='CPU',
    inference_engine='tensorflow',
    net=TensorflowNetwork(),
    web_engine='FastApi'
)


@app.get('/')
def hello():
    return 'Hello World'


@app.get('/remoteImage')
def remote_image(item: dict):
    res = api.remote_image_request(item)
    return {
        'result': res
    }


@app.post('/image')
def cloud_storage_handler(item: dict):
    api.cloud_storage_request(item)
    return '200'
