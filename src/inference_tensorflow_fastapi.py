from src.service.api import Api
from fastapi import FastAPI
from src.service.tensorflow_network import TensorflowNetwork
from src.service.system_track import SystemTrack

app = FastAPI()
api = Api(
    sys=SystemTrack(
        docker=True,
        cloud=True,
        processor_unit='CPU',
        inference_engine='tensorflow',
        web_engine='FastApi'
    ),
    net=TensorflowNetwork(),
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
