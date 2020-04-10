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


@app.post('/image')
def default(item: dict):
    api.process_request(item)
    return '200'
