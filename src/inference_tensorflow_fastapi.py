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


@app.post('/image')
def default(item: dict):
    api.process_request(item)
    return '200'
