from src.service.openvino_network import OpenVinoNetwork
from src.service.api import Api
from fastapi import FastAPI
from src.service.system_track import SystemTrack

app = FastAPI()
api = Api(
    sys=SystemTrack(
        docker=True,
        cloud=True,
        processor_unit='CPU',
        inference_engine='openvino',
        web_engine='FastApi'
    ),
    net=OpenVinoNetwork()
)


@app.post('/image')
def cloud_storage_handler(item: dict):
    api.cloud_storage_request(item)
    return '200'
