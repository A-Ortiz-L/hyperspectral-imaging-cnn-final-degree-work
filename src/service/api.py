import time
from config.cfg import data_dir, bucket
from src.service.google_storage import GoogleStorage
from src.service.google_big_query import GoogleBigQuery
from src.service.system_track import SystemTrack
import os

class Api:
    def __init__(self, net, sys: SystemTrack):
        self.__storage = GoogleStorage()
        self.__big_query = GoogleBigQuery()
        self.__net = net
        self.sys_track = sys

    def cloud_storage_request(self, item: dict):
        start = time.time()
        image_name = item['name']
        size = item['size']
        file_type = item['contentType']
        time_created = item['timeCreated']
        image_path = f'{data_dir}{image_name}'
        self.__storage.download_blob(bucket, image_name, image_path)
        prediction, inference_time = self.__net.process_image(image_path)
        total_time = time.time() - start
        row = [
            (
                image_name, size, file_type,
                time_created, prediction, inference_time,
                total_time, self.sys_track.physical_cores,
                self.sys_track.total_cores, self.sys_track.system, self.sys_track.processor,
                self.sys_track.system_memory,
                self.sys_track.system_memory_available,
                self.sys_track.so_version, self.sys_track.so_release, self.sys_track.inference_engine,
                self.sys_track.web_engine, self.sys_track.processor_unit,
                self.sys_track.docker, self.sys_track.cloud
            )
        ]
        self.__big_query.insert_row(row)
        os.remove(image_path)

    def remote_image_request(self, item: dict):
        pass
