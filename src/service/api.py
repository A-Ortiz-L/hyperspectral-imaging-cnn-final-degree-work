import time
from config.cfg import data_dir, bucket
from src.service.google_storage import GoogleStorage
from src.service.google_big_query import GoogleBigQuery
import psutil
import platform
import os
import cv2
import requests
import numpy as np


class Api:
    def __init__(self, inference_engine: str, web_engine: str,
                 cloud: bool, processor_unit: str, net, docker=True):
        self.__docker = docker
        self.__inference_engine = inference_engine
        self.__web_engine = web_engine
        self.__cloud = cloud
        self.__processor_unit = processor_unit
        self.__storage = GoogleStorage()
        self.__big_query = GoogleBigQuery()
        self.__sys_information = platform.uname()
        self.__net = net

    def process_request(self, item: dict):
        start = time.time()
        image_name = item['name']
        size = item['size']
        file_type = item['contentType']
        time_created = item['timeCreated']
        image_path = f'{data_dir}{image_name}'
        self.__storage.download_blob(bucket, image_name, image_path)
        prediction, inference_time = self.__net.process_image(image_path)

        system = self.__sys_information.system
        processor = self.__sys_information.processor
        sys_memory = psutil.virtual_memory()
        physical_cores = psutil.cpu_count(logical=False)
        total_cores = psutil.cpu_count(logical=True)
        system_memory = self.__get_size(sys_memory.total)
        system_memory_available = self.__get_size(sys_memory.available)
        so_version = self.__sys_information.version
        so_release = self.__sys_information.release
        total_time = time.time() - start

        row = [
            (
                image_name, size, file_type,
                time_created, prediction, inference_time,
                total_time, physical_cores,
                total_cores, system, processor, system_memory, system_memory_available,
                so_version, so_release, self.__inference_engine,
                self.__web_engine, self.__processor_unit,
                self.__docker, self.__cloud
            )
        ]
        self.__big_query.insert_row(row)
        os.remove(image_path)

    def shape_image(self, file_route):
        start = time.time()
        img_array = cv2.imread(file_route, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (128, 128))
        img = new_array.reshape(-1, 128, 128, 1) / 255.0
        res = requests.post('localhost:8500', data=img)
        print(res)
        predict = True if res[0][0] >= 0.5 else False
        return predict, (time.time() - start)

    @staticmethod
    def __get_size(num_bytes, suffix="B"):
        factor = 1024
        for unit in ['', 'K', 'M', 'G', 'T', 'P']:
            if num_bytes < factor:
                return f'{num_bytes:.2f}{unit}{suffix}'
            num_bytes /= factor
