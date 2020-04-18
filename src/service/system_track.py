import psutil
import platform


class SystemTrack:
    def __init__(self, docker: bool, inference_engine: str, web_engine: str, cloud: bool, processor_unit: str):
        self.sys_information = platform.uname()
        self.sys_memory = psutil.virtual_memory()
        self.physical_cores = psutil.cpu_count(logical=False)
        self.total_cores = psutil.cpu_count(logical=True)
        self.system = self.sys_information.system
        self.processor = self.sys_information.processor
        self.system_memory = self.__get_size(self.sys_memory.total)
        self.system_memory_available = self.__get_size(self.sys_memory.available)
        self.so_version = self.sys_information.version
        self.so_release = self.sys_information.release

        self.docker = docker
        self.inference_engine = inference_engine
        self.web_engine = web_engine
        self.cloud = cloud
        self.processor_unit = processor_unit

    @staticmethod
    def __get_size(num_bytes, suffix="B"):
        factor = 1024
        for unit in ['', 'K', 'M', 'G', 'T', 'P']:
            if num_bytes < factor:
                return f'{num_bytes:.2f}{unit}{suffix}'
            num_bytes /= factor
