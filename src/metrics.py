import pandas as pd
from os import getcwd
import os

openvino_fastapi_4 = pd.read_csv(getcwd() + '/result/snapshot/openvino_fastapi_4core_cpu.csv')
openvino_flask_4 = pd.read_csv(getcwd() + '/result/snapshot/openvino_flask_4core_cpu.csv')
openvino_fastapi_8 = pd.read_csv(getcwd() + '/result/snapshot/openvino_fastapi_8core_cpu.csv')
openvino_flask_8 = pd.read_csv(getcwd() + '/result/snapshot/openvino_flask_8core_cpu.csv')

tf_fastapi_4 = pd.read_csv(getcwd() + '/result/snapshot/tensroflow_fastapi_4core_cpu_tensor_serve.csv')
tf_flask_4 = pd.read_csv(getcwd() + '/result/snapshot/tensorflow_flask_4core_cpu_tensor_serve.csv')
tf_fastapi_8 = pd.read_csv(getcwd() + '/result/snapshot/tensorflow_fastapi_8core_cpu_tensor_serve.csv')
tf_flask_8 = pd.read_csv(getcwd() + '/result/snapshot/tensorflow_flask_8core_cpu_tensor_serve.csv')

damaged = pd.read_csv(getcwd() + '/result/snapshot/damaged.csv')
df = pd.concat([openvino_fastapi_4, openvino_fastapi_8, openvino_flask_4, openvino_flask_8,
                tf_fastapi_4, tf_fastapi_8, tf_flask_4, tf_flask_8])

# print(df.groupby(['inference_engine']).mean()['inference_time'])
# print(df.groupby(['inference_engine', 'physical_core']).mean()['inference_time'])
# print(df.groupby(['inference_engine', 'physical_core', 'web_engine']).mean()['inference_time'])
# print(df.groupby(['inference_engine', 'physical_core', 'web_engine']).mean()['total_execution_time'])


r = df[(df['inference_engine'] == 'openvino')]
print(len(r.index))
skr = r.set_index('image_name').join(damaged.set_index('name'))
print(skr.head())
print(len(skr.index))
