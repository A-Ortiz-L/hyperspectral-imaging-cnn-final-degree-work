from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
import cv2
from config.cfg import data_dir
import os
#from src.entity.KerasModel import KerasModel
import time


def pre_process_image(image_path):
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.resize(img_array, (128, 128))
    return img_array


def process_folder_openvino(folder):
    for img in os.listdir(folder):
        processed_img = pre_process_image(f'{folder}{img}')
        res = exec_net.infer(inputs={input_layer: processed_img})
        # print(res)
        output_node_name = list(res.keys())[0]
        res = res[output_node_name]
        idx = np.argsort(res[0])[-1]
        # print(res)
        # print(idx)


def process_folder_tensforflow(folder, model_keras):
    for img in os.listdir(folder):
        img = k.prepare_file(f'{folder}{img}')
        res = model_keras.predict(img)
        # print(res)


if __name__ == '__main__':
    plugin = IEPlugin(device="CPU")
    net = IENetwork(model='./data/xml_model/tf_model1.xml',
                    weights='./data/xml_model/tf_model1.bin')
    input_layer = next(iter(net.inputs))
    input_shape = net.inputs[input_layer].shape
    # image = cv2.imread('./data/damaged/post_001_073.png')
    # pre_img = pre_processing(image, 128, 128)
    exec_net = plugin.load(network=net)

    #k = KerasModel()
    #k.load_model()
    acc = 0
    '''for i in range(1, 100):
        start = time.time()
        process_folder_tensforflow(f'{data_dir}damaged/', k.model)
        process_folder_tensforflow(f'{data_dir}undamaged/', k.model)
        end = time.time()
        print(f'Time : {end - start} seconds')
        acc += (end - start)

    print(acc, f'Mean {acc / 100}')
    '''
    for i in range(1, 100):
        start = time.time()
        process_folder_openvino(f'{data_dir}undamaged/')
        process_folder_openvino(f'{data_dir}damaged/')
        end = time.time()
        print(f'Time : {end - start} seconds')
        acc += (end - start)
    print(acc, f'Mean {acc / 100}')
