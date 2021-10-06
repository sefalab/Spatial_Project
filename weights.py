import os
import sys
import json
import multiprocessing
from utils import config,util
import numpy as np

def calculate_class_weight(file_name):
    _, palette = util.get_label_info(config.class_dict)
    label_to_frequency = {}
    for i, label in enumerate(palette):
        label_to_frequency[i] = 0
#     print(config.data_dir+'/'+file_name.split('/')[0]+'/collapsed/'+file_name.split('/')[-1])
    image = util.load_label(config.data_dir+'/'+file_name.split('/')[0]+'/collapsed/'+file_name.split('/')[-1])
    for i, label in enumerate(palette):
        class_mask = (image == label)
        class_mask = np.all(class_mask, axis=2)
        class_mask = class_mask.astype(np.float32)
        class_frequency = np.sum(class_mask)
        label_to_frequency[i] += class_frequency
    return label_to_frequency

gt = {}
provinces =config.provinces
for prov in provinces:

    path =os.path.join(config.data_dir,config.data_splits)
    with open(path, 'r') as JSON:
        json_dict = json.load(JSON)
    file_names =json_dict[prov][1] 

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        frequencies = pool.map(calculate_class_weight, file_names)
    pool.close()
    _, palette = util.get_label_info(config.class_dict)
    label_to_frequency = {}
    for i, label in enumerate(palette):
        label_to_frequency[i] = 0

    for frequency in frequencies:
        label_to_frequency[0] += frequency[0]
        label_to_frequency[1] += frequency[1]
        label_to_frequency[2] += frequency[2]
        label_to_frequency[3] += frequency[3]

    class_weights = []
    total_frequency = sum(label_to_frequency.values())
    for label, frequency in label_to_frequency.items():
        class_weight = 1 / np.log(1.02 + (frequency / total_frequency))
        class_weights.append(class_weight)
    #class_weights = np.array(class_weights, np.float32)


    gt[prov] =class_weights

with open(config.weights, 'w') as fp:
    json.dump(gt, fp)