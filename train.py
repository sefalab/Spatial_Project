import os
import sys
import multiprocessing

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from net import nn
from utils import config, util
from utils.dataset import input_fn
import json, multiprocessing

# Import libraries
from absl import app
from absl import flags
from absl import logging
from official.common import distribute_utils
from official.utils.flags import core as flags_core

FLAGS = flags.FLAGS


def calculate_class_weight(file_name):
    _, palette = util.get_label_info(os.path.join(config.data_dir, 'class_dict.csv'))
    label_to_frequency = {}
    for i, label in enumerate(palette):
        label_to_frequency[i] = 0

    image = util.load_label(file_name)
    for i, label in enumerate(palette):
        class_mask = (image == label)
        class_mask = np.all(class_mask, axis=2)
        class_mask = class_mask.astype(np.float32)
        class_frequency = np.sum(class_mask)
        label_to_frequency[i] += class_frequency
    return label_to_frequency


def model_n(prov):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    #Sets the threshold for what messages will be logged.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    strategy = tf.distribute.MirroredStrategy()

    _, palette = util.get_label_info(os.path.join(config.data_dir, 'class_dict.csv'))

    import json
    path =os.path.join(config.data_dir,config.data_splits)
    with open(path, 'r') as JSON:
        json_dict = json.load(JSON)
    file_names =json_dict[prov][1]
    #file_names = [file_name[:-4] for file_name in os.listdir(os.path.join(config.data_dir, config.image_dir))]
    
    with open('data_.json', 'r') as JSON:
        json_dict_ = json.load(JSON)
    weights = np.array(json_dict_[prov], np.float32)

    dataset = input_fn(file_names, palette)
    dataset = strategy.experimental_distribute_dataset(dataset)
    #weights = util.get_class_weights(file_names)

    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(lr =0.0001)
        model = nn.build_model((config.height, config.width, 3), len(palette))
        model(tf.zeros((1, config.height, config.width, 3)))

    with strategy.scope():
        loss_fn = nn.segmentation_loss(weights)


        def compute_loss(y_true, y_pred):
            return tf.reduce_sum(loss_fn(y_true, y_pred)) * 1. / config.batch_size

    with strategy.scope():
        def train_step(image, y_true):
            with tf.GradientTape() as tape:
                y_pred = model(image)
                loss = compute_loss(y_true, y_pred)
            train_variable = model.trainable_variables
            gradient = tape.gradient(loss, train_variable)
            optimizer.apply_gradients(zip(gradient, train_variable))

            return loss

    with strategy.scope():
        @tf.function
        def distribute_train_step(image, y_true):
            loss = strategy.experimental_run_v2(train_step, args=(image, y_true))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)



    steps = len(file_names) // config.batch_size
    if not os.path.exists('weights/collapsed/V2'):
        os.makedirs('weights/collapsed/V2')
    pb = tf.keras.utils.Progbar(steps, stateful_metrics=['loss',tf.keras.metrics.MeanIoU(num_classes=4)])
    for step, inputs in enumerate(dataset):
        if step % steps == 0:
            print(f'Epoch {step // steps + 1}/{config.epochs}')
            pb = tf.keras.utils.Progbar(steps, stateful_metrics=['loss',tf.keras.metrics.MeanIoU(num_classes=4)])
        step += 1
        image, y_true = inputs
        loss = distribute_train_step(image, y_true)
        pb.add(1, [('loss', loss)])
        if step % steps == 0:
            model.save_weights(os.path.join("weights","collapsed", prov+f"_model_v2.h5"))
        if step // steps == config.epochs:
            sys.exit("--- Stop Training ---")
    return prov+':---done'


def main():


#     gt = {}
#     provinces =['EC','FS','GT','KZN','LIM','MP','NC','NW','WC']
#     for prov in provinces:

#         path =os.path.join(config.data_dir,config.data_splits)
#         with open(path, 'r') as JSON:
#             json_dict = json.load(JSON)
#         file_names =json_dict[prov][1] 

#         with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
#             frequencies = pool.map(calculate_class_weight, file_names)
#         pool.close()
#         _, palette = util.get_label_info(os.path.join(config.data_dir, 'class_dict.csv'))
#         label_to_frequency = {}
#         for i, label in enumerate(palette):
#             label_to_frequency[i] = 0

#         for frequency in frequencies:
#             label_to_frequency[0] += frequency[0]
#             label_to_frequency[1] += frequency[1]
#             label_to_frequency[2] += frequency[2]
#             label_to_frequency[3] += frequency[3]

#         class_weights = []
#         total_frequency = sum(label_to_frequency.values())
#         for label, frequency in label_to_frequency.items():
#             class_weight = 1 / np.log(1.02 + (frequency / total_frequency))
#             class_weights.append(class_weight)
#         #class_weights = np.array(class_weights, np.float32)


#         gt[prov] =class_weights

#     with open('data_.json', 'w') as fp:
#         json.dump(gt, fp)

    provinces =['EC','FS','GT','KZN','LIM','MP','NC','NW','WC']
    complete_provs =[]
    for prov in provinces:
        tmp = model_n(prov)
        complete_provs.append(tmp)
        
    print(complete_provs)
    
#     with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
#         frequencies = pool.map(model_n, provinces)
#     pool.close()
    
    
    
if __name__ == '__main__':
    main()
