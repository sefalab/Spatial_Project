# Import libraries
import os
import sys
import numpy as np
import json, multiprocessing

from net import nn
from utils import config, util
from utils.dataset import input_fn

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from absl import app
from absl import flags
from absl import logging
from official.common import distribute_utils
from official.utils.flags import core as flags_core

def model_n(prov):
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    #Sets the threshold for what messages will be logged.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    
    #load mask_info
    _, palette = util.get_label_info(config.class_dict)

    #Load file_names from Json

    with open(config.data_splits, 'r') as JSON:
        json_dict_file_names = json.load(JSON)
    file_names =json_dict_file_names[prov][1]
    
    #file_names = [file_name[:-4] for file_name in os.listdir(os.path.join(config.data_dir, config.image_dir))]
    
    #Load weights for each class
    with open(config.weights, 'r') as JSON:
        json_dict_weights = json.load(JSON)
    weights = np.array(json_dict_weights[prov], np.float32)

#     FILENAMES = tf.io.gfile.glob("/spatial/data/DeepLab-tf/Dataset/tfRecords/test/*.tfrecord")
#     dataset = util.get_dataset(FILENAMES, False)

    #Get tfrecords
    FILENAMES = tf.io.gfile.glob(config.tf_data_dir+ "tfrecords/EC/*.tfrecord")
#     for file_ in FILENAMES:
#         os.system(config.tpu_bucket_service + file_)

    #Load data
    dataset = util.get_dataset(FILENAMES, False)
#     split_ind = int(0.75 * len(FILENAMES))
#     TRAINING_FILENAMES, VALID_FILENAMES = FILENAMES[:split_ind], FILENAMES[split_ind:]

    print("Train TFRecord Files:", len(FILENAMES))
#     print("Validation TFRecord Files:", len(VALID_FILENAMES))

#    dataset = util.get_dataset(TRAINING_FILENAMES, False)
#     valid_dataset = util.get_dataset(VALID_FILENAMES, False)
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu = prov.lower())
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    #strategy = tf.distribute.MirroredStrategy()    
    dataset = strategy.experimental_distribute_dataset(dataset)
    #weights = util.get_class_weights(file_names)

    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate =0.0001)
        model = nn.build_model((config.height, config.width, 3), len(palette))
        model(tf.zeros((1, config.height, config.width, 3)))

    with strategy.scope():
        loss_fn = nn.segmentation_loss(weights)


        def compute_loss(y_true, y_pred):
            return tf.reduce_sum(loss_fn(y_true, y_pred)) * 1. / config.batch_size

    with strategy.scope():
        @tf.function
        def train_step(image, y_true):
            with tf.GradientTape() as tape:
                y_pred = model(image)
                #tf.print("a:", y_true[0,0,0,:])
                loss = compute_loss(y_true, y_pred)
                
            train_variable = model.trainable_variables
            gradient = tape.gradient(loss, train_variable)
            optimizer.apply_gradients(zip(gradient, train_variable))

            return loss

    with strategy.scope():
        @tf.function
        def distribute_train_step(image, y_true):
            import keras.backend as KB
            
#             y_true = KB.one_hot(tf.cast(KB.flatten(y_true), tf.int32),
#                                 KB.int_shape(y_pred)[-1]+1)
#             unpacked = tf.unstack(y_true, axis=-1)
#             y_true = tf.stack(unpacked[:-1], axis=-1)
#             print(y_true)
            loss = strategy.run(train_step, args=(image, y_true))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)



    steps = len(file_names) // config.batch_size
    if not os.path.exists('weights/collapsed/test'):
        os.makedirs('weights/collapsed/test')
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
            model.save_weights(os.path.join("weights","collapsed", prov+f"_model_test.h5"))
        if step // steps == config.epochs:
            sys.exit("--- Stop Training ---")
    return prov+':---done'


def main():

    provinces =config.provinces
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