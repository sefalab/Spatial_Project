import os
import numpy as np
import tensorflow as tf

from utils import config, util
from tensorflow.keras import utils


class Generator(utils.Sequence):
    def __init__(self, file_names, palette):
        self.palette = palette
        self.file_names = file_names

    def __len__(self):
        return int(np.floor(len(self.file_names) / config.batch_size))

    def __getitem__(self, index):
        image = util.load_image(self.file_names[index])
        label = util.load_label(self.file_names[index])
        image, label = util.random_crop(image, label)
        image, label = util.random_augmentation(image, label)
        one_hot = util.one_hot(label, self.palette)
        return np.float32(image) / 255.0, np.float32(one_hot)

    def on_epoch_end(self):
        np.random.shuffle(self.file_names)


def input_fn(file_names, palette):
    def generator_fn():
        generator = utils.OrderedEnqueuer(Generator(file_names, palette), True)
        generator.start(workers=os.cpu_count() - 4)
        while True:
            image, label = generator.get().__next__()
            yield image, label

    dataset = tf.data.Dataset.from_generator(generator_fn,
                                             (tf.float32, tf.float32),
                                             (tf.TensorShape([config.height, config.width, 3]),
                                              tf.TensorShape([config.height, config.width, len(palette)])))

    dataset = dataset.batch(config.batch_size)
    dataset = dataset.repeat(config.epochs)
    dataset = dataset.prefetch(10 * config.batch_size)
    return dataset