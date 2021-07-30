import os
import tensorflow as tf
import numpy as np

from net import nn
from utils import config
from utils import util
import tqdm


def main():
    _, palette = util.get_label_info(os.path.join(config.data_dir, "class_dict.csv"))

    model = nn.build_model(classes=len(palette))
    model(tf.zeros((1, config.height, config.width, 3)))

    file_names = [file_name[:-4] for file_name in os.listdir(os.path.join(config.data_dir, config.test_img_dir))]
    for file_name in tqdm.tqdm(file_names):
        image = util.load_image(file_name)
        label = util.load_label(file_name)
        image, label = util.random_crop(image, label)

        image = np.expand_dims(image, 0).astype('float32')

        output = model.predict(image / 255.0)
        output = np.array(output[0, :, :, :])
        output = np.argmax(output, axis=-1)
        output = util.colour_code_segmentation(output, palette)
        output = np.uint8(output)
        util.save_images([output, label], os.path.join('results', f'{file_name}.png'), titles=['Pred', 'Label'])


if __name__ == '__main__':
    if not os.path.exists('results'):
        os.makedirs('results')
    main()
