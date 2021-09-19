  
import csv
import multiprocessing
import os
import random

import cv2
import numpy as np

from utils import config


def load_image(file_name):
    path = os.path.join(config.data_dir, config.image_dir, file_name + '.png')
    return cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)


def load_label(file_name):
    path = os.path.join(config.data_dir, config.label_dir, file_name + '_L.png')
    return cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)


def write_image(path, image):
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def get_label_info(csv_path):
    class_names = []
    label_values = []
    with open(csv_path, 'r') as reader:
        file_reader = csv.reader(reader, delimiter=',')
        next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
    return class_names, label_values


def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)
    return x


def colour_code_segmentation(image, label_values):
    label_values = np.array(label_values)
    x = label_values[image.astype(int)]
    return x


def save_images(images, path, cols=1, titles=None):
    from matplotlib import pyplot
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    pyplot.axis('off')
    fig = pyplot.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            pyplot.gray()
        pyplot.imshow(image)
        pyplot.axis("off")
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    pyplot.savefig(path)
    pyplot.close(fig)


def calculate_class_weight(file_name):
    _, palette = get_label_info(os.path.join(config.data_dir, 'class_dict.csv'))
    label_to_frequency = {}
    for i, label in enumerate(palette):
        label_to_frequency[i] = 0

    image = load_label(file_name)
    for i, label in enumerate(palette):
        class_mask = (image == label)
        class_mask = np.all(class_mask, axis=2)
        class_mask = class_mask.astype(np.float32)
        class_frequency = np.sum(class_mask)
        label_to_frequency[i] += class_frequency
    return label_to_frequency


def get_class_weights(file_names):
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        frequencies = pool.map(calculate_class_weight, file_names)
    pool.close()
    _, palette = get_label_info(os.path.join(config.data_dir, 'class_dict.csv'))
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
    class_weights = np.array(class_weights, np.float32)
    return class_weights


def random_augmentation(image, label):
    if random.uniform(0, 1) > 0.5:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
    if random.uniform(0, 1) > 0.5:
        factor = 1.0 + random.uniform(-0.1, 0.1)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        image = cv2.LUT(image, table)
    if random.uniform(0, 1) > 0.5:
        angle = random.uniform(-20, 20)
        matrix = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
        label = cv2.warpAffine(label, matrix, (label.shape[1], label.shape[0]), flags=cv2.INTER_NEAREST)

    return image, label


def random_crop(image, label, crop_height=config.height, crop_width=config.width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Dimension Exception')

    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1] - crop_width)
        y = random.randint(0, image.shape[0] - crop_height)

        if len(label.shape) == 3:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width, :]
        else:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width]
    else:
        raise Exception('Exception')


def one_hot(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map

def get_weights():
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
        _, palette = util.get_label_info(os.path.join(config.data_dir, 'class_dict.csv'))
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

    with open('data_.json', 'w') as fp:
        json.dump(gt, fp)