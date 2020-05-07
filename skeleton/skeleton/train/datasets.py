import csv
import multiprocessing
import os

import tensorflow as tf

from skeleton import constants


def create_dataset(dataset_path, images_dir, num_epochs, batch_size, shuffle=500, prefetch=10):
    dataset = tf.data.Dataset.from_generator(lambda: _generator(dataset_path, images_dir),
                                             output_types=(tf.string, tf.int32),
                                             output_shapes=(tf.TensorShape([]), tf.TensorShape([])))
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.shuffle(shuffle)  # Shuffling buffer
    dataset = dataset.map(_create_sample, num_parallel_calls=multiprocessing.cpu_count())
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(prefetch)  # Pipelining
    return dataset


def _generator(path, images_dir):
    with open(path) as f:
        reader = csv.reader(f)
        for label, image_path in reader:
            image_path = os.path.join(images_dir, image_path)
            yield image_path, int(label)


def _create_sample(image_path, label):
    with tf.name_scope('create_sample'):
        with tf.name_scope('read_image'):
            raw_image = tf.read_file(image_path)
            image = tf.image.decode_jpeg(raw_image, channels=3)

        with tf.name_scope('preprocessing'):
            image = tf.cast(image, dtype=tf.float32)
            image = tf.subtract(image, constants.IMAGENET_MEAN, name='mean_substraction')
            image = tf.image.resize_images(image, size=(256, 256))

        with tf.name_scope('data_augmentation'):
            image = tf.image.random_crop(image, size=(227, 227, 3))
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=20)

    return image, label
