import argparse
import time

import tensorflow as tf

from skeleton.train import datasets


def main(dataset_path, images_dir, num_epochs, batch_size):
    with tf.device('/cpu:0'):  # To force the graph operations of the input pipeline to be placed in the CPU
        with tf.name_scope('input_pipeline'):
            dataset = datasets.create_dataset(dataset_path, images_dir, num_epochs, batch_size)
            iterator = dataset.make_one_shot_iterator()
            batch = iterator.get_next()

    with tf.Session() as sess:
        try:
            while True:
                start = time.time()
                images, labels = sess.run(batch)
                duration = time.time() - start
                print('Time per batch: {}'.format(duration))
        except tf.errors.OutOfRangeError:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('dataset_path', help='Path to dataset description')
    parser.add_argument('images_dir', help='Image directory')
    parser.add_argument('-e', '--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=5, help='Batch size')

    args = parser.parse_args()
    main(args.dataset_path, args.images_dir, args.num_epochs, args.batch_size)
