import argparse
import datetime
import logging
import os

import tensorflow as tf
import yaml
from tensorflow.python.util import deprecation

from skeleton.train import datasets, models

# Disable tensorflow (almost all) logging bullshit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Logging
logger = logging.getLogger('skeleton')
logger.setLevel(logging.INFO)
logger.addFilter(logging.Filter(logger.name))
loghandler = logging.StreamHandler()
loghandler.setLevel(logging.DEBUG)
loghandler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s: %(message)s'))
logger.addHandler(loghandler)


def main(dataset_csv, images_dir, experiment_config, with_gpu):
    # ----------------- TRAINING SETUP ---------------- #
    with open(experiment_config, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(str(config))

    experiment_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logdir = os.path.expanduser(config['logdir'])
    logdir = os.path.join(logdir, experiment_id)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    checkpointdir = os.path.expanduser(config['checkpoint'])
    checkpointdir = os.path.join(checkpointdir, experiment_id)
    checkpoint = os.path.join(checkpointdir, 'model')
    if not os.path.isdir(checkpointdir):
        os.makedirs(checkpointdir)

    # ----------------- DEFINITION PHASE ------------------- #
    global_step = tf.get_variable('global_step', dtype=tf.int32, initializer=0, trainable=False)

    # Input pipeline
    with tf.device('/cpu:0'):
        with tf.name_scope('input_pipeline'):
            dataset = datasets.create_dataset(dataset_csv, images_dir, config['num_epochs'], config['batch_size'])
            iterator = dataset.make_one_shot_iterator()
            images, labels = iterator.get_next()

            tf.summary.image('input', images, max_outputs=4)

    images = tf.identity(images, name='images')
    labels = tf.identity(labels, name='labels')

    train_device = '/gpu:0' if with_gpu else '/cpu:0'
    with tf.device(train_device):
        # Model
        logits = models.alexnet(images, config['num_classes'])
        predictions = tf.nn.softmax(logits, name='predictions')  # For inference purposes

        # Loss
        loss_op = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        tf.summary.scalar('Loss', loss_op)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate'])
        train_step = optimizer.minimize(loss_op, global_step=global_step)

    # Summary op
    summary_op = tf.summary.merge_all()

    # ----------------- RUN PHASE ------------------- #
    # Weight saver
    saver = tf.train.Saver(max_to_keep=None)

    # Summary writer
    writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while True:
                # Run the train step
                _, loss, step, summ_val = sess.run([train_step, loss_op, global_step, summary_op])
                # Print how the loss is evolving per step in order to check if the model is converging
                if step < 20 or step % config['log_iters'] == 0:
                    logger.info('Step %d\tLoss=%0.4f', step, loss)
                    writer.add_summary(summ_val, global_step=step)
                # Save the graph definition and its weights
                if step % config['checkpoint_iters'] == 0:
                    saver.save(sess, save_path=checkpoint, global_step=step)
        except tf.errors.OutOfRangeError:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('dataset_csv', help='Path to the CSV decribing the dataset')
    parser.add_argument('images_dir', help='Path to the images directory')
    parser.add_argument('experiment_config', help='Path to the experiment configuration')
    parser.add_argument('--gpu', '-g', action='store_true', help='Either to train with GPU or not')
    args = parser.parse_args()

    main(args.dataset_csv, args.images_dir, args.experiment_config, args.gpu)
