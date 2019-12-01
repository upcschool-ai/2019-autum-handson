import argparse
import os

import tensorflow as tf

from sessions.session_01 import exercise_01


def main(learning_rate, logdir):
    graph = tf.Graph()
    with graph.as_default():
        # Inputs and labels
        x = tf.placeholder(dtype=tf.float32, name='x')
        y = tf.placeholder(dtype=tf.float32, name='y')

        # ------------------------ FORWARD PASS -----------------------------------
        # Linear regression forward pass
        W = tf.get_variable('W', shape=[], dtype=tf.float32)
        b = tf.get_variable('b', shape=[], dtype=tf.float32)
        z = W * x
        y_ = z + b

        # Compute loss
        diff = y_ - y
        loss = tf.pow(diff, 2)

        # ------------------------ BACKWARD PASS -----------------------------------
        # Loss backprop
        loss_grad = 2 * diff

        # Linear regression backprop
        W_grad = x * loss_grad
        b_grad = 1 * loss_grad
        x_grad = W * loss_grad

        # ------------------------ OPTIMIZATION -----------------------------------
        W_update = W.assign(W - learning_rate * W_grad)
        b_update = b.assign(b - learning_rate * b_grad)
        train_op = tf.group(W_update, b_update)

    sess = tf.Session(graph=graph)
    with sess:
        sess.run(tf.global_variables_initializer())
        data = exercise_01.DataDistribution()
        for input_data, label in data.generate(num_iters=5000):
            prediction, loss_val, _ = sess.run([y_, loss, train_op], feed_dict={x: input_data, y: label})
            print('[Loss: {}] Input: {} --> Prediction {}. Label {}'.format(loss_val, input_data, prediction, label))
        W_pred, b_pred = sess.run([W, b])
        print('W GT: {}. W pred: {}'.format(data.W, W_pred))
        print('b GT: {}. b pred: {}'.format(data.b, b_pred))

    writer = tf.summary.FileWriter(os.path.expanduser(logdir), graph=graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=10e-6,
                        help='Learning rate for the optimization step')
    parser.add_argument('-l', '--logdir', help='Log dir for tfevents')
    args = parser.parse_args()
    main(args.learning_rate, args.logdir)
