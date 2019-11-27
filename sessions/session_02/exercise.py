import argparse
import os

import tensorflow as tf

from sessions.session_01 import exercise_01


def main(learning_rate, logdir):
    graph = tf.Graph()
    with graph.as_default():
        # Inputs and labels
        with tf.name_scope('input'):
            x = tf.placeholder(dtype=tf.float32, name='x')
            y = tf.placeholder(dtype=tf.float32, name='y')

        # ------------------------ FORWARD PASS -----------------------------------
        # Linear regression forward pass
        with tf.variable_scope('LinearRegressor'):
            W = tf.get_variable('W', shape=[], dtype=tf.float32)
            b = tf.get_variable('b', shape=[], dtype=tf.float32)
            z = W * x
            y_ = z + b

        # Compute loss
        with tf.name_scope('MSELoss'):
            diff = y_ - y
            loss = tf.pow(diff, 2)

        # ------------------------ BACKWARD PASS -----------------------------------
        with tf.name_scope('SGD'):
            with tf.name_scope('compute_gradients'):
                with tf.name_scope('MSELoss'):
                    # Loss backprop
                    loss_grad = 2 * diff

                with tf.name_scope('LinearRegressor'):
                    # Linear regression backprop
                    W_grad = x * loss_grad
                    b_grad = 1 * loss_grad
                    x_grad = W * loss_grad

        # ------------------------ OPTIMIZATION -----------------------------------
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)

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
