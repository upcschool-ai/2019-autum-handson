import tensorflow as tf

from sessions.session_01 import exercise_01


def main():
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=[], name='x')
        W = tf.get_variable('W', shape=[], dtype=tf.float32, initializer=tf.initializers.random_normal())
        b = tf.get_variable('b', shape=[], dtype=tf.float32, initializer=tf.initializers.random_normal())
        z = W * x
        y = z + b

    sess = tf.Session(graph=graph)
    with sess:
        sess.run(tf.global_variables_initializer())
        data = exercise_01.DataDistribution()
        for input_data, label in data.generate(num_iters=10):
            prediction = sess.run(y, feed_dict={x: input_data})
            print('Input: {} --> Prediction {}. Label {}'.format(input_data, prediction, label))

    # TODO: E04: Implement optimization step manually!
    pass


if __name__ == '__main__':
    main()
