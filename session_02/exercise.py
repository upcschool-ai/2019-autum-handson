import tensorflow as tf


def main():
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=[], name='x')
        W = tf.get_variable('W', shape=[], dtype=tf.float32, initializer=tf.initializers.random_normal())
        b = tf.get_variable('b', shape=[], dtype=tf.float32, initializer=tf.initializers.random_normal())
        z = W * x
        y = z + b

    # TODO: E03: Run a forward pass --> Run phase (use session and the DataDistribution class from previous exercise)

    # TODO: E04: Implement optimization step manually!
    pass


if __name__ == '__main__':
    main()
