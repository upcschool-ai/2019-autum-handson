import tensorflow as tf


def alexnet(images, num_classes, training=True, name='AlexNet'):
    with tf.variable_scope(name):
        conv1 = _conv_layer(images, filters=96, kernel_size=(11, 11), strides=(4, 4), lrn=True, max_pool=True,
                            name='conv1')
        tf.summary.histogram('conv1', conv1)
        conv2 = _conv_layer(conv1, filters=256, kernel_size=(5, 5), lrn=True, max_pool=True, padding='same',
                            name='conv2')
        tf.summary.histogram('conv2', conv2)
        conv3 = _conv_layer(conv2, filters=384, kernel_size=(3, 3), padding='same', name='conv3')
        tf.summary.histogram('conv3', conv3)
        conv4 = _conv_layer(conv3, filters=384, kernel_size=(3, 3), padding='same', name='conv4')
        tf.summary.histogram('conv4', conv4)
        conv5 = _conv_layer(conv4, filters=256, kernel_size=(3, 3), max_pool=True, name='conv5')
        tf.summary.histogram('conv5', conv5)

        flat = tf.reshape(conv5, [-1, 5 * 5 * 256])
        fc1 = _fully_connected(flat, units=4096, dropout_rate=0.5, training=training, name='fc1')
        tf.summary.histogram('fc1', fc1)
        fc2 = _fully_connected(fc1, units=4096, dropout_rate=0.5, training=training, name='fc2')
        tf.summary.histogram('fc2', fc2)
        logits = _fully_connected(fc2, units=num_classes, activation=None, training=training, name='fc3')
        tf.summary.histogram('logits', logits)

        return logits


def _conv_layer(inputs, filters, kernel_size, strides=(1, 1), lrn=False, max_pool=False, padding='valid',
                name='conv_layer'):
    with tf.variable_scope(name):
        output = tf.layers.conv2d(
            inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=tf.nn.relu, padding=padding,
            kernel_initializer=tf.initializers.random_normal(mean=0, stddev=0.01),
            bias_initializer=tf.initializers.ones(),
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        )
        if lrn:
            output = tf.nn.lrn(output, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)
        if max_pool:
            output = tf.layers.max_pooling2d(output, pool_size=(3, 3), strides=(2, 2))
        return output


def _fully_connected(inputs, units, activation=tf.nn.relu, dropout_rate=None, training=True, name='fully_connected'):
    with tf.variable_scope(name):
        output = tf.layers.dense(
            inputs, units=units, activation=activation,
            kernel_initializer=tf.initializers.random_normal(mean=0, stddev=0.01),
            bias_initializer=tf.initializers.ones(),
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        )
        if dropout_rate:
            output = tf.layers.dropout(output, training=training, rate=dropout_rate)

        return output
