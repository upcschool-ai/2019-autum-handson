import argparse

import tensorflow as tf


# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py


def freeze_graph(checkpoint_path, output_node_names, output_path):
    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(checkpoint_path + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, checkpoint_path)

        print('{} ops in the original graph'.format(len(graph.as_graph_def().node)))

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            graph.as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(',')  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('%d ops in the final graph.' % len(output_graph_def.node))

    return output_graph_def


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', help='Path to the checkpoint from which to build the frozen model')
    parser.add_argument('output_node_names', help='Coma separated list of output tensor names')
    parser.add_argument('output_path', help='Path for the generated frozen model')
    args = parser.parse_args()

    freeze_graph(args.checkpoint_path, args.output_node_names, args.output_path)
