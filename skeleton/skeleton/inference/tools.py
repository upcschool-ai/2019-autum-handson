import tensorflow as tf


def load_graph(model_path, graph=None, prefix='', **kwargs):
    """Loads the graph stored in pb file.

    :param str model_path: Path to frozen model (.pb) file
    :param tf.Graph graph: Graph where the model should be loaded. Defaults to the default graph
    :param str prefix: Prefix append at the beginning of tensor variables
    :param kwargs: Params that will be passthrough to tf.import_graph_def
    :return: The loaded graph
    """
    try:
        # We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a the given graph (or the default)
        graph = graph if graph is not None else tf.Graph()
        with graph.as_default():
            # The name var will prefix every op/nodes in your graph
            tf.import_graph_def(graph_def, name=prefix, **kwargs)
        return graph
    except tf.errors.NotFoundError as e:
        raise IOError(e.message)
    except Exception as e:
        raise IOError('Could not decode graph. {}'.format(e.message))
