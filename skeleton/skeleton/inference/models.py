import cv2
import numpy as np
import pkg_resources
import tensorflow as tf

from skeleton import constants
from .tools import load_graph


class AlexNet:

    def __init__(self):
        self._model_path = pkg_resources.resource_filename(__name__, u'optimized_frozen_model.pb')
        self._graph = load_graph(self._model_path)
        self._sess = tf.Session(graph=self._graph)

        self._image_placeholder = self._graph.get_tensor_by_name('input_pipeline/IteratorGetNext:0')
        self._prediction = self._graph.get_tensor_by_name('AlexNet/fc3/dense/BiasAdd:0')

    def predict(self, image):
        image = np.asarray(image)
        normalized_image = np.subtract(image, constants.IMAGENET_MEAN)
        resized_image = cv2.resize(normalized_image, dsize=(227, 227))
        resized_image = np.expand_dims(resized_image, axis=0)  # Fake batch dimension
        result = self._sess.run(self._prediction, feed_dict={self._image_placeholder: resized_image})
        return np.squeeze(result)  # Remove batch dim
