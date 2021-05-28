from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gzip
import numpy as np
from collections import OrderedDict

from tframe import tf

from tframe import console, context, hub
from tframe import pedia
from tframe.data.base_classes import ImageDataAgent


class MNIST(ImageDataAgent):
  """THE MNIST DATABASE of handwritten digits"""

  DATA_NAME = 'MNIST'
  DATA_URL = 'http://yann.lecun.com/exdb/mnist/'
  TFD_FILE_NAME = 'mnist.tfd'

  PROPERTIES = {pedia.num_classes: 10}

  @classmethod
  def load(cls, data_dir, train_size=-1, validate_size=5000, test_size=10000,
           flatten=False, one_hot=True, **kwargs):
    return super().load(
      data_dir, train_size, validate_size, test_size, flatten, one_hot)

  @classmethod
  def load_as_numpy_arrays(cls, data_dir):
    """Load 70000 samples (60000 training samples and 10000 test samples)
       of shape [28, 28, 1] with dense labels"""
    # gz file names
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    # Check gs files
    for file_name in (TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS):
      cls._check_raw_data(data_dir, file_name, cls.DATA_URL + file_name)
    # Extract images and labels
    get_file_path = lambda fn: os.path.join(data_dir, fn)
    # 60000 training samples
    train_images = cls._extract_images(get_file_path(TRAIN_IMAGES))
    train_labels = cls._extract_labels(get_file_path(TRAIN_LABELS))
    # 10000 test samples
    test_images = cls._extract_images(get_file_path(TEST_IMAGES))
    test_lables = cls._extract_labels(get_file_path(TEST_LABELS))
    # Merge data into regular numpy arrays
    images = np.concatenate((train_images, test_images))
    labels = np.concatenate((train_labels, test_lables))
    # Return data tuple
    return images, labels

  # region : Private Methods

  @classmethod
  def _read32(cls, bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

  @classmethod
  def _extract_images(cls, file_path):
    """Returns MNIST images of shape [28, 28, 1] """
    with open(file_path, 'rb') as f:
      console.show_status('Extracting {} ...'.format(f.name))
      with gzip.GzipFile(fileobj=f) as bytestream:
        magic = cls._read32(bytestream)
        if magic != 2051: raise ValueError(
          '!! Invalid magic number {} in MNIST image file: {}'.format(
            magic, f.name))
        num_images = cls._read32(bytestream)
        rows = cls._read32(bytestream)
        cols = cls._read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data

  @classmethod
  def _extract_labels(cls, file_path):
    """Returns MNIST dense labels"""
    with open(file_path, 'rb') as f:
      console.show_status('Extracting {} ...'.format(f.name))
      with gzip.GzipFile(fileobj=f) as bytestream:
        magic = cls._read32(bytestream)
        if magic != 2049: raise ValueError(
          '!! Invalid magic number {} in MNIST label file: {}'.format(
            magic, f.name))
        num_items = cls._read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

  @staticmethod
  def connection_heat_map_extractor(_):
    """This method will be called in net.variable_extractor during linking
       if has been registered to model(a Net)
    """
    if not hub.export_input_connection: return
    weights = context.weights_list[0]
    assert isinstance(weights, (tf.Variable, tf.Tensor))
    assert weights.shape.as_list()[0] == 784
    weights = tf.abs(weights)
    heat_map = tf.reshape(tf.reduce_sum(weights, axis=1), [28, 28])
    context.variables_to_export['connection_heat_map'] = heat_map

  @staticmethod
  def flatten_researcher(grads):
    """This method will be used in Monitor.grad_dict property when trainer is
       taking down tensors to export if has been registered to monitor
    """
    assert isinstance(grads, OrderedDict)
    key = list(grads.keys())[0]
    if grads[key].shape[0] != 784: return
    grad = grads.get(key)
    assert isinstance(grad, np.ndarray)
    grad = np.sum(grad, axis=1).reshape([28, 28])
    grads['|grad(inputs)|'] = grad

  # endregion : Private Methods


if __name__ == '__main__':
  from tframe.data.images.image_viewer import ImageViewer
  data_dir = '../../examples/00-MNIST/data'
  data_set = MNIST.load_as_tframe_data(data_dir)
  viewer = ImageViewer(data_set)
  viewer.show()


