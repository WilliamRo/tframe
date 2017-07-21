from __future__ import absolute_import

import numpy as np
import six

from . import console
from .. import pedia


class TFData(object):
  """"""
  def __init__(self, features, targets=None, name=None, **kwargs):
    """"""
    if not hasattr(features, 'shape'):
      raise ValueError('features must have attribute "shape"')

    self.name = name

    self._data = {pedia.features: features}
    self.attachment = {}
    if targets is not None:
      kwargs[pedia.targets] = targets

    for k in kwargs.keys():
      if hasattr(kwargs[k], 'shape') and kwargs[k].shape[0] == self.sample_num:
        self._data[k] = kwargs[k]
      else:
        self.attachment[k] = kwargs[k]

    # Check data shape
    for k in self._data:
      shape = self._data[k].shape
      if len(shape) == 1:
        self._data[k] = self._data[k].reshape((shape[0], 1))

    self._batch_size = None
    self._cursor = 0

  @property
  def sample_num(self):
    return self.features.shape[0]

  @property
  def features(self):
    return self._data[pedia.features]

  @property
  def feature_shape(self):
    return self.features[0].shape

  @property
  def batches_per_epoch(self):
    return 1.0 * self.sample_num / self._batch_size

  @property
  def progress(self):
    cursor = self._cursor - 1
    if cursor <= self._batch_size:
      return 1.0
    else:
      return 1.0 * cursor / self.sample_num

  def __getattr__(self, attrname):
    if attrname in self._data.keys():
      return self._data[attrname]
    elif attrname in self.attachment.keys():
      return self.attachment[attrname]
    else:
      return self.__dict__[attrname]

  def __getitem__(self, item):
    if isinstance(item, six.string_types):
      if item in self._data.keys():
        return self._data[item]
      elif item in self.attachment.keys():
        return self.attachment[item]
      else:
        raise ValueError('Can not resolve "{}"'.format(item))
    # item is an array
    item = np.mod(item, self.sample_num)
    data = {}
    for k in self._data.keys():
      data[k] = self._data[k][item]

    return data

  def set_batch_size(self, batch_size):
    if not isinstance(batch_size, int) or batch_size < 1:
      raise TypeError('batch size must be a positive integer')
    self._batch_size = batch_size

  def next_batch(self, batch_size=None, shuffle=False):
    if batch_size is None:
      batch_size = self._batch_size
    else:
      self.set_batch_size(batch_size)

    if batch_size is None:
      raise ValueError('batch size not specified')

    indices = (np.random.randint(self.sample_num, size=(batch_size, ))
               if shuffle else range(self._cursor, self._cursor + batch_size))

    self._cursor += batch_size
    end_epoch = self._cursor >= self.sample_num
    if end_epoch:
      self._cursor -= self.sample_num

    return self[indices], end_epoch


def load_mnist(data_dir, flatten=False, one_hot=False,
               validation_size=5000):
  console.show_status('Loading MNIST ...')
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets(data_dir, one_hot=one_hot, reshape=flatten,
                                    validation_size=validation_size)
  console.show_status('MNIST loaded')
  console.supplement('Training Set:')
  console.supplement('images: {}'.format(mnist.train.images.shape), 2)
  console.supplement('labels: {}'.format(mnist.train.labels.shape), 2)
  console.supplement('Validation Set:')
  console.supplement('images: {}'.format(mnist.validation.images.shape), 2)
  console.supplement('labels: {}'.format(mnist.validation.labels.shape), 2)
  console.supplement('Test Set:')
  console.supplement('images: {}'.format(mnist.test.images.shape), 2)
  console.supplement('labels: {}'.format(mnist.test.labels.shape), 2)

  data = {}
  data['train'] = TFData(mnist.train.images, targets=mnist.train.labels)
  data['validation'] = TFData(mnist.validation.images,
                              targets=mnist.validation.labels)
  data['test'] = TFData(mnist.test.images, targets=mnist.test.labels)

  return data


