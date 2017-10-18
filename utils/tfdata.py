from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pickle
import platform
import six
from six.moves import urllib
import time
import tarfile

from . import console
from .. import pedia


class TFData(object):
  """
  Only valid data container for tframe
  """
  # region : Constructor

  def __init__(self, features, targets=None, name=None, **kwargs):
    """
    Pack data into an instance of TFData
    :param features: features in numpy array
    :param targets: targets or labels, must be provided in supervised learning
    :param name: name for dataset, i.e. 'MNIST'
    :param kwargs: Other data to be pack into tha data dict
    
    Examples:
    (1) images, labels = get_data()
        data = TFDate(images, targets=labels)
        display(data[features])
    
    """
    if not hasattr(features, 'shape'):
      raise ValueError('features must have attribute "shape"')

    self.name = name

    self._data = {pedia.features: features}
    self.attachment = {}
    if targets is not None:
      if targets.shape[0] != features.shape[0]:
        raise ValueError(
          "targets number {} doesn't match feature number".format(
            targets.shape[0], features.shape[0]))
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

  # endregion : Constructor

  # region : Properties

  @property
  def scalar_labels(self):
    if self.targets is None:
      return False
    # Here self.targets is not None
    target_shape = self.targets.shape
    shape_len = len(target_shape)
    if shape_len == 1 or shape_len == 2 and target_shape[1] == 1:
      return self.targets
    elif shape_len > 2:
      return None
    # At this point targets may be one-hot
    return np.argmax(self.targets, axis=1)

  @property
  def feature_is_image(self):
    feature_shape = self.features.shape
    flag = False
    if len(feature_shape) == 4:
      # Channel last images are supported only
      flag = feature_shape[-1] in [1, 3]
    elif len(feature_shape) == 3:
      flag = feature_shape[1] > 1 and feature_shape[2] > 1

    return flag

  @property
  def sample_num(self):
    return self.features.shape[0]

  @property
  def cursor(self):
    return self._cursor

  @property
  def features(self):
    return self._data[pedia.features]

  @property
  def targets(self):
    return (None if pedia.targets not in self._data.keys()
            else self._data[pedia.targets])

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

  # TODO: ...
  # def __getattr__(self, attrname):
  #   if attrname in self._data.keys():
  #     return self._data[attrname]
  #   elif attrname in self.attachment.keys():
  #     return self.attachment[attrname]
  #   else:
  #     return object.__getattribute__(self, attrname)

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

  # endregion : Properties

  # region : Public Methods

  def set_cursor(self, position):
    if not 0 <= position < self.sample_num:
      raise ValueError('Invalid position for cursor')
    self._cursor = position

  def move_cursor(self, step):
    assert step in [-1, 1]
    self._cursor += step
    if self._cursor < 0:
      self._cursor += self.sample_num
    elif self._cursor >= self.sample_num:
      self._cursor -= self.sample_num

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

  def save(self, filename):
    with open(filename, 'wb') as output:
      pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

  @staticmethod
  def load(filename):
    with open(filename, 'rb') as input_:
      return pickle.load(input_)

  # endregion : Public Methods

  pass


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
  data[pedia.training] = TFData(mnist.train.images, targets=mnist.train.labels)
  data[pedia.validation] = TFData(mnist.validation.images,
                              targets=mnist.validation.labels)
  data[pedia.test] = TFData(mnist.test.images, targets=mnist.test.labels)

  return data


def load_cifar10(data_dir, flatten=False, one_hot=False, validation_size=0):
  console.show_status('Loading CIFAR-10 ...')

  # region : Download, tar data

  # Check data directory
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  # Get data file name and path
  DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(data_dir, filename)
  # If data does not exists, download from Alex's website
  if not os.path.exists(filepath):
    console.show_status('Downloading ...')
    start_time = time.time()
    def _progress(count, block_size, total_size):
      console.clear_line()
      console.print_progress(count*block_size, total_size, start_time)
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    statinfo = os.stat(filepath)
    console.show_status('Successfully downloaded {} {} bytes.'.format(
      filename, statinfo.st_size))

  # Tar file
  tarfile.open(filepath, 'r:gz').extractall(data_dir)
  # Update data directory
  data_dir = os.path.join(data_dir, 'cifar-10-batches-py')

  # endregion : Download, tar data

  # Define functions
  def pickle_load(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
      return pickle.load(f)
    elif version[0] == '3':
      return pickle.load(f, encoding='latin1')
    raise ValueError('Invalid python version: {}'.format(version))

  def load_batch(filename):
    with open(filename, 'rb') as f:
      datadict = pickle_load(f)
      X = datadict['data']
      X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
      Y = np.array(datadict['labels'])
      if flatten:
        X = X.reshape(10000, -1)
      if one_hot:
        def dense_to_one_hot(labels_dense, num_classes):
          """Convert class labels from scalars to one-hot vectors."""
          num_labels = labels_dense.shape[0]
          index_offset = np.arange(num_labels) * num_classes
          labels_one_hot = np.zeros((num_labels, num_classes))
          labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
          return labels_one_hot
        Y = dense_to_one_hot(Y, 10)

      return X, Y

  # Load data from files
  xs = []
  ys = []
  for b in range(1, 6):
    f = os.path.join(data_dir,  'data_batch_{}'.format(b))
    X, Y = load_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_batch(os.path.join(data_dir, 'test_batch'))

  # Pack data into instances of TFData and form a data dict
  data = {}
  total = 50000
  training_size = total - validation_size

  mask = list(range(training_size))
  data[pedia.training] = TFData(Xtr[mask], targets=Ytr[mask])
  mask = list(range(training_size, total))
  data[pedia.validation] = TFData(Xtr[mask], targets=Ytr[mask])
  data[pedia.test] = TFData(Xte, targets=Yte)

  console.show_status('CIFAR-10 loaded')
  console.supplement('Training Set:')
  console.supplement('images: {}'.format(
    data[pedia.training][pedia.features].shape), 2)
  console.supplement('labels: {}'.format(
    data[pedia.training][pedia.targets].shape), 2)
  console.supplement('Validation Set:')
  console.supplement('images: {}'.format(
    data[pedia.validation][pedia.features].shape), 2)
  console.supplement('labels: {}'.format(
    data[pedia.validation][pedia.targets].shape), 2)
  console.supplement('Test Set:')
  console.supplement('images: {}'.format(
    data[pedia.test][pedia.features].shape), 2)
  console.supplement('labels: {}'.format(
    data[pedia.test][pedia.targets].shape), 2)

  return data




