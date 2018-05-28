from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tarfile
import platform
import pickle

import tframe.utils.misc as misc

from tframe import console
from tframe.data.dataset import DataSet
from tframe.data.base_classes import ImageDataAgent


class CIFAR10(ImageDataAgent):
  """"""
  DATA_NAME = 'CIFAR-10'
  DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  TFD_FILE_NAME = 'cifar-10.tfd'

  PROPERTIES = {
    'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
                'horse', 'ship', 'truck']
  }

  @classmethod
  def load(cls, data_dir, train_size=40000, validate_size=10000,
           test_size=10000, flatten=False, one_hot=True):
    return super().load(
      data_dir, train_size, validate_size, test_size, flatten, one_hot)

  @classmethod
  def load_as_tframe_data(cls, data_dir):
    file_path = os.path.join(data_dir, cls.TFD_FILE_NAME)
    if os.path.exists(file_path): return DataSet.load(file_path)
    # If .tfd file does not exist, try to convert from raw data
    console.show_status('Trying to convert raw data to tfr DataSet ...')
    features, targets = cls.load_as_numpy_arrays(data_dir)
    data_set = DataSet(features, targets, name=cls.DATA_NAME, **cls.PROPERTIES)
    console.show_status('Successfully converted {} samples'.format(
      data_set.size))
    # Save DataSet
    data_set.save(file_path)
    return data_set

  @classmethod
  def load_as_numpy_arrays(cls, data_dir):
    # Make sure tar.gz file is ready
    file_path = cls._check_raw_data(data_dir)
    # Extract file
    tarfile.open(file_path, 'r:gz').extractall(data_dir)
    # Update data directory
    data_dir = os.path.join(data_dir, 'cifar-10-batches-py')

    # Load data from files
    xs, ys = [], []
    # - Load train set
    for b in range(1, 6):
      f = os.path.join(data_dir, 'data_batch_{}'.format(b))
      X, Y = cls._load_batch(f)
      xs.append(X)
      ys.append(Y)
    # - Load test set
    X, Y = cls._load_batch(os.path.join(data_dir, 'test_batch'))
    xs.append(X)
    ys.append(Y)

    # Merge all data together
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    return X, Y

  # region : Private Methods

  @staticmethod
  def _load_batch(file_name):
    """Load 10000 samples of shape [32. 32, 3] with dense labels"""
    def pickle_load(f):
      version = platform.python_version_tuple()
      if version[0] == '2': return pickle.load(f)
      elif version[0] == '3': return pickle.load(f, encoding='latin1')
      raise ValueError('!! Invalid python version: {}'.format(version))
    with open(file_name, 'rb') as f:
      data_dict = pickle_load(f)
      X = data_dict['data']
      X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
      Y = np.array(data_dict['labels'])
      return X, Y

  # endregion : Private Methods


if __name__ == '__main__':
  from tframe.data.images.image_viewer import ImageViewer
  data_dir = '../../examples/cifar-10/data'
  data_set = CIFAR10.load_as_tframe_data(data_dir)
  viewer = ImageViewer(data_set)
  viewer.show()


