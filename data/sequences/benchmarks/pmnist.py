from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import random

from tframe import tf

from tframe import checker
from tframe import console
from tframe import context
from tframe.utils.misc import convert_to_one_hot

from tframe.data.images.mnist import MNIST
from tframe.data.sequences.seq_set import SequenceSet
from tframe.data.base_classes import DataAgent


class pMNIST(DataAgent):
  """(Permuted) MNIST"""

  DATA_NAME = 'pMNIST'

  @classmethod
  def load(cls, data_dir, train_size=55000, val_size=5000, test_size=10000,
           permute=False, permute_mark='alpha', file_name=None, **kwargs):
    data_set = cls.load_as_tframe_data(
      data_dir, file_name=file_name, permute=permute, permute_mark=permute_mark)
    if val_size > 0:
      train_set, val_set, test_set = data_set.split(
        train_size, val_size, test_size,
        names=('train_set', 'val_set', 'test_set'))
      return train_set, val_set, test_set
    else:
      train_set, test_set = data_set.split(
        train_size, test_size, names=('train_set', 'test_set'))
      return train_set, test_set

  @classmethod
  def load_as_tframe_data(cls, data_dir, file_name=None, permute=False,
                          permute_mark='alpha', **kwargs):
    # Check file name
    if file_name is None:
      file_name = cls._get_file_name(permute, permute_mark) + '.tfds'
    data_path = os.path.join(data_dir, file_name)
    if os.path.exists(data_path): return SequenceSet.load(data_path)
    # If data does not exist, create a new data set
    console.show_status('Creating data ...')
    images, labels = MNIST.load_as_numpy_arrays(data_dir)
    # images (70000, 784, 1), np.float64
    images = images.reshape(images.shape[0], -1, 1) / 255.
    # permute images if necessary
    if permute:
      images = np.swapaxes(images, 0, 1)
      images = np.random.permutation(images)
      images = np.swapaxes(images, 0, 1)
    # labels (70000, 10), np.float64
    labels = convert_to_one_hot(labels, 10)
    # Wrap data into a Sequence Set
    features = [image for image in images]
    targets = [label for label in labels]
    data_set = SequenceSet(features, summ_dict={'targets': targets},
                           n_to_one=True, name='pMNIST')
    console.show_status('Saving data set ...')
    data_set.save(data_path)
    console.show_status('Data set saved to `{}`'.format(data_path))
    return data_set

  @classmethod
  def _get_file_name(cls, permute, mark='alpha'):
    file_name = 'MNIST'
    if permute: file_name = 'p' + file_name + '_' + mark
    return file_name


