from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from tframe import console
from tframe.utils.local import check_path
from tframe.utils.misc import convert_to_one_hot

from tframe.data.sequences.seq_set import SequenceSet
from tframe.data.base_classes import DataAgent


class IMDB(DataAgent):
  """The IMDB movie review data set.

     References
     [1] Maas, Andrew L. etc. Learning Word Vectors for Sentiment Analysis. 2011
     [2] MGU16
  """

  DATA_NAME = 'IMDB'

  @classmethod
  def load(cls, data_dir, train_size=15000, val_size=10000, test_size=25000,
           num_words=10000, max_len=None, **kwargs):
    data_set = cls.load_as_tframe_data(data_dir, num_words=num_words)
    if max_len is not None:
      data_set.features = [s[:max_len] for s in data_set.features]
    return data_set.split(
      train_size, val_size, test_size,
      names=('train_set', 'val_set', 'test_set'))

  @classmethod
  def load_as_tframe_data(cls, data_dir, num_words=10000, **kwargs):
    # Load directly if data set exists
    data_path = cls._get_data_path(data_dir, num_words)
    if os.path.exists(data_path): return SequenceSet.load(data_path)
    # If data does not exist, create from raw data
    console.show_status('Creating data sets ...')
    (train_data, train_labels), (test_data, test_labels) = cls._load_raw_data(
      data_dir, num_words=num_words)
    data_list = list(train_data) + list(test_data)
    features = [np.array(cmt).reshape([-1, 1]) for cmt in data_list]

    targets = list(np.concatenate((train_labels, test_labels)))

    data_set = SequenceSet(features, summ_dict={'targets': targets},
                           n_to_one=True, name='IMDB')
    console.show_status('Saving data set ...')
    data_set.save(data_path)
    console.show_status('Data set saved to `{}`'.format(data_path))
    return data_set

  # region : Private Methods

  @classmethod
  def _load_raw_data(cls, data_dir, num_words=10000):
    from tensorflow import keras
    imdb = keras.datasets.imdb
    data_path = os.path.join(check_path(data_dir), 'imdb.npz')
    return imdb.load_data(data_path, num_words=num_words)

  @classmethod
  def _get_data_path(cls, data_dir, num_words):
    assert isinstance(num_words, int) and num_words > 0
    return os.path.join(data_dir, 'IMDB_{}.tfds'.format(num_words))

  # endregion : Private Methods


