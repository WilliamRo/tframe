from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tframe import checker
from tframe import console
from tframe.data.base_classes import DataAgent
from tframe.data.sequences.seq_set import SequenceSet
from tframe.data.perpetual_machine import PerpetualMachine


def engine(T, fixed_length):
  # Check inputs
  checker.check_positive_integer(T)
  checker.check_type(fixed_length, bool)
  # Decide sequence length
  if not fixed_length:
    T_min, T_max = T, int(np.round(T * 1.1))
    T = np.random.randint(T_min, T_max + 1)
  # Generate values
  values = np.random.rand(T)
  # Initialize indicators
  indicators = np.zeros_like(values)
  # Randomly select 2 indices
  mid = int(np.round(T / 2))
  k1 = np.random.randint(0, mid)
  k2 = np.random.randint(mid, T)
  indicators[k1] = 1.0
  indicators[k2] = 1.0

  sequence = np.stack([values, indicators], axis=1)
  # target = (values[k1] + values[k2]) / 2.
  target = values[k1] + values[k2]
  return sequence, np.array([target])


class AP(DataAgent):
  """Adding Problem. References: LSTM97, IRNN15, uRNN16.
     The implementation is mostly based on uRNN16"""

  DATA_NAME = 'AddingProblem'

  @classmethod
  def engine(cls, T, fixed_length):
    def _engine(size):
      return cls._synthesize(size, T, fixed_length)
    return _engine

  @classmethod
  def load(cls, data_dir, val_size=200, test_size=10000, T=150,
           fixed_length=True, **kwargs):
    # Load train set
    train_set = PerpetualMachine('APPM', cls.engine(T, fixed_length))
    # Load test set
    val_set = cls.load_as_tframe_data(
      data_dir, val_size, T, fixed_length, prefix='val_')
    test_set = cls.load_as_tframe_data(
      data_dir, test_size, T, fixed_length, prefix='test_')
    # Return
    return train_set, val_set, test_set

  @classmethod
  def load_as_tframe_data(cls, data_dir, size=10000, T=150, fixed_length=True,
                          file_name=None, prefix='', **kwargs):
    """In IRNN15: `..., we noticed that both LSTMs and RNNs started to
        struggle when T is around 150.`"""
    # Check file_name
    if file_name is None:
      file_name = cls._get_file_name(size, T, fixed_length)
      file_name = prefix + file_name + '.tfds'
    data_path = os.path.join(data_dir, file_name)
    if os.path.exists(data_path): return SequenceSet.load(data_path)
    # If data does not exist, create a new data set
    console.show_status('Creating data ...')
    data_set = cls._synthesize(size, T, fixed_length, verbose=True)
    console.show_status('Saving data set ...')
    data_set.save(data_path)
    console.show_status('Data set saved to `{}`'.format(data_path))
    return data_set

  @classmethod
  def _synthesize(cls, size, T, fixed_length, verbose=False):
    features, targets = [], []
    for i in range(size):
      x, y = engine(T, fixed_length)
      features.append(x)
      targets.append(y)
      if verbose:
        console.clear_line()
        console.print_progress(i + 1, size)
    # Wrap data into a SequenceSet
    data_set = SequenceSet(
      features, summ_dict={'targets': targets}, n_to_one=True,
      name='Adding Toys')
    return data_set

  @classmethod
  def _get_file_name(cls, test_size, T, fixed_length):
    checker.check_positive_integer(T)
    # `fL` for fixed length and `vL` for variant length
    file_name = 'AP_{}_T{}_{}'.format(
      test_size, T, 'fL' if fixed_length else 'vL')
    return file_name


if __name__ == '__main__':
  x, y = engine(20, True)
  print()


