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


_span_width = 10
_noise_symbols_num = 4
_symbol_kinds = 2 + _noise_symbols_num

def engine(L, N, fixed_length=True):
  # Check inputs
  assert isinstance(N, int) and N > 1
  assert isinstance(L, int) and L >= _span_width * N
  # Decide sequence length
  if not fixed_length:
    L_min, L_max = L, int(np.round(L * 1.1))
    L = np.random.randint(L_min, L_max + 1)
  # Generate features
  # ... 0-X, 1-Y, 2-a, 3-b, 4-c, 5-d, ...
  sequence = np.random.randint(2, 2 + _noise_symbols_num, size=[L])
  # ... insert N key symbols
  M = int(np.round(L / N))
  alpha, target = 1, 0
  for k in range(N):
    index = np.random.randint(k * M, k * M + _span_width)
    symbol = np.random.randint(0, 2)
    target += symbol * alpha
    sequence[index] = symbol
    alpha *= 2
  # Set trigger symbols  # not necessary
  # sequence[0], sequence[-1] = 6, 7
  # Convert sequence and target to one-hot
  onehot_sequence = np.eye(_symbol_kinds)[sequence]
  onehot_target = np.eye(alpha)[target]
  return onehot_sequence, onehot_target


class TO(DataAgent):
  """Temporal Order Problem. Reference: LSTM97
  """

  DATA_NAME = 'TemporalOrder'

  @classmethod
  def engine(cls, L, N, fixed_length):
    def _engine(size):
      return cls._synthesize(size, L, N, fixed_length)
    return _engine

  @classmethod
  def load(cls, data_dir, val_size=200, test_size=1000, L=100, N=3,
           fixed_length=True, **kwargs):
    # Load train set
    train_set = PerpetualMachine('TOPM', cls.engine(L, N, fixed_length))
    # Load test set
    val_set = cls.load_as_tframe_data(
      data_dir, val_size, L, N, fixed_length, prefix='val_')
    test_set = cls.load_as_tframe_data(
      data_dir, test_size, L, N, fixed_length, prefix='test_')
    # Return
    return train_set, val_set, test_set

  @classmethod
  def load_as_tframe_data(
      cls, data_dir, size=1000, L=100, N=3, fixed_length=True, file_name=None,
      prefix='', **kwargs):
    # Check file_name
    if file_name is None:
      file_name = cls._get_file_name(size, L, N, fixed_length)
      file_name = prefix + file_name + '.tfds'
    data_path = os.path.join(data_dir, file_name)
    if os.path.exists(data_path): return SequenceSet.load(data_path)
    # If data does not exist, create a new data set
    console.show_status('Creating data ...')
    data_set = cls._synthesize(size, L, N, fixed_length, verbose=True)
    console.show_status('Saving data set ...')
    data_set.save(data_path)
    console.show_status('Data set saved to `{}`'.format(data_path))
    return data_set

  @classmethod
  def _synthesize(cls, size, L, N, fixed_length, verbose=False):
    features, targets = [], []
    for i in range(size):
      x, y = engine(L, N, fixed_length)
      features.append(x)
      targets.append(y)
      if verbose:
        console.clear_line()
        console.print_progress(i + 1, size)
    # Wrap data into a SequenceSet
    data_set = SequenceSet(
      features, summ_dict={'targets': targets}, n_to_one=True,
      name='TemporalOrder')
    return data_set

  @classmethod
  def _get_file_name(cls, size, L, N, fixed_length):
    # `fL` for fixed length and `vL` for variant length
    checker.check_positive_integer(size)
    checker.check_positive_integer(L)
    checker.check_positive_integer(N)
    file_name = 'TO_{}_L{}_N{}_{}'.format(
      size, L, N, 'fL' if fixed_length else 'vL')
    return file_name


if __name__ == '__main__':
  print()
