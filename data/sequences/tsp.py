from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from collections import OrderedDict

from tframe import checker
from tframe import console
from tframe.data.base_classes import DataAgent
from tframe.data.sequences.seq_set import SequenceSet
from tframe.data.perpetual_machine import PerpetualMachine


def engine(number, N=3, T_or_interval=100, mu=0., sigma2=0.2, add_noise=False):
  # Check input
  if isinstance(T_or_interval, int):
    L_min = T_or_interval
    L_max = int(np.round(L_min * 1.1))
  else:
    checker.check_type(T_or_interval, int)
    L_min, L_max = T_or_interval
  checker.check_positive_integer(N)
  assert number in (1, -1) and 0 < N <= L_min <= L_max
  # Decide the length
  L = np.random.randint(L_min, L_max + 1)
  sequence = np.random.randn(L) * np.sqrt(sigma2) + mu
  sequence[:N] = number
  target = (number + 1.) / 2.
  if add_noise: target -= number * 0.2
  return sequence, np.array([target])


class TSP(DataAgent):
  """Two Sequence Problem"""
  DATA_NAME = 'TwoSequenceProblem'

  @classmethod
  def engine(cls, N, T, mu, var, noisy):
    def _engine(size):
      return cls._get_one_data_set(size, N, T, mu, var, noisy)
    return _engine

  @classmethod
  def load(cls, data_dir, validate_size=512, test_size=2560, N=3, T=100,
           mu=0., sigma2=0.2, add_noise=False, **kwargs):
    # Load train set
    train_set = PerpetualMachine(
      'TSPPM', cls.engine(N, T, mu, sigma2, add_noise))
    # Load validation set and test set
    load_as_tfd = lambda size, prefix: cls.load_as_tframe_data(
      data_dir, size, N=N, T=T, mu=mu, sigma2=sigma2, add_noise=add_noise,
      prefix=prefix)
    val_set = load_as_tfd(validate_size, 'val_')
    test_set = load_as_tfd(test_size, 'test_')
    return train_set, val_set, test_set

  @classmethod
  def load_as_tframe_data(cls, data_dir, size=2560, file_name=None, N=3, T=100,
                          mu=0., sigma2=0.2, add_noise=False, prefix=''):
    # Check file_name
    if file_name is None:
      file_name = cls._get_file_name(size, N, T, mu, sigma2, add_noise)
      file_name = prefix + file_name
    data_path = os.path.join(data_dir, file_name)
    if os.path.exists(data_path): return SequenceSet.load(data_path)
    # If data does not exist, create a new data set
    console.show_status('Creating data ...')
    data_set = cls._get_one_data_set(size, N, T, mu, sigma2, add_noise)
    console.show_status('Saving data set ...')
    data_set.save(data_path)
    console.show_status('Data set saved to `{}`'.format(data_path))
    return data_set

  @classmethod
  def _get_one_data_set(cls, size, N, T, mu, var, noisy):
    features, targets = [], []
    for _ in range(size):
      number = np.random.choice([-1, 1])
      x, y = engine(number, N, T, mu, var, noisy)
      features.append(x)
      targets.append(y)
    # Wrap data into a SequenceSet
    data_set = SequenceSet(
      features, summ_dict={'targets': targets}, n_to_one=True,
      name='Noisy Sequences' if noisy else 'Noise-free Sequences',
      N=N, T=T, mu=mu, var=var, noisy=noisy)
    return data_set

  @classmethod
  def _get_file_name(cls, size, N, T, mu, sigma2, add_noise):
    checker.check_positive_integer(N)
    checker.check_positive_integer(T)
    file_name = '{}_{}_N{}T{}_mu{}var{}_{}'.format(
      'TSP', size, N, T, mu, sigma2, 'noisy' if add_noise else 'noise-free')
    return file_name


if __name__ == '__main__':
  train_set, val_set, test_set = TSP.load('E:/tmp')

