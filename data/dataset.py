from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe import checker
from tframe import pedia
from tframe import hub

from tframe.data.base_classes import TFRData
# from tframe.data.sequences.paral_engine import ParallelEngine


class DataSet(TFRData):

  EXTENSION = 'tfd'

  FEATURES = pedia.features
  TARGETS = pedia.targets

  def __init__(self, features=None, targets=None, data_dict=None,
               name='dataset', is_rnn_input=False, **kwargs):
    """
    A DataSet is the only data structure which can be fed into tframe model
    directly. Data stored in data_dict must be a regular numpy array with the
    same length.
    """
    # Call parent's constructor
    super().__init__(name)

    # Attributes
    self.data_dict = {} if data_dict is None else data_dict
    self.features = features
    self.targets = targets
    self.properties.update(kwargs)

    self.is_rnn_input = is_rnn_input
    self.should_reset_state = False
    self.reset_batch_indices = None
    self.reset_values = None

    # Sanity checks
    self._check_data()


  # region : Properties
  
  @property
  def features(self): return self.data_dict.get(self.FEATURES, None)

  @features.setter
  def features(self, val):
    if val is not None:
      if isinstance(val, np.ndarray): self.data_dict[self.FEATURES] = val
      else: raise TypeError('!! Unsupported feature type {}'.format(type(val)))

  @property
  def targets(self): return self.data_dict.get(self.TARGETS, None)

  @targets.setter
  def targets(self, val):
    if val is not None:
      if isinstance(val, np.ndarray): self.data_dict[self.TARGETS] = val
      else: raise TypeError('!! Unsupported target type {}'.format(type(val)))

  @property
  def representative(self):
    array = list(self.data_dict.values())[0]
    assert isinstance(array, np.ndarray)
    assert len(array.shape) > 2 if self.is_rnn_input else 1
    return array

  @property
  def should_partially_reset_state(self):
    return self.reset_batch_indices is not None

  @property
  def structure(self): return [1]

  @property
  def size(self): return len(self.representative)

  @property
  def total_steps(self):
    assert self.is_rnn_input
    return self.representative.shape[1]

  @property
  def is_regular_array(self): return True

  @property
  def stack(self): return self

  @property
  def as_rnn_batch(self):
    """Convert a regular array to RNN batch format"""
    if self.is_rnn_input: return self
    return self._convert_to_rnn_input()

  # endregion : Properties

  # region : Overriden Methods

  def __len__(self): return self.size

  def __getitem__(self, item):
    if isinstance(item, str):
      if item in self.data_dict.keys(): return self.data_dict[item]
      elif item in self.properties.keys(): return self.properties[item]
      else: raise KeyError('!! Can not resolve "{}"'.format(item))

    # If item is index array
    f = lambda x: x[item]
    data_set = DataSet(data_dict=self._apply(f), name=self.name + '(slice)')
    return self._finalize(data_set, item)

  # endregion : Overriden Methods

  # region : Basic APIs

  def get_round_length(self, batch_size, num_steps=None):
    assert isinstance(batch_size, int)
    if batch_size < 0: batch_size = self.size

    if num_steps is None: round_len = np.ceil(self.size / batch_size)
    else:
      if self.is_rnn_input:
        if num_steps < 0: num_steps = self.total_steps
        round_len = np.ceil(self.total_steps / num_steps)
      else:
        if num_steps < 0: round_len = 1
        else: round_len = np.ceil(self.size // batch_size / num_steps)

    return int(round_len)

  def gen_batches(self, batch_size, shuffle=False):
    """Yield batches of data with the specific size"""
    round_len = self.get_round_length(batch_size)
    if batch_size == -1: batch_size = self.size

    # Generate batches
    for i in range(round_len):
      indices = self._select(i, batch_size, shuffle)
      # Yield data batch
      data_batch = self.stack[indices]
      # batch_preprocessor should not change self.size
      if self.batch_preprocessor is not None:
        data_batch = self.batch_preprocessor(data_batch)
      yield data_batch

  def gen_rnn_batches(self, batch_size=1, num_steps=-1, shuffle=False):
    """Each numpy array will be regarded as a single sequence and will be
       partitioned into batches of sequences with corresponding steps.

      :param batch_size: Batch size, positive integer
      :param num_steps: steps of each RNN data batch
      :param shuffle: Whether to shuffle the partitioned sequences
     """
    # Check batch_size and shuffle
    checker.check_positive_integer(batch_size)
    assert shuffle is False
    # Generate partitioned data set
    if self.is_rnn_input: rnn_data = self
    else:
      data_set = (self if self.batch_preprocessor is None
                  else self.batch_preprocessor(self))
      rnn_data = data_set._convert_to_rnn_input(batch_size)

    round_len = self.get_round_length(batch_size, num_steps)
    if num_steps < 0: num_steps = rnn_data.total_steps

    # Generate batches
    for i in range(round_len):
      f = lambda x: x[:, i*num_steps:(i + 1)*num_steps]
      batch = DataSet(
        data_dict=rnn_data._apply(f), is_rnn_input=True,
        name=self.name + '_batch_{}_of_{}'.format(i + 1, round_len),
        **self.properties)
      if i == 0: batch.should_reset_state = True
      yield batch

  # endregion : Basic APIs

  # region : Public Methods

  def split(self, *sizes, names=None):
    # Sanity check
    if len(sizes) == 0: raise ValueError('!! split sizes not specified')
    elif len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
      sizes = sizes[0]
    if names is not None:
      if not isinstance(names, (tuple, list)):
        raise TypeError('!! names must be a tuple or list of strings')
      if len(names) != len(sizes):
        raise ValueError('!! length of name list and sizes list does not match')
    # Check sizes
    sizes, auto_index, total_size = list(sizes), -1, 0
    for i, size in enumerate(sizes):
      if size is None or size < 0:
        if auto_index < 0:
          auto_index = i
          continue
        else: raise ValueError(
          '!! only one split size can be calculated automatically')
      if not isinstance(size, int) and size < 0:
        raise ValueError('!! size must be a non-negative integer')
      total_size += size
    # Calculate size automatically if necessary
    if auto_index >= 0:
      sizes[auto_index] = self.size - total_size
      if sizes[auto_index] <= 0: raise ValueError(
        '!! negative value appears when calculating size automatically')
    elif total_size != self.size: raise ValueError(
      '!! total size does not match size of the data set to split')
    # Split data set
    data_sets, cursor = (), 0
    for i, size in enumerate(sizes):
      if size == 0: continue
      indices = slice(cursor, cursor + size)
      data_set = self[indices]
      if names is not None: data_set.name = names[i]
      data_sets += (data_set,)
      cursor += size

    return data_sets

  # endregion : Public Methods

  # region : Private Methods

  def _finalize(self, data_set, indices=None):
    assert isinstance(data_set, DataSet)
    data_set.__class__ = self.__class__
    data_set.properties = self.properties
    if indices is not None:
      for k, v in self.properties.items():
        if isinstance(v, tuple) and len(v) == self.size:
          data_set.properties[k] = v[indices]
    return data_set

  def _select(self, batch_index, batch_size, shuffle, upper_bound=None):
    if upper_bound is None: upper_bound = self.size
    assert isinstance(batch_index, int) and batch_index >= 0
    checker.check_positive_integer(batch_size)
    if shuffle:
      indices = self._rand_indices(upper_bound=upper_bound, size=batch_size)
    else:
      from_index = batch_index * batch_size
      to_index = min((batch_index + 1) * batch_size, upper_bound)
      indices = list(range(from_index, to_index))
    return indices

  def _check_data(self):
    """data_dict should be a non-empty dictionary containing regular numpy
       arrays with the same length"""
    # Make sure data_dict is a non-empty dictionary
    if not isinstance(self.data_dict, dict) or len(self.data_dict) == 0:
      raise TypeError('!! data_dict must be a non-empty dictionary')

    data_length = len(list(self.data_dict.values())[0])

    # Check each item in data_dict
    for name, array in self.data_dict.items():
      # Check type and length
      if not isinstance(array, np.ndarray) or len(array) != data_length:
        raise ValueError('!! {} should be a numpy array with length {}'.format(
          name, data_length))
      # Check sample shape
      if len(array.shape) == 1:
        self.data_dict[name] = np.reshape(array, (-1, 1))

  def _apply(self, f, data_dict=None):
    """Apply callable method f to all data in data_dict. If data_dict is not
       provided, self.data_dict will be used as default"""
    assert callable(f)
    if data_dict is None: data_dict = self.data_dict
    result_dict = {}
    for k, v in data_dict.items(): result_dict[k] = f(v)
    return result_dict

  def _convert_to_rnn_input(self, batch_size=1):
    checker.check_positive_integer(batch_size)
    def f(array):
      assert isinstance(array, np.ndarray) and len(array.shape) > 1
      L = len(array) // batch_size
      data = np.zeros(shape=(batch_size, L, *array.shape[1:]))
      for i in range(batch_size): data[i] = array[i * L:(i + 1) * L]
      return data
    return DataSet(data_dict=self._apply(f), is_rnn_input=True,
                   name=self.name, **self.properties)

  def _rand_indices(self, upper_bound=None, size=1):
    if upper_bound is None: upper_bound = self.size
    assert self.features is not None
    if not hub.rand_over_classes:
      indices = np.random.randint(upper_bound, size=size)
    else:
      classes = np.random.randint(self.num_classes, size=size)
      indices = []
      for cls in classes:
        group_index = np.random.randint(len(self.groups[cls]))
        indices.append(self.groups[cls][group_index])

    if len(indices) == 1: return int(indices[0])
    else: return indices

  # endregion : Private Methods


if __name__ == '__main__':
  features = np.arange(12)
  data_set = DataSet(features)


