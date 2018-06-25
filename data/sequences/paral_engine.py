import numpy as np

from tframe import checker
from tframe.data.dataset import DataSet


class ParallelEngine(object):
  def __init__(self, batch_size):
    self._data_sets = [None] * checker.check_positive_integer(batch_size)
    self._cursors = [0] * self.size

  # region : Properties

  @property
  def size(self): return len(self._data_sets)

  @property
  def remainders(self):
    return [len(d) - c for d, c in zip(self._data_sets, self._cursors)]

  @property
  def max_emit_length(self): return min(self.remainders)

  @property
  def is_ready(self): return self.next_inactive_index is None

  @property
  def inactive_indices(self):
    indices = np.argwhere(
      [d is None or len(d) - c == 0
       for d, c in zip(self._data_sets, self._cursors)]).ravel()
    return list(indices)

  @property
  def next_inactive_index(self):
    indices = self.inactive_indices
    if len(indices) == 0: return None
    else: return indices[0]

  @property
  def flameout(self): return self.size == 0

  # endregion : Properties

  # region : Private Methods

  def _set_data(self, index, data_set):
    if data_set is None:
      self._data_sets.pop(index)
      self._cursors.pop(index)
    else:
      assert isinstance(data_set, DataSet)
      self._data_sets[index] = data_set
      self._cursors[index] = 0

  # endregion : Private Methods

  # region : Public Methods

  def set_data(self, data_set):
    # Sanity check
    assert not self.is_ready
    # Find next position to replace sequence
    index = self.next_inactive_index
    # Substitution
    self._set_data(index, data_set)
    return index

  def emit(self, num_steps):
    assert self.is_ready
    assert isinstance(num_steps, int)
    if num_steps < 0: num_steps = self.max_emit_length

    # Determine steps
    steps = min(self.max_emit_length, num_steps)

    template = self._data_sets[0]
    assert isinstance(template, DataSet)
    data_dict = template.data_dict.copy()
    cursors = None
    for key, data in data_dict.items():
      assert isinstance(data, np.ndarray)
      sample_shape = data.shape[1:]
      data_dict[key] = np.zeros(shape=(self.size, steps, *sample_shape))

      cursors = self._cursors.copy()
      for i in range(self.size):
        array = self._data_sets[i][key]
        assert isinstance(array, np.ndarray)
        c = cursors[i]
        data_dict[key][i] = array[c:c + steps]
        # Move cursor
        assert 0 < c + steps <= len(array)
        cursors[i] += steps

    # Update cursors
    assert cursors is not None
    self._cursors = cursors
    # Wrap data into a DataSet and return
    return DataSet(data_dict=data_dict, is_rnn_input=True)

  @staticmethod
  def get_round_length(batch_size, num_steps, lengths, len_f=None):
    checker.check_type(lengths, int)
    pe = ParallelEngine(batch_size)
    round_len, cursor = 0, 0

    while True:
      # Set sequences and targets if necessary
      while not pe.is_ready:
        if cursor < len(lengths):
          length = lengths[cursor] if len_f is None else len_f(lengths[cursor])
          ds = DataSet(features=np.zeros(shape=(length, 1)))
          cursor += 1
        else: ds = None
        pe.set_data(ds)

      if pe.flameout: break
      pe.emit(num_steps)
      round_len += 1

    return round_len

  # endregion : Public Methods


if __name__ == '__main__':
  batch_size = 3
  num_steps = 7
  lengths = [15, 8, 23]
  print('>> Calculating ...')
  print('>> round_len = {}'.format(
    ParallelEngine.get_round_length(batch_size, num_steps, lengths)))

