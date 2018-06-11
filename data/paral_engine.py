import numpy as np

from tframe import checker


class ParallelEngine(object):
  def __init__(self, batch_size):
    self._sequences = [None] * checker.check_positive_integer(batch_size)
    self._targets = [None] * self.size
    self._cursors = [0] * self.size

  # region : Properties

  @property
  def size(self):
    return len(self._sequences)

  @property
  def remainders(self):
    return [len(s) - c for s, c in zip(self._sequences, self._cursors)]

  @property
  def max_emit_length(self):
    return min(self.remainders)

  @property
  def is_ready(self):
    return self.next_inactive_index is None

  @property
  def inactive_indices(self):
    indices = np.argwhere(
      [s is None or len(s) - c == 0
       for s, c in zip(self._sequences, self._cursors)]).ravel()
    return list(indices)

  @property
  def next_inactive_index(self):
    indices = self.inactive_indices
    if len(indices) == 0: return None
    else: return indices[0]

  @property
  def flameout(self):
    return self.size == 0

  # endregion : Properties

  # region : Private Methods

  def _set_sequence(self, index, sequence, target):
    if sequence is None:
      self._sequences.pop(index)
      self._targets.pop(index)
      self._cursors.pop(index)
    else:
      self._sequences[index] = sequence
      self._targets[index] = target
      self._cursors[index] = 0

  # endregion : Private Methods

  # region : Public Methods

  def set_sequence(self, sequence, target):
    # Sanity check
    assert not self.is_ready
    # Find next position to replace sequence
    index = self.next_inactive_index
    # Substitution
    self._set_sequence(index, sequence, target)
    return index

  def emit(self, num_steps):
    assert self.is_ready
    assert isinstance(num_steps, int)
    if num_steps < 0: num_steps = self.max_emit_length

    steps = min(self.max_emit_length, num_steps)
    feature_shape = self._sequences[0].shape[1:]
    target_shape = self._targets[0].shape[1:]

    features = np.zeros(shape=(self.size, steps, *feature_shape))
    targets = np.zeros(shape=(self.size, steps, *target_shape))
    for i in range(self.size):
      # Fill in features
      sequence_i = self._sequences[i]
      assert isinstance(sequence_i, np.ndarray)
      c = self._cursors[i]
      features[i] = sequence_i[c:c + steps]
      # Fill in targets
      target_i = self._targets[i]
      assert isinstance(target_i, np.ndarray)
      if len(target_i) == 1:
        targets[i] = np.tile(target_i, (steps, 1))
      else:
        assert len(target_i) == len(sequence_i)
        targets[i] = target_i[c:c + steps]

      # Move cursor
      assert 0 < c + steps <= len(sequence_i)
      self._cursors[i] += steps

    # Return features and targets
    return features, targets

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
          x = np.zeros(shape=(length, 1))
          y = x
          cursor += 1
        else: x, y = None, None
        pe.set_sequence(x, y)

      if pe.flameout: break
      pe.emit(num_steps)
      round_len += 1

    return round_len

  # endregion : Public Methods


if __name__ == '__main__':
  batch_size = 3
  num_steps = 28
  lengths = [15, 8, 23]
  print('>> Calculating ...')
  print('>> round_len = {}'.format(
    ParallelEngine.get_round_length(batch_size, num_steps, lengths)))

