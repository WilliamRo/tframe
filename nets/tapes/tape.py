from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf

from tframe import checker
from tframe.nets.rnn_cells.cell_base import CellBase


class Tape(CellBase):

  net_name = 'tape'

  def __init__(
      self,
      length,
      depth,
      num_tapes=1,
      dropout=0.0,
      pos_encoding=None,
      output_last=False,
      **kwargs):

    # Call parent's constructor
    super().__init__(**kwargs)
    # Specific attributes
    self._depth = checker.check_positive_integer(depth)
    self._length = checker.check_positive_integer(length)
    self._num_tapes = checker.check_positive_integer(num_tapes)
    self._dropout = dropout
    self._output_last = output_last
    assert pos_encoding in (None, 'absolute', 'relative', 'abs', 'rel')
    self._pos_encoding = pos_encoding

  # region : Properties

  @property
  def _scale_tail(self): return self.tape_tail

  @property
  def tape_tail(self): return '({}x{})'.format(self._length, self._depth)

  @property
  def init_state(self):
    states = [self._get_placeholder(
      name='s{}'.format(i+1) if self._num_tapes > 1 else 'tape_state',
      size=[self._length, self._depth]) for i in range(self._num_tapes)]
    if self._num_tapes == 1: return states[0]
    else: return states

  # endregion : Properties

  # region : Static Methods

  @staticmethod
  def get_angles(pos, i, depth):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(depth))
    return pos * angle_rates

  @staticmethod
  def absolute_positional_encoding(position, depth):
    angle_rads = Tape.get_angles(
      np.arange(position)[:, np.newaxis],
      np.arange(depth)[np.newaxis, :], depth)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

  # endregion : Static Methods

  # region : Core Methods

  def _record(self, tape, h):
    # Currently only single tape is supported
    assert self._num_tapes == 1
    # Check tensor shape
    assert isinstance(tape, tf.Tensor) and isinstance(h, tf.Tensor)
    # Make sure tape.shape = [bs, length, depth] and x.shape = [bs, depth]
    assert len(tape.shape) == 3 and len(h.shape) in (2, 3)
    assert tape.shape.as_list()[-1] == h.shape.as_list()[-1] == self._depth

    # Record
    tape, _ = tf.split(tape, (self._length - 1, 1), axis=1)
    # .. expand h if necessary
    if len(h.shape) == 2: h = tf.expand_dims(h, 1)
    assert h.shape.as_list()[-2] == 1
    # .. concatenate
    tape = tf.concat([h, tape], axis=1)

    # Return tape
    return tape

  def _process(self, tape, x):
    return x

  def _encode(self, x):
    if self._pos_encoding in ('absolute', 'abs'):
      pos_encoding = self.absolute_positional_encoding(
        self._length, self._depth)
      x += pos_encoding
    elif self._pos_encoding in ('relative', 'rel'):
      raise NotImplementedError
    elif self._pos_encoding is not None:
      raise KeyError('Unknown encoding type {}'.format(self._pos_encoding))
    # Return
    return x

  def _link(self, tape, x, **kwargs):
    # Check tensor shape
    assert isinstance(tape, tf.Tensor) and isinstance(x, tf.Tensor)
    # Make sure tape.shape = [bs, length, depth]
    # Allow x to be a tape or single-time data
    assert len(tape.shape) == 3 and len(x.shape) in (2, 3)
    # Process
    new_h = self._process(tape, x)
    # Record
    new_tape = self._record(tape, new_h)
    # Encode output if necessary
    output = self._encode(new_tape)
    # Output last if necessary
    if self._output_last: output = output[..., -1, :]
    # Apply dropout to output if necessary
    if self._dropout_rate > 0: output = self.dropout(output, self._dropout_rate)
    return output, new_tape

  # endregion : Core Methods
