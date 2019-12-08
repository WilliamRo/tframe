from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import checker
from tframe.nets.rnn_cells.cell_base import CellBase


class Tape(CellBase):

  def __init__(self, length, depth, num_tapes=1, **kwargs):
    # Call parent's constructor
    super().__init__(**kwargs)
    # Specific attributes
    self._depth = checker.check_positive_integer(depth)
    self._length = checker.check_positive_integer(length)
    self._num_tapes = checker.check_positive_integer(num_tapes)

  @property
  def tape_tail(self): return '({}x{})'.format(self._length, self._depth)

  @property
  def init_state(self):
    states = [self._get_placeholder(
      name='s{}'.format(i+1) if self._num_tapes > 1 else 'tape_state',
      size=[self._length, self._depth]) for i in range(self._num_tapes)]
    if self._num_tapes == 1: return states[0]
    else: return states

  def _record(self, tape, h):
    # Currently only single tape is supported
    assert self._num_tapes == 1
    # Check tensor shape
    assert isinstance(tape, tf.Tensor) and isinstance(h, tf.Tensor)
    # Make sure tape.shape = [bs, length, depth] and x.shape = [bs, depth]
    assert len(tape.shape) == 3 and len(h.shape) == 2
    assert tape.shape.as_list()[-1] == h.shape.as_list()[-1] == self._depth

    # Record
    tape, _ = tf.split(tape, (self._length - 1, 1), axis=1)
    tape = tf.concat([tf.expand_dims(h, 1), tape], axis=1)

    # Return tape
    return tape

  def _process(self, tape, x):
    raise NotImplementedError

  def _link(self, tape, x, **kwargs):
    # Check tensor shape
    assert isinstance(tape, tf.Tensor) and isinstance(x, tf.Tensor)
    # Make sure tape.shape = [bs, length, depth]
    # Allow x to be a tape or single-time data
    assert len(tape.shape) == 3 and len(x.shape) in (2, 3)
    # Process, record and return
    new_h = self._process(tape, x)
    new_tape = self._record(tape, new_h)
    return new_tape, new_tape
