from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker, hub
from tframe.nets.rnn_cells.cell_base import CellBase


class Conveyor(CellBase):
  """
  TODO: to be deprecated. Use the more general api--tape--instead.
  """

  net_name = 'conveyor'

  def __init__(self, length, **kwargs):
    # Call parent's constructor
    CellBase.__init__(self, **kwargs)
    # Specific attributes
    self._length = checker.check_positive_integer(length)
    self._input_shape = hub.conveyor_input_shape
    # conveyor_input_shape must be provided in xx_core.py module
    assert isinstance(self._input_shape, list)
    self._state_shape = [self._length] + self._input_shape

  @property
  def _scale_tail(self):
    return '({})'.format('x'.join([str(s) for s in self._state_shape]))

  @property
  def init_state(self):
    return tf.placeholder(
      dtype=hub.dtype, shape=[None] + self._state_shape, name='conveyor_state')

  def _link(self, s, x, **kwargs):
    # Typically, s.shape = [B, L, S], x.shape = [B, S]
    assert isinstance(x, tf.Tensor)
    s, _ = tf.split(s, (self._length - 1, 1), axis=1)
    output = tf.concat([tf.expand_dims(x, 1), s], axis=1)
    return output, output


