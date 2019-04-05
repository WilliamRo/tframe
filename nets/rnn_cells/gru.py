from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import checker
from tframe.nets.rnn_cells.cell_base import CellBase
from tframe import initializers


class GRU(CellBase):
  """Gated Recurrent Unit"""
  net_name = 'gru'

  def __init__(
      self,
      state_size,
      use_reset_gate=True,
      activation='tanh',
      weight_initializer='xavier_normal',
      use_bias=True,
      bias_initializer='zeros',
      z_bias_initializer='zeros',
      **kwargs):
    # Call parent's constructor
    CellBase.__init__(self, activation, weight_initializer,
                      use_bias, bias_initializer, **kwargs)

    # Specific attributes
    self._state_size = checker.check_positive_integer(state_size)
    self._use_reset_gate = checker.check_type(use_reset_gate, bool)
    self._z_bias_initializer = initializers.get(z_bias_initializer)


  @property
  def _scale_tail(self):
    return '[{}]({})'.format(
      '-' if self._use_reset_gate is None else 'r', self._state_size)


  def _link(self, prev_s, x, **kwargs):
    """s(pre_states) is state_array of size 'state_size'"""
    self._check_state(prev_s)
    # - Calculate r gate and z gate
    r = None
    if self._use_reset_gate:
      r = self.neurons(x, prev_s, is_gate=True, scope='reset_gate')
      self._gate_dict['reset_gate'] = r

    z = self.neurons(x, prev_s, is_gate=True, scope='update_gate',
                     bias_initializer=self._z_bias_initializer)
    self._gate_dict['update_gate'] = z

    # - Read
    s_w = prev_s
    if self._use_reset_gate:
      with tf.name_scope('read'): s_w = tf.multiply(r, prev_s)
    # - Calculate candidates to write
    s_bar = self.neurons(x, s_w, activation=self._activation, scope='s_bar')
    with tf.name_scope('write'): new_s = tf.add(
      tf.multiply(z, prev_s), tf.multiply(tf.subtract(1., z), s_bar))

    return new_s, new_s
