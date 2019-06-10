from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import checker
from tframe.nets.rnn_cells.cell_base import CellBase
from tframe import initializers
from tframe.operators.apis.dynamic_weights import DynamicWeights


class GRU(CellBase, DynamicWeights):
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
      reset_who='s',
      **kwargs):
    """
    :param reset_who: in ('x', 'y')
           'x': a_h = W_h * (h_{t-1} \odot r_t)
           'y': a_h = r_t \odot (W_h * h_{t-1})
           \hat{h}_t = \varphi(Wx*x + a_h + b)
           in which r_t is the reset gate at time step t,
           \odot is the Hadamard product, W_h is the hidden-to-hidden matrix
    """
    # Call parent's constructor
    CellBase.__init__(self, activation, weight_initializer,
                      use_bias, bias_initializer, **kwargs)

    # Specific attributes
    self._state_size = checker.check_positive_integer(state_size)
    self._use_reset_gate = checker.check_type(use_reset_gate, bool)
    self._z_bias_initializer = initializers.get(z_bias_initializer)

    assert reset_who in ('s', 'a')
    self._reset_who = reset_who


  @property
  def _scale_tail(self):
    return '[{}]({})'.format(
      '-' if not self._use_reset_gate else ('r' + self._reset_who),
      self._state_size)


  def _link(self, prev_s, x, **kwargs):
    """s(pre_states) is state_array of size 'state_size'"""
    self._check_state(prev_s)
    # - Calculate z gate
    z = self.neurons(x, prev_s, is_gate=True, scope='update_gate',
                     bias_initializer=self._z_bias_initializer)
    self._gate_dict['update_gate'] = z

    # - Calculate s_bar
    if self._use_reset_gate:
      s_bar = self.neurons_with_reset_gate(x, prev_s, self._reset_who)
    else:
      s_bar = self.neurons(
        x, prev_s, activation=self._activation, scope='s_bar')

    # - Update state
    with tf.name_scope('update_state'): new_s = tf.add(
      tf.multiply(z, prev_s), tf.multiply(tf.subtract(1., z), s_bar))

    return new_s, new_s

