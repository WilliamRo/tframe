from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.nets.rnn_cells.basic_cell import BasicRNNCell


class GRU(BasicRNNCell):
  """Gated Recurrent Unit
  """
  net_name = 'gru'

  def _link(self, s, x, **kwargs):
    """s(pre_states) is state_array of size 'state_size'"""
    self._check_state(s)
    input_size = self._get_external_shape(x)
    Wrz, Ws, brz, bs = (None,) * 4

    # r gate and z gate
    Wrz = self._get_variable(
      'Wrz', [self._state_size + input_size, 2 * self._state_size])
    if self._use_bias: brz = self._get_bias('brz', 2 * self._state_size)
    rz = tf.sigmoid(tf.nn.bias_add(tf.matmul(
      tf.concat([s, x], axis=1), Wrz), brz))
    r, z = tf.split(rz, num_or_size_splits=2, axis=1)
    # - Read
    with tf.name_scope('read'): s_w = tf.multiply(r, s)
    # - Calculate candidates to write
    Ws = self._get_variable(
      'Ws', [self._state_size + input_size, self._state_size])
    if self._use_bias: bs = self._get_bias('bs', self._state_size)
    s_bar = self._activation(tf.nn.bias_add(tf.matmul(
      tf.concat([s_w, x], axis=1), Ws), bs))
    with tf.name_scope('write'):
      new_s = tf.add(tf.multiply(z, s), tf.multiply(tf.subtract(1., z), s_bar))

    self._kernel = (Wrz, Ws)
    self._bias = (brz, bs)
    return new_s, new_s


