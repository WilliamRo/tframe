from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.nets.rnn_cells.srn import BasicRNNCell


class GRU(BasicRNNCell):
  """Gated Recurrent Unit
  """
  net_name = 'gru'

  def _link(self, prev_s, x, **kwargs):
    """s(pre_states) is state_array of size 'state_size'"""
    self._check_state(prev_s)
    # - Calculate r gate and z gate
    r, z = self.neurons(
      x, prev_s, num_or_size_splits=2, is_gate=True, scope='gates')
    self._gate_dict['reset_gate'] = r
    self._gate_dict['update_gate'] = z

    # - Read
    with tf.name_scope('read'): s_w = tf.multiply(r, prev_s)
    # - Calculate candidates to write
    s_bar = self.neurons(x, s_w, activation=self._activation, scope='s_bar')
    with tf.name_scope('write'): new_s = tf.add(
      tf.multiply(z, prev_s), tf.multiply(tf.subtract(1., z), s_bar))

    return new_s, new_s
