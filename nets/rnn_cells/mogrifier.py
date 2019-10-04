from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import hub as th
from tframe import checker

from tframe.nets.rnn_cells.lstms import LSTM


class Mogrifier(LSTM):
  """Reference: Melis, etc. Mogrifier LSTM. 2019"""

  net_name = 'mogrifier'

  def __init__(
      self,
      state_size,
      rounds=5,
      lower_rank=None,
      activation='tanh',
      weight_initializer='xavier_normal',
      use_bias=True,
      cell_bias_initializer='zeros',
      input_bias_initializer='zeros',
      output_bias_initializer='zeros',
      forget_bias_initializer='zeros',
      **kwargs):
    # Call parent's constructor
    LSTM.__init__(
      state_size, activation, weight_initializer, use_bias,
      cell_bias_initializer, input_bias_initializer, output_bias_initializer,
      forget_bias_initializer, **kwargs)

    # Specific attributes
    self._rounds = checker.check_positive_integer(rounds, allow_zero=True)
    self._lower_rank = lower_rank

  def _mogrify(self, x, h):
    for i in range(self._rounds):
      if i % 2 == 0:
        # update x
        pass
      else:
        # update h
        pass
    return x, h

  def _gate_input(self, evidence, target, i):
    with tf.variable_scope('round_{}'.format(i + 1)):
      pass

  def _link(self, pre_states, x, **kwargs):
    # Check and unpack previous states
    self._check_state(pre_states, 2)
    h, c = pre_states
    # Modulate x and h
    x, h = self._mogrify(x, h)
    # Get gates and g
    f, i, o, g = self._get_fiog_fast(x, h)
    # Calculate new_c
    new_c = tf.add(tf.multiply(f, c), tf.multiply(i, g), 'new_c')
    # Calculate new_h
    new_h = tf.multiply(o, tf.tanh(new_c))

    # Register gates and return
    self._gate_dict['input_gate'] = i
    self._gate_dict['forget_gate'] = f
    self._gate_dict['output_gate'] = o
    return new_h, (new_h, new_c)



