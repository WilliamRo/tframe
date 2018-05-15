from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tframe import activations
from tframe import initializers
from tframe import hub

from tframe.nets import RNet


class BasicLSTMCell(RNet):
  """Basic LSTM cell
  """
  net_name = 'basic_lstm'

  def __init__(
      self,
      state_size,
      activation='tanh',
      use_bias=True,
      weight_initializer='xavier_uniform',
      bias_initializer='zeros',
      **kwargs):
    """
    :param state_size: state size: positive int
    :param activation: activation: string or callable
    :param use_bias: whether to use bias
    :param weight_initializer: weight initializer identifier
    :param bias_initializer: bias initializer identifier
    """
    # Call parent's constructor
    RNet.__init__(self, BasicLSTMCell.net_name)

    # Attributes
    self._state_size = state_size
    self._activation = activations.get(activation, **kwargs)
    self._use_bias = use_bias
    self._weight_initializer = initializers.get(weight_initializer)
    self._bias_initializer = initializers.get(bias_initializer)
    self._output_scale = state_size

  # region : Properties

  def structure_string(self, detail=True, scale=True):
    return self.net_name + '_{}'.format(self._state_size) if scale else ''

  @property
  def init_state(self):
    if self._init_state is not None: return self._init_state
    assert self._state_size is not None
    place_holder = lambda name: tf.placeholder(
      dtype=hub.dtype, shape=(None, self._state_size), name=name)
    self._init_state = (place_holder('h'), place_holder('c'))
    return self._init_state

  # endregion : Properties

  # region : Private Methods

  def _link(self, pre_states, input_, **kwargs):
    """pre_states = (h_{t-1}, c_{t-1})"""
    self._check_state(pre_states, 2)
    h, c = pre_states
    input_size = self._get_external_shape(input_)

    # Initiate bias
    bias = (tf.get_variable('b', shape=[self._state_size * 4], dtype=hub.dtype,
                            initializer=self._bias_initializer)
            if self._use_bias else None)

    get_variable = lambda name, shape: tf.get_variable(
      name, shape, dtype=hub.dtype, initializer=self._weight_initializer)

    W = get_variable('W', [self._state_size + input_size, self._state_size * 4])
    gate_inputs = tf.matmul(tf.concat([input_, h], axis=1), W)
    if self._use_bias: gate_inputs = tf.nn.bias_add(gate_inputs, bias)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, f, o, g = tf.split(gate_inputs, num_or_size_splits=4, axis=1)

    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    add, multiply = tf.add, tf.multiply
    new_c = add(multiply(tf.sigmoid(f), c),
                multiply(tf.sigmoid(i), self._activation(g)))
    new_h = multiply(tf.sigmoid(o), self._activation(new_c))

    self._kernel, self._bias = W, bias
    # Return a tuple with the same structure as input pre_states
    return new_h, (new_h, new_c)

  def _get_zero_state(self, batch_size):
    assert not self.is_root
    return (np.zeros(shape=(batch_size, self._state_size)),
            np.zeros(shape=(batch_size, self._state_size)))

  # endregion : Private Methods




