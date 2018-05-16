from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tframe import checker
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
      input_gate=True,
      output_gate=True,
      forget_gate=True,
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

    self._input_gate = checker.check_type(input_gate, bool)
    self._output_gate = checker.check_type(output_gate, bool)
    self._forget_gate = checker.check_type(forget_gate, bool)

    self._output_scale = state_size

  # region : Properties

  def structure_string(self, detail=True, scale=True):
    gates = '[{}{}{}g]'.format('i' if self._input_gate else '',
                               'f' if self._forget_gate else '',
                               'o' if self._output_gate else '')
    return self.net_name + gates + '_{}'.format(
      self._state_size) if scale else ''

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

    # Determine the size of W and b according to gates to be used
    size_splits = 1 + self._input_gate + self._output_gate + self._forget_gate
    dim = self._state_size * size_splits

    # Initiate bias
    bias = None
    if self._use_bias: bias = tf.get_variable(
      'b', shape=[dim], dtype=hub.dtype, initializer=self._bias_initializer)

    get_variable = lambda name, shape: tf.get_variable(
      name, shape, dtype=hub.dtype, initializer=self._weight_initializer)

    W = get_variable('W', [self._state_size + input_size, dim])
    gate_inputs = tf.matmul(tf.concat([input_, h], axis=1), W)
    if self._use_bias: gate_inputs = tf.nn.bias_add(gate_inputs, bias)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, f, o = (None,) * 3
    splits = list(tf.split(gate_inputs, num_or_size_splits=size_splits, axis=1))
    if self._input_gate: i = tf.sigmoid(splits.pop(0))
    if self._forget_gate: f = tf.sigmoid(splits.pop(0))
    if self._output_gate: o = tf.sigmoid(splits.pop(0))
    g = self._activation(splits.pop())
    assert len(splits) == 0

    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    # - Forget
    if f is not None: c = tf.multiply(f, c)
    # - Write gate
    if i is not None: g = tf.multiply(i, g)
    # - Write
    new_c = tf.add(c, g)
    # - Read
    new_h = self._activation(new_c)
    if o is not None: new_h = tf.multiply(o, new_h)

    self._kernel, self._bias = W, bias
    # Return a tuple with the same structure as input pre_states
    return new_h, (new_h, new_c)

  def _get_zero_state(self, batch_size):
    assert not self.is_root
    return (np.zeros(shape=(batch_size, self._state_size)),
            np.zeros(shape=(batch_size, self._state_size)))

  # endregion : Private Methods




