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
      with_peepholes=False,
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
    self._use_bias = checker.check_type(use_bias, bool)
    self._weight_initializer = initializers.get(weight_initializer)
    self._bias_initializer = initializers.get(bias_initializer)

    self._input_gate = checker.check_type(input_gate, bool)
    self._output_gate = checker.check_type(output_gate, bool)
    self._forget_gate = checker.check_type(forget_gate, bool)
    self._with_peepholes = checker.check_type(with_peepholes, bool)

    self._output_scale = state_size

  # region : Properties

  def structure_string(self, detail=True, scale=True):
    gates = '[{}{}{}g{}]'.format('i' if self._input_gate else '',
                                 'f' if self._forget_gate else '',
                                 'o' if self._output_gate else '',
                                 'p' if self._with_peepholes else '')
    return self.net_name + gates + '({})'.format(
      self._state_size) if scale else ''

  @property
  def init_state(self):
    if self._init_state is not None: return self._init_state
    assert self._state_size is not None
    get_placeholder = lambda name: self._get_placeholder(name, self._state_size)
    self._init_state = (get_placeholder('h'), get_placeholder('c'))
    return self._init_state

  # endregion : Properties

  # region : Private Methods

  def _link(self, pre_states, input_, **kwargs):
    """pre_states = (h_{t-1}, c_{t-1})"""
    self._check_state(pre_states, 2)
    h, c = pre_states

    # Link
    if self._with_peepholes:
      new_h, new_c = self._link_with_peepholes(input_, h, c)
    else: new_h, new_c = self._basic_link(input_, h, c)

    # Return a tuple with the same structure as input pre_states
    return new_h, (new_h, new_c)

  def _get_zero_state(self, batch_size):
    assert not self.is_root
    return (np.zeros(shape=(batch_size, self._state_size)),
            np.zeros(shape=(batch_size, self._state_size)))

  # endregion : Private Methods

  # region : Link Cores

  def _basic_link(self, x, h, c):
    input_size = self._get_external_shape(x)

    # Determine the size of W and b according to gates to be used
    size_splits = 1 + self._input_gate + self._output_gate + self._forget_gate
    dim = self._state_size * size_splits

    # Initiate bias
    bias = None
    if self._use_bias: bias = self._get_bias('b', dim)

    W = self._get_variable('W', [self._state_size + input_size, dim])
    gate_inputs = tf.matmul(tf.concat([x, h], axis=1), W)
    if self._use_bias: gate_inputs = tf.nn.bias_add(gate_inputs, bias)

    # i = input_gate, g = new_input, f = forget_gate, o = output_gate
    i, f, o = (None,) * 3
    splits = list(tf.split(gate_inputs, num_or_size_splits=size_splits, axis=1))
    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    # - Calculate candidates to write
    g = self._activation(splits.pop())
    if self._input_gate:
      with tf.name_scope('write_gate'):
        i = tf.sigmoid(splits.pop(0))
        g = tf.multiply(i, g)
    # - Forget
    if self._forget_gate:
      with tf.name_scope('forget_gate'):
        f = tf.sigmoid(splits.pop(0))
        c = tf.multiply(f, c)
    # - Write
    with tf.name_scope('write'): new_c = tf.add(c, g)
    # - Read
    new_h = self._activation(new_c)
    if self._output_gate:
      with tf.name_scope('read_gate'):
        o = tf.sigmoid(splits.pop(0))
        new_h = tf.multiply(o, new_h)
    assert len(splits) == 0

    self._kernel, self._bias = W, bias
    return new_h, new_c

  def _link_with_peepholes(self, x, h, c):
    input_size = self._get_external_shape(x)
    Wi, Wf, W, Wo = (None,) * 4
    bi, bf, b, bo = (None,) * 4

    # Forget and write
    fi_inputs = tf.concat([h, x, c], axis=1)
    weight_shape = [input_size + self._state_size * 2, self._state_size]
    # - Calculate candidates to write
    W = self._get_variable(
      'W', [input_size + self._state_size, self._state_size])
    if self._use_bias: b = self._get_bias('b', self._state_size)
    g = self._activation(tf.nn.bias_add(
      tf.matmul(tf.concat([h, x], axis=1), W), b))
    # - Forget
    if self._forget_gate:
      with tf.name_scope('forget_gate'):
        Wf = self._get_variable('Wf', weight_shape)
        if self._use_bias: bf = self._get_bias('bf', self._state_size)
        f = tf.sigmoid(tf.nn.bias_add(tf.matmul(fi_inputs, Wf), bf))
        c = tf.multiply(f, c)
    # - Write
    if self._input_gate:
      with tf.name_scope('write_gate'):
        Wi = self._get_variable('Wi', weight_shape)
        if self._use_bias: bi = self._get_bias('bi', self._state_size)
        i = tf.sigmoid(tf.nn.bias_add(tf.matmul(fi_inputs, Wi), bi))
        g = tf.multiply(i, g)
    with tf.name_scope('write'): new_c = tf.add(c, g)
    # - Read
    new_h = self._activation(new_c)
    if self._output_gate:
      with tf.name_scope('read_gate'):
        Wo = self._get_variable('Wo', weight_shape)
        if self._use_bias: bo = self._get_bias('bo', self._state_size)
        o = tf.sigmoid(tf.nn.bias_add(
          tf.matmul(tf.concat([h, x, new_c], axis=1), Wo), bo))
        new_h = tf.multiply(new_h, o)

    self._kernel, self._bias = (Wi, Wf, W, Wo), (bi, bf, b, bo)
    return new_h, new_c

  # endregion : Link Cores




