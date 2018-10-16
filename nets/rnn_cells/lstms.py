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
      output_gate_bias_initializer=None,
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
    self._output_gate_bias_initializer = initializers.get(
      output_gate_bias_initializer)

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
    if self._use_bias:
      if self._output_gate_bias_initializer is None:
        bias = self._get_bias('b', dim)
      else:
        gif_bias = self._get_bias('gif_bias', dim - self._state_size)
        o_bias = self._get_bias(
          'o_bias', self._state_size, self._output_gate_bias_initializer)
        bias = tf.concat([gif_bias, o_bias], axis=0)

    W = self._get_variable('W', [self._state_size + input_size, dim])
    gate_inputs = tf.matmul(tf.concat([x, h], axis=1), W)
    if self._use_bias: gate_inputs = tf.nn.bias_add(gate_inputs, bias)

    # i = input_gate, g = new_input, f = forget_gate, o = output_gate
    i, f, o = (None,) * 3
    splits = list(tf.split(gate_inputs, num_or_size_splits=size_splits, axis=1))
    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    # - Calculate candidates to write
    g = self._activation(splits.pop(0))
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


class OriginalLSTMCell(RNet):
  """The original LSTM cell proposed in 1997, which has no forget gate and
     and cell blocks are of size 1
  """
  net_name = 'origin_lstm'

  def __init__(
      self,
      state_size,
      cell_activation='sigmoid',
      cell_activation_range=(-2, 2),
      memory_activation='sigmoid',
      memory_activation_range=(-1, 1),
      weight_initializer='xavier_uniform',
      weight_initial_range=None,
      use_cell_bias=False,
      cell_bias_initializer='zeros',
      use_in_bias=True,
      in_bias_initializer='zeros',
      use_out_bias=True,
      out_bias_initializer='zeros',
      truncate=True,
      forward_gate=True,
      **kwargs):

    # Call parent's constructor
    RNet.__init__(self, OriginalLSTMCell.net_name)

    # Set state size
    self._state_size = state_size

    # Set activation
    self._cell_activation = activations.get(
      cell_activation, range=cell_activation_range)
    self._memory_activation = activations.get(
      memory_activation, range=memory_activation_range)
    self._gate_activation = activations.get('sigmoid')

    # Set weight and bias configs
    self._weight_initializer = initializers.get(
      weight_initializer, range=weight_initial_range)
    self._use_cell_bias = use_cell_bias
    self._cell_bias_initializer = initializers.get(cell_bias_initializer)
    self._use_in_bias = use_in_bias
    self._in_bias_initializer = initializers.get(in_bias_initializer)
    self._use_out_bias = use_out_bias
    self._out_bias_initializer = initializers.get(out_bias_initializer)

    # Additional options
    self._truncate = truncate
    self._forward_gate = forward_gate

    # ...
    self._num_splits = 3
    self._output_scale = state_size
    self._h_size = (state_size * self._num_splits if self._forward_gate else
                    state_size)

  # region : Properties

  def structure_string(self, detail=True, scale=True):
    str = self.net_name
    if scale: str += '({})'.format(self._state_size)
    return str

  @property
  def init_state(self):
    if self._init_state is not None: return self._init_state
    assert self._state_size is not None
    self._init_state = (self._get_placeholder('h', self._h_size),
                        self._get_placeholder('c', self._state_size))
    return self._init_state

  # endregion : Properties

  # region : Private Methods

  def _get_zero_state(self, batch_size):
    assert not self.is_root
    return (np.zeros(shape=(batch_size, self._h_size)),
            np.zeros(shape=(batch_size, self._state_size)))

  @staticmethod
  @tf.custom_gradient
  def _truncate_matmul(x, W):
    def grad(dy):
      return tf.zeros_like(x), tf.matmul(tf.transpose(x), dy)
    return tf.matmul(x, W), grad

  def _link(self, pre_states, x, **kwargs):
    # Get input size and previous states
    input_size = self._get_external_shape(x)
    self._check_state(pre_states, (self._h_size, self._state_size))
    h, c = pre_states

    # :: Calculate net_c, net_in, net_out using h = [y_c, y_in, y_out] when
    #    forward gate (otherwise h = y_c) and x
    # .. Calculate net chunk before adding bias
    W = self._get_variable(
      'W', [self._h_size + input_size, self._num_splits * self._state_size])
    input_chunk = tf.concat([x, h], axis=1)

    if self._truncate:
      net_chunk = self._truncate_matmul(input_chunk, W)
    else:
      net_chunk = tf.matmul(input_chunk, W)

    # .. Unpack the chunk and add bias if necessary
    net_c, net_in, net_out = tf.split(
      net_chunk, num_or_size_splits=self._num_splits, axis=1)

    # Calculate input and output gate
    in_bias = None
    if self._use_in_bias:
      in_bias = self._get_bias(
        'in_bias', self._state_size, initializer=self._in_bias_initializer)
      net_in = tf.nn.bias_add(net_in, in_bias)
    y_in = self._gate_activation(net_in)

    out_bias = None
    if self._use_out_bias:
      out_bias = self._get_bias(
        'out_bias', self._state_size, initializer=self._out_bias_initializer)
      net_out = tf.nn.bias_add(net_out, out_bias)
    y_out = self._gate_activation(net_out)

    # Calculate new state
    cell_bias = None
    if self._use_cell_bias:
      cell_bias = self._get_bias(
        'cell_bias', self._state_size, initializer=self._cell_bias_initializer)
      net_c = tf.nn.bias_add(net_c, cell_bias)
    new_c = tf.add(c, tf.multiply(y_in, self._cell_activation(net_c)))

    # Calculate output
    y_c = tf.multiply(y_out, self._memory_activation(new_c))

    self._kernel = W
    self._bias = [b for b in (in_bias, out_bias, cell_bias) if b is not None]
    # Generate new_h and return
    new_h = (tf.concat([y_c, y_in, y_out], axis=1) if self._forward_gate
             else y_c)
    return y_c, (new_h, new_c)

  # endregion : Private Methods








