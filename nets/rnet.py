from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe import hub
from tframe import checker
from tframe.nets.net import Net


class RNet(Net):
  """Recurrent net which outputs states besides common result"""
  net_name = 'rnet'
  # If not None, compute_gradients should be callable, it accepts a dL/dy
  # .. tensor with shape (num_steps(=1), batch_size, *(y.shape))
  # .. and returns (grads_and_vars, new_grad_buffer)
  compute_gradients = None

  def __init__(self, name):
    # Call parent's constructor
    Net.__init__(self, name)

    # Attributes
    self._inter_type = self.RECURRENT
    self._state_array = None
    self._state_size = None
    self._init_state = None
    self._kernel = None
    self._bias = None
    self._weight_initializer = None
    self._bias_initializer = None

    # For real-time training TODO: BETA
    self.repeater_tensors = None  # registered in sub-classes
    self._grad_tensors = None

    self._new_state_tensor = None  #
    self._gradient_buffer_placeholder = None
    self._gradient_buffer_array = None

    self._custom_vars = None

  # region : Properties

  @property
  def custom_var_list(self):
    assert self.linked
    if self._custom_vars is not None:
      assert isinstance(self._custom_vars, (set, list, tuple))
      return list(self._custom_vars)
    else:
      return self.parameters

  @property
  def default_var_list(self):
    assert self.linked
    return list(set(self.parameters) - set(self.custom_var_list))

  @property
  def rnn_cells(self):
    assert self.is_root
    return [net for net in self.children if isinstance(net, RNet)]

  @property
  def rnn_cell_num(self):
    return len(self.rnn_cells)

  @property
  def init_state(self):
    if self._init_state is not None: return self._init_state
    # Initiate init_state
    if self.is_root:
      states = []
      with tf.name_scope('InitStates'):
        for rnn_cell in self.rnn_cells:
          states.append(rnn_cell.init_state)
      assert len(states) == self.rnn_cell_num
      self._init_state = tuple(states)
    else:
      # If initial state is a tuple, this property must be overriden
      assert self._state_size is not None
      # The initialization of init_state must be done under with_graph
      # .. decorator
      self._init_state = tf.placeholder(
        dtype=hub.dtype, shape=(None, self._state_size), name='init_state')

    return self._init_state

  @property
  def gradient_buffer_placeholder(self):
    if self._gradient_buffer_placeholder is not None: return self._gradient_buffer_placeholder
    # Initiate gradient buffer
    if self.is_root:
      buffer = []
      with tf.name_scope('GradientBuffer'):
        for rnn_cell in self.rnn_cells:
          if rnn_cell.compute_gradients is not None:
            buffer.append(rnn_cell.gradient_buffer_placeholder)
      self._gradient_buffer_placeholder = tuple(buffer)
    else:
      raise NotImplementedError('!! Properties not implemented.')

    return self._gradient_buffer_placeholder

  @property
  def repeater_containers(self):
    """TODO: BETA"""
    assert self.is_root
    rcs = []
    for child in self.children:
      assert isinstance(child, Net)
      if isinstance(child, RNet) and child.compute_gradients is not None:
        rcs.append(child)
    return tuple(rcs)

  # endregion : Properties

  # region : Overriden Methods

  def _link(self, pre_outputs, input_, **kwargs):
    """
    This methods should be called only by tf.scan
    :param pre_outputs: (output:Tensor, states:List of Tensors)
                          set None to use default initializer
    :param input_: data input in a time step
    :return: (output:Tensor, states:List of Tensors)
    """
    # Check inputs
    if pre_outputs is not None:
      assert isinstance(pre_outputs, tuple)
      if hub.use_rtrl:
        # TODO: BETA
        assert len(pre_outputs) == 3
      else: assert len(pre_outputs) == 2
      pre_states = pre_outputs[1]
      # The assertion below is not held by rnn_cells
      assert isinstance(pre_states, (tuple, list))
      assert len(pre_states) == self.rnn_cell_num
    else: raise ValueError('!! pre_outputs can not be None')  # TODO
    assert isinstance(input_, tf.Tensor)

    # Link
    states = []
    output = input_
    state_cursor = 0
    for net in self.children:
      assert isinstance(net, Net)
      if isinstance(net, RNet):
        # rnn_cells in self.children accept state and input
        # .. and gives (output, state)
        output, state = net(pre_states[state_cursor], output)
        states.append(state)
        state_cursor += 1
      else:
        output = net(output)

    assert len(states) == len(pre_states)

    result_tuple = output, tuple(states)

    # TODO: BETA
    if hub.use_rtrl:
      result_tuple += self._get_grad_tensor_tuple(output),

    return result_tuple

  # endregion : Overriden Methods

  # region : Public Methods

  def reset_buffers(self, batch_size):
    assert self.is_root
    self._state_array = self._get_zero_state(batch_size)
    # TODO: BETA
    if hub.use_rtrl:
      self._gradient_buffer_array = self._get_zero_gradient_buffer(
        batch_size)

  def reset_part_buffer(self, indices, values=None):
    """This method is first designed for parallel training of RNN model with
        irregular sequence input"""
    assert isinstance(indices, (list, tuple))
    assert self.is_root

    # Separate indices
    if values is None: values = [0] * len(indices)
    zero_indices = [i for i, v in zip(indices, values) if v is not None]
    none_indices = [i for i in indices if i not in zero_indices]
    assert len(zero_indices) + len(none_indices) == len(indices)

    def _reset(state):
      if isinstance(state, np.ndarray):
        if len(zero_indices) > 0: state[np.array(zero_indices), :] = 0
        if len(none_indices) > 0: state = np.delete(state, none_indices, axis=0)
        return state
      elif isinstance(state, (list, tuple)):
        # tf.scan returns a list of states
        state = list(state)
        for i, s in enumerate(state): state[i] = _reset(s)
        return tuple(state)
      else:
        raise TypeError('!! Unknown type of states: {}'.format(type(state)))

    self._state_array = _reset(self._state_array)
    # TODO: BETA
    if hub.use_rtrl:
      self._gradient_buffer_array = _reset(self._gradient_buffer_array)

  # endregion : Public Methods

  # region : Private Methods

  def _get_grad_tensor_tuple(self, y):
    """TODO: BETA
    :param y: output tensor (B, D)
    :return: a tuple: (dy/dR1, ..., dy/dRn)
    """
    assert self.is_root
    containers = self.repeater_containers
    assert len(containers) > 0

    mascot = tf.placeholder(dtype=tf.float32)

    grad_tensors = []
    y_size = y.get_shape().as_list()[1]
    y_list = tf.split(y, num_or_size_splits=y.shape[1], axis=1)

    for r in [c.repeater_tensors for c in containers]:
      # r.shape is (B, D_r) while y.shape is (B, D_y)
      assert isinstance(r, tf.Tensor)
      dy_dr = []
      for yi in y_list:
        dyi_dr = tf.gradients(yi, r)[0]  # (B, R)
        dy_dr.append(tf.expand_dims(dyi_dr, 1))  # (B, 1, R)
      dy_dr = tf.concat(dy_dr, axis=1)  # (B, D, R)
      grad_tensors.append(dy_dr)
    return tuple(grad_tensors)

  def _get_zero_state(self, batch_size):
    if self.is_root:
      state_array = []
      for rnn_cell in self.rnn_cells:
        state_array.append(rnn_cell._get_zero_state(batch_size))
      return tuple(state_array)
    else:
      # Get zero state according to self.init_state
      tf_states = self.init_state
      if not isinstance(tf_states, tuple): tf_states = (tf_states,)
      state = []
      for tf_state in tf_states:
        assert isinstance(tf_state, tf.Tensor)
        shape_list = tf_state.shape.as_list()
        shape_list[0] = batch_size
        state.append(np.zeros(shape=shape_list))

      state = tuple(state)
      if len(state) == 1: state = state[0]
      return state

  def _get_zero_gradient_buffer(self, batch_size):
    if self.is_root:
      buffer = []
      for rnn_cell in self.rnn_cells:
        if rnn_cell.compute_gradients is not None:
          buffer.append(rnn_cell._get_zero_gradient_buffer(batch_size))
      return tuple(buffer)
    else:
      # Get zero gradient buffer according to self.gradient buffer
      buffer = []
      for dS_dWj in self.gradient_buffer_placeholder:
        assert isinstance(dS_dWj, tf.Tensor)
        shape_list = dS_dWj.shape.as_list()
        shape_list[0] = batch_size
        buffer.append(np.zeros(shape=shape_list))
      # for dsk_dws in self.gradient_buffer_placeholder:
      #   np_dws = []
      #   for dw in dsk_dws:
      #     assert isinstance(dw, tf.tensor)
      #     shape_list = dw.shape.as_list()
      #     shape_list[0] = batch_size
      #     np_dws.append(np.zeros(shape=shape_list))
      #   buffer.append(tuple(np_dws))
      return tuple(buffer)

  def _get_rnn_dict(self, batch_size=None):
    """Get state dict together with gradient buffer if necessary
    """
    assert self.is_root

    rnn_dict = {}
    # During training, batch size is not None
    if batch_size is None:
      # During training
      state = self._state_array
      assert state is not None
      # TODO: BETA
      if hub.use_rtrl:
        rnn_dict[self.gradient_buffer_placeholder] = self._gradient_buffer_array
    else:
      # While is_training == False
      checker.check_positive_integer(batch_size)
      state = self._get_zero_state(batch_size)

    rnn_dict[self.init_state] = state
    return rnn_dict

  def _check_state(self, state, num_or_sizes=1):
    # Check num_or_sizes
    if isinstance(num_or_sizes, int):
      assert num_or_sizes > 0 and self._state_size is not None
      sizes = (self._state_size,) * num_or_sizes
    else:
      assert isinstance(num_or_sizes, tuple)
      sizes = num_or_sizes
    # Check state
    if not isinstance(state, tuple): state = (state,)
    # Check state
    assert len(state) == len(sizes)
    for s, size in zip(state, sizes):
      assert isinstance(s, tf.Tensor) and isinstance(size, int)
      assert s.shape.as_list()[1] == size

  @staticmethod
  def _get_external_shape(input_):
    assert isinstance(input_, tf.Tensor)
    input_shape = input_.shape.as_list()
    assert len(input_shape) == 2
    return input_shape[1]

  @staticmethod
  def _get_placeholder(name, size):
    return tf.placeholder(dtype=hub.dtype, shape=(None, size), name=name)

  def _get_variable(self, name, shape):
    initializer = self._weight_initializer
    if initializer is None:
      initializer = tf.glorot_normal_initializer()
    return tf.get_variable(
      name, shape, dtype=hub.dtype, initializer=initializer)

  def _get_bias(self, name, dim, initializer=None):
    if initializer is None: initializer = self._bias_initializer
    if initializer is None:
      initializer = tf.zeros_initializer()
    return tf.get_variable(
      name, shape=[dim], dtype=hub.dtype, initializer=initializer)

  def _get_weight_and_bias(self, weight_shape, use_bias, symbol=''):
    assert isinstance(weight_shape, (list, tuple)) and len(weight_shape) == 2
    W_name, b_name = 'W{}'.format(symbol), 'b{}'.format(symbol)
    W = self._get_variable(W_name, weight_shape)
    b = self._get_bias(b_name, weight_shape[1]) if use_bias else None
    return W, b

  def _net(self, x, W, b):
    return tf.nn.bias_add(tf.matmul(x, W), b)

  def _gate(self, x, W, b):
    return tf.sigmoid(self._net(x, W, b))

  def _distribute_last_tensors(self):
    assert self.is_root
    s_cursor, g_cursor = 0, 0
    for child in self.children:
      if isinstance(child, RNet):
        child._new_state_tensor = self._new_state_tensor[s_cursor]
        s_cursor += 1
        # TODO: BETA
        if hub.use_rtrl:
          child._grad_tensors = self._grad_tensors[g_cursor]
          g_cursor += 1
    assert s_cursor == len(self._new_state_tensor)
    # TODO: BETA
    if hub.use_rtrl:
      assert g_cursor == len(self._grad_tensors)

  def _easy_neurons(self, x, name, f, output_dim=None, use_bias=True):
    assert name is not None and callable(f)
    x_size = self._get_external_shape(x)
    dim = self._state_size if output_dim is None else output_dim

    with tf.variable_scope(name):
      bias = self._get_bias('bias', dim) if use_bias else None
      W = self._get_variable('W', shape=[x_size, dim])
      net = tf.nn.bias_add(tf.matmul(x, W), bias)
      return f(net)

  # endregion : Private Methods

  # region : Customized Ops

  # TODO: can be merged into a single method

  @staticmethod
  @tf.custom_gradient
  def _truncate_matmul(x, W):
    assert len(x.shape) == len(W.shape) == 2
    y = tf.matmul(x, W)
    def grad(dy):
      dx = tf.zeros_like(x)
      # dW = tf.gradients(y, W, grad_ys=dy)[0]
      dW = tf.matmul(tf.transpose(x), dy)
      return dx, dW
    return y, grad

  @staticmethod
  @tf.custom_gradient
  def _truncate_multiply(x, W):
    """x is usually larger than W"""
    x_shape = x.shape.as_list()
    W_shape = W.shape.as_list()
    assert len(x_shape) == len(W_shape) == 2 and W_shape[0] == 1
    y = tf.multiply(x, W)
    def grad(dy):
      dx = tf.zeros_like(x)
      # dx = tf.multiply(dy, W)
      # dW = tf.gradients(y, W, grad_ys=dy)[0]
      dW = tf.reduce_sum(tf.multiply(dy, x), axis=0, keep_dims=True)
      return dx, dW
    return y, grad

  # endregion : Customized Ops

