from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from collections import OrderedDict

import tframe as tfr
from tframe import hub
from tframe import context
from tframe import checker
from tframe import linker
from tframe.nets.net import Net
from tframe.utils.misc import ravel_nested_stuff


class RNet(Net):
  """Recurrent net which outputs states besides common result"""
  net_name = 'rnet'
  # If not None, compute_gradients should be callable, it accepts a dL/dy
  # .. tensor with shape (num_steps(=1), batch_size, *(y.shape))
  # .. and returns (grads_and_vars, new_grad_buffer)
  compute_gradients = None

  MEMORY_TENSOR_DICT = 'MEMORY_TENSOR_DICT'

  def __init__(self, name):
    # Call parent's constructor
    Net.__init__(self, name)

    # Attributes
    self._inter_type = self.RECURRENT
    self._state_array = None
    self._state_size = None
    self._init_state = None
    self._weights = None
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

    # Gate activations should be registered here
    self._gate_dict = OrderedDict()

  # region : Properties

  @property
  def rnn_cells(self):
    assert self.is_root
    return [net for net in self.children if isinstance(net, RNet)]

  @property
  def rnn_cell_num(self):
    return len(self.rnn_cells)

  @property
  def memory_block_num(self):
    return sum([1 if not isinstance(cell.init_state, (tuple, list))
                else len(cell.init_state)
                for cell in self.rnn_cells])

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

      # if hub.use_rtrl or hub.export_tensors_to_note:
      #   # TODO: BETA & ++export_tensors
      #   assert len(pre_outputs) == 3
      # else: assert len(pre_outputs) == 2
      pre_states = pre_outputs[1]

      # TODO: ++export_tensors
      if hub.export_dy_ds: self._register_memories(pre_states)

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

    # Register gates
    self._register_gates()

    # Add logits if necessary
    if self.logits_tensor is not None:
      result_tuple += self.logits_tensor,

    # TODO: ++export_tensors
    if hub.export_tensors_to_note:
      y = self.logits_tensor if self.logits_tensor is not None else output
      result_tuple += self._get_tensors_to_export(y),

    # Add extra loss to result
    extra_loss = self._get_extra_loss()
    if extra_loss is not None:
      result_tuple += extra_loss,

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

  def _register_gates(self):
    assert self.is_root
    for cell in self.rnn_cells:
      for k, g in cell._gate_dict.items():
        if hub.export_gates: context.add_tensor_to_export(k, g)
        if hub.train_gates:
          coef = hub.gate_loss_strength
          context.add_loss_tensor(tf.multiply(
            coef, tf.reduce_sum(g), name='{}_loss'.format(k)))

  # endregion : Private Methods

  # region : Link tools

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

  def neurons(self, x, s=None, num=None, fc_memory=True,
              is_gate=False, activation=None,
              scope=None, truncate=False,
              num_or_size_splits=None,
              weight_initializer=None,
              use_bias=None,
              bias_initializer=None,
              weight_regularizer=None,
              bias_regularizer=None,
              activity_regularizer=None,
              **kwargs):
    if num is None:
      multiplier = (1 if not isinstance(num_or_size_splits, int)
                    else num_or_size_splits)
      num = multiplier * self._state_size
    if activation is None:
      activation = tf.sigmoid if is_gate else tf.tanh
    if weight_initializer is None:
      weight_initializer = getattr(self, '_weight_initializer', None)
    if use_bias is None:
      use_bias = getattr(self, '_use_bias', None)
    if bias_initializer is None:
      bias_initializer = getattr(self, '_bias_initializer')
    if weight_regularizer is None:
      weight_regularizer = getattr(self, '_weight_regularizer', None)
    if bias_regularizer is None:
      bias_regularizer = getattr(self, '_bias_regularizer', None)
    if activity_regularizer is None:
      activity_regularizer = getattr(self, '_activity_regularizer', None)
    return linker.neurons(
      num=num, external_input=x, activation=activation,
      memory=s, fc_memory=fc_memory, scope=scope,
      use_bias=use_bias, truncate=truncate,
      num_or_size_splits=num_or_size_splits,
      weight_initializer=weight_initializer,
      bias_initializer=bias_initializer,
      weight_regularizer=weight_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      **kwargs)

  # region : To be superceded

  def _neurons_forward(
      self, x, name, f, output_dim=None, use_bias=True, truncate=False,
      w_reg=None):
    assert name is not None and callable(f)
    dim = self._state_size if output_dim is None else output_dim
    return linker.neurons(
      dim, x, activation=f, use_bias=use_bias, truncate=truncate, scope=name,
      weight_regularizer=w_reg)

  def _neurons_forward_with_memory(
      self, x, s, name, f, fc_mem, output_dim=None, use_bias=True,
      truncate=False, w_reg=None):
    # If fully connect memory
    if fc_mem:
      return self._neurons_forward(
        tf.concat([x, s], axis=1), name, f, output_dim, use_bias, truncate,
        w_reg=w_reg)
    # Otherwise
    dim = self._state_size if output_dim is None else output_dim
    return linker.neurons(
      dim, x, activation=f, memory=s, fc_memory=fc_mem, use_bias=use_bias,
      scope=name, truncate=truncate, weight_regularizer=w_reg)

  # endregion : To be superceded

  # endregion : Link tools

  # region : Real-time recurrent learning

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

      if (isinstance(child, RNet) and
          getattr(child, 'compute_gradients', None) is not None):
        rcs.append(child)
    return tuple(rcs)

  def _get_grad_tensor_tuple(self, y):
    """TODO: BETA
    :param y: output tensor (B, D)
    :return: a tuple: (dy/dR1, ..., dy/dRn)
    """
    assert self.is_root
    containers = self.repeater_containers
    assert len(containers) > 0

    # mascot = tf.placeholder(dtype=tf.float32)

    grad_tensors = []
    # y_size = y.get_shape().as_list()[1]
    y_list = tf.split(y, num_or_size_splits=y.shape[1], axis=1)

    for r in [getattr(c, 'repeater_tensor') for c in containers]:
      # r.shape is (B, D_r) while y.shape is (B, D_y)
      assert isinstance(r, tf.Tensor)
      dy_dr = []
      for yi in y_list:
        dyi_dr = tf.gradients(yi, r)[0]  # (B, R)
        dy_dr.append(tf.expand_dims(dyi_dr, 1))  # (B, 1, R)
      dy_dr = tf.concat(dy_dr, axis=1)  # (B, D, R)
      grad_tensors.append(dy_dr)
    return tuple(grad_tensors)

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
      return tuple(buffer)

  # endregion : Real-time recurrent learning

  # region : Export tensors TODO ++export_tensor

  @property
  def num_tensors_to_export(self):
    # For dy/dS
    if hub.use_default_s_in_dy_ds:
      num_dydS = self.memory_block_num
    else:
      # TODO: this doesn't work since rnet has not been built yet
      num_dydS = len(context.get_collection_by_key(
      RNet.MEMORY_TENSOR_DICT, True, val_type=dict))
    # For gates
    num_gates = 0
    if hub.export_gates:
      for cell in self.rnn_cells:
        if hasattr(cell, 'gate_number'): num_gates += cell.gate_number
    # Return
    return num_dydS + num_gates

  @staticmethod
  def _register_memories(pre_states):
    """Register memory tensors as a dict into tfr.collections"""
    if not hub.use_default_s_in_dy_ds: return
    assert isinstance(pre_states, (list, tuple))
    # d = OrderedDict()
    for tensor, index in zip(
        *ravel_nested_stuff(pre_states, with_indices=True)):
      assert isinstance(index, list)
      key = 'S{}'.format('-'.join([str(i + 1) for i in index]))
      context.add_to_dict_collection(context.S_IN_DYDS, key, tensor)

  @staticmethod
  def _get_tensors_to_export(output):
    tensors = []
    # For tensors already registered
    tensors_to_export = context.tensors_to_export
    for t_name, t in tensors_to_export.items():
      tensors.append(t)
    # For tensors need to be calculated using output
    # For dy/dS
    for s_name, s in context.get_collection_by_key(
        context.S_IN_DYDS, True, val_type=dict).items():
      tensor = tf.gradients(output, s, name=s_name)[0]
      tensors.append(tensor)
      key = 'dy/d{}'.format(s_name)
      context.add_tensor_to_export(key, None)
    return tuple(tensors)

  @staticmethod
  def _set_tensors_to_export(scan_output):
    tensor_dict = context.tensors_to_export
    for key, tensor in zip(tensor_dict.keys(), scan_output):
      tensor_dict[key] = tensor

  # endregion : Export tensors

