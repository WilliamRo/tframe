from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf

from collections import OrderedDict

import tframe as tfr
from tframe import hub
from tframe import console
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
    self._train_state_buffer = None
    self._eval_state_buffer = None
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
  def loss_in_loop(self):
    from tframe.models import Predictor
    return isinstance(self, Predictor) and hub.allow_loss_in_loop

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
      if hub.export_dy_ds or hub.export_states:
        self._register_memories(pre_states)

      # The assertion below is not held by rnn_cells
      assert isinstance(pre_states, (tuple, list))
      assert len(pre_states) == self.rnn_cell_num
    else: raise ValueError('!! pre_outputs can not be None')  # TODO

    # Unwrap input_
    inputs, targets = None, None
    if isinstance(input_, (list, tuple)):
      assert len(input_) == 2
      inputs, targets = input_
    else:
      assert isinstance(input_, tf.Tensor)
      inputs = input_
    assert isinstance(inputs, tf.Tensor)
    # Here if `targets` is not None, self.loss_in_while_loop must be True

    # Link
    states = []
    outputs = inputs
    state_cursor = 0
    for net in self.children:
      assert isinstance(net, Net)
      if isinstance(net, RNet):
        # rnn_cells in self.children accept state and input
        # .. and gives (output, state)
        outputs, state = net(pre_states[state_cursor], outputs)
        states.append(state)
        state_cursor += 1
      else:
        outputs = net(outputs)

    assert len(states) == len(pre_states)
    # Set y for future use (4 & 7 below)
    # Use logits if possible for logits provide larger slope
    y = self.logits_tensor if self.logits_tensor is not None else outputs

    # The order from 1 to 6 should be the same as that in recurrent.py
    # 1&2. outputs and states
    result_tuple = outputs, tuple(states)

    # Register gates
    self._register_gates()

    # 3. Add logits if necessary
    if self.logits_tensor is not None:
      result_tuple += self.logits_tensor,

    # 4. TODO: ++export_tensors
    if hub.export_tensors_to_note:
      result_tuple += self._get_tensors_to_export(y),

    # 5. Add extra loss to result
    extra_loss = self._get_extra_loss()
    if extra_loss is not None:
      result_tuple += extra_loss,

    # 6. TODO: BETA to be deprecated
    if hub.use_rtrl:
      result_tuple += self._get_grad_tensor_tuple(outputs),

    # 7. Loss related tensors
    if targets is not None:
      # Handle while-free situation
      if 'mascot' in targets.name: targets = y
      assert self.loss_in_loop and callable(context.loss_function)
      # Raw loss does not need to be exported
      loss = context.loss_function(targets, y)
      # result_tuple += loss,

      # dl_t/dx_{t-1}
      if hub.export_dl_dx or hub.export_dl_ds_stat:
        result_tuple += self._calc_dL_dS_prev(loss, pre_states),

    # 8. Jacobian dx_t/dx_{t-1}
    if (hub.export_dl_dx or hub.export_dl_ds_stat or
        hub.export_jacobian_norm):
      result_tuple += self._calc_dS_dS_prev(states, pre_states),

    # Run build-in extractors
    if not kwargs.get('pseudo', False): self.variable_extractor()

    return result_tuple

  # endregion : Overriden Methods

  # region : Public Methods

  def set_buffers(self, state_array, is_training=True):
    if hub.state_nan_protection:
      state_array = self._apply_to_nested_array(
        state_array, self._reset_on_nan)
    if is_training: self._train_state_buffer = state_array
    else: self._eval_state_buffer = state_array

  def reset_buffers(self, batch_size, is_training=True):
    assert self.is_root
    if is_training: self._train_state_buffer = self._get_zero_state(batch_size)
    else: self._eval_state_buffer = self._get_zero_state(batch_size)
    if 'reset_buffer' in hub.verbose_config:
      prefix = 'Train' if is_training else 'Eval'
      console.show_status(
        '{} state buffer has been reset.'.format(prefix), '[RNet]')
    # TODO: BETA
    assert not hub.use_rtrl
    if hub.use_rtrl:
      self._gradient_buffer_array = self._get_zero_gradient_buffer(
        batch_size)

  def decrease_buffer_size(self, indices, is_training):
    assert self.is_root and isinstance(indices, (list, tuple))

    def _decrease(state):
      assert isinstance(state, np.ndarray) and len(state) > len(indices)
      return state[np.array(indices)]

    # def _decrease(state):
    #   if isinstance(state, np.ndarray):
    #     assert len(state) > len(indices)
    #     return state[np.array(indices)]
    #   elif isinstance(state, (list, tuple)):
    #     # tf.scan returns a list of states
    #     state = list(state)
    #     for i, s in enumerate(state): state[i] = _decrease(s)
    #     return tuple(state)
    #   else: raise TypeError('!! Unknown type of states: {}'.format(type(state)))
    # if is_training:
    #   self._train_state_buffer = _decrease(self._train_state_buffer)
    # else: self._eval_state_buffer = _decrease(self._eval_state_buffer)

    if is_training:
      self._train_state_buffer = self._apply_to_nested_array(
        self._train_state_buffer, _decrease)
    else: self._eval_state_buffer = self._apply_to_nested_array(
      self._eval_state_buffer, _decrease)

    # Display info if necessary
    if 'partial_reset' in hub.verbose_config:
      prefix = 'train' if is_training else 'eval'
      console.show_status(
        'Batch size of {} state buffer decreased'.format(prefix), '[RNet]')

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
      assert isinstance(state, np.ndarray)
      if len(zero_indices) > 0: state[np.array(zero_indices), :] = 0
      if len(none_indices) > 0: state = np.delete(state, none_indices, axis=0)
      return state

    # def _reset(state):
    #   if isinstance(state, np.ndarray):
    #     if len(zero_indices) > 0: state[np.array(zero_indices), :] = 0
    #     if len(none_indices) > 0: state = np.delete(state, none_indices, axis=0)
    #     return state
    #   elif isinstance(state, (list, tuple)):
    #     # tf.scan returns a list of states
    #     state = list(state)
    #     for i, s in enumerate(state): state[i] = _reset(s)
    #     return tuple(state)
    #   else:
    #     raise TypeError('!! Unknown type of states: {}'.format(type(state)))

    # self._train_state_buffer = _reset(self._train_state_buffer)
    self._train_state_buffer = self._apply_to_nested_array(
      self._train_state_buffer, _reset)
    # TODO: BETA
    if hub.use_rtrl:
      # self._gradient_buffer_array = _reset(self._gradient_buffer_array)
      self._gradient_buffer_array = self._apply_to_nested_array(
        self._gradient_buffer_array, _reset)

  # endregion : Public Methods

  # region : Private Methods

  @staticmethod
  def _reset_on_nan(state):
    assert isinstance(state, np.ndarray)
    if not np.isnan(state).any(): return state
    # Scan each batch
    for i, s in enumerate(state):
      if np.isnan(s).any():
        state[i] = 0.0
        if 'reset_buffer' in hub.verbose_config:
          console.show_status('state[{}] reset due to NaN'.format(i), '[RNet]')
    return state

  def _apply_to_nested_array(self, array, f):
    assert callable(f)
    if isinstance(array, np.ndarray):
      return f(array)
    elif isinstance(array, (list, tuple)):
      array = list(array)
      for i, a in enumerate(array):
        array[i] = self._apply_to_nested_array(a, f)
      return tuple(array)
    assert False

  def _get_zero_state(self, batch_size):
    if self.is_root:
      state_array = []
      for rnn_cell in self.rnn_cells:
        state_array.append(rnn_cell._get_zero_state(batch_size))
      return tuple(state_array)
    else:
      def zeros_like(state):
        if isinstance(state, tf.Tensor):
          shape_list = state.shape.as_list()
          shape_list[0] = batch_size
          return np.zeros(shape=shape_list)
        assert isinstance(state, (list, tuple))
        return tuple([zeros_like(s) for s in state])

      # Get zero state according to self.init_state
      return zeros_like(self.init_state)

  def _get_rnn_dict(self, is_training, batch_size=None):
    """Get state dict together with gradient buffer if necessary
    """
    assert self.is_root and isinstance(is_training, bool)

    rnn_dict = {}
    if is_training:
      state = self._train_state_buffer
      assert state is not None
      # TODO: BETA
      # if hub.use_rtrl:
      #   rnn_dict[self.gradient_buffer_placeholder] = self._gradient_buffer_array
    else:
      checker.check_positive_integer(batch_size)
      state = self._eval_state_buffer

    rnn_dict[self.init_state] = state
    return rnn_dict

  def _check_state(self, state, num_or_sizes=1):
    """Check cell state
    :param state: a tuple of states or a state array
    :param num_or_sizes: can be
        (1) an integer: number of tensor arrays of size `self._state_size`
        (2) a tuple: tuple([get_size(s) for s in `state tuple`])
    """
    # Check num_or_sizes
    if isinstance(num_or_sizes, (int, np.int32)):
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
      assert isinstance(s, tf.Tensor) and isinstance(size, (int, np.int32))
      if size == 1:
        assert len(s.shape.as_list()) == 1 or s.shape.as_list()[1] == 1
      else: assert s.shape.as_list()[1] == size

  @staticmethod
  def _get_size(tensor):
    assert isinstance(tensor, tf.Tensor)
    input_shape = tensor.shape.as_list()
    assert len(input_shape) == 2
    return input_shape[1]

  @staticmethod
  def _get_placeholder(name, size):
    if isinstance(size, (list, tuple)): shape = tuple([None] + list(size))
    else: shape = (None, checker.check_positive_integer(size))
    return tf.placeholder(dtype=hub.dtype, shape=shape, name=name)

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

  def add_bias(self, x, bias_initializer=None, name='unnamed'):
    if bias_initializer is None:
      bias_initializer = getattr(self, '_bias_initializer')
    initializer = linker.initializers.get(bias_initializer)
    bias = linker.get_bias(name + 'bias', linker.get_dimension(x), initializer)
    return tf.nn.bias_add(x, bias, name=name + '_add_bias')

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
      if isinstance(num_or_size_splits, int):
        num = num_or_size_splits * self._state_size
      elif isinstance(num_or_size_splits, (list, tuple)):
        num = sum(num_or_size_splits)
      else: num = self._state_size
    if activation is None and is_gate:
      activation = tf.sigmoid
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
    _kwargs = getattr(self, '_kwargs', {})
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
      **_kwargs, **kwargs)

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

  @staticmethod
  def _register_memories(pre_states):
    """Register memory tensors as a dict into tfr.collections"""
    if not hub.use_default_s_in_dy_ds and not hub.export_states: return
    assert isinstance(pre_states, (list, tuple))
    tensors, indices = ravel_nested_stuff(pre_states, with_indices=True)

    for tensor, index in zip(tensors, indices):
      assert isinstance(index, list)
      if len(tensors) == 1: key = 'S'
      else: key = 'S{}'.format('-'.join([str(i + 1) for i in index]))
      if hub.export_states:
        context.add_tensor_to_export(key, tensor)
      if hub.use_default_s_in_dy_ds:
        context.add_to_dict_collection(context.S_IN_DYDS, key, tensor)

  @staticmethod
  def _get_tensors_to_export(y):
    """In RNN, tensors, not variables, should be set to context outside
       while_loop"""
    tensors = []
    # For tensors already registered
    tensors_to_export = context.tensors_to_export
    for t_name, t in tensors_to_export.items():
      tensors.append(t)
    # For tensors need to be calculated using output
    # For dy/dS
    for s_name, s in context.get_collection_by_key(
        context.S_IN_DYDS, True, val_type=dict).items():
      y_shape = y.shape.as_list()
      assert len(y_shape) == 2
      # y here must be splitted
      for i in range(y_shape[1]):
        tensor = tf.gradients(y[0, i], s, name=s_name)[0]
        tensors.append(tensor)
        key = 'dy{}/d{}'.format(i + 1, s_name)
        context.add_tensor_to_export(key, None)

    return tuple(tensors)

  @staticmethod
  def _set_tensors_to_export(tensor_list):
    # TODO: the order of tensors in tensor_list somehow matches
    #       that in context
    tensor_dict = context.tensors_to_export
    for key, tensor in zip(tensor_dict.keys(), tensor_list):
      tensor_dict[key] = tensor

  # endregion : Export tensors

  # region : Export dL/dx

  def _calc_dL_dS_prev(self, loss, pre_states):
    """dS in dL/dS must be an integral whole"""
    assert isinstance(loss, tf.Tensor)
    dL_dS = tf.gradients(loss, ravel_nested_stuff(pre_states))
    assert isinstance(dL_dS, (tuple, list))
    if len(dL_dS) > 1: dL_dS = [tf.concat(dL_dS, axis=-1)]
    return tuple(dL_dS)

    # dL_dS = []
    # for state in pre_states:
    #   # state is a single tensor or a list of tensors
    #   checker.check_type(state, tf.Tensor)
    #   dL_dS.append(tuple(tf.gradients(loss, state)))
    # return tuple(dL_dS)

  def _calc_dS_dS_prev(self, states, pre_states):
    # Ravel states and pre_states
    assert isinstance(states, (tuple, list))
    assert isinstance(pre_states, (tuple, list))
    states = ravel_nested_stuff(states)
    pre_states = ravel_nested_stuff(pre_states)

    # Split states for calculating Jacobian later
    split_states = []
    for s in states: split_states += tf.split(s, s.shape[1], axis=-1)
    return (tf.stack(
      [tf.concat(tf.gradients(s, pre_states), axis=-1) for s in split_states],
      axis=-1),)

    # states & pre_states can be tuples/lists or tensors
    # if isinstance(states, tf.Tensor):
    #   # states.shape is [batch_size, state_size].
    #   assert isinstance(pre_states, tf.Tensor)
    #   assert len(pre_states.shape) == len(states.shape) == 2
    #   # each entry has a shape of [batch_size, 1]
    #   split_states = tf.split(states, states.shape[1], axis=1)
    #   # output has shape [batch_size, pre_s_size, s_size]
    #   # i.e. the output is a standard Jacobian
    #   return tf.stack(
    #     [tf.gradients(s, pre_states)[0] for s in split_states], axis=2)
    # else:
    #   assert isinstance(states, (tuple, list))
    #   assert isinstance(pre_states, (tuple, list))
    #   assert len(states) == len(pre_states)
    #   results = []
    #   for s, pre_s in zip(states, pre_states):
    #     results.append(self._calc_dS_dS_prev(s, pre_s))
    #   return tuple(results)

  # endregion : Export dL/dx

