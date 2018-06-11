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

  def __init__(self, name):
    # Call parent's constructor
    Net.__init__(self, name)

    # Attributes
    self._state_array = None
    self._state_size = None
    self._init_state = None
    self._kernel = None
    self._bias = None
    self._weight_initializer = None
    self._bias_initializer = None

  # region : Properties

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
      return tuple(states)
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
      assert isinstance(pre_outputs, tuple) and len(pre_outputs) == 2
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
    return output, tuple(states)

  # endregion : Overriden Methods

  # region : Public Methods

  def reset_state(self, batch_size):
    assert self.is_root
    self._state_array = self._get_zero_state(batch_size)

  def reset_part_state(self, indices, values=None):
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

  # endregion : Public Methods

  # region : Private Methods

  def _get_zero_state(self, batch_size):
    if self.is_root:
      state_array = []
      for rnn_cell in self.rnn_cells:
        state_array.append(rnn_cell._get_zero_state(batch_size))
      return tuple(state_array)
    else:
      # If state is a tuple, this method must be overriden
      assert self._state_size is not None
      state = np.zeros(shape=(batch_size, self._state_size))
      return state

  def _get_state_dict(self, batch_size=None):
    assert self.is_root

    if batch_size is None:
      # During training
      state = self._state_array
      assert state is not None
    else:
      # While is_training == False
      checker.check_positive_integer(batch_size)
      state = self._get_zero_state(batch_size)

    return {self.init_state: state}

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
    assert self._weight_initializer is not None
    return tf.get_variable(
      name, shape, dtype=hub.dtype, initializer=self._weight_initializer)

  def _get_bias(self, name, dim):
    assert self._bias_initializer is not None
    return tf.get_variable(
      name, shape=[dim], dtype=hub.dtype, initializer=self._bias_initializer)

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

  # endregion : Private Methods

