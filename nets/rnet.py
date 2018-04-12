from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import config
from tframe.nets.net import Net


class RecurrentNet(Net):
  """Recurrent net which outputs states besides common result"""
  def __init__(self, name):
    # Call parent's constructor
    Net.__init__(self, name)

    # Attributes
    self._state_size = None
    self._init_state = None

  # region : Properties

  @property
  def rnn_cells(self):
    assert self.is_root
    return [net for net in self.children if isinstance(net, RecurrentNet)]

  @property
  def rnn_cell_num(self):
    return len(self.rnn_cells)

  @property
  def init_state(self):
    if self.is_root:
      states = []
      with tf.name_scope('InitStates'):
        for rnn_cell in self.rnn_cells:
          states.append(rnn_cell.init_state)
      assert len(states) == self.rnn_cell_num
      return states
    else:
      # If initial state is a tuple or list, this property must be overrode
      if self._init_state is None:
        assert self._state_size is not None
        # The initialization of init_state must be done under with_graph
        # .. decorator
        self._init_state = tf.placeholder(
          dtype=config.dtype, shape=(None, self._state_size), name='init_state')
      return self._init_state

  # endregion : Properties

  # region : Methods Overrode

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
    else: pre_states = None
    assert isinstance(input_, tf.Tensor)

    # Link
    states = []
    output = input_
    state_cursor = 0
    for net in self.children:
      assert isinstance(net, Net)
      if isinstance(net, RecurrentNet):
        # rnn_cells in self.children accept state and input
        # .. and gives (output, state)
        output, state = net(pre_states[state_cursor], output)
        states.append(state)
        state_cursor += 1
      else:
        output = net(output)

    assert len(states) == len(pre_states)
    return output, states

  # endregion : Methods Overrode

  # region : Public Methods

  def get_state_dict(self, state=None, batch_size=None):
    """
    Get state dictionary
    :param state: if provided, batch_size will be ignored
    :param batch_size: if specified, a zero-state dictionary will be returned
    :return: state dictionary
    """
    if self.is_root:
      state_dict = {}
      if state is not None:
        assert len(state) == self.rnn_cell_num
        for i, rnn_cell in enumerate(self.rnn_cells):
          state_dict.update(rnn_cell.get_state_dict(state=state[i]))
      else:
        assert batch_size is not None
        for rnn_cell in self.rnn_cells:
          state_dict.update(rnn_cell.get_state_dict(batch_size=batch_size))
      return state_dict
    else:
      # If init state is a tuple or a list, this method must be overrode
      assert self._init_state is not None
      assert self._state_size is not None
      if state is None:
        assert batch_size is not None
        state = tf.zeros(shape=(batch_size, self._state_size),
                         dtype=config.dtype)
      return {self._init_state: state}

  # endregion : Public Methods

