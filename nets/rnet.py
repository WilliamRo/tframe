from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.nets.net import Net


class RecurrentNet(Net):
  """Recurrent net which outputs states besides common result"""
  def __init__(self, name):
    # Call parent's constructor
    Net.__init__(self, name)
    # Attributes

  # region : Properties

  @property
  def rnn_cell_num(self):
    assert self.is_root
    return len([net for net in self.children if isinstance(net, RecurrentNet)])


  def zero_state(self, batch_size):
    assert self.is_root
    states = []
    for child in self.children:
      if isinstance(child, RecurrentNet):
        states.append(child.zero_state(batch_size))
    assert len(states) == self.rnn_cell_num
    return states

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

