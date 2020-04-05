from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import context
from tframe import checker
from tframe import hub as th
from tframe.activations import sog
from tframe.layers.hyper.hyper_base import HyperBase


class SparseSOG(HyperBase):

  full_name = 'sparse_sog_n'
  abbreviation = 'sparsogn'

  def __init__(
      self,
      num_neurons,
      group_size,
      head_size=-1,
      activation=None,
      use_bias=True,
      weight_initializer='xavier_normal',
      bias_initializer='zeros',
      **kwargs):
    """
    Softmax over groups applied to neurons.
    Case 1: head_size < 0: does not use extra neurons
    Case 2: head_size = 0: use extra neurons without a head
    Case 3: head_size > 0: use extra neurons with a head
    """
    # Call parent's constructor
    super().__init__(activation, weight_initializer, use_bias,
                     bias_initializer, **kwargs)

    # Specific attributes
    self._num_neurons = checker.check_positive_integer(num_neurons)
    self._group_size = checker.check_positive_integer(group_size)
    self._head_size = checker.check_type(head_size, int)

    # Developer options
    options = th.developer_options


  @property
  def structure_tail(self):
    activation = ''
    if self._activation is not None:
      activation = '->act'
      if isinstance(self._activation_string, str):
        activation = '->' + self._activation_string
    if self._group_size == 1:
      return '({})'.format(self._num_neurons) + activation
    return '({}|S{}H{})'.format(
      self._num_neurons, self._group_size, self._head_size) + activation


  def forward(self, x, **kwargs):
    # Densely forward
    y_bar = self.dense_v2(self._num_neurons, 'y_bar', x)

    # Calculate SOG gate
    # .. get seed
    if self._head_size < 0: net_gate = y_bar
    elif self._head_size == 0:
      net_gate = self.dense_v2(self._num_neurons, 'seed', x)
    else:
      head = self.dense_v2(self._head_size, 'head', x)
      net_gate = self.dense_v2(self._num_neurons, 'seed', head)
    gate = sog(net_gate, self._group_size)

    # Export gates if necessary
    if th.export_gates: context.add_tensor_to_export('sog_gate', gate)

    # Apply gate
    y = tf.multiply(y_bar, gate, 'y')
    # ~
    return y
