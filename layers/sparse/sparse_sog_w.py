from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import checker
from tframe.layers.hyper.hyper_base import HyperBase


class SparseSOG(HyperBase):

  full_name = 'sparse_sog_w'
  abbreviation = 'sparsogw'

  def __init__(
      self,
      num_neurons,
      group_size,
      axis=0,
      activation=None,
      use_bias=True,
      weight_initializer='xavier_normal',
      bias_initializer='zeros',
      **kwargs):
    """
    axis: 0 or 1.
    axis = 0: Partition input neurons.
              Each output neuron uses only 1 input activation in each group.
    axis = 1: Partition output neurons.
              Each input neuron passes value to only 1 output neuron in each
              group.
    """
    # Call parent's constructor
    super().__init__(activation, weight_initializer, use_bias,
                     bias_initializer, **kwargs)

    # Specific attributes
    assert axis in (0, 1)
    self._axis = axis
    self._num_neurons = checker.check_positive_integer(num_neurons)
    self._group_size = checker.check_positive_integer(group_size)


  @property
  def structure_tail(self):
    activation = ''
    if self._activation is not None:
      activation = '->act'
      if isinstance(self._activation_string, str):
        activation = '->' + self._activation_string
    if self._group_size == 1:
      return '({})'.format(self._num_neurons) + activation
    return '({}|P{}-S{})'.format(
      self._num_neurons, 'IO'[self._axis], self._group_size) + activation


  def forward(self, x, **kwargs):
    return self.sparse_sog(
      self._num_neurons, self._group_size, x, scope='psi_sparse_sog',
      activation=self._activation, axis=self._axis)
