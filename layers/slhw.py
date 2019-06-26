from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import checker, linker
from tframe import hub, context
from tframe import initializers
from tframe.layers.layer import LayerWithNeurons
from tframe.operators.apis.distributor import Distributor
from tframe.operators.apis.hard_driver import HardDriver


class SLHighway(LayerWithNeurons, HardDriver):

  full_name = 'sl-highway'
  abbreviation = 'slhw'

  def __init__(
      self,
      config_string,
      num_layers,
      head_size,
      activation='tanh',
      use_bias=True,
      weight_initializer='xavier_normal',
      bias_initializer='zeros',
      **kwargs):

    # Call parent's constructor
    LayerWithNeurons.__init__(self, activation, weight_initializer, use_bias,
                              bias_initializer, **kwargs)
    HardDriver.__init__(self, config_string, head_size)

    self._num_layers = checker.check_positive_integer(num_layers)
    self._activation_string = activation
    self.output_scale = [self.total_size]


  @property
  def structure_tail(self):
    tail = '{}H{}'.format(self.group_string, self._arm_size)
    if self._activation_string:
      tail = '{}=>{}'.format(tail, self._activation_string)
    return '(({})x{})'.format(tail, self._num_layers)


  def forward(self, x, **kwargs):
    """input x -> read_head -> read -> write_head -> write
    """
    # Sanity check
    assert self.get_dimension(x) == self.total_size

    # Iteration
    for i in range(self._num_layers):
      scope = lambda pre: '{}_{}'.format(pre, i + 1)
      head = self.dense(self._arm_size, x, scope('head'))
      # Read
      x_hat = self._read(head, x, scope('read'))
      h = self.dense(self.num_groups, x_hat, scope('h'), self._activation)
      # Write
      x = self._write(head, x, h, scope('write'))

    return x



