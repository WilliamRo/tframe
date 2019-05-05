from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import checker, linker
from tframe import hub, context
from tframe.layers.layer import LayerWithNeurons


class BitMax(LayerWithNeurons):

  full_name = 'bitmax'
  abbreviation = 'bitmax'

  def __init__(
      self,
      num_classes,
      heads=1,
      normalize=False,
      use_bias=True,
      weight_initializer='xavier_normal',
      bias_initializer='zeros',
      **kwargs):
    # Call parent's constructor
    LayerWithNeurons.__init__(
      self, None, weight_initializer, use_bias, bias_initializer, **kwargs)

    self.num_classes = checker.check_positive_integer(num_classes)
    self._heads = checker.check_positive_integer(heads)
    self._normalize = checker.check_type(normalize, bool)
    self.neuron_scale = [num_classes]


  @property
  def structure_tail(self):
    return '({}h>{})'.format(self._heads, self.num_classes)


  def forward(self, x, **kwargs):
    bit_max = linker.bit_max(
      x, self.num_classes, heads=self._heads,
      weight_initializer=self._weight_initializer,
      use_bias=self._use_bias, bias_initializer=self._bias_initializer)
    if self._heads and self._normalize:
      bit_max = tf.divide(bit_max, self._heads)
    return bit_max

