from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

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
      sum_method=None,
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
    self._sum_method = sum_method
    self._normalize = checker.check_type(normalize, bool)
    self.neuron_scale = [num_classes]


  @property
  def structure_tail(self):
    token = 'sum' if self._sum_method is None else self._sum_method
    return '({}h>{}>{})'.format(self._heads, token, self.num_classes)


  def forward(self, x, **kwargs):
    # bit_max.shape = ([heads, ]batch_size, num_classes)
    bit_max = linker.bit_max(
      x, self.num_classes, heads=self._heads, sum_heads=False,
      weight_initializer=self._weight_initializer,
      use_bias=self._use_bias, bias_initializer=self._bias_initializer)

    # Calculate weighted-sum of bit_max
    if self._heads > 1:
      if self._sum_method is None:
        # simply sum up if sum_method is not provided
        bit_max = tf.reduce_sum(bit_max, axis=0)
        division_ = self._heads
      else:
        assert isinstance(self._sum_method, str)
        # weights.shape = (batch_size, heads)
        weights = self.neurons(
          x, self._heads, activation=self._sum_method, scope='head_weight')
        # weights.shape => (heads, batch_size, 1)
        weights = tf.reshape(tf.transpose(weights), [self._heads, -1, 1])
        # calculate weighted sum
        bit_max = tf.reduce_sum(tf.multiply(weights, bit_max), axis=0)
        # calculate division if necessary
        if self._normalize: division_ = tf.reduce_sum(weights, axis=0)

    # Normalize if necessary
    # bit_max.shape = (batch_size, num_classes)
    if self._heads > 1 and self._normalize:
      bit_max = tf.divide(bit_max, division_)

    return bit_max

