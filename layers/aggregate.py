from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.layers.layer import Layer
from tframe.layers.layer import single_input


class Sum(Layer):

  abbreviation = 'sum'
  full_name = abbreviation

  def __init__(self, axis=-1):
    self._axis = axis

  @property
  def structure_tail(self):
    return '(ax={})'.format(self._axis)

  @single_input
  def _link(self, x, **kwargs):
    return tf.reduce_sum(x, axis=self._axis)
