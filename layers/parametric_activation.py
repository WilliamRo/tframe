from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.layers.layer import Layer
from tframe.layers.layer import single_input

from tframe import initializers


class Polynomial(Layer):
  """Polynomial activation function. ref: Vasilis Z. 1997"""
  is_nucleus = True

  full_name = 'polynomial'
  abbreviation = 'poly'

  def __init__(self, order, initializer='xavier_uniform'):
    # Check input
    if order < 0: raise TypeError('!! order must be a non-negative integer')

    self.order = order
    self.coefs = []
    self._initializer = initializers.get(initializer)
    tail = '{}'.format(order)
    self.full_name += tail


  @single_input
  def _link(self, input_, **kwargs):
    assert isinstance(input_, tf.Tensor)
    if len(self.coefs) > 0: tf.get_variable_scope().reuse_variables()
    # Get input dimension
    D = input_.get_shape().as_list()[1]
    self.neuron_scale = [D]
    # Get variable and calculate output
    order_list = []
    x = tf.ones_like(input_)
    for order in range(self.order + 1):
      coefs = tf.get_variable('coefs_{}'.format(order), shape=[1, D],
                              dtype=input_.dtype, initializer=self._initializer)
      self.coefs.append(coefs)
      order_list.append(tf.multiply(coefs, x))
      if order != self.order:
        x = tf.multiply(x, input_, name='x_{}'.format(order + 1))

    return tf.add_n(order_list)



