from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe.layers.layer import Layer
from tframe.layers.layer import single_input

from tframe import pedia


class Quadric(Layer):
  """Quadric layer"""
  is_nucleus = True

  full_name = 'quadric'
  abbreviation = 'quad'

  def __init__(self):
    self.weights = None
    self.neuron_scale = [1]

  @single_input
  def _link(self, input_, **kwargs):
    assert isinstance(input_, tf.Tensor)
    if self.weights is not None: tf.get_variable_scope().reuse_variables()
    # Get the shape and data type of input
    dim = input_.get_shape().as_list()[1]
    # Get variable
    self.weights = tf.get_variable('weights', shape=(dim, dim))
    # Calculate output
    tmp = tf.matmul(input_, self.weights)
    output = tf.reduce_sum(
      tmp * input_, axis=1, keep_dims=True, name='quad_output')
    return output


if __name__ == '__main__':
  x = tf.constant([1, 2], shape=(1, 2))
  A = tf.constant([[1, 2], [3, 4]])
  y = tf.matmul(x, A)
  s = tf.matmul(y, x, transpose_b=True)
  ss = tf.reduce_sum(y * x)
  sess = tf.Session()
  print(sess.run(y))
  print(sess.run(ss))



