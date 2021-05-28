from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf

from tframe.layers.layer import Layer
from tframe.layers.layer import single_input

from tframe import pedia


class Homogeneous(Layer):
  """Homogeneous layer"""
  is_nucleus = True

  full_name = 'homogeneous'
  abbreviation = 'homo'
  MAX_ORDER = 5

  def __init__(self, order):
    # Check input
    if order not in range(1, self.MAX_ORDER + 1):
      raise TypeError('!! order must be an integer between 1 and {}'.format(
        self.MAX_ORDER))

    self.coefs = None
    self.order = order
    self.abbreviation = self.poly_name
    self.full_name = self.poly_name
    self.output_scale = [1]


  @property
  def poly_name(self):
    return {1: 'linear', 2: 'quadratic', 3: 'cubic',
             4: 'quartic', 5: 'quintic', 6: 'sixtic',
             7: 'septic'}[self.order]


  @single_input
  def _link(self, input_, **kwargs):
    assert isinstance(input_, tf.Tensor)
    if self.coefs is not None: tf.get_variable_scope().reuse_variables()
    # Get input dimension
    D = input_.get_shape().as_list()[1]
    self.neuron_scale = [D] * self.order
    # Get variable
    self.coefs = tf.get_variable('coefs', shape=(D,) * self.order)
    # Calculate output
    result = self.coefs
    for dim in range(self.order - 1, -1, -1):
      shape = [-1, D] + [1] * dim
      result = tf.reshape(input_, shape=shape) * result
      name = ('d{}'.format(dim) if dim > 0
              else '{}_output'.format(self.poly_name))
      result = tf.reduce_sum(result, axis=1, name=name, keep_dims=dim is 0)

    return result


if __name__ == '__main__':
  from tframe import console
  console.section('homogeneous.py test')

