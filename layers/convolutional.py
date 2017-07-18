from __future__ import absolute_import

import tensorflow as tf

from .layer import Layer
from .layer import single_input

from tensorflow.python.layers.convolutional import Conv2D as Conv2D_


class Conv2D(Layer, Conv2D_):
  """"""
  is_nucleus = True

  full_name = 'convolutional2d'
  abbreviation = 'conv'

  @single_input
  def _link(self, input_, **kwargs):
    assert isinstance(input_, tf.Tensor)
    return Conv2D_.__call__(self, input_, scope=self.full_name)


