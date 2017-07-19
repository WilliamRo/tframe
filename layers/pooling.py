from __future__ import absolute_import

import tensorflow as tf

from .layer import Layer
from .layer import single_input

from ..utils import get_scale

from tensorflow.python.layers.pooling import MaxPool2D as MaxPool2D_


class MaxPool2D(Layer, MaxPool2D_):
  """"""
  full_name = 'maxpool2d'
  abbreviation = 'maxpool'

  @single_input
  def __call__(self, input_=None, **kwargs):
    assert isinstance(input_, tf.Tensor)
    output = MaxPool2D_.__call__(self, input_, scope=self.full_name)
    self.neuron_scale = get_scale(output)
    return output

