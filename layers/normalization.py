from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.layers.normalization import BatchNorm as _BarchNorm

from .layer import Layer
from .layer import single_input

from ..utils import get_scale

from .. import pedia


class BatchNormalization(Layer, _BarchNorm):
  full_name = 'batchnorm'
  abbreviation = 'bn'

  @single_input
  def __call__(self, input_=None, **kwargs):
    assert isinstance(input_, tf.Tensor)
    return _BarchNorm.__call__(self, input_, scope=self.full_name,
                                training=pedia.memo[pedia.is_training])
