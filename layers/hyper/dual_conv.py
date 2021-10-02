from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe import tf

from .conv import ConvBase


class DualConv2D(ConvBase):

  full_name = 'dualconv2d'
  abbreviation = 'duconv2d'

  class Configs(ConvBase.Configs):
    kernel_dim = 2

  def forward(self, x: tf.Tensor, filter=None, **kwargs):
    # Currently only hyper filter is supported
    assert isinstance(filter, (list, tuple))

    # Generate dual kernel
    # dual = tf.sqrt(1.0 - tf.square(filter))
    real, imag = filter
    assert isinstance(real, tf.Tensor)
    N = np.prod(real.shape.as_list()[1:3])

    # Convolve
    y_1 = self.conv2d(
      x, self.channels, self.kernel_size, 'DualReal', strides=self.strides,
      padding=self.padding, dilations=self.dilations, filter=real, **kwargs)

    y_2 = self.conv2d(
      x, self.channels, self.kernel_size, 'DualImag', strides=self.strides,
      padding=self.padding, dilations=self.dilations, filter=imag, **kwargs)

    return tf.sqrt(tf.square(y_1) + tf.square(y_2)) / N
