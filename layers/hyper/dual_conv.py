from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from .conv import ConvBase


class DualConv2D(ConvBase):

  full_name = 'dualconv2d'
  abbreviation = 'duconv2d'

  class Configs(ConvBase.Configs):
    kernel_dim = 2

  def forward(self, x: tf.Tensor, filter=None, **kwargs):
    # Currently only hyper filter is supported
    assert isinstance(filter, tf.Tensor)

    # Generate dual kernel
    dual = tf.sqrt(1.0 - tf.square(filter))

    # Convolve
    y_1 = self.conv2d(
      x, self.channels, self.kernel_size, 'DualReal', strides=self.strides,
      padding=self.padding, dilations=self.dilations, filter=filter, **kwargs)

    y_2 = self.conv2d(
      x, self.channels, self.kernel_size, 'DualImag', strides=self.strides,
      padding=self.padding, dilations=self.dilations, filter=dual, **kwargs)

    return tf.sqrt(tf.square(y_1) + tf.square(y_2))
