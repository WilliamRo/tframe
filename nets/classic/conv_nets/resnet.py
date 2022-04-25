from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe.layers.advanced import Dense
from tframe.layers.convolutional import Conv2D
from tframe.layers.pooling import GlobalAveragePooling2D
from tframe.layers.common import Flatten, Dropout

from .conv_net import ConvNet


class ResNet(ConvNet):

  def __init__(self, num_blocks, in_channels=16, kernel_size=3):
    self.num_blocks = num_blocks
    self.in_channels = in_channels
    self.kernel_size = kernel_size


  def _get_layers(self):
    # Add first layer
    layers = [Conv2D(self.in_channels, kernel_size=self.kernel_size,
                     activation='relu', use_batchnorm=True, use_bias=False)]

    # Add residual blocks
    filters = self.in_channels
    for i, num_block in enumerate(self.num_blocks):
      # At the beginning of the big block (except the 1st one),
      #   contract feature size
      strides = [1 if i == 0 else 2] + [1] * (num_block - 1)
      for s in strides:
        shortcut = (
          Conv2D(filters, kernel_size=1, strides=s, use_batchnorm=True)
          if s != 1 else None)
        layers.append(self.residual_block(
          filters, self.kernel_size, s, shortcut=shortcut))

      # Double filters in the next big block
      filters *= 2

    # Add last few layers
    layers.append(GlobalAveragePooling2D())
    layers.append(Flatten())

    # Return layer list
    return layers

