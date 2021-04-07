from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import checker
from tframe.layers.convolutional import Conv2D, Deconv2D
from tframe.layers.pooling import MaxPool2D
from tframe.layers.merge import Bridge
from tframe.layers.normalization import BatchNormalization
from tframe.nets.classic.conv_nets.conv_net import ConvNet

from typing import List, Optional, Union


class UNet2D(ConvNet):

  def __init__(self, filters: Optional[int] = None,
               kernel_size: Optional[int] = 3,
               activation: str = 'relu',
               height: Optional[int] = 4,
               thickness: Optional[int] = 2,
               use_maxpool: bool = True,
               use_batchnorm: bool = False,
               link_indices: Union[List[int], str, None] = 'a',
               auto_crop=False, arc_string: Optional[str] = None):
    """This class provides some generalization to the traditional U-Net

    Example U-Net, height=3, thickness=2, link_indices=2,3

           Left Tower                                            Right Tower
                               bridge index 3
    3F  0   |->|->| --------------------------------------------> |->|->|
                  \                                              /
               contract                                      expand
                   \              bridge index 2              /
    2F  1          |->|->| --------------------------> |->|->|
                         \                            /
                      contract                    expand
                          \                        /
    1F  2                 |->|->|           |->|->|
                                \          /
                             contract  expand
                                 \      /
    GF                           |->|->|         # thickness=2 means 2 `->`s


    :param filters: initial filters, will be doubled after contracting, and
                    halved after expanding
    :param kernel_size: kernel size for each [De]Conv2D layer
    :param activation: activation used in each [Dd]Conv2D layer
    :param height: height of each tower
    :param thickness: number of convolutional layers used on each floor
    :param use_maxpool: whether to use MaxPool2D layer for contracting
    :param use_batchnorm: whether to use BatchNorm layer before activation
    :param link_indices: specifies the floor number to build bridge between
                         2 towers
    :param arc_string: architecture string, if provided, some of the arguments
                       will be overwrote
    """

    self.filters = filters
    self.kernel_size = kernel_size
    self.activation = activation
    self.height = height
    self.thickness = thickness
    self.use_maxpool = use_maxpool
    self.use_batchnorm = use_batchnorm
    self.link_indices = link_indices
    self.arc_string = arc_string
    self.auto_crop = auto_crop

    self.parse_arc_str_and_check()


  def _get_conv(self, filters, strides=1, transpose=False):
    Conv = Deconv2D if transpose else Conv2D
    return Conv(filters, self.kernel_size, strides, padding='same',
                activation=self.activation if strides == 1 else None,
                use_bias=False)


  def _get_layers(self):
    layers, floors = [], []

    # Define some utilities
    contract = lambda channels: layers.append(
      MaxPool2D(2, 2) if self.use_maxpool else self._get_conv(
        channels, strides=2))
    expand = lambda channels: layers.append(
      self._get_conv(channels, strides=2, transpose=True))

    # Build left tower for contracting
    filters = self.filters
    for i in range(self.height):   # (height - i)-th floor
      # Add front layers on each floor
      for _ in range(self.thickness):
        layers.append(self._get_conv(filters))
      # Remember the last layer in each floor before contracting
      floors.append(layers[-1])
      # Contract
      contract(filters)
      # Double filters
      filters *= 2

    # Build ground floor (GF)
    for _ in range(self.thickness):
      layers.append(self._get_conv(filters))

    # Build right tower for expanding
    for i in range(1, self.height + 1):    # i-th floor
      # Halve filters
      filters = filters // 2
      # Expand
      expand(filters)
      # Build a bridge if necessary
      if i in self.link_indices:
        guest_is_larger = None
        if self.auto_crop: guest_is_larger = not self.use_maxpool
        layers.append(Bridge(floors[self.height - i], guest_is_larger))
      # Increase thickness
      for _ in range(self.thickness):
        layers.append(self._get_conv(filters))

    return layers


  def parse_arc_str_and_check(self):
    """The format of arc_string is
      {filters}-{kernel_size}-{height}-{thickness}-[link_indices]-[mp]-[bn]
    in which
      {link_indices} can be `a` or `f` indicating linking all layers on the same
      floor, or indices separated by `,` indicating which floor to link, e.g.,
      `0,2,4`, in which case the given height must be greater than 4.
    """
    if self.arc_string is not None:
      options = self.arc_string.split('-')
      assert len(options) >= 5
      self.filters, self.kernel_size, self.height, self.thickness = [
        int(op) for op in options[:4]]
      self.activation = options[4]
      # For optional settings
      for op in options[5:]:
        if op in ('mp', 'maxpool'): self.use_maxpool = True
        elif op in ('bn', 'batchnorm'): self.use_batchnorm = True
        elif op in ('a', 'f', 'all', 'full'): self.link_indices = op
        else:
          # Parse link_indices with weak format checking
          assert ','in op and self.link_indices is None
          self.link_indices = [int(ind) for ind in op.split(',')]
          assert len(self.link_indices) == len(set(self.link_indices))

    # Check types
    checker.check_positive_integer(self.height)
    checker.check_positive_integer(self.filters)
    checker.check_type(self.activation, str)
    checker.check_positive_integer(self.kernel_size)
    checker.check_positive_integer(self.thickness)
    checker.check_type(self.use_maxpool, bool)
    checker.check_type(self.use_batchnorm, bool)
    if self.link_indices in (None, 'none', '-', ''):
      self.link_indices = []
    elif self.link_indices in ('a', 'f', 'all', 'full'):
      self.link_indices = list(range(1, self.height + 1))
    checker.check_type(self.link_indices, int)


  def __str__(self):
    result = '{}-{}-{}-{}-{}'.format(
      self.filters, self.kernel_size, self.height, self.thickness,
      self.activation)
    if len(self.link_indices) < self.height:
      result += '-' + ','.join([str(i) for i in self.link_indices])
    if self.use_maxpool: result += '-mp'
    if self.use_batchnorm: result += '-bn'
    return result


if __name__ == '__main__':
  unet = UNet2D(arc_string='64-3-4-3-relu-mp')
  print(unet)

