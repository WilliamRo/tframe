from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import checker
from tframe.layers.merge import Bridge
from tframe.nets.classic.conv_nets.conv_net import ConvNet
from tframe.layers.hyper.conv import Conv1D, Conv2D, Conv3D
from tframe.layers.hyper.conv import Deconv1D, Deconv2D, Deconv3D
from tframe.layers.pooling import MaxPool1D, MaxPool2D, MaxPool3D

from typing import List, Optional, Union



class UNet(ConvNet):

  conv_classes = [Conv1D, Conv2D, Conv3D]
  deconv_classes = [Deconv1D, Deconv2D, Deconv3D]
  pool_classes = [MaxPool1D, MaxPool2D, MaxPool3D]

  def __init__(self,
               dimension: int,
               filters: Optional[int] = None,
               kernel_size: Optional[int] = 3,
               activation: str = 'relu',
               use_bias: bool = False,
               height: Optional[int] = 4,
               thickness: Optional[int] = 2,
               use_maxpool: bool = True,
               use_batchnorm: bool = False,
               link_indices: Union[List[int], str, None] = 'a',
               guest_first=True,
               bottle_neck_after_bridge = False,
               filter_generator=None,
               contraction_kernel_size: Optional[int] = None,
               expansion_kernel_size: Optional[int] = None,
               auto_crop=False,
               arc_string: Optional[str] = None):
    """This class provides a general U-Net API for 1D/2D/3D image processing.

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


    :param dimension: should be 1, 2, or 3, corresponding to 1D, 2D, and 3D
                      UNet, correspondingly
    :param filters: initial filters, will be doubled after contracting, and
                    halved after expanding
    :param kernel_size: kernel size for each [De]Conv layer
    :param activation: activation used in each [De]Conv layer
    :param use_bias: whether to use bias in conv layers
    :param height: height of each tower
    :param thickness: number of convolutional layers used on each floor
    :param use_maxpool: whether to use MaxPool layer for contracting
    :param use_batchnorm: whether to use BatchNorm layer before activation
    :param link_indices: specifies the floor number to build bridge between
                         2 towers
    :param guest_first: option to put output from left tower first while
                        concatenating
    :param bottle_neck_after_bridge: TODO
    :param filter_generator: function to generate conv kernel
    :param contraction_kernel_size: global kernel size used in left tower
    :param expansion_kernel_size: global kernel size used in right tower
    :param auto_crop: TODO currently can not be implemented
    :param arc_string: architecture string, if provided, some of the arguments
                       will be overwritten
    """

    self.dimension = dimension
    self.filters = filters
    self.kernel_size = kernel_size
    self.activation = activation
    self.use_bias = use_bias

    self.height = height
    self.thickness = thickness
    self.use_maxpool = use_maxpool
    self.use_batchnorm = use_batchnorm
    self.link_indices = link_indices
    self.arc_string = arc_string
    self.auto_crop = auto_crop
    self.filter_generator = filter_generator
    self.bottle_net_after_bridge = bottle_neck_after_bridge
    self.guest_first = guest_first

    self.parse_arc_str_and_check()

    self.contraction_kernel_size = (
      self.kernel_size if contraction_kernel_size is None
      else contraction_kernel_size)
    self.expansion_kernel_size = (
      self.kernel_size if expansion_kernel_size is None
      else expansion_kernel_size)

    # TODO: not working and there's no solution
    assert not auto_crop


  def _get_conv(self, filters, kernel_size, strides=1, transpose=False):
    Conv = (self.deconv_classes if transpose
            else self.conv_classes)[self.dimension - 1]

    # TODO: make sure this is an appropriate logic
    activation = self.activation if strides == 1 else None

    return Conv(filters, kernel_size, strides, padding='same',
                activation=activation, use_bias=self.use_bias)

  def _get_pooling(self): return self.pool_classes[self.dimension - 1](2, 2)


  def _get_layers(self):
    """Abstract method defined by ConvNet, returns a list of layers"""
    layers, floors = [], []

    # Define some utilities
    contract = lambda channels: layers.append(
      self._get_pooling() if self.use_maxpool
      else self._get_conv(channels, self.contraction_kernel_size, strides=2))
    expand = lambda channels: layers.append(self._get_conv(
      channels, self.expansion_kernel_size, strides=2, transpose=True))

    # (1) Build left tower for contracting
    filters = self.filters
    for i in range(self.height):   # (height - i)-th floor
      # Add front layers on each floor
      for _ in range(self.thickness):
        layers.append(self._get_conv(filters, self.contraction_kernel_size))
      # Register the last layer in each floor before contracting
      floors.append(layers[-1])
      # Contract
      contract(filters)
      # Double filters
      filters *= 2

    # (2) Build ground floor (GF)
    for _ in range(self.thickness):
      layers.append(self._get_conv(filters, self.expansion_kernel_size))

    # (3) Build right tower for expanding
    for i in range(1, self.height + 1):    # i-th floor
      # Halve filters
      filters = filters // 2
      # Expand
      expand(filters)
      # Build a bridge if necessary
      if i in self.link_indices:
        guest_is_larger = None
        if self.auto_crop: guest_is_larger = not self.use_maxpool
        layers.append(Bridge(floors[self.height - i], guest_is_larger,
                             guest_first=self.guest_first))
        if self.bottle_net_after_bridge:
          layers.append(self._get_conv(filters, kernel_size=1))
      # Increase thickness
      for _ in range(self.thickness):
        layers.append(self._get_conv(filters, self.expansion_kernel_size))

    return layers


  def parse_arc_str_and_check(self):
    """The format of arc_string is
      {filters}-{kernel_size}-{height}-{thickness}-[link_indices]-[mp]-[bn]
    in which
      {link_indices} can be `a` or `f` indicating linking all layers on the same
      floor, or indices separated by `,` indicating which floor to link, e.g.,
      `0,2,4`, in which case the given height must be greater than 4.

    Note that arc_string does not support different kernel size for left and
      right tower.
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
    if self.link_indices: checker.check_type(self.link_indices, int)


  def __str__(self):
    result = '{}D-k{}c{}e{}h{}t{}-{}'.format(
      self.dimension, self.filters, self.contraction_kernel_size,
      self.expansion_kernel_size, self.height, self.thickness, self.activation)
    if len(self.link_indices) < self.height:
      if len(self.link_indices) == 0: result += '-o'
      else: result += '-' + ','.join([str(i) for i in self.link_indices])
    if self.use_maxpool: result += '-mp'
    if self.use_batchnorm: result += '-bn'
    return result


if __name__ == '__main__':
  from tframe import Predictor
  from tframe import console
  from tframe import hub as th
  from tframe import mu

  console.suppress_logging()
  th.save_model = False

  model = Predictor('UNet')
  model.add(mu.Input(sample_shape=[None, None, None, 1]))

  unet = UNet(dimension=3, arc_string='16-3-4-2-relu-mp')
  unet.add_to(model)

  print(unet)
  model.rehearse()


