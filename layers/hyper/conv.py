from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import checker
from tframe import hub as th
from tframe.layers.layer import single_input
from tframe.layers.normalization import BatchNormalization

from .hyper_base import HyperBase


class ConvBase(HyperBase):

  abbreviation = 'conv'

  class Configs(object):
    data_format = 'channels_last'
    kernel_dim = None
    transpose = False

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='same',
               dilations=1,
               activation=None,
               use_bias=False,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               expand_last_dim=False,
               use_batchnorm=False,
               filter_generator=None,
               **kwargs):
    """If filter_generator is provided, it should have the signature:
          def filter_generator(self, filter_shape):
            ...
        in which self is an instance of ConvBase
    """

    # Call parent's initializer
    super(ConvBase, self).__init__(
      activation=activation,
      weight_initializer=kernel_initializer,
      use_bias=use_bias,
      bias_initializer=bias_initializer,
      layer_normalization=False,
      **kwargs)

    # Specific attributes
    self.channels = filters
    self.kernel_size = self._check_size(kernel_size)
    self.strides = self._check_size(strides)
    self.padding = padding
    self.dilations = dilations
    self.expand_last_dim = expand_last_dim
    self.use_batchnorm = use_batchnorm

    # Set filter generator
    if filter_generator is not None: assert callable(filter_generator)
    self.filter_generator = filter_generator

    # Set neuron scale as filter shape
    knl_shape = (tuple(kernel_size) if isinstance(kernel_size, (list, tuple))
                 else (kernel_size, kernel_size))
    self.neuron_scale = knl_shape + (filters,)

  def get_layer_string(self, scale, full_name=False, suffix=''):
    activation = self._activation_string
    if self.use_batchnorm: suffix += '->bn'
    if isinstance(activation, str): suffix += '->{}'.format(activation)
    result = super().get_layer_string(scale, full_name, suffix)
    return result

  def _check_size(self, size):
    return checker.check_conv_size(size, self.Configs.kernel_dim)

  @single_input
  def _link(self, x: tf.Tensor, **kwargs):
    # Expand last dimension if necessary
    if self.expand_last_dim: x = tf.expand_dims(x, -1)

    # Generate filter if filter_generator is provided
    filter = None
    if callable(self.filter_generator):
      # Currently customized filter is only supported by 2-D convolution
      assert self.Configs.kernel_dim == 2
      input_dim = x.shape.as_list()[-1]
      filter_shape = self.kernel_size
      if self.Configs.transpose: filter_shape += (self.channels, input_dim)
      else: filter_shape += (input_dim, self.channels)
      with tf.variable_scope('filter-generator'):
        filter = self.filter_generator(self, filter_shape)

    # Convolve
    y = self.forward(x, filter=filter, **kwargs)

    # Apply batchnorm if required
    if self.use_batchnorm:
      momentum = th.bn_momentum if th.bn_momentum is not None else 0.99
      y = BatchNormalization(momentum=momentum)(y)

    # Activate if required
    if self._activation is None: return y
    assert callable(self._activation)
    return self._activation(y)


class Conv2D(ConvBase):
  """Perform 2D convolution on a channel-last image"""

  full_name = 'conv2d'
  abbreviation = 'conv2d'

  class Configs(ConvBase.Configs):
    kernel_dim = 2

  def forward(self, x: tf.Tensor, filter=None, **kwargs):
    return self.conv2d(
      x, self.channels, self.kernel_size, 'HyperConv2D', strides=self.strides,
      padding=self.padding, dilations=self.dilations, filter=filter, **kwargs)


class Deconv2D(ConvBase):

  full_name = 'deconv2d'
  abbreviation = 'deconv2d'

  class Configs(ConvBase.Configs):
    kernel_dim = 2
    transpose = True

  def forward(self, x: tf.Tensor, filter=None, **kwargs):
    return self.deconv2d(
      x, self.channels, self.kernel_size, 'HyperDeconv2D', strides=self.strides,
      padding=self.padding, dilations=self.dilations, filter=filter, **kwargs)


