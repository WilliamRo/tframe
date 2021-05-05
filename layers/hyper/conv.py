from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import activations
from tframe import hub as th
from tframe.layers.normalization import BatchNormalization

from .hyper_base import HyperBase


class ConvBase(HyperBase):

  abbreviation = 'conv'

  class Configs(object):
    pass

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
               **kwargs):

    # Call parent's initializer
    super(ConvBase, self).__init__(
      activation=activation,
      weight_initializer=kernel_initializer,
      use_bias=use_bias,
      bias_initializer=bias_initializer,
      layer_normalization=False,
      **kwargs)

    # Specific attributes
    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.dilations = dilations
    self.expand_last_dim = expand_last_dim
    self.use_batchnorm = use_batchnorm

    # Set neuron scale as filter shape
    knl_shape = (list(kernel_size) if isinstance(kernel_size, (list, tuple))
                 else [kernel_size, kernel_size])
    self.neuron_scale = knl_shape + [filters]

  def get_layer_string(self, scale, full_name=False, suffix=''):
    activation = self._activation_string
    if self.use_batchnorm: suffix += '->bn'
    if isinstance(activation, str): suffix += '->{}'.format(activation)
    result = super().get_layer_string(scale, full_name, suffix)
    return result


class Conv2D(ConvBase):
  """Perform 2D convolution on a channel-last image
  """

  full_name = 'conv2d'
  abbreviation = 'conv2d'

  class Configs(ConvBase.Configs):
    data_format = 'channels_last'

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
               **kwargs):

    # Call parent's initializer
    super(Conv2D, self).__init__(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      dilations=dilations,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      expand_last_dim=expand_last_dim,
      use_batchnorm=use_batchnorm,
      **kwargs)

  def forward(self, x: tf.Tensor, **kwargs):
    # Expand last dimension if necessary
    if self.expand_last_dim: x = tf.expand_dims(x, -1)

    # Convolve
    y = self.conv2d(x, self.filters, self.kernel_size, 'HyperConv2D',
                    strides=self.strides, padding=self.padding,
                    dilations=self.dilations, **kwargs)

    # Apply batchnorm if required
    if self.use_batchnorm:
      momentum = th.bn_momentum if th.bn_momentum is not None else 0.99
      y = BatchNormalization(momentum=momentum)(y)

    # Activate if required
    if self._activation is None: return y
    assert callable(self._activation)
    return self._activation(y)
