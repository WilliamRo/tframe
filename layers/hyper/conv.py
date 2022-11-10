from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker
from tframe import hub as th
from tframe.layers.layer import single_input
from tframe.layers.normalization import BatchNormalization

from .hyper_base import HyperBase

import typing as tp


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
               name: tp.Optional[str] = None,
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
                 else (kernel_size,) * self.Configs.kernel_dim)
    self.neuron_scale = knl_shape + (filters,)

    # Set name if provided
    if name is not None: self.full_name = name

  def get_layer_string(self, scale, full_name=False, suffix=''):
    activation = self._activation_string
    if self.dilations not in (None, 1): suffix += f'(di{self.dilations})'
    if callable(self.filter_generator): suffix += '[H]'
    if self.use_batchnorm: suffix += '->bn'
    if isinstance(activation, str): suffix += '->{}'.format(activation)
    result = super().get_layer_string(scale, full_name, suffix)
    return result

  def _check_size(self, size):
    return checker.check_conv_size(size, self.Configs.kernel_dim)

  def _get_filter_shape(self, input_dim):
    filter_shape = self.kernel_size
    if self.Configs.transpose:
      filter_shape += (self.channels, input_dim)
    else:
      filter_shape += (input_dim, self.channels)
    return filter_shape

  @single_input
  def _link(self, x: tf.Tensor, **kwargs):
    # Expand last dimension if necessary
    if self.expand_last_dim: x = tf.expand_dims(x, -1)

    # Generate filter if filter_generator is provided
    filter = None
    if callable(self.filter_generator):
      filter_shape = self._get_filter_shape(x.shape.as_list()[-1])
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


class Conv1D(ConvBase):
  """Perform 1D convolution on a channel-last data"""

  full_name = 'conv1d'
  abbreviation = 'conv1d'

  class Configs(ConvBase.Configs):
    kernel_dim = 1

  def forward(self, x: tf.Tensor, filter=None, **kwargs):
    return self.conv1d(
      x, self.channels, self.kernel_size, 'HyperConv1D', strides=self.strides,
      padding=self.padding, dilations=self.dilations, filter=filter, **kwargs)


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


class Conv3D(ConvBase):
  """Perform 3D convolution on a channel-last image"""

  full_name = 'conv3d'
  abbreviation = 'conv3d'

  class Configs(ConvBase.Configs):
    kernel_dim = 3

  def forward(self, x: tf.Tensor, filter=None, **kwargs):
    return self.conv3d(
      x, self.channels, self.kernel_size, 'HyperConv3D', strides=self.strides,
      padding=self.padding, dilations=self.dilations, filter=filter, **kwargs)


class Deconv1D(ConvBase):

  full_name = 'deconv1d'
  abbreviation = 'deconv1d'

  class Configs(ConvBase.Configs):
    kernel_dim = 1
    transpose = True

  def forward(self, x: tf.Tensor, filter=None, **kwargs):
    return self.deconv1d(
      x, self.channels, self.kernel_size, 'HyperDeconv1D', strides=self.strides,
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


class Deconv3D(ConvBase):

  full_name = 'deconv3d'
  abbreviation = 'deconv3d'

  class Configs(ConvBase.Configs):
    kernel_dim = 3
    transpose = True

  def forward(self, x: tf.Tensor, filter=None, **kwargs):
    return self.deconv3d(
      x, self.channels, self.kernel_size, 'HyperDeconv3D', strides=self.strides,
      padding=self.padding, dilations=self.dilations, filter=filter, **kwargs)


class DenseUpsampling2D(Conv2D):
  """DUC proposed in
  [1] Wang, et al., Understanding Convolution for Semantic Segmentation,
     https://arxiv.org/abs/1702.08502 """

  full_name = 'DUConv2d'
  abbreviation = 'DUC2d'
  
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
    # Sanity check
    self.duc_strides = self._check_strides(strides)
    sh, sw = self.duc_strides

    # Call parent's constructor
    super(DenseUpsampling2D, self).__init__(
      filters=filters * sh * sw,
      kernel_size=kernel_size,
      strides=1,
      padding=padding,
      dilations=dilations,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      expand_last_dim=expand_last_dim,
      use_batchnorm=use_batchnorm,
      filter_generator=filter_generator,
      **kwargs)

  def _check_strides(self, strides: tp.Union[int, list, tuple]):
    """Currently only strides of up-to-2 dimension is supported"""
    if isinstance(strides, int) and strides > 0:
      strides = [strides] * 2
    elif isinstance(strides, (tuple, list)) and len(strides) == 2:
      assert all([isinstance(s, int) and s > 0 for s in strides])
    else: raise ValueError(f'!! Illegal strides `{strides}`.')
    return strides

  def _reshape(self, x: tf.Tensor):
    sh, sw = self.duc_strides
    _, h, w, F = x.shape.as_list()
    H, W = h * sh, w * sw
    f = F // sh // sw

    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, shape=[-1, f, sh, sw, h, w])
    # x = tf.transpose(x, [0, 1, 2, 4, 3, 5])   # Case I
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3])  # Case II
    x = tf.reshape(x, shape=[-1, f, H, W])
    x = tf.transpose(x, [0, 2, 3, 1])
    return x

  def forward(self, x: tf.Tensor, filter=None, **kwargs):
    y = super(DenseUpsampling2D, self).forward(x, filter, **kwargs)

    return self._reshape(y)



