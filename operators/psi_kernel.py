from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import checker
from tframe import context
from tframe import hub
from tframe import linker
from tframe import initializers

from typing import Optional

from .kernel_base import KernelBase


class PsiKernel(KernelBase):

  def __init__(self,
               kernel_key,
               num_units,
               input_,
               suffix,
               weight_initializer='glorot_normal',
               prune_frac=0,
               LN=False,
               gain_initializer='ones',
               etch: Optional[str] = None,
               weight_dropout=0.0,
               filter_size=None,
               strides=1,
               padding='SAME',
               dilations=1,
               **kwargs):

    # Call parent's initializer
    super().__init__(kernel_key, num_units, weight_initializer, prune_frac,
                     etch=etch, weight_dropout=weight_dropout, **kwargs)

    self.input_ = checker.check_type(input_, tf.Tensor)
    self.suffix = checker.check_type(suffix, str)
    self.LN = checker.check_type(LN, bool)
    self.gain_initializer = initializers.get(gain_initializer)

    # Attributes for convolutional operators
    # Check filter size for convolutional operations
    self.filter_size = self._check_size(filter_size)
    self.strides = self._check_size(strides)
    self.padding = padding.upper()
    self.dilations = self._check_size(dilations)

  # region : Properties

  @property
  def input_dim(self):
    """Get the last dimension of the input tensor."""
    return linker.get_dimension(self.input_)

  # endregion : Properties

  # region : Public Methods

  def __call__(self):
    with tf.variable_scope('psi_' + self.suffix):
      a = self.kernel(**self.kwargs)
      if self.LN: a = self._layer_normalization(a)
    return a

  # endregion : Public Methods

  # region : Private Methods

  @staticmethod
  def _check_size(size, dim=2):
    if size is None: return None
    return checker.check_conv_size(size, dim)

  def _get_kernel(self, identifier):
    assert isinstance(identifier, str)
    identifier = identifier.lower()
    if identifier in ('dense', 'fc'): return self.dense
    elif identifier in ('mul', 'multiplicative'): return self.multiplicative
    elif identifier in ('row_mask', 'hyper16'): return self.row_mask
    elif identifier in ('elect', 'election'): return self.elect
    elif identifier in ('sparse_sog', 'sparsog'): return self.sparse_sog
    elif identifier in ('conv2d',): return self.conv2d
    elif identifier in ('deconv2d',): return self.deconv2d
    else: raise ValueError('!! Unknown kernel `{}`'.format(identifier))

  def _layer_normalization(self, a):
    return self.layer_normalization(a, self.gain_initializer, False)

  # endregion : Private Methods

  # region : Static Methods

  @staticmethod
  def layer_normalization(a, gain_initializer, use_bias=False):
    from tframe.operators.apis.neurobase import NeuroBase

    assert not use_bias
    gain_initializer = initializers.get(gain_initializer)
    return NeuroBase.layer_normalize(
      a, axis=1, center=False, gamma_initializer=gain_initializer)

  # endregion : Static Methods

  # region : Kernels

  def _conv_common(self, conv, name, transpose=False,
                   filter=None, **kwargs):
    # Find filter shape
    assert callable(conv) and self.filter_size is not None
    filter_shape = list(self.filter_size)
    if transpose: filter_shape += [self.num_units, self.input_dim]
    else: filter_shape += [self.input_dim, self.num_units]

    # Get filter if not provided
    if filter is None: filter = self._get_weights('kernel', shape=filter_shape)
    # Check filter shape
    assert filter.shape.as_list() == filter_shape

    return conv(self.input_, filter,
                strides=self.strides,
                padding=self.padding,
                dilations=self.dilations,
                data_format='NHWC', name=name, **kwargs)

  def conv2d(self, filter=None):
    return self._conv_common(tf.nn.conv2d, name='conv2d_kernel', filter=filter)

  def deconv2d(self, filter=None):
    """This remedy for output tensor shape is from keras.layers.Conv2DTranspose
    """
    from tensorflow.python.keras.utils.conv_utils import deconv_output_length

    # Define utility
    get_len = lambda i, shape: deconv_output_length(
      shape[i+1], self.filter_size[i], padding=self.padding.lower(),
      output_padding=None, stride=self.strides[i], dilation=self.dilations[i])
    get_hw = lambda shape: [get_len(i, shape) for i in (0, 1)]

    # Infer the dynamic output shape:
    assert self.filter_size is not None and self.strides is not None
    inputs_shape = tf.shape(self.input_)
    out_height, out_width = get_hw(inputs_shape)
    output_shape = (inputs_shape[0], out_height, out_width, self.num_units)
    output_shape_tensor = tf.stack(output_shape)

    # Compute
    y = self._conv_common(
      tf.nn.conv2d_transpose, name='deconv2d_kernel',
      output_shape=output_shape_tensor, transpose=True, filter=filter)

    # Compute and set static output shape
    out_shape = self.input_.shape.as_list()
    out_shape[1], out_shape[2] = get_hw(out_shape)
    out_shape[3] = self.num_units
    y.set_shape(out_shape)

    return y

  def dense(self):
    W = self._get_weights('W', shape=[self.input_dim, self.num_units])
    rank = len(self.input_.shape)
    if rank > 2: return tf.tensordot(self.input_, W, [[rank - 1], [0]])
    return self.input_ @ W

  def sparse_sog(self, axis, group_size):
    """Given x of shape (bs, dim_x)
       y = x @ (W_bar \odot C)
       where W_bar and C has shape (dim_x, dim_y), and
       C = \eta_{SxN}(C_bar, axis), \eta is the operator of softmax over groups

    `axis` may be passed from
        SparseSOG(..., axis, ...) -> neuro_base.sparse_sog(..., axis, ...)
        -> neural_array.add_kernel(..., axis=axis) -> PsiKernel(..., axis=axis)
        -> kernel_base.kwargs['axis]
    So does `group_size`
    """
    S = group_size
    # Check dim and calculate N (num_groups)
    dim_to_be_partitioned = self.input_dim if axis == 0 else self.num_units
    assert dim_to_be_partitioned % S == 0
    N = dim_to_be_partitioned // S

    # Prepare weight matrix W_bar
    W_bar = self._get_weights('W_bar', shape=[self.input_dim, self.num_units])
    if S == 1: return self.input_ @ W_bar
    # .. make sure inputs are vectors
    assert len(self.input_.shape) == 2
    # .. create connection matrix C according to axis
    # .. (While shape_C can be determined by 1 line of code, readability is
    #     of more importance)
    if axis == 0:
      assert S * N == self.input_dim
      shape_C = [S, self.num_units * N]
    elif axis == 1:
      assert S * N == self.num_units
      shape_C = [self.input_dim * N, S]
    else: raise AssertionError('`axis` must be either 0 or 1')
    C_tilde = self._get_weights('C_tilde', shape=shape_C)
    C_bar = tf.nn.softmax(C_tilde, axis=axis, name='C_bar')
    C = tf.reshape(C_bar, shape=[self.input_dim, self.num_units], name='C')
    # assert all(tf.reduce_sum(C, axis) == N)
    W = tf.multiply(W_bar, C, name='W')
    # Codes for exporting weights
    if hub.export_sparse_weights:
      context.add_var_to_export('connection', C)
    # Encourage saturation
    if hub.saturation_penalty is not None and hub.saturation_penalty > 0:
      from tframe.losses import saturate_loss
      sta_loss = saturate_loss(C)
      context.add_loss_tensor(sta_loss)
      # TODO: STILL DEVELOPING
      # from tframe.losses import saturate_loss
      # sta_loss = saturate_loss(C, mu=1/S) * hub.saturation_penalty
      # vips = tf.reduce_max(C_bar, axis=axis)
      # right_loss = tf.reduce_mean(1. - vips)
      # left = C_bar[tf.less(C_bar, 1 / S)]
      # left_loss = tf.reduce_mean(left)
      # sta_loss = (left_loss + right_loss) * hub.saturation_penalty
      # context.add_loss_tensor(sta_loss)
    # Calculate output and return
    return tf.matmul(self.input_, W)

  def multiplicative(self, seed, fd):
    """Generate weights using seed. Theoretically,

       [Dy, 1] [Dy, Dx] [Dx, 1]   [Dy, Df]   [Df, Df]   [Df, Dx]
          y   =   W   @   x     =  Wyf @ diag(Wfs @ s) @ Wfx @ x

       Yet practically,

       [bs, Dy] [bs, Dx]  [Dx, Dy]    [Dx, Df]    [bs, Ds] [Ds, Df]  [Df, Dy]
          y   =    x    @    W  = ((x @ Wxf) \odot (seed  @  Wsf))  @  Wfy

      in which bs is batch size, Dx is input dimension, Dy is output dimension,
       s is seed and Df is factorization dimension.

      Ref: Sutskever, etc. Generating text with recurrent neural networks, 2011
    """
    xd, sd, yd = self.input_dim, linker.get_dimension(seed), self.num_units

    # Get weights
    Wxf = self._get_weights('Wxf', [xd, fd])
    Wsf = self._get_weights('Wsf', [sd, fd])
    Wfy = self._get_weights('Wfy', [fd, yd])

    # Calculate output
    x = self.input_
    a = ((x @ Wxf) * (seed @ Wsf)) @ Wfy
    return a

  def row_mask(self, seed, seed_weight_initializer):
    """Generate weights with rows being masked.
       y = (diag(row_mask) @ W) @ x
       Note that during implementation, weight matrix is actually masked
       by columns.
       Used in (1) Ha, etc. Hyper Networks. 2016.
               (2) a GRU variant: reset_gate \odot (Ws @ s_{t-1})
    """
    xd, sd, yd = self.input_dim, linker.get_dimension(seed), self.num_units

    # Get weights
    Wsy = self._get_weights(
      'Wsy', [sd, yd], initializer=seed_weight_initializer)
    Wxy = self._get_weights('Wxy', shape=[xd, yd])

    # Calculate output
    x = self.input_
    a = (seed @ Wsy) * (x @ Wxy)
    return a

  def elect(self, groups, votes):
    """Given a vector with group specification, one representative will be
       elected.
       groups = ((size1, num1), (size2, num2), ...)
       x.shape = [batch_size, Dx]
       y.shape = [batch_size, num_groups]
    """
    # Sanity check
    assert isinstance(groups, (list, tuple))
    groups = [g[:2] for g in groups]
    total_units = sum([s*n for s, n in groups])
    assert total_units == self.input_dim

    # Get votes
    # initializer = tf.constant_initializer(np.concatenate(
    #     [np.ones([1, s * n], dtype=np.float32) / s for s, n in groups], axis=1))
    if votes is None:
      initializer = 'glorot_uniform'
      votes = self._get_weights(
        'V', [1, self.input_dim], initializer=initializer)

    # Calculate output
    splitted_x = linker.split(self.input_, groups)
    splitted_v = linker.split(votes, groups)
    output_list = []
    for (s, n), x, v in zip(groups, splitted_x, splitted_v):
      if s == 1:
        output_list.append(x)
        continue
      y = tf.multiply(v, x)
      if n > 1: y = tf.reshape(y, [-1, s])
      y = tf.reduce_sum(y, axis=1, keepdims=True)
      if n > 1: y = tf.reshape(y, [-1, n])
      output_list.append(y)

    return linker.concatenate(output_list)

  # endregion : Kernels
