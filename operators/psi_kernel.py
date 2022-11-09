from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker
from tframe import context
from tframe import hub
from tframe import linker
from tframe import pedia
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

  def _check_size(self, size):
    if size is None: return None
    dim = len(self.input_.shape) - 2
    # inputs to conv-layers is at least 3-D tensors
    if dim < 1: return None
    return checker.check_conv_size(size, dim)

  def _get_kernel(self, identifier):
    assert isinstance(identifier, str)
    identifier = identifier.lower()
    if identifier in ('dense', 'fc'): return self.dense
    elif identifier in ('mul', 'multiplicative'): return self.multiplicative
    elif identifier in ('row_mask', 'hyper16'): return self.row_mask
    elif identifier in ('elect', 'election'): return self.elect
    elif identifier in ('sparse_sog', 'sparsog'): return self.sparse_sog
    elif identifier in ('conv1d',): return self.conv1d
    elif identifier in ('conv2d',): return self.conv2d
    elif identifier in ('conv3d',): return self.conv3d
    elif identifier in ('deconv1d',): return self.deconv1d
    elif identifier in ('deconv2d',): return self.deconv2d
    elif identifier in ('deconv3d',): return self.deconv3d
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

  def _conv_common(self, conv, name, transpose=False, kernel=None, **kwargs):
    """The reference for convolution with different filter for each sample in
    mini-batch:
       https://stackoverflow.com/questions/42068999/tensorflow-convolutions-with-different-filter-for-each-sample-in-the-mini-batch
    """
    # Find filter shape
    assert callable(conv) and self.filter_size is not None
    filter_shape = list(self.filter_size)
    if transpose: filter_shape += [self.num_units, self.input_dim]
    else: filter_shape += [self.input_dim, self.num_units]

    # Get filter if not provided
    if kernel is None: kernel = self._get_weights('kernel', shape=filter_shape)

    # Define convolution method
    if conv.__name__  == 'conv1d':
      kwargs['stride'] = self.strides
      kwargs['data_format'] = 'NWC'
    elif conv.__name__ =='conv1d_transpose':
      kwargs['strides'] = self.strides
      kwargs['data_format'] = 'NWC'
    elif conv.__name__ in ('conv2d', 'conv2d_transpose'):
      kwargs['strides'] = self.strides
      kwargs['data_format'] = 'NHWC'
    elif conv.__name__ in ('conv3d_v1', 'conv3d_transpose'):
      # For some reason, conv3d_v1 only accepts 5D `strides` and
      #  strides[0] == strides[4] == 1. So as `dilations`
      self.dilations = [1] + list(self.dilations) + [1]
      kwargs['strides'] = [1] + list(self.strides) + [1]
      kwargs['data_format'] = 'NDHWC'
    else: raise KeyError(f'!! Unknown conv op `{conv.__name__}`')

    _conv = lambda tupl: conv(
      tupl[0], tupl[1], padding=self.padding, dilations=self.dilations,
      name=name, **kwargs)

    # Check filter shape
    if kernel.shape.as_list() == filter_shape:
      return _conv((self.input_, kernel))
    else:
      assert len(kernel.shape) == len(filter_shape) + 1
      # Add filter to context for future use
      context.add_to_list_collection(pedia.hyper_kernels, kernel)
      return tf.squeeze(tf.map_fn(
        _conv, (tf.expand_dims(self.input_, 1), kernel), dtype=hub.dtype),
        axis=1)

  def conv1d(self, filter=None) -> tf.Tensor:
    return self._conv_common(tf.nn.conv1d, name='conv1d_kernel', kernel=filter)

  def conv2d(self, filter=None) -> tf.Tensor:
    return self._conv_common(tf.nn.conv2d, name='conv2d_kernel', kernel=filter)

  def conv3d(self, filter=None) -> tf.Tensor:
    return self._conv_common(tf.nn.conv3d, name='conv3d_kernel', kernel=filter)

  def _deconvXd(self, X: int, func, name: str, filter=None) -> tf.Tensor:
    """This remedy for output tensor shape is from keras.layers.Conv2DTranspose
    """
    from tensorflow.python.keras.utils.conv_utils import deconv_output_length

    # Define function for calculating deconv_output.shape[i+1]
    # For X=1,2,3, deconvXd uses `strides`. (Unlike conv1d uses `stride`)
    get_len = lambda i, shape: deconv_output_length(
      shape[i+1], self.filter_size[i], padding=self.padding.lower(),
      output_padding=None, stride=self.strides[i], dilation=self.dilations[i])

    # Infer the dynamic output shape as TENSOR
    assert self.filter_size is not None and self.strides is not None
    inputs_shape = tf.shape(self.input_)
    major_shape_tensor = [get_len(i, inputs_shape) for i in range(X)]
    output_shape_tensor = tf.stack(
      [inputs_shape[0]] + major_shape_tensor + [self.num_units])

    # Compute deconv output
    y = self._conv_common(func, name=name, output_shape=output_shape_tensor,
                          transpose=True, kernel=filter)

    # Compute and set static output shape as LIST of scalars
    # in_shape = (?, L, C) or (?, H, W, C) or (?, D, H, W, C)
    out_shape = self.input_.shape.as_list()
    out_shape[1:X+1] = [get_len(i, out_shape) for i in range(X)]
    out_shape[-1] = self.num_units
    y.set_shape(out_shape)

    return y

  def deconv1d(self, filter=None) -> tf.Tensor:
    return self._deconvXd(1, tf.nn.conv1d_transpose, name='deconv1d_kernel',
                          filter=filter)

  def deconv2d(self, filter=None) -> tf.Tensor:
    return self._deconvXd(2, tf.nn.conv2d_transpose, name='deconv2d_kernel',
                          filter=filter)

  def deconv3d(self, filter=None) -> tf.Tensor:
    return self._deconvXd(3, tf.nn.conv3d_transpose, name='deconv3d_kernel',
                          filter=filter)

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
