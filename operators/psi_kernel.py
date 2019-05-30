from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import tensorflow as tf

from tframe import context
from tframe import checker
from tframe import hub
from tframe import linker
from tframe import initializers


class PsyKernel(object):

  def __init__(self,
               kernel_key,
               num_neurons,
               input_,
               suffix,
               weight_initializer='glorot_normal',
               prune_frac=0,
               LN=False,
               gain_initializer='ones',
               **kwargs):

    self.kernel_key = checker.check_type(kernel_key, str)
    self.kernel = self._get_kernel(kernel_key)
    self.num_neurons = checker.check_positive_integer(num_neurons)
    self.input_ = input_
    self.suffix = suffix

    self.weight_initializer = initializers.get(weight_initializer)
    assert 0 <= prune_frac <= 1
    self.prune_frac = prune_frac
    self.LN = LN
    self.gain_initializer = initializers.get(gain_initializer)

    self.kwargs = kwargs

    self._check_arguments()

  # region : Properties

  @property
  def prune_is_on(self):
    assert 0 <= self.prune_frac <= 1
    return self.prune_frac > 0 and hub.prune_on
  
  @property
  def input_dim(self):
    return linker.get_dimension(self.input_)

  # endregion : Properties

  # region : Public Methods

  def __call__(self):
    with tf.variable_scope('psy_' + self.suffix):
      a = self.kernel(**self.kwargs)
      if self.LN: a = self._layer_normalization(a)
    return a

  # endregion : Public Methods

  # region : Private Methods

  def _get_kernel(self, identifier):
    assert isinstance(identifier, str)
    identifier = identifier.lower()
    if identifier in ('dense', 'fc'): return self.dense
    elif identifier in ('mul', 'multiplicative'): return self.multiplicative
    else: raise ValueError('!! Unknown kernel `{}`'.format(identifier))

  def _get_weights(self, name, shape, dtype=None):
    # Set default dtype if not specified
    if dtype is None: dtype = hub.dtype
    # Get weights
    weights = tf.get_variable(
      name, shape, dtype=dtype, initializer=self.weight_initializer)
    if not self.prune_is_on: return weights
    # Register, context.pruner should be created in early model.build
    assert context.pruner is not None
    masked_weights = context.pruner.register_to_dense(weights, self.prune_frac)
    # Return
    assert isinstance(masked_weights, tf.Tensor)
    return masked_weights

  def _layer_normalization(self, a):
    return self.layer_normalization(a, self.gain_initializer, False)

  def _check_arguments(self):
    # The 1st argument is self
    arg_names = inspect.getfullargspec(self.kernel).args[1:]
    for arg_name in arg_names:
      if not arg_name in self.kwargs: raise AssertionError(
        '!! kernel ({}) argument `{}` should be provided.'.format(
          self.kernel_key, arg_name))

    # Make sure provided arguments matches kernel argument specification
    # .. exactly
    if len(self.kwargs) != len(arg_names):
      raise AssertionError(
        '!! kernel `{}` requires {} additional arguments but {} are '
        'provided.'.format(self.kernel_key, len(arg_names), len(self.kwargs)))

  # endregion : Private Methods

  # region : Static Methods

  @staticmethod
  def layer_normalization(a, gain_initializer, use_bias=False):
    # use_bias is forced to be False for now
    assert not use_bias

    # Get gain
    gain_initializer = initializers.get(gain_initializer)
    gain_shape = [1, linker.get_dimension(a)]
    gain = tf.get_variable(
      'gain', shape=gain_shape, dtype=hub.dtype, initializer=gain_initializer)

    # Get mean and standard deviation
    mu, variance = tf.nn.moments(a, axes=1, keep_dims=True)
    sigma = tf.sqrt(variance + hub.variance_epsilon)
    # Normalize and rescale
    a = gain * (a - mu) / sigma
    return a

  # endregion : Static Methods

  # region : Kernels

  def dense(self):
    W = self._get_weights('W', shape=[self.input_dim, self.num_neurons])
    return self.input_ @ W

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
    xd, sd, yd = self.input_dim, linker.get_dimension(seed), self.num_neurons

    # Get weights
    Wxf = self._get_weights('Wxf', [xd, fd])
    Wsf = self._get_weights('Wsf', [sd, fd])
    Wfy = self._get_weights('Wfy', [fd, yd])

    # Calculate output
    x = self.input_
    a = ((x @ Wxf) * (seed @ Wsf)) @ Wfy
    return a

# endregion : Kernels
