from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import checker
from tframe import hub
from tframe import linker
from tframe import initializers

from .kernel_base import KernelBase


class PsyKernel(KernelBase):

  def __init__(self,
               kernel_key,
               num_neurons,
               input_,
               suffix,
               weight_initializer='glorot_normal',
               prune_frac=0,
               LN=False,
               gain_initializer='ones',
               etch=None,
               **kwargs):

    # Call parent's initializer
    super().__init__(kernel_key, num_neurons, weight_initializer, prune_frac,
                     etch=etch, **kwargs)

    self.input_ = checker.check_type(input_, tf.Tensor)
    self.suffix = checker.check_type(suffix, str)
    self.LN = checker.check_type(LN, bool)
    self.gain_initializer = initializers.get(gain_initializer)

  # region : Properties

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
    elif identifier in ('row_mask', 'hyper16'): return self.row_mask
    else: raise ValueError('!! Unknown kernel `{}`'.format(identifier))

  def _layer_normalization(self, a):
    return self.layer_normalization(a, self.gain_initializer, False)

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

  def row_mask(self, seed, seed_weight_initializer):
    """Generate weights with rows being masked.
       y = (diag(row_mask) @ W) @ x
       Note that during implementation, this is actually called column mask.
       Used in (1) Ha, etc. Hyper Networks. 2016.
               (2) a GRU variant: reset_gate \odot (Ws @ s_{t-1})
    """
    xd, sd, yd = self.input_dim, linker.get_dimension(seed), self.num_neurons

    # Get weights
    Wsy = self._get_weights(
      'Wsy', [sd, yd], initializer=seed_weight_initializer)
    Wxy = self._get_weights('Wxy', shape=[xd, yd])

    # Calculate output
    x = self.input_
    a = (seed @ Wsy) * (x @ Wxy)
    return a

  # endregion : Kernels
