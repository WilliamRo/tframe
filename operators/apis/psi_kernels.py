from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import hub
from tframe import linker
from tframe import initializers


"""
The argument list of a \psi kernel function must follow:
  1. output_dimension
  2. input tensor with shape [batch_size, input_dim]
  3. other parameters
"""

def dense(output_dim, x, weight_initializer):
  input_dim = linker.get_dimension(x)
  weight_initializer = initializers.get(weight_initializer)
  W = tf.get_variable('W', shape=[input_dim, output_dim], dtype=hub.dtype,
                      initializer=weight_initializer)
  return x @ W


def multiplicative(yd, x, seed, fd, weight_initializer):
  """Generate weights using seed. Theoretically,

     [Dy, 1] [Dy, Dx] [Dx, 1]   [Dy, Df]   [Df, Df]   [Df, Dx]
        y   =   W   @   x     =  Wyf @ diag(Wfs @ s) @ Wfx @ x

     Yet practically,

     [bs, Dy] [bs, Dx]  [Dx, Dy]    [Dx, Df]    [bs, Ds] [Ds, Df]  [Df, Dy]
        y   =    x    @    W  = ((x @ Wxf) \odot (seed  @  Wsf))  @  Wfy

    in which bs is batch size, Dx is input dimension, Dy is output dimension,
     s is seed and Df is factorization dimension.

    Ref: Sutskever, etc. Gernerating text with recurrent neural networks, 2011
  """
  xd, sd = linker.get_dimension(x), linker.get_dimension(seed)
  weight_initializer = initializers.get(weight_initializer)

  # Get weights
  get_W = lambda name, shape: tf.get_variable(
    name, shape, dtype=hub.dtype, initializer=weight_initializer)
  Wxf = get_W('Wxf', [xd, fd])
  Wsf = get_W('Wsf', [sd, fd])
  Wfy = get_W('Wfy', [fd, yd])

  # Calculate output
  y = ((x @ Wxf) * (seed @ Wsf)) @ Wfy
  return y


class PsyKernels(object):

  directory = {
    'fc': dense,
    'dense': dense,
    'mul': multiplicative,
    'multiplicative': multiplicative,
  }

  @staticmethod
  def get_kernel(identifier, suffix, **kwargs):
    """If identifier is callable, the arguments used in linking should be
       taken care of.
    """
    assert callable(identifier) or identifier in PsyKernels.directory
    assert isinstance(suffix, str)
    # Wrap kernel
    def kernel(num, x):
      with tf.variable_scope('psy_' + suffix):
        if callable(identifier): return identifier(num, x)
        return PsyKernels.directory[identifier](num, x, **kwargs)
    return kernel


