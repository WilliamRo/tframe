from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tframe import tf

from tframe.utils import checker
from tframe.utils.arg_parser import Parser



def relu(input_):
  if input_.dtype in [tf.complex64, tf.complex128]:
    with tf.name_scope("c_relu"):
      eps = 1e-7
      real = tf.real(input_)
      # Handle small value issue
      sign = real / tf.maximum(eps, tf.abs(real))
      mask = tf.maximum(sign, 0)
      mask = tf.complex(mask, tf.zeros_like(mask, dtype=mask.dtype))
      return input_ * mask
  else:
    return tf.nn.relu(input_, name='relu')


def leaky_relu(input_, **kwargs):
  if input_.dtype in [tf.complex64, tf.complex128]:
    raise TypeError('leaky-relu currently does not support complex input')
  leak = kwargs.get('leak', 0.1)
  return tf.maximum(input_, input_ * leak, name='lrelu')


def softmax(input_):
  if input_.dtype in [tf.complex64, tf.complex128]:
    raise TypeError('softmax currently does not support complex input')
  return tf.nn.softmax(input_, name='softmax')


def cumax(x):
  assert isinstance(x, tf.Tensor) and len(x.shape) == 2
  return tf.cumsum(tf.nn.softmax(x), axis=1, name='cumax')


def sigmoid(input_, **kwargs):
  sig = tf.sigmoid(input_, name='sigmoid')
  rng = kwargs.get('range', None)
  if rng is None: return sig
  # If range is specified
  low, high = checker.get_range(rng)
  return (high - low) * sig + low


@tf.custom_gradient
def sign_st(x):
  """Sign function with straight-through estimator"""
  from tframe import hub as th
  def sign(v):
    return (tf.cast(tf.math.greater_equal(v, 0), th.dtype) - 0.5) * 2
  def grad(dy):
    return dy * tf.cast(tf.logical_and(
      tf.greater_equal(x, -1.0), tf.less_equal(x, 1.0)), dtype=th.dtype)
  return sign(x), grad


def sog(x, groups_size):
  """Softmax over groups. All groups share the same group size."""
  # Sanity check
  assert isinstance(x, tf.Tensor)
  x_shape = x.shape.as_list()
  assert len(x_shape) == 2
  num_neurons = x_shape[-1]
  S = checker.check_positive_integer(groups_size)
  assert num_neurons % S == 0

  # Reshape neurons
  x_reshaped = tf.reshape(x, shape=[-1, S])
  # Apply softmax over each group
  a = tf.nn.softmax(x_reshaped, axis=-1)
  # Reshape back
  outputs = tf.reshape(a, shape=[-1, num_neurons])
  return outputs


def get(identifier, **kwargs):
  # Sanity check
  # assert len(kwargs) == 0
  # Return identifier directly if it is callable
  if callable(identifier): return identifier
  elif isinstance(identifier, six.string_types):
    # Parse identifier
    p = Parser.parse(identifier)
    identifier = p.name.lower()
    if identifier in ['relu']: return relu
    elif identifier in ['lrelu', 'leakyrelu', 'leaky-relu']:
      leak = p.get_arg(float, default=0.1)
      return lambda x: leaky_relu(x, leak=leak)
    elif identifier in ['id']: return lambda x: x
    elif identifier in ['softmax']: return softmax
    elif identifier in ['cumax']: return cumax
    elif identifier in ['sigmoid']: return lambda x: sigmoid(x, **kwargs)
    elif identifier in ['signst', 'sign_st', 'sign-st']: return sign_st
    elif identifier in ['retanh', 'relutanh']: return lambda x: relu(tf.tanh(x))
    else:
      # Try to find activation in tf.nn
      activation = getattr(tf, identifier, None)
      if activation is None:
        raise ValueError('Can not resolve {}'.format(identifier))
      return activation
  else:
    raise TypeError('identifier must be callable or a string')
