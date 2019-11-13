from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import tensorflow as tf

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


def get(identifier, **kwargs):
  # Sanity check
  assert len(kwargs) == 0
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
    elif identifier in ['softmax']: return softmax
    elif identifier in ['cumax']: return cumax
    elif identifier in ['sigmoid']: return lambda x: sigmoid(x, **kwargs)
    else:
      # Try to find activation in tf.nn
      activation = getattr(tf, identifier, None)
      if activation is None:
        raise ValueError('Can not resolve {}'.format(identifier))
      return activation
  else:
    raise TypeError('identifier must be callable or a string')
