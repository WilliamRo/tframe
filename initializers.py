from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import numpy as np

from tframe import tf
from tensorflow.python.ops import init_ops

from tframe.utils import checker


def glorot_uniform():
  return init_ops.glorot_uniform_initializer()


def identity():
  return init_ops.identity_initializer()


def get(identifier, **kwargs):
  if identifier is None or isinstance(identifier, init_ops.Initializer):
    return identifier
  if np.isscalar(identifier) and identifier == 0.: identifier = 'zeros'

  # TODO: ...
  if callable(identifier): return identifier

  elif isinstance(identifier, six.string_types):
    # If identifier is a string
    identifier = identifier.lower()
    if identifier in ['random_uniform']:
      rng = kwargs.get('range', None)
      low, high = checker.get_range(rng)
      return init_ops.RandomUniform(minval=low, maxval=high)
    elif identifier in ['random_norm', 'random_normal']:
      mean = kwargs.get('mean', 0.)
      stddev = kwargs.get('stddev', 1.)
      return init_ops.truncated_normal_initializer(mean=mean, stddev=stddev)
    elif identifier in ['glorot_uniform', 'xavier_uniform']:
      return glorot_uniform()
    elif identifier in ['glorot_normal', 'xavier_normal']:
      return init_ops.glorot_normal_initializer()
    elif identifier in ['id', 'identity']:
      return identity()
    else:
      # Find initializer in tensorflow.python.ops.init_ops
      initializer = (
        init_ops.__dict__.get(identifier, None) or
        init_ops.__dict__.get('{}_initializer'.format(identifier), None))
      # If nothing is found
      if initializer is None:
        raise ValueError('Can not resolve "{}"'.format(identifier))
      # Return initializer with default parameters
      return initializer
  elif np.isscalar(identifier):
    # Note string is scalar
    return tf.initializers.constant(value=identifier)
  else:
    raise TypeError('identifier must be a Initializer or a string')


if __name__ == '__main__':
  print(get('glorot_normal'))
