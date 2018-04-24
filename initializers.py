from __future__ import  absolute_import

import six

from tensorflow.python.ops import init_ops


def glorot_uniform():
  return init_ops.glorot_uniform_initializer()


def identity():
  return init_ops.identity_initializer()


def get(identifier):
  if identifier is None or isinstance(identifier, init_ops.Initializer):
    return identifier
  elif isinstance(identifier, six.string_types):
    # If identifier is a string
    identifier = identifier.lower()
    if identifier in ['glorot_uniform', 'xavier_uniform']:
      return glorot_uniform()
    elif identifier in ['id', 'identity']:
      return identity()
    else:
      # Find initializer in tensorflow.python.ops.init_ops
      initializer = (
        init_ops.__dict__.get(identifier, None) or
        init_ops.__dict__.get('{}_initializer'.format(identifier),None))
      # If nothing is found
      if initializer is None:
        raise ValueError('Can not resolve "{}"'.format(identifier))
      # Return initializer with default parameters
      return initializer
  else:
    raise TypeError('identifier must be a Initializer or a string')


if __name__ == '__main__':
  print(get('glorot_normal'))
