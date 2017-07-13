from __future__ import absolute_import

import six

import tensorflow as tf


def L2(strength=0.1):
  return lambda x: strength * tf.norm(x)


def get(identifier):
  if identifier is None or callable(identifier):
    return identifier
  elif isinstance(identifier, six.string_types):
    identifier = identifier.lower()
    if identifier in ['l2']:
      return L2()
    else:
      raise ValueError('Can not resolve "{}"'.format(identifier))
  else:
    raise TypeError('identifier must be a function or a string')


if __name__ == '__main__':
  print(callable(L2()))