from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tframe import tf

from tframe.utils.arg_parser import Parser


def L1(penalty):
  return lambda x: penalty * tf.norm(x, ord=1)

def L2(penalty):
  return lambda x: penalty * tf.norm(x, ord=2)


def get(identifier):
  if identifier is None or callable(identifier): return identifier
  if not isinstance(identifier, six.string_types):
    raise TypeError('identifier must be a function or a string')

  p = Parser.parse(identifier)
  key = p.name.lower()
  if key in ['l1']: return L1(penalty=p.get_arg(float))
  elif key in ['l2']: return L2(penalty=p.get_arg(float))
  else: raise ValueError('Can not resolve "{}"'.format(key))
