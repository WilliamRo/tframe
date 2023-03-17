from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf
from tframe.utils.arg_parser import Parser
import six



Constraint = tf.keras.constraints.Constraint

class MaxValue(Constraint):

  def __init__(self, max_value=1.0):
    self.max_value = max_value

  def __call__(self, w):
    return tf.clip_by_value(w, -self.max_value, self.max_value)



def get(identifier):
  if identifier is None or isinstance(identifier, Constraint): return identifier
  if not isinstance(identifier, six.string_types):
    raise TypeError('identifier must be a function or a string')

  p = Parser.parse(identifier)
  key = p.name.lower()

  if key in ['max_norm']:
    max_value = p.get_arg(float, default=2.0)
    axis = p.get_kwarg('axis', int, default=0)
    return tf.keras.constraints.max_norm(max_value=max_value, axis=axis)
  elif key in ['value', 'max_value']:
    max_value = p.get_arg(float, default=1.0)
    return MaxValue(max_value)
  else:
    KeyError('Unknown constraint name `{}`'.format(p.name))

