from __future__ import absolute_import

import tensorflow as tf

from ..core import Function


class Layer(Function):
  """Abstract definition for layers"""
  is_nucleus = False

  full_name = None
  abbreviation = None

  @property
  def group_name(self):
    return self.full_name

  def _link(self, inputs):
    raise NotImplementedError('_link method not implemented')

  def _get_variable(self, name, shape, fixed_zero=False,
                      initializer='xavier_uniform', regularizer=None):
    return tf.get_variable(
      name, shape, dtype=tf.float32, trainable=not fixed_zero,
      initializer=tf.zeros_initializer() if fixed_zero else initializer,
      regularizer=None if fixed_zero else regularizer)


def single_input(_link):

  def wrapper(*args):
    # Currently not sure if the decorator is for class method only
    input_ = args[1] if isinstance(args[0], Layer) else args[0]
    if isinstance(input_, list):
      if len(input_) != 1:
        raise ValueError('This layer only accept single input')
      input_ = input_[0]

    if not isinstance(input_, tf.Tensor):
      raise TypeError('This layer only accept a Tensor as input')

    args = ((args[0], input_) + args[2:] if isinstance(args[0], Layer)
            else (input_, ) + args[1:])

    return _link(*args)

  return wrapper


