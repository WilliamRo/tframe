from __future__ import absolute_import

import tensorflow as tf


class Function(object):
  """A core concept in tframe"""
  def __init__(self):
    self.inputs = None
    self.outputs = None
    self.chain = []

  @property
  def group_name(self):
    raise NotImplementedError('Property "group_name" has not implemented yet')

  def __call__(self, inputs=None, reuse=False):
    if inputs is None:
      if self.inputs is None:
        raise ValueError('Input not defined')
      inputs = self.inputs
    if self.group_name is not None:
      with tf.variable_scope(self.group_name):
        # TODO: reuse optional should be removed
        if reuse:
          tf.get_variable_scope().reuse_variables()
        return self._link(inputs)
    else:
      return self._link(inputs)

  def _link(self, inputs):
    # TODO: this may be not necessary
    # Otherwise link functions in chain list
    if not isinstance(self.chain, list):
      raise TypeError('Function.chain must be a list')
    else:
      if len(self.chain) == 0:
        raise ValueError('chain is empty')
      outputs = inputs
      for f in self.chain:
        outputs = f(outputs)

      return outputs

