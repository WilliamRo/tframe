from __future__ import absolute_import

import tensorflow as tf


class Function(object):
  """A core concept in tframe"""

  def group_name(self):
    raise NotImplementedError('Property "group_name" has not implemented yet')

  def __call__(self, *inputs, **kwargs):
    if self.group_name is not None:
      with tf.variable_scope(self.group_name):
        return self._link(*inputs, **kwargs)
    else:
      return self._link(*inputs, **kwargs)

  def _link(self, *inputs, **kwargs):
    raise NotImplementedError('_link method not implemented')

