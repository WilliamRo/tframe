from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Function(object):
  """A core concept in tframe"""
  master = None

  def group_name(self):
    raise NotImplementedError('Property "group_name" has not implemented yet')

  def __call__(self, *inputs, **kwargs):
    link = lambda: self._link(*inputs, **kwargs)
    if self.master is not None:
      assert issubclass(self.master, Function)
      link = lambda: self.master._link(self, *inputs, **kwargs)

    # Call _link
    if self.group_name is not None:
      with tf.variable_scope(self.group_name, reuse=tf.AUTO_REUSE):
        return link()
    else: return link()

  def _link(self, *inputs, **kwargs):
    raise NotImplementedError('_link method not implemented')

