from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Metric(object):

  def __init__(self, f=None, as_loss=True, tensor=None):
    if f is not None:
      # As prototype
      assert tensor is None and callable(f)
    else:
      # As operator
      assert f is None and isinstance(tensor, tf.Tensor)

    self._f = f
    self._as_loss = as_loss
    self._tensor = tensor

  @property
  def tensor(self):
    return self._tensor

  def is_better_than(self, metric1, metric2, gap=0):
    if self._as_loss: return metric1 < metric2 - gap
    else: return metric1 > metric2 + gap

  def __call__(self, *args, **kwargs):
    tensor = self._f(*args, **kwargs)
    assert isinstance(tensor, tf.Tensor)
    return Metric(as_loss=self._as_loss, tensor=tensor)
