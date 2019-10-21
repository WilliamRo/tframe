from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe import checker
from tframe import hub as th
from tframe.layers.layer import Layer, single_input


class Significance(Layer):

  full_name = 'significance'
  abbreviation = 'signif'

  def __init__(self, init_weights=None, numerator=4.0):
    self._max_level = checker.check_positive_integer(th.max_level)
    if init_weights is None:
      init_weights = numerator / np.arange(1, self._max_level + 1)
    self._init_weights = np.array(init_weights, np.float32).reshape(-1, 1)
    assert len(self._init_weights) == self._max_level
    self._U = None

  @property
  def U(self):
    if self._U is not None: return self._U
    N = self._max_level
    U = np.zeros([N, N], dtype=np.float32)
    Y, X = np.meshgrid(range(N), range(N))
    U[X <= Y] = 1.0
    self._U = U
    return self._U

  @single_input
  def _link(self, x, **kwargs):
    # Sanity check
    assert isinstance(x, tf.Tensor)
    dim = x.shape.as_list()[-1]
    repeats = int(dim / self._max_level)
    assert repeats * self._max_level == dim
    # Get tf weights
    weights = tf.get_variable(
      name='sig_weights', dtype=th.dtype, initializer=self._init_weights)
    # Apply significance
    U = tf.constant(self.U, dtype=th.dtype)
    s = U @ tf.nn.softmax(weights)
    s = tf.reshape(s, [1, self._max_level])
    coef = tf.concat([s] * repeats, axis=1)
    return x * coef


if __name__ == '__main__':
  import matplotlib.pyplot as plt

  max_level = 10
  N = max_level
  U = np.zeros([N, N], dtype=np.float32)
  Y, X = np.meshgrid(range(N), range(N))
  U[X <= Y] = 1.0

  x = np.arange(1, max_level + 1)
  init_weights = 4. / x
  exps = np.exp(init_weights)
  softmax = exps / np.sum(exps)
  s = U @ softmax

  plt.plot(s)
  plt.show()


