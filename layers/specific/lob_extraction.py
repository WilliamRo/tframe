from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf

from tframe import checker, context
from tframe import hub as th
from tframe.layers.layer import Layer, single_input


class Significance(Layer):

  full_name = 'significance'
  abbreviation = 'signif'

  def __init__(self, init_weights=None, numerator=1.0):
    self._max_level = checker.check_positive_integer(th.max_level)
    if init_weights is None:
      init_weights = numerator / np.arange(1, self._max_level + 1)
    self._init_weights = np.array(init_weights, th.np_dtype).reshape(-1, 1)
    assert len(self._init_weights) == self._max_level
    self._U = None

  @property
  def U(self):
    if self._U is not None: return self._U
    N = self._max_level
    U = np.zeros([N, N], dtype=th.np_dtype)
    Y, X = np.meshgrid(range(N), range(N))
    U[X <= Y] = 1.0
    self._U = U
    return self._U

  @single_input
  def _link(self, x, **kwargs):
    """x = {P_ask[i], V_ask[i], P_bid[i], V_bid[i]}_i=1^10"""
    # Sanity check
    assert isinstance(x, tf.Tensor)
    dim = x.shape.as_list()[-1]
    repeats = int(dim / self._max_level)
    assert repeats * self._max_level == dim

    # Get tf weights
    weights = tf.get_variable(
      name='sig_weights', dtype=th.dtype, initializer=self._init_weights,
      trainable=not th.lob_fix_sig_curve)
    # Apply significance
    U = tf.constant(self.U, dtype=th.dtype)
    def square_max(x):
      square = x * x
      return square / (tf.reduce_sum(square) + 1e-6)
    # s = tf.matmul(U, tf.nn.softmax(weights)) TODO softmax ?
    s = tf.matmul(U, square_max(weights))
    coef = tf.reshape(tf.concat([s] * repeats, axis=-1), [1, -1])
    if th.export_var_alpha: context.add_var_to_export('Significance', weights)
    return x * coef


if __name__ == '__main__':
  import matplotlib.pyplot as plt

  max_level = 10
  N = max_level
  U = np.zeros([N, N], dtype=np.float32)
  Y, X = np.meshgrid(range(N), range(N))
  U[X <= Y] = 1.0

  x = np.arange(1, max_level + 1)
  init_weights = 1. / x
  # exps = np.exp(init_weights)
  exps = init_weights * init_weights
  softmax = exps / np.sum(exps)
  s = U @ softmax

  plt.plot(init_weights)
  plt.plot(softmax)
  plt.plot(s)
  plt.legend(['w', 'softmax', 's'])
  plt.show()


