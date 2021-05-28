from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf

from tframe import hub
from tframe.layers.layer import Layer, single_input


class Normalize(Layer):
  """"""
  full_name = 'normalize'
  abbreviation = 'norm'

  def __init__(self, mu=None, sigma=None):
    if mu is not None:
      assert isinstance(mu, (np.ndarray, float))
    if sigma is not None:
      assert isinstance(sigma, (np.ndarray, float))
    self._mu = mu
    self._sigma = sigma


  @single_input
  def _link(self, x, **kwargs):
    y = x
    if self._mu is not None:
      self._mu = tf.constant(self._mu, dtype=hub.dtype)
      y = tf.subtract(x, self._mu)
    if self._sigma is not None:
      self._sigma = tf.constant(self._sigma, dtype=hub.dtype)
      y = tf.divide(y, self._sigma)
    return y

