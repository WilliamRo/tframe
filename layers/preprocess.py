from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe import hub
from tframe.layers.layer import Layer, single_input


class Normalize(Layer):
  """"""
  full_name = 'normalize'
  abbreviation = 'norm'

  def __init__(self, mu, sigma=None):
    self._mu = mu
    self._sigma = sigma


  @single_input
  def _link(self, x, **kwargs):
    if isinstance(self._mu, np.ndarray):
      self._mu = tf.constant(self._mu, dtype=hub.dtype)
    if isinstance(self._sigma, np.ndarray):
      self._sigma = tf.constant(self._sigma, dtype=hub.dtype)
    y = tf.subtract(x, self._mu)
    if self._sigma is not None: y = tf.divide(y, self._sigma)
    return y

