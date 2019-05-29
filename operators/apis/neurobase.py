from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import checker
from tframe import activations
from tframe import initializers


class NeuroBase(object):

  def __init__(
      self,
      activation=None,
      weight_initializer='xavier_normal',
      use_bias=False,
      bias_initializer='zeros'):

    if activation: activation = activations.get(activation)
    self._activation = activation
    self._weight_initializer = initializers.get(weight_initializer)
    self._use_bias = checker.check_type(use_bias, bool)
    self._bias_initializer = initializers.get(bias_initializer)
