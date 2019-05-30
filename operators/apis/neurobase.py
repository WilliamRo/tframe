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
      bias_initializer='zeros',
      layer_normalization=False,
      **kwargs):

    if activation: activation = activations.get(activation)
    self._activation = activation
    self._weight_initializer = initializers.get(weight_initializer)
    self._use_bias = checker.check_type(use_bias, bool)
    self._bias_initializer = initializers.get(bias_initializer)
    self._layer_normalization = checker.check_type(layer_normalization, bool)
    self._gain_initializer = initializers.get(
      kwargs.get('gain_initializer', 'ones'))
    self._normalize_each_psi = kwargs.pop('normalize_each_psi', False)
    self._nb_kwargs = kwargs


  @property
  def _prune_frac(self):
    return self._nb_kwargs.get('prune_frac', 0)

  @property
  def _s_prune_frac(self):
    frac = self._nb_kwargs.get('s_prune_frac', 0)
    if frac == 0: return self._prune_frac
    else: return frac

  @property
  def _x_prune_frac(self):
    frac =  self._nb_kwargs.get('x_prune_frac', 0)
    if frac == 0: return self._prune_frac
    else: return frac
