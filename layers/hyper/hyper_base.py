from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe.layers.layer import Layer, single_input
from tframe.operators.apis.neurobase import NeuroBase


class HyperBase(Layer, NeuroBase):

  is_nucleus = True

  def __init__(
      self,
      activation=None,
      weight_initializer='xavier_normal',
      use_bias=True,
      bias_initializer='zeros',
      layer_normalization=False,
      **kwargs):

    # Call parent's constructor
    NeuroBase.__init__(self, activation, weight_initializer, use_bias,
                       bias_initializer, layer_normalization, **kwargs)
    self._activation_string = activation

  @single_input
  def _link(self, x, **kwargs):
    return self.forward(x, **kwargs)

  def forward(self, x, **kwargs):
    raise NotImplemented

