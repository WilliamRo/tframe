from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker
from tframe import hub, context, console

from .hyper_base import HyperBase


class Dense(HyperBase):

  full_name = 'dense'
  abbreviation = 'dense'

  def __init__(
      self,
      num_neurons,
      activation=None,
      use_bias=True,
      weight_initializer='xavier_normal',
      bias_initializer='zeros',
      layer_normalization=False,
      etch=None,
      **kwargs):
    """
    :param etch: if this argument is not None, it will be passed to the
                 neuron array and
                 (1) a masked weight matrix will be created
                 (2) the corresponding weight and gradient will be registered
                     to monitor
                 (3) corresponding etch method will be called during training
    """

    # Call parent's constructor
    super().__init__(activation, weight_initializer, use_bias,
                     bias_initializer, layer_normalization, **kwargs)

    self.num_neurons = checker.check_positive_integer(num_neurons)
    self.neuron_scale = [num_neurons]
    self.etch = etch


  @property
  def structure_tail(self):
    activation = ''
    if self._activation is not None:
      activation = '->act'
      if isinstance(self._activation_string, str):
        activation = '->' + self._activation_string
    return '({})'.format(self.num_neurons) + activation


  def forward(self, x, **kwargs):
    return self.dense(self.num_neurons, x, scope='neuron_array',
                      activation=self._activation, etch=self.etch)
