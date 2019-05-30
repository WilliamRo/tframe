from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import activations
from tframe import checker
from tframe import context
from tframe import initializers
from tframe import linker
from tframe import hub
from tframe.nets import RNet
from tframe.operators.apis.neurobase import NeuroBase
from tframe.operators.neurons import NeuronArray


class CellBase(RNet, NeuroBase):
  """Base class for RNN cells.
     TODO: all rnn cells are encouraged to in inherit this class
  """
  net_name = 'cell_base'

  def __init__(
      self,
      activation='tanh',
      weight_initializer='xavier_normal',
      use_bias=True,
      bias_initializer='zeros',
      layer_normalization=False,
      **kwargs):
    # Call parent's constructor
    RNet.__init__(self, self.net_name)
    NeuroBase.__init__(self, activation, weight_initializer, use_bias,
                       bias_initializer, layer_normalization, **kwargs)

    self._output_scale_ = None

  @property
  def _output_scale(self):
    if self._output_scale_ is not None: return self._output_scale_
    return self._state_size

  # TODO: this property is a compromise to avoid error in Net.
  @_output_scale.setter
  def _output_scale(self, val): self._output_scale_ = val

  @property
  def _scale_tail(self):
    assert self._state_size is not None
    return '({})'.format(self._state_size)

  @property
  def prune_is_on(self):
    """This property is decided in linker.neuron"""
    sum = 0
    for k in ('prune_frac', 's_prune_frac', 'x_prune_frac'):
      sum += self._kwargs.get(k, 0)
    return hub.pruning_rate_fc > 0 and sum > 0

  def structure_string(self, detail=True, scale=True):
    return self.net_name + self._scale_tail if scale else ''

  # def _update_gate(self, num_units, unit_size, x=None, s=None, net_z=None,
  #                  reverse=True, bias_initializer='zeros',
  #                  saturation_penalty=0.):
  #   # Sanity check
  #   checker.check_positive_integer(unit_size)
  #
  #   # Check net_z
  #   if net_z is None:
  #     assert x is not None
  #     net_z = self.neurons(
  #       x, s, scope='net_z', bias_initializer=bias_initializer)
  #
  #   # Calculate z
  #   z = linker.softmax_over_groups(net_z, [(unit_size, num_units)], 'z')
  #
  #   # Calculate opposite
  #   z_opposite = tf.subtract(1., z)
  #   if reverse: z, z_opposite = z_opposite, z
  #
  #   # Add saturation loss to context if necessary
  #   if saturation_penalty > 0:
  #     context.add_loss_tensor(
  #       saturation_penalty * tf.reduce_mean(tf.abs(z - 0.5)))
  #   elif saturation_penalty < 0:
  #     context.add_loss_tensor(
  #       -saturation_penalty * tf.reduce_mean(tf.minimum(z, z_opposite)))
  #
  #   return z, z_opposite


