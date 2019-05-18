from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import checker, linker
from tframe import hub, context
from tframe import initializers

from tframe.layers.layer import LayerWithNeurons, Layer, single_input


class Dense(LayerWithNeurons):

  full_name = 'dense'
  abbreviation = 'dense'

  def __init__(
      self,
      num_neurons,
      activation=None,
      use_bias=True,
      weight_initializer='xavier_normal',
      bias_initializer='zeros',
      prune_frac=0,
      **kwargs):
    # Call parent's constructor
    LayerWithNeurons.__init__(
      self, activation, weight_initializer, use_bias, bias_initializer,
      **kwargs)

    self.num_neurons = checker.check_positive_integer(num_neurons)
    self._prune_frac = checker.check_gate(prune_frac)
    self.neuron_scale = [num_neurons]

  @property
  def structure_tail(self):
    activation = ''
    if self._activation is not None:
      activation = '->act'
      if isinstance(self._activation_string, str):
        activation = '->' + self._activation_string
    return '({})'.format(self.num_neurons) + activation

  def forward(self, x, **kwargs):
    return self.neurons(x, self.num_neurons, activation=self._activation,
                        prune_frac=self._prune_frac)


class SparseAffine(Layer):

  full_name = 'sparse'
  abbreviation = 'sparse'

  def __init__(
      self,
      num_neurons,
      heads=1,
      logits_initializer='random_normal',
      coef_initializer='random_normal',
      use_bias=True,
      bias_initializer='zeros',
      **kwargs):

    self.num_neurons = checker.check_positive_integer(num_neurons)
    self.heads = checker.check_positive_integer(heads)
    self._logits_initializer = initializers.get(logits_initializer)
    self._coef_initializer = initializers.get(coef_initializer)
    self._use_bias = checker.check_type(use_bias, bool)
    self._bias_initializer = initializers.get(bias_initializer)

    self.neuron_scale = [self.num_neurons]
    self._kwargs = kwargs

  @property
  def structure_tail(self):
    return '({}->{})'.format(self.heads, self.num_neurons)

  @single_input
  def _link(self, x, **kwargs):
    return linker.sparse_affine(
      x, self.num_neurons, self.heads, self._logits_initializer,
      self._coef_initializer, self._use_bias, self._bias_initializer)
