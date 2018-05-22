from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tframe import checker
from tframe import activations
from tframe import initializers

from tframe.nets.rnet import RNet


class AMU(RNet):
  """Recurrent Neural Networks With Auxiliary Memory Units
     ref: https://ieeexplore.ieee.org/document/7883962/
  """
  net_name = 'amu'

  def __init__(
      self,
      output_dim,
      neurons_per_amu=3,
      activation='tanh',
      use_bias=True,
      weight_initializer='xavier_uniform',
      bias_initializer='zeros',
      **kwargs):
    # Call parent's constructor
    RNet.__init__(self, AMU.net_name)

    # Attributes
    self._output_dim = output_dim
    self._neurons_per_amu = neurons_per_amu
    self._activation = activations.get(activation, **kwargs)
    self._use_bias = checker.check_type(use_bias, bool)
    self._weight_initializer = initializers.get(weight_initializer)
    self._bias_initializer = initializers.get(bias_initializer)

    self._output_scale = output_dim

  # region : Properties

  def structure_string(self, detail=True, scale=True):
    return self.net_name + '({}x{})'.format(
      self._output_dim, self._neurons_per_amu)

  @property
  def init_state(self):
    if self._init_state is not None: return self._init_state
    self._init_state = (self._get_placeholder('h', self.num_neurons),
                        self._get_placeholder('s', self._output_dim))
    return self._init_state

  @property
  def num_neurons(self):
    assert isinstance(self._output_dim, int)
    assert isinstance(self._neurons_per_amu, int)
    return self._output_dim * self._neurons_per_amu

  # endregion : Properties

  # region : Private Methods

  def _link(self, pre_states, x, **kwargs):
    """pre_states = (h, s) in which
          len(h) = output_dim * neuron_per_amu
          len(s) = output_dim
    """
    self._check_state(pre_states, (self.num_neurons, self._output_dim))
    h, s = pre_states
    input_size = self._get_external_shape(x)

    W = self._get_variable(
      'W', [self.num_neurons + input_size + self._output_dim, self.num_neurons])
    bias = None
    if self._use_bias: bias = self._get_bias('b', self.num_neurons)
    new_h = self._activation(tf.nn.bias_add(tf.matmul(
      tf.concat([h, x, s], axis=1), W), bias))

    # Form AMUs
    r = tf.reshape(new_h, [-1, self._neurons_per_amu, self._output_dim], 'amus')
    # - Write
    with tf.name_scope('write'): new_s = tf.add(s, tf.reduce_prod(r, axis=1))

    # return outputs, states
    self._kernel, self._bias = W, bias
    return r[:, 0, :], (new_h, new_s)

  def _get_zero_state(self, batch_size):
    assert not self.is_root
    return (np.zeros(shape=(batch_size, self.num_neurons)),
            np.zeros(shape=(batch_size, self._output_dim)))

  # endregion : Private Methods

