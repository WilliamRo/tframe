from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tframe import checker
from tframe import activations
from tframe import initializers

from tframe.nets.rnet import RNet
from tframe.nets.rnn_cells.basic_cell import BasicRNNCell


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


class PAMU(BasicRNNCell):
  """Practical AMU"""
  net_name = 'pamu'

  # region : Properties

  @property
  def init_state(self):
    if self._init_state is not None: return self._init_state
    assert self._state_size is not None
    get_placeholder = lambda name: self._get_placeholder(name, self._state_size)
    self._init_state = (get_placeholder('h'), get_placeholder('c'))
    return self._init_state

  # endregion : Properties

  # region : Private Methods

  def _link(self, pre_states, x, **kwargs):
    """pre_state = (h, c)"""
    self._check_state(pre_states, 2)
    h, c  = pre_states
    x_size = self._get_external_shape(x)

    # :: Update memory
    fi_inputs = tf.concat([h, x, c], axis=1)
    fio_weight_shape = [x_size + self._state_size * 2, self._state_size]
    # - Calculate output
    W, b = self._get_weight_and_bias(
      [x_size + self._state_size, self._state_size], self._use_bias)
    output = self._activation(self._net(tf.concat([x, c], axis=1), W, b))
    # - Calculate candidates to write
    Wc, bc = self._get_weight_and_bias(
      [x_size + self._state_size, self._state_size], self._use_bias, 'c')
    candidates = self._activation(self._net(tf.concat([h, x], axis=1), Wc, bc))
    # - Forget
    with tf.name_scope('forget'):
      Wf, bf = self._get_weight_and_bias(fio_weight_shape, self._use_bias, 'f')
      f = self._gate(fi_inputs, Wf, bf)
      c = tf.multiply(f, c)
    # - Write
    with tf.name_scope('write'):
      Wi, bi = self._get_weight_and_bias(fio_weight_shape, self._use_bias, 'i')
      i = self._gate(fi_inputs, Wi, bi)
      new_c = tf.add(c, tf.multiply(candidates, i))
    # :: Read
    with tf.name_scope('read'):
      Wo, bo = self._get_weight_and_bias(fio_weight_shape, self._use_bias, 'o')
      o = self._gate(tf.concat([h, x, new_c], axis=1), Wo, bo)
      new_h = tf.multiply(new_c, o)

    self._kernel, self._bias = (W, Wc, Wi, Wf, Wo), (b, bc, bi, bf, bo)
    # Return output, state
    return output, (new_h, new_c)

  def _get_zero_state(self, batch_size):
    assert not self.is_root
    return (np.zeros(shape=(batch_size, self._state_size)),
            np.zeros(shape=(batch_size, self._state_size)))

  # endregion : Private Methods
