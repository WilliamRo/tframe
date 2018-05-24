from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import activations
from tframe import initializers
from tframe import checker

from tframe.nets import RNet


class BasicRNNCell(RNet):
  """Basic RNN cell
     TODO: Temporarily defined as a net
  """
  net_name = 'basic_cell'

  def __init__(
      self,
      state_size,
      activation='tanh',
      use_bias=True,
      weight_initializer='xavier_uniform',
      bias_initializer='zeros',
      **kwargs):
    """
    :param state_size: state size: positive int
    :param activation: activation: string or callable
    :param use_bias: whether to use bias
    :param weight_initializer: weight initializer identifier
    :param bias_initializer: bias initializer identifier
    """
    # Call parent's constructor
    RNet.__init__(self, self.net_name)

    # Attributes
    self._state_size = state_size
    self._activation = activations.get(activation, **kwargs)
    self._use_bias = checker.check_type(use_bias, bool)
    self._weight_initializer = initializers.get(weight_initializer)
    self._bias_initializer = initializers.get(bias_initializer)
    self._output_scale = state_size


  def structure_string(self, detail=True, scale=True):
    return self.net_name + '({})'.format(self._state_size) if scale else ''


  def _link(self, pre_state, input_, **kwargs):
    self._check_state(pre_state)
    input_size = self._get_external_shape(input_)

    # Initiate bias
    bias = None
    if self._use_bias: bias = self._get_bias('b', self._state_size)
    W = self._get_variable(
      'W', [self._state_size + input_size, self._state_size])

    net = tf.nn.bias_add(
      tf.matmul(tf.concat([input_, pre_state], axis=1), W), bias)
    state = self._activation(net, name='state')

    self._kernel, self._bias = W, bias
    return state, state



