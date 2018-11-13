from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import activations
from tframe import initializers
from tframe import checker

from tframe.nets import RNet


class Noah(RNet):
  """Vanilla RNN cell with a linear auxiliary memory.
  """
  net_name = 'noah'

  def __init__(
      self,
      state_size,
      mem_fc=True,
      **kwargs):
    # Call parent's constructor
    RNet.__init__(self, self.net_name)

    # Attributes
    self._state_size = state_size
    self._activation = activations.get('tanh', **kwargs)
    # self._use_bias = True
    self._weight_initializer = initializers.get('xavier_uniform')
    self._bias_initializer = initializers.get('zeros')
    self._output_scale = state_size
    self._fully_connect_memories = mem_fc


  @property
  def init_state(self):
    if self._init_state is not None: return self._init_state
    self._init_state = (self._get_placeholder('h', self._state_size),
                        self._get_placeholder('s', self._state_size))
    return self._init_state


  def structure_string(self, detail=True, scale=True):
    return self.net_name + '({})'.format(self._state_size) if scale else ''


  def _link(self, pre_states, x, **kwargs):
    h, s = pre_states
    x_size = self._get_external_shape(x)

    # Calculate net_{xh}
    Wxh = self._get_variable(
      'Wxh', [self._state_size + x_size, self._state_size])
    x_h = tf.concat([x, h], axis=1)
    net_xh = tf.matmul(x_h, Wxh)

    # Calculate net_s
    if self._fully_connect_memories:
      Ws = self._get_variable('Ws', [self._state_size, self._state_size])
      net_s = tf.matmul(s, Ws)
    else:
      Ws = self._get_variable('Ws', self._state_size)
      net_s = tf.multiply(s, Ws)

    # Calculate new_h and new_s
    bias = self._get_variable('bias', self._state_size)
    net = tf.nn.bias_add(tf.add(net_xh, net_s), bias)
    new_h = self._activation(net, name='y')
    new_s = tf.add(s, new_h)

    return new_h, (new_h, new_s)


class Shem(RNet):
  net_name = 'shem'

  def __init__(self, state_size, **kwargs):
    # Call parent's constructor
    RNet.__init__(self, self.net_name)

    # Attributes
    self._state_size = state_size
    self._activation = activations.get('tanh', **kwargs)
    self._kwargs = kwargs

    # Key word arguments
    self._use_forget_gate = kwargs.get('forget_gate', False)
    self._use_input_gate = kwargs.get('input_gate', False)


  def structure_string(self, detail=True, scale=True):
    return self.net_name + '({})'.format(self._state_size) if scale else ''


  def _link(self, s, x, **kwargs):
    # Input for all units
    xs = tf.concat([x, s], axis=1, name='xs')

    # Calculate output
    y = self._easy_neurons(xs, 'output', self._activation)

    # Calculate memory
    s_to_add = self._easy_neurons(xs, 's_add', self._activation)
    if self._use_input_gate:
      i = self._easy_neurons(xs, 'input_gate', tf.sigmoid)
      s_to_add = tf.multiply(s_to_add, i)
    if self._use_forget_gate:
      f = self._easy_neurons(xs, 'forget_gate', tf.sigmoid)
      s = tf.multiply(s, f)
    new_s = tf.add(s, s_to_add)

    return y, new_s


class Ham(RNet):
  net_name = 'Ham'


  def structure_string(self, detail=True, scale=True):
    return self.net_name + '({})'.format(self._state_size) if scale else ''

class Japheth(RNet):
  net_name = 'Japheth'


  def structure_string(self, detail=True, scale=True):
    return self.net_name + '({})'.format(self._state_size) if scale else ''
