from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import activations
from tframe import initializers
from tframe import hub

from tframe.nets import RecurrentNet


class BasicRNNCell(RecurrentNet):
  """Basic RNN cell
     TODO: Temporarily defined as a net
  """

  def __init__(
      self,
      state_size,
      activation='tanh',
      inner_struct='add',
      use_bias=True,
      weight_initializer='xavier_uniform',
      bias_initializer='zeros',
      **kwargs):
    """

    :param state_size: state size: positive int
    :param activation: activation: string or callable
    :param inner_struct: \in {'add', 'concat'}
    :param use_bias: whether to use bias
    :param weight_initializer: weight initializer identifier
    :param bias_initializer: bias initializer identifier
    """
    # Call parent's constructor
    RecurrentNet.__init__(self, 'basicell')

    # Attributes
    self._state_size = state_size
    self._inner_struct = inner_struct
    self._activation = activations.get(activation, **kwargs)
    self._use_bias = use_bias
    self._weight_initializer = initializers.get(weight_initializer)
    self._bias_initializer = initializers.get(bias_initializer)
    self._output_scale = state_size


  def structure_string(self, detail=True, scale=True):
    return self.group_name + '_{}'.format(self._state_size) if scale else ''


  def _link(self, pre_state, input_, **kwargs):
    assert isinstance(pre_state, tf.Tensor)
    state_shape = pre_state.shape.as_list()
    assert state_shape[1] == self._state_size
    assert isinstance(input_, tf.Tensor)
    input_shape = input_.shape.as_list()
    assert len(input_shape) == 2
    input_size = input_shape[1]

    # Initiate bias
    bias = (tf.get_variable('b', shape=[self._state_size], dtype=hub.dtype,
                            initializer=self._bias_initializer)
            if self._use_bias else None)

    get_variable = lambda name, shape: tf.get_variable(
      name, shape, dtype=hub.dtype, initializer=self._weight_initializer)
    if self._inner_struct == 'add':
      W_h = get_variable('W_h', [self._state_size, self._state_size])
      W_x = get_variable('W_x', [input_size, self._state_size])
      tmp = tf.matmul(pre_state, W_h) + tf.matmul(input_, W_x)
    elif self._inner_struct == 'concat':
      W = get_variable('W', [self._state_size + input_size, self._state_size])
      tmp = tf.matmul(tf.concat([input_, pre_state], axis=1), W)
    else:
      raise TypeError('!! Unknown inner structure {}'.format(
        self._inner_struct))

    if self._use_bias: tmp += bias
    state = self._activation(tmp, name='state')

    return state, state



