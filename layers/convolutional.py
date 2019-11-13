from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.core.function import Function
from tframe.core.decorators import init_with_graph
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input

# from tframe.utils import get_scale

from tensorflow.python.layers.convolutional import Conv1D as _Conv1D
from tensorflow.python.layers.convolutional import Conv2D as _Conv2D
from tensorflow.python.layers.convolutional import Deconv2D as _Deconv2D


def _get_neuron_scale(filters, kernel_size):
  if not isinstance(kernel_size, (list, tuple)):
    kernel_size = [kernel_size]
  assert not isinstance(filters, (list, tuple))
  return list(kernel_size) + [filters]


class _Conv(Layer):
  is_nucleus = True
  abbreviation = 'conv'

  @init_with_graph
  def __init__(self, *args, expand_last_dim=False, **kwargs):
    # IDEs such as pycharm should be able to find the noumenon's para infos
    self.noumenon = super(Function, self)
    self.noumenon.__init__(*args, **kwargs)
    self.expand_last_dim = expand_last_dim

  @single_input
  def _link(self, input_=None, **kwargs):
    assert isinstance(input_, tf.Tensor)
    # Expand last dimension if necessary
    if self.expand_last_dim: input_ = tf.expand_dims(input_, -1)
    # TODO: too violent ?
    output = self.noumenon.__call__(input_, scope=self.full_name)
    # self.neuron_scale = get_scale(output)
    return output

  def __call__(self, *args, **kwargs):
    return Layer.__call__(self, *args, **kwargs)


# The tensorflow class is next to Function in the __mro__ list of the
#  classes below

class Conv1D(_Conv, _Conv1D):
  full_name = 'convolutional1d'
  abbreviation = 'conv1d'

  def __init__(self, filters,
               kernel_size,
               strides=1,
               padding='same',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    _Conv.__init__(
      self,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      trainable=trainable,
      name=name, **kwargs)
    self.neuron_scale = _get_neuron_scale(self.filters, self.kernel_size)


class Conv2D(_Conv, _Conv2D):
  full_name = 'convolutional2d'
  abbreviation = 'conv2d'

  def __init__(self, filters,
               kernel_size,
               strides=(1, 1),
               padding='same',
               data_format='channels_last',
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               expand_last_dim=False,
               **kwargs):
    _Conv.__init__(
      self,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      trainable=trainable,
      expand_last_dim=expand_last_dim,
      name=name, **kwargs)
    self.neuron_scale = _get_neuron_scale(self.filters, self.kernel_size)


class Deconv2D(_Conv, _Deconv2D):
  full_name = 'deconvolutional2d'
  abbreviation = 'deconv2d'

  def __init__(self, filters,
               kernel_size,
               strides=(1, 1),
               padding='same',
               data_format='channels_last',
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    _Conv.__init__(
      self,
      filters,
      kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      trainable=trainable,
      name=name,
      **kwargs)
    self.neuron_scale = _get_neuron_scale(self.filters, self.kernel_size)


if __name__ == '__main__':
  print(Conv2D.__mro__)


