from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe.layers.layer import Layer
from tframe.layers.layer import single_input

from tframe.utils import get_scale
from tframe.core.decorators import init_with_graph
from tframe.core.function import Function

from tensorflow.python.layers.pooling import MaxPooling1D as MaxPool1D_
from tensorflow.python.layers.pooling import MaxPool2D as MaxPool2D_
from tensorflow.python.layers.pooling import AveragePooling2D as AveragePooling2D_

# TODO: this module needs to be refactored



class MaxPool1D(Layer, MaxPool1D_):

  full_name = 'maxpoold'
  abbreviation = 'maxpool'

  @init_with_graph
  def __init__(self, pool_size, strides,
               padding='same', data_format='channels_last',
               name=None, **kwargs):
    MaxPool1D_.__init__(
      self, pool_size, strides, padding, data_format, name, **kwargs)

  @property
  def structure_tail(self):
    size = lambda inputs: 'x'.join([str(n) for n in inputs])
    return '({}>{})'.format(size(self.pool_size), size(self.strides))

  @single_input
  def _link(self, input_=None, **kwargs):
    assert isinstance(input_, tf.Tensor)
    output = MaxPool1D_.__call__(self, input_, scope=self.full_name)
    # self.neuron_scale = get_scale(output)
    return output

  def __call__(self, *args, **kwargs):
    return Layer.__call__(self, *args, **kwargs)


class MaxPool2D(Layer, MaxPool2D_):

  full_name = 'maxpool2d'
  abbreviation = 'maxpool'

  @init_with_graph
  def __init__(self, pool_size, strides,
               padding='same', data_format='channels_last',
               name=None, **kwargs):
    MaxPool2D_.__init__(
      self, pool_size, strides, padding, data_format, name, **kwargs)

  @property
  def structure_tail(self):
    size = lambda inputs: 'x'.join([str(n) for n in inputs])
    return '({}>{})'.format(size(self.pool_size), size(self.strides))

  @single_input
  def _link(self, input_=None, **kwargs):
    assert isinstance(input_, tf.Tensor)
    output = MaxPool2D_.__call__(self, input_, scope=self.full_name)
    # self.neuron_scale = get_scale(output)
    return output

  def __call__(self, *args, **kwargs):
    return Layer.__call__(self, *args, **kwargs)


class AveragePooling2D(Layer, AveragePooling2D_):

  full_name = 'avgpool2d'
  abbreviation = 'avgpool2d'

  @init_with_graph
  def __init__(self, pool_size, strides,
               padding='same', data_format='channels_last',
               name=None, **kwargs):
    AveragePooling2D_.__init__(
      self, pool_size, strides, padding, data_format, name, **kwargs)

  @property
  def structure_tail(self):
    size = lambda inputs: 'x'.join([str(n) for n in inputs])
    return '({}>{})'.format(size(self.pool_size), size(self.strides))

  @single_input
  def _link(self, input_=None, **kwargs):
    assert isinstance(input_, tf.Tensor)
    output = AveragePooling2D_.__call__(self, input_, scope=self.full_name)
    return output

  def __call__(self, *args, **kwargs):
    return Layer.__call__(self, *args, **kwargs)


class GlobalAveragePooling2D(Layer):

  full_name = 'globalavgpool2d'
  abbreviation = 'gap2d'

  @init_with_graph
  def __init__(self, data_format='channels_last', flatten=True, **kwargs):
    self._data_format = data_format
    assert data_format == 'channels_last'
    self._flatten = flatten
    self._kwargs = kwargs

  @single_input
  def _link(self, input_, **kwargs):
    assert isinstance(input_, tf.Tensor)
    shape = input_.shape.as_list()
    assert len(shape) == 4
    output = tf.layers.average_pooling2d(
      input_, pool_size=shape[1:3], strides=(1, 1),
      data_format=self._data_format)
    output = tf.reshape(output, shape=[-1, output.shape.as_list()[-1]])
    return output

